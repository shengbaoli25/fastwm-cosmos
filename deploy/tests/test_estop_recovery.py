"""Emergency-stop recovery tests use fake adapters, never real hardware."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realman_deploy.contracts import ActionChunk, Observation
from realman_deploy.normalization import DatasetNormalizer
from realman_deploy.runtime import DeploymentRuntime
from realman_deploy.web import DeploymentWeb


class FakeHardware:
    def __init__(self):
        self.armed = False
        self.estop_latched = False
        self.gripper = None
        self.reset_error = None
        self.reset_hook = None
        self.reset_calls = 0
        self.estops = 0
        self.arm = Mock(side_effect=AssertionError('recovery must not arm'))
        self.command = Mock(side_effect=AssertionError('recovery must not move'))
        self.observe = Mock(side_effect=AssertionError('recovery must not sample'))
        self.slow_stop = Mock()
        self.close = Mock()

    def status(self):
        return dict(action_space='joint8', force_preprocessing=None,
                    armed=self.armed, estop_latched=self.estop_latched,
                    robot_ready=True, cameras_ready=True, force_ready=True,
                    gripper_ready=True)

    def emergency_stop(self):
        self.estops += 1
        self.armed = False
        self.estop_latched = True

    def reset_emergency_stop(self):
        self.reset_calls += 1
        if self.reset_hook:
            self.reset_hook()
        if self.reset_error:
            raise self.reset_error
        self.estop_latched = False


@pytest.fixture
def runtime():
    normalizer = DatasetNormalizer({
        'state': {'mean': [0.] * 8, 'std': [1.] * 8},
        'action': {'mean': [0.] * 8, 'std': [1.] * 8},
        'force': {'mean': [0.] * 6, 'std': [1.] * 6},
    }, action_space='joint8')
    value = DeploymentRuntime(normalizer, require_cuda=False, clock=lambda: 100.,
                              execution_mode='synchronous', sampling_profile='161616')
    value._cuda_status = lambda: {'available': True, 'checking': False}
    yield value
    value._infer_active = value._gripper_debug_busy = value._hardware_connecting = False
    value._load_thread = None
    value.close()


def attach(runtime):
    hardware = FakeHardware()
    policy = SimpleNamespace(close=Mock(), diagnostics={'checkpoint': 'fake.pt'})
    runtime.attach(hardware=hardware, policy=policy)
    return hardware, policy


def background(call):
    outcome = {}

    def run():
        try:
            outcome['result'] = call()
        except Exception as exc:
            outcome['error'] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


def release_and_join(thread, release):
    release.set()
    thread.join(2)
    assert not thread.is_alive(), 'fake operation did not finish'


def test_reset_clears_old_actions_and_history_without_resuming_or_unloading(runtime):
    hardware, policy = attach(runtime)
    obs = Observation(100., 4, (0.,) * 7 + (0.5,), (0.,) * 6,
                      {'global': b'rgb'}, action_space='joint8')
    actions = tuple([obs.state] * 32)
    old_epoch = runtime.gate.current()
    chunk = ActionChunk(4, 100., old_epoch, actions, 100., 'joint8')
    runtime.timeline.replace(chunk, 100.)
    runtime.observations.put(obs)
    # The legacy rolling-reference object is EEF7 even in a Joint8 runtime.
    # Seed its old buffer explicitly to prove recovery clears it too.
    runtime.reference.update([row[:7] for row in actions], source_stamp=100.)
    runtime._history.append(obs.state, obs.force)
    old_history = runtime._history
    runtime._input_history = old_history.tensors()
    runtime._pending_chunk = chunk
    runtime._running_since = 90.
    runtime._run_inference_count = 5
    runtime.inference_count = 12
    runtime.emergency_stop()
    result = runtime.reset_emergency_stop(confirmed=True)
    assert result['state'] == 'paused' and result['error'] is None
    assert result['estop_latched'] is False and result['estop_reset_busy'] is False
    assert result['model_ready'] is True and runtime.policy is policy
    assert result['action_queue'] == 0 and result['pending_chunk'] is False
    assert result['observation_sequence'] == -1 and result['bootstrap'] is True
    assert runtime.reference.last_raw() is None
    assert runtime._input_history is None and runtime._history is not old_history
    assert all(runtime._history.tensors()[1]), 'history must contain only masked padding'
    assert runtime._running_since is None and runtime._run_inference_count == 0
    assert runtime.inference_count == 12 and runtime._threads == []
    assert runtime.hardware is hardware and hardware.reset_calls == 1
    assert hardware.armed is False and result['sampling_profile'] == '161616'
    assert result['execution_mode'] == 'synchronous'
    assert runtime._publish([list(row) for row in actions], obs, old_epoch, 99.) is False
    assert len(runtime.timeline) == 0 and runtime.reference.last_raw() is None
    for method in [hardware.arm, hardware.command, hardware.observe, policy.close]:
        method.assert_not_called()


@pytest.mark.parametrize('confirmed', [False, None, 0, 1, 'true', [], {}])
def test_confirmation_requires_literal_true(runtime, confirmed):
    hardware, _ = attach(runtime)
    runtime.emergency_stop()
    with pytest.raises(ValueError, match='确认'):
        runtime.reset_emergency_stop(confirmed=confirmed)
    assert runtime._estop_latched and hardware.reset_calls == 0


@pytest.mark.parametrize('busy', [
    'inference', 'loading', 'hardware_connect', 'gripper_debug',
    'gripper_worker', 'emergency_stop', 'command',
])
def test_busy_work_cannot_be_cleared_by_reset(runtime, busy):
    hardware, _ = attach(runtime)
    runtime.emergency_stop()
    if busy == 'inference':
        runtime._infer_active = True
    elif busy == 'loading':
        runtime._load_thread = SimpleNamespace(is_alive=lambda: True)
    elif busy == 'hardware_connect':
        runtime._hardware_connecting = True
    elif busy == 'gripper_debug':
        runtime._gripper_debug_busy = True
    elif busy == 'gripper_worker':
        hardware.gripper = SimpleNamespace(status=lambda: {'busy': True}, hold=Mock())
    elif busy == 'emergency_stop':
        runtime._estop_inflight = 1
    else:
        runtime._command_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='进行|动作接口'):
            runtime.reset_emergency_stop(confirmed=True)
        assert runtime._estop_latched and runtime.state == 'estop'
        assert hardware.reset_calls == 0 and not runtime._estop_reset_busy
    finally:
        if busy == 'command':
            runtime._command_lock.release()
        runtime._estop_inflight = 0


@pytest.mark.parametrize('state', ['stopping', 'stopped'])
def test_service_shutdown_rejects_reset(runtime, state):
    hardware, _ = attach(runtime)
    runtime.emergency_stop()
    runtime.state = state
    with pytest.raises(RuntimeError, match='服务正在停止'):
        runtime.reset_emergency_stop(confirmed=True)
    assert hardware.reset_calls == 0


def test_shutdown_event_rejects_reset(runtime):
    runtime.emergency_stop()
    runtime.stop_event.set()
    with pytest.raises(RuntimeError, match='服务正在停止'):
        runtime.reset_emergency_stop(confirmed=True)
    assert runtime._estop_latched


@pytest.mark.parametrize('failure', ['exception', 'latched', 'armed', 'missing_feedback', 'gripper_hold'])
def test_failed_hardware_recovery_keeps_runtime_locked(runtime, failure):
    hardware, _ = attach(runtime)
    runtime.emergency_stop()
    if failure == 'exception':
        hardware.reset_error = RuntimeError('physical stop still active')
    elif failure == 'latched':
        hardware.reset_emergency_stop = Mock()
    elif failure == 'armed':
        hardware.reset_hook = lambda: setattr(hardware, 'armed', True)
    elif failure == 'missing_feedback':
        hardware.status = lambda: {'action_space': 'joint8'}
    else:
        hardware.gripper = SimpleNamespace(status=lambda: {'busy': False},
            hold=Mock(side_effect=RuntimeError('gripper hold not acknowledged')))
    with pytest.raises(RuntimeError):
        runtime.reset_emergency_stop(confirmed=True)
    assert runtime.state == 'estop' and runtime._estop_latched
    assert runtime.error.startswith('急停恢复失败，保持锁定:')
    assert runtime._estop_reset_busy is False
    assert runtime._command_lock.acquire(blocking=False)
    runtime._command_lock.release()
    with pytest.raises(RuntimeError, match='急停已锁存'):
        runtime.start()
    hardware.arm.assert_not_called()
    hardware.command.assert_not_called()


def test_armed_hardware_rejected_before_reset(runtime):
    hardware, _ = attach(runtime)
    runtime.emergency_stop()
    hardware.armed = True
    with pytest.raises(RuntimeError, match='控制使能'):
        runtime.reset_emergency_stop(confirmed=True)
    assert hardware.reset_calls == 0 and runtime._estop_latched


def test_unsupported_adapter_keeps_stop_latched(runtime):
    hardware, _ = attach(runtime)
    hardware.reset_emergency_stop = None
    runtime.emergency_stop()
    assert runtime.status()['estop_reset_supported'] is False
    with pytest.raises(RuntimeError, match='不支持安全恢复'):
        runtime.reset_emergency_stop(confirmed=True)
    assert runtime.state == 'estop' and runtime._estop_latched


def test_new_emergency_stop_preempts_blocked_reset(runtime):
    hardware, policy = attach(runtime)
    runtime.emergency_stop()
    entered, release = threading.Event(), threading.Event()

    def blocked_reset():
        entered.set()
        assert release.wait(2), 'fake reset was not released'

    hardware.reset_hook = blocked_reset
    thread, outcome = background(lambda: runtime.reset_emergency_stop(confirmed=True))
    try:
        assert entered.wait(1)
        runtime.emergency_stop()
        stop_epoch = runtime.gate.current()
        assert hardware.estops == 2
    finally:
        release_and_join(thread, release)
    assert isinstance(outcome.get('error'), RuntimeError)
    assert '取消' in str(outcome['error'])
    assert runtime._estop_latched and runtime.state == 'estop'
    assert runtime.gate.current() == stop_epoch and runtime.policy is policy
    assert runtime._estop_reset_busy is False
    with pytest.raises(RuntimeError, match='急停已锁存'):
        runtime.start()
    hardware.arm.assert_not_called()


def test_duplicate_reset_and_other_controls_rejected_while_reset_in_progress(runtime):
    hardware, _ = attach(runtime)
    runtime.emergency_stop()
    entered, release = threading.Event(), threading.Event()

    def blocked_reset():
        entered.set()
        assert release.wait(2), 'fake reset was not released'

    hardware.reset_hook = blocked_reset
    thread, outcome = background(lambda: runtime.reset_emergency_stop(confirmed=True))
    try:
        assert entered.wait(1)
        assert runtime.status()['estop_reset_busy'] is True
        with pytest.raises(RuntimeError, match='进行'):
            runtime.reset_emergency_stop(confirmed=True)
        factory = Mock(side_effect=AssertionError('must not create adapters'))
        for operation in [runtime.start, lambda: runtime.attach(policy=object()),
            lambda: runtime.configure_sampling('4416', 'synchronous'),
            lambda: runtime.load_policy_async(factory, 'unused.pt'),
            lambda: runtime.connect_hardware(factory),
            lambda: runtime.debug_gripper('connect', factory=factory)]:
            with pytest.raises(RuntimeError, match='急停恢复正在进行'):
                operation()
        factory.assert_not_called()
    finally:
        release_and_join(thread, release)
    assert 'error' not in outcome and outcome['result']['state'] == 'paused'
    assert hardware.reset_calls == 1


def test_no_hardware_can_release_software_only_latch(runtime):
    runtime.emergency_stop()
    assert runtime.status()['estop_reset_supported'] is True
    result = runtime.reset_emergency_stop(confirmed=True)
    assert result['state'] == 'paused' and result['estop_latched'] is False
    assert runtime.policy is runtime.hardware is None and runtime._threads == []


def test_confirmed_repeated_reset_is_idempotent_only_when_already_paused(runtime):
    hardware, _ = attach(runtime)
    runtime.emergency_stop()
    runtime.reset_emergency_stop(confirmed=True)
    epoch = runtime.gate.current()
    result = runtime.reset_emergency_stop(confirmed=True)
    assert result['state'] == 'paused' and result['epoch'] == epoch
    assert hardware.reset_calls == 1
    with pytest.raises(ValueError):
        runtime.reset_emergency_stop()
    runtime.state = 'ready'
    with pytest.raises(RuntimeError, match='未处于软件急停'):
        runtime.reset_emergency_stop(confirmed=True)


@pytest.fixture
def http_service(tmp_path, runtime):
    config = tmp_path / 'runtime.json'
    config.write_text(json.dumps(dict(checkpoint_directory=str(tmp_path / 'weights'),
        hardware_config='hardware.json', sampling_profile='161616',
        execution_mode='synchronous')), encoding='utf-8')
    catalog = tmp_path / 'catalog.json'
    catalog.write_text('{"checkpoints": []}', encoding='utf-8')
    web = DeploymentWeb(runtime, catalog_path=catalog, config_path=config, port=0)
    thread = threading.Thread(target=web.serve_forever, daemon=True)
    thread.start()
    assert web.serving.wait(1)
    base = f'http://127.0.0.1:{web.server.server_address[1]}'

    def request(path, data=None, *, headers=None):
        req = Request(base + path, data=None if data is None else json.dumps(data).encode(),
                      headers={'Content-Type': 'application/json', **(headers or {})})
        try:
            response = urlopen(req, timeout=3)
        except HTTPError as error:
            response = error
        with response:
            return response.code, json.load(response)

    yield request
    web.close()
    thread.join(2)
    assert not thread.is_alive()


def test_http_reset_returns_paused_and_retains_loaded_model(http_service, runtime):
    hardware, policy = attach(runtime)
    assert http_service('/api/estop', {})[0] == 200
    code, result = http_service('/api/estop/reset', {'confirmed': True})
    assert code == 200, result
    assert result['state'] == 'paused' and result['model_ready'] is True
    assert result['estop_latched'] is False and result['estop_reset_busy'] is False
    assert runtime.policy is policy and hardware.reset_calls == 1
    hardware.arm.assert_not_called()
    hardware.command.assert_not_called()


@pytest.mark.parametrize('payload', [
    {}, [], True, {'confirmed': False}, {'confirmed': 1}, {'confirmed': 'true'},
    {'confirmed': None}, {'confirmed': True, 'resume': True},
])
def test_http_reset_rejects_missing_false_or_extra_fields(http_service, runtime, payload):
    hardware, _ = attach(runtime)
    runtime.emergency_stop()
    code, result = http_service('/api/estop/reset', payload)
    assert code == 400 and result.get('error')
    assert hardware.reset_calls == 0 and runtime._estop_latched


def test_http_cross_origin_cannot_reset_emergency_stop(http_service, runtime):
    hardware, _ = attach(runtime)
    runtime.emergency_stop()
    code, result = http_service('/api/estop/reset', {'confirmed': True},
                               headers={'Origin': 'http://unrelated.invalid'})
    assert code == 403 and 'cross-origin' in result['error']
    assert runtime._estop_latched and hardware.reset_calls == 0


def test_status_polling_never_attempts_recovery_or_actuator_commands(http_service, runtime):
    hardware, _ = attach(runtime)
    runtime.emergency_stop()
    for _ in range(3):
        code, result = http_service('/api/status')
        assert code == 200 and result['estop_latched'] is True
        assert result['estop_reset_supported'] is True and result['estop_reset_busy'] is False
    assert hardware.reset_calls == 0 and hardware.estops == 1
    for method in [hardware.arm, hardware.command, hardware.observe, hardware.slow_stop]:
        method.assert_not_called()
