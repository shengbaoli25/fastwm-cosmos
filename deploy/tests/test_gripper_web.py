"""Gripper diagnostics use fake workers only; no CUDA, SDK or camera I/O."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realman_deploy.normalization import DatasetNormalizer
from realman_deploy.runtime import DeploymentRuntime
from realman_deploy.web import DeploymentWeb


class FakeWorker:
    def __init__(self, *, events=None):
        self.events = events if events is not None else []
        self.calls = []
        self.connected = False
        self.opening = 0.5
        self.enabled = False
        self.alarm = 0
        self.busy = False
        self.close_error = None
        self.block = None
        self.entered = threading.Event()
        self.cancel_event = None
        self.holds = 0

    def start(self):
        self.events.append('start')
        self.connected = True

    def close(self):
        self.events.append('close')
        if self.close_error:
            raise self.close_error
        self.connected = False

    def wait_connected(self, timeout_s):
        self.events.append(('wait_connected', timeout_s))
        self.entered.set()
        if self.block is not None and not self.block.wait(3):
            raise RuntimeError('fake connection was not released')
        return self.status()

    def status(self):
        return dict(connected=self.connected, ready=self.connected and self.enabled,
                    enabled=self.enabled, alarm=self.alarm, opening=self.opening,
                    busy=self.busy, error=None)

    def debug_command(self, action, *, opening, speed, torque, timeout_s, cancel_event):
        self.calls.append(dict(action=action, opening=opening, speed=speed, torque=torque,
                               timeout_s=timeout_s))
        self.cancel_event = cancel_event
        self.entered.set()
        if self.block is not None and not self.block.wait(3):
            raise RuntimeError('fake command was not released')
        # A delayed success deliberately ignores cancellation: runtime must reject it.
        if action in {'open', 'close', 'move'}:
            self.opening = {'open': 1.0, 'close': 0.0}.get(action, opening)
        elif action in {'enable', 'disable'}:
            self.enabled = action == 'enable'
        elif action == 'clear_error':
            self.alarm = 0
        return self.status()

    def hold(self):
        self.holds += 1
        self.events.append('hold')


class FakeHardware:
    def __init__(self, *, gripper=None, events=None):
        self.gripper = gripper
        self.events = events if events is not None else []
        self.armed = False
        self.estop_latched = False
        self.closed = False
        self.slow_stops = self.estops = 0

    def status(self):
        return dict(action_space='joint8', force_preprocessing=None,
                    armed=self.armed, estop_latched=self.estop_latched,
                    robot_ready=True, cameras_ready=True, force_ready=True,
                    gripper_ready=self.gripper is not None)

    def slow_stop(self):
        self.slow_stops += 1
        self.armed = False

    def emergency_stop(self):
        self.estops += 1
        self.estop_latched = True
        self.armed = False

    def close(self):
        self.events.append('hardware.close')
        self.closed = True


@pytest.fixture
def runtime():
    normalizer = DatasetNormalizer({
        'state': {'mean': [0.] * 8, 'std': [1.] * 8},
        'action': {'mean': [0.] * 8, 'std': [1.] * 8},
        'force': {'mean': [0.] * 6, 'std': [1.] * 6},
    }, action_space='joint8')
    result = DeploymentRuntime(normalizer, require_cuda=False,
                               execution_mode='synchronous', sampling_profile='161616')
    result._cuda_status = lambda: {'available': True, 'checking': False}
    yield result
    worker = result._gripper_worker()
    if worker is not None:
        worker.close_error = None
        if worker.block is not None:
            worker.block.set()
    result.close()


def attach_worker(runtime, worker=None):
    worker = worker or FakeWorker()
    runtime.debug_gripper('connect', factory=lambda: worker)
    return worker


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


def finish(thread, worker):
    if worker.block is not None:
        worker.block.set()
    thread.join(3)
    assert not thread.is_alive(), 'fake operation did not finish'


def test_standalone_connect_and_status_need_no_policy_or_cameras(runtime):
    worker = attach_worker(runtime)
    assert runtime.policy is runtime.hardware is None
    assert worker.events == ['start', ('wait_connected', 8.0)]
    assert runtime.gripper_debug_status()['connected'] is True
    result = runtime.debug_gripper('status')
    assert result['supported'] is True
    assert worker.calls[-1]['action'] == 'status'
    assert runtime.state == 'idle'


@pytest.mark.parametrize('action,kwargs,expected', [
    ('open', {'speed': 20, 'torque': 30}, 1.0),
    ('close', {'speed': 10, 'torque': 40}, 0.0),
    ('move', {'opening': 0.35, 'speed': 15, 'torque': 25}, 0.35),
    ('move', {'opening': 0}, 0.0),
    ('move', {'opening': 1}, 1.0),
])
def test_movement_payload_is_forwarded_to_worker(runtime, action, kwargs, expected):
    worker = attach_worker(runtime)
    result = runtime.debug_gripper(action, **kwargs)
    assert result['opening'] == expected
    assert worker.calls[-1] == dict(action=action, timeout_s=12.0,
        opening=kwargs.get('opening'), speed=kwargs.get('speed'), torque=kwargs.get('torque'))
    assert worker.cancel_event is not None and not worker.cancel_event.is_set()


@pytest.mark.parametrize('action', ['enable', 'disable', 'clear_error'])
def test_nonmovement_commands_reach_worker(runtime, action):
    worker = attach_worker(runtime)
    runtime.debug_gripper(action)
    assert worker.calls[-1]['action'] == action


@pytest.mark.parametrize('action,kwargs', [
    ('unknown', {}), (None, {}), (True, {}),
    ('move', {}), ('move', {'opening': True}), ('move', {'opening': '0.5'}),
    ('move', {'opening': -0.1}), ('move', {'opening': 1.1}),
    ('move', {'opening': math.nan}), ('move', {'opening': math.inf}),
    ('open', {'opening': 0.5}), ('status', {'speed': 30}),
    ('enable', {'torque': 30}), ('open', {'speed': True}),
    ('open', {'speed': 0}), ('open', {'speed': 101}), ('open', {'speed': 30.0}),
    ('close', {'torque': False}), ('close', {'torque': -1}),
    ('close', {'torque': 101}), ('close', {'torque': '30'}),
])
def test_invalid_commands_do_not_enter_worker(runtime, action, kwargs):
    worker = attach_worker(runtime)
    with pytest.raises(ValueError):
        runtime.debug_gripper(action, **kwargs)
    assert worker.calls == []
    assert runtime.gripper_debug_status()['busy'] is False


@pytest.mark.parametrize('state', ['running', 'loading', 'stopping', 'stopped'])
def test_active_runtime_rejects_gripper_commands(runtime, state):
    worker = attach_worker(runtime)
    runtime.state = state
    with pytest.raises(RuntimeError):
        runtime.debug_gripper('open')
    assert worker.calls == []


@pytest.mark.parametrize('guard', ['estop', 'inference', 'worker_busy', 'load_thread', 'hardware_connecting'])
def test_busy_and_estop_guards_reject_gripper_commands(runtime, guard):
    worker = attach_worker(runtime)
    if guard == 'estop':
        runtime.emergency_stop()
    elif guard == 'inference':
        runtime._infer_active = True
    elif guard == 'worker_busy':
        worker.busy = True
    elif guard == 'load_thread':
        runtime._load_thread = Mock(is_alive=lambda: True)
    else:
        runtime._hardware_connecting = True
    with pytest.raises(RuntimeError):
        runtime.debug_gripper('open')
    assert worker.calls == []


@pytest.mark.parametrize('flag', ['armed', 'estop_latched'])
def test_hardware_latches_reject_manual_gripper_commands(runtime, flag):
    worker = FakeWorker()
    runtime.hardware = hardware = FakeHardware(gripper=worker)
    setattr(hardware, flag, True)
    with pytest.raises(RuntimeError):
        runtime.debug_gripper('open')
    assert worker.calls == []


def test_gripper_operation_blocks_other_lifecycle_changes(runtime):
    worker = attach_worker(runtime)
    worker.block = threading.Event()
    worker.entered.clear()
    thread, outcome = background(lambda: runtime.debug_gripper('open'))
    assert worker.entered.wait(1)
    factory = Mock()
    persist = Mock()
    try:
        assert runtime.gripper_debug_status()['busy'] is True
        for command in (runtime.start,
                        lambda: runtime.configure_sampling('4416', 'synchronous', persist=persist),
                        lambda: runtime.load_policy_async(factory, 'fake.pt'),
                        lambda: runtime.connect_hardware(factory),
                        lambda: runtime.debug_gripper('connect', factory=factory),
                        lambda: runtime.debug_gripper('status')):
            with pytest.raises(RuntimeError, match='正在进行'):
                command()
        factory.assert_not_called()
        persist.assert_not_called()
    finally:
        finish(thread, worker)
    assert 'error' not in outcome
    assert runtime.gripper_debug_status()['busy'] is False


@pytest.mark.parametrize('stop,expected', [('pause', 'paused'), ('emergency_stop', 'estop')])
def test_pause_and_estop_cancel_pending_command_and_reject_late_success(runtime, stop, expected):
    worker = attach_worker(runtime)
    worker.block = threading.Event()
    worker.entered.clear()
    thread, outcome = background(lambda: runtime.debug_gripper('open'))
    assert worker.entered.wait(1)
    try:
        getattr(runtime, stop)()
        assert runtime.state == expected
        assert worker.cancel_event.is_set()
        assert worker.holds == 1
    finally:
        finish(thread, worker)
    assert isinstance(outcome.get('error'), RuntimeError)
    assert '取消' in str(outcome['error'])
    assert 'result' not in outcome
    assert runtime.gripper_debug_status()['busy'] is False


def test_standalone_handoff_closes_old_connection_before_factory(runtime):
    events = []
    worker = attach_worker(runtime, FakeWorker(events=events))
    events.clear()
    hardware = FakeHardware()

    def factory():
        events.append('factory')
        assert worker.connected is False
        return hardware

    assert runtime.connect_hardware(factory)['robot_ready'] is True
    assert events == ['close', 'factory']
    assert runtime._debug_gripper is None and runtime.hardware is hardware
    second_factory = Mock()
    runtime.connect_hardware(second_factory)
    second_factory.assert_not_called()


def test_close_failure_keeps_standalone_and_never_creates_second_connection(runtime):
    worker = attach_worker(runtime)
    worker.close_error = RuntimeError('worker did not exit')
    factory = Mock()
    with pytest.raises(RuntimeError, match='did not exit'):
        runtime.connect_hardware(factory)
    assert runtime._debug_gripper is worker and runtime.hardware is None
    assert runtime._hardware_connecting is False
    factory.assert_not_called()
    assert worker.connected is True


def test_reconnect_close_failure_keeps_existing_worker(runtime):
    worker = attach_worker(runtime)
    worker.close_error = RuntimeError('worker did not exit')
    factory = Mock()
    with pytest.raises(RuntimeError, match='did not exit'):
        runtime.debug_gripper('connect', factory=factory)
    assert runtime._debug_gripper is worker
    assert runtime.gripper_debug_status()['busy'] is False
    factory.assert_not_called()


@pytest.mark.parametrize('stop', ['pause', 'emergency_stop'])
def test_stop_during_connect_wait_rejects_late_connection(runtime, stop):
    worker = FakeWorker()
    worker.block = threading.Event()
    thread, outcome = background(lambda: runtime.debug_gripper('connect', factory=lambda: worker))
    assert worker.entered.wait(1)
    cancel = runtime._gripper_debug_cancel
    try:
        getattr(runtime, stop)()
        assert cancel.is_set()
        assert worker.holds == 1
    finally:
        finish(thread, worker)
    assert isinstance(outcome.get('error'), RuntimeError)
    assert '取消' in str(outcome['error'])


@pytest.mark.parametrize('stop', ['pause', 'emergency_stop'])
def test_stop_can_preempt_worker_factory_and_discard_unused_worker(runtime, stop):
    worker = FakeWorker()
    entered, release = threading.Event(), threading.Event()

    def factory():
        entered.set()
        assert release.wait(3)
        return worker

    thread, outcome = background(lambda: runtime.debug_gripper('connect', factory=factory))
    assert entered.wait(1)
    stopper, stopped = background(getattr(runtime, stop))
    try:
        stopper.join(1)
        assert not stopper.is_alive(), 'worker factory held the runtime lock during stop'
        assert 'error' not in stopped
    finally:
        release.set()
        thread.join(3)
        stopper.join(3)
    assert not thread.is_alive()
    assert isinstance(outcome.get('error'), RuntimeError)
    assert worker.events == ['close'], 'cancelled worker must never start'
    assert runtime._debug_gripper is None


def test_pause_during_standalone_close_prevents_full_hardware_factory(runtime):
    worker = attach_worker(runtime)
    entered, release = threading.Event(), threading.Event()
    original_close = worker.close

    def blocked_close():
        entered.set()
        assert release.wait(3)
        original_close()

    worker.close = blocked_close
    factory = Mock(return_value=FakeHardware())
    thread, outcome = background(lambda: runtime.connect_hardware(factory))
    assert entered.wait(1)
    try:
        runtime.pause()
    finally:
        release.set()
        thread.join(3)
        worker.close = original_close
    assert not thread.is_alive()
    assert isinstance(outcome.get('error'), RuntimeError)
    factory.assert_not_called()
    assert runtime.hardware is None
    assert runtime._hardware_connecting is False


@pytest.mark.parametrize('stop', ['pause', 'emergency_stop'])
def test_stop_during_full_hardware_factory_closes_late_adapter(runtime, stop):
    hardware = FakeHardware()
    entered, release = threading.Event(), threading.Event()

    def factory():
        entered.set()
        assert release.wait(3)
        return hardware

    thread, outcome = background(lambda: runtime.connect_hardware(factory))
    assert entered.wait(1)
    try:
        getattr(runtime, stop)()
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert isinstance(outcome.get('error'), RuntimeError)
    assert hardware.closed is True and runtime.hardware is None
    assert runtime._hardware_connecting is False


@pytest.mark.parametrize('stop,counter', [('pause', 'slow_stops'), ('emergency_stop', 'estops')])
def test_failed_gripper_hold_still_attempts_robot_stop(runtime, stop, counter):
    worker = FakeWorker()
    hardware = runtime.hardware = FakeHardware(gripper=worker)
    runtime.state = 'running'
    worker.hold = Mock(side_effect=RuntimeError('gripper feedback unavailable'))
    with pytest.raises(RuntimeError, match='feedback unavailable'):
        getattr(runtime, stop)()
    assert getattr(hardware, counter) == 1
    assert runtime.state == ('error' if stop == 'pause' else 'estop')
    worker.hold = Mock()


@pytest.fixture
def http_service(tmp_path, runtime, monkeypatch):
    config = tmp_path / 'runtime.json'
    config.write_text(json.dumps(dict(checkpoint_directory=str(tmp_path / 'weights'),
        hardware_config='hardware.json', sampling_profile='161616',
        execution_mode='synchronous')), encoding='utf-8')
    catalog = tmp_path / 'catalog.json'
    catalog.write_text('{"checkpoints": []}', encoding='utf-8')
    (tmp_path / 'hardware.json').write_text('{}', encoding='utf-8')
    worker = FakeWorker()
    factory = Mock(return_value=worker)
    monkeypatch.setattr(DeploymentWeb, '_new_debug_gripper', lambda _self: factory())
    web = DeploymentWeb(runtime, catalog_path=catalog, config_path=config,
                        hardware_factory='fake:factory', port=0)
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

    yield web, worker, factory, request
    worker.close_error = None
    if worker.block is not None:
        worker.block.set()
    web.close()
    thread.join(2)
    assert not thread.is_alive()


def test_http_standalone_connection_commands_and_cached_get(http_service, runtime):
    web, worker, factory, request = http_service
    assert request('/api/gripper')[1]['connected'] is False
    code, result = request('/api/gripper', {'action': 'connect'})
    assert code == 200 and result['connected'] is True
    factory.assert_called_once_with()
    assert runtime.hardware is runtime.policy is None
    for action in ['enable', 'open', 'close', 'clear_error', 'disable', 'status']:
        code, result = request('/api/gripper', {'action': action})
        assert code == 200, result
        assert worker.calls[-1]['action'] == action
    code, result = request('/api/gripper',
                           {'action': 'move', 'opening': 0.3, 'speed': 20, 'torque': 25})
    assert code == 200 and result['opening'] == 0.3
    calls = list(worker.calls)
    for _ in range(3):
        assert request('/api/gripper')[1]['opening'] == 0.3
        assert request('/api/status')[1]['gripper_debug']['opening'] == 0.3
    assert worker.calls == calls, 'GET must never perform hardware commands'


@pytest.mark.parametrize('payload', [
    {}, [], {'action': True}, {'action': []}, {'action': 'wat'},
    {'action': 'open', 'extra': 1}, {'action': 'open', 'opening': 0.5},
    {'action': 'move', 'opening': True}, {'action': 'move', 'opening': '0.5'},
    {'action': 'move', 'opening': -1}, {'action': 'move', 'opening': math.nan},
    {'action': 'open', 'speed': True}, {'action': 'open', 'speed': 30.5},
    {'action': 'close', 'torque': 101}, {'action': 'enable', 'speed': 10},
])
def test_http_invalid_payload_never_reaches_worker(http_service, payload):
    _, worker, factory, request = http_service
    code, result = request('/api/gripper', payload)
    assert code == 400 and result.get('error')
    assert worker.calls == []
    factory.assert_not_called()


def test_http_cross_origin_control_is_rejected(http_service):
    _, worker, factory, request = http_service
    code, result = request('/api/gripper', {'action': 'connect'},
                           headers={'Origin': 'http://unrelated.invalid'})
    assert code == 403 and 'cross-origin' in result['error']
    factory.assert_not_called()
    assert worker.calls == []


def test_http_busy_blocks_start_config_load_and_full_connect(http_service, runtime, monkeypatch):
    web, worker, _, request = http_service
    assert request('/api/gripper', {'action': 'connect'})[0] == 200
    (web.store.root / 'dummy.pt').write_bytes(b'fake checkpoint')
    hardware_factory = Mock(return_value=FakeHardware())
    monkeypatch.setattr('realman_deploy.adapters.load_factory', hardware_factory)
    worker.block = threading.Event()
    worker.entered.clear()
    thread, outcome = background(lambda: request('/api/gripper', {'action': 'open'}))
    assert worker.entered.wait(1)
    old_config = web.config_path.read_bytes()
    try:
        for path, payload in [('/api/start', {}),
            ('/api/settings', {'sampling_profile': '4416', 'execution_mode': 'synchronous'}),
            ('/api/select', {'id': 'dummy'}), ('/api/connect', {}),
            ('/api/gripper', {'action': 'close'})]:
            code, result = request(path, payload)
            assert code == 400 and '正在进行' in result['error'], (path, result)
        assert web.config_path.read_bytes() == old_config
        assert runtime.policy is None
        hardware_factory.assert_not_called()
    finally:
        finish(thread, worker)
    assert outcome.get('result', (None,))[0] == 200


@pytest.mark.parametrize('endpoint,state', [('/api/pause', 'paused'), ('/api/estop', 'estop')])
def test_http_stop_preempts_blocked_operation(http_service, runtime, endpoint, state):
    _, worker, _, request = http_service
    request('/api/gripper', {'action': 'connect'})
    worker.block = threading.Event()
    worker.entered.clear()
    thread, outcome = background(lambda: request('/api/gripper', {'action': 'open'}))
    assert worker.entered.wait(1)
    try:
        code, result = request(endpoint, {})
        assert code == 200 and result['state'] == state
        assert worker.cancel_event.is_set()
    finally:
        finish(thread, worker)
    code, result = outcome['result']
    assert code == 400 and '取消' in result['error']
    assert runtime.state == state
