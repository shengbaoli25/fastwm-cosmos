"""Dashboard behavior tests; no SDK, camera, GPU or real inference server."""
from collections import deque
from copy import deepcopy
from dataclasses import replace
from http.client import HTTPConnection
from types import SimpleNamespace
import json
import threading
import time
import unittest
from unittest.mock import Mock, patch

from realman_sync_client import clock, webapp
from realman_sync_client.client import SynchronousClient
from realman_sync_client.mock import MockHardware
from realman_sync_client.wire import PROTOCOL, decode_request


META = dict(camera_names=['global', 'wrist'], width=8, height=6,
            action_space='joint8', action_horizon=32, history_length=32,
            control_hz=30, video_context_frames=1, video_context_stride=1,
            backend='test', force_preprocessing={'id':'test-force-contract'})
COSMOS_PROFILES = [dict(id=identifier, video_steps=video, force_steps=force, action_steps=action)
                   for identifier, video, force, action in
                   [('161616', 16, 16, 16), ('4416', 4, 4, 16), ('3310', 3, 3, 10)]]
DENSE_PROFILES = [dict(id='dense'+str(steps), video_steps=None, video_mode='single_forward',
                       force_steps=steps, action_steps=steps) for steps in (16, 10)]


def until(predicate, seconds=2):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('condition was not reached before timeout')


class GateTransport:
    """Each request waits for its own test-controlled completion event."""
    server_url = 'http://127.0.0.1:1'
    timeout_s = 1.0
    token = 'SECRET_SERVER_BEARER_MUST_NOT_APPEAR_IN_UI'

    def __init__(self):
        self.calls = []
        self.gates = []
        self.active = self.max_active = 0
        self.released = []
        self.lock = threading.Lock()
        self.fail = False
        self.metadata = deepcopy(META)
        self.sampling = None

    def health(self):
        result = dict(protocol=PROTOCOL, ready=True, metadata=deepcopy(self.metadata))
        if self.sampling is not None:
            result['sampling'] = deepcopy(self.sampling)
        return result

    def infer(self, request):
        decode_request(request, self.metadata)
        gate = threading.Event()
        with self.lock:
            self.calls.append(request)
            self.gates.append(gate)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if not gate.wait(3):
                raise TimeoutError('test completion gate timed out')
            if self.fail:
                raise RuntimeError('test inference failure')
            return dict(protocol=PROTOCOL, session_id=request['session_id'],
                        request_id=request['request_id'],
                        source_sequence=request['observation']['sequence'],
                        action_space='joint8', control_hz=30,
                        actions=[[.01]*7+[.4] for _ in range(32)], inference_ms=12.5)
        finally:
            with self.lock:
                self.active -= 1

    def release(self, session_id):
        self.released.append(session_id)


class Collector:
    def __init__(self, hardware, metadata):
        self.hardware, self.metadata = hardware, metadata
        self.condition = threading.RLock()
        self.history = deque(maxlen=32)
        self.closed = False
        self.fail = None
        self.before_snapshot = None
        self.snapshot_entered = threading.Event()

    def start(self):
        pass

    def snapshot_after(self, *_args, **_kwargs):
        self.snapshot_entered.set()
        if self.before_snapshot:
            self.before_snapshot()
        now = clock.now()
        with self.condition:
            self.history.clear()
            self.history.extend(self.hardware.observe_at(now-(31-i)/30) for i in range(32))
            return self.history[-1], tuple(self.history), (self.history[-1],)

    def check(self):
        if self.fail:
            raise RuntimeError(self.fail)

    def close(self):
        self.closed = True


class Client:
    instances = []
    collector_hook = None

    def __init__(self, transport, hardware, metadata, *, options):
        self.transport, self.hardware, self.metadata = transport, hardware, metadata
        self.options = options
        self.session_id = 'test-session-' + str(len(self.instances))
        self.stop_event = threading.Event()
        self.collector = Collector(hardware, metadata)
        if self.collector_hook:
            self.collector_hook(self.collector)
        self.instances.append(self)

    _request = SynchronousClient._request

    def _execute(self, actions):
        self.hardware.arm()
        for action in actions[:self.options.execute_steps]:
            if self.stop_event.is_set():
                raise InterruptedError('stop')
            self.hardware.command(action)
        self.hardware.slow_stop()
        return clock.now()


class DashboardTests(unittest.TestCase):
    def setUp(self):
        Client.instances = []
        Client.collector_hook = None
        self.transport = GateTransport()
        self.hardware = MockHardware(width=8, height=6,
            force_preprocessing_id=META['force_preprocessing']['id'],
            force_preprocessing=META['force_preprocessing'])
        self.dashboard = webapp.Dashboard(self.transport, mode='mock',
                                         hardware_factory=lambda meta: self.hardware)
        self.client_patch = patch.object(webapp, 'SynchronousClient', Client)
        self.client_patch.start()

    def tearDown(self):
        self.dashboard.stop_event.set()
        for gate in self.transport.gates:
            gate.set()
        self.dashboard.close()
        self.client_patch.stop()
        Client.collector_hook = None
        self.assertFalse(self.dashboard.thread and self.dashboard.thread.is_alive())

    def start_request(self):
        self.dashboard.start()
        until(lambda: len(self.transport.calls) == 1)

    def declare_sampling(self, profile='161616', profiles=COSMOS_PROFILES):
        selected = next(item for item in profiles if item['id'] == profile)
        self.transport.sampling = dict(deepcopy(selected), supported=True,
                                       sampling_profile=profile, profiles=deepcopy(profiles))
        self.dashboard.sampling = deepcopy(self.transport.sampling)

    def test_defaults_auto_infer_but_never_execute(self):
        self.start_request()
        self.assertFalse(self.dashboard.execute)
        self.transport.gates[0].set()
        until(lambda: self.dashboard.inference['requests_completed'] == 1)
        self.dashboard.control('inference_stop')
        self.assertEqual(self.hardware.commands, [])
        self.assertFalse(self.hardware.armed)
        self.assertEqual(self.dashboard.inference['inference_ms'], 12.5)

    def test_only_one_inference_is_in_flight(self):
        self.start_request()
        time.sleep(.06)
        self.assertEqual(len(self.transport.calls), 1)
        self.transport.gates[0].set()
        until(lambda: len(self.transport.calls) == 2)
        self.assertEqual(self.transport.max_active, 1)

    def test_stopping_inflight_drops_response(self):
        self.start_request()
        self.dashboard.control('inference_stop')
        self.transport.gates[0].set()
        until(lambda: self.dashboard.inference['phase'] == 'paused')
        self.assertEqual(self.dashboard.inference['requests_completed'], 0)
        self.assertEqual(self.hardware.commands, [])
        self.assertEqual(len(self.transport.calls), 1)

    def test_pause_closes_collector_and_resume_rebuilds_history_without_old_error(self):
        self.start_request()
        old_client = self.dashboard.client
        old_history = old_client.collector.history
        self.dashboard.control('inference_stop')
        self.transport.gates[0].set()
        until(lambda: self.dashboard.inference['phase'] == 'paused'
              and old_client.collector.closed)
        self.assertIsNone(self.dashboard.client)
        self.assertIs(self.dashboard.hardware, self.hardware)
        self.assertFalse(self.hardware.closed)
        # A prior collector's latched failure must not contaminate a resume.
        old_client.collector.fail = 'gripper feedback stale while paused'
        self.dashboard.control('inference_start')
        until(lambda: len(self.transport.calls) == 2)
        new_client = self.dashboard.client
        self.assertIsNot(new_client, old_client)
        self.assertIsNot(new_client.collector.history, old_history)
        self.assertEqual(len(new_client.collector.history), META['history_length'])
        self.assertIsNone(new_client.collector.fail)
        self.assertNotEqual(new_client.session_id, old_client.session_id)
        self.transport.gates[1].set()
        until(lambda: self.dashboard.inference['requests_completed'] == 1)
        self.dashboard.control('inference_stop')
        self.assertIsNone(self.dashboard.inference['last_error'])
        self.assertEqual(self.hardware.commands, [])

    def test_execution_toggle_requires_a_new_response(self):
        self.start_request()
        self.dashboard.control('execute_start')
        self.transport.gates[0].set()
        until(lambda: len(self.transport.calls) == 2)
        self.assertEqual(self.hardware.commands, [])
        self.dashboard.control('execute_stop')
        self.transport.gates[1].set()
        until(lambda: len(self.transport.calls) == 3)
        self.assertEqual(self.hardware.commands, [])
        self.assertEqual(self.dashboard.inference['requests_completed'], 0)

    def test_execution_is_possible_only_after_explicit_control(self):
        self.start_request()
        self.dashboard.control('execute_start')
        self.transport.gates[0].set()
        until(lambda: len(self.transport.calls) == 2)
        self.transport.gates[1].set()
        until(lambda: len(self.hardware.commands) == 32)
        self.dashboard.control('execute_stop')
        self.assertFalse(self.hardware.armed)

    def test_status_and_preview_do_not_sample_hardware(self):
        self.start_request()
        sequence = self.hardware.sequence
        for _ in range(3):
            status = self.dashboard.status()
            jpeg = self.dashboard.jpeg('global')
            self.assertEqual(jpeg[:2], b'\xff\xd8')
            self.assertEqual(status['sensors']['state']['samples'], sequence)
        self.assertEqual(self.hardware.sequence, sequence)

    def test_server_credentials_never_appear_in_status(self):
        self.start_request()
        self.assertNotIn(self.transport.token, json.dumps(self.dashboard.status()))

    def test_disconnect_closes_collector_hardware_and_releases_session(self):
        self.start_request()
        client = self.dashboard.client
        self.dashboard.control('disconnect')
        self.transport.gates[0].set()
        until(lambda: self.dashboard.phase == 'disconnected')
        self.assertTrue(client.collector.closed)
        self.assertTrue(self.hardware.closed)
        self.assertIn(client.session_id, self.transport.released)
        self.assertIsNone(self.dashboard.hardware)

    def test_http_error_disables_inference_and_execution(self):
        self.start_request()
        self.dashboard.control('execute_start')
        # The old generation is discarded after a user transition. Exercise
        # the generic failure on the new, active execution generation.
        self.transport.gates[0].set()
        until(lambda: len(self.transport.calls) == 2)
        self.transport.fail = True
        self.transport.gates[1].set()
        until(lambda: self.dashboard.inference['last_error'] is not None)
        self.assertFalse(self.dashboard.auto_infer)
        self.assertFalse(self.dashboard.execute)
        self.assertEqual(self.hardware.commands, [])
        self.assertTrue(Client.instances[0].collector.closed)

    def test_force_epoch_change_discards_response(self):
        self.start_request()
        with self.dashboard.client.collector.condition:
            collector = self.dashboard.client.collector
            old = collector.history[-1]
            collector.history[-1] = replace(old, force_meta=dict(old.force_meta, epoch=1))
        # Request contains an independent force_meta copy, so changing the
        # live epoch must invalidate it before execution.
        self.transport.gates[0].set()
        until(lambda: self.dashboard.inference['last_error'] is not None)
        self.assertEqual(self.hardware.commands, [])
        self.assertIn('力滤波器', self.dashboard.inference['last_error'])

    def test_invalid_execution_start_is_rejected(self):
        with self.assertRaises(ValueError):
            self.dashboard.control('execute_start')
        self.assertFalse(self.dashboard.execute)

    def test_sampling_waits_for_inflight_and_releases_session_before_change(self):
        self.check_sampling_waits_for_inflight('4416')

    def test_3310_waits_for_inflight_and_releases_session_before_change(self):
        self.check_sampling_waits_for_inflight('3310')

    def test_original_cosmos_profile_can_be_restored(self):
        self.check_sampling_waits_for_inflight('161616', initial_profile='3310')

    def test_dense_profile_uses_server_declaration_and_single_forward_metadata(self):
        self.check_sampling_waits_for_inflight('dense10', profiles=DENSE_PROFILES, initial_profile='dense16')
        sampling = self.dashboard.status()['sampling']
        self.assertEqual(sampling['video_mode'], 'single_forward')
        self.assertIsNone(sampling['video_steps'])
        self.assertEqual((sampling['force_steps'], sampling['action_steps']), (10, 10))
        self.assertEqual(sampling['profiles'], DENSE_PROFILES)

    def check_sampling_waits_for_inflight(self, target_profile, *, profiles=COSMOS_PROFILES, initial_profile='161616'):
        self.declare_sampling(initial_profile, profiles)
        self.start_request()
        old_client = self.dashboard.client
        self.dashboard.inference.update(actions=[[1]], latencies=[{'inference_ms': 1}])
        def change(profile):
            self.assertEqual(self.transport.active, 0)
            self.assertIn(old_client.session_id, self.transport.released)
            self.assertTrue(old_client.collector.closed)
            self.assertFalse(self.dashboard.auto_infer)
            selected = next(item for item in profiles if item['id'] == profile)
            self.transport.sampling = dict(deepcopy(selected), supported=True,
                                           sampling_profile=profile, profiles=deepcopy(profiles))
            return {'sampling': deepcopy(self.transport.sampling)}
        self.transport.set_sampling = Mock(side_effect=change)
        self.dashboard.control('sampling_set', sampling_profile=target_profile)
        self.transport.set_sampling.assert_not_called()
        self.transport.gates[0].set()
        until(lambda: self.dashboard.sampling.get('sampling_profile') == target_profile)
        self.assertFalse(self.dashboard.auto_infer)
        self.assertFalse(self.dashboard.execute)
        self.assertEqual(self.dashboard.inference['actions'], [])
        self.assertEqual(self.dashboard.inference['latencies'], [])
        self.assertEqual(self.dashboard.inference['requests_completed'], 0)
        self.assertEqual(self.hardware.commands, [])
        self.transport.set_sampling.assert_called_once_with(target_profile)

    def test_sampling_failure_stays_paused_and_retains_confirmed_profile(self):
        self.dashboard.auto_infer = False
        self.declare_sampling()
        self.transport.set_sampling = Mock(side_effect=RuntimeError('warmup failed'))
        self.dashboard.control('sampling_set', sampling_profile='4416')
        self.dashboard.start()
        until(lambda: self.dashboard.sampling.get('error'))
        self.assertEqual(self.dashboard.sampling['sampling_profile'], '161616')
        self.assertIn('warmup failed', self.dashboard.sampling['error'])
        self.assertFalse(self.dashboard.auto_infer)
        self.assertEqual(self.hardware.commands, [])

    def test_sampling_rejects_execution_and_cancelled_queue_never_runs(self):
        self.dashboard.hardware = self.hardware
        self.declare_sampling()
        self.transport.set_sampling = Mock()
        self.dashboard.execute = True
        with self.assertRaises(ValueError):
            self.dashboard.control('sampling_set', sampling_profile='4416')
        self.dashboard.execute = False
        with self.assertRaises(ValueError):
            self.dashboard.control('sampling_set', sampling_profile='garbage')
        self.dashboard.control('sampling_set', sampling_profile='4416')
        with self.assertRaises(ValueError):
            self.dashboard.control('inference_start')
        self.dashboard.control('disconnect')
        self.dashboard.start()
        until(lambda: self.dashboard.phase == 'disconnected')
        self.transport.set_sampling.assert_not_called()

    def test_sampling_accepts_any_explicitly_declared_string_id_without_client_allowlist(self):
        self.declare_sampling('vendor/custom-profile', [{'id': 'vendor/custom-profile'}])
        self.transport.set_sampling = Mock()
        self.dashboard.control('sampling_set', sampling_profile='vendor/custom-profile')
        self.assertEqual(self.dashboard._sampling_action, 'vendor/custom-profile')
        self.assertFalse(self.dashboard.auto_infer)
        self.transport.set_sampling.assert_not_called()

    def test_sampling_rejects_undeclared_ids_and_malformed_declarations_without_state_change(self):
        self.declare_sampling('dense16', DENSE_PROFILES)
        self.transport.set_sampling = Mock()
        for value in ('161616', '4416', '3310', 'garbage', None, 10, True, [], {}, ''):
            with self.subTest(profile=value):
                with self.assertRaisesRegex(ValueError, 'profiles'):
                    self.dashboard.control('sampling_set', sampling_profile=value)
                self.assertEqual(self.dashboard.generation, 0)
                self.assertTrue(self.dashboard.auto_infer)
                self.assertIsNone(self.dashboard._sampling_action)
        for malformed in (None, {}, {'dense10': {}}, 'dense10', [], [None, 'dense10', {'id': 10}, {'id': ''}, {'id': '   '}]):
            with self.subTest(declaration=malformed):
                self.dashboard.sampling = dict(supported=True, sampling_profile='dense10', profiles=malformed)
                with self.assertRaisesRegex(ValueError, 'profiles'):
                    self.dashboard.control('sampling_set', sampling_profile='dense10')
        self.transport.set_sampling.assert_not_called()

    def test_sampling_revoked_while_queued_is_not_dispatched(self):
        self.declare_sampling('dense16', DENSE_PROFILES)
        self.transport.set_sampling = Mock()
        self.start_request()
        self.dashboard.control('sampling_set', sampling_profile='dense10')
        self.declare_sampling('dense16', DENSE_PROFILES[:1])
        self.transport.gates[0].set()
        until(lambda: bool(self.dashboard.sampling.get('error')))
        self.assertIn('不再声明', self.dashboard.sampling['error'])
        self.assertFalse(self.dashboard.execute)
        self.assertFalse(self.dashboard.auto_infer)
        self.transport.set_sampling.assert_not_called()

    def test_health_without_sampling_clears_previous_server_declarations(self):
        self.declare_sampling()
        self.transport.sampling = None
        self.dashboard.auto_infer = False
        self.dashboard.start()
        until(lambda: self.dashboard.server['ready'])
        self.assertFalse(self.dashboard.sampling['supported'])
        self.assertEqual(self.dashboard.sampling['profiles'], [])
        self.transport.set_sampling = Mock()
        with self.assertRaises(ValueError):
            self.dashboard.control('sampling_set', sampling_profile='4416')
        self.transport.set_sampling.assert_not_called()

    def test_health_can_read_sampling_declarations_nested_in_metadata(self):
        self.declare_sampling('dense16', DENSE_PROFILES)
        self.transport.metadata['sampling'] = self.transport.sampling
        self.transport.sampling = None
        self.dashboard.sampling = {}
        self.dashboard.auto_infer = False
        self.dashboard.start()
        until(lambda: self.dashboard.server['ready'])
        self.assertEqual(self.dashboard.sampling['profiles'], DENSE_PROFILES)
        self.assertIsNone(self.dashboard.sampling['video_steps'])

    def test_manual_gripper_drops_inflight_action_and_runs_once_on_worker(self):
        self.hardware.gripper_diagnostics = lambda: {'motion_available': True}
        caller = threading.get_ident()
        def manual(operation):
            self.assertNotEqual(caller, threading.get_ident())
            self.assertIsNone(self.dashboard.client)
            self.assertEqual(self.transport.active, 0)
            return {'operation': operation, 'confirmed': False}
        self.hardware.manual_gripper = Mock(side_effect=manual)
        self.start_request()
        self.dashboard.control('gripper_close')
        self.hardware.manual_gripper.assert_not_called()
        self.transport.gates[0].set()
        until(lambda: self.hardware.manual_gripper.call_count == 1)
        self.hardware.manual_gripper.assert_called_once_with('close')
        self.assertFalse(self.dashboard.auto_infer)
        self.assertFalse(self.dashboard.execute)
        self.assertEqual(self.hardware.commands, [])

    def test_manual_gripper_guards_are_checked_before_queuing(self):
        self.dashboard.hardware = self.hardware
        self.hardware.manual_gripper = Mock()
        self.hardware.gripper_diagnostics = lambda: {'enable_available': True}
        with self.assertRaises(ValueError):
            self.dashboard.control('gripper_open')
        self.dashboard.execute = True
        with self.assertRaises(ValueError):
            self.dashboard.control('gripper_enable')
        self.hardware.manual_gripper.assert_not_called()
        self.assertIsNone(self.dashboard._hardware_action)

    def test_manual_open_busy_blocks_execution_and_timeout_keeps_everything_paused(self):
        entered, finish = threading.Event(), threading.Event()
        self.hardware.gripper_diagnostics = lambda: {'motion_available': True}
        def manual(operation):
            self.assertEqual(operation, 'open')
            self.assertIsNone(self.dashboard.client)
            entered.set()
            if not finish.wait(1):
                raise RuntimeError('test manual opening was not released')
            raise TimeoutError('manual opening cancelled after timeout')
        self.hardware.manual_gripper = Mock(side_effect=manual)
        self.start_request()
        self.dashboard.control('gripper_open')
        self.transport.gates[0].set()
        try:
            self.assertTrue(entered.wait(1))
            self.assertTrue(self.dashboard._hardware_busy)
            for action in ('execute_start', 'inference_start', 'gripper_open', 'connect'):
                with self.subTest(action=action), self.assertRaises(ValueError):
                    self.dashboard.control(action)
            self.assertFalse(self.dashboard.execute)
            self.assertFalse(self.dashboard.auto_infer)
            self.assertEqual(self.hardware.commands, [])
        finally:
            finish.set()
        until(lambda: not self.dashboard._hardware_busy)
        until(lambda: any('manual opening cancelled after timeout' in row['message']
                         for row in self.dashboard.events))
        self.hardware.manual_gripper.assert_called_once_with('open')
        self.assertFalse(self.dashboard.execute)
        self.assertFalse(self.dashboard.auto_infer)
        self.assertEqual(self.hardware.commands, [])

    def test_disconnect_during_session_release_cancels_manual_dispatch(self):
        entered, finish = threading.Event(), threading.Event()
        self.hardware.gripper_diagnostics = lambda: {'enable_available': True}
        self.hardware.manual_gripper = Mock()
        def release(_session):
            entered.set()
            finish.wait(1)
        self.transport.release = release
        self.start_request()
        self.dashboard.control('gripper_enable')
        self.transport.gates[0].set()
        self.assertTrue(entered.wait(1))
        self.dashboard.control('disconnect')
        finish.set()
        until(lambda: self.dashboard.phase == 'disconnected')
        self.hardware.manual_gripper.assert_not_called()
        self.assertEqual(self.hardware.commands, [])

    def test_emergency_during_session_release_cancels_sampling_dispatch(self):
        entered, finish = threading.Event(), threading.Event()
        self.declare_sampling()
        self.transport.set_sampling = Mock()
        def release(_session):
            entered.set()
            finish.wait(1)
        self.transport.release = release
        self.start_request()
        self.dashboard.control('sampling_set', sampling_profile='4416')
        self.transport.gates[0].set()
        self.assertTrue(entered.wait(1))
        self.dashboard.control('emergency_stop')
        finish.set()
        until(lambda: self.dashboard.inference['phase'] == 'paused')
        self.transport.set_sampling.assert_not_called()

    def test_gripper_clear_rejects_unknown_or_unsafe_conditions_without_writes(self):
        self.dashboard.hardware = self.hardware
        self.hardware.clear_gripper_error = Mock()
        self.hardware.gripper_diagnostics = Mock()
        for label, diag, execute, armed in (
            ('unknown status', {}, False, False),
            ('blocked status', {'clear_available': False, 'clear_reason': 'unknown bus'}, False, False),
            ('execution enabled', {'clear_available': True}, True, False),
            ('hardware armed', {'clear_available': True}, False, True),
            ('emergency latched', {'clear_available': False, 'clear_reason': 'Emergency stop is latched'}, False, False),
        ):
            with self.subTest(label=label):
                self.hardware.gripper_diagnostics.return_value = diag
                self.dashboard.execute = execute
                self.hardware.armed = armed
                with self.assertRaises(ValueError):
                    self.dashboard.control('gripper_clear_error')
                self.assertIsNone(self.dashboard._hardware_action)
                self.assertEqual(self.dashboard.generation, 0)
                self.hardware.clear_gripper_error.assert_not_called()
        self.hardware.armed = self.dashboard.execute = False

    def test_gripper_clear_does_not_queue_when_hardware_refuses_diagnostics(self):
        self.dashboard.hardware = self.hardware
        self.hardware.clear_gripper_error = Mock()
        self.hardware.gripper_diagnostics = Mock(side_effect=RuntimeError('Emergency stop is latched'))
        with self.assertRaisesRegex(RuntimeError, 'Emergency stop is latched'):
            self.dashboard.control('gripper_clear_error')
        self.assertIsNone(self.dashboard._hardware_action)
        self.hardware.clear_gripper_error.assert_not_called()

    def test_disconnect_and_emergency_cancel_queued_gripper_clear(self):
        for stop_action in ('disconnect', 'emergency_stop'):
            with self.subTest(action=stop_action):
                transport = GateTransport()
                hardware = MockHardware(width=8, height=6,
                    force_preprocessing_id=META['force_preprocessing']['id'],
                    force_preprocessing=META['force_preprocessing'])
                hardware.gripper_diagnostics = lambda: {'clear_available': True}
                hardware.clear_gripper_error = Mock()
                dashboard = webapp.Dashboard(transport, mode='mock',
                    hardware_factory=lambda meta: hardware)
                dashboard.hardware = hardware
                dashboard.control('gripper_clear_error')
                dashboard.control(stop_action)
                self.assertIsNone(dashboard._hardware_action)
                dashboard.start()
                try:
                    if stop_action == 'disconnect':
                        until(lambda: dashboard.phase == 'disconnected')
                    else:
                        until(lambda: dashboard.inference['phase'] == 'paused')
                    hardware.clear_gripper_error.assert_not_called()
                    self.assertFalse(dashboard.auto_infer)
                    self.assertFalse(dashboard.execute)
                    self.assertEqual(hardware.commands, [])
                    self.assertEqual(len(transport.calls), 0)
                finally:
                    dashboard.close()

    def test_gripper_operation_runs_on_worker_after_collector_closes(self):
        entered, finish = threading.Event(), threading.Event()
        call_thread = []
        self.hardware.gripper_diagnostics = lambda: {'clear_available': True}
        def clear():
            call_thread.append(threading.get_ident())
            self.assertTrue(Client.instances[0].collector.closed)
            self.assertIsNone(self.dashboard.client)
            entered.set()
            if not finish.wait(1):
                raise RuntimeError('test did not release hardware operation')
            return {'cleared': True}
        self.hardware.clear_gripper_error = clear
        self.start_request()
        result = self.dashboard.control('gripper_clear_error')
        self.assertIn('message', result)
        self.assertFalse(self.dashboard.auto_infer)
        self.assertFalse(self.dashboard.execute)
        self.assertEqual(call_thread, [])
        for action in ('inference_start', 'execute_start', 'connect', 'gripper_clear_error'):
            with self.subTest(queued_action=action), self.assertRaises(ValueError):
                self.dashboard.control(action)
        self.transport.gates[0].set()
        try:
            self.assertTrue(entered.wait(1))
            self.assertEqual(call_thread, [self.dashboard.thread.ident])
            for action in ('inference_start', 'execute_start', 'connect', 'gripper_clear_error'):
                with self.subTest(busy_action=action), self.assertRaises(ValueError):
                    self.dashboard.control(action)
            self.assertEqual(self.hardware.commands, [])
        finally:
            finish.set()
        until(lambda: any('夹爪操作结果' in row['message'] for row in self.dashboard.events))
        self.assertFalse(self.dashboard.auto_infer)
        self.assertFalse(self.dashboard.execute)

    def test_gripper_operation_error_is_reported_and_keeps_inference_off(self):
        self.hardware.diagnose_gripper = Mock(side_effect=RuntimeError('gripper read failed'))
        self.start_request()
        collector = self.dashboard.client.collector
        self.dashboard.control('gripper_diagnose')
        self.hardware.diagnose_gripper.assert_not_called()
        self.transport.gates[0].set()
        until(lambda: any('gripper read failed' in row['message'] for row in self.dashboard.events))
        event = next(row for row in self.dashboard.events if 'gripper read failed' in row['message'])
        until(lambda: not self.dashboard._hardware_busy)
        self.assertEqual(event['level'], 'error')
        self.assertTrue(collector.closed)
        self.assertFalse(self.dashboard.auto_infer)
        self.assertFalse(self.dashboard.execute)
        self.assertFalse(self.dashboard._hardware_busy)
        self.assertEqual(self.hardware.commands, [])
        self.hardware.diagnose_gripper.assert_called_once_with()

    def test_emergency_latch_blocks_immediate_reenable_while_inflight(self):
        self.hardware.execution_available = True
        def latch():
            self.hardware.execution_available = False
            self.hardware.execution_reason = 'Emergency stop is latched'
            self.hardware.slow_stop()
        self.hardware.emergency_stop = latch
        self.start_request()
        self.dashboard.control('execute_start')
        self.dashboard.control('emergency_stop')
        self.assertFalse(self.dashboard.execute)
        self.dashboard.control('inference_start')
        with self.assertRaises(ValueError):
            self.dashboard.control('execute_start')
        self.assertFalse(self.dashboard.execute)
        self.assertEqual(self.hardware.commands, [])

    def test_outer_worker_failure_clears_all_control_flags_and_closes_devices(self):
        self.start_request()
        self.dashboard.control('execute_start')
        def fatal():
            raise RuntimeError('fatal readiness failure')
        self.dashboard._ready = fatal
        self.transport.gates[0].set()
        until(lambda: not self.dashboard.thread.is_alive())
        state = self.dashboard.status()
        self.assertEqual(state['phase'], 'error')
        self.assertFalse(state['execution']['enabled'])
        self.assertFalse(state['execution']['available'])
        self.assertFalse(state['inference']['enabled'])
        self.assertTrue(self.hardware.closed)
        self.assertTrue(Client.instances[0].collector.closed)

    def test_failed_health_after_inflight_clears_execute(self):
        self.start_request()
        self.dashboard.control('execute_start')
        def offline():
            raise ConnectionError('server offline')
        self.transport.health = offline
        # A clock advance makes the next loop run its ordinary 2-second
        # health check without adding real sleep to the suite.
        real_now = clock.now
        with patch.object(clock, 'now', side_effect=lambda: real_now() + 3):
            self.transport.gates[0].set()
            until(lambda: not self.dashboard.server['ready'])
            self.assertFalse(self.dashboard.execute)
            self.assertFalse(self.dashboard.execution_available)
            self.assertEqual(self.hardware.commands, [])

    def test_paused_session_released_only_once(self):
        self.start_request()
        self.dashboard.control('inference_stop')
        self.transport.gates[0].set()
        until(lambda: self.dashboard.inference['phase'] == 'paused')
        time.sleep(.27)
        self.assertEqual(self.transport.released, [Client.instances[0].session_id])

    def test_stop_during_production_execute_cancels_remaining_actions(self):
        waiting = threading.Event()
        def wait_control(client, _duration):
            waiting.set()
            if not client.stop_event.wait(1):
                raise AssertionError('test did not send stop')
            raise InterruptedError('client stop requested')
        # Exercise production chunk execution with a deterministic command
        # slot wait; only hardware and time waiting are replaced.
        with patch.object(Client, '_execute', SynchronousClient._execute), \
             patch.object(Client, '_wait_control', wait_control, create=True):
            self.start_request()
            self.dashboard.control('execute_start')
            self.transport.gates[0].set()
            until(lambda: len(self.transport.calls) == 2)
            self.transport.gates[1].set()
            self.assertTrue(waiting.wait(1))
            self.dashboard.control('execute_stop')
            until(lambda: len(self.transport.calls) == 3)
            self.assertEqual(len(self.hardware.commands), 1)
            self.assertFalse(self.hardware.armed)

    def test_pause_during_history_wait_does_not_send_new_request(self):
        entered, release = threading.Event(), threading.Event()
        def install(_self, collector):
            def block():
                entered.set()
                release.wait(1)
            collector.before_snapshot = block
        Client.collector_hook = install
        self.dashboard.start()
        self.assertTrue(entered.wait(1))
        self.dashboard.control('inference_stop')
        release.set()
        until(lambda: self.dashboard.inference['phase'] == 'paused' or bool(self.transport.calls))
        self.assertEqual(len(self.transport.calls), 0, 'Pause must be rechecked after history waiting')


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.transport = GateTransport()
        self.dashboard = webapp.Dashboard(self.transport, mode='mock')
        self.server = webapp.DashboardServer(('127.0.0.1', 0), self.dashboard)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval':.01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(1)

    def request(self, method, path, body=None, headers=None):
        conn = HTTPConnection('127.0.0.1', self.port, timeout=2)
        try:
            # Send the complete small fixture request together. On Windows a
            # header-only rejection can close the socket before http.client's
            # separate body send, resetting it before we can read the status.
            # Keep the real HTTP encoder and all response/error assertions.
            outgoing = []
            with patch.object(conn, 'send', side_effect=outgoing.append):
                conn.request(method, path, body=body, headers=headers or {})
            conn.send(b''.join(outgoing))
            reply = conn.getresponse()
            return reply.status, reply.read(), dict(reply.getheaders())
        finally:
            conn.close()

    def headers(self, **changes):
        values = {'Content-Type':'application/json', 'X-Client-Key':self.server.client_key,
                  'Origin':f'http://127.0.0.1:{self.port}'}
        values.update(changes)
        return values

    def test_status_is_available_without_exposing_server_token(self):
        code, body, headers = self.request('GET','/api/status')
        self.assertEqual(code,200)
        self.assertNotIn(self.transport.token.encode(),body)
        self.assertNotIn('Access-Control-Allow-Origin',headers)
        self.assertEqual(headers['Cache-Control'],'no-store')

    def test_wrong_host_cannot_read_status_or_control(self):
        code, _, _ = self.request('GET','/api/status',headers={'Host':'attacker.example'})
        self.assertEqual(code,403)
        code, _, _ = self.request('POST','/api/control',json.dumps({'action':'disconnect'}),
                                  self.headers(Host='attacker.example'))
        self.assertEqual(code,403)
        self.assertTrue(self.dashboard.want_connect)

    def test_cross_origin_and_missing_key_cannot_control(self):
        for headers in (self.headers(Origin='http://attacker.example'),
                        self.headers(**{'X-Client-Key':''}),self.headers(Origin='null')):
            code, _, _ = self.request('POST','/api/control',json.dumps({'action':'disconnect'}),headers)
            self.assertEqual(code,403)
            self.assertTrue(self.dashboard.want_connect)

    def test_json_same_origin_control_succeeds(self):
        code, body, _ = self.request('POST','/api/control',json.dumps({'action':'disconnect'}),self.headers())
        self.assertEqual(code,200)
        self.assertTrue(json.loads(body)['ok'])
        self.assertFalse(self.dashboard.want_connect)

    def test_sampling_http_accepts_only_ids_declared_by_current_server(self):
        self.dashboard.sampling = dict(supported=True, sampling_profile='dense16',
                                       profiles=deepcopy(DENSE_PROFILES))
        self.transport.set_sampling = Mock()
        for identifier in ('4416', '3310', 'undeclared', None, {}, 10):
            with self.subTest(identifier=identifier):
                code, body, _ = self.request('POST', '/api/control', json.dumps(
                    dict(action='sampling_set', sampling_profile=identifier)), self.headers())
                self.assertEqual(code, 409)
                self.assertIn('profiles', json.loads(body)['error'])
                self.assertIsNone(self.dashboard._sampling_action)
        code, _, _ = self.request('POST', '/api/control', json.dumps(
            dict(action='sampling_set', sampling_profile='dense10')), self.headers())
        self.assertEqual(code, 200)
        self.assertEqual(self.dashboard._sampling_action, 'dense10')
        self.assertFalse(self.dashboard.auto_infer)
        self.assertFalse(self.dashboard.execute)
        self.transport.set_sampling.assert_not_called()

    def test_gripper_clear_http_only_queues_and_does_not_write_hardware(self):
        hardware = Mock(armed=False)
        hardware.gripper_diagnostics.return_value = {'clear_available': True}
        self.dashboard.hardware = hardware
        code, body, _ = self.request('POST', '/api/control',
            json.dumps({'action': 'gripper_clear_error'}), self.headers())
        self.assertEqual(code, 200)
        self.assertTrue(json.loads(body)['ok'])
        self.assertEqual(self.dashboard._hardware_action, 'clear_gripper_error')
        self.assertFalse(self.dashboard.auto_infer)
        self.assertFalse(self.dashboard.execute)
        hardware.clear_gripper_error.assert_not_called()
        hardware.command.assert_not_called()

    def test_form_content_type_is_rejected(self):
        code, _, _ = self.request('POST','/api/control','action=disconnect',
                                 self.headers(**{'Content-Type':'application/x-www-form-urlencoded'}))
        self.assertEqual(code,400)
        self.assertTrue(self.dashboard.want_connect)

    def test_http_stop_cancels_unarmed_manual_preparation_and_ramp_without_native_writes(self):
        for action in ('execute_stop', 'inference_stop', 'disconnect'):
            for phase in ('preparing', 'ramping', 'queued'):
                with self.subTest(action=action, phase=phase):
                    entered, cancelled = threading.Event(), threading.Event()
                    late_targets = []
                    gripper = SimpleNamespace(desired=1., hold=Mock(), command=Mock(), set_enabled=Mock())
                    def hold():
                        gripper.desired = None
                        cancelled.set()
                    gripper.hold.side_effect = hold
                    hardware = SimpleNamespace(armed=False, gripper=gripper,
                        slow_stop=Mock(), command=Mock(), manual_gripper=Mock())
                    self.dashboard.hardware = hardware
                    self.dashboard.want_connect = True
                    self.dashboard.execute = self.dashboard.auto_infer = False
                    self.dashboard._hardware_busy = phase == 'preparing'
                    self.dashboard._hardware_operation = 'gripper_open' if phase == 'preparing' else None
                    self.dashboard._hardware_action = 'gripper_open' if phase == 'queued' else None
                    def preparation_wait():
                        entered.set()
                        if not cancelled.wait(1):
                            late_targets.append('would enqueue after preparation')
                    pending = threading.Thread(target=preparation_wait)
                    pending.start()
                    try:
                        self.assertTrue(entered.wait(.5))
                        code, _, _ = self.request('POST', '/api/control', json.dumps({'action':action}), self.headers())
                        self.assertEqual(code, 200)
                        pending.join(.5)
                        self.assertFalse(pending.is_alive())
                        gripper.hold.assert_called_once()
                        self.assertIsNone(gripper.desired)
                        self.assertIsNone(self.dashboard._hardware_action)
                        self.assertFalse(self.dashboard.execute)
                        self.assertEqual(late_targets, [])
                        hardware.slow_stop.assert_not_called()
                        hardware.command.assert_not_called()
                        hardware.manual_gripper.assert_not_called()
                        gripper.command.assert_not_called()
                        gripper.set_enabled.assert_not_called()
                    finally:
                        cancelled.set()
                        pending.join(1)
                        self.dashboard._hardware_busy = False
                        self.dashboard._hardware_operation = None

    def test_bad_control_is_rejected_without_state_change(self):
        for body in ('not json', json.dumps({'action':'shutdown_system'}), json.dumps({'action':{}})):
            code, _, _ = self.request('POST','/api/control',body,self.headers())
            self.assertEqual(code,409)
            self.assertTrue(self.dashboard.want_connect)

    def test_oversized_control_rejected(self):
        code, _, _ = self.request('POST','/api/control',' '*4097,self.headers())
        self.assertEqual(code,409)
        self.assertTrue(self.dashboard.want_connect)

    def test_unknown_camera_and_asset_are_not_exposed(self):
        for path in ('/api/camera/other.jpg','/server-token.txt','/../config/server-token.txt'):
            self.assertEqual(self.request('GET',path)[0],404)

    def test_dashboard_cannot_bind_public_interface(self):
        with self.assertRaises(ValueError):
            webapp.DashboardServer(('0.0.0.0',0), self.dashboard)


if __name__ == '__main__':
    unittest.main()
