"""Recovery integration/race tests: fake sensors and gated inference only."""
from io import BytesIO
import json
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from realman_force.contract import source_at
from realman_sync_client import webapp
from realman_sync_client.http import InferenceHTTPClient
from realman_sync_client.mock import MockHardware
from realman_sync_client.recovery import transient_failure
from realman_sync_client.wire import PROTOCOL
from test_webapp import Client, GateTransport, META, until


TRANSIENT = '关节目标未写入：夹爪通信曾中断；反馈恢复后请重新点击开始执行'
FORCE_GAP = 'No fresh force sample at or before observation grid time'


class RecoveryHardware(MockHardware):
    """Opaque-token API double, without a robot SDK or transport connection."""
    def __init__(self):
        super().__init__(width=8, height=6,
                         force_preprocessing_id=META['force_preprocessing']['id'],
                         force_preprocessing=META['force_preprocessing'])
        self.feedback_ready = True
        self.ready_since = time.monotonic()
        self.execution_available = True
        self.calls = []
        self.token = None
        self.reject_recovery = False
        self.recovery_ack_times = []
        self.command_error_once = self.stop_error_once = None

    def set_feedback_ready(self, ready):
        self.feedback_ready = ready
        self.ready_since = time.monotonic() if ready else None

    def can_infer(self, _metadata):
        return self.feedback_ready, None if self.feedback_ready else '夹爪运动反馈过期'

    def acknowledge_gripper_faults(self):
        self.calls.append('explicit_ack')
        if not self.feedback_ready:
            raise RuntimeError('夹爪反馈尚未恢复，不能恢复动作执行')

    def capture_feedback_recovery_token(self):
        self.calls.append('capture')
        self.token = object()
        return self.token

    def acknowledge_recovered_feedback(self, token):
        self.calls.append('recovery_ack')
        self.recovery_ack_times.append(time.monotonic())
        return self.feedback_ready and not self.reject_recovery and token is self.token

    def invalidate_feedback_recovery(self):
        self.calls.append('invalidate')
        self.token = None

    def command(self, action):
        if self.command_error_once:
            message, self.command_error_once = self.command_error_once, None
            self.set_feedback_ready(False)
            raise RuntimeError(message)
        return super().command(action)

    def slow_stop(self):
        if self.stop_error_once:
            message, self.stop_error_once = self.stop_error_once, None
            raise RuntimeError(message)
        # A normal chunk completion does not revoke explicit execution intent.
        return super().slow_stop()


class RecoveryTransport(GateTransport):
    def __init__(self):
        super().__init__()
        self.invalid_response_once = False

    def infer(self, request):
        response = super().infer(request)
        if self.invalid_response_once:
            self.invalid_response_once = False
            response['actions'][0][0] = float('nan')
        return response


class DashboardRecoveryTests(unittest.TestCase):
    def setUp(self):
        Client.instances = []
        Client.collector_hook = None
        self.fixtures = []
        self.client_patch = patch.object(webapp, 'SynchronousClient', Client)
        self.stable_patch = patch.object(webapp.Dashboard, 'RECOVERY_STABLE_S', .06)
        self.timeout_patch = patch.object(webapp.Dashboard, 'RECOVERY_TIMEOUT_S', 1.2)
        for active_patch in (self.client_patch, self.stable_patch, self.timeout_patch):
            active_patch.start()
            self.addCleanup(active_patch.stop)
        self.transport, self.hardware, self.dashboard = self.make_fixture()

    def make_fixture(self):
        transport, hardware = RecoveryTransport(), RecoveryHardware()
        dashboard = webapp.Dashboard(transport, mode='mock', hardware_factory=lambda _meta: hardware)
        self.fixtures.append((transport, hardware, dashboard))
        return transport, hardware, dashboard

    def close_fixture(self, fixture):
        transport, _hardware, dashboard = fixture
        dashboard.stop_event.set()
        for gate in transport.gates:
            gate.set()
        dashboard.close()
        self.assertFalse(dashboard.thread and dashboard.thread.is_alive())

    def tearDown(self):
        for fixture in self.fixtures:
            self.close_fixture(fixture)
        Client.collector_hook = None

    def recovery(self):
        return self.dashboard.status()['recovery']

    def start_execution_request(self):
        self.dashboard.start()
        until(lambda: len(self.transport.calls) == 1)
        self.dashboard.control('execute_start')
        self.transport.gates[0].set()
        until(lambda: len(self.transport.calls) == 2)
        self.assertEqual([call for call in self.hardware.calls if call != 'invalidate'],
                         ['explicit_ack', 'capture'])
        return self.dashboard.client

    def enter_execution_recovery(self):
        old_client = self.start_execution_request()
        self.hardware.command_error_once = TRANSIENT
        self.transport.gates[1].set()
        until(lambda: self.recovery()['active'])
        return old_client

    def test_transient_execution_recovers_with_new_session_and_never_replays_old_action(self):
        old_client = self.enter_execution_recovery()
        state = self.dashboard.status()
        self.assertTrue(state['inference']['enabled'])
        self.assertFalse(state['execution']['enabled'])
        self.assertTrue(state['recovery']['resume_execution'])
        self.assertIn('通信', state['recovery']['reason'])
        self.assertEqual(self.hardware.commands, [])
        until(lambda: old_client.collector.closed)
        self.assertIn(old_client.session_id, self.transport.released)
        self.hardware.set_feedback_ready(True)
        until(lambda: len(self.transport.calls) == 3)
        new_client = self.dashboard.client
        self.assertIsNot(new_client, old_client)
        self.assertIsNot(new_client.collector.history, old_client.collector.history)
        self.assertNotEqual(new_client.session_id, old_client.session_id)
        self.assertGreater(self.transport.calls[2]['observation']['sequence'],
                           self.transport.calls[1]['observation']['sequence'])
        self.assertFalse(self.recovery()['active'])
        self.assertIsNotNone(self.recovery()['last_recovered_at'])
        self.assertTrue(self.dashboard.execute)
        self.assertEqual(self.hardware.commands, [], 'Old action must not be replayed at recovery')
        self.assertIsNotNone(self.dashboard._recovery_deadline,
                             'Ready feedback alone must not reset the recovery budget')
        self.assertGreaterEqual(self.hardware.recovery_ack_times[0] - self.hardware.ready_since, .05)
        self.transport.gates[2].set()
        until(lambda: len(self.hardware.commands) == 32)
        until(lambda: self.dashboard._recovery_deadline is None)
        self.assertEqual(self.hardware.calls.count('capture'), 1)
        self.assertEqual(self.hardware.calls.count('explicit_ack'), 1)
        self.assertIsNotNone(self.hardware.token, 'Normal chunk slow_stop must preserve the token')

    def test_inference_only_recovery_does_not_enable_actions_or_capture_execution_token(self):
        self.dashboard.start()
        until(lambda: len(self.transport.calls) == 1)
        old_client = self.dashboard.client
        old_client.collector.fail = 'sensor collection failed: 夹爪运动反馈过期'
        self.hardware.set_feedback_ready(False)
        self.transport.gates[0].set()
        until(lambda: self.recovery()['active'])
        self.assertFalse(self.recovery()['resume_execution'])
        self.hardware.set_feedback_ready(True)
        until(lambda: len(self.transport.calls) == 2)
        self.assertFalse(self.dashboard.execute)
        self.assertTrue(self.dashboard.auto_infer)
        self.assertNotEqual(old_client.session_id, self.dashboard.client.session_id)
        self.assertNotIn('capture', self.hardware.calls)
        self.transport.gates[1].set()
        until(lambda: self.dashboard.inference['requests_completed'] == 1)
        self.assertEqual(self.hardware.commands, [])

    def test_transport_rebuild_during_recovery_resumes_only_inference(self):
        self.enter_execution_recovery()
        self.hardware.recovery_requires_execution_restart = lambda _token: True
        self.hardware.set_feedback_ready(True)
        until(lambda: len(self.transport.calls) == 3)
        self.assertFalse(self.dashboard.execute)
        self.assertFalse(self.recovery()['resume_execution'])
        self.assertTrue(self.dashboard.auto_infer)
        self.assertIsNone(self.dashboard._execution_recovery_token)
        self.transport.gates[2].set()
        until(lambda: self.dashboard.inference['requests_completed'] >= 1)
        self.assertEqual(self.hardware.commands, [])
        self.assertEqual(self.hardware.recovery_ack_times, [])

    def test_transport_already_rebuilt_at_failure_does_not_preserve_execution_intent(self):
        old_client = self.start_execution_request()
        self.hardware.recovery_requires_execution_restart = lambda _token: True
        old_client.collector.fail = '夹爪通信自动重连中'
        self.transport.gates[1].set()
        until(lambda: self.recovery()['active'])
        self.assertFalse(self.recovery()['resume_execution'])
        until(lambda: len(self.transport.calls) == 3)
        self.assertFalse(self.dashboard.execute)
        self.assertEqual(self.hardware.commands, [])

    def test_user_stop_cancels_pending_execution_recovery(self):
        self.enter_execution_recovery()
        invalidations = self.hardware.calls.count('invalidate')
        self.dashboard.control('execute_stop')
        self.assertFalse(self.recovery()['active'])
        self.assertGreater(self.hardware.calls.count('invalidate'), invalidations)
        self.hardware.set_feedback_ready(True)
        until(lambda: len(self.transport.calls) == 3)
        self.transport.gates[2].set()
        until(lambda: len(self.transport.calls) == 4)
        self.assertFalse(self.dashboard.execute)
        self.assertFalse(self.recovery()['active'])
        self.assertEqual(self.hardware.commands, [])
        self.assertNotIn('recovery_ack', self.hardware.calls)

    def test_observation_feedback_gap_rebuilds_session_without_enabling_motion(self):
        # Exercise the real force selector: accepting an old/future sample is
        # not the recovery mechanism. The failed cycle must be discarded.
        with self.assertRaisesRegex(ValueError, FORCE_GAP) as caught:
            source_at([{'time': 99.96}, {'time': 100.01}], 100.)
        messages = (str(caught.exception), '夹爪观测反馈过期', '夹爪反馈与机械臂状态不同步')
        for message in messages:
            with self.subTest(error=message):
                transport, hardware, dashboard = self.make_fixture()
                dashboard.start()
                until(lambda: len(transport.calls) == 1)
                old_client = dashboard.client
                old_client.collector.fail = 'sensor collection failed: ' + message
                hardware.set_feedback_ready(False)
                transport.gates[0].set()
                until(lambda: dashboard.status()['recovery']['active'])
                until(lambda: old_client.collector.closed)
                self.assertTrue(dashboard.auto_infer)
                self.assertFalse(dashboard.status()['recovery']['resume_execution'])
                self.assertEqual(dashboard.inference['requests_completed'], 0)
                hardware.set_feedback_ready(True)
                until(lambda: len(transport.calls) == 2)
                self.assertNotEqual(old_client.session_id, dashboard.client.session_id)
                self.assertIn(old_client.session_id, transport.released)
                self.assertGreater(transport.calls[1]['observation']['sequence'],
                                   transport.calls[0]['observation']['sequence'])
                transport.gates[1].set()
                until(lambda: dashboard.inference['requests_completed'] == 1)
                self.assertFalse(dashboard.execute)
                self.assertNotIn('capture', hardware.calls)
                self.assertEqual(hardware.commands, [])
                self.close_fixture((transport, hardware, dashboard))

    def test_late_transient_error_after_user_stop_cannot_recreate_recovery(self):
        old_client = self.start_execution_request()
        old_client.collector.fail = TRANSIENT
        self.dashboard.control('execute_stop')
        self.transport.gates[1].set()
        until(lambda: len(self.transport.calls) == 3)
        self.assertFalse(self.recovery()['active'])
        self.assertFalse(self.dashboard.execute)
        self.assertTrue(self.dashboard.auto_infer)
        self.assertEqual(self.hardware.commands, [])
        self.assertNotIn('recovery_ack', self.hardware.calls)

    def test_recovery_deadline_stops_inference_and_execution(self):
        with patch.object(webapp.Dashboard, 'RECOVERY_TIMEOUT_S', .15):
            self.enter_execution_recovery()
            until(lambda: not self.dashboard.auto_infer)
        self.assertFalse(self.recovery()['active'])
        self.assertFalse(self.dashboard.execute)
        self.assertIsNotNone(self.dashboard.inference['last_error'])
        self.assertEqual(self.hardware.commands, [])

    def test_rejected_execution_token_never_rearms(self):
        self.enter_execution_recovery()
        self.hardware.reject_recovery = True
        self.hardware.set_feedback_ready(True)
        until(lambda: not self.dashboard.auto_infer)
        self.assertFalse(self.recovery()['active'])
        self.assertFalse(self.dashboard.execute)
        self.assertIn('recovery_ack', self.hardware.calls)
        self.assertEqual(self.hardware.commands, [])

    def test_unknown_hard_and_mixed_errors_are_terminal(self):
        cases = ('unexpected parser bug', '关节目标超出软限位', '夹爪报警 0x0001',
                 'RM75 存在未使能关节', 'Controller response timeout; 夹爪报警 0x0001',
                 FORCE_GAP + '; RM75 控制器或关节反馈报错',
                 '夹爪观测反馈过期; 夹爪报警 0x0001',
                 '夹爪反馈与机械臂状态不同步; 夹爪未使能')
        for message in cases:
            with self.subTest(error=message):
                transport, hardware, dashboard = self.make_fixture()
                dashboard.start()
                until(lambda: len(transport.calls) == 1)
                dashboard.client.collector.fail = message
                transport.gates[0].set()
                until(lambda: not dashboard.auto_infer)
                self.assertFalse(dashboard.status()['recovery']['active'])
                self.assertFalse(dashboard.execute)
                self.assertEqual(hardware.commands, [])
                self.close_fixture((transport, hardware, dashboard))

    def test_invalid_model_action_is_terminal(self):
        self.start_execution_request()
        self.transport.invalid_response_once = True
        self.transport.gates[1].set()
        until(lambda: not self.dashboard.auto_infer)
        self.assertFalse(self.recovery()['active'])
        self.assertFalse(self.dashboard.execute)
        self.assertEqual(self.hardware.commands, [])

    def test_missing_sample_does_not_hide_a_hard_cause(self):
        error = RuntimeError('sensor collection failed: ' + FORCE_GAP)
        error.__cause__ = RuntimeError('RM75 存在未使能关节')
        self.assertFalse(transient_failure(error))

    def test_stop_failure_prevents_automatic_recovery(self):
        self.start_execution_request()
        self.hardware.command_error_once = TRANSIENT
        self.hardware.stop_error_once = 'slow_stop failed'
        self.transport.gates[1].set()
        until(lambda: not self.dashboard.auto_infer)
        self.assertFalse(self.recovery()['active'])
        self.assertFalse(self.dashboard.execute)
        self.assertEqual(self.hardware.commands, [])

    def test_active_recovery_does_not_swallow_a_new_unknown_or_hard_http_error(self):
        for error in (RuntimeError('unexpected health parser bug'),
                      HTTPError('http://127.0.0.1:1/health', 401, 'Unauthorized', {}, None)):
            with self.subTest(error=str(error)):
                _transport, hardware, dashboard = self.make_fixture()
                dashboard.hardware = hardware
                self.assertTrue(dashboard._begin_recovery(RuntimeError(TRANSIENT), dashboard.generation))
                self.assertTrue(dashboard.status()['recovery']['active'])
                # False tells the normal worker exception handler to stop;
                # being already in recovery must not authorize this new error.
                self.assertFalse(dashboard._begin_recovery(error, dashboard.generation))
                self.assertEqual(hardware.commands, [])

    def test_short_ready_interval_and_second_collector_failure_share_one_budget(self):
        simulated_now = [100.0]
        with patch.object(webapp.clock, 'now', side_effect=lambda: simulated_now[0]), \
                patch.object(webapp.Dashboard, 'RECOVERY_STABLE_S', .01), \
                patch.object(webapp.Dashboard, 'RECOVERY_TIMEOUT_S', .05):
            self.dashboard.start()
            until(lambda: len(self.transport.calls) == 1)
            self.dashboard.client.collector.fail = 'sensor collection failed: 夹爪运动反馈过期'
            self.hardware.set_feedback_ready(False)
            self.transport.gates[0].set()
            until(lambda: self.recovery()['active'])
            original_deadline = self.dashboard._recovery_deadline
            self.assertAlmostEqual(original_deadline, 100.05)
            self.hardware.set_feedback_ready(True)
            until(lambda: self.dashboard._recovery_ready_since is not None)
            simulated_now[0] = 100.02
            until(lambda: len(self.transport.calls) == 2)
            self.assertFalse(self.recovery()['active'])
            self.dashboard.client.collector.fail = 'sensor collection failed: ' + FORCE_GAP
            self.transport.gates[1].set()
            until(lambda: self.recovery()['active'] and self.recovery()['attempts'] == 2)
            self.assertEqual(self.dashboard._recovery_deadline, original_deadline)
            simulated_now[0] = 100.051
            until(lambda: not self.dashboard.auto_infer)
            self.assertFalse(self.recovery()['active'])
            self.assertEqual(self.dashboard.inference['requests_completed'], 0)
            self.assertEqual(self.hardware.commands, [])

    def test_http_503_body_with_hard_failure_overrides_transient_status_code(self):
        for message, expected in (('non-finite model action', False),
                                  ('夹爪报警 0x0001', False),
                                  ('server is not ready for inference', True)):
            with self.subTest(body=message):
                body = BytesIO(json.dumps({'protocol': PROTOCOL, 'error': message}).encode('utf8'))
                failure = HTTPError('http://127.0.0.1:1/health', 503, 'Service Unavailable', {}, body)
                transport = InferenceHTTPClient('http://127.0.0.1:1', token='')
                transport.opener = Mock()
                transport.opener.open.side_effect = failure
                with self.assertRaises(RuntimeError) as caught:
                    transport.health()
                self.assertIn(message, str(caught.exception))
                self.assertEqual(transient_failure(caught.exception), expected)
                transport.opener.open.assert_called_once()


if __name__ == '__main__':
    unittest.main()
