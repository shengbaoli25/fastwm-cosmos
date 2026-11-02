"""Read-only reconnection and runtime cancellation; no real hardware/socket."""
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

from realman_sync_client import web_hardware as web, webapp


class ReadOnlySDK:
    def __init__(self, enabled=True, mismatch=None, alarm=0):
        self.enabled, self.mismatch, self.calls = enabled, mismatch, []
        self.alarm = alarm

    def rm_get_current_arm_state(self):
        self.calls.append(('arm_state',))
        return 0, {}

    def rm_get_tool_voltage(self):
        self.calls.append(('voltage',))
        return 0, 0 if self.mismatch == 'voltage' else 3

    def rm_get_tool_rs485_mode(self):
        self.calls.append(('bus',))
        return dict(return_code=0, mode=1, baudrate=9600 if self.mismatch == 'bus' else 115200, timeout=2)

    def rm_read_holding_registers(self, params):
        assert params == (1, 0x0100, 1, 1)
        self.calls.append(('read', params))
        return 0, int(self.enabled)

    def rm_read_multiple_holding_registers(self, params):
        _, address, _, count = params
        self.calls.append(('read_many', params))
        words = {0x0610:[0, 0, self.alarm], 0x0404:[0, 0, 0, 12000],
            0x0301:[0, 0, 0, 12000], 0x0104:[30 if self.mismatch == 'speed' else 100, 50, 100, 100],
            0x060D:[0, 6000, 0, 50, 0, self.alarm]}[address][:count]
        return 0, [byte for value in words for byte in value.to_bytes(2, 'big')]

    def __getattr__(self, name):
        raise AssertionError(f'Unexpected SDK operation during read-only reconnect: {name}')


class FakeThread:
    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive


class Candidate(web._CheckedGripper):
    """Run the actual read-only setup/parser without launching a thread."""
    def __init__(self, config, sdk, events):
        super().__init__(config)
        self.sdk, self.events = sdk, events
        self.wait_error = None
        self.stuck = False
        self.closed_count = 0

    def start(self):
        self.events.append('candidate_start')
        self.thread = FakeThread()
        self.mapping_min, self.mapping_max, _ = self._setup(self.sdk, lambda *args: args)
        self._poll_feedback(self.sdk, lambda *args: args, self.mapping_min, self.mapping_max)

    def wait_initialized(self, _timeout):
        if self.wait_error:
            raise self.wait_error
        return self.status()

    def close(self):
        self.closed_count += 1
        self.events.append('candidate_close')
        if self.thread is not None and not self.stuck:
            self.thread.alive = False
        self.thread = None  # Emulate legacy join-timeout behavior too.


class ReconnectTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.hardware = object.__new__(web.WebHardware)
        self.hardware.lock, self.hardware._diagnostic_lock = threading.RLock(), threading.Lock()
        self.hardware.connected = self.hardware._robot_identity_valid = True
        self.hardware.closed = self.hardware._closing = self.hardware.armed = False
        self.hardware.estop_latched = self.hardware._diagnostic_active = False
        self.hardware._clearing_gripper = self.hardware._reconnecting_gripper = False
        self.hardware._last_gripper_diagnosis = self.hardware._last_gripper_result = None
        self.hardware._manual_gripper_operation = self.hardware._last_manual_gripper_result = None
        self.hardware._gripper_error = None
        self.hardware._acknowledged_gripper_faults = 10
        self.hardware.config = dict(gripper_speed=100, gripper_torque=50, gripper_enable_poll_s=.25)
        self.hardware.sdk = ReadOnlySDK()
        self.old = Candidate(self.hardware.config, self.hardware.sdk, self.events)
        self.old.start()
        self.old.fault_generation = 7
        self.hardware.gripper = self.old
        self.events.clear()

    def run_candidate(self, candidate):
        with patch.object(web, '_CheckedGripper', return_value=candidate) as factory:
            result = self.hardware.reconnect_gripper()
        factory.assert_called_once_with(self.hardware.config)
        return result

    def test_reconnect_preserves_real_enabled_or_disabled_without_any_write(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                candidate = Candidate(self.hardware.config, ReadOnlySDK(enabled=enabled), self.events)
                previous = self.hardware.gripper
                previous_thread = previous.thread
                result = self.run_candidate(candidate)
                self.assertTrue(result['ok'])
                self.assertIs(result['enabled'], enabled)
                self.assertFalse(result['writes_performed'])
                self.assertEqual(result['writes'], [])
                self.assertFalse(previous_thread.is_alive())
                self.assertEqual(self.events[-2:], ['candidate_close', 'candidate_start'])
                self.assertIs(self.hardware.gripper, candidate)
                self.assertGreater(candidate.fault_generation, 10)
                self.assertEqual(self.hardware._acknowledged_gripper_faults, 10)
                self.assertIsNone(candidate.desired)
                self.assertTrue(candidate.requests.empty())
                self.assertFalse(self.hardware._diagnostic_active)
                self.assertFalse(self.hardware._reconnecting_gripper)
                self.assertTrue(self.hardware.gripper_diagnostics()['reconnect_available'])

    def test_alarm_reconnect_keeps_reading_without_clearing_or_enabling_motion(self):
        candidate = Candidate(self.hardware.config, ReadOnlySDK(alarm=32), self.events)
        result = self.run_candidate(candidate)
        self.assertTrue(result['ok'])
        self.assertEqual(result['after_alarm'], 32)
        self.assertIn('设备仍报警 0x0020', result['message'])
        self.assertFalse(result['writes_performed'])
        self.assertEqual(result['writes'], [])
        self.assertTrue(candidate.thread.is_alive())
        self.assertEqual(candidate.closed_count, 0)
        before = len(candidate.history)
        candidate._poll_feedback(candidate.sdk, lambda *args: args, 0, 12000)
        self.assertEqual(len(candidate.history), before + 1)
        state = candidate.status()
        self.assertTrue(state['monitoring_ready'])
        self.assertEqual(state['alarm'], 32)
        self.assertEqual(state['opening'], .5)
        for field in ('ready', 'measurement_ready', 'control_feedback_ready'):
            self.assertFalse(state[field])
        diagnostic = self.hardware.gripper_diagnostics()
        self.assertTrue(diagnostic['monitoring_ready'])
        for field in ('enable_available', 'motion_available'):
            self.assertFalse(diagnostic[field])
        for operation in ('enable', 'open', 'close'):
            with self.subTest(operation=operation), self.assertRaises(RuntimeError):
                self.hardware.manual_gripper(operation)
        with patch.object(web.RealManHardwareAdapter, 'command') as command:
            with self.assertRaisesRegex(RuntimeError, '关节目标未写入'):
                self.hardware.command([0.] * 7 + [.5])
            command.assert_not_called()
        with self.assertRaises(RuntimeError):
            candidate.command(.8)
        # Fake SDK rejects every method except the explicitly allowed reads.
        self.assertTrue(all(call[0] in ('arm_state', 'voltage', 'bus', 'read', 'read_many')
                            for call in candidate.sdk.calls))
        self.assertTrue(candidate.requests.empty())
        self.assertIsNone(candidate.desired)

    def test_old_worker_must_exit_and_live_reference_survives_repeated_click(self):
        self.old.stuck = True
        old_thread = self.old.thread
        for _ in range(2):
            with patch.object(web, '_CheckedGripper') as factory:
                with self.assertRaisesRegex(RuntimeError, '旧夹爪线程未退出'):
                    self.hardware.reconnect_gripper()
                factory.assert_not_called()
            self.assertIs(self.hardware.gripper, self.old)
            self.assertIs(self.old.thread, old_thread)
            self.assertTrue(old_thread.is_alive())

    def test_monitoring_reconnect_still_requires_position_and_enable_within_500ms(self):
        for stale_source in ('position', 'enable'):
            with self.subTest(stale_source=stale_source):
                candidate = Candidate(self.hardware.config, ReadOnlySDK(), self.events)
                def stale_feedback(_timeout):
                    old = web.clock.now() - .6
                    if stale_source == 'enable':
                        candidate.enabled_received = old
                    else:
                        arrival, _ = candidate.history[-1]
                        candidate.history_started.clear()
                        candidate.history_started.append((arrival, old))
                        candidate.state['measurement_started'] = old
                    return candidate.status()
                candidate.wait_initialized = stale_feedback
                with patch.object(web, '_CheckedGripper', return_value=candidate):
                    with self.assertRaisesRegex(RuntimeError, '反馈过期'):
                        self.hardware.reconnect_gripper()
                self.assertEqual(candidate.closed_count, 1)
                self.assertFalse(candidate.status()['monitoring_ready'])
                self.assertFalse(self.hardware._last_gripper_result['writes_performed'])

    def test_read_only_setup_rejects_mismatch_without_repairing_it(self):
        for mismatch in ('voltage', 'bus', 'speed'):
            with self.subTest(mismatch=mismatch):
                candidate = Candidate(self.hardware.config, ReadOnlySDK(mismatch=mismatch), self.events)
                with patch.object(web, '_CheckedGripper', return_value=candidate):
                    with self.assertRaisesRegex(RuntimeError, '只读通信重连失败'):
                        self.hardware.reconnect_gripper()
                self.assertEqual(candidate.closed_count, 1)
                self.assertIsNone(candidate.desired)
                self.assertFalse(self.hardware._last_gripper_result['ok'])
                self.assertFalse(self.hardware._last_gripper_result['writes_performed'])
                self.assertFalse(self.hardware._diagnostic_active)
                self.assertTrue(self.hardware._diagnostic_lock.acquire(blocking=False))
                self.hardware._diagnostic_lock.release()

    def test_failed_candidate_is_closed_and_failure_reason_is_visible(self):
        candidate = Candidate(self.hardware.config, ReadOnlySDK(), self.events)
        candidate.wait_error = TimeoutError('new feedback unavailable')
        with patch.object(web, '_CheckedGripper', return_value=candidate):
            with self.assertRaisesRegex(RuntimeError, 'new feedback unavailable'):
                self.hardware.reconnect_gripper()
        self.assertEqual(candidate.closed_count, 1)
        self.assertIn('new feedback unavailable', self.hardware.gripper_diagnostics()['message'])
        self.assertIsNone(candidate.thread)
        self.assertIsNone(candidate.desired)

    def test_failed_candidate_with_live_thread_blocks_another_bus_owner(self):
        candidate = Candidate(self.hardware.config, ReadOnlySDK(), self.events)
        candidate.wait_error, candidate.stuck = TimeoutError('failed wait'), True
        with patch.object(web, '_CheckedGripper', return_value=candidate):
            with self.assertRaisesRegex(RuntimeError, '新线程清理未确认'):
                self.hardware.reconnect_gripper()
        self.assertIs(self.hardware.gripper, candidate)
        self.assertTrue(candidate.thread.is_alive())
        with patch.object(web, '_CheckedGripper') as factory:
            with self.assertRaisesRegex(RuntimeError, '旧夹爪线程未退出'):
                self.hardware.reconnect_gripper()
            factory.assert_not_called()

    def test_armed_closed_busy_or_emergency_cannot_begin_reconnection(self):
        for attr in ('armed', 'closed', '_closing', 'estop_latched', '_diagnostic_active'):
            with self.subTest(attr=attr), patch.object(self.hardware, attr, True), \
                    patch.object(web, '_CheckedGripper') as factory:
                self.assertFalse(self.hardware.gripper_diagnostics()['reconnect_available'])
                with self.assertRaises(RuntimeError):
                    self.hardware.reconnect_gripper()
                factory.assert_not_called()
        self.hardware._diagnostic_lock.acquire()
        try:
            with patch.object(web, '_CheckedGripper') as factory:
                with self.assertRaisesRegex(RuntimeError, '正在进行'):
                    self.hardware.reconnect_gripper()
                factory.assert_not_called()
        finally:
            self.hardware._diagnostic_lock.release()


class RuntimeReconnectTests(unittest.TestCase):
    def runtime(self):
        transport = SimpleNamespace(server_url='http://127.0.0.1:1',
            health=Mock(side_effect=RuntimeError('offline test')))
        dashboard = webapp.Dashboard(transport)
        hardware = SimpleNamespace(armed=False, execution_available=False,
            reconnect_gripper=Mock(return_value={'ok':True}),
            gripper_diagnostics=lambda: dict(reconnect_available=True), close=Mock(),
            emergency_stop=Mock(), slow_stop=Mock())
        dashboard.hardware = hardware
        return dashboard, hardware

    def test_queue_pauses_inference_invalidates_requests_and_defers_reconnection(self):
        dashboard, hardware = self.runtime()
        client = SimpleNamespace(stop_event=threading.Event())
        dashboard.client = client
        result = dashboard.control('gripper_reconnect')
        self.assertIn('只读', result['message'])
        self.assertFalse(dashboard.auto_infer)
        self.assertFalse(dashboard.execute)
        self.assertTrue(client.stop_event.is_set())
        self.assertEqual(dashboard.generation, 1)
        self.assertEqual(dashboard._hardware_action, 'reconnect_gripper')
        hardware.reconnect_gripper.assert_not_called()
        for action in ('connect', 'inference_start', 'execute_start', 'gripper_reconnect'):
            with self.subTest(action=action), self.assertRaises(ValueError):
                dashboard.control(action)

    def test_stop_actions_cancel_queued_reconnect(self):
        for action in ('disconnect', 'emergency_stop'):
            with self.subTest(action=action):
                dashboard, hardware = self.runtime()
                dashboard.control('gripper_reconnect')
                dashboard.control(action)
                self.assertIsNone(dashboard._hardware_action)
                hardware.reconnect_gripper.assert_not_called()
                self.assertFalse(dashboard.execute)

    def test_cancellation_during_client_release_prevents_reconnect_dispatch(self):
        for action in ('disconnect', 'emergency_stop'):
            with self.subTest(action=action):
                dashboard, hardware = self.runtime()
                dashboard.control('gripper_reconnect')
                def release():
                    if not dashboard.stop_event.is_set():
                        dashboard.control(action)
                        dashboard.stop_event.set()
                with patch.object(dashboard, '_close_client', side_effect=release):
                    dashboard._run()
                hardware.reconnect_gripper.assert_not_called()
                self.assertTrue(any('已取消' in row['message'] for row in dashboard.events))

    def test_reconnect_dispatch_occurs_only_after_client_release(self):
        dashboard, hardware = self.runtime()
        order = []
        dashboard.control('gripper_reconnect')
        def reconnect():
            order.append('reconnect')
            dashboard.stop_event.set()
            return {'ok':True}
        hardware.reconnect_gripper.side_effect = reconnect
        with patch.object(dashboard, '_close_client', side_effect=lambda: order.append('release')):
            dashboard._run()
        self.assertEqual(order[:2], ['release', 'reconnect'])
        hardware.reconnect_gripper.assert_called_once()
        self.assertFalse(dashboard.auto_infer)
        self.assertFalse(dashboard.execute)


if __name__ == '__main__':
    unittest.main()
