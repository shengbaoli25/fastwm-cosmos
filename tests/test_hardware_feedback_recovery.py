"""Session recovery permits, using real feedback parsers and no hardware I/O."""
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

from realman_sync_client import clock
from realman_sync_client.gripper import FeedbackUnavailable
from realman_sync_client.hardware import RealManHardwareAdapter
from realman_sync_client.web_hardware import WebHardware, _CheckedGripper, _MonitoredFeedback
from test_web_hardware import force_config, native_state
import test_gripper_recovery_loop as loop


class Bus:
    def __init__(self, timer):
        self.timer, self.enabled, self.alarm, self.reads = timer, 1, 0, 0

    def rm_read_holding_registers(self, params):
        self.timer.value += .01
        self.reads += 1
        return 0, self.enabled

    def rm_read_multiple_holding_registers(self, params):
        self.timer.value += .01
        self.reads += 1
        return 0, [byte for word in [0, 6000, 0, 50, 0, self.alarm]
                   for byte in word.to_bytes(2, 'big')]


class RecoveryPermitTests(unittest.TestCase):
    def setUp(self):
        self.timer = SimpleNamespace(value=10.)
        self.clock_patch = patch.object(clock, 'now', lambda: self.timer.value)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.worker = _CheckedGripper({'gripper_transport': 'json'})
        self.worker.thread = Mock(is_alive=Mock(return_value=True))
        self.bus = Bus(self.timer)
        self.read()
        self.feedback = _MonitoredFeedback(Mock(), force_config())
        self.feedback.ingest(native_state(), self.timer.value)
        self.hardware = object.__new__(WebHardware)
        self.hardware.lock = threading.RLock()
        self.hardware.gripper, self.hardware.feedback = self.worker, self.feedback
        self.hardware.closed = self.hardware._closing = self.hardware.estop_latched = False
        self.hardware._diagnostic_active = False
        self.hardware._feedback_recovery_generation = 0
        self.hardware._acknowledged_gripper_faults = 0
        self.hardware.sdk = Mock()
        self.hardware.status = Mock(return_value={})
        self.hardware._local_ready = Mock(return_value=(True, None))
        self.token = self.hardware.capture_feedback_recovery_token()

    def read(self):
        self.worker.enabled_received = None
        self.worker._poll_feedback(self.bus, lambda *args: args, 0, 12000)

    def fail_and_recover(self, error=None):
        self.worker._feedback_failed(error or TimeoutError('read timed out'))
        self.read()

    def test_transient_feedback_ack_changes_only_acknowledged_generation(self):
        for error in (TimeoutError('read timed out'), FeedbackUnavailable('freshness')):
            self.fail_and_recover(error)
            before = (self.bus.reads, self.worker.command_sequence, list(self.worker.history))
            self.assertTrue(self.hardware.acknowledge_recovered_feedback(self.token))
            self.assertEqual(self.hardware._acknowledged_gripper_faults, self.worker.fault_generation)
            self.assertEqual(before, (self.bus.reads, self.worker.command_sequence, list(self.worker.history)))
            self.assertIsNone(self.worker.desired)
            self.assertTrue(self.worker.requests.empty())
            self.assertEqual(self.hardware.sdk.mock_calls, [])

    def test_disabled_seen_after_timeout_remains_hard_after_reenabled(self):
        self.worker._feedback_failed(TimeoutError())
        self.bus.enabled = 0
        self.read()
        self.bus.enabled = 1
        self.read()
        self.assertTrue(self.worker.status()['ready'])
        self.assertGreater(self.worker.hard_fault_generation, 0)
        self.assertFalse(self.hardware.acknowledge_recovered_feedback(self.token))

    def test_alarm_latches_even_if_later_reply_is_zero(self):
        self.bus.alarm = 32
        self.read()
        self.bus.alarm = 0
        self.read()
        self.assertTrue(self.worker.status()['ready'])
        self.assertFalse(self.hardware.acknowledge_recovered_feedback(self.token))

    def test_unknown_error_cannot_become_transient_after_good_feedback(self):
        self.fail_and_recover(RuntimeError('unknown parser or mapping failure'))
        self.assertTrue(self.worker.status()['ready'])
        self.assertFalse(self.hardware.acknowledge_recovered_feedback(self.token))

    def test_false_ready_stale_malformed_alarm_and_closed_worker_do_not_ack(self):
        for case in ('not_ready', 'position_old', 'enable_old', 'false_alarm', 'thread_stopped'):
            with self.subTest(case=case):
                self.read()
                self.worker.thread.is_alive.return_value = True
                if case == 'not_ready':
                    self.worker.state['ready'] = False
                elif case == 'position_old':
                    self.timer.value += .501
                elif case == 'enable_old':
                    self.worker.enabled_received = self.timer.value-.501
                elif case == 'false_alarm':
                    self.worker.state['alarm'] = False
                else:
                    self.worker.thread.is_alive.return_value = False
                self.assertFalse(self.hardware.acknowledge_recovered_feedback(self.token))
        self.assertEqual(self.hardware._acknowledged_gripper_faults, 0)

    def test_replacing_worker_or_feedback_invalidates_object_bound_token(self):
        old = self.hardware.gripper
        self.hardware.gripper = _CheckedGripper({})
        self.assertFalse(self.hardware.acknowledge_recovered_feedback(self.token))
        self.hardware.gripper = old
        self.hardware.feedback = _MonitoredFeedback(Mock(), force_config())
        self.assertFalse(self.hardware.acknowledge_recovered_feedback(self.token))
        self.assertFalse(self.hardware.acknowledge_recovered_feedback(None))

    def test_closing_then_reusing_same_worker_cannot_restore_old_permit(self):
        self.worker.close()
        self.worker.stop.clear()
        self.read()
        self.assertFalse(self.hardware.acknowledge_recovered_feedback(self.token))

    def test_arm_system_joint_disable_and_unknown_feedback_latch_across_good_packet(self):
        for case in ('system', 'joint', 'disable', 'unknown', 'force_frame'):
            with self.subTest(case=case):
                token = self.hardware.capture_feedback_recovery_token()
                bad = native_state()
                if case == 'system':
                    bad.err.err_len, bad.err.err[0] = 1, 32
                elif case == 'joint':
                    bad.joint_status.joint_err_code[0] = 1
                elif case == 'disable':
                    bad.joint_status.joint_en_flag[0] = 0
                elif case == 'unknown':
                    bad.errCode = 1
                else:
                    bad.force_sensor.coordinate = 0
                self.feedback.ingest(bad, self.timer.value)
                self.feedback.ingest(native_state(), self.timer.value)
                self.assertFalse(self.hardware.acknowledge_recovered_feedback(token))

    def test_changed_fault_between_status_and_final_lock_cannot_be_acknowledged(self):
        original = self.worker.status
        called = [False]
        def race(*args, **kwargs):
            result = original(*args, **kwargs)
            if not called[0]:
                called[0] = True
                self.fail_and_recover()
            return result
        with patch.object(self.worker, 'status', side_effect=race):
            self.assertFalse(self.hardware.acknowledge_recovered_feedback(self.token))
        self.assertEqual(self.hardware._acknowledged_gripper_faults, 0)

    def test_new_hard_fault_after_ack_is_rechecked_before_dispatch(self):
        self.fail_and_recover()
        self.assertTrue(self.hardware.acknowledge_recovered_feedback(self.token))
        self.feedback.hard_fault_generation += 1
        with self.assertRaisesRegex(RuntimeError, '执行会话已失效'):
            self.hardware._check_command_dispatch()
        self.assertEqual(self.hardware.sdk.mock_calls, [])

    def test_user_invalidation_blocks_recovery_but_chunk_slow_stop_does_not(self):
        with patch.object(RealManHardwareAdapter, '_stop') as stop:
            self.hardware.slow_stop()
        stop.assert_called_once_with(False)
        self.fail_and_recover()
        self.assertTrue(self.hardware.acknowledge_recovered_feedback(self.token))
        self.hardware.invalidate_feedback_recovery()
        self.assertFalse(self.hardware.acknowledge_recovered_feedback(self.token))

    def test_emergency_and_manual_operations_invalidate_even_if_not_completed(self):
        for operation in ('emergency', 'manual', 'reconnect', 'clear'):
            with self.subTest(operation=operation):
                token = self.hardware.capture_feedback_recovery_token()
                self.hardware._diagnostic_lock = Mock(acquire=Mock(return_value=False))
                if operation == 'emergency':
                    with patch.object(RealManHardwareAdapter, 'emergency_stop'):
                        self.hardware.emergency_stop()
                elif operation == 'manual':
                    with self.assertRaises(ValueError):
                        self.hardware.manual_gripper('invalid')
                else:
                    with self.assertRaises(RuntimeError):
                        getattr(self.hardware, 'reconnect_gripper' if operation == 'reconnect' else 'clear_gripper_error')()
                self.assertFalse(self.hardware.acknowledge_recovered_feedback(token))

    def test_current_arm_not_ready_or_emergency_state_blocks_ack(self):
        self.hardware._local_ready.return_value = (False, 'arm not ready')
        self.assertFalse(self.hardware.acknowledge_recovered_feedback(self.token))
        self.hardware._local_ready.return_value = (True, None)
        self.hardware.estop_latched = True
        self.assertFalse(self.hardware.acknowledge_recovered_feedback(self.token))

    def test_freshness_dispatch_failure_keeps_reader_alive_without_replaying_target(self):
        helper = loop.RecoveryLoopTests()
        worker, timer, sdk, io = helper.make()
        accepted = []
        def intent():
            if io.positions == 1 and not accepted:
                worker.command(0.)
                accepted.append(True)
            elif io.positions >= 3:
                self.assertTrue(worker.status()['ready'])
                self.assertIsNone(worker.desired)
                worker.stop.set()
        worker.wake.on_wait = intent
        with patch.object(worker, '_send_target', side_effect=FeedbackUnavailable('freshness')) as send:
            helper.run_loop(worker, timer, sdk, io)
        send.assert_called_once()
        self.assertEqual(io.positions, 3)
        self.assertEqual(io.writes, [])
        self.assertEqual(worker.fault_generation, 1)
        self.assertEqual(worker.hard_fault_generation, 0)


if __name__ == '__main__':
    unittest.main()
