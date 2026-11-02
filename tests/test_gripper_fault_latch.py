"""Offline regressions for fault generations at the robot dispatch boundary.

No SDK is initialized, no sockets/threads are started, and all native motion
methods are mocks. Recovery uses the real measured-feedback parser.
"""
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

from realman_sync_client import clock, webapp
from realman_sync_client.web_hardware import WebHardware, _CheckedGripper


class ReadOnlyBus:
    def __init__(self, timer):
        self.timer = timer
        self.enabled = 1
        self.reads = []

    def rm_read_holding_registers(self, params):
        assert params == (1, 0x0100, 1, 1)
        self.reads.append(params)
        self.timer.value += .01
        return 0, self.enabled

    def rm_read_multiple_holding_registers(self, params):
        assert params == (1, 0x060D, 1, 6)
        self.reads.append(params)
        self.timer.value += .01
        words = [0, 6000, 0, 50, 0, 0]
        return 0, [byte for value in words for byte in value.to_bytes(2, 'big')]


class AfterReleaseLock:
    """Run one race hook after a snapshot releases its ordinary state lock."""
    def __init__(self):
        self.inner = threading.Lock()
        self.callback = None

    def __enter__(self):
        self.inner.acquire()
        return self

    def __exit__(self, *_):
        self.inner.release()
        callback, self.callback = self.callback, None
        if callback:
            callback()


class GripperFaultLatchTests(unittest.TestCase):
    def setUp(self):
        self.timer = SimpleNamespace(value=10.)
        self.clock_patch = patch.object(clock, 'now', lambda: self.timer.value)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.worker = _CheckedGripper(dict(gripper_enable_poll_s=.25))
        self.worker.mapping_min, self.worker.mapping_max = 0, 12000
        self.bus = ReadOnlyBus(self.timer)
        self.read_feedback()

        # Bypass construction deliberately: no configuration discovery, SDK,
        # cameras or native connection. Exercise the actual command validator.
        self.hardware = object.__new__(WebHardware)
        self.hardware.lock = threading.RLock()
        self.hardware.gripper = self.worker
        self.hardware._closing = self.hardware.closed = False
        self.hardware.estop_latched = False
        self.hardware.connected = self.hardware.armed = True
        self.hardware._ever_armed = False
        self.hardware._acknowledged_gripper_faults = 0
        self.hardware.config = dict(action_space='joint8', feedback_max_age_s=.03,
            joint_limit_margin_deg=2., max_joint_step_deg=8., follow=False,
            trajectory_mode=0, trajectory_radio=0)
        self.hardware.joint_min, self.hardware.joint_max = [-180.] * 7, [180.] * 7
        self.hardware.last_command = None
        self.hardware.feedback = Mock()
        self.hardware.feedback.snapshot.return_value = ([0.] * 7, [0.] * 6, [0.] * 6, 10.)
        self.hardware.sdk = Mock()
        self.hardware.sdk.rm_movej_canfd.return_value = 0
        self.action = (0.,) * 7 + (.45,)

    def read_feedback(self):
        return self.worker._poll_feedback(self.bus, lambda *args: args, 0, 12000)

    def fail_and_recover(self):
        self.worker._feedback_failed(TimeoutError('read response timeout'))
        self.read_feedback()
        self.assertTrue(self.worker.status()['ready'])

    def test_recovered_read_does_not_acknowledge_fault_or_write_joint(self):
        self.worker.command(.1)
        self.worker.ramp_target = .1
        sequence = self.worker.command_sequence
        self.fail_and_recover()
        self.assertIsNone(self.worker.desired)
        self.assertIsNone(self.worker.ramp_target)
        self.assertGreater(self.worker.command_sequence, sequence)
        self.assertGreater(self.worker.fault_generation, 0)
        self.assertTrue(self.worker.status()['measurement_ready'])
        with self.assertRaisesRegex(RuntimeError, '关节目标未写入'):
            self.hardware.command(self.action)
        self.hardware.sdk.rm_movej_canfd.assert_not_called()
        self.assertIsNone(self.worker.desired)

    def test_explicit_ack_only_changes_generation_then_new_command_can_dispatch(self):
        self.fail_and_recover()
        reads_before = list(self.bus.reads)
        history_before = list(self.worker.history)
        state_before = dict(self.worker.state)
        sequence_before = self.worker.command_sequence
        self.hardware.acknowledge_gripper_faults()
        self.assertEqual(self.hardware._acknowledged_gripper_faults, self.worker.fault_generation)
        self.assertEqual(self.bus.reads, reads_before)
        self.assertEqual(list(self.worker.history), history_before)
        self.assertEqual(self.worker.state, state_before)
        self.assertEqual(self.worker.command_sequence, sequence_before)
        self.assertIsNone(self.worker.desired)
        self.assertTrue(self.worker.requests.empty())
        self.hardware.sdk.assert_not_called()
        self.assertEqual(self.hardware.sdk.mock_calls, [])
        self.hardware.command(self.action)
        self.hardware.sdk.rm_movej_canfd.assert_called_once()
        self.assertEqual(self.worker.desired, self.action[-1])

    def test_per_chunk_arm_never_acknowledges_recovered_fault(self):
        self.fail_and_recover()
        healthy = dict(robot_ready=True, cameras_ready=True, force_ready=True, gripper_ready=True)
        with patch.object(self.hardware, 'status', return_value=healthy), \
                patch.object(self.hardware, '_local_ready', return_value=(True, None)):
            self.hardware.arm()
            self.hardware.arm()
        self.assertEqual(self.hardware._acknowledged_gripper_faults, 0)
        with self.assertRaisesRegex(RuntimeError, '关节目标未写入'):
            self.hardware.command(self.action)
        self.hardware.sdk.rm_movej_canfd.assert_not_called()

    def test_bad_or_stale_feedback_cannot_be_acknowledged_or_auto_enabled(self):
        for scenario in ('false_ready', 'disabled', 'expired_enable', 'missing_enable', 'expired_position'):
            with self.subTest(scenario=scenario):
                self.bus.enabled = 1
                self.worker.enabled_received = None
                self.read_feedback()
                if scenario == 'false_ready':
                    self.worker.state['ready'] = False
                elif scenario == 'disabled':
                    self.bus.enabled = 0
                    self.worker.enabled_received = None
                    self.read_feedback()
                elif scenario == 'expired_enable':
                    self.worker.enabled_received = self.timer.value - .501
                elif scenario == 'missing_enable':
                    self.worker.enabled_received = None
                else:
                    self.timer.value += .501
                previous = self.hardware._acknowledged_gripper_faults
                reads_before = list(self.bus.reads)
                with self.assertRaisesRegex(RuntimeError, '尚未恢复'):
                    self.hardware.acknowledge_gripper_faults()
                self.assertEqual(self.hardware._acknowledged_gripper_faults, previous)
                self.assertEqual(self.bus.reads, reads_before)
                self.assertTrue(self.worker.requests.empty())
                self.assertIsNone(self.worker.desired)
        self.assertEqual(self.hardware.sdk.mock_calls, [])

    def test_new_fault_after_ack_requires_another_explicit_ack(self):
        self.fail_and_recover()
        self.hardware.acknowledge_gripper_faults()
        acknowledged = self.hardware._acknowledged_gripper_faults
        self.fail_and_recover()
        self.assertGreater(self.worker.fault_generation, acknowledged)
        with self.assertRaisesRegex(RuntimeError, '关节目标未写入'):
            self.hardware.command(self.action)
        self.hardware.sdk.rm_movej_canfd.assert_not_called()

    def test_ack_cannot_mix_old_ready_snapshot_with_new_fault_generation(self):
        lock = AfterReleaseLock()
        self.worker.lock = lock
        previous_generation = self.worker.fault_generation
        # The fault and new measured recovery happen just after status copies
        # ready. Its generation must describe that same snapshot, not the future.
        lock.callback = self.fail_and_recover
        self.hardware.acknowledge_gripper_faults()
        self.assertEqual(self.hardware._acknowledged_gripper_faults, previous_generation)
        self.assertGreater(self.worker.fault_generation, previous_generation)
        with self.assertRaisesRegex(RuntimeError, '关节目标未写入'):
            self.hardware.command(self.action)
        self.hardware.sdk.rm_movej_canfd.assert_not_called()

    def test_fault_during_joint_validation_is_rechecked_before_native_dispatch(self):
        snapshot = self.hardware.feedback.snapshot.return_value

        def state_snapshot(*_):
            self.fail_and_recover()
            return snapshot

        self.hardware.feedback.snapshot.side_effect = state_snapshot
        with self.assertRaisesRegex(RuntimeError, '关节目标未写入'):
            self.hardware.command(self.action)
        self.hardware.sdk.rm_movej_canfd.assert_not_called()

    def test_fault_between_gripper_admission_snapshot_and_queue_cannot_replay_target(self):
        lock = AfterReleaseLock()
        self.worker.lock = lock
        lock.callback = self.fail_and_recover
        # This command began before the communication fault. Fresh later
        # measurements must not make its old admission snapshot valid again.
        with self.assertRaises(RuntimeError):
            self.worker.command(.1, source='policy')
        self.assertIsNone(self.worker.desired)
        self.assertIsNone(self.worker.ramp_target)
        self.assertTrue(self.worker.requests.empty())

    def test_hold_between_gripper_snapshot_and_queue_cancels_pending_intent(self):
        for had_target in (False, True):
            with self.subTest(had_target=had_target):
                if had_target:
                    self.worker.command(.2)
                else:
                    self.assertIsNone(self.worker.desired)
                lock = AfterReleaseLock()
                self.worker.lock = lock
                lock.callback = self.worker.hold
                # Even an empty queue can have a caller between readiness
                # admission and enqueue. An operator hold cancels that caller.
                with self.assertRaises(RuntimeError):
                    self.worker.command(.1, source='policy')
                self.assertIsNone(self.worker.desired)
                self.assertIsNone(self.worker.ramp_target)

    def dashboard(self):
        dashboard = webapp.Dashboard(SimpleNamespace(server_url='http://127.0.0.1:1'), auto_infer=True)
        # Dashboard availability was measured separately; the actual adapter's
        # explicit acknowledgement remains the operation under test.
        dashboard.hardware = SimpleNamespace(execution_available=True, armed=False,
            acknowledge_gripper_faults=self.hardware.acknowledge_gripper_faults)
        dashboard.execution_available = True
        return dashboard

    def test_only_explicit_execute_start_acknowledges_in_dashboard(self):
        self.fail_and_recover()
        dashboard = self.dashboard()
        dashboard.control('inference_start')
        dashboard.control('connect')
        self.assertEqual(self.hardware._acknowledged_gripper_faults, 0)
        self.assertFalse(dashboard.execute)
        dashboard.control('execute_start')
        self.assertTrue(dashboard.execute)
        self.assertEqual(self.hardware._acknowledged_gripper_faults, self.worker.fault_generation)
        self.assertEqual(self.hardware.sdk.mock_calls, [])
        self.assertIsNone(self.worker.desired)

    def test_execute_start_rejects_failed_ack_and_does_not_enable_execution(self):
        self.worker._feedback_failed(TimeoutError('read response timeout'))
        dashboard = self.dashboard()
        with self.assertRaisesRegex(RuntimeError, '尚未恢复'):
            dashboard.control('execute_start')
        self.assertFalse(dashboard.execute)
        self.assertEqual(self.hardware._acknowledged_gripper_faults, 0)
        self.assertEqual(self.hardware.sdk.mock_calls, [])


if __name__ == '__main__':
    unittest.main()
