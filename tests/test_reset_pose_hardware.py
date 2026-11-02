"""Offline planned-reset tests. Every SDK operation and wait is mocked."""
from copy import deepcopy
import threading
import unittest
from unittest.mock import Mock, patch

from realman_sync_client import web_hardware as web
from test_web_hardware import native_state, force_config
import test_hardware_feedback_recovery as recovery


class ResetPoseTests(unittest.TestCase):
    def setUp(self):
        fixture = recovery.RecoveryPermitTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.hardware, self.feedback, self.timer = fixture.hardware, fixture.feedback, fixture.timer
        self.hardware.config = force_config()
        self.hardware.connected = self.hardware._robot_identity_valid = True
        self.hardware._ever_armed = self.hardware._reset_active = False
        self.hardware._diagnostic_lock = threading.Lock()
        self.hardware.joint_min, self.hardware.joint_max = [-180.]*7, [180.]*7
        self.hardware.slow_stop = Mock(side_effect=self.hardware.gripper.hold)
        self.hardware.sdk.rm_get_current_arm_state.return_value = (0, {'err': {'err': []}})
        self.hardware.sdk.rm_get_joint_min_pos.return_value = (0, [-180.]*7)
        self.hardware.sdk.rm_get_joint_max_pos.return_value = (0, [180.]*7)
        self.hardware.sdk.rm_movej.return_value = 0
        self.pose = dict(schema='realman-reset-pose-v1', robot_ip=self.hardware.config['robot_ip'],
            model=self.hardware.config.get('expected_model', 'RM_75'), joint_degrees=[0.]*7,
            speed_percent=10, gripper_action='unchanged')
        self.token = self.hardware.capture_reset_token()

    def tick(self, duration, *, target=True, invalid_force=False):
        self.assertFalse(self.hardware.lock._is_owned(), 'motion wait must not block Stop')
        self.timer.value += duration
        packet = native_state(coordinate=0 if invalid_force else 1)
        packet.joint_status.joint_position = list(self.pose['joint_degrees']) if target else [5.]*7
        self.feedback.ingest(packet, self.timer.value)

    def execute(self, sleep=None):
        with patch.object(web.time, 'sleep', side_effect=sleep or self.tick):
            return self.hardware.reset_to_pose(self.pose, expected_token=self.token)

    def test_one_nonblocking_joint_plan_completes_from_stable_fresh_measured_q(self):
        result = self.execute()
        self.hardware.sdk.rm_movej.assert_called_once_with([0.]*7, v=10, r=0, connect=0, block=0)
        self.assertTrue(result['completed'])
        self.assertGreaterEqual(result['duration_ms'], 200.)
        self.assertEqual(result['max_error_deg'], 0.)
        self.assertEqual(result['gripper_action'], 'unchanged')
        self.assertIsNone(self.hardware.gripper.desired)
        self.hardware.slow_stop.assert_called_once()
        self.assertFalse(self.hardware._reset_active)

    def test_broken_gripper_camera_and_force_do_not_block_joint_reset(self):
        self.hardware.gripper._feedback_failed(RuntimeError('tool mismatch'))
        self.hardware.cameras = []
        self.feedback.ingest(native_state(coordinate=0), self.timer.value)
        self.assertTrue(self.hardware.reset_available()[0])
        result = self.execute(lambda delay: self.tick(delay, invalid_force=True))
        self.assertTrue(result['completed'])
        self.hardware.sdk.rm_movej.assert_called_once()

    def test_stop_before_dispatch_rejects_old_epoch_without_moving(self):
        self.hardware.invalidate_feedback_recovery()
        with self.assertRaisesRegex(RuntimeError, '取消'):
            self.execute()
        self.hardware.sdk.rm_movej.assert_not_called()

    def test_stop_during_native_limit_reads_is_rechecked_at_dispatch(self):
        def limits():
            self.hardware.invalidate_feedback_recovery()
            return 0, [-180.]*7
        self.hardware.sdk.rm_get_joint_min_pos.side_effect = limits
        with self.assertRaisesRegex(RuntimeError, '取消'):
            self.execute()
        self.hardware.sdk.rm_movej.assert_not_called()
        self.assertGreaterEqual(self.hardware.slow_stop.call_count, 2)

    def test_stop_during_motion_stops_once_more_and_never_resends(self):
        def stop(delay):
            self.tick(delay)
            self.hardware.invalidate_feedback_recovery()
        with self.assertRaisesRegex(RuntimeError, '取消'):
            self.execute(stop)
        self.hardware.sdk.rm_movej.assert_called_once()
        self.assertEqual(self.hardware.slow_stop.call_count, 2)

    def test_unknown_or_rejected_move_ack_never_retries(self):
        for value in (1, True, TimeoutError('uncertain ACK')):
            with self.subTest(value=value):
                self.hardware.sdk.rm_movej.reset_mock(side_effect=True)
                if isinstance(value, Exception):
                    self.hardware.sdk.rm_movej.side_effect = value
                else:
                    self.hardware.sdk.rm_movej.return_value = value
                with self.assertRaises((RuntimeError, TimeoutError)):
                    self.execute()
                self.hardware.sdk.rm_movej.assert_called_once()

    def test_native_limits_rechecked_and_out_of_range_target_never_sent(self):
        self.hardware.sdk.rm_get_joint_max_pos.return_value = (0, [1.]*7)
        with self.assertRaisesRegex(ValueError, '软限位'):
            self.execute()
        self.hardware.sdk.rm_movej.assert_not_called()

    def test_invalid_pose_cannot_trigger_any_motion(self):
        for key, value in (('joint_degrees', [float('nan')]*7), ('joint_degrees', [True]*7),
                           ('speed_percent', 100), ('speed_percent', True),
                           ('robot_ip', 'wrong'), ('model', 'wrong'), ('gripper_action', 'open')):
            with self.subTest(key=key, value=value):
                bad = deepcopy(self.pose)
                bad[key] = value
                with self.assertRaises(ValueError):
                    self.hardware.reset_to_pose(bad, expected_token=self.token)
        self.hardware.sdk.rm_movej.assert_not_called()

    def test_system_joint_disabled_and_stale_udp_block_reset(self):
        for kind in ('system', 'joint', 'disabled', 'stale'):
            with self.subTest(kind=kind):
                packet = native_state()
                if kind == 'system':
                    packet.err.err_len, packet.err.err[0] = 1, 32
                elif kind == 'joint':
                    packet.joint_status.joint_err_code[0] = 1
                elif kind == 'disabled':
                    packet.joint_status.joint_en_flag[0] = 0
                self.feedback.ingest(packet, self.timer.value)
                if kind == 'stale':
                    self.timer.value += .1
                self.assertFalse(self.hardware.reset_available()[0])
                with self.assertRaises(RuntimeError):
                    self.hardware.capture_reset_token()
                self.feedback.ingest(native_state(), self.timer.value)
        self.hardware.sdk.rm_movej.assert_not_called()

    def test_historical_arm_fault_after_token_is_not_hidden_by_good_packet(self):
        bad = native_state()
        bad.joint_status.joint_en_flag[0] = 0
        self.feedback.ingest(bad, self.timer.value)
        self.feedback.ingest(native_state(), self.timer.value)
        self.assertTrue(self.hardware.reset_available()[0])
        with self.assertRaisesRegex(RuntimeError, '取消'):
            self.execute()
        self.hardware.sdk.rm_movej.assert_not_called()

    def test_no_new_feedback_cannot_confirm_completion(self):
        self.feedback.state_values = (0.,)*7
        with self.assertRaisesRegex(RuntimeError, '过期'):
            self.execute(lambda delay: setattr(self.timer, 'value', self.timer.value+delay))
        self.hardware.sdk.rm_movej.assert_called_once()
        self.assertEqual(self.hardware.slow_stop.call_count, 2)

    def test_30_second_timeout_stops_with_no_repeated_move(self):
        with self.assertRaisesRegex(RuntimeError, '30 秒'):
            self.execute(lambda delay: self.tick(delay, target=False))
        self.hardware.sdk.rm_movej.assert_called_once()
        self.assertEqual(self.hardware.slow_stop.call_count, 2)

    def test_emergency_closed_identity_and_replaced_connection_reject_reset(self):
        for field in ('estop_latched', 'closed', '_closing'):
            setattr(self.hardware, field, True)
            self.assertFalse(self.hardware.reset_available()[0])
            setattr(self.hardware, field, False)
        self.hardware._robot_identity_valid = False
        self.assertFalse(self.hardware.reset_available()[0])
        self.hardware._robot_identity_valid = True
        self.hardware.sdk = Mock()
        with self.assertRaisesRegex(RuntimeError, '取消'):
            self.execute()
        self.hardware.sdk.rm_movej.assert_not_called()


if __name__ == '__main__':
    unittest.main()
