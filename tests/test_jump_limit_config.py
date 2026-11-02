"""Explicit jump-limit opt-out, using mock SDK calls only."""
import json
import math
import threading
import unittest
from unittest.mock import Mock

from realman_sync_client.client import validate_hardware_chunk
from realman_sync_client.hardware import RealManHardwareAdapter


class JointJumpOptOutTests(unittest.TestCase):
    def hardware(self, limit):
        hardware = object.__new__(RealManHardwareAdapter)
        hardware.config = dict(feedback_max_age_s=.03, joint_limit_margin_deg=2.,
            max_joint_step_deg=limit, follow=False, trajectory_mode=0, trajectory_radio=0)
        hardware.connected = hardware.armed = True
        hardware.closed = hardware.estop_latched = False
        hardware.lock = threading.RLock()
        hardware.last_command = None
        hardware.feedback = Mock()
        hardware.feedback.snapshot.return_value = ((0.,)*7, (0.,)*6, (0.,)*6, 10.)
        hardware.joint_min, hardware.joint_max = (-90.,)*7, (90.,)*7
        hardware.sdk = Mock()
        hardware.sdk.rm_movej_canfd.return_value = 0
        hardware.gripper = Mock()
        return hardware

    def test_explicit_null_removes_both_jump_gates(self):
        hardware = self.hardware(None)
        action = (math.radians(8.16),) + (0.,)*6 + (.5,)
        validate_hardware_chunk(hardware, [action], (0.,)*7+(.5,))
        hardware._command_joint(action)
        target = hardware.sdk.rm_movej_canfd.call_args.args[0]
        self.assertAlmostEqual(target[0], 8.16)
        hardware.gripper.command.assert_called_once_with(.5)

    def test_existing_numeric_limit_still_rejects_before_sdk(self):
        hardware = self.hardware(8.)
        action = (math.radians(8.16),) + (0.,)*6 + (.5,)
        with self.assertRaisesRegex(ValueError, 'step limit'):
            validate_hardware_chunk(hardware, [action], (0.,)*7+(.5,))
        with self.assertRaisesRegex(ValueError, '跳变'):
            hardware._command_joint(action)
        hardware.sdk.rm_movej_canfd.assert_not_called()

    def test_null_keeps_joint_soft_limits(self):
        hardware = self.hardware(None)
        action = (math.radians(89.),) + (0.,)*6 + (.5,)
        with self.assertRaisesRegex(ValueError, 'soft-limit'):
            validate_hardware_chunk(hardware, [action], (0.,)*7+(.5,))
        with self.assertRaisesRegex(ValueError, '软限位'):
            hardware._command_joint(action)
        hardware.sdk.rm_movej_canfd.assert_not_called()

    def test_null_keeps_estop_and_finite_values(self):
        hardware = self.hardware(None)
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            hardware._command_joint((float('nan'),)+(0.,)*6+(.5,))
        hardware.estop_latched = True
        with self.assertRaisesRegex(RuntimeError, '急停'):
            hardware._command_joint((0.,)*7+(.5,))
        hardware.sdk.rm_movej_canfd.assert_not_called()

    def test_log_distinguishes_tracking_error_from_adjacent_targets(self):
        hardware = self.hardware(None)
        hardware.config['log_motion_trace'] = True
        with self.assertLogs('realman_sync_client.hardware', level='INFO') as messages:
            hardware._command_joint((math.radians(8.16),)+(0.,)*6+(.5,))
            hardware._command_joint((math.radians(8.17),)+(0.,)*6+(.5,))
        first, second = [json.loads(line.split('Motion trace ', 1)[1]) for line in messages.output]
        self.assertIsNone(first['adjacent_command_delta_deg'])
        self.assertAlmostEqual(second['measured_tracking_error_deg'], 8.17)
        self.assertAlmostEqual(second['adjacent_command_delta_deg'], .01)
        self.assertIsNone(second['configured_jump_limit_deg'])


if __name__ == '__main__':
    unittest.main()
