"""Exercise the command boundary using fake SDK/feedback only."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy"))

from realman_deploy.hardware import RealManHardwareAdapter


class JointStepLimitTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / "deploy/realman_hardware.example.json").read_text())

    def hardware(self, limit, *, solution=None, ik_code=0):
        config = dict(self.config(), max_joint_step_deg=limit)
        with patch.object(RealManHardwareAdapter, "_connect"):
            hardware = RealManHardwareAdapter(config)
        self.addCleanup(hardware._stop_pool.shutdown)
        hardware.connected = hardware.armed = True
        hardware.sdk = Mock()
        hardware.sdk.rm_algo_inverse_kinematics.return_value = (
            ik_code, solution if solution is not None else [21.55, 0, 0, 0, 0, 0, 0])
        hardware.sdk.rm_movep_canfd.return_value = 0
        hardware.feedback = Mock()
        hardware.feedback.snapshot.return_value = ([0.0] * 7, [0.0] * 6, [0.0] * 6, 100.0)
        hardware.ik_params_type = lambda *args: args
        hardware.joint_min, hardware.joint_max = [-180.0] * 7, [180.0] * 7
        hardware.gripper = Mock()
        return hardware

    def test_explicit_null_accepts_jump_that_numeric_limit_rejects(self):
        action = (0.001, 0, 0, 0, 0, 0, 0.5)
        limited = self.hardware(20.0)
        with self.assertRaisesRegex(ValueError, "跳变过大"):
            limited.command(action)
        limited.sdk.rm_movep_canfd.assert_not_called()
        unlimited = self.hardware(None)
        unlimited.command(action)
        unlimited.sdk.rm_movep_canfd.assert_called_once()
        unlimited.gripper.command.assert_called_once_with(0.5)

    def test_disabled_jump_check_keeps_soft_limits(self):
        hardware = self.hardware(None, solution=[179, 0, 0, 0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "关节软限位"):
            hardware.command((0.001, 0, 0, 0, 0, 0, 0.5))
        hardware.sdk.rm_movep_canfd.assert_not_called()

    def test_disabled_jump_check_keeps_ik_failure_and_nonfinite_rejection(self):
        for solution, code in (([0.0] * 7, 1), ([float("nan")] * 7, 0)):
            with self.subTest(code=code):
                hardware = self.hardware(None, solution=solution, ik_code=code)
                with self.assertRaises((RuntimeError, ValueError)):
                    hardware.command((0.001, 0, 0, 0, 0, 0, 0.5))
                hardware.sdk.rm_movep_canfd.assert_not_called()

    def test_disabled_jump_check_keeps_estop_latch(self):
        hardware = self.hardware(None)
        hardware.estop_latched = True
        with self.assertRaisesRegex(RuntimeError, "急停已锁存"):
            hardware.command((0.001, 0, 0, 0, 0, 0, 0.5))
        hardware.sdk.rm_movep_canfd.assert_not_called()

    def test_joint8_uses_radians_and_sdk_joint_command_without_ik(self):
        import math
        hardware=self.hardware(20.)
        hardware.config.update(action_space='joint8',trajectory_mode=2,trajectory_radio=17)
        hardware.sdk.rm_movej_canfd.return_value=0
        action=(math.radians(10),0.,0.,0.,0.,0.,0.,.6)
        hardware.command(action)
        hardware.sdk.rm_algo_inverse_kinematics.assert_not_called()
        hardware.sdk.rm_movep_canfd.assert_not_called()
        args,kwargs=hardware.sdk.rm_movej_canfd.call_args
        self.assertAlmostEqual(args[0][0],10.)
        self.assertEqual(kwargs,dict(expand=0.,trajectory_mode=2,radio=17))
        hardware.gripper.command.assert_called_once_with(.6)
        hardware.config['max_joint_step_deg']=5.
        with self.assertRaisesRegex(ValueError,'跳变过大'):hardware.command(action)
        hardware.config['max_joint_step_deg']=None
        with self.assertRaisesRegex(ValueError,'软限位'):hardware.command((math.pi,0,0,0,0,0,0,.6))

    def test_limit_requires_positive_finite_number_or_explicit_null(self):
        config = self.config()
        config.pop("max_joint_step_deg", None)
        self.assertEqual(RealManHardwareAdapter._validate_config(config)["max_joint_step_deg"], 8.0)
        for invalid in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                RealManHardwareAdapter._validate_config(dict(config, max_joint_step_deg=invalid))

    def test_motion_trace_distinguishes_model_target_from_sent_and_measured_pose(self):
        hardware = self.hardware(None)
        hardware.config["log_motion_trace"] = True
        action = (0.10, 0, 0, 0, 0, 0, 0.5)
        with self.assertLogs("realman_deploy.hardware", level="INFO") as logs:
            hardware.command(action)
        trace = json.loads(logs.output[0].split("Motion trace ", 1)[1])
        self.assertEqual(trace["requested_pose"], list(action[:6]))
        self.assertEqual(trace["measured_pose"], [0.0] * 6)
        self.assertEqual(trace["feedback_stamp"], 100.0)
        self.assertAlmostEqual(trace["sent_pose"][0], 0.10 / 30.0)
        self.assertEqual(trace["sent_pose"], hardware.sdk.rm_movep_canfd.call_args.args[0])

    def test_disabled_radius_accepts_outside_workspace_and_keeps_speed_limit(self):
        hardware = self.hardware(None)
        hardware.workspace_center = (0.0,) * 6
        current = [0.20, 0, 0, 0, 0, 0]
        hardware.feedback.snapshot.return_value = ([0.0] * 7, current, [0.0] * 6, 100.0)
        action = (0.30, 0, 0, 0, 0, 0, 0.5)
        with self.assertRaisesRegex(ValueError, "工作半径"):
            hardware.command(action)
        hardware.sdk.rm_movep_canfd.assert_not_called()
        hardware.config["workspace_radius_m"] = None
        hardware.command(action)
        sent = hardware.sdk.rm_movep_canfd.call_args.args[0]
        self.assertAlmostEqual(sent[0], 0.20 + 0.10 / 30)

    def test_disabled_radius_keeps_orientation_deviation_check(self):
        hardware = self.hardware(None)
        hardware.workspace_center = (0.0,) * 6
        hardware.config["workspace_radius_m"] = None
        pose = [0.20, 0, 0, 0.9, 0, 0]
        hardware.feedback.snapshot.return_value = ([0.0] * 7, pose, [0.0] * 6, 100.0)
        with self.assertRaisesRegex(ValueError, "姿态偏离"):
            hardware.command(tuple(pose) + (0.5,))
        hardware.sdk.rm_movep_canfd.assert_not_called()

    def test_radius_requires_positive_finite_number_or_null(self):
        config = self.config()
        config.pop("workspace_radius_m", None)
        self.assertEqual(RealManHardwareAdapter._validate_config(config)["workspace_radius_m"], 0.15)
        self.assertIsNone(RealManHardwareAdapter._validate_config(dict(config, workspace_radius_m=None))["workspace_radius_m"])
        for invalid in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                RealManHardwareAdapter._validate_config(dict(config, workspace_radius_m=invalid))


if __name__ == "__main__":
    unittest.main()
