"""Deterministic measured-motion barriers; no robot, network, or wall-clock waits."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from realman_sync_client.client import (
    ClientOptions, MotionNotSettledError, ObservationDriftError, SynchronousClient,
)
from realman_sync_client.mock import MockHardware


META = dict(camera_names=["global", "wrist"], width=4, height=3,
            action_space="joint8", action_horizon=32, history_length=32,
            control_hz=30, video_context_frames=2, video_context_stride=2)


class VirtualClock:
    def __init__(self):
        self.time = 100.0

    def now(self):
        return self.time


class MeasuredHardware(MockHardware):
    def __init__(self, timer):
        super().__init__(width=4, height=3)
        self.timer = timer
        self.arm_count = 0
        self.feedback_values = (0.,) * 7
        self.feedback_age_s = 0.
        self.drift_on_arm = 0.

    def joint_feedback(self):
        return self.feedback_values, self.timer.time-self.feedback_age_s

    def arm(self):
        self.arm_count += 1
        super().arm()
        self.feedback_values = (math.radians(self.drift_on_arm),) + (0.,) * 6


class MotionFixture(unittest.TestCase):
    def setUp(self):
        self.timer = VirtualClock()
        self.clock_patch = patch("realman_sync_client.client.clock.now", self.timer.now)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)

    def make_client(self, **options):
        hardware = MeasuredHardware(self.timer)
        client = SynchronousClient(None, hardware, META,
            options=ClientOptions(execute_steps=1, **options))
        self.prime(client)
        return client, hardware

    def append(self, client, degrees=0., gripper=.5):
        observation = client.hardware.observe_at(self.timer.time)
        observation = replace(observation, state=(math.radians(degrees),) + (0.,)*6 + (gripper,))
        client.collector._append(observation, self.timer.time)
        return observation

    def prime(self, client):
        for _ in range(32):
            self.timer.time += 1/30
            self.append(client)

    def driver(self, client, position, *, stop_at=None):
        count = 0
        def advance(_timeout):
            nonlocal count
            count += 1
            self.timer.time += 1/30
            self.append(client, position(count))
            if stop_at == count:
                client.stop_event.set()
        return advance


class SettlingTests(MotionFixture):
    def test_stop_ack_requires_a_complete_stationary_window_after_motion(self):
        client, hardware = self.make_client()
        after = self.timer.time
        old_sequence = client.collector.history[-1].sequence
        # The stop command is already acknowledged, but measured q continues
        # moving for 300 ms. Its entire subsequent stable window must be real.
        drive = self.driver(client, lambda step: min(step, 9)*.1)
        with patch.object(client.collector.condition, "wait", side_effect=drive):
            obs, history, video = client.snapshot_for_inference(after, old_sequence)
        self.assertGreaterEqual(obs.stamp-after, .5-1e-9)
        self.assertAlmostEqual(math.degrees(obs.state[0]), .9)
        self.assertEqual(len(history), 32)
        self.assertEqual(len(video), 2)
        self.assertIs(video[-1], obs)
        self.assertGreater(client.last_settle_metrics["window_s"], .199999)
        self.assertEqual(hardware.commands, [])
        self.assertEqual(hardware.arm_count, 0)

    def test_old_stationary_history_cannot_satisfy_new_stop_boundary(self):
        client, _ = self.make_client()
        after = self.timer.time
        with patch.object(client.collector.condition, "wait",
                          side_effect=self.driver(client, lambda _: 0.)) as wait:
            obs, _, _ = client.snapshot_for_inference(after, -1)
        self.assertGreaterEqual(obs.stamp-after, .2)
        self.assertGreaterEqual(wait.call_count, 7)

    def test_small_excursion_with_excess_velocity_never_passes(self):
        client, hardware = self.make_client(settle_timeout_s=.5)
        after = self.timer.time
        drive = self.driver(client, lambda step: .02 if step % 2 else -.02)
        with patch.object(client.collector.condition, "wait", side_effect=drive):
            with self.assertRaises(MotionNotSettledError):
                client.snapshot_for_inference(after, -1)
        self.assertLess(client.last_settle_metrics["peak_to_peak_deg"][0], .1)
        self.assertGreater(client.last_settle_metrics["max_velocity_deg_s"][0], .5)
        self.assertEqual(hardware.commands, [])
        self.assertFalse(MotionNotSettledError.recoverable)

    def test_peak_to_peak_rejects_slow_creep_even_below_velocity_limit(self):
        client, _ = self.make_client(settle_window_s=.5, settle_timeout_s=.9)
        drive = self.driver(client, lambda step: step*.3/30)
        with patch.object(client.collector.condition, "wait", side_effect=drive):
            with self.assertRaises(MotionNotSettledError):
                client.snapshot_for_inference(self.timer.time, -1)
        self.assertGreater(client.last_settle_metrics["peak_to_peak_deg"][0], .1)
        self.assertLess(client.last_settle_metrics["max_velocity_deg_s"][0], .5)

    def test_stop_interrupts_settling_without_arm_or_command(self):
        client, hardware = self.make_client()
        drive = self.driver(client, lambda step: step, stop_at=2)
        with patch.object(client.collector.condition, "wait", side_effect=drive):
            with self.assertRaises(InterruptedError):
                client.snapshot_for_inference(self.timer.time, -1)
        self.assertEqual(hardware.arm_count, 0)
        self.assertEqual(hardware.commands, [])

    def test_large_window_uses_measured_buffer_without_expanding_model_history(self):
        client, _ = self.make_client(settle_window_s=1.5, settle_timeout_s=3.)
        after = self.timer.time
        with patch.object(client.collector.condition, "wait",
                          side_effect=self.driver(client, lambda _: 0.)):
            _, history, _ = client.snapshot_for_inference(after, -1)
        self.assertEqual(len(history), 32)
        self.assertGreaterEqual(client.last_settle_metrics["window_s"], 1.5)
        self.assertLessEqual(client.collector.stability_history.maxlen, 152)

    def test_bad_new_options_rejected_before_hardware_access(self):
        for name in ("settle_window_s", "settle_tolerance_deg", "settle_velocity_deg_s",
                     "settle_timeout_s", "max_inference_drift_deg"):
            for value in (False, True, 0, -1, float("nan"), float("inf"), "1"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    ClientOptions(**{name: value})
        with self.assertRaises(ValueError):
            ClientOptions(settle_window_s=5.01)

    def test_sensor_warmup_keeps_its_own_budget_before_settle_timeout(self):
        client, _ = self.make_client(sensor_timeout_s=10., settle_timeout_s=.5)
        collector = client.collector
        collector.history.clear()
        collector.video.clear()
        collector.stability_history.clear()
        collector.started = False
        after = self.timer.time
        steps = 0
        def warmup(_timeout):
            nonlocal steps
            steps += 1
            self.timer.time += 1/30
            if steps > 180:  # Six seconds of camera startup, then real data.
                self.append(client)
        with patch.object(collector.condition, "wait", side_effect=warmup):
            obs, history, _ = client.snapshot_for_inference(after, -1)
        self.assertGreater(obs.stamp-after, 7.)
        self.assertEqual(len(history), 32)

    def test_explicit_timeout_caps_sensor_startup(self):
        client, _ = self.make_client(sensor_timeout_s=10.)
        client.collector.history.clear()
        client.collector.video.clear()
        def advance(_timeout):
            self.timer.time += .05
        with patch.object(client.collector.condition, "wait", side_effect=advance):
            with self.assertRaisesRegex(TimeoutError, "observation history"):
                client.snapshot_for_inference(self.timer.time, -1, timeout_s=.1)


class DriftGuardTests(MotionFixture):
    def test_inference_drift_is_rejected_before_arm(self):
        client, hardware = self.make_client()
        source = client.collector.history[-1]
        hardware.feedback_values = (math.radians(.6),) + (0.,)*6
        with self.assertRaises(ObservationDriftError):
            client._execute([source.state]*32, source_observation=source)
        self.assertEqual(hardware.arm_count, 0)
        self.assertEqual(hardware.commands, [])
        self.assertFalse(ObservationDriftError.recoverable)
        self.assertAlmostEqual(client.last_execution_guard["max_drift_deg"], .6)

    def test_drift_during_arm_is_rechecked_before_first_command(self):
        client, hardware = self.make_client()
        source = client.collector.history[-1]
        hardware.drift_on_arm = 1.
        with self.assertRaises(ObservationDriftError):
            client._execute([source.state]*32, source_observation=source)
        self.assertEqual(hardware.arm_count, 1)
        self.assertEqual(hardware.commands, [])
        self.assertEqual(client.last_execution_guard["phase"], "before_first_command")

    def test_stop_precedes_arm(self):
        client, hardware = self.make_client()
        client.stop_event.set()
        with self.assertRaises(InterruptedError):
            client._execute([client.collector.history[-1].state]*32)
        self.assertEqual(hardware.arm_count, 0)
        self.assertEqual(hardware.commands, [])

    def test_missing_request_source_cannot_bypass_guard(self):
        client, hardware = self.make_client()
        with self.assertRaisesRegex(ValueError, "source observation"):
            client._execute([client.collector.history[-1].state]*32)
        self.assertEqual(hardware.arm_count, 0)

    def test_request_source_fallback_and_gripper_change_preserve_joint_actions(self):
        client, hardware = self.make_client()
        source = client.collector.history[-1]
        client._request(source, tuple(client.collector.history), tuple(client.collector.video)[::2])
        hardware.state = (0.,)*7 + (1.,)
        action = (0.,)*7 + (.1,)
        with patch.object(client, "_wait_control"):
            client._execute([action]*32)
        self.assertEqual(hardware.commands[0][1], action)
        self.assertEqual(client.last_execution_guard["max_drift_deg"], 0.)
        self.assertEqual(hardware.stop_count, 1)

    def test_stale_or_nonfinite_feedback_is_rejected(self):
        for kind in ("stale", "nan", "short"):
            client, hardware = self.make_client()
            source = client.collector.history[-1]
            if kind == "stale":
                hardware.feedback_age_s = .2
            elif kind == "nan":
                hardware.feedback_values = (float("nan"),) + (0.,)*6
            else:
                hardware.feedback_values = (0.,)*6
            with self.subTest(kind=kind), self.assertRaises((ValueError, RuntimeError)):
                client._execute([source.state]*32, source_observation=source)
            self.assertEqual(hardware.arm_count, 0)
            self.assertEqual(hardware.commands, [])


if __name__ == "__main__":
    unittest.main()
