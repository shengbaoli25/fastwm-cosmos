from __future__ import annotations

import math
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy"))

from realman_deploy.contracts import Observation
from realman_deploy.model_adapter import CosmosJointPolicy
from realman_deploy.normalization import DatasetNormalizer
from realman_deploy.reference import RollingActionReference
from realman_deploy.runtime import DeploymentRuntime


NORMALIZER = DatasetNormalizer({
    "state": {"mean": [0.0] * 7, "std": [1.0] * 7},
    "action": {"mean": [0.0] * 7, "std": [1.0] * 7},
    "force": {"mean": [0.0] * 6, "std": [1.0] * 6},
})


def observation(sequence: int, stamp: float | None = None) -> Observation:
    return Observation(stamp or time.monotonic(), sequence, (0.11, 0.12, 0.13, 0.0, 0.0, 0.0, 0.5),
                      (0.0,) * 6, {"global": b"rgb", "wrist": b"rgb"})


class MockHardware:
    def __init__(self):
        self.sequence = 0
        self.armed = False
        self.commands = []
        self.slow_stops = 0
        self.estops = 0
        self.closed = False

    def status(self):
        return {"robot_ready": True, "cameras_ready": True, "force_ready": True,
                "gripper_ready": True}

    def arm(self):
        if self.closed:
            raise RuntimeError("closed")
        self.armed = True

    def observe(self):
        result = observation(self.sequence)
        self.sequence += 1
        return result

    def command(self, action):
        if not self.armed:
            raise RuntimeError("not armed")
        self.commands.append(action)

    def slow_stop(self):
        self.armed = False
        self.slow_stops += 1

    def emergency_stop(self):
        self.armed = False
        self.estops += 1

    def close(self):
        self.closed = True


class MockPolicy:
    def __init__(self, *, blocked: threading.Event | None = None):
        self.references = []
        self.histories = []
        self.closed = False
        self.blocked = blocked

    def infer(self, obs, normalized_reference, history):
        self.references.append([row[:] for row in normalized_reference])
        self.histories.append(history)
        if self.blocked:
            self.blocked.wait(3.0)
        return [[0.10 + i / 1000.0, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5 + i / 1000.0]
                for i in range(32)]

    def close(self):
        self.closed = True


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


class RollingReferenceTests(unittest.TestCase):
    def test_first_chunk_holds_current_pose_and_later_chunk_shifts_elapsed_slots(self):
        ref = RollingActionReference()
        first = observation(0, stamp=100.0)
        held = ref.get(first, NORMALIZER)
        for row in held:
            for got, want in zip(row, first.state):
                self.assertAlmostEqual(got, want, places=6)
        predicted = [[step / 10.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] for step in range(32)]
        ref.update(predicted, source_stamp=100.0)
        later = observation(1, stamp=100.1)
        rolled = ref.get(later, NORMALIZER)
        expected = predicted[3:] + [predicted[-1]] * 3
        for got, want in zip(rolled, expected):
            for x, y in zip(got, want):
                self.assertAlmostEqual(x, y, places=5)

    def test_reason_prompt_index_uses_manifest_order(self):
        manifest = {"texts": [{"prompt": "z_task"}, {"prompt": "a_task"}]}
        self.assertEqual(CosmosJointPolicy.prompt_index(manifest, "z_task"), 0)
        self.assertEqual(CosmosJointPolicy.prompt_index(manifest, "a_task"), 1)

    def test_dataset_normalization_matches_zscore_and_rejects_nan(self):
        n = DatasetNormalizer({
            "state": {"mean": [0.5] * 7, "std": [2.0] * 7},
            "action": {"mean": [0.5] * 7, "std": [2.0] * 7},
            "force": {"mean": [0.5] * 6, "std": [2.0] * 6},
        })
        self.assertAlmostEqual(n.encode("action", [2.5] * 7)[0], 1.0, places=6)
        self.assertAlmostEqual(n.decode("action", [1.0] * 7)[0], 2.5, places=6)
        with self.assertRaises(ValueError):
            n.encode("action", [math.nan] * 7)


class RuntimeTests(unittest.TestCase):
    def make_runtime(self, policy=None):
        runtime = DeploymentRuntime(NORMALIZER, require_cuda=False)
        hardware = MockHardware()
        policy = policy or MockPolicy()
        runtime.attach(policy=policy, hardware=hardware)
        return runtime, policy, hardware

    def test_no_reference_runtime_never_reads_or_updates_rolling_actions(self):
        class Policy(MockPolicy):
            reference_mode = 'none'

            def infer(self, obs, normalized_reference, history):
                self.references.append(normalized_reference)
                return [list(obs.state)] * 32

        runtime, policy, _ = self.make_runtime(Policy())
        runtime.reference.get = Mock(side_effect=AssertionError('No rolling input'))
        runtime.reference.update = Mock(side_effect=AssertionError('No rolling update'))
        try:
            runtime.start()
            self.assertTrue(wait_until(lambda: runtime.inference_count >= 2))
            runtime.pause()
            self.assertTrue(all(value is None for value in policy.references))
            self.assertEqual(runtime.status()['reference_mode'], 'none')
            self.assertIsNone(runtime.status()['bootstrap'])
        finally:
            runtime.close()

    def test_real_async_loop_rolls_ae_chunk_into_next_world_model_input(self):
        runtime, policy, hardware = self.make_runtime()
        try:
            runtime.start()
            self.assertTrue(wait_until(lambda: len(policy.references) >= 2 and hardware.commands))
            runtime.pause()
            first, second = policy.references[:2]
            current = (0.11, 0.12, 0.13, 0.0, 0.0, 0.0, 0.5)
            for row in first:
                for got, want in zip(row, current):
                    self.assertAlmostEqual(got, want, places=5)
            # 30 Hz observations advance one or two slots while an inference is
            # running; the next action condition is a time-aligned AE prediction.
            self.assertGreaterEqual(second[0][0], 0.10)
            self.assertLessEqual(second[0][0], 0.13)
            self.assertEqual(len(policy.histories[0][0]), 32)
            self.assertTrue(hardware.commands)
            self.assertEqual(hardware.slow_stops, 1)
        finally:
            runtime.close()

    def test_result_finishing_after_pause_is_discarded_and_cannot_reseed_reference(self):
        release = threading.Event()
        runtime, policy, _ = self.make_runtime(MockPolicy(blocked=release))
        try:
            runtime.start()
            self.assertTrue(wait_until(lambda: runtime._infer_active))
            runtime.pause()
            release.set()
            self.assertTrue(wait_until(lambda: not runtime._infer_active))
            self.assertTrue(runtime.reference.bootstrap())
            self.assertEqual(runtime.status()["action_queue"], 0)
            self.assertEqual(runtime.state, "paused")
            self.assertGreaterEqual(runtime.discarded_results, 1)
            self.assertIsNotNone(runtime.status()["latency_ms"])
        finally:
            release.set()
            runtime.close()

    def test_emergency_stop_is_latched(self):
        runtime, _, hardware = self.make_runtime()
        try:
            runtime.start()
            runtime.emergency_stop()
            with self.assertRaisesRegex(RuntimeError, "急停已锁存"):
                runtime.start()
            self.assertEqual(hardware.estops, 1)
            self.assertFalse(hardware.armed)
        finally:
            runtime.close()


class RelaxedRuntimeTests(unittest.TestCase):
    def manual_runtime(self, mode="async_relaxed"):
        now = [100.0]
        runtime = DeploymentRuntime(NORMALIZER, require_cuda=False,
                                    execution_mode=mode, clock=lambda: now[0])
        hardware = MockHardware()
        runtime.attach(hardware=hardware)
        # Drive the same publish/control paths deterministically without threads.
        runtime.state = "running"
        runtime._running_since = now[0]
        hardware.arm()
        runtime.observations.put(observation(0, stamp=now[0]))
        return runtime, hardware, now

    def rows(self, start=0.1):
        return [[start + i / 1000, 0.2, 0.3, 0, 0, 0, 0.5] for i in range(32)]

    def tick(self, runtime, now, stamp):
        now[0] = stamp
        last, _ = runtime.observations.get()
        runtime.observations.put(observation(last.sequence + 1, stamp=stamp))
        runtime._control_tick()

    def test_delayed_publish_anchors_rolling_reference_to_execution(self):
        runtime, hardware, now = self.manual_runtime()
        try:
            source = observation(0, stamp=100.0)
            now[0] = 101.5
            rows = self.rows()
            self.assertTrue(runtime._publish(rows, source, runtime.gate.current(), 100.0))
            self.assertEqual(runtime.input_age_at_result_ms, 1500.0)
            self.assertEqual(runtime.latency_ms, 1500.0)
            self.assertEqual(len(runtime.timeline), 32)
            reference = runtime.reference.get(observation(1, stamp=101.5), NORMALIZER)
            self.assertAlmostEqual(reference[0][0], rows[0][0], places=5)
            reference = runtime.reference.get(observation(2, stamp=101.6), NORMALIZER)
            self.assertAlmostEqual(reference[0][0], rows[3][0], places=5)
            self.tick(runtime, now, 101.5)
            self.assertAlmostEqual(hardware.commands[0][0], rows[0][0], places=5)
        finally:
            runtime.close()

    def test_strict_runtime_still_rejects_expired_chunk(self):
        runtime, _, now = self.manual_runtime("timestamped")
        try:
            now[0] = 101.5
            with self.assertRaisesRegex(RuntimeError, "仅剩 0 个动作"):
                runtime._publish(self.rows(), observation(0, stamp=100.0), runtime.gate.current(), 100.0)
            self.assertTrue(runtime.reference.bootstrap())
            self.assertEqual(len(runtime.timeline), 0)
        finally:
            runtime.close()

    def test_empty_queue_stops_once_and_later_result_resumes(self):
        runtime, hardware, now = self.manual_runtime()
        try:
            self.tick(runtime, now, 100.01)
            self.tick(runtime, now, 100.05)
            self.assertEqual(hardware.slow_stops, 1)
            self.assertEqual(runtime.state, "running")
            self.assertFalse(hardware.armed)
            now[0] = 101.5
            runtime._publish(self.rows(), observation(0, stamp=100.0), runtime.gate.current(), 100.0)
            for i in range(32):
                self.tick(runtime, now, 101.5 + i / 30)
            self.assertEqual(len(hardware.commands), 32)
            self.assertTrue(hardware.armed)
            # Last issued action gets a full slot before stopping.
            self.tick(runtime, now, 101.5 + 31 / 30 + 0.01)
            self.assertEqual(hardware.slow_stops, 1)
            self.tick(runtime, now, 102.6)
            self.tick(runtime, now, 102.7)
            self.assertEqual(hardware.slow_stops, 2)
            self.assertEqual(runtime.state, "running")
            now[0] = 103.0
            runtime._publish(self.rows(0.4), observation(100, stamp=101.5), runtime.gate.current(), 101.5)
            self.tick(runtime, now, 103.0)
            self.assertEqual(len(hardware.commands), 33)
            self.assertAlmostEqual(hardware.commands[-1][0], 0.4, places=5)
            self.assertTrue(hardware.armed)
        finally:
            runtime.close()

    def test_waiting_does_not_disable_live_sensor_freshness(self):
        runtime, hardware, now = self.manual_runtime()
        try:
            self.tick(runtime, now, 100.01)
            now[0] = 100.5
            try:
                runtime._control_tick()
            except RuntimeError as exc:
                self.assertIn("观测中断/过期", str(exc))
                runtime._fail("control", exc, runtime.gate.current())
            else:
                self.fail("stale live sensor must fail even while waiting")
            self.assertEqual(runtime.state, "error")
            self.assertFalse(hardware.armed)
            self.assertFalse(hardware.commands)
        finally:
            runtime.close()

    def test_pause_or_estop_discards_late_result_and_reference(self):
        for stop in ("pause", "emergency_stop"):
            with self.subTest(stop=stop):
                runtime, hardware, now = self.manual_runtime()
                try:
                    epoch = runtime.gate.current()
                    getattr(runtime, stop)()
                    now[0] = 101.5
                    self.assertFalse(runtime._publish(self.rows(), observation(0, stamp=100.0), epoch, 100.0))
                    runtime._control_tick()
                    self.assertTrue(runtime.reference.bootstrap())
                    self.assertEqual(len(runtime.timeline), 0)
                    self.assertFalse(hardware.commands)
                    self.assertFalse(hardware.armed)
                    self.assertEqual(runtime.discarded_results, 1)
                finally:
                    runtime.close()

    def test_pause_during_rearm_cannot_dispatch_action(self):
        runtime, hardware, now = self.manual_runtime()
        entered, release = threading.Event(), threading.Event()
        tick_thread = pause_thread = None
        try:
            self.tick(runtime, now, 100.01)
            now[0] = 101.5
            runtime._publish(self.rows(), observation(0, stamp=100.0), runtime.gate.current(), 100.0)
            original_arm = hardware.arm
            def blocked_arm():
                entered.set()
                release.wait(1.0)
                original_arm()
            hardware.arm = blocked_arm
            tick_thread = threading.Thread(target=lambda: self.tick(runtime, now, 101.5))
            tick_thread.start()
            self.assertTrue(entered.wait(1.0))
            pause_thread = threading.Thread(target=runtime.pause)
            pause_thread.start()
            self.assertTrue(wait_until(lambda: runtime.state == "stopping", timeout=0.5))
            release.set()
            tick_thread.join(1.0)
            pause_thread.join(1.0)
            self.assertEqual(runtime.state, "paused")
            self.assertFalse(hardware.commands)
            self.assertFalse(hardware.armed)
        finally:
            release.set()
            for thread in (tick_thread, pause_thread):
                if thread:
                    thread.join(1.0)
            runtime.close()

    def test_slow_async_loop_executes_previous_chunk_during_next_inference(self):
        class SlowPolicy(MockPolicy):
            def infer(self, obs, normalized_reference, history):
                rows = super().infer(obs, normalized_reference, history)
                time.sleep(1.5)
                return rows

        runtime = DeploymentRuntime(NORMALIZER, require_cuda=False, execution_mode="async_relaxed")
        hardware, policy = MockHardware(), SlowPolicy()
        runtime.attach(hardware=hardware, policy=policy)
        try:
            runtime.start()
            self.assertTrue(wait_until(lambda: len(policy.references) >= 2 and len(hardware.commands) >= 5))
            self.assertTrue(runtime._infer_active)
            self.assertEqual(runtime.inference_count, 1)
            self.assertTrue(wait_until(lambda: len(hardware.commands) == 32 and runtime._relaxed_holding,
                                       timeout=2.0))
            self.assertEqual(runtime.state, "running")
            self.assertTrue(wait_until(lambda: runtime.inference_count >= 2 and len(hardware.commands) > 32,
                                       timeout=2.0))
            self.assertEqual(runtime.discarded_results, 0)
            self.assertIsNone(runtime.error)
            self.assertGreaterEqual(runtime.latency_ms, 1490)
            runtime.pause()
            count = len(hardware.commands)
            self.assertTrue(wait_until(lambda: not runtime._infer_active))
            self.assertEqual(len(hardware.commands), count)
            self.assertTrue(runtime.reference.bootstrap())
        finally:
            runtime.close()

    def test_fast_async_keeps_one_completed_next_chunk_until_boundary(self):
        class FastPolicy(MockPolicy):
            pass

        runtime = DeploymentRuntime(NORMALIZER, require_cuda=False, execution_mode="async_relaxed")
        hardware, policy = MockHardware(), FastPolicy()
        runtime.attach(hardware=hardware, policy=policy)
        try:
            runtime.start()
            self.assertTrue(wait_until(lambda: runtime._pending_chunk and len(hardware.commands) >= 3))
            # A sub-second model must not repeatedly replace the active action
            # horizon with results from every 30 Hz camera frame.
            calls = len(policy.references)
            self.assertGreaterEqual(calls, 2)
            time.sleep(0.2)
            self.assertEqual(len(policy.references), calls)
            self.assertTrue(runtime._pending_chunk)
            self.assertTrue(wait_until(lambda: runtime.inference_count > calls and len(hardware.commands) > 32,
                                       timeout=2.5))
            self.assertIsNone(runtime.error)
            runtime.pause()
        finally:
            runtime.close()


class SynchronousRuntimeTests(unittest.TestCase):
    def test_full_chunk_finishes_before_next_post_stop_observation_and_inference(self):
        hardware = MockHardware()
        release_stop = threading.Event()
        stop_started = threading.Event()
        acknowledged = []
        original_stop = hardware.slow_stop

        def slow_stop():
            stop_started.set()
            release_stop.wait(1.0)
            original_stop()
            acknowledged.append(time.monotonic())

        hardware.slow_stop = slow_stop

        class RecordedPolicy(MockPolicy):
            def __init__(self):
                super().__init__()
                self.calls = []
                self.overlap = False

            def infer(self, obs, reference, history):
                count = len(hardware.commands)
                self.calls.append((count, obs.stamp, acknowledged[-1], hardware.armed))
                result = super().infer(obs, reference, history)
                time.sleep(0.15)
                self.overlap |= len(hardware.commands) != count or hardware.armed
                return result

        runtime = DeploymentRuntime(NORMALIZER, require_cuda=False, execution_mode="synchronous")
        policy = RecordedPolicy()
        runtime.attach(hardware=hardware, policy=policy)
        try:
            runtime.start()
            self.assertTrue(stop_started.wait(1.0))
            time.sleep(0.08)
            self.assertFalse(policy.calls, "inference cannot begin before stop acknowledgement")
            release_stop.set()
            self.assertTrue(wait_until(lambda: len(policy.calls) >= 2))
            self.assertEqual([call[0] for call in policy.calls[:2]], [0, 32])
            for count, stamp, stopped_at, armed in policy.calls[:2]:
                self.assertGreater(stamp, stopped_at)
                self.assertFalse(armed)
            self.assertFalse(policy.overlap)
            self.assertEqual(runtime.status()["execution_phase"], "inferring")
            runtime.pause()
            self.assertTrue(wait_until(lambda: not runtime._infer_active))
            self.assertEqual(len(hardware.commands), 32)
            self.assertEqual(runtime.inference_count, 1)
            self.assertEqual(runtime.discarded_results, 1)
            self.assertTrue(runtime.reference.bootstrap())
        finally:
            release_stop.set()
            runtime.close()

    def test_pause_or_estop_during_inference_invalidates_pending_chunk(self):
        for stop in ("pause", "emergency_stop"):
            with self.subTest(stop=stop):
                release = threading.Event()
                hardware = MockHardware()
                policy = MockPolicy(blocked=release)
                runtime = DeploymentRuntime(NORMALIZER, require_cuda=False, execution_mode="synchronous")
                runtime.attach(policy=policy, hardware=hardware)
                try:
                    runtime.start()
                    self.assertTrue(wait_until(lambda: runtime._infer_active))
                    self.assertFalse(hardware.armed)
                    getattr(runtime, stop)()
                    release.set()
                    self.assertTrue(wait_until(lambda: not runtime._infer_active))
                    self.assertFalse(hardware.commands)
                    self.assertEqual(len(runtime.timeline), 0)
                    self.assertTrue(runtime.reference.bootstrap())
                    self.assertEqual(runtime.discarded_results, 1)
                finally:
                    release.set()
                    runtime.close()

    def test_stop_failure_cannot_start_inference(self):
        hardware = MockHardware()
        hardware.slow_stop = Mock(side_effect=RuntimeError("stop acknowledgement failed"))
        policy = MockPolicy()
        runtime = DeploymentRuntime(NORMALIZER, require_cuda=False, execution_mode="synchronous")
        runtime.attach(policy=policy, hardware=hardware)
        try:
            runtime.start()
            self.assertTrue(wait_until(lambda: runtime.state == "error"))
            self.assertFalse(policy.references)
            self.assertFalse(hardware.commands)
        finally:
            hardware.slow_stop = lambda: None
            runtime.close()


if __name__ == "__main__":
    unittest.main()
