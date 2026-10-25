"""Remote scheduling and local actuator guards, with a manual clock and fake I/O."""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realman_deploy.contracts import Observation
from realman_deploy.normalization import DatasetNormalizer
from realman_deploy.runtime import DeploymentRuntime


NORMALIZER = DatasetNormalizer({
    "state": {"mean": [0.] * 8, "std": [1.] * 8},
    "action": {"mean": [2.] * 7 + [0.], "std": [2.] * 7 + [1.]},
    "force": {"mean": [0.] * 6, "std": [1.] * 6},
}, action_space="joint8")


def observation(sequence, stamp, joint=1.):
    return Observation(stamp, sequence, (joint,) * 7 + (0.2,), (0.,) * 6,
                       {"global": b"rgb", "wrist": b"rgb"}, action_space="joint8")


class Hardware:
    def __init__(self):
        self.armed = False
        self.commands = []
        self.stops = 0

    def status(self):
        return dict(action_space="joint8", force_preprocessing=None,
                    robot_ready=True, cameras_ready=True, force_ready=True, gripper_ready=True)

    def arm(self):
        self.armed = True

    def slow_stop(self):
        self.armed = False
        self.stops += 1

    def emergency_stop(self):
        self.armed = False

    def command(self, action):
        assert self.armed
        self.commands.append(action)

    def close(self):
        pass


class RemoteRuntimeTests(unittest.TestCase):
    def runtime(self, **settings):
        now = [100.]
        settings.setdefault("execution_mode", "async_relaxed")
        settings.setdefault("inference_device", "qizhi_h200")
        runtime = DeploymentRuntime(NORMALIZER, clock=lambda: now[0], **settings)
        hardware = Hardware()
        policy = SimpleNamespace(requires_local_cuda=False, normalizer=NORMALIZER,
                                 reference_mode="none", close=Mock())
        runtime.attach(policy=policy, hardware=hardware)
        self.addCleanup(runtime.close)
        # Tests execute the actual publish/control paths, without background I/O.
        runtime._threads = [threading.current_thread()]
        runtime._cuda_status = Mock(return_value={"available": False})
        runtime.start()
        runtime.observations.put(observation(0, now[0]))
        return runtime, hardware, now

    @staticmethod
    def rows():
        return [[0.5] * 7 + [0.8] for _ in range(32)]

    @staticmethod
    def tick(runtime, now, stamp, joint=1.):
        now[0] = stamp
        last, _ = runtime.observations.get()
        runtime.observations.put(observation(last.sequence + 1, stamp, joint))
        runtime._control_tick()

    def test_remote_bypasses_only_local_cuda_requirement(self):
        runtime, hardware, _ = self.runtime()
        runtime._cuda_status.assert_not_called()
        self.assertTrue(hardware.armed)
        runtime.pause()
        for value in (True, None, 0, ""):
            runtime.policy.requires_local_cuda = value
            with self.assertRaisesRegex(RuntimeError, "CUDA 未就绪"):
                runtime.start()
        self.assertFalse(hardware.armed)

    def test_late_return_fails_closed(self):
        runtime, hardware, now = self.runtime()
        now[0] = 100.61
        epoch = runtime.gate.current()
        with self.assertRaisesRegex(RuntimeError, "返回时已过期") as caught:
            runtime._publish(self.rows(), observation(0, 100.), epoch, 100.)
        runtime._fail("inference", caught.exception, epoch)
        self.assertEqual(runtime.state, "error")
        self.assertFalse(hardware.commands)
        self.assertFalse(hardware.armed)
        self.assertEqual(runtime.discarded_results, 1)

    def test_accepted_result_must_be_fresh_at_first_command(self):
        runtime, hardware, now = self.runtime()
        now[0] = 100.2
        runtime._publish(self.rows(), observation(0, 100.), runtime.gate.current(), 100.)
        with self.assertRaisesRegex(RuntimeError, "首次执行时已过期"):
            self.tick(runtime, now, 100.61)
        self.assertFalse(hardware.commands)

    def test_deadline_does_not_truncate_started_full_action_chunk(self):
        runtime, hardware, now = self.runtime()
        now[0] = 100.2
        runtime._publish(self.rows(), observation(0, 100.), runtime.gate.current(), 100.)
        for i in range(32):
            self.tick(runtime, now, 100.2 + i / 30.)
        self.assertEqual(len(hardware.commands), 32)
        self.assertEqual(runtime.input_age_at_first_action_ms, 200.)
        self.assertGreater(now[0] - 100., runtime.max_result_age_s)

    def test_stale_pending_chunk_cannot_be_started_after_queue_wait(self):
        runtime, hardware, now = self.runtime()
        now[0] = 100.2
        runtime._publish(self.rows(), observation(0, 100.), runtime.gate.current(), 100.)
        self.tick(runtime, now, 100.2)
        now[0] = 100.5
        runtime._publish(self.rows(), observation(2, 100.3), runtime.gate.current(), 100.3)
        self.assertIsNotNone(runtime._pending_chunk)
        for i in range(1, 32):
            self.tick(runtime, now, 100.2 + i / 30.)
        with self.assertRaisesRegex(RuntimeError, "等待结束时已过期"):
            self.tick(runtime, now, 100.2 + 32 / 30. + 0.001)
        self.assertEqual(len(hardware.commands), 32)

    def test_fresh_pending_chunk_switches_without_stopping_between_chunks(self):
        runtime, hardware, now = self.runtime()
        now[0] = 100.2
        runtime._publish(self.rows(), observation(0, 100.), runtime.gate.current(), 100.)
        for i in range(31):
            self.tick(runtime, now, 100.2 + i / 30.)
        now[0] = 101.2
        runtime._publish(self.rows(), observation(40, 101.), runtime.gate.current(), 101.)
        self.tick(runtime, now, 100.2 + 31 / 30.)
        self.tick(runtime, now, 100.2 + 32 / 30. + 0.001)
        self.assertEqual(len(hardware.commands), 33)
        self.assertEqual(hardware.stops, 0)
        self.assertLess(runtime.input_age_at_first_action_ms, 300.)

    def test_prefetch_waits_for_late_chunk_observation(self):
        runtime, _, now = self.runtime()
        now[0] = 100.2
        runtime._publish(self.rows(), observation(0, 100.), runtime.gate.current(), 100.)
        self.assertFalse(runtime._remote_prefetch_ready(now[0]))
        for i in range(25):
            self.tick(runtime, now, 100.2 + i / 30.)
        self.assertTrue(runtime._remote_prefetch_ready(now[0]))
        self.assertLess(runtime._remote_prefetch_lead_s(), runtime.max_result_age_s)

    def test_inference_worker_defers_then_uses_latest_observation_once(self):
        runtime, _, now = self.runtime()
        now[0] = 100.2
        runtime._publish(self.rows(), observation(0, 100.), runtime.gate.current(), 100.)
        checked, entered, release = threading.Event(), threading.Event(), threading.Event()
        original_ready = runtime._remote_prefetch_ready
        calls = []

        def ready(stamp):
            result = original_ready(stamp)
            checked.set()
            return result

        def infer(obs, reference, history):
            calls.append(obs.sequence)
            entered.set()
            release.wait(2.)
            return self.rows()

        runtime._remote_prefetch_ready = ready
        runtime.policy.infer = infer
        runtime._history.append(observation(0, now[0]).state, (0.,) * 6)
        runtime._input_history = runtime._history.tensors()
        runtime.observations.put(observation(1, now[0]))
        worker = threading.Thread(target=runtime._inference_loop)
        worker.start()
        try:
            runtime.wake.set()
            self.assertTrue(checked.wait(1.))
            self.assertFalse(entered.is_set(), "early observations must not start remote inference")
            with runtime.lock:
                for i in range(25):
                    self.tick(runtime, now, 100.2 + i / 30.)
                runtime.observations.put(observation(100, now[0]))
            runtime.wake.set()
            self.assertTrue(entered.wait(1.))
            with runtime.lock:
                runtime.observations.put(observation(101, now[0]))
                runtime.observations.put(observation(102, now[0]))
            runtime.wake.set()
            runtime.stop_event.set()
            release.set()
            worker.join(1.)
            self.assertFalse(worker.is_alive())
            self.assertEqual(calls, [100])
            self.assertEqual(runtime._pending_chunk.source_sequence, 100)
        finally:
            runtime.stop_event.set()
            release.set()
            runtime.wake.set()
            worker.join(2.)

    def test_smoothing_uses_decoded_joints_measured_bootstrap_and_unfiltered_gripper(self):
        runtime, hardware, now = self.runtime(action_smoothing_alpha=0.25)
        now[0] = 100.2
        runtime._publish(self.rows(), observation(0, 100.), runtime.gate.current(), 100.)
        self.tick(runtime, now, 100.2)
        self.tick(runtime, now, 100.2 + 1 / 30.)
        self.assertAlmostEqual(hardware.commands[0][0], 1.5, places=5)
        self.assertAlmostEqual(hardware.commands[1][0], 1.875, places=5)
        self.assertAlmostEqual(hardware.commands[0][-1], 0.8, places=5)
        runtime.pause()
        self.assertIsNone(runtime._smoothed_action)
        self.assertIsNone(runtime.input_age_at_first_action_ms)
        runtime.start()
        now[0] = 102.
        runtime.observations.put(observation(50, now[0], joint=2.))
        runtime._publish(self.rows(), observation(50, now[0], joint=2.), runtime.gate.current(), now[0])
        self.tick(runtime, now, 102., joint=2.)
        self.assertAlmostEqual(hardware.commands[-1][0], 2.25, places=5)
        runtime.emergency_stop()
        self.assertIsNone(runtime._smoothed_action)

    def test_configuration_persistence_failure_leaves_existing_policy_and_settings(self):
        runtime, _, _ = self.runtime()
        runtime.pause()
        policy = runtime.policy
        with self.assertRaises(OSError):
            runtime.configure_sampling("4416", "synchronous", inference_device="local",
                                       action_smoothing_alpha=0.5,
                                       persist=Mock(side_effect=OSError("read only")))
        self.assertIs(runtime.policy, policy)
        self.assertEqual(runtime.inference_device, "qizhi_h200")
        self.assertEqual(runtime.action_smoothing_alpha, 1.)
        policy.close.assert_not_called()
        runtime.configure_sampling("4416", "synchronous", inference_device="local",
                                   action_smoothing_alpha=0.5)
        self.assertIsNone(runtime.policy)
        self.assertIsNone(runtime.max_result_age_s)
        self.assertEqual(runtime.action_smoothing_alpha, 0.5)
        policy.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
