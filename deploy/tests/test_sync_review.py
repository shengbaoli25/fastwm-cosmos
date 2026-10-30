"""GPU-free regression checks for the additive Cosmos wire adapter."""
from pathlib import Path
import copy
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realman_deploy.normalization import DatasetNormalizer
from realman_sync.cosmos_policy import CosmosPolicy
from realman_sync.mock_policy import MockPolicy
from realman_sync.server import Conflict, InferenceService
from realman_sync.wire import PROTOCOL, decode_request, encode_observation


class CosmosAdapterReviewTests(unittest.TestCase):
    def setUp(self):
        self.normalizer = DatasetNormalizer({
            "state": {"mean": [1.] * 7 + [.5], "std": [2.] * 8},
            "action": {"mean": [2.] * 7 + [.5], "std": [4.] * 8},
            "force": {"mean": [3.] * 6, "std": [2.] * 6},
        }, action_space="joint8")
        self.output = [[.25] * 7 + [2.] for _ in range(32)]
        self.inner = SimpleNamespace(normalizer=self.normalizer,
                                     infer=Mock(return_value=self.output), close=Mock())
        self.policy = CosmosPolicy.__new__(CosmosPolicy)
        self.policy.policy = self.inner
        self.history = [dict(sequence=i, stamp=100 + i / 30,
                             state=[1. + i] * 7 + [.5], force=[3. + i] * 6,
                             force_meta={}) for i in range(2)]
        self.observation = dict(self.history[-1], action_space="joint8",
            frames={name: dict(data=b"rgb", width=1, height=1)
                    for name in ("global", "wrist")}, frame_meta={})

    def test_raw_history_normalized_once_with_zero_padded_mask(self):
        actions = self.policy.predict(self.observation, self.history, [self.observation])
        obs, reference, assembled = self.inner.infer.call_args.args
        self.assertIsNone(reference)
        self.assertEqual(obs.state, tuple(self.history[-1]["state"]))
        self.assertEqual(obs.frames, {"global": b"rgb", "wrist": b"rgb"})
        rows, pads, channels = assembled
        self.assertEqual(rows[:30], [[0.] * 14] * 30)
        self.assertEqual(pads, [True] * 30 + [False] * 2)
        self.assertEqual(channels, [[False] * 14] * 30 + [[True] * 14] * 2)
        for i, raw in enumerate(self.history):
            self.assertEqual(rows[30 + i], self.normalizer.encode("state", raw["state"])
                             + self.normalizer.encode("force", raw["force"]))
        expected = self.normalizer.decode("action", self.output)
        self.assertEqual(len(actions), 32)
        self.assertEqual(actions[0][:-1], expected[0][:-1])
        self.assertEqual(actions[0][-1], 1.)
        self.assertGreater(expected[0][-1], 1.)

    def test_gripper_only_clipped_joint_radians_preserved(self):
        self.output[0] = [20.] * 7 + [-20.]
        actions = self.policy.predict(self.observation, self.history, [self.observation])
        expected = self.normalizer.decode("action", self.output[0])
        self.assertEqual(actions[0][:-1], expected[:-1])
        self.assertEqual(actions[0][-1], 0.)

    def test_nonfinite_model_output_cannot_be_hidden_by_gripper_clamp(self):
        self.output[0][-1] = float("nan")
        with self.assertRaises(ValueError):
            self.policy.predict(self.observation, self.history, [self.observation])

    def test_short_model_output_rejected_and_close_delegates(self):
        self.inner.infer.return_value = self.output[:31]
        with self.assertRaises(ValueError):
            self.policy.predict(self.observation, self.history, [self.observation])
        self.policy.close()
        self.inner.close.assert_called_once_with()


def request_fixture():
    sample = dict(sequence=0, stamp=100., state=[0.] * 7 + [.5],
                  force=[0.] * 6, force_meta={})
    obs = encode_observation(dict(sample, action_space="joint8",
        frames={name: dict(width=1, height=1, data=b"rgb") for name in ("global", "wrist")},
        frame_meta={name: dict(name=name, width=1, height=1, color_order="RGB", received=99.99)
                    for name in ("global", "wrist")}))
    return dict(protocol=PROTOCOL, session_id="review-session", request_id="review-request",
                force_preprocessing=None, observation=obs, history=[sample])


class ServiceBoundaryReviewTests(unittest.TestCase):
    def test_camera_identity_and_client_clock_alignment(self):
        policy = MockPolicy(width=1, height=1)
        request = request_fixture()
        decode_request(request, policy.metadata)
        for key, value in (("color_order", "BGR"), ("width", 2),
                           ("received", 100.01), ("received", 99.9)):
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(request)
                bad["observation"]["frame_meta"]["global"][key] = value
                with self.assertRaises(ValueError):
                    decode_request(bad, policy.metadata)

    def test_close_waits_for_inference_then_rejects_late_reader(self):
        policy = MockPolicy(width=1, height=1)
        entered, finish, closed = threading.Event(), threading.Event(), threading.Event()
        errors = []
        def predict(observation, history, video_context):
            entered.set()
            if not finish.wait(2):
                raise RuntimeError("test inference wait timed out")
            self.assertFalse(closed.is_set())
            return [observation["state"][:] for _ in range(32)]
        policy.predict = predict
        policy.close = closed.set
        service = InferenceService(policy)
        def infer():
            try:
                service.infer(request_fixture())
            except BaseException as exc:
                errors.append(exc)
        active = threading.Thread(target=infer)
        closing = threading.Thread(target=service.close)
        active.start()
        try:
            self.assertTrue(entered.wait(1))
            closing.start()
            self.assertTrue(service.closing.wait(1))
            self.assertFalse(closed.is_set())
            with self.assertRaises(Conflict):
                service.infer(request_fixture())
        finally:
            finish.set()
            active.join(2)
            if closing.ident is not None:
                closing.join(2)
        self.assertFalse(active.is_alive())
        self.assertFalse(closing.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(closed.is_set())
        with self.assertRaisesRegex(Conflict, "shutting down"):
            service.infer(request_fixture())


if __name__ == "__main__":
    unittest.main()
