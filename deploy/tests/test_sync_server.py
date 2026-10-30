"""CPU-only contract and real HTTP tests; no robot SDK, Torch or checkpoint."""
import copy
import json
from pathlib import Path
import sys
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "deploy"), str(ROOT / "src")]

from realman_sync.cosmos_policy import CosmosPolicy
from realman_sync.mock_policy import MockPolicy
from realman_sync.server import Conflict, InferenceService, PolicyHTTPServer
from realman_sync.wire import PROTOCOL, decode_request, encode_observation, validate_response
from realman_deploy.normalization import DatasetNormalizer


def request_for(policy, *, sequence=100, session="test-session"):
    metadata = policy.metadata
    samples = []
    for i in range(sequence - 31, sequence + 1):
        samples.append(dict(sequence=i, stamp=100.0 + i / 30, state=[.1] * 7 + [.5],
                            force=[float(i)] * 6, force_meta={}))
    def observation(sample):
        return encode_observation(dict(copy.deepcopy(sample), action_space="joint8", frame_meta={}, frames={
            name: dict(width=metadata["width"], height=metadata["height"],
                       data=bytes(metadata["width"] * metadata["height"] * 3))
            for name in metadata["camera_names"]}))
    request = dict(protocol=PROTOCOL, session_id=session, request_id=str(sequence),
                   observation=observation(samples[-1]), history=samples, force_preprocessing=None)
    if metadata["video_context_frames"] == 2:
        request["video_context"] = [observation(samples[-5]), observation(samples[-1])]
    return request


class WireTests(unittest.TestCase):
    def setUp(self):
        self.policy = MockPolicy(width=2, height=2)
        self.request = request_for(self.policy)

    def test_lossless_frames_and_history(self):
        decoded = decode_request(self.request, self.policy.metadata)
        self.assertEqual(decoded["observation"]["frames"]["global"]["data"], bytes(12))
        self.assertEqual(len(decoded["history"]), 32)
        self.assertEqual(decoded["video_context"], [decoded["observation"]])

    def test_rejects_bad_payloads(self):
        variants = []
        for key, value in (("state", [0] * 7), ("force", [float("nan")] * 6),
                           ("sequence", True), ("stamp", -1), ("action_space", "eef_pose7")):
            payload = copy.deepcopy(self.request)
            payload["observation"][key] = value
            variants.append(payload)
        payload = copy.deepcopy(self.request)
        payload["observation"]["frames"]["global"]["data"] = "!" * 16
        variants.append(payload)
        payload = copy.deepcopy(self.request)
        payload["history"][1]["stamp"] += .015
        variants.append(payload)
        payload = copy.deepcopy(self.request)
        payload["history"][1]["force_meta"] = {"epoch": 3}
        variants.append(payload)
        payload = copy.deepcopy(self.request)
        payload["history"][-1]["state"][0] = 9
        variants.append(payload)
        for payload in variants:
            with self.subTest(payload=payload["observation"]["sequence"]), self.assertRaises(ValueError):
                decode_request(payload, self.policy.metadata)

    def test_real_training_video_stride(self):
        policy = MockPolicy(width=2, height=2, video_context_frames=2)
        request = request_for(policy)
        self.assertEqual(len(decode_request(request, policy.metadata)["video_context"]), 2)
        request["video_context"][0]["sequence"] = 99
        with self.assertRaisesRegex(ValueError, "stride"):
            decode_request(request, policy.metadata)

    def test_force_identity_and_source_time(self):
        metadata = dict(self.policy.metadata, force_preprocessing={"id": "trained-force"})
        self.request["force_preprocessing"] = metadata["force_preprocessing"]
        for row in self.request["history"] + [self.request["observation"]]:
            row["force_meta"] = dict(epoch=2, preprocessing_id="trained-force", source_time=row["stamp"] - .005)
        decode_request(self.request, metadata)
        self.request["history"][0]["force_meta"]["source_time"] -= .1
        with self.assertRaisesRegex(ValueError, "stale"):
            decode_request(self.request, metadata)

    def test_response_correlation(self):
        response = InferenceService(self.policy).infer(self.request)
        self.assertEqual(len(validate_response(response, self.request, self.policy.metadata)), 32)
        response["request_id"] = "old"
        with self.assertRaises(ValueError):
            validate_response(response, self.request, self.policy.metadata)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.policy = MockPolicy(width=2, height=2)
        self.service = InferenceService(self.policy)
        self.request = request_for(self.policy)

    def test_duplicate_and_stale_are_rejected(self):
        self.service.infer(self.request)
        with self.assertRaises(Conflict):
            self.service.infer(self.request)
        self.request["request_id"] = "new-id-old-observation"
        with self.assertRaises(Conflict):
            self.service.infer(self.request)
        self.service.infer(request_for(self.policy, sequence=132))

    def test_exclusive_session_and_release(self):
        self.service.infer(self.request)
        with self.assertRaises(Conflict):
            self.service.infer(request_for(self.policy, session="other"))
        with self.assertRaises(Conflict):
            self.service.release(dict(protocol=PROTOCOL, session_id="other"))
        self.service.release(dict(protocol=PROTOCOL, session_id="test-session"))
        self.service.infer(request_for(self.policy, session="other"))

    def test_only_one_inference_and_no_release_while_busy(self):
        self.service.lock.acquire()
        try:
            with self.assertRaises(Conflict):
                self.service.infer(self.request)
            with self.assertRaises(Conflict):
                self.service.release(dict(protocol=PROTOCOL, session_id="test-session"))
        finally:
            self.service.lock.release()

    def test_late_reader_cannot_infer_after_shutdown(self):
        self.service.close()
        with self.assertRaisesRegex(Conflict, "shutting down"):
            self.service.infer(self.request)

    def test_failed_policy_cannot_continue_session(self):
        def broken(*args):
            raise RuntimeError("model failed")
        self.policy.predict = broken
        with self.assertRaises(RuntimeError):
            self.service.infer(self.request)
        with self.assertRaises(Conflict):
            self.service.infer(request_for(self.policy, sequence=132))

    def test_clock_origins_are_never_compared(self):
        for row in self.request["history"] + [self.request["observation"]]:
            row["stamp"] += 100000000
        self.service.infer(self.request)


class CosmosWrapperTests(unittest.TestCase):
    def test_normalizes_history_once_and_decodes_action_once(self):
        normalizer = DatasetNormalizer({name: {"mean": [2] * width, "std": [2] * width}
                                        for name, width in (("state", 8), ("action", 8), ("force", 6))},
                                       action_space="joint8")
        calls = []
        class Policy:
            def infer(self, observation, reference, history):
                calls.append((observation, reference, history))
                return [[1] * 7 + [-3] for _ in range(32)]
        inner = Policy()
        inner.normalizer = normalizer
        wrapper = CosmosPolicy.__new__(CosmosPolicy)
        wrapper.policy = inner
        mock = MockPolicy(width=2, height=2)
        request = decode_request(request_for(mock), mock.metadata)
        result = wrapper.predict(request["observation"], request["history"], request["video_context"])
        self.assertEqual(result[0], [4.0] * 7 + [0.0])
        self.assertIsNone(calls[0][1])
        self.assertEqual(calls[0][2][0][-1][:8], normalizer.encode("state", request["observation"]["state"]))
        self.assertEqual(calls[0][0].frames["global"], bytes(12))


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.policy = MockPolicy(width=2, height=2)
        self.server = PolicyHTTPServer(("127.0.0.1", 0), InferenceService(self.policy), token="test-token")
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:%s" % self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def call(self, path, payload=None, token="test-token"):
        request = Request(self.url + path, data=None if payload is None else json.dumps(payload).encode(),
                          headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        with urlopen(request, timeout=2) as response:
            return json.load(response)

    def test_health_infer_release_real_http(self):
        self.assertTrue(self.call("/health")["ready"])
        request = request_for(self.policy)
        response = self.call("/infer", request)
        self.assertEqual(len(validate_response(response, request, self.policy.metadata)), 32)
        with self.assertRaises(HTTPError) as exc:
            self.call("/infer", request)
        self.assertEqual(exc.exception.code, 409)
        self.assertTrue(self.call("/session/release", dict(protocol=PROTOCOL, session_id="test-session"))["released"])

    def test_auth_and_nonfinite_json(self):
        with self.assertRaises(HTTPError) as exc:
            self.call("/health", token="wrong")
        self.assertEqual(exc.exception.code, 401)
        request = request_for(self.policy)
        request["observation"]["state"][0] = float("nan")
        with self.assertRaises(HTTPError) as exc:
            self.call("/infer", request)
        self.assertEqual(exc.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
