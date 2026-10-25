from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy"))

from realman_deploy.contracts import Observation
from realman_deploy.normalization import DatasetNormalizer, OnlineHistory
from realman_deploy.remote_transport import (
    RemoteInferenceError, RemoteJointPolicy, RemotePolicyService, decode_request, encode_request,
    make_server, normalizer_identity,
)


class FakePolicy:
    device = "cuda:0"
    reference_mode = "none"

    def __init__(self, normalizer):
        self.normalizer = normalizer
        self.diagnostics = {}
        self.calls = 0

    def infer(self, observation, reference, history):
        self.calls += 1
        return [[0.1] * self.normalizer.action_dim for _ in range(32)]

    def close(self):
        pass


class SlowPolicy(FakePolicy):
    def infer(self, observation, reference, history):
        time.sleep(0.25)
        return super().infer(observation, reference, history)


def normalizer_file(folder):
    payload = {
        "action_space": "joint8",
        "action_units": ["rad"] * 7 + ["openness_0_1"],
        "state_columns": [0, 1, 2, 3, 4, 5, 6, 19],
        "statistics": {
            "state": {"mean": [0.0] * 8, "std": [1.0] * 8},
            "action": {"mean": [0.0] * 8, "std": [1.0] * 8},
            "force": {"mean": [0.0] * 6, "std": [1.0] * 6},
        },
    }
    path = Path(folder) / "normalization.json"
    path.write_text(json.dumps(payload))
    return path


class RemoteTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.normalization = normalizer_file(self.tmp.name)
        self.normalizer = DatasetNormalizer.from_json(self.normalization)
        self.checkpoint = Path(self.tmp.name) / "checkpoint.pt"
        self.checkpoint.write_bytes(b"same-checkpoint")
        self.sha = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.config = {
            "normalization_file": str(self.normalization), "action_space": "joint8",
            "camera_order": ["global", "wrist"], "camera_width": 8, "camera_height": 4,
            "sampling_profile": "4416", "remote_inference": {
                "url": "http://127.0.0.1", "token_env": "TEST_REMOTE_TOKEN",
                "checkpoint_sha256": self.sha, "timeout_s": 2, "max_latency_s": 2,
            },
        }
        os.environ["TEST_REMOTE_TOKEN"] = "test-secret-with-more-than-16"
        self.server = None

    def tearDown(self):
        if self.server:
            self.server.shutdown(); self.server.server_close()
        os.environ.pop("TEST_REMOTE_TOKEN", None)
        self.tmp.cleanup()

    def make_server(self):
        service = RemotePolicyService(FakePolicy(self.normalizer), self.config, checkpoint_sha256=self.sha)
        self.server = make_server(service, host="127.0.0.1", port=0, token=os.environ["TEST_REMOTE_TOKEN"])
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.config["remote_inference"]["url"] = f"http://127.0.0.1:{self.server.server_port}"
        return service

    def observation(self, seq=0):
        bytes_per_frame = self.config["camera_width"] * self.config["camera_height"] * 3
        return Observation.now(seq, [0.0] * 7 + [0.5], [0.0] * 6,
                              {"global": bytes(bytes_per_frame), "wrist": bytes(bytes_per_frame)},
                              action_space="joint8")

    def request_only_policy(self):
        policy = object.__new__(RemoteJointPolicy)
        policy._closed = False
        policy._connection = None
        policy._last_used_at = 0.0
        policy.timeout_s = 2.0
        policy.url = "http://127.0.0.1:8000/private-forward-path"
        policy._parsed = urlsplit(policy.url)
        policy._prefix = policy._parsed.path
        policy._token = os.environ["TEST_REMOTE_TOKEN"]
        return policy

    def test_proxy_html_error_preserves_http_status_and_content_type(self):
        policy = self.request_only_policy()
        response = Mock(status=403)
        response.read.return_value = ("<html>Access denied " + policy._token + " " + policy.url + "</html>").encode()
        response.getheader.side_effect = lambda name: {"Content-Type": "text/html; charset=utf-8"}.get(name)
        connection = Mock(sock=None)
        connection.getresponse.return_value = response
        with patch("realman_deploy.remote_transport.http.client.HTTPConnection", return_value=connection):
            with self.assertRaises(RemoteInferenceError) as error:
                policy._request("GET", "/health", None)
        message = str(error.exception)
        self.assertIn("HTTP 403", message)
        self.assertIn("Content-Type: text/html; charset=utf-8", message)
        self.assertIn("non-JSON", message)
        self.assertNotIn(policy._token, message)
        self.assertNotIn(policy.url, message)
        self.assertEqual(connection.request.call_count, 1)
        connection.close.assert_called_once()

    def test_proxy_connection_close_updates_timeout_before_reconnecting(self):
        policy = self.request_only_policy()
        response = Mock(status=200)
        response.read.return_value = b"{}"
        response.getheader.side_effect = lambda name: {"Content-Type": "application/json", "Connection": "close"}.get(name)
        # http.client.getresponse() sets sock=None after Connection: close. The
        # object is reused, and its next request reconnects using .timeout.
        connection = Mock(sock=None, timeout=2.0)
        connection.getresponse.return_value = response
        timeouts_at_request = []
        connection.request.side_effect = lambda *args, **kwargs: timeouts_at_request.append(connection.timeout)
        with patch("realman_deploy.remote_transport.http.client.HTTPConnection", return_value=connection) as factory:
            policy._request("GET", "/health", None)
            self.assertIsNone(connection.sock)
            policy._request("POST", "/v1/infer", b"data", timeout_s=0.05)
        self.assertEqual(timeouts_at_request, [2.0, 0.05])
        self.assertEqual(connection.timeout, 0.05)
        factory.assert_called_once()

    def test_roundtrip_and_metadata_identity(self):
        service = self.make_server()
        policy = RemoteJointPolicy(self.checkpoint, self.config, config_path=self.normalization)
        history = OnlineHistory(self.normalizer, 32); history.append([0.0] * 7 + [0.5], [0.0] * 6)
        result = policy.infer(self.observation(), None, history.tensors())
        self.assertEqual((len(result), len(result[0])), (32, 8))
        self.assertEqual(service.policy.calls, 1)
        self.assertEqual(policy.diagnostics["last_inference"]["image_encoding"], "zlib_rgb")
        self.assertGreater(policy.diagnostics["last_inference"]["upload_bytes"], 8)
        policy.close()

    def test_client_socket_timeout_bounds_slow_remote_request(self):
        service = RemotePolicyService(SlowPolicy(self.normalizer), self.config, checkpoint_sha256=self.sha)
        self.server = make_server(service, host="127.0.0.1", port=0, token=os.environ["TEST_REMOTE_TOKEN"])
        thread = threading.Thread(target=self.server.serve_forever, daemon=True); thread.start()
        self.config["remote_inference"].update(url=f"http://127.0.0.1:{self.server.server_port}", max_latency_s=0.05)
        policy = RemoteJointPolicy(self.checkpoint, self.config, config_path=self.normalization)
        history = OnlineHistory(self.normalizer, 32); history.append([0.0] * 7 + [0.5], [0.0] * 6)
        started = time.monotonic()
        with self.assertRaises(Exception):
            policy.infer(self.observation(), None, history.tensors())
        elapsed = time.monotonic() - started
        policy.close()
        self.assertLess(elapsed, 0.20)

    def test_duplicate_request_is_rejected(self):
        service = self.make_server()
        obs = self.observation()
        history = OnlineHistory(self.normalizer, 32); history.append(obs.state, obs.force)
        body, _ = encode_request(obs, None, history.tensors(), width=8, height=4,
            camera_order=["global", "wrist"], identity=normalizer_identity(self.normalizer),
            sampling_profile="4416", session_id="same", request_index=1)
        service.infer(body)
        with self.assertRaises(ValueError):
            service.infer(body)

    def test_mismatched_normalization_is_rejected(self):
        service = self.make_server()
        obs = self.observation()
        history = OnlineHistory(self.normalizer, 32); history.append(obs.state, obs.force)
        body, _ = encode_request(obs, None, history.tensors(), width=8, height=4,
            camera_order=["global", "wrist"], identity="0" * 64, sampling_profile="4416")
        with self.assertRaises(ValueError):
            service.infer(body)

    def test_random_payload_roundtrip(self):
        obs = self.observation()
        # zlib output must decompress exactly to the configured RGB byte count.
        import random
        raw = random.Random(7).randbytes(8 * 4 * 3)
        obs = Observation(obs.stamp, obs.sequence, obs.state, obs.force,
                          {"global": raw, "wrist": raw}, action_space="joint8")
        history = OnlineHistory(self.normalizer, 32); history.append(obs.state, obs.force)
        body, _ = encode_request(obs, None, history.tensors(), width=8, height=4,
            camera_order=["global", "wrist"], identity=normalizer_identity(self.normalizer),
            sampling_profile="4416")
        decoded, _, _, _ = decode_request(body, width=8, height=4, camera_order=["global", "wrist"],
            action_space="joint8", identity=normalizer_identity(self.normalizer), sampling_profile="4416")
        self.assertEqual(decoded.frames["global"], raw)


if __name__ == "__main__":
    unittest.main()
