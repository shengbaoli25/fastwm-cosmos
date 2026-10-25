"""Remote/local device selection through the existing deployment dashboard."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "deploy")]

from realman_deploy.normalization import DatasetNormalizer
from realman_deploy.runtime import DeploymentRuntime
from realman_deploy.web import DeploymentWeb

SHA = "9a560012cc277d5b4755cb8e461edb5b238850503a63fb57699567bc9dab823d"
OTHER_SHA = "0" * 64


class FakePolicy:
    reference_mode = "none"
    requires_local_cuda = False

    def __init__(self, normalizer):
        self.normalizer = normalizer
        self.closed = False
        self.diagnostics = {}

    def infer(self, observation, reference, history):
        return [[0.1] * 8 for _ in range(32)]

    def close(self):
        self.closed = True


class RemoteWebTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.normalization_path = self.folder / "normalization.json"
        self.normalization_path.write_text(json.dumps({
            "action_space": "joint8", "action_units": ["rad"] * 7 + ["openness_0_1"],
            "state_columns": [0, 1, 2, 3, 4, 5, 6, 19],
            "statistics": {
                "state": {"mean": [0.] * 8, "std": [1.] * 8},
                "action": {"mean": [0.] * 8, "std": [1.] * 8},
                "force": {"mean": [0.] * 6, "std": [1.] * 6},
            },
        }))
        self.normalizer = DatasetNormalizer.from_json(self.normalization_path)
        self.config_path = self.folder / "runtime.json"
        self.catalog_path = self.folder / "catalog.json"
        self.config = {
            "normalization_file": "normalization.json", "camera_order": ["global", "wrist"],
            "camera_width": 640, "camera_height": 480,
            "sampling_profile": "4416", "execution_mode": "synchronous",
            "inference_device": "qizhi_h200", "checkpoint_directory": str(self.folder / "weights"),
            "remote_inference": {"url": "http://127.0.0.1:8000/forward/8000", "checkpoint_sha256": SHA,
                                 "token_env": "TEST_DASHBOARD_REMOTE_TOKEN"},
        }
        self.web = self.runtime = self.thread = None
        os.environ["TEST_DASHBOARD_REMOTE_TOKEN"] = "not-a-value-to-return-in-settings"

    def tearDown(self):
        if self.web:
            self.web.close()
            if self.thread:
                self.thread.join(timeout=2)
        os.environ.pop("TEST_DASHBOARD_REMOTE_TOKEN", None)
        self.tmp.cleanup()

    def start_web(self, *, catalog=None):
        self.config_path.write_text(json.dumps(self.config))
        self.catalog_path.write_text(json.dumps({"checkpoints": catalog or []}))
        self.runtime = DeploymentRuntime(self.normalizer, execution_mode="synchronous", sampling_profile="4416",
                                         inference_device=self.config["inference_device"])
        self.runtime._cuda_status = Mock(return_value={"available": False, "error": "test has no CUDA"})
        self.web = DeploymentWeb(self.runtime, catalog_path=self.catalog_path, config_path=self.config_path, port=0)
        self.thread = threading.Thread(target=self.web.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.web.server.server_port}"

    def request(self, route, body=None):
        request = Request(self.base + route, data=json.dumps(body).encode() if body is not None else None,
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=3) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    def join_load(self):
        self.assertIsNotNone(self.runtime._load_thread)
        self.runtime._load_thread.join(timeout=2)
        self.assertFalse(self.runtime._load_thread.is_alive())

    def test_remote_without_cuda_or_download_uses_catalog_sha(self):
        self.config["remote_inference"].pop("checkpoint_sha256")
        self.start_web(catalog=[dict(id="step5000", label="Step 5000", source="https://example.invalid/weights.pt", sha256=SHA)])
        loaded = FakePolicy(self.normalizer)
        with patch("realman_deploy.remote_transport.RemoteJointPolicy", return_value=loaded) as factory:
            code, response = self.request("/api/select", {"id": "step5000"})
            self.assertEqual(code, 202, response)
            self.join_load()
        self.assertIs(self.runtime.policy, loaded)
        self.assertEqual(self.runtime.state, "ready")
        self.runtime._cuda_status.assert_not_called()
        args, kwargs = factory.call_args
        self.assertIsNone(args[0])
        self.assertEqual(args[1]["remote_inference"]["checkpoint_sha256"], SHA)
        self.assertNotIn("model_kwargs", args[1])
        self.assertNotIn("tokenizer", args[1])
        self.assertEqual(kwargs["config_path"], self.config_path)
        self.assertFalse((self.folder / "weights/step5000.pt").exists())

    def test_remote_without_catalog_or_download_uses_config_sha(self):
        self.start_web()
        loaded = FakePolicy(self.normalizer)
        with patch("realman_deploy.remote_transport.RemoteJointPolicy", return_value=loaded) as factory:
            code, response = self.request("/api/select", {"id": ""})
            self.assertEqual(code, 202, response)
            self.join_load()
        self.assertIs(self.runtime.policy, loaded)
        self.runtime._cuda_status.assert_not_called()
        self.assertIsNone(factory.call_args.args[0])
        self.assertEqual(factory.call_args.args[1]["remote_inference"]["checkpoint_sha256"], SHA)

    def test_minimal_config_connects_real_remote_adapter_without_local_weights(self):
        from realman_deploy.remote_transport import RemoteJointPolicy, RemotePolicyService, make_server
        policy = FakePolicy(self.normalizer)
        service = RemotePolicyService(policy, self.config, checkpoint_sha256=SHA)
        server = make_server(service, host="127.0.0.1", port=0,
                             token=os.environ["TEST_DASHBOARD_REMOTE_TOKEN"])
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        remote = None
        try:
            self.config["remote_inference"]["url"] = f"http://127.0.0.1:{server.server_port}"
            remote = RemoteJointPolicy(None, self.config, config_path=self.config_path)
            self.assertIs(remote.requires_local_cuda, False)
            self.assertEqual(remote.normalizer.action_space, "joint8")
            self.assertEqual(remote.expected_checkpoint, SHA)
            self.assertNotIn("model_kwargs", remote.config)
            self.assertNotIn("tokenizer", remote.config)
        finally:
            if remote:
                remote.close()
            server.shutdown(); server.server_close(); thread.join(timeout=2)

    def test_local_selection_still_requires_cuda(self):
        self.config["inference_device"] = "local"
        self.start_web(catalog=[dict(id="step5000", source="https://example.invalid/weights.pt", sha256=SHA)])
        old = FakePolicy(self.normalizer)
        self.runtime.attach(policy=old)
        with patch("realman_deploy.web.CosmosJointPolicy") as factory:
            code, response = self.request("/api/select", {"id": "step5000"})
        self.assertEqual(code, 400)
        self.assertIn("CUDA", response["error"])
        factory.assert_not_called()
        self.assertIs(self.runtime.policy, old)
        self.assertFalse(old.closed)

    def test_config_catalog_sha_conflict_preserves_current_policy(self):
        self.config["remote_inference"]["checkpoint_sha256"] = OTHER_SHA
        self.start_web(catalog=[dict(id="step5000", source="https://example.invalid/weights.pt", sha256=SHA)])
        old = FakePolicy(self.normalizer)
        self.runtime.attach(policy=old)
        before = self.config_path.read_bytes()
        with patch("realman_deploy.remote_transport.RemoteJointPolicy") as factory:
            code, response = self.request("/api/select", {"id": "step5000"})
        self.assertEqual(code, 400)
        self.assertIn("SHA256", response["error"])
        factory.assert_not_called()
        self.assertIs(self.runtime.policy, old)
        self.assertFalse(old.closed)
        self.assertEqual(before, self.config_path.read_bytes())

    def test_settings_persist_device_smoothing_url_and_only_token_env(self):
        self.config["inference_device"] = "local"
        self.start_web()
        old = FakePolicy(self.normalizer)
        self.runtime.attach(policy=old)
        code, response = self.request("/api/settings", {
            "sampling_profile": "4416", "execution_mode": "async_relaxed",
            "inference_device": "qizhi_h200", "action_smoothing_alpha": 0.35,
            "max_result_age_s": 0.55,
            "remote_inference": {"url": "https://forward.example.org/worker/ports/8000/", "timeout_s": 3},
        })
        self.assertEqual(code, 200, response)
        self.assertTrue(response["reload_required"])
        self.assertIsNone(self.runtime.policy)
        self.assertTrue(old.closed)
        self.assertEqual(self.runtime.inference_device, "qizhi_h200")
        self.assertEqual(self.runtime.action_smoothing_alpha, .35)
        self.assertEqual(self.runtime.execution_mode, "async_relaxed")
        self.assertEqual(self.runtime.max_result_age_s, .55)
        saved = json.loads(self.config_path.read_text())
        self.assertEqual(saved["remote_inference"]["url"], "https://forward.example.org/worker/ports/8000")
        self.assertEqual(saved["remote_inference"]["checkpoint_sha256"], SHA)
        self.assertEqual(saved["remote_inference"]["token_env"], "TEST_DASHBOARD_REMOTE_TOKEN")
        self.assertEqual(saved["camera_order"], ["global", "wrist"])
        self.assertNotIn(os.environ["TEST_DASHBOARD_REMOTE_TOKEN"], json.dumps(response))
        self.assertNotIn(os.environ["TEST_DASHBOARD_REMOTE_TOKEN"], self.config_path.read_text())

    def test_invalid_settings_leave_policy_and_disk_config_unchanged(self):
        self.start_web()
        old = FakePolicy(self.normalizer)
        self.runtime.attach(policy=old)
        before = self.config_path.read_bytes()
        original = copy.deepcopy(self.web.config)
        for invalid in (
                {"action_smoothing_alpha": 0},
                {"remote_inference": {"image_encoding": "unsupported"}},
                {"remote_inference": {"url": "not a URL"}},
                {"inference_device": "unknown_device"},
                {"max_result_age_s": -1},
        ):
            with self.subTest(invalid=invalid):
                body = {"sampling_profile": "4416", "execution_mode": "synchronous", **invalid}
                code, response = self.request("/api/settings", body)
                self.assertEqual(code, 400, response)
                self.assertIs(self.runtime.policy, old)
                self.assertFalse(old.closed)
                self.assertEqual(before, self.config_path.read_bytes())
                self.assertEqual(original, self.web.config)

    def test_remote_start_requires_explicit_sampling_profile_and_execution_mode(self):
        from realman_deploy.deployment_settings import deployment_settings
        for absent in ("sampling_profile", "execution_mode"):
            with self.subTest(absent=absent):
                config = copy.deepcopy(self.config)
                config.pop(absent)
                with self.assertRaises(ValueError) as error:
                    deployment_settings(config)
                self.assertIn(absent, str(error.exception))

    def test_minimal_remote_launch_passes_device_and_deadline_to_runtime(self):
        # No model_kwargs, tokenizer, CUDA environment or local checkpoint is needed.
        from deploy import serve_realman
        self.config.pop("checkpoint_directory")
        self.config_path.write_text(json.dumps(self.config))
        self.catalog_path.write_text('{"checkpoints": []}')
        argv = ["serve_realman.py", "--config", str(self.config_path), "--catalog", str(self.catalog_path), "--port", "0"]
        fake_runtime = Mock()
        fake_web = Mock()
        fake_web.serve_forever.side_effect = KeyboardInterrupt
        with patch.object(sys, "argv", argv), \
                patch.object(serve_realman, "DeploymentRuntime", return_value=fake_runtime) as runtime_factory, \
                patch.object(serve_realman, "DeploymentWeb", return_value=fake_web) as web_factory:
            result = serve_realman.main()
        self.assertEqual(result, 0)
        self.assertEqual(runtime_factory.call_args.kwargs["inference_device"], "qizhi_h200")
        self.assertEqual(runtime_factory.call_args.kwargs["max_result_age_s"], .6)
        self.assertEqual(runtime_factory.call_args.kwargs["sampling_profile"], "4416")
        self.assertEqual(web_factory.call_args.args[0], fake_runtime)
        fake_web.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
