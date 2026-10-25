"""Remote benchmark preserves partial results and distinguishes failed requests."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy"))

import benchmark_remote
from realman_deploy.normalization import DatasetNormalizer
from realman_deploy.remote_transport import RemoteInferenceError


class ScriptedPolicy:
    reference_mode = "none"
    image_encoding = "zlib_rgb"
    width, height = 4, 4
    camera_order = ["global", "wrist"]

    def __init__(self, outcomes):
        self.normalizer = DatasetNormalizer({
            "state": {"mean": [0.] * 8, "std": [1.] * 8},
            "action": {"mean": [0.] * 8, "std": [1.] * 8},
            "force": {"mean": [0.] * 6, "std": [1.] * 6},
        }, action_space="joint8")
        self.outcomes = iter(outcomes)
        self.diagnostics = {"remote": {"model_name": "ScriptedPolicy", "checkpoint_sha256": "a" * 64}}
        self.sequences = []
        self.closed = False

    def infer(self, observation, reference, history):
        self.sequences.append(observation.sequence)
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        self.diagnostics["last_inference"] = {
            "transport_overhead_ms": float(outcome), "server_timing_ms": {"inference": 0.0},
            "upload_bytes": 4096, "source_sequence": observation.sequence,
        }
        return [[0.1] * 8 for _ in range(32)]

    def close(self):
        self.closed = True


class RemoteBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.path = self.folder / "config.json"
        self.config = {"remote_inference": {"max_latency_s": 0.6}}
        self.path.write_text(json.dumps(self.config))
        self.output = self.folder / "results.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_mixed_success_timeout_and_error_complete_without_retries(self):
        # Warmup timeout is retained separately and cannot skew measured rates.
        policy = ScriptedPolicy([RemoteInferenceError("timed out"), 10.0,
            RemoteInferenceError("remote HTTP 504: deadline"),
            RemoteInferenceError("remote HTTP 429: model is busy"), 30.0])
        with patch.object(benchmark_remote, "RemoteJointPolicy", return_value=policy) as factory, \
                contextlib.redirect_stdout(io.StringIO()):
            result = benchmark_remote.run(self.path, runs=4, warmup=1, patterns=["blank"], output=self.output)
        self.assertTrue(result["completed"])
        self.assertEqual(policy.sequences, list(range(5)))
        self.assertTrue(policy.closed)
        self.assertEqual(factory.call_args.args[1], self.config)
        self.assertEqual(json.loads(self.path.read_text()), self.config)
        self.assertEqual(len(result["samples"]), 4)
        self.assertEqual(len(result["warmup_samples"]), 1)
        self.assertEqual([row["success"] for row in result["samples"]], [True, False, False, True])
        summary = result["summary"]["blank"]
        self.assertEqual(summary["attempts"], 4)
        self.assertEqual(summary["successes"], 2)
        self.assertEqual(summary["failures"], 2)
        self.assertEqual(summary["timeouts"], 1)
        self.assertEqual(summary["success_rate"], .5)
        self.assertEqual(summary["timeout_rate"], .25)
        self.assertEqual(summary["transport_overhead_ms"]["p50"], 20.0)
        self.assertEqual(summary["overhead_under_100ms_fraction"], .5)
        self.assertEqual(result["warmup_summary"]["blank"]["timeout_rate"], 1.)
        for row in result["samples"]:
            if not row["success"]:
                self.assertNotIn("server_timing_ms", row)
                self.assertNotIn("first_action_normalized", row)
        self.assertEqual(json.loads(self.output.read_text()), result)

    def test_all_failed_requests_have_null_latency_percentiles(self):
        policy = ScriptedPolicy([TimeoutError("request timed out"), RuntimeError("remote service unavailable")])
        with patch.object(benchmark_remote, "RemoteJointPolicy", return_value=policy), \
                contextlib.redirect_stdout(io.StringIO()):
            result = benchmark_remote.run(self.path, runs=1, warmup=0, patterns=["blank", "random"], output=self.output)
        self.assertTrue(result["completed"])
        self.assertEqual(policy.sequences, [0, 1])
        for pattern in ("blank", "random"):
            summary = result["summary"][pattern]
            self.assertEqual(summary["success_rate"], 0.)
            self.assertEqual(summary["non_model_under_100ms_fraction"], 0.)
            self.assertIsNone(summary["total_client_ms"]["p50"])
            self.assertIsNone(summary["server_inference_ms"]["p95"])
            self.assertIsNone(summary["transport_overhead_ms"]["p99"])
            self.assertIsNone(summary["upload_bytes"])
        self.assertEqual(result["summary"]["blank"]["timeout_rate"], 1.)
        self.assertEqual(result["summary"]["random"]["timeout_rate"], 0.)
        self.assertNotIn("NaN", self.output.read_text())

    def test_interrupted_run_retains_completed_samples_on_disk(self):
        policy = ScriptedPolicy([10.0, KeyboardInterrupt()])
        with patch.object(benchmark_remote, "RemoteJointPolicy", return_value=policy), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                benchmark_remote.run(self.path, runs=3, warmup=0, patterns=["blank"], output=self.output)
        saved = json.loads(self.output.read_text())
        self.assertFalse(saved["completed"])
        self.assertEqual(len(saved["samples"]), 1)
        self.assertEqual(saved["summary"]["blank"]["attempts"], 1)
        self.assertTrue(policy.closed)


if __name__ == "__main__":
    unittest.main()
