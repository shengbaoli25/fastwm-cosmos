import contextlib
import copy
import hashlib
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pathlib
import threading
import time
import unittest
from unittest.mock import patch

from realman_sync_client.cli import main
from realman_sync_client.client import (ClientOptions, ObservationCollector, SynchronousClient,
                                       validate_actions, validate_metadata)
from realman_sync_client.http import InferenceHTTPClient
from realman_sync_client.mock import MockHardware
from realman_sync_client.wire import PROTOCOL, decode_request


def metadata(**overrides):
    return dict(camera_names=["global", "wrist"], width=4, height=3, action_space="joint8",
                action_horizon=32, history_length=32, control_hz=30,
                video_context_frames=overrides.pop("video_context_frames", 1),
                video_context_stride=overrides.pop("video_context_stride", 1),
                force_preprocessing=None, **overrides)


def response(request):
    return dict(protocol=PROTOCOL, session_id=request["session_id"], request_id=request["request_id"],
                source_sequence=request["observation"]["sequence"], action_space="joint8",
                control_hz=30, actions=[list(request["observation"]["state"]) for _ in range(32)])


class RecordingTransport:
    def __init__(self, meta, mutate=None, delay=.08):
        self.meta, self.mutate, self.delay = meta, mutate, delay
        self.requests = []
        self.infer_times = []
        self.release_ids = []

    def infer(self, request):
        decode_request(request, self.meta)
        self.requests.append(copy.deepcopy(request))
        self.infer_times.append(time.monotonic())
        time.sleep(self.delay)
        result = response(request)
        if self.mutate:
            self.mutate(result)
        return result

    def release(self, session_id):
        self.release_ids.append(session_id)


class RecordingHardware(MockHardware):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stop_acknowledgements = []

    def slow_stop(self):
        super().slow_stop()
        self.stop_acknowledgements.append(time.monotonic())


class SyncClientTests(unittest.TestCase):
    def test_hardware_sources_and_force_modules_match_original_hashes(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "HARDWARE_SOURCES.json").read_text())
        for name, expected in manifest["files"].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected, name)

    def test_continuous_history_strided_video_and_stop_before_fresh_request(self):
        meta = metadata(video_context_frames=2, video_context_stride=4)
        hardware = RecordingHardware(width=4, height=3)
        transport = RecordingTransport(meta)
        client = SynchronousClient(transport, hardware, meta, options=ClientOptions(execute_steps=2))
        self.assertEqual(client.run(cycles=2), 2)
        self.assertEqual(len(hardware.commands), 4)
        self.assertEqual(len(transport.release_ids), 1)
        for request in transport.requests:
            history = request["history"]
            self.assertEqual(len(history), 32)
            self.assertEqual(history[-1]["sequence"], request["observation"]["sequence"])
            for previous, current in zip(history, history[1:]):
                self.assertEqual(current["sequence"], previous["sequence"] + 1)
                self.assertAlmostEqual(current["stamp"] - previous["stamp"], 1 / 30, places=6)
            self.assertEqual(request["video_context"][1]["sequence"] - request["video_context"][0]["sequence"], 4)
        first, second = transport.requests
        self.assertGreater(second["observation"]["sequence"] - first["observation"]["sequence"], 2)
        self.assertGreater(second["observation"]["stamp"], hardware.stop_acknowledgements[1])
        self.assertGreater(transport.infer_times[1], hardware.stop_acknowledgements[1])
        self.assertGreaterEqual(hardware.commands[1][0] - hardware.commands[0][0], 1 / 30)
        self.assertTrue(hardware.closed)
        self.assertFalse(hardware.armed)

    def test_invalid_last_row_rejected_before_first_action_even_with_prefix_execution(self):
        meta = metadata()
        hardware = RecordingHardware(width=4, height=3)
        transport = RecordingTransport(meta, mutate=lambda result: result["actions"][-1].__setitem__(0, float("nan")))
        client = SynchronousClient(transport, hardware, meta, options=ClientOptions(execute_steps=1))
        with self.assertRaisesRegex(ValueError, "finite"):
            client.run(cycles=1)
        self.assertEqual(hardware.commands, [])
        self.assertTrue(hardware.closed)
        self.assertGreaterEqual(hardware.stop_count, 2)

    def test_network_failure_is_not_retried(self):
        meta = metadata()
        hardware = RecordingHardware(width=4, height=3)
        transport = RecordingTransport(meta)
        calls = []
        def fail(request):
            calls.append(request)
            raise TimeoutError("network stalled")
        transport.infer = fail
        client = SynchronousClient(transport, hardware, meta)
        with self.assertRaisesRegex(TimeoutError, "network stalled"):
            client.run(cycles=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(hardware.commands, [])
        self.assertTrue(hardware.closed)

    def test_response_identity_and_local_age_fail_closed(self):
        request = dict(protocol=PROTOCOL, session_id="session", request_id="request",
                       observation=dict(sequence=31, stamp=100.0, state=[0.] * 7 + [.5]))
        result = response(request)
        with self.assertRaisesRegex(ValueError, "expired"):
            validate_actions(result, request, metadata(), now=105.1, max_source_age_s=5)
        for key, value in (("request_id", "old"), ("source_sequence", 30),
                           ("session_id", "another"), ("action_space", "eef_pose7")):
            bad = dict(result, **{key: value})
            with self.assertRaises(ValueError):
                validate_actions(bad, request, metadata(), now=101, max_source_age_s=5)

    def test_sensor_fault_and_command_overrun_stop_execution(self):
        meta = metadata()
        hardware = RecordingHardware(width=4, height=3)
        client = SynchronousClient(RecordingTransport(meta), hardware, meta, options=ClientOptions(execute_steps=2))
        observation = hardware.observe_at(time.monotonic())
        client.collector.history.append(observation)
        original = hardware.command
        def slow(action):
            original(action)
            time.sleep(.04)
        hardware.command = slow
        with self.assertRaisesRegex(RuntimeError, "overrun"):
            client._execute([observation.state] * 32)
        self.assertEqual(len(hardware.commands), 1)
        hardware.slow_stop()
        hardware.close()

    def test_force_epoch_resets_history(self):
        meta = metadata()
        hardware = MockHardware(width=4, height=3)
        collector = ObservationCollector(hardware, meta)
        now = time.monotonic()
        collector._append(hardware.observe_at(now), now)
        later = hardware.observe_at(now + 1 / 30)
        later.force_meta["epoch"] = 1
        collector._append(later, now + 1 / 30)
        self.assertEqual(len(collector.history), 1)

    def test_release_timeout_is_bounded_separately_from_inference(self):
        transport = InferenceHTTPClient("http://127.0.0.1:8766", timeout_s=120)
        with patch.object(transport, "_request", return_value={}) as request:
            transport.release("session")
        self.assertEqual(request.call_args.kwargs["timeout_s"], 2.0)

    def test_joint_soft_limits_checked_for_all_rows_before_arm(self):
        from realman_sync_client.client import validate_hardware_chunk
        hardware = MockHardware(width=4, height=3)
        hardware.joint_min = (-90.,) * 7
        hardware.joint_max = (90.,) * 7
        hardware.config = dict(joint_limit_margin_deg=2., max_joint_step_deg=8.)
        rows = [hardware.state] * 31 + [(2.,) + hardware.state[1:]]
        with self.assertRaisesRegex(ValueError, "soft-limit"):
            validate_hardware_chunk(hardware, rows, hardware.state)
        self.assertFalse(hardware.armed)

    def test_real_hardware_requires_explicit_execution_before_network(self):
        with contextlib.redirect_stderr(io.StringIO()), patch("realman_sync_client.cli.InferenceHTTPClient") as transport:
            with self.assertRaises(SystemExit):
                main(["--hardware", "realman"])
        transport.assert_not_called()

    def test_metadata_rejects_training_cadence_mismatch(self):
        for change in ({"control_hz": 20}, {"action_space": "eef_pose7"}, {"history_length": 8}):
            with self.assertRaises(ValueError):
                validate_metadata(dict(protocol=PROTOCOL, metadata=dict(metadata(), **change)))


class HTTPIntegrationTests(unittest.TestCase):
    def test_mock_cli_through_real_http_with_bearer_and_release(self):
        meta = metadata()
        calls = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def reply(self, payload):
                raw = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def do_GET(self):
                calls.append((self.path, self.headers.get("Authorization")))
                self.reply(dict(protocol=PROTOCOL, ready=True, metadata=meta))
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append((self.path, self.headers.get("Authorization")))
                if self.path == "/infer":
                    decode_request(request, meta)
                    self.reply(response(request))
                else:
                    self.reply(dict(protocol=PROTOCOL, released=True))
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict("os.environ", {"REALMAN_SERVER_TOKEN": "test-token"}), contextlib.redirect_stdout(io.StringIO()):
                code = main(["--server", f"http://127.0.0.1:{server.server_port}",
                             "--cycles", "1", "--execute-steps", "1"])
            self.assertEqual(code, 0)
            self.assertEqual([item[0] for item in calls], ["/health", "/infer", "/session/release"])
            self.assertTrue(all(item[1] == "Bearer test-token" for item in calls))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
