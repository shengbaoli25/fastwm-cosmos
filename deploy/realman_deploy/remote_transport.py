"""Bounded HTTP transport for the existing local control/inference runtime.

Only observations and normalized action chunks cross the connection. Robot I/O,
clock deadlines, stop handling and action execution remain on the client host.
The same blocking ``infer`` method fits both existing synchronous and asynchronous
runtime modes; the runtime owns the single background inference worker.
"""
from __future__ import annotations

import hashlib
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import socket
import struct
import threading
import time
from typing import Any
from urllib.parse import urlsplit
import uuid
from collections import OrderedDict
import zlib

from .contracts import Observation
from .normalization import DatasetNormalizer

PROTOCOL = "realman-wam/1"
CONTENT_TYPE = "application/vnd.realman.observation"
MAGIC = b"RMW1"
MAX_BODY = 32 * 1024 * 1024
MAX_HEADER = 1024 * 1024
MAX_RESPONSE = 2 * 1024 * 1024


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def checkpoint_identity(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalizer_identity(normalizer: DatasetNormalizer) -> str:
    payload = dict(statistics=normalizer.stats, action_space=normalizer.action_space,
                   force_preprocessing=normalizer.force_preprocessing)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _matrix(value: Any, rows: int, columns: int, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{label} must contain {rows} rows")
    result = []
    for row in value:
        if not isinstance(row, (tuple, list)) or len(row) != columns:
            raise ValueError(f"{label} must contain {columns} values per row")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in row):
            raise ValueError(f"{label} contains non-finite/non-numeric data")
        result.append([float(x) for x in row])
    return result


def _history(value: Any, width: int):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("history must contain values, padding and valid-channel mask")
    rows = _matrix(value[0], 32, width + 6, "history")
    padding, valid = value[1:]
    if not isinstance(padding, list) or len(padding) != 32 or any(type(x) is not bool for x in padding):
        raise ValueError("history padding must contain 32 booleans")
    if (not isinstance(valid, list) or len(valid) != 32 or
            any(not isinstance(row, list) or len(row) != width + 6 or
                any(type(x) is not bool for x in row) for row in valid)):
        raise ValueError("history valid-channel mask has the wrong shape/type")
    return rows, padding, valid


def encode_request(observation: Observation, reference, history, *, width: int, height: int,
                   camera_order: list[str], identity: str, sampling_profile: str,
                   encoding: str = "zlib_rgb", request_id: str | None = None,
                   session_id: str = "codec", request_index: int = 0, deadline_ms: float = 1000.0) -> tuple[bytes, dict]:
    if encoding not in {"raw_rgb", "zlib_rgb"}:
        raise ValueError("image_encoding must be raw_rgb or zlib_rgb")
    if not 1 <= width <= 4096 or not 1 <= height <= 4096:
        raise ValueError("invalid image dimensions")
    frame_specs, buffers = [], []
    for name in camera_order:
        raw = observation.frames.get(name)
        if raw is None or len(raw) != width * height * 3:
            raise ValueError(f"camera {name} does not match configured RGB dimensions")
        payload = zlib.compress(raw, level=1) if encoding == "zlib_rgb" else raw
        buffers.append(payload)
        frame_specs.append(dict(name=name, encoding=encoding, width=width, height=height,
                                bytes=len(payload), raw_bytes=len(raw)))
    header = dict(protocol=PROTOCOL, request_id=request_id or uuid.uuid4().hex,
                  sequence=observation.sequence, source_stamp=observation.stamp,
                  session_id=session_id, request_index=request_index, deadline_ms=deadline_ms,
                  action_space=observation.action_space, state=observation.state,
                  force=observation.force, frame_meta=dict(observation.frame_meta),
                  force_meta=dict(observation.force_meta), frames=frame_specs,
                  history=_history(history, len(observation.state)), reference=reference,
                  normalization_identity=identity, sampling_profile=sampling_profile)
    packed = _json(header)
    body = MAGIC + struct.pack("!I", len(packed)) + packed + b"".join(buffers)
    if len(packed) > MAX_HEADER or len(body) > MAX_BODY:
        raise ValueError("observation exceeds transport size limit")
    return body, header


def decode_request(body: bytes, *, width: int, height: int, camera_order: list[str],
                   action_space: str, identity: str, sampling_profile: str):
    if len(body) > MAX_BODY or len(body) < 8 or body[:4] != MAGIC:
        raise ValueError("invalid observation framing")
    size = struct.unpack("!I", body[4:8])[0]
    if size > MAX_HEADER or 8 + size > len(body):
        raise ValueError("invalid observation metadata size")
    header = json.loads(body[8:8 + size])
    if not isinstance(header, dict) or header.get("protocol") != PROTOCOL:
        raise ValueError("unsupported observation protocol")
    if header.get("normalization_identity") != identity:
        raise ValueError("client/server normalization mismatch")
    if header.get("sampling_profile") != sampling_profile:
        raise ValueError("client/server sampling profile mismatch; restart server with selected profile")
    if header.get("action_space") != action_space:
        raise ValueError("client/server action space mismatch")
    request_id = header.get("request_id")
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
        raise ValueError("invalid request ID")
    if (not isinstance(header.get("session_id"), str) or not 1 <= len(header["session_id"]) <= 128 or
            type(header.get("request_index")) is not int or header["request_index"] < 0):
        raise ValueError("invalid client session/request index")
    deadline = header.get("deadline_ms")
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline) or not 10 <= deadline <= 10000:
        raise ValueError("invalid request deadline")
    specs = header.get("frames")
    if not isinstance(specs, list) or len(specs) != len(camera_order):
        raise ValueError("camera count mismatch")
    offset, frames = 8 + size, {}
    for name, spec in zip(camera_order, specs):
        raw_size = width * height * 3
        if (not isinstance(spec, dict) or spec.get("name") != name or
                spec.get("width") != width or spec.get("height") != height or
                spec.get("raw_bytes") != raw_size):
            raise ValueError("camera order/dimensions mismatch")
        length = spec.get("bytes")
        if type(length) is not int or length <= 0 or offset + length > len(body):
            raise ValueError("invalid image payload size")
        payload = body[offset:offset + length]
        offset += length
        if spec.get("encoding") == "raw_rgb":
            raw = payload
        elif spec.get("encoding") == "zlib_rgb":
            decoder = zlib.decompressobj()
            raw = decoder.decompress(payload, raw_size + 1)
            if decoder.unconsumed_tail or decoder.unused_data or not decoder.eof:
                raise ValueError("invalid/compressed image exceeds expected size")
        else:
            raise ValueError("unsupported image encoding")
        if len(raw) != raw_size:
            raise ValueError("RGB byte count mismatch")
        frames[name] = raw
    if offset != len(body):
        raise ValueError("unexpected trailing observation bytes")
    source_stamp = header.get("source_stamp")
    if isinstance(source_stamp, bool) or not isinstance(source_stamp, (int, float)) or not math.isfinite(source_stamp) or source_stamp <= 0:
        raise ValueError("invalid client source timestamp")
    # Host monotonic clocks have unrelated origins. The server's Observation gets
    # a server-local timestamp; source_stamp is only echoed to the owning client.
    observation = Observation(time.monotonic(), header["sequence"], tuple(header["state"]),
                              tuple(header["force"]), frames, header.get("frame_meta", {}),
                              header.get("force_meta", {}), action_space)
    history = _history(header.get("history"), len(observation.state))
    reference = header.get("reference")
    if reference is not None:
        reference = _matrix(reference, 32, len(observation.state), "reference")
    return observation, reference, history, header


class RemoteInferenceError(RuntimeError):
    pass


class RemoteJointPolicy:
    """CPU-only policy adapter; the existing runtime chooses sync/async execution."""

    # DeploymentRuntime may use this to avoid rejecting a client-only process
    # before it has attached its local camera/robot adapter.
    requires_local_cuda = False

    def __init__(self, checkpoint: str | Path | None, config: dict, *, config_path: str | Path):
        self.config = dict(config)
        self.options = dict(config.get("remote_inference", {}))
        self.device = "qizhi_h200"
        self.checkpoint = str(checkpoint or "remote")
        self.normalizer = DatasetNormalizer.from_json(self._path(config["normalization_file"], config_path))
        self.identity = normalizer_identity(self.normalizer)
        self.url = str(self.options.get("url", "")).rstrip("/")
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.query or parsed.fragment or parsed.username:
            raise ValueError("remote_inference.url must be an http(s) service URL without credentials/query")
        self._parsed = parsed
        self._prefix = parsed.path.rstrip("/")
        self._lock = threading.Lock()
        self._connection = None
        self._last_used_at = 0.0
        self._closed = False
        self._session_id = uuid.uuid4().hex
        self._request_index = 0
        self.max_latency_s = float(self.options.get("max_latency_s", 1.0))
        if not math.isfinite(self.max_latency_s) or not 0.01 <= self.max_latency_s <= 10:
            raise ValueError("max remote latency must be 0.01..10 seconds")
        self.timeout_s = float(self.options.get("timeout_s", 5.0))
        if not math.isfinite(self.timeout_s) or not 0.1 <= self.timeout_s <= 120:
            raise ValueError("remote timeout must be 0.1..120 seconds")
        self._token = os.environ.get(str(self.options.get("token_env", "REALMAN_REMOTE_TOKEN")), "")
        if not self._token:
            raise ValueError("remote token environment variable is empty")
        self.sampling_profile = str(config.get("sampling_profile", "4416"))
        self.image_encoding = str(self.options.get("image_encoding", "zlib_rgb"))
        self.camera_order = list(config.get("camera_order", ["global", "wrist"]))
        self.width, self.height = int(config.get("camera_width", 640)), int(config.get("camera_height", 480))
        declared_sha = self.options.get("checkpoint_sha256")
        if checkpoint and Path(checkpoint).expanduser().is_file():
            actual_sha = checkpoint_identity(checkpoint)
            if declared_sha and declared_sha != actual_sha:
                raise ValueError("configured remote checkpoint SHA differs from selected local checkpoint")
            self.expected_checkpoint = actual_sha
        elif declared_sha:
            self.expected_checkpoint = str(declared_sha)
        else:
            raise ValueError("remote inference requires the selected local checkpoint or checkpoint_sha256")
        if len(self.expected_checkpoint) != 64 or any(x not in "0123456789abcdef" for x in self.expected_checkpoint):
            raise ValueError("invalid checkpoint SHA256")
        try:
            metadata = self._request("GET", "/health", None, timeout_s=self.timeout_s)
            self._validate_metadata(metadata)
        except Exception:
            self.close()
            raise
        self.reference_mode = metadata["reference_mode"]
        self.diagnostics = dict(inference_device=self.device, remote=metadata, transport=PROTOCOL)

    @staticmethod
    def _path(value, config_path):
        path = Path(value).expanduser()
        return path if path.is_absolute() else Path(config_path).resolve().parent / path

    def _validate_metadata(self, metadata):
        expected = dict(protocol=PROTOCOL, ready=True, normalization_identity=self.identity,
                        action_space=self.normalizer.action_space, sampling_profile=self.sampling_profile,
                        camera_order=self.camera_order, camera_width=self.width, camera_height=self.height)
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RemoteInferenceError(f"remote metadata mismatch: {key}")
        if metadata.get("reference_mode") not in {"none", "rolling_ae"}:
            raise RemoteInferenceError("remote reference mode unsupported")
        if metadata.get("checkpoint_sha256") != self.expected_checkpoint:
            raise RemoteInferenceError("remote checkpoint identity mismatch")

    def _request(self, method, path, body, *, timeout_s: float | None = None):
        if self._closed:
            raise RemoteInferenceError("remote policy is closed")
        request_timeout = self.timeout_s if timeout_s is None else min(self.timeout_s, float(timeout_s))
        # The server closes idle HTTP connections after ten seconds. Reconnect
        # proactively after a pause, before sending any new inference request.
        if self._connection is not None and time.monotonic() - self._last_used_at > 5.0:
            self._connection.close()
            self._connection = None
        if self._connection is None:
            cls = http.client.HTTPSConnection if self._parsed.scheme == "https" else http.client.HTTPConnection
            self._connection = cls(self._parsed.hostname, self._parsed.port, timeout=request_timeout)
        # A proxy may close the connection after any response. http.client then
        # reconnects this same object on the next request using its timeout
        # attribute, so update both the object and any still-open socket.
        self._connection.timeout = request_timeout
        if self._connection.sock is not None:
            self._connection.sock.settimeout(request_timeout)
        headers = {"Authorization": "Bearer " + self._token, "Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = CONTENT_TYPE
        connection = self._connection
        try:
            connection.request(method, self._prefix + path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read(MAX_RESPONSE + 1)
            if len(payload) > MAX_RESPONSE:
                raise RemoteInferenceError("server response exceeds size limit")
            response_bytes = int(response.getheader("Content-Length") or len(payload))
            content_type = response.getheader("Content-Type") or "unspecified"
            status = f"remote HTTP {response.status} (Content-Type: {content_type})"
            try:
                data = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                # Platform gateways often return an HTML login/error page.
                # Preserve its status/type while avoiding body/URL/secret echo.
                raise RemoteInferenceError(status + ": non-JSON response") from exc
            if response.status != 200:
                detail = str(data.get("error", "request failed")) if isinstance(data, dict) else "request failed"
                detail = detail.replace(self._token, "[redacted]").replace(self.url, "[remote URL]")
                raise RemoteInferenceError(f"{status}: {detail}")
            if not isinstance(data, dict):
                raise RemoteInferenceError("invalid server response")
            data["_response_bytes"] = response_bytes
            self._last_used_at = time.monotonic()
            return data
        except (OSError, http.client.HTTPException, json.JSONDecodeError, RemoteInferenceError) as exc:
            connection.close()
            self._connection = None
            # Never automatically retry an inference: a timed-out request may
            # still be computing remotely, and duplicate/stale chunks are unsafe.
            raise RemoteInferenceError(str(exc)) from exc

    def infer(self, observation: Observation, normalized_reference, history):
        with self._lock:
            started = time.monotonic()
            self._request_index += 1
            body, header = encode_request(observation, normalized_reference, history,
                width=self.width, height=self.height, camera_order=self.camera_order,
                identity=self.identity, sampling_profile=self.sampling_profile, encoding=self.image_encoding,
                session_id=self._session_id, request_index=self._request_index, deadline_ms=self.max_latency_s * 1000)
            encoded = time.monotonic()
            response = self._request("POST", "/v1/infer", body, timeout_s=self.max_latency_s)
            received = time.monotonic()
            if (response.get("request_id") != header["request_id"] or
                    response.get("session_id") != self._session_id or
                    response.get("request_index") != self._request_index or
                    response.get("checkpoint_sha256") != self.expected_checkpoint or
                    response.get("source_sequence") != observation.sequence or
                    response.get("source_stamp") != observation.stamp or
                    response.get("normalization_identity") != self.identity or
                    response.get("action_space") != observation.action_space or
                    response.get("sampling_profile") != self.sampling_profile):
                raise RemoteInferenceError("stale/mismatched action response")
            actions = _matrix(response.get("actions"), 32, self.normalizer.action_dim, "actions")
            elapsed = (time.monotonic() - started) * 1000
            if elapsed > self.max_latency_s * 1000:
                raise RemoteInferenceError(f"remote result exceeded end-to-end deadline ({elapsed:.1f} ms)")
            server = response.get("server_timing_ms", {})
            processing_ms = float(server.get("processing", 0))
            self.diagnostics["last_inference"] = dict(
                elapsed_ms=elapsed, encode_ms=(encoded - started) * 1000,
                roundtrip_ms=(received - encoded) * 1000,
                transport_overhead_ms=max(0.0, elapsed - processing_ms),
                non_model_overhead_ms=max(0.0, elapsed - float(server.get("inference", 0))),
                upload_bytes=len(body), download_bytes=int(response.get("_response_bytes", 0)),
                server_timing_ms=server, remote_model=response.get("model_diagnostics", {}),
                source_sequence=observation.sequence, image_encoding=self.image_encoding)
            return actions

    def close(self):
        # close may be called while a network operation is pending; socket close
        # interrupts it. Runtime epoch checks discard any result after pause/stop.
        self._closed = True
        connection, self._connection = self._connection, None
        if connection:
            if connection.sock is not None:
                try:
                    connection.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            connection.close()


class RemotePolicyService:
    """One loaded CUDA policy with no request backlog or robot interfaces."""

    def __init__(self, policy, config: dict, *, checkpoint_sha256: str = ""):
        self.policy, self.config = policy, dict(config)
        if (len(checkpoint_sha256) != 64 or
                any(ch not in "0123456789abcdef" for ch in checkpoint_sha256)):
            raise ValueError("server checkpoint SHA256 must be a lowercase 64-character hex string")
        self.normalizer = policy.normalizer
        self.identity = normalizer_identity(self.normalizer)
        self.lock = threading.Lock()
        self._sessions = OrderedDict()
        self.metadata = dict(protocol=PROTOCOL, ready=True,
            normalization_identity=self.identity, action_space=self.normalizer.action_space,
            sampling_profile=str(config.get("sampling_profile", "4416")),
            camera_order=list(config.get("camera_order", ["global", "wrist"])),
            camera_width=int(config.get("camera_width", 640)), camera_height=int(config.get("camera_height", 480)),
            reference_mode=getattr(policy, "reference_mode", "rolling_ae"),
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_name=Path(getattr(policy, "checkpoint", "remote")).name,
            device=getattr(policy, "device", "unknown"),
            gpu_name=dict(getattr(policy, "diagnostics", {}).get("device", {})).get("name", "unknown"),
            dtype=dict(getattr(policy, "diagnostics", {}).get("device", {})).get("dtype", "bfloat16"),
            inference_backend=getattr(policy, "inference_backend", "remote_policy"),
            requires_local_cuda=False,
            model_name=type(policy).__name__,
            optimization=dict(getattr(policy, "diagnostics", {}).get("optimization", {})),
            model_diagnostics=dict(getattr(policy, "diagnostics", {})))

    def infer(self, body):
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("model is busy; no request queue is kept")
        try:
            started = time.monotonic()
            obs, reference, history, header = decode_request(body,
                width=self.metadata["camera_width"], height=self.metadata["camera_height"],
                camera_order=self.metadata["camera_order"], action_space=self.normalizer.action_space,
                identity=self.identity, sampling_profile=self.metadata["sampling_profile"])
            decoded = time.monotonic()
            session, index = header["session_id"], header["request_index"]
            if index <= self._sessions.get(session, -1):
                raise ValueError("duplicate/out-of-order request")
            self._sessions[session] = index
            self._sessions.move_to_end(session)
            while len(self._sessions) > 1024:
                self._sessions.popitem(last=False)
            actions = _matrix(self.policy.infer(obs, reference, history), 32,
                              self.normalizer.action_dim, "actions")
            finished = time.monotonic()
            if (finished - started) * 1000 > header["deadline_ms"]:
                raise TimeoutError("server processing exceeded request deadline")
            diagnostics = getattr(self.policy, "diagnostics", {})
            return dict(protocol=PROTOCOL, request_id=header["request_id"], source_sequence=header["sequence"],
                source_stamp=header["source_stamp"], session_id=session, request_index=index,
                checkpoint_sha256=self.metadata["checkpoint_sha256"], actions=actions, action_space=self.normalizer.action_space,
                sampling_profile=self.metadata["sampling_profile"], normalization_identity=self.identity,
                server_timing_ms=dict(decode=(decoded - started) * 1000,
                                      inference=(finished - decoded) * 1000,
                                      processing=(finished - started) * 1000),
                model_diagnostics={key: diagnostics[key] for key in ("last_inference", "optimization", "sampling")
                                   if key in diagnostics})
        finally:
            self.lock.release()


def make_server(service: RemotePolicyService, *, host: str = "127.0.0.1", port: int = 8000,
                token: str, read_timeout_s: float = 10.0) -> ThreadingHTTPServer:
    if not isinstance(token, str) or len(token) < 16 or "\n" in token or "\r" in token:
        raise ValueError("server token must contain at least 16 characters")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(read_timeout_s)
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def _authorized(self):
            supplied = self.headers.get("Authorization", "")
            return hmac.compare_digest(supplied.encode(), ("Bearer " + token).encode())

        def _reply(self, status, data):
            body = _json(data)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if status != 200:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                self.close_connection = True

        def do_GET(self):
            if not self._authorized():
                return self._reply(401, dict(error="authentication required"))
            if self.path.rstrip("/") != "/health":
                return self._reply(404, dict(error="not found"))
            self._reply(200, dict(service.metadata))

        def do_POST(self):
            if not self._authorized():
                return self._reply(401, dict(error="authentication required"))
            if self.path != "/v1/infer":
                return self._reply(404, dict(error="not found"))
            if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Type") != CONTENT_TYPE:
                return self._reply(400, dict(error="use Content-Length and the observation content type"))
            try:
                size = int(self.headers.get("Content-Length", "-1"))
                if not 8 <= size <= MAX_BODY:
                    return self._reply(413, dict(error="invalid observation size"))
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError("truncated request")
                result = service.infer(body)
            except BlockingIOError as exc:
                return self._reply(429, dict(error=str(exc)))
            except (ValueError, TypeError, KeyError, zlib.error) as exc:
                return self._reply(400, dict(error=str(exc)))
            except TimeoutError as exc:
                return self._reply(504, dict(error=str(exc)))
            except OSError:
                self.close_connection = True
                return
            except Exception:
                import logging
                logging.getLogger(__name__).exception("remote inference failed")
                return self._reply(500, dict(error="model inference failed; inspect server log"))
            self._reply(200, result)

        def log_message(self, fmt, *args):
            import logging
            logging.getLogger(__name__).info("%s %s", self.address_string(), fmt % args)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
