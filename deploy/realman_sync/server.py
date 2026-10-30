"""Single in-flight inference worker with a bounded HTTP request interface."""
from __future__ import annotations

import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import threading
import time

from .wire import MAX_BODY_BYTES, PROTOCOL, decode_request, validate_actions

logger = logging.getLogger(__name__)


class Conflict(ValueError):
    pass


class InferenceService:
    """One resident policy; caller owns all robot control and capture timing."""
    def __init__(self, policy, *, session_timeout_s=300):
        self.policy = policy
        self.metadata = dict(policy.metadata)
        self.lock = threading.Lock()
        self.closing = threading.Event()
        self.session_id = None
        self.last_sequence = -1
        self.last_stamp = 0.0
        self.request_ids = set()
        self.last_activity = 0.0
        self.session_timeout_s = float(session_timeout_s)
        if self.session_timeout_s <= 0:
            raise ValueError("session timeout must be positive")
        self.failed = False

    def infer(self, payload):
        if not self.lock.acquire(blocking=False):
            raise Conflict("Inference busy; synchronous requests must not overlap")
        try:
            if self.closing.is_set():
                raise Conflict("Server is shutting down")
            request = decode_request(payload, self.metadata)
            now = time.monotonic()
            session = request["session_id"]
            obs = request["observation"]
            if session != self.session_id:
                if self.session_id is not None and now - self.last_activity <= self.session_timeout_s:
                    raise Conflict("Another client owns this policy session")
                self.session_id = session
                self.last_sequence, self.last_stamp = -1, 0.0
                self.request_ids.clear()
                self.failed = False
            if self.failed:
                raise Conflict("Session failed; restart the client and release the session before continuing")
            if (request["request_id"] in self.request_ids or obs["sequence"] <= self.last_sequence
                    or obs["stamp"] <= self.last_stamp):
                raise Conflict("Duplicate or out-of-order request; inference is never retried")
            if len(self.request_ids) >= 100000:
                raise Conflict("Session request limit reached; release and start a fresh session")
            # Consume before inference; a failed call must not accidentally be replayed.
            self.request_ids.add(request["request_id"])
            self.last_sequence, self.last_stamp = obs["sequence"], obs["stamp"]
            self.last_activity = now
            started = time.monotonic()
            try:
                actions = self.policy.predict(obs, request["history"], request["video_context"])
                actions = validate_actions(actions, self.metadata["action_space"], self.metadata["action_horizon"])
            except Exception:
                self.failed = True
                raise
            finally:
                self.last_activity = time.monotonic()
            return dict(protocol=PROTOCOL, session_id=session, request_id=request["request_id"],
                        source_sequence=obs["sequence"], action_space=self.metadata["action_space"],
                        control_hz=self.metadata["control_hz"], actions=actions,
                        inference_ms=(time.monotonic() - started) * 1000)
        finally:
            self.lock.release()

    def close(self):
        # Reject late readers before waiting for the current GPU call to finish.
        self.closing.set()
        with self.lock:
            self.policy.close()

    def release(self, payload):
        if not self.lock.acquire(blocking=False):
            raise Conflict("Cannot release while inference is running")
        try:
            if (not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL
                    or not isinstance(payload.get("session_id"), str)):
                raise ValueError("Invalid session release")
            if self.session_id is not None and payload["session_id"] != self.session_id:
                raise Conflict("Session ownership mismatch")
            self.session_id = None
            self.request_ids.clear()
            self.failed = False
            return dict(protocol=PROTOCOL, released=True)
        finally:
            self.lock.release()


def _reject_constant(value):
    raise ValueError("Nonfinite JSON number: " + value)


class PolicyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, service, *, token=None, request_timeout_s=30):
        self.service = service
        self.token = token
        self.request_timeout_s = request_timeout_s
        # Bound simultaneous connection handlers, including slow request readers.
        self.slots = threading.BoundedSemaphore(8)
        super().__init__(address, PolicyHandler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class PolicyHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(self.server.request_timeout_s)

    def log_message(self, fmt, *args):
        logger.info("%s %s", self.client_address[0], fmt % args)

    def _reply(self, status, payload):
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def _authorized(self):
        expected = self.server.token
        if expected and not hmac.compare_digest(
                self.headers.get("Authorization", "").encode(), ("Bearer " + expected).encode()):
            self._reply(401, dict(error="Unauthorized", protocol=PROTOCOL))
            return False
        return True

    def do_GET(self):
        if not self._authorized():
            return
        if self.path != "/health":
            self._reply(404, dict(error="Unknown endpoint"))
            return
        self._reply(200, dict(protocol=PROTOCOL, ready=not self.server.service.closing.is_set(),
                              metadata=self.server.service.metadata))

    def do_POST(self):
        if not self._authorized():
            return
        if self.path not in {"/infer", "/session/release"}:
            self._reply(404, dict(error="Unknown endpoint"))
            return
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Content-Type must be application/json")
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Transfer-Encoding is unsupported")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1:
                raise ValueError("One Content-Length header is required")
            length = int(lengths[0])
            if not 0 < length <= MAX_BODY_BYTES:
                self._reply(413, dict(error="Request exceeds body limit"))
                return
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete request body")
            payload = json.loads(raw, parse_constant=_reject_constant)
            result = (self.server.service.infer(payload) if self.path == "/infer"
                      else self.server.service.release(payload))
            self._reply(200, result)
        except Conflict as exc:
            self._reply(409, dict(protocol=PROTOCOL, error=str(exc)))
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
            self._reply(400, dict(protocol=PROTOCOL, error=str(exc)))
        except TimeoutError:
            self._reply(408, dict(protocol=PROTOCOL, error="Request read timeout"))
        except Exception:
            logger.exception("Policy request failed")
            self._reply(500, dict(protocol=PROTOCOL, error="Inference failed; inspect server log"))
