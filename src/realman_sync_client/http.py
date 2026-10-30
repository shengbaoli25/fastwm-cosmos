"""Bounded, one-request-at-a-time JSON HTTP transport without retries."""
from __future__ import annotations

import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .wire import MAX_BODY_BYTES, PROTOCOL


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "redirects are disabled", headers, fp)


class InferenceHTTPClient:
    def __init__(self, server_url: str, *, timeout_s: float = 120.0, token: str | None = None):
        parts = urlsplit(server_url)
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username or parts.password or parts.query or parts.fragment
                or parts.path not in {"", "/"}):
            raise ValueError("server URL must be an http(s) origin without credentials or path")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        self.server_url = server_url.rstrip("/")
        self.timeout_s = timeout_s
        self.token = os.environ.get("REALMAN_SERVER_TOKEN") if token is None else token
        if self.token and any(ord(character) < 33 or ord(character) > 126 for character in self.token):
            raise ValueError("bearer token must contain printable ASCII without whitespace")
        # Robot traffic uses the explicitly selected origin directly. Redirects
        # and environment proxies cannot replay or reroute an inference request.
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    def _request(self, path, data=None, *, timeout_s=None):
        body = None if data is None else json.dumps(
            data, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if body is not None and len(body) > MAX_BODY_BYTES:
            raise ValueError("request exceeds 64 MiB")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = Request(self.server_url + path, data=body, headers=headers,
            method="GET" if body is None else "POST")
        try:
            with self.opener.open(request, timeout=self.timeout_s if timeout_s is None else timeout_s) as response:
                raw = response.read(MAX_BODY_BYTES + 1)
                if len(raw) > MAX_BODY_BYTES:
                    raise ValueError("response exceeds 64 MiB")
                result = json.loads(raw, parse_constant=_reject_constant)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"inference HTTP request failed; no retry: {exc}") from exc
        if not isinstance(result, dict):
            raise ValueError("server response must be a JSON object")
        if result.get("protocol") != PROTOCOL:
            raise ValueError("server protocol mismatch")
        return result

    def health(self):
        return self._request("/health")

    def infer(self, request):
        return self._request("/infer", request)

    def release(self, session_id):
        return self._request("/session/release", dict(protocol=PROTOCOL, session_id=session_id),
                             timeout_s=min(2.0, self.timeout_s))
