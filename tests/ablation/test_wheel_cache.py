"""Offline checks for the verified/resumable CUDA wheel staging path."""
import hashlib
import importlib.util
import io
from pathlib import Path
import time
from unittest import mock

import pytest

spec = importlib.util.spec_from_file_location("wheel_cache", Path(__file__).resolve().parents[2] / "scripts/ablation/prefetch_fastwam_wheels.py")
wheel_cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wheel_cache)


class Response(io.BytesIO):
    def __init__(self, data, status, content_range=None):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range} if content_range else {}


def test_pins_include_all_supported_interpreters_and_sha256():
    assert set(wheel_cache.WHEELS) == {"cp310", "cp311", "cp312"}
    for records in wheel_cache.WHEELS.values():
        assert [record[0] for record in records] == ["torch", "torchvision"]
        for _, _, size, digest in records:
            assert size > 8_000_000
            assert len(bytes.fromhex(digest)) == 32


def test_http_cache_reuse_rejects_same_length_corruption(tmp_path):
    body = b"valid wheel bytes"
    digest = hashlib.sha256(body).hexdigest()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "bad.body").write_bytes(b"X" * len(body))
    (cache / "good.body").write_bytes(body)
    destination = tmp_path / "example.whl"
    assert wheel_cache.reuse(destination, len(body), digest, [cache])
    assert destination.read_bytes() == body
    (cache / "good.body").write_bytes(b"Z" * len(body))
    assert destination.read_bytes() == body  # Copy, no link to mutable shared cache.


@pytest.mark.parametrize("ignores_range", [False, True])
def test_resume_or_restart_if_server_ignores_range(tmp_path, ignores_range):
    body = b"complete wheel bytes"
    destination = tmp_path / "wheel.whl"
    partial = destination.with_suffix(".whl.part")
    partial.write_bytes(body[:7])
    response = Response(body if ignores_range else body[7:], 200 if ignores_range else 206,
                        None if ignores_range else f"bytes 7-{len(body)-1}/{len(body)}")
    with mock.patch.object(wheel_cache.urllib.request, "urlopen", return_value=response) as request:
        wheel_cache.download(destination, len(body), hashlib.sha256(body).hexdigest(), ["https://official.example"], time.monotonic() + 60)
    assert request.call_args.args[0].get_header("Range") == "bytes=7-"
    assert destination.read_bytes() == body
    assert not partial.exists()


def test_bad_digest_never_promoted(tmp_path):
    body = b"corrupt file"
    destination = tmp_path / "wheel.whl"
    with mock.patch.object(wheel_cache.urllib.request, "urlopen", return_value=Response(body, 200)):
        with pytest.raises(RuntimeError, match="official pin"):
            wheel_cache.download(destination, len(body), "0" * 64, ["https://official.example"], time.monotonic() + 60)
    assert not destination.exists()
    assert len(list(tmp_path.glob("*.invalid-*"))) == 1


def test_expired_total_budget_does_not_start_network(tmp_path):
    with mock.patch.object(wheel_cache.urllib.request, "urlopen") as request:
        with pytest.raises(RuntimeError, match="time budget"):
            wheel_cache.download(tmp_path / "wheel.whl", 99, "0" * 64, ["https://official.example"], time.monotonic() - 1)
    request.assert_not_called()
