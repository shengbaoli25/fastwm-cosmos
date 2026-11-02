"""Stage official CUDA 12.8 wheels before pip resolves dependencies on its configured mirror.

Pure standard library, no installation or model downloads. Resumable downloads
are promoted only after their size and the official index SHA256 both match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import urllib.parse
import urllib.request

# Official https://download.pytorch.org/whl/cu128/{torch,torchvision}/ index
# SHA256 fragments and official CDN Content-Length, checked 2026-11-02.
WHEELS = {
    "cp310": [
        ("torch", "2.7.1", 1039365846, "d6c3cba198dc93f93422a8545f48a6697890366e4b9701f54351fc27e2304bd3"),
        ("torchvision", "0.22.1", 8730539, "538f4db667286d939b4eee0a66d31ed21b51186668006b0e0ffe20338ecc7e00"),
    ],
    "cp311": [
        ("torch", "2.7.1", 1039389795, "c301dc280458afd95450af794924c98fe07522dd148ff384739b810e3e3179f2"),
        ("torchvision", "0.22.1", 8732234, "92568ac46b13a8c88b61589800b1b9c4629be091ea7ce080fc6fc622e11e0915"),
    ],
    "cp312": [
        ("torch", "2.7.1", 1039245529, "0b64f7d0a6f2a739ed052ba959f7b67c677028c9566ce51997f9f90fe573ddaa"),
        ("torchvision", "0.22.1", 8730624, "f64ef9bb91d71ab35d8384912a19f7419e35928685bc67544d58f45148334373"),
    ],
}
OFFICIAL_BASES = ("https://download-r2.pytorch.org/whl/cu128", "https://download.pytorch.org/whl/cu128")


def verify(path: Path, size: int, digest: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest() == digest


def reuse(destination: Path, size: int, digest: str, roots: list[Path]) -> bool:
    for root in roots:
        if not root.is_dir():
            continue
        # Only operator-specified cache directories are traversed. Pip HTTP cache
        # bodies are reusable wheel bytes even though they have opaque filenames.
        for directory, _, names in os.walk(root):
            for name in names:
                if name != destination.name and not name.endswith(".body"):
                    continue
                candidate = Path(directory) / name
                if candidate.resolve() == destination.resolve() or not verify(candidate, size, digest):
                    continue
                temporary = destination.with_suffix(destination.suffix + ".copying")
                shutil.copyfile(candidate, temporary)
                if not verify(temporary, size, digest):
                    raise ValueError(f"Cached wheel changed while copying: {candidate}")
                os.replace(temporary, destination)
                print(f"Verified cached wheel: {candidate} -> {destination}", flush=True)
                return True
    return False


def download(destination: Path, size: int, digest: str, bases: list[str], deadline: float,
             *, low_rate_seconds: float = 90, min_bytes_per_second: int = 128 * 1024):
    partial = destination.with_suffix(destination.suffix + ".part")
    if verify(partial, size, digest):
        os.replace(partial, destination)
        return
    if partial.exists() and partial.stat().st_size >= size:
        # Keep corrupt bytes available for inspection; never mistake a complete
        # length for a completed artifact or append to an invalid complete file.
        os.replace(partial, partial.with_name(partial.name + f".invalid-{time.time_ns()}"))
    failures = []
    for base in bases:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        offset = partial.stat().st_size if partial.exists() else 0
        url = base.rstrip("/") + "/" + urllib.parse.quote(destination.name)
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-", "Accept-Encoding": "identity"})
        print(f"Download {url}, resume={offset}/{size}, remaining budget={remaining:.0f}s", flush=True)
        try:
            with urllib.request.urlopen(request, timeout=min(20, remaining)) as response:
                if response.status == 206:
                    expected = f"bytes {offset}-{size - 1}/{size}"
                    if response.headers.get("Content-Range") != expected:
                        raise ValueError(f"Unexpected Content-Range: {response.headers.get('Content-Range')}")
                elif response.status == 200:
                    offset = 0  # Server ignored Range: overwrite this owned partial.
                else:
                    raise ValueError(f"Unexpected HTTP status: {response.status}")
                started = last_log = time.monotonic()
                transferred = 0
                with partial.open("ab" if offset else "wb") as handle:
                    while True:
                        now = time.monotonic()
                        if now >= deadline:
                            raise TimeoutError("Wheel download total time budget exhausted")
                        # read1 returns available bytes; read(1MiB) could block for
                        # minutes on a continuously trickling connection.
                        block = response.read1(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                        transferred += len(block)
                        if offset + transferred > size:
                            raise ValueError("Download exceeds the pinned official wheel size")
                        elapsed = time.monotonic() - started
                        if time.monotonic() - last_log >= 15:
                            rate = transferred / max(elapsed, 0.001)
                            print(f"{destination.name}: {offset + transferred}/{size} bytes, {rate/1024:.0f} KiB/s", flush=True)
                            last_log = time.monotonic()
                        if elapsed >= low_rate_seconds and transferred / elapsed < min_bytes_per_second:
                            raise TimeoutError("Sustained transfer below 128KiB/s; keeping .part and trying the next source")
            if not verify(partial, size, digest):
                raise ValueError("Downloaded wheel size/SHA256 differs from the official pin")
            os.replace(partial, destination)
            print(f"Verified downloaded wheel SHA256: {digest}", flush=True)
            return
        except (OSError, ValueError) as error:
            failures.append(f"{base}: {error}")
            print(f"Wheel source failed: {failures[-1]}", flush=True)
            if partial.exists() and partial.stat().st_size >= size:
                os.replace(partial, partial.with_name(partial.name + f".invalid-{time.time_ns()}"))
    raise RuntimeError(f"Unable to stage {destination.name} within the time budget. Resume by rerunning; place a verified wheel in the wheelhouse or select another HTTPS base. " + "; ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", required=True, type=Path)
    parser.add_argument("--reuse-root", action="append", type=Path, default=[])
    parser.add_argument("--base-url", action="append", default=[], help="Optional faster mirror; bytes must match the official SHA256")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if sys.implementation.name != "cpython" or platform.system() != "Linux" or platform.machine() != "x86_64" or tag not in WHEELS:
        parser.error("Requires Linux x86_64 CPython 3.10, 3.11 or 3.12")
    if args.timeout_seconds <= 0 or any(not base.startswith("https://") for base in args.base_url):
        parser.error("Use a positive total timeout and HTTPS mirror URLs")
    wheelhouse = args.wheelhouse.resolve()
    wheelhouse.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.timeout_seconds
    receipts = []
    for package, version, size, digest in WHEELS[tag]:
        destination = wheelhouse / f"{package}-{version}+cu128-{tag}-{tag}-manylinux_2_28_x86_64.whl"
        if destination.exists() and not verify(destination, size, digest):
            raise ValueError(f"Existing wheel differs from the official pin; inspect/move it before retrying: {destination}")
        if not destination.exists() and not reuse(destination, size, digest, args.reuse_root):
            download(destination, size, digest, args.base_url + list(OFFICIAL_BASES), deadline)
        receipts.append({"file": str(destination), "size": size, "sha256": digest})
    (wheelhouse / "verified-wheels.json").write_text(json.dumps(receipts, indent=2) + "\n")
    # Do not add pip --hash flags: that would demand hashes for every transitive
    # dependency too. Both explicit local wheels were already verified above.
    (wheelhouse / "requirements.txt").write_text("".join(Path(item["file"]).as_uri() + "\n" for item in receipts))


if __name__ == "__main__":
    main()
