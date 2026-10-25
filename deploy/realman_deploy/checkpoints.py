"""Checkpoint catalog and verified desktop downloads."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointSpec:
    id: str
    label: str
    source: str
    sha256: str | None = None
    size_bytes: int | None = None
    manifest: str | None = None

    def as_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "source": self.source,
                "sha256": self.sha256, "size_bytes": self.size_bytes,
                "manifest": self.manifest}


class CheckpointCatalog:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.specs: dict[str, CheckpointSpec] = {}
        self.reload()

    def reload(self) -> None:
        payload = json.loads(self.path.read_text()) if self.path.is_file() else {"checkpoints": []}
        entries = payload if isinstance(payload, list) else payload.get("checkpoints", [])
        if not isinstance(entries, list):
            raise ValueError("checkpoint catalog must contain a list")
        specs = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("checkpoint catalog entries must be objects")
            ident = str(entry.get("id", ""))
            if not ident or ident in specs or "/" in ident or "\\" in ident or ".." in ident:
                raise ValueError("checkpoint ids must be unique safe names")
            source = str(entry.get("source", ""))
            if not source:
                raise ValueError(f"checkpoint {ident} has no source")
            digest = entry.get("sha256")
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                       or any(c not in "0123456789abcdef" for c in digest.lower())):
                raise ValueError(f"checkpoint {ident} has an invalid sha256")
            specs[ident] = CheckpointSpec(ident, str(entry.get("label", ident)), source,
                                          digest.lower() if digest else None,
                                          int(entry["size_bytes"]) if entry.get("size_bytes") is not None else None,
                                          str(entry["manifest"]) if entry.get("manifest") else None)
        self.specs = specs

    def public(self, root: Path) -> list[dict]:
        result = []
        known = set()
        for spec in self.specs.values():
            path = root / f"{spec.id}.pt"
            installed = path.is_file()
            size_ok = installed and (spec.size_bytes is None or path.stat().st_size == spec.size_bytes)
            result.append({**spec.as_dict(), "installed": size_ok,
                           "verified": False, "path": str(path) if size_ok else None})
            known.add(path.name)
        for path in sorted(root.glob("*.pt")):
            if path.name not in known and path.is_file():
                result.append({"id": path.stem, "label": path.stem, "source": "local",
                               "sha256": None, "size_bytes": path.stat().st_size,
                               "manifest": None, "installed": True, "path": str(path)})
        return result

    def get(self, ident: str) -> CheckpointSpec:
        try:
            return self.specs[ident]
        except KeyError as exc:
            raise KeyError(f"unknown checkpoint: {ident}") from exc


class CheckpointStore:
    """Atomic downloads to ``~/桌面/checkpoints`` by default.

    Existing files are never replaced unless their verified digest differs;
    replacement is staged in a temporary file and atomically renamed.
    """

    def __init__(self, catalog: CheckpointCatalog, root: str | Path | None = None):
        self.catalog = catalog
        self.root = Path(root or (Path.home() / "桌面" / "checkpoints")).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._active: str | None = None

    def list(self) -> list[dict]:
        self.catalog.reload()
        return self.catalog.public(self.root)

    def path_for(self, ident: str) -> Path:
        try:
            name = self.catalog.get(ident).id
        except KeyError:
            name = ident
            if (not name or "/" in name or "\\" in name or ".." in name
                    or not (self.root / f"{name}.pt").is_file()):
                raise
        return self.root / f"{name}.pt"

    @staticmethod
    def _local_preprocessing_receipt(spec: CheckpointSpec, target: Path) -> bytes | None:
        """Preserve an exported binding when a local checkpoint is renamed.

        This is not a new binding: normalization/force identity are still checked
        by the policy loader against the runtime's actual normalization file.
        """
        parsed = urlparse(spec.source)
        if parsed.scheme not in ("", "file"):
            return None
        source = Path(parsed.path if parsed.scheme == "file" else spec.source).expanduser().resolve()
        source_receipt = Path(str(source) + ".force.json")
        if not source_receipt.is_file():
            return None
        receipt_bytes = source_receipt.read_bytes()
        receipt = json.loads(receipt_bytes)
        if (not isinstance(receipt, dict)
                or receipt.get("schema") != "realman_checkpoint_preprocessing_v1"
                or receipt.get("checkpoint_sha256") != spec.sha256):
            raise ValueError("checkpoint preprocessing binding differs from catalog SHA256")
        for key in ("normalization_sha256", "force_preprocessing_id"):
            value = receipt.get(key)
            if (not isinstance(value, str) or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)):
                raise ValueError("invalid checkpoint preprocessing binding: " + key)
        target_receipt = Path(str(target) + ".force.json")
        if target_receipt.exists() and json.loads(target_receipt.read_bytes()) != receipt:
            raise ValueError("existing checkpoint preprocessing binding differs; refusing to overwrite")
        return receipt_bytes

    @staticmethod
    def _install_preprocessing_receipt(target: Path, receipt: bytes | None) -> None:
        if receipt is None:
            return
        destination = Path(str(target) + ".force.json")
        if destination.exists():
            return
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent,
                    prefix="." + target.name + ".force-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(receipt)
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def download(self, ident: str, progress=None) -> Path:
        spec = self.catalog.get(ident)
        if spec.sha256 is None:
            raise ValueError("catalog downloads require a pinned SHA256")
        with self._lock:
            if self._active is not None:
                raise RuntimeError(f"checkpoint download already active: {self._active}")
            self._active = ident
        try:
            target = self.path_for(ident)
            receipt = self._local_preprocessing_receipt(spec, target)
            if target.is_file() and (spec.sha256 is None or sha256_file(target) == spec.sha256):
                self._install_preprocessing_receipt(target, receipt)
                return target
            temporary = self.root / f".{ident}.download-{os.getpid()}"
            if temporary.exists():
                temporary.unlink()
            parsed = urlparse(spec.source)
            if parsed.scheme in ("", "file"):
                source = Path(parsed.path if parsed.scheme == "file" else spec.source).expanduser().resolve()
                if not source.is_file():
                    raise FileNotFoundError(source)
                total = source.stat().st_size
                with source.open("rb") as src, temporary.open("xb") as dst:
                    copied = 0
                    for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
                        dst.write(block); copied += len(block)
                        if progress: progress(copied, total)
            elif parsed.scheme in ("http", "https"):
                request = Request(spec.source, headers={"User-Agent": "fastwam-realman-deploy/1"})
                with urlopen(request, timeout=60) as response, temporary.open("xb") as dst:
                    total = int(response.headers.get("Content-Length") or spec.size_bytes or 0)
                    copied = 0
                    for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        dst.write(block); copied += len(block)
                        if progress: progress(copied, total)
            else:
                raise ValueError("checkpoint source must be a local path, file://, http:// or https:// URL")
            if spec.size_bytes is not None and temporary.stat().st_size != spec.size_bytes:
                raise ValueError("downloaded checkpoint size differs from catalog")
            digest = sha256_file(temporary)
            if spec.sha256 is not None and digest != spec.sha256:
                raise ValueError("downloaded checkpoint SHA256 differs from catalog")
            temporary.replace(target)
            self._install_preprocessing_receipt(target, receipt)
            return target
        finally:
            temporary = self.root / f".{ident}.download-{os.getpid()}"
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            with self._lock:
                self._active = None
