"""Explicit official asset staging, with immutable revisions and byte verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fastwam.ablation.contracts import write_json


def verify_file(path, record):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != record["size"]:
        raise ValueError(f"Official asset size differs: {path}")
    sha = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {record['size']}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
            git_blob.update(block)
    if "sha256" in record and sha.hexdigest() != record["sha256"]:
        raise ValueError(f"Official asset SHA256 differs: {path}")
    if "git_blob_sha1" in record and git_blob.hexdigest() != record["git_blob_sha1"]:
        raise ValueError(f"Official Git blob differs: {path}")
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--reuse-root", action="append", type=Path, default=[])
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    asset_root.mkdir(parents=True, exist_ok=True)
    lock = json.loads((ROOT / "configs/ablation/fastwam-assets.lock.json").read_text())
    receipts = []
    for record in lock["files"]:
        destination = asset_root / record["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = "existing_verified"
        if not destination.exists():
            reused = None
            for directory in args.reuse_root:
                candidates = [directory / record["destination"], directory / record["filename"],
                              directory / Path(record["filename"]).name,
                              directory / "fastwam" / Path(record["filename"]).name]
                for candidate in candidates:
                    if not candidate.is_file() or candidate.stat().st_size != record["size"]:
                        continue
                    try:
                        verify_file(candidate, record)
                    except ValueError:
                        continue
                    temporary = destination.with_name(destination.name + ".copying-" + uuid.uuid4().hex)
                    # Copy rather than link: future shared-directory changes cannot
                    # mutate this experiment's verified initialization files.
                    shutil.copyfile(candidate, temporary)
                    verify_file(temporary, record)
                    os.replace(temporary, destination)
                    reused = str(candidate.resolve())
                    break
                if reused is not None:
                    break
            if reused is not None:
                source = reused
            else:
                print(f"Download official {record['provider']} {record['repo_id']}@{record['revision']}:{record['filename']}", flush=True)
                local_dir = asset_root / record["repo_id"]
                if record["provider"] == "huggingface":
                    from huggingface_hub import hf_hub_download
                    result = hf_hub_download(record["repo_id"], record["filename"], revision=record["revision"],
                                             local_dir=local_dir, token=False)
                else:
                    from modelscope.hub.file_download import model_file_download
                    result = model_file_download(record["repo_id"], record["filename"], revision=record["revision"],
                                                 local_dir=str(local_dir))
                if Path(result).resolve() != destination.resolve():
                    raise ValueError(f"Unexpected asset destination: {result}")
                source = record["provider"] + ":" + record["repo_id"] + "@" + record["revision"]
        digest = verify_file(destination, record)
        receipts.append({**record, "local_path": str(destination), "verified_sha256": digest, "source": source})
        write_json(asset_root / "fastwam-assets-receipt.json", {
            "schema": "fastwam-official-assets-receipt-v1", "complete": len(receipts) == len(lock["files"]),
            "files": receipts, "training_started": False,
        })
        print(f"Verified {destination}: {digest}", flush=True)


if __name__ == "__main__":
    main()
