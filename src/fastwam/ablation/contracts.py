"""Data and deployment identities, without importing model frameworks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "rm71d78-ablation-v1"
PROMPT = "拔下插头并插入另一插座"
SOURCE_SHA256 = "b24b701239f54ad38d16c819b9ea9f6fad5f9374a683a84a68d2cd7ef27067a8"
CAMERAS = ["global", "wrist"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def identity_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(root):
    root = Path(root).resolve()
    files = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Symlinks are not allowed in bound assets")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = file_sha256(path)
    require(bool(files), f"Empty asset directory: {root}")
    return files


def path_sha256(path):
    path = Path(path)
    return identity_sha256(inventory(path)) if path.is_dir() else file_sha256(path)


def confined_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), f"Path escapes asset root: {relative}")
    return path


def load_dataset_manifest(root, verify_files=True):
    root = Path(root).resolve()
    data = read_json(root / "manifest.json")
    require(data.get("schema") == SCHEMA, "Unexpected dataset manifest schema")
    require(data.get("identity_sha256") == identity_sha256({k: v for k, v in data.items() if k != "identity_sha256"}), "Dataset manifest identity mismatch")
    for key, expected in {"source_episodes": 78, "fps": 30, "action_space": "joint8", "state_dim": 8,
                          "action_dim": 8, "camera_names": CAMERAS, "image_size": [640, 480], "prompt": PROMPT}.items():
        require(data.get(key) == expected, f"Dataset contract mismatch: {key}")
    require(data.get("source_inventory_sha256") == SOURCE_SHA256, "Dataset is not the frozen RM71D78 source")
    episodes = data["episodes"]
    ids = list(range(len(episodes)))
    require([e["episode_index"] for e in episodes] == ids and bool(ids), "Invalid converted episode indices")
    require({e["source_episode"] for e in episodes} == set(range(78)), "All 78 source demonstrations must contribute")
    require(all(e["stop"] > e["start"] >= 0 and e["length"] == e["stop"] - e["start"] for e in episodes), "Invalid segment bounds")
    require(data.get("train_episode_ids") == ids and data.get("validation_episode_ids") == [], "This profile uses all demonstrations for training; replay is training-fit only")
    from realman_force.contract import validate_contract_identity
    validate_contract_identity(data["force_preprocessing"])
    files = data["files"]
    require("meta/info.json" in files and "meta/stats.json" in files, "Incomplete dataset inventory")
    if verify_files:
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
        require(actual == set(files), "Dataset inventory changed")
        for relative, digest in files.items():
            require(file_sha256(confined_path(root, relative)) == digest, f"Dataset file changed: {relative}")
    info = read_json(root / "meta/info.json")
    require(info["codebase_version"] == "v2.1" and info["total_episodes"] == len(episodes)
            and info["total_frames"] == sum(e["length"] for e in episodes), "Export metadata disagrees with manifest")
    return data
