#!/usr/bin/env python3
"""Strict, lossless two-camera ForceVLA conversion for stock FastWAM LIBERO packing.

No resampling, window selection, rotation conversion or image recompression is
performed. Runtime preprocessing/statistics remain the official FastWAM code.

Adapted from realman-fastwam/scripts/dualwm/prepare_forcevla.py at commit
5cb2eb7148ac3fa215f5790211019ede51e62d7d (original SHA256 below). The pinned
wipe50 inputs observation.image and observation.wrist_image become cam_main
and cam_wrist in that order, for the stock LIBERO horizontal two-view packing.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


SOURCE_REPOSITORY = "qiaojunyu/ForceVLA-real-data"
SOURCE_REVISION = "6070cbef017e95d695fc804e6fb28cc8baabb1cd"
SOURCE_SUBSET = "flexiv_wipe_board_inputForce"
SOURCE_EPISODES = 50
SOURCE_FRAMES = 34808
CAMERA_OUTPUT_ORDER = ("cam_main", "cam_wrist")
PACKING_POSITIONS = ("left", "right")
CAMERA_MAPPING = {"observation.image": "cam_main", "observation.wrist_image": "cam_wrist"}
ORIGINAL_CONVERTER_SHA256 = "9395a9eebfe5d100264cbfed6e27be9637c536d69e604f3089fd0fab241b15f4"
PINNED_INFO_SHA256 = "6fc2ee8733c13cd1a4ddbabb3f7a09dc4d44593eb67a09a6b57fb45cf69fe757"
PINNED_PUBLIC_CAMERAS = ("observation.image", "observation.wrist_image")
PROFILE = "forcevla-libero2cam-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _scalar_column(table, name, dtype=None):
    values = np.asarray(table[name].to_pylist(), dtype=dtype)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(f"{name} must be scalar per frame")
    return values


def _feature(dim, names=None, dtype="float32"):
    return {"dtype": dtype, "shape": [dim], "names": names}


def _numeric_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"min": values.min(0).tolist(), "max": values.max(0).tolist(),
            "mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
            "count": [len(values)]}


def _image_column(table, name, source_root, expected_shape, stream_digest):
    # Preserve embedded image bytes. Paths are materialized without re-encoding
    # so the new dataset never depends on an external source directory.
    cells = []
    for scalar in table[name]:
        cell = scalar.as_py()
        if isinstance(cell, dict) and cell.get("bytes") is not None:
            payload = cell["bytes"]
        elif isinstance(cell, (bytes, bytearray)):
            payload = bytes(cell)
        elif isinstance(cell, dict) and cell.get("path"):
            path = (source_root / cell["path"]).resolve()
            if not path.is_relative_to(source_root.resolve()):
                raise ValueError("Image path escapes source root")
            payload = path.read_bytes()
        else:
            raise ValueError(f"{name} contains an absent or unsupported image")
        if not isinstance(payload, bytes) or not payload:
            raise ValueError(f"{name} contains an absent or unsupported image")
        # Decode only for validation. The original payload is written unchanged.
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                if image.mode != "RGB" or [image.height, image.width, 3] != expected_shape:
                    raise ValueError(f"{name} image is not RGB with metadata shape {expected_shape}")
        except (OSError, SyntaxError) as error:
            raise ValueError(f"{name} contains an invalid encoded image") from error
        stream_digest.update(len(payload).to_bytes(8, "big"))
        stream_digest.update(payload)
        cells.append({"bytes": payload, "path": None})
    return pa.array(cells, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]))


def _manifest_entries(manifest, source_root, revision):
    if manifest is None:
        return None
    value = json.loads(Path(manifest).read_text())
    if value.get("repo_id") != SOURCE_REPOSITORY or value.get("revision") != revision:
        raise ValueError("Download manifest repository/revision does not match")
    if value.get("subset") != source_root.name:
        raise ValueError("Download manifest subset does not match the input directory")
    entries = {}
    prefix = "data_lerobot/" + source_root.name + "/"
    for entry in value["files"]:
        if not isinstance(entry, dict) or not entry.get("path", "").startswith(prefix):
            raise ValueError("Expected a payload-integrity manifest with per-file metadata")
        entries[entry["path"][len(prefix):]] = entry
    return entries


def _camera_mapping(cameras, source_info):
    """Validate before dict construction so repeated --camera inputs cannot vanish."""
    if cameras is None:
        cameras = CAMERA_MAPPING
    pairs = list(cameras.items()) if isinstance(cameras, dict) else list(cameras)
    if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in pairs):
        raise ValueError("Camera mappings must be source_key=output_name pairs")
    sources, outputs = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    if len(pairs) != 2 or len(set(sources)) != 2 or len(set(outputs)) != 2:
        raise ValueError("LIBERO two-camera conversion requires exactly two distinct source and output cameras; duplicate mappings are forbidden")
    if set(outputs) != set(CAMERA_OUTPUT_ORDER):
        raise ValueError(f"LIBERO two-camera output cameras must be {CAMERA_OUTPUT_ORDER}")
    if dict(pairs) != CAMERA_MAPPING:
        raise ValueError(f"LIBERO two-camera profile requires the verified exact source mapping {CAMERA_MAPPING}")
    features = source_info["features"]
    available = sorted(key for key, feature in features.items() if feature.get("dtype") == "image")
    for key in sources:
        feature = features.get(key, {})
        if feature.get("dtype") != "image":
            raise ValueError(f"Missing real source camera {key!r}; available image features: {available}. A view cannot be synthesized or duplicated")
        shape = feature.get("shape")
        if not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3 or any(not isinstance(dim, int) or dim < 1 for dim in shape):
            raise ValueError(f"Camera {key!r} must have a positive HWC RGB metadata shape")
    by_output = {output: source for source, output in pairs}
    return {by_output[output]: output for output in CAMERA_OUTPUT_ORDER}


def _numeric_column(table, name, width, declared_dtype):
    column = table[name]
    arrow_type = column.type
    if not (pa.types.is_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type)):
        raise ValueError(f"{name} must be a numeric list column")
    expected_type = {"float32": pa.float32(), "float64": pa.float64()}.get(declared_dtype)
    if expected_type is None or arrow_type.value_type != expected_type:
        raise ValueError(f"{name} must match its declared float32/float64 dtype")
    values = np.asarray(column.to_pylist(), dtype=declared_dtype)
    if values.shape != (len(table), width):
        raise ValueError(f"Invalid {name} shape; expected {width} numeric values per frame")
    return values, expected_type


def convert_dataset(source_root, output_root, *, cameras=None, source_revision=SOURCE_REVISION,
                    manifest=None, test_fixture=False):
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_root}")
    if output_root.is_relative_to(source_root) or source_root.is_relative_to(output_root):
        raise ValueError("Input and output dataset directories must be separate")
    info_path = source_root / "meta/info.json"
    source_info = json.loads(info_path.read_text())
    cameras = _camera_mapping(cameras, source_info)
    marked_fixture = source_info.get("forcevla_test_fixture") is True
    if test_fixture != marked_fixture:
        raise ValueError("TEST-only conversion requires --test-fixture and forcevla_test_fixture=true metadata together")
    if source_revision != SOURCE_REVISION:
        raise ValueError("LIBERO two-camera profile requires the pinned ForceVLA source revision")
    if not test_fixture:
        if (source_root.name != SOURCE_SUBSET or int(source_info["total_episodes"]) != SOURCE_EPISODES
                or int(source_info["total_frames"]) != SOURCE_FRAMES):
            raise ValueError("Production conversion requires the complete pinned wipe50 subset (all 50 episodes, 34808 original frames)")
        if sha256(info_path) != PINNED_INFO_SHA256:
            raise ValueError("Production info.json must match the pinned public ForceVLA metadata SHA256")
        if manifest is None:
            raise ValueError("Production conversion requires a pinned payload-integrity --manifest")
        for key in cameras:
            if source_info["features"][key]["shape"] != [480, 640, 3]:
                raise ValueError("Production source cameras must match pinned 480x640 RGB metadata")
    if source_info.get("codebase_version") != "v2.1" or float(source_info["fps"]) != 30:
        raise ValueError("This converter requires ForceVLA LeRobot v2.1 at 30 Hz")
    expected = {"observation.state": ([13], ["current_eef_pose", "gripper_width", "f_ext_base_frame"]),
                "action": ([7], ["target_eef_pose", "target_gripper_width"])}
    for name, (shape, names) in expected.items():
        feature = source_info["features"].get(name, {})
        if feature.get("shape") != shape or feature.get("names") != names:
            raise ValueError(f"Unverified ForceVLA field layout: {name}")
        if feature.get("dtype") not in ("float32", "float64") or (not test_fixture and feature.get("dtype") != "float64"):
            raise ValueError(f"Unverified ForceVLA numeric dtype: {name}")
    tasks_rows = _rows(source_root / "meta/tasks.jsonl")
    tasks = {int(row["task_index"]): row["task"] for row in tasks_rows}
    if len(tasks) != len(tasks_rows) or not tasks or any(not isinstance(v, str) or not v.strip() for v in tasks.values()):
        raise ValueError("Source tasks must have unique indices and nonempty instructions")
    files = sorted(source_root.glob("data/chunk-*/episode_*.parquet"))
    if len(files) != int(source_info["total_episodes"]) or not files:
        raise ValueError("Conversion requires every published episode; partial downloads are not accepted")
    if [int(path.stem.removeprefix("episode_")) for path in files] != list(range(len(files))):
        raise ValueError("Source episode indices must start at zero and be contiguous; refusing to reindex rows")
    entries = _manifest_entries(manifest, source_root, source_revision)
    input_files = []
    for path in [*sorted((source_root / "meta").glob("*")), *files]:
        if not path.is_file():
            continue
        relative = path.relative_to(source_root).as_posix()
        digest = sha256(path)
        if entries is not None:
            entry = entries.get(relative)
            if entry is None or path.stat().st_size != entry["size_bytes"]:
                raise ValueError(f"Source file absent from manifest or size mismatch: {relative}")
            if entry.get("sha256"):
                if digest != entry["sha256"]:
                    raise ValueError(f"Source payload SHA256 mismatch: {relative}")
            else:
                payload = path.read_bytes()
                git_hash = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
                if git_hash != entry.get("git_blob_sha1"):
                    raise ValueError(f"Source metadata Git blob hash mismatch: {relative}")
        input_files.append({"path": relative, "sha256": digest, "size_bytes": path.stat().st_size})
    if entries is not None and set(entries) != {row["path"] for row in input_files}:
        raise ValueError("Source file inventory differs from the manifest")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_root.name + ".tmp-", dir=output_root.parent))
    try:
        (temporary / "meta").mkdir()
        state_dtype = source_info["features"]["observation.state"]["dtype"]
        action_dtype = source_info["features"]["action"]["dtype"]
        features = {
            "observation.state": _feature(7, ["current_eef_pose", "gripper_width"], state_dtype),
            "observation.history_state": _feature(7, ["current_eef_pose", "gripper_width"], state_dtype),
            "observation.wrench": _feature(6, ["fx", "fy", "fz", "tx", "ty", "tz"], state_dtype),
            "action": _feature(7, ["target_eef_pose", "target_gripper_width"], action_dtype),
            **{key: {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1], "names": None}
               for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")},
        }
        for source_key, output_key in cameras.items():
            feature = source_info["features"].get(source_key)
            if feature is None or feature.get("dtype") != "image":
                raise ValueError(f"Expected an embedded ForceVLA image feature: {source_key}")
            features["observation.images." + output_key] = dict(feature)
        episode_rows, episode_stats, output_files = [], [], []
        camera_digests = {key: hashlib.sha256() for key in cameras}
        total_frames = 0
        for output_episode, path in enumerate(files):
            table = pq.read_table(path)
            n = len(table)
            if n < 2:
                raise ValueError("Each episode needs at least two frames for official variance statistics")
            required = {"observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index", *cameras}
            if not required.issubset(table.column_names):
                raise ValueError(f"Missing synchronized source columns in {path.name}: {sorted(required - set(table.column_names))}")
            for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
                expected_type = pa.float32() if key == "timestamp" else pa.int64()
                if table[key].type != expected_type:
                    raise ValueError(f"{key} must retain its pinned source scalar dtype {expected_type}")
            state, state_arrow_type = _numeric_column(table, "observation.state", 13, state_dtype)
            action, action_arrow_type = _numeric_column(table, "action", 7, action_dtype)
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError("Nonfinite ForceVLA values need an explicit missing-label policy; refusing to impute")
            timestamp = _scalar_column(table, "timestamp")
            if not np.isfinite(timestamp).all() or not np.all(np.diff(timestamp) > 0):
                raise ValueError("Episode timestamps must be finite and strictly increasing")
            if not np.allclose(np.diff(timestamp), 1 / 30, atol=1e-4, rtol=1e-3):
                raise ValueError("Expected the original contiguous 30 Hz timeline; no implicit resampling")
            if "frame_index" in table.column_names and not np.array_equal(_scalar_column(table, "frame_index"), np.arange(n)):
                raise ValueError("Episode frame indices must start at zero and be contiguous")
            source_episode = int(path.stem.removeprefix("episode_"))
            if "episode_index" in table.column_names and not np.all(_scalar_column(table, "episode_index") == source_episode):
                raise ValueError("Parquet filename and episode_index disagree")
            if not np.array_equal(_scalar_column(table, "index"), np.arange(total_frames, total_frames + n)):
                raise ValueError("Source global index must be contiguous in original episode/frame order")
            task_key = "task_index" if "task_index" in table.column_names else "task.index"
            if task_key in table.column_names:
                task_index = _scalar_column(table, task_key, np.int64)
            elif len(tasks) == 1:
                task_index = np.full(n, next(iter(tasks)), dtype=np.int64)
            else:
                raise ValueError("Ambiguous source task assignment")
            if len(np.unique(task_index)) != 1 or int(task_index[0]) not in tasks:
                raise ValueError("Each episode must reference exactly one known task")
            numeric = {"observation.state": state[:, :7], "observation.history_state": state[:, :7],
                       "observation.wrench": state[:, 7:], "action": action}
            columns = {key: pa.array(value.tolist(), type=pa.list_(action_arrow_type if key == "action" else state_arrow_type, value.shape[1]))
                       for key, value in numeric.items()}
            # Keep original timestamp values/types and all encoded image bytes.
            columns.update({key: table[key] for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")})
            for source_key, output_key in cameras.items():
                columns["observation.images." + output_key] = _image_column(table, source_key, source_root,
                    source_info["features"][source_key]["shape"], camera_digests[source_key])
            target = temporary / f"data/chunk-{output_episode // 1000:03d}/episode_{output_episode:06d}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            # Do not preserve obsolete HuggingFace column metadata after rename.
            pq.write_table(pa.table(columns), target, compression="snappy")
            episode_rows.append({"episode_index": output_episode, "tasks": [tasks[int(task_index[0])]], "length": n})
            episode_stats.append({"episode_index": output_episode,
                                  "stats": {key: _numeric_stats(value) for key, value in numeric.items()}})
            output_files.append({"path": target.relative_to(temporary).as_posix(), "sha256": sha256(target),
                                 "frames": n, "source_path": path.relative_to(source_root).as_posix(),
                                 "source_episode_index": source_episode, "episode_index": output_episode})
            total_frames += n
        if total_frames != int(source_info["total_frames"]):
            raise ValueError("Converted frame count differs from source metadata")
        if len({digest.hexdigest() for digest in camera_digests.values()}) != 2:
            raise ValueError("Two-camera source contains duplicate complete camera payload streams; refusing fake duplicated views")
        output_info = {
            "codebase_version": "v2.1", "robot_type": source_info["robot_type"], "fps": source_info["fps"],
            "total_episodes": len(files), "total_frames": total_frames, "total_tasks": len(tasks),
            "total_videos": 0, "total_chunks": (len(files) + 999) // 1000, "chunks_size": 1000,
            "splits": {"train": f"0:{len(files)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": None, "features": features,
            "forcevla": {"schema": PROFILE, "state_action_mapping": "identity_pose_gripper",
                       "action_provenance": "recorded_target_pose", "action_alignment_verified": False,
                       "pose_rotation": "source_3d_rotation_unverified", "position_units": "m",
                       "wrench_frame": "base", "wrench_units": ["N", "N", "N", "Nm", "Nm", "Nm"],
                       "wrench_provenance": "robot_estimated_external_wrench", "resampled": False,
                       "history_state_alias": "observation.state", "normalization": "official_runtime"},
        }
        _json(temporary / "meta/info.json", output_info)
        _jsonl(temporary / "meta/tasks.jsonl", tasks_rows)
        _jsonl(temporary / "meta/episodes.jsonl", episode_rows)
        _jsonl(temporary / "meta/episodes_stats.jsonl", episode_stats)
        metadata_files = [{"path": path.relative_to(temporary).as_posix(), "sha256": sha256(path),
                           "size_bytes": path.stat().st_size} for path in sorted((temporary / "meta").glob("*"))]
        conversion = {"schema": "forcevla-fastwam-conversion-v2", "profile": PROFILE,
                      "test_fixture": test_fixture, "usage": "TEST-only synthetic fixture" if test_fixture else "production",
                      "source_repo": SOURCE_REPOSITORY,
                      "source_revision": source_revision, "source_revision_provenance": "verified_manifest" if entries else "declared_only",
                      "source_subset": source_root.name, "input_files": input_files, "output_files": output_files,
                      "metadata_files": metadata_files,
                      "source_manifest_sha256": sha256(manifest) if manifest else None,
                      "cameras": cameras, "episodes": len(files), "frames": total_frames,
                      "camera_output_order": list(CAMERA_OUTPUT_ORDER),
                      "camera_packing": dict(zip(CAMERA_OUTPUT_ORDER, PACKING_POSITIONS)),
                      "camera_identity_provenance": "verified_exact_pinned_source_feature_mapping",
                      "camera_payload_sha256": {cameras[key]: value.hexdigest() for key, value in camera_digests.items()},
                      "numeric_processing": "original_float_dtype_and_values; state13_split_into_state7_wrench6",
                      "converter_provenance": {"source_path": "realman-fastwam/scripts/dualwm/prepare_forcevla.py",
                          "source_git_commit": "5cb2eb7148ac3fa215f5790211019ede51e62d7d", "source_sha256": ORIGINAL_CONVERTER_SHA256},
                      "pinned_public_evidence": {"revision": SOURCE_REVISION, "info_sha256": PINNED_INFO_SHA256,
                          "source_image_features": list(PINNED_PUBLIC_CAMERAS)},
                      "image_processing": "original_encoded_bytes", "sampling": "all_original_frames",
                      "training_stats": "computed later by official BaseLerobotDataset.get_dataset_stats"}
        _json(temporary / "conversion_manifest.json", conversion)
        temporary.rename(output_root)
        return conversion
    except BaseException:
        shutil.rmtree(temporary)
        raise


def validate_converted_libero2cam(root):
    """Verify a production conversion without importing any model/GPU code.

    All output parquet and metadata payload hashes are checked. Synthetic test
    fixtures are deliberately rejected even if their metadata claims full50.
    The returned conversion manifest is safe for launcher preflight checks.
    """
    root = Path(root).resolve()
    conversion = json.loads((root / "conversion_manifest.json").read_text())
    required = {"schema": "forcevla-fastwam-conversion-v2", "profile": PROFILE,
                "test_fixture": False, "usage": "production", "source_repo": SOURCE_REPOSITORY,
                "source_revision": SOURCE_REVISION, "source_revision_provenance": "verified_manifest",
                "source_subset": SOURCE_SUBSET, "episodes": SOURCE_EPISODES, "frames": SOURCE_FRAMES,
                "cameras": CAMERA_MAPPING, "camera_output_order": list(CAMERA_OUTPUT_ORDER),
                "camera_packing": dict(zip(CAMERA_OUTPUT_ORDER, PACKING_POSITIONS)),
                "image_processing": "original_encoded_bytes", "sampling": "all_original_frames"}
    for key, expected in required.items():
        if conversion.get(key) != expected:
            raise ValueError(f"Production LIBERO conversion manifest mismatch: {key}")
    source_info_entries = [entry for entry in conversion.get("input_files", []) if entry.get("path") == "meta/info.json"]
    if len(source_info_entries) != 1 or source_info_entries[0].get("sha256") != PINNED_INFO_SHA256:
        raise ValueError("Converted data lacks the pinned public source metadata hash")
    source_manifest_hash = conversion.get("source_manifest_sha256")
    if not isinstance(source_manifest_hash, str) or len(source_manifest_hash) != 64:
        raise ValueError("Converted data lacks its verified source download manifest hash")
    camera_hashes = conversion.get("camera_payload_sha256", {})
    if set(camera_hashes) != set(CAMERA_OUTPUT_ORDER) or len(set(camera_hashes.values())) != 2:
        raise ValueError("Converted data lacks two distinct original camera payload streams")
    output_files = conversion.get("output_files", [])
    expected_parquet = [f"data/chunk-000/episode_{episode:06d}.parquet" for episode in range(SOURCE_EPISODES)]
    if ([entry.get("path") for entry in output_files] != expected_parquet
            or [entry.get("episode_index") for entry in output_files] != list(range(SOURCE_EPISODES))
            or [entry.get("source_episode_index") for entry in output_files] != list(range(SOURCE_EPISODES))
            or sum(entry.get("frames", 0) for entry in output_files) != SOURCE_FRAMES):
        raise ValueError("Converted output manifest must preserve all 50 original episodes and 34808 rows")
    actual_parquet = sorted(path.relative_to(root).as_posix() for path in root.glob("data/chunk-*/episode_*.parquet"))
    if actual_parquet != expected_parquet:
        raise ValueError("Converted parquet inventory differs from the full50 manifest")
    metadata_files = conversion.get("metadata_files", [])
    expected_meta = ["meta/episodes.jsonl", "meta/episodes_stats.jsonl", "meta/info.json", "meta/tasks.jsonl"]
    if [entry.get("path") for entry in metadata_files] != expected_meta:
        raise ValueError("Converted metadata inventory differs from the conversion manifest")
    for entry in [*metadata_files, *output_files]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Converted output file is missing or escapes dataset root: {entry['path']}")
        if "size_bytes" in entry and path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Converted output size mismatch: {entry['path']}")
        if sha256(path) != entry.get("sha256"):
            raise ValueError(f"Converted output SHA256 mismatch: {entry['path']}")
    info = json.loads((root / "meta/info.json").read_text())
    if (info.get("codebase_version") != "v2.1" or info.get("fps") != 30
            or info.get("total_episodes") != SOURCE_EPISODES or info.get("total_frames") != SOURCE_FRAMES
            or info.get("splits") != {"train": "0:50"} or info.get("forcevla", {}).get("schema") != PROFILE):
        raise ValueError("Converted production metadata is inconsistent with pinned wipe50")
    features = info["features"]
    for key, width in (("observation.state", 7), ("observation.history_state", 7), ("observation.wrench", 6), ("action", 7)):
        if features.get(key, {}).get("dtype") != "float64" or features.get(key, {}).get("shape") != [width]:
            raise ValueError(f"Converted production numeric metadata is invalid: {key}")
    for name in CAMERA_OUTPUT_ORDER:
        feature = features.get("observation.images." + name, {})
        if feature.get("dtype") != "image" or feature.get("shape") != [480, 640, 3]:
            raise ValueError(f"Converted production camera metadata is invalid: {name}")
    episodes = _rows(root / "meta/episodes.jsonl")
    if (len(episodes) != SOURCE_EPISODES
            or [episode.get("episode_index") for episode in episodes] != list(range(SOURCE_EPISODES))
            or [episode.get("length") for episode in episodes] != [entry["frames"] for entry in output_files]):
        raise ValueError("Converted episode metadata does not match the full50 output manifest")
    return conversion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--profile", choices=["libero2cam"], default="libero2cam")
    parser.add_argument("--camera", action="append",
                        help="Verified mapping: observation.image=cam_main, observation.wrist_image=cam_wrist (default); LIBERO left/right order")
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--manifest", help="Required in production: pinned payload-integrity manifest; verify every source file")
    parser.add_argument("--test-fixture", action="store_true", help="TEST-only: allow small synthetic inputs explicitly marked in metadata")
    args = parser.parse_args()
    cameras = [value.split("=", 1) for value in args.camera] if args.camera else None
    result = convert_dataset(args.source_root, args.output_root, cameras=cameras,
                             source_revision=args.source_revision, manifest=args.manifest, test_fixture=args.test_fixture)
    print(json.dumps({key: result[key] for key in ("episodes", "frames", "source_revision_provenance")}))


if __name__ == "__main__":
    main()
