"""Audit frozen LeRobot v3 captures and export one shared, lossless-RGB v2.1 view.

The original 78 demonstrations remain immutable. Each contiguous valid segment
becomes a separate training episode; action windows cannot cross capture gaps.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile

import numpy as np

from .contracts import (CAMERAS, PROMPT, SCHEMA, SOURCE_SHA256, confined_path, identity_sha256,
                        inventory, read_json, require, write_json)

STATE_COLUMNS = [0, 1, 2, 3, 4, 5, 6, 19]


def continuous_segments(table):
    """Same causal/freshness boundaries as cosmos.realman_joint, without Torch."""
    grid = np.asarray(table["observation.monotonic_time_s"], dtype=np.float64).reshape(-1)
    n = len(grid)
    require(n >= 2 and np.isfinite(grid).all() and (np.diff(grid) > 0).all(), "Invalid capture clock")
    ft = np.asarray(table["observation.force_timestamp_s"], dtype=np.float64).reshape(n)
    fresh = np.asarray(table["observation.capture_fresh"])
    require(fresh.shape == (n, 4), "Missing four-source freshness flags")
    valid = (fresh == 1).all(1) & (np.asarray(table["action.applied"]).reshape(n) == 1)
    valid &= np.isfinite(ft) & (grid - ft >= -1e-6) & (grid - ft <= .03)
    valid &= np.asarray(table["observation.force_filter_input_count"]).reshape(n) >= 100
    breaks = np.zeros(n, dtype=bool)
    breaks[0] = True
    breaks[1:] = np.abs(np.diff(grid) - 1 / 30) > 2e-5
    breaks |= np.asarray(table["observation.capture_discontinuity"]).reshape(n) != 0
    breaks |= np.asarray(table["observation.capture_gap_s"]).reshape(n) > 2e-5
    for key in ("observation.force_filter_epoch", "action.follow_session_id", "action.leader_anchor", "action.follower_anchor"):
        values = np.asarray(table[key])
        changes = values[1:] != values[:-1]
        breaks[1:] |= changes if changes.ndim == 1 else changes.any(1)
    segments, start = [], None
    for j in range(n):
        if start is not None and (breaks[j] or not valid[j]):
            if j - start >= 2:
                segments.append((start, j))
            start = None
        if valid[j] and start is None:
            start = j
    if start is not None and n - start >= 2:
        segments.append((start, n))
    return segments


def numeric_episode(table, episode, cursor):
    n = episode["length"]
    require(len(table["frame_index"]) == n and table["frame_index"] == list(range(n)), "Missing source rows")
    require(table["index"] == list(range(cursor, cursor + n)), "Wrong global row order")
    require(np.allclose(table["timestamp"], np.arange(n) / 30, atol=2e-6, rtol=0), "Wrong frame timeline")
    state = np.asarray(table["observation.state"], dtype=np.float32)
    action = np.asarray(table["action"], dtype=np.float32)
    force = np.asarray(table["observation.force_filtered"], dtype=np.float32)
    require(state.shape == (n, 20) and action.shape == (n, 8) and force.shape == (n, 6), "Wrong joint8/force dimensions")
    require(all(np.isfinite(a).all() for a in (state, action, force)), "Nonfinite source values")
    require(np.array_equal(state[:, 13:19], force), "Force columns disagree")
    require(all(((x >= 0) & (x <= 1)).all() for x in (state[:, 19], action[:, 7])), "Gripper outside [0,1]")
    ft = np.asarray(table["observation.force_timestamp_s"]).reshape(n)
    require(np.isfinite(ft).all() and (np.diff(ft) >= 0).all(), "Invalid force timeline")
    for key in ("observation.force_filter_timestamp_s", "observation.robot_timestamp_s"):
        require(np.array_equal(ft, np.asarray(table[key]).reshape(n)), "Force/robot callback mismatch")
    for key in ("action.timestamp_s", "observation.leader_timestamp_s", "observation.gripper_timestamp_s",
                "action.gripper_timestamp_s", "observation.camera.arrival_time_s"):
        values = np.asarray(table[key])
        bound = ft[:, None] if values.ndim == 2 else ft
        require(np.isfinite(values).all() and (values <= bound + 1e-6).all(), "Future input: " + key)
    camera_times = np.asarray(table["observation.camera.arrival_time_s"])
    require(camera_times.shape == (n, 2), "Both camera timestamps required")
    # Invalid rows are excluded below, but all retained frames must obey the live wire alignment.
    segments = continuous_segments(table)
    require(bool(segments), "A source demonstration has no usable contiguous segment")
    for lo, hi in segments:
        age = ft[lo:hi, None] - camera_times[lo:hi]
        require((age >= -1e-6).all() and (age <= .050001).all(), "Camera/robot alignment exceeds live contract")
    return {"observation.state": state[:, STATE_COLUMNS], "action": action, "observation.force": force}, segments


def png_frames(root, info, episode, camera):
    import av
    from PIL import Image
    feature = "observation.images." + camera
    key = "videos/" + feature + "/"
    relative = info["video_path"].format(video_key=feature, chunk_index=episode[key + "chunk_index"], file_index=episode[key + "file_index"])
    path = confined_path(root, relative)
    first = round(episode[key + "from_timestamp"] * 30)
    n = episode["length"]
    require(abs(episode[key + "from_timestamp"] * 30 - first) < 1e-3, "Video offset is off the frame grid")
    if key + "to_timestamp" in episode:
        require(abs((episode[key + "to_timestamp"] - episode[key + "from_timestamp"]) * 30 - n) < 1e-3, "Video duration differs from rows")
    result = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        require(float(stream.average_rate) == 30, "Wrong video FPS")
        for index, frame in enumerate(container.decode(video=0)):
            if first <= index < first + n:
                require((frame.width, frame.height) == (640, 480), "Wrong raw RGB size")
                if frame.time is not None:
                    require(abs(frame.time - index / 30) < 1e-4, "Video PTS differs from row grid")
                image = Image.fromarray(frame.to_ndarray(format="rgb24"))
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                result.append({"bytes": buffer.getvalue(), "path": None})
            if index >= first + n:
                break
    require(len(result) == n, f"Missing camera frames: {camera}")
    return result


def numeric_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": values.mean(0).tolist(), "std": values.std(0).tolist(), "min": values.min(0).tolist(),
            "max": values.max(0).tolist(), "q01": np.quantile(values, .01, axis=0).tolist(),
            "q99": np.quantile(values, .99, axis=0).tolist(), "count": [len(values)]}


def jsonl(path, rows):
    import json
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def prepare_dataset(source_root, output_root=None):
    """Audit only if output_root is absent. Export refuses any existing destination."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from realman_force.contract import validate_contract
    source_root = Path(source_root).resolve()
    files = inventory(source_root)
    source_hash = identity_sha256(files)
    require(source_hash == SOURCE_SHA256, "Source inventory differs from frozen 78-episode dataset")
    info = read_json(source_root / "meta/info.json")
    require((info["codebase_version"], info["fps"], info["total_episodes"]) == ("v3.0", 30, 78), "Expected LeRobot v3 / 30 Hz / 78 episodes")
    force = read_json(source_root / "meta/force.json")
    leader = read_json(source_root / "meta/leader.json")
    require(force["schema"] == "realman_leader_force_gripper_v2", "Wrong force schema")
    require(leader.get("training_demonstration") is True and leader.get("mode") == "execute", "Only executed demonstrations accepted")
    episodes = sorted([e for p in sorted((source_root / "meta/episodes").rglob("*.parquet"))
                       for e in pq.read_table(p).to_pylist()], key=lambda e: e["episode_index"])
    require([e["episode_index"] for e in episodes] == list(range(78)), "Missing or duplicate source episodes")
    require(sum(e["length"] for e in episodes) == info["total_frames"], "Source total frame mismatch")
    output, stage = None, None
    if output_root is not None:
        output = Path(output_root).resolve()
        require(not output.exists() and not output.is_relative_to(source_root), "Output must be new and outside source")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=output.name + ".incomplete-", dir=output.parent))
        (stage / "meta").mkdir()
    converted, stats_rows, all_numeric, timing = [], [], {}, []
    source_cursor = target_cursor = 0
    try:
        for episode in episodes:
            ep, n = episode["episode_index"], episode["length"]
            require(episode["tasks"] == [PROMPT], "Unexpected task instruction")
            require((episode["dataset_from_index"], episode["dataset_to_index"]) == (source_cursor, source_cursor + n), "Source episode bounds mismatch")
            validate_contract(force["model_input"], read_json(source_root / "meta/force_sessions" / f"episode_{ep:06d}.json"))
            relative = info["data_path"].format(chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"])
            original = pq.read_table(confined_path(source_root, relative)).to_pydict()
            indices = [j for j, value in enumerate(original["episode_index"]) if value == ep]
            table = {key: [values[j] for j in indices] for key, values in original.items()}
            numeric, segments = numeric_episode(table, episode, source_cursor)
            # Decode both cameras during audit too: metadata alone is not image validation.
            images = {camera: png_frames(source_root, info, episode, camera) for camera in CAMERAS}
            timing.append({"source_episode": ep, "source_frames": n, "retained_frames": sum(hi - lo for lo, hi in segments), "segments": segments})
            for lo, hi in segments:
                idx, length = len(converted), hi - lo
                converted.append(dict(episode_index=idx, source_episode=ep, start=lo, stop=hi, length=length))
                arrays = {key: value[lo:hi] for key, value in numeric.items()}
                for key, values in arrays.items():
                    all_numeric.setdefault(key, []).append(values)
                stats_rows.append({"episode_index": idx, "stats": {key: numeric_stats(value) for key, value in arrays.items()}})
                if stage is not None:
                    columns = {key: pa.array(values.tolist(), type=pa.list_(pa.float32(), values.shape[1])) for key, values in arrays.items()}
                    columns.update({"timestamp": pa.array(np.arange(length, dtype=np.float32) / 30, type=pa.float32()),
                                    "frame_index": pa.array(range(length), type=pa.int64()),
                                    "episode_index": pa.array([idx] * length, type=pa.int64()),
                                    "index": pa.array(range(target_cursor, target_cursor + length), type=pa.int64()),
                                    "task_index": pa.array([0] * length, type=pa.int64())})
                    for camera in CAMERAS:
                        columns["observation.images." + camera] = pa.array(images[camera][lo:hi], type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
                    destination = stage / f"data/chunk-{idx // 1000:03d}/episode_{idx:06d}.parquet"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    # LeRobot's pinned loader relies on HF schema metadata to
                    # decode embedded image structs into PIL images.
                    hf_features = {key: {"_type": "Sequence", "length": values.shape[1],
                                         "feature": {"_type": "Value", "dtype": "float32"}}
                                   for key, values in arrays.items()}
                    hf_features.update({key: {"_type": "Value", "dtype": "float32" if key == "timestamp" else "int64"}
                                        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")})
                    hf_features.update({"observation.images." + camera: {"_type": "Image"} for camera in CAMERAS})
                    arrow_table = pa.table(columns).replace_schema_metadata({
                        b"huggingface": json.dumps({"info": {"features": hf_features}}).encode()})
                    pq.write_table(arrow_table, destination)
                target_cursor += length
            source_cursor += n
        summary = dict(schema=SCHEMA, source_episodes=78, source_inventory_sha256=source_hash, source_frames=source_cursor,
                       retained_frames=target_cursor, discarded_frames=source_cursor - target_cursor, fps=30,
                       action_space="joint8", state_dim=8, action_dim=8, action_units=["rad"] * 7 + ["openness_0_1"],
                       camera_names=CAMERAS, image_size=[640, 480], prompt=PROMPT, force_preprocessing=force["model_input"],
                       train_episode_ids=list(range(len(converted))), validation_episode_ids=[], episodes=converted,
                       timing=timing, image_encoding="lossless PNG of original decoded RGB; no resize",
                       state_columns=STATE_COLUMNS, force_policy="recorded calibrated causal filtered force; no refilter",
                       evaluation_scope="training-fit replay; no held-out or robot-success claim")
        # Reject changes made while the potentially long conversion was in progress.
        require(inventory(source_root) == files, "Source changed during conversion")
        if stage is not None:
            features = {key: {"dtype": "float32", "shape": [8 if key != "observation.force" else 6], "names": None}
                        for key in all_numeric}
            features.update({"observation.images." + camera: {"dtype": "image", "shape": [480, 640, 3], "names": ["height", "width", "channels"]} for camera in CAMERAS})
            features.update({key: {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1], "names": None}
                             for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")})
            write_json(stage / "meta/info.json", dict(codebase_version="v2.1", robot_type="rm71d", fps=30,
                       total_episodes=len(converted), total_frames=target_cursor, total_tasks=1, total_videos=0,
                       total_chunks=(len(converted) + 999) // 1000, chunks_size=1000, splits={"train": f"0:{len(converted)}"},
                       data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet", video_path=None, features=features))
            write_json(stage / "meta/stats.json", {key: numeric_stats(np.concatenate(values)) for key, values in all_numeric.items()})
            jsonl(stage / "meta/tasks.jsonl", [{"task_index": 0, "task": PROMPT}])
            jsonl(stage / "meta/episodes.jsonl", [{"episode_index": e["episode_index"], "tasks": [PROMPT], "length": e["length"]} for e in converted])
            jsonl(stage / "meta/episodes_stats.jsonl", stats_rows)
            summary["files"] = inventory(stage)
            summary["identity_sha256"] = identity_sha256(summary)
            write_json(stage / "manifest.json", summary)
            stage.rename(output)
        return summary
    except Exception:
        # Keep an explicitly incomplete directory for diagnosis; never delete source data.
        if stage is not None:
            write_json(stage / "INCOMPLETE.json", {"status": "failed", "usable_for_training": False})
        raise
