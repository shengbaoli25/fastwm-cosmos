"""Fixed training-set fit protocol in the original ForceVLA coordinates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PROTOCOL = "official-fastwam-forcevla-five-episode-fit-v1"
EPISODES = (0, 12, 24, 36, 49)
AXES = ("x", "y", "z", "rotation_0", "rotation_1", "rotation_2", "gripper")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def source_identity(root):
    root = Path(root).resolve()
    files = sorted([*root.glob("src/fastwam/**/*.py"), *root.glob("scripts/baseline/*.py"),
                    *root.glob("scripts/forcevla/*.py"),
                    root / "scripts/train.py", root / "scripts/accelerate_configs/accelerate_zero1_ds.yaml",
                    root / "scripts/ds_configs/ds_zero1_config.json"])
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in files}


def select_five_windows(starts, ends, *, horizon=32, history=32, mode="middle"):
    """Freeze episode identities before inspecting model output.

    Middle mode matches the current dual-WM complete-window coordinates.
    Sequence mode produces non-overlapping prediction chunks for each chosen
    episode. Uncovered beginning/end frames are disclosed, never imputed.
    """
    if len(starts) != 50 or len(ends) != 50:
        raise ValueError("This comparison requires all 50 original wipe episodes")
    if mode not in {"middle", "sequence"} or min(horizon, history) < 1:
        raise ValueError("Invalid selection settings")
    windows = []
    for episode in EPISODES:
        start, end = int(starts[episode]), int(ends[episode])
        first, last = start + history - 1, end - horizon - 1
        if first > last:
            raise ValueError(f"Episode {episode} lacks a complete observation/future window")
        anchors = [(first + last) // 2] if mode == "middle" else range(first, last + 1, horizon)
        for anchor in anchors:
            frame = anchor - start
            windows.append({"key": f"episode_{episode:06d}_frame_{frame:06d}",
                            "episode_ordinal": episode, "index": anchor, "frame_index": frame,
                            "episode_start_index": start, "episode_end_index": end})
    return windows


def action_metrics(prediction, target, is_pad):
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or target.ndim != 2 or target.shape[1] != 7:
        raise ValueError("Action arrays must have equal [K,7] shapes")
    valid = ~np.asarray(is_pad, dtype=bool)
    if valid.shape != target.shape[:1] or not valid.any():
        raise ValueError("Invalid action padding")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite action prediction/GT")
    error = prediction[valid] - target[valid]
    mae = np.abs(error).mean(0)
    return {"valid_steps": int(valid.sum()), "mae_source_units_per_axis": mae.tolist(),
            "xyz_mean_axis_mae_mm": float(mae[:3].mean() * 1000),
            "xyz_l2_mm": float(np.linalg.norm(error[:, :3], axis=1).mean() * 1000),
            "rotation_mae_source_units_per_axis": mae[3:6].tolist(),
            "rotation_convention": "source coordinates unverified; per-axis only, no wrapping or SO(3) claim",
            "gripper_mae_source_units": float(mae[6]),
            "prediction_temporal_std_per_axis": prediction[valid].std(0).tolist(),
            "target_temporal_std_per_axis": target[valid].std(0).tolist()}


def video_metrics(prediction, target):
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or target.ndim != 4 or target.shape[0] != 3 or target.shape[1] < 2:
        raise ValueError("Video arrays must have equal [3,T,H,W] shapes")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite video values")
    error = (prediction[:, 1:] - target[:, 1:]) / 2
    mse = float(np.square(error).mean())
    return {"future_frames": target.shape[1] - 1, "mse_0_1": mse,
            "psnr_db": float(-10 * np.log10(max(mse, 1e-12)))}


def window_metrics(arrays):
    hold = np.broadcast_to(arrays["raw_current_state"], arrays["raw_action_future"].shape)
    static = np.repeat(arrays["video_target"][:, :1], arrays["video_target"].shape[1], axis=1)
    metrics = {"action": action_metrics(arrays["action_prediction"], arrays["raw_action_future"], arrays["action_is_pad"]),
               "hold_action": action_metrics(hold, arrays["raw_action_future"], arrays["action_is_pad"]),
               "video": video_metrics(arrays["video_prediction"], arrays["video_target"]),
               "static_video": video_metrics(static, arrays["video_target"])}
    metrics["beats_hold_action_xyz"] = metrics["action"]["xyz_mean_axis_mae_mm"] < metrics["hold_action"]["xyz_mean_axis_mae_mm"]
    metrics["beats_static_video"] = metrics["video"]["psnr_db"] > metrics["static_video"]["psnr_db"]
    return metrics


def summarize(rows):
    complete = [row for row in rows if row["status"] == "complete"]
    result = {"completed_windows": len(complete), "aggregation": "equal complete-window weight"}
    if complete:
        for group in ("action", "hold_action", "video", "static_video"):
            result[group] = {key + "_mean": float(np.mean([row["metrics"][group][key] for row in complete]))
                             for key, value in complete[0]["metrics"][group].items()
                             if isinstance(value, (int, float)) and not isinstance(value, bool)}
        result["beats_hold_action_xyz_windows"] = sum(row["metrics"]["beats_hold_action_xyz"] for row in complete)
        result["beats_static_video_windows"] = sum(row["metrics"]["beats_static_video"] for row in complete)
    return result
