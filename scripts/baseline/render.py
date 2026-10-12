#!/usr/bin/env python3
"""Render audited GT/prediction curves and video strips for the fixed five episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fastwam.baseline.analysis import EPISODES, AXES, file_sha256, json_sha256, window_metrics


def render(evaluation, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, destination = Path(evaluation).resolve(), Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((root / "metrics.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if metrics["status"] != "complete" or manifest["status"] != "complete":
        raise ValueError("Only complete frozen five-episode results can be rendered as a finished baseline")
    if file_sha256(root / "metrics.json") != manifest["metrics_sha256"] or metrics["identity_sha256"] != manifest["identity"]["sha256"]:
        raise ValueError("Evaluation manifest/metrics changed")
    episodes = {episode: [] for episode in EPISODES}
    for row in metrics["windows"]:
        path = (root / row["npz"]).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != row["npz_sha256"]:
            raise ValueError("Numeric payload hash differs")
        with np.load(path, allow_pickle=False) as payload:
            arrays = dict(payload)
        if json_sha256(window_metrics(arrays)) != json_sha256(row["metrics"]):
            raise ValueError("Recomputed metrics differ")
        episodes[row["episode_ordinal"]].append((row, arrays))
    outputs = []
    colors = {"GT": "#253e5e", "Prediction": "#df6e38", "Hold": "#90959d"}
    for episode, chunks in episodes.items():
        if not chunks:
            raise ValueError("Expected all five episode identities")
        chunks.sort(key=lambda item: item[0]["frame_index"])
        figure, axes = plt.subplots(4, 2, figsize=(12, 10), constrained_layout=True)
        for axis_index, axis in enumerate(axes.flat):
            if axis_index == 7:
                axis.axis("off")
                axis.text(0, .9, "Training-set fit only\nOriginal 30-video / 30-action FastWAM\nGaussian-noise action inference\nRotation: unverified source coordinates\nGT current observations at each forecast anchor", va="top", fontsize=11)
                continue
            factor, unit = (1000, "mm") if axis_index < 3 else (1, "source units")
            for chunk_index, (row, arrays) in enumerate(chunks):
                times = arrays["time_seconds"]
                hold = np.broadcast_to(arrays["raw_current_state"], arrays["raw_action_future"].shape)
                for key, values, style in (("GT", arrays["raw_action_future"], "-"),
                                           ("Prediction", arrays["action_prediction"], "-"), ("Hold", hold, "--")):
                    axis.plot(times, values[:, axis_index] * factor, style, lw=1.6, color=colors[key],
                              label=key if chunk_index == 0 else None)
                if len(chunks) > 1:
                    axis.axvline(times[0], color="#d7dce2", lw=.6)
            axis.set_title(AXES[axis_index])
            axis.set_ylabel(unit)
            axis.set_xlabel("Source episode time (s)")
            axis.grid(alpha=.2)
            if axis_index == 0:
                axis.legend(loc="best")
        figure.suptitle(f"Episode {episode:03d} — GT vs original FastWAM prediction ({len(chunks)} forecast chunks)", fontsize=14)
        path = destination / f"episode_{episode:03d}_action_curves.png"
        figure.savefig(path, dpi=190)
        plt.close(figure)
        outputs.append(path)
        row, arrays = chunks[len(chunks) // 2]
        figure, axes = plt.subplots(2, arrays["video_target"].shape[1], figsize=(16, 4), constrained_layout=True)
        for lane, (name, key) in enumerate((("GT", "video_target"), ("Prediction", "video_prediction"))):
            frames = ((arrays[key].transpose(1, 2, 3, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
            for index, frame in enumerate(frames):
                axes[lane, index].imshow(frame)
                axes[lane, index].set_xticks([])
                axes[lane, index].set_yticks([])
                if lane == 0:
                    axes[lane, index].set_title(f"+{index * 4 / 30:.2f}s" + ("\ncondition" if index == 0 else ""), fontsize=10)
                if index == 0:
                    axes[lane, index].set_ylabel(name, fontsize=11)
        figure.suptitle(f"Episode {episode:03d}, frame {row['frame_index']} — original FastWAM video forecast", fontsize=14)
        path = destination / f"episode_{episode:03d}_video_strip.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)
    result = {"evaluation_identity_sha256": metrics["identity_sha256"], "metrics_sha256": file_sha256(root / "metrics.json"),
              "plots": [{"path": p.name, "sha256": file_sha256(p)} for p in outputs],
              "scope": "training-set fit; chunk forecasts use GT current observations, not a closed-loop rollout"}
    (destination / "plots_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.evaluation, args.output), indent=2))
