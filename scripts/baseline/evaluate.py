#!/usr/bin/env python3
"""Original FastWAM inference on five preselected training episodes."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

from fastwam.baseline.analysis import (PROTOCOL, EPISODES, file_sha256, json_sha256, source_identity,
                                      select_five_windows, summarize, window_metrics)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def data_identity(directory):
    root = Path(directory).resolve()
    manifest_path = root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") == "forcevla-fastwam-conversion-v2":
        # New two-view conversion has a separate strict production validator.
        # Preserve the old single-view v1 protocol rather than accepting any
        # new manifest solely on its episode count.
        converter_path = Path(__file__).resolve().parents[1] / "forcevla/prepare_forcevla.py"
        spec = importlib.util.spec_from_file_location("forcevla_two_view_conversion", converter_path)
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        verified = converter.validate_converted_libero2cam(root)
        return {"conversion_manifest_sha256": file_sha256(manifest_path),
                "episodes": 50, "frames": 34808, "source_revision": verified["source_revision"],
                "payloads_verified": True, "profile": verified["profile"], "cameras": verified["cameras"],
                "camera_output_order": verified["camera_output_order"]}
    if (manifest.get("schema") != "forcevla-fastwam-conversion-v1" or
            manifest.get("episodes") != 50 or manifest.get("frames") != 34808):
        raise ValueError("Baseline requires the exact complete 50-episode wipe conversion")
    for row in manifest["output_files"]:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != row["sha256"]:
            raise ValueError(f"Converted data changed: {path}")
    return {"conversion_manifest_sha256": file_sha256(manifest_path),
            "episodes": 50, "frames": 34808, "source_revision": manifest["source_revision"],
            "payloads_verified": True}


def load_exact_checkpoint(model, path, expected_step):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("step") != expected_step or "mot" not in payload or "proprio_encoder" not in payload:
        raise ValueError("Checkpoint step or original FastWAM payload does not match the training target")
    # Upstream load_checkpoint accepts missing MoT keys; comparison requires an
    # exact complete checkpoint. This only tightens loading, never inference.
    model.mot.load_state_dict(payload["mot"], strict=True)
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    del payload


def evaluate(args):
    import torch
    import pyarrow.parquet as pq
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import default_collate
    from fastwam.baseline.evaluation import infer_unconditioned
    from fastwam.utils import misc

    started = time.monotonic()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config)
    if len(cfg.data.train.dataset_dirs) != 1 or float(cfg.data.train.val_set_proportion) != 0:
        raise ValueError("Five-episode fit requires one complete unsplit wipe dataset")
    source_root = Path(__file__).resolve().parents[2]
    code_hashes = source_identity(source_root)
    data_root = Path(cfg.data.train.dataset_dirs[0]).resolve()
    data = data_identity(data_root)
    is_two_view = data.get("profile") == "forcevla-libero2cam-v1"
    if is_two_view and (list(cfg.data.train.video_size) != [224, 448] or
                       [image.key for image in cfg.data.train.shape_meta.images] != ["cam_main", "cam_wrist"] or
                       int(cfg.data.train.processor.num_output_cameras) != 2 or
                       str(cfg.data.train.concat_multi_camera) != "horizontal"):
        raise ValueError("Two-view evidence requires the complete LIBERO left/right image configuration")
    misc.register_work_dir(str(root / "dataset_runtime"))
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    dataset_cfg._target_ = "fastwam.baseline.evaluation.VerifiedEvaluationDataset"
    dataset = instantiate(dataset_cfg, pretrained_norm_stats=str(Path(args.stats).resolve()))
    dataset.lerobot_dataset.processor.eval()
    boundaries = dataset.lerobot_dataset.episode_data_index
    windows = select_five_windows(boundaries["from"], boundaries["to"], mode=args.mode,
                                  horizon=dataset.lerobot_dataset.action_size)
    identity = {"protocol": PROTOCOL, "scope": "training_set_fit_only", "episodes": list(EPISODES),
                "config_sha256": file_sha256(args.config), "checkpoint_sha256": file_sha256(args.checkpoint),
                "checkpoint_step": args.expected_step, "stats_sha256": file_sha256(args.stats),
                "source_files": code_hashes, "source_sha256": json_sha256(code_hashes), "data": data,
                "selection": windows, "mode": args.mode, "seed": args.seed, "steps": args.steps,
                "inference": "original FastWAM.infer_joint; action=None; observation+T5+current proprio only",
                "curve_semantics": "32-step open-loop forecasts; sequence mode uses GT observations at each chunk anchor"}
    identity["sha256"] = json_sha256(identity)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["identity"]["sha256"] != identity["sha256"]:
            raise ValueError("Evaluation identity changed; use a fresh output directory")
    results = []
    for window in windows:
        row_path = root / "windows" / (window["key"] + ".json")
        if row_path.exists():
            row = json.loads(row_path.read_text())
            if row.get("identity_sha256") != identity["sha256"]:
                raise ValueError("Saved window identity changed")
            if row["status"] == "complete":
                payload_path = root / row["npz"]
                if file_sha256(payload_path) != row["npz_sha256"]:
                    raise ValueError("Saved numeric payload changed")
                with np.load(payload_path, allow_pickle=False) as payload:
                    if json_sha256(window_metrics(dict(payload))) != json_sha256(row["metrics"]):
                        raise ValueError("Saved metrics disagree with numeric payload")
                results.append(row)
    done = {row["key"] for row in results}

    def save(reason):
        status = "complete" if len(results) == len(windows) and all(r["status"] == "complete" for r in results) else "partial"
        atomic_json(root / "metrics.json", {"status": status, "scope": "training_set_fit_only", "expected_windows": len(windows),
                                            "identity_sha256": identity["sha256"], "summary": summarize(results), "windows": results})
        atomic_json(manifest_path, {"identity": identity, "status": status, "reason": reason,
                                   "metrics_sha256": file_sha256(root / "metrics.json")})
        return status

    save("initialized_or_resumed")
    if len(done) == len(windows):
        return 0
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[str(cfg.mixed_precision)]
    # The fine-tuned checkpoint is strictly complete; avoid reading the 12GB
    # original release again when every trainable tensor is about to be replaced.
    model = instantiate(cfg.model, model_dtype=dtype, device=args.device,
                        pretrained_fastwam_checkpoint=None, require_pretrained_fastwam=False)
    if model.__class__.__name__ != "FastWAM" or model.mot.num_layers != 30:
        raise ValueError("Baseline must be the original 30-layer FastWAM class")
    load_exact_checkpoint(model, args.checkpoint, args.expected_step)
    model.eval().requires_grad_(False)
    processor = dataset.lerobot_dataset.processor
    raw_episodes = {}
    for window in windows:
        if window["key"] in done:
            continue
        if args.deadline_seconds and time.monotonic() - started >= args.deadline_seconds:
            save("deadline")
            return 2
        row = {**window, "identity_sha256": identity["sha256"]}
        tick = time.monotonic()
        try:
            sample = default_collate([dataset._get(window["index"])])
            if dataset.lerobot_dataset.last_actual_index != window["index"]:
                raise ValueError("Official reader substituted a different evaluation sample")
            if bool(sample["action_is_pad"].any()) or bool(sample["image_is_pad"].any()) or bool(sample["proprio_is_pad"].any()):
                raise ValueError("Selected window must be complete")
            episode = window["episode_ordinal"]
            if episode not in raw_episodes:
                files = list(data_root.glob(f"data/chunk-*/episode_{episode:06d}.parquet"))
                if len(files) != 1:
                    raise ValueError("Expected one converted episode parquet")
                table = pq.read_table(files[0], columns=["action", "observation.state"])
                raw_dtype = np.float64 if is_two_view else np.float32
                raw_episodes[episode] = {key: np.asarray(table[key].to_pylist(), dtype=raw_dtype) for key in table.column_names}
            frame, horizon = window["frame_index"], sample["action"].shape[1]
            raw_action = raw_episodes[episode]["action"][frame:frame + horizon]
            captured = dataset.lerobot_dataset.last_raw_action.numpy()
            # Stock reader converts to float32 for the model. New raw curves
            # preserve the original float64 labels independently of that cast.
            if not np.array_equal(captured, raw_action.astype(captured.dtype)):
                raise ValueError("Official successful reader raw action differs from converted source")
            device = torch.device(args.device)
            with torch.inference_mode(), torch.autocast(device.type, dtype=dtype,
                    enabled=device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}):
                output = infer_unconditioned(model, sample, seed=args.seed, steps=args.steps)
            action_normalized = output["action"].detach().cpu().float()
            action = processor.normalizer.normalizers["action"]["default"].backward(action_normalized)
            predicted_video = np.stack([np.asarray(frame, dtype=np.float32) for frame in output["video"]], axis=0)
            predicted_video = predicted_video.transpose(3, 0, 1, 2) / 127.5 - 1
            arrays = {"raw_action_future": raw_action, "action_prediction": action.numpy(),
                      "action_prediction_normalized": action_normalized.numpy(),
                      "raw_current_state": raw_episodes[episode]["observation.state"][frame],
                      "action_is_pad": sample["action_is_pad"][0].numpy(),
                      "video_prediction": predicted_video, "video_target": sample["video"][0].numpy(),
                      "source_frame_indices": np.arange(frame, frame + horizon),
                      "time_seconds": np.arange(frame, frame + horizon, dtype=np.float64) / 30}
            row["metrics"] = window_metrics(arrays)
            row["npz"] = "windows/" + window["key"] + ".npz"
            (root / "windows").mkdir(exist_ok=True)
            np.savez_compressed(root / row["npz"], **arrays)
            row.update(status="complete", npz_sha256=file_sha256(root / row["npz"]))
        except Exception as error:
            row.update(status="failed", error_type=type(error).__name__, error=str(error))
        row["elapsed_seconds"] = time.monotonic() - tick
        atomic_json(root / "windows" / (window["key"] + ".json"), row)
        results.append(row)
        save("window_finished")
        print(json.dumps({"key": row["key"], "status": row["status"], "finished": len(results), "expected": len(windows)}), flush=True)
    return 0 if save("finished") == "complete" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "stats", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--expected-step", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--mode", choices=("middle", "sequence"), default="middle")
    parser.add_argument("--deadline-seconds", type=float, default=1200)
    args = parser.parse_args()
    if min(args.expected_step, args.steps, args.deadline_seconds) <= 0:
        parser.error("Step and budget must be positive")
    raise SystemExit(evaluate(args))


if __name__ == "__main__":
    main()
