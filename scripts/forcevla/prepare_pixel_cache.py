#!/usr/bin/env python3
"""Plan, explicitly build, or exhaustively audit new stock-equivalent CPU pixels.

Default only prints a plan. Never loads a VAE, queries CUDA, or starts training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--raw-config", default=str(ROOT / "configs/data/forcevla_libero_2cam.yaml"))
    result.add_argument("--pixel-cache-root", required=True)
    result.add_argument("--dataset")
    result.add_argument("--stats")
    result.add_argument("--receipt", help="Fresh external JSON path required for exhaustive audit")
    result.add_argument("--shard-size", type=int, default=128)
    result.add_argument("--max-new-frames", type=int)
    result.add_argument("--cpu-threads", type=int, default=2)
    result.add_argument("--resume", action="store_true")
    modes = result.add_mutually_exclusive_group()
    modes.add_argument("--execute-pixels", action="store_true")
    modes.add_argument("--audit-pixels", action="store_true")
    return result


def load_raw(args):
    from omegaconf import OmegaConf
    from fastwam.utils.config_resolvers import register_default_resolvers
    register_default_resolvers()
    data = OmegaConf.load(args.raw_config)
    wrapper = OmegaConf.create({"data": OmegaConf.to_container(data.get("data", data), resolve=False)})
    raw = wrapper.data.train
    for key in ("dataset_dirs",):
        raw[key] = [str((Path(value) if Path(value).is_absolute() else ROOT / value).resolve()) for value in raw[key]]
    for key in ("pretrained_norm_stats", "text_embedding_cache_dir"):
        value = Path(raw[key])
        raw[key] = str((value if value.is_absolute() else ROOT / value).resolve())
    if args.dataset:
        raw.dataset_dirs = [str(Path(args.dataset).resolve())]
    if args.stats:
        raw.pretrained_norm_stats = str(Path(args.stats).resolve())
    return raw


def main(argv=None):
    args = parser().parse_args(argv)
    if args.cpu_threads < 1 or args.shard_size < 1 or (args.max_new_frames is not None and args.max_new_frames < 1):
        raise ValueError("CPU threads and pixel bounds must be positive")
    if not args.execute_pixels and not args.audit_pixels:
        print(json.dumps({"status": "pixel_preprocessing_planned", "pixel_cache_root": str(Path(args.pixel_cache_root).resolve()),
            "samples": 34808, "episodes": 50, "pixel_dtype": "float32", "frame_shape": [3, 224, 448],
            "pixel_bytes": 34808 * 3 * 224 * 448 * 4, "cpu_threads": args.cpu_threads,
            "unique_frame_processing": "original stock operator/cast order exactly once per committed recorded frame",
            "window_policy": "gather complete nine original frames per anchor; stock strides and BF16 cast at unchanged VAE boundary",
            "cpu_audit": "exhaustive independent stock batches plus every original full anchor/numeric/mask/task association",
            "execution_required": "exact --execute-pixels or --audit-pixels", "training_started": False,
            "cuda_queried": False, "vae_constructed": False}, indent=2))
        return 0
    import torch
    from fastwam.cache.deduplicated_source import RecordedFrameSource, pixel_identity_from_source
    from fastwam.cache.pixels import write_pixel_cache
    torch.set_num_threads(args.cpu_threads)
    raw = load_raw(args)
    if args.audit_pixels:
        if not args.receipt:
            raise ValueError("Exhaustive audit requires a fresh external --receipt path")
        if args.resume or args.max_new_frames is not None:
            raise ValueError("Exhaustive audit cannot be resumed or restricted to a frame subset")
        from fastwam.cache.pixel_audit import audit_pixel_cache
        receipt = audit_pixel_cache(raw, args.pixel_cache_root, args.receipt,
            on_episode=lambda event: print(json.dumps({"event": "stock_pixel_episode_verified", **event}), file=sys.stderr, flush=True))
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        return 0
    identity = pixel_identity_from_source(raw)
    recorded = RecordedFrameSource(raw.dataset_dirs[0], identity)
    manifest = write_pixel_cache(args.pixel_cache_root, identity, recorded, shard_size=args.shard_size,
        resume=args.resume, max_new_frames=args.max_new_frames,
        on_shard_commit=lambda manifest, entry: print(json.dumps({"event": "unique_pixel_shard_committed",
            "completed_frames": manifest["completed_frames"], "status": manifest["status"], "shard": entry["path"],
            "identity_sha256": manifest["identity_sha256"], "training_started": False}), file=sys.stderr, flush=True))
    print(json.dumps({"status": manifest["status"], "completed_frames": manifest["completed_frames"],
                      "identity_sha256": manifest["identity_sha256"], "training_started": False}, indent=2))
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
