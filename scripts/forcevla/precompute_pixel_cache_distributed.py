#!/usr/bin/env python3
"""Plan, explicitly encode one bounded rank, or CPU-merge audited pixel windows.

Default prints a static plan without constructing a dataset or loading Torch.
The exhaustive CPU pixel audit must pass before any VAE is constructed. Each
worker uses the original batch-one complete-window BF16 device cast/encoder.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "src/fastwam/models/wan22/wan_video_vae.py",
    "src/fastwam/models/wan22/helpers/loader.py", "src/fastwam/models/wan22/helpers/io.py",
    "src/fastwam/datasets/lerobot/robot_video_dataset.py", "src/fastwam/datasets/lerobot/base_lerobot_dataset.py",
    "src/fastwam/datasets/dataset_utils.py", "src/fastwam/datasets/lerobot/processors/fastwam_processor.py",
    "src/fastwam/datasets/lerobot/processors/base_processor.py", "src/fastwam/datasets/lerobot/utils/normalizer.py",
    "src/fastwam/datasets/lerobot/transforms/action_state_merger.py", "src/fastwam/datasets/lerobot/transforms/image.py",
    "src/fastwam/baseline/evaluation.py", "src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py",
    "src/fastwam/datasets/lerobot/lerobot/datasets/utils.py", "src/fastwam/cache/source.py",
    "src/fastwam/cache/latent.py", "src/fastwam/cache/pixels.py", "src/fastwam/cache/deduplicated_source.py",
    "src/fastwam/cache/pixel_audit.py", "src/fastwam/cache/distributed.py",
    "scripts/forcevla/precompute_vae_cache.py", "scripts/forcevla/precompute_pixel_cache_distributed.py")


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--prepare-plan", action="store_true", help="Explicit CPU-only source/asset/audit preparation")
    mode.add_argument("--execute-worker", action="store_true", help="Explicit independent VAE encoding; never training")
    mode.add_argument("--execute-merge", action="store_true", help="Explicit CPU-only strict full-rank merge")
    result.add_argument("--plan", help="New plan output for preparation; exact saved plan for worker/merge")
    result.add_argument("--pixel-cache")
    result.add_argument("--pixel-audit-receipt")
    result.add_argument("--workers-root")
    result.add_argument("--output", help="Fresh merged complete latent cache")
    result.add_argument("--merge-receipt")
    result.add_argument("--world-size", type=int)
    result.add_argument("--rank", type=int)
    result.add_argument("--device")
    result.add_argument("--physical-device")
    result.add_argument("--physical-devices", help="Preparation only: four distinct CUDA_VISIBLE_DEVICES identifiers")
    result.add_argument("--config")
    result.add_argument("--task", default="forcevla_libero_2cam_cached_4gpu_30k")
    result.add_argument("--raw-config")
    result.add_argument("--dataset")
    result.add_argument("--stats")
    result.add_argument("--text-cache")
    result.add_argument("--vae-weights")
    result.add_argument("--t5-weights")
    result.add_argument("--vae-backend", choices=("cudagraphs", "eager"), default="cudagraphs")
    result.add_argument("--shard-size", type=int, default=128)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--max-new-samples", type=int)
    result.add_argument("--max-worker-seconds", type=float)
    result.add_argument("--complete-assigned-episodes", action="store_true", help="Allow all assigned anchors; requires an explicit time bound")
    return result


def action_arguments(args):
    if not (args.prepare_plan or args.execute_worker or args.execute_merge):
        return None
    for key in ("plan", "pixel_cache", "pixel_audit_receipt", "workers_root", "vae_weights", "t5_weights"):
        if not getattr(args, key):
            raise ValueError("Explicit action requires --" + key.replace("_", "-"))
    if args.world_size != 4 or type(args.shard_size) is not int or args.shard_size <= 0:
        raise ValueError("Explicit world-size4 and positive shard-size are required")
    pixel_root = Path(args.pixel_cache).resolve()
    guarded = [("workers-root", Path(args.workers_root).resolve()), ("plan", Path(args.plan).resolve()),
               ("pixel-audit-receipt", Path(args.pixel_audit_receipt).resolve())]
    guarded.extend(("worker rank root", Path(args.workers_root).resolve() / f"rank_{rank:03d}") for rank in range(4))
    if args.output:
        output = Path(args.output).resolve()
        guarded.extend([("output", output), ("merge lock", output.with_name(output.name + "_merge.lock")),
                        ("merge receipt", Path(args.merge_receipt).resolve() if args.merge_receipt else output.with_name(output.name + "_merge.json"))])
    elif args.merge_receipt:
        guarded.append(("merge receipt", Path(args.merge_receipt).resolve()))
    for label, path in guarded:
        if path.is_relative_to(pixel_root) or (label in ("output", "worker rank root") and pixel_root.is_relative_to(path)):
            raise ValueError(label + " overlaps immutable pixel cache")
    if args.prepare_plan:
        devices = args.physical_devices.split(",") if args.physical_devices else []
        if len(devices) != 4 or any(not value or value.strip() != value for value in devices) or len(set(devices)) != 4:
            raise ValueError("Preparation requires four distinct physical device identifiers")
        return [{"device": "cuda:0", "physical_device": value, "visible_devices": value} for value in devices]
    if args.execute_merge:
        if not args.output:
            raise ValueError("Merge requires a fresh --output")
        return None
    if type(args.rank) is not int or not 0 <= args.rank < 4 or args.device != "cuda:0" or not args.physical_device:
        raise ValueError("Worker requires explicit rank0..3, devicecuda:0 and physical-device")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.physical_device or os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError("Each independent worker must expose exactly its bound physical device and run outside a distributed launcher")
    if args.max_new_samples is not None and args.max_new_samples <= 0:
        raise ValueError("Positive max-new-samples required")
    if args.complete_assigned_episodes and args.max_new_samples is not None:
        raise ValueError("Complete episode mode cannot also limit samples")
    if args.max_worker_seconds is not None and (not math.isfinite(args.max_worker_seconds) or args.max_worker_seconds <= 0):
        raise ValueError("Positive finite max-worker-seconds required")
    if args.complete_assigned_episodes and args.max_worker_seconds is None:
        raise ValueError("Full assigned episode encoding requires an explicit remaining time bound")
    return None


def _stock_cli():
    spec = importlib.util.spec_from_file_location("stock_pixel_window_precompute", ROOT / "scripts/forcevla/precompute_vae_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(args):
    """All dataset, pixel proof and asset checks precede model construction."""
    from omegaconf import OmegaConf
    from fastwam.cache import make_cache_identity
    from fastwam.cache.latent import CacheIntegrityError, file_sha256
    from fastwam.cache.deduplicated_source import build_deduplicated_source
    from fastwam.cache.pixel_audit import validate_pixel_audit_receipt

    stock = _stock_cli()
    _, raw = stock.load_config(args)
    source = build_deduplicated_source(raw, args.pixel_cache)
    pixel_sha = source.pixel_manifest["identity_sha256"]
    audit_path = Path(args.pixel_audit_receipt).resolve()
    audit_sha = file_sha256(audit_path)
    audit = validate_pixel_audit_receipt(audit_path, args.pixel_cache, pixel_sha)
    if file_sha256(audit_path) != audit_sha:
        raise CacheIntegrityError("Pixel audit receipt changed during fresh validation")
    for key in ("numeric_reference_sha256", "dataset_stats_sha256", "conversion_manifest_sha256"):
        if audit.get("source_provenance", {}).get(key) != source.source_provenance.get(key):
            raise CacheIntegrityError("Pixel audit/current raw numeric provenance differs: " + key)
    paths = {key: Path(value).resolve() for key, value in
             (("vae", args.vae_weights), ("t5", args.t5_weights), ("stats", raw.pretrained_norm_stats))}
    expected = {"vae": stock.EXPECTED_VAE_SHA256, "t5": stock.EXPECTED_T5_SHA256, "stats": stock.EXPECTED_STATS_SHA256}
    for key, path in paths.items():
        if not path.is_file() or file_sha256(path) != expected[key]:
            raise CacheIntegrityError("Exact local stock asset SHA differs: " + key)
    texts, text_hashes = stock._task_texts(raw)
    if set(texts) != set(source.task_prompts.values()):
        raise CacheIntegrityError("Original source prompts and stock cached T5 payloads differ")
    raw_values = OmegaConf.to_container(raw, resolve=True)
    preprocessing = {key: value for key, value in raw_values.items()
                     if key not in ("dataset_dirs", "pretrained_norm_stats", "text_embedding_cache_dir", "use_text_embed_cache")}
    import importlib.metadata
    preprocessing["runtime_versions"] = {name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "datasets", "pyarrow", "pillow", "safetensors")}
    preprocessing["source_v2"] = {"provenance": source.source_provenance,
        "pixel_audit_receipt_sha256": audit_sha, "world_size": 4,
        "pixel_dtype": "float32", "BF16_cast": "original_device_cast_after_full_window_gather",
        "stock_pilot_imported": False}
    identity = make_cache_identity(data_conversion_sha256=source.source_provenance["conversion_manifest_sha256"],
        dataset_stats_sha256=expected["stats"], vae_weights_sha256=expected["vae"],
        text_encoder_weights_sha256=expected["t5"], preprocessing=preprocessing,
        source_code_sha256={relative: file_sha256(ROOT / relative) for relative in SOURCE_FILES},
        text_payload_sha256=text_hashes, episode_lengths=source.episode_lengths, encoding_backend=args.vae_backend)
    return source, identity, texts, paths


def execute(args, devices):
    from fastwam.cache.distributed import (make_distributed_plan, merge_worker_caches,
                                          validate_distributed_plan, write_worker_cache)
    from fastwam.cache.latent import CacheIntegrityError, cache_identity_sha256

    source, identity, texts, paths = inputs(args)
    plan_path = Path(args.plan).resolve()
    if args.prepare_plan:
        if plan_path.is_relative_to(Path(args.pixel_cache).resolve()) or plan_path.is_relative_to(Path(args.workers_root).resolve()):
            raise ValueError("Plan must be external to pixel and worker payload roots")
        plan = make_distributed_plan(identity, source.pixel_manifest["identity_sha256"], source.index_rows,
            source.numeric_reference, source.task_prompts, world_size=4, device_bindings=devices,
            source_provenance=source.source_provenance)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with plan_path.open("x") as stream:
            stream.write(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        return {"status": "distributed_plan_prepared", "plan_sha256": plan["sha256"], "identity_sha256": plan["identity_sha256"], "formal_training_started": False}
    plan = json.loads(plan_path.read_text())
    validate_distributed_plan(plan, source.index_rows, source.numeric_reference)
    if plan["identity_sha256"] != cache_identity_sha256(identity) or plan["source_provenance"] != source.source_provenance:
        raise CacheIntegrityError("Actual source/config/pixel/audit/assets differ from saved rank plan")
    worker_roots = [Path(args.workers_root).resolve() / f"rank_{rank:03d}" for rank in range(4)]
    if args.execute_merge:
        return merge_worker_caches(worker_roots, args.output, plan, source.index_rows,
                                   source.numeric_reference, receipt_path=args.merge_receipt)
    binding = {"device": args.device, "physical_device": args.physical_device, "visible_devices": args.physical_device}
    if plan["device_bindings"][args.rank] != binding:
        raise CacheIntegrityError("Actual worker device differs from prepared rank binding")
    import torch
    from fastwam.models.wan22.helpers.loader import _load_registered_model
    # Exactly the frozen stock component loader and outer encoder, no policy.
    vae = _load_registered_model(str(paths["vae"]), "wan_video_vae", torch_dtype=torch.bfloat16, device=args.device)
    vae.eval().requires_grad_(False)
    encode = vae.model.encode
    if args.vae_backend == "cudagraphs":
        encode = torch.compile(encode, backend="cudagraphs", fullgraph=True)
    def encoder(video):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return encode(video.to(device=args.device, dtype=torch.bfloat16), vae.scale).clone()
    started = time.monotonic()
    def progress(manifest, entry):
        print(json.dumps({"event": "pixel_worker_shard_committed", "rank": args.rank,
            "plan_sha256": plan["sha256"], "completed_samples": manifest["completed_samples"],
            "assigned_samples": manifest["assigned_samples"], "status": manifest["status"],
            "shard": entry["path"], "elapsed_seconds": time.monotonic() - started,
            "formal_training_started": False}), file=sys.stderr, flush=True)
    max_samples = None if args.complete_assigned_episodes else args.max_new_samples or 3
    max_seconds = args.max_worker_seconds if args.max_worker_seconds is not None else 120.
    worker = write_worker_cache(worker_roots[args.rank], plan, args.rank, source, source.index_rows,
        encoder, texts, source.numeric_reference, shard_size=args.shard_size, resume=args.resume,
        max_new_samples=max_samples, max_seconds=max_seconds, on_shard_commit=progress)
    return {"status": worker["status"], "rank": args.rank, "completed_samples": worker["completed_samples"],
            "assigned_samples": worker["assigned_samples"], "plan_sha256": plan["sha256"],
            "formal_training_started": False, "hardware_equivalence_verified": False}


def main(argv=None):
    args = parser().parse_args(argv)
    devices = action_arguments(args)
    if not (args.prepare_plan or args.execute_worker or args.execute_merge):
        print(json.dumps({"status": "planning_only", "world_size": 4, "samples": 34808,
            "execute_worker": False, "model_loaded": False, "dataset_constructed": False,
            "required_gate": "exhaustive current-source CPU pixel audit; independent original full9 VAE per anchor",
            "default_worker_bounds": {"new_samples": 3, "seconds": 120, "checked_between_windows": True},
            "formal_training_started": False, "hardware_equivalence_verified": False}))
        return 0
    result = execute(args, devices)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 2 if args.execute_worker and result["status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
