#!/usr/bin/env python3
"""Plan or explicitly execute frozen VAE preprocessing. Never starts training.

Default: print a plan without constructing a raw dataset or any model.
Execution: use local verified VAE/T5 weights only, encode each full stock
nine-frame anchor independently, and write resumable feature/numeric shards.
"""
from __future__ import annotations

import argparse
import importlib.util
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_VAE_SHA256 = "0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996"
EXPECTED_T5_SHA256 = "d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23"
EXPECTED_STATS_SHA256 = "6d0ba7d47ce5bf5f290e3b5392ec8c13c07c2139e9a4f18d65a46cd5cb3a0dd4"


def load_config(args):
    from omegaconf import OmegaConf
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    if args.config:
        config = OmegaConf.load(args.config)
    else:
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
            config = compose(config_name="train", overrides=[f"task={args.task}"])
    path = Path(args.raw_config or config.get("cache_source_config", REPO_ROOT / "configs/data/forcevla_libero_2cam.yaml"))
    if not path.is_absolute():
        path = REPO_ROOT / path
    data = OmegaConf.load(path)
    raw_container = data.get("data", data)
    wrapper = OmegaConf.create({"data": OmegaConf.to_container(raw_container, resolve=False)})
    raw = wrapper.data.train
    if args.dataset:
        raw.dataset_dirs = [str(Path(args.dataset).resolve())]
    if args.stats:
        raw.pretrained_norm_stats = str(Path(args.stats).resolve())
    if args.text_cache:
        raw.text_embedding_cache_dir = str(Path(args.text_cache).resolve())
    return config, raw


def plan(args, raw):
    return {"status": "preprocessing_planned", "training_started": False,
            "execute_cache": bool(args.execute_cache), "cache_root": str(Path(args.cache_root).resolve()),
            "raw_dataset": list(raw.dataset_dirs), "raw_config": args.raw_config or "configs/data/forcevla_libero_2cam.yaml",
            "samples": 34808, "episodes": 50, "window_RGB_shape": [1, 3, 9, 224, 448],
            "window_latent_shape": [1, 48, 3, 14, 28], "latent_dtype": "bfloat16",
            "numeric_dtype": "float32", "raw_numeric_dtype": "float64", "shard_size": args.shard_size,
            "latent_bytes": 48 * 3 * 14 * 28 * 2 * 34808,
            "encoding": "independent full9frame window; VAE clears causal cache per call; scaled posterior mean",
            "vae_backend": args.vae_backend, "requires_local_weights": True,
            "resume_policy": "same identity only; validate every committed payload; no retry/replacement/RGB fallback",
            "max_new_samples": args.max_new_samples, "max_cache_seconds": args.max_cache_seconds,
            "GPU_preprocessing": "explicit --execute-cache only; no policy/optimizer/training"}


def _task_texts(raw):
    import hashlib
    import torch
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from fastwam.cache.latent import CacheIntegrityError, file_sha256

    texts, hashes = {}, {}
    root = Path(raw.dataset_dirs[0]).resolve()
    for line in (root / "meta/tasks.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        prompt = DEFAULT_PROMPT.format(task=json.loads(line)["task"])
        path = Path(raw.text_embedding_cache_dir) / (hashlib.sha256(prompt.encode()).hexdigest() + ".t5_len128.wan22ti2v5b.pt")
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen task text cache: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        context, mask = payload["context"].clone(), payload["mask"].bool()
        if tuple(context.shape) != (128, 4096) or tuple(mask.shape) != (128,) or context.dtype != torch.bfloat16:
            raise CacheIntegrityError("Source T5 cache is not the original BF16 128x4096 payload")
        context[~mask] = 0
        texts[prompt] = {"context": context, "context_mask": torch.ones_like(mask)}
        hashes[prompt] = file_sha256(path)
    return texts, hashes


def execute(args, config, raw):
    started = time.monotonic()
    import torch
    from omegaconf import OmegaConf
    from fastwam.cache import make_cache_identity, cache_identity_sha256, validate_cache, write_window_cache
    from fastwam.cache.latent import CacheIntegrityError, file_sha256
    from fastwam.cache.source import build_index_rows, build_strict_source

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise CacheIntegrityError("Cache writer is single-process; do not launch it through distributed training")
    if not args.device.startswith("cuda"):
        raise ValueError("Production VAE preprocessing requires an explicitly selected CUDA device")
    if not args.vae_weights or not args.t5_weights:
        raise ValueError("Execution requires explicit local --vae-weights and --t5-weights paths")
    vae_path, t5_path = Path(args.vae_weights).resolve(), Path(args.t5_weights).resolve()
    if not vae_path.is_file() or not t5_path.is_file():
        raise FileNotFoundError("Local VAE and T5 weights must exist; precompute never downloads weights")
    if file_sha256(vae_path) != EXPECTED_VAE_SHA256:
        raise CacheIntegrityError("VAE full payload SHA256 differs from the exact frozen official component")
    if file_sha256(t5_path) != EXPECTED_T5_SHA256:
        raise CacheIntegrityError("T5 full payload SHA256 differs from the exact frozen official encoder")
    dataset_root = Path(raw.dataset_dirs[0]).resolve()
    spec = importlib.util.spec_from_file_location("forcevla_cache_conversion", REPO_ROOT / "scripts/forcevla/prepare_forcevla.py")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    converter.validate_converted_libero2cam(dataset_root)
    stats_path = Path(raw.pretrained_norm_stats).resolve()
    if not stats_path.is_file():
        raise FileNotFoundError("Frozen official normalization stats must already exist")
    if file_sha256(stats_path) != EXPECTED_STATS_SHA256:
        raise CacheIntegrityError("Normalization stats SHA256 differs from the frozen complete wipe50 stats")
    texts, text_hashes = _task_texts(raw)
    episodes = [json.loads(line)["length"] for line in (dataset_root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    raw_resolved = OmegaConf.to_container(raw, resolve=True)
    preprocessing = {key: value for key, value in raw_resolved.items()
                     if key not in ("dataset_dirs", "pretrained_norm_stats", "text_embedding_cache_dir", "use_text_embed_cache")}
    preprocessing["runtime_versions"] = {name: importlib.metadata.version(name)
                                         for name in ("torch", "torchvision", "datasets", "pyarrow", "pillow", "safetensors")}
    source_files = ["src/fastwam/models/wan22/wan_video_vae.py", "src/fastwam/datasets/lerobot/robot_video_dataset.py",
                    "src/fastwam/datasets/lerobot/base_lerobot_dataset.py", "src/fastwam/datasets/dataset_utils.py",
                    "src/fastwam/datasets/lerobot/processors/fastwam_processor.py",
                    "src/fastwam/datasets/lerobot/processors/base_processor.py",
                    "src/fastwam/datasets/lerobot/utils/normalizer.py", "src/fastwam/datasets/lerobot/transforms/action_state_merger.py",
                    "src/fastwam/datasets/lerobot/transforms/image.py", "src/fastwam/baseline/evaluation.py",
                    "src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py",
                    "src/fastwam/datasets/lerobot/lerobot/datasets/utils.py",
                    "src/fastwam/cache/source.py", "src/fastwam/cache/latent.py", "scripts/forcevla/precompute_vae_cache.py"]
    identity = make_cache_identity(data_conversion_sha256=file_sha256(dataset_root / "conversion_manifest.json"),
        dataset_stats_sha256=file_sha256(stats_path), vae_weights_sha256=file_sha256(vae_path),
        text_encoder_weights_sha256=file_sha256(t5_path), preprocessing=preprocessing,
        source_code_sha256={path: file_sha256(REPO_ROOT / path) for path in source_files},
        text_payload_sha256=text_hashes, episode_lengths=episodes, encoding_backend=args.vae_backend)
    cache_root = Path(args.cache_root).resolve()
    if cache_root.exists() and args.resume:
        manifest = validate_cache(cache_root, cache_identity_sha256(identity), require_complete=False)
        if manifest["status"] == "complete":
            return manifest
    rows = build_index_rows(dataset_root)
    getter = build_strict_source(raw, rows, cache_root.parent / (cache_root.name + "_source_runtime"))
    # The registered component loader constructs only WanVideoVAE38, never a policy.
    from fastwam.models.wan22.helpers.loader import _load_registered_model
    vae = _load_registered_model(str(vae_path), "wan_video_vae", torch_dtype=torch.bfloat16, device=args.device)
    vae.eval().requires_grad_(False)
    encode = vae.model.encode
    if args.vae_backend == "cudagraphs":
        encode = torch.compile(encode, backend="cudagraphs", fullgraph=True)

    def encoder(video):
        # Stock build_inputs casts RGB to model BF16 before its autocast encode.
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return encode(video.to(device=args.device, dtype=torch.bfloat16), vae.scale).clone()

    def progress(manifest, entry):
        event = {"event": "cache_shard_committed", "identity_sha256": manifest["identity_sha256"],
                 "completed_samples": manifest["completed_samples"], "expected_samples": 34808,
                 "status": manifest["status"], "shard": entry["path"],
                 "elapsed_seconds": time.monotonic() - started,
                 "preprocessing_seconds": manifest["preprocessing_seconds"],
                 "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device), "training_started": False}
        progress_path = cache_root.parent / (cache_root.name + "_progress.json")
        temporary = progress_path.with_name(progress_path.name + ".tmp")
        temporary.write_text(json.dumps(event, indent=2) + "\n")
        temporary.replace(progress_path)
        print(json.dumps(event), file=sys.stderr, flush=True)

    manifest = write_window_cache(cache_root, identity, getter, rows, encoder, texts,
                                  shard_size=args.shard_size, resume=args.resume,
                                  max_new_samples=args.max_new_samples, max_seconds=args.max_cache_seconds,
                                  on_shard_commit=progress)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--config", help="Resolved training config; raw source is loaded separately")
    group.add_argument("--task", default="forcevla_libero_2cam_cached_4gpu_30k")
    parser.add_argument("--raw-config", help="Original RGB data yaml, default forcevla_libero_2cam.yaml")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--stats")
    parser.add_argument("--text-cache")
    parser.add_argument("--vae-weights")
    parser.add_argument("--t5-weights")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vae-backend", choices=("cudagraphs", "eager"), default="cudagraphs")
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--max-new-samples", type=int, help="Optional bounded preprocessing pilot; partial cache must be resumed before training")
    parser.add_argument("--max-cache-seconds", type=float, help="Optional encode-time limit, checked between whole windows; commits valid partial shards")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute-cache", action="store_true", help="Explicitly execute frozen GPU preprocessing; no training")
    args = parser.parse_args()
    config, raw = load_config(args)
    if not args.execute_cache:
        print(json.dumps(plan(args, raw), indent=2, ensure_ascii=False))
        return
    manifest = execute(args, config, raw)
    print(json.dumps({"status": manifest["status"], "completed_samples": manifest["completed_samples"],
                      "identity_sha256": manifest["identity_sha256"], "preprocessing_seconds": manifest.get("preprocessing_seconds"),
                      "stopped_reason": manifest.get("stopped_reason"), "training_started": False}, indent=2))
    if manifest["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
