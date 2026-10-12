#!/usr/bin/env python3
"""Export the eight-GPU feature-only recipe; never launch training or a model.

Without --check-cache this is a configuration-only artifact, not a ready run.
--check-cache verifies every completed cache shard and binds its identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex

TASK = "forcevla_libero_2cam_cached_4gpu_30k"
RELEASE_SHA = "1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579"
VAE_SHA = "0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996"
STATS_SHA = "6d0ba7d47ce5bf5f290e3b5392ec8c13c07c2139e9a4f18d65a46cd5cb3a0dd4"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def config_overrides(shared_repo, cache_root, checkpoint, output, identity=None, num_processes=4):
    if num_processes not in (4, 8):
        raise ValueError("Prepared distributed recipes support four or eight GPUs")
    task = f"forcevla_libero_2cam_cached_{num_processes}gpu_30k"
    shared = Path(shared_repo).resolve()
    args = [f"task={task}", f"output_dir={Path(output).resolve()}",
            f"data.train.cache_root={Path(cache_root).resolve()}",
            f"model.pretrained_fastwam_checkpoint={Path(checkpoint).resolve()}",
            f"model.action_dit_pretrained_path={shared / 'checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt'}"]
    if identity is not None:
        args.append(f"data.train.expected_identity_sha256={identity}")
    return args


def validate_config(config):
    data, model = config["data"]["train"], config["model"]
    expected = {"batch_size": 16, "gradient_accumulation_steps": 1,
                "max_steps": 30000, "save_every": 5000, "eval_every": 0,
                "learning_rate": 1e-4, "weight_decay": 1e-2, "mixed_precision": "bf16",
                "lr_scheduler_type": "cosine", "seed": 42, "resume": None}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("The explicit 8x16/no-accumulation/30000/save5000 recipe changed")
    if (data["_target_"] != "fastwam.cache.LatentCacheDataset" or
            not data["require_production"] or "val" in config["data"] or
            any(key in data for key in ("processor", "dataset_dirs", "text_embedding_cache_dir"))):
        raise ValueError("Training must read only a complete production latent cache, without an RGB reader")
    if (model["proprio_dim"] != 8 or model["video_dit_config"]["action_dim"] != 7 or
            model["action_dit_config"]["action_dim"] != 7 or
            model["video_dit_config"]["num_layers"] != 30 or
            model["action_dit_config"]["num_layers"] != 30 or
            model["video_dit_config"]["action_conditioned"] or
            model["video_dit_config"]["video_attention_mask_mode"] != "first_frame_causal" or
            model["mot_checkpoint_mixed_attn"] or model["compile_training_denoise"] or
            not model["require_pretrained_fastwam"] or not model["skip_dit_load_from_pretrain"] or
            model["pretrained_fastwam_sha256"] != RELEASE_SHA):
        raise ValueError("Original Video30/Action30, attention mask or strict release initialization changed")


def prepare(shared_repo, cache_root, checkpoint, output_plan, run_output, *, check_cache=False, num_processes=4):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[2]
    if num_processes not in (4, 8):
        raise ValueError("Prepared distributed recipes support four or eight GPUs")
    task = f"forcevla_libero_2cam_cached_{num_processes}gpu_30k"
    global_batch = 16 * num_processes
    output_plan = Path(output_plan).resolve()
    if output_plan.suffix != ".json":
        raise ValueError("Plan filename must end with .json")
    yaml_path = output_plan.with_suffix(".yaml")
    if any(path.exists() or path.is_symlink() for path in (output_plan, yaml_path)):
        raise FileExistsError("Refusing to overwrite a prepared plan or its YAML companion")
    cache_check = {"status": "not_checked", "training_ready": False}
    identity = None
    if check_cache:
        from fastwam.cache import validate_cache
        manifest = validate_cache(cache_root)
        if (manifest["identity"]["scope"] != "production" or
                manifest["identity"]["sample_count"] != 34808 or
                manifest["identity"]["vae_weights_sha256"] != VAE_SHA or
                manifest["identity"]["dataset_stats_sha256"] != STATS_SHA or
                manifest["identity"]["numeric_dtype"] != "float32"):
            raise ValueError("Expected all 34808 real ForceVLA wipe anchors")
        identity = manifest["identity_sha256"]
        cache_check = {"status": "verified_complete", "training_ready": True,
                       "identity_sha256": identity, "samples": 34808,
                       "manifest_sha256": sha256(Path(cache_root) / "manifest.json")}
    overrides = config_overrides(shared_repo, cache_root, checkpoint, run_output, identity, num_processes)
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    config = OmegaConf.to_container(cfg, resolve=True)
    validate_config(config)
    command = ["accelerate", "launch", "--config_file", f"scripts/accelerate_configs/accelerate_zero1_ds_{num_processes}gpu.yaml",
               "--num_processes", str(num_processes), "--num_machines", "1", "--machine_rank", "0", "scripts/train.py", *overrides]
    sources = [root / "configs/data/forcevla_libero_2cam_cached.yaml",
               root / "configs/task" / f"{task}.yaml",
               root / "configs/task/forcevla_libero_2cam_cached_8gpu_30k.yaml",
               root / "configs/data/forcevla_libero_2cam.yaml",
               root / "scripts/accelerate_configs" / f"accelerate_zero1_ds_{num_processes}gpu.yaml",
               root / "scripts/ds_configs/ds_zero1_config.json"]
    plan = {"status": "cache_verified_configuration" if check_cache else "configuration_only",
            "training_started": False, "task": task, "config": config,
            "cache_check": cache_check, "world_size": num_processes, "per_gpu_batch_size": 16,
            "gradient_accumulation_steps": 1, "effective_batch": global_batch, "max_steps": 30000,
            "warmup_updates": 1500, "save_every": 5000,
            "periodic_checkpoint_steps": [5000, 10000, 15000, 20000, 25000, 30000],
            "nominal_max_sample_draws": 30000 * global_batch, "nominal_anchor_passes": 30000 * global_batch / 34808,
            "cache_contract": {"sample_count": 34808, "episodes": 50, "split": "none",
                               "video_latents": {"shape": [48, 3, 14, 28], "dtype": "bfloat16",
                                                 "scaled": True, "video_num_frames": 9},
                               "action_shape": [32, 7], "proprio_shape": [32, 8], "target_dtype": "float32",
                               "raw_numeric_dtype": "float64",
                               "padding_shapes": [9, 32, 33], "vae_payload_sha256": VAE_SHA,
                               "stats_sha256": STATS_SHA,
                               "encoding": "Each anchor's nine-frame causal window is encoded independently once",
                               "rgb_decode_during_training": False, "vae_encode_during_training": False},
            "train_argv": command, "train_shell": shlex.join(command), "cwd": str(root),
            "environment": {"DIFFSYNTH_MODEL_BASE_PATH": str(Path(shared_repo).resolve() / "checkpoints"),
                            "DIFFSYNTH_SKIP_DOWNLOAD": "true", "PYTHONPATH": str(root / "src"),
                            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                            "TOKENIZERS_PARALLELISM": "false", "NNODES": "1", "NODE_RANK": "0"},
            "resource_limit": {"branch_total_cap_seconds": 18000, "gpu": f"{num_processes} H200 141GB (H100 permitted)",
                               "target_fit_within_cap": "unverified; training deliberately not started",
                               "allocation_includes_preparation": True},
            "verification_limit": "CPU bridge equivalence uses real small DiTs and fake VAE; full VAE hardware agreement unverified here",
            "evaluation": {"episodes": [0, 12, 24, 36, 49], "mode": "sequence", "expected_windows": 119,
                           "split": "training-fit only", "video_shape": [3, 9, 224, 448]},
            "source_config_files": {str(path.relative_to(root)): sha256(path) for path in sources}}
    output_plan.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, yaml_path, resolve=True)
    plan["resolved_yaml_sha256"] = sha256(yaml_path)
    output_plan.write_text(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-repo", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--output-plan", required=True)
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--check-cache", action="store_true")
    parser.add_argument("--num-processes", type=int, choices=(4, 8), default=4)
    args = parser.parse_args()
    plan = prepare(args.shared_repo, args.cache_root, args.pretrained_checkpoint,
                   args.output_plan, args.run_output, check_cache=args.check_cache, num_processes=args.num_processes)
    print(json.dumps({key: plan[key] for key in ("status", "training_started", "cache_check", "world_size", "effective_batch", "max_steps")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
