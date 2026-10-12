#!/usr/bin/env python3
"""Write the complete two-view/30k training plan without starting training.

--check-data additionally checks the freshly converted two-camera dataset.
No torch/model construction, GPU allocation or subprocess launch occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path

TASK = "forcevla_libero_2cam_30k"
CAMERAS = {"observation.image": "cam_main", "observation.wrist_image": "cam_wrist"}
STATS_SHA256 = "6d0ba7d47ce5bf5f290e3b5392ec8c13c07c2139e9a4f18d65a46cd5cb3a0dd4"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_overrides(shared_repo, dataset, checkpoint, output):
    shared = Path(shared_repo).resolve()
    return [f"task={TASK}", f"output_dir={Path(output).resolve()}",
            f"data.train.dataset_dirs=[{Path(dataset).resolve()}]",
            f"data.train.pretrained_norm_stats={shared / 'data/forcevla_fastwam/shared_stats/dataset_stats.json'}",
            f"data.train.text_embedding_cache_dir={shared / 'data/text_embeds_cache/forcevla'}",
            f"model.pretrained_fastwam_checkpoint={Path(checkpoint).resolve()}",
            f"model.action_dit_pretrained_path={shared / 'checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt'}"]


def validate_config(config):
    data, model = config["data"]["train"], config["model"]
    processor = data["processor"]
    images = data["shape_meta"]["images"]
    if ([image["key"] for image in images] != list(CAMERAS.values()) or
            any(image["raw_shape"] != [3, 480, 640] or image["shape"] != [3, 224, 224] for image in images) or
            data["concat_multi_camera"] != "horizontal" or data["video_size"] != [224, 448] or
            processor["num_output_cameras"] != 2 or data["camera_key"] is not None):
        raise ValueError("Must use both real views in the exact LIBERO left/right image recipe")
    for key in ("train_transforms", "val_transforms"):
        if processor[key] != [{"_target_": "fastwam.datasets.lerobot.transforms.image.ToTensor"},
                              {"_target_": "torchvision.transforms.Resize", "size": [224, 224]}]:
            raise ValueError("Per-camera transforms differ from the official LIBERO recipe")
    if (data["num_frames"] != 33 or data["action_video_freq_ratio"] != 4 or
            data["global_sample_stride"] != 1 or data["val_set_proportion"] != 0 or
            not data["is_training_set"] or "val" in config["data"]):
        raise ValueError("Expected complete wipe training set without a validation/test split")
    if (processor["action_output_dim"] != 7 or processor["proprio_output_dim"] != 8 or
            processor["action_state_merger"]["state_target_dim"] != 8 or
            processor["norm_default_mode"] != "z-score" or
            processor["delta_action_dim_mask"]["default"] != [False] * 7):
        raise ValueError("ForceVLA physical targets/normalization/padding changed")
    if (config["max_steps"] != 30000 or config["batch_size"] != 1 or
            config["gradient_accumulation_steps"] != 16 or config["resume"] is not None or
            config["lr_scheduler_type"] != "cosine" or config["mixed_precision"] != "bf16" or
            config["learning_rate"] != 1e-4 or config["weight_decay"] != 1e-2):
        raise ValueError("Training horizon or optimizer recipe differs from the prepared fresh run")
    if (model["video_dit_config"]["num_layers"] != 30 or model["action_dit_config"]["num_layers"] != 30 or
            not model["require_pretrained_fastwam"] or not model["skip_dit_load_from_pretrain"] or
            model["video_dit_config"]["action_conditioned"] or
            model["pretrained_fastwam_sha256"] != "1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579"):
        raise ValueError("Must preserve the complete released original Video30/Action30 policy and mask")


def prepare(shared_repo, dataset, checkpoint, output_plan, run_output, *, check_data=False):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[2]
    output_plan = Path(output_plan).resolve()
    if output_plan.suffix != ".json":
        raise ValueError("Training plan must use a .json filename with a separate .yaml companion")
    companion = output_plan.with_suffix(".yaml")
    if any(path.exists() or path.is_symlink() for path in (output_plan, companion)):
        raise FileExistsError("Refusing to overwrite an existing training plan")
    overrides = config_overrides(shared_repo, dataset, checkpoint, run_output)
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    config = OmegaConf.to_container(cfg, resolve=True)
    validate_config(config)
    data_check = {"status": "not_checked", "note": "New two-view data has not been converted/verified by this plan command"}
    if check_data:
        from prepare_forcevla import validate_converted_libero2cam
        conversion = validate_converted_libero2cam(dataset)
        data_check = {"status": "verified_complete", "episodes": conversion["episodes"],
                      "frames": conversion["frames"], "cameras": conversion["cameras"],
                      "conversion_manifest_sha256": sha256(Path(dataset) / "conversion_manifest.json")}
        stats = Path(config["data"]["train"]["pretrained_norm_stats"])
        if sha256(stats) != STATS_SHA256:
            raise ValueError("Frozen ForceVLA normalization statistics changed")
    command = ["accelerate", "launch", "--config_file", "scripts/accelerate_configs/accelerate_zero1_ds.yaml",
               "--num_processes", "1", "scripts/train.py", *overrides]
    plan = {"status": "configuration_only", "training_started": False, "task": TASK,
            "camera_mapping": CAMERAS, "camera_order": ["left:cam_main", "right:cam_wrist"],
            "video_shape_BCTHW": [1, 3, 9, 224, 448], "expected_vae_latent_shape": [1, 48, 3, 14, 28],
            "expected_video_tokens": 294, "latent_shape_status": "source-derived; not a real VAE execution",
            "max_steps": 30000, "effective_batch": 16, "warmup_updates": 1500,
            "nominal_max_sample_draws": 480000, "nominal_anchor_passes": 480000 / 34808,
            "data_check": data_check, "config": config, "train_argv": command,
            "train_shell": shlex.join(command), "cwd": str(root),
            "environment": {"DIFFSYNTH_MODEL_BASE_PATH": str(Path(shared_repo).resolve() / "checkpoints"),
                            "DIFFSYNTH_SKIP_DOWNLOAD": "true", "PYTHONPATH": str(root / "src"),
                            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"},
            "preflight_required": "New two real optimizer updates using this exact 30000-step/two-camera recipe",
            "resource_limit": {"preferred_gpu": "H200 141GB", "previous_branch_cap_seconds": 18000,
                               "target_fit_within_cap": "not demonstrated; never silently shorten max_steps"},
            "evaluation": {"episodes": [0, 12, 24, 36, 49], "mode": "sequence", "expected_windows": 119,
                           "split": "training-fit only", "video_shape": [3, 9, 224, 448],
                           "evidence_collector": "Old external 224x224 collector must not validate this two-view run"},
            "asset_verification": "Not performed here; initialization factory hashes and strictly loads all policy tensors",
            "source_config_files": {str(path.relative_to(root)): sha256(path) for path in
                                    [root / 'configs/data/forcevla_libero_2cam.yaml',
                                     root / 'configs/task/forcevla_libero_2cam_30k.yaml',
                                     root / 'configs/data/libero_2cam.yaml',
                                     root / 'src/fastwam/datasets/lerobot/robot_video_dataset.py']}}
    output_plan.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, companion, resolve=True)
    output_plan.write_text(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-repo", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--output-plan", required=True)
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    plan = prepare(args.shared_repo, args.dataset, args.pretrained_checkpoint,
                   args.output_plan, args.run_output, check_data=args.check_data)
    print(json.dumps({key: plan[key] for key in ("status", "training_started", "max_steps", "video_shape_BCTHW", "data_check")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
