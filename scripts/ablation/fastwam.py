"""Explicit plan/stats/text-cache/train/resume commands; default is plan only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("plan", "check-hardware", "stats", "text-cache", "train", "resume", "worker"), default="plan")
    parser.add_argument("--config", default=str(ROOT / "configs/ablation/fastwam.yaml"))
    parser.add_argument("--dataset-root")
    parser.add_argument("--output", help="stats file for stats; run directory for training")
    parser.add_argument("--resume", help="Full Accelerate state directory; weights-only restart is not resume")
    parser.add_argument("--num-processes", type=int, default=None,
                        help="Override configured ablation_num_processes (default: 4)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--stop-after-steps", type=int, help="Save and stop early while preserving the configured scheduler and run identity")
    parser.add_argument("--smoke-audit", action="store_true", help="Collect and enforce actual update/restore evidence for the separate 3-step smoke run")
    args = parser.parse_args(argv)
    from fastwam.ablation.fastwam import (UPSTREAM_REVISION, RELEASE_SHA256, write_stats,
                                         make_training_config, build_binding, semantic_recipe,
                                         execution_profile, check_hardware, bounded_training_invocation)
    from fastwam.ablation.contracts import file_sha256, write_json
    if args.stop_after_steps is not None and args.command not in {"train", "resume", "worker"}:
        parser.error("--stop-after-steps only applies to train/resume")
    if args.smoke_audit and args.command not in {"train", "resume", "worker"}:
        parser.error("--smoke-audit only applies to train/resume")

    if args.command == "plan":
        print(json.dumps({"baseline": "fastwam", "official_revision": UPSTREAM_REVISION,
                          "official_release_sha256": RELEASE_SHA256,
                          "training_started": False, "config": str(Path(args.config).resolve()),
                          "commands": ["check-hardware", "stats", "text-cache", "train", "resume"],
                          "initialization": "Retain all official backbones; reset action input/output and proprio projections",
                          "training": "Linux CUDA + Accelerate/DeepSpeed ZeRO1; stage local assets first",
                          "inputs": "global/wrist RGB + absolute joint8; no force conditioning",
                          "actions": "32 x 8 absolute joint radians + gripper [0,1]"}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "stats":
        if not args.dataset_root or not args.output:
            parser.error("stats requires --dataset-root and --output")
        write_stats(args.dataset_root, args.output)
        print(f"Wrote official-format joint8 statistics: {Path(args.output).resolve()}")
        return 0

    overrides = list(args.set)
    if args.dataset_root:
        overrides.append(f"dataset_root={Path(args.dataset_root).resolve().as_posix()}")
    if args.output:
        overrides.append(f"output_dir={Path(args.output).resolve().as_posix()}")
    if args.resume:
        overrides.append(f"resume={Path(args.resume).resolve().as_posix()}")
    cfg = make_training_config(args.config, overrides)
    if args.smoke_audit and int(cfg.max_steps) != 3:
        parser.error("--smoke-audit requires the separate 3-step configuration")
    if args.command == "check-hardware":
        print(json.dumps(check_hardware(cfg), ensure_ascii=False, indent=2))
        return 0
    if args.command == "text-cache":
        # Calls the original T5 precompute routine with the resolved data config.
        # This is preprocessing, not optimization; it is never run by plan/stats.
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(Path(cfg.asset_root).resolve())
        namespace = runpy.run_path(str(ROOT / "scripts/precompute_text_embeds.py"))
        namespace["main"].__wrapped__(cfg)
        return 0

    num_processes = int(cfg.ablation_num_processes if args.num_processes is None else args.num_processes)
    if num_processes < 1:
        parser.error("--num-processes must be positive")
    if args.command == "resume" and not args.resume:
        parser.error("resume requires --resume with a full Accelerate state directory")
    if cfg.resume and (not Path(cfg.resume).is_dir() or not (Path(cfg.resume) / "trainer_state.json").is_file()):
        parser.error("Resume requires a complete saved state directory containing trainer_state.json")
    if args.command != "worker":
        cfg.ablation_num_processes = num_processes
    execution_profile(cfg)
    if args.stop_after_steps is not None and not 2 <= args.stop_after_steps <= int(cfg.max_steps):
        parser.error("--stop-after-steps must be between 2 and configured max_steps")
    hardware = check_hardware(cfg)
    cfg.model.training_recipe = semantic_recipe(cfg)
    from omegaconf import OmegaConf
    binding = build_binding(cfg.dataset_root, cfg.normalization_path, cfg.asset_root, cfg.text_embedding_cache_dir,
                            training_recipe=OmegaConf.to_container(cfg.model.training_recipe, resolve=True))
    if file_sha256(cfg.pretrained_fastwam_checkpoint) != RELEASE_SHA256:
        raise ValueError("Official complete FastWAM initialization checkpoint SHA256 differs")
    out = Path(cfg.output_dir).resolve()
    binding_path = out / "fastwam_binding.json"
    if cfg.resume:
        # Full state has no dataset metadata; bind it to the original run directory.
        old_run = Path(cfg.resume).resolve().parents[2]
        old_binding = json.loads((old_run / "fastwam_binding.json").read_text(encoding="utf-8"))
        if old_binding != binding:
            raise ValueError("Resume state belongs to another dataset/normalizer/asset recipe")
    if binding_path.exists() and json.loads(binding_path.read_text(encoding="utf-8")) != binding:
        raise ValueError("Refusing to mix an existing run with different data or assets")
    if args.command != "worker":
        # Fresh runs never overwrite checkpoints; resume is explicit.
        if out.exists() and any((out / "checkpoints").glob("**/*.pt")) and not cfg.resume:
            raise ValueError("Run already has checkpoints; use resume or a fresh output directory")
        out.mkdir(parents=True, exist_ok=True)
        # Native dataset_stats.json is a reserialized convenience copy. Bind the
        # byte-exact input separately so export never picks changed float formatting.
        normalized = out / "normalization.json"
        if Path(cfg.normalization_path).resolve() != normalized:
            shutil.copyfile(cfg.normalization_path, normalized)
        cfg.normalization_path = normalized.as_posix()
        cfg.model.normalization_path = normalized.as_posix()
        cfg.data.train.pretrained_norm_stats = normalized.as_posix()
        write_json(binding_path, binding)
        write_json(out / "training-hardware.json", hardware)
        from omegaconf import OmegaConf
        resolved = out / "launch_config.yaml"
        OmegaConf.save(cfg, resolved)
        command = [sys.executable, "-m", "accelerate.commands.launch", "--config_file",
                   str(ROOT / "scripts/accelerate_configs/accelerate_zero1_ds.yaml"),
                   "--num_processes", str(num_processes), str(Path(__file__).resolve()),
                   "worker", "--config", str(resolved)]
        if args.stop_after_steps is not None:
            command.extend(["--stop-after-steps", str(args.stop_after_steps)])
        if args.smoke_audit:
            command.append("--smoke-audit")
        env = os.environ.copy()
        env["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode

    # Only accelerate's explicit worker command enters the training routine.
    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "").lower() != "true":
        raise RuntimeError("Use train/resume to launch the existing DeepSpeed trainer; do not invoke worker directly")
    from fastwam.runtime import run_training, Wan22Trainer
    with bounded_training_invocation(Wan22Trainer, cfg, args.stop_after_steps, smoke_audit=args.smoke_audit):
        run_training(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
