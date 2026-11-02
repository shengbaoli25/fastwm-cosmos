#!/usr/bin/env python3
"""Read-only Cosmos full-state inventory. Never imports torch or unpickles state.

Run under the intended training Python. JSON is written only to stdout; no files
are modified. File metadata is evidence of presence, not successful restoration.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

SMALL_LIMIT = 2 * 1024 * 1024


def small_text(path: Path) -> str:
    if path.stat().st_size > SMALL_LIMIT:
        raise ValueError(f"Refusing text larger than {SMALL_LIMIT} bytes: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def file_info(path: Path, archive_index: bool = False) -> dict:
    result = {"path": str(path), "exists": path.is_file(), "symlink": path.is_symlink()}
    if not result["exists"]:
        return result
    stat = path.stat()
    result.update(bytes=stat.st_size, mtime_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat())
    if archive_index and stat.st_size and not path.is_symlink():
        # ZipFile reads only the central directory here, never tensor members.
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                result["archive_index"] = {
                    "entries": len(names),
                    "has_data_pickle": any(name.endswith("/data.pkl") for name in names),
                    "tensor_storage_entries": sum("/data/" in name for name in names),
                }
        except (OSError, zipfile.BadZipFile) as error:
            result["archive_index_error"] = str(error)
    return result


def configuration(run: Path) -> dict:
    path = run / "config.yaml"
    result = file_info(path)
    if not path.is_file():
        return result
    text = small_text(path)
    result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Return exact bounded text: no YAML parser, interpolation or imports execute.
    # The caller can inspect original scheduler/model/data semantics verbatim.
    result["yaml"] = text
    result["top_level_scalar_lines"] = [
        line for line in text.splitlines()
        if re.match(r"^(batch_size|gradient_accumulation_steps|max_steps|seed|num_workers|"
                    r"mixed_precision|learning_rate|weight_decay|lr_scheduler_type|save_every|"
                    r"output_dir|resume|gradient_clip_val|ablation_num_processes)\s*:", line)
    ]
    return result


def state_inventory(run: Path, state: Path, world_size: int, archive_index: bool) -> dict:
    step = int(state.name.removeprefix("step_"))
    result = {"step": step, "path": str(state), "issues": [], "warnings": []}
    metadata = state / "trainer_state.json"
    try:
        progress = json.loads(small_text(metadata))
        result["trainer_state"] = progress
        if (not isinstance(progress, dict)
                or any(type(progress.get(key)) is not int or progress[key] < 0
                       for key in ("global_step", "epoch", "batch_in_epoch"))
                or progress["global_step"] != step):
            result["issues"].append("trainer_state progress does not match the directory")
    except (OSError, ValueError, TypeError) as error:
        result["issues"].append(f"trainer_state: {error}")
    try:
        tag = small_text(state / "latest").strip()
        result["latest_tag"] = tag
        if tag != "pytorch_model":
            result["issues"].append("Expected DeepSpeed latest tag pytorch_model")
    except (OSError, ValueError) as error:
        result["issues"].append(f"latest: {error}")

    model = state / "pytorch_model"
    expected_optimizer = {f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt" for rank in range(world_size)}
    actual_optimizer = {path.name for path in model.glob("*optim_states.pt")}
    if actual_optimizer != expected_optimizer:
        result["issues"].append("Optimizer partition names differ from expected world size/BF16 ZeRO1")
    expected_rng = {f"random_states_{rank}.pkl" for rank in range(world_size)}
    if {path.name for path in state.glob("random_states_*.pkl")} != expected_rng:
        result["issues"].append("RNG rank files differ from expected world size")
    required = [state / "scheduler.bin", model / "mp_rank_00_model_states.pt",
                run / f"checkpoints/weights/step_{step:06d}.pt"]
    required += [state / name for name in sorted(expected_rng)]
    required += [model / name for name in sorted(expected_optimizer)]
    files = [file_info(path, archive_index) for path in required]
    result["files"] = files
    for info in files:
        if not info["exists"] or info.get("bytes", 0) == 0 or info["symlink"]:
            result["issues"].append(f"Missing/empty/symlink required file: {info['path']}")
        if archive_index and (info.get("archive_index_error")
                              or not info.get("archive_index", {}).get("has_data_pickle")):
            result["issues"].append(f"Torch archive index not established: {info['path']}")
    result["sampler"] = file_info(state / "sampler.bin", archive_index)
    if not result["sampler"]["exists"]:
        result["warnings"].append("sampler.bin absent; verify actual saved loader/sampler implementation")
    result["required_file_bytes"] = sum(info.get("bytes", 0) for info in files)
    result["metadata_candidate"] = not result["issues"]
    return result


def log_evidence(path: Path, run: Path, states: list[dict]) -> dict:
    text = small_text(path)
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    compact = re.sub(r"trainer\.py:\d+", "", re.sub(r"\s+", "", text))
    completed = []
    for state in states:
        step = state["step"]
        weights_path = run / f"checkpoints/weights/step_{step:06d}.pt"
        state_path = run / f"checkpoints/state/step_{step:06d}"
        marker = f"step={step}weights={weights_path}state={state_path}"
        state["completed_save_log"] = marker in compact
        if state["completed_save_log"]:
            completed.append(step)
    records = re.findall(r"epoch=(\d+)\s+step=(\d+)/(\d+)", text)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "completed_save_steps": completed,
            "last_training_record": dict(zip(("epoch", "step", "original_target"), map(int, records[-1]))) if records else None}


def source_evidence(root: Path) -> dict:
    result = {"path": str(root), "files": {}}
    for command in (["git", "rev-parse", "HEAD"], ["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=no"]):
        try:
            process = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=15, check=False)
            result["git_head" if command[1] == "rev-parse" else "git_tracked_changes"] = {
                "exit_code": process.returncode, "stdout": process.stdout.strip(), "stderr": process.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            result.setdefault("git_errors", []).append(str(error))
    for relative in ("src/fastwam/trainer.py", "src/fastwam/runtime.py",
                     "src/fastwam/utils/samplers.py", "src/fastwam/cosmos/assets.py",
                     "src/fastwam/cosmos/cache.py", "src/fastwam/cosmos/native.py",
                     "src/fastwam/cosmos/noaction.py", "src/fastwam/cosmos/state.py",
                     "scripts/cosmos/train.py", "scripts/cosmos/train_realman_cached.py",
                     "scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml",
                     "scripts/accelerate_configs/accelerate_zero1_ds.yaml", "scripts/ds_configs/ds_zero1_config.json"):
        path = root / relative
        info = file_info(path)
        if path.is_file():
            text = small_text(path)
            info["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if relative.endswith("trainer.py"):
                tree = ast.parse(text)
                lines = text.splitlines()
                names = {"load_training_state", "_resume_or_load_checkpoint", "_build_scheduler", "_build_loader"}
                info["resume_methods"] = {
                    node.name: {"first_line": node.lineno, "source": "\n".join(lines[node.lineno - 1:node.end_lineno])}
                    for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names}
            elif path.suffix in (".yaml", ".json"):
                info["text"] = text
        result["files"][relative] = info
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oldrun", type=Path, help="Original runs/long/training directory")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--stop-at-step", type=int, default=30000)
    parser.add_argument("--log", type=Path, help="Original training.log; skipped unless supplied")
    parser.add_argument("--source-root", type=Path, help="Original source checkout; no automatic filesystem search")
    parser.add_argument("--archive-index", action="store_true", help="Inspect ZIP central directories only; no tensor reads or unpickling")
    args = parser.parse_args(argv)
    if args.world_size <= 0 or args.stop_at_step <= 0 or not args.oldrun.is_dir():
        parser.error("oldrun must exist and world size/stop step must be positive")
    run = args.oldrun.resolve()
    states = [state_inventory(run, path, args.world_size, args.archive_index)
              for path in sorted((run / "checkpoints/state").glob("step_*"), reverse=True)
              if path.is_dir() and re.fullmatch(r"step_\d{6}", path.name)]
    report = {"schema": "cosmos-readonly-resume-audit-v1", "oldrun": str(run),
              "observed_utc": datetime.now(timezone.utc).isoformat(),
              "world_size_expected": args.world_size, "stop_at_step": args.stop_at_step,
              "inspection": "metadata and optional ZIP index only; no torch import/load, no tensor validation, no mutations",
              "config": configuration(run), "states": states}
    if args.log:
        report["log"] = log_evidence(args.log.resolve(), run, states)
    if args.source_root:
        report["source"] = source_evidence(args.source_root.resolve())
    versions = {}
    for package in ("torch", "torchvision", "accelerate", "deepspeed", "omegaconf", "hydra-core", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report["audit_execution_environment_not_original_launch"] = {
        "python": sys.executable, "python_version": sys.version, "packages": versions,
        "distributed_environment": {key: os.environ[key] for key in
            ("TRAIN_JOB_ID", "CUDA_VISIBLE_DEVICES", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
             "ACCELERATE_USE_DEEPSPEED", "NCCL_NVLS_ENABLE") if key in os.environ}}
    candidates = [state for state in states if state["metadata_candidate"] and state["step"] < args.stop_at_step
                  and (not args.log or state.get("completed_save_log"))]
    report["latest_metadata_candidate"] = candidates[0]["path"] if candidates else None
    report["ready_to_resume"] = False
    report["remaining_gpu_checks"] = [
        "Confirm original job is stopped and original runtime/command/source/dataset identities",
        "Restore full model, all optimizer partitions, scheduler and RNG on four ranks",
        "Verify trainer and engine step and exact sharded dataloader progress before first optimizer update",
        "Keep original scheduler recipe; set total stopping cap only after restore",
    ]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
