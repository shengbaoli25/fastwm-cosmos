#!/usr/bin/env python3
"""Budgeted original FastWAM training followed by the frozen five-episode fit."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from fastwam.baseline.analysis import EPISODES, file_sha256, json_sha256, source_identity


def config_overrides(shared_repo, target_steps, output, pretrained_checkpoint, save_every=0):
    shared = Path(shared_repo).resolve()
    return ["task=forcevla_baseline", f"max_steps={target_steps}", f"save_every={save_every}", f"output_dir={Path(output).resolve()}",
            f"data.train.dataset_dirs=[{shared / 'data/forcevla_fastwam/wipe'}]",
            f"data.train.pretrained_norm_stats={shared / 'data/forcevla_fastwam/shared_stats/dataset_stats.json'}",
            f"data.train.text_embedding_cache_dir={shared / 'data/text_embeds_cache/forcevla'}",
            f"model.pretrained_fastwam_checkpoint={Path(pretrained_checkpoint).resolve()}",
            f"model.action_dit_pretrained_path={shared / 'checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt'}"]


def _comparable_config(config):
    value = dict(config)
    for key in ("output_dir", "save_every", "eval_every", "log_every"):
        value.pop(key, None)
    return value


def verify_preflight(preflight, config, root):
    report = json.loads(Path(preflight).read_text())
    if (report.get("status") != "complete" or report.get("actual_engine_updates") != 2 or
            report.get("microbatches") != 32 or not report.get("all_parameters_finite") or
            report.get("actual_adam_step_calls") != report.get("expected_adam_step_calls") or
            not report.get("adam_master_and_moments_finite") or report.get("optimizer_step_was_skipped") or
            not report.get("adam_state_steps") or any(step != 2 for step in report["adam_state_steps"])):
        raise ValueError("Fresh original FastWAM two-update GPU preflight is required")
    if _comparable_config(report["resolved_config_except_output"]) != _comparable_config(config):
        raise ValueError("Formal model/data/optimizer/schedule differs from the preflight")
    current = source_identity(root)
    if current != report["source_files"] or json_sha256(current) != report["source_sha256"]:
        raise ValueError("Source changed after the two-update preflight")
    if file_sha256(config["data"]["train"]["pretrained_norm_stats"]) != report["stats_sha256"]:
        raise ValueError("Shared statistics changed after the preflight")
    manifest = Path(config["data"]["train"]["dataset_dirs"][0]) / "conversion_manifest.json"
    if file_sha256(manifest) != report["data_conversion_manifest_sha256"]:
        raise ValueError("Converted dataset manifest changed after the preflight")
    checkpoint = config["model"]["pretrained_fastwam_checkpoint"]
    if (str(Path(checkpoint).resolve()) != report["pretrained_fastwam"]["path"] or
            config["model"]["pretrained_fastwam_sha256"] != report["pretrained_fastwam"]["sha256"]):
        raise ValueError("Full original policy initialization changed after the preflight")
    return report


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stop_process_group(process):
    # The launcher can exit while an Accelerate child still owns the GPU.
    # start_new_session pins this owned group's ID even after its leader exits.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)
    # poll() only describes the launcher; check the owned group separately.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def has_nonfinite_training_log(line):
    line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
    # Rich markup can remove the literal [train] tag from the log.
    if not re.search(r"\bstep=\d+/\d+\b", line):
        return False
    values = re.findall(r"\b(?:loss(?:_\w+)?|lr)=([^\s]+)", line)
    return any(not math.isfinite(float(value)) for value in values)


def run_process(command, log, seconds, environment, *, guard_loss=False):
    started = time.monotonic()
    log = Path(log)
    with log.open("wb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        print(json.dumps({"pid": process.pid, "process_group": process.pid, "log": str(log), "command": command}), flush=True)
        reason = None
        consumed = 0
        carry = ""
        while process.poll() is None:
            if time.monotonic() - started >= seconds:
                reason = "budget_timeout"
                break
            if guard_loss:
                with log.open("r", errors="replace") as reader:
                    reader.seek(consumed)
                    carry += reader.read()
                    consumed = reader.tell()
                lines = carry.split("\n")
                carry = lines.pop()
                for line in lines:
                    if has_nonfinite_training_log(line):
                        reason = "nonfinite_training_log"
                        break
                if reason:
                    break
            time.sleep(2)
        if reason:
            stop_process_group(process)
        else:
            process.wait()
            stop_process_group(process)
    if guard_loss and reason is None:
        for line in log.read_text(errors="replace").splitlines():
            if has_nonfinite_training_log(line):
                reason = "nonfinite_training_log"
                break
    return {"status": "complete" if process.returncode == 0 and reason is None else "failed",
            "reason": reason, "exit_code": process.returncode, "elapsed_seconds": time.monotonic() - started,
            "log_sha256": file_sha256(log)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-repo", required=True, help="Existing asset/data checkout; read-only")
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True, help="Pinned complete official LIBERO FastWAM release")
    parser.add_argument("--target-steps", type=int, default=2176)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--budget-seconds", type=float, default=18000)
    parser.add_argument("--train-seconds", type=float, default=16200)
    parser.add_argument("--eval-seconds", type=float, default=1200)
    parser.add_argument("--eval-mode", choices=("middle", "sequence"), default="sequence")
    args = parser.parse_args()
    if min(args.target_steps, args.train_seconds, args.eval_seconds, args.budget_seconds) <= 0:
        parser.error("Steps and budgets must be positive")
    if args.train_seconds + args.eval_seconds + 60 > args.budget_seconds:
        parser.error("Reserve at least 60 seconds inside the five-hour cap")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Formal run output must be fresh; do not silently reuse optimizer states")
    overrides = config_overrides(args.shared_repo, args.target_steps, output, args.pretrained_checkpoint, args.save_every)
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    config = OmegaConf.to_container(cfg, resolve=True)
    preflight = verify_preflight(args.preflight, config, root)
    output.mkdir(parents=True)
    plan = {"status": "initialized", "target_steps": args.target_steps, "microbatch": 1, "accumulation": 16,
            "planned_maximum_sample_draws": args.target_steps * 16, "training_frames": 34808,
            "maximum_sample_draws_per_anchor": args.target_steps * 16 / 34808,
            "one_pass_updates_ceil": math.ceil(34808 / 16), "episodes": 50, "split": "all_train",
            "selected_evaluation_episodes": list(EPISODES), "evaluation_mode": args.eval_mode,
            "budget_seconds": args.budget_seconds, "train_seconds": args.train_seconds, "eval_seconds": args.eval_seconds,
            "preflight_sha256": file_sha256(args.preflight), "source_sha256": preflight["source_sha256"],
            "config": config, "save_every": args.save_every,
            "note": "2176 is the official ceil(34808/16) update count for one anchor pass. Verify actual last-batch sync, observed draws and reader retries; a pass alone is not evidence of overfitting."}
    atomic_json(output / "plan.json", plan)
    environment = dict(os.environ)
    environment["DIFFSYNTH_MODEL_BASE_PATH"] = str(Path(args.shared_repo).resolve() / "checkpoints")
    environment["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    started = time.monotonic()
    train_command = ["accelerate", "launch", "--config_file", "scripts/accelerate_configs/accelerate_zero1_ds.yaml",
                     "--num_processes", "1", "scripts/train.py", *overrides]
    plan["train"] = run_process(train_command, output / "train.log", args.train_seconds, environment, guard_loss=True)
    checkpoint = output / "checkpoints/weights" / f"step_{args.target_steps:06d}.pt"
    state = output / "checkpoints/state" / f"step_{args.target_steps:06d}/trainer_state.json"
    if (plan["train"]["status"] != "complete" or not checkpoint.exists() or not state.exists() or
            json.loads(state.read_text())["global_step"] != args.target_steps):
        plan["status"] = "incomplete_training"
        atomic_json(output / "plan.json", plan)
        raise SystemExit(2)
    remaining = args.budget_seconds - (time.monotonic() - started) - 60
    if remaining <= 0:
        plan["status"] = "no_evaluation_budget"
        atomic_json(output / "plan.json", plan)
        raise SystemExit(2)
    eval_seconds = min(args.eval_seconds, remaining)
    eval_command = [sys.executable, "scripts/baseline/evaluate.py", "--config", str(output / "config.yaml"),
                    "--checkpoint", str(checkpoint), "--stats", str(output / "dataset_stats.json"),
                    "--output", str(output / "evaluation"), "--expected-step", str(args.target_steps),
                    "--mode", args.eval_mode, "--deadline-seconds", str(max(eval_seconds - 30, 1))]
    plan["eval"] = run_process(eval_command, output / "eval.log", eval_seconds, environment)
    metrics_path = output / "evaluation/metrics.json"
    evaluated = metrics_path.is_file() and json.loads(metrics_path.read_text()).get("status") == "complete"
    plan["status"] = "complete" if plan["eval"]["status"] == "complete" and evaluated else "incomplete_evaluation"
    plan["elapsed_seconds"] = time.monotonic() - started
    atomic_json(output / "plan.json", plan)
    print(json.dumps({"status": plan["status"], "output": str(output)}), flush=True)
    raise SystemExit(0 if plan["status"] == "complete" else 2)


if __name__ == "__main__":
    main()
