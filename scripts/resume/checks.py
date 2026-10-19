"""Checkpoint and platform-job checks shared by the resume controller and ranks.

Archive checks establish readable structure and the trainer's completed-save log,
not tensor correctness. Each fresh GPU rank must still load and verify its state.
"""
import hashlib
import json
import math
import re
from pathlib import Path
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete_state(output, log, *, final=False, max_steps=30000, save_every=5000):
    output, log = Path(output), Path(log)
    text = log.read_text(errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    # Rich strips bracket tags and can wrap paths/source locations across lines.
    compact = re.sub(r"\s+", "", text)
    compact = re.sub(r"trainer\.py:\d+", "", compact)
    saved = set()
    for step in sorted(set(range(save_every, max_steps+1, save_every)) | {max_steps}):
        marker = (f"step={step}weights={output}/checkpoints/weights/step_{step:06d}.pt"
                  f"state={output}/checkpoints/state/step_{step:06d}")
        if marker in compact:
            saved.add(step)
    # At 30000 the native trainer overwrites the periodic checkpoint once more.
    # Require the final save's log, not merely the earlier periodic-save marker.
    if f"max_stepsreachedstep={max_steps}weights=" not in compact:
        saved.discard(max_steps)
    if not final and max_steps in saved:
        raise ValueError("This run already logged its final save; verify completion instead of resuming")
    rejected = []
    for state in sorted((output / "checkpoints/state").glob("step_[0-9]*"), reverse=True):
        try:
            if not re.fullmatch(r"step_\d{6}", state.name):
                continue
            step = int(state.name[5:])
            if step not in saved or (step % save_every and step != max_steps) or step <= 0 or step > (max_steps if final else max_steps-1):
                continue
            data = json.loads((state / "trainer_state.json").read_text())
            if any(type(data.get(k)) is not int or data[k] < 0
                   for k in ("global_step", "epoch", "batch_in_epoch")) or data["global_step"] != step:
                raise ValueError("Invalid trainer progress")
            tag = (state / "latest").read_text().strip()
            if tag != "pytorch_model":
                raise ValueError("Unexpected DeepSpeed latest tag")
            model = state / tag
            files = [state / "scheduler.bin", model / "mp_rank_00_model_states.pt",
                     output / f"checkpoints/weights/step_{step:06d}.pt"]
            files += [state / f"random_states_{r}.pkl" for r in range(4)]
            files += [model / f"bf16_zero_pp_rank_{r}_mp_rank_00_optim_states.pt" for r in range(4)]
            if {p.name for p in model.glob("*optim_states.pt")} != {
                    f"bf16_zero_pp_rank_{r}_mp_rank_00_optim_states.pt" for r in range(4)}:
                raise ValueError("Exactly four ZeRO partitions required")
            for p in files:
                if p.is_symlink() or not p.is_file() or p.stat().st_size <= 0:
                    raise ValueError(f"Missing state file: {p.name}")
                with zipfile.ZipFile(p) as z:
                    names = z.namelist()
                    if not any(n.endswith("/data.pkl") for n in names) or not any("/data/" in n for n in names):
                        # Scheduler contains only Python scalars and no tensor data.
                        if p.name != "scheduler.bin" or not any(n.endswith("/data.pkl") for n in names):
                            raise ValueError(f"Invalid torch archive: {p.name}")
            return state, step
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as e:
            rejected.append(f"{state.name}: {e}")
    raise ValueError("No completed four-rank full-state checkpoint; " + "; ".join(rejected))


def stopped_evidence(path, prior):
    value = json.loads(Path(path).read_text())
    if (value.get("prior_output") != str(prior) or value.get("status") not in ("STOPPED", "FAILED", "SUCCEEDED")
            or value.get("source") != "platform_ui" or not value.get("resource_id")
            or not value.get("observed_at_utc") or not value.get("evidence_file")):
        raise ValueError("Observed platform termination evidence required; stale running status is insufficient")
    evidence = Path(path).parent / value["evidence_file"]
    if not evidence.is_file() or sha(evidence) != value.get("evidence_sha256"):
        raise ValueError("Platform termination evidence file changed")
    return value


def platform_identity(env):
    job = env.get("TRAIN_JOB_ID", "")
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{3,127}", job)
            or env.get("PET_NNODES", "1") != "1" or env.get("PET_NODE_RANK", "0") != "0"):
        raise ValueError("Foreground single-node platform distributed training job required")
    return job


def launch_environment(env, source):
    env = dict(env)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT"):
        env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["PYTHONPATH"] = ":".join((str(source / "src"), str(source), env.get("PYTHONPATH", "")))
    return env


def force_deadline(path, now):
    b = json.loads(Path(path).read_text())
    # Original, actually observed v9 launcher. A new job cannot re-anchor it.
    if (b.get("launcher_started_unix") != 1791795479.8201158
            or b.get("absolute_deadline_unix") != 1791975479.8201158
            or b.get("duration_seconds") != 180000
            or b.get("budget_anchor") != "formal_launcher_start"
            or b.get("execution_identity") != {"kind": "interactive_modeling",
               "allocation_id": "ab5d6993-795b-4bf7-9084-e59f089e7788"}
            or not math.isfinite(now) or now >= b["absolute_deadline_unix"] - 60):
        raise ValueError("Original ForceWM 50-hour authorization must not reset on resume")
    return b["absolute_deadline_unix"]


def verify_restored(trainer, state, step, helpers):
    import torch
    saved = json.loads((state / "trainer_state.json").read_text())
    actual = {"global_step": trainer.global_step, "epoch": trainer.epoch,
              "batch_in_epoch": trainer.batch_in_epoch}
    if saved != actual or trainer.model.global_steps != step:
        raise ValueError("Trainer/engine/dataloader restore mismatch")
    scheduler = helpers.inspect_scheduler(trainer.scheduler)
    if scheduler != {"last_epoch": step, "warmup_milestones": [1500], "cosine_steps": 28500}:
        raise ValueError("Full 30000-step scheduler was not restored")
    restored_scheduler = trainer.scheduler.state_dict()
    saved_scheduler = torch.load(state / "scheduler.bin", map_location="cpu", weights_only=False)
    if restored_scheduler != saved_scheduler:
        raise ValueError("Scheduler state differs from the saved state")
    adam = helpers.unwrap_optimizer(trainer.optimizer)
    details = helpers.inspect_adam(adam)
    if (not details["adam_state_steps"] or set(details["adam_state_steps"]) != {float(step)}
            or not details["adam_master_and_moments_finite"]):
        raise ValueError("Adam state is incomplete, nonfinite, or at another step")
    if not all(helpers._finite(p) for p in trainer.model.parameters()):
        raise ValueError("Restored model contains nonfinite parameters")
    from continuation import verify_rng_and_sampler
    rng = verify_rng_and_sampler(trainer, state, saved)
    return {"rng_and_sampler": rng, "trainer": actual, "engine_step": trainer.model.global_steps,
            "scheduler": scheduler, "adam": details}



def verify_code(source):
    """The resume scripts may change; the running training implementation may not."""
    manifest = json.loads((source / "scripts/resume/frozen_code.json").read_text())
    for name, digest in manifest["files"].items():
        if sha(source / name) != digest:
            raise ValueError("Frozen training code changed: " + name)
    return manifest


def prior_resource(prior, initial):
    path = prior / "resume_origin.json"
    return json.loads(path.read_text())["job"] if path.exists() else initial


def final_check(output, log, job):
    state, step = complete_state(output, log, final=True)
    if step != 30000:
        raise ValueError("Final completed 30000 save absent")
    for rank in range(4):
        r = json.loads((output / "origins" / f"rank_{rank:03d}.json").read_text())
        if (r.get("rank") != rank or r.get("job") != job
                or r.get("status") != "complete" or r.get("global_step") != 30000):
            raise ValueError("All four completed resume ranks required")
    return {"status": "completed_30000", "state": str(state), "job": job}


def storage_estimate(state, output, step):
    import shutil
    size = sum(p.stat().st_size for p in state.rglob("*") if p.is_file())
    size += (state.parents[2] / f"checkpoints/weights/step_{step:06d}.pt").stat().st_size
    required = int(size * ((30000-step)//5000) * 1.1)
    free = shutil.disk_usage(output.parent).free
    if free < required:
        raise ValueError("Insufficient disk for remaining full-state checkpoints")
    return {"free_bytes": free, "estimated_required_bytes": required}


def budget_estimate(log, step, deadline, now):
    speeds = re.findall(r"speed=([0-9.]+)\s+step/s", Path(log).read_text(errors="replace"))
    speed = float(speeds[-1]) if speeds else 0.
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Actual previous training speed required")
    # Planning allowance, not a claim that full save/evaluation time was measured.
    required = (30000-step) / speed * 1.05 + 600 + 1800 + 3600 + 60
    if required > deadline-now:
        raise ValueError("Remaining training/save/eval estimate exceeds the original authorization")
    return {"previous_logged_steps_per_second": speed, "estimated_required_seconds": required,
            "available_seconds": deadline-now, "save_eval_seconds_are_allowances": True}
