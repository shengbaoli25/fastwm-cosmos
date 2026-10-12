#!/usr/bin/env python3
"""Plan, or explicitly run two disposable cached-training optimizer updates.

Run execution through the prepared four/eight-process Accelerate ZeRO1 config.
The stock trainer builds the actual optimizer, sampler and 30000-step scheduler.
No checkpoint is saved and this probe is never continued as the formal run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from functools import wraps
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _training_plan_module():
    spec = importlib.util.spec_from_file_location("cached_training_plan", Path(__file__).with_name("prepare_cached_training.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_inputs(args):
    from omegaconf import OmegaConf

    plan = json.loads(Path(args.plan).read_text()) if args.plan else None
    cfg = OmegaConf.create(plan["config"]) if plan else OmegaConf.load(args.config)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    _training_plan_module().validate_config(resolved)
    world_size = args.num_processes or (plan["world_size"] if plan else 4)
    if world_size not in (4, 8) or (plan and world_size != plan["world_size"]):
        raise ValueError("Preflight world size must match the four/eight-process training plan")
    identity = resolved["data"]["train"].get("expected_identity_sha256")
    bound = _valid_sha256(identity)
    if plan and bound and plan.get("cache_check", {}).get("identity_sha256") != identity:
        raise ValueError("Plan cache identity differs from the resolved training config")
    return cfg, resolved, plan, world_size, bound


def metadata(args):
    _, config, _, world_size, bound = load_inputs(args)
    return {"status": "preflight_planned", "execute_preflight": False,
            "formal_training_started": False, "diagnostic_updates": 2,
            "world_size": world_size, "batch_per_rank": 16, "gradient_accumulation_steps": 1,
            "effective_batch": world_size * 16, "planned_schedule_steps": config["max_steps"],
            "warmup_updates": 1500, "cache_identity_bound": bound,
            "cache_identity_sha256": config["data"]["train"].get("expected_identity_sha256"),
            "checkpoint_saved": False, "output": str(Path(args.output).resolve()),
            "execution_requirement": "Use the matching Accelerate ZeRO1 launch and add --execute-preflight",
            "persistence": "Two disposable updates; formal training starts fresh with the unchanged 30000-step config"}


def ancestor_launcher_config():
    """Extract only the launch config argument, without exposing command lines."""
    pid = os.getppid()
    for _ in range(8):
        if pid <= 1:
            break
        directory = Path("/proc") / str(pid)
        try:
            argv = directory.joinpath("cmdline").read_bytes().decode().split("\0")
            if ("launch" in argv or "accelerate.commands.launch" in argv
                    or any(argument.endswith("/accelerate/commands/launch.py") for argument in argv)):
                for index, argument in enumerate(argv):
                    value = (argv[index + 1] if argument == "--config_file" and index + 1 < len(argv)
                             else argument.split("=", 1)[1] if argument.startswith("--config_file=") else None)
                    if value:
                        path = Path(value)
                        return (path if path.is_absolute() else directory.joinpath("cwd").resolve() / path).resolve()
            parent = next(line for line in directory.joinpath("status").read_text().splitlines()
                          if line.startswith("PPid:"))
            pid = int(parent.split()[1])
        except (OSError, StopIteration, UnicodeDecodeError):
            break
    return None


def launcher_identity(world_size, explicit_path=None):
    from omegaconf import OmegaConf

    detected = ancestor_launcher_config()
    declared = Path(explicit_path).resolve() if explicit_path else None
    if detected is not None and declared is not None and detected != declared:
        raise ValueError("Declared launcher YAML differs from the actual ancestor Accelerate command")
    expected = ROOT / "scripts/accelerate_configs" / f"accelerate_zero1_ds_{world_size}gpu.yaml"
    path = detected or declared or expected
    if path != expected.resolve():
        raise ValueError("Use the dedicated launcher YAML matching this cached preflight")
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if (config["num_processes"] != world_size or config["num_machines"] != 1
            or config["machine_rank"] != 0 or config["distributed_type"] != "DEEPSPEED"):
        raise ValueError("Dedicated launcher topology differs from the exact preflight")
    ds_path = ROOT / config["deepspeed_config"]["deepspeed_config_file"]
    return {"path": str(path), "sha256": _sha256(path),
            "evidence": "ancestor_accelerate_command" if detected else "explicit_cli" if declared else "expected_dedicated_config",
            "deepspeed_config_path": str(ds_path), "deepspeed_config_sha256": _sha256(ds_path)}


def gradient_clipping_metadata(configured, engine_value, ds_config):
    return {"configured_max_grad_norm": float(configured), "effective_gradient_clipping": float(engine_value),
            "deepspeed_gradient_clipping_field_present": "gradient_clipping" in ds_config,
            "deepspeed_gradient_clipping_field": ds_config.get("gradient_clipping"),
            "recipe_change_applied": False,
            "note": "The original ZeRO1 JSON has no clipping field; cfg max_grad_norm=1 does not establish engine clipping=1"}


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _valid_launcher(launch):
    return (isinstance(launch, dict) and _valid_sha256(launch.get("sha256"))
            and _valid_sha256(launch.get("deepspeed_config_sha256"))
            and isinstance(launch.get("path"), str) and bool(launch["path"])
            and isinstance(launch.get("deepspeed_config_path"), str) and bool(launch["deepspeed_config_path"])
            and launch.get("evidence") in ("ancestor_accelerate_command", "explicit_cli", "expected_dedicated_config"))


def _valid_sample_indices(indices):
    return (isinstance(indices, list) and len(indices) == 16
            and all(type(index) is int and 0 <= index < 34808 for index in indices)
            and len(set(indices)) == 16)


def _finite_update_evidence(rank):
    losses, updates = rank.get("losses"), rank.get("update_timings")
    if not isinstance(losses, list) or not isinstance(updates, list) or len(losses) != 2 or len(updates) != 2:
        return False
    for index, (loss, update) in enumerate(zip(losses, updates), 1):
        if (not isinstance(loss, dict) or not _finite_number(loss.get("loss"))
                or not all(_finite_number(value) for value in loss.values())
                or not isinstance(update, dict) or update.get("update") != index
                or update.get("optimizer_step_was_skipped") is not False
                or update.get("batch_size") != 16 or update.get("microbatches") != 1
                or not _valid_sample_indices(update.get("sample_indices"))
                or not _finite_number(update.get("global_loss"))
                or not _finite_number(update.get("gradient_norm")) or update["gradient_norm"] < 0
                or not _finite_number(update.get("seconds")) or update["seconds"] < 0
                or not isinstance(update.get("global_components"), dict)
                or not all(_finite_number(value) for value in update["global_components"].values())):
            return False
    return True


def ranks_complete(ranks, world_size, identity, failure_paths=(), launcher=None):
    if (failure_paths or type(world_size) is not int or world_size not in (4, 8) or not _valid_sha256(identity)
            or not isinstance(ranks, list) or len(ranks) != world_size
            or any(not isinstance(rank, dict) or type(rank.get("rank")) is not int for rank in ranks)
            or sorted(rank["rank"] for rank in ranks) != list(range(world_size))):
        return False
    launcher = ranks[0].get("launcher_config") if launcher is None else launcher
    if not _valid_launcher(launcher):
        return False
    for rank in ranks:
        device = rank.get("device")
        if (not isinstance(device, dict)
                or any(type(device.get(key)) is not int or device[key] != rank["rank"] for key in (
                    "env_LOCAL_RANK", "accelerator_local_process_index", "current_cuda_device"))
                or device.get("chosen_factory_device") != f"cuda:{rank['rank']}"
                or not isinstance(device.get("cuda_device_name"), str) or not device["cuda_device_name"]
                or device.get("CUDA_VISIBLE_DEVICES") != ranks[0].get("device", {}).get("CUDA_VISIBLE_DEVICES")
                or rank.get("launcher_config") != launcher or not _finite_update_evidence(rank)):
            return False
        steps = rank.get("adam_state_steps")
        if (rank.get("status") != "complete" or rank.get("world_size") != world_size
                or rank.get("cache_identity_sha256") != identity
                or rank.get("actual_engine_updates") != 2 or rank.get("trainer_updates") != 2
                or rank.get("planned_schedule_steps") != 30000 or rank.get("batch_per_rank") != 16
                or rank.get("microbatches") != 2 or rank.get("optimizer_step_was_skipped") is not False
                or rank.get("all_parameters_finite") is not True or rank.get("adam_master_and_moments_finite") is not True
                or not isinstance(steps, list) or not steps or any(not _finite_number(step) or step != 2 for step in steps)
                or type(rank.get("actual_adam_step_calls")) is not int
                or type(rank.get("expected_adam_step_calls")) is not int
                or rank.get("actual_adam_step_calls", 0) != rank.get("expected_adam_step_calls", -1)
                or rank.get("actual_adam_step_calls", 0) <= 0
                or rank.get("scheduler") != {"last_epoch": 2, "warmup_milestones": [1500], "cosine_steps": 28500}):
            return False
    # Disjoint per distributed update; reuse in a later update is permitted.
    return all(len({sample_index for rank in ranks
                    for sample_index in rank["update_timings"][update_index]["sample_indices"]}) == world_size * 16
               for update_index in range(2))


def receipt_complete(receipt, failure_paths=()):
    return (isinstance(receipt, dict) and receipt.get("status") == "complete" and receipt.get("world_size") in (4, 8)
            and receipt.get("effective_batch") == receipt["world_size"] * 16
            and receipt.get("planned_schedule_steps") == 30000 and receipt.get("actual_engine_updates") == 2
            and receipt.get("formal_training_started") is False and receipt.get("checkpoint_saved") is False
            and _valid_launcher(receipt.get("launcher_config"))
            and ranks_complete(receipt.get("ranks", []), receipt["world_size"],
                               receipt.get("cache_identity_sha256"), failure_paths, receipt.get("launcher_config")))


def preflight_success(exit_code, receipt, failure_paths=()):
    """External acceptance also requires the entire distributed command to exit 0."""
    return type(exit_code) is int and exit_code == 0 and receipt_complete(receipt, failure_paths)


def unwrap_optimizer(optimizer):
    visited = set()
    while hasattr(optimizer, "optimizer") and id(optimizer) not in visited:
        visited.add(id(optimizer))
        optimizer = optimizer.optimizer
    return optimizer


def run_two_updates(trainer, *, synchronize, clock=time.monotonic):
    """Bound only this loop; retain the stock trainer's full scheduler horizon."""
    import torch

    if trainer.max_steps != 30000 or trainer.global_step != 0:
        raise ValueError("Probe requires a fresh trainer with the complete 30000-update schedule")
    accelerator = trainer.accelerator
    iterator = iter(trainer.train_loader)
    updates, losses = [], []
    for _ in range(2):
        synchronize()
        started = clock()
        sample = next(iterator)
        if ("video" in sample or tuple(sample["video_latents"].shape) != (16, 48, 3, 14, 28)
                or tuple(sample["action"].shape) != (16, 32, 7)
                or tuple(sample["proprio"].shape) != (16, 32, 8)):
            raise ValueError("Each rank must consume the exact feature-only batch16 cache contract")
        trainer.batch_in_epoch += 1
        with accelerator.accumulate(trainer.model):
            train_model = (trainer.model if hasattr(trainer.model, "training_loss")
                           else accelerator.unwrap_model(trainer.model))
            with accelerator.autocast():
                loss, components = train_model.training_loss(sample)
            if not bool(torch.isfinite(loss.detach()).all()):
                raise FloatingPointError("Nonfinite original FastWAM cached training loss")
            accelerator.backward(loss)
            if not accelerator.sync_gradients:
                raise RuntimeError("The exact accumulation1 recipe must synchronize every microbatch")
            grad_norm = accelerator.clip_grad_norm_(trainer.model.parameters(), trainer.max_grad_norm)
            trainer.optimizer.step()
            skipped = bool(accelerator.optimizer_step_was_skipped)
            if not skipped:
                trainer.scheduler.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer.global_step += 1
            # Preserve the stock loop's cross-rank metric synchronization.
            global_loss = float(accelerator.gather(loss.detach().float().reshape(1)).mean().item())
            global_components = {key: float(accelerator.gather(
                torch.tensor(float(value), device=loss.device).reshape(1)).mean().item())
                for key, value in components.items()}
            norm = float(accelerator.gather(torch.as_tensor(grad_norm, device=loss.device).reshape(1)).mean().item())
        synchronize()
        losses.append({"loss": float(loss.detach().item()), **components})
        updates.append({"update": trainer.global_step, "seconds": clock() - started,
                        "batch_size": 16, "microbatches": 1, "global_loss": global_loss,
                        "global_components": global_components, "gradient_norm": norm,
                        "optimizer_step_was_skipped": skipped,
                        "lr": [float(group["lr"]) for group in trainer.optimizer.param_groups],
                        "sample_indices": sample["sample_index"].detach().cpu().tolist()})
    if trainer.max_steps != 30000:
        raise RuntimeError("The preflight shortened the formal scheduler horizon")
    return losses, updates


def _finite(tensor, chunk_size=2_000_000):
    import torch
    flat = tensor.detach().reshape(-1)
    return all(bool(torch.isfinite(flat[start:start + chunk_size]).all())
               for start in range(0, flat.numel(), chunk_size))


def inspect_adam(adam):
    steps, local_bytes, finite, dtypes = [], 0, True, set()
    for group in adam.param_groups:
        for parameter in group["params"]:
            state = adam.state[parameter]
            if any(key not in state for key in ("step", "exp_avg", "exp_avg_sq")):
                raise RuntimeError("An actual ZeRO Adam partition has no moments")
            steps.append(float(state["step"]))
            for tensor in (parameter, state["exp_avg"], state["exp_avg_sq"]):
                finite = _finite(tensor) and finite
                local_bytes += tensor.numel() * tensor.element_size()
                dtypes.add(str(tensor.dtype))
    return {"adam_state_steps": steps, "adam_master_and_moments_finite": finite,
            "local_adam_master_and_moments_bytes": local_bytes,
            "adam_master_and_moments_dtypes": sorted(dtypes)}


def inspect_scheduler(scheduler):
    visited = set()
    while hasattr(scheduler, "scheduler") and id(scheduler) not in visited:
        visited.add(id(scheduler))
        scheduler = scheduler.scheduler
    return {"last_epoch": int(scheduler.last_epoch),
            "warmup_milestones": list(scheduler._milestones),
            "cosine_steps": int(scheduler._schedulers[-1].T_max)}


def _unique_state_bytes(state):
    storages = {}
    for tensor in state.values():
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def io_probe(output, mib):
    if mib == 0:
        return {"status": "not_measured", "bytes_per_second": None}
    path = output / ".bounded-io-probe.tmp"
    block = b"\0" * 1024 * 1024
    started = time.monotonic()
    try:
        with path.open("xb") as stream:
            for _ in range(mib):
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        elapsed = time.monotonic() - started
        return {"status": "measured_bounded_write", "bytes": mib * 1024**2,
                "seconds": elapsed, "bytes_per_second": mib * 1024**2 / elapsed,
                "note": "Small fsync write; full checkpoint serialization and sustained storage bandwidth remain unmeasured"}
    finally:
        path.unlink(missing_ok=True)


def budget_report(ranks, storage, *, remaining_seconds=None, probe_elapsed_seconds=0.):
    warm_seconds = max(rank["update_timings"][-1]["seconds"] for rank in ranks)
    cold_extra = max(max(rank["update_timings"][0]["seconds"] for rank in ranks) - warm_seconds, 0.)
    startup_seconds = max(rank["startup_seconds"] for rank in ranks)
    policy_bytes = max(rank["policy_checkpoint_bytes_estimate"] for rank in ranks)
    module_bytes = max(rank["state_module_bytes_estimate"] for rank in ranks)
    optimizer_bytes = sum(rank["local_adam_master_and_moments_bytes"] for rank in ranks)
    one_checkpoint = policy_bytes + module_bytes + optimizer_bytes
    bandwidth = storage.get("bytes_per_second")
    # Stock trainer saves at each of six periodic steps, then rewrites step30000.
    save_seconds = one_checkpoint * 7 / bandwidth if bandwidth else None
    training_seconds = warm_seconds * 30000 + cold_extra
    estimated = startup_seconds + training_seconds + save_seconds if save_seconds is not None else None
    after_preflight = remaining_seconds - probe_elapsed_seconds if remaining_seconds is not None else None
    free_bytes = storage.get("free_bytes")
    return {"startup_seconds_measured": startup_seconds, "warm_update_seconds_slowest_rank": warm_seconds,
            "cold_first_update_extra_seconds": cold_extra, "training_seconds_estimate": training_seconds,
            "one_weights_and_state_checkpoint_bytes_estimate": one_checkpoint,
            "retained_six_checkpoints_bytes_estimate": one_checkpoint * 6,
            "checkpoint_write_calls": 7, "checkpoint_seconds_estimate": save_seconds,
            "startup_training_and_save_seconds_estimate": estimated,
            "remaining_budget_seconds": remaining_seconds,
            "preflight_elapsed_seconds_measured": probe_elapsed_seconds,
            "remaining_budget_after_preflight_seconds": after_preflight,
            "fit_within_remaining_budget_estimate": (estimated <= after_preflight
                if estimated is not None and after_preflight is not None else None),
            "retained_checkpoints_fit_free_disk_estimate": (one_checkpoint * 6 <= free_bytes
                if free_bytes is not None else None),
            "projection_is_verified": False,
            "note": "Two updates and a bounded IO sample do not establish sustained 30000-step throughput or full save time"}


def execute(args):
    started = time.monotonic()
    args._owns_output = False
    cfg, formal_config, plan, world_size, bound = load_inputs(args)
    if not bound:
        raise ValueError("Execution requires a verified cache identity bound in the resolved config")
    if int(os.environ.get("WORLD_SIZE", "1")) != world_size:
        raise RuntimeError("Launch this preflight through the matching four/eight-rank Accelerate ZeRO1 config")
    launch = launcher_identity(world_size, getattr(args, "launcher_config", None))
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    if plan:
        for key in ("DIFFSYNTH_MODEL_BASE_PATH", "TOKENIZERS_PARALLELISM"):
            if key in plan.get("environment", {}):
                os.environ[key] = str(plan["environment"][key])

    import torch
    from accelerate import Accelerator
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam import runtime
    from fastwam.baseline.analysis import json_sha256, source_identity
    from fastwam.trainer import Wan22Trainer
    from fastwam.utils import misc
    from fastwam.utils.pytorch_utils import set_global_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Explicit hardware preflight requires CUDA")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.reset_peak_memory_stats()
    coordinator = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16",
                              step_scheduler_with_optimizer=False)
    if coordinator.num_processes != world_size or coordinator.state.deepspeed_plugin is None:
        raise RuntimeError("The exact distributed DeepSpeed launch is required")
    output = Path(args.output).resolve()
    if coordinator.is_main_process:
        output.mkdir(parents=True, exist_ok=False)
        (output / "formal_config.json").write_text(json.dumps(formal_config, indent=2) + "\n")
    coordinator.wait_for_everyone()
    rank_output = output / f"rank_{coordinator.process_index:03d}"
    rank_output.mkdir(exist_ok=False)
    args._owns_output = True
    cfg.output_dir = str(rank_output)
    misc.register_work_dir(str(rank_output))
    set_global_seed(int(cfg.seed))
    OmegaConf.save(cfg, rank_output / "config.yaml", resolve=True)
    factory_device = runtime._resolve_train_device()
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=factory_device)
    pretrained = deepcopy(model.pretrained_fastwam_metadata)

    def forbidden_encode(*args, **kwargs):
        raise RuntimeError("Cached hardware preflight attempted VAE encoding")

    model._encode_video_latents = forbidden_encode
    model.vae.model.encode = forbidden_encode
    train_ds, val_ds = runtime.build_datasets(cfg.data)
    pinned = _training_plan_module()
    identity = train_ds.manifest["identity"]
    if (identity["vae_weights_sha256"] != pinned.VAE_SHA
            or identity["dataset_stats_sha256"] != pinned.STATS_SHA):
        raise ValueError("Bound cache VAE/statistics identity differs from the official recipe")
    trainer = Wan22Trainer(model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    engine = trainer.model
    zero = trainer.accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]
    if (int(zero["stage"]) != 1 or trainer.accelerator.num_processes != world_size
            or engine.train_micro_batch_size_per_gpu() != 16 or engine.gradient_accumulation_steps() != 1
            or engine.train_batch_size() != world_size * 16
            or any(zero.get(key, {}).get("device", "none") != "none" for key in ("offload_optimizer", "offload_param"))):
        raise RuntimeError("Actual engine batch/accumulation/ZeRO1 differs from the prepared recipe")
    parameters = [*model.dit.parameters(), *model.proprio_encoder.parameters()]
    if not all(parameter.requires_grad for parameter in parameters):
        raise RuntimeError("An original trainable policy parameter was frozen")
    groups = model.optimizer_param_groups(parameters)
    schedule = inspect_scheduler(trainer.scheduler)
    if schedule != {"last_epoch": 0, "warmup_milestones": [1500], "cosine_steps": 28500}:
        raise RuntimeError("Actual learning-rate scheduler differs from the complete 30000-step recipe")
    adam = unwrap_optimizer(trainer.optimizer)
    if not isinstance(adam, torch.optim.AdamW):
        raise RuntimeError("Expected the trainer's actual underlying Torch AdamW")
    original_step, adam_calls = adam.step, []

    @wraps(original_step)
    def measured_step(*step_args, **step_kwargs):
        norm = float(engine.optimizer._global_grad_norm)
        if not math.isfinite(norm) or norm < 0:
            raise FloatingPointError("Actual ZeRO optimizer gradient norm is invalid")
        tick = time.monotonic()
        result = original_step(*step_args, **step_kwargs)
        adam_calls.append({"seconds": time.monotonic() - tick, "gradient_norm": norm,
                           "engine_update": int(engine.global_steps) + 1,
                           "lr": [float(group["lr"]) for group in adam.param_groups]})
        return result

    adam.step = measured_step
    # Guard against future loop changes writing any production-sized checkpoint.
    trainer.save_checkpoint = lambda: (_ for _ in ()).throw(RuntimeError("Preflight cannot save checkpoints"))
    torch.cuda.synchronize()
    startup_seconds = time.monotonic() - started
    losses, updates = run_two_updates(trainer, synchronize=torch.cuda.synchronize)
    verification_started = time.monotonic()
    all_finite = all(_finite(parameter) for parameter in parameters)
    optimizer_report = inspect_adam(adam)
    schedule = inspect_scheduler(trainer.scheduler)
    actual_adam_verified = (len(adam_calls) == len(groups) * 2 and bool(optimizer_report["adam_state_steps"])
                            and all(step == 2 for step in optimizer_report["adam_state_steps"]))
    skipped = any(update["optimizer_step_was_skipped"] for update in updates)
    report = {"rank": coordinator.process_index, "world_size": world_size,
              "status": "complete" if all_finite and actual_adam_verified
                 and optimizer_report["adam_master_and_moments_finite"] and not skipped
                 and schedule == {"last_epoch": 2, "warmup_milestones": [1500], "cosine_steps": 28500}
                 and int(engine.global_steps) == 2 and len(losses) == 2 else "failed",
              "planned_schedule_steps": trainer.max_steps, "actual_engine_updates": int(engine.global_steps),
              "trainer_updates": trainer.global_step, "microbatches": len(losses), "batch_per_rank": 16,
              "scheduler": schedule,
              "all_parameters_finite": all_finite, **optimizer_report,
              "actual_adam_step_calls": len(adam_calls), "expected_adam_step_calls": len(groups) * 2,
              "adam_calls": adam_calls, "optimizer_step_was_skipped": skipped,
              "update_timings": updates, "losses": losses, "startup_seconds": startup_seconds,
              "verification_seconds": time.monotonic() - verification_started,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "gradient_clipping": gradient_clipping_metadata(cfg.max_grad_norm, engine.gradient_clipping(),
                  trainer.accelerator.state.deepspeed_plugin.deepspeed_config),
              "effective_gradient_clipping": float(engine.gradient_clipping()),
              "device": {"env_LOCAL_RANK": int(os.environ.get("LOCAL_RANK", "0")),
                         "accelerator_local_process_index": coordinator.local_process_index,
                         "current_cuda_device": torch.cuda.current_device(),
                         "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
                         "chosen_factory_device": factory_device,
                         "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")},
              "launcher_config": launch,
              "pretrained_fastwam": pretrained, "cache_identity_sha256": train_ds.identity_sha256,
              "deepspeed_zero_stage": int(zero["stage"]),
              "optimizer_group_numel": [sum(parameter.numel() for parameter in group["params"]) for group in groups],
              "policy_checkpoint_bytes_estimate": _unique_state_bytes(model.mot.state_dict()) + _unique_state_bytes(model.proprio_encoder.state_dict()),
              "state_module_bytes_estimate": _unique_state_bytes(model.state_dict()),
              "formal_training_started": False, "checkpoint_saved": False}
    (rank_output / "preflight.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    ranks = [None] * world_size
    torch.distributed.all_gather_object(ranks, report)
    if coordinator.is_main_process:
        storage = io_probe(output, args.io_benchmark_mib)
        storage["free_bytes"] = shutil.disk_usage(output).free
        code = source_identity(ROOT)
        combined = {"status": "complete" if ranks_complete(ranks, world_size, train_ds.identity_sha256,
                        tuple(output.glob("failure_rank_*.json"))) else "failed",
                    "world_size": world_size, "effective_batch": world_size * 16,
                    "planned_schedule_steps": 30000, "actual_engine_updates": 2,
                    "cache_identity_sha256": train_ds.identity_sha256, "ranks": ranks,
                    "launcher_config": launch,
                    "formal_config": formal_config, "source_files": code, "source_sha256": json_sha256(code),
                    "storage": storage, "budget": budget_report(ranks, storage,
                        remaining_seconds=args.remaining_budget_seconds, probe_elapsed_seconds=time.monotonic() - started),
                    "formal_training_started": False, "checkpoint_saved": False,
                    "persistence": "Disposable two-update probe; formal run starts fresh seed and release weights"}
        (output / "preflight.json").write_text(json.dumps(combined, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: combined[key] for key in ("status", "world_size", "effective_batch", "actual_engine_updates", "budget")}), flush=True)
    coordinator.wait_for_everyone()
    receipt = json.loads((output / "preflight.json").read_text())
    return receipt_complete(receipt, tuple(output.glob("failure_rank_*.json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan", help="Verified plan JSON exported by prepare_cached_training.py")
    source.add_argument("--config", help="Resolved YAML with exact expected_identity_sha256")
    parser.add_argument("--output", required=True, help="Fresh disposable report directory")
    parser.add_argument("--num-processes", type=int, choices=(4, 8))
    parser.add_argument("--launcher-config", help="Optional actual --config_file path, cross-checked against the ancestor launch")
    parser.add_argument("--remaining-budget-seconds", type=float)
    parser.add_argument("--io-benchmark-mib", type=int, default=64)
    parser.add_argument("--execute-preflight", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.io_benchmark_mib <= 256:
        parser.error("Bounded IO benchmark must be between 0 and 256 MiB")
    if (args.remaining_budget_seconds is not None
            and (not math.isfinite(args.remaining_budget_seconds) or args.remaining_budget_seconds <= 0)):
        parser.error("Remaining budget must be positive")
    if not args.execute_preflight:
        print(json.dumps(metadata(args), indent=2, allow_nan=False))
        return
    try:
        passed = execute(args)
    except Exception as error:
        if getattr(args, "_owns_output", False):
            failed = {"status": "failed", "rank": int(os.environ.get("RANK", "0")),
                      "error_type": type(error).__name__, "error": str(error),
                      "formal_training_started": False, "checkpoint_saved": False}
            torch_module = sys.modules.get("torch")
            if torch_module is not None and torch_module.cuda.is_initialized():
                failed["peak_allocated_gib"] = torch_module.cuda.max_memory_allocated() / 2**30
            path = Path(args.output) / f"failure_rank_{failed['rank']:03d}.json"
            path.write_text(json.dumps(failed, indent=2, allow_nan=False) + "\n")
        raise
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
