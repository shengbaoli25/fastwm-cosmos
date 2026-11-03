"""Read-only smoke evidence around the actual Accelerate/ZeRO1 training calls.

Tensor fingerprints are deterministic samples, not a claim of full-tensor
equality. Adam counters, scheduler, trainer progress and RNG are checked exactly.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import time
from unittest.mock import patch

from .contracts import identity_sha256, write_json

SCHEMA = "fastwam-real-smoke-audit-v1"


def tensor_fingerprint(tensor, limit=4096):
    import torch
    flat = tensor.detach().reshape(-1)
    count = min(flat.numel(), limit)
    if count == 0:
        raise ValueError("Cannot audit an empty tensor")
    indices = torch.arange(count, device=flat.device, dtype=torch.int64)
    if count > 1:
        indices = indices * (flat.numel() - 1) // (count - 1)
    sample = flat[indices].contiguous().cpu()
    if not torch.isfinite(sample).all():
        raise ValueError("Nonfinite parameter/optimizer sample")
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": tensor.numel(),
            "sampled_numel": count, "sha256": hashlib.sha256(sample.view(torch.uint8).numpy().tobytes()).hexdigest()}


def _plain(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _adam(trainer):
    optimizer = trainer.model.optimizer
    seen = set()
    while hasattr(optimizer, "optimizer"):
        if id(optimizer) in seen:
            raise ValueError("Cyclic optimizer wrapper")
        seen.add(id(optimizer))
        optimizer = optimizer.optimizer
    if type(optimizer).__name__ != "AdamW":
        raise ValueError("Smoke audit expects the configured Torch AdamW under ZeRO1")
    return optimizer


def policy_samples(model):
    selected = {}
    selectors = {
        "action_input": lambda key: key.endswith("mixtures.action.action_encoder.weight"),
        "action_output": lambda key: key.endswith("mixtures.action.head.weight"),
        "video_backbone": lambda key: ".mixtures.video." in key,
        "proprio": lambda key: key.startswith("proprio_encoder."),
    }
    # Official FastWAM registers video_expert/action_expert before their shared
    # mot.mixtures aliases. Parent named_parameters() deduplicates those aliases
    # and therefore never emits mot.*. Enumerate the actual checkpoint roots so
    # the same trainable tensors are audited under their saved payload names.
    for root_name in ("mot", "proprio_encoder"):
        module = getattr(model, root_name, None)
        if module is None:
            continue
        for name, parameter in module.named_parameters(prefix=root_name):
            if not parameter.requires_grad or parameter.ndim < 2:
                continue
            for label, matches in selectors.items():
                if label not in selected and matches(name):
                    selected[label] = {"name": name, **tensor_fingerprint(parameter)}
    if set(selected) != set(selectors):
        raise ValueError(f"Missing audited trainable parameter categories: {set(selectors) - set(selected)}")
    return selected


def optimizer_samples(trainer):
    import torch
    optimizer = _adam(trainer)
    partitions = []
    for group_index, group in enumerate(optimizer.param_groups):
        for param_index, parameter in enumerate(group["params"]):
            if parameter.dtype != torch.float32:
                raise ValueError("Expected actual FP32 Adam master partitions under the configured ZeRO1 engine")
            state = optimizer.state.get(parameter, {})
            item = {"group": group_index, "parameter": param_index, "master": tensor_fingerprint(parameter)}
            if state:
                if not {"step", "exp_avg", "exp_avg_sq"} <= state.keys():
                    raise ValueError("Actual Adam partition is missing its counter or moments")
                item.update(step=float(state["step"]), exp_avg=tensor_fingerprint(state["exp_avg"]),
                            exp_avg_sq=tensor_fingerprint(state["exp_avg_sq"]))
            else:
                item["step"] = 0.0
            partitions.append(item)
    if not partitions:
        raise ValueError("No actual Adam partitions found")
    return {"partitions": partitions,
            "param_groups": [_plain({k: v for k, v in g.items() if k != "params"}) for g in optimizer.param_groups]}


def rng_fingerprint(device):
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    payload = {"python": random.getstate(), "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
               "torch_cpu": torch.get_rng_state().tolist(), "torch_cuda_local": torch.cuda.get_rng_state(device).tolist()}
    return identity_sha256(payload)


def capture_state(trainer):
    model = trainer.accelerator.unwrap_model(trainer.model)
    return {"global_step": int(trainer.global_step), "engine_global_steps": int(trainer.model.global_steps),
            "epoch": int(trainer.epoch), "batch_in_epoch": int(trainer.batch_in_epoch),
            "accelerator_step": int(trainer.accelerator.step),
            "policy_samples": policy_samples(model), "optimizer": optimizer_samples(trainer),
            "scheduler": _plain(trainer.scheduler.state_dict()),
            "rng_sha256": rng_fingerprint(trainer.accelerator.device)}


def assert_restored(current, saved):
    if current != saved:
        changed = [key for key in current if current[key] != saved.get(key)]
        raise ValueError(f"Loaded training state differs from checkpoint audit: {changed}")


def memory_record(device):
    import torch
    torch.cuda.synchronize(device)
    return {"allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


@contextmanager
def audit_prepared_training(trainer, cfg, limit):
    """Enable only on the separate three-step smoke run, after real restore."""
    import torch
    if int(cfg.max_steps) != 3 or (int(trainer.global_step), limit) not in {(0, 2), (2, 3)}:
        raise ValueError("Smoke audit requires a separate 3-step configuration, train 0->2 then resume 2->3")
    rank = int(trainer.accelerator.process_index)
    start_step = int(trainer.global_step)
    model = trainer.accelerator.unwrap_model(trainer.model)
    binding_hash = identity_sha256(model.ablation_binding)
    from .fastwam import verify_prepared_batch
    prepared_batch = verify_prepared_batch(trainer, cfg)
    initial = capture_state(trainer)
    if {p["step"] for p in initial["optimizer"]["partitions"]} != {float(start_step)}:
        raise ValueError("Initial actual Adam counters differ from the trainer step")
    restored = False
    if start_step:
        state_dir = Path(cfg.resume)
        saved = json.loads((state_dir / f"smoke-audit-rank-{rank:03d}.json").read_text())
        if saved["schema"] != SCHEMA or saved["binding_sha256"] != binding_hash or saved["rank"] != rank:
            raise ValueError("Checkpoint audit belongs to a different recipe or rank")
        assert_restored(initial, saved["state"])
        sampler = trainer.train_sampler
        if sampler.epoch_offset != trainer.epoch or sampler.resume_batch_offset != trainer.batch_in_epoch:
            raise ValueError("Actual data sampler did not restore the saved position")
        restored = True
    torch.cuda.reset_peak_memory_stats(trainer.accelerator.device)
    beginning = time.monotonic()
    steps, checkpoints = [], []
    original_backward, original_save = trainer.accelerator.backward, trainer.save_checkpoint

    def backward(loss, *args, **kwargs):
        loss_value = float(loss.detach().float())
        if not math.isfinite(loss_value) or not loss.requires_grad:
            raise ValueError("Actual training loss is nonfinite or detached")
        before = optimizer_samples(trainer)
        engine_step = int(trainer.model.global_steps)
        started = time.monotonic()
        result = original_backward(loss, *args, **kwargs)
        # Accelerate 1.12 DeepSpeedEngineWrapper.backward invokes engine.step;
        # engine.get_global_grad_norm is the real pre-clipping reduced norm.
        norm = float(trainer.model.get_global_grad_norm())
        after = optimizer_samples(trainer)
        expected_step = start_step + len(steps) + 1
        if (not math.isfinite(norm) or norm <= 0 or trainer.accelerator.optimizer_step_was_skipped
                or int(trainer.model.global_steps) != engine_step + 1
                or {p["step"] for p in after["partitions"]} != {float(expected_step)}):
            raise ValueError("Actual backward/optimizer update failed, skipped or had nonfinite/zero gradients")
        updated = [i for i, (old, new) in enumerate(zip(before["partitions"], after["partitions"], strict=True))
                   if old["master"]["sha256"] != new["master"]["sha256"]]
        if not updated:
            raise ValueError("No sampled FP32 master parameters changed after the actual optimizer step")
        steps.append({"step": expected_step, "loss": loss_value, "global_grad_norm": norm,
                      "changed_master_partitions": updated, "optimizer_before": before, "optimizer_after": after,
                      "backward_and_update_seconds": time.monotonic() - started,
                      "cuda_memory": memory_record(trainer.accelerator.device)})
        return result

    def save_checkpoint():
        result = original_save()
        state = capture_state(trainer)
        write_json(Path(result["state_path"]) / f"smoke-audit-rank-{rank:03d}.json",
                   {"schema": SCHEMA, "rank": rank, "binding_sha256": binding_hash, "state": state})
        checkpoints.append({"step": int(trainer.global_step), "state_path": result["state_path"]})
        trainer.accelerator.wait_for_everyone()
        return result

    with patch.object(trainer.accelerator, "backward", backward), patch.object(trainer, "save_checkpoint", save_checkpoint):
        yield
    final = capture_state(trainer)
    if (trainer.global_step != limit or len(steps) != limit - start_step
            or not any(record["step"] == limit for record in checkpoints)):
        raise ValueError("Smoke invocation did not execute and save every required optimizer step")
    changed_policy = [key for key in initial["policy_samples"]
                      if initial["policy_samples"][key]["sha256"] != final["policy_samples"][key]["sha256"]]
    if not changed_policy:
        raise ValueError("No sampled deployed policy parameter changed during the real smoke invocation")
    write_json(Path(cfg.output_dir) / "smoke-audit" / f"from_{start_step:06d}-to_{limit:06d}.rank_{rank:03d}.json", {
        "schema": SCHEMA, "completed": True, "rank": rank, "binding_sha256": binding_hash,
        "prepared_batch": prepared_batch,
        "start_step": start_step, "end_step": limit, "restored_checkpoint_verified": restored,
        "initial": initial, "final": final, "steps": steps, "checkpoints": checkpoints,
        "changed_policy_samples": changed_policy, "elapsed_seconds": time.monotonic() - beginning,
        "cuda_memory": memory_record(trainer.accelerator.device),
        "evidence_scope": "Real loss/gradient norm/Adam counters; sampled policy/master/moment equality; exact scheduler/progress/RNG; not full-tensor equality",
    })
