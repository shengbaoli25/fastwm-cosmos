#!/usr/bin/env python3
"""Two disposable real updates with the planned official schedule; no weights saved."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import MethodType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-steps", type=int, default=2176)
    parser.add_argument("--task", default="forcevla_baseline",
                        help="Hydra task to probe; must match the formal image/data recipe")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.target_steps < 2:
        parser.error("Target schedule must contain at least two updates")
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam import runtime
    from fastwam.trainer import Wan22Trainer
    from fastwam.utils import misc
    from fastwam.utils.pytorch_utils import set_global_seed
    from fastwam.baseline.analysis import file_sha256, json_sha256, source_identity

    root = Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={args.task}", f"max_steps={args.target_steps}",
                       f"output_dir={output}", "eval_every=0", "save_every=0", *args.override])
    if not torch.cuda.is_available():
        raise RuntimeError("This full-model preflight requires a GPU")
    misc.register_work_dir(str(output))
    set_global_seed(int(cfg.seed))
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    pretrained_metadata = model.pretrained_fastwam_metadata
    groups = model.optimizer_param_groups(list(model.dit.parameters()) + list(model.proprio_encoder.parameters()))
    original_loss = model.training_loss
    losses = []

    def checked_loss(self, sample, **kwargs):
        loss, components = original_loss(sample, **kwargs)
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError("Nonfinite original FastWAM training loss")
        losses.append({"loss": float(loss.detach()), **components})
        return loss, components

    model.training_loss = MethodType(checked_loss, model)
    train_ds, val_ds = runtime.build_datasets(cfg.data)
    trainer = Wan22Trainer(model=model, train_dataset=train_ds, val_dataset=val_ds, cfg=cfg)
    engine = trainer.model
    plugin = trainer.accelerator.state.deepspeed_plugin
    if plugin is None:
        raise RuntimeError("Official DeepSpeed initialization is required")
    ds_config = plugin.deepspeed_config
    zero = ds_config["zero_optimization"]
    if int(zero["stage"]) != 1 or any(zero.get(key, {}).get("device", "none") != "none"
                                        for key in ("offload_optimizer", "offload_param")):
        raise RuntimeError("Original ZeRO1 without offload is required")
    if float(engine.gradient_clipping()) != 0:
        raise RuntimeError("Original DS JSON has effective gradient clipping disabled")
    if not all(p.requires_grad for p in model.dit.parameters()):
        raise RuntimeError("An original DiT parameter was silently frozen")
    adam = trainer.optimizer
    visited = set()
    while hasattr(adam, "optimizer") and id(adam) not in visited:
        visited.add(id(adam))
        adam = adam.optimizer
    if not isinstance(adam, torch.optim.AdamW):
        raise RuntimeError("The actual original AdamW must be inspected")
    original_adam_step = adam.step
    adam_calls = []

    def measured_adam_step(*step_args, **step_kwargs):
        norm = float(engine.optimizer._global_grad_norm)
        if not torch.isfinite(torch.tensor(norm)) or norm < 0:
            raise FloatingPointError("The actual optimizer gradient norm is invalid")
        tick = time.monotonic()
        result = original_adam_step(*step_args, **step_kwargs)
        adam_calls.append({"seconds": time.monotonic() - tick, "gradient_norm": norm,
                           "engine_update": int(engine.global_steps) + 1,
                           "lr": [float(group["lr"]) for group in adam.param_groups]})
        return result

    adam.step = measured_adam_step
    original_step = engine.step
    boundaries = []
    last_boundary = time.monotonic()
    last_micro = 0

    def measured_step(self, *step_args, **step_kwargs):
        nonlocal last_boundary, last_micro
        old_steps = int(self.global_steps)
        result = original_step(*step_args, **step_kwargs)
        if int(self.global_steps) != old_steps:
            torch.cuda.synchronize()
            now = time.monotonic()
            boundaries.append({"engine_step": int(self.global_steps), "seconds": now - last_boundary,
                               "microbatches": len(losses) - last_micro})
            last_boundary, last_micro = now, len(losses)
        return result

    engine.step = MethodType(measured_step, engine)
    planned_steps = trainer.max_steps
    trainer.max_steps = 2
    trainer.save_checkpoint = lambda: {"weights_path": None, "state_path": None}
    trainer.train()
    unwrapped = trainer.accelerator.unwrap_model(engine)
    finite = True
    for parameter in [*unwrapped.dit.parameters(), *unwrapped.proprio_encoder.parameters()]:
        flat = parameter.detach().reshape(-1)
        for offset in range(0, flat.numel(), 2_000_000):
            finite = finite and bool(torch.isfinite(flat[offset:offset + 2_000_000]).all())
    global_steps = int(getattr(engine, "global_steps", -1))
    adam_state_finite = True
    adam_steps = []
    for group in adam.param_groups:
        for parameter in group["params"]:
            state = adam.state[parameter]
            if any(key not in state for key in ("step", "exp_avg", "exp_avg_sq")):
                raise RuntimeError("A ZeRO parameter group has no actual Adam moments")
            adam_steps.append(float(state["step"]))
            for tensor in (parameter, state["exp_avg"], state["exp_avg_sq"]):
                flat = tensor.detach().reshape(-1)
                for offset in range(0, flat.numel(), 2_000_000):
                    adam_state_finite = adam_state_finite and bool(torch.isfinite(flat[offset:offset + 2_000_000]).all())
    config_identity = OmegaConf.to_container(cfg, resolve=True)
    config_identity.pop("output_dir")
    code = source_identity(root)
    skipped = bool(trainer.accelerator.optimizer_step_was_skipped)
    actual_adam_verified = len(adam_calls) == len(groups) * 2 and adam_steps and all(step == 2 for step in adam_steps)
    report = {"status": "complete" if finite and global_steps == 2 and len(losses) == 32 and
              actual_adam_verified and adam_state_finite and not skipped else "failed",
              "planned_schedule_steps": planned_steps, "actual_engine_updates": global_steps,
              "microbatches": len(losses), "all_parameters_finite": finite,
              "optimizer_group_numel": [sum(p.numel() for p in group["params"]) for group in groups],
              "actual_adam_step_calls": len(adam_calls), "expected_adam_step_calls": len(groups) * 2,
              "adam_state_steps": adam_steps, "adam_master_and_moments_finite": adam_state_finite,
              "optimizer_step_was_skipped": skipped, "adam_calls": adam_calls,
              "deepspeed_zero_stage": int(zero["stage"]),
              "pretrained_fastwam": pretrained_metadata,
              "update_timings": boundaries,
              "timing_note": "Two diagnostic updates including synchronization; first includes cold kernels/Adam allocation. Formal throughput remains unmeasured.",
              "stats_sha256": file_sha256(cfg.data.train.pretrained_norm_stats),
              "data_conversion_manifest_sha256": file_sha256(Path(cfg.data.train.dataset_dirs[0]) / "conversion_manifest.json"),
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "effective_gradient_clipping": getattr(engine, "gradient_clipping", lambda: None)(),
              "resolved_config_except_output": config_identity, "source_files": code,
              "source_sha256": json_sha256(code), "losses": losses,
              "persistence": "disposable two-update probe; no checkpoint/state saved; formal run must start fresh seed"}
    (output / "preflight.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "actual_engine_updates", "microbatches", "all_parameters_finite", "peak_allocated_gib")}))
    raise SystemExit(0 if report["status"] == "complete" else 2)


if __name__ == "__main__":
    main()
