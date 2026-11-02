#!/usr/bin/env python3
"""Resume the original Cosmos run with its 50k recipe; stop at total step 30000.

Default: print a plan without importing torch. --execute must run under four-rank
Accelerate/DeepSpeed after reviewing audit_resume.py's report of the actual host.
The old training directory and state are never modified by this wrapper.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

CONFIG_SHA256 = "6941b393f453a57d42942a70e0780853405bbdeb37e38fe3b6d4393deb6e6898"
JOB_ID = "job-187fc8ee-5443-4786-963b-d48882d3f8ff"
ORIGINAL_STEPS = 50000
STOP_STEP = 30000
RESTORE_STEP = 25000
SOURCE_FILES = (
    "src/fastwam/trainer.py", "src/fastwam/runtime.py", "src/fastwam/utils/samplers.py",
    "src/fastwam/cosmos/assets.py", "src/fastwam/cosmos/cache.py", "src/fastwam/cosmos/native.py",
    "src/fastwam/cosmos/noaction.py", "src/fastwam/cosmos/state.py",
    "scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml", "scripts/ds_configs/ds_zero1_config.json",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def validate_config(cfg):
    expected = dict(batch_size=64, num_workers=8, lr_scheduler_type="cosine", learning_rate=1e-4,
                    max_steps=ORIGINAL_STEPS, gradient_accumulation_steps=1, mixed_precision="bf16",
                    seed=42, max_grad_norm=1.0, weight_decay=0.01, save_every=5000, eval_every=0, resume=None)
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError("Original 4-rank B64/accum1/50k configuration must remain unchanged")
    if cfg.model.get("training_recipe") != {"batch_per_rank": 64, "max_steps": ORIGINAL_STEPS}:
        raise ValueError("Original model training_recipe must retain the 50k manifest")
    if cfg.model.get("_target_") != "fastwam.cosmos.assets.create_noaction_joint_cosmos":
        raise ValueError("Expected original noaction joint8 Cosmos factory")
    if cfg.data.train.get("_target_") != "fastwam.cosmos.cache.CosmosCachedDataset" or cfg.data.get("val") is not None:
        raise ValueError("Expected original cached dataset with no separate validation split")


def validate_audit(args, oldrun, state):
    from audit_resume import state_inventory
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    if (audit.get("schema") != "cosmos-readonly-resume-audit-v1"
            or Path(audit.get("oldrun", "")).resolve() != oldrun
            or audit.get("config", {}).get("sha256") != CONFIG_SHA256
            or audit.get("world_size_expected") != 4 or audit.get("stop_at_step") != STOP_STEP):
        raise ValueError("Audit does not bind the original run/config and requested four-rank stop target")
    source = Path(args.source_root).resolve()
    source_audit = audit.get("source", {})
    if Path(source_audit.get("path", "")).resolve() != source:
        raise ValueError("Audit must include --source-root for the actual imported source")
    hashes = {}
    for relative in SOURCE_FILES:
        expected = source_audit.get("files", {}).get(relative, {}).get("sha256")
        path = source / relative
        if not expected or not path.is_file() or sha(path) != expected:
            raise ValueError("Source or launcher changed after audit: " + relative)
        hashes[relative] = expected
    audited = next((value for value in audit.get("states", []) if value.get("step") == RESTORE_STEP), None)
    current = state_inventory(oldrun, state, 4, True)
    if (not audited or Path(audited["path"]).resolve() != state or not audited.get("metadata_candidate")
            or not current["metadata_candidate"]
            or audited.get("trainer_state") != current["trainer_state"]):
        raise ValueError("Selected full-state checkpoint is missing, changed or structurally incomplete")
    # Detect accidental replacement/truncation after audit, without hashing 20+ GB.
    before = {entry["path"]: (entry.get("bytes"), entry.get("mtime_utc")) for entry in audited["files"]}
    after = {entry["path"]: (entry.get("bytes"), entry.get("mtime_utc")) for entry in current["files"]}
    if before != after:
        raise ValueError("Checkpoint file size/time changed after audit")
    return current, {"audit_sha256": sha(args.audit), "source_hashes": hashes,
                     "audited_source_head": source_audit.get("git_head")}


def unwrap(value, attribute):
    seen = set()
    while hasattr(value, attribute) and id(value) not in seen:
        seen.add(id(value))
        value = getattr(value, attribute)
    return value


def finite(tensor):
    import torch
    flat = tensor.detach().reshape(-1)
    return all(bool(torch.isfinite(flat[start:start + 2_000_000]).all())
               for start in range(0, flat.numel(), 2_000_000))


def equal_state(left, right):
    import numpy as np
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, (list, tuple)) and len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


def rng_snapshot():
    import random
    import numpy as np
    import torch
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()


def restore_rng(snapshot):
    import random
    import numpy as np
    import torch
    py, npy, cpu, cuda = snapshot
    random.setstate(py)
    np.random.set_state(npy)
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state_all(cuda)


class FirstEpochContinuation:
    """Skip already consumed sharded batches while retaining original end padding."""
    def __init__(self, loader, sampler, epoch, batches):
        if not hasattr(loader, "iteration") or not 0 <= batches <= len(loader):
            raise ValueError("Expected native Accelerate loader and a valid saved batch offset")
        self.loader, self.sampler = loader, sampler
        self.epoch, self.batches, self.first = epoch, batches, True
        sampler.set_epoch_offset(0)
        sampler.clear_resume_batch_offset()

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if not self.first:
            yield from self.loader
            return
        from accelerate import skip_first_batches
        self.first = False
        snapshot = rng_snapshot()
        shorter = skip_first_batches(self.loader, self.batches)
        shorter.set_epoch(self.epoch)
        self.sampler.set_epoch(self.epoch)
        iterator = iter(shorter)
        try:
            first = next(iterator)
        except StopIteration:
            restore_rng(snapshot)
            self.loader.set_epoch(self.epoch + 1)
            return
        # Creating replacement workers must not consume the restored model RNG.
        restore_rng(snapshot)
        yield first
        yield from iterator
        self.loader.set_epoch(self.epoch + 1)


def verify_restored(trainer, state, progress):
    import torch
    actual = {key: getattr(trainer, key) for key in ("global_step", "epoch", "batch_in_epoch")}
    if actual != progress or trainer.model.global_steps != RESTORE_STEP:
        raise ValueError("Trainer/DeepSpeed progress differs from the saved full state")
    engine = trainer.model
    if (engine.train_micro_batch_size_per_gpu(), engine.gradient_accumulation_steps(), engine.train_batch_size()) != (64, 1, 256):
        raise ValueError("Restored engine must use real microbatch64, accumulation1, globalbatch256")
    scheduler = unwrap(trainer.scheduler, "scheduler")
    scheduler_summary = {"last_epoch": int(scheduler.last_epoch), "warmup_milestones": list(scheduler._milestones),
                         "cosine_steps": int(scheduler._schedulers[-1].T_max)}
    if scheduler_summary != {"last_epoch": RESTORE_STEP, "warmup_milestones": [2500], "cosine_steps": 47500}:
        raise ValueError("Original 50k scheduler was not restored")
    saved_scheduler = torch.load(state / "scheduler.bin", map_location="cpu", weights_only=False)
    if not equal_state(trainer.scheduler.state_dict(), saved_scheduler):
        raise ValueError("Restored scheduler differs from the saved scheduler")
    adam = unwrap(trainer.optimizer, "optimizer")
    steps, partitions = [], 0
    for group in adam.param_groups:
        for parameter in group["params"]:
            saved = adam.state[parameter]
            if any(key not in saved for key in ("step", "exp_avg", "exp_avg_sq")):
                raise ValueError("An actual Adam partition is missing step or moments")
            steps.append(float(saved["step"]))
            if not all(finite(tensor) for tensor in (parameter, saved["exp_avg"], saved["exp_avg_sq"])):
                raise ValueError("Restored Adam master weights or moments are nonfinite")
            partitions += 1
    if not steps or set(steps) != {float(RESTORE_STEP)} or not all(finite(p) for p in engine.parameters()):
        raise ValueError("Restored optimizer step or model parameters are invalid")
    rank = trainer.accelerator.process_index
    saved_rng = torch.load(state / f"random_states_{rank}.pkl", map_location="cpu", weights_only=False)
    py, npy, cpu, cuda = rng_snapshot()
    if (not equal_state(py, saved_rng["random_state"]) or not equal_state(npy, saved_rng["numpy_random_seed"])
            or not equal_state(cpu, saved_rng["torch_manual_seed"])
            or not equal_state(cuda, saved_rng["torch_cuda_manual_seed"])
            or trainer.accelerator.step != saved_rng["step"]):
        raise ValueError("Per-rank Python/NumPy/Torch/CUDA/Accelerate RNG state was not restored")
    sampler = trainer.train_sampler
    if (sampler.epoch_offset != progress["epoch"] or sampler.resume_batch_offset != progress["batch_in_epoch"]
            or (sampler.seed, sampler.batch_size, sampler.num_processes) != (42, 64, 4)
            or not 0 <= progress["batch_in_epoch"] <= len(trainer.train_loader)):
        raise ValueError("Sampler identity or saved data progress differs")
    # mmap keeps the large model archive out of an extra resident CPU copy.
    module = torch.load(state / "pytorch_model/mp_rank_00_model_states.pt", map_location="cpu",
                        weights_only=False, mmap=True)["module"]
    current = trainer.accelerator.unwrap_model(engine).state_dict()
    keys = [key for key in current if key.endswith("rng_state") or key.endswith("manifest_bytes")]
    if not any(key.endswith("rng_state") for key in keys) or not any(key.endswith("manifest_bytes") for key in keys):
        raise ValueError("Persistent Force RNG/protocol buffers are missing")
    if any(key not in module or not torch.equal(current[key].detach().cpu(), module[key]) for key in keys):
        raise ValueError("Persistent Force RNG/protocol buffers differ from the original checkpoint")
    del module, current
    return {"trainer": actual, "engine_step": int(engine.global_steps), "scheduler": scheduler_summary,
            "adam_partitions": partitions, "adam_steps": sorted(set(steps)), "finite_model_and_adam": True,
            "per_rank_rng_restored": True, "force_buffers_verified": keys,
            "configured_max_grad_norm": trainer.max_grad_norm,
            "actual_engine_gradient_clipping": float(engine.gradient_clipping())}


def execute(args):
    if not args.audit:
        raise ValueError("--execute requires the reviewed --audit JSON from the original host/source")
    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "").lower() != "true" or os.environ.get("WORLD_SIZE") != "4":
        raise ValueError("Launch through the audited four-process Accelerate/DeepSpeed configuration")
    from omegaconf import OmegaConf
    config = Path(args.config).resolve()
    if sha(config) != CONFIG_SHA256:
        raise ValueError("Original config SHA256 differs")
    cfg = OmegaConf.load(config)
    validate_config(cfg)
    oldrun = Path(cfg.output_dir).resolve()
    if oldrun.name != "training" or JOB_ID not in oldrun.parts or config.parent != oldrun:
        raise ValueError("Use the actual original run/config.yaml, not a relocated/reconstructed recipe")
    state = Path(args.state).resolve()
    if state != oldrun / "checkpoints/state/step_025000":
        raise ValueError("This wrapper is scoped to the original completed step25000 full state")
    out = Path(args.output).resolve()
    if out == oldrun or out.is_relative_to(oldrun) or oldrun.is_relative_to(out):
        raise ValueError("New output must be separate from the immutable old run")
    checkpoint, provenance = validate_audit(args, oldrun, state)
    source = Path(args.source_root).resolve()
    sys.path[:0] = [str(source / "src"), str(source)]
    import torch
    from accelerate import Accelerator
    from hydra.utils import instantiate
    import fastwam.trainer as trainer_module
    from fastwam.runtime import build_datasets
    from fastwam.utils import misc
    from fastwam.utils.logging_config import setup_logging
    from fastwam.utils.pytorch_utils import set_global_seed
    if Path(trainer_module.__file__).resolve() != source / "src/fastwam/trainer.py":
        raise ValueError("Imported trainer is not the audited original source")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    acc = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16", step_scheduler_with_optimizer=False)
    ds_config = acc.state.deepspeed_plugin.deepspeed_config
    if (acc.num_processes != 4 or acc.process_index != int(os.environ["RANK"])
            or ds_config["zero_optimization"]["stage"] != 1 or "H200" not in torch.cuda.get_device_name()):
        raise ValueError("Exactly four H200 ranks with original ZeRO1 are required")
    audited_ds = json.loads((source / "scripts/ds_configs/ds_zero1_config.json").read_text())
    for key in ("zero_optimization", "gradient_clipping"):
        if ds_config.get(key) != audited_ds.get(key):
            raise ValueError("Actual DeepSpeed settings differ from the audited original config: " + key)
    if acc.is_main_process:
        out.mkdir(parents=True, exist_ok=False)
        (out / "origins").mkdir()
    acc.wait_for_everyone()
    cfg.output_dir, cfg.resume = str(out), str(state)
    misc.register_work_dir(str(out))
    set_global_seed(42)
    import logging
    setup_logging(log_level=logging.INFO, is_main_process=acc.is_main_process)
    if acc.is_main_process:
        handler = logging.FileHandler(out / "training.log", mode="x", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s %(filename)s:%(lineno)d"))
        logging.getLogger().addHandler(handler)
    receipt = {"schema": "cosmos-original-50k-resume-stop30k-v1", "rank": acc.process_index,
               "job_id": os.environ.get("TRAIN_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
               "original_config_sha256": CONFIG_SHA256, "original_run": str(oldrun), "source": str(source),
               "resume_state": str(state), "original_recipe_steps": ORIGINAL_STEPS, "stop_at_step": STOP_STEP,
               "batch_per_rank": 64, "global_batch": 256, "gradient_accumulation_steps": 1,
               "gpu": torch.cuda.get_device_name(), "python": sys.executable, **provenance,
               "packages": {name: importlib.metadata.version(name) for name in ("torch", "accelerate", "deepspeed")}}
    origin = out / f"origins/rank_{acc.process_index:03d}.json"
    write_json(origin, dict(receipt, status="restoring"))
    if acc.is_main_process:
        OmegaConf.save(cfg, out / "config.yaml")
        write_json(out / "resume_control.json", {"original_recipe_steps": ORIGINAL_STEPS, "stop_at_step": STOP_STEP,
                                                  "resume_state": str(state), "original_config_sha256": CONFIG_SHA256})
    try:
        model = instantiate(cfg.model, device=f"cuda:{torch.cuda.current_device()}")
        dataset, validation = build_datasets(cfg.data)
        if len(dataset) != 64462 or validation is not dataset:
            raise ValueError("Original cached dataset length/no-split identity differs")
        trainer = trainer_module.Wan22Trainer(model, dataset, validation, cfg=cfg)
        restored = verify_restored(trainer, state, checkpoint["trainer_state"])
        trainer.train_loader = FirstEpochContinuation(trainer.train_loader, trainer.train_sampler,
                                                       trainer.epoch, trainer.batch_in_epoch)
        trainer.max_steps = STOP_STEP
        receipt.update(status="resumed", restored=restored)
        write_json(origin, receipt)
        acc.wait_for_everyone()
        trainer.train()
        if trainer.global_step != STOP_STEP or trainer.model.global_steps != STOP_STEP:
            raise ValueError("Training did not stop at cumulative step30000")
        acc.wait_for_everyone()
        from audit_resume import state_inventory
        final = state_inventory(out, out / "checkpoints/state/step_030000", 4, True)
        if not final["metadata_candidate"] or final["trainer_state"]["global_step"] != STOP_STEP:
            raise ValueError("Final full-state checkpoint is incomplete")
        receipt.update(status="complete", global_step=trainer.global_step, finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(origin, receipt)
        acc.wait_for_everyone()
        torch.distributed.destroy_process_group()
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(origin, receipt)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--audit", help="Reviewed audit_resume.py report including --source-root")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"mode": "plan_only", "training_started": False, "config_sha256_required": CONFIG_SHA256,
                          "config": args.config, "state": args.state, "new_output": args.output,
                          "source": args.source_root, "audit": args.audit, "world_size": 4,
                          "gpu": "H200", "microbatch_per_gpu": 64, "gradient_accumulation_steps": 1,
                          "original_scheduler_and_model_recipe_steps": ORIGINAL_STEPS,
                          "restore_step": RESTORE_STEP, "stop_at_total_step": STOP_STEP,
                          "additional_updates": STOP_STEP - RESTORE_STEP}, indent=2))
        return 0
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
