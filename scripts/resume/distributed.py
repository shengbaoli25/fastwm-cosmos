"""Resume a stopped four-rank Cosmos run in a foreground platform training job.

The existing interactive output is read-only. Submit this only after the old
allocation is confirmed stopped; the new job writes to a distinct output.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from checks import (complete_state, launch_environment, platform_identity, stopped_evidence,
                    verify_restored, verify_code, prior_resource, final_check, storage_estimate)


def recipe(cfg):
    from omegaconf import OmegaConf
    from fastwam.cosmos.preflight import validate_config

    data = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    data.pop("output_dir", None)
    data["resume"] = None
    validate_config(data)
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_inputs(source: Path, prior: Path, output: Path, preflight: Path, prior_log: Path):
    from omegaconf import OmegaConf
    from fastwam.cosmos.binding import binding
    from fastwam.cosmos.preflight import validate_config

    if not source.is_dir() or not prior.is_dir() or output.exists():
        raise ValueError("Source/prior output missing or new output already exists")
    config = prior / "config.yaml"
    cfg = OmegaConf.load(config)
    digest = recipe(cfg)
    if Path(cfg.output_dir) != prior:
        raise ValueError("Prior config belongs to another output")
    manifest_path = prior / "resume_origin.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    root = Path(manifest["root_output"]) if manifest else prior
    if root != Path("/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/cosmos-dd805094-4rank-30k-v2"):
        raise ValueError("Resume lineage must bind the actual original experiment")
    root_config = root / "config.yaml"
    if (not root_config.is_file() or recipe(OmegaConf.load(root_config)) != digest
            or (manifest and (manifest.get("output") != str(prior)
                              or manifest.get("root_config_sha256") != hashlib.sha256(root_config.read_bytes()).hexdigest()
                              or manifest.get("recipe_sha256") != digest))):
        raise ValueError("Original Cosmos recipe/source chain changed")
    receipt = json.loads(preflight.read_text())
    if receipt.get("status") != "two_adam_verified" or receipt.get("binding") != binding(root_config):
        raise ValueError("Current code and frozen config do not match the actual GPU preflight")
    if manifest:
        if manifest.get("preflight_sha256") != hashlib.sha256(preflight.read_bytes()).hexdigest():
            raise ValueError("Preflight provenance changed across resume jobs")
        origins = prior / "origins"
        for rank in range(4):
            r = json.loads((origins / f"rank_{rank:03d}.json").read_text())
            if (r.get("rank") != rank or r.get("job") != manifest.get("job")
                    or r.get("resume_step") != manifest.get("resume_step")
                    or r.get("status") not in ("resumed", "complete")):
                raise ValueError("Prior Cosmos resume has incomplete rank lineage")
        if str(cfg.resume) != manifest.get("resume_state"):
            raise ValueError("Prior resume config differs from its lineage")
    else:
        if cfg.resume is not None:
            raise ValueError("Original Cosmos run must start fresh")
        for rank in range(4):
            r = json.loads((prior / "origins" / f"rank_{rank:03d}.json").read_text())
            if (r.get("rank") != rank or r.get("world_size") != 4
                    or r.get("status") not in ("running", "complete", "failed")
                    or r.get("gate", {}).get("binding") != receipt["binding"]
                    or r.get("gate", {}).get("preflight_sha256") != hashlib.sha256(preflight.read_bytes()).hexdigest()
                    or r.get("resource") != receipt.get("resource")):
                raise ValueError("Four matching original Cosmos ranks required")
    state, step = complete_state(prior, prior_log)
    return config, state, step, root, digest


def worker(args):
    import logging
    import torch
    from accelerate import Accelerator
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.cosmos.preflight import build_probe_datasets, validate_config
    from fastwam.trainer import Wan22Trainer
    from fastwam.utils import misc
    from fastwam.utils.logging_config import setup_logging
    from fastwam.utils.pytorch_utils import set_global_seed

    verify_code(args.source)
    cfg = OmegaConf.load(args.config)
    if recipe(cfg) != args.recipe_sha256 or Path(cfg.output_dir) != args.prior_output:
        raise ValueError("Checkpoint does not belong to this config")
    if complete_state(args.prior_output, args.prior_log) != (args.state, args.step):
        raise ValueError("Selected full-state checkpoint changed before rank startup")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    acc = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16",
                      step_scheduler_with_optimizer=False)
    if acc.num_processes != 4 or acc.state.deepspeed_plugin is None:
        raise ValueError("Resume requires exactly four DeepSpeed ranks")
    if acc.process_index != int(os.environ["RANK"]):
        raise ValueError("Rank identity mismatch")
    cfg.output_dir = str(args.output)
    cfg.resume = str(args.state)
    if acc.is_main_process:
        OmegaConf.save(cfg, args.output / "config.yaml")
    acc.wait_for_everyone()
    misc.register_work_dir(str(args.output))
    set_global_seed(42)
    setup_logging(log_level=logging.INFO, is_main_process=acc.is_main_process)
    model = instantiate(cfg.model, device=f"cuda:{torch.cuda.current_device()}")
    train_ds, val_ds = build_probe_datasets(cfg.data)
    trainer = Wan22Trainer(model, train_ds, val_ds, cfg=cfg)
    from fastwam.cosmos.preflight import helpers
    restored = verify_restored(trainer, args.state, args.step, helpers())
    engine = trainer.model
    if (engine.train_micro_batch_size_per_gpu(), engine.gradient_accumulation_steps(),
            engine.train_batch_size()) != (16, 1, 64):
        raise ValueError("Actual distributed batch differs from B16/global64")
    origins = args.output / "origins"
    origins.mkdir(exist_ok=True)
    receipt = {"rank": acc.process_index, "status": "resumed", "resume_step": args.step,
               "restored": restored, "state": str(args.state), "job": os.environ["TRAIN_JOB_ID"],
               "started_at": datetime.now(timezone.utc).isoformat()}
    origin = origins / f"rank_{acc.process_index:03d}.json"
    with origin.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    from continuation import install_continuation
    install_continuation(trainer)
    acc.wait_for_everyone()
    trainer.train()
    if trainer.global_step != 30000:
        raise RuntimeError("Resume ended before 30000 steps")
    receipt.update(status="complete", global_step=trainer.global_step)
    origin.write_text(json.dumps(receipt, indent=2))
    acc.wait_for_everyone()
    torch.distributed.destroy_process_group()


def controller(args):
    job = platform_identity(os.environ)
    termination = stopped_evidence(args.stop_evidence, args.prior_output)
    if termination["resource_id"] != prior_resource(args.prior_output, "dd805094-b321-4f0a-9793-8c5b585ff7a9"):
        raise ValueError("Termination evidence belongs to another allocation/job")
    source = args.source.resolve()
    if source != Path(__file__).resolve().parents[2]:
        raise ValueError("Entry point must belong to the selected source")
    frozen = verify_code(source)
    sys.path[:0] = [str(source / "src"), str(source), str(source / "scripts/cosmos")]
    config, state, step, root, digest = verify_inputs(source, args.prior_output, args.output, args.preflight, args.prior_log)
    native = args.native_python
    if (native != Path("/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-fastwam/.venv/bin/python")
            or not native.is_absolute() or not native.is_file() or native.parent.name != "bin"
            or native.parent.parent.name != ".venv"):
        raise ValueError("Original absolute .venv/bin/python invocation required")
    inventory = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                               check=True, capture_output=True, text=True).stdout.splitlines()
    if len(inventory) not in (4, 8) or any("H200" not in name for name in inventory):
        raise ValueError("Allocate four or eight H200 GPUs; run exactly four training ranks")
    if args.output.exists() or args.watch.exists():
        raise FileExistsError("Resume output/watch must be fresh")
    storage = storage_estimate(state, args.output, step)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          check=True, capture_output=True, text=True).stdout
    if apps.strip():
        raise ValueError("Allocated GPUs already have compute processes")
    from bounded_probe import execute
    env = launch_environment(os.environ, source)
    command = [str(native), "-m", "accelerate.commands.launch", "--config_file",
               str(source / "scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml"),
               "--num_processes", "4", "--num_machines", "1", str(Path(__file__).resolve()),
               "--worker", "--source", str(source), "--prior-output", str(args.prior_output),
               "--prior-log", str(args.prior_log), "--output", str(args.output), "--state", str(state), "--step", str(step),
               "--config", str(config), "--recipe-sha256", digest]
    plan = {"frozen_code_commit": frozen["commit"], "storage": storage, "prior_output": str(args.prior_output), "job": job, "source": str(source), "source_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "resume_state": str(state), "resume_step": step, "output": str(args.output),
            "root_output": str(root), "root_config_sha256": hashlib.sha256((root / "config.yaml").read_bytes()).hexdigest(),
            "stop_evidence_sha256": hashlib.sha256(args.stop_evidence.read_bytes()).hexdigest(),
            "recipe_sha256": digest, "preflight_sha256": hashlib.sha256(args.preflight.read_bytes()).hexdigest(),
            "argv": command, "gpu_names": inventory, "started_at": datetime.now(timezone.utc).isoformat()}
    print("RESUME_PLAN", json.dumps(plan), flush=True)
    if not args.execute:
        return
    args.watch.mkdir(parents=True, exist_ok=False)
    (args.watch / "resume-plan.json").write_text(json.dumps(plan, indent=2))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "resume_origin.json").write_text(json.dumps(plan, indent=2))
    os.environ.clear()
    os.environ.update(env)
    result = execute(command, str(source), str(args.watch / "supervision"), None,
                     maximum_seconds=None, log_name="training.log", phase="cosmos-distributed-resume")
    print("RESUME_SUPERVISOR", json.dumps(result), flush=True)
    if result["status"] != "process_exited_zero":
        raise RuntimeError("Distributed resume did not exit cleanly")
    completion = final_check(args.output, args.watch / "supervision/training.log", job)
    (args.watch / "completion.json").write_text(json.dumps(completion, indent=2))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--execute", action="store_true")
    for name in ("source", "prior-output", "prior-log", "stop-evidence", "output", "preflight", "watch", "native-python", "state", "config", "recipe-sha256"):
        parser.add_argument("--" + name)
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    for name in ("source", "prior_output", "prior_log", "stop_evidence", "output", "preflight", "watch", "native_python", "state", "config"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value))
    if args.worker:
        worker(args)
    else:
        if any(getattr(args, name) is None for name in
               ("source", "prior_output", "prior_log", "stop_evidence", "output", "preflight", "watch", "native_python")):
            parser.error("Controller requires source, prior-output, output, preflight, watch and native-python")
        controller(args)


if __name__ == "__main__":
    main()
