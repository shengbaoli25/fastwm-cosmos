"""RealMan Joint8 adapter for the pinned, official JAX ForceVLA release.

The upstream pi0 force expert, LIMoE, loss, optimizer and checkpoint machinery
remain upstream code. Heavy imports are deferred so plan/tests need only NumPy.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from typing import Any

import numpy as np

UPSTREAM_URL = "https://github.com/ft-robotic/ForceVLA"
UPSTREAM_REVISION = "79de88fdd0651d957a2b5212cae8995159c5ad65"
MODEL_SOURCE_SHA256 = "15ac88d43012fce1fe64b0b59c74f08c704b0aefe58fb3c7dad63f386b697aad"
LEROBOT_REVISION = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
CONFIG_NAME = "rm71d78_forcevla_joint8"
PROMPT = "拔下插头并插入另一插座"


def execution_profile(config: dict) -> dict:
    """JAX's batch_size is global; DATA_AXIS shards it across all devices."""
    count = int(config.get("gpu_count", 4))
    per_gpu = int(config.get("per_gpu_batch_size", 64))
    batch = int(config.get("batch_size", 256))
    fsdp = int(config.get("fsdp_devices", 4))
    steps = int(config.get("num_train_steps", 30000))
    if min(count, per_gpu, fsdp, steps) < 1 or batch != count * per_gpu or count % fsdp:
        raise ValueError("Require global batch = gpu_count * per_gpu_batch_size and an FSDP divisor of gpu_count")
    return dict(gpu_count=count, gpu_model=config.get("gpu_model", "H200"),
                per_gpu_batch_size=per_gpu, global_batch_size=batch, fsdp_devices=fsdp,
                mesh_shape=[count // fsdp, fsdp], data_axes=["batch", "fsdp"],
                gradient_accumulation_steps=1, num_train_steps=steps)


def verify_training_devices(config: dict, devices, process_count: int) -> dict:
    """Reject silently running the requested physical batch on fewer GPUs."""
    profile = execution_profile(config)
    if process_count != 1:
        raise RuntimeError("The pinned ForceVLA loader supports one process on one multi-GPU host")
    devices = list(devices)
    if len(devices) != profile["gpu_count"] or any(d.platform != "gpu" for d in devices):
        raise RuntimeError(f"Expected exactly {profile['gpu_count']} visible CUDA GPUs")
    kinds = [str(d.device_kind) for d in devices]
    if any(str(profile["gpu_model"]).upper() not in kind.upper() for kind in kinds):
        raise RuntimeError(f"Expected {profile['gpu_model']} devices, found {kinds}")
    return {**profile, "visible_devices": kinds, "jax_process_count": process_count}


def _vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite [{size}], got {result.shape}")
    return result


def _image(value) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError("RGB image must have three dimensions")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError("RGB image must have three channels")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("LeRobot float RGB must be in [0,1]")
        image = np.rint(image * 255).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError("RGB must be uint8 or LeRobot float [0,1]")
    return np.ascontiguousarray(image)


@dataclasses.dataclass(frozen=True)
class Joint8Inputs:
    """One shared transform for training and serving; force is state[8:14]."""

    def __call__(self, data: dict) -> dict:
        state = _vector(data["state"], 8, "state")
        force = _vector(data["force"], 6, "force")
        if not 0 <= state[7] <= 1:
            raise ValueError("state gripper openness must be in [0,1]")
        packed = np.zeros(32, dtype=np.float32)
        packed[:8], packed[8:14] = state, force
        base, wrist = _image(data["image"]), _image(data["wrist_image"])
        result = {
            "state": packed,
            "image": {"base_0_rgb": base, "left_wrist_0_rgb": wrist,
                      "right_wrist_0_rgb": np.zeros_like(base)},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_,
                           "right_wrist_0_rgb": np.False_},
            "prompt": data.get("prompt", PROMPT),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape != (32, 8) or not np.isfinite(actions).all():
                raise ValueError("actions must be finite [32,8]")
            if np.any((actions[:, 7] < 0) | (actions[:, 7] > 1)):
                raise ValueError("action gripper openness must be in [0,1]")
            result["actions"] = np.pad(actions, ((0, 0), (0, 24)))
        return result


@dataclasses.dataclass(frozen=True)
class Joint8Outputs:
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.shape != (32, 32) or not np.isfinite(actions).all():
            raise ValueError("ForceVLA must return finite [32,32] padded actions")
        return {"actions": actions[:, :8].copy()}


def patch_model_source(source: str) -> str:
    """Apply three reviewed fixes to an exact upstream source, in memory only.

    This preserves the pretrained 32-channel projections. Joint8 needs eight
    state channels and moves force to [8:14]. Upstream sampling incorrectly
    concatenates a None cached-prefix return; use the saved prefix tensor.
    """
    source = source.replace("\r\n", "\n")
    if hashlib.sha256(source.encode()).hexdigest() != MODEL_SOURCE_SHA256:
        raise ValueError("ForceVLA pi0_force.py does not match the reviewed upstream revision")
    old = "observations = observations.at[:, :7].set(obs.state[:, :7])"
    new = "observations = observations.at[:, :8].set(obs.state[:, :8])"
    if source.count(old) != 1 or source.count("obs.state[:, 7:13]") != 1:
        raise ValueError("ForceVLA state/force split changed upstream")
    source = source.replace(old, new).replace("obs.state[:, 7:13]", "obs.state[:, 8:14]")
    before, after = source.split("    def sample_actions(", 1)
    bug = "jnp.concatenate([prefix_out, force_tokens], axis=1)"
    if after.count(bug) != 1:
        raise ValueError("ForceVLA cached-prefix sampling implementation changed")
    after = after.replace(bug, "jnp.concatenate([prefix_out_fix, force_tokens], axis=1)")
    return before + "    def sample_actions(" + after


def _activate_upstream(upstream_root: str | Path):
    root = Path(upstream_root).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"ForceVLA requires upstream commit {UPSTREAM_REVISION}, got {revision}")
    subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", "src", "scripts", "flaxformer"], check=True)
    existing = sys.modules.get("openpi")
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Another openpi is loaded; use the dedicated ForceVLA process/environment")
    for relative in ("flaxformer", "packages/openpi-client/src", "src"):
        sys.path.insert(0, str(root / relative))
    model_source = patch_model_source((root / "src/openpi/models/pi0_force.py").read_text(encoding="utf-8"))
    name = "fastwam.ablation._forcevla_joint8_model"
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__file__ = str(root / "src/openpi/models/pi0_force.py")
        sys.modules[name] = module
        try:
            exec(compile(model_source, module.__file__, "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def _manifest(config: dict) -> dict:
    from fastwam.ablation.contracts import load_dataset_manifest
    manifest = load_dataset_manifest(config["dataset_root"])
    if manifest["identity_sha256"] != config.get("dataset_manifest_sha256"):
        raise ValueError("ForceVLA config is not bound to this dataset manifest")
    return manifest


def _normalization(config: dict, *, required: bool = True):
    from fastwam.ablation.contracts import file_sha256
    path = Path(config["normalization_path"]).resolve()
    if not path.exists() and not required:
        return None
    if file_sha256(path) != config.get("normalization_sha256"):
        raise ValueError("ForceVLA normalization hash mismatch; bind generated statistics before training/serving")
    from openpi.shared import normalize
    stats = normalize.deserialize_json(path.read_text(encoding="utf-8"))
    if set(stats) != {"state", "actions"}:
        raise ValueError("ForceVLA normalization must contain state and actions only")
    for key in stats:
        for field in ("mean", "std"):
            array = np.asarray(getattr(stats[key], field))
            if array.shape != (32,) or not np.isfinite(array).all():
                raise ValueError(f"Invalid ForceVLA {key} {field}")
    if np.any(np.asarray(stats["state"].std) < 0) or np.any(np.asarray(stats["actions"].std) < 0):
        raise ValueError("Negative normalization standard deviation")
    return stats


def _verify_lerobot():
    """Do not silently train with a newer incompatible LeRobot package."""
    import importlib.metadata
    from urllib.parse import unquote, urlparse
    dist = importlib.metadata.distribution("lerobot")
    raw = dist.read_text("direct_url.json")
    metadata = json.loads(raw) if raw else {}
    revision = metadata.get("vcs_info", {}).get("commit_id")
    if revision is None and metadata.get("url", "").startswith("file:"):
        path = unquote(urlparse(metadata["url"]).path)
        if os.name == "nt" and path.startswith("/") and len(path) > 2 and path[2] == ":":
            path = path[1:]
        revision = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    if revision != LEROBOT_REVISION:
        raise ValueError(f"Install the reviewed LeRobot commit {LEROBOT_REVISION}; found {revision!r}")


def build_train_config(config: dict, *, resume: bool = False, for_stats: bool = False, for_inference: bool = False):
    manifest = {"identity_sha256": config["dataset_manifest_sha256"]} if for_inference else _manifest(config)
    dataset = Path(config["dataset_root"]).resolve()
    # ForceVLA's pinned LeRobot loader resolves repo_id relative to this variable.
    # Set it before importing any LeRobot modules.
    if not for_inference:
        os.environ["HF_LEROBOT_HOME"] = str(dataset.parent)
    model_module = _activate_upstream(config["upstream_root"])
    from openpi.training import config as upstream_config
    from openpi.training import weight_loaders
    from openpi import transforms
    method = config.get("training_method", "lora")
    if method not in {"lora", "full"}:
        raise ValueError("training_method must be lora or full")
    model = model_module.Pi0_GuidanceConfig(
        paligemma_variant="gemma_2b_lora" if method == "lora" else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if method == "lora" else "gemma_300m",
        action_horizon=32, action_dim=32,
    )
    stats = None if for_stats else _normalization(config)

    @dataclasses.dataclass(frozen=True)
    class RealManForceData(upstream_config.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            delta_mask = transforms.make_bool_mask(7, -1)
            return upstream_config.DataConfig(
                repo_id=dataset.name, asset_id=dataset.name, norm_stats=stats,
                repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({
                    "state": "observation.state", "force": "observation.force",
                    "image": "observation.images.global", "wrist_image": "observation.images.wrist",
                    "actions": "action", "prompt": "prompt",
                })]),
                data_transforms=transforms.Group(
                    inputs=[Joint8Inputs(), transforms.DeltaActions(delta_mask)],
                    outputs=[transforms.AbsoluteActions(delta_mask), Joint8Outputs()],
                ),
                model_transforms=upstream_config.ModelTransformFactory()(model_config),
                action_sequence_keys=("action",), prompt_from_task=True, use_quantile_norm=False,
            )

    run = Path(config["run_root"]).resolve()
    values = dict(
        name=CONFIG_NAME, exp_name=config.get("experiment_name", "rm71d78"),
        project_name="rm71d78-ablation", model=model, data=RealManForceData(repo_id=dataset.name),
        weight_loader=weight_loaders.Pi0GuidanceWeightLoader(config.get("base_params_path", "")),
        freeze_filter=model.get_freeze_filter(), ema_decay=None if method == "lora" else 0.99,
        assets_base_dir=str(run / "assets"), checkpoint_base_dir=str(run / "checkpoints"),
        num_train_steps=int(config.get("num_train_steps", 30000)), batch_size=int(config.get("batch_size", 256)),
        save_interval=int(config.get("save_interval", 2000)), keep_period=int(config.get("keep_period", 10000)),
        log_interval=int(config.get("log_interval", 10)), seed=int(config.get("seed", 42)),
        num_workers=0, fsdp_devices=int(config.get("fsdp_devices", 4)), resume=resume,
        overwrite=False, wandb_enabled=False,
        policy_metadata={"dataset_manifest_sha256": manifest["identity_sha256"], "action_space": "joint8",
                         "force_input": True, "upstream_revision": UPSTREAM_REVISION},
    )
    if any(values[k] < 1 for k in ("num_train_steps", "batch_size", "save_interval", "keep_period", "log_interval", "fsdp_devices")):
        raise ValueError("Training step, batch and interval counts must be positive")
    return upstream_config.TrainConfig(**values)


def load_predictor(config: dict):
    """Load a real Orbax training checkpoint and return raw joint8 predictions."""
    from fastwam.ablation.contracts import file_sha256
    binding = json.loads(Path(config["training_binding_path"]).read_text(encoding="utf-8"))
    if binding.get("upstream_revision") != UPSTREAM_REVISION:
        raise ValueError("Checkpoint training binding belongs to a different ForceVLA implementation")
    for key in ("dataset_manifest_sha256", "normalization_sha256"):
        if binding.get(key) != config.get(key):
            raise ValueError(f"Checkpoint training binding mismatch: {key}")
    checkpoint = Path(config["checkpoint_path"]).resolve()
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError("ForceVLA checkpoint_path must be a saved step containing params/")
    embedded_binding = json.loads((checkpoint / "assets/ablation_training_binding.json").read_text(encoding="utf-8"))
    if embedded_binding != binding:
        raise ValueError("Checkpoint embedded training binding differs from the selected experiment")
    bound_stats = checkpoint / "assets" / Path(binding["dataset_root"]).name / "norm_stats.json"
    if file_sha256(bound_stats) != file_sha256(config["normalization_path"]):
        raise ValueError("Checkpoint normalization does not match the declared training statistics")
    # Architecture/method come from the frozen training record. Only runtime
    # asset locations may change when copying the deployment to another host.
    inference_config = dict(binding)
    for key in ("upstream_root", "normalization_path"):
        inference_config[key] = config[key]
    train_config = build_train_config(inference_config, for_inference=True)
    from openpi.policies import policy_config
    steps = int(config.get("num_inference_steps", 10))
    if steps < 1:
        raise ValueError("num_inference_steps must be positive")
    policy = policy_config.create_trained_policy(train_config, checkpoint, sample_kwargs={"num_steps": steps})

    def predict(sample: dict) -> np.ndarray:
        inputs = {"state": sample["state"], "force": sample["force"],
                  "image": sample["images"]["global"], "wrist_image": sample["images"]["wrist"],
                  "prompt": sample.get("prompt", PROMPT)}
        result = np.asarray(policy.infer(inputs)["actions"], dtype=np.float32)
        if result.shape != (32, 8) or not np.isfinite(result).all():
            raise ValueError("Invalid ForceVLA prediction; refusing to serve it")
        return result

    return predict


def _script(root: str, filename: str):
    path = Path(root).resolve() / "scripts" / filename
    spec = importlib.util.spec_from_file_location(f"_forcevla_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _check_resume_checkpoint(checkpoint_dir: Path, contract: dict):
    candidates = sorted((p for p in checkpoint_dir.glob("[0-9]*")
                         if p.name.isdigit() and (p / "train_state").is_dir()),
                        key=lambda p: int(p.name))
    if not candidates:
        raise FileNotFoundError("Resume requires a real saved train_state checkpoint")
    # The pinned upstream manager treats (0,) as no checkpoint and starts fresh.
    if int(candidates[-1].name) == 0:
        raise ValueError("The official trainer cannot resume a step-0-only run; refusing a silent fresh start")
    for checkpoint in candidates:
        embedded = json.loads((checkpoint / "assets/ablation_training_binding.json").read_text(encoding="utf-8"))
        if embedded != contract:
            raise ValueError(f"Resume checkpoint has a different embedded training binding: {checkpoint}")


def _bounded_training_config(config, stop_after_steps: int | None):
    if stop_after_steps is None:
        return config
    if not 2 <= stop_after_steps <= config.num_train_steps:
        raise ValueError("stop_after_steps must be between 2 and the configured target steps")
    # Only bound this invocation; retain the original schedule and run identity.
    return dataclasses.replace(config, num_train_steps=stop_after_steps)


def run(config: dict, operation: str, *, stop_after_steps: int | None = None) -> dict[str, Any]:
    """Only explicit train/resume operations call the official training loop."""
    if stop_after_steps is not None and operation not in {"train", "resume"}:
        raise ValueError("stop_after_steps is only supported for train/resume")
    if operation == "plan":
        return {"baseline": "forcevla", "upstream": UPSTREAM_URL, "revision": UPSTREAM_REVISION,
                "training_started": False, "method": config.get("training_method", "lora"),
                "num_train_steps": int(config.get("num_train_steps", 30000)),
                "batch_size": int(config.get("batch_size", 256)), "action_horizon": 32,
                "execution_profile": execution_profile(config),
                "pretrained_initialization": "official pi0_base + initialized force/LIMoE and LoRA",
                "force_input": "normalized [Fx,Fy,Fz,Tx,Ty,Tz] from observation.force",
                "source_episodes": 78, "evaluation": "training-fit, no held-out claim",
                "runtime_validation": "not_run", "platform": "Linux / separate ForceVLA JAX environment"}
    if operation not in {"stats", "train", "resume"}:
        raise ValueError(f"Unknown operation: {operation}")
    cfg = build_train_config(config, resume=operation == "resume", for_stats=operation == "stats")
    cfg = _bounded_training_config(cfg, stop_after_steps)
    _verify_lerobot()
    from fastwam.ablation.contracts import file_sha256, path_sha256
    if operation == "stats":
        from openpi.shared import normalize
        from openpi.training import data_loader
        dataset = data_loader.create_torch_dataset(cfg.data.create(cfg.assets_dirs, cfg.model), 32, cfg.model)
        data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
        transformed = data_loader.TransformedDataset(dataset, [*data_cfg.repack_transforms.inputs,
                                                                *data_cfg.data_transforms.inputs])
        accumulators = {key: normalize.RunningStats() for key in ("state", "actions")}
        for index in range(len(transformed)):
            item = transformed[index]
            for key, accumulator in accumulators.items():
                accumulator.update(np.asarray(item[key]))
        path = Path(config["normalization_path"]).resolve()
        if path.name != "norm_stats.json":
            raise ValueError("ForceVLA normalization_path must end with norm_stats.json")
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing training statistics: {path}")
        normalize.save(path.parent, {key: value.get_statistics() for key, value in accumulators.items()})
        receipt = {"operation": "stats", "training_started": False,
                   "dataset_manifest_sha256": config["dataset_manifest_sha256"],
                   "normalization_path": str(path), "normalization_sha256": file_sha256(path),
                   "samples": len(transformed), "upstream_revision": UPSTREAM_REVISION}
        path.with_suffix(".binding.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt
    base = Path(config["base_params_path"]).resolve()
    if path_sha256(base) != config.get("base_params_sha256"):
        raise ValueError("Base pi0 params hash mismatch")
    binding = Path(config["normalization_path"]).with_suffix(".binding.json")
    norm_binding = json.loads(binding.read_text(encoding="utf-8"))
    if norm_binding["dataset_manifest_sha256"] != config["dataset_manifest_sha256"] or norm_binding["normalization_sha256"] != config["normalization_sha256"]:
        raise ValueError("Normalization was computed for a different dataset")
    import jax
    device_receipt = verify_training_devices(config, jax.devices(), jax.process_count())
    run_root = Path(config["run_root"]).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    contract_path = run_root / "forcevla_training_config.json"
    contract = {**config, "upstream_revision": UPSTREAM_REVISION}
    if operation == "resume":
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("Resume config differs from the frozen training configuration")
        _check_resume_checkpoint(cfg.checkpoint_dir, contract)
    else:
        with contract_path.open("x", encoding="utf-8") as handle:
            json.dump(contract, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    from datetime import datetime, timezone
    launch_receipt = dict(device_receipt, operation=operation, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                          upstream_revision=UPSTREAM_REVISION, dataset_manifest_sha256=config["dataset_manifest_sha256"],
                          checkpoint_dir=str(cfg.checkpoint_dir), optimization_completed=False,
                          invocation_stop_after_steps=stop_after_steps,
                          effective_num_train_steps=cfg.num_train_steps)
    (run_root / "forcevla_launch_receipt.json").write_text(json.dumps(launch_receipt, indent=2) + "\n", encoding="utf-8")
    training = _script(config["upstream_root"], "train.py")
    import logging
    logging.basicConfig(level=logging.INFO)
    from fastwam.ablation.contracts import bind_checkpoint_assets
    with bind_checkpoint_assets(training._checkpoints._normalize, contract):
        training.main(cfg)
    return {"operation": operation, "training_started": True, "training_returned": True,
            "checkpoint_dir": str(cfg.checkpoint_dir), "upstream_revision": UPSTREAM_REVISION,
            "configured_target_steps": execution_profile(config)["num_train_steps"],
            "invocation_stop_after_steps": stop_after_steps}
