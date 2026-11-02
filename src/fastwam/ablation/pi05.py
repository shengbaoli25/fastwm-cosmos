"""RM71D adapter for the pinned, unmodified official OpenPI JAX trainer.

Importing this module does not import JAX, allocate a GPU, or download assets.
The HTTP layer supplies RGB uint8 images and consumes absolute joint8 actions.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np

UPSTREAM_URL = "https://github.com/Physical-Intelligence/openpi"
UPSTREAM_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"
PRETRAINED_SOURCE = "gs://openpi-assets/checkpoints/pi05_base"
PROMPT = "拔下插头并插入另一插座"
HORIZON = 32


def _rgb(value):
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError("Expected an RGB image with three axes")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError("Expected three RGB channels")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("LeRobot floating RGB must be finite and in [0,1]")
        image = (image * 255).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError("RGB must be uint8 or normalized LeRobot floating RGB")
    return np.ascontiguousarray(image)


@dataclasses.dataclass(frozen=True)
class Joint8Inputs:
    """Shared by the official training pipeline and the loaded policy."""

    def __call__(self, data):
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Expected finite absolute joint8 state")
        global_image = _rgb(data["images"]["global"])
        wrist_image = _rgb(data["images"]["wrist"])
        result = {
            "state": state.copy(),
            "image": {
                "base_0_rgb": global_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(global_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
            "prompt": data.get("prompt", PROMPT),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape != (HORIZON, 8) or not np.isfinite(actions).all():
                raise ValueError("Expected finite 32 x joint8 training actions")
            result["actions"] = actions.copy()
        # Force is deliberately absent: this baseline has no force conditioning.
        return result


@dataclasses.dataclass(frozen=True)
class Joint8Outputs:
    def __call__(self, data):
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[0] != HORIZON or actions.shape[1] < 8:
            raise ValueError("Unexpected pi05 action output shape")
        return {"actions": actions[:, :8].astype(np.float32, copy=True)}


def data_transforms(_model_config):
    from openpi import transforms

    mask = transforms.make_bool_mask(7, -1)
    return transforms.Group(inputs=[Joint8Inputs()], outputs=[Joint8Outputs()]).push(
        inputs=[transforms.DeltaActions(mask)],
        outputs=[transforms.AbsoluteActions(mask)],
    )


def settings(config):
    """Only the fields that determine training/model/data semantics."""
    mode = config.get("finetune_mode", "lora")
    if mode not in {"lora", "full"}:
        raise ValueError("finetune_mode must be lora or full")
    out = {
        "finetune_mode": mode,
        "num_train_steps": int(config.get("num_train_steps", 30000)),
        "batch_size": int(config.get("batch_size", 256)),
        "num_gpus": int(config.get("num_gpus", 4)),
        "per_device_batch_size": int(config.get("per_device_batch_size", 64)),
        "gradient_accumulation_steps": int(config.get("gradient_accumulation_steps", 1)),
        "gpu_model": str(config.get("gpu_model", "H200")),
        "save_interval": int(config.get("save_interval", 5000)),
        "log_interval": int(config.get("log_interval", 10)),
        "seed": int(config.get("seed", 42)),
        "num_workers": int(config.get("num_workers", 2)),
        "fsdp_devices": int(config.get("fsdp_devices", 4)),
    }
    for key in ("num_train_steps", "batch_size", "save_interval", "log_interval", "fsdp_devices", "num_gpus", "per_device_batch_size"):
        if out[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if out["num_workers"] < 0:
        raise ValueError("num_workers cannot be negative")
    if out["batch_size"] != out["num_gpus"] * out["per_device_batch_size"]:
        raise ValueError("batch_size must equal num_gpus * per_device_batch_size")
    if out["gradient_accumulation_steps"] != 1:
        raise ValueError("The official JAX trainer uses one real batch per update; gradient accumulation is unsupported")
    if out["num_gpus"] % out["fsdp_devices"]:
        raise ValueError("num_gpus must be divisible by fsdp_devices")
    if not out["gpu_model"]:
        raise ValueError("gpu_model must identify the requested hardware")
    return out


def verify_training_layout(config, devices, *, process_count, local_device_count):
    """Validate actual allocation independently from optional model imports."""
    opts = settings(config)
    if process_count != 1 or local_device_count != len(devices):
        raise RuntimeError("Use one Python process with all requested GPUs visible; do not use torchrun or multiple hosts")
    if len(devices) != opts["num_gpus"]:
        raise RuntimeError(f"Expected exactly {opts['num_gpus']} visible GPUs, found {len(devices)}")
    if any(d.platform != "gpu" or opts["gpu_model"].lower() not in d.device_kind.lower() for d in devices):
        raise RuntimeError(f"All training devices must be {opts['gpu_model']} GPUs")
    return {
        "device_kinds": [d.device_kind for d in devices],
        "num_gpus": len(devices), "process_count": process_count,
        "global_batch_size": opts["batch_size"], "per_device_batch_size": opts["per_device_batch_size"],
        "gradient_accumulation_steps": 1,
        "mesh_shape": [len(devices) // opts["fsdp_devices"], opts["fsdp_devices"]],
        "mesh_axes": ["batch", "fsdp"], "data_axes": ["batch", "fsdp"],
    }


def check_hardware(config):
    """Check the same official mesh and batch slices used by scripts/train.py."""
    _activate(config)
    import jax
    from openpi.training import sharding

    opts = settings(config)
    report = verify_training_layout(config, jax.devices(), process_count=jax.process_count(),
                                    local_device_count=jax.local_device_count())
    mesh = sharding.make_mesh(opts["fsdp_devices"])
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    indices = data_sharding.devices_indices_map((opts["batch_size"], HORIZON, 32))
    batches = [len(range(opts["batch_size"])[index[0]]) for index in indices.values()]
    if len(batches) != opts["num_gpus"] or any(size != opts["per_device_batch_size"] for size in batches):
        raise RuntimeError(f"Official JAX sharding does not produce the requested per-GPU batch: {batches}")
    report.update(jax_version=jax.__version__, actual_per_device_batch_sizes=batches,
                  upstream_revision=UPSTREAM_REVISION)
    return report


def _name(config):
    return "pi05_rm71d78_" + settings(config)["finetune_mode"]


def _dataset_repo_id(config):
    return config["dataset_repo_id"] if "dataset_repo_id" in config else Path(config["dataset_root"]).name


def _activate(config, *, dataset=False):
    """Require the reviewed source and reject another openpi installation."""
    root = Path(config["upstream_root"]).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"OpenPI revision must be {UPSTREAM_REVISION}, found {revision}")
    if subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"], check=False).returncode:
        raise ValueError("The pinned OpenPI source has tracked modifications")
    imported = sys.modules.get("openpi")
    if imported is not None and not Path(imported.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Another openpi implementation is already imported; use a separate environment/process")
    if dataset:
        data_root = Path(config["dataset_root"]).resolve()
        os.environ["HF_LEROBOT_HOME"] = str(data_root.parent)
    sys.path.insert(0, str(root / "src"))
    import openpi

    if not Path(openpi.__file__).resolve().is_relative_to(root):
        raise RuntimeError("OpenPI was not imported from the pinned source")
    if dataset:
        from lerobot.common.constants import HF_LEROBOT_HOME

        if Path(HF_LEROBOT_HOME).resolve() != data_root.parent:
            raise RuntimeError("LeRobot was imported before dataset-root setup; use a fresh process")
    return root


def make_train_config(config, *, resume=False):
    """Construct official dataclasses; no architecture or optimizer reimplementation."""
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config as official
    from openpi.training import weight_loaders

    opts = settings(config)
    lora = opts["finetune_mode"] == "lora"
    model = Pi0Config(
        pi05=True, action_dim=32, action_horizon=HORIZON,
        paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
    )
    repo_id = _dataset_repo_id(config)
    run_root = Path(config.get("run_root", ".")).resolve()
    data = official.SimpleDataConfig(
        repo_id=repo_id,
        assets=official.AssetsConfig(asset_id=repo_id),
        data_transforms=data_transforms,
        model_transforms=official.ModelTransformFactory(default_prompt=PROMPT),
        base_config=official.DataConfig(
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({
                "state": "observation.state", "actions": "action",
                "images": {"global": "observation.images.global", "wrist": "observation.images.wrist"},
                "prompt": "prompt",
            })]),
            prompt_from_task=True,
            action_sequence_keys=("action",),
        ),
    )
    return official.TrainConfig(
        name=_name(config), project_name="rm71d78-ablation", exp_name="train",
        model=model, data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(Path(config.get("pretrained_checkpoint_path", ".")) / "params")),
        freeze_filter=model.get_freeze_filter(), ema_decay=None if lora else 0.99,
        assets_base_dir=str(run_root / "assets"), checkpoint_base_dir=str(run_root / "checkpoints"),
        num_train_steps=opts["num_train_steps"], batch_size=opts["batch_size"],
        save_interval=opts["save_interval"], keep_period=opts["save_interval"],
        seed=opts["seed"], num_workers=opts["num_workers"], fsdp_devices=opts["fsdp_devices"],
        log_interval=opts["log_interval"], wandb_enabled=False, overwrite=False, resume=resume,
        policy_metadata={"action_space": "joint8", "control_hz": 30, "uses_force": False},
    )


def normalization_path(config):
    return Path(config["run_root"]).resolve() / "assets" / _name(config) / Path(config["dataset_root"]).name / "norm_stats.json"


def _script(root, name):
    spec = importlib.util.spec_from_file_location("_rm71d_pi05_" + name, root / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _binding_path(config):
    return Path(config["training_binding_path"]) if "training_binding_path" in config else Path(config["run_root"]) / "pi05-training-binding.json"


def prepare_statistics(config):
    from .contracts import file_sha256, load_dataset_manifest

    manifest = load_dataset_manifest(config["dataset_root"], verify_files=True)
    root = _activate(config, dataset=True)
    train_config = make_train_config(config)
    output = normalization_path(config)
    if output.exists():
        raise FileExistsError(f"Statistics already exist: {output}; use a new run_root")
    from openpi.training import config as official

    # The released statistics script selects a config by name. Register only for
    # this call, without changing official files or any other config.
    previous = official._CONFIGS_DICT.get(train_config.name)
    official._CONFIGS_DICT[train_config.name] = train_config
    try:
        _script(root, "compute_norm_stats").main(train_config.name)
    finally:
        if previous is None:
            official._CONFIGS_DICT.pop(train_config.name, None)
        else:
            official._CONFIGS_DICT[train_config.name] = previous
    binding = {
        "schema": "rm71d78-pi05-training-v1", "upstream_revision": UPSTREAM_REVISION,
        "run_id": uuid.uuid4().hex,
        "dataset_identity_sha256": manifest["identity_sha256"],
        "dataset_repo_id": Path(config["dataset_root"]).name,
        "settings": settings(config), "normalization_sha256": file_sha256(output),
        "pretrained_source": PRETRAINED_SOURCE, "formal_training_started": False,
    }
    _write_json(_binding_path(config), binding)
    return binding


def train(config, *, resume=False, stop_after_steps=None):
    from .contracts import bind_checkpoint_assets, file_sha256, load_dataset_manifest, path_sha256

    manifest = load_dataset_manifest(config["dataset_root"], verify_files=True)
    binding_path = _binding_path(config)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    _verify_binding(config, binding)
    if binding["dataset_identity_sha256"] != manifest["identity_sha256"]:
        raise ValueError("Dataset differs from the statistics binding")
    if file_sha256(normalization_path(config)) != binding["normalization_sha256"]:
        raise ValueError("Normalization changed since preparation")
    base = Path(config["pretrained_checkpoint_path"]).resolve()
    if not (base / "params").is_dir():
        raise FileNotFoundError("Local official pi05_base/params is required; download it separately")
    base_sha = path_sha256(base / "params")
    if resume and binding.get("pretrained_params_sha256") != base_sha:
        raise ValueError("Resume must use the originally recorded pretrained parameters")
    root = _activate(config, dataset=True)
    train_config = make_train_config(config, resume=resume)
    # This only requests an earlier final save/exit; the planned recipe and
    # learning-rate schedule stay bound so the same run can resume to its target.
    if stop_after_steps is not None:
        stop_after_steps = int(stop_after_steps)
        if not 2 <= stop_after_steps <= train_config.num_train_steps:
            raise ValueError("stop_after_steps must be between 2 and the configured num_train_steps")
        train_config = dataclasses.replace(train_config, num_train_steps=stop_after_steps)
    if resume:
        verify_resume_checkpoint(train_config.checkpoint_dir, binding)
    if not resume and train_config.checkpoint_dir.exists():
        raise FileExistsError("Checkpoint directory already exists; use resume or a new run_root")
    import logging
    logging.basicConfig(level=logging.INFO)
    hardware = check_hardware(config)
    _write_json(Path(config["run_root"]) / "training-hardware.json", hardware)
    _write_json(Path(config["run_root"]) / "last-train-invocation.json", {
        "resume": resume, "stop_after_steps": stop_after_steps,
        "planned_num_train_steps": settings(config)["num_train_steps"],
        "effective_num_train_steps": train_config.num_train_steps,
    })
    logging.info("Verified GPU batch allocation: %s", hardware)
    binding.update(pretrained_params_sha256=base_sha, formal_training_started=True)
    _write_json(binding_path, binding)
    training = _script(root, "train")
    with bind_checkpoint_assets(training._checkpoints._normalize, binding):
        training.main(train_config)


def verify_resume_checkpoint(checkpoint_dir, binding):
    candidates = [p for p in Path(checkpoint_dir).glob("*") if p.name.isdigit()
                  and (p / "train_state").is_dir() and (p / "params").is_dir()]
    # The pinned official initializer treats an isolated step 0 as a fresh run.
    if not candidates or max(int(p.name) for p in candidates) == 0:
        raise FileNotFoundError("No resumable optimizer checkpoint exists; refusing a fresh run disguised as resume")
    latest = max(candidates, key=lambda p: int(p.name))
    embedded = json.loads((latest / "assets" / "ablation_training_binding.json").read_text(encoding="utf-8"))
    if embedded != binding:
        raise ValueError("Resume checkpoint belongs to a different pi05 training run or recipe")
    return latest


def _verify_binding(config, binding):
    if binding.get("schema") != "rm71d78-pi05-training-v1" or binding.get("upstream_revision") != UPSTREAM_REVISION:
        raise ValueError("Invalid pi05 training binding")
    try:
        uuid.UUID(hex=binding["run_id"])
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        raise ValueError("Pi05 training binding requires a unique run_id") from error
    if binding.get("settings") != settings(config):
        raise ValueError("Pi05 settings differ from the saved training settings")
    if ("dataset_repo_id" in config or "dataset_root" in config) and binding.get("dataset_repo_id") != _dataset_repo_id(config):
        raise ValueError("Dataset asset ID differs from the trained asset ID")
    if config.get("dataset_manifest_sha256") and config["dataset_manifest_sha256"] != binding["dataset_identity_sha256"]:
        raise ValueError("Deployment manifest identity differs from the trained dataset")


def wrap_policy(policy):
    def predict(sample):
        # Do not pass force/history through even if supplied by the wire protocol.
        observation = {k: sample[k] for k in ("state", "images", "prompt")}
        actions = np.asarray(policy.infer(observation)["actions"], dtype=np.float32)
        if actions.shape != (HORIZON, 8) or not np.isfinite(actions).all():
            raise ValueError("Pi05 returned non-finite or incorrectly shaped actions")
        # The common server applies the shared physical gripper [0,1] bound.
        return actions
    return predict


def load_predictor(config):
    from .contracts import file_sha256

    binding = json.loads(_binding_path(config).read_text(encoding="utf-8"))
    _verify_binding(config, binding)
    if not binding.get("formal_training_started"):
        raise ValueError("This run has no recorded real training start")
    checkpoint = Path(config["checkpoint_path"]).resolve()
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError("Expected a JAX step checkpoint with params/")
    embedded = json.loads((checkpoint / "assets" / "ablation_training_binding.json").read_text(encoding="utf-8"))
    if embedded != binding:
        raise ValueError("Checkpoint belongs to a different pi05 training run or recipe")
    # Read the stats bundled by the official trainer, never another experiment.
    checkpoint_stats = checkpoint / "assets" / binding["dataset_repo_id"] / "norm_stats.json"
    if file_sha256(checkpoint_stats) != binding["normalization_sha256"]:
        raise ValueError("Checkpoint normalization differs from training")
    if config.get("normalization_path") and file_sha256(config["normalization_path"]) != binding["normalization_sha256"]:
        raise ValueError("Requested normalization does not match the checkpoint")
    _activate(config)
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import AssetsConfig

    inference_steps = int(config.get("num_inference_steps", 10))
    if inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    runtime = dict(config, dataset_repo_id=binding["dataset_repo_id"])
    train_config = make_train_config(runtime)
    train_config = dataclasses.replace(train_config, data=dataclasses.replace(
        train_config.data, assets=AssetsConfig(assets_dir=str(checkpoint / "assets"), asset_id=binding["dataset_repo_id"]),
    ))
    policy = create_trained_policy(
        train_config, checkpoint,
        sample_kwargs={"num_steps": inference_steps},
    )
    return wrap_policy(policy)


def plan(config):
    opts = settings(config)
    return {
        "backend": "pi05", "upstream_url": UPSTREAM_URL, "upstream_revision": UPSTREAM_REVISION,
        "pretrained_source": PRETRAINED_SOURCE, "settings": opts,
        "normalization_path": str(normalization_path(config)),
        "checkpoint_directory": str(Path(config["run_root"]) / "checkpoints" / _name(config) / "train"),
        "uses_force": False, "action_space": "joint8", "action_horizon": HORIZON,
        "model_action_dim": 32, "control_hz": 30,
        "training_started": False, "runtime_verified": False,
        "hardware_budget": (f"Single Linux host, {opts['num_gpus']} x {opts['gpu_model']}; global batch {opts['batch_size']} / "
                            f"per-GPU batch {opts['per_device_batch_size']}, no accumulation; measure actual memory"),
        "launch_processes": 1, "mesh_shape": [opts["num_gpus"] // opts["fsdp_devices"], opts["fsdp_devices"]],
    }
