"""RealMan joint8 adaptation around the repository's official FastWAM core.

This module is cheap to import: planning/stats never load Torch or model weights.
The baseline does not consume force or synthesize future frames at inference.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from types import MethodType
from unittest.mock import patch
from contextlib import contextmanager, nullcontext

from .contracts import file_sha256, identity_sha256, load_dataset_manifest, write_json

UPSTREAM_REVISION = "7faa71108368fbb3b6885649f112af607427a2d4"
RELEASE_SHA256 = "1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579"
SCHEMA = "fastwam-realman-joint8-v1"
CAMERAS = ("global", "wrist")
HEAD_KEYS = frozenset({
    "mixtures.action.action_encoder.weight", "mixtures.action.action_encoder.bias",
    "mixtures.action.head.weight", "mixtures.action.head.bias",
})
VAE_RELATIVE_PATH = "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"
MODEL_RECIPE_FIELDS = ("model_id", "tokenizer_model_id", "tokenizer_max_len", "load_text_encoder",
                       "proprio_dim", "redirect_common_files", "mot_checkpoint_mixed_attn",
                       "skip_dit_load_from_pretrain", "video_dit_config", "action_dit_config",
                       "video_scheduler", "action_scheduler", "loss")
TRAIN_RECIPE_FIELDS = ("batch_size", "gradient_accumulation_steps", "seed", "mixed_precision",
                       "learning_rate", "weight_decay", "max_grad_norm", "max_steps", "num_epochs",
                       "lr_scheduler_type", "num_workers", "eval_every", "ablation_num_processes")


@contextmanager
def rng_neutral_training_logging(runtime):
    """Keep native checkpoint RNG intact across Rich save/restore log records."""
    original = runtime.setup_logging

    def setup_logging(*args, **kwargs):
        # Rich path hyperlinks allocate IDs through Python's global random
        # module, including Accelerate's log immediately after restoring RNG.
        # Keep visible source paths, but disable that link allocation for the
        # complete ablation worker lifetime. Never reset RNG to hide drift.
        options = dict(kwargs.pop("rich_handler_kwargs", None) or {})
        options["enable_link_path"] = False
        return original(*args, rich_handler_kwargs=options, **kwargs)

    with patch.object(runtime, "setup_logging", setup_logging):
        yield


def execution_profile(cfg):
    profile = {"num_processes": int(cfg.get("ablation_num_processes", 4)),
               "per_gpu_batch_size": int(cfg.get("batch_size", 64)),
               "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 1)),
               "max_steps": int(cfg.get("max_steps", 30000)), "gpu_model": "H200"}
    if (profile["num_processes"], profile["per_gpu_batch_size"], profile["gradient_accumulation_steps"]) != (4, 64, 1):
        raise ValueError("This run requires four H200 GPUs with real per-GPU batch 64 and accumulation 1")
    if profile["max_steps"] < 1:
        raise ValueError("max_steps must be positive; use a separate short smoke config before the 30000-step run")
    profile["global_batch_size"] = 256
    return profile


def check_hardware(cfg):
    import torch
    profile = execution_profile(cfg)
    if not torch.cuda.is_available() or torch.cuda.device_count() != profile["num_processes"]:
        raise RuntimeError("Expected exactly four visible CUDA H200 GPUs")
    devices = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    if any("H200" not in device.name.upper() for device in devices):
        raise RuntimeError(f"Expected H200 GPUs, found {[device.name for device in devices]}")
    return {**profile, "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
            "device_names": [d.name for d in devices], "device_memory_bytes": [d.total_memory for d in devices]}


def verify_prepared_batch(trainer, cfg):
    """Read the actual DeepSpeed engine after Accelerate resolved its auto fields."""
    profile = execution_profile(cfg)
    engine = trainer.model
    actual = {"num_processes": int(trainer.accelerator.num_processes),
              "per_gpu_batch_size": int(engine.train_micro_batch_size_per_gpu()),
              "gradient_accumulation_steps": int(engine.gradient_accumulation_steps()),
              "global_batch_size": int(engine.train_batch_size())}
    if any(actual[k] != profile[k] for k in actual):
        raise RuntimeError(f"DeepSpeed did not retain the requested physical batch: {actual}")
    return actual


@contextmanager
def bounded_training_invocation(trainer_class, cfg, stop_after_steps=None, *, smoke_audit=False):
    """Keep full-target scheduler/recipe creation, bound only the prepared loop."""
    target = execution_profile(cfg)["max_steps"]
    if stop_after_steps is not None and not 2 <= int(stop_after_steps) <= target:
        raise ValueError("stop_after_steps must be between 2 and the configured max_steps")
    original = trainer_class.train

    def train(trainer):
        actual = verify_prepared_batch(trainer, cfg)
        planned = trainer.max_steps
        limit = target if stop_after_steps is None else int(stop_after_steps)
        if planned != target or int(trainer.global_step) >= limit:
            raise ValueError("Training target differs from the frozen recipe or resume already reached this invocation's limit")
        if trainer.accelerator.is_main_process:
            write_json(Path(cfg.output_dir) / "training-execution.json", {
                **actual, "configured_max_steps": target, "stop_after_steps": stop_after_steps,
                "restored_global_step": int(trainer.global_step), "effective_loop_target": limit,
            })
        # Wan22Trainer creates scheduler and restores optimizer/RNG before train().
        # Its original final-save path now exits at the invocation bound.
        trainer.max_steps = limit
        try:
            if smoke_audit:
                from .training_audit import audit_prepared_training
                audit = audit_prepared_training(trainer, cfg, limit)
            else:
                audit = nullcontext()
            with audit:
                return original(trainer)
        finally:
            trainer.max_steps = planned

    with patch.object(trainer_class, "train", train):
        yield


def semantic_recipe(cfg):
    """Bind train/serve semantics while allowing assets and run folders to move."""
    from omegaconf import OmegaConf
    plain = OmegaConf.to_container(cfg, resolve=True)
    data = {k: v for k, v in plain["data"]["train"].items()
            if k not in {"dataset_dirs", "pretrained_norm_stats", "text_embedding_cache_dir"}}
    return {"model": {k: plain["model"][k] for k in MODEL_RECIPE_FIELDS}, "data": data,
            "training": {k: plain[k] for k in TRAIN_RECIPE_FIELDS},
            "model_init_seed": int(plain["model"]["model_init_seed"])}


def text_cache_path(directory, prompt):
    digest = hashlib.sha256(PROMPT_TEMPLATE.format(task=prompt).encode("utf-8")).hexdigest()
    return Path(directory) / f"{digest}.t5_len128.wan22ti2v5b.pt"


def convert_stats(raw, dataset_identity):
    """Convert verified canonical raw-space stats to the official normalizer schema."""
    result = {"ablation": {"schema": SCHEMA, "dataset_identity": dataset_identity,
                           "normalization": "z-score", "action_space": "joint8"}}
    for kind, key in (("state", "observation.state"), ("action", "action")):
        field = {}
        for name in ("mean", "std", "min", "max", "q01", "q99"):
            values = raw[key][name]
            if len(values) != 8 or not all(math.isfinite(float(v)) for v in values):
                raise ValueError(f"{key}.{name} must contain eight finite values")
            if name == "std" and any(float(v) < 0 for v in values):
                raise ValueError("Negative standard deviation")
            field[f"global_{name}"] = values
        result[kind] = {"default": field}
    return result


def write_stats(dataset_root, output_path):
    manifest = load_dataset_manifest(dataset_root, verify_files=True)
    raw = json.loads((Path(dataset_root) / "meta/stats.json").read_text(encoding="utf-8"))
    stats = convert_stats(raw, manifest["identity_sha256"])
    write_json(output_path, stats)
    return stats


def _verified_stats(path, manifest):
    stats = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {"schema": SCHEMA, "dataset_identity": manifest["identity_sha256"],
                "normalization": "z-score", "action_space": "joint8"}
    if stats.get("ablation") != expected:
        raise ValueError("FastWAM statistics do not match the canonical dataset or joint8 recipe")
    # Re-run numeric validation instead of trusting the identity label alone.
    raw = {key: {name: stats[kind]["default"][f"global_{name}"]
                 for name in ("mean", "std", "min", "max", "q01", "q99")}
           for kind, key in (("state", "observation.state"), ("action", "action"))}
    convert_stats(raw, manifest["identity_sha256"])
    return stats


def adaptation_state(target, source):
    """Retain every backbone tensor; only explicitly named robot heads reset.

    Even the 7D source head's bias is discarded: its pose semantics are not
    joint8. The complete proprio projection resets for the same reason.
    """
    if set(source) != set(target):
        raise ValueError(f"Official policy keys differ: missing={sorted(set(target)-set(source))}, "
                         f"unexpected={sorted(set(source)-set(target))}")
    if not HEAD_KEYS.issubset(target):
        raise ValueError("Official FastWAM action head names changed")
    source_input = source["mixtures.action.action_encoder.weight"]
    source_output = source["mixtures.action.head.weight"]
    if source_input.shape[1] != 7 or source_output.shape[0] != 7:
        raise ValueError("Expected the pinned 7D LIBERO release")
    if target["mixtures.action.action_encoder.weight"].shape[1] != 8:
        raise ValueError("Target action encoder must have eight inputs")
    if target["mixtures.action.head.weight"].shape[0] != 8:
        raise ValueError("Target action head must have eight outputs")
    retained = {}
    for key, value in source.items():
        if key in HEAD_KEYS:
            continue
        if value.shape != target[key].shape:
            raise ValueError(f"Backbone tensor shape differs: {key}")
        retained[key] = value
    # Full strict state load: omitted heads are replaced with seeded target init.
    return {key: target[key] if key in HEAD_KEYS else retained[key] for key in target}


def load_release(model, path):
    import torch
    if file_sha256(path) != RELEASE_SHA256:
        raise ValueError("FastWAM release SHA256 does not match the pinned official LIBERO checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "mot" not in payload or "proprio_encoder" not in payload:
        raise ValueError("Expected the complete official policy, not an ActionDiT/Wan initialization")
    model.mot.load_state_dict(adaptation_state(model.mot.state_dict(), payload["mot"]), strict=True)
    # Do not load the LIBERO proprio projection: its eight fields differ from joint8.


def _save_bound_checkpoint(self, path, optimizer=None, step=None):
    import torch
    payload = {"mot": self.mot.state_dict(), "proprio_encoder": self.proprio_encoder.state_dict(),
               "step": step, "torch_dtype": str(self.torch_dtype), "ablation": self.ablation_binding}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_joint8_checkpoint(model, path, expected_binding, optimizer=None):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("ablation") != expected_binding:
        raise ValueError("Checkpoint/data/statistics/VAE/text identity mismatch; not a deployable joint8 checkpoint")
    if "mot" not in payload or "proprio_encoder" not in payload:
        raise ValueError("Joint8 checkpoint must contain complete MoT and proprio weights")
    model.mot.load_state_dict(payload["mot"], strict=True)
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def _load_bound_checkpoint(self, path, optimizer=None):
    return load_joint8_checkpoint(self, path, self.ablation_binding, optimizer)


def build_binding(dataset_root, normalization_path, asset_root, text_embedding_cache_dir,
                  *, training_recipe, deployment_binding=None):
    manifest = (load_dataset_manifest(dataset_root, verify_files=True) if deployment_binding is None
                else {"identity_sha256": deployment_binding["dataset_identity"],
                      "prompt": deployment_binding["prompt"]})
    _verified_stats(normalization_path, manifest)
    vae_path = Path(asset_root) / VAE_RELATIVE_PATH
    cache_path = text_cache_path(text_embedding_cache_dir, manifest["prompt"])
    if not vae_path.is_file() or not cache_path.is_file():
        raise FileNotFoundError(f"Stage local VAE and T5 cache before training/inference: {vae_path}; {cache_path}")
    binding = {"schema": SCHEMA, "baseline": "fastwam", "action_space": "joint8",
            "action_dim": 8, "state_dim": 8, "action_horizon": 32, "control_hz": 30,
            "camera_names": list(CAMERAS), "video_size": [224, 448],
            "dataset_identity": manifest["identity_sha256"],
            "prompt": manifest["prompt"],
            "normalization_sha256": file_sha256(normalization_path),
            "vae_sha256": file_sha256(vae_path), "text_sha256": file_sha256(cache_path),
            "upstream_revision": UPSTREAM_REVISION, "release_sha256": RELEASE_SHA256,
            "training_recipe": training_recipe,
            "training_recipe_sha256": identity_sha256(training_recipe),
            "initialization": "retain_all_backbones_reset_action_io_and_proprio"}
    if deployment_binding is not None and binding != deployment_binding:
        raise ValueError("Deployment binding differs from current data statistics, VAE, text or recipe")
    return binding


def create_joint8_fastwam(*, dataset_root, normalization_path, asset_root,
                         text_embedding_cache_dir, pretrained_fastwam_checkpoint=None,
                         initialize_from_official=True, model_init_seed=42,
                         deployment_binding=None, training_recipe=None, **kwargs):
    import torch
    from fastwam.models.wan22.fastwam import FastWAM
    from fastwam.baseline.optimizer import bounded_optimizer_param_groups
    from fastwam.utils.pytorch_utils import set_global_seed
    from omegaconf import DictConfig, OmegaConf
    if int(kwargs.get("proprio_dim", 0)) != 8 or int(kwargs["action_dit_config"]["action_dim"]) != 8:
        raise ValueError("FastWAM RealMan requires exactly 8D state and action")
    kwargs = {k: OmegaConf.to_container(v, resolve=True) if isinstance(v, DictConfig) else v
              for k, v in kwargs.items()}
    if isinstance(training_recipe, DictConfig):
        training_recipe = OmegaConf.to_container(training_recipe, resolve=True)
    if (training_recipe is None or training_recipe["model"] != {k: kwargs[k] for k in MODEL_RECIPE_FIELDS}
            or training_recipe["model_init_seed"] != int(model_init_seed)):
        raise ValueError("Model constructor differs from the bound semantic training recipe")
    binding = build_binding(dataset_root, normalization_path, asset_root, text_embedding_cache_dir,
                            training_recipe=training_recipe, deployment_binding=deployment_binding)
    if initialize_from_official and not pretrained_fastwam_checkpoint:
        raise ValueError("Fresh training requires the pinned complete FastWAM release")
    set_global_seed(int(model_init_seed))
    video_scheduler = kwargs.pop("video_scheduler")
    action_scheduler = kwargs.pop("action_scheduler")
    losses = kwargs.pop("loss", {})
    kwargs["torch_dtype"] = kwargs.pop("model_dtype", torch.bfloat16)
    for prefix, schedule in (("video", video_scheduler), ("action", action_scheduler)):
        kwargs[f"{prefix}_train_shift"] = float(schedule["train_shift"])
        kwargs[f"{prefix}_infer_shift"] = float(schedule["infer_shift"])
        kwargs[f"{prefix}_num_train_timesteps"] = int(schedule["num_train_timesteps"])
    kwargs["loss_lambda_video"] = float(losses.get("lambda_video", 1.0))
    kwargs["loss_lambda_action"] = float(losses.get("lambda_action", 1.0))
    # Only explicitly staged assets are allowed; creating a model never downloads.
    with patch.dict(os.environ, {"DIFFSYNTH_SKIP_DOWNLOAD": "true",
                                "DIFFSYNTH_MODEL_BASE_PATH": str(Path(asset_root).resolve())}):
        model = FastWAM.from_wan22_pretrained(**kwargs)
    if initialize_from_official:
        load_release(model, pretrained_fastwam_checkpoint)
    model.ablation_binding = binding
    model.save_checkpoint = MethodType(_save_bound_checkpoint, model)
    model.load_checkpoint = MethodType(_load_bound_checkpoint, model)
    model.optimizer_param_groups = MethodType(bounded_optimizer_param_groups, model)
    return model


def make_training_config(config_path, overrides=()):
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[3]
    base = OmegaConf.load(root / "configs/train.yaml")
    del base["defaults"]
    base.model = OmegaConf.load(root / "configs/model/fastwam.yaml")
    cfg = OmegaConf.merge(base, OmegaConf.load(config_path), OmegaConf.from_dotlist(list(overrides)))
    # The launcher intentionally changes cwd to the repository for DeepSpeed's
    # relative JSON path. Resolve user paths before that change, while model/data
    # interpolation still tracks these top-level values.
    for key in ("dataset_root", "normalization_path", "asset_root", "text_embedding_cache_dir",
                "pretrained_fastwam_checkpoint", "output_dir", "resume"):
        if not OmegaConf.is_missing(cfg, key) and cfg.get(key) is not None:
            cfg[key] = Path(str(cfg[key])).expanduser().resolve().as_posix()
    OmegaConf.resolve(cfg)
    validate_recipe(cfg)
    cfg.model.training_recipe = semantic_recipe(cfg)
    return cfg


def validate_recipe(cfg):
    if cfg.model.get("_recursive_", True):
        raise ValueError("Factory metadata must remain inert: model._recursive_=false is required")
    if cfg.data.train.processor.action_output_dim != 8 or cfg.data.train.processor.proprio_output_dim != 8:
        raise ValueError("joint8 shape override is not allowed")
    if cfg.data.train.num_frames != 33 or cfg.data.train.action_video_freq_ratio != 4:
        raise ValueError("This baseline uses a 32-action horizon and nine video frames")
    if cfg.data.train.val_set_proportion != 0 or cfg.data.train.processor.norm_default_mode != "z-score":
        raise ValueError("This recipe uses all canonical training episodes and z-score normalization")
    if list(cfg.data.train.video_size) != [224, 448] or cfg.data.train.concat_multi_camera != "horizontal":
        raise ValueError("FastWAM requires the global/wrist horizontal 224x448 layout")
    if [item.key for item in cfg.data.train.shape_meta.images] != list(CAMERAS):
        raise ValueError("Camera order must be global, wrist")
    if cfg.data.train.processor.action_state_transforms is not None or any(cfg.data.train.processor.delta_action_dim_mask.default):
        raise ValueError("joint8 must remain absolute radians/gripper; no delta or coordinate conversion")
    for split in ("train_transforms", "val_transforms"):
        transforms = cfg.data.train.processor[split]
        if (len(transforms) != 2 or transforms[0]._target_ != "fastwam.datasets.lerobot.transforms.image.ToTensor"
                or transforms[1]._target_ != "torchvision.transforms.Resize" or list(transforms[1].size) != [224, 224]):
            raise ValueError("Image preprocessing must match the deployed two-camera resize")
    if float(cfg.model.action_scheduler.infer_shift) != float(cfg.model.action_scheduler.train_shift):
        raise ValueError("Use matched FastWAM action train/inference schedules")


def prepare_observation(sample):
    """Exactly the stock per-camera ToTensor/Resize, horizontal concat, Normalize."""
    import numpy as np
    import torch
    from torchvision.transforms import functional as F
    state = np.asarray(sample["state"], dtype=np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("State must contain eight finite joint8 values")
    if not 0 <= state[7] <= 1:
        raise ValueError("State gripper must be in [0,1]")
    frames = []
    for camera in CAMERAS:
        array = np.asarray(sample["images"][camera])
        if array.dtype != np.uint8 or array.shape != (480, 640, 3):
            raise ValueError(f"{camera} must be uint8 RGB [480,640,3]")
        frame = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0
        frames.append(F.resize(frame, [224, 224], interpolation=F.InterpolationMode.BILINEAR, antialias=True))
    image = torch.cat(frames, dim=-1)
    return (image - 0.5) / 0.5, torch.from_numpy(state.copy())


def normalization_parameters(stats, field):
    """Official SingleFieldLinearNormalizer z-score arithmetic, on CPU FP32.

    Keep this tiny inference operation free of dataset/Arrow/Git dependencies.
    std_reg and clipping match datasets/lerobot/utils/normalizer.py exactly.
    """
    import torch
    values = stats[field]["default"]
    mean = torch.tensor(values["global_mean"], dtype=torch.float32)
    std = torch.tensor(values["global_std"], dtype=torch.float32)
    return 1.0 / (std + 1e-8), -mean / (std + 1e-8)


def load_predictor(config):
    """Load real trained weights and return sample -> raw absolute float32[32,8]."""
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config["training_config_path"])
    validate_recipe(cfg)
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.training_recipe = semantic_recipe(cfg)
    for key in ("normalization_path", "asset_root", "text_embedding_cache_dir"):
        model_cfg[key] = config[key]
    binding = json.loads(Path(config["binding_path"]).read_text(encoding="utf-8"))
    if config.get("dataset_manifest_sha256", binding["dataset_identity"]) != binding["dataset_identity"]:
        raise ValueError("Deployment dataset identity differs from the training binding")
    model_cfg.dataset_root = None
    model_cfg.deployment_binding = binding
    model_cfg.initialize_from_official = False
    model_cfg.pretrained_fastwam_checkpoint = None
    model_cfg.compile_training_denoise = False
    # Unlike upstream's permissive load_checkpoint, the bound loader is strict.
    model = instantiate(model_cfg, device=config.get("device", "cuda:0"), model_dtype=torch.bfloat16)
    model.load_checkpoint(config["checkpoint_path"])
    model.eval()
    context_payload = torch.load(text_cache_path(config["text_embedding_cache_dir"], binding["prompt"]),
                                 map_location="cpu", weights_only=True)
    context, mask = context_payload["context"].clone(), context_payload["mask"].bool()
    if context.shape != (128, 4096) or mask.shape != (128,) or not torch.isfinite(context).all():
        raise ValueError("Invalid T5 cache; expected context [128,4096] and mask [128]")
    context[~mask] = 0  # Match official RobotVideoDataset behavior exactly.
    context_mask = torch.ones_like(mask)
    stats = json.loads(Path(config["normalization_path"]).read_text(encoding="utf-8"))
    state_scale, state_offset = normalization_parameters(stats, "state")
    action_scale, action_offset = normalization_parameters(stats, "action")
    steps = int(config.get("num_inference_steps", 16))
    if steps < 1:
        raise ValueError("num_inference_steps must be positive")

    def predict(sample):
        if sample["prompt"] != binding["prompt"]:
            raise ValueError("Prompt differs from the bound single-task T5 cache")
        image, state = prepare_observation(sample)
        norm_state = torch.clamp(state * state_scale + state_offset, -5.0, 5.0)
        with torch.inference_mode():
            result = model.infer_action(prompt=None, input_image=image, action_horizon=32,
                                        proprio=norm_state, context=context, context_mask=context_mask,
                                        text_cfg_scale=1.0, num_inference_steps=steps,
                                        seed=int(config.get("seed", 42)), compile_action_infer=False)
        output = ((result["action"].float().cpu() - action_offset) / action_scale).numpy()
        if output.shape != (32, 8) or not np.isfinite(output).all():
            raise ValueError("FastWAM returned invalid joint8 actions")
        # The physical gripper is bounded; joint angles are never silently clipped.
        output[:, 7] = np.clip(output[:, 7], 0.0, 1.0)
        return output.astype(np.float32, copy=False)

    return predict
