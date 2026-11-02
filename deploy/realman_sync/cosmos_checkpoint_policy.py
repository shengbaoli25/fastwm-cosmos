"""Explicit inference-only bootstrap from a complete Cosmos policy checkpoint.

The training constructors, model class, native kernels and sampling code remain
unchanged. The saved policy contains the complete Video/Force/Action state, so
its original initialization checkpoint does not need to be read again.
"""
from __future__ import annotations

from copy import copy, deepcopy
import inspect
import json
import logging
import os
from pathlib import Path
import tempfile

from realman_deploy.model_adapter import CosmosJointPolicy
from .cosmos_policy import CosmosPolicy


logger = logging.getLogger(__name__)
SAMPLING_PROFILES = {
    "161616": dict(id="161616", label="标准 · 16 / 16 / 16",
                   video_steps=16, force_steps=16, action_steps=16),
    "4416": dict(id="4416", label="平衡 · 4 / 4 / 16",
                 video_steps=4, force_steps=4, action_steps=16),
    "3310": dict(id="3310", label="快速 · 3 / 3 / 10",
                 video_steps=3, force_steps=3, action_steps=10),
}


def checkpoint_sampling_config(config, profile=None):
    """Inference-only settings; never edit the saved model constructor inputs."""
    selected = config.get("sampling_profile", "161616") if profile is None else profile
    if not isinstance(selected, str) or selected not in SAMPLING_PROFILES:
        raise ValueError("Sampling profile must be 161616, 4416 or 3310")
    if config.get("execution_mode", "synchronous") != "synchronous":
        raise ValueError("The synchronous checkpoint server requires synchronous execution")
    backend = config.get("inference_backend", "joint_static_cache")
    if backend not in {"joint_static_cache", "joint_multirate", "joint_expert_schedule"}:
        raise ValueError("Unsupported checkpoint sampling backend")
    if backend == "joint_expert_schedule" and config.get("sampling_profile") != "3310":
        raise ValueError("Checkpoint joint_expert_schedule is reserved for profile 3310")
    if type(config.get("joint_steps", 16)) is not int or config.get("joint_steps", 16) != 16:
        raise ValueError("Checkpoint sampling profiles require joint_steps=16")
    options = config.get("inference_optimization", {})
    if backend == "joint_expert_schedule":
        valid_options = (isinstance(options, dict)
            and not set(options) - {"compile_layers", "cuda_graph"}
            and options.get("compile_layers", False) is False
            and options.get("cuda_graph", False) is False)
    else:
        valid_options = (isinstance(options, dict) and not set(options) - {"mode"}
                         and options.get("mode", "eager") == "eager")
    if not valid_options:
        raise ValueError("Checkpoint sampling currently supports eager optimization only")
    result = deepcopy(config)
    result.update(execution_mode="synchronous", joint_steps=16,
                  inference_backend="joint_multirate" if selected == "4416" else "joint_static_cache",
                  inference_optimization={"mode": "eager"})
    if selected == "3310":
        # 3 and 10 require independent Euler grids (30 merged ticks). The
        # original multirate runner requires divisibility and cannot use them.
        result.update(inference_backend="joint_expert_schedule",
                      inference_optimization={"compile_layers": False, "cuda_graph": False})
    # The base constructor interprets any profile as multirate. Removing it for
    # 161616 deliberately restores the exact original static-cache dispatch.
    if selected == "161616":
        result.pop("sampling_profile", None)
    else:
        result["sampling_profile"] = selected
    return result


def _persist_sampling(path, config):
    """Merge only deployment sampler fields and atomically replace the JSON."""
    path = Path(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict):
        raise ValueError("Runtime configuration must be a JSON object")
    for key in ("sampling_profile", "execution_mode", "joint_steps", "inference_backend", "inference_optimization"):
        if key in config:
            saved[key] = deepcopy(config[key])
        else:
            saved.pop(key, None)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(saved, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _sha(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be a lowercase SHA256 identity")
    return value


def checkpoint_contract(payload, configured):
    """Rebuild the original factory inputs, rejecting changed identities early."""
    from fastwam.cosmos import assets
    from fastwam.cosmos.noaction import NoActionJointModel, SCHEMA

    if (not isinstance(payload, dict) or payload.get("schema") != SCHEMA
            or not isinstance(payload.get("model"), dict) or not payload["model"]
            or not isinstance(payload.get("manifest"), dict)):
        raise ValueError("A complete no-reference Cosmos Joint8 policy checkpoint is required")
    if not isinstance(configured, dict):
        raise ValueError("model_kwargs must be the verified training model configuration")
    manifest = payload["manifest"]
    json.dumps(manifest, allow_nan=False)
    pinned = dict(upstream=assets.UPSTREAM_COMMIT, model=assets.MODEL_ID,
        revision=assets.REVISION, video_variant="base/post-trained",
        video_sha256=assets.NONACTION_WEIGHT_SHA, tokenizer_sha256=assets.TOKENIZER_SHA)
    if any(manifest.get(key) != value for key, value in pinned.items()):
        raise ValueError("Checkpoint official Cosmos source/asset identities differ from the pinned implementation")
    if (manifest.get("schema") != SCHEMA or manifest.get("joint_protocol") != SCHEMA
            or manifest.get("reference_mode") != "none" or manifest.get("action_space") != "joint8"
            or manifest.get("action_dim") != 8 or manifest.get("state_dim") != 8
            or manifest.get("history_dim") != 14):
        raise ValueError("Checkpoint must preserve the no-reference Joint8 protocol")
    for setting, field in (("expected_text_identity", "text_identity"),
                           ("expected_data_identity", "dataset_identity"),
                           ("expected_normalization_sha256", "normalization_sha256")):
        expected = _sha(manifest.get(field), field)
        if configured.get(setting) != expected:
            raise ValueError(f"Runtime {setting} differs from the trained checkpoint")
    kwargs = deepcopy(manifest.get("extra_config"))
    if not isinstance(kwargs, dict):
        raise ValueError("Checkpoint is missing the original model constructor arguments")
    parameters = inspect.signature(NoActionJointModel.__init__).parameters
    allowed = set(parameters) - {"self", "net", "manifest", "text_bank"}
    if set(kwargs) - allowed:
        raise ValueError("Checkpoint contains constructor arguments unsupported by this exact model implementation")
    asset_keys = {"upstream", "checkpoint", "text_cache", "expected_text_identity",
                  "expected_data_identity", "expected_normalization_sha256", "training_recipe"}
    for key, value in configured.items():
        if key in asset_keys:
            continue
        if key not in allowed:
            raise ValueError(f"Unsupported deployment model argument: {key}")
        # This affects autograd only; the saved manifest itself stays untouched.
        if key == "gradient_checkpointing" and value is False:
            continue
        trained = kwargs.get(key, parameters[key].default)
        if value != trained or type(value) is not type(trained):
            raise ValueError(f"Deployment model argument differs from training: {key}")
    override = configured.get("training_recipe")
    if override is not None and not isinstance(override, dict):
        raise ValueError("training_recipe must be a dictionary")
    recipe = dict(batch_per_rank=16, max_steps=30000)
    recipe.update(override or {})
    if type(recipe.get("batch_per_rank")) is not int or recipe["batch_per_rank"] < 1:
        raise ValueError("training_recipe requires a positive integer batch_per_rank")
    recipe.update(ranks=4, accum=1, global_batch=4 * recipe["batch_per_rank"])
    if recipe != manifest.get("recipe"):
        raise ValueError("Runtime training_recipe differs from the checkpoint; use the actual saved training configuration")
    if not any(key.startswith("dit.video.net.") for key in payload["model"]):
        raise ValueError("Checkpoint is missing its trained video network")
    initial = dict(pinned, dataset_identity=manifest["dataset_identity"],
        normalization_sha256=manifest["normalization_sha256"], text_identity=manifest["text_identity"],
        extra_config=deepcopy(kwargs), recipe=recipe)
    return initial, kwargs


def _finite_tensors(values, label):
    """Bound temporary allocations while checking large CPU checkpoint tensors."""
    import torch
    for name, value in values:
        if not isinstance(value, torch.Tensor):
            continue  # Native RMSNorm includes its original non-tensor _extra_state.
        if value.is_meta:
            raise ValueError(f"Unmaterialized meta tensor in {label}: {name}")
        if value.is_floating_point() or value.is_complex():
            flat = value.reshape(-1)
            for block in flat.split(1 << 20):
                if not bool(torch.isfinite(block).all()):
                    raise ValueError(f"Nonfinite tensor in {label}: {name}")


def strict_materialize(model, payload):
    """Use original load hooks and reject partial states before device transfer."""
    if payload.get("schema") != model.checkpoint_schema or payload.get("manifest") != model.manifest:
        raise ValueError("Checkpoint architecture/assets/protocol differ from the reconstructed model")
    _finite_tensors(payload["model"].items(), "checkpoint")
    import torch
    for name, expected in model.state_dict().items():
        actual = payload["model"].get(name)
        if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor) and actual.dtype != expected.dtype:
            raise ValueError(f"Checkpoint tensor dtype differs from the original model precision strategy: {name}")
    model.load_state_dict(payload["model"], strict=True, assign=True)
    _finite_tensors(list(model.named_parameters()) + list(model.named_buffers()), "loaded model")
    return model


def move_materialized(model, device):
    # Never cast the already-loaded FP32 modulation/final-layer tensors through
    # BF16: converting back to float would restore dtype but lose trained bits.
    model = model.to(device=device).eval()
    _finite_tensors(list(model.named_parameters()) + list(model.named_buffers()), "device model")
    return model


def create_checkpoint_model(payload, *, upstream, text_cache, configured, device="cuda"):
    """Build the unchanged joint model without reading initialization weights.

    The adapter must first verify its checkpoint/normalization sidecar. All
    checkpoint, source, text and reconstructed model identities are checked here.
    """
    import torch
    from fastwam.cosmos.assets import read_text_bank
    from fastwam.cosmos.native import nonaction_native_imports, nonaction_net_config
    from fastwam.cosmos.noaction import NoActionJointModel

    initial, kwargs = checkpoint_contract(payload, configured)
    bank, text_manifest, digest = read_text_bank(text_cache)
    if digest != initial["text_identity"]:
        raise ValueError("Verified Reason1 text cache differs from the training identity")
    # Existing native import checks all pinned upstream core hashes.
    _, module, _, Net, _ = nonaction_native_imports(Path(upstream).resolve())
    config = nonaction_net_config()
    config["sac_config"] = module.SACConfig(mode="none")
    with torch.device("meta"):
        net = Net(**config)
    # CPU construction is intentional: scheduler scalar computations, runtime
    # manifest/RNG hooks and ActionDiT's non-buffer frequencies require CPU.
    with torch.device("cpu"):
        model = NoActionJointModel(net, manifest=initial, text_bank=bank, **kwargs)
    # Establish the original BF16 + protected FP32 strategy on the untrained
    # allocations/meta tensors, before assigning any saved parameter values.
    model.to(dtype=torch.bfloat16)
    strict_materialize(model, payload)
    model = move_materialized(model, device)
    model.gradient_checkpointing = False
    return model, text_manifest


class CheckpointJointPolicy(CosmosJointPolicy):
    """Keep the existing observation, normalization, sampler and warmup behavior."""

    def __init__(self, checkpoint, config, *, config_path):
        super().__init__(checkpoint, checkpoint_sampling_config(config), config_path=config_path)

    def sampling_settings(self):
        selected = self.config.get("sampling_profile", "161616")
        counts = SAMPLING_PROFILES[selected]
        return dict(supported=True, sampling_profile=selected,
                    **{key: counts[key] for key in ("video_steps", "force_steps", "action_steps")},
                    inference_backend=self.inference_backend, execution_mode="synchronous",
                    optimization="eager", profiles=deepcopy(list(SAMPLING_PROFILES.values())))

    def configure_sampling(self, profile, *, persist=None):
        """Caller holds the inference lock and has released the client session.

        Stage a new runner around the resident model/tokenizer. Neither live
        sampler settings nor the runtime file change until warm-up succeeds.
        """
        updated = checkpoint_sampling_config(self.config, profile)
        if updated == self.config:
            return self.sampling_settings()
        candidate = copy(self)
        candidate.config = updated
        candidate.inference_backend = updated["inference_backend"]
        candidate.runner = None
        candidate.diagnostics = deepcopy(self.diagnostics)
        candidate.diagnostics.pop("sampling", None)
        candidate.diagnostics.pop("optimization", None)
        try:
            candidate._configure_inference()
            candidate._warmup()
            if persist is not None:
                persist(updated)
        except BaseException:
            if candidate.runner is not None:
                try:
                    candidate.runner.close()
                except Exception:
                    logger.exception("Candidate sampler cleanup failed; retaining previous sampler")
            raise
        old_runner = self.runner
        self.config, self.inference_backend = candidate.config, candidate.inference_backend
        self.runner, self.diagnostics = candidate.runner, candidate.diagnostics
        # Expert schedules bind their counts at runner construction. Publish
        # that binding in the same transaction so the first live infer agrees.
        if candidate.inference_backend == "joint_expert_schedule":
            self._expert_steps = candidate._expert_steps
        else:
            self.__dict__.pop("_expert_steps", None)
        # _warmup resets only candidate.inference_index; live RNG progression
        # is unchanged, and neither warm-up action chunk is returned to a client.
        if old_runner is not None and old_runner is not self.runner:
            try:
                old_runner.close()
            except Exception:
                logger.exception("Previous sampler cleanup failed after successful switch")
        return self.sampling_settings()

    def _load(self):
        import torch
        from torchvision.transforms import InterpolationMode, Resize
        from realman_deploy.device import validate_device
        from realman_deploy.preprocessing import file_sha256
        from fastwam.cosmos.assets import load_tokenizer, checked, TOKENIZER_SHA
        from fastwam.cosmos.native import verify_upstream

        expected_sampling = checkpoint_sampling_config(self.config)
        if self.inference_backend != expected_sampling["inference_backend"]:
            raise ValueError("Checkpoint inference backend differs from its sampling profile")
        if self.normalizer.action_space != "joint8" or self.normalizer.force_preprocessing is None:
            raise ValueError("The exact Joint8 normalization and force preprocessing contract are required")
        # CosmosJointPolicy.__init__ already verifies the full checkpoint SHA,
        # normalization SHA and force identity using the mandatory .force.json.
        self._resize = Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        configured = deepcopy(self.config.get("model_kwargs"))
        initial, _ = checkpoint_contract(payload, configured)
        normalization = self._path(self.config["normalization_file"])
        if file_sha256(normalization) != initial["normalization_sha256"]:
            raise ValueError("Use the exact normalization.json bound during this training run")
        if payload["manifest"]["action_dim"] != self.normalizer.action_dim:
            raise ValueError("Checkpoint and normalization action widths differ")
        upstream = self._path(configured["upstream"])
        text_cache = self._path(configured["text_cache"])
        tokenizer = self._path(self.config["tokenizer"])
        verify_upstream(upstream)
        checked(tokenizer, TOKENIZER_SHA)
        if type(self.config.get("joint_steps", 16)) is not int or self.config.get("joint_steps", 16) < 1:
            raise ValueError("Joint sampler step count must be a positive integer")
        self.reference_mode = "none"
        device_info = validate_device(self.device)
        self.device = device_info["device"]
        self.diagnostics["device"] = device_info
        self.model, text_manifest = create_checkpoint_model(payload, upstream=upstream,
            text_cache=text_cache, configured=configured, device=self.device)
        self.text_index = self.prompt_index(text_manifest, self.config.get("task_prompt"))
        del payload
        self.tokenizer = load_tokenizer(upstream, tokenizer, device=self.device)
        self.diagnostics["checkpoint_bootstrap"] = "complete_policy_state_strict_without_base_file"
        self._configure_inference()
        self._warmup()


class CheckpointCosmosPolicy(CosmosPolicy):
    """Use the original synchronous wire adapter with an explicit new backend."""

    def __init__(self, config, config_path, checkpoint):
        self.config_path = Path(config_path).resolve()
        self.policy = CheckpointJointPolicy(checkpoint, config, config_path=Path(config_path))
        try:
            if self.policy.normalizer.action_space != "joint8" or self.policy.reference_mode != "none":
                raise ValueError("Synchronous server requires a no-reference Joint8 checkpoint")
            self.metadata = dict(camera_names=config.get("camera_order", ["global", "wrist"]),
                width=int(config.get("camera_width", 640)), height=int(config.get("camera_height", 480)),
                action_space="joint8", action_horizon=32, control_hz=30, history_length=32,
                video_context_frames=1, video_context_stride=1,
                force_preprocessing=self.policy.normalizer.force_preprocessing, backend="cosmos_checkpoint")
        except Exception:
            self.policy.close()
            raise

    def sampling_settings(self):
        return self.policy.sampling_settings()

    def configure_sampling(self, profile):
        return self.policy.configure_sampling(profile,
            persist=lambda updated: _persist_sampling(self.config_path, updated))
