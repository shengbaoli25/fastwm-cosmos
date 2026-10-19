"""Cosmos joint model adapter.

The heavy torch/Cosmos imports are intentionally lazy.  The HTTP dashboard
can therefore start on a CPU-only machine and report that a GPU is absent
without importing or allocating the 2B model.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import time

from .contracts import Observation
from .normalization import DatasetNormalizer, OnlineHistory


logger = logging.getLogger(__name__)


class CosmosJointPolicy:
    def __init__(self, checkpoint: str | Path, config: dict, *, config_path: str | Path):
        self.checkpoint = Path(checkpoint).resolve()
        self.config = dict(config)
        self.config_dir = Path(config_path).resolve().parent
        self.normalizer = DatasetNormalizer.from_json(self._path(config["normalization_file"]))
        if self.normalizer.force_preprocessing is not None:
            from .preprocessing import verify_binding
            verify_binding(self.checkpoint, self._path(config['normalization_file']))
        self.device = str(config.get("device", "cuda"))
        self.model = None
        self.tokenizer = None
        self.runner = None
        self.inference_backend = str(config.get("inference_backend", "reference"))
        if self.inference_backend not in {"reference", "video_isolated_reuse", "joint_static_cache"}:
            raise ValueError(f"Unknown inference_backend: {self.inference_backend}")
        self.text_index = 0
        self.inference_index = 0
        self.diagnostics = {}
        try:
            self._load()
        except Exception:
            self.close()
            raise

    def _path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.config_dir / path).resolve()

    @staticmethod
    def prompt_index(manifest: dict, task_prompt: str | None) -> int:
        # text_bank is stacked in manifest order, not lexicographic order.
        prompts = [str(item["prompt"]) for item in manifest.get("texts", [])]
        if not prompts or len(prompts) != len(set(prompts)):
            raise ValueError("text cache requires unique nonempty prompts")
        if task_prompt is None and len(prompts) == 1:
            return 0
        if task_prompt not in prompts:
            raise ValueError("task_prompt does not exist in the verified Reason1 text cache")
        return prompts.index(task_prompt)

    def _load(self) -> None:
        import torch
        # Importing torchvision may hold the GIL longer than the sensor deadline.
        # Complete this work before the runtime can enter closed-loop control.
        from torchvision.transforms import InterpolationMode, Resize
        self._resize = Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True)
        if not torch.cuda.is_available() and self.device.startswith("cuda"):
            raise RuntimeError("CUDA is unavailable; connect the external GPU before loading a checkpoint")
        from fastwam.cosmos.assets import create_joint_cosmos_forcewm, create_noaction_joint_cosmos, load_tokenizer
        from fastwam.cosmos.noaction import SCHEMA
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        self.reference_mode = "none" if payload.get("schema") == SCHEMA else "rolling_ae"
        if self.reference_mode == "none" and self.inference_backend == "video_isolated_reuse":
            raise ValueError("New joint diffusion uses reference or joint_static_cache backend")
        if self.reference_mode != "none" and self.inference_backend == "joint_static_cache":
            raise ValueError("joint_static_cache requires the no-reference diffusion checkpoint")
        kwargs = dict(self.config["model_kwargs"])
        if self.reference_mode == "none":
            from .preprocessing import file_sha256
            normalization_sha = payload['manifest'].get('normalization_sha256')
            if not normalization_sha or file_sha256(self._path(self.config['normalization_file'])) != normalization_sha:
                raise ValueError('Use the exact normalization.json bound during this RealMan training run')
            if int(self.config.get('joint_steps', 16)) != payload['manifest']['num_inference_steps']:
                raise ValueError('Joint sampler step count differs from the trained configuration')
            trained = payload["manifest"]["extra_config"]
            if (payload['manifest']['action_space'] != self.normalizer.action_space
                    or payload['manifest']['action_dim'] != self.normalizer.action_dim):
                raise ValueError('Checkpoint and normalization action space differ')
            for key, value in trained.items():
                if key in kwargs and kwargs[key] != value and key != "gradient_checkpointing":
                    raise ValueError(f"Deployment architecture differs from trained checkpoint: {key}")
            kwargs.update(trained)
            kwargs["expected_data_identity"] = payload["manifest"]["dataset_identity"]
            kwargs["expected_normalization_sha256"] = normalization_sha
        del payload
        for key in ("upstream", "checkpoint", "text_cache"):
            kwargs[key] = str(self._path(kwargs[key]))
        kwargs.update({"device": self.device})
        tokenizer_path = self._path(self.config["tokenizer"])
        if not tokenizer_path.is_file():
            raise FileNotFoundError("deployment config must point to the verified Cosmos tokenizer.pth")
        text_cache = Path(kwargs["text_cache"]).expanduser().resolve()
        text_manifest_path = text_cache / "manifest.json"
        if not text_manifest_path.is_file():
            raise FileNotFoundError(f"verified Reason1 text cache manifest is missing: {text_manifest_path}")
        text_manifest = json.loads(text_manifest_path.read_text())
        self.text_index = self.prompt_index(text_manifest, self.config.get("task_prompt"))
        factory = create_noaction_joint_cosmos if self.reference_mode == "none" else create_joint_cosmos_forcewm
        self.model = factory(**kwargs)
        self.model.load_checkpoint(self.checkpoint)
        self.model.eval()
        self.model.gradient_checkpointing = False
        self.tokenizer = load_tokenizer(kwargs["upstream"], tokenizer_path, device=self.device)
        self._configure_inference()
        self._warmup()

    def _configure_inference(self) -> None:
        if self.inference_backend in {"reference", "joint_static_cache"}:
            return
        from fastwam.cosmos.inference import JointInferenceRunner
        options = self.config.get("inference_optimization", {})
        compile_future = bool(options.get("compile_future", True))
        if compile_future and self.device.startswith("cuda"):
            import torch._inductor.config as compiler_config
            # This process owns the model. Bound compiler workers on this host;
            # compilation and graph capture finish before readiness is published.
            compiler_config.compile_threads = max(1, int(options.get("compile_threads", 2)))
        self.runner = JointInferenceRunner(self.model,
            use_cuda_graph=bool(options.get("cuda_graph", True)),
            compile_future=compile_future,
            compile_mode=str(options.get("compile_mode", "max-autotune-no-cudagraphs")))

    def _warmup(self) -> None:
        """Initialize kernels while loading; synthetic outputs never enter a queue."""
        width, height = self.config.get("camera_width", 640), self.config.get("camera_height", 480)
        frames = {name: bytes(width * height * 3)
                  for name in self.config.get("camera_order", ["global", "wrist"])}
        state = tuple(self.normalizer.stats["state"]["mean"])
        force = tuple(self.normalizer.stats["force"]["mean"])
        obs = Observation.now(0, state, force, frames, action_space=self.normalizer.action_space)
        history = OnlineHistory(self.normalizer, 32)
        history.append(state, force)
        reference = (None if getattr(self, 'reference_mode', 'rolling_ae') == 'none'
                     else [self.normalizer.encode("action", state) for _ in range(32)])
        profiles = []
        try:
            # The second pass distinguishes initialization from cached execution.
            for _ in range(2):
                self.inference_index = 0
                self.infer(obs, reference, history.tensors())
                profiles.append(dict(self.diagnostics["last_inference"]))
        finally:
            self.inference_index = 0
        self.diagnostics = {"inference_backend": self.inference_backend, "warmup": profiles}
        logger.info("Model initialization completed; synthetic warmup timings: %s", profiles)

    def _encode_image(self, observation: Observation):
        import torch
        names = self.config.get("camera_order", ["global", "wrist"])
        width, height = self.config.get("camera_width", 640), self.config.get("camera_height", 480)
        images = []
        for name in names:
            raw = observation.frames.get(name)
            if raw is None or len(raw) != width * height * 3:
                raise ValueError(f"missing camera frame: {name}")
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(height, width, 3)
            tensor = tensor.permute(2, 0, 1).float().div(255.0)
            tensor = self._resize(tensor)
            images.append(tensor)
        return torch.cat(images, dim=-1).sub_(0.5).div_(0.5).unsqueeze(0).unsqueeze(2).to(self.device, torch.bfloat16)

    def infer(self, observation: Observation, normalized_reference: list[list[float]] | None,
              history: tuple[list[list[float]], list[bool], list[list[bool]]]) -> list[list[float]]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Cosmos policy is not loaded with a tokenizer")
        if observation.action_space != self.normalizer.action_space:
            raise ValueError('Observation and policy action space differ')
        import torch
        started = time.monotonic()
        pixels = self._encode_image(observation)
        with torch.inference_mode():
            first = self.tokenizer.encode(pixels[:, :, :1])
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(self.device)
            encoded = time.monotonic()
            raw = torch.tensor(history[0], device=self.device, dtype=torch.float32).unsqueeze(0)
            hpad = torch.tensor(history[1], device=self.device, dtype=torch.bool).unsqueeze(0)
            hvalid = torch.tensor(history[2], device=self.device, dtype=torch.bool).unsqueeze(0)
            state = torch.tensor([self.normalizer.encode("state", observation.state)],
                                 device=self.device, dtype=torch.float32)
            sample = {
                "first_frame_latents": first,
                "history": raw.to(torch.bfloat16), "history_is_pad": hpad,
                "history_channel_valid": hvalid,
                "proprio": torch.cat((state, torch.zeros((1, 1), device=self.device)), -1).unsqueeze(1),
                "action_is_pad": torch.zeros((1, 32), device=self.device, dtype=torch.bool),
                "cosmos_text_index": torch.tensor([self.text_index], device=self.device),
            }
            if getattr(self, "reference_mode", "rolling_ae") == "none":
                result = self.model.infer_joint(sample,
                    num_inference_steps=int(self.config.get("joint_steps", 16)),
                    seed=int(self.config.get("action_seed", 42)) + self.inference_index,
                    video_seed=int(self.config.get("video_seed", 44)),
                    force_seed=int(self.config.get("force_seed", 43)),
                    use_cache=self.inference_backend == "joint_static_cache")
            else:
                reference = torch.tensor(normalized_reference, device=self.device,
                                        dtype=torch.bfloat16).unsqueeze(0)
                inference = self.runner if self.runner is not None else self.model
                result = inference.infer_forcewm(sample, action_condition=reference,
                                                  num_inference_steps=int(self.config.get("action_steps", 16)),
                                                  seed=int(self.config.get("action_seed", self.config.get("model_kwargs", {}).get("action_seed", 42))) + self.inference_index,
                                                  video_seed=int(self.config.get("video_seed",
                                                      self.config.get("model_kwargs", {}).get("memory_seed", 44))),
                                                  generate_action=True)
            actions = result["action"][0].float().cpu().tolist()
            self.inference_index += 1
        finished = time.monotonic()
        profile = {"backend": self.inference_backend,
                   "image_and_tokenizer_ms": round((encoded - started) * 1000, 1),
                   "world_models_and_action_ms": round((finished - encoded) * 1000, 1),
                   "total_ms": round((finished - started) * 1000, 1)}
        self.diagnostics = {**self.diagnostics, "last_inference": profile}
        logger.info("Inference sequence=%s timings=%s", observation.sequence, profile)
        if (len(actions) != 32 or any(len(row) != self.normalizer.action_dim for row in actions)
                or not all(math.isfinite(float(value)) for row in actions for value in row)):
            raise ValueError(f"AE output must be a finite Action32x{self.normalizer.action_dim} chunk")
        return actions

    def close(self) -> None:
        if self.runner is not None:
            self.runner.close()
            self.runner = None
        self.model = self.tokenizer = None
