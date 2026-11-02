"""Deployment exporter and policy interface for the ForceVLA ablation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .contracts import (CAMERAS, PROMPT, identity_sha256, load_dataset_manifest,
                        path_sha256, read_json, require, write_json)

BACKENDS = ("forcevla",)
ASSET_KEYS = ("checkpoint_path", "normalization_path", "training_binding_path")


def required_assets(backend):
    require(backend in BACKENDS, "This branch supports only the ForceVLA ablation")
    return set(ASSET_KEYS)


def verify_training_binding(backend, model, dataset_identity):
    require(backend in BACKENDS, "This branch supports only the ForceVLA ablation")
    binding = read_json(model["training_binding_path"])
    require(binding.get("dataset_manifest_sha256") == dataset_identity, "Training binding belongs to another dataset")
    require(binding.get("normalization_sha256") == path_sha256(model["normalization_path"]), "Training normalization differs from selected deployment statistics")
    embedded = read_json(Path(model["checkpoint_path"]) / "assets/ablation_training_binding.json")
    require(embedded == binding, "Checkpoint belongs to another training run or model recipe")


def export_deployment(backend, model_config, dataset_root, output_path):
    require(backend in BACKENDS, "Unknown ablation backend")
    manifest = load_dataset_manifest(dataset_root)
    model_config = dict(model_config)
    require("checkpoint_path" in model_config and "normalization_path" in model_config, "Export requires a real training checkpoint and normalization")
    require("training_binding_path" in model_config, "Export requires the training binding file")
    require(required_assets(backend).issubset(model_config), "Missing required model-specific assets")
    verify_training_binding(backend, model_config, manifest["identity_sha256"])
    assets = {}
    for key in ASSET_KEYS:
        if model_config.get(key):
            path = Path(model_config[key]).expanduser().resolve()
            require(path.exists(), f"Missing deployment asset: {key}: {path}")
            assets[key] = {"path": str(path), "sha256": path_sha256(path)}
            model_config[key] = str(path)
    model_config["dataset_manifest_sha256"] = manifest["identity_sha256"]
    model_config.pop("dataset_root", None)
    payload = dict(schema="rm71d78-deployment-v1", backend=backend, model=model_config, assets=assets,
                   dataset_identity_sha256=manifest["identity_sha256"], source_inventory_sha256=manifest["source_inventory_sha256"],
                   force_preprocessing=manifest["force_preprocessing"], prompt=manifest["prompt"],
                   camera_names=CAMERAS, image_size=[640, 480], action_space="joint8", action_horizon=32, control_hz=30,
                   force_conditioning=True, model_load_verified=False)
    payload["identity_sha256"] = identity_sha256(payload)
    require(not Path(output_path).exists(), "Deployment output already exists; choose a new path")
    write_json(output_path, payload)
    return payload


def load_deployment(path):
    config = read_json(path)
    require(config.get("schema") == "rm71d78-deployment-v1", "Wrong deployment schema")
    require(config.get("identity_sha256") == identity_sha256({k: v for k, v in config.items() if k != "identity_sha256"}), "Deployment identity mismatch")
    require(config["backend"] in BACKENDS, "Unknown backend")
    for key, expected in {"camera_names": CAMERAS, "image_size": [640, 480], "action_space": "joint8", "action_horizon": 32,
                          "control_hz": 30, "prompt": PROMPT}.items():
        require(config.get(key) == expected, f"Invalid deployment contract: {key}")
    require(config["model"]["dataset_manifest_sha256"] == config["dataset_identity_sha256"], "Dataset identity mismatch")
    from realman_force.contract import validate_contract_identity
    validate_contract_identity(config["force_preprocessing"])
    for key, asset in config["assets"].items():
        require(config["model"].get(key) == asset["path"], f"Model asset path mismatch: {key}")
        require(path_sha256(asset["path"]) == asset["sha256"], f"Deployment asset changed: {key}")
    require(required_assets(config["backend"]).issubset(config["assets"]), "Unbound model assets")
    require({k for k in ASSET_KEYS if config["model"].get(k)} == set(config["assets"]), "Additional model assets must also be bound")
    verify_training_binding(config["backend"], config["model"], config["dataset_identity_sha256"])
    return config


class AblationPolicy:
    def __init__(self, deployment_path):
        self.config = load_deployment(deployment_path)
        from .forcevla import load_predictor
        self.predictor = load_predictor(dict(self.config["model"]))
        self.metadata = dict(camera_names=CAMERAS, width=640, height=480, action_space="joint8",
                             action_horizon=32, control_hz=30, history_length=32, video_context_frames=1,
                             video_context_stride=1, force_preprocessing=self.config["force_preprocessing"],
                             backend=self.config["backend"], force_conditioning=self.config["force_conditioning"],
                             deployment_identity=self.config["identity_sha256"], dataset_identity=self.config["dataset_identity_sha256"])

    def predict(self, observation, history, video_context):
        images = {name: np.frombuffer(frame["data"], dtype=np.uint8).reshape(frame["height"], frame["width"], 3).copy()
                  for name, frame in observation["frames"].items()}
        sample = dict(state=np.asarray(observation["state"], dtype=np.float32),
                      force=np.asarray(observation["force"], dtype=np.float32), images=images, prompt=self.config["prompt"])
        actions = np.asarray(self.predictor(sample), dtype=np.float32)
        require(actions.shape == (32, 8) and np.isfinite(actions).all(), "Model returned invalid joint8 actions")
        # The physical gripper has a bounded scalar output; joint angles are never clipped.
        actions[:, 7] = np.clip(actions[:, 7], 0, 1)
        return actions.tolist()

    def close(self):
        close = getattr(self.predictor, "close", None)
        if close is not None:
            close()
