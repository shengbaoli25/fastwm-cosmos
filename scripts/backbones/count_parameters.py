#!/usr/bin/env python3
"""Audit exact backbone-comparison parameter counts without weights or a GPU.

Example (from the repository root)::

    python scripts/backbones/count_parameters.py

The configured joint model and the pinned upstream teacher constructors run on
PyTorch's meta device. Parameter-free flow schedulers alone use CPU because
their scalar normalization calls Tensor.item(). No weights are downloaded or
loaded. BF16 bytes are weight storage only, not inference peak memory.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = (
    "configs/experiments/rm71d78_vjepa_vitb_28l_50k.yaml",
    "configs/experiments/rm71d78_dinov3_vithplus_28l_50k.yaml",
)
COUNTED_SOURCE_FILES = (
    "scripts/backbones/count_parameters.py",
    "src/fastwam/cosmos/dense_features.py",
    "src/fastwam/cosmos/dense_encoder.py",
    "src/fastwam/forcewm/diffusion.py",
    "src/fastwam/forcewm/model.py",
    "src/fastwam/models/wan22/action_dit.py",
    "src/fastwam/models/wan22/wan_video_dit.py",
    "src/fastwam/baseline/optimizer.py",
    "src/fastwam/trainer.py",
)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_summary(module):
    """Count registered nn.Parameters once by identity; never count buffers."""
    unique = list(module.named_parameters())
    occurrences = list(module.named_parameters(remove_duplicate=False))
    names_by_id = defaultdict(list)
    for name, parameter in occurrences:
        names_by_id[id(parameter)].append(name)
    specs = [dict(name=name, shape=list(parameter.shape),
                  requires_grad=parameter.requires_grad, dtype=str(parameter.dtype))
             for name, parameter in unique]
    count = sum(parameter.numel() for _, parameter in unique)
    return dict(
        parameters=count,
        trainable_parameters=sum(p.numel() for _, p in unique if p.requires_grad),
        frozen_parameters=sum(p.numel() for _, p in unique if not p.requires_grad),
        unique_parameter_tensors=len(unique),
        all_registered_parameter_paths=len(occurrences),
        duplicate_parameter_aliases=[names for names in names_by_id.values() if len(names) > 1],
        parameter_specs_sha256=hashlib.sha256(
            json.dumps(specs, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        bf16_parameter_bytes=2 * count,
        registered_buffer_elements=sum(buffer.numel() for buffer in module.buffers()),
    )


def verify_optimizer_coverage(model):
    """Use the training optimizer grouping and reject omissions or duplicates."""
    candidates = list(model.dit.parameters())
    proprio = getattr(model, "proprio_encoder", None)
    if proprio is not None:
        candidates.extend(proprio.parameters())
    groups = model.optimizer_param_groups(candidates)
    optimized = [parameter for group in groups for parameter in group["params"]]
    actual_ids = [id(parameter) for parameter in optimized]
    expected_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise ValueError("Optimizer must cover every trainable parameter exactly once")
    return dict(group_count=len(groups), parameters=sum(p.numel() for p in optimized),
                covers_each_trainable_parameter_exactly_once=True)


def apply_training_freeze(model):
    """Execute the repository's exact stateless method without importing Accelerate."""
    source = ROOT / "src/fastwam/trainer.py"
    tree = ast.parse(source.read_text())
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == "_apply_dit_only_train_mode"]
    if len(matches) != 1:
        raise ValueError("Expected one trainer freeze method")
    method = matches[0]
    method.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    namespace[method.name](model)


@contextmanager
def parameter_free_cpu_schedulers(dense, torch):
    original = dense.WanContinuousFlowMatchScheduler

    def cpu_scheduler(*args, **kwargs):
        with torch.device("cpu"):
            return original(*args, **kwargs)

    dense.WanContinuousFlowMatchScheduler = cpu_scheduler
    try:
        yield
    finally:
        dense.WanContinuousFlowMatchScheduler = original


def audit_config(config_path, teacher_source=None):
    import torch
    from omegaconf import OmegaConf
    from fastwam.cosmos import dense_features as dense
    from fastwam.cosmos.dense_encoder import build_teacher_architecture, teacher_spec

    config_path = Path(config_path).resolve()
    cfg = OmegaConf.load(config_path)
    model_config = OmegaConf.to_container(cfg.model, resolve=True, throw_on_missing=True)
    target = model_config.pop("_target_")
    if target != "fastwam.cosmos.dense_features.create_dense_feature_joint":
        raise ValueError(f"Unsupported model target: {target}")
    teacher_name = str(cfg.teacher)
    spec = teacher_spec(teacher_name)
    if model_config["feature_dim"] != spec["feature_dim"]:
        raise ValueError("Teacher and model feature dimensions differ")
    if str(cfg.data.train.expected_teacher) != teacher_name:
        raise ValueError("Dataset and model teacher identities differ")
    with parameter_free_cpu_schedulers(dense, torch), torch.device("meta"):
        model = dense.create_dense_feature_joint(**model_config, device="meta")
        teacher = build_teacher_architecture(teacher_name, source=teacher_source)
        teacher.eval().requires_grad_(False).to(dtype=torch.bfloat16)
    apply_training_freeze(model)
    if any(parameter.device.type != "meta" for module in (model, teacher)
           for parameter in module.parameters()):
        raise ValueError("All counted parameter tensors must remain on the meta device")
    model_count = parameter_summary(model)
    teacher_count = parameter_summary(teacher)
    components = {name: parameter_summary(module) for name, module in model.dit.items()}
    if sum(item["parameters"] for item in components.values()) != model_count["parameters"]:
        raise ValueError("Unexpected shared or ungrouped parameters across model components")
    if teacher_count["trainable_parameters"]:
        raise ValueError("Cache teacher must be fully frozen")
    optimizer = verify_optimizer_coverage(model)
    teacher_ids = {id(parameter) for parameter in teacher.parameters()}
    if teacher_ids.intersection(id(parameter) for parameter in model.parameters()):
        raise ValueError("Teacher must be separate from the training model")
    try:
        config_name = str(config_path.relative_to(ROOT))
    except ValueError:
        config_name = str(config_path)
    return dict(
        config=config_name,
        config_sha256=sha256_file(config_path),
        teacher=teacher_name,
        teacher_spec=spec,
        model_config=model_config,
        training_model=model_count,
        components=components,
        optimizer=optimizer,
        frozen_vision_encoder=teacher_count,
        teacher_is_registered_in_training_model=False,
        logical_model_parameters=model_count["parameters"] + teacher_count["parameters"],
        logical_model_bf16_parameter_bytes=2 * (model_count["parameters"] + teacher_count["parameters"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append",
                        help="Repeat to audit multiple experiment YAMLs (default: both new variants)")
    parser.add_argument("--teacher-source", type=Path,
                        help="Optional pinned upstream source; use with one --config")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/cosmos/backbone-parameter-counts.json")
    args = parser.parse_args()
    configs = args.config or [ROOT / path for path in DEFAULT_CONFIGS]
    if args.teacher_source and len(configs) != 1:
        parser.error("--teacher-source requires exactly one --config")
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="offline")
    import torch

    variants = [audit_config(path, args.teacher_source) for path in configs]
    result = dict(
        schema="fastwam-backbone-parameter-counts-v1",
        audited_at_utc=datetime.now(timezone.utc).isoformat(),
        torch_version=torch.__version__,
        git_head_at_audit=subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        method="Actual configured PyTorch constructors on meta; exact trainer freeze method; unique nn.Parameter identity count",
        source_files_sha256={path: sha256_file(ROOT / path) for path in COUNTED_SOURCE_FILES},
        variants=variants,
        exclusions=[
            "Registered buffers, positional/RoPE caches, gradients, optimizer state and activations are not parameters.",
            "The frozen vision encoder is cached offline during training and is not part of the optimizer.",
            "Logical-model totals include the predictor, Action, Force, adapters and frozen vision encoder only.",
            "No text encoder, video decoder/tokenizer or visualization decoder is instantiated by these variants.",
            "DINO official combined QKV biases include masked K-bias entries; all registered entries count as parameters.",
            "BF16 parameter bytes are weight storage, not an inference-memory measurement; GPU count does not multiply model size.",
            "No model weights, GPU allocations, network requests or forward/training steps were used for this audit.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "variants": [dict(
        teacher=variant["teacher"],
        trainable_parameters=variant["training_model"]["trainable_parameters"],
        frozen_vision_parameters=variant["frozen_vision_encoder"]["parameters"],
        logical_model_parameters=variant["logical_model_parameters"],
    ) for variant in variants]}, indent=2))


if __name__ == "__main__":
    main()
