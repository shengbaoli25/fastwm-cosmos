"""Attach the proven Torch 2.7.1 oversized-flat compatibility workaround."""
from types import MethodType
from pathlib import Path

import torch

from fastwam.runtime import create_fastwam
from fastwam.utils.pytorch_utils import set_global_seed
from .optimizer import bounded_optimizer_param_groups
from .analysis import file_sha256


def _optimizer_groups(self, parameters):
    return bounded_optimizer_param_groups(self, parameters)


def load_pretrained_fastwam(model, path, expected_sha256):
    path = Path(path).resolve()
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError("The pinned complete official FastWAM checkpoint is absent or its SHA256 differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "mot" not in payload or "proprio_encoder" not in payload:
        raise ValueError("A full trained FastWAM policy checkpoint is required, not Wan or ActionDiT initialization")
    model.mot.load_state_dict(payload["mot"], strict=True)
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    model.pretrained_fastwam_metadata = {"path": str(path), "sha256": expected_sha256,
                                         "source_step": payload.get("step"), "loading": "strict complete MoT and proprio"}
    del payload


def create_fastwam_baseline(model_init_seed=42, pretrained_fastwam_checkpoint=None,
                           pretrained_fastwam_sha256=None, require_pretrained_fastwam=True, **kwargs):
    # Seed before the stock loader initializes task-specific input/output heads.
    set_global_seed(int(model_init_seed))
    if require_pretrained_fastwam and not pretrained_fastwam_checkpoint:
        raise ValueError("Training must initialize from the complete official FastWAM release")
    model = create_fastwam(**kwargs)
    if pretrained_fastwam_checkpoint:
        if not pretrained_fastwam_sha256:
            raise ValueError("Official full-policy initialization requires its pinned SHA256")
        load_pretrained_fastwam(model, pretrained_fastwam_checkpoint, pretrained_fastwam_sha256)
    model.optimizer_param_groups = MethodType(_optimizer_groups, model)
    return model
