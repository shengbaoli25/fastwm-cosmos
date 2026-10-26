"""The DINO backbone must load without optional evaluation dependencies."""
import builtins
import hashlib
from pathlib import Path

import torch

from fastwam.cosmos import dense_encoder


ROOT = Path(__file__).resolve().parents[2]


def test_dinov3_loads_backbone_without_hub_or_evaluation_imports(tmp_path, monkeypatch):
    weights = tmp_path / 'model.safetensors'
    payload = b'checkpoint loading is isolated from this import regression'
    weights.write_bytes(payload)
    monkeypatch.setenv('DINOV3_LICENSE_ACCEPTED', '1')
    monkeypatch.setattr(dense_encoder, '_load_dinov3_safetensors', lambda *args: None)
    original_import = builtins.__import__

    def backbone_import_only(name, *args, **kwargs):
        if name.startswith('dinov3.eval') or name.startswith('torchmetrics'):
            raise AssertionError('Dense teacher imported optional evaluation modules')
        return original_import(name, *args, **kwargs)

    def no_torch_hub(*args, **kwargs):
        raise AssertionError('Dense teacher attempted a hub lookup')

    monkeypatch.setattr(builtins, '__import__', backbone_import_only)
    monkeypatch.setattr(torch.hub, 'load', no_torch_hub)
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', no_torch_hub)
    teacher = dense_encoder.load_dense_teacher(
        'dinov3', ROOT / 'third_party/dinov3', weights,
        hashlib.sha256(payload).hexdigest(), device='cpu')
    assert teacher.feature_tokens == 392
    assert teacher.feature_dim == 768
    assert len(teacher.encoder.blocks) == 12
    assert not teacher.encoder.training
    assert all(not parameter.requires_grad for parameter in teacher.encoder.parameters())
