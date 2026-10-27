"""Checkpoint fingerprints must survive a fresh object's deserialization."""
import copy
import sys
import types
from dataclasses import make_dataclass

import pytest
import torch

from fastwam.cosmos.persistence import state_digest


@pytest.fixture
def checkpoint_objects(monkeypatch):
    # Match the actual DeepSpeed 0.18.7 checkpoint classes without importing
    # DeepSpeed's CUDA extensions in the CPU unit suite. torch.save/load below
    # genuinely pickles and reconstructs these objects at the production path.
    name = 'deepspeed.runtime.fp16.loss_scaler'
    for package in ('deepspeed', 'deepspeed.runtime', 'deepspeed.runtime.fp16',
                    'deepspeed.utils'):
        module = types.ModuleType(package)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, package, module)
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    for cls_name in ('LossScaler', 'DynamicLossScaler'):
        setattr(module, cls_name, type(cls_name, (), {'__module__': name}))
    static = module.LossScaler()
    static.cur_scale = 1.0
    static.dynamic = False
    dynamic = module.DynamicLossScaler()
    dynamic.__dict__.update(cur_scale=65536.0, dynamic=True, cur_iter=2,
        last_overflow_iter=-1, scale_factor=2.0, scale_window=1000, min_scale=1,
        delayed_shift=2, cur_hysteresis=2, consecutive_hysteresis=False,
        raise_error_at_min_scale=True, dtype=torch.float16)
    fragment_module = types.ModuleType('deepspeed.utils.tensor_fragment')
    monkeypatch.setitem(sys.modules, fragment_module.__name__, fragment_module)
    fragment_module.fragment_address = make_dataclass('fragment_address',
        [('numel', int), ('start', int)], namespace={'__module__': fragment_module.__name__})
    return static, dynamic, fragment_module.fragment_address(numel=8, start=4)


@pytest.mark.parametrize('scaler_index', [0, 1])
def test_optimizer_scaler_fingerprint_roundtrip_and_continuation(tmp_path, checkpoint_objects, scaler_index):
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=1000)
    inputs = torch.randn(3, 4)
    def update(net, opt, sched):
        opt.zero_grad()
        net(inputs).square().mean().backward()
        opt.step()
        sched.step()
    for _ in range(2):
        update(model, optimizer, scheduler)
    state = {'model': model.state_dict(),
             'optimizer': {'base_optimizer_state': optimizer.state_dict(),
                           'loss_scaler': checkpoint_objects[scaler_index],
                           'param_slice_mappings': [{'weight': checkpoint_objects[2]}],
                           'dynamic_loss_scale': bool(scaler_index)},
             'scheduler': scheduler.state_dict(), 'rng': torch.get_rng_state()}
    original = state_digest(state)
    path = tmp_path / 'state.pt'
    torch.save(state, path)
    loaded = torch.load(path, weights_only=False)
    assert loaded['optimizer']['loss_scaler'] is not state['optimizer']['loss_scaler']
    assert state_digest(loaded) == original

    restored = torch.nn.Linear(4, 2)
    other = torch.optim.AdamW(restored.parameters(), lr=1e-4, betas=(0.9, 0.95))
    other_schedule = torch.optim.lr_scheduler.LinearLR(other, total_iters=1000)
    restored.load_state_dict(loaded['model'])
    other.load_state_dict(loaded['optimizer']['base_optimizer_state'])
    other_schedule.load_state_dict(loaded['scheduler'])
    assert state_digest(other.state_dict()) == state_digest(optimizer.state_dict())
    assert state_digest(other_schedule.state_dict()) == state_digest(scheduler.state_dict())
    update(model, optimizer, scheduler)
    update(restored, other, other_schedule)
    assert state_digest(model.state_dict()) == state_digest(restored.state_dict())
    assert state_digest(optimizer.state_dict()) == state_digest(other.state_dict())
    assert state_digest(scheduler.state_dict()) == state_digest(other_schedule.state_dict())

    loaded['optimizer']['loss_scaler'].cur_scale *= 2
    assert state_digest(loaded) != original


def test_scaler_fingerprint_covers_every_state_field_and_class(checkpoint_objects):
    for scaler in checkpoint_objects:
        baseline = state_digest(scaler)
        assert state_digest(copy.deepcopy(scaler)) == baseline
        for field, value in vars(scaler).items():
            changed = copy.deepcopy(scaler)
            replacement = (not value if isinstance(value, bool) else
                           torch.bfloat16 if isinstance(value, torch.dtype) else value + 1)
            setattr(changed, field, replacement)
            assert state_digest(changed) != baseline, field
    static, dynamic, _ = copy.deepcopy(checkpoint_objects)
    dynamic.__dict__.clear()
    dynamic.__dict__.update(vars(static))
    assert state_digest(static) != state_digest(dynamic)


def test_fingerprint_does_not_silently_stringify_unknown_objects():
    with pytest.raises(TypeError):
        state_digest({'unsupported': object()})
