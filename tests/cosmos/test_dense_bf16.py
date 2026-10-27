"""Predictive video conditioning and Cosmos expert regression coverage."""
from contextlib import contextmanager
import json
from pathlib import Path

import pytest
import torch

from fastwam.cosmos.dense_features import DenseFeatureJointModel
from fastwam.cosmos.jepa_video_cache import episode_splits, valid_windows


@contextmanager
def cpu_bf16():
    with torch.backends.mkldnn.flags(enabled=False), torch.autocast('cpu', enabled=False):
        yield


def tiny_model(checkpointing=False, lambda_future=1.):
    return DenseFeatureJointModel(manifest={'scope': 'jepa-regression'},
        feature_dim=8, hidden_dim=16, depth=2, heads=2, view_grid=1,
        action_hidden=16, action_ffn=32, force_hidden=16, force_ffn=32,
        joint_heads=2, joint_head_dim=8, context_dim=16,
        lambda_future=lambda_future, gradient_checkpointing=checkpointing).to(dtype=torch.bfloat16)


def sample(batch=2):
    generator = torch.Generator().manual_seed(123)
    normal = lambda *shape: torch.randn(*shape, generator=generator)
    return dict(visual_features=normal(batch, 5, 2, 8),
        action=normal(batch, 32, 8), history=normal(batch, 32, 14),
        proprio=normal(batch, 32, 9), wrench_future=normal(batch, 32, 6),
        action_is_pad=torch.zeros(batch, 32, dtype=torch.bool),
        image_is_pad=torch.zeros(batch, 9, dtype=torch.bool),
        history_is_pad=torch.zeros(batch, 32, dtype=torch.bool),
        force_is_pad=torch.zeros(batch, 32, dtype=torch.bool),
        history_channel_valid=torch.ones(batch, 32, 14, dtype=torch.bool),
        wrench_valid=torch.ones(batch, 32, 6, dtype=torch.bool))


@pytest.mark.parametrize('checkpointing', [False, True])
@pytest.mark.parametrize('future_weight', [0., 1.])
def test_bf16_backward_reaches_queries_and_all_experts(checkpointing, future_weight):
    model = tiny_model(checkpointing, future_weight)
    with cpu_bf16():
        loss, terms = model.training_loss(sample())
        assert torch.isfinite(loss) and loss.dtype == torch.float32
        assert terms['loss_future_weighted'] == pytest.approx(future_weight*terms['loss_future_l1'])
        loss.backward()
    assert model.dit['video'].mask_token.grad.float().abs().sum() > 0
    for name in ('video', 'force', 'action', 'action_source_time', 'proprio_encoder'):
        gradients = [p.grad for p in model.dit[name].parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients), name
        assert sum(g.float().square().sum() for g in gradients) > 0


def test_ablation_initialization_is_identical():
    first, second = tiny_model(lambda_future=1.), tiny_model(lambda_future=0.)
    assert first.state_dict().keys() == second.state_dict().keys()
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


def test_inference_caches_predictor_once_and_ignores_future_labels_and_padding():
    model = tiny_model()
    data = sample()
    count = []
    handle = model.dit['video'].register_forward_hook(lambda *args: count.append(1))
    with cpu_bf16():
        original = model.infer_joint(data, num_inference_steps=2)
        assert len(count) == 1
        changed = {k: v.clone() for k, v in data.items()}
        changed['visual_features'][:, 1:] = float('nan')
        changed['action'][:] = float('nan')
        changed['wrench_future'][:] = float('nan')
        changed['proprio'][:, 1:] = float('nan')
        changed['force_is_pad'][:] = True
        changed['action_is_pad'][:] = True
        changed['wrench_valid'][:] = False
        later = model.infer_joint(changed, num_inference_steps=2)
        assert len(count) == 2
    handle.remove()
    for key in original:
        assert torch.isfinite(original[key]).all()
        torch.testing.assert_close(original[key], later[key], rtol=0, atol=0)
    assert original['future_features'].shape == (2, 4, 2, 8)


def test_queries_are_not_exported_in_expert_caches_and_action_loss_trains_queries():
    model = tiny_model(lambda_future=0.)
    data = sample()
    with cpu_bf16():
        prepared = model.prepare_context(data['visual_features'][:, 0].bfloat16(), data)
        assert all(k.shape == (2, 2, 2, 8) for k, v, hk, hv in prepared[1])
        tf, ta = torch.full((2,), 200.), torch.full((2,), 500.)
        force, action = model.joint_denoise_core(data['wrench_future'].bfloat16(),
            data['action'].bfloat16(), tf, ta, data, prepared)
        action.float().square().mean().backward()
    assert model.dit['video'].mask_token.grad.float().abs().sum() > 0
    assert model.dit['video'].blocks[0].qkv.weight.grad.float().abs().sum() > 0


def test_episode_splits_are_disjoint_and_windows_respect_discontinuities():
    splits = episode_splits()
    assert [len(splits[k]) for k in ('train', 'val', 'test')] == [55, 8, 8]
    assert sorted(sum(splits.values(), [])) == list(range(71))
    from types import SimpleNamespace
    source = SimpleNamespace(rows=[(0, f, 0, 80) for f in range(80)],
                             bounds=[(0, 40) if f < 40 else (40, 80) for f in range(80)])
    assert [f for _, _, f in valid_windows(source)] == list(range(4, 8)) + list(range(44, 48))


def test_checkpoint_rejects_old_schema_and_other_experiment(tmp_path):
    main = tiny_model(lambda_future=1.)
    path = tmp_path / 'model.pt'
    main.save_checkpoint(path, step=3)
    assert tiny_model(lambda_future=1.).load_checkpoint(path) == 3
    with pytest.raises(ValueError, match='experiment identity'):
        tiny_model(lambda_future=0.).load_checkpoint(path)
