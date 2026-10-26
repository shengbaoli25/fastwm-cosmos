"""Exercise custom dense-model methods without an autocast forward wrapper."""
from contextlib import contextmanager

import pytest
import torch

from fastwam.cosmos.dense_features import DenseFeatureJointModel


@contextmanager
def cpu_bf16():
    # Some CPU oneDNN builds support BF16 forward but reject its backward.
    with torch.backends.mkldnn.flags(enabled=False), torch.autocast('cpu', enabled=False):
        yield


def tiny_model(checkpointing):
    return DenseFeatureJointModel(manifest={'scope': 'bf16-regression'},
        feature_dim=8, hidden_dim=16, depth=2, heads=2, view_grid=1,
        action_hidden=16, action_ffn=32, force_hidden=16, force_ffn=32,
        gradient_checkpointing=checkpointing).to(device='cpu', dtype=torch.bfloat16)


def sample(batch=2):
    generator = torch.Generator().manual_seed(123)
    normal = lambda *shape: torch.randn(*shape, generator=generator)
    result = dict(visual_features=normal(batch, 3, 2, 8),
        action=normal(batch, 32, 8), history=normal(batch, 32, 14),
        proprio=normal(batch, 32, 9), wrench_future=normal(batch, 32, 6),
        action_is_pad=torch.zeros(batch, 32, dtype=torch.bool),
        image_is_pad=torch.zeros(batch, 9, dtype=torch.bool),
        history_is_pad=torch.zeros(batch, 32, dtype=torch.bool),
        force_is_pad=torch.zeros(batch, 32, dtype=torch.bool),
        history_channel_valid=torch.ones(batch, 32, 14, dtype=torch.bool),
        wrench_valid=torch.ones(batch, 32, 6, dtype=torch.bool))
    result['action_is_pad'][0, -2:] = True
    result['force_is_pad'][0, -2:] = True
    result['history_channel_valid'][0, :2, :6] = False
    result['image_is_pad'][0, 8] = True
    return result


@pytest.mark.parametrize('checkpointing', [False, True])
def test_bf16_training_loss_backward_and_update_without_autocast(checkpointing):
    model = tiny_model(checkpointing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, foreach=False)
    before = model.dit['video'].time[0].weight.detach().clone()
    with cpu_bf16():
        loss, components = model.training_loss(sample())
        assert loss.dtype == torch.float32 and torch.isfinite(loss)
        assert set(components) == {'loss_video_fm', 'loss_force_fm', 'loss_action_fm'}
        assert all(torch.isfinite(torch.tensor(value)) and value > 0 for value in components.values())
        loss.backward()
        for name in ('video', 'force', 'action', 'action_source_time', 'proprio_encoder'):
            gradients = [parameter.grad for parameter in model.dit[name].parameters()
                         if parameter.requires_grad]
            assert gradients and all(gradient is not None for gradient in gradients), name
            assert all(torch.isfinite(gradient.float()).all() for gradient in gradients), name
            assert sum(gradient.float().square().sum() for gradient in gradients) > 0, name
        optimizer.step()
    assert not torch.equal(before, model.dit['video'].time[0].weight)
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())


def test_bf16_direct_core_accepts_fp32_clocks_and_inference_is_finite():
    model = tiny_model(False)
    data = sample()
    visual = data['visual_features'].to(torch.bfloat16)
    force = data['wrench_future'].to(torch.bfloat16)
    action = data['action'].to(torch.bfloat16)
    clocks = [torch.tensor([250., 750.]), torch.tensor([100., 900.]),
              torch.tensor([400., 600.])]
    with cpu_bf16(), torch.no_grad():
        conditions = model._conditions(data)
        outputs = model.joint_denoise_core(visual[:, 0], visual[:, 1:], force, action,
            *clocks, data, conditions)
        assert [output.shape for output in outputs] == [
            (2, 2, 2, 8), (2, 32, 6), (2, 32, 8)]
        for output in outputs:
            assert output.dtype == torch.bfloat16
            assert torch.isfinite(output.float()).all()
        expert = model.dit['action']
        context, mask = conditions[:2]
        prepared = expert.prepare(action, clocks[2], context, mask)
        legacy = expert.pre_dit(action, clocks[2], context, mask)
        torch.testing.assert_close(prepared[1], legacy['t'], rtol=0, atol=0)
        torch.testing.assert_close(prepared[2], legacy['t_mod'], rtol=0, atol=0)
        inferred = model.infer_joint(data, num_inference_steps=1)
        for output in inferred.values():
            assert output.dtype == torch.bfloat16
            assert torch.isfinite(output.float()).all()
