"""Weak routing regularizer: real attention gradients and legacy compatibility."""
import copy
import math
from unittest.mock import patch

import pytest
import torch

from fastwam.cosmos.jepa_force_latent import (
    ActionGuidedVisualReader, JepaForceLatentModel, spatial_query_redundancy,
)


def projected_logits(logits):
    """Use identity Q to generate arbitrary known actual scaled QK logits."""
    queries, count = logits.shape
    q = torch.eye(queries)[None, None]
    k = (logits.T * math.sqrt(queries))[None, None]
    return q, k, torch.ones(1, count, dtype=torch.bool)


def test_diverse_spatial_maps_have_less_penalty_than_collapsed_maps():
    collapsed = torch.tensor([[.7, .1, .1, .1]]).repeat(3, 2)
    diverse = torch.tensor([[.7, .1, .1, .1], [.1, .7, .1, .1], [.1, .1, .7, .1]]).repeat(1, 2)
    first = spatial_query_redundancy(*projected_logits(collapsed.log()))
    second = spatial_query_redundancy(*projected_logits(diverse.log()))
    assert first['loss'].item() == pytest.approx(.05)
    assert second['loss'].item() == 0.
    # Orthogonal query features alone cannot evade the same spatial routing.
    q, _, _ = projected_logits(collapsed.log())
    torch.testing.assert_close(q @ q.transpose(-1, -2), torch.eye(3)[None, None])
    assert first['loss'].item() > second['loss'].item()


def test_camera_only_separation_cannot_lower_spatial_penalty():
    logits = torch.tensor([[3., 1., 0., 2.]]).repeat(3, 2)
    separated = logits.clone()
    separated[0, :4] += 30.
    separated[0, 4:] -= 30.
    separated[1, :4] -= 20.
    separated[1, 4:] += 20.
    a = spatial_query_redundancy(*projected_logits(logits))
    b = spatial_query_redundancy(*projected_logits(separated))
    torch.testing.assert_close(a['loss'], b['loss'], atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_real_q_and_k_receive_finite_nonzero_gradients_even_under_autocast(dtype):
    torch.manual_seed(17)
    q = (torch.randn(2, 2, 1, 8) + .1 * torch.randn(2, 2, 4, 8)).to(dtype).requires_grad_()
    k = torch.randn(2, 2, 8, 8).to(dtype).requires_grad_()
    valid = torch.ones(2, 8, dtype=torch.bool)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        penalty = spatial_query_redundancy(q, k, valid)
    assert penalty['loss'].dtype == torch.float32 and penalty['loss'] > 0
    penalty['loss'].backward()
    for gradient in (q.grad, k.grad):
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_padding_and_views_without_spatial_choice_are_ignored():
    torch.manual_seed(12)
    q = torch.randn(1, 2, 3, 8, requires_grad=True)
    k = torch.randn(1, 2, 8, 8, requires_grad=True)
    # First camera has 2 valid patches; second has just one, hence no penalty.
    valid = torch.tensor([[True, True, False, False, False, True, False, False]])
    first = spatial_query_redundancy(q, k, valid, margin=0.)
    changed = k.detach().clone()
    changed[:, :, 2:] += 1e3
    second = spatial_query_redundancy(q, changed, valid, margin=0.)
    torch.testing.assert_close(first['loss'], second['loss'], atol=0, rtol=0)
    first['loss'].backward()
    assert k.grad[:, :, 2:].count_nonzero() == 0
    for empty in (torch.zeros_like(valid), torch.tensor([[True, False, False, False, False, False, False, False]])):
        result = spatial_query_redundancy(q, k, empty)
        assert torch.isfinite(result['loss']) and result['loss'] == 0
    assert spatial_query_redundancy(q[:, :, :1], k, valid)['loss'] == 0


def test_loss_can_reduce_near_duplicate_routing_without_forcing_disjoint_maps():
    torch.manual_seed(67)
    q = torch.nn.Parameter(torch.randn(1, 1, 1, 8) + .1 * torch.randn(1, 1, 4, 8))
    k = torch.nn.Parameter(torch.randn(1, 1, 16, 8))
    valid = torch.ones(1, 16, dtype=torch.bool)
    optimizer = torch.optim.Adam([q, k], lr=.05)
    initial = float(spatial_query_redundancy(q, k, valid)['loss'].detach())
    assert initial > .02
    for _ in range(30):
        optimizer.zero_grad()
        loss = spatial_query_redundancy(q, k, valid)['loss']
        loss.backward()
        optimizer.step()
    final = spatial_query_redundancy(q, k, valid)
    assert final['loss'].item() < initial * .1
    assert final['mean_cosine'].item() > .3  # no orthogonal/disjoint requirement


def tiny_reader(dtype=torch.float32):
    torch.manual_seed(22)
    reader = ActionGuidedVisualReader(feature_dim=12, action_dim=8, state_dim=8,
        history_steps=4, text_dim=6, force_hidden=16, width=16, query_count=4,
        depth=2, heads=2, horizon=5).to(dtype=dtype)
    normal = lambda *s: torch.randn(*s).to(dtype)
    observation = reader.prepare_observation(normal(2, 8, 12), torch.ones(2, 8, dtype=torch.bool),
        normal(2, 4, 14), torch.ones(2, 4, 14, dtype=torch.bool), normal(2, 9), normal(2, 3, 6))
    return reader, normal(2, 5, 8), torch.tensor([250., 750.], dtype=dtype), observation


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_enabled_side_loss_preserves_exact_sdpa_reader_outputs_and_projects_gradients(dtype):
    reader, action, ta, observation = tiny_reader(dtype)
    baseline = reader(action, ta, observation)
    changed = reader(action, ta, observation, query_redundancy_margin=.95)
    for key in ('visual', 'features', 'wrench'):
        torch.testing.assert_close(baseline[key], changed[key], atol=0, rtol=0)
    assert 'query_redundancy' not in baseline
    changed['query_redundancy']['loss'].backward()
    for layer in reader.readers:
        for parameter in (layer.query.weight, layer.key.weight):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert reader.patch[-1].weight.grad.abs().sum() > 0


def make_model(**extra):
    from test_noaction import model as baseline_model
    baseline = baseline_model(False)
    return JepaForceLatentModel(copy.deepcopy(baseline.video.net), manifest={'scope': 'test'},
        action_hidden=24, action_ffn=48, force_hidden=24, force_ffn=48, force_heads=2,
        gradient_checkpointing=False, action_space='eef_pose7',
        allow_preencoded_patches=True, jepa_feature_dim=12, reader_width=16,
        reader_heads=2, query_count=4, reader_depth=2, injection_layers=(1, 2), **extra)


def test_disabled_legacy_manifest_state_and_checkpoint_are_compatible(tmp_path):
    from test_jepa_force_latent import observed_sample
    model = make_model()
    explicit = make_model(query_redundancy_weight=0.)
    assert model.manifest == explicit.manifest
    assert 'query_redundancy' not in model.manifest['jepa_force_latent']
    legacy_jepa_fields = {'feature_dim', 'reader_width', 'query_count', 'reader_depth', 'reader_heads',
        'injection_layers', 'seed', 'encoder', 'allow_preencoded_patches', 'latent_loss_weight',
        'wrench_loss_weight', 'latent_warmup_steps', 'target', 'target_normalization', 'visibility',
        'feedback', 'query_residual_bypass'}
    assert set(model.manifest['jepa_force_latent']) == legacy_jepa_fields
    enabled = make_model(query_redundancy_weight=.01)
    assert set(model.state_dict()) == set(enabled.state_dict())
    assert set(dict(model.named_parameters())) == set(dict(enabled.named_parameters()))
    path = tmp_path / 'legacy.pt'
    model.save_checkpoint(path, step=123)
    assert explicit.load_checkpoint(path) == 123
    with pytest.raises(ValueError, match='differ'):
        enabled.load_checkpoint(path)
    sample = observed_sample()
    with patch('fastwam.cosmos.jepa_force_latent.spatial_query_redundancy', side_effect=AssertionError('slow path')):
        torch.manual_seed(71)
        first, fm = model.training_loss(sample)
        torch.manual_seed(71)
        second, sm = explicit.training_loss(sample)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert fm == sm and 'loss_query_redundancy' not in fm


def test_enabled_inference_keeps_fastpath_and_matches_legacy_exactly():
    from test_jepa_force_latent import observed_sample
    model = make_model(query_redundancy_weight=.01)
    # Identical initialized tensors; only protocol metadata differs.
    legacy = make_model()
    data = observed_sample()
    model.dit['jepa_adapter'].gates.data.fill_(.2)
    legacy.dit['jepa_adapter'].gates.data.fill_(.2)
    with patch('fastwam.cosmos.jepa_force_latent.spatial_query_redundancy', side_effect=AssertionError('eval slow path')):
        result = model.infer_joint(data, num_inference_steps=2)
        reference = legacy.infer_joint(data, num_inference_steps=2)
    for key in reference:
        torch.testing.assert_close(result[key], reference[key], atol=0, rtol=0)


def test_enabled_training_adds_only_warmup_scaled_penalty_and_gradients():
    from test_jepa_force_latent import observed_sample
    enabled = make_model(query_redundancy_weight=.01, query_redundancy_warmup_steps=2500)
    legacy = make_model()
    data = observed_sample()
    for step, expected_weight in ((0, 0.), (1250, .005), (2500, .01), (5000, .01)):
        enabled.set_auxiliary_step(step)
        legacy.set_auxiliary_step(step)
        torch.manual_seed(77)
        base_loss, base_metrics = legacy.training_loss(data)
        torch.manual_seed(77)
        loss, metrics = enabled.training_loss(data)
        assert metrics['query_redundancy_weight'] == expected_weight
        assert all(metrics[k] == value for k, value in base_metrics.items())
        assert 0 <= metrics['loss_query_redundancy'] <= .050001
        torch.testing.assert_close(loss - base_loss,
            loss.new_tensor(expected_weight * metrics['loss_query_redundancy']), atol=2e-7, rtol=1e-3)
    loss.backward()
    for layer in enabled.dit['jepa_reader'].readers:
        assert layer.query.weight.grad.abs().sum() > 0
        assert layer.key.weight.grad.abs().sum() > 0


def test_enabled_loss_checkpoint_recomputation_and_optimizer_resume_match(tmp_path):
    from test_jepa_force_latent import observed_sample
    plain = make_model(query_redundancy_weight=.01)
    plain.dit['jepa_adapter'].gates.data.fill_(.2)
    recomputed = copy.deepcopy(plain)
    recomputed.gradient_checkpointing = True
    data = observed_sample()
    for model in (plain, recomputed):
        model.set_auxiliary_step(1250)
        torch.manual_seed(19)
        loss, _ = model.training_loss(data)
        loss.backward()
    for (name, first), (_, second) in zip(plain.named_parameters(), recomputed.named_parameters()):
        if first.grad is None:
            assert second.grad is None, name
        else:
            torch.testing.assert_close(first.grad, second.grad, atol=2e-6, rtol=3e-5, msg=name)
    optimizer = torch.optim.AdamW(plain.parameters(), lr=1e-4)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    path = tmp_path / 'regularized.pt'
    plain.save_checkpoint(path, optimizer=optimizer, step=1251)

    def update(model, opt, step):
        model.set_auxiliary_step(step)
        opt.zero_grad(set_to_none=True)
        loss, metrics = model.training_loss(data)
        loss.backward()
        opt.step()
        return loss.detach(), metrics

    expected, expected_metrics = update(plain, optimizer, 1251)
    restored = make_model(query_redundancy_weight=.01)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    restored_step = restored.load_checkpoint(path, restored_optimizer)
    actual, actual_metrics = update(restored, restored_optimizer, restored_step)
    assert actual_metrics['query_redundancy_weight'] == .01 * (1251 / 2500)
    assert actual_metrics == expected_metrics
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for first, second in zip(plain.parameters(), restored.parameters()):
        torch.testing.assert_close(first, second, atol=0, rtol=0)


@pytest.mark.parametrize('settings', [dict(query_redundancy_weight=-.1), dict(query_redundancy_weight=float('nan')),
    dict(query_redundancy_margin=1.), dict(query_redundancy_margin=float('inf')),
    dict(query_redundancy_warmup_steps=True), dict(query_redundancy_warmup_steps=1.5)])
def test_model_rejects_invalid_redundancy_settings(settings):
    with pytest.raises(ValueError, match='redundancy'):
        make_model(**settings)


@pytest.mark.parametrize('margin', [-.1, 1., float('nan'), float('inf')])
def test_helper_rejects_invalid_margin(margin):
    with pytest.raises(ValueError, match='margin'):
        spatial_query_redundancy(*projected_logits(torch.zeros(3, 8)), margin=margin)
