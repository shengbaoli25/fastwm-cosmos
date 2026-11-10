"""Native tiny-network tests of composed reference/JEPA interfaces, not quality."""
import copy
from unittest.mock import patch

import pytest
import torch

from test_noaction import net
from test_reference_model import reference_sample, core_inputs, model as reference_only
from test_jepa_force_latent import CountingEncoder
from fastwam.cosmos.jepa_force_latent import ObservationConditions
from fastwam.cosmos.reference_model import (ReferenceJepaModel, ReferenceObservationConditions,
    create_reference_jepa_cosmos, JEPA_REFERENCE_SCHEMA)


def model(checkpointing=False, encoder=None):
    torch.manual_seed(7)
    return ReferenceJepaModel(net(), manifest={'scope': 'test'},
        action_hidden=24, action_ffn=48, force_hidden=24, force_ffn=48, force_heads=2,
        action_space='joint8', action_representation='delta_current_joint7',
        reference_hidden=16, reference_heads=2, reference_layers=1,
        expected_reference_identity='b' * 64, gradient_checkpointing=checkpointing,
        observation_encoder=encoder, allow_preencoded_patches=encoder is None,
        jepa_feature_dim=12, reader_width=16, reader_heads=2, query_count=4,
        injection_layers=(1, 2))


def observed_sample(batch=1):
    data = reference_sample(batch=batch)
    data['jepa_observed_patches'] = torch.randn(batch, 8, 12, generator=torch.Generator().manual_seed(81))
    return data


def test_composition_retains_action_conditioned_force_residual_and_reference_isolation():
    m = model()
    m.dit['jepa_adapter'].gates.data.fill_(.3)
    sample = observed_sample()
    args = core_inputs(sample)
    condition = m._conditions(sample)
    assert isinstance(condition, (ObservationConditions, ReferenceObservationConditions))
    assert len(condition) == 5 and condition.reference_condition.shape == (1, 32, 24)
    before = m.joint_denoise_core(*args, condition, return_auxiliary=True)
    changed = copy.deepcopy(sample)
    changed['reference_inputs']['reference_pose'][..., :3] += .2
    after = m.joint_denoise_core(*args, m._conditions(changed), return_auxiliary=True)
    for first, second in zip(before[:2], after[:2]):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert not torch.allclose(before[2], after[2])
    # The same-step JEPA query is still noisy action plus observations, never R.
    torch.testing.assert_close(before[3]['action_force_context']['features'],
                               after[3]['action_force_context']['features'], atol=0, rtol=0)
    changed_args = list(args)
    changed_args[2] = changed_args[2] + 3
    changed_args[5] = changed_args[5] + 100
    action_changed = m.joint_denoise_core(*changed_args, condition, return_auxiliary=True)
    for first, second in zip(before[:2], action_changed[:2]):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert not torch.allclose(before[3]['action_force_context']['features'],
                              action_changed[3]['action_force_context']['features'])


def test_reference_dropout_restores_unchanged_jepa_path_and_zero_gate_matches_reference_only():
    m = model()
    sample = observed_sample()
    args = core_inputs(sample)
    # Before a JEPA gate opens the reference-only forward is exactly preserved.
    original = reference_only()
    for actual, expected in zip(m.joint_denoise_core(*args, m._conditions(sample)),
                                original.joint_denoise_core(*args, original._conditions(sample))):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    m.dit['jepa_adapter'].gates.data.fill_(.4)
    sample['reference_inputs']['reference_present'][:] = False
    sample['reference_inputs']['reference_pose'][:] = float('nan')
    condition = m._conditions(sample)
    assert condition.reference_condition.count_nonzero() == 0
    result = m.joint_denoise_core(*args, condition)
    with patch.object(m, '_condition_action_input', side_effect=lambda tokens, _: tokens):
        unchanged_jepa = m.joint_denoise_core(*args, condition)
    for actual, expected in zip(result, unchanged_jepa):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_training_composes_both_auxiliary_losses_and_gradient_checkpointing():
    m = model()
    sample = observed_sample(batch=2)
    m.set_auxiliary_step(1250)
    with patch.object(m.dit['reference_auditor'], 'forward', wraps=m.dit['reference_auditor'].forward) as audit:
        torch.manual_seed(99)
        loss, metrics = m.training_loss(sample)
        loss.backward()
        assert audit.call_count == 1
    assert metrics['force_feature_weight'] == .5
    assert metrics['loss_reference_alignment'] > 0
    for parameter in (m.dit['jepa_reader'].feature_head.weight,
                      m.dit['jepa_adapter'].gates,
                      m.dit['reference_auditor'].reference_embedding[0].weight,
                      m.dit['reference_condition'][-1].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    recomputed = model(True)
    recomputed.load_state_dict(m.state_dict())
    recomputed.set_auxiliary_step(1250)
    torch.manual_seed(99)
    other, _ = recomputed.training_loss(sample)
    other.backward()
    torch.testing.assert_close(loss, other)
    for (name, first), (_, second) in zip(m.named_parameters(), recomputed.named_parameters()):
        if first.grad is None:
            assert second.grad is None, name
        else:
            torch.testing.assert_close(first.grad, second.grad, atol=2e-6, rtol=3e-5, msg=name)


def test_cached_uncached_use_observations_only_and_refresh_jepa_query_each_step():
    encoder = CountingEncoder()
    m = model(encoder=encoder)
    m.dit['jepa_adapter'].gates.data.fill_(.4)
    sample = observed_sample()
    del sample['jepa_observed_patches']
    sample['jepa_observation'] = torch.randn(1, 8, 3)
    with patch.object(m.dit['reference_auditor'], 'forward', wraps=m.dit['reference_auditor'].forward) as audit, \
         patch.object(m.dit['jepa_reader'], 'forward', wraps=m.dit['jepa_reader'].forward) as reader:
        before = m.infer_joint(sample, num_inference_steps=3, use_cache=False)
        assert encoder.calls == 1 and audit.call_count == 1 and reader.call_count == 3
    other = copy.deepcopy(sample)
    for key in ('action', 'video_latents', 'wrench_future'):
        other[key].fill_(float('nan'))
    other['proprio'][:, 1:] = float('nan')
    del other['reference_supervision']
    after = m.infer_joint(other, num_inference_steps=3, use_cache=True)
    for key in before:
        torch.testing.assert_close(before[key], after[key], atol=3e-6, rtol=3e-5)
    assert encoder.calls == 2


def test_checkpoint_schema_reference_identity_and_resumed_ramp_are_bound(tmp_path):
    m = model()
    sample = observed_sample()
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4)

    def update(module, opt, step):
        module.set_auxiliary_step(step)
        opt.zero_grad(set_to_none=True)
        loss, _ = module.training_loss(sample)
        loss.backward()
        opt.step()
        return loss.detach()

    update(m, optimizer, 0)
    path = tmp_path / 'combined.pt'
    m.save_checkpoint(path, optimizer, 1)
    expected = update(m, optimizer, 1)
    restored = model()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    step = restored.load_checkpoint(path, restored_optimizer)
    assert step == 1 and restored.checkpoint_schema == JEPA_REFERENCE_SCHEMA
    torch.testing.assert_close(update(restored, restored_optimizer, step), expected, atol=0, rtol=0)
    for first, second in zip(m.parameters(), restored.parameters()):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    incompatible = model()
    incompatible.manifest['reference_identity'] = 'c' * 64
    with pytest.raises(ValueError, match='differ'):
        incompatible.load_checkpoint(path)
    with pytest.raises(ValueError, match='differ'):
        reference_only().load_checkpoint(path)


def test_combined_factory_uses_official_jepa_construction_and_strict_identity(monkeypatch):
    from fastwam.cosmos import assets
    monkeypatch.setattr(assets, 'read_text_bank', lambda _: (torch.randn(1, 3, 24), {}, 'a' * 64))
    monkeypatch.setattr(assets, 'load_nonaction_video', lambda *args, **kwargs: net())
    kwargs = dict(upstream='test', checkpoint='test', text_cache='test',
        expected_text_identity='a' * 64, expected_data_identity='b' * 64,
        expected_reference_identity='c' * 64, expected_normalization_sha256='d' * 64,
        device='cpu', action_space='joint8', action_representation='delta_current_joint7',
        allow_preencoded_patches=True, jepa_feature_dim=12, reader_width=16, reader_heads=2,
        query_count=4, injection_layers=(1, 2), action_hidden=24, action_ffn=48,
        force_hidden=24, force_ffn=48, force_heads=2,
        reference_hidden=16, reference_heads=2, reference_layers=1)
    m = create_reference_jepa_cosmos(**kwargs)
    assert isinstance(m, ReferenceJepaModel)
    assert m.manifest['schema'] == JEPA_REFERENCE_SCHEMA
    assert m.manifest['reference_identity'] == 'c' * 64
    assert 'jepa_force_latent' in m.manifest and 'reference' in m.manifest
    assert m.manifest['reference']['local_condition_tokens'] == 32
    with pytest.raises(ValueError, match='Pinned'):
        create_reference_jepa_cosmos(**dict(kwargs, expected_reference_identity='unpinned'))


def test_combined_bf16_loss_gradients_and_cached_sampling_are_finite():
    m = model(True).to(dtype=torch.bfloat16)
    m.dit['jepa_adapter'].gates.data.fill_(.2)
    m.set_auxiliary_step(1250)
    sample = observed_sample()
    with torch.backends.mkldnn.flags(enabled=False):
        loss, _ = m.training_loss(sample)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                   for parameter in m.parameters())
        uncached = m.infer_joint(sample, num_inference_steps=2, use_cache=False)
        cached = m.infer_joint(sample, num_inference_steps=2, use_cache=True)
    for key in cached:
        assert torch.isfinite(cached[key]).all() and torch.isfinite(uncached[key]).all()
        torch.testing.assert_close(cached[key], uncached[key], atol=.04, rtol=.04)


def test_combined_real_lora_encoder_stays_frozen_and_receives_auxiliary_gradient(monkeypatch):
    from test_jepa_observation import tiny_official
    from fastwam.cosmos import jepa_observation as jo
    monkeypatch.setitem(jo.VARIANT_SPECS, 'vitb', dict(
        width=24, depth=12, heads=3, factory='vit_base', filename='tiny-test.pt'))
    encoder = jo.JEPAObservationEncoder(tiny_official(), variant='vitb', weight_sha256='e' * 64)
    m = model(encoder=encoder)
    from fastwam.trainer import Wan22Trainer
    Wan22Trainer._apply_dit_only_train_mode(m)
    sample = observed_sample()
    del sample['jepa_observed_patches']
    sample['jepa_observation'] = torch.rand(1, 2, 3, 2, 224, 224) * 2 - 1
    m.set_auxiliary_step(1250)
    loss, _ = m.training_loss(sample)
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in encoder.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if name.endswith(('_q_b', '_v_b')):
                assert parameter.grad.abs().sum() > 0, name
        else:
            assert parameter.grad is None, name
