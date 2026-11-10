"""Reference conditioning contracts, using native tiny networks (not task quality)."""
import copy
from unittest.mock import patch

import pytest
import torch

from conftest import sample
from test_noaction import net
from fastwam.cosmos.reference_auditor import ObservationReferenceAuditor, alignment_loss
from fastwam.cosmos.reference_model import ReferenceJointModel


def reference_sample(batch=1, length=19):
    s = sample(batch)
    generator = torch.Generator().manual_seed(154)
    normal = lambda *shape: torch.randn(*shape, generator=generator)
    s.update(action=normal(batch, 32, 8), proprio=normal(batch, 32, 9),
             history=normal(batch, 32, 14),
             history_channel_valid=torch.ones(batch, 32, 14, dtype=torch.bool))
    order = torch.linspace(0, 1, length).expand(batch, -1).clone()
    pose = torch.zeros(batch, length, 8)
    pose[..., 0] = .2 + .2 * order
    pose[..., 1] = .1 * torch.sin(order * 5)
    pose[..., 2] = .3 + .1 * order
    pose[..., 6] = 1
    pose[..., 7] = (order < .4).float()
    current = pose[:, 0].clone()
    s['reference_inputs'] = dict(reference_pose=pose,
        reference_valid=torch.ones(batch, length, dtype=torch.bool),
        reference_channel_valid=torch.ones(batch, length, 3, dtype=torch.bool),
        reference_order=order, reference_break_before=torch.zeros(batch, length, dtype=torch.bool),
        reference_present=torch.ones(batch, dtype=torch.bool), current_tcp=current)
    target = torch.zeros(batch, length)
    target[:, length // 3] = 1
    s['reference_supervision'] = dict(alignment_target=target,
                                     alignment_enabled=torch.ones(batch, dtype=torch.bool))
    return s


def observations(s):
    return {key: s[key] for key in ('first_frame_latents', 'proprio', 'history',
                                    'history_is_pad', 'history_channel_valid')}


def auditor():
    torch.manual_seed(13)
    return ObservationReferenceAuditor(hidden=16, heads=2, layers=1)


def model(checkpointing=False):
    torch.manual_seed(7)
    return ReferenceJointModel(net(), manifest={'scope': 'test'},
        action_hidden=24, action_ffn=48, force_hidden=24, force_ffn=48, force_heads=2,
        action_space='joint8', action_representation='delta_current_joint7',
        reference_hidden=16, reference_heads=2, reference_layers=1,
        gradient_checkpointing=checkpointing)


def core_inputs(s):
    video = s['video_latents'].clone()
    video[:, :, :1] = s['first_frame_latents']
    batch = len(video)
    return (video, s['wrench_future'], s['action'], torch.full((batch,), 700.),
            torch.full((batch,), 450.), torch.full((batch,), 200.))


@pytest.mark.parametrize('length', [7, 32, 81])
def test_full_variable_length_plan_produces_32_condition_slots(length):
    s = reference_sample(length=length)
    result = auditor()(s['reference_inputs'], **observations(s))
    assert result['local_tokens'].shape == (1, 32, 16)
    assert result['alignment_probabilities'].shape == (1, length)
    torch.testing.assert_close(result['alignment_probabilities'].sum(-1), torch.ones(1))
    assert all(torch.isfinite(result[key]).all() for key in ('local_tokens', 'progress', 'span'))


def test_padding_and_masked_geometry_cannot_change_condition():
    a = auditor(); s = reference_sample(length=11)
    original = a(s['reference_inputs'], **observations(s))
    padded = copy.deepcopy(s['reference_inputs'])
    for key in ('reference_pose', 'reference_order'):
        shape = list(padded[key].shape); shape[1] = 9
        padded[key] = torch.cat((padded[key], torch.full(shape, float('nan'))), 1)
    for key in ('reference_valid', 'reference_channel_valid', 'reference_break_before'):
        shape = list(padded[key].shape); shape[1] = 9
        padded[key] = torch.cat((padded[key], torch.zeros(shape, dtype=torch.bool)), 1)
    actual = a(padded, **observations(s))
    for key in ('local_tokens', 'progress', 'span'):
        torch.testing.assert_close(original[key], actual[key], atol=2e-6, rtol=2e-5)
    assert actual['alignment_probabilities'][:, 11:].count_nonzero() == 0


def test_observation_changes_alignment_and_target_metadata_is_rejected():
    a = auditor(); s = reference_sample()
    first = a(s['reference_inputs'], **observations(s))
    other = copy.deepcopy(s)
    other['first_frame_latents'] += 4
    other['history'] += 2
    second = a(other['reference_inputs'], **observations(other))
    assert not torch.allclose(first['alignment_probabilities'], second['alignment_probabilities'])
    assert not torch.allclose(first['local_tokens'], second['local_tokens'])
    leaked = dict(s['reference_inputs'], source_frame=torch.arange(19)[None])
    with pytest.raises(ValueError, match='explicit observation/reference contract'):
        a(leaked, **observations(s))


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_all_reference_dropout_is_finite_and_returns_zero_condition(dtype):
    a = auditor().to(dtype=dtype); s = reference_sample(batch=2)
    s['reference_inputs']['reference_present'][:] = False
    s['reference_inputs']['reference_pose'][:] = float('nan')
    target = torch.zeros_like(s['reference_supervision']['alignment_target'])
    with torch.backends.mkldnn.flags(enabled=False):
        output = a(s['reference_inputs'], **observations(s))
        loss = alignment_loss(output['alignment_log_probabilities'], target, torch.zeros(2, dtype=torch.bool))
        loss = loss + output['local_tokens'].float().square().mean()
        loss.backward()
    assert torch.isfinite(loss) and loss == 0
    assert output['local_tokens'].count_nonzero() == 0
    assert output['alignment_probabilities'].count_nonzero() == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.parameters())


def test_reference_affects_only_action_and_action_loss_trains_auditor():
    m = model(); s = reference_sample(); args = core_inputs(s)
    first = m.joint_denoise_core(*args, m._conditions(s))
    other = copy.deepcopy(s)
    other['reference_inputs']['reference_pose'][..., :3] += .2
    second = m.joint_denoise_core(*args, m._conditions(other))
    torch.testing.assert_close(first[0], second[0], atol=0, rtol=0)
    torch.testing.assert_close(first[1], second[1], atol=0, rtol=0)
    assert not torch.allclose(first[2], second[2])
    assert first[2].shape == (1, 32, 8)
    # No alignment CE: action-only gradients must reach soft addressing itself.
    first[2].square().mean().backward()
    audit = m.dit['reference_auditor']
    for parameter in (audit.reference_embedding[0].weight, audit.query.weight,
                      audit.key.weight, audit.span.weight, m.dit['reference_condition'][0].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    optimized = {id(p) for p in m.dit.parameters()} | {id(p) for p in m.proprio_encoder.parameters()}
    assert all(id(p) in optimized for p in audit.parameters())


def test_joint_training_calls_auditor_once_and_checkpointing_preserves_gradient():
    m = model(); s = reference_sample(batch=2)
    # Exercise mixed present/dropped references in one batch.
    s['reference_inputs']['reference_present'][1] = False
    s['reference_supervision']['alignment_target'][1] = 0
    s['reference_supervision']['alignment_enabled'][1] = False
    with patch.object(m.dit['reference_auditor'], 'forward', wraps=m.dit['reference_auditor'].forward) as audit, \
         patch.object(m, 'infer_joint', side_effect=AssertionError('No action prepass')):
        torch.manual_seed(99); loss, metrics = m.training_loss(s); loss.backward()
        assert audit.call_count == 1 and torch.isfinite(loss)
        assert metrics['reference_present_fraction'] == .5
    checkpointed = model(True); checkpointed.load_state_dict(m.state_dict())
    torch.manual_seed(99); other_loss, _ = checkpointed.training_loss(s); other_loss.backward()
    torch.testing.assert_close(loss, other_loss)
    for (name, p), (_, q) in zip(m.named_parameters(), checkpointed.named_parameters()):
        if p.grad is None:
            assert q.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=3e-5, msg=name)


def test_inference_cache_parity_and_no_teacher_future_dependency():
    m = model(); s = reference_sample()
    with torch.no_grad():
        audit = m._conditions(s)[2]
    other = copy.deepcopy(s)
    for key in ('action', 'video_latents', 'wrench_future'):
        other[key] = torch.randn_like(other[key]) * 100
    other['proprio'][:, 1:] = 999
    del other['reference_supervision']
    with torch.no_grad():
        changed = m._conditions(other)[2]
    for key in audit:
        torch.testing.assert_close(audit[key], changed[key], atol=0, rtol=0)
    uncached = m.infer_joint(s, num_inference_steps=2, use_cache=False)
    cached = m.infer_joint(other, num_inference_steps=2, use_cache=True)
    for key in uncached:
        torch.testing.assert_close(uncached[key], cached[key], atol=3e-6, rtol=3e-5)
    with pytest.raises(ValueError, match='gradients'):
        m.prefill_static(s['first_frame_latents'], m._conditions(s))


def test_reference_checkpoint_restores_next_update_and_binds_auditor_config(tmp_path):
    m = model(); s = reference_sample()
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4)
    def update(module, opt):
        opt.zero_grad(); loss, _ = module.training_loss(s); loss.backward(); opt.step()
        return loss.detach()
    update(m, optimizer)
    path = tmp_path / 'reference.pt'
    m.save_checkpoint(path, optimizer, 1)
    expected = update(m, optimizer)
    restored = model(); restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    assert restored.load_checkpoint(path, restored_optimizer) == 1
    torch.testing.assert_close(update(restored, restored_optimizer), expected, atol=0, rtol=0)
    for p, q in zip(m.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    incompatible = model(); incompatible.manifest['reference']['alignment_weight'] = .2
    with pytest.raises(ValueError, match='differ'):
        incompatible.load_checkpoint(path)


def test_bf16_training_and_sampling_are_finite():
    m = model(True).to(dtype=torch.bfloat16); s = reference_sample()
    with torch.backends.mkldnn.flags(enabled=False):
        loss, _ = m.training_loss(s); loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        for value in m.infer_joint(s, num_inference_steps=2).values():
            assert torch.isfinite(value).all()
