"""CPU architecture contracts; no claims of GPU speed or model quality."""
import copy
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from conftest import tiny_net, tiny_model, sample
from fastwam.cosmos.joint import JointCosmosForceWM, joint_attention_mask, grouped_attention_mask, joint_token_layout, action_prefix_mask, GROUP_READS, warm_start_joint
from fastwam.cosmos.preflight import validate_config
from fastwam.forcewm.masks import build_force_attention_mask
from fastwam.forcewm.model import masked_force_mse


def joint_model(checkpoint=True, force_hidden=24):
    torch.manual_seed(7)
    return JointCosmosForceWM(tiny_net(), manifest={'scope': 'TEST-joint', 'depth': 2},
        action_hidden=32, action_ffn=64, force_hidden=force_hidden, force_ffn=64,
        force_heads=2, memory_width=16, memory_heads=2,
        generated_video_steps=1, gradient_checkpointing=checkpoint)


def world_models(model, s):
    raw, ctx, cmask = model._contexts(s)
    return model._world_models(s, s['first_frame_latents'].to(model.torch_dtype),
        raw, ctx, cmask, s['action'].to(model.torch_dtype), s['action_is_pad'])


def test_joint_mask_matches_requested_groups_and_excludes_unavailable_tokens():
    valid = torch.tensor([[False, True, True, False, True, True]])
    groups, all_valid = joint_token_layout(3, 1, valid, 2, 2)
    mask = joint_attention_mask(3, 1, valid, 2, 2)
    reads = torch.tensor(GROUP_READS, dtype=torch.bool)
    for i in range(9):
        for j in range(9):
            expected = (bool(reads[groups[i], groups[j]]) and bool(all_valid[0, i])
                        and bool(all_valid[0, j])) if all_valid[0, i] else i == j
            assert bool(mask[0, 0, i, j]) == expected


def test_joint_block_is_single_concatenated_attention_with_both_directions():
    m = joint_model(False)
    block = m.dit['force'].blocks[0]
    x = torch.randn(2, 6, 24, requires_grad=True)
    q, k, v = [torch.randn(2, 2, 4, 16, requires_grad=True) for _ in range(3)]
    mask = torch.ones(2, 1, 10, 10, dtype=torch.bool)
    actual_video, actual_force = block(x, q, k, v, mask)
    a = block.self_attention
    y = block.norm1(x)
    fk, fv = a.kv(y)
    reference = F.scaled_dot_product_attention(torch.cat((q, a.split(a.q(y))), 2),
        torch.cat((k, fk), 2), torch.cat((v, fv), 2), attn_mask=mask)
    fx = x + a.out(reference[:, :, 4:].transpose(1, 2).flatten(2))
    torch.testing.assert_close(actual_video, reference[:, :, :4])
    torch.testing.assert_close(actual_force, fx + block.ff(block.norm2(fx)))
    assert torch.autograd.grad(actual_video.square().mean(), x, retain_graph=True)[0].abs().sum() > 0
    assert torch.autograd.grad(actual_force.square().mean(), v)[0].abs().sum() > 0
    assert not any('video_attention' in n for n, _ in m.named_parameters())


def test_force_only_loss_updates_native_cosmos_and_queries():
    m = joint_model()
    s = sample(1)
    _, f = world_models(m, s)
    masked_force_mse(f.wrench, s['wrench_future'], s['force_is_pad'], s['wrench_valid']).backward()
    gradients = [p.grad for n, p in m.video.named_parameters() if 'self_attn' in n]
    assert any(g is not None and g.abs().sum() > 0 for g in gradients)
    assert (m.dit['force'].queries.grad.abs().sum(-1) > 0).all()
    assert all(p.grad is None for p in m.dit['action'].parameters())


def test_video_only_loss_updates_force_attention():
    m = joint_model()
    s = sample(1)
    raw, ctx, cmask = m._contexts(s)
    prediction = m._supervised_video(s, s['video_latents'], torch.tensor([500.]),
        raw, ctx, cmask, s['action'], s['action_is_pad'])
    prediction.square().mean().backward()
    assert m.dit['force'].blocks[0].self_attention.v.weight.grad.abs().sum() > 0


def test_shared_initialization_matches_baseline_except_joint_attention():
    old, new = tiny_model(), joint_model(force_hidden=32)
    a, b = old.state_dict(), new.state_dict()
    for name in a.keys() & b.keys():
        if (name.startswith('dit.runtime.') or '.self_attention.' in name
                or '.video_attention.' in name or '.norm_video.' in name):
            continue
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0, msg=name)


def test_inference_future_labels_cannot_change_any_prediction_and_rng_restores():
    m = joint_model()
    s = sample(1)
    changed = copy.deepcopy(s)
    for key in ('video_latents', 'wrench_future', 'action'):
        changed[key].fill_(float('nan'))
    for key in ('wrench_valid', 'force_is_pad', 'image_is_pad'):
        changed[key].logical_not_()
    m.dit['action'].eval()
    state = m.dit['runtime'].get_state()
    first = m.infer_forcewm(s, action_condition=s['action'], num_inference_steps=2)
    second = m.infer_forcewm(changed, action_condition=s['action'], num_inference_steps=2)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert torch.equal(state, m.dit['runtime'].get_state())
    assert not m.dit['action'].training and m.training


def test_history_changes_video_and_observation_changes_force():
    m = joint_model(False)
    s = sample(1)
    a = m.infer_world_model(s, action_condition=s['action'])
    changed = copy.deepcopy(s)
    changed['history'] += 5
    b = m.infer_world_model(changed, action_condition=s['action'])
    assert not torch.allclose(a['video_latents'], b['video_latents'])
    changed = copy.deepcopy(s)
    changed['first_frame_latents'] += 5
    c = m.infer_world_model(changed, action_condition=s['action'])
    assert not torch.allclose(a['force'], c['force'])


def test_unavailable_channels_and_padding_do_not_leak_through_joint_attention():
    m = joint_model(False)
    s = sample(1)
    s['history_is_pad'][:, :5] = True
    s['history_channel_valid'][:, 8:10, :3] = False
    s['action_is_pad'][:, -5:] = True
    t = copy.deepcopy(s)
    t['history'][:, :5] = float('nan')
    t['history'][:, 8:10, :3] = float('nan')
    # Native control adapter requires finite Action, even in padded slots.
    t['action'][:, -5:] = 999.
    a = m.infer_world_model(s, action_condition=s['action'])
    b = m.infer_world_model(t, action_condition=t['action'])
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_ae_noise_and_clock_never_enter_either_wm():
    m = joint_model(False)
    s = sample(1)
    rng, runtime = torch.get_rng_state(), m.dit['runtime'].get_state()
    captured = []
    hook = m.dit['controls'].register_forward_pre_hook(lambda _, args: captured.append(args[0].clone()))
    metrics = []
    for clock in (100., 900.):
        torch.set_rng_state(rng)
        m.dit['runtime'].set_state(runtime)
        with patch.object(m.a_scheduler, 'sample_training_t', return_value=torch.tensor([clock])), \
             patch.object(m.dit['force'], 'prepare', wraps=m.dit['force'].prepare) as fp, \
             patch.object(m.dit['action'], 'prepare', wraps=m.dit['action'].prepare) as ap:
            _, metric = m.training_loss(s)
            metrics.append(metric)
            assert fp.call_count > 0
            for call in fp.call_args_list:
                torch.testing.assert_close(call.kwargs['action_condition'], s['action'])
                assert 'clock' not in call.kwargs and 'wrench_future' not in call.kwargs
            assert torch.equal(ap.call_args.args[1], torch.tensor([clock]))
            assert not torch.equal(ap.call_args.args[0], s['action'])
    hook.remove()
    assert captured and all(torch.equal(x, s['action']) for x in captured)
    assert metrics[0]['loss_video'] == metrics[1]['loss_video']
    assert metrics[0]['loss_force'] == metrics[1]['loss_force']
    assert metrics[0]['loss_action'] != metrics[1]['loss_action']


def test_training_future_labels_only_change_their_losses():
    m = joint_model(False)
    s = sample(1)
    changed = copy.deepcopy(s)
    changed['video_latents'][:, :, 1:] *= 10
    rng, runtime = torch.get_rng_state(), m.dit['runtime'].get_state()
    _, a = m.training_loss(s)
    torch.set_rng_state(rng)
    m.dit['runtime'].set_state(runtime)
    _, b = m.training_loss(changed)
    assert a['loss_video'] != b['loss_video']
    assert a['loss_force'] == b['loss_force'] and a['loss_action'] == b['loss_action']
    changed = copy.deepcopy(s)
    changed['wrench_future'] *= 10
    torch.set_rng_state(rng)
    m.dit['runtime'].set_state(runtime)
    _, c = m.training_loss(changed)
    assert a['loss_force'] != c['loss_force']
    assert a['loss_video'] == c['loss_video'] and a['loss_action'] == c['loss_action']


def test_joint_checkpoint_two_updates_and_exact_resume(tmp_path):
    m = joint_model(False)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, betas=(.9, .95))
    s = sample(1)
    for _ in range(2):
        opt.zero_grad()
        loss, _ = m.training_loss(s)
        loss.backward()
        opt.step()
    assert {int(v['step']) for v in opt.state.values()} == {2}
    p = tmp_path/'joint.pt'
    m.save_checkpoint(p, opt, 2)
    restored = joint_model(False)
    other = torch.optim.AdamW(restored.parameters(), lr=1e-4, betas=(.9, .95))
    assert restored.load_checkpoint(p, other) == 2
    rng = torch.get_rng_state()
    for model, optimizer in ((m, opt), (restored, other)):
        torch.set_rng_state(rng)
        optimizer.zero_grad()
        loss, _ = model.training_loss(s)
        loss.backward()
        optimizer.step()
    for name, value in m.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0, msg=name)
    with pytest.raises(ValueError, match='differ'):
        tiny_model().load_checkpoint(p)
    tiny_model().save_checkpoint(tmp_path/'old.pt')
    with pytest.raises(ValueError, match='differ'):
        m.load_checkpoint(tmp_path/'old.pt')
    restored.manifest = {'wrong': True}
    with pytest.raises(ValueError, match='differ'):
        restored.load_checkpoint(p)


def test_joint_checkpointing_gradients_match():
    a, b = joint_model(True), joint_model(False)
    b.load_state_dict(a.state_dict())
    rng = torch.get_rng_state()
    for m in (a, b):
        torch.set_rng_state(rng)
        loss, _ = m.training_loss(sample(1))
        loss.backward()
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        if p.grad is None:
            assert q.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5, msg=name)


def test_joint_bf16_with_native_fp32_no_autocast():
    m = joint_model()
    m.video.net.use_wan_fp32_strategy = True
    for block in m.video.blocks:
        block.use_wan_fp32_strategy = True
    m.video.net.final_layer.use_wan_fp32_strategy = True
    m.bfloat16()
    with torch.backends.mkldnn.flags(enabled=False):
        loss, metrics = m.training_loss(sample(1))
        loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    for group in ('video', 'force', 'action', 'controls'):
        grads = [p.grad for p in m.dit[group].parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
    s = sample(1)
    assert all(torch.isfinite(x).all() for x in m.infer_forcewm(
        s, action_condition=s['action'], num_inference_steps=1).values())


def test_joint_configuration_diff_is_only_architecture_and_output():
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
        configs = [OmegaConf.to_container(compose(config_name='train', overrides=[
            'task='+task, 'model.expected_text_identity='+'a'*64,
            'data.train.expected_identity_sha256='+'b'*64]), resolve=True, throw_on_missing=True)
            for task in ('cosmos_forcewm_4gpu_30k', 'cosmos_forcewm_joint_4gpu_30k')]
    for cfg in configs:
        validate_config(cfg)
    old, joint = configs
    assert joint['model']['_target_'] == 'fastwam.cosmos.assets.create_joint_cosmos_forcewm'
    assert old.pop('output_dir') != joint.pop('output_dir')
    old['model'].pop('_target_')
    joint['model'].pop('_target_')
    assert old == joint


def test_factory_binds_new_architecture_and_runtime_manifest(monkeypatch):
    from fastwam.cosmos import assets
    from fastwam.cosmos.joint import JOINT_PROTOCOL
    monkeypatch.setattr(assets, 'read_text_bank', lambda _: (None, {}, 'a'*64))
    monkeypatch.setattr(assets, 'load_video', lambda *args, **kwargs: tiny_net())
    m = assets.create_joint_cosmos_forcewm(upstream='test', checkpoint='test',
        text_cache='test', expected_text_identity='a'*64, device='cpu',
        action_hidden=32, action_ffn=64, force_hidden=24, force_ffn=64,
        force_heads=2, memory_width=16, memory_heads=2)
    assert isinstance(m, JointCosmosForceWM)
    assert m.manifest['schema'] == 'cosmos-orvfa-joint28-v2'
    assert m.manifest['joint_protocol'] == JOINT_PROTOCOL
    assert m.manifest['attention'] == dict(heads=2, video_width=32, force_width=24, force_qkv_width=32)
    invalid = copy.deepcopy(m.state_dict())
    invalid['dit.runtime.manifest_bytes'][0] ^= 1
    with pytest.raises(ValueError, match='manifest differs'):
        m.load_state_dict(invalid)


def test_cached_prefix_equals_full_three_expert_attention_and_gradients():
    torch.manual_seed(71)
    groups = torch.tensor([0, 2, 2, 0, 1, 3, 4, 4, 4])
    valid = torch.tensor([[True, True, True, False, True, True, True, True, False]])
    full_mask = grouped_attention_mask(groups, valid)
    q, k, v = [torch.randn(1, 2, 9, 16, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    full = F.scaled_dot_product_attention(q, k, v, attn_mask=full_mask)
    prefix = F.scaled_dot_product_attention(q[:, :, :6], k[:, :, :6], v[:, :, :6],
                                          attn_mask=full_mask[:, :, :6, :6])
    action = F.scaled_dot_product_attention(q[:, :, 6:], k, v,
        attn_mask=action_prefix_mask(valid[:, :6], ~valid[:, 6:]))
    split = torch.cat((prefix, action), 2)
    torch.testing.assert_close(full, split, atol=1e-6, rtol=1e-5)
    g1 = torch.autograd.grad(full.square().sum(), (q, k, v), retain_graph=True)
    g2 = torch.autograd.grad(split.square().sum(), (q, k, v))
    for a, b in zip(g1, g2):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)


def test_observed_and_reference_do_not_leak_future_across_layers():
    m = joint_model(False)
    s = sample(1)
    raw, ctx, cmask = m._contexts(s)
    def read(latents, reference):
        return m._joint_video(latents, torch.tensor([600.]), raw, ctx, cmask,
                              s, reference, s['action_is_pad'], collect=True)[1]
    latents = s['video_latents'].clone()
    latents[:, :, :1] = s['first_frame_latents']
    a = read(latents, s['action'])
    # First four video patches + 32 history tokens belong to O.
    observed = list(range(4)) + list(range(12, 44))
    reference = list(range(44, 76))
    changed = latents.clone()
    changed[:, :, 1:] += 100
    with torch.no_grad():
        m.dit['force'].queries.add_(100)
    b = read(changed, s['action'])
    c = read(changed, s['action'] + 100)
    for layer in range(m.depth):
        for i in (0, 1):
            torch.testing.assert_close(a[layer][i][:, :, observed + reference],
                                       b[layer][i][:, :, observed + reference], rtol=0, atol=0)
            torch.testing.assert_close(a[layer][i][:, :, observed],
                                       c[layer][i][:, :, observed], rtol=0, atol=0)
    assert not torch.allclose(b[-1][0][:, :, reference], c[-1][0][:, :, reference])


def test_action_loss_trains_both_wms_without_cross_memory_modules():
    m = joint_model(False)
    s = sample(1)
    memory, force = world_models(m, s)
    raw, ctx, cmask = m._contexts(s)
    action, _ = m.action_velocity(s['action'] * .1, torch.tensor([300.]), ctx, cmask,
                                 (), memory, force, s['action_is_pad'])
    action.square().mean().backward()
    assert 'memories' not in m.dit
    for group in ('video', 'force', 'action'):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.dit[group].parameters())


def test_model_only_warm_start_preserves_all_retained_tensors_and_rebinds_runtime(tmp_path):
    from fastwam.cache.latent import file_sha256
    m = joint_model(False)
    state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in m.state_dict().items()}
    state['dit.memories.0.removed_parameter'] = torch.ones(2)
    path = tmp_path / 'step5000.pt'
    torch.save(dict(schema='fastwam-cosmos-force-joint-self-attention28-v1', step=5000,
                    model=state), path)
    restored = joint_model(False)
    receipt = warm_start_joint(restored, path, file_sha256(path))
    assert receipt['optimizer'] == 'fresh' and receipt['step'] == 5000
    for key, value in restored.state_dict().items():
        if not key.startswith('dit.runtime.'):
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match='SHA256'):
        warm_start_joint(restored, path, '0' * 64)
    state['dit.action.head.weight'] = torch.full_like(state['dit.action.head.weight'], float('nan'))
    torch.save(dict(schema='fastwam-cosmos-force-joint-self-attention28-v1', step=5000,
                    model=state), path)
    with pytest.raises(ValueError, match='nonfinite'):
        warm_start_joint(restored, path, file_sha256(path))
