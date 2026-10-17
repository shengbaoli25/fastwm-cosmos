"""Real tiny-network gradient contracts; no claim of trained localization."""
import copy

import pytest
import torch

from conftest import sample, tiny_net
from fastwam.cosmos.model import temporal_mse
from fastwam.cosmos.region_queries import RegionQueryCosmosForceWM, RegionQueryReader
from fastwam.forcewm.model import masked_force_mse


def model(checkpoint=False):
    torch.manual_seed(7)
    return RegionQueryCosmosForceWM(tiny_net(), manifest={'scope': 'TEST-region32'},
        action_hidden=32, action_ffn=64, force_hidden=32, force_ffn=64,
        force_heads=2, memory_width=16, memory_heads=2, generated_video_steps=1,
        gradient_checkpointing=checkpoint, region_layer=0, region_count=32,
        region_width=16, region_heads=2)


def world(m, s):
    raw, ctx, cmask = m._contexts(s)
    memory, force = m._world_models(s, s['first_frame_latents'], raw, ctx, cmask,
                                     s['action'], s['action_is_pad'])
    return memory, force, raw, ctx, cmask


def assert_query_grad(m):
    grad = m.dit['regions'].queries.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert (grad.abs().sum(-1) > 0).all()
    assert m.dit['regions'].k.weight.grad.abs().sum() > 0


def test_count_and_joint_camera_normalization():
    reader = RegionQueryReader(24, count=32, width=16, heads=2)
    visual = torch.randn(1, 2, 2, 14, 14, 24)
    tokens, maps = reader(visual, torch.randn(1, 32, 7), torch.zeros(1, 32, dtype=torch.bool))
    assert reader.queries.shape == (32, 16)
    assert tokens.shape == (1, 2, 32, 16)
    assert maps.shape == (1, 2, 32, 2, 14, 14)
    torch.testing.assert_close(maps.sum((-1, -2, -3)), torch.ones(1, 2, 32))
    assert not any('decode' in key or 'pair' in key for key in reader.state_dict())
    changed = reader(visual.flip(-2), torch.randn(1, 32, 7), torch.zeros(1, 32, dtype=torch.bool))[1]
    assert not torch.equal(maps, changed)


def test_force_loss_alone_trains_query_selection():
    m = model(); s = sample(1)
    _, force, *_ = world(m, s)
    loss = masked_force_mse(force.wrench, s['wrench_future'], s['force_is_pad'], s['wrench_valid'])
    loss.backward(); assert_query_grad(m)
    assert all(p.grad is None for p in m.dit['action'].parameters())
    for group in ('video', 'force', 'controls', 'region_bridge'):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in m.dit[group].parameters()), group


@pytest.mark.parametrize('gradient_checkpointing', [False, True])
def test_action_loss_trains_reader_but_cannot_update_either_world_model(gradient_checkpointing):
    m = model(gradient_checkpointing); s = sample(1)
    memory, force, raw, ctx, cmask = world(m, s)
    time = torch.tensor([500.]); noise = torch.randn_like(s['action'])
    noisy = m.a_scheduler.add_noise(s['action'], noise, time)
    _, _, kv = m._video(s['first_frame_latents'], torch.zeros_like(time), raw, observed=True)
    predicted, _ = m.action_velocity(noisy, time, ctx, cmask, kv, memory, force, s['action_is_pad'])
    loss = temporal_mse(predicted, m.a_scheduler.training_target(s['action'], noise, time),
                        ~s['action_is_pad'], m.a_scheduler, time)
    loss.backward(); assert_query_grad(m)
    assert m.dit['region_action_read'].cross.k.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.dit['action'].parameters())
    assert m.proprio_encoder.weight.grad.abs().sum() > 0
    for group in ('video', 'force', 'controls', 'region_bridge'):
        assert all(p.grad is None for p in m.dit[group].parameters()), group
    assert not memory[1][0].requires_grad
    assert all(not k.requires_grad and not v.requires_grad for k, v in kv)


def test_no_auxiliary_loss_and_clean_controls_only():
    m = model(); s = sample(1); seen = []
    hook = m.dit['regions'].register_forward_pre_hook(lambda _, args: seen.append(args[1].detach().clone()))
    loss, metrics = m.training_loss(s); loss.backward(); hook.remove()
    assert set(metrics) == {'loss_video', 'loss_action', 'loss_force'}
    assert seen
    for actual in seen:
        torch.testing.assert_close(actual, s['action'], rtol=0, atol=0)
    assert not any('reconstruction' in key or 'diversity' in key for key in m.manifest['region_queries'])


def test_padded_action_is_ignored():
    reader = RegionQueryReader(24, count=32, width=16, heads=2)
    visual = torch.randn(1, 2, 2, 3, 4, 24); action = torch.randn(1, 32, 7)
    pad = torch.zeros(1, 32, dtype=torch.bool); pad[:, 20:] = True
    expected = reader(visual, action, pad); action[:, 20:] = float('nan')
    for a, b in zip(expected, reader(visual, action, pad)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_inference_ignores_future_labels_and_ae_seed():
    m = model(); s = sample(1); changed = copy.deepcopy(s)
    for key in ('video_latents', 'wrench_future', 'action'): changed[key].fill_(float('nan'))
    for key in ('force_is_pad', 'wrench_valid', 'image_is_pad'): changed[key].logical_not_()
    before = m.dit['runtime'].get_state()
    a = m.infer_with_regions(s, action_condition=s['action'], seed=1, num_inference_steps=1)
    b = m.infer_with_regions(changed, action_condition=s['action'], seed=2, num_inference_steps=1)
    for key in ('force', 'video_latents'):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    torch.testing.assert_close(a['regions']['attention'], b['regions']['attention'], rtol=0, atol=0)
    assert torch.equal(before, m.dit['runtime'].get_state())
    assert not torch.equal(a['action'], b['action'])


def test_checkpointing_matches_gradients():
    a, b = model(True), model(False); b.load_state_dict(a.state_dict()); s = sample(1)
    rng = torch.get_rng_state(); la, _ = a.training_loss(s); la.backward()
    torch.set_rng_state(rng); lb, _ = b.training_loss(s); lb.backward()
    torch.testing.assert_close(la, lb)
    for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters()):
        assert name == other
        if p.grad is None: assert q.grad is None, name
        else: torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4, msg=name)


def test_save_restore_and_reject_old_models(tmp_path):
    a = model(); s = sample(1); opt = torch.optim.AdamW(a.parameters(), lr=1e-4)
    for _ in range(2):
        opt.zero_grad(); loss, _ = a.training_loss(s); loss.backward(); opt.step()
    path = tmp_path / 'regions.pt'; a.save_checkpoint(path, opt, 2)
    b = model(); other = torch.optim.AdamW(b.parameters(), lr=1e-4)
    assert b.load_checkpoint(path, other) == 2
    rng = torch.get_rng_state()
    opt.zero_grad(); la, _ = a.training_loss(s); la.backward(); opt.step()
    torch.set_rng_state(rng)
    other.zero_grad(); lb, _ = b.training_loss(s); lb.backward(); other.step()
    torch.testing.assert_close(la, lb, rtol=0, atol=0)
    for p, q in zip(a.parameters(), b.parameters()): torch.testing.assert_close(p, q, rtol=0, atol=0)
    payload = torch.load(path, weights_only=True)
    for schema in ('fastwam-cosmos-paired-regions28-v1', 'fastwam-cosmos-clean-action-force-chunk28-v3'):
        payload['schema'] = schema; torch.save(payload, path)
        with pytest.raises(ValueError, match='differ'): b.load_checkpoint(path)


def test_bf16_with_native_fp32_video():
    m = model(); m.video.net.use_wan_fp32_strategy = True
    for block in m.video.blocks: block.use_wan_fp32_strategy = True
    m.video.net.final_layer.use_wan_fp32_strategy = True; m.bfloat16()
    with torch.backends.mkldnn.flags(enabled=False):
        loss, _ = m.training_loss(sample(1)); loss.backward()
    assert torch.isfinite(loss); assert_query_grad(m)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_production_config_pins_task_only_32_total_queries():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from fastwam.cosmos.preflight import validate_config
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / 'configs'), version_base=None):
        cfg = compose(config_name='train', overrides=['task=cosmos_forcewm_regions32_4gpu_30k',
            'model.expected_text_identity=' + 'a' * 64, 'data.train.expected_identity_sha256=' + 'b' * 64])
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True); validate_config(cfg)
    assert cfg['model']['region_count'] == 32
    assert cfg['batch_size'] == 16 and cfg['max_steps'] == 30000 and cfg['save_every'] == 5000
    cfg['model']['region_pairs'] = 32
    with pytest.raises(ValueError, match='task-only'): validate_config(cfg)
