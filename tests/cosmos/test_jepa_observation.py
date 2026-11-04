"""Real upstream RoPE/patch arithmetic, with small weights and no downloads."""
import copy

import pytest
import torch
from torch import nn

from fastwam.cosmos import jepa_observation as jo


def tiny_official():
    from fastwam.cosmos._vendor.vjepa21.vision_transformer import VisionTransformer
    # Keep all twelve official blocks and the actual 224x224 patch geometry.
    return VisionTransformer(img_size=(384, 384), patch_size=16, num_frames=64,
                             tubelet_size=2, embed_dim=24, depth=12, num_heads=3,
                             use_rope=True, interpolate_rope=True,
                             img_temporal_dim_size=1).eval()


@pytest.fixture
def tiny_spec(monkeypatch):
    monkeypatch.setitem(jo.VARIANT_SPECS, 'vitb', dict(
        width=24, depth=12, heads=3, factory='vit_base', filename='tiny-test.pt'))


def wrapper(checkpoint_suffix=True):
    return jo.JEPAObservationEncoder(tiny_official(), variant='vitb',
                                    weight_sha256='a' * 64,
                                    checkpoint_suffix=checkpoint_suffix)


def test_qv_lora_exact_initialization_and_no_key_or_base_update():
    torch.manual_seed(5)
    base = nn.Linear(12, 36)
    x = torch.randn(2, 7, 12)
    expected = base(x).detach()
    original = copy.deepcopy(base.state_dict())
    adapter = jo.QVLoRALinear(base)
    assert torch.equal(adapter(x), expected)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=.1)
    for _ in range(2):
        optimizer.zero_grad()
        adapter(x).square().mean().backward()
        assert adapter.base.weight.grad is None and adapter.base.bias.grad is None
        assert adapter.lora_q_b.grad.abs().sum() > 0
        assert adapter.lora_v_b.grad.abs().sum() > 0
        optimizer.step()
    actual = adapter(x)
    assert torch.equal(actual[..., 12:24], expected[..., 12:24])
    assert not torch.equal(actual[..., :12], expected[..., :12])
    assert not torch.equal(actual[..., 24:], expected[..., 24:])
    assert adapter.lora_q_a.grad.abs().sum() > 0
    assert adapter.lora_v_a.grad.abs().sum() > 0
    for name, value in original.items():
        assert torch.equal(value, adapter.base.state_dict()[name])


def test_observed_encoder_matches_official_normalization_and_camera_order(tiny_spec):
    torch.manual_seed(3)
    encoder = tiny_official()
    reference = copy.deepcopy(encoder)
    m = jo.JEPAObservationEncoder(encoder, variant='vitb', weight_sha256='b' * 64)
    x = torch.rand(1, 2, 3, 2, 224, 224) * 2 - 1
    views = (x.flatten(0, 1) + 1) / 2
    mean = views.new_tensor(jo.IMAGENET_MEAN)[None, :, None, None, None]
    std = views.new_tensor(jo.IMAGENET_STD)[None, :, None, None, None]
    with torch.no_grad():
        expected = reference((views - mean) / std).reshape(1, 392, 24)
        actual = m(x)
        swapped = m(x.flip(1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(swapped[:, :196], actual[:, 196:], rtol=0, atol=0)
    torch.testing.assert_close(swapped[:, 196:], actual[:, :196], rtol=0, atol=0)


def test_only_last_four_qv_trainable_and_parent_train_keeps_base_eval(tiny_spec):
    m = wrapper()
    parent = nn.ModuleDict({'jepa': m})
    parent.train()
    assert m.training
    assert not any(sub.training for sub in m.encoder.modules())
    expected_names = {f'encoder.blocks.{i}.attn.qkv.lora_{part}_{factor}'
                      for i in range(8, 12) for part in ('q', 'v') for factor in ('a', 'b')}
    assert set(m.trainable_names) == expected_names
    assert sum(p.numel() for p in m.trainable_parameters()) == 4 * 4 * 8 * 24
    parent.requires_grad_(True)
    m.configure_trainable_parameters()
    assert set(m.trainable_names) == expected_names
    parent.requires_grad_(False)
    m.configure_trainable_parameters()
    assert set(m.trainable_names) == expected_names


def test_suffix_checkpointing_preserves_lora_gradients_and_frozen_prefix(tiny_spec):
    torch.manual_seed(13)
    direct = wrapper(False).train()
    checkpointed = copy.deepcopy(direct)
    checkpointed.checkpoint_suffix = True
    x = torch.rand(1, 2, 3, 2, 224, 224) * 2 - 1
    grad_modes = []
    hook1 = checkpointed.encoder.blocks[0].register_forward_pre_hook(
        lambda module, inputs: grad_modes.append(('prefix', torch.is_grad_enabled())))
    hook2 = checkpointed.encoder.blocks[8].register_forward_pre_hook(
        lambda module, inputs: grad_modes.append(('suffix', torch.is_grad_enabled())))
    targets = torch.randn(1, 392, 24)
    y = direct(x)
    z = checkpointed(x)
    torch.testing.assert_close(y, z, rtol=0, atol=0)
    (y * targets).mean().backward()
    (z * targets).mean().backward()
    hook1.remove(); hook2.remove()
    assert ('prefix', False) in grad_modes and ('suffix', True) in grad_modes
    for (name, p), (_, q) in zip(direct.named_parameters(), checkpointed.named_parameters()):
        if p.requires_grad:
            assert p.grad is not None and q.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad)
            if name.endswith(('_q_b', '_v_b')):
                assert p.grad.abs().sum() > 0, name
        else:
            assert p.grad is None and q.grad is None, name


@pytest.mark.parametrize('shape', [(1, 2, 3, 10, 224, 224), (1, 3, 2, 224, 448)])
def test_encoder_rejects_future_clip_or_ambiguous_camera_layout(tiny_spec, shape):
    with pytest.raises(ValueError, match='observed RGB'):
        wrapper()(torch.zeros(shape))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 255.])
def test_encoder_rejects_invalid_rgb(tiny_spec, value):
    with pytest.raises(ValueError, match='finite'):
        wrapper()(torch.full((1, 2, 3, 2, 224, 224), value))


def test_factory_strict_ema_loading_precedes_lora_and_binds_hash(tiny_spec, monkeypatch, tmp_path):
    original = tiny_official()
    state = {'module.backbone.' + k: v for k, v in original.state_dict().items()}
    path = tmp_path / 'official-test.pt'
    torch.save({'ema_encoder': state}, path)
    monkeypatch.setattr(jo, 'build_jepa_architecture', lambda variant: tiny_official())
    m = jo.create_jepa_observation(variant='vitb', checkpoint=path,
                                 device='cpu', dtype=torch.float32)
    assert m.identity['weight_sha256'] == jo.sha256_file(path)
    torch.testing.assert_close(m.encoder.blocks[-1].attn.qkv.base.weight,
                               original.blocks[-1].attn.qkv.weight, rtol=0, atol=0)
    with pytest.raises(ValueError, match='SHA256'):
        jo.create_jepa_observation(variant='vitb', checkpoint=path,
                                  expected_sha256='0' * 64, device='cpu')
    state.pop(next(iter(state)))
    torch.save({'ema_encoder': state}, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        jo.create_jepa_observation(variant='vitb', checkpoint=path, device='cpu')


def test_ema_key_contract_and_prefix_collision():
    with pytest.raises(ValueError, match='ema_encoder'):
        jo._ema_state({'target_encoder': {}})
    with pytest.raises(ValueError, match='collision'):
        jo._ema_state({'ema_encoder': {'module.x': torch.zeros(1), 'x': torch.ones(1)}})


def test_production_variant_geometry():
    assert jo.variant_spec('vitb')['width'] == 768
    assert jo.variant_spec('vitb')['depth'] == 12
    assert jo.variant_spec('vitl')['width'] == 1024
    assert jo.variant_spec('vitl')['depth'] == 24


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA BF16 smoke needs GPU')
def test_cuda_bf16_real_rope_lora_gradients(tiny_spec):
    m = wrapper().to(device='cuda', dtype=torch.bfloat16).train()
    x = torch.rand(1, 2, 3, 2, 224, 224, device='cuda') * 2 - 1
    output = m(x)
    assert output.shape == (1, 392, 24) and torch.isfinite(output).all()
    (output.float() * torch.randn_like(output.float())).mean().backward()
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            if name.endswith(('_q_b', '_v_b')):
                assert p.grad.abs().sum() > 0, name
        else:
            assert p.grad is None, name
