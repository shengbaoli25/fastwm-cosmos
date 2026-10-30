"""Exact frozen architectures, strict H+ loading, and causal video geometry."""
import hashlib
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

from fastwam.cosmos import dense_encoder


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize('name,width,depth,count', [
    ('vjepa21', 768, 12, 86833152),
    ('vjepa21_vitg', 1408, 40, 1013267968),
    ('dinov3', 768, 12, 85669632),
    ('dinov3_vithplus', 1280, 32, 840633600),
])
def test_pinned_factories_build_frozen_meta_models_without_weights_or_network(
        monkeypatch, name, width, depth, count):
    def forbidden(*args, **kwargs):
        raise AssertionError('Unweighted architecture attempted checkpoint or network loading')

    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', forbidden)
    with torch.device('meta'):
        encoder = dense_encoder.build_teacher_architecture(name)
    assert encoder.embed_dim == width and len(encoder.blocks) == depth
    assert sum(p.numel() for p in encoder.parameters()) == count
    assert all(p.device.type == 'meta' and not p.requires_grad for p in encoder.parameters())
    assert not encoder.training
    spec = dense_encoder.teacher_spec(name)
    assert spec['feature_dim'] == width and spec['video_input_size'] == 224
    spec['feature_dim'] = -1
    assert dense_encoder.teacher_spec(name)['feature_dim'] == width


def _tiny_hplus():
    sys.path.insert(0, str(ROOT / 'third_party/dinov3'))
    from dinov3.models.vision_transformer import DinoVisionTransformer
    return DinoVisionTransformer(img_size=32, patch_size=16, embed_dim=32,
        depth=2, num_heads=4, ffn_ratio=6., ffn_layer='swiglu', layerscale_init=1e-5,
        n_storage_tokens=4, mask_k_bias=True, pos_embed_rope_dtype='fp32')


def _hf_state(encoder):
    state = encoder.state_dict()
    source = {
        'embeddings.cls_token': state['cls_token'],
        'embeddings.mask_token': state['mask_token'].unsqueeze(1),
        'embeddings.register_tokens': state['storage_tokens'],
        'embeddings.patch_embeddings.weight': state['patch_embed.proj.weight'],
        'embeddings.patch_embeddings.bias': state['patch_embed.proj.bias'],
        'norm.weight': state['norm.weight'], 'norm.bias': state['norm.bias'],
    }
    for layer in range(len(encoder.blocks)):
        src, dst = f'blocks.{layer}.', f'layer.{layer}.'
        for name in ('norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias'):
            source[dst + name] = state[src + name]
        source[dst + 'layer_scale1.lambda1'] = state[src + 'ls1.gamma']
        source[dst + 'layer_scale2.lambda1'] = state[src + 'ls2.gamma']
        q, k, v = state[src + 'attn.qkv.weight'].chunk(3, 0)
        qb, _, vb = state[src + 'attn.qkv.bias'].chunk(3, 0)
        for name, value in (('q_proj.weight', q), ('k_proj.weight', k),
                ('v_proj.weight', v), ('q_proj.bias', qb), ('v_proj.bias', vb),
                ('o_proj.weight', state[src + 'attn.proj.weight']),
                ('o_proj.bias', state[src + 'attn.proj.bias'])):
            source[dst + 'attention.' + name] = value
        for hf, official in (('gate_proj', 'w1'), ('up_proj', 'w2'), ('down_proj', 'w3')):
            for suffix in ('weight', 'bias'):
                source[dst + f'mlp.{hf}.{suffix}'] = state[src + f'mlp.{official}.{suffix}']
    return source


def test_hplus_swiglu_checkpoint_mapping_preserves_gate_math_and_zero_k_bias():
    encoder = _tiny_hplus()
    source = _hf_state(encoder)
    mapped = dense_encoder._map_dinov3_state(encoder, source)
    x = torch.randn(2, 5, 32)
    linear = torch.nn.functional.linear
    gate = linear(x, source['layer.0.mlp.gate_proj.weight'], source['layer.0.mlp.gate_proj.bias'])
    up = linear(x, source['layer.0.mlp.up_proj.weight'], source['layer.0.mlp.up_proj.bias'])
    expected = linear(torch.nn.functional.silu(gate) * up,
        source['layer.0.mlp.down_proj.weight'], source['layer.0.mlp.down_proj.bias'])
    encoder.load_state_dict(mapped, strict=True)
    torch.testing.assert_close(encoder.blocks[0].mlp(x), expected)
    assert torch.count_nonzero(encoder.blocks[0].attn.qkv.bias[32:64]) == 0


@pytest.mark.parametrize('corruption', ['missing_gate', 'unexpected_k_bias', 'wrong_shape'])
def test_hplus_mapping_rejects_incompatible_weights(corruption):
    encoder = _tiny_hplus()
    source = _hf_state(encoder)
    if corruption == 'missing_gate':
        del source['layer.1.mlp.gate_proj.weight']
    elif corruption == 'unexpected_k_bias':
        source['layer.1.attention.k_proj.bias'] = torch.zeros(32)
    else:
        source['layer.1.mlp.gate_proj.weight'] = torch.zeros(1, 32)
    with pytest.raises(ValueError, match='key set differs|shape mismatch'):
        dense_encoder._map_dinov3_state(encoder, source)


class IndependentFrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.inputs = []

    def forward_features(self, frames):
        self.inputs.append(frames.detach().clone())
        # Each output depends only on its own frame; dense channels are cheap views.
        values = frames.float().mean((1, 2, 3))[:, None, None]
        return {'x_norm_patchtokens': values.expand(-1, 196, 1280)}


def _hplus_teacher():
    return dense_encoder.DenseTeacher('dinov3_vithplus',
        IndependentFrameEncoder().eval().requires_grad_(False), 224, 14, 14, 'test-hash')


def _clip():
    video = torch.empty(1, 3, 10, 224, 448)
    for frame in range(10):
        video[:, :, frame, :, :224] = -0.8 + frame * .1
        video[:, :, frame, :, 224:] = -0.2 + frame * .1
    return video


def test_hplus_independent_frames_mean_adjacent_pairs_and_preserve_camera_order():
    teacher = _hplus_teacher()
    result = teacher.encode_video(_clip())
    assert result.shape == (1, 5, 392, 1280)
    frames = teacher.encoder.inputs[0]
    assert frames.shape == (20, 3, 224, 224)
    expected = frames.float().mean((1, 2, 3)).reshape(2, 5, 2).mean(-1).bfloat16()
    torch.testing.assert_close(result[0, :, 0, 0], expected[0])
    torch.testing.assert_close(result[0, :, 196, 0], expected[1])
    assert not result.requires_grad
    manifest = teacher.manifest()
    assert manifest['temporal_aggregation'] == 'independent_frame_encoding_adjacent_pair_mean'
    assert manifest['video_input_size'] == 224 and manifest['feature_dim'] == 1280


def test_hplus_observation_never_receives_future_frames_and_both_observed_frames_matter():
    teacher = _hplus_teacher()
    video = _clip()
    changed = video.clone()
    changed[:, :, 2:] = .99
    observed = teacher.encode_video(video[:, :, :2], context_only=True)
    perturbed = teacher.encode_video(changed[:, :, :2], context_only=True)
    assert observed.shape == (1, 1, 392, 1280)
    torch.testing.assert_close(observed, perturbed, rtol=0, atol=0)
    assert all(frames.shape[0] == 4 for frames in teacher.encoder.inputs)
    full, changed_full = teacher.encode_video(video), teacher.encode_video(changed)
    torch.testing.assert_close(full[:, :1], observed, rtol=0, atol=0)
    assert not torch.equal(full[:, 1:], changed_full[:, 1:])
    for frame in (0, 1):
        altered = video[:, :, :2].clone()
        altered[:, :, frame] = .99
        assert not torch.equal(observed, teacher.encode_video(altered, context_only=True))
    with pytest.raises(ValueError, match='2\\|10'):
        teacher.encode_video(video, context_only=True)


def test_hplus_license_hash_and_filename_are_checked_before_model_construction(tmp_path, monkeypatch):
    weights = tmp_path / 'model.safetensors'
    weights.write_bytes(b'not actually loaded by this asset test')
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    source = ROOT / 'third_party/dinov3'
    monkeypatch.delenv('DINOV3_LICENSE_ACCEPTED', raising=False)
    with pytest.raises(PermissionError, match='LICENSE_ACCEPTED'):
        dense_encoder.validate_teacher_assets('dinov3_vithplus', source, weights, digest)
    monkeypatch.setenv('DINOV3_LICENSE_ACCEPTED', '1')
    dense_encoder.validate_teacher_assets('dinov3_vithplus', source, weights, digest)
    with pytest.raises(ValueError, match='SHA256'):
        dense_encoder.validate_teacher_assets('dinov3_vithplus', source, weights, 'wrong')
