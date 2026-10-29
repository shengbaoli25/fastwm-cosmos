"""The DINO backbone must load without optional evaluation dependencies."""
import builtins
import hashlib
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from fastwam.cosmos import dense_encoder


ROOT = Path(__file__).resolve().parents[2]


def test_dinov3_loads_backbone_without_hub_or_evaluation_imports(tmp_path, monkeypatch):
    weights = tmp_path / 'model.safetensors'
    payload = b'checkpoint loading is isolated from this import regression'
    weights.write_bytes(payload)
    monkeypatch.setenv('DINOV3_LICENSE_ACCEPTED', '1')
    monkeypatch.setattr(dense_encoder, '_load_dinov3_safetensors', lambda *args: None)
    monkeypatch.syspath_prepend(str(ROOT / 'third_party/dinov3'))
    from dinov3.hub import backbones

    class FakeHplus(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.ones(1))
            self.blocks = nn.ModuleList(nn.Linear(1, 1) for _ in range(32))

        def forward_features(self, images):
            return {'x_norm_patchtokens': images}

    monkeypatch.setattr(backbones, 'dinov3_vith16plus', lambda pretrained=False: FakeHplus())
    original_import = builtins.__import__

    def backbone_import_only(name, *args, **kwargs):
        if name.startswith('dinov3.eval') or name.startswith('torchmetrics'):
            raise AssertionError('Dense teacher imported optional evaluation modules')
        return original_import(name, *args, **kwargs)

    def no_torch_hub(*args, **kwargs):
        raise AssertionError('Dense teacher attempted a hub lookup')

    monkeypatch.setattr(builtins, '__import__', backbone_import_only)
    monkeypatch.setattr(torch.hub, 'load', no_torch_hub)
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', no_torch_hub)
    teacher = dense_encoder.load_dense_teacher(
        'dinov3_vith16plus', ROOT / 'third_party/dinov3', weights,
        hashlib.sha256(payload).hexdigest(), device='cpu')
    assert teacher.feature_tokens == 392
    assert teacher.feature_dim == 1280
    assert len(teacher.encoder.blocks) == 32
    assert not teacher.encoder.training
    assert all(not parameter.requires_grad for parameter in teacher.encoder.parameters())


def test_dinov3_hplus_architecture_has_official_geometry_without_allocating_weights(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'third_party/dinov3'))
    from dinov3.hub.backbones import dinov3_vith16plus

    with torch.device('meta'):
        encoder = dinov3_vith16plus(pretrained=False).eval()
        features = encoder.forward_features(torch.empty(1, 3, 224, 224))

    assert all(value.device.type == 'meta' for value in encoder.state_dict().values())
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 840_633_600
    # Native QKV includes zero-masked key biases absent from the converted checkpoint.
    assert sum(parameter.numel() for parameter in encoder.parameters()) - 32 * 1280 == 840_592_640
    assert len(encoder.blocks) == 32
    assert encoder.embed_dim == 1280
    assert encoder.num_heads == 20
    assert encoder.patch_embed.proj.weight.shape == (1280, 3, 16, 16)
    assert encoder.patch_embed.patches_resolution == (14, 14)
    assert encoder.storage_tokens.shape == (1, 4, 1280)
    assert encoder.rope_embed.periods.shape == (16,)
    for block in encoder.blocks:
        assert block.attn.num_heads == 20
        assert block.mlp.w1.weight.shape == (5120, 1280)
        assert block.mlp.w2.weight.shape == (5120, 1280)
        assert block.mlp.w3.weight.shape == (1280, 5120)
    assert features['x_norm_clstoken'].shape == (1, 1280)
    assert features['x_storage_tokens'].shape == (1, 4, 1280)
    assert features['x_norm_patchtokens'].shape == (1, 196, 1280)
    assert features['x_prenorm'].shape == (1, 201, 1280)


def _tiny_encoder(monkeypatch, ffn_layer, seed):
    monkeypatch.syspath_prepend(str(ROOT / 'third_party/dinov3'))
    from dinov3.models.vision_transformer import DinoVisionTransformer

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = DinoVisionTransformer(img_size=32, patch_size=16, embed_dim=24,
            depth=2, num_heads=3, ffn_ratio=6 if ffn_layer == 'swiglu' else 4,
            layerscale_init=1e-5, norm_layer='layernormbf16',
            ffn_layer=ffn_layer, n_storage_tokens=4, mask_k_bias=True,
            pos_embed_rope_dtype='fp32', pos_embed_rope_rescale_coords=2)
        encoder.init_weights()
    return encoder.eval()


def _modelscope_checkpoint(encoder):
    target = {key: value.detach().clone() for key, value in encoder.state_dict().items()}
    source = {
        'embeddings.cls_token': target['cls_token'],
        'embeddings.mask_token': target['mask_token'].unsqueeze(1),
        'embeddings.register_tokens': target['storage_tokens'],
        'embeddings.patch_embeddings.weight': target['patch_embed.proj.weight'],
        'embeddings.patch_embeddings.bias': target['patch_embed.proj.bias'],
        'norm.weight': target['norm.weight'],
        'norm.bias': target['norm.bias'],
    }
    for layer in range(len(encoder.blocks)):
        prefix, block = f'layer.{layer}.', f'blocks.{layer}.'
        for norm in ('norm1', 'norm2'):
            source[prefix + norm + '.weight'] = target[block + norm + '.weight']
            source[prefix + norm + '.bias'] = target[block + norm + '.bias']
        source[prefix + 'layer_scale1.lambda1'] = target[block + 'ls1.gamma']
        source[prefix + 'layer_scale2.lambda1'] = target[block + 'ls2.gamma']
        q, k, v = target[block + 'attn.qkv.weight'].chunk(3, 0)
        qb, _, vb = target[block + 'attn.qkv.bias'].chunk(3, 0)
        for name, value in (('q', q), ('k', k), ('v', v)):
            source[prefix + f'attention.{name}_proj.weight'] = value
        source[prefix + 'attention.q_proj.bias'] = qb
        source[prefix + 'attention.v_proj.bias'] = vb
        source[prefix + 'attention.o_proj.weight'] = target[block + 'attn.proj.weight']
        source[prefix + 'attention.o_proj.bias'] = target[block + 'attn.proj.bias']
        projections = (('gate_proj', 'w1'), ('up_proj', 'w2'), ('down_proj', 'w3')) \
            if hasattr(encoder.blocks[layer].mlp, 'w1') else (('up_proj', 'fc1'), ('down_proj', 'fc2'))
        for converted_name, native_name in projections:
            for kind in ('weight', 'bias'):
                source[prefix + f'mlp.{converted_name}.{kind}'] = target[block + f'mlp.{native_name}.{kind}']
    return source


@pytest.mark.parametrize('ffn_layer', ['swiglu', 'mlp'])
def test_dinov3_safetensors_roundtrip_loads_independent_backbone(tmp_path, monkeypatch, ffn_layer):
    original = _tiny_encoder(monkeypatch, ffn_layer, seed=10)
    loaded = _tiny_encoder(monkeypatch, ffn_layer, seed=20)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(30)
        for parameter in original.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.02)
        for block in original.blocks:
            block.attn.qkv.bias[24:48].zero_()
    for name, parameter in original.named_parameters():
        assert not torch.equal(parameter, loaded.get_parameter(name)), name
    source = _modelscope_checkpoint(original)
    weights = tmp_path / 'model.safetensors'
    save_file({key: value.contiguous().clone() for key, value in source.items()}, str(weights))

    dense_encoder._load_dinov3_safetensors(loaded, weights)

    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)
    images = torch.linspace(-1, 1, 2 * 3 * 32 * 32).reshape(2, 3, 32, 32)
    with torch.inference_mode():
        expected = original.forward_features(images)['x_norm_patchtokens']
        actual = loaded.forward_features(images)['x_norm_patchtokens']
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('ffn_layer', ['swiglu', 'mlp'])
@pytest.mark.parametrize('corruption', ['missing', 'unexpected', 'shape'])
def test_dinov3_safetensors_rejects_corrupted_checkpoints(
        tmp_path, monkeypatch, ffn_layer, corruption):
    encoder = _tiny_encoder(monkeypatch, ffn_layer, seed=10)
    source = _modelscope_checkpoint(_tiny_encoder(monkeypatch, ffn_layer, seed=20))
    key = 'layer.1.mlp.gate_proj.weight' if ffn_layer == 'swiglu' else 'layer.1.mlp.up_proj.weight'
    if corruption == 'missing':
        source.pop(key)
        message = r'key set differs; missing=\['
    elif corruption == 'unexpected':
        source['layer.0.attention.k_proj.bias'] = torch.zeros(24)
        message = r"key set differs; missing=\[\], unexpected=\['layer.0.attention.k_proj.bias'\]"
    else:
        source[key] = source[key][:-1]
        message = 'tensor shape mismatch for blocks.1.mlp.'
    before = {name: value.detach().clone() for name, value in encoder.state_dict().items()}
    weights = tmp_path / 'model.safetensors'
    save_file({name: value.contiguous().clone() for name, value in source.items()}, str(weights))

    with pytest.raises(ValueError, match=message) as error:
        dense_encoder._load_dinov3_safetensors(encoder, weights)
    if corruption == 'missing':
        assert key in str(error.value)
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_dinov3_tiny_bfloat16_forward_returns_normalized_patch_tokens(tmp_path, monkeypatch):
    original = _tiny_encoder(monkeypatch, 'swiglu', seed=10)
    loaded = _tiny_encoder(monkeypatch, 'swiglu', seed=20)
    weights = tmp_path / 'model.safetensors'
    save_file({key: value.contiguous().clone() for key, value in _modelscope_checkpoint(original).items()},
              str(weights))
    dense_encoder._load_dinov3_safetensors(loaded, weights)
    loaded.to(dtype=torch.bfloat16)
    images = torch.linspace(-1, 1, 2 * 3 * 32 * 32, dtype=torch.bfloat16).reshape(2, 3, 32, 32)

    with torch.inference_mode():
        features = loaded.forward_features(images)
        expected = loaded.norm(features['x_prenorm'])[:, 5:]

    patches = features['x_norm_patchtokens']
    assert patches.shape == (2, 4, 24)
    assert patches.dtype == torch.bfloat16
    assert torch.isfinite(patches).all()
    torch.testing.assert_close(patches, expected, rtol=0, atol=0)
    torch.testing.assert_close(patches.float().mean(-1), torch.zeros(2, 4), rtol=0, atol=0.005)
    torch.testing.assert_close(patches.float().var(-1, unbiased=False), torch.ones(2, 4), rtol=0, atol=0.01)
