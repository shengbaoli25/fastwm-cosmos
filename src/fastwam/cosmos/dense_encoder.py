"""Pinned DINOv3 and V-JEPA 2.1 dense feature teachers."""
import hashlib
import os
from pathlib import Path
import sys

import torch
from torch.nn import functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VJEPA_SOURCE_COMMIT = '204698b45b3712590f06245fbfba32d3be539812'
DINOV3_SOURCE_COMMIT = '6876159a11b4df116f30f667f8c9888617df0751'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(path):
    import subprocess
    revision = path / 'REVISION'
    if revision.is_file():
        return revision.read_text().strip()
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def _load_dinov3_safetensors(encoder, weights):
    from safetensors.torch import load_file

    if weights.name != 'model.safetensors':
        raise ValueError('DINOv3 expects the pinned Hugging Face model.safetensors artifact')
    source = load_file(str(weights), device='cpu')
    expected_source = {
        'embeddings.cls_token', 'embeddings.mask_token',
        'embeddings.register_tokens', 'embeddings.patch_embeddings.weight',
        'embeddings.patch_embeddings.bias', 'norm.weight', 'norm.bias',
    }
    for layer in range(12):
        prefix = f'layer.{layer}.'
        expected_source.update({
            prefix + 'norm1.weight', prefix + 'norm1.bias',
            prefix + 'norm2.weight', prefix + 'norm2.bias',
            prefix + 'layer_scale1.lambda1', prefix + 'layer_scale2.lambda1',
            prefix + 'attention.q_proj.weight', prefix + 'attention.q_proj.bias',
            prefix + 'attention.k_proj.weight',
            prefix + 'attention.v_proj.weight', prefix + 'attention.v_proj.bias',
            prefix + 'attention.o_proj.weight', prefix + 'attention.o_proj.bias',
            prefix + 'mlp.up_proj.weight', prefix + 'mlp.up_proj.bias',
            prefix + 'mlp.down_proj.weight', prefix + 'mlp.down_proj.bias',
        })
    if set(source) != expected_source:
        missing, unexpected = sorted(expected_source - set(source)), sorted(set(source) - expected_source)
        raise ValueError(f'DINOv3 safetensors key set differs; missing={missing}, unexpected={unexpected}')

    target = encoder.state_dict()
    converted = {
        'cls_token': source['embeddings.cls_token'],
        'mask_token': source['embeddings.mask_token'].squeeze(1),
        'storage_tokens': source['embeddings.register_tokens'],
        'patch_embed.proj.weight': source['embeddings.patch_embeddings.weight'],
        'patch_embed.proj.bias': source['embeddings.patch_embeddings.bias'],
        'norm.weight': source['norm.weight'],
        'norm.bias': source['norm.bias'],
    }
    for layer in range(12):
        src = f'layer.{layer}.'
        dst = f'blocks.{layer}.'
        for name in ('norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias'):
            converted[dst + name] = source[src + name]
        converted[dst + 'ls1.gamma'] = source[src + 'layer_scale1.lambda1']
        converted[dst + 'ls2.gamma'] = source[src + 'layer_scale2.lambda1']
        converted[dst + 'attn.qkv.weight'] = torch.cat((
            source[src + 'attention.q_proj.weight'],
            source[src + 'attention.k_proj.weight'],
            source[src + 'attention.v_proj.weight']), dim=0)
        converted[dst + 'attn.qkv.bias'] = torch.cat((
            source[src + 'attention.q_proj.bias'],
            torch.zeros_like(source[src + 'attention.q_proj.bias']),
            source[src + 'attention.v_proj.bias']), dim=0)
        converted[dst + 'attn.proj.weight'] = source[src + 'attention.o_proj.weight']
        converted[dst + 'attn.proj.bias'] = source[src + 'attention.o_proj.bias']
        converted[dst + 'mlp.fc1.weight'] = source[src + 'mlp.up_proj.weight']
        converted[dst + 'mlp.fc1.bias'] = source[src + 'mlp.up_proj.bias']
        converted[dst + 'mlp.fc2.weight'] = source[src + 'mlp.down_proj.weight']
        converted[dst + 'mlp.fc2.bias'] = source[src + 'mlp.down_proj.bias']

    derived = {key: value for key, value in target.items()
               if key == 'rope_embed.periods' or key.endswith('.attn.qkv.bias_mask')}
    expected_derived = {'rope_embed.periods'} | {
        f'blocks.{layer}.attn.qkv.bias_mask' for layer in range(12)}
    if set(derived) != expected_derived:
        raise ValueError(f'Pinned DINOv3 source has unexpected derived buffers: {sorted(derived)}')
    converted.update(derived)
    if set(converted) != set(target):
        missing, unexpected = sorted(set(target) - set(converted)), sorted(set(converted) - set(target))
        raise ValueError(f'DINOv3 mapped key set differs; missing={missing}, unexpected={unexpected}')
    for key, value in converted.items():
        if tuple(value.shape) != tuple(target[key].shape):
            raise ValueError(f'DINOv3 tensor shape mismatch for {key}: '
                             f'{tuple(value.shape)} != {tuple(target[key].shape)}')
    encoder.load_state_dict(converted, strict=True)


def load_dense_teacher(name, source, weights, expected_sha256, device='cuda'):
    source, weights = Path(source).resolve(), Path(weights).resolve()
    if not source.is_dir() or not weights.is_file() or sha256_file(weights) != expected_sha256:
        raise ValueError('Teacher source/checkpoint missing or checkpoint SHA256 differs')
    expected_commit = VJEPA_SOURCE_COMMIT if name == 'vjepa21' else DINOV3_SOURCE_COMMIT
    if _git_commit(source) != expected_commit:
        raise ValueError('Teacher upstream source commit differs from the pinned revision')
    if name == 'dinov3' and os.environ.get('DINOV3_LICENSE_ACCEPTED') != '1':
        raise PermissionError('Set DINOV3_LICENSE_ACCEPTED=1 only after accepting Meta DINOv3 terms')
    if name not in ('vjepa21', 'dinov3'):
        raise ValueError('Unsupported dense teacher')
    sys.path.insert(0, str(source))
    if name == 'vjepa21':
        from app.vjepa_2_1.models.vision_transformer import vit_base
        encoder = vit_base(patch_size=16, img_size=(384, 384), num_frames=64,
            tubelet_size=2, use_sdpa=True, use_silu=False, wide_silu=True,
            uniform_power=True, use_rope=True, img_temporal_dim_size=1,
            interpolate_rope=True)
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('ema_encoder'), dict):
            raise ValueError('V-JEPA 2.1 official ema_encoder checkpoint entry is missing')
        state = checkpoint['ema_encoder'].copy()
        for key in list(state):
            clean = key.replace('module.', '').replace('backbone.', '')
            if clean != key:
                if clean in state:
                    raise ValueError('V-JEPA 2.1 checkpoint key normalization collision')
                state[clean] = state.pop(key)
        result = encoder.load_state_dict(state, strict=False)
        if (set(result.missing_keys) - {'pos_embed'}
                or set(result.unexpected_keys) - {'pos_embed'}):
            raise ValueError('V-JEPA 2.1 EMA encoder key mismatch')
        input_size, source_grid, output_grid = 384, 24, 14
    else:
        encoder = torch.hub.load(str(source), 'dinov3_vitb16', source='local', pretrained=False)
        _load_dinov3_safetensors(encoder, weights)
        input_size, source_grid, output_grid = 224, 14, 14
    encoder.eval().to(device=device, dtype=torch.bfloat16)
    encoder.requires_grad_(False)
    return DenseTeacher(name, encoder, input_size, source_grid, output_grid, expected_sha256)


class DenseTeacher:
    def __init__(self, name, encoder, input_size, source_grid, output_grid, weight_sha256):
        self.name = name
        self.encoder = encoder
        self.input_size = int(input_size)
        self.source_grid = int(source_grid)
        self.output_grid = int(output_grid)
        self.weight_sha256 = weight_sha256
        self.feature_dim = 768
        self.feature_tokens = 2 * self.output_grid * self.output_grid

    @torch.inference_mode()
    def encode(self, paired_views):
        if (paired_views.ndim != 4 or paired_views.shape[1:] != (3, 224, 448)
                or not bool(torch.isfinite(paired_views).all())):
            raise ValueError('Paired RGB frames must be finite [-1,1] [B,3,224,448]')
        device = next(self.encoder.parameters()).device
        left, right = paired_views[..., :224].to(device), paired_views[..., 224:].to(device)
        views = torch.stack((left, right), dim=1).flatten(0, 1).float().add_(1).mul_(0.5)
        views = F.interpolate(views, size=(self.input_size, self.input_size), mode='bicubic',
                              align_corners=False, antialias=True)
        mean = views.new_tensor(IMAGENET_MEAN)[None, :, None, None]
        std = views.new_tensor(IMAGENET_STD)[None, :, None, None]
        views = ((views - mean) / std).to(dtype=torch.bfloat16)
        if self.name == 'vjepa21':
            # Upstream RoPE promotes Q/K to FP32; its normal BF16 autocast
            # context aligns SDPA inputs while preserving the pretrained math.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                features = self.encoder(views.unsqueeze(2))
        else:
            features = self.encoder.forward_features(views)
            if isinstance(features, dict):
                features = features.get('x_norm_patchtokens')
        if (not isinstance(features, torch.Tensor) or features.ndim != 3
                or features.shape[1:] != (self.source_grid ** 2, self.feature_dim)):
            raise ValueError('Teacher output must be its normalized dense patch tokens')
        if self.source_grid != self.output_grid:
            grid = features.transpose(1, 2).reshape(-1, self.feature_dim,
                self.source_grid, self.source_grid).float()
            grid = F.adaptive_avg_pool2d(grid, (self.output_grid, self.output_grid))
            features = grid.flatten(2).transpose(1, 2).to(torch.bfloat16)
        return features.reshape(len(paired_views), 2, self.output_grid ** 2,
                                self.feature_dim).flatten(1, 2).contiguous()

    def manifest(self):
        return dict(name=self.name, upstream_commit=VJEPA_SOURCE_COMMIT if self.name == 'vjepa21'
            else DINOV3_SOURCE_COMMIT, weight_sha256=self.weight_sha256,
            checkpoint_format='official_vjepa2_1_ema_encoder' if self.name == 'vjepa21'
            else 'modelscope_huggingface_safetensors_mapped_to_official_dinov3',
            preprocessing='two-view square resize bicubic antialias; ImageNet mean/std',
            input_size=self.input_size, source_grid=self.source_grid,
            output_grid=self.output_grid, feature_dim=self.feature_dim,
            feature_tokens=self.feature_tokens, token_order='cam_main then cam_wrist, row-major')
