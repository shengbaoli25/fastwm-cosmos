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
TEACHER_SPECS = {
    'vjepa21': dict(feature_dim=768, input_size=384, source_grid=24,
        source_commit=VJEPA_SOURCE_COMMIT, factory='vit_base', depth=12,
        checkpoint_filename='vjepa2_1_vitb_dist_vitG_384.pt', checkpoint_key='ema_encoder',
        temporal_aggregation='joint_video_tubelet2'),
    'vjepa21_vitg': dict(feature_dim=1408, input_size=384, source_grid=24,
        source_commit=VJEPA_SOURCE_COMMIT, factory='vit_giant_xformers', depth=40,
        checkpoint_filename='vjepa2_1_vitg_384.pt', checkpoint_key='target_encoder',
        temporal_aggregation='joint_video_tubelet2'),
    'dinov3': dict(feature_dim=768, input_size=224, source_grid=14,
        source_commit=DINOV3_SOURCE_COMMIT, factory='dinov3_vitb16', depth=12,
        checkpoint_filename='model.safetensors', checkpoint_key=None,
        temporal_aggregation='independent_frame_encoding_adjacent_pair_mean'),
    'dinov3_vithplus': dict(feature_dim=1280, input_size=224, source_grid=14,
        source_commit=DINOV3_SOURCE_COMMIT, factory='dinov3_vith16plus', depth=32,
        checkpoint_filename='model.safetensors', checkpoint_key=None,
        temporal_aggregation='independent_frame_encoding_adjacent_pair_mean'),
}


def teacher_spec(name):
    """Return a copy of the pinned architecture and temporal geometry."""
    try:
        spec = TEACHER_SPECS[name].copy()
    except KeyError as error:
        raise ValueError(f'Unsupported dense teacher: {name}') from error
    return dict(spec, name=name, upstream_commit=spec['source_commit'],
        output_grid=14, feature_tokens=392, video_input_size=224)


def build_teacher_manifest(name, weight_sha256):
    """Preserve legacy cache identities; new H+ binds its temporal aggregation."""
    spec = teacher_spec(name)
    manifest = dict(name=name, upstream_commit=spec['source_commit'],
        weight_sha256=weight_sha256,
        checkpoint_format=(f"official_vjepa2_1_{spec['checkpoint_key']}" if name.startswith('vjepa21')
            else 'modelscope_huggingface_safetensors_mapped_to_official_dinov3'),
        preprocessing='two-view square resize bicubic antialias; ImageNet mean/std',
        input_size=spec['input_size'], source_grid=spec['source_grid'],
        output_grid=spec['output_grid'], feature_dim=spec['feature_dim'],
        feature_tokens=spec['feature_tokens'], token_order='cam_main then cam_wrist, row-major')
    if name == 'dinov3_vithplus':
        manifest.update(video_input_size=spec['video_input_size'],
            temporal_aggregation=spec['temporal_aggregation'])
    return manifest


class _FactoryScalarsOnCPU(torch.overrides.TorchFunctionMode):
    """Upstream V-JEPA converts its drop-path schedule to Python scalars.

    Keep only this parameter-free schedule on CPU during meta construction.
    Weight tensors still follow the caller's active torch.device context.
    """
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if func is torch.linspace:
            kwargs['device'] = 'cpu'
        return func(*args, **kwargs)


def build_teacher_architecture(name, source=None):
    """Build the pinned, unweighted encoder without network or checkpoint I/O.

    Supports ``with torch.device('meta')`` for exact parameter accounting.
    The factory represents the frozen encoder, never the trainable predictor.
    """
    spec = teacher_spec(name)
    if source is None:
        source = Path(__file__).resolve().parents[3] / 'third_party' / (
            'vjepa21' if name.startswith('vjepa21') else 'dinov3')
    source = Path(source).resolve()
    if not source.is_dir() or _git_commit(source) != spec['source_commit']:
        raise ValueError('Teacher upstream source commit differs from the pinned revision')
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    # Some optional torch infrastructure initializes scalar CPU tensors at
    # import time. Import under CPU even when the architecture lives on meta.
    with torch.device('cpu'):
        if name.startswith('vjepa21'):
            from app.vjepa_2_1.models.vision_transformer import vit_base, vit_giant_xformers
            factory = vit_base if name == 'vjepa21' else vit_giant_xformers
        else:
            from dinov3.hub.backbones import dinov3_vitb16, dinov3_vith16plus
            from dinov3.models import vision_transformer  # noqa: F401
            factory = dinov3_vith16plus if name == 'dinov3_vithplus' else dinov3_vitb16
    if name.startswith('vjepa21'):
        with _FactoryScalarsOnCPU():
            encoder = factory(patch_size=16, img_size=(384, 384), num_frames=64,
                tubelet_size=2, use_sdpa=True, use_silu=False, wide_silu=True,
                uniform_power=name == 'vjepa21', use_rope=True, img_temporal_dim_size=1,
                interpolate_rope=True)
    else:
        encoder = factory(pretrained=False)
    return encoder.eval().requires_grad_(False)


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
    encoder.load_state_dict(_map_dinov3_state(encoder, source), strict=True)


def _map_dinov3_state(encoder, source):
    """Strict HF-to-official mapping for Base MLP and H+ SwiGLU backbones."""
    depth = len(encoder.blocks)
    gated = hasattr(encoder.blocks[0].mlp, 'w1')
    mlp_names = ('gate_proj', 'up_proj', 'down_proj') if gated else ('up_proj', 'down_proj')
    expected_source = {
        'embeddings.cls_token', 'embeddings.mask_token',
        'embeddings.register_tokens', 'embeddings.patch_embeddings.weight',
        'embeddings.patch_embeddings.bias', 'norm.weight', 'norm.bias',
    }
    for layer in range(depth):
        prefix = f'layer.{layer}.'
        expected_source.update({
            prefix + 'norm1.weight', prefix + 'norm1.bias',
            prefix + 'norm2.weight', prefix + 'norm2.bias',
            prefix + 'layer_scale1.lambda1', prefix + 'layer_scale2.lambda1',
            prefix + 'attention.q_proj.weight', prefix + 'attention.q_proj.bias',
            prefix + 'attention.k_proj.weight',
            prefix + 'attention.v_proj.weight', prefix + 'attention.v_proj.bias',
            prefix + 'attention.o_proj.weight', prefix + 'attention.o_proj.bias',
        })
        expected_source.update(prefix + f'mlp.{name}.{suffix}'
            for name in mlp_names for suffix in ('weight', 'bias'))
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
    for layer in range(depth):
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
        # HF gate_proj is the SiLU branch: official w3(silu(w1(x)) * w2(x)).
        mlp_mapping = (('gate_proj', 'w1'), ('up_proj', 'w2'), ('down_proj', 'w3')) \
            if gated else (('up_proj', 'fc1'), ('down_proj', 'fc2'))
        for source_name, target_name in mlp_mapping:
            for suffix in ('weight', 'bias'):
                converted[dst + f'mlp.{target_name}.{suffix}'] = source[src + f'mlp.{source_name}.{suffix}']

    derived = {key: value for key, value in target.items()
               if key == 'rope_embed.periods' or key.endswith('.attn.qkv.bias_mask')}
    expected_derived = {'rope_embed.periods'} | {
        f'blocks.{layer}.attn.qkv.bias_mask' for layer in range(depth)}
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
    return converted


def validate_teacher_assets(name, source, weights, expected_sha256):
    """Validate a cache assembler's inputs without constructing an encoder."""
    spec = teacher_spec(name)
    source, weights = Path(source).resolve(), Path(weights).resolve()
    if not source.is_dir() or not weights.is_file() or sha256_file(weights) != expected_sha256:
        raise ValueError('Teacher source/checkpoint missing or checkpoint SHA256 differs')
    if _git_commit(source) != spec['source_commit']:
        raise ValueError('Teacher upstream source commit differs from the pinned revision')
    if name.startswith('dinov3') and os.environ.get('DINOV3_LICENSE_ACCEPTED') != '1':
        raise PermissionError('Set DINOV3_LICENSE_ACCEPTED=1 only after accepting Meta DINOv3 terms')
    if weights.name != spec['checkpoint_filename']:
        raise ValueError('Teacher checkpoint filename differs from pinned variant')


def load_dense_teacher(name, source, weights, expected_sha256, device='cuda'):
    spec = teacher_spec(name)
    validate_teacher_assets(name, source, weights, expected_sha256)
    source, weights = Path(source).resolve(), Path(weights).resolve()
    encoder = build_teacher_architecture(name, source)
    if name in ('vjepa21', 'vjepa21_vitg'):
        checkpoint_key = spec['checkpoint_key']
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get(checkpoint_key), dict):
            raise ValueError(f'V-JEPA 2.1 official {checkpoint_key} checkpoint entry is missing')
        state = checkpoint[checkpoint_key].copy()
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
        del checkpoint, state
    else:
        _load_dinov3_safetensors(encoder, weights)
    encoder.eval().to(device=device, dtype=torch.bfloat16)
    if name == 'dinov3_vithplus':
        # Keep the official RoPE frequencies in FP32 after casting weights.
        encoder.rope_embed._init_weights()
    encoder.requires_grad_(False)
    return DenseTeacher(name, encoder, spec['input_size'], spec['source_grid'],
                        spec['output_grid'], expected_sha256)


class DenseTeacher:
    def __init__(self, name, encoder, input_size, source_grid, output_grid, weight_sha256):
        self.name = name
        self.encoder = encoder
        self.input_size = int(input_size)
        self.source_grid = int(source_grid)
        self.output_grid = int(output_grid)
        self.weight_sha256 = weight_sha256
        self.feature_dim = teacher_spec(name)['feature_dim']
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
        if self.name.startswith('vjepa21'):
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

    @torch.inference_mode()
    def encode_video(self, paired_video, *, context_only=False):
        """Encode an observed-only clip or a separate full target clip.

        Context accepts exactly two frames so a caller cannot inadvertently
        encode future target frames along the observation path. DINO encodes
        each frame independently, then means adjacent pairs into five slots.
        """
        expected_frames = 2 if context_only else 10
        if (self.name not in TEACHER_SPECS or paired_video.ndim != 5
                or paired_video.shape[1:] != (3, expected_frames, 224, 448)
                or not bool(torch.isfinite(paired_video).all())):
            raise ValueError('Teacher video must be [B,3,2|10,224,448]')
        device = next(self.encoder.parameters()).device
        views = torch.stack((paired_video[..., :224], paired_video[..., 224:]), 1)
        views = views.flatten(0, 1).to(device=device, dtype=torch.float32).add_(1).mul_(0.5)
        b2, channels, times, height, width = views.shape
        views = F.interpolate(views.transpose(1, 2).reshape(b2*times, channels, height, width),
                              size=(224, 224), mode='bicubic', align_corners=False,
                              antialias=True).reshape(b2, times, channels, 224, 224).transpose(1, 2)
        mean = views.new_tensor(IMAGENET_MEAN)[None, :, None, None, None]
        std = views.new_tensor(IMAGENET_STD)[None, :, None, None, None]
        views = ((views - mean) / std).to(torch.bfloat16)
        tubelets = expected_frames // 2
        grid = self.output_grid ** 2
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            if self.name.startswith('vjepa21'):
                features = self.encoder(views)
                if (not isinstance(features, torch.Tensor) or features.ndim != 3
                        or features.shape[1:] != (tubelets * grid, self.feature_dim)):
                    raise ValueError('V-JEPA video output tubelet geometry differs')
            else:
                frames = views.transpose(1, 2).reshape(b2 * times, channels, 224, 224)
                features = self.encoder.forward_features(frames)
                if isinstance(features, dict):
                    features = features.get('x_norm_patchtokens')
                if (not isinstance(features, torch.Tensor) or features.shape !=
                        (b2 * times, grid, self.feature_dim)):
                    raise ValueError('DINOv3 video output frame geometry differs')
                features = (features.reshape(b2, tubelets, 2, grid, self.feature_dim)
                            .float().mean(2).to(torch.bfloat16))
        return (features.reshape(len(paired_video), 2, tubelets, grid, self.feature_dim)
                .permute(0, 2, 1, 3, 4).flatten(2, 3).contiguous())

    def manifest(self):
        manifest = build_teacher_manifest(self.name, self.weight_sha256)
        if any(manifest[key] != getattr(self, key) for key in
               ('input_size', 'source_grid', 'output_grid', 'feature_dim', 'feature_tokens')):
            raise ValueError('Teacher runtime geometry differs from the pinned variant')
        return manifest
