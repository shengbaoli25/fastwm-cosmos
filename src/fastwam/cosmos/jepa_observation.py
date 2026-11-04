"""Observed-only V-JEPA 2.1 patches with last-four-block Q/V LoRA.

Input is floating RGB [-1, 1], [B, camera=2, C=3, observed_time=2, 224, 224].
The dataset owns causal frame selection; the encoder rejects longer clips.
Output is [B, 392, width], main-camera patches then wrist-camera patches,
row-major within each 14x14 grid. No global feature or future frame is used.
"""
from contextlib import nullcontext
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


SOURCE_COMMIT = '204698b45b3712590f06245fbfba32d3be539812'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VARIANT_SPECS = {
    'vitb': dict(width=768, depth=12, heads=12, factory='vit_base',
                 filename='vjepa2_1_vitb_dist_vitG_384.pt'),
    'vitl': dict(width=1024, depth=24, heads=16, factory='vit_large',
                 filename='vjepa2_1_vitl_dist_vitG_384.pt'),
}


def variant_spec(variant):
    if variant not in VARIANT_SPECS:
        raise ValueError('JEPA observation variant must be vitb or vitl')
    return dict(VARIANT_SPECS[variant])


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


class QVLoRALinear(nn.Module):
    """Keep an official fused QKV projection, adapting only Q and V rows."""

    def __init__(self, base, rank=8, alpha=8.):
        super().__init__()
        if not isinstance(base, nn.Linear) or base.out_features != 3 * base.in_features:
            raise ValueError('Expected the official square-width fused QKV projection')
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('LoRA rank and alpha must be positive')
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.scale = int(rank), float(alpha) / rank
        factory = dict(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_q_a = nn.Parameter(torch.empty(rank, self.in_features, **factory))
        self.lora_q_b = nn.Parameter(torch.zeros(self.in_features, rank, **factory))
        self.lora_v_a = nn.Parameter(torch.empty(rank, self.in_features, **factory))
        self.lora_v_b = nn.Parameter(torch.zeros(self.in_features, rank, **factory))
        nn.init.kaiming_uniform_(self.lora_q_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_v_a, a=math.sqrt(5))

    def forward(self, x):
        original = self.base(x)
        q, k, v = original.chunk(3, dim=-1)
        dq = F.linear(F.linear(x, self.lora_q_a), self.lora_q_b) * self.scale
        dv = F.linear(F.linear(x, self.lora_v_a), self.lora_v_b) * self.scale
        return torch.cat((q + dq, k, v + dv), dim=-1)

    def configure_trainable_parameters(self):
        self.requires_grad_(False)
        for name in ('lora_q_a', 'lora_q_b', 'lora_v_a', 'lora_v_b'):
            getattr(self, name).requires_grad_(True)


class JEPAObservationEncoder(nn.Module):
    def __init__(self, encoder, *, variant, weight_sha256, checkpoint_suffix=True):
        super().__init__()
        spec = variant_spec(variant)
        if len(encoder.blocks) != spec['depth'] or encoder.embed_dim != spec['width']:
            raise ValueError('Official JEPA encoder depth/width differs from variant')
        if not encoder.use_rope or encoder.tubelet_size != 2 or encoder.patch_size != 16:
            raise ValueError('Expected official RoPE, patch16, temporal tubelet2 encoder')
        self.encoder = encoder.eval().requires_grad_(False)
        self.variant = variant
        self.feature_dim = self.width = spec['width']
        self.feature_tokens = 392
        self.first_lora_block = spec['depth'] - 4
        self.checkpoint_suffix = bool(checkpoint_suffix)
        for block in self.encoder.blocks[self.first_lora_block:]:
            block.attn.qkv = QVLoRALinear(block.attn.qkv, rank=8, alpha=8.)
        self.identity = dict(
            schema='observed-vjepa21-qv-lora-v1', variant=variant,
            upstream_commit=SOURCE_COMMIT, checkpoint_key='ema_encoder',
            checkpoint_filename=spec['filename'], weight_sha256=weight_sha256,
            width=self.width, depth=spec['depth'], pretrained_input_size=384,
            input_shape=[2, 3, 2, 224, 224], input_range=[-1., 1.],
            normalization=dict(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
            output_shape=[392, self.width], token_order='camera_main_then_wrist_row_major',
            temporal_aggregation='observed_two_frame_tubelet2',
            position='official_interpolated_rope',
            lora=dict(rank=8, alpha=8., targets=['q', 'v'], last_blocks=4),
        )
        self.configure_trainable_parameters()

    def train(self, mode=True):
        # Evaluation mode disables all pretrained stochastic layers; it does
        # not disable autograd through the trainable LoRA suffix.
        super().train(mode)
        self.encoder.eval()
        return self

    def configure_trainable_parameters(self):
        """Restore the contract after a parent freezes/unfreezes its modules."""
        self.encoder.requires_grad_(False)
        for block in self.encoder.blocks[self.first_lora_block:]:
            block.attn.qkv.configure_trainable_parameters()
        self.encoder.eval()

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    @property
    def trainable_names(self):
        return tuple(name for name, p in self.named_parameters() if p.requires_grad)

    @staticmethod
    def _block(block, tokens):
        return block(tokens, mask=None, T=1, H_patches=14, W_patches=14,
                     return_attn=False, mode='video')[0]

    def forward(self, observed):
        if not isinstance(observed, torch.Tensor) or observed.ndim != 6 or tuple(observed.shape[1:]) != (2, 3, 2, 224, 224):
            raise ValueError('JEPA requires observed RGB [B,2,3,2,224,224] only')
        if not observed.is_floating_point():
            raise ValueError('JEPA observation RGB must be floating point in [-1,1]')
        if not bool(torch.isfinite(observed).all()) or bool((observed.abs() > 1.00001).any()):
            raise ValueError('JEPA observation RGB must be finite and in [-1,1]')
        parameter = next(self.encoder.parameters())
        device, dtype = parameter.device, parameter.dtype
        views = observed.to(device=device, dtype=torch.float32).flatten(0, 1)
        mean = views.new_tensor(IMAGENET_MEAN)[None, :, None, None, None]
        std = views.new_tensor(IMAGENET_STD)[None, :, None, None, None]
        views = ((views.add(1).mul(.5) - mean) / std).to(dtype=dtype)
        # Official RoPE promotes Q/K to FP32. Autocast realigns SDPA Q/K/V
        # when the stored backbone and LoRA weights are BF16/FP16.
        autocast = (torch.autocast(device_type=device.type, dtype=dtype)
                    if dtype in (torch.bfloat16, torch.float16) else nullcontext())
        with autocast:
            with torch.no_grad():
                tokens = self.encoder.patch_embed(views)
                if self.encoder.modality_embedding:
                    tokens = tokens + self.encoder.video_mod_embed
                for block in self.encoder.blocks[:self.first_lora_block]:
                    tokens = self._block(block, tokens)
            for block in self.encoder.blocks[self.first_lora_block:]:
                if self.checkpoint_suffix and self.training and torch.is_grad_enabled():
                    tokens = activation_checkpoint(self._block, block, tokens, use_reentrant=False)
                else:
                    tokens = self._block(block, tokens)
            tokens = self.encoder.norms_block[-1](tokens)
        expected = (len(observed) * 2, 196, self.width)
        if tuple(tokens.shape) != expected:
            raise ValueError(f'JEPA dense output {tuple(tokens.shape)} differs from {expected}')
        return tokens.reshape(len(observed), 392, self.width).contiguous()


def build_jepa_architecture(variant):
    """Construct only official encoder architecture; never download weights."""
    from ._vendor.vjepa21 import vision_transformer
    spec = variant_spec(variant)
    factory = getattr(vision_transformer, spec['factory'])
    return factory(patch_size=16, img_size=(384, 384), num_frames=64,
                   tubelet_size=2, use_sdpa=True, use_silu=False, wide_silu=True,
                   uniform_power=False, use_rope=True, img_temporal_dim_size=1,
                   interpolate_rope=True)


def _ema_state(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get('ema_encoder'), dict):
        raise ValueError('Official V-JEPA 2.1 ema_encoder checkpoint entry is required')
    state = {}
    for name, value in payload['ema_encoder'].items():
        clean = name
        while clean.startswith(('module.', 'backbone.')):
            clean = clean.split('.', 1)[1]
        if clean in state:
            raise ValueError('EMA checkpoint prefix normalization collision')
        state[clean] = value
    return state


def create_jepa_observation(*, variant, checkpoint, expected_sha256=None,
                            device='cuda', dtype=torch.bfloat16, checkpoint_suffix=True):
    """Load strict official EMA weights before adding any trainable adapter."""
    spec = variant_spec(variant)
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f'JEPA checkpoint missing: {path}')
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('JEPA checkpoint SHA256 differs from the pinned asset')
    encoder = build_jepa_architecture(variant)
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    state = _ema_state(payload)
    # RoPE has no learned absolute positional embedding. Only the unused
    # upstream pos_embed entry may be discarded; every actual weight is strict.
    state.pop('pos_embed', None)
    encoder.load_state_dict(state, strict=True)
    del payload, state
    encoder.to(device=device, dtype=dtype)
    return JEPAObservationEncoder(encoder, variant=variant,
                                  weight_sha256=digest, checkpoint_suffix=checkpoint_suffix)
