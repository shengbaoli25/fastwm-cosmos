"""Future-wrench flow matching; observed history never receives a noise clock."""
import torch
from torch import nn
from .model import Attention
from .masks import mask_invalid_channels
from fastwam.models.wan22.wan_video_dit import sinusoidal_embedding_1d


class ForceDiffusionBlock(nn.Module):
    def __init__(self, dim, ffn, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = Attention(dim, dim, width, heads)
        self.ff = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Linear(ffn, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**.5)

    def qkv(self, x, time_mod=None):
        mod = self.modulation if time_mod is None else self.modulation + time_mod
        shift, scale, gate, ms, mc, mg = mod.unbind(1)
        norm = self.norm1(x) * (1 + scale[:, None]) + shift[:, None]
        q = self.attention.split(self.attention.q(norm))
        k, v = self.attention.kv(norm)
        return q, k, v, (x, gate, ms, mc, mg)

    def forward(self, mixed, state):
        x, gate, shift, scale, mg = state
        x = x + gate[:, None] * self.attention.out(mixed.transpose(1, 2).flatten(2))
        return x + mg[:, None] * self.ff(self.norm2(x) * (1 + scale[:, None]) + shift[:, None])


class ForceDiffusionExpert(nn.Module):
    def __init__(self, *, hidden_dim=768, ffn_dim=2048, num_layers=28,
                 width=2048, num_heads=16, text_dim=1024, horizon=32, history_steps=32):
        super().__init__()
        self.hidden_dim, self.horizon, self.history_steps = hidden_dim, horizon, history_steps
        self.num_heads = num_heads
        self.history_encoder = nn.Linear(26, hidden_dim)
        self.force_encoder = nn.Linear(6, hidden_dim)
        self.text_encoder = nn.Linear(text_dim, hidden_dim)
        self.position = nn.Parameter(torch.randn(history_steps + horizon, hidden_dim) / hidden_dim**.5)
        self.time_embedding = nn.Sequential(nn.Linear(512, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))
        self.blocks = nn.ModuleList(ForceDiffusionBlock(hidden_dim, ffn_dim, width, num_heads) for _ in range(num_layers))
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 6))

    def prepare_history(self, history, is_pad, channel_valid, context):
        h, valid = mask_invalid_channels(history, is_pad, channel_valid)
        if h.shape[1:] != (self.history_steps, 13):
            raise ValueError('Measured History32x13 required')
        text = self.text_encoder(context.mean(1))[:, None]
        x = self.history_encoder(torch.cat((h, valid.to(h.dtype)), -1)) + text + self.position[:self.history_steps]
        return torch.where(valid.any(-1)[..., None], x, 0), valid.any(-1)

    def prepare_future(self, force, tf, tv, context):
        if force.shape[1:] != (self.horizon, 6):
            raise ValueError('Noisy Force32x6 required')
        x = self.force_encoder(force) + self.position[self.history_steps:] + self.text_encoder(context.mean(1))[:, None]
        time = torch.cat((sinusoidal_embedding_1d(256, tf), sinusoidal_embedding_1d(256, tv)), -1)
        mod = self.time_projection(self.time_embedding(time)).unflatten(-1, (6, self.hidden_dim))
        return x, mod
