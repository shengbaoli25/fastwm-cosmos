"""Direct future-wrench chunk: measured history + clean controls + learned queries.

All horizon queries execute together. No previous predicted/teacher wrench or
Action diffusion clock is an input. Target validity is owned by the loss.
"""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .model import ForceBlock, ForcePrediction
from .masks import build_force_attention_mask, mask_invalid_channels


class ChunkForceWM(nn.Module):
    def __init__(self, video_hidden_dim=2048, text_dim=4096, hidden_dim=1280,
                 ffn_dim=4608, num_heads=16, num_layers=28, horizon=32,
                 history_steps=32, checkpoint_layers=True, observation_channels=16):
        super().__init__()
        self.horizon, self.history_steps = horizon, history_steps
        self.hidden_dim, self.num_heads = hidden_dim, num_heads
        self.checkpoint_layers = checkpoint_layers
        self.history_encoder = nn.Linear(26, hidden_dim)
        self.action_encoder = nn.Linear(14, hidden_dim)
        self.observation_channels = observation_channels
        self.observation_encoder = nn.Linear(observation_channels, hidden_dim)
        self.text_encoder = nn.Linear(text_dim, hidden_dim)
        self.queries = nn.Parameter(torch.randn(horizon, hidden_dim) / math.sqrt(hidden_dim))
        self.position = nn.Parameter(torch.randn(history_steps+2*horizon, hidden_dim) / math.sqrt(hidden_dim))
        self.blocks = nn.ModuleList([ForceBlock(hidden_dim, ffn_dim, num_heads, video_hidden_dim)
                                    for _ in range(num_layers)])
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 6))

    def forward(self, *, history, action_condition, context, context_mask,
                history_is_pad, candidate_is_pad, history_channel_valid,
                observation_latents, video_features, video_is_pad):
        b = history.shape[0]
        if (history.shape != (b, self.history_steps, 13)
                or action_condition.shape != (b, self.horizon, 7)
                or len(video_features) != len(self.blocks)
                or observation_latents.ndim != 5
                or observation_latents.shape[1:3] != (self.observation_channels, 1)):
            raise ValueError('Force history/clean action/video geometry differs')
        h, hv = mask_invalid_channels(history, history_is_pad, history_channel_valid)
        u, uv = mask_invalid_channels(action_condition, candidate_is_pad)
        if not bool(hv[:, -1, 7:].all()):
            raise ValueError('Measured current wrench required')
        if (context_mask.dtype != torch.bool or video_is_pad.dtype != torch.bool
                or not bool(context_mask.any(-1).all()) or not bool((~video_is_pad).any(-1).all())):
            raise ValueError('Available text/video context required')
        text = torch.where(context_mask[..., None], context, 0).sum(1)
        text = self.text_encoder(text / context_mask.sum(-1).clamp_min(1)[:, None])
        obs = self.observation_encoder(observation_latents.detach().mean((2, 3, 4)))
        h = self.history_encoder(torch.cat((h, hv.to(h.dtype)), -1)) + text[:, None] + obs[:, None]
        u = self.action_encoder(torch.cat((u, uv.to(u.dtype)), -1)) + text[:, None]
        first_w = self.history_steps + self.horizon
        w = self.queries[None].expand(b, -1, -1) + text[:, None]
        x = torch.cat((h, u, w), 1) + self.position
        wpad = torch.zeros((b, self.horizon), dtype=torch.bool, device=x.device)
        valid = torch.cat((hv.any(-1), uv.any(-1), ~wpad), 1)
        x = torch.where(valid[..., None], x, 0)
        mask = build_force_attention_mask(history_is_pad, candidate_is_pad, wpad,
                                          history_channel_valid=history_channel_valid)
        vmask = (~video_is_pad)[:, None, None, :]
        layers = []
        for block, video in zip(self.blocks, video_features):
            def run(tokens, features, block=block):
                out = block(tokens, features, mask, vmask, first_w)
                return torch.where(valid[..., None], out, 0)
            features = torch.where((~video_is_pad)[..., None], video.detach(), 0)
            x = checkpoint(run, x, features, use_reentrant=False) if self.training and self.checkpoint_layers else run(x, features)
            layers.append(x[:, first_w:])
        return ForcePrediction(self.head(x[:, first_w:]), tuple(layers), wpad)
