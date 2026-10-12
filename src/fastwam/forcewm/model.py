"""Causal continuous-wrench decoder with shifted supervision and cached rollout.

H reads causal H. U reads bidirectional U. W reads H/U and causal W. Only W
reads generated same-layer video. A W input is the previous wrench (w0 for
the first token), never the wrench being predicted. Force has no noise input.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .masks import build_force_attention_mask, mask_invalid_channels


@dataclass
class ForcePrediction:
    wrench: Tensor
    layers: tuple[Tensor, ...]
    is_pad: Tensor


def masked_force_mse(prediction, target, is_pad, channel_valid):
    if (prediction.ndim != 3 or prediction.shape[-1] != 6 or target.shape != prediction.shape
            or channel_valid.shape != prediction.shape or channel_valid.dtype != torch.bool
            or is_pad.shape != prediction.shape[:2] or is_pad.dtype != torch.bool):
        raise ValueError("Force loss requires [B,T,6] values and Boolean validity")
    valid = channel_valid.to(prediction.device) & ~is_pad.to(prediction.device)[..., None]
    p = torch.where(valid, prediction.float(), 0)
    y = torch.where(valid, target.to(prediction.device, torch.float32), 0)
    if not bool(torch.isfinite(p).all() & torch.isfinite(y).all()):
        raise ValueError("Measured Force loss must be finite")
    return ((p-y).square().sum((1,2)) / valid.sum((1,2)).clamp_min(1)).mean()


def shifted_wrench_inputs(history, future, future_is_pad, future_valid):
    """[w0,w1,...,w31] predicts [w1,...,w32]; never pass unshifted labels."""
    if history.ndim != 3 or history.shape[-1] != 13 or future.shape[-1] != 6:
        raise ValueError("History13 and future wrench6 required")
    if future_valid.dtype != torch.bool or future_is_pad.dtype != torch.bool:
        raise ValueError("Boolean future validity required")
    valid = future_valid & ~future_is_pad[..., None]
    clean = torch.where(valid, future, 0)
    if not bool(torch.isfinite(clean).all()) or not bool(torch.isfinite(history[:, -1, 7:]).all()):
        raise ValueError("Force teacher inputs must be finite")
    return torch.cat((history[:, -1:, 7:], clean[:, :-1]), dim=1).detach()


class Attention(nn.Module):
    def __init__(self, query_dim, kv_dim, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError("Attention width must be divisible by heads")
        self.heads, self.head_dim = heads, width // heads
        self.q = nn.Linear(query_dim, width)
        self.k = nn.Linear(kv_dim, width)
        self.v = nn.Linear(kv_dim, width)
        self.out = nn.Linear(width, query_dim)

    def split(self, x):
        return x.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)

    def kv(self, x):
        return self.split(self.k(x)), self.split(self.v(x))

    def attend(self, x, kv, mask=None):
        y = F.scaled_dot_product_attention(self.split(self.q(x)), *kv,
                                          attn_mask=mask, dropout_p=0.)
        return self.out(y.transpose(1, 2).flatten(-2))


class ForceBlock(nn.Module):
    def __init__(self, dim, ffn, heads, video_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attention = Attention(dim, dim, dim, heads)
        self.norm_video = nn.LayerNorm(dim)
        self.video_attention = Attention(dim, video_dim, dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Linear(ffn, dim))

    def finish(self, x):
        return x + self.ff(self.norm2(x))

    def forward(self, x, video, self_mask, video_mask, first_w):
        y = self.norm1(x)
        x = x + self.self_attention.attend(y, self.self_attention.kv(y), self_mask)
        w = x[:, first_w:]
        w = w + self.video_attention.attend(self.norm_video(w),
                    self.video_attention.kv(video), video_mask)
        return self.finish(torch.cat((x[:, :first_w], w), dim=1))


class AutoregressiveForceWM(nn.Module):
    def __init__(self, video_hidden_dim=3072, text_dim=4096, hidden_dim=1280,
                 ffn_dim=4608, num_heads=16, num_layers=30, horizon=32,
                 history_steps=32, checkpoint_layers=True, observation_channels=48):
        super().__init__()
        self.horizon, self.history_steps = horizon, history_steps
        self.hidden_dim, self.num_heads = hidden_dim, num_heads
        self.checkpoint_layers = checkpoint_layers
        self.history_encoder = nn.Linear(26, hidden_dim)
        self.action_encoder = nn.Linear(14, hidden_dim)
        self.wrench_encoder = nn.Linear(6, hidden_dim)
        self.observation_channels = observation_channels
        self.observation_encoder = nn.Linear(observation_channels, hidden_dim)
        self.text_encoder = nn.Linear(text_dim, hidden_dim)
        self.action_clock = nn.Sequential(nn.Linear(2, hidden_dim), nn.SiLU(),
                                          nn.Linear(hidden_dim, hidden_dim))
        self.position = nn.Parameter(torch.randn(history_steps+2*horizon, hidden_dim) / math.sqrt(hidden_dim))
        self.blocks = nn.ModuleList([ForceBlock(hidden_dim, ffn_dim, num_heads, video_hidden_dim)
                                    for _ in range(num_layers)])
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 6))

    def _condition(self, history, noisy_action, action_clock, context, context_mask,
                   history_is_pad, candidate_is_pad, history_channel_valid,
                   observation_latents, video_features, video_is_pad, **unused):
        b = history.shape[0]
        if (history.shape != (b,self.history_steps,13) or noisy_action.shape != (b,self.horizon,7)
                or len(video_features) != len(self.blocks) or observation_latents.ndim != 5
                or observation_latents.shape[1:3] != (self.observation_channels,1)):
            raise ValueError("Force history/action/video geometry differs")
        if not bool((~history_is_pad[:, -1]).all()):
            raise ValueError("Measured anchor wrench w0 required")
        h, hv = mask_invalid_channels(history, history_is_pad, history_channel_valid)
        if not bool(hv[:, -1, 7:].all()):
            raise ValueError("All six current wrench channels are required")
        u, uv = mask_invalid_channels(noisy_action, candidate_is_pad)
        if not bool((~video_is_pad).any(-1).all()) or not bool(context_mask.any(-1).all()):
            raise ValueError("Available video and text context required")
        text = torch.where(context_mask[..., None], context, 0).sum(1)
        text = text / context_mask.sum(-1).clamp_min(1)[:, None]
        text = self.text_encoder(text)
        clock = action_clock.to(history.dtype).reshape(b, 1) / 1000.
        clock = self.action_clock(torch.cat((clock, clock.square()), -1))
        obs = self.observation_encoder(observation_latents.detach().mean((2,3,4)))
        h = self.history_encoder(torch.cat((h,hv.to(h.dtype)), -1)) + text[:,None] + obs[:,None]
        u = self.action_encoder(torch.cat((u,uv.to(u.dtype)), -1)) + text[:,None] + clock[:,None]
        hu = torch.cat((h,u), 1) + self.position[:self.history_steps+self.horizon]
        valid = torch.cat((hv.any(-1), uv.any(-1)), 1)
        return torch.where(valid[...,None], hu, 0), valid, text+clock

    def forward_teacher(self, *, previous_wrench, **condition):
        """Parallel causal teacher-forced pass; loss labels remain caller-owned."""
        hu, hv, cond = self._condition(**condition)
        if previous_wrench.shape != (hu.shape[0], self.horizon,6) or not bool(torch.isfinite(previous_wrench).all()):
            raise ValueError("Finite shifted previous-wrench sequence required")
        h = self.history_steps + self.horizon
        w = self.wrench_encoder(previous_wrench.detach()) + self.position[h:] + cond[:,None]
        x = torch.cat((hu,w),1)
        # Requested horizon is fixed and available; target masks belong to loss.
        wpad = torch.zeros(previous_wrench.shape[:2], device=x.device, dtype=torch.bool)
        mask = build_force_attention_mask(condition['history_is_pad'], condition['candidate_is_pad'], wpad,
                                          history_channel_valid=condition['history_channel_valid'])
        vmask = (~condition['video_is_pad'])[:,None,None,:]
        valid = torch.cat((hv, ~wpad),1)
        layers=[]
        for block, video in zip(self.blocks,condition['video_features']):
            def run(tokens, vi, block=block):
                out=block(tokens, vi, mask, vmask, h)
                return torch.where(valid[...,None],out,0)
            x = checkpoint(run,x,video.detach(),use_reentrant=False) if self.training and self.checkpoint_layers else run(x,video.detach())
            layers.append(x[:,h:])
        return ForcePrediction(self.head(x[:,h:]), tuple(layers), wpad)

    @torch.no_grad()
    def rollout(self, **condition):
        """Generate w1..w32 from measured w0 using exact per-layer KV caches.

        No future wrench, future mask, random generator, or noise is accepted.
        Each call refreshes H/U under the supplied current Action and clock.
        """
        forbidden={'previous_wrench','wrench_future','target','future_is_pad','wrench_valid','noise'} & condition.keys()
        if forbidden:
            raise ValueError('Rollout does not accept future labels/noise: '+str(sorted(forbidden)))
        hu, valid, cond=self._condition(**condition)
        b=hu.shape[0]; h=self.history_steps+self.horizon
        wpad=torch.zeros((b,self.horizon),dtype=torch.bool,device=hu.device)
        full_mask=build_force_attention_mask(condition['history_is_pad'],condition['candidate_is_pad'],wpad,
                                             history_channel_valid=condition['history_channel_valid'])
        caches=[]
        for block,video in zip(self.blocks,condition['video_features']):
            n=block.norm1(hu); kv=block.self_attention.kv(n)
            hu=hu+block.self_attention.attend(n,kv,full_mask[:,:,:h,:h])
            hu=torch.where(valid[...,None],block.finish(hu),0)
            caches.append([kv,block.video_attention.kv(video.detach())])
        previous=condition['history'][:,-1:,7:]
        wrenches=[]; layers=[[] for _ in self.blocks]
        for step in range(self.horizon):
            x=self.wrench_encoder(previous)+self.position[h+step:h+step+1]+cond[:,None]
            for i,block in enumerate(self.blocks):
                n=block.norm1(x); new_kv=block.self_attention.kv(n)
                kv=tuple(torch.cat((old,new),dim=2) for old,new in zip(caches[i][0],new_kv))
                caches[i][0]=kv
                x=x+block.self_attention.attend(n,kv,full_mask[:,:,h+step:h+step+1,:h+step+1])
                x=x+block.video_attention.attend(block.norm_video(x),caches[i][1],(~condition['video_is_pad'])[:,None,None,:])
                x=block.finish(x); layers[i].append(x)
            previous=self.head(x)
            wrenches.append(previous)
        return ForcePrediction(torch.cat(wrenches,1),tuple(torch.cat(x,1) for x in layers),wpad)


class ActionDualMemory(nn.Module):
    """Zero-gated residuals: Action reads generated Video and free Force only."""
    def __init__(self, action_config, video_hidden_dim, force_hidden_dim=1280,
                 attention_width=512, heads=8):
        super().__init__()
        dim=action_config['hidden_dim']
        self.norm=nn.LayerNorm(dim,elementwise_affine=False)
        self.video_cross=Attention(dim,video_hidden_dim,attention_width,heads)
        self.force_cross=Attention(dim,force_hidden_dim,attention_width,heads)
        self.video_gate=nn.Parameter(torch.zeros(()))
        self.force_gate=nn.Parameter(torch.zeros(()))

    def forward(self,action,video,force,video_is_pad,force_is_pad,action_is_pad):
        q=self.norm(action)
        v=self.video_cross.attend(q,self.video_cross.kv(video.detach()),(~video_is_pad)[:,None,None,:])
        f=self.force_cross.attend(q,self.force_cross.kv(force.detach()),(~force_is_pad)[:,None,None,:])
        return action+torch.where((~action_is_pad)[...,None],self.video_gate*v+self.force_gate*f,0)


def forward_joint_with_memories(mot, *, video_features, force_prediction, video_is_pad,
                                action_is_pad, **kwargs):
    video, action=kwargs.pop('video_tokens'),kwargs.pop('action_tokens')
    if len(video_features)!=mot.num_layers or len(force_prediction.layers)!=mot.num_layers:
        raise ValueError('Same-layer memory must cover all 30 blocks')
    for i in range(mot.num_layers):
        video,action=mot._forward_joint_layer(video_block=mot.mixtures['video'].blocks[i],
            action_block=mot.mixtures['action'].blocks[i],video_tokens=video,action_tokens=action,**kwargs)
        module=mot.action_dual_memory[i]
        def memory(a,v,f,module=module):
            return module(a,v,f,video_is_pad,force_prediction.is_pad,action_is_pad)
        if torch.is_grad_enabled():
            action=checkpoint(memory,action,video_features[i],force_prediction.layers[i],use_reentrant=False)
        else:
            action=memory(action,video_features[i],force_prediction.layers[i])
    return video,action
