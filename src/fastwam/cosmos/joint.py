"""Bidirectional Video/Force attention, with clean Action in both world models.

Force supervision uses a differentiable refresh of independently generated
video. The diffusion rollout is detached, never replaced by target latents.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fastwam.forcewm.chunk import ChunkForceWM
from fastwam.forcewm.model import Attention, ForcePrediction
from .model import CosmosForceWM


JOINT_PROTOCOL = 'video-force-joint-huw-bidirectional-generated-refresh-v1'


def joint_attention_mask(video_tokens, force_mask, force_valid):
    """V<->valid H/U/W, native bidirectional V, inherited Force direct edges.

    Invalid Force queries retain their own diagonal sentinel only. H/U/W can
    exchange information indirectly through V across layers; the old strict
    multi-layer isolation consequently does not apply to this experiment.
    """
    batch, size = force_valid.shape
    if (video_tokens < 1 or force_valid.dtype != torch.bool
            or force_mask.dtype != torch.bool
            or force_mask.shape != (batch, 1, size, size)):
        raise ValueError('Boolean Force mask and positive Video token count required')
    result = torch.ones((batch, 1, video_tokens+size, video_tokens+size),
                        device=force_valid.device, dtype=torch.bool)
    result[:, :, :video_tokens, video_tokens:] = force_valid[:, None, None, :]
    result[:, :, video_tokens:, :video_tokens] = force_valid[:, None, :, None]
    result[:, :, video_tokens:, video_tokens:] = force_mask
    return result


class JointForceBlock(nn.Module):
    def __init__(self, original, dim, width, heads):
        super().__init__()
        # Reuse the matched initialized residual/FFN weights. The previous
        # one-way video cross-attention is replaced, not left unused in Adam.
        self.norm1, self.norm2, self.ff = original.norm1, original.norm2, original.ff
        self.self_attention = Attention(dim, dim, width, heads)

    def forward(self, force, video_q, video_k, video_v, mask):
        attention = self.self_attention
        normalized = self.norm1(force)
        fq = attention.split(attention.q(normalized))
        fk, fv = attention.kv(normalized)
        q = torch.cat((video_q, fq), 2)
        k = torch.cat((video_k, fk), 2)
        v = torch.cat((video_v, fv), 2)
        mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        video_tokens = video_q.shape[2]
        force = force + attention.out(mixed[:, :, video_tokens:].transpose(1, 2).flatten(2))
        force = force + self.ff(self.norm2(force))
        return mixed[:, :, :video_tokens], force


class JointChunkForceWM(ChunkForceWM):
    def __init__(self, *, video_hidden_dim=2048, hidden_dim=1280, num_heads=16, **kwargs):
        super().__init__(video_hidden_dim=video_hidden_dim, hidden_dim=hidden_dim,
                         num_heads=num_heads, **kwargs)
        # Do not advance the baseline RNG stream used to initialize AE memory.
        with torch.random.fork_rng(devices=[]):
            self.blocks = nn.ModuleList([JointForceBlock(block, hidden_dim,
                video_hidden_dim, num_heads) for block in self.blocks])

    def forward(self, **kwargs):
        raise RuntimeError('Joint Force must execute together with Cosmos through _joint_video')


class JointCosmosForceWM(CosmosForceWM):
    force_type = JointChunkForceWM
    checkpoint_schema = 'fastwam-cosmos-force-joint-self-attention28-v1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.dit['force'].num_heads != self.video.num_heads
                or self.video.hidden_dim != self.video.num_heads*self.video.attn_head_dim):
            raise ValueError('Joint attention requires matching native Cosmos head geometry')

    def _joint_video(self, latents, clock, raw, context, cmask, sample, action, apad,
                     *, collect=False):
        prepared = self.video.prepare(latents, clock.float()/1000., raw,
            candidate_actions=self.dit['controls'](action.detach(), apad),
            fps=self.fps, context_is_projected=True)
        force = self.dit['force']
        f, fmask, valid, wpad, first_w = force.prepare(
            history=sample['history'].to(self.device, self.torch_dtype),
            action_condition=action.detach(), context=context.detach(), context_mask=cmask,
            history_is_pad=sample['history_is_pad'].to(self.device, torch.bool),
            candidate_is_pad=apad,
            history_channel_valid=sample['history_channel_valid'].to(self.device, torch.bool),
            observation_latents=sample['first_frame_latents'].to(self.device, self.torch_dtype))
        x = prepared['tokens']
        mask = joint_attention_mask(x.shape[1], fmask, valid)
        video_layers, force_layers = [], []
        for i, block in enumerate(force.blocks):
            def run(video_tokens, force_tokens, i=i, block=block):
                q, k, v, state = self.video.block_qkv(i, video_tokens, prepared)
                mixed, next_force = block(force_tokens, q, k, v, mask)
                next_video = self.video.block_finish(i, mixed, state, prepared)
                return next_video, torch.where(valid[..., None], next_force, 0)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x, f = checkpoint(run, x, f, use_reentrant=False)
            else:
                x, f = run(x, f)
            if collect:
                video_layers.append(x[:, prepared['tokens_per_frame']:].detach())
                force_layers.append(f[:, first_w:])
        prediction = (ForcePrediction(force.head(f[:, first_w:]), tuple(force_layers), wpad)
                      if collect else None)
        return self.video.finish(x, prepared), tuple(video_layers), prediction

    def _supervised_video(self, sample, noisy, clock, raw, context, cmask, action, apad):
        return self._joint_video(noisy, clock, raw, context, cmask, sample, action, apad)[0]

    @torch.no_grad()
    def generate_video_memory(self, first, raw_context, action_condition, action_is_pad,
                              *, sample, context, cmask):
        """Free rollout only. A separate refresh supplies Force loss gradients."""
        shape = (*first.shape[:2], 3, *first.shape[3:])
        x = self.dit['runtime'].normal(shape).to(self.device, self.torch_dtype)
        x[:, :, :1] = first.detach()
        times, deltas = self.v_scheduler.build_inference_schedule(
            self.generated_video_steps, self.device, x.dtype)
        for t, dt in zip(times, deltas):
            velocity, _, _ = self._joint_video(x, t.expand(len(x)), raw_context,
                context, cmask, sample, action_condition, action_is_pad)
            x = self.v_scheduler.step(velocity, dt, x)
            x[:, :, :1] = first.detach()
        return x.detach()

    def _world_models(self, sample, first, raw, context, cmask, action, apad):
        x = self.generate_video_memory(first, raw.detach(), action, apad,
                                      sample=sample, context=context, cmask=cmask)
        # This pass is differentiable during training, including native Cosmos
        # attention. No future Video/Force label or AE state enters the refresh.
        zero = torch.zeros(len(x), device=self.device, dtype=self.torch_dtype)
        _, layers, force = self._joint_video(x, zero, raw, context, cmask,
                                           sample, action, apad, collect=True)
        pad = torch.zeros(layers[0].shape[:2], device=self.device, dtype=torch.bool)
        return (x, layers, pad), force
