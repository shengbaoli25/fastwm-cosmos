"""O/R/V/F/A masked joint attention with a reusable WM prefix.

O reads O; R reads O/R; V and F read O/R/V/F; A reads everything.
The clean reference is currently recorded Action32, never noisy AE states.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fastwam.forcewm.chunk import ChunkForceWM
from fastwam.forcewm.model import Attention, ForcePrediction
from .model import CosmosForceWM

JOINT_PROTOCOL = 'orvfa-masked-joint-native-kv-generated-refresh-v2'
GROUP_READS = ((1, 0, 0, 0, 0), (1, 1, 0, 0, 0),
               (1, 1, 1, 1, 0), (1, 1, 1, 1, 0), (1, 1, 1, 1, 1))


def grouped_attention_mask(groups, valid):
    """Boolean allowed edges; unavailable rows have only a diagonal sentinel."""
    if (groups.ndim != 1 or valid.ndim != 2 or groups.numel() != valid.shape[1]
            or groups.dtype != torch.long or valid.dtype != torch.bool
            or not bool(((groups >= 0) & (groups < 5)).all())):
        raise ValueError('O/R/V/F/A group IDs and boolean token validity required')
    reads = torch.tensor(GROUP_READS, device=valid.device, dtype=torch.bool)
    allowed = reads[groups[:, None], groups[None, :]][None]
    allowed = allowed & valid[:, :, None] & valid[:, None, :]
    sentinel = torch.eye(len(groups), device=valid.device, dtype=torch.bool)[None]
    return (allowed | ((~valid)[:, :, None] & sentinel))[:, None]


def joint_token_layout(video_tokens, observed_tokens, force_valid, history_steps, horizon):
    """Physical order: [observed video, future video, history, reference, force]."""
    if (not 0 < observed_tokens < video_tokens
            or force_valid.shape[1] != history_steps + 2 * horizon):
        raise ValueError('Observed/future Video and H/R/F geometry differs')
    groups = torch.tensor([0] * observed_tokens + [2] * (video_tokens - observed_tokens)
                          + [0] * history_steps + [1] * horizon + [3] * horizon,
                          device=force_valid.device, dtype=torch.long)
    valid = torch.cat((torch.ones(len(force_valid), video_tokens, device=force_valid.device,
                                 dtype=torch.bool), force_valid), 1)
    return groups, valid


def joint_attention_mask(video_tokens, observed_tokens, force_valid, history_steps=32, horizon=32):
    groups, valid = joint_token_layout(video_tokens, observed_tokens, force_valid, history_steps, horizon)
    return grouped_attention_mask(groups, valid)


def action_prefix_mask(prefix_valid, action_is_pad):
    """A query rows of the exact same full O/R/V/F/A attention matrix."""
    if (prefix_valid.dtype != torch.bool or action_is_pad.dtype != torch.bool
            or prefix_valid.shape[0] != action_is_pad.shape[0]):
        raise ValueError('Boolean prefix/action validity required')
    b, steps = action_is_pad.shape
    valid = ~action_is_pad
    keys = torch.cat((prefix_valid, valid), 1)
    allowed = valid[:, :, None] & keys[:, None, :]
    sentinel = torch.zeros((b, steps, keys.shape[1]), device=keys.device, dtype=torch.bool)
    sentinel[:, :, prefix_valid.shape[1]:] = torch.eye(steps, device=keys.device, dtype=torch.bool)
    return (allowed | ((~valid)[:, :, None] & sentinel))[:, None]


class JointForceBlock(nn.Module):
    def __init__(self, original, dim, width, heads):
        super().__init__()
        self.norm1, self.norm2, self.ff = original.norm1, original.norm2, original.ff
        self.self_attention = Attention(dim, dim, width, heads)

    def forward(self, force, video_q, video_k, video_v, mask, *, return_kv=False):
        attention = self.self_attention
        normalized = self.norm1(force)
        fq = attention.split(attention.q(normalized))
        fk, fv = attention.kv(normalized)
        q = torch.cat((video_q, fq), 2)
        k = torch.cat((video_k, fk), 2)
        v = torch.cat((video_v, fv), 2)
        mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        count = video_q.shape[2]
        force = force + attention.out(mixed[:, :, count:].transpose(1, 2).flatten(2))
        force = force + self.ff(self.norm2(force))
        result = mixed[:, :, :count], force
        return (*result, k, v) if return_kv else result


class JointChunkForceWM(ChunkForceWM):
    def __init__(self, *, video_hidden_dim=2048, hidden_dim=1280, num_heads=16, **kwargs):
        super().__init__(video_hidden_dim=video_hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads, **kwargs)
        with torch.random.fork_rng(devices=[]):
            self.blocks = nn.ModuleList([JointForceBlock(block, hidden_dim, video_hidden_dim, num_heads)
                                         for block in self.blocks])

    def forward(self, **kwargs):
        raise RuntimeError('Joint Force must execute together with Cosmos through _joint_video')


class JointCosmosForceWM(CosmosForceWM):
    force_type = JointChunkForceWM
    checkpoint_schema = 'fastwam-cosmos-orvfa-joint28-v2'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.dit['force'].num_heads != self.video.num_heads
                or self.video.hidden_dim != self.video.num_heads * self.video.attn_head_dim):
            raise ValueError('Joint attention requires matching native Cosmos head geometry')
        del self.dit['memories']
        self.manifest.update(schema='cosmos-orvfa-joint28-v2', joint_protocol=JOINT_PROTOCOL,
            attention_groups=['O', 'R', 'V', 'F', 'A'], allowed_reads=GROUP_READS,
            reference='explicit clean recorded Action32; future GPT reference uses the same input',
            memory='native per-layer WM prefix KV; differentiable clock0 refresh; no-grad rollout',
            action_gradients='AE loss reaches prefix projections and WMs; no AE state in WM forward',
            observed_clock='clean zero; observed/history tokens cannot read reference or predictions')
        self.dit['runtime'].bind_manifest(self.manifest)

    def _action_observation(self, first, raw):
        # O is already in the masked WM prefix; no duplicate observed pass.
        return ()

    def _joint_video(self, latents, clock, raw, context, cmask, sample, action, apad, *, collect=False):
        prepared = self.video.prepare(latents, clock.float() / 1000., raw,
            candidate_actions=self.dit['controls'](action.detach(), apad),
            fps=self.fps, context_is_projected=True, observed_time='clean_zero')
        force = self.dit['force']
        f, _, valid, wpad, first_w = force.prepare(
            history=sample['history'].to(self.device, self.torch_dtype),
            action_condition=action.detach(), context=context.detach(), context_mask=cmask,
            history_is_pad=sample['history_is_pad'].to(self.device, torch.bool), candidate_is_pad=apad,
            history_channel_valid=sample['history_channel_valid'].to(self.device, torch.bool),
            observation_latents=sample['first_frame_latents'].to(self.device, self.torch_dtype))
        x = prepared['tokens']
        groups, prefix_valid = joint_token_layout(x.shape[1], prepared['tokens_per_frame'],
            valid, force.history_steps, force.horizon)
        mask = grouped_attention_mask(groups, prefix_valid)
        caches, force_layers = [], []
        for i, block in enumerate(force.blocks):
            def run(video_tokens, force_tokens, i=i, block=block):
                q, k, v, state = self.video.block_qkv(i, video_tokens, prepared)
                outputs = block(force_tokens, q, k, v, mask, return_kv=collect)
                next_video = self.video.block_finish(i, outputs[0], state, prepared)
                result = (next_video, torch.where(valid[..., None], outputs[1], 0))
                return (*result, *outputs[2:])
            outputs = (checkpoint(run, x, f, use_reentrant=False)
                       if torch.is_grad_enabled() and self.gradient_checkpointing else run(x, f))
            x, f = outputs[:2]
            if collect:
                caches.append(outputs[2:])
                force_layers.append(f[:, first_w:])
        prediction = (ForcePrediction(force.head(f[:, first_w:]), tuple(force_layers), wpad) if collect else None)
        return self.video.finish(x, prepared), tuple(caches), prediction

    def _supervised_video(self, sample, noisy, clock, raw, context, cmask, action, apad):
        return self._joint_video(noisy, clock, raw, context, cmask, sample, action, apad)[0]

    @torch.no_grad()
    def generate_video_memory(self, first, raw_context, action_condition, action_is_pad, *, sample, context, cmask):
        shape = (*first.shape[:2], 3, *first.shape[3:])
        x = self.dit['runtime'].normal(shape).to(self.device, self.torch_dtype)
        x[:, :, :1] = first.detach()
        times, deltas = self.v_scheduler.build_inference_schedule(self.generated_video_steps, self.device, x.dtype)
        for t, dt in zip(times, deltas):
            velocity, _, _ = self._joint_video(x, t.expand(len(x)), raw_context,
                context, cmask, sample, action_condition, action_is_pad)
            x = self.v_scheduler.step(velocity, dt, x)
            x[:, :, :1] = first.detach()
        return x.detach()

    def _world_models(self, sample, first, raw, context, cmask, action, apad):
        x = self.generate_video_memory(first, raw.detach(), action, apad, sample=sample, context=context, cmask=cmask)
        zero = torch.zeros(len(x), device=self.device, dtype=self.torch_dtype)
        _, caches, force = self._joint_video(x, zero, raw, context, cmask, sample, action, apad, collect=True)
        # Availability comes only from input masks, never future target masks.
        from fastwam.forcewm.masks import mask_invalid_channels
        _, hv = mask_invalid_channels(sample['history'].to(self.device, self.torch_dtype),
            sample['history_is_pad'].to(self.device, torch.bool),
            sample['history_channel_valid'].to(self.device, torch.bool))
        video_count = caches[0][0].shape[2] - hv.shape[1] - 2 * force.wrench.shape[1]
        valid = torch.cat((torch.ones(len(x), video_count, device=self.device, dtype=torch.bool),
                           hv.any(-1), ~apad, ~force.is_pad), 1)
        return (x, caches, ~valid), force

    def action_velocity(self, action, clock, context, cmask, observation_kv, memory, force, apad):
        expert = self.dit['action']
        x, _, mod, text, text_mask, freq = expert.prepare(action, clock, context, cmask)
        allowed = action_prefix_mask(~memory[2], apad)
        for i, (pk, pv) in enumerate(memory[1]):
            block = expert.blocks[i]
            def run(tokens, prefix_k, prefix_v, block=block):
                q, k, v, res, gate, shift, scale, mg, _ = self.ops._build_expert_attention_io(
                    expert, block, tokens, freq, mod)
                split = lambda t: t.unflatten(-1, (expert.num_heads, self.video.attn_head_dim)).transpose(1, 2)
                mixed = F.scaled_dot_product_attention(split(q), torch.cat((prefix_k, split(k)), 2),
                    torch.cat((prefix_v, split(v)), 2), attn_mask=allowed, dropout_p=0.).transpose(1, 2).flatten(2)
                out = self.ops._apply_expert_post_block_tensor(block, res, mixed, gate, shift, scale, mg, text, text_mask)
                return torch.where((~apad)[..., None], out, 0)
            x = (checkpoint(run, x, pk, pv, use_reentrant=False)
                 if torch.is_grad_enabled() and self.gradient_checkpointing else run(x, pk, pv))
        return expert.post(x), force


def warm_start_joint(model, path, digest):
    """Explicit model-only v1 migration. Fresh optimizer/scheduler/runtime RNG."""
    from fastwam.cache.latent import file_sha256
    if file_sha256(path) != digest:
        raise ValueError('Warm-start checkpoint SHA256 differs')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (payload.get('schema') != 'fastwam-cosmos-force-joint-self-attention28-v1'
            or payload.get('step') != 5000):
        raise ValueError('Verified original joint step5000 required')
    original = payload['model']
    current = model.state_dict()
    runtime = 'dit.runtime.'
    retained = {k for k in current if not k.startswith(runtime)}
    missing = retained - original.keys()
    unexpected = {k for k in original if k not in retained
                  and not k.startswith(('dit.memories.', runtime))}
    if missing or unexpected:
        raise ValueError(f'Warm-start key mismatch: missing={missing}, extra={unexpected}')
    for key in retained:
        value = original[key]
        if value is None and current[key] is None:
            continue
        if (not isinstance(value, torch.Tensor) or not isinstance(current[key], torch.Tensor)
                or value.shape != current[key].shape
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Warm-start shape/nonfinite tensor: ' + key)
    receipt = dict(schema=payload['schema'], step=5000, sha256=digest,
        retained_tensors=len(retained), removed='dit.memories.*', optimizer='fresh',
        scheduler='new 0..30000', memory_rng='fresh seed44')
    model.manifest['warm_start'] = receipt
    model.manifest['initialization'] = 'model-only migration from SHA-bound joint v1 step5000'
    model.dit['runtime'].bind_manifest(model.manifest)
    state = model.state_dict()
    state.update({k: original[k] for k in retained})
    model.load_state_dict(state, strict=True)
    return receipt
