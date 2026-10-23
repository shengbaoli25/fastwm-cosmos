"""Independent V/F/A Euler grids with explicit, dependency-aware feature reuse.

Each branch samples its velocity at the start of its own interval and commits
the update at the end. States/clocks remain fixed within that interval. Force
features needed by Action are refreshed when their Video/Force inputs change;
those auxiliary evaluations never replace the stored Force integration slope.
This is a defined multirate sampler, not a claim of parity with equal-step grids.
"""
from math import lcm
import torch
from torch import nn
from torch.nn import functional as F

from .noaction import BRANCH_READS
from .noaction_inference import NoActionInferenceRunner, _action_rope


def expert_plan(video_steps, force_steps, action_steps):
    counts = tuple(int(x) for x in (video_steps, force_steps, action_steps))
    if min(counts) < 1:
        raise ValueError('Expert step counts must be positive')
    total = lcm(*counts)
    if total > 64:
        raise ValueError('Use commensurate grids with at most 64 merged ticks')
    ratios = tuple(total // x for x in counts)
    plan = []
    force_key = None
    for i in range(total):
        starts = tuple(i % ratio == 0 for ratio in ratios)
        ends = tuple((i + 1) % ratio == 0 for ratio in ratios)
        key = (i // ratios[0], i // ratios[1])
        force_features = starts[1] or (starts[2] and key != force_key)
        if force_features:
            force_key = key
        plan.append(dict(start=starts, end=ends, force_features=force_features))
    return counts, ratios, plan


class _ForceLayer(nn.Module):
    def __init__(self, block, heads, head_dim):
        super().__init__()
        self.block, self.heads, self.head_dim = block, heads, head_dim

    def split(self, x):
        return x.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)

    def forward(self, force, mods, history, video, mask):
        fb = self.block
        shift, scale, gate, ffshift, ffscale, ffgate = mods
        normalized = fb.norm1(force) * (1 + scale[:, None]) + shift[:, None]
        q = self.split(fb.attention.q(normalized))
        k, v = self.split(fb.attention.k(normalized)), self.split(fb.attention.v(normalized))
        shared_k = torch.cat((video[0], history[0], k), 2)
        shared_v = torch.cat((video[1], history[1], v), 2)
        mixed = F.scaled_dot_product_attention(q, shared_k, shared_v, attn_mask=mask)
        force = force + gate[:, None] * fb.attention.out(mixed.transpose(1, 2).flatten(2))
        force = force + ffgate[:, None] * fb.ff(fb.norm2(force) * (1 + ffscale[:, None]) + ffshift[:, None])
        return force, shared_k, shared_v


class _ActionLayer(nn.Module):
    def __init__(self, block, heads, head_dim):
        super().__init__()
        self.block, self.heads, self.head_dim = block, heads, head_dim

    def split(self, x):
        return x.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)

    def forward(self, action, mods, shared, text, positions, mask):
        ab = self.block
        shift, scale, gate, ffshift, ffscale, ffgate = mods
        normalized = ab.norm1(action) * (1 + scale) + shift
        cosine, sine = positions
        q = _action_rope(ab.self_attn.norm_q(ab.self_attn.q(normalized)), cosine, sine, self.heads)
        k = _action_rope(ab.self_attn.norm_k(ab.self_attn.k(normalized)), cosine, sine, self.heads)
        v = self.split(ab.self_attn.v(normalized))
        mixed = F.scaled_dot_product_attention(q, torch.cat((shared[0], k), 2),
            torch.cat((shared[1], v), 2), attn_mask=mask)
        action = action + gate * ab.self_attn.o(mixed.transpose(1, 2).flatten(2))
        q = self.split(ab.cross_attn.norm_q(ab.cross_attn.q(ab.norm3(action))))
        cross = F.scaled_dot_product_attention(q, text[0], text[1])
        action = action + ab.cross_attn.o(cross.transpose(1, 2).flatten(2))
        return action + ffgate * ab.ffn(ab.norm2(action) * (1 + ffscale) + ffshift)


class ExpertScheduleRunner(NoActionInferenceRunner):
    def __init__(self, model, *, video_steps=16, force_steps=16, action_steps=16,
                 compile_layers=True, cuda_graph=True, compile_threads=4):
        if model.manifest.get('branch_reads') != BRANCH_READS:
            raise ValueError('Verified one-way dependency graph required')
        self.counts, self.ratios, self.plan = expert_plan(video_steps, force_steps, action_steps)
        self.merged_steps = len(self.plan)
        super().__init__(model, compile_layers=False, cuda_graph=cuda_graph, compile_threads=compile_threads)
        if compile_layers:
            import torch._inductor.config as compiler_config
            compiler_config.compile_threads = max(1, int(compile_threads))
        def prepare(module):
            module.eval()
            return (torch.compile(module, fullgraph=True, dynamic=False,
                options={'triton.cudagraphs': False, 'emulate_precision_casts': True})
                if compile_layers else module)
        self.force_layers = [prepare(_ForceLayer(f, model.video.num_heads, model.video.attn_head_dim))
                             for f in model.dit['force'].blocks]
        self.action_layers = [prepare(_ActionLayer(a, model.video.num_heads, model.video.attn_head_dim))
                              for a in model.dit['action'].blocks]
        self.stats.update(compile_layers=compile_layers, sampler='independent_expert_block_hold',
            nominal_steps=dict(zip(('video', 'force', 'action'), self.counts)),
            merged_ticks=self.merged_steps, velocity_cache=False,
            video_forward_evaluations=self.counts[0], action_forward_evaluations=self.counts[2],
            force_sampling_evaluations=self.counts[1],
            force_forward_evaluations=sum(p['force_features'] for p in self.plan))

    def _build_schedules(self, steps):
        if steps != self.merged_steps:
            raise ValueError('Merged step count is fixed by the three expert grids')
        schedules = []
        for scheduler, count, ratio in zip(
                (self.model.v_scheduler, self.model.f_scheduler, self.model.a_scheduler), self.counts, self.ratios):
            times, deltas = scheduler.build_inference_schedule(count, self.model.device, self.model.torch_dtype)
            expanded = torch.zeros(steps, device=self.model.device, dtype=self.model.torch_dtype)
            expanded[ratio - 1::ratio] = deltas
            schedules.append((times.repeat_interleave(ratio), expanded))
        return schedules

    def infer_joint(self, sample, *, num_inference_steps=None, **kwargs):
        if num_inference_steps is not None and int(num_inference_steps) != self.merged_steps:
            raise ValueError('Pass the merged expert grid size or omit num_inference_steps')
        return super().infer_joint(sample, num_inference_steps=self.merged_steps, **kwargs)

    def _rollout(self, video, force, action, first, static, texts, ftext, masks):
        m, fe, ae = self.model, self.model.dit['force'], self.model.dit['action']
        video_kv, shared_kv = None, None
        pv, pf, pa = None, None, None
        for tick, ((prepared, mods, deltas), plan) in enumerate(zip(self.schedule, self.plan)):
            sv, sf, sa = plan['start']
            evaluate_force = plan['force_features']
            if sv:
                vt = m.video.net.x_embedder(torch.cat(
                    (video[:, :, 1:].to(first.dtype), self.pad_channels), 1))
                video_kv = []
            if evaluate_force:
                ft = fe.force_encoder(force) + fe.position[fe.history_steps:] + ftext
                shared_kv = []
            if sa:
                at = ae.action_encoder(action)
            for index, (layer, mod, kv, text) in enumerate(zip(self.layers, mods, static, texts)):
                if sv:
                    vt, vk, vv = layer.video_forward(vt, mod[0], kv, text[0], self.positions)
                    video_kv.append((vk, vv))
                if evaluate_force:
                    ft, sk, svalue = self.force_layers[index](ft, mod[1], kv[2:], video_kv[index], masks[0])
                    shared_kv.append((sk, svalue))
                if sa:
                    at = self.action_layers[index](at, mod[2], shared_kv[index], text[1],
                                                   self.positions[2:], masks[1])
            if sv:
                pv = m.video.finish(vt.flatten(1, 3), prepared)
            if sf:
                pf = fe.head(ft)
            if sa:
                pa = ae.head(at)
            ev, ef, ea = plan['end']
            if ev:
                video = torch.cat((first, video[:, :, 1:] + pv * deltas[0]), 2)
            if ef:
                force = force + pf * deltas[1]
            if ea:
                action = action + pa * deltas[2]
        return video, force, action


@torch.no_grad()
def infer_expert_reference(model, sample, *, video_steps=16, force_steps=16, action_steps=16,
                           seed=42, video_seed=44, force_seed=43):
    """Slow reference recomputes full joint core on every merged tick."""
    counts, ratios, plan = expert_plan(video_steps, force_steps, action_steps)
    first = sample['first_frame_latents'].to(model.device, model.torch_dtype)
    conditions = model._conditions(sample)
    static = model.prefill_static(first, conditions)
    schedules = [s.build_inference_schedule(n, model.device, model.torch_dtype)
                 for s, n in zip((model.v_scheduler, model.f_scheduler, model.a_scheduler), counts)]
    def noise(shape, value):
        return torch.randn(shape, generator=torch.Generator(device='cpu').manual_seed(value)).to(model.device, model.torch_dtype)
    video = noise((*first.shape[:2], 3, *first.shape[3:]), video_seed)
    video[:, :, :1] = first
    force = noise((len(first), 32, 6), force_seed)
    action = noise((len(first), 32, model.action_dim), seed)
    slopes = [None, None, None]
    for tick, entry in enumerate(plan):
        times = [schedules[j][0][tick // ratios[j]].expand(len(first)) for j in range(3)]
        predicted = model.joint_denoise_core(video, force, action, *times, conditions, static)
        for j, start in enumerate(entry['start']):
            if start:
                slopes[j] = predicted[j]
        states = [video, force, action]
        for j, end in enumerate(entry['end']):
            if end:
                states[j] = states[j] + slopes[j] * schedules[j][1][tick // ratios[j]]
        video, force, action = states
        video[:, :, :1] = first
    return dict(video_latents=video, force=force, action=action)
