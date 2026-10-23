"""No-reference V/F/A flow matching, with one layer core for training and inference.

Video reads O/V; Force reads O/H/V/F; Action reads O/H/V/F/A.
All clocks are 0..1000, with 0 clean. There is no future action condition.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .model import CosmosForceWM, _ActionOps, temporal_mse
from .expert import CosmosVideoExpert
from .state import ForceRuntimeState
from fastwam.forcewm.diffusion import ForceDiffusionExpert
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.wan_video_dit import sinusoidal_embedding_1d, modulate
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

SCHEMA = 'cosmos-noaction-vfa-diffusion28-joint8-v2'
QUERY_POSTTRAIN_SCHEMA = 'cosmos-noaction-vfa-diffusion28-joint8-query-posttrain-v1'
BRANCH_READS = ((1, 0, 0), (1, 1, 0), (1, 1, 1))


def frame_slice(prepared, start, end):
    """Keep absolute RoPE while slicing frame-local modulation and token grids."""
    count = prepared['tokens_per_frame']
    state = dict(prepared['state'])
    state['time'] = state['time'][:, start:end]
    for name in ('adaln', 'extra'):
        if state[name] is not None:
            state[name] = state[name][:, start:end]
    state['rope'] = state['rope'][start * count:end * count]
    return {**prepared, 'tokens': prepared['tokens'][:, start * count:end * count],
            'grid': (end - start, *prepared['grid'][1:]), 'state': state}


def attend(q, k, v, valid=None, bias=None):
    if bias is not None:
        mask = bias.to(dtype=q.dtype)
        if valid is not None:
            mask = mask.masked_fill(~valid[:, None, None, :], -torch.inf)
    else:
        mask = None if valid is None else valid[:, None, None, :]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)


class ActionConditionedVisualQueries(nn.Module):
    """Action-conditioned distributions over current-frame visual keys."""
    def __init__(self, *, video_dim, action_dim, query_dim=256, num_queries=4,
                 temperature=0.07, diversity_margin=0.7, route_scale_init=0.05):
        super().__init__()
        if num_queries < 2 or query_dim < 8 or temperature <= 0:
            raise ValueError('At least two queries, positive query width and temperature required')
        self.num_queries = int(num_queries)
        self.temperature = float(temperature)
        self.diversity_margin = float(diversity_margin)
        self.visual_norm = nn.LayerNorm(query_dim)
        self.action_norm = nn.LayerNorm(query_dim)
        self.visual_proj = nn.Linear(video_dim, query_dim, bias=False)
        self.action_proj = nn.Linear(action_dim, query_dim, bias=False)
        self.query_norm = nn.LayerNorm(query_dim)
        self.queries = nn.Parameter(torch.randn(num_queries, query_dim) / query_dim ** 0.5)
        self.route_scale = nn.Parameter(torch.tensor(float(route_scale_init)))

    def forward(self, observed_keys, action_tokens, action_valid):
        if (observed_keys.ndim != 3 or action_tokens.ndim != 3
                or action_valid.shape != action_tokens.shape[:2]
                or action_valid.dtype != torch.bool):
            raise ValueError('Visual keys [B,N,D], action tokens [B,T,D], and boolean validity required')
        visual = F.normalize(self.visual_norm(self.visual_proj(observed_keys).float()), dim=-1)
        action = self.action_norm(self.action_proj(action_tokens.float()))
        query = self.query_norm(self.queries.float()[None, None] + action[:, :, None])
        query = F.normalize(query, dim=-1)
        logits = torch.einsum('btqd,bnd->btqn', query, visual) / self.temperature
        maps = logits.softmax(-1)

        # A soft pairwise cosine ceiling avoids forcing unrelated queries to be
        # orthogonal while still penalizing near-duplicate spatial maps.
        qmap = F.normalize(maps.float(), dim=-1)
        similarity = qmap @ qmap.transpose(-1, -2)
        pairs = torch.triu(torch.ones(self.num_queries, self.num_queries,
                                      device=maps.device, dtype=torch.bool), diagonal=1)
        penalty = F.relu(similarity[..., pairs] - self.diversity_margin).square().mean(-1)
        valid = action_valid.float()
        diversity = (penalty * valid).sum() / valid.sum().clamp_min(1)

        # At route_scale=0 the prior is exactly neutral; a small positive
        # initialization gives the task loss a gradient into the queries.
        n = observed_keys.shape[1]
        mixture = maps.mean(2)
        prior = (1. + self.route_scale.tanh() * (n * mixture - 1.)).clamp_min(1e-6)
        observed_bias = prior.log()
        return maps, observed_bias, diversity


class NoActionJointModel(CosmosForceWM):
    checkpoint_schema = SCHEMA
    reference_mode = 'none'

    def __init__(self, net, *, manifest, text_bank=None, action_hidden=768, action_ffn=3072,
                 force_hidden=768, force_ffn=2048, force_heads=16, action_seed=42,
                 force_seed=43, memory_seed=44, num_inference_steps=16,
                 force_shift=1., gradient_checkpointing=True, fps=7.5, action_space='joint8',
                 visual_query_posttrain=False, query_count=4, query_dim=256,
                 query_temperature=0.07, query_diversity_weight=0.01,
                 query_diversity_margin=0.7, query_route_scale_init=0.05):
        nn.Module.__init__(self)
        self.gradient_checkpointing = gradient_checkpointing
        self.num_inference_steps, self.fps = int(num_inference_steps), float(fps)
        if action_space not in ('joint8', 'eef_pose7'):
            raise ValueError('Explicit joint8 or legacy eef_pose7 action space required')
        self.action_space = action_space
        self.action_dim = self.state_dim = 8 if action_space == 'joint8' else 7
        if self.num_inference_steps < 1:
            raise ValueError('Positive joint sampling steps required')
        video = CosmosVideoExpert(net)
        if force_heads != video.num_heads:
            raise ValueError('Force and Video attention head geometry must match')
        self.depth = video.num_layers
        text_dim = net.blocks[0].cross_attn.k_proj.in_features
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(action_seed)
            action = ActionDiT(hidden_dim=action_hidden, action_dim=self.action_dim, ffn_dim=action_ffn,
                text_dim=text_dim, freq_dim=256, eps=1e-6, num_heads=video.num_heads,
                attn_head_dim=video.attn_head_dim, num_layers=self.depth)
            self.proprio_encoder = nn.Linear(self.state_dim + 1, text_dim)
            source_time = nn.Sequential(nn.Linear(512, action_hidden), nn.SiLU(), nn.Linear(action_hidden, 6 * action_hidden))
            torch.manual_seed(force_seed)
            force = ForceDiffusionExpert(hidden_dim=force_hidden, ffn_dim=force_ffn,
                num_layers=self.depth, width=video.hidden_dim, num_heads=force_heads, text_dim=text_dim,
                state_dim=self.state_dim)
        self.visual_query_posttrain = bool(visual_query_posttrain)
        self.query_diversity_weight = float(query_diversity_weight)
        if self.query_diversity_weight < 0:
            raise ValueError('Query diversity weight must be non-negative')
        self.manifest = dict(manifest, schema=SCHEMA, joint_protocol=SCHEMA, reference_mode='none',
            branch_reads=BRANCH_READS, loss='video/action/force FM, sigma0 clean, target noise-clean',
            clocks=dict(video_shift=5., action_shift=1., force_shift=float(force_shift)),
            architecture=dict(depth=self.depth, action_hidden=action_hidden, action_ffn=action_ffn,
                force_hidden=force_hidden, force_ffn=force_ffn, heads=video.num_heads, head_dim=video.attn_head_dim),
            initialization='official non-action-conditioned Video2World; fresh AE/Force',
            num_inference_steps=self.num_inference_steps)
        self.manifest.update(action_space=action_space, action_dim=self.action_dim,
            state_dim=self.state_dim, history_dim=self.state_dim + 6,
            action_units='7 absolute joint radians + gripper openness' if action_space == 'joint8' else 'xyz metres + Euler radians + gripper openness')
        modules = dict(video=video, action=action, force=force, action_source_time=source_time,
                       runtime=ForceRuntimeState(memory_seed, self.manifest))
        if self.visual_query_posttrain:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(46)
                modules['visual_queries'] = ActionConditionedVisualQueries(
                    video_dim=video.hidden_dim, action_dim=action_hidden, query_dim=query_dim,
                    num_queries=query_count, temperature=query_temperature,
                    diversity_margin=query_diversity_margin, route_scale_init=query_route_scale_init)
            self.checkpoint_schema = QUERY_POSTTRAIN_SCHEMA
            self.manifest.update(schema=QUERY_POSTTRAIN_SCHEMA, posttrain_method='action-conditioned visual query routing',
                visual_queries=dict(count=query_count, dim=query_dim, temperature=query_temperature,
                    diversity_loss='off-diagonal spatial-map cosine hinge',
                    diversity_margin=query_diversity_margin, diversity_weight=self.query_diversity_weight,
                    initial_route_scale=query_route_scale_init,
                    route='same-layer Action tokens to observed-frame O keys; additive bias on Force reads only'),
                trainable_policy='Force/Action/visual_queries; frozen Video and proprio encoder',
                warm_start='explicit model-only step20000; fresh optimizer/scheduler')
            modules['runtime'].bind_manifest(self.manifest)
        self.dit = nn.ModuleDict(modules)
        self.register_buffer('text_bank', text_bank, persistent=False)
        self.v_scheduler = WanContinuousFlowMatchScheduler(shift=5.)
        self.a_scheduler = WanContinuousFlowMatchScheduler(shift=1.)
        self.f_scheduler = WanContinuousFlowMatchScheduler(shift=force_shift)
        self.compile_training_denoise = False
        self.ops = _ActionOps()

    def configure_training_parameters(self):
        """Keep the learned interaction path trainable without moving Video."""
        if not self.visual_query_posttrain:
            return False
        self.eval()
        self.requires_grad_(False)
        for name in ('action', 'force', 'action_source_time', 'visual_queries'):
            module = self.dit[name]
            module.train()
            module.requires_grad_(True)
        return True

    def _conditions(self, sample):
        if any(k in sample for k in ('action_condition', 'candidate_actions', 'reference', 'action_reference')):
            raise ValueError('This schema supports no future action/reference conditions')
        raw, context, cmask = self._contexts(sample)
        force = self.dit['force']
        history, hvalid = force.prepare_history(sample['history'].to(self.device, self.torch_dtype),
            sample['history_is_pad'].to(self.device, torch.bool),
            sample['history_channel_valid'].to(self.device, torch.bool), context)
        return raw, context, cmask, history, hvalid

    def _contexts(self, sample):
        # The native text projection is unchanged; state width is an explicit contract.
        if 'cosmos_context' in sample:
            raw = sample['cosmos_context'].to(self.device, self.torch_dtype)
        else:
            if self.text_bank is None:
                raise ValueError('Verified Cosmos Reason1 text bank required')
            ids = sample['cosmos_text_index'].to(self.device, torch.long).reshape(-1)
            if bool((ids < 0).any() | (ids >= len(self.text_bank)).any()):
                raise ValueError('Text index outside pinned bank')
            raw = self.text_bank[ids[0]].unsqueeze(0) if bool((ids == ids[0]).all()) else self.text_bank[ids]
        text = self.video.net.crossattn_proj(raw) if self.video.net.use_crossattn_projection else raw
        proprio = sample['proprio'].to(self.device, self.torch_dtype)
        text = text.expand(proprio.shape[0], -1, -1)
        if proprio.ndim == 3:
            proprio = proprio[:, 0]
        if proprio.shape != (text.shape[0], self.state_dim + 1):
            raise ValueError('Current proprio width differs from the action-space contract')
        context = torch.cat((text, self.proprio_encoder(proprio)[:, None]), 1)
        return text, context, torch.ones(context.shape[:2], device=self.device, dtype=torch.bool)

    def _video_prepare(self, video, tv, raw):
        return self.video.prepare(video, tv.float() / 1000, raw, fps=self.fps,
            observed_time='clean_zero', context_is_projected=True, validate_values=False)

    def _split(self, tensor):
        return tensor.unflatten(-1, (self.video.num_heads, self.video.attn_head_dim)).transpose(1, 2)

    def _static_layer(self, i, observed, history, prepared, hvalid):
        oq, ok, ov, os = self.video.block_qkv(i, observed, prepared)
        block = self.dit['force'].blocks[i]
        hq, hk, hv, hs = block.qkv(history)
        observed = self.video.block_finish(i, attend(oq, ok, ov), os, prepared)
        valid = torch.cat((torch.ones(len(history), ok.shape[2], device=history.device, dtype=torch.bool), hvalid), 1)
        history = block(attend(hq, torch.cat((ok, hk), 2), torch.cat((ov, hv), 2), valid), hs)
        history = torch.where(hvalid[..., None], history, 0)
        return observed, history, (ok, ov, hk, hv)

    def _dynamic_layer(self, i, video, force, action, prepared, fm, am, text, text_mask, freq, hvalid, static,
                       action_valid=None, text_cache=None):
        ok, ov, hk, hv = static
        vq, vk, vv, vs = self.video.block_qkv(i, video, prepared)
        block = self.dit['force'].blocks[i]
        fq, fk, fv, fs = block.qkv(force, fm)
        ae = self.dit['action']; ab = ae.blocks[i]
        aq, ak, av, residual, gate, shift, scale, mg, _ = self.ops._build_expert_attention_io(ae, ab, action, freq, am)
        aq, ak, av = (self._split(x) for x in (aq, ak, av))
        # Every Q/K/V above comes from the same layer input, before any update.
        video = self.video.block_finish(i, attend(vq, torch.cat((ok, vk), 2), torch.cat((ov, vv), 2)), vs, prepared, None if text_cache is None else text_cache[0])
        k, v = torch.cat((ok, vk, hk, fk), 2), torch.cat((ov, vv, hv, fv), 2)
        valid = torch.cat((torch.ones(len(force), ok.shape[2] + vk.shape[2], device=force.device, dtype=torch.bool),
                           hvalid, torch.ones(len(force), fk.shape[2], device=force.device, dtype=torch.bool)), 1)
        selector = self.dit['visual_queries'] if self.visual_query_posttrain else None
        if selector is None:
            force = block(attend(fq, k, v, valid), fs)
            diversity = None
        else:
            if action_valid is None:
                raise ValueError('Action validity is required by visual query routing')
            observed_keys = ok.transpose(1, 2).flatten(2)
            maps, observed_bias, diversity = selector(observed_keys, action, action_valid)
            rest = torch.zeros((*observed_bias.shape[:2], k.shape[2] - observed_bias.shape[-1]),
                               device=observed_bias.device, dtype=observed_bias.dtype)
            bias = torch.cat((observed_bias, rest), -1)[:, None]
            force = block(attend(fq, k, v, valid, bias), fs)
        av_valid = torch.cat((valid, torch.ones(len(action), ak.shape[2], device=action.device, dtype=torch.bool)), 1)
        mixed = attend(aq, torch.cat((k, ak), 2), torch.cat((v, av), 2), av_valid).transpose(1, 2).flatten(2)
        if text_cache is None:
            action = self.ops._apply_expert_post_block_tensor(ab, residual, mixed, gate, shift, scale, mg, text, text_mask)
        else:
            action = ab.gate(residual, gate, ab.self_attn.o(mixed))
            q = self._split(ab.cross_attn.norm_q(ab.cross_attn.q(ab.norm3(action))))
            k, v = text_cache[1]
            cross = F.scaled_dot_product_attention(q, k, v, attn_mask=text_mask[:, None], dropout_p=0.)
            action = action + ab.cross_attn.o(cross.transpose(1, 2).flatten(2))
            action = ab.gate(action, mg, ab.ffn(modulate(ab.norm2(action), shift, scale)))
        return (video, force, action, diversity) if selector is not None else (video, force, action)

    def prefill_static(self, first, conditions):
        """Request-local O/H cache. Used only for inference, with fixed weights."""
        if torch.is_grad_enabled():
            raise ValueError('Static inference cache must not detach training gradients')
        raw, context, cmask, history, hvalid = conditions
        zero = torch.zeros(len(first), device=first.device, dtype=first.dtype)
        prepared = self._video_prepare(first, zero, raw)
        observed = prepared['tokens']; caches = []
        for i in range(self.depth):
            observed, history, kv = self._static_layer(i, observed, history, prepared, hvalid)
            caches.append(kv)
        texts = []
        action_text = self.dit['action'].text_embedding(context)
        for vb, ab in zip(self.video.blocks, self.dit['action'].blocks):
            va = vb.cross_attn; aa = ab.cross_attn
            split = lambda x: x.unflatten(-1, (va.n_heads, va.head_dim))
            vkv = (va.k_norm(split(va.k_proj(raw))), va.v_norm(split(va.v_proj(raw))))
            akv = (self._split(aa.norm_k(aa.k(action_text))), self._split(aa.v(action_text)))
            texts.append((vkv, akv))
        return dict(layers=tuple(caches), text=tuple(texts))

    def joint_denoise_core(self, video, force, action, tv, tf, ta, conditions, static_cache=None,
                           action_valid=None):
        raw, context, cmask, history, hvalid = conditions
        full = self._video_prepare(video, tv, raw)
        op = frame_slice(full, 0, 1); vp = frame_slice(full, 1, full['grid'][0])
        observed, vt = op['tokens'], vp['tokens']
        ft, fm = self.dit['force'].prepare_future(force, tf, tv, context)
        at, _, am, text, text_mask, freq = self.dit['action'].prepare(action, ta, context, cmask)
        source_time = torch.cat((sinusoidal_embedding_1d(256, tv), sinusoidal_embedding_1d(256, tf)), -1)
        am = am + self.dit['action_source_time'](source_time).unflatten(-1, (6, self.dit['action'].hidden_dim))
        if static_cache is not None and (torch.is_grad_enabled() or len(static_cache["layers"]) != self.depth):
            raise ValueError('Request-local inference cache with matching depth required')
        query_diversities = []
        if self.visual_query_posttrain:
            if action_valid is None:
                action_valid = torch.ones(action.shape[:2], device=action.device, dtype=torch.bool)
            if action_valid.shape != action.shape[:2] or action_valid.dtype != torch.bool:
                raise ValueError('Action validity must match the action token grid')
        for i in range(self.depth):
            def layer(o, h, v, f, a, i=i):
                if static_cache is None:
                    o, h, kv = self._static_layer(i, o, h, op, hvalid)
                else:
                    kv = static_cache["layers"][i]
                result = self._dynamic_layer(i, v, f, a, vp, fm, am, text, text_mask, freq, hvalid, kv,
                    action_valid=action_valid,
                    text_cache=None if static_cache is None else static_cache["text"][i])
                if self.visual_query_posttrain:
                    v, f, a, diversity = result
                    return o, h, v, f, a, diversity
                v, f, a = result
                return o, h, v, f, a
            args = (observed, history, vt, ft, at)
            result = (checkpoint(layer, *args, use_reentrant=False)
                if self.gradient_checkpointing and torch.is_grad_enabled() else layer(*args))
            if self.visual_query_posttrain:
                observed, history, vt, ft, at, diversity = result
                query_diversities.append(diversity)
            else:
                observed, history, vt, ft, at = result
        future_velocity = self.video.finish(vt, vp)
        vv = torch.cat((torch.zeros_like(video[:, :, :1]), future_velocity), 2)
        outputs = (vv, self.dit['force'].head(ft), self.dit['action'].post(at))
        return (*outputs, torch.stack(query_diversities).mean()) if self.visual_query_posttrain else outputs

    def training_loss(self, sample):
        clean = sample['video_latents'].to(self.device, self.torch_dtype)
        first = sample['first_frame_latents'].to(self.device, self.torch_dtype)
        action = sample['action'].to(self.device, self.torch_dtype)
        force = sample['wrench_future'].to(self.device, self.torch_dtype)
        if clean.shape[1:3] != (16, 3) or first.shape != (*clean.shape[:2], 1, *clean.shape[3:]) or action.shape[1:] != (32, self.action_dim) or force.shape[1:] != (32, 6):
            raise ValueError(f'Expected Video16x3, independent first frame, Action32x{self.action_dim}, Force32x6')
        conditions = self._conditions(sample)
        tv, tf, ta = (s.sample_training_t(len(clean), self.device, self.torch_dtype)
                      for s in (self.v_scheduler, self.f_scheduler, self.a_scheduler))
        vn, fn, an = torch.randn_like(clean), torch.randn_like(force), torch.randn_like(action)
        fvalid = sample['wrench_valid'].to(self.device, torch.bool) & ~sample['force_is_pad'].to(self.device, torch.bool)[..., None]
        # Missing target channels are neutral; their values never enter loss/input.
        force = torch.where(fvalid, force, 0)
        v = self.v_scheduler.add_noise(clean, vn, tv); v[:, :, :1] = first
        f = self.f_scheduler.add_noise(force, fn, tf)
        a = self.a_scheduler.add_noise(action, an, ta)
        denoised = self.joint_denoise_core(v, f, a, tv, tf, ta, conditions,
            action_valid=~sample['action_is_pad'].to(self.device, torch.bool))
        if self.visual_query_posttrain:
            pv, pf, pa, query_diversity = denoised
        else:
            pv, pf, pa = denoised
        imagepad = sample['image_is_pad'].to(self.device, torch.bool)
        vvalid = ~imagepad[:, 1:].reshape(len(clean), 2, 4).all(-1)
        lv = temporal_mse(pv[:, :, 1:].transpose(1, 2).flatten(2),
            self.v_scheduler.training_target(clean, vn, tv)[:, :, 1:].transpose(1, 2).flatten(2), vvalid, self.v_scheduler, tv)
        la = temporal_mse(pa, self.a_scheduler.training_target(action, an, ta),
            ~sample['action_is_pad'].to(self.device, torch.bool), self.a_scheduler, ta)
        delta = torch.where(fvalid, pf.float() - self.f_scheduler.training_target(force, fn, tf).float(), 0)
        lf = ((delta.square().sum((1, 2)) / fvalid.sum((1, 2)).clamp_min(1)) * self.f_scheduler.training_weight(tf)).mean()
        components = dict(loss_video_fm=float(lv.detach()), loss_force_fm=float(lf.detach()),
                          loss_action_fm=float(la.detach()))
        if self.visual_query_posttrain:
            lq = query_diversity * self.query_diversity_weight
            components['loss_query_diversity'] = float(lq.detach())
            components['query_diversity_raw'] = float(query_diversity.detach())
            return lv + lf + la + lq, components
        return lv + lf + la, components

    @torch.no_grad()
    def infer_joint(self, sample, *, num_inference_steps=None, seed=42, video_seed=44, force_seed=43, use_cache=True):
        steps = self.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        if steps < 1:
            raise ValueError('Positive joint steps required')
        prior_modes = {m: m.training for m in self.modules()}
        self.eval()
        try:
            first = sample['first_frame_latents'].to(self.device, self.torch_dtype)
            conditions = self._conditions(sample)
            def noise(shape, seed):
                return torch.randn(shape, generator=torch.Generator(device='cpu').manual_seed(seed)).to(self.device, self.torch_dtype)
            v = noise((*first.shape[:2], 3, *first.shape[3:]), video_seed); v[:, :, :1] = first
            f = noise((len(first), 32, 6), force_seed)
            a = noise((len(first), 32, self.action_dim), seed)
            static = self.prefill_static(first, conditions) if use_cache else None
            action_valid = (~sample['action_is_pad'].to(self.device, torch.bool)
                            if 'action_is_pad' in sample else
                            torch.ones((len(first), 32), device=self.device, dtype=torch.bool))
            schedules = [s.build_inference_schedule(steps, self.device, self.torch_dtype) for s in (self.v_scheduler, self.f_scheduler, self.a_scheduler)]
            for i in range(steps):
                tv, tf, ta = (pair[0][i].expand(len(first)) for pair in schedules)
                denoised = self.joint_denoise_core(v, f, a, tv, tf, ta, conditions, static,
                    action_valid=action_valid)
                if self.visual_query_posttrain:
                    pv, pf, pa, _ = denoised
                else:
                    pv, pf, pa = denoised
                v = self.v_scheduler.step(pv, schedules[0][1][i], v); v[:, :, :1] = first
                f = self.f_scheduler.step(pf, schedules[1][1][i], f)
                a = self.a_scheduler.step(pa, schedules[2][1][i], a)
            return dict(action=a, force=f, video_latents=v)
        finally:
            for module, mode in prior_modes.items():
                module.training = mode

    def infer_forcewm(self, sample, *, num_inference_steps=None, seed=42, video_seed=44, force_seed=43, use_cache=True):
        return self.infer_joint(sample, num_inference_steps=num_inference_steps, seed=seed,
            video_seed=video_seed, force_seed=force_seed, use_cache=use_cache)

    def infer_world_model(self, sample, **kwargs):
        result = self.infer_joint(sample, **kwargs)
        return {k: v for k, v in result.items() if k != 'action'}

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = dict(schema=self.checkpoint_schema, manifest=self.manifest, model=self.state_dict(), step=step,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
        if optimizer is not None:
            payload['optimizer'] = optimizer.state_dict()
        path = Path(path); temp = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, temp); temp.replace(path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if payload.get('schema') != self.checkpoint_schema or payload.get('manifest') != self.manifest:
            raise ValueError('Checkpoint architecture/assets/protocol differ')
        self.load_state_dict(payload['model'], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(payload['optimizer'])
            torch.set_rng_state(payload['torch_rng'])
            if payload['cuda_rng']:
                torch.cuda.set_rng_state_all(payload['cuda_rng'])
        return payload.get('step')


def warm_start_visual_query_posttrain(model, path, digest, *, expected_step=20000):
    """Strict model-only migration from a verified baseline V/F/A checkpoint."""
    from fastwam.cache.latent import file_sha256
    if not model.visual_query_posttrain or model.checkpoint_schema != QUERY_POSTTRAIN_SCHEMA:
        raise ValueError('Visual-query post-training model required')
    if file_sha256(path) != digest:
        raise ValueError('Warm-start checkpoint SHA256 differs')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (payload.get('schema') != SCHEMA or payload.get('step') != expected_step
            or 'model' not in payload or 'manifest' not in payload):
        raise ValueError('Verified baseline step-20000 model checkpoint required')
    old_manifest = payload['manifest']
    for key in ('upstream', 'video_sha256', 'dataset_identity', 'normalization_sha256',
                'text_identity', 'action_space', 'action_dim', 'state_dim', 'history_dim'):
        if old_manifest.get(key) != model.manifest.get(key):
            raise ValueError('Warm-start provenance differs: ' + key)
    for key in ('joint_protocol', 'reference_mode', 'branch_reads', 'architecture', 'clocks',
                'action_units', 'num_inference_steps'):
        if old_manifest.get(key) != model.manifest.get(key):
            raise ValueError('Warm-start behavior differs: ' + key)

    original, current = payload['model'], model.state_dict()
    new_prefixes = ('dit.visual_queries.',)
    runtime_prefix = 'dit.runtime.'
    retained = {key for key in current
                if not key.startswith(new_prefixes) and not key.startswith(runtime_prefix)}
    missing = retained - original.keys()
    unexpected = {key for key in original
                  if key not in retained and not key.startswith(runtime_prefix)}
    if missing or unexpected:
        raise ValueError(f'Warm-start key mismatch: missing={sorted(missing)}, extra={sorted(unexpected)}')
    for key in retained:
        source, target = original[key], current[key]
        if (not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor)
                or source.shape != target.shape or not bool(torch.isfinite(source).all())):
            raise ValueError('Warm-start shape/nonfinite tensor: ' + key)

    receipt = dict(source_schema=SCHEMA, source_step=expected_step, source_sha256=digest,
        retained_tensors=len(retained), new_parameters='dit.visual_queries.*',
        runtime='fresh; rebound to post-train manifest', optimizer='fresh AdamW',
        scheduler='fresh 5000-step cosine with 250-step warmup')
    model.manifest['warm_start'] = receipt
    model.dit['runtime'].bind_manifest(model.manifest)
    state = model.state_dict()
    state.update({key: original[key] for key in retained})
    model.load_state_dict(state, strict=True)
    return receipt
