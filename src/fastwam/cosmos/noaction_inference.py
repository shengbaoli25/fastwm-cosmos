"""Inference-only acceleration for the no-reference V/F/A joint sampler.

Reuses the existing model parameters. Training and the reference sampler remain
unchanged. Captured inputs are copied on every request; only geometry, fixed
schedules and versioned text/weights may persist across requests.
"""
from __future__ import annotations

import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils._pytree import tree_flatten, tree_unflatten

from .noaction import SCHEMA, frame_slice
from fastwam.models.wan22.wan_video_dit import sinusoidal_embedding_1d


def _video_rope(x, cosine, sine):
    value = x.float()
    half = value.shape[-1] // 2
    rotated = torch.cat((-value[..., half:], value[..., :half]), -1)
    return (value * cosine + rotated * sine).to(x.dtype)


def _action_rope(x, cosine, sine, heads):
    value = x.unflatten(-1, (heads, -1)).float().unflatten(-1, (-1, 2))
    real, imag = value[..., 0], value[..., 1]
    result = torch.stack((real * cosine - imag * sine, real * sine + imag * cosine), -1)
    return result.flatten(-2).transpose(1, 2).to(x.dtype)


class _ControlLayer(nn.Module):
    """Compiled control math consumes K/V from the original video layer input."""
    def __init__(self, force, action, heads, head_dim):
        super().__init__()
        self.force, self.action = force, action
        self.heads, self.head_dim = heads, head_dim

    def split(self, x):
        return x.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)

    def forward(self, force, action, mods, static, text, positions, masks, video_k, video_v):
        fb, ab = self.force, self.action
        fmods, amods = mods
        hk, hv = static
        atext_k, atext_v = text
        acos, asin = positions
        fmask, amask = masks
        shift, scale, fg, fs, fc, fmg = fmods
        normalized_f = fb.norm1(force) * (1 + scale[:, None]) + shift[:, None]
        fq = self.split(fb.attention.q(normalized_f))
        fk, fv = self.split(fb.attention.k(normalized_f)), self.split(fb.attention.v(normalized_f))

        shift, scale, ag, ashift, ascale, amg = amods
        normalized_a = ab.norm1(action) * (1 + scale) + shift
        aq = _action_rope(ab.self_attn.norm_q(ab.self_attn.q(normalized_a)), acos, asin, self.heads)
        ak = _action_rope(ab.self_attn.norm_k(ab.self_attn.k(normalized_a)), acos, asin, self.heads)
        av = self.split(ab.self_attn.v(normalized_a))

        shared_k = torch.cat((video_k, hk, fk), 2)
        shared_v = torch.cat((video_v, hv, fv), 2)
        mixed_f = F.scaled_dot_product_attention(fq, shared_k, shared_v, attn_mask=fmask)
        force = force + fg[:, None] * fb.attention.out(mixed_f.transpose(1, 2).flatten(2))
        force = force + fmg[:, None] * fb.ff(fb.norm2(force) * (1 + fc[:, None]) + fs[:, None])

        mixed_a = F.scaled_dot_product_attention(aq, torch.cat((shared_k, ak), 2),
                                                torch.cat((shared_v, av), 2), attn_mask=amask)
        action = action + ag * ab.self_attn.o(mixed_a.transpose(1, 2).flatten(2))
        q = self.split(ab.cross_attn.norm_q(ab.cross_attn.q(ab.norm3(action))))
        cross = F.scaled_dot_product_attention(q, atext_k, atext_v)
        action = action + ab.cross_attn.o(cross.transpose(1, 2).flatten(2))
        action = action + amg * ab.ffn(ab.norm2(action) * (1 + ascale) + ashift)
        return force, action


class _JointLayer(nn.Module):
    """Keep native video numerics; CUDA Graph captures its launch sequence.

    Full video compilation changes reduction/rounding order enough to amplify
    BF16 error over 28 layers x 16 steps. Compile the control block separately.
    Every attention uses K/V formed from the original layer inputs.
    """
    def __init__(self, video, force, action, compile_control=False):
        super().__init__()
        self.video = video
        self.heads, self.head_dim = video.self_attn.n_heads, video.self_attn.head_dim
        control = _ControlLayer(force, action, self.heads, self.head_dim).eval()
        self.control = (torch.compile(control, fullgraph=True, dynamic=False,
            options={'triton.cudagraphs': False, 'emulate_precision_casts': True})
            if compile_control else control)

    def video_forward(self, video, vmods, static, text, positions):
        vb = self.video
        ok, ov = static[:2]
        vtext_k, vtext_v = text
        vcos, vsin = positions[:2]
        shift, scale, _ = vmods[0]
        normalized = (vb.layer_norm_self_attn(video) * (1 + scale) + shift).flatten(1, 3)
        sa = vb.self_attn
        split = lambda x: x.unflatten(-1, (self.heads, self.head_dim))
        q, k = sa.q_norm(split(sa.q_proj(normalized))), sa.k_norm(split(sa.k_proj(normalized)))
        vq = _video_rope(q, vcos, vsin).transpose(1, 2)
        vk = _video_rope(k, vcos, vsin).transpose(1, 2)
        vv = split(sa.v_proj(normalized)).transpose(1, 2)
        video_k, video_v = torch.cat((ok, vk), 2), torch.cat((ov, vv), 2)
        mixed = F.scaled_dot_product_attention(vq, video_k, video_v)
        video = video + vmods[0][2] * sa.output_proj(mixed.transpose(1, 2).flatten(2)).reshape_as(video)
        shift, scale, gate = vmods[1]
        normalized = (vb.layer_norm_cross_attn(video) * (1 + scale) + shift).flatten(1, 3)
        ca = vb.cross_attn
        q = ca.q_norm(split(ca.q_proj(normalized))).transpose(1, 2)
        cross = F.scaled_dot_product_attention(q, vtext_k.transpose(1, 2), vtext_v.transpose(1, 2))
        video = video + gate * ca.output_proj(cross.transpose(1, 2).flatten(2)).reshape_as(video)
        shift, scale, gate = vmods[2]
        video = video + gate * vb.mlp(vb.layer_norm_mlp(video) * (1 + scale) + shift)
        return video, video_k, video_v

    def forward(self, video, force, action, mods, static, text, positions, masks):
        video, video_k, video_v = self.video_forward(video, mods[0], static, text[0], positions)
        force, action = self.control(force, action, mods[1:], static[2:], text[1],
                                    positions[2:], masks, video_k, video_v)
        return video, force, action


class _Graph:
    def __init__(self, function, args):
        flat, self.spec = tree_flatten(args)
        self.device = flat[0].device
        if self.device.type != 'cuda' or any(x.device != self.device for x in flat):
            raise ValueError('CUDA graph inputs must be on the same CUDA device')
        self.signature = [(x.shape, x.dtype, x.device) for x in flat]
        with torch.cuda.device(self.device):
            self.inputs = [x.clone() for x in flat]
            values = tree_unflatten(self.inputs, self.spec)
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    function(*values)
            stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                try:
                    self.outputs = function(*values)
                except Exception:
                    # Preserve the first failure if capture_end also fails.
                    import logging
                    logging.getLogger(__name__).exception('Joint sampling graph capture body failed')
                    raise
            torch.cuda.current_stream(self.device).wait_stream(stream)

    def __call__(self, args):
        flat, spec = tree_flatten(args)
        if spec != self.spec or [(x.shape, x.dtype, x.device) for x in flat] != self.signature:
            raise ValueError('CUDA graph request shape/dtype/device changed')
        with torch.cuda.device(self.device):
            for target, source in zip(self.inputs, flat):
                target.copy_(source)
            self.graph.replay()
            # Public results must survive the next replay.
            return tuple(value.clone() for value in self.outputs)


class NoActionInferenceRunner:
    def __init__(self, model, *, compile_layers=True, cuda_graph=True, compile_threads=4):
        if model.checkpoint_schema != SCHEMA or model.video.net.extra_per_block_abs_pos_emb:
            raise ValueError('Requires the no-reference model with native rope3d positions')
        if model.video.net.patch_temporal != 1:
            raise ValueError('Temporal patching must preserve each latent frame')
        self.model = model
        self.compile_layers, self.cuda_graph = compile_layers, cuda_graph
        if compile_layers:
            import torch._inductor.config as config
            config.compile_threads = max(1, int(compile_threads))
        self.layers = [_JointLayer(v, f, a, compile_layers).eval() for v, f, a in
                       zip(model.video.blocks, model.dit['force'].blocks, model.dit['action'].blocks)]
        self.stats = dict(compile_layers=compile_layers, cuda_graph=cuda_graph,
                          compile_scope='Force/Action blocks; native Video captured in CUDA Graph',
                          text_cache_hits=0, text_cache_builds=0, schedule_builds=0,
                          graph_builds=0, graph_setup_seconds=0., action_rope='FP32 real arithmetic')
        self._weights = None
        self._schedule_key = self._text_key = None
        self._text_input = self._text_cache = None
        self._graph = None

    def close(self):
        self._graph = None
        self._schedule_key = self._text_key = None
        self._text_input = self._text_cache = None
        self.schedule = None
        self.positions = self.pad_channels = None

    def _check_weights(self):
        # Detect in-place loads, new parameter objects, device or dtype changes.
        stamp = tuple((p.data_ptr(), p._version, p.dtype, p.device) for p in self.model.parameters())
        if self._weights != stamp:
            self.close()
            self._weights = stamp

    def _conditions(self, sample):
        m = self.model
        if any(key in sample for key in ('action_condition', 'candidate_actions', 'reference', 'action_reference')):
            raise ValueError('Future action/reference conditions are unsupported')
        if 'cosmos_context' in sample:
            source, selection = sample['cosmos_context'], None
        else:
            if m.text_bank is None:
                raise ValueError('Verified text bank required')
            selection = tuple(sample['cosmos_text_index'].reshape(-1).tolist())
            if not selection or any(i < 0 or i >= len(m.text_bank) for i in selection):
                raise ValueError('Text index out of range')
            source = m.text_bank
        try:
            version = source._version
        except RuntimeError:
            version = None
        key = (source.data_ptr(), version, tuple(source.shape), source.stride(),
               source.dtype, source.device, selection)
        if version is not None and key == self._text_key:
            text, vkv = self._text_cache
            self.stats['text_cache_hits'] += 1
        else:
            if selection is None:
                raw = source
            elif len(set(selection)) == 1:
                raw = source[selection[0]:selection[0] + 1]
            else:
                raw = source[list(selection)]
            raw = raw.to(m.device, m.torch_dtype)
            text = m.video.net.crossattn_proj(raw) if m.video.net.use_crossattn_projection else raw
            vkv = []
            for block in m.video.blocks:
                a = block.cross_attn
                split = lambda x: x.unflatten(-1, (a.n_heads, a.head_dim))
                vkv.append((a.k_norm(split(a.k_proj(text))), a.v_norm(split(a.v_proj(text)))))
            self._text_input = source  # Prevent allocator-address reuse from producing a false hit.
            self._text_key, self._text_cache = key, (text, tuple(vkv))
            self.stats['text_cache_builds'] += 1
        proprio = sample['proprio'].to(m.device, m.torch_dtype)
        if proprio.ndim == 3:
            proprio = proprio[:, 0]
        text = text.expand(len(proprio), -1, -1)
        if proprio.shape != (len(text), m.state_dim + 1):
            raise ValueError('Invalid proprio width')
        context = torch.cat((text, m.proprio_encoder(proprio)[:, None]), 1)
        cmask = torch.ones(context.shape[:2], device=m.device, dtype=torch.bool)
        force = m.dit['force']
        history, valid = force.prepare_history(sample['history'].to(m.device, m.torch_dtype),
            sample['history_is_pad'].to(m.device, torch.bool),
            sample['history_channel_valid'].to(m.device, torch.bool), context)
        return (text, context, cmask, history, valid), vkv

    def _prefill(self, first, conditions, vkv):
        m = self.model
        text, context, cmask, history, hvalid = conditions
        prepared = m._video_prepare(first, torch.zeros(len(first), device=m.device, dtype=m.torch_dtype), text)
        observed, layers = prepared['tokens'], []
        for i in range(m.depth):
            observed, history, kv = m._static_layer(i, observed, history, prepared, hvalid)
            layers.append(kv)
        action_text = m.dit['action'].text_embedding(context)
        texts = []
        for video_kv, block in zip(vkv, m.dit['action'].blocks):
            a = block.cross_attn
            texts.append((video_kv, (m._split(a.norm_k(a.k(action_text))), m._split(a.v(action_text)))))
        ftext = m.dit['force'].text_encoder(context.mean(1))[:, None]
        return tuple(layers), tuple(texts), ftext

    def _build_schedules(self, steps):
        m = self.model
        return [s.build_inference_schedule(steps, m.device, m.torch_dtype)
                for s in (m.v_scheduler, m.f_scheduler, m.a_scheduler)]

    def _prepare_schedule(self, first, text, steps):
        m, ae, fe = self.model, self.model.dit['action'], self.model.dit['force']
        key = (tuple(first.shape), tuple(text.shape), first.dtype, first.device, steps, m.fps,
               m.v_scheduler.shift, m.f_scheduler.shift, m.a_scheduler.shift)
        if key == self._schedule_key:
            return
        self._graph = None
        schedules = self._build_schedules(steps)
        dummy = torch.zeros((*first.shape[:2], 3, *first.shape[3:]), device=m.device, dtype=m.torch_dtype)
        entries = []
        for step in range(steps):
            tv, tf, ta = (pair[0][step].expand(len(first)) for pair in schedules)
            full = m._video_prepare(dummy, tv, text)
            vp = frame_slice(full, 1, 3)
            state = vp['state']
            ftime = torch.cat((sinusoidal_embedding_1d(256, tf), sinusoidal_embedding_1d(256, tv)), -1)
            fm = fe.time_projection(fe.time_embedding(ftime)).unflatten(-1, (6, fe.hidden_dim))
            am = ae.time_projection(ae.time_embedding(sinusoidal_embedding_1d(ae.freq_dim, ta))).unflatten(-1, (6, ae.hidden_dim))
            source_time = torch.cat((sinusoidal_embedding_1d(256, tv), sinusoidal_embedding_1d(256, tf)), -1)
            am = am + m.dit['action_source_time'](source_time).unflatten(-1, (6, ae.hidden_dim))
            mods = []
            for vb, fb, ab in zip(m.video.blocks, fe.blocks, ae.blocks):
                vm = []
                with m.video._fp32_region(vb, first):
                    for branch in ('self_attn', 'cross_attn', 'mlp'):
                        value = getattr(vb, 'adaln_modulation_' + branch)(state['time'])
                        if vb.use_adaln_lora:
                            value = value + state['adaln']
                        vm.append(tuple(x[:, :, None, None].to(first.dtype) for x in value.chunk(3, -1)))
                mods.append((tuple(vm), tuple((fb.modulation + fm).unbind(1)), m.ops._split_modulation(ab, am)))
            entries.append((vp, tuple(mods), tuple(pair[1][step] for pair in schedules)))
        angles = entries[0][0]['state']['rope'].transpose(0, 1)
        afreq = ae.get_freqs(32)
        self.positions = (angles.cos(), angles.sin(), afreq.real.float(), afreq.imag.float())
        self.pad_channels = torch.zeros((len(first), 2, 2, *first.shape[3:]), device=m.device, dtype=m.torch_dtype)
        self.schedule = entries
        self._schedule_key = key
        self.stats['schedule_builds'] += 1

    def _rollout(self, video, force, action, first, static, texts, ftext, masks):
        m, fe, ae = self.model, self.model.dit['force'], self.model.dit['action']
        for prepared, mods, deltas in self.schedule:
            # Native prepare casts the FP32 Euler video state back to the
            # patch projection dtype; keep the state itself in FP32.
            vt = m.video.net.x_embedder(torch.cat((video[:, :, 1:].to(first.dtype), self.pad_channels), 1))
            ft = fe.force_encoder(force) + fe.position[fe.history_steps:] + ftext
            at = ae.action_encoder(action)
            for layer, mod, kv, text in zip(self.layers, mods, static, texts):
                vt, ft, at = layer(vt, ft, at, mod, kv, text, self.positions, masks)
            pv = m.video.finish(vt.flatten(1, 3), prepared)
            pf, pa = fe.head(ft), ae.head(at)
            video = torch.cat((first, video[:, :, 1:] + pv * deltas[0]), 2)
            force = force + pf * deltas[1]
            action = action + pa * deltas[2]
        return video, force, action

    @torch.no_grad()
    def infer_joint(self, sample, *, num_inference_steps=None, seed=42, video_seed=44, force_seed=43, use_cache=True):
        m = self.model
        if m.training or not use_cache:
            raise ValueError('Optimized runner requires eval mode and static cache')
        self._check_weights()
        steps = m.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        if steps < 1:
            raise ValueError('Positive sampling steps required')
        first = sample['first_frame_latents'].to(m.device, m.torch_dtype)
        if first.ndim != 5 or first.shape[1:3] != (16, 1):
            raise ValueError('Independent observed latent [B,16,1,H,W] required')
        conditions, video_text = self._conditions(sample)
        self._prepare_schedule(first, conditions[0], steps)
        static, texts, ftext = self._prefill(first, conditions, video_text)
        count = 3 * static[0][0].shape[2]
        valid = conditions[-1]
        fvalid = torch.cat((torch.ones((len(first), count), device=m.device, dtype=torch.bool),
                            valid, torch.ones((len(first), 32), device=m.device, dtype=torch.bool)), 1)
        avalid = torch.cat((fvalid, torch.ones((len(first), 32), device=m.device, dtype=torch.bool)), 1)
        masks = (fvalid[:, None, None, :], avalid[:, None, None, :])
        def noise(shape, value):
            return torch.randn(shape, generator=torch.Generator(device='cpu').manual_seed(value)).to(m.device, m.torch_dtype)
        video = noise((*first.shape[:2], 3, *first.shape[3:]), video_seed)
        video[:, :, :1] = first
        force, action = noise((len(first), 32, 6), force_seed), noise((len(first), 32, m.action_dim), seed)
        args = (video, force, action, first, static, texts, ftext, masks)
        if self.cuda_graph:
            if self._graph is None:
                started = time.perf_counter()
                self._graph = _Graph(self._rollout, args)
                self.stats['graph_builds'] += 1
                self.stats['graph_setup_seconds'] += time.perf_counter() - started
            video, force, action = self._graph(args)
        else:
            video, force, action = self._rollout(*args)
        return dict(video_latents=video, force=force, action=action)
