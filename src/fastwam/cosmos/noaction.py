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


def attend(q, k, v, valid=None):
    mask = None if valid is None else valid[:, None, None, :]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)


class NoActionJointModel(CosmosForceWM):
    checkpoint_schema = SCHEMA
    reference_mode = 'none'

    def __init__(self, net, *, manifest, text_bank=None, action_hidden=768, action_ffn=3072,
                 force_hidden=768, force_ffn=2048, force_heads=16, action_seed=42,
                 force_seed=43, memory_seed=44, num_inference_steps=16,
                 force_shift=1., gradient_checkpointing=True, fps=7.5, action_space='joint8'):
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
        self.dit = nn.ModuleDict(dict(video=video, action=action, force=force, action_source_time=source_time,
                                     runtime=ForceRuntimeState(memory_seed, self.manifest)))
        self.register_buffer('text_bank', text_bank, persistent=False)
        self.v_scheduler = WanContinuousFlowMatchScheduler(shift=5.)
        self.a_scheduler = WanContinuousFlowMatchScheduler(shift=1.)
        self.f_scheduler = WanContinuousFlowMatchScheduler(shift=force_shift)
        self.compile_training_denoise = False
        self.ops = _ActionOps()

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

    def _static_layer(self, i, observed, history, prepared, hvalid, text_kv=None):
        oq, ok, ov, os = self.video.block_qkv(i, observed, prepared)
        block = self.dit['force'].blocks[i]
        hq, hk, hv, hs = block.qkv(history)
        observed = self.video.block_finish(i, attend(oq, ok, ov), os, prepared, text_kv)
        valid = torch.cat((torch.ones(len(history), ok.shape[2], device=history.device, dtype=torch.bool), hvalid), 1)
        history = block(attend(hq, torch.cat((ok, hk), 2), torch.cat((ov, hv), 2), valid), hs)
        history = torch.where(hvalid[..., None], history, 0)
        return observed, history, (ok, ov, hk, hv)

    def _dynamic_layer(self, i, video, force, action, prepared, fm, am, text, text_mask, freq, hvalid, static, text_cache=None):
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
        force = block(attend(fq, k, v, valid), fs)
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
        return video, force, action

    def prefill_static(self, first, conditions):
        """Request-local O/H cache. Used only for inference, with fixed weights."""
        if torch.is_grad_enabled():
            raise ValueError('Static inference cache must not detach training gradients')
        raw, context, cmask, history, hvalid = conditions
        zero = torch.zeros(len(first), device=first.device, dtype=first.dtype)
        prepared = self._video_prepare(first, zero, raw)
        texts = []
        action_text = self.dit['action'].text_embedding(context)
        for vb, ab in zip(self.video.blocks, self.dit['action'].blocks):
            va = vb.cross_attn; aa = ab.cross_attn
            split = lambda x: x.unflatten(-1, (va.n_heads, va.head_dim))
            vkv = (va.k_norm(split(va.k_proj(raw))), va.v_norm(split(va.v_proj(raw))))
            akv = (self._split(aa.norm_k(aa.k(action_text))), self._split(aa.v(action_text)))
            texts.append((vkv, akv))
        observed = prepared['tokens']; caches = []
        for i in range(self.depth):
            observed, history, kv = self._static_layer(
                i, observed, history, prepared, hvalid, texts[i][0])
            caches.append(kv)
        return dict(layers=tuple(caches), text=tuple(texts))

    def joint_denoise_core(self, video, force, action, tv, tf, ta, conditions, static_cache=None):
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
        for i in range(self.depth):
            def layer(o, h, v, f, a, i=i):
                if static_cache is None:
                    o, h, kv = self._static_layer(i, o, h, op, hvalid)
                else:
                    kv = static_cache["layers"][i]
                v, f, a = self._dynamic_layer(i, v, f, a, vp, fm, am, text, text_mask, freq, hvalid, kv,
                    None if static_cache is None else static_cache["text"][i])
                return o, h, v, f, a
            args = (observed, history, vt, ft, at)
            observed, history, vt, ft, at = (checkpoint(layer, *args, use_reentrant=False)
                if self.gradient_checkpointing and torch.is_grad_enabled() else layer(*args))
        future_velocity = self.video.finish(vt, vp)
        vv = torch.cat((torch.zeros_like(video[:, :, :1]), future_velocity), 2)
        return vv, self.dit['force'].head(ft), self.dit['action'].post(at)

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
        pv, pf, pa = self.joint_denoise_core(v, f, a, tv, tf, ta, conditions)
        imagepad = sample['image_is_pad'].to(self.device, torch.bool)
        vvalid = ~imagepad[:, 1:].reshape(len(clean), 2, 4).all(-1)
        lv = temporal_mse(pv[:, :, 1:].transpose(1, 2).flatten(2),
            self.v_scheduler.training_target(clean, vn, tv)[:, :, 1:].transpose(1, 2).flatten(2), vvalid, self.v_scheduler, tv)
        la = temporal_mse(pa, self.a_scheduler.training_target(action, an, ta),
            ~sample['action_is_pad'].to(self.device, torch.bool), self.a_scheduler, ta)
        delta = torch.where(fvalid, pf.float() - self.f_scheduler.training_target(force, fn, tf).float(), 0)
        lf = ((delta.square().sum((1, 2)) / fvalid.sum((1, 2)).clamp_min(1)) * self.f_scheduler.training_weight(tf)).mean()
        return lv + lf + la, dict(loss_video_fm=float(lv.detach()), loss_force_fm=float(lf.detach()), loss_action_fm=float(la.detach()))

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
            schedules = [s.build_inference_schedule(steps, self.device, self.torch_dtype) for s in (self.v_scheduler, self.f_scheduler, self.a_scheduler)]
            for i in range(steps):
                tv, tf, ta = (pair[0][i].expand(len(first)) for pair in schedules)
                pv, pf, pa = self.joint_denoise_core(v, f, a, tv, tf, ta, conditions, static)
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
