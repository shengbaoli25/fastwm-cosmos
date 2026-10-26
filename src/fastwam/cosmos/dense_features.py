"""Dense frozen-teacher features with a fresh Joint8 V/F/A flow model."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fastwam.forcewm.diffusion import ForceDiffusionExpert
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from fastwam.models.wan22.wan_video_dit import sinusoidal_embedding_1d
from .model import _ActionOps, temporal_mse


SCHEMA = 'fastwam-dense-feature-joint8-vfa-v1'
GROUP_READS = ((1, 0, 0, 0, 0), (1, 1, 0, 0, 0),
               (1, 0, 1, 0, 0), (1, 1, 1, 1, 0), (1, 1, 1, 1, 1))


def build_joint_attention_mask(lengths, valid):
    if (len(lengths) != 5 or any(type(n) is not int or n < 1 for n in lengths)
            or valid.dtype != torch.bool or valid.ndim != 2 or sum(lengths) != valid.shape[1]):
        raise ValueError('Five positive O/H/V/F/A lengths and matching bool validity required')
    groups = torch.cat([torch.full((n,), i, device=valid.device, dtype=torch.long)
                        for i, n in enumerate(lengths)])
    reads = torch.tensor(GROUP_READS, device=valid.device, dtype=torch.bool)
    topology = reads[groups[:, None], groups[None, :]]
    allowed = topology[None] & valid[:, :, None] & valid[:, None, :]
    unavailable = torch.eye(len(groups), device=valid.device, dtype=torch.bool)[None]
    return (allowed | (unavailable & ~valid[:, :, None]))[:, None]


def _split(x, heads, head_dim):
    return x.unflatten(-1, (heads, head_dim)).transpose(1, 2)


def _attend(q, k, v, mask):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)


class DenseFeatureBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError('Dense feature width must divide attention heads')
        self.heads, self.head_dim = heads, dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def project(self, x):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        return (_split(q, self.heads, self.head_dim), _split(k, self.heads, self.head_dim),
                _split(v, self.heads, self.head_dim))

    def finish(self, x, mixed):
        x = x + self.out(mixed.transpose(1, 2).flatten(2))
        return x + self.ff(self.norm2(x))


class DenseFeatureVideoExpert(nn.Module):
    def __init__(self, *, feature_dim=768, hidden_dim=768, depth=12, heads=12, view_grid=14):
        super().__init__()
        self.feature_dim, self.hidden_dim = feature_dim, hidden_dim
        self.num_layers, self.num_heads = depth, heads
        self.attn_head_dim = hidden_dim // heads
        self.view_grid = view_grid
        self.tokens_per_view = view_grid * view_grid
        self.tokens_per_frame = 2 * self.tokens_per_view
        self.input = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim))
        self.camera = nn.Parameter(torch.randn(1, 1, 2, 1, hidden_dim) / hidden_dim ** .5)
        self.spatial = nn.Parameter(torch.randn(1, 1, 1, self.tokens_per_view, hidden_dim) / hidden_dim ** .5)
        self.temporal = nn.Parameter(torch.randn(1, 3, 1, 1, hidden_dim) / hidden_dim ** .5)
        self.time = nn.Sequential(nn.Linear(256, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([DenseFeatureBlock(hidden_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, feature_dim)

    def _position(self, batch, frames, device, dtype):
        camera = self.camera.to(device=device, dtype=dtype).expand(batch, frames, -1, -1, -1)
        spatial = self.spatial.to(device=device, dtype=dtype).expand(batch, frames, 2, -1, -1)
        temporal = self.temporal[:, :frames].to(device=device, dtype=dtype).expand(batch, -1, -1, -1, -1)
        return (camera + spatial + temporal).flatten(2, 3)

    def observed_tokens(self, observed):
        if observed.ndim != 3 or observed.shape[1:] != (self.tokens_per_frame, self.feature_dim):
            raise ValueError('Current dense feature frame geometry differs')
        x = self.input(observed)
        return x + self._position(len(x), 1, x.device, x.dtype)[:, 0]

    def future_tokens(self, future, clock):
        b, steps, tokens, width = future.shape
        if (steps != 2 or tokens != self.tokens_per_frame or width != self.feature_dim
                or clock.shape != (b,)):
            raise ValueError('Future features must be [B,2,2*grid^2,D] with a batch clock')
        x = self.input(future.flatten(1, 2))
        position = self._position(b, 3, x.device, x.dtype)[:, 1:].flatten(1, 2)
        time_embedding = sinusoidal_embedding_1d(256, clock.float() / 1000.)
        time_embedding = time_embedding.to(
            device=self.time[0].weight.device, dtype=self.time[0].weight.dtype)
        time = self.time(time_embedding)
        time = time[:, None, None, :].expand(-1, 2, self.tokens_per_frame, -1).flatten(1, 2)
        return x + position + time

    def velocity(self, hidden):
        b = len(hidden)
        x = self.output(self.norm(hidden))
        return x.reshape(b, 2, self.tokens_per_frame, self.feature_dim)


class DenseFeatureJointModel(nn.Module):
    """Freshly initialized Joint8 Action/Force/feature flow model.

    The visual teacher is intentionally outside this module: its verified
    dense features are immutable cache inputs, while this model predicts the
    two future feature slots and jointly denoises Force and Action.
    """
    checkpoint_schema = SCHEMA

    def __init__(self, *, manifest, feature_dim=768, hidden_dim=768, depth=12,
                 heads=12, view_grid=14, action_hidden=768, action_ffn=3072,
                 force_hidden=768, force_ffn=2048, force_shift=1.,
                 action_seed=42, force_seed=43, proprio_seed=44,
                 gradient_checkpointing=True, num_inference_steps=16):
        super().__init__()
        if hidden_dim % heads or num_inference_steps < 1:
            raise ValueError('Invalid dense model geometry or inference schedule')
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.num_inference_steps = int(num_inference_steps)
        self.depth = int(depth)
        self.action_dim = 8
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(action_seed)
            video = DenseFeatureVideoExpert(feature_dim=feature_dim, hidden_dim=hidden_dim,
                depth=depth, heads=heads, view_grid=view_grid)
            action = ActionDiT(hidden_dim=action_hidden, action_dim=8, ffn_dim=action_ffn,
                text_dim=hidden_dim, freq_dim=256, eps=1e-6, num_heads=heads,
                attn_head_dim=hidden_dim // heads, num_layers=depth)
            torch.manual_seed(proprio_seed)
            proprio = nn.Linear(9, hidden_dim)
            source_time = nn.Sequential(nn.Linear(512, action_hidden), nn.SiLU(),
                                        nn.Linear(action_hidden, 6 * action_hidden))
            torch.manual_seed(force_seed)
            force = ForceDiffusionExpert(hidden_dim=force_hidden, ffn_dim=force_ffn,
                num_layers=depth, width=hidden_dim, num_heads=heads, text_dim=hidden_dim,
                state_dim=8)
        self.dit = nn.ModuleDict(dict(video=video, action=action, force=force,
            proprio_encoder=proprio, action_source_time=source_time))
        self.manifest = dict(manifest, schema=SCHEMA,
            attention_groups=['O', 'H', 'V', 'F', 'A'], allowed_reads=GROUP_READS,
            group_semantics=['O->O', 'H->O,H', 'V->O,V', 'F->O,H,V,F', 'A->O,H,V,F,A'],
            action_space='joint8', action_dim=8, state_dim=8, history_dim=14,
            visual_targets='t=16,32 from 30Hz input; one target per four stride-4 sampled frames',
            visual_teacher='frozen precomputed patch features; teacher weights excluded from trainable model',
            initialization='fresh random video-dynamics, ForceDiffusionExpert and ActionDiT; independent optimizers',
            architecture=dict(depth=depth, feature_dim=feature_dim, hidden_dim=hidden_dim,
                heads=heads, view_grid=view_grid, action_hidden=action_hidden,
                force_hidden=force_hidden, num_inference_steps=num_inference_steps))
        self.v_scheduler = WanContinuousFlowMatchScheduler(shift=5.)
        self.f_scheduler = WanContinuousFlowMatchScheduler(shift=force_shift)
        self.a_scheduler = WanContinuousFlowMatchScheduler(shift=1.)
        self.ops = _ActionOps()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def torch_dtype(self):
        return self.dit['action'].action_encoder.weight.dtype

    def optimizer_param_groups(self, parameters):
        from fastwam.baseline.optimizer import bounded_optimizer_param_groups
        return bounded_optimizer_param_groups(self, parameters)

    def _conditions(self, sample):
        proprio = sample['proprio'].to(self.device, self.torch_dtype)
        if proprio.shape[1:] != (32, 9):
            raise ValueError('Joint8 Proprio32x9 required')
        context = self.dit['proprio_encoder'](proprio[:, 0])[:, None]
        context_mask = torch.ones(context.shape[:2], device=self.device, dtype=torch.bool)
        force = self.dit['force']
        history, hvalid = force.prepare_history(sample['history'].to(self.device, self.torch_dtype),
            sample['history_is_pad'].to(self.device, torch.bool),
            sample['history_channel_valid'].to(self.device, torch.bool), context)
        return context, context_mask, history, hvalid

    def _run_static(self, observed, history, hvalid, full_mask, o_slice, h_slice):
        caches = []
        for i, video_block in enumerate(self.dit['video'].blocks):
            force_block = self.dit['force'].blocks[i]
            oq, ok, ov = video_block.project(observed)
            hq, hk, hv, hs = force_block.qkv(history)
            observed = video_block.finish(observed, _attend(oq, ok, ov, full_mask[:, :, o_slice, o_slice]))
            keys = torch.cat((ok, hk), dim=2)
            values = torch.cat((ov, hv), dim=2)
            history = force_block(_attend(hq, keys, values, full_mask[:, :, h_slice, :h_slice.stop]), hs)
            history = torch.where(hvalid[..., None], history, 0)
            caches.append((ok, ov, hk, hv))
        return observed, history, tuple(caches)

    def joint_denoise_core(self, observed_features, video_noise, force_noise, action_noise,
                           tv, tf, ta, sample, conditions):
        context, cmask, history, hvalid = conditions
        b = len(observed_features)
        observed = self.dit['video'].observed_tokens(observed_features)
        future = self.dit['video'].future_tokens(video_noise, tv)
        force, fm = self.dit['force'].prepare_future(force_noise, tf, tv, context)
        action, _, am, text, text_mask, freq = self.dit['action'].prepare(action_noise, ta, context, cmask)
        source_time = torch.cat((sinusoidal_embedding_1d(256, tv.float()),
                                 sinusoidal_embedding_1d(256, tf.float())), -1)
        source_time = source_time.to(
            device=self.dit['action_source_time'][0].weight.device,
            dtype=self.dit['action_source_time'][0].weight.dtype)
        am = am + self.dit['action_source_time'](source_time).unflatten(-1,
            (6, self.dit['action'].hidden_dim))

        n_obs, n_future = observed.shape[1], future.shape[1]
        n_hist = history.shape[1]
        n_force, n_action = force.shape[1], action.shape[1]
        lengths = (n_obs, n_hist, n_future, n_force, n_action)
        image_pad = sample['image_is_pad'].to(self.device, torch.bool)
        target_valid = ~image_pad[:, [4, 8]]
        force_valid = (~sample['force_is_pad'].to(self.device, torch.bool)
                       & sample['wrench_valid'].to(self.device, torch.bool).any(-1))
        action_valid = ~sample['action_is_pad'].to(self.device, torch.bool)
        valid = torch.cat((torch.ones(b, n_obs, device=self.device, dtype=torch.bool),
            hvalid, target_valid.repeat_interleave(self.dit['video'].tokens_per_frame, dim=1),
            force_valid, action_valid), 1)
        mask = build_joint_attention_mask(lengths, valid)
        o0 = 0
        h0 = o0 + n_obs
        v0 = h0 + n_hist
        f0 = v0 + n_future
        a0 = f0 + n_force
        o_slice, h_slice = slice(o0, h0), slice(h0, v0)
        v_slice, f_slice, a_slice = slice(v0, f0), slice(f0, a0), slice(a0, a0+n_action)
        observed, history, static = self._run_static(observed, history, hvalid,
            mask, o_slice, h_slice)
        for i, video_block in enumerate(self.dit['video'].blocks):
            force_block = self.dit['force'].blocks[i]
            action_block = self.dit['action'].blocks[i]
            ok, ov, hk, hv = static[i]
            def run_layer(vcur, fcur, acur, ok, ov, hk, hv,
                          video_block=video_block, force_block=force_block,
                          action_block=action_block):
                vq, vk, vv = video_block.project(vcur)
                fq, fk, fv, fs = force_block.qkv(fcur, fm)
                aq, ak, av, residual, gate, shift, scale, mg, _ = self.ops._build_expert_attention_io(
                    self.dit['action'], action_block, acur, freq, am)
                aq, ak, av = (_split(x, self.dit['action'].num_heads,
                                      self.dit['action'].attn_head_dim) for x in (aq, ak, av))
                keys = torch.cat((ok, hk, vk, fk, ak), dim=2)
                values = torch.cat((ov, hv, vv, fv, av), dim=2)
                vnext = video_block.finish(vcur, _attend(vq, keys, values, mask[:, :, v_slice, :]))
                vnext = torch.where(target_valid.repeat_interleave(self.dit['video'].tokens_per_frame, dim=1)[..., None], vnext, 0)
                fnext = force_block(_attend(fq, keys, values, mask[:, :, f_slice, :]), fs)
                fnext = torch.where(force_valid[..., None], fnext, 0)
                mixed = _attend(aq, keys, values, mask[:, :, a_slice, :]).transpose(1, 2).flatten(2)
                anext = self.ops._apply_expert_post_block_tensor(action_block, residual, mixed,
                    gate, shift, scale, mg, text, text_mask)
                anext = torch.where(action_valid[..., None], anext, 0)
                return vnext, fnext, anext
            args = (future, force, action, ok, ov, hk, hv)
            future, force, action = (checkpoint(run_layer, *args, use_reentrant=False)
                if self.gradient_checkpointing and torch.is_grad_enabled() else run_layer(*args))
        if not bool(torch.isfinite(future).all() & torch.isfinite(force).all() & torch.isfinite(action).all()):
            raise FloatingPointError('Nonfinite V/F/A layer output')
        return (self.dit['video'].velocity(future), self.dit['force'].head(force),
                self.dit['action'].post(action))

    def training_loss(self, sample):
        required = ('visual_features', 'action', 'history', 'proprio', 'wrench_future',
                    'action_is_pad', 'image_is_pad', 'history_is_pad', 'force_is_pad',
                    'history_channel_valid', 'wrench_valid')
        if any(key not in sample for key in required):
            raise ValueError('Dense visual cache and Joint8 numeric supervision required')
        visual = sample['visual_features'].to(self.device, self.torch_dtype)
        if (visual.ndim != 4 or visual.shape[1:] != (3, self.dit['video'].tokens_per_frame,
                                                      self.dit['video'].feature_dim)):
            raise ValueError('Visual features must contain O,t=16,t=32 slots')
        action = sample['action'].to(self.device, self.torch_dtype)
        force = sample['wrench_future'].to(self.device, self.torch_dtype)
        if action.shape[1:] != (32, 8) or force.shape[1:] != (32, 6):
            raise ValueError('Action32x8 and Force32x6 required')
        b = len(visual)
        context = self._conditions(sample)
        tv, tf, ta = (scheduler.sample_training_t(b, self.device, self.torch_dtype)
                      for scheduler in (self.v_scheduler, self.f_scheduler, self.a_scheduler))
        clean_v = visual[:, 1:]
        noise_v, noise_f, noise_a = torch.randn_like(clean_v), torch.randn_like(force), torch.randn_like(action)
        fvalid = sample['wrench_valid'].to(self.device, torch.bool) & ~sample['force_is_pad'].to(self.device, torch.bool)[..., None]
        clean_f = torch.where(fvalid, force, 0)
        noisy_v = self.v_scheduler.add_noise(clean_v, noise_v, tv)
        noisy_f = self.f_scheduler.add_noise(clean_f, noise_f, tf)
        noisy_a = self.a_scheduler.add_noise(action, noise_a, ta)
        pv, pf, pa = self.joint_denoise_core(visual[:, 0], noisy_v, noisy_f, noisy_a,
            tv, tf, ta, sample, context)
        vvalid = ~sample['image_is_pad'].to(self.device, torch.bool)[:, [4, 8]]
        target_v = self.v_scheduler.training_target(clean_v, noise_v, tv)
        lv = temporal_mse(pv.flatten(2), target_v.flatten(2), vvalid, self.v_scheduler, tv)
        av = ~sample['action_is_pad'].to(self.device, torch.bool)
        la = temporal_mse(pa, self.a_scheduler.training_target(action, noise_a, ta),
                          av, self.a_scheduler, ta)
        target_f = self.f_scheduler.training_target(clean_f, noise_f, tf)
        delta = torch.where(fvalid, pf.float() - target_f.float(), 0)
        lf = ((delta.square().sum((1, 2)) / fvalid.sum((1, 2)).clamp_min(1))
              * self.f_scheduler.training_weight(tf)).mean()
        if not bool(torch.isfinite(lv) & torch.isfinite(la) & torch.isfinite(lf)):
            raise FloatingPointError('Nonfinite dense feature/Force/Action objective')
        return lv + la + lf, dict(loss_video_fm=float(lv.detach()),
            loss_force_fm=float(lf.detach()), loss_action_fm=float(la.detach()))

    @torch.no_grad()
    def infer_joint(self, sample, *, num_inference_steps=None, seed=42, video_seed=44, force_seed=43):
        steps = self.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        if steps < 1:
            raise ValueError('Positive inference step count required')
        modes = {module: module.training for module in self.modules()}
        self.eval()
        try:
            if sample['visual_features'].ndim == 3:
                sample = {key: (value.unsqueeze(0) if isinstance(value, torch.Tensor) and value.ndim > 0
                                else value) for key, value in sample.items()}
            visual = sample['visual_features'].to(self.device, self.torch_dtype)
            b = len(visual)
            conditions = self._conditions(sample)
            def noise(shape, noise_seed):
                generator = torch.Generator(device='cpu').manual_seed(noise_seed)
                return torch.randn(shape, generator=generator).to(self.device, self.torch_dtype)
            v = noise((b, 2, visual.shape[2], visual.shape[3]), video_seed)
            f = noise((b, 32, 6), force_seed)
            a = noise((b, 32, 8), seed)
            schedules = [scheduler.build_inference_schedule(steps, self.device, self.torch_dtype)
                         for scheduler in (self.v_scheduler, self.f_scheduler, self.a_scheduler)]
            for index in range(steps):
                tv, tf, ta = (pair[0][index].expand(b) for pair in schedules)
                pv, pf, pa = self.joint_denoise_core(visual[:, 0], v, f, a, tv, tf, ta, sample, conditions)
                v = self.v_scheduler.step(pv, schedules[0][1][index], v)
                f = self.f_scheduler.step(pf, schedules[1][1][index], f)
                a = self.a_scheduler.step(pa, schedules[2][1][index], a)
            return dict(action=a, force=f, future_features=v)
        finally:
            for module, mode in modes.items():
                module.training = mode

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = dict(schema=self.checkpoint_schema, manifest=self.manifest,
                       model=self.state_dict(), step=step)
        if optimizer is not None:
            payload['optimizer'] = optimizer.state_dict()
        path = Path(path)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if payload.get('schema') != self.checkpoint_schema or payload.get('manifest') != self.manifest:
            raise ValueError('Checkpoint schema/teacher/data identity differs')
        self.load_state_dict(payload['model'], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(payload['optimizer'])
        return payload.get('step')


def create_dense_feature_joint(**kwargs):
    device = kwargs.pop('device', 'cuda')
    dtype = kwargs.pop('model_dtype', kwargs.pop('torch_dtype', torch.bfloat16))
    return DenseFeatureJointModel(**kwargs).to(device=device, dtype=dtype)
