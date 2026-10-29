"""Predictive video context with unchanged Cosmos Joint8 Action/Force experts."""
from pathlib import Path
from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fastwam.forcewm.diffusion import ForceDiffusionExpert
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from fastwam.models.wan22.wan_video_dit import sinusoidal_embedding_1d
from .model import _ActionOps, temporal_mse

SCHEMA = 'fastwam-jepa-video-context-cosmos-joint8-v1'


def _plain_config(value):
    if isinstance(value, Mapping):
        return {key: _plain_config(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_config(item) for item in value]
    return value


def _split(x, heads):
    return x.unflatten(-1, (heads, x.shape[-1] // heads)).transpose(1, 2)


def _attend(q, k, v, valid=None):
    mask = None if valid is None else valid[:, None, None, :]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)


def _video_coordinates(grid, device=None):
    t, view, row, col = torch.meshgrid(torch.arange(5, device=device),
        torch.arange(2, device=device), torch.arange(grid, device=device),
        torch.arange(grid, device=device), indexing='ij')
    # Each camera resets its local spatial coordinates; view embeddings distinguish cameras.
    return torch.stack((t, row, col), -1).reshape(-1, 3)


def _rope3d(x, coordinates):
    dim = x.shape[-1]
    first = (dim // 6) * 2
    if first < 2 or dim % 2:
        raise ValueError('3D rotary head dimension must be even and at least six')
    parts, offset = [], 0
    for axis, width in enumerate((first, first, dim - 2 * first)):
        block = x[..., offset:offset+width]
        frequency = torch.pow(10000., -torch.arange(0, width, 2, device=x.device).float() / width)
        angle = coordinates[:, axis].float()[:, None] * frequency
        cosine, sine = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
        even, odd = block[..., 0::2], block[..., 1::2]
        parts.append(torch.stack((even*cosine-odd*sine, even*sine+odd*cosine), -1).flatten(-2))
        offset += width
    return torch.cat(parts, -1)


class PredictiveBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.condition_norm = nn.LayerNorm(width)
        self.condition_q = nn.Linear(width, width)
        self.condition_kv = nn.Linear(width, 2 * width)
        self.condition_out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, x, conditions, condition_valid, coordinates):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q, k, v = [_split(t, self.heads) for t in (q, k, v)]
        x = x + self.out(_attend(_rope3d(q, coordinates), _rope3d(k, coordinates), v)
                        .transpose(1, 2).flatten(2))
        ck, cv = self.condition_kv(conditions).chunk(2, -1)
        cq = self.condition_q(self.condition_norm(x))
        x = x + self.condition_out(_attend(*[_split(t, self.heads) for t in (cq, ck, cv)],
                                          condition_valid).transpose(1, 2).flatten(2))
        x = x + self.ff(self.norm2(x))
        return x


class DenseFeatureVideoExpert(nn.Module):
    """Bidirectional observed/future-query predictor; no teacher targets enter it."""
    def __init__(self, *, feature_dim=1408, hidden_dim=768, depth=28, heads=12,
                 view_grid=14, context_dim=1024, joint_width=2048, joint_heads=16):
        super().__init__()
        self.feature_dim, self.hidden_dim = feature_dim, hidden_dim
        self.num_layers, self.num_heads = depth, heads
        self.joint_heads = joint_heads
        self.tokens_per_view = view_grid ** 2
        self.tokens_per_frame = 2 * self.tokens_per_view
        self.input = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.camera = nn.Parameter(torch.randn(1, 1, 2, 1, hidden_dim) * 0.02)
        self.register_buffer('coordinates', _video_coordinates(view_grid), persistent=False)
        self.context_input = nn.Linear(context_dim, hidden_dim)
        self.history_input = nn.Linear(28, hidden_dim)
        self.history_position = nn.Parameter(torch.randn(1, 32, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(PredictiveBlock(hidden_dim, heads) for _ in range(depth))
        # Bridges belong to the visual interface, preserving Cosmos expert tensor shapes.
        self.context_kv = nn.ModuleList(nn.Sequential(nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2 * joint_width)) for _ in range(depth))
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, feature_dim)

    def prepare(self, observed, context, history, channel_valid):
        if observed.shape[1:] != (self.tokens_per_frame, self.feature_dim):
            raise ValueError('Video context must be one two-frame tubelet per camera')
        batch = len(observed)
        position = self.camera.expand(batch, 5, -1, self.tokens_per_view, -1)
        position = position.flatten(2, 3)
        observed = self.input(observed) + position[:, 0]
        queries = self.mask_token.expand(batch, 4*self.tokens_per_frame, -1)
        queries = queries + position[:, 1:].flatten(1, 2)
        history = torch.where(channel_valid, history, 0)
        history = self.history_input(torch.cat((history, channel_valid.to(history.dtype)), -1))
        history = history + self.history_position
        conditions = torch.cat((self.context_input(context), history), 1)
        condition_valid = torch.cat((torch.ones(context.shape[:2], dtype=torch.bool, device=context.device),
                                     channel_valid.any(-1)), 1)
        return torch.cat((observed, queries), 1), conditions, condition_valid

    def forward(self, observed, context, history, channel_valid, checkpointing=False):
        x, conditions, valid = self.prepare(observed, context, history, channel_valid)
        caches = []
        for block, bridge in zip(self.blocks, self.context_kv):
            def layer(x, conditions, block=block, bridge=bridge):
                x = block(x, conditions, valid, self.coordinates)
                # Only observed positions are exported; future queries stay private.
                k, v = bridge(x[:, :self.tokens_per_frame]).chunk(2, -1)
                return x, _rope3d(_split(k, self.joint_heads),
                                  self.coordinates[:self.tokens_per_frame]), _split(v, self.joint_heads)
            x, k, v = (checkpoint(layer, x, conditions, use_reentrant=False)
                       if checkpointing and torch.is_grad_enabled() else layer(x, conditions))
            caches.append((k, v))
        predicted = self.output(self.norm(x[:, self.tokens_per_frame:]))
        return predicted.reshape(len(x), 4, self.tokens_per_frame, self.feature_dim), tuple(caches)


class DenseFeatureJointModel(nn.Module):
    """Single-pass visual prediction followed by coupled Cosmos Force/Action FM."""
    checkpoint_schema = SCHEMA

    def __init__(self, *, manifest, feature_dim=1408, hidden_dim=768, depth=28,
                 heads=12, view_grid=14, action_hidden=768, action_ffn=3072,
                 force_hidden=768, force_ffn=2048, joint_heads=16, joint_head_dim=128,
                 context_dim=1024, force_shift=1., lambda_future=1.,
                 action_seed=42, force_seed=43, proprio_seed=44,
                 gradient_checkpointing=True, num_inference_steps=16):
        super().__init__()
        if hidden_dim % heads or num_inference_steps < 1 or lambda_future not in (0., 1.):
            raise ValueError('Invalid predictor geometry, inference schedule or ablation weight')
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.num_inference_steps = int(num_inference_steps)
        self.depth, self.action_dim = int(depth), 8
        self.lambda_future = float(lambda_future)
        joint_width = joint_heads * joint_head_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(action_seed)
            video = DenseFeatureVideoExpert(feature_dim=feature_dim, hidden_dim=hidden_dim,
                depth=depth, heads=heads, view_grid=view_grid, context_dim=context_dim,
                joint_width=joint_width, joint_heads=joint_heads)
            torch.manual_seed(action_seed)
            action = ActionDiT(hidden_dim=action_hidden, action_dim=8, ffn_dim=action_ffn,
                text_dim=context_dim, freq_dim=256, eps=1e-6, num_heads=joint_heads,
                attn_head_dim=joint_head_dim, num_layers=depth)
            # Same Action/proprio/source-time initialization order as NoActionJointModel.
            proprio = nn.Linear(9, context_dim)
            source_time = nn.Sequential(nn.Linear(512, action_hidden), nn.SiLU(),
                                        nn.Linear(action_hidden, 6 * action_hidden))
            torch.manual_seed(force_seed)
            force = ForceDiffusionExpert(hidden_dim=force_hidden, ffn_dim=force_ffn,
                num_layers=depth, width=joint_width, num_heads=joint_heads,
                text_dim=context_dim, state_dim=8)
            torch.manual_seed(proprio_seed)
            task = nn.Embedding(1, context_dim)
        self.dit = nn.ModuleDict(dict(video=video, action=action, force=force,
            proprio_encoder=proprio, action_source_time=source_time, task=task))
        self.manifest = dict(_plain_config(manifest), schema=SCHEMA,
            architecture=dict(depth=depth, feature_dim=feature_dim, hidden_dim=hidden_dim,
                heads=heads, view_grid=view_grid, action_hidden=action_hidden,
                force_hidden=force_hidden, action_ffn=action_ffn, force_ffn=force_ffn,
                joint_heads=joint_heads, joint_head_dim=joint_head_dim, context_dim=context_dim),
            lambda_future=float(lambda_future), action_space='joint8',
            visual_input=('[-4,0] mean of independent DINO frame features; future pair means only in loss'
                if manifest.get('teacher', {}).get('name') == 'dinov3_vith16plus' else
                '[-4,0] two-frame context; full10-frame future targets only in loss'),
            visual_interface='post-layer observed-position K/V only; learned 768-to-2048 bridge',
            visual_clock='deterministic clean zero', action_force='original Cosmos28 modules',
            initialization='fresh predictor and experts; frozen official teacher cached separately',
            num_inference_steps=num_inference_steps)
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
        if proprio.ndim == 3:
            proprio = proprio[:, 0]
        if proprio.shape[1:] != (9,):
            raise ValueError('Current Joint8 proprio9 required')
        batch = len(proprio)
        task = self.dit['task'](torch.zeros(batch, device=self.device, dtype=torch.long))[:, None]
        context = torch.cat((task, self.dit['proprio_encoder'](proprio)[:, None]), 1)
        mask = torch.ones(context.shape[:2], device=self.device, dtype=torch.bool)
        raw_history = sample['history'].to(self.device, self.torch_dtype)
        channels = sample['history_channel_valid'].to(self.device, torch.bool)
        pads = sample['history_is_pad'].to(self.device, torch.bool)
        channels = channels & ~pads[..., None]
        history, valid = self.dit['force'].prepare_history(raw_history, pads, channels, context)
        return context, mask, history, valid, raw_history, channels

    def prepare_context(self, observed, sample):
        conditions = self._conditions(sample)
        context, _, history, valid, raw_history, channels = conditions
        future, visual = self.dit['video'](observed, context, raw_history, channels,
                                          self.gradient_checkpointing)
        caches = []
        for force_block, (ok, ov) in zip(self.dit['force'].blocks, visual):
            def layer(history, ok, ov, force_block=force_block):
                hq, hk, hv, hs = force_block.qkv(history)
                key_valid = torch.cat((torch.ones(len(history), ok.shape[2], device=self.device,
                                                  dtype=torch.bool), valid), 1)
                history = force_block(_attend(hq, torch.cat((ok, hk), 2),
                                               torch.cat((ov, hv), 2), key_valid), hs)
                return torch.where(valid[..., None], history, 0), hk, hv
            history, hk, hv = (checkpoint(layer, history, ok, ov, use_reentrant=False)
                if self.gradient_checkpointing and torch.is_grad_enabled() else layer(history, ok, ov))
            caches.append((ok, ov, hk, hv))
        return future, tuple(caches), conditions

    def joint_denoise_core(self, force_noise, action_noise, tf, ta, sample, prepared):
        _, static, conditions = prepared
        context, cmask, _, hvalid = conditions[:4]
        batch = len(force_noise)
        tv = torch.zeros(batch, device=self.device, dtype=torch.float32)
        force, fm = self.dit['force'].prepare_future(force_noise, tf, tv, context)
        action, _, am, text, text_mask, freq = self.dit['action'].prepare(action_noise, ta, context, cmask)
        source_time = torch.cat((sinusoidal_embedding_1d(256, tv),
                                 sinusoidal_embedding_1d(256, tf.float())), -1)
        source_time = source_time.to(self.dit['action_source_time'][0].weight)
        am = am + self.dit['action_source_time'](source_time).unflatten(-1, (6, self.dit['action'].hidden_dim))
        force_valid = (~sample['force_is_pad'].to(self.device, torch.bool)
                       & sample['wrench_valid'].to(self.device, torch.bool).any(-1))
        action_valid = ~sample['action_is_pad'].to(self.device, torch.bool)
        heads = self.dit['action'].num_heads
        for i, (force_block, action_block) in enumerate(zip(self.dit['force'].blocks, self.dit['action'].blocks)):
            ok, ov, hk, hv = static[i]
            def layer(force, action, ok, ov, hk, hv, force_block=force_block, action_block=action_block):
                fq, fk, fv, fs = force_block.qkv(force, fm)
                aq, ak, av, residual, gate, shift, scale, mg, _ = self.ops._build_expert_attention_io(
                    self.dit['action'], action_block, action, freq, am)
                aq, ak, av = [_split(x, heads) for x in (aq, ak, av)]
                keys = torch.cat((ok, hk, fk), 2)
                values = torch.cat((ov, hv, fv), 2)
                valid = torch.cat((torch.ones(batch, ok.shape[2], device=self.device, dtype=torch.bool),
                                   hvalid, force_valid), 1)
                fnext = force_block(_attend(fq, keys, values, valid), fs)
                mixed = _attend(aq, torch.cat((keys, ak), 2), torch.cat((values, av), 2),
                                 torch.cat((valid, action_valid), 1)).transpose(1, 2).flatten(2)
                anext = self.ops._apply_expert_post_block_tensor(action_block, residual, mixed,
                    gate, shift, scale, mg, text, text_mask)
                return torch.where(force_valid[..., None], fnext, 0), torch.where(action_valid[..., None], anext, 0)
            args = (force, action, ok, ov, hk, hv)
            force, action = (checkpoint(layer, *args, use_reentrant=False)
                            if self.gradient_checkpointing and torch.is_grad_enabled() else layer(*args))
        return self.dit['force'].head(force), self.dit['action'].post(action)

    def training_loss(self, sample):
        visual = sample['visual_features'].to(self.device, self.torch_dtype)
        video = self.dit['video']
        if visual.shape[1:] != (5, video.tokens_per_frame, video.feature_dim):
            raise ValueError('V-JEPA video cache must contain context plus four future tubelets')
        action = sample['action'].to(self.device, self.torch_dtype)
        force = sample['wrench_future'].to(self.device, self.torch_dtype)
        if action.shape[1:] != (32, 8) or force.shape[1:] != (32, 6):
            raise ValueError('Action32x8 and Force32x6 required')
        prepared = self.prepare_context(visual[:, 0], sample)
        tf, ta = [scheduler.sample_training_t(len(visual), self.device, self.torch_dtype)
                  for scheduler in (self.f_scheduler, self.a_scheduler)]
        noise_f, noise_a = torch.randn_like(force), torch.randn_like(action)
        fvalid = sample['wrench_valid'].to(self.device, torch.bool) & ~sample['force_is_pad'].to(self.device, torch.bool)[..., None]
        clean_f = torch.where(fvalid, force, 0)
        pf, pa = self.joint_denoise_core(self.f_scheduler.add_noise(clean_f, noise_f, tf),
            self.a_scheduler.add_noise(action, noise_a, ta), tf, ta, sample, prepared)
        # Future teachers are read here only, after all action/force predictions.
        target = F.layer_norm(visual[:, 1:].float(), (video.feature_dim,))
        lv = F.l1_loss(prepared[0].float(), target)
        la = temporal_mse(pa, self.a_scheduler.training_target(action, noise_a, ta),
                          ~sample['action_is_pad'].to(self.device, torch.bool), self.a_scheduler, ta)
        delta = torch.where(fvalid, pf.float() - self.f_scheduler.training_target(clean_f, noise_f, tf).float(), 0)
        lf = ((delta.square().sum((1, 2)) / fvalid.sum((1, 2)).clamp_min(1))
              * self.f_scheduler.training_weight(tf)).mean()
        loss = self.lambda_future * lv + la + lf
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite JEPA future/Force/Action objective')
        return loss, dict(loss_future_l1=float(lv.detach()), loss_future_weighted=float((self.lambda_future*lv).detach()),
                          loss_force_fm=float(lf.detach()), loss_action_fm=float(la.detach()))

    @torch.no_grad()
    def infer_joint(self, sample, *, num_inference_steps=None, seed=42, force_seed=43, **unused):
        steps = self.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        if steps < 1:
            raise ValueError('Positive inference step count required')
        modes = {module: module.training for module in self.modules()}
        self.eval()
        try:
            visual = sample['visual_features']
            if visual.ndim == 3:
                sample = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) and value.ndim else value
                          for key, value in sample.items()}
            visual = sample['visual_features'].to(self.device, self.torch_dtype)
            batch = len(visual)
            # Future labels and label validity must never condition an inference call.
            sample = dict(sample, action_is_pad=torch.zeros(batch, 32, device=self.device, dtype=torch.bool),
                          force_is_pad=torch.zeros(batch, 32, device=self.device, dtype=torch.bool),
                          wrench_valid=torch.ones(batch, 32, 6, device=self.device, dtype=torch.bool))
            prepared = self.prepare_context(visual[:, 0], sample)
            def noise(shape, noise_seed):
                return torch.randn(shape, generator=torch.Generator().manual_seed(noise_seed)).to(self.device, self.torch_dtype)
            force, action = noise((batch, 32, 6), force_seed), noise((batch, 32, 8), seed)
            schedules = [scheduler.build_inference_schedule(steps, self.device, self.torch_dtype)
                         for scheduler in (self.f_scheduler, self.a_scheduler)]
            for index in range(steps):
                tf, ta = [pair[0][index].expand(batch) for pair in schedules]
                pf, pa = self.joint_denoise_core(force, action, tf, ta, sample, prepared)
                force = self.f_scheduler.step(pf, schedules[0][1][index], force)
                action = self.a_scheduler.step(pa, schedules[1][1][index], action)
            return dict(action=action, force=force, future_features=prepared[0])
        finally:
            for module, mode in modes.items():
                module.training = mode

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = dict(schema=self.checkpoint_schema, manifest=self.manifest, model=self.state_dict(), step=step)
        if optimizer is not None:
            payload['optimizer'] = optimizer.state_dict()
        path = Path(path)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if payload.get('schema') != self.checkpoint_schema or payload.get('manifest') != self.manifest:
            raise ValueError('Checkpoint schema/teacher/data/experiment identity differs')
        self.load_state_dict(payload['model'], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(payload['optimizer'])
        return payload.get('step')


def create_dense_feature_joint(**kwargs):
    device = kwargs.pop('device', 'cuda')
    dtype = kwargs.pop('model_dtype', kwargs.pop('torch_dtype', torch.bfloat16))
    return DenseFeatureJointModel(**kwargs).to(device=device, dtype=dtype)
