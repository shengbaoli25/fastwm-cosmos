"""Action-guided observed JEPA features modify only Action's Force memory.

The Force branch keeps the original no-action dependency graph. A clean Force
label pass reuses the current model weights under no_grad; it needs no teacher
checkpoint. Future wrench labels are consumed exclusively by the training losses.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .noaction import NoActionJointModel, attend
from fastwam.models.wan22.wan_video_dit import sinusoidal_embedding_1d


SCHEMA = 'cosmos-jepa-action-force-latent-joint8-v1'


class ObservationConditions(tuple):
    """Preserve the baseline five-tensor API while carrying observations only."""
    def __new__(cls, baseline, observation):
        result = super().__new__(cls, baseline)
        result.observation = observation
        return result


def _split(x, heads):
    return x.unflatten(-1, (heads, x.shape[-1] // heads)).transpose(1, 2)


def spatial_query_redundancy(query, key, valid, *, margin=.95):
    """Weak duplicate-routing penalty on actual per-camera spatial attention.

    query/key are the same projected [B,H,Q/P,D] tensors used by SDPA. For
    each camera separately, condition the spatial distribution on that view;
    camera preference alone therefore cannot satisfy this objective. Only
    pairwise cosine similarities above ``margin`` are penalized. Shared
    evidence and overlap remain allowed; there is no orthogonality target.

    All arithmetic after the projections is FP32, including under autocast,
    and is differentiable into Q/K. Views with fewer than two valid patches
    cannot support distinct spatial reads and are excluded. Views are weighted
    equally, including ones with small global attention mass: no learnable
    camera mass can switch this penalty off. This regularizer does not supply
    contact labels and cannot guarantee task-relevant attention or escape an
    exactly symmetric, identical-query stationary point by itself.
    """
    if not math.isfinite(margin) or not 0 <= margin < 1:
        raise ValueError('Query redundancy margin must be finite in [0,1)')
    if (query.ndim != 4 or key.ndim != 4 or query.shape[:2] != key.shape[:2]
            or query.shape[-1] != key.shape[-1] or key.shape[-2] % 2
            or valid.dtype != torch.bool or valid.shape != (key.shape[0], key.shape[-2])):
        raise ValueError('Expected compatible Q/K and boolean two-view patch validity')
    batch, heads, queries, width = query.shape
    patches = key.shape[-2] // 2
    if min(heads, queries, width, patches) < 1:
        raise ValueError('Nonempty attention geometry required')
    with torch.autocast(device_type=query.device.type, enabled=False):
        logits = torch.matmul(query.float(), key.float().transpose(-2, -1)) / math.sqrt(width)
        # [B,H,view,Q,patch] follows main-then-wrist row-major encoder order.
        logits = logits.reshape(batch, heads, queries, 2, patches).transpose(2, 3)
        mask = valid.reshape(batch, 2, patches)[:, None, :, None]
        # A finite floor also makes an entirely masked view well-defined;
        # multiply by the mask afterwards and exclude that view from the mean.
        logits = logits.masked_fill(~mask, -torch.finfo(torch.float32).max)
        spatial = logits.softmax(-1) * mask
        unit = F.normalize(spatial, dim=-1, eps=1e-12)
        similarity = (unit @ unit.transpose(-2, -1)).clamp(0., 1.)
        pairs = torch.ones(queries, queries, device=query.device, dtype=torch.bool).triu(1)
        eligible = (mask.sum(-1) >= 2)[..., None] & pairs
        eligible = eligible.expand_as(similarity)
        denominator = eligible.sum().clamp_min(1)
        excess = F.relu(similarity - margin)
        loss = torch.where(eligible, excess, 0.).sum() / denominator
        mean = torch.where(eligible, similarity, 0.).sum() / denominator
        active = (eligible & (similarity > margin)).sum() / denominator
    return dict(loss=loss, mean_cosine=mean.detach(), active_fraction=active.detach())


class VisualValueRead(nn.Module):
    """No query residual: output values must pass through observed patches."""
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.query_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width),
                                nn.GELU(), nn.Linear(2 * width, width))

    def project(self, values, positions):
        return (_split(self.key(values + positions), self.heads),
                _split(self.value(values), self.heads))

    def forward(self, query, kv, valid, *, redundancy_margin=None):
        q = _split(self.query(self.query_norm(query)), self.heads)
        k, v = kv
        result = F.scaled_dot_product_attention(q, k, v,
            attn_mask=valid[:, None, None, :], dropout_p=0.)
        result = self.output(result.transpose(1, 2).flatten(2))
        result = result + self.ff(result)
        if redundancy_margin is None:
            return result
        # The original fast SDPA result remains the policy output. This small
        # training-only branch penalizes routing, not cosmetic query features.
        return result, spatial_query_redundancy(q, k, valid, margin=redundancy_margin)


class ActionGuidedVisualReader(nn.Module):
    def __init__(self, *, feature_dim, action_dim, state_dim, history_steps,
                 text_dim, force_hidden, width=256, query_count=8, depth=2, heads=4, horizon=32):
        super().__init__()
        if min(width, query_count, depth, heads) < 1 or width % heads:
            raise ValueError('Positive reader geometry and width divisible by heads required')
        self.feature_dim, self.width, self.horizon = feature_dim, width, horizon
        self.history_dim, self.history_steps = state_dim + 6, history_steps
        self.patch = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, width))
        self.camera = nn.Embedding(2, width)
        self.spatial = nn.Linear(2, width, bias=False)
        self.action = nn.Linear(horizon * action_dim, width)
        self.history = nn.Linear(history_steps * 2 * self.history_dim, width)
        self.proprio = nn.Linear(state_dim + 1, width)
        self.task = nn.Linear(text_dim, width)
        self.clock = nn.Sequential(nn.Linear(128, width), nn.SiLU(), nn.Linear(width, width))
        self.query_slots = nn.Parameter(torch.randn(query_count, width) / width**.5)
        self.readers = nn.ModuleList(VisualValueRead(width, heads) for _ in range(depth))
        self.horizon_slots = nn.Parameter(torch.randn(horizon, width) / width**.5)
        self.horizon_read = VisualValueRead(width, heads)
        self.feature_head = nn.Linear(width, force_hidden)
        self.wrench_head = nn.Linear(width, 6)

    def prepare_observation(self, patches, patch_valid, history, history_valid, proprio, task):
        if patches.ndim != 3 or patches.shape[-1] != self.feature_dim:
            raise ValueError('Observed JEPA patches must be [B,N,feature_dim]')
        batch, count, _ = patches.shape
        side = math.isqrt(count // 2)
        if count != 2 * side * side or side < 1:
            raise ValueError('JEPA patches must contain two equal square camera grids')
        if patch_valid.shape != patches.shape[:2] or not bool(patch_valid.any(1).all()):
            raise ValueError('At least one valid observed patch per example required')
        if not bool(torch.isfinite(patches[patch_valid]).all()):
            raise ValueError('Observed JEPA patches contain nonfinite values')
        patches = torch.where(patch_valid[..., None], patches, 0)
        values = self.patch(patches)
        coordinates = torch.linspace(-1, 1, side, device=patches.device, dtype=patches.dtype)
        row, col = torch.meshgrid(coordinates, coordinates, indexing='ij')
        xy = torch.stack((col, row), -1).reshape(1, side * side, 2).repeat(1, 2, 1)
        cameras = torch.arange(2, device=patches.device).repeat_interleave(side * side)
        position = self.spatial(xy) + self.camera(cameras)[None]
        if history.shape != (batch, self.history_steps, self.history_dim) or history_valid.shape != history.shape:
            raise ValueError('Query history shape differs from the baseline history contract')
        measured = torch.where(history_valid, history, 0)
        if not bool(torch.isfinite(measured).all()):
            raise ValueError('Measured history contains a nonfinite valid channel')
        history_input = torch.cat((measured, history_valid.to(measured.dtype)), -1).flatten(1)
        condition = self.history(history_input) + self.proprio(proprio) + self.task(task.mean(1))
        return dict(kv=tuple(reader.project(values, position) for reader in self.readers),
                    valid=patch_valid, condition=condition)

    def forward(self, action, ta, observation, *, query_redundancy_margin=None):
        condition = observation['condition'] + self.action(action.flatten(1))
        condition = condition + self.clock(sinusoidal_embedding_1d(128, ta).to(condition.dtype))
        query = self.query_slots[None] + condition[:, None]
        visual = None
        redundancy = []
        for reader, kv in zip(self.readers, observation['kv']):
            query_input = query if visual is None else query + visual
            if query_redundancy_margin is None:
                visual = reader(query_input, kv, observation['valid'])
            else:
                visual, penalty = reader(query_input, kv, observation['valid'],
                                         redundancy_margin=query_redundancy_margin)
                redundancy.append(penalty)
        # The horizon query is learned position only; no raw action skip enters U.
        slots = self.horizon_slots[None].expand(len(action), -1, -1)
        valid = torch.ones(visual.shape[:2], dtype=torch.bool, device=visual.device)
        u = self.horizon_read(slots, self.horizon_read.project(visual, torch.zeros_like(visual)), valid)
        result = dict(visual=u, features=self.feature_head(u), wrench=self.wrench_head(u))
        if redundancy:
            result['query_redundancy'] = {name: torch.stack([item[name] for item in redundancy]).mean()
                                        for name in ('loss', 'mean_cosine', 'active_fraction')}
        return result


class ActionForceResidual(nn.Module):
    def __init__(self, force_hidden, width, depth, injection_layers):
        super().__init__()
        layers = tuple(int(layer) for layer in injection_layers)
        if not layers or len(set(layers)) != len(layers) or any(layer < 1 or layer > depth for layer in layers):
            raise ValueError('Injection layers must be distinct 1-based model layer numbers')
        self.injection_layers = layers
        self.index = {layer - 1: index for index, layer in enumerate(layers)}
        self.input = nn.Sequential(nn.LayerNorm(force_hidden), nn.Linear(force_hidden, width))
        self.time = nn.Sequential(nn.Linear(3 * 64, width), nn.SiLU(), nn.Linear(width, width))
        self.layer = nn.Embedding(depth, width)
        self.output = nn.Sequential(nn.SiLU(), nn.Linear(width, force_hidden))
        # Exactly one zero initialization: projection weights remain nonzero.
        self.gates = nn.Parameter(torch.zeros(len(layers)))

    def clocks(self, tv, tf, ta, dtype):
        return self.time(torch.cat([sinusoidal_embedding_1d(64, t) for t in (tv, tf, ta)], -1).to(dtype))

    def forward(self, i, features, clocks):
        gate = self.gates[self.index[i]].tanh()
        layer = self.layer.weight[i]
        return gate * self.output(self.input(features) + clocks[:, None] + layer)


class JepaForceLatentModel(NoActionJointModel):
    checkpoint_schema = SCHEMA

    def __init__(self, net, *, observation_encoder=None, jepa_feature_dim=768,
                 reader_width=256, query_count=8, reader_depth=2, reader_heads=4,
                 injection_layers=(7, 14, 21, 28), latent_loss_weight=1.,
                 wrench_loss_weight=.1, latent_warmup_steps=2500, jepa_seed=47,
                 query_redundancy_weight=0., query_redundancy_margin=.95,
                 query_redundancy_warmup_steps=2500,
                 allow_preencoded_patches=False, **kwargs):
        super().__init__(net, **kwargs)
        if latent_loss_weight < 0 or wrench_loss_weight < 0:
            raise ValueError('Auxiliary loss weights must be nonnegative')
        self.latent_loss_weight = float(latent_loss_weight)
        self.wrench_loss_weight = float(wrench_loss_weight)
        if isinstance(latent_warmup_steps, bool) or int(latent_warmup_steps) != latent_warmup_steps or latent_warmup_steps < 0:
            raise ValueError('Latent warmup steps must be a nonnegative integer')
        self.latent_warmup_steps = int(latent_warmup_steps)
        if (not math.isfinite(query_redundancy_weight) or query_redundancy_weight < 0
                or not math.isfinite(query_redundancy_margin) or not 0 <= query_redundancy_margin < 1):
            raise ValueError('Nonnegative finite query redundancy weight and margin in [0,1) required')
        if (isinstance(query_redundancy_warmup_steps, bool)
                or not math.isfinite(query_redundancy_warmup_steps)
                or int(query_redundancy_warmup_steps) != query_redundancy_warmup_steps
                or query_redundancy_warmup_steps < 0):
            raise ValueError('Query redundancy warmup steps must be a nonnegative integer')
        self.query_redundancy_weight = float(query_redundancy_weight)
        self.query_redundancy_margin = float(query_redundancy_margin)
        self.query_redundancy_warmup_steps = int(query_redundancy_warmup_steps)
        self._auxiliary_step = 0
        self.allow_preencoded_patches = bool(allow_preencoded_patches)
        if observation_encoder is None and not allow_preencoded_patches:
            raise ValueError('Online JEPA encoder required unless preencoded mode is explicit')
        feature_dim = int(getattr(observation_encoder, 'feature_dim', jepa_feature_dim))
        force = self.dit['force']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(jepa_seed)
            self.dit['jepa_reader'] = ActionGuidedVisualReader(feature_dim=feature_dim,
                action_dim=self.action_dim, state_dim=self.state_dim, history_steps=force.history_steps,
                text_dim=force.text_encoder.in_features, force_hidden=force.hidden_dim,
                width=reader_width, query_count=query_count, depth=reader_depth, heads=reader_heads)
            self.dit['jepa_adapter'] = ActionForceResidual(force.hidden_dim, reader_width, self.depth, injection_layers)
        if observation_encoder is not None:
            self.dit['jepa_observation'] = observation_encoder
        identity = dict(getattr(observation_encoder, 'identity', {}) or {})
        self.manifest.update(schema=SCHEMA, joint_protocol=SCHEMA,
            jepa_force_latent=dict(feature_dim=feature_dim, reader_width=reader_width,
                query_count=query_count, reader_depth=reader_depth, reader_heads=reader_heads,
                injection_layers=list(injection_layers), seed=jepa_seed,
                encoder=identity, allow_preencoded_patches=self.allow_preencoded_patches,
                latent_loss_weight=self.latent_loss_weight, wrench_loss_weight=self.wrench_loss_weight,
                latent_warmup_steps=self.latent_warmup_steps,
                target='online current Force weights; clean force clocks 0; observed O/H/F attention only; final state',
                target_normalization='per-position LayerNorm without affine',
                visibility='Query: noisy action + clock + measured history/proprio/task + observed JEPA only',
                feedback='Action-only Force KV copy; no writeback to Video/Force/history',
                query_residual_bypass=False))
        # Keep the exact original manifest and state keys when disabled, so
        # historical checkpoints still load with strict identity checking.
        if self.query_redundancy_weight > 0:
            self.manifest['jepa_force_latent']['query_redundancy'] = dict(
                objective='per_camera_spatial_attention_cosine_hinge_v1',
                weight=self.query_redundancy_weight, margin=self.query_redundancy_margin,
                warmup_steps=self.query_redundancy_warmup_steps,
                reduction='mean unordered query pairs, eligible views, heads, layers, examples',
                views='equal view weights; skip fewer than two valid patches',
                scope='training_only; observed patch QK; no contact labels')
        # Runtime is metadata-only, but its identity must describe this checkpoint.
        self.dit['runtime'].bind_manifest(self.manifest)

    def configure_trainable_parameters(self):
        if 'jepa_observation' in self.dit:
            configure = getattr(self.dit['jepa_observation'], 'configure_trainable_parameters', None)
            if configure is None:
                raise ValueError('Observation encoder must explicitly restore its LoRA-only parameter policy')
            configure()

    def set_auxiliary_step(self, global_step):
        """The trainer supplies its restored global step before every loss call."""
        if isinstance(global_step, bool) or int(global_step) != global_step or global_step < 0:
            raise ValueError('Auxiliary step must be a nonnegative integer')
        self._auxiliary_step = int(global_step)

    @torch.no_grad()
    def clean_force_target(self, sample, conditions):
        """Label-only O/H/F pass, using current weights and no future video.

        This target differs from the full O/H/V/F main Force pass. It is never
        returned as input to the visual query, Action memory, or deployed policy.
        Main Force FM trains the reused parameters; this target has no gradient.
        """
        raw, context, _, history, hvalid = conditions
        first = sample['first_frame_latents'].to(self.device, self.torch_dtype)
        clean = sample['wrench_future'].to(self.device, self.torch_dtype)
        valid = sample['wrench_valid'].to(self.device, torch.bool)
        valid = valid & ~sample['force_is_pad'].to(self.device, torch.bool)[..., None]
        clean = torch.where(valid, clean, 0)
        if not bool(torch.isfinite(clean).all()):
            raise ValueError('Nonfinite valid clean force label')
        zero = torch.zeros(len(first), device=self.device, dtype=self.torch_dtype)
        prepared = self._video_prepare(first, zero, raw)
        observed = prepared['tokens']
        force, fm = self.dit['force'].prepare_future(clean, zero, zero, context)
        for i, block in enumerate(self.dit['force'].blocks):
            observed, history, static = self._static_layer(i, observed, history, prepared, hvalid)
            ok, ov, hk, hv = static
            fq, fk, fv, state = block.qkv(force, fm)
            key_valid = torch.cat((torch.ones(ok.shape[0], ok.shape[2], device=self.device, dtype=torch.bool),
                                   hvalid, torch.ones(fk.shape[0], fk.shape[2], device=self.device, dtype=torch.bool)), 1)
            force = block(attend(fq, torch.cat((ok, hk, fk), 2), torch.cat((ov, hv, fv), 2), key_valid), state)
        return force.detach()

    def _conditions(self, sample):
        baseline = super()._conditions(sample)
        if 'jepa_observed_patches' in sample:
            if not self.allow_preencoded_patches or 'jepa_observation' in sample:
                raise ValueError('Preencoded patches require explicit mode and cannot override online observations')
            patches = sample['jepa_observed_patches'].to(self.device, self.torch_dtype)
        else:
            if 'jepa_observation' not in self.dit or 'jepa_observation' not in sample:
                raise ValueError('Observed-only jepa_observation RGB is required')
            # Keep RGB precision until the encoder has applied normalization.
            patches = self.dit['jepa_observation'](sample['jepa_observation'].to(device=self.device))
            # Official encoder normalization can return FP32 under BF16 autocast.
            # This cast preserves LoRA gradients and matches the reader weights.
            patches = patches.to(device=self.device, dtype=self.torch_dtype)
        valid = sample.get('jepa_patch_valid')
        valid = (torch.ones(patches.shape[:2], device=self.device, dtype=torch.bool) if valid is None
                 else valid.to(self.device, torch.bool))
        history = sample['history'].to(self.device, self.torch_dtype)
        hvalid = sample['history_channel_valid'].to(self.device, torch.bool)
        hvalid = hvalid & ~sample['history_is_pad'].to(self.device, torch.bool)[..., None]
        proprio = sample['proprio'].to(self.device, self.torch_dtype)
        proprio = proprio[:, 0] if proprio.ndim == 3 else proprio
        observation = self.dit['jepa_reader'].prepare_observation(patches, valid, history, hvalid, proprio, baseline[0])
        return ObservationConditions(baseline, observation)

    def _action_force_context(self, action, tv, tf, ta, conditions):
        if not isinstance(conditions, ObservationConditions):
            raise ValueError('JEPA model conditions require observation context')
        if self.query_redundancy_weight > 0 and self.training:
            context = self.dit['jepa_reader'](action, ta, conditions.observation,
                query_redundancy_margin=self.query_redundancy_margin)
        else:
            context = self.dit['jepa_reader'](action, ta, conditions.observation)
        context['clocks'] = self.dit['jepa_adapter'].clocks(tv, tf, ta, action.dtype)
        return context

    def _action_force_kv(self, i, force, fm, fk, fv, action_force_context):
        adapter = self.dit['jepa_adapter']
        if i not in adapter.index:
            return fk, fv
        delta = adapter(i, action_force_context['features'], action_force_context['clocks'])
        # This private copy is never returned as the Force state or put in static KV.
        _, action_fk, action_fv, _ = self.dit['force'].blocks[i].qkv(force + delta, fm)
        return action_fk, action_fv

    def _auxiliary_training_loss(self, sample, auxiliary):
        if 'force_teacher_features' in sample or 'force_teacher_valid' in sample:
            raise ValueError('External teacher features are unsupported; use the online clean Force label pass')
        context = auxiliary['action_force_context']
        prediction, wrench = context['features'], context['wrench']
        channel_valid = sample['wrench_valid'].to(self.device, torch.bool)
        channel_valid = channel_valid & ~sample['force_is_pad'].to(self.device, torch.bool)[..., None]
        target = self.clean_force_target(sample, auxiliary['conditions']).float()
        token_valid = channel_valid.all(-1)
        pred = F.layer_norm(prediction.float(), (prediction.shape[-1],))
        target = F.layer_norm(target, (target.shape[-1],))
        per_token = (pred - target).square().mean(-1)
        latent_loss = torch.where(token_valid, per_token, 0).sum() / token_valid.sum().clamp_min(1)
        target_force = sample['wrench_future'].to(self.device).float().detach()
        target_force = torch.where(channel_valid, target_force, 0)
        difference = F.smooth_l1_loss(wrench.float(), target_force, reduction='none')
        wrench_loss = torch.where(channel_valid, difference, 0).sum() / channel_valid.sum().clamp_min(1)
        ramp = 1. if self.latent_warmup_steps == 0 else min(1., self._auxiliary_step / self.latent_warmup_steps)
        latent_weight = self.latent_loss_weight * ramp
        loss = latent_weight * latent_loss + self.wrench_loss_weight * wrench_loss
        metrics = dict(loss_force_feature=float(latent_loss.detach()),
                       loss_query_wrench=float(wrench_loss.detach()),
                       force_feature_weight=latent_weight, auxiliary_step=float(self._auxiliary_step))
        if self.query_redundancy_weight > 0:
            if self.training and 'query_redundancy' not in context:
                raise RuntimeError('Enabled training query redundancy statistics are missing')
            if 'query_redundancy' in context:
                redundancy = context['query_redundancy']
                ramp = (1. if self.query_redundancy_warmup_steps == 0 else
                        min(1., self._auxiliary_step / self.query_redundancy_warmup_steps))
                weight = self.query_redundancy_weight * ramp
                loss = loss + weight * redundancy['loss']
                metrics.update(loss_query_redundancy=float(redundancy['loss'].detach()),
                    query_redundancy_weight=weight,
                    query_redundancy_cosine=float(redundancy['mean_cosine']),
                    query_redundancy_active_fraction=float(redundancy['active_fraction']))
        return loss, metrics


def create_jepa_force_latent_cosmos(*, observation_encoder=None, device='cuda',
                                  reader_width=256, query_count=8, reader_depth=2,
                                  reader_heads=4, injection_layers=(7, 14, 21, 28),
                                  latent_loss_weight=1., wrench_loss_weight=.1,
                                  latent_warmup_steps=2500, jepa_seed=47,
                                  query_redundancy_weight=0., query_redundancy_margin=.95,
                                  query_redundancy_warmup_steps=2500,
                                  allow_preencoded_patches=False, jepa_feature_dim=768, **kwargs):
    """Reuse the verified official-asset factory, then add a checkpoint-isolated side path."""
    from .assets import create_noaction_joint_cosmos
    if observation_encoder is not None and not isinstance(observation_encoder, nn.Module):
        if not isinstance(observation_encoder, Mapping):
            raise ValueError('observation_encoder must be a module or factory configuration')
        from .jepa_observation import create_jepa_observation
        # B/L construction initializes different-sized temporary pretrained
        # weights before loading them. Keep that work and LoRA initialization
        # from shifting the baseline's subsequent training-noise RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(jepa_seed)
            observation_encoder = create_jepa_observation(**dict(observation_encoder), device='cpu', dtype=torch.float32)
    model_kwargs = dict(observation_encoder=observation_encoder, jepa_feature_dim=jepa_feature_dim,
        reader_width=reader_width, query_count=query_count, reader_depth=reader_depth,
        reader_heads=reader_heads, injection_layers=injection_layers,
        latent_loss_weight=latent_loss_weight, wrench_loss_weight=wrench_loss_weight,
        latent_warmup_steps=latent_warmup_steps, jepa_seed=jepa_seed,
        query_redundancy_weight=query_redundancy_weight,
        query_redundancy_margin=query_redundancy_margin,
        query_redundancy_warmup_steps=query_redundancy_warmup_steps,
        allow_preencoded_patches=allow_preencoded_patches)
    model = create_noaction_joint_cosmos(device=device, _model_class=JepaForceLatentModel,
                                        _model_kwargs=model_kwargs, **kwargs)
    model.configure_trainable_parameters()
    return model
