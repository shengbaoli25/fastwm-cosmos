"""Independent spatial queries supervised by Force and Action task losses.

There are Q query identities TOTAL across both cameras, shared over the two
future latent groups. No paired endpoints, feature decoder or auxiliary loss.
Only explicit clean controls enter the world models and the region reader.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fastwam.forcewm.model import Attention, ForcePrediction
from fastwam.models.wan22.wan_video_dit import flash_attention
from .model import CosmosForceWM
from .regions import merge_views, region_force_mask, view_grid


class RegionQueryReader(nn.Module):
    """Content-adaptive soft local reads with inspectable joint-camera maps."""

    def __init__(self, video_dim, *, count=32, width=512, heads=8):
        super().__init__()
        if count < 1 or width % heads:
            raise ValueError('Positive query count and divisible width required')
        self.count, self.width, self.heads = count, width, heads
        self.queries = nn.Parameter(torch.randn(count, width) / math.sqrt(width))
        self.time = nn.Parameter(torch.randn(2, width) / math.sqrt(width))
        self.action = nn.Sequential(nn.Linear(32 * 14, width), nn.SiLU(), nn.Linear(width, width))
        self.video_norm = nn.LayerNorm(video_dim)
        self.position = nn.Linear(3, width)
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(video_dim, width)
        self.v = nn.Linear(video_dim, width)
        self.out = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        # A bounded soft locality bias. The centre comes from a coarse visual
        # read, so it can move with image content, not just Action/time.
        self.log_radius = nn.Parameter(torch.full((count, 2), math.log(.25)))

    def forward(self, future, action, action_is_pad):
        b, times, views, height, width, _ = future.shape
        if ((times, views) != (2, 2) or action.shape != (b, 32, 7)
                or action_is_pad.shape != (b, 32) or action_is_pad.dtype != torch.bool):
            raise ValueError('Two future groups/views and explicit clean Action32 required')
        valid = (~action_is_pad)[..., None].expand_as(action)
        clean = torch.where(valid, action.detach(), 0)
        if not bool(torch.isfinite(clean).all()):
            raise ValueError('Valid clean Action must be finite')
        condition = self.action(torch.cat((clean, valid.to(clean.dtype)), -1).flatten(1))
        queries = self.queries[None, None] + self.time[None, :, None] + condition[:, None, None]
        pixels = self.video_norm(future.flatten(2, 4)).reshape(b * times, views * height * width, -1)
        y, x = torch.meshgrid((torch.arange(height, device=future.device).float() + .5) / height,
                              (torch.arange(width, device=future.device).float() + .5) / width,
                              indexing='ij')
        local = torch.stack((x, y), -1).flatten(0, 1)
        # Feature order is [view, row, column]; global x is for locality only.
        xy = torch.cat((local * local.new_tensor([.5, 1.]),
                        local * local.new_tensor([.5, 1.]) + local.new_tensor([.5, 0.])))
        camera = torch.arange(views, device=future.device).repeat_interleave(height * width).float()
        position = torch.cat((xy, camera[:, None]), -1)
        split = lambda value: value.unflatten(-1, (self.heads, self.width // self.heads)).transpose(1, 2)
        q = split(self.q(queries.reshape(b * times, self.count, self.width)))
        k = split(self.k(pixels) + self.position(position.to(pixels.dtype))[None])
        logits = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(self.width // self.heads)
        coarse = logits.softmax(-1).mean(1)
        centers = coarse @ xy
        radius = self.log_radius.exp().clamp(.05, 1.)
        distance = ((xy[None, None] - centers[:, :, None]) / radius[None, :, None]).square().sum(-1)
        attention = (logits - .5 * distance[:, None]).softmax(-1)
        values = attention.to(pixels.dtype) @ split(self.v(pixels))
        values = values.transpose(1, 2).flatten(2)
        # No query/clean-Action residual is copied into region values: the
        # selected visual features must carry the information sent downstream.
        tokens = self.norm(self.out(values)).reshape(b, times, self.count, self.width)
        maps = attention.mean(1).reshape(b, times, self.count, views, height, width)
        return tokens, maps


class RegionQueryBridge(nn.Module):
    """Region/Force self-attention with spatial Video writeback."""

    def __init__(self, video_dim, force_dim, width, heads, gate=.01):
        super().__init__()
        self.force_in = nn.Linear(force_dim, width)
        self.force_out = nn.Linear(width, force_dim)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.video_out = nn.Linear(width, video_dim, bias=False)
        self.force_gate = nn.Parameter(torch.tensor(float(gate)))
        self.video_gate = nn.Parameter(torch.tensor(float(gate)))
        self.heads = heads

    def forward(self, visual, force, maps, force_mask, valid):
        flat = visual.flatten(1, 2)
        joined = torch.cat((flat, self.force_in(force)), 1)
        blocked = (~region_force_mask(flat.shape[1], force_mask, valid)[:, 0]).repeat_interleave(self.heads, 0)
        norm = self.norm(joined)
        update, _ = self.attention(norm, norm, norm, attn_mask=blocked, need_weights=False)
        fused = joined + update
        fused = fused + self.ff(fused)
        regions = fused[:, :flat.shape[1]].reshape_as(visual)
        next_force = force + self.force_gate * self.force_out(fused[:, flat.shape[1]:] - joined[:, flat.shape[1]:])
        next_force = torch.where(valid[..., None], next_force, 0)
        b, times, count, views, height, width = maps.shape
        assignment = maps.flatten(3)
        assignment = assignment / assignment.sum(2, keepdim=True).clamp_min(1e-8)
        delta = self.video_out(regions - visual)
        writeback = (assignment.to(delta.dtype).transpose(-1, -2) @ delta).reshape(b, times, views, height, width, -1)
        return regions, next_force, self.video_gate * merge_views(writeback)


class ActionRegionRead(nn.Module):
    """Differentiable AE read: deliberately DOES NOT detach region features."""

    def __init__(self, action_dim, region_dim, width, heads):
        super().__init__()
        self.norm = nn.LayerNorm(action_dim, elementwise_affine=False)
        self.cross = Attention(action_dim, region_dim, width, heads)
        self.gate = nn.Parameter(torch.tensor(.01))

    def forward(self, action, regions, action_is_pad):
        delta = self.cross.attend(self.norm(action), self.cross.kv(regions.flatten(1, 2)))
        return action + torch.where((~action_is_pad)[..., None], self.gate * delta, 0)


class RegionQueryCosmosForceWM(CosmosForceWM):
    checkpoint_schema = 'fastwam-cosmos-region-queries32-v1'

    def __init__(self, *args, region_count=32, region_width=512, region_heads=8,
                 region_layer=13, region_seed=46, region_writeback=True, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= region_layer < self.depth:
            raise ValueError('Region layer outside Cosmos depth')
        self.region_layer, self.region_writeback = region_layer, region_writeback
        self.base_manifest = copy.deepcopy(self.manifest)
        config = dict(protocol='independent-regions-force-ae-task-supervision-v1',
            count_total=region_count, cameras=2, future_region_features=2 * region_count,
            width=region_width, heads=region_heads, read_layer_zero_based=region_layer,
            seed=region_seed, spatial_writeback=region_writeback,
            supervision='existing ForceMSE and ActionFM gradients; no auxiliary losses',
            maps='jointly normalized over both camera grids; two future latent groups',
            parameter_sharing=f'one bridge and one AE reader shared across layers {region_layer}..{self.depth - 1}',
            action_read='shared query reader on detached pre-fusion video; WMs isolated from AE')
        self.manifest = dict(self.manifest, schema='cosmos-region-queries32-v1', region_queries=config,
            memory='Force differentiates through generated refresh; AE trains shared reader on detached video; no denoising unroll',
            loss='VideoFM + ActionFM + ForceMSE, weights 1/1/1; no region reconstruction/diversity loss')
        self.dit['runtime'].bind_manifest(self.manifest)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(region_seed)
            self.dit['regions'] = RegionQueryReader(self.video.hidden_dim, count=region_count, width=region_width, heads=region_heads)
            self.dit['region_bridge'] = RegionQueryBridge(
                self.video.hidden_dim, self.dit['force'].hidden_dim, region_width, region_heads)
            self.dit['region_action_read'] = ActionRegionRead(
                self.dit['action'].head.in_features, region_width, region_width, region_heads)
        if not region_writeback:
            self.dit['region_bridge'].video_out.requires_grad_(False)
            self.dit['region_bridge'].video_gate.requires_grad_(False)
        self._region_trace = None

    def _contexts(self, sample):
        raw, context, mask = super()._contexts(sample)
        # Construct the AE context from detached video-owned text BEFORE
        # concatenation. Slicing an existing concatenation still backpropagates
        # zero gradients into the text projection (and AdamW could decay it).
        proprio = sample['proprio'].to(self.device, self.torch_dtype)
        if proprio.ndim == 3:
            proprio = proprio[:, 0]
        context = torch.cat((raw.detach().expand(context.shape[0], -1, -1),
                             self.proprio_encoder(proprio)[:, None]), 1)
        return raw, context, mask

    def _video(self, *args, observed=False, **kwargs):
        # Observation KV is another path from AE to Cosmos. Isolate it as
        # well as the generated memory, without allocating a backward graph.
        if observed:
            with torch.no_grad():
                return super()._video(*args, observed=True, **kwargs)
        return super()._video(*args, observed=False, **kwargs)

    def _region_video(self, latents, clock, raw, context, cmask, sample, action, apad, *, collect=False):
        prepared = self.video.prepare(latents, clock.float() / 1000., raw,
            candidate_actions=self.dit['controls'](action.detach(), apad), fps=self.fps, context_is_projected=True)
        fmodel = self.dit['force']
        force, fmask, valid, wpad, first_w = fmodel.prepare(
            history=sample['history'].to(self.device, self.torch_dtype), action_condition=action.detach(),
            context=context.detach(), context_mask=cmask,
            history_is_pad=sample['history_is_pad'].to(self.device, torch.bool), candidate_is_pad=apad,
            history_channel_valid=sample['history_channel_valid'].to(self.device, torch.bool),
            observation_latents=sample['first_frame_latents'].to(self.device, self.torch_dtype))
        x, count = prepared['tokens'], prepared['tokens_per_frame']
        vmask = torch.ones(len(x), 1, 1, x.shape[1] - count, device=x.device, dtype=torch.bool)
        layers, flayers = [], []
        regions = maps = ae_source = None
        for i, fblock in enumerate(fmodel.blocks):
            def video_step(tokens, layer=i):
                q, k, v, state = self.video.block_qkv(layer, tokens, prepared)
                return self.video.block_finish(layer, F.scaled_dot_product_attention(q, k, v, dropout_p=0.), state, prepared)
            use_checkpoint = torch.is_grad_enabled() and self.gradient_checkpointing
            x = checkpoint(video_step, x, use_reentrant=False) if use_checkpoint else video_step(x)
            if i == self.region_layer:
                future = view_grid(x, prepared['grid'])
                if collect:
                    ae_source = future.detach()
                reader = self.dit['regions']
                regions, maps = (checkpoint(reader, future, action, apad, use_reentrant=False)
                                 if use_checkpoint else reader(future, action, apad))
            def force_step(tokens, features, block=fblock):
                return torch.where(valid[..., None], block(tokens, features, fmask, vmask, first_w), 0)
            force = (checkpoint(force_step, force, x[:, count:], use_reentrant=False)
                     if use_checkpoint else force_step(force, x[:, count:]))
            if regions is not None:
                bridge = self.dit['region_bridge']
                regions, force, delta = (checkpoint(bridge, regions, force, maps, fmask, valid, use_reentrant=False)
                                         if use_checkpoint else bridge(regions, force, maps, fmask, valid))
                if self.region_writeback:
                    x = torch.cat((x[:, :count], x[:, count:] + delta), 1)
            if collect:
                layers.append(x[:, count:].detach())
                flayers.append(force[:, first_w:])
        predicted = ForcePrediction(fmodel.head(force[:, first_w:]), tuple(flayers), wpad) if collect else None
        trace = None if not collect else dict(attention=maps.detach(), read_layer_zero_based=self.region_layer, grid=prepared['grid'])
        return self.video.finish(x, prepared), tuple(layers), predicted, trace, ae_source

    def _supervised_video(self, sample, noisy, clock, raw, context, cmask, action, apad):
        return self._region_video(noisy, clock, raw, context, cmask, sample, action, apad)[0]

    @torch.no_grad()
    def generate_video_memory(self, first, raw_context, action_condition, action_is_pad, *, sample, context, cmask):
        x = self.dit['runtime'].normal((*first.shape[:2], 3, *first.shape[3:])).to(self.device, self.torch_dtype)
        x[:, :, :1] = first.detach()
        times, deltas = self.v_scheduler.build_inference_schedule(self.generated_video_steps, self.device, x.dtype)
        for time, delta in zip(times, deltas):
            velocity = self._region_video(x, time.expand(len(x)), raw_context, context, cmask,
                                          sample, action_condition, action_is_pad)[0]
            x = self.v_scheduler.step(velocity, delta, x)
            x[:, :, :1] = first.detach()
        return x.detach()

    def _world_models(self, sample, first, raw, context, cmask, action, apad):
        x = self.generate_video_memory(first, raw.detach(), action, apad, sample=sample, context=context, cmask=cmask)
        zero = torch.zeros(len(x), device=self.device, dtype=self.torch_dtype)
        _, layers, force, self._region_trace, ae_source = self._region_video(
            x, zero, raw, context, cmask, sample, action, apad, collect=True)
        # Same learned selection parameters receive Force and Action losses.
        # Detach the SOURCE, never the reader output: this prevents AE from
        # changing either WM while retaining gradients for query selection.
        reader = self.dit['regions']
        ae_regions, _ = (checkpoint(reader, ae_source, action, apad, use_reentrant=False)
                         if torch.is_grad_enabled() and self.gradient_checkpointing
                         else reader(ae_source, action, apad))
        pad = torch.zeros(layers[0].shape[:2], device=self.device, dtype=torch.bool)
        return (x, layers, pad, ae_regions), force

    def action_velocity(self, action, clock, context, cmask, observation_kv, memory, force, apad):
        expert = self.dit['action']
        # The text prefix was projected by Cosmos; only the final proprio
        # token is an AE-owned context path and may retain its gradient.
        ae_context = torch.cat((context[:, :-1].detach(), context[:, -1:]), 1)
        x, _, mod, text, mask, freq = expert.prepare(action, clock, ae_context, cmask)
        for i, (kobs, vobs) in enumerate(observation_kv):
            kobs, vobs = kobs.detach(), vobs.detach()
            block, mem = expert.blocks[i], self.dit['memories'][i]
            region_read = self.dit['region_action_read'] if i >= self.region_layer else None
            def run(tokens, observed_k, observed_v, regions, block=block, mem=mem, region_read=region_read, layer=i):
                q, k, v, res, gate, shift, scale, mg, _ = self.ops._build_expert_attention_io(expert, block, tokens, freq, mod)
                allowed = torch.cat((torch.ones(len(tokens), observed_k.shape[1], device=self.device, dtype=torch.bool), ~apad), 1)[:, None, None]
                mixed = flash_attention(q, torch.cat((observed_k, k), 1), torch.cat((observed_v, v), 1), expert.num_heads, ctx_mask=allowed)
                out = self.ops._apply_expert_post_block_tensor(block, res, mixed, gate, shift, scale, mg, text, mask)
                out = mem(out, memory[1][layer], force.layers[layer], memory[2], force.is_pad, apad)
                return out if region_read is None else region_read(out, regions, apad)
            regions = memory[3]
            x = (checkpoint(run, x, kobs, vobs, regions, use_reentrant=False)
                 if torch.is_grad_enabled() and self.gradient_checkpointing else run(x, kobs, vobs, regions))
        return expert.post(x), force

    @torch.no_grad()
    def infer_with_regions(self, sample, *, action_condition, **kwargs):
        self._region_trace = None
        result = super().infer_forcewm(sample, action_condition=action_condition, **kwargs)
        if self._region_trace is None:
            raise RuntimeError('No region-query trace captured')
        result['regions'] = self._region_trace
        return result
