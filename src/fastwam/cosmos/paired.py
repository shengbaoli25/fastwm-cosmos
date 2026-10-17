"""RealAction with a paired visual bottleneck and bidirectional Force exchange."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fastwam.forcewm.model import ForcePrediction
from .model import CosmosForceWM
from .regions import PairedRegionReader, RegionForceBridge, view_grid


PAIRED_PROTOCOL = 'paired-regions-generated-refresh-spatial-writeback-v1'


class PairedCosmosForceWM(CosmosForceWM):
    checkpoint_schema = 'fastwam-cosmos-paired-regions28-v1'

    def __init__(self, *args, region_pairs=32, region_width=512, region_heads=8,
                 region_layer=13, region_seed=46, region_reconstruction_weight=.05,
                 region_diversity_weight=.001, region_writeback=True, **kwargs):
        super().__init__(*args, **kwargs)
        if (not 0 <= region_layer < self.depth or region_pairs < 1
                or region_reconstruction_weight < 0 or region_diversity_weight < 0):
            raise ValueError('Invalid paired region geometry/loss configuration')
        self.region_layer = region_layer
        self.region_reconstruction_weight = region_reconstruction_weight
        self.region_diversity_weight = region_diversity_weight
        self.region_writeback = region_writeback
        self.base_manifest = copy.deepcopy(self.manifest)
        config = dict(protocol=PAIRED_PROTOCOL, pairs_per_view=region_pairs,
            width=region_width, heads=region_heads, read_layer_zero_based=region_layer,
            seed=region_seed, reconstruction_weight=region_reconstruction_weight,
            diversity_weight=region_diversity_weight, spatial_writeback=region_writeback,
            basic_queries=2*region_pairs*2, future_region_features=2*2*region_pairs*2,
            temporal_mapping='two future latent groups; no per-RGB-frame localization',
            auxiliary='generated-feature reconstruction through pre-fusion slots; no contact annotations')
        self.manifest = dict(self.manifest, schema='cosmos-paired-regions28-v1', paired_regions=config,
            memory='generated paired Video/Force refresh differentiable for WM loss; detached and fixed for AE',
            initialization='official Cosmos; original RealAction random modules; independently seeded paired reader/bridges',
            loss='VideoFM + ActionFM + ForceMSE + weighted generated-feature reconstruction and target diversity')
        self.dit['runtime'].bind_manifest(self.manifest)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(region_seed)
            self.dit['regions'] = PairedRegionReader(self.video.hidden_dim,
                pairs=region_pairs, width=region_width, heads=region_heads)
            self.dit['region_bridges'] = nn.ModuleList([
                RegionForceBridge(self.video.hidden_dim, self.dit['force'].hidden_dim,
                                  region_width, region_heads)
                for _ in range(self.depth-region_layer)])
        if not region_writeback:
            # This ablation still exchanges region/Force features, but must not
            # advertise a learned Video writeback or optimize unused parameters.
            for bridge in self.dit['region_bridges']:
                bridge.video_out.requires_grad_(False)
                bridge.video_gate.requires_grad_(False)
        self._region_trace = None
        self._region_auxiliary = None
        self._region_loss_requested = False

    def _paired_video(self, latents, clock, raw, context, cmask, sample, action, apad,
                      *, collect=False):
        prepared = self.video.prepare(latents, clock.float()/1000., raw,
            candidate_actions=self.dit['controls'](action.detach(), apad),
            fps=self.fps, context_is_projected=True)
        fmodel = self.dit['force']
        f, fmask, valid, wpad, first_w = fmodel.prepare(
            history=sample['history'].to(self.device, self.torch_dtype),
            action_condition=action.detach(), context=context.detach(), context_mask=cmask,
            history_is_pad=sample['history_is_pad'].to(self.device, torch.bool),
            candidate_is_pad=apad,
            history_channel_valid=sample['history_channel_valid'].to(self.device, torch.bool),
            observation_latents=sample['first_frame_latents'].to(self.device, self.torch_dtype))
        x = prepared['tokens']
        count = prepared['tokens_per_frame']
        vmask = torch.ones(len(x), 1, 1, x.shape[1]-count, device=x.device, dtype=torch.bool)
        layers, flayers = [], []
        regions = maps = coords = auxiliary = None
        for i, fblock in enumerate(fmodel.blocks):
            def video_step(tokens, layer=i):
                q, k, v, state = self.video.block_qkv(layer, tokens, prepared)
                return self.video.block_finish(layer,
                    F.scaled_dot_product_attention(q, k, v, dropout_p=0.), state, prepared)
            use_checkpoint = torch.is_grad_enabled() and self.gradient_checkpointing
            x = checkpoint(video_step, x, use_reentrant=False) if use_checkpoint else video_step(x)
            if i == self.region_layer:
                # First read precedes ANY Force->Video writeback in this pass.
                # The fixed maps remain attributable to this visual read even
                # while the associated feature vectors subsequently fuse.
                future = view_grid(x, prepared['grid'])
                reader = self.dit['regions']
                regions, maps, coords = (checkpoint(reader, future, action, apad, use_reentrant=False)
                    if use_checkpoint else reader(future, action, apad))
                if collect and self._region_loss_requested:
                    auxiliary = reader.auxiliary_loss(regions, maps, future)

            def force_step(tokens, features, block=fblock):
                return torch.where(valid[..., None],
                    block(tokens, features, fmask, vmask, first_w), 0)
            # Preserve all dense future tokens; this experiment opens the
            # generated-refresh gradient, not the old detached memory path.
            f = (checkpoint(force_step, f, x[:, count:], use_reentrant=False)
                 if use_checkpoint else force_step(f, x[:, count:]))
            if regions is not None:
                bridge = self.dit['region_bridges'][i-self.region_layer]
                def exchange(region_tokens, force_tokens, read_maps, block=bridge):
                    return block(region_tokens, force_tokens, read_maps, fmask, valid)
                regions, f, delta = (checkpoint(exchange, regions, f, maps, use_reentrant=False)
                    if use_checkpoint else exchange(regions, f, maps))
                if self.region_writeback:
                    x = torch.cat((x[:, :count], x[:, count:] + delta), 1)
            if collect:
                layers.append(x[:, count:].detach())
                flayers.append(f[:, first_w:])
        force = ForcePrediction(fmodel.head(f[:, first_w:]), tuple(flayers), wpad) if collect else None
        trace = None if not collect else dict(attention=maps.detach(), xy=coords.detach(),
            read_layer_zero_based=self.region_layer, grid=prepared['grid'])
        return self.video.finish(x, prepared), tuple(layers), force, trace, auxiliary

    def _supervised_video(self, sample, noisy, clock, raw, context, cmask, action, apad):
        return self._paired_video(noisy, clock, raw, context, cmask, sample, action, apad)[0]

    @torch.no_grad()
    def generate_video_memory(self, first, raw_context, action_condition, action_is_pad,
                              *, sample, context, cmask):
        x = self.dit['runtime'].normal((*first.shape[:2], 3, *first.shape[3:])).to(self.device, self.torch_dtype)
        x[:, :, :1] = first.detach()
        times, deltas = self.v_scheduler.build_inference_schedule(self.generated_video_steps, self.device, x.dtype)
        for t, dt in zip(times, deltas):
            velocity = self._paired_video(x, t.expand(len(x)), raw_context, context,
                cmask, sample, action_condition, action_is_pad)[0]
            x = self.v_scheduler.step(velocity, dt, x)
            x[:, :, :1] = first.detach()
        return x.detach()

    def _world_models(self, sample, first, raw, context, cmask, action, apad):
        x = self.generate_video_memory(first, raw.detach(), action, apad,
                                      sample=sample, context=context, cmask=cmask)
        zero = torch.zeros(len(x), device=self.device, dtype=self.torch_dtype)
        _, layers, force, trace, auxiliary = self._paired_video(x, zero, raw, context,
            cmask, sample, action, apad, collect=True)
        self._region_trace, self._region_auxiliary = trace, auxiliary
        pad = torch.zeros(layers[0].shape[:2], device=self.device, dtype=torch.bool)
        return (x, layers, pad), force

    def training_loss(self, sample):
        self._region_auxiliary = None
        self._region_loss_requested = True
        try:
            loss, metrics = super().training_loss(sample)
        finally:
            self._region_loss_requested = False
        if self._region_auxiliary is None:
            raise RuntimeError('Generated pre-fusion region reconstruction was not computed')
        reconstruction, diversity = self._region_auxiliary
        loss = loss + self.region_reconstruction_weight*reconstruction + self.region_diversity_weight*diversity
        metrics.update(loss_region_reconstruction=float(reconstruction.detach()),
                       loss_region_diversity=float(diversity.detach()))
        self._region_auxiliary = None
        return loss, metrics

    @torch.no_grad()
    def infer_with_regions(self, sample, *, action_condition, **kwargs):
        self._region_trace = None
        result = super().infer_forcewm(sample, action_condition=action_condition, **kwargs)
        if self._region_trace is None:
            raise RuntimeError('No paired-region trace captured')
        result['regions'] = self._region_trace
        return result
