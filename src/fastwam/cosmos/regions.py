"""Paired, spatially identifiable visual queries; no contact boxes required.

Query identity is shared across the two future latent groups, not across views.
Read maps are captured BEFORE Force fusion. They describe patch contributions,
not calibrated contact probabilities or independently observed RGB frames.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def view_grid(tokens, grid):
    """Split concatenated main-left / wrist-right columns without mixing them."""
    times, height, width = grid
    if times != 3 or width % 2 or tokens.shape[1] != times * height * width:
        raise ValueError('Three latent groups and two equally wide camera regions required')
    x = tokens.reshape(tokens.shape[0], times, height, 2, width // 2, -1)
    return x[:, 1:].permute(0, 1, 3, 2, 4, 5).contiguous()


def merge_views(future):
    b, times, views, height, width, dim = future.shape
    if times != 2 or views != 2:
        raise ValueError('Two future groups and two camera regions required')
    return future.permute(0, 1, 3, 2, 4, 5).reshape(b, times*height*views*width, dim)


def patch_coordinates(height, width, *, device):
    y = (torch.arange(height, device=device).float() + .5) / height
    x = (torch.arange(width, device=device).float() + .5) / width
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack((xx, yy), -1).reshape(-1, 2)


class PairedRegionReader(nn.Module):
    def __init__(self, video_dim, *, pairs=32, width=512, heads=8):
        super().__init__()
        if pairs < 1 or width % heads:
            raise ValueError('Positive pair count and divisible attention width required')
        self.pairs, self.width, self.heads = pairs, width, heads
        self.queries = nn.Parameter(torch.randn(2, pairs, 2, width) / math.sqrt(width))
        self.time = nn.Parameter(torch.randn(2, width) / math.sqrt(width))
        self.action = nn.Sequential(nn.Linear(32*14, width), nn.SiLU(), nn.Linear(width, width))
        self.video_norm = nn.LayerNorm(video_dim)
        self.position = nn.Linear(2, width)
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(video_dim, width)
        self.v = nn.Linear(video_dim, width)
        self.out = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.pair_mix = nn.Sequential(nn.Linear(2*width+2, 2*width), nn.GELU(),
                                      nn.Linear(2*width, 2*width))
        self.offset = nn.Linear(width, 2)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        # 32 pairs -> 4 rows x 8 columns; other counts remain useful ablations.
        rows = max(1, math.floor(math.sqrt(pairs / 2)))
        cols = math.ceil(pairs / rows)
        centers = [((i % cols + .5)/cols, (i // cols + .5)/rows) for i in range(pairs)]
        self.register_buffer('anchors', torch.tensor(centers))
        self.register_buffer('radius', torch.tensor([1.5/cols, 1.5/rows]))
        self.decode = nn.Linear(width, video_dim)

    def forward(self, future, action, action_is_pad):
        b, times, views, height, width, dim = future.shape
        if ((times, views) != (2, 2) or action.shape != (b, 32, 7)
                or action_is_pad.shape != (b, 32) or action_is_pad.dtype != torch.bool):
            raise ValueError('Paired reader needs two views/groups and explicit clean Action32')
        valid = (~action_is_pad)[..., None].expand_as(action)
        clean = torch.where(valid, action.detach(), 0)
        if not bool(torch.isfinite(clean).all()):
            raise ValueError('Valid clean action must be finite')
        condition = self.action(torch.cat((clean, valid.to(clean.dtype)), -1).flatten(1))
        queries = (self.queries[None, None] + self.time[None, :, None, None, None]
                   + condition[:, None, None, None, None])
        qflat = queries.reshape(b*times*views, self.pairs*2, self.width)
        pixels = future.reshape(b*times*views, height*width, dim)
        norm = self.video_norm(pixels)
        coords = patch_coordinates(height, width, device=future.device)
        k = self.k(norm) + self.position(coords.to(norm.dtype))[None]
        q = self.q(qflat)
        split = lambda x: x.unflatten(-1, (self.heads, self.width//self.heads)).transpose(1, 2)
        logits = split(q).float() @ split(k).float().transpose(-1, -2)
        logits = logits / math.sqrt(self.width//self.heads)
        target_q = queries[..., 1, :]
        anchors = torch.sigmoid(torch.logit(self.anchors)[None, None, None]
                                + self.offset(target_q).float())
        distance = ((coords - anchors[..., None, :]) / self.radius).square().sum(-1)
        bias = torch.stack((torch.zeros_like(distance), -.5*distance), -2)
        logits = logits + bias.reshape(b*times*views, 1, self.pairs*2, height*width)
        attention = logits.softmax(-1)
        values = attention.to(norm.dtype) @ split(self.v(norm))
        values = values.transpose(1, 2).flatten(2)
        tokens = self.norm(qflat + self.out(values)).reshape(b, times, views, self.pairs, 2, self.width)
        maps = attention.mean(1).reshape(b, times, views, self.pairs, 2, height, width)
        xy = maps.flatten(-2) @ coords
        relation = torch.cat((tokens.flatten(-2), xy[..., 1, :] - xy[..., 0, :]), -1)
        tokens = tokens + self.pair_mix(relation.to(tokens.dtype)).unflatten(-1, (2, self.width))
        return tokens, maps, xy

    def auxiliary_loss(self, tokens, maps, future):
        """Reconstruct detached normalized features through slots only.

        This is generated-feature self-distillation, not DINO/track supervision.
        No dense-feature bypass or future force label reaches the decoder.
        """
        b, t, v, p, _, height, width = maps.shape
        assignment = maps.reshape(b, t, v, p*2, height*width)
        assignment = assignment / assignment.sum(-2, keepdim=True).clamp_min(1e-8)
        decoded = self.decode(tokens.flatten(3, 4)).float()
        reconstructed = assignment.transpose(-1, -2) @ decoded
        target = F.layer_norm(future.detach().float(), (future.shape[-1],)).flatten(3, 4)
        recon = (reconstructed - target).square().mean()
        # Soft redundancy penalty on target endpoints only; never separates
        # the two endpoints of a pair, which can coincide at actual contact.
        target_maps = F.normalize(maps[..., 1, :, :].flatten(-2), dim=-1)
        overlap = target_maps @ target_maps.transpose(-1, -2)
        off = ~torch.eye(p, device=maps.device, dtype=torch.bool)
        diversity = overlap[..., off].square().mean() if p > 1 else overlap.sum()*0
        return recon, diversity


def region_force_mask(region_count, force_mask, valid):
    b, size = valid.shape
    if force_mask.shape != (b, 1, size, size) or valid.dtype != torch.bool:
        raise ValueError('Force validity and direct-edge mask required')
    mask = torch.ones(b, 1, region_count+size, region_count+size,
                      device=valid.device, dtype=torch.bool)
    mask[:, :, :region_count, region_count:] = valid[:, None, None, :]
    mask[:, :, region_count:, :region_count] = valid[:, None, :, None]
    mask[:, :, region_count:, region_count:] = force_mask
    return mask


class RegionForceBridge(nn.Module):
    """Bidirectional fusion + spatial residual writeback to actual Video tokens."""
    def __init__(self, video_dim, force_dim, width, heads, gate=.01):
        super().__init__()
        self.force_in = nn.Linear(force_dim, width)
        self.force_out = nn.Linear(width, force_dim)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2*width),
                                nn.GELU(), nn.Linear(2*width, width))
        self.video_out = nn.Linear(width, video_dim, bias=False)
        self.force_gate = nn.Parameter(torch.tensor(float(gate)))
        self.video_gate = nn.Parameter(torch.tensor(float(gate)))
        self.heads = heads

    def forward(self, visual, force, maps, force_mask, valid):
        flat = visual.flatten(1, 4)
        joined = torch.cat((flat, self.force_in(force)), 1)
        mask = region_force_mask(flat.shape[1], force_mask, valid)
        # MultiheadAttention bool masks have inverse semantics to SDPA.
        blocked = (~mask[:, 0]).repeat_interleave(self.heads, 0)
        norm = self.norm(joined)
        update, _ = self.attention(norm, norm, norm, attn_mask=blocked, need_weights=False)
        fused = joined + update
        fused = fused + self.ff(fused)
        next_regions = fused[:, :flat.shape[1]].reshape_as(visual)
        next_force = force + self.force_gate * self.force_out(fused[:, flat.shape[1]:] - joined[:, flat.shape[1]:])
        next_force = torch.where(valid[..., None], next_force, 0)
        b, t, v, pairs, _, h, w = maps.shape
        assignment = maps.reshape(b, t, v, pairs*2, h*w)
        assignment = assignment / assignment.sum(-2, keepdim=True).clamp_min(1e-8)
        delta = self.video_out(next_regions.flatten(3, 4) - visual.flatten(3, 4))
        writeback = assignment.to(delta.dtype).transpose(-1, -2) @ delta
        writeback = writeback.reshape(b, t, v, h, w, -1)
        return next_regions, next_force, self.video_gate * merge_views(writeback)
