"""Observation-only progress inference and fixed-width soft reference reading.

The full input plan has variable M; only the condition output has 32 slots.
No source frame, episode clock, action target, or teacher future is accepted.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


INPUT_KEYS = frozenset(('reference_pose', 'reference_valid', 'reference_channel_valid',
    'reference_order', 'reference_break_before', 'reference_present', 'current_tcp'))


def alignment_loss(log_probabilities, target, enabled):
    """Finite masked CE, including an all-dropped minibatch with zero gradient."""
    target = target.to(device=log_probabilities.device, dtype=torch.float32)
    enabled = enabled.to(device=log_probabilities.device, dtype=torch.bool).reshape(-1)
    if target.shape != log_probabilities.shape or enabled.shape != target.shape[:1]:
        raise ValueError('Alignment target dimensions differ from the padded reference')
    # Invalid keys have -inf log probability. Never evaluate 0 * -inf.
    terms = torch.where(target > 0, log_probabilities, 0.) * target
    losses = -terms.sum(-1)
    return torch.where(enabled, losses, 0.).sum() / enabled.sum().clamp_min(1)


class ObservationReferenceAuditor(nn.Module):
    """Infer progress from causal observations; read 32 local condition tokens.

    A learnt normalized look-ahead span maps the current reference progress to
    32 ordered soft addresses. This is not GT-time resampling: reference order
    belongs to the augmented plan, and every address depends on the observation.
    """
    def __init__(self, *, hidden=128, heads=4, layers=2, local_tokens=32,
                 state_dim=8, history_length=32, visual_grid=4):
        super().__init__()
        if local_tokens != 32 or hidden % heads or layers < 1:
            raise ValueError('Auditor requires 32 local slots and divisible attention width')
        self.hidden, self.local_tokens = hidden, local_tokens
        self.state_dim, self.history_length, self.visual_grid = state_dim, history_length, visual_grid
        self.reference_embedding = nn.Sequential(nn.Linear(20, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(hidden, heads, 4*hidden, dropout=0.,
                                          activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(hidden),
                                             enable_nested_tensor=False)
        self.null_token = nn.Parameter(torch.zeros(1, 1, hidden))
        # Keep spatial information from the independent current-image latent.
        observation_width = 16*visual_grid**2 + state_dim+1 + 8 + 2*history_length*(state_dim+6)
        self.observation = nn.Sequential(nn.Linear(observation_width, hidden), nn.SiLU(),
                                        nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.span = nn.Linear(hidden, 1)
        self.local_projection = nn.Sequential(nn.Linear(hidden+3, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.register_buffer('local_fraction', torch.linspace(0, 1, local_tokens), persistent=False)

    def forward(self, inputs, *, first_frame_latents, proprio, history,
                history_is_pad, history_channel_valid):
        if set(inputs) != INPUT_KEYS:
            raise ValueError('Auditor inputs must contain only the explicit observation/reference contract')
        dtype = self.reference_embedding[0].weight.dtype
        device = self.reference_embedding[0].weight.device
        pose = inputs['reference_pose'].to(device=device, dtype=dtype)
        current = inputs['current_tcp'].to(device=device, dtype=dtype)
        if pose.ndim != 3 or pose.shape[-1] != 8 or current.shape != (len(pose), 8) or pose.shape[1] < 1:
            raise ValueError('Expected full reference [B,M,8] and current measured TCP [B,8]')
        b, m, _ = pose.shape
        valid = inputs['reference_valid'].to(device=device, dtype=torch.bool)
        channels = inputs['reference_channel_valid'].to(device=device, dtype=torch.bool)
        order = inputs['reference_order'].to(device=device, dtype=torch.float32)
        breaks = inputs['reference_break_before'].to(device=device, dtype=torch.bool)
        present = inputs['reference_present'].to(device=device, dtype=torch.bool).reshape(b)
        if valid.shape != (b,m) or channels.shape != (b,m,3) or order.shape != (b,m) or breaks.shape != (b,m):
            raise ValueError('Reference masks/order must be padded to the full-plan length M')
        valid = valid & channels.all(-1) & present[:, None]
        present = valid.any(-1)
        # Remove every masked value before projection, including arbitrary/NaN padding.
        pose = torch.where(valid[...,None], pose, 0.)
        delta_xyz = (pose[..., :3] - current[:,None,:3]) / .1
        q, c = pose[...,3:7], current[:,None,3:7]
        # q * inverse(current); canonical sign removes antipodal ambiguity.
        v = c[...,3:4]*q[...,:3] - q[...,3:4]*c[...,:3] - torch.linalg.cross(q[...,:3],c[...,:3],dim=-1)
        w = (q*c).sum(-1,keepdim=True)
        relative_q = torch.cat((v,w),-1)
        relative_q = torch.where(relative_q[...,3:4] < 0,-relative_q,relative_q)
        canonical_q = torch.where(pose[...,6:7] < 0,-pose[...,3:7],pose[...,3:7])
        pose = torch.cat((pose[...,:3],canonical_q,pose[...,7:]),-1)
        safe_order = torch.where(valid, order, 0.)
        features = torch.cat((pose,delta_xyz,relative_q,safe_order[...,None].to(dtype),
                              breaks[...,None].to(dtype),channels.to(dtype)),-1)
        features = torch.where(valid[...,None],features,0.)
        encoded = self.reference_embedding(features)
        keys_valid = torch.cat((valid, ~present[:,None]),1)
        encoded = self.encoder(torch.cat((encoded,self.null_token.expand(b,-1,-1)),1),
                               src_key_padding_mask=~keys_valid)
        first = first_frame_latents.to(device=device,dtype=dtype)
        if first.ndim != 5 or first.shape[:3] != (b,16,1):
            raise ValueError('Auditor visual input must be the independently encoded current frame only')
        visual = F.adaptive_avg_pool2d(first[:,:,0],self.visual_grid).flatten(1)
        prop = proprio.to(device=device,dtype=dtype)
        prop = prop[:,0] if prop.ndim == 3 else prop
        hist = history.to(device=device,dtype=dtype)
        if prop.shape != (b,self.state_dim+1) or hist.shape != (b,self.history_length,self.state_dim+6):
            raise ValueError('Causal observation dimensions differ')
        hm = history_channel_valid.to(device=device,dtype=torch.bool) & ~history_is_pad.to(device=device,dtype=torch.bool)[...,None]
        hist = torch.where(hm,hist,0.)
        obs = self.observation(torch.cat((visual,prop,current,hist.flatten(1),hm.to(dtype).flatten(1)),-1))
        scores = torch.einsum('bd,bmd->bm',self.query(obs).float(),self.key(encoded).float())/math.sqrt(self.hidden)
        scores = scores.masked_fill(~keys_valid,float('-inf'))
        logp = scores.log_softmax(-1)
        probability = logp.exp()
        plan_probability = probability[:,:m]
        progress = (plan_probability*safe_order).sum(-1,keepdim=True)
        # At most one quarter of the plan's normalized order; span is predicted,
        # not a fixed ratio of source frames or the current 32-action horizon.
        span = .01 + .24*torch.sigmoid(self.span(obs).float())
        centres = (progress + span*self.local_fraction[None]).clamp(0,1)
        bandwidth = (span/4).clamp_min(.01)
        distance = (safe_order[:,None,:]-centres[:,:,None])/bandwidth[:,:,None]
        segment = breaks.long().cumsum(-1)
        same_segment = segment[:,:,None] == segment[:,None,:]
        segment_mass = torch.bmm(same_segment.float(),plan_probability[:,:,None]).squeeze(-1)
        # The alignment posterior locates the anchor, not every future slot.
        # Reusing its sharp logits here would collapse all 32 reads onto the
        # current waypoint after alignment training instead of reading ahead.
        local_scores = -.5*distance.square() + segment_mass[:,None,:].clamp_min(1e-8).log()
        local_scores = local_scores.masked_fill(~valid[:,None,:],float('-inf'))
        null_score = torch.where(present,torch.full_like(progress[:,0],float('-inf')),0.)
        local_weights = torch.cat((local_scores,null_score[:,None,None].expand(-1,self.local_tokens,1)),-1).softmax(-1)
        local = torch.bmm(local_weights.to(dtype),encoded)
        geometry = torch.stack((centres,span.expand(-1,self.local_tokens),self.local_fraction[None].expand(b,-1)),-1)
        local = self.local_projection(torch.cat((local,geometry.to(dtype)),-1))
        local = local * present[:,None,None].to(dtype)
        return dict(local_tokens=local,alignment_log_probabilities=logp[:,:m],
                    alignment_probabilities=plan_probability,progress=progress,
                    span=span,reference_present=present)
