"""Learned dataset-to-Cosmos controls at the sampled video-frame rate.

The policy predicts 32 normalized commands; RGB is sampled at 0,4,...,32.
Eight disjoint groups [4j:4j+4] therefore condition those eight intervals.
This is a newly trained projection, not a claim that dataset units/rotations
match the official robot controls. Never sum/average poses as physical deltas.
"""
import torch
from torch import nn


CONTROL_PROTOCOL = 'normalized-action32-group4-validbits-linear32to7-v1'


class VideoControlAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4 * 7 + 4, 7)

    def forward(self, action, is_pad):
        if (action.ndim != 3 or action.shape[1:] != (32, 7)
                or not action.is_floating_point() or not torch.isfinite(action).all()
                or is_pad.shape != action.shape[:2] or is_pad.dtype != torch.bool):
            raise ValueError('Video controls require finite Action[B,32,7] and boolean padding[B,32]')
        valid = (~is_pad).to(device=action.device).reshape(len(action), 8, 4)
        grouped = action.reshape(len(action), 8, 4, 7)
        grouped = torch.where(valid[..., None], grouped, 0).flatten(2)
        features = torch.cat((grouped, valid.to(action.dtype)), dim=-1)
        output = self.projection(features.to(self.projection.weight.dtype))
        # Fully padded intervals are neutral, including the learned bias.
        return torch.where(valid.any(-1, keepdim=True), output, 0)
