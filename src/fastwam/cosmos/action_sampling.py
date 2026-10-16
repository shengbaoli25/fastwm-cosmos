"""Explicit Action FM starting states; world-model controls are independent."""
from dataclasses import dataclass, asdict
import math

import torch


@dataclass(frozen=True)
class ActionSampling:
    initialization: str = 'noisy_reference'
    noise_strength: float = 0.1
    num_inference_steps: int = 16
    seed: int = 0

    def __post_init__(self):
        if self.initialization not in ('gaussian', 'noisy_reference'):
            raise ValueError('Action initialization must be gaussian or noisy_reference')
        if (type(self.noise_strength) not in (int, float)
                or not math.isfinite(self.noise_strength)
                or not 0 <= self.noise_strength <= 1):
            raise ValueError('Action noise_strength must be finite in [0,1]')
        if type(self.num_inference_steps) is not int or self.num_inference_steps < 1:
            raise ValueError('Positive integer Action inference steps required')
        if type(self.seed) is not int:
            raise ValueError('Integer Action seed required')
        if self.initialization == 'gaussian' and self.noise_strength != 1:
            raise ValueError('Gaussian initialization requires noise_strength=1')

    def metadata(self):
        return dict(**asdict(self), scope='GT/reference-conditioned trajectory reconstruction',
                    reference_coordinates='original training normalized Action32x7',
                    wm_controls='separate explicit clean action_condition',
                    noise_contract='(1-sigma)*reference+sigma*epsilon')

    def prepare(self, scheduler, *, batch_size, device, dtype, reference=None):
        shape = (batch_size, 32, 7)
        if self.initialization == 'noisy_reference':
            if not torch.is_tensor(reference) or reference.shape != shape:
                raise ValueError('Explicit normalized finite Action reference [B,32,7] required')
            if not reference.is_floating_point() or not bool(torch.isfinite(reference).all()):
                raise ValueError('Explicit normalized finite Action reference [B,32,7] required')
            reference = reference.detach().to(device=device, dtype=dtype)
            if not bool(torch.isfinite(reference).all()):
                raise ValueError('Action reference overflows model dtype')
        elif reference is not None:
            raise ValueError('Gaussian mode must not consume an Action reference')
        generator = torch.Generator(device='cpu').manual_seed(self.seed)
        noise = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
        times, deltas = scheduler.build_inference_schedule(self.num_inference_steps, device, dtype)
        if self.initialization == 'gaussian':
            return noise, times, deltas
        if self.noise_strength == 0:
            return reference.clone(), times[:0], deltas[:0]
        # Scale the original schedule, preserving its forward count and endpoint.
        # The first Action clock must describe the actual mixed starting state.
        times = times * self.noise_strength
        deltas = deltas * self.noise_strength
        action = scheduler.add_noise(reference, noise, times[0].expand(batch_size))
        return action, times, deltas
