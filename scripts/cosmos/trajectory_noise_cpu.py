"""Prepare a real trajectory starting state on CPU; never loads model weights.

Input: a verified evaluation NPZ containing raw_action_future/action_is_pad and
the original dataset_stats.json. Outputs do not constitute denoising results.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fastwam.cosmos.action_sampling import ActionSampling
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json, SingleFieldLinearNormalizer
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--window',type=Path,required=True)
    p.add_argument('--stats',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--noise-strength',type=float,default=.1)
    p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    spec=ActionSampling(noise_strength=a.noise_strength,seed=a.seed)
    with np.load(a.window,allow_pickle=False) as data:
        raw=data['raw_action_future'].copy();pad=data['action_is_pad'].copy()
    if raw.shape!=(32,7) or raw.dtype!=np.float64 or pad.shape!=(32,) or pad.dtype!=np.bool_:
        raise ValueError('Original FP64 Action32x7 and boolean padding mask required')
    if not np.isfinite(raw).all():raise ValueError('Nonfinite raw reference')
    stats=load_dataset_stats_from_json(a.stats)['action']['default']
    normalizer=SingleFieldLinearNormalizer({k[7:]:v for k,v in stats.items() if k.startswith('global_')},'z-score')
    reference=normalizer.forward(torch.from_numpy(raw).float()).unsqueeze(0)
    initial,times,deltas=spec.prepare(WanContinuousFlowMatchScheduler(shift=1),
        batch_size=1,device='cpu',dtype=torch.float32,reference=reference)
    initial_physical=normalizer.backward(initial[0]).numpy()
    if a.output.exists():raise ValueError('Refusing to overwrite CPU evidence')
    a.output.mkdir(parents=True)
    np.savez(a.output/'initial_state.npz',raw_action_future=raw,
        reference_normalized=reference[0].numpy(),initial_normalized=initial[0].numpy(),
        initial_physical=initial_physical,action_is_pad=pad,clock=times.numpy(),delta=deltas.numpy())
    valid=~pad
    record=dict(status='CPU real-trajectory perturbation prepared; model not run',
        **spec.metadata(),window=str(a.window.resolve()),stats=str(a.stats.resolve()),
        window_sha256=hashlib.sha256(a.window.read_bytes()).hexdigest(),
        stats_sha256=hashlib.sha256(a.stats.read_bytes()).hexdigest(),
        valid_steps=int(valid.sum()),perturbed_valid_values=int((initial[0].numpy()[valid]!=reference[0].numpy()[valid]).sum()),
        initial_xyz_mae_mm=float(np.abs(initial_physical[valid,:3]-raw[valid,:3]).mean()*1000),
        gpu_used=False,denoising_performed=False)
    (a.output/'receipt.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
