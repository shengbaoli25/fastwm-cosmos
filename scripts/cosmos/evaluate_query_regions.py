"""Inspect trained independent queries with explicit clean Action32 conditions.

Metrics are in normalized training space; physical-unit/whole-episode reports
must also use the dataset's original FP64/statistics protocol. RGB references
here are decoded GT latents, not original camera recordings.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.cache.latent import file_sha256
from fastwam.cosmos.assets import load_tokenizer
from fastwam.cosmos.query_visualization import write_query_bundle
from fastwam.forcewm.model import masked_force_mse


def normalized_action_mse(predicted, target, is_pad):
    valid = (~is_pad)[..., None].expand_as(target)
    error = torch.where(valid, predicted.float() - target.float(), 0)
    if not bool(torch.isfinite(error).all()):
        raise ValueError('Nonfinite valid Action predictions')
    return float(error.square().sum() / valid.sum().clamp_min(1))


def rgb_frames(decoded):
    if decoded.shape != (1, 3, 9, 224, 448) or not torch.isfinite(decoded).all():
        raise ValueError('Expected finite decoded [1,3,9,224,448]')
    return (((decoded[0].permute(1, 2, 3, 0).float().cpu() + 1) / 2)
            .clamp(0, 1) * 255).round().numpy().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'checkpoint-sha256', 'tokenizer', 'output', 'indices'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--video-seed', type=int, default=44)
    parser.add_argument('--action-seed', type=int, default=0)
    parser.add_argument('--query', type=int, default=0)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if cfg.model._target_ != 'fastwam.cosmos.assets.create_cosmos_region_queries':
        raise ValueError('Independent-query model configuration required')
    if file_sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('Actual checkpoint SHA256 differs')
    indices = [int(x) for x in args.indices.split(',')]
    if not indices or len(indices) != len(set(indices)) or min(indices) < 0:
        raise ValueError('Unique nonnegative cache indices required')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('schema') != 'fastwam-cosmos-region-queries32-v1':
        raise ValueError('Trained independent-query checkpoint required')
    for name, value in payload['model'].items():
        if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('Nonfinite checkpoint tensor ' + name)
    del payload
    dataset = instantiate(cfg.data.train)
    model = instantiate(cfg.model, device=args.device)
    step = model.load_checkpoint(args.checkpoint)
    if type(step) is not int or step < 1:
        raise ValueError('Completed training step required')
    decoder = load_tokenizer(cfg.model.upstream, args.tokenizer, device=args.device)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    reports = []
    for index in indices:
        row = dataset[index]
        sample = {key: value.unsqueeze(0) for key, value in row.items()}
        with torch.inference_mode():
            result = model.infer_with_regions(sample, action_condition=sample['action'],
                video_seed=args.video_seed, seed=args.action_seed, num_inference_steps=16)
            generated = decoder.decode(result['video_latents'])
            reference = decoder.decode(sample['video_latents'].to(args.device, model.torch_dtype))
        predicted = result['force'].float().cpu()
        hold = sample['history'][:, -1:, 7:].expand_as(predicted)
        maps = result['regions']['attention'][0].float().cpu().numpy()
        report = dict(sample_index=index, episode_index=int(row['episode_index']), anchor=int(row['frame_index']),
            checkpoint_step=step, checkpoint_sha256=args.checkpoint_sha256,
            config_sha256=file_sha256(args.config), cache_identity=dataset.identity_sha256,
            video_seed=args.video_seed, action_seed=args.action_seed,
            action_normalized_mse=normalized_action_mse(result['action'].cpu(), sample['action'], sample['action_is_pad']),
            force_normalized_mse=float(masked_force_mse(predicted, sample['wrench_future'], sample['force_is_pad'], sample['wrench_valid'])),
            hold_normalized_mse=float(masked_force_mse(hold, sample['wrench_future'], sample['force_is_pad'], sample['wrench_valid'])),
            model_manifest=model.manifest,
            units='normalized training space, not physical units',
            action_scope='GT-action-conditioned reconstruction, not observation-only policy',
            source_frame_indices=[int(row['frame_index']) + i for i in range(0, 33, 4)],
            source_image_is_pad=row['image_is_pad'].tolist(),
            video_reference='GT latent VAE reconstruction, not raw recorded RGB',
            map_source='generated clock0 refresh at layer13 before first Force writeback',
            scope='offline query read visualization, not verified contact localization')
        target = root / f'sample_{index:06d}'
        write_query_bundle(target, rgb_frames(generated), maps, report,
                           query=args.query, reference=rgb_frames(reference))
        np.savez_compressed(target / 'predictions.npz', force=predicted.numpy(),
            target=sample['wrench_future'].numpy(), force_valid=sample['wrench_valid'].numpy(),
            force_is_pad=sample['force_is_pad'].numpy(), action=result['action'].float().cpu().numpy(),
            action_target=sample['action'].numpy(), action_is_pad=sample['action_is_pad'].numpy())
        reports.append(report)
    (root / 'summary.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
