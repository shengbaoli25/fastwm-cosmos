"""Evaluate actual trained paired queries on a checked cache; no training launch.

Writes maps, normalized force metrics, generated videos and a standalone HTML
viewer. A new paired checkpoint is mandatory; baseline checkpoints cannot
produce trained region queries merely by loading random new parameters.
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
from fastwam.cosmos.region_visualization import write_bundle
from fastwam.forcewm.model import masked_force_mse


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','checkpoint','checkpoint-sha256','tokenizer','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--indices',required=True,help='Comma-separated actual cache sample indices')
    p.add_argument('--device',default='cuda')
    p.add_argument('--video-seed',type=int,default=44)
    p.add_argument('--pair',type=int,default=0,help='Focus video ID; viewer always includes ALL pairs')
    a=p.parse_args();cfg=OmegaConf.load(a.config)
    if cfg.model._target_!='fastwam.cosmos.assets.create_cosmos_paired_regions':
        raise ValueError('The trained paired-region architecture is required')
    if file_sha256(a.checkpoint)!=a.checkpoint_sha256:
        raise ValueError('Checkpoint SHA256 differs')
    indices=[int(v) for v in a.indices.split(',')]
    if len(indices)!=len(set(indices)) or min(indices)<0:
        raise ValueError('Unique nonnegative cache indices required')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    dataset=instantiate(cfg.data.train)
    model=instantiate(cfg.model,device=a.device);step=model.load_checkpoint(a.checkpoint)
    if not isinstance(step,int) or step < 1:
        raise ValueError('A checkpoint with completed training steps is required')
    decoder=load_tokenizer(cfg.model.upstream,a.tokenizer,device=a.device)
    reports=[]
    for index in indices:
        row=dataset[index]
        sample={key:value.unsqueeze(0) for key,value in row.items()}
        with torch.inference_mode():
            result=model.infer_with_regions(sample,action_condition=sample['action'],
                                            video_seed=a.video_seed,generate_action=False)
            rgb=decoder.decode(result['video_latents'])
        if rgb.shape!=(1,3,9,224,448) or not torch.isfinite(rgb).all():
            raise ValueError('Unexpected actual decoded video')
        frames=(((rgb[0].permute(1,2,3,0).float().cpu()+1)/2).clamp(0,1)*255).round().numpy().astype(np.uint8)
        force=result['force'].cpu()
        mse=float(masked_force_mse(force,sample['wrench_future'],sample['force_is_pad'],sample['wrench_valid']))
        hold=sample['history'][:,-1:,7:].expand_as(force)
        hold_mse=float(masked_force_mse(hold,sample['wrench_future'],sample['force_is_pad'],sample['wrench_valid']))
        report=dict(sample_index=index,episode_index=int(row['episode_index']),anchor=int(row['frame_index']),
            checkpoint_step=step,checkpoint_sha256=a.checkpoint_sha256,
            cache_identity=dataset.identity_sha256,video_seed=a.video_seed,
            force_normalized_mse=mse,hold_normalized_mse=hold_mse,
            units='normalized training space; not Newton or Newton-metre',
            architecture=model.manifest['paired_regions'],video_kind='generated prediction',
            source_frame_indices=[int(row['frame_index'])+i for i in range(0,33,4)],
            source_image_is_pad=row['image_is_pad'].tolist(),
            config_sha256=file_sha256(a.config),model_manifest=model.manifest,
            map_source='clock0 generated-video refresh before first Force writeback in that pass',
            scope='GT-action-conditioned offline inspection; not a contact detection accuracy claim')
        target=out/f'sample_{index:06d}'
        maps=result['regions']['attention'][0].float().cpu().numpy()
        write_bundle(target,frames,maps,report,pair=a.pair)
        np.savez_compressed(target/'predictions.npz',force=force.float().numpy(),
                            target=sample['wrench_future'].float().numpy(),
                            valid=sample['wrench_valid'].numpy(),is_pad=sample['force_is_pad'].numpy())
        reports.append(report)
    (out/'summary.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


if __name__=='__main__':main()
