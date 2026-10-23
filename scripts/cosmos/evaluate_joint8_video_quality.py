"""Decode independent video-expert outputs; score future frames against recorded RGB."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'deploy')]
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from fastwam.cosmos.assets import load_tokenizer

def quality(p,g):
    x=p.astype(np.float64)/255;y=g.astype(np.float64)/255
    mse=float(np.mean((x-y)**2));mx=gaussian_filter(x,(0,1.5,1.5,0),truncate=3.333333)
    my=gaussian_filter(y,(0,1.5,1.5,0),truncate=3.333333)
    vx=gaussian_filter(x*x,(0,1.5,1.5,0),truncate=3.333333)-mx*mx
    vy=gaussian_filter(y*y,(0,1.5,1.5,0),truncate=3.333333)-my*my
    cov=gaussian_filter(x*y,(0,1.5,1.5,0),truncate=3.333333)-mx*my
    ssim=((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    return dict(mse=mse,psnr_db=float(-10*np.log10(max(mse,1e-15))),ssim=float(ssim[:,5:-5,5:-5].mean()))

def main():
    p=argparse.ArgumentParser();
    for k in ('prepared','results','runtime','output'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True);torch.set_num_threads(4)
    rows=[r for r in torch.load(args.prepared,weights_only=False) if 'midpoint' in r['strata']]
    cfg=json.loads(args.runtime.read_text());vae=load_tokenizer(cfg['model_kwargs']['upstream'],cfg['tokenizer'])
    output=args.output/'video-quality.jsonl';done=set()
    if output.exists():done={(r['sample'],r['steps'],r['seed_offset']) for r in map(json.loads,output.read_text().splitlines())}
    for row in rows:
        for v in [4,8,16]:
            for seed in range(3):
                if (row['sample'],v,seed) in done:continue
                with np.load(args.results/f'sample_{row["sample"]:06d}_seed{seed}_v{v}_f{v}_a{v}.npz') as saved:
                    latent=torch.tensor(saved['video_latents'],device='cuda',dtype=torch.bfloat16)
                with torch.inference_mode():video=vae.decode(latent).float().cpu()[0].permute(1,2,3,0).add(1).mul(127.5).clamp(0,255).byte().numpy()
                gt=row['gt_pixels'].numpy();assert video.shape==gt.shape
                # Midpoint windows all have complete future video horizons, checked here.
                # future action mask alone omits offset32; source selection midpoint remains internal.
                if not row['target']['action_valid'].all() or not row['target']['force_valid'].all():
                    raise ValueError('Need full future horizon for video comparison')
                record=dict(sample=row['sample'],episode=row['episode'],steps=v,seed_offset=seed,
                    global_view=quality(video[1:,:,:224],gt[1:,:,:224]),wrist_view=quality(video[1:,:,224:],gt[1:,:,224:]))
                with output.open('a') as stream:stream.write(json.dumps(record)+'\n')
                del latent,video
        print(json.dumps(dict(stage='video_quality',sample=row['sample'])),flush=True)
    records=[json.loads(x) for x in output.read_text().splitlines()]
    summary={str(v):{camera:{key:float(np.mean([r[camera][key] for r in records if r['steps']==v]))
        for key in ('mse','psnr_db','ssim')} for camera in ('global_view','wrist_view')} for v in [4,8,16]}
    (args.output/'video-quality.json').write_text(json.dumps(dict(summary=summary,records=len(records),scope='Only 8 future frames; two cameras scored separately; PSNR and 11x11 Gaussian population SSIM averaged per window. Offline raw RGB reference; includes VAE reconstruction error; latent archive FP16.'),indent=2)+'\n')
if __name__=='__main__':main()
