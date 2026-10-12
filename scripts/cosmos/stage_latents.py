"""Offline image-only preparation, independent of the task-text download.

These shards are NOT a training cache. A later assembly binds audited numeric
labels and the complete Reason1 text identity. No Wan latent is consumed.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import time

import torch
from safetensors.torch import save_file

from fastwam.cache.latent import cache_identity_sha256, file_sha256
from fastwam.cache.pixels import PixelCacheDataset, pixel_rows
from fastwam.cosmos.assets import TOKENIZER_SHA, load_tokenizer
from fastwam.cosmos.native import verify_upstream


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def rank_range(total, rank, world):
    if total < 1 or world < 1 or not 0 <= rank < world:
        raise ValueError('Invalid partition')
    return total * rank // world, total * (rank + 1) // world


def worker(rank, args):
    torch.set_num_threads(4)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    out = Path(args.output) / f'rank{rank}'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    pixels = PixelCacheDataset(args.pixel_cache, args.pixel_identity)
    if pixels.manifest['identity']['scope'] != 'production':
        raise ValueError('Production RGB required')
    rows = pixel_rows(pixels.manifest['identity'])
    lower, upper = rank_range(len(pixels), rank, args.workers)
    end = min(upper, lower + args.limit) if args.limit else upper
    identity = dict(schema='cosmos-image-only-stage-v1', pixel_identity=pixels.identity_sha256,
                    pixel_manifest_sha=file_sha256(pixels.root/'manifest.json'),
                    tokenizer_sha=TOKENIZER_SHA, upstream=verify_upstream(Path(args.upstream)),
                    exporter_sha=file_sha256(__file__), offsets=list(range(0,33,4)),
                    camera_order=['cam_main','cam_wrist'], video_shape=[16,3,28,56],
                    first_shape=[16,1,28,56], encoding='independent9frame_and_independent1frame',
                    start=lower, stop=upper, rank=rank, workers=args.workers)
    m = dict(identity=identity, identity_sha256=cache_identity_sha256(identity),
             status='partial', trainable=False, completed_samples=0, shards=[],
             hostname=socket.gethostname(), pid=os.getpid(), gpu=torch.cuda.get_device_name(rank))
    atomic(out/'manifest.json', m)
    vae = load_tokenizer(args.upstream, args.tokenizer, device=str(device))
    print(json.dumps(dict(rank=rank, phase='tokenizer_loaded', start=lower, stop=end)), flush=True)
    for start in range(lower, end, args.shard_size):
        stop = min(start + args.shard_size, end)
        zs, firsts = [], []
        for j in range(start, stop):
            x = pixels.window(j).unsqueeze(0).to(device, torch.bfloat16)
            with torch.inference_mode():
                z = vae.encode(x)
                first = vae.encode(x[:,:,:1])
                if z.shape != (1,16,3,28,56) or first.shape != (1,16,1,28,56):
                    raise ValueError('Unexpected native Cosmos latent shape')
                if not bool(torch.isfinite(z).all() & torch.isfinite(first).all()):
                    raise ValueError('Nonfinite native Cosmos latent')
                if j == lower:
                    decoded = vae.decode(z).float()
                    if decoded.shape != x.shape or not bool(torch.isfinite(decoded).all()):
                        raise ValueError('Invalid reconstruction')
                    quality = []
                    for a,b in [(0,224),(224,448)]:
                        mse = ((decoded[...,a:b] - x.float()[...,a:b])/2).square().mean()
                        quality.append(float(-10 * torch.log10(mse.clamp_min(1e-12))))
                    m['pilot'] = dict(sample=j, main_psnr=quality[0], wrist_psnr=quality[1],
                                      shape=list(z.shape), decode_shape=list(decoded.shape))
                    from PIL import Image
                    canvas = torch.cat((x[0,:,4].float(), decoded[0,:,4]),dim=1)
                    rgb = ((canvas.clamp(-1,1)+1)*127.5).byte().permute(1,2,0).cpu().numpy()
                    Image.fromarray(rgb).save(out/'pilot_gt_above_reconstruction.png')
                    print(json.dumps(dict(rank=rank, phase='real_window_pilot', **m['pilot'])),flush=True)
            zs.append(z[0].to('cpu',torch.bfloat16).contiguous())
            firsts.append(first[0].to('cpu',torch.bfloat16).contiguous())
        payload = dict(video_latents=torch.stack(zs), first_frame_latents=torch.stack(firsts),
                       sample_index=torch.arange(start,stop),
                       episode_index=torch.tensor([rows[j][1] for j in range(start,stop)]),
                       frame_index=torch.tensor([rows[j][2] for j in range(start,stop)]))
        name = f'{start:06d}-{stop:06d}.safetensors'
        tmp = out/(name+'.tmp')
        save_file(payload,str(tmp)); tmp.replace(out/name)
        m['shards'].append(dict(start=start,stop=stop,path=name,sha256=file_sha256(out/name)))
        m['completed_samples'] = stop-lower
        m['status'] = 'complete' if stop == upper else 'partial'
        m['elapsed_seconds'] = time.monotonic()-started
        atomic(out/'manifest.json',m)
        print(json.dumps(dict(rank=rank, completed=stop-lower, total=upper-lower,
                              seconds=round(m['elapsed_seconds'],2))),flush=True)
    del vae
    torch.cuda.synchronize(device)


def main():
    p=argparse.ArgumentParser()
    for k in ['pixel-cache','pixel-identity','tokenizer','upstream','output']:
        p.add_argument('--'+k,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--shard-size',type=int,default=32)
    p.add_argument('--limit',type=int)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not a.execute or a.workers!=4 or a.shard_size<1 or (a.limit is not None and a.limit<1):
        p.error('Explicit execution, four GPUs and positive bounds required')
    if torch.cuda.device_count()!=4: raise ValueError('Expected exactly four visible GPUs')
    root=Path(a.output); root.mkdir(parents=True,exist_ok=False)
    atomic(root/'run.json',dict(status='running',trainable=False,argv=vars(a),started_unix=time.time()))
    try:
        torch.multiprocessing.spawn(worker,args=(a,),nprocs=a.workers,join=True)
    except BaseException as e:
        atomic(root/'run.json',dict(status='failed',trainable=False,error=repr(e),argv=vars(a)))
        raise
    atomic(root/'run.json',dict(status='complete' if a.limit is None else 'pilot_complete',
                               trainable=False,argv=vars(a),finished_unix=time.time()))


if __name__=='__main__': main()
