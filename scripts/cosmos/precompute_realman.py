"""Four independent GPU encoders, then strict full-cache assembly."""
import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file
from fastwam.cache.latent import cache_identity_sha256, file_sha256
from fastwam.cosmos.realman import RealManSource, PROFILE, require
from fastwam.cosmos.assets import load_tokenizer, read_text_bank, TOKENIZER_SHA
from fastwam.cosmos.native import verify_upstream
from fastwam.cosmos.cache import SCHEMA, VIDEO_SHAPE, FIRST_SHAPE, _atomic, validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','source-sha','text-cache','tokenizer','upstream','output','stage'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--rank',type=int)
    p.add_argument('--assemble',action='store_true')
    a=p.parse_args(); torch.set_num_threads(4)
    s=RealManSource(a.dataset,a.source_sha)
    bank,tm,td=read_text_bank(a.text_cache); del bank
    require([x['prompt'] for x in tm['texts']]==sorted(set(s.prompts)), 'Recorded prompt mismatch')
    i=dict(schema=SCHEMA,scope='production',dataset_profile=PROFILE,
        tokenizer_sha256=TOKENIZER_SHA, video_shape=list(VIDEO_SHAPE),rgb_shape=[3,9,224,448],
        camera_order=['cam_main','cam_wrist'],encoding='independent9frame_and_independent1frame',
        pixel_identity=s.identity_sha256,numeric_identity=s.identity_sha256,
        numeric_sidecar_identity=s.identity_sha256,text_identity=td,
        upstream_hashes=verify_upstream(Path(a.upstream)),episode_lengths=s.lengths,
        prompts=sorted(set(s.prompts)),exporter_sha256=file_sha256(__file__),realman_source=s.identity)
    digest=cache_identity_sha256(i); output=Path(a.output); stage=Path(a.stage)
    if a.assemble:
        shards=[]; cursor=0
        for rank in range(4):
            m=json.loads((stage/f'rank{rank}.json').read_text())
            require(m['status']=='complete' and m['identity_sha256']==digest
                    and m['rank']==rank and m['start']==len(s)*rank//4
                    and m['stop']==len(s)*(rank+1)//4,'Incomplete/mismatched encoder rank')
            for e in m['shards']:
                require(e['start']==cursor, 'Rank overlap/gap')
                shards.append(e);cursor=e['stop']
        require(cursor==len(s),'Missing encoded anchors')
        _atomic(output/'manifest.json',dict(identity=i,identity_sha256=digest,status='complete',
            completed_samples=cursor,shards=shards))
        verified=validate(output,digest)
        print(json.dumps({'status':'cache_verified','samples':cursor,'identity':verified['identity_sha256']}),flush=True)
        return
    require(a.rank in range(4),'Exactly four encoder ranks required')
    torch.cuda.set_device(0)
    require(torch.cuda.device_count()==1, 'Isolate one allocated GPU per encoder process')
    vae=load_tokenizer(a.upstream,a.tokenizer)
    start=len(s)*a.rank//4; stop=len(s)*(a.rank+1)//4
    output.mkdir(parents=True,exist_ok=True); (output/'shards').mkdir(exist_ok=True)
    stage.mkdir(parents=True,exist_ok=True)
    receipt=stage/f'rank{a.rank}.json'
    require(not receipt.exists(),'Existing encoder receipt; use a new attempt directory')
    m=dict(rank=a.rank,start=start,stop=stop,status='running',identity_sha256=digest,shards=[])
    _atomic(receipt,m)
    for first in range(start,stop,32):
        end=min(stop,first+32); samples=[]
        for j in range(first,end):
            x=s.window(j).unsqueeze(0).to('cuda',torch.bfloat16)
            with torch.inference_mode():
                z=vae.encode(x); z0=vae.encode(x[:,:,:1])
            require(z.shape==(1,*VIDEO_SHAPE) and z0.shape==(1,*FIRST_SHAPE),'Tokenizer shapes differ')
            require(bool(torch.isfinite(z).all() & torch.isfinite(z0).all()),'Nonfinite tokenizer output')
            item=s[j];item.update(video_latents=z[0].to('cpu',torch.bfloat16),
                                  first_frame_latents=z0[0].to('cpu',torch.bfloat16))
            samples.append(item)
        payload={k:torch.stack([v[k] for v in samples]).contiguous() for k in samples[0]}
        rel=f'shards/{first:06d}-{end:06d}.safetensors'; path=output/rel
        require(not path.exists(),'Refuse to overwrite a feature shard')
        tmp=path.with_suffix('.tmp');save_file(payload,str(tmp));tmp.replace(path)
        m['shards'].append(dict(start=first,stop=end,path=rel,sha256=file_sha256(path)))
        m['status']='complete' if end==stop else 'running';m['completed_samples']=end-start
        _atomic(receipt,m)
        print(json.dumps({'rank':a.rank,'done':end-start,'total':stop-start}),flush=True)


if __name__=='__main__':main()
