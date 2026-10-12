"""Offline only: original RGB cache -> independent Cosmos features + numeric labels."""
import argparse,json
from pathlib import Path
import torch
from fastwam.cache.pixels import PixelCacheDataset
from fastwam.forcewm.data import ForceWMCachedDataset
from fastwam.cosmos.assets import load_tokenizer,read_text_bank
from fastwam.cosmos.native import verify_upstream
from fastwam.cosmos.cache import build_cache


def main():
    p=argparse.ArgumentParser()
    for k in ['pixel-cache','pixel-identity','numeric-cache','numeric-identity','stats','text-cache','output','tokenizer']:
        p.add_argument('--'+k,required=True)
    p.add_argument('--upstream',default='third_party/cosmos-predict2.5')
    p.add_argument('--shard-size',type=int,default=32);p.add_argument('--limit',type=int)
    p.add_argument('--staged-latents',help='Complete audited image-only shards; no GPU/re-encoding during assembly')
    p.add_argument('--resume',action='store_true');p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not a.execute:p.error('Offline GPU encoding requires explicit --execute')
    if a.shard_size<1 or a.limit is not None and a.limit<1:p.error('Positive limits required')
    bank,texts,td=read_text_bank(a.text_cache);del bank
    pixels=PixelCacheDataset(a.pixel_cache,a.pixel_identity)
    numeric=ForceWMCachedDataset(a.numeric_cache,a.stats,a.numeric_identity,require_production=True)
    staged=None
    if a.staged_latents:
        from fastwam.cosmos.latent_stage import StagedLatents
        staged=StagedLatents(a.staged_latents,pixels,verify_upstream(Path(a.upstream)))
        encode=None
    else:
        vae=load_tokenizer(a.upstream,a.tokenizer)
        def encode(x):return vae.encode(x.to('cuda',torch.bfloat16)).to(torch.bfloat16)
    receipt=build_cache(a.output,pixels=pixels,numeric=numeric,encoder=encode,
        text_manifest=texts,text_identity=td,upstream_hashes=verify_upstream(Path(a.upstream)),
        shard_size=a.shard_size,resume=a.resume,limit=a.limit,staged=staged)
    print(json.dumps({'status':receipt['status'],'samples':receipt['completed_samples'],'identity':receipt['identity_sha256']},indent=2))
if __name__=='__main__':main()
