"""Bind actual GPU evidence to the code and complete configured inputs."""
from pathlib import Path
import hashlib,json
from fastwam.cache.latent import file_sha256,cache_identity_sha256
ROOT=Path(__file__).resolve().parents[3]

def binding(config):
    from omegaconf import OmegaConf
    cfg=OmegaConf.to_container(OmegaConf.load(config),resolve=True,throw_on_missing=True)
    cfg.pop('output_dir',None)
    paths=[p for p in (ROOT/'src').rglob('*.py')]
    paths.extend((ROOT/'scripts/cosmos').glob('*.py'))
    paths.extend([ROOT/'scripts/ds_configs/ds_zero1_config.json',ROOT/'scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml'])
    code={p.relative_to(ROOT).as_posix():file_sha256(p) for p in sorted(paths)}
    from .native import verify_upstream
    code.update(verify_upstream(Path(cfg['model']['upstream']).resolve()))
    return {'config':cache_identity_sha256(cfg),'code':cache_identity_sha256(code)}

def verify_preflight(config,receipt,outer):
    r=json.loads(Path(receipt).read_text());o=json.loads(Path(outer).read_text());b=binding(config)
    if r.get('status')!='two_adam_verified' or r.get('binding')!=b:raise ValueError('Current Cosmos GPU preflight required')
    ranks=r.get('ranks',[])
    if len(ranks)!=4 or {x.get('rank') for x in ranks}!={0,1,2,3} or any(not x.get('accepted') or x.get('binding')!=b for x in ranks):raise ValueError('All four actual ranks required')
    verify_outer(o)
    return {'binding':b,'preflight_sha256':file_sha256(receipt),'outer_sha256':file_sha256(outer)}

def verify_outer(o):
    tree=o.get('ownership',{})
    if (o.get('status')!='process_exited_zero' or o.get('returncode')!=0
            or not o.get('subreaper_restored') or not tree.get('closure_verified')
            or tree.get('signals') or tree.get('errors') or tree.get('live_after_cleanup')):
        raise ValueError('Actual unforced whole-process-tree exit required')
