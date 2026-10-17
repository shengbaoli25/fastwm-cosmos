"""Resolve from actual verified caches; never invent a training-ready identity."""
import argparse
from pathlib import Path
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from fastwam.cosmos.assets import read_text_bank,checked,WEIGHT_SHA
from fastwam.cosmos.cache import validate
from fastwam.cosmos.native import verify_upstream
from fastwam.cosmos.preflight import validate_config

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',default='cosmos_forcewm_4gpu_30k',
        choices=['cosmos_forcewm_4gpu_30k','cosmos_forcewm_paired_4gpu_30k'])
    for key in ('text-cache','feature-cache','checkpoint','upstream','output-dir','config-output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();bank,tm,td=read_text_bank(a.text_cache);del bank
    fm=validate(a.feature_cache);checked(a.checkpoint,WEIGHT_SHA);verify_upstream(Path(a.upstream))
    if fm['identity']['text_identity']!=td:raise ValueError('Feature/text identities differ')
    root=Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        cfg=compose(config_name='train',overrides=['task='+a.task])
    cfg.model.expected_text_identity=td;cfg.model.text_cache=str(Path(a.text_cache).resolve())
    cfg.model.checkpoint=str(Path(a.checkpoint).resolve());cfg.model.upstream=str(Path(a.upstream).resolve())
    cfg.data.train.cache_root=str(Path(a.feature_cache).resolve());cfg.data.train.expected_identity_sha256=fm['identity_sha256']
    cfg.output_dir=str(Path(a.output_dir).absolute())
    resolved=OmegaConf.to_container(cfg,resolve=True,throw_on_missing=True);validate_config(resolved)
    with Path(a.config_output).open('x') as f:OmegaConf.save(OmegaConf.create(resolved),f)
    print('Resolved actual complete cache inputs; GPU preflight still required.')
if __name__=='__main__':main()
