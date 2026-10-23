"""Resolve from actual verified caches; never invent a training-ready identity."""
import argparse
import json
from pathlib import Path
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from fastwam.cosmos.assets import read_text_bank,checked,WEIGHT_SHA,NONACTION_WEIGHT_SHA
from fastwam.cosmos.cache import validate
from fastwam.cosmos.native import verify_upstream
from fastwam.cosmos.preflight import validate_config
from fastwam.cache.latent import cache_identity_sha256, file_sha256

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task', choices=['cosmos_forcewm_4gpu_30k', 'cosmos_forcewm_joint_4gpu_30k', 'cosmos_noaction_joint_4gpu_30k', 'cosmos_noaction_query_posttrain_4gpu_5k'],
                   default='cosmos_forcewm_4gpu_30k')
    for key in ('text-cache','feature-cache','checkpoint','upstream','output-dir','config-output'):p.add_argument('--'+key,required=True)
    p.add_argument('--normalization', help='Exact RealMan normalization exported for this training cache')
    p.add_argument('--batch-size',type=int,choices=[16,32,64])
    p.add_argument('--max-steps',type=int,choices=[1000,5000,20000,30000])
    p.add_argument('--initial-checkpoint')
    p.add_argument('--initial-checkpoint-sha256')
    a=p.parse_args();bank,tm,td=read_text_bank(a.text_cache);del bank
    fm=validate(a.feature_cache);checked(a.checkpoint,NONACTION_WEIGHT_SHA if a.task in ('cosmos_noaction_joint_4gpu_30k','cosmos_noaction_query_posttrain_4gpu_5k') else WEIGHT_SHA);verify_upstream(Path(a.upstream))
    if fm['identity']['text_identity']!=td:raise ValueError('Feature/text identities differ')
    root=Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        cfg=compose(config_name='train',overrides=['task='+a.task])
    query_posttrain = a.task == 'cosmos_noaction_query_posttrain_4gpu_5k'
    if a.batch_size is not None or a.max_steps is not None:
        if a.task not in ('cosmos_noaction_joint_4gpu_30k', 'cosmos_noaction_query_posttrain_4gpu_5k'):
            raise ValueError('Overrides require a joint8 model')
        if a.batch_size is not None:cfg.batch_size=a.batch_size
        if a.max_steps is not None:cfg.max_steps=a.max_steps
        cfg.save_every=1000 if cfg.max_steps == 1000 else 5000
    cfg.model.expected_text_identity=td;cfg.model.text_cache=str(Path(a.text_cache).resolve())
    cfg.model.checkpoint=str(Path(a.checkpoint).resolve());cfg.model.upstream=str(Path(a.upstream).resolve())
    cfg.data.train.cache_root=str(Path(a.feature_cache).resolve());cfg.data.train.expected_identity_sha256=fm['identity_sha256']
    if a.task in ('cosmos_noaction_joint_4gpu_30k', 'cosmos_noaction_query_posttrain_4gpu_5k'):
        cfg.model.expected_data_identity=fm['identity_sha256']
        source=fm['identity'].get('realman_source')
        if source is not None:
            if source.get('action_space') != cfg.model.action_space or source.get('action_dim') != 8:
                raise ValueError('Training cache must contain joint8 action targets and state')
            if a.normalization is None:
                raise ValueError('RealMan no-reference training requires --normalization')
            norm=json.loads(Path(a.normalization).read_text())
            if (norm.get('source_identity_sha256') != cache_identity_sha256(source)
                    or norm.get('statistics') != source['statistics']
                    or norm.get('force_preprocessing') != source.get('force_preprocessing')
                    or norm.get('action_space') != source['action_space']
                    or norm.get('action_units') != source['action_units']
                    or norm.get('state_columns') != source['state_columns']):
                raise ValueError('Normalization and force preprocessing differ from training source')
            cfg.model.expected_normalization_sha256=file_sha256(a.normalization)
    if query_posttrain:
        if (a.batch_size not in (None,64) or a.max_steps not in (None,5000)
                or not a.initial_checkpoint or not a.initial_checkpoint_sha256
                or len(a.initial_checkpoint_sha256) != 64):
            raise ValueError('The 5k query post-train requires batch64 and a SHA-bound step20000 checkpoint')
        initial = Path(a.initial_checkpoint).resolve()
        if not initial.is_file() or file_sha256(initial) != a.initial_checkpoint_sha256:
            raise ValueError('Initial 20k model-only checkpoint is missing or its SHA differs')
        cfg.model.initial_checkpoint=str(initial)
        cfg.model.initial_checkpoint_sha256=a.initial_checkpoint_sha256
    elif a.initial_checkpoint is not None or a.initial_checkpoint_sha256 is not None:
        raise ValueError('Initial checkpoint arguments are exclusive to query post-training')
    cfg.output_dir=str(Path(a.output_dir).absolute())
    resolved=OmegaConf.to_container(cfg,resolve=True,throw_on_missing=True);validate_config(resolved)
    with Path(a.config_output).open('x') as f:OmegaConf.save(OmegaConf.create(resolved),f)
    print('Resolved actual complete cache inputs; GPU preflight still required.')
if __name__=='__main__':main()
