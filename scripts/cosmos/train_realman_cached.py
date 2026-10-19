"""Train the no-reference joint diffusion model from verified existing RealMan caches."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/cosmos')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'source-sha', 'prepared', 'work', 'run-root', 'checkpoint'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--batch-size',type=int,choices=[16,32,64],default=16)
    parser.add_argument('--max-steps',type=int,choices=[1000,20000,30000],default=30000)
    args = parser.parse_args()
    from distributed_fresh import clean_environment
    env = clean_environment(os.environ, ROOT)
    os.environ.clear()
    os.environ.update(env)
    if not os.environ.get('TRAIN_JOB_ID'):
        raise ValueError('Platform distributed training job required')

    import torch
    from fastwam.cosmos.cache import _atomic
    from fastwam.cosmos.noaction import SCHEMA as JOINT_PROTOCOL, BRANCH_READS
    from fastwam.cosmos.realman_joint import open_realman_source
    from omegaconf import OmegaConf
    torch.set_num_threads(4)
    if (JOINT_PROTOCOL != 'cosmos-noaction-vfa-diffusion28-joint8-v2'
            or BRANCH_READS != ((1, 0, 0), (1, 1, 0), (1, 1, 1))):
        raise ValueError('The no-reference joint diffusion architecture is required')

    prepared = Path(args.prepared).resolve()
    work = Path(args.work).resolve()
    run = Path(args.run_root).resolve()
    if (prepared == work or prepared in work.parents or work in prepared.parents
            or run.exists()):
        raise ValueError('Use fresh work/run directories outside the immutable cache')
    work.mkdir(parents=True, exist_ok=False)
    source = open_realman_source(args.dataset, args.source_sha)
    if source.identity.get('action_space') != 'joint8':
        raise ValueError('The new cached training entrypoint requires joint8 demonstrations')
    source.export_normalization(work/'normalization.json')
    manifest = json.loads((prepared / 'features/manifest.json').read_text())
    if (manifest['status'] != 'complete'
            or manifest['completed_samples'] != len(source)
            or manifest['identity'].get('realman_source') != source.identity):
        raise ValueError('Existing cache does not match the complete current RealMan source')
    _atomic(work / 'cache_reuse.json', dict(
        prepared=str(prepared), dataset_source_sha=source.source_sha,
        cache_identity=manifest['identity_sha256'], samples=len(source),
        joint_protocol=JOINT_PROTOCOL, branch_reads=BRANCH_READS,
        model_warm_start=False, optimizer_resume=False))
    del source
    # The resolver verifies every shard, text pairing, official weights and
    # pinned upstream before this independent run may enter its GPU gates.
    with (work / 'resolve.log').open('w') as log:
        subprocess.run([
            sys.executable, '-u', str(ROOT / 'scripts/cosmos/resolve_config.py'),
            '--task', 'cosmos_noaction_joint_4gpu_30k',
            '--batch-size',str(args.batch_size),'--max-steps',str(args.max_steps),
            '--normalization', str(work / 'normalization.json'),
            '--text-cache', str(prepared / 'text'),
            '--feature-cache', str(prepared / 'features'),
            '--checkpoint', args.checkpoint,
            '--upstream', str(ROOT / 'third_party/cosmos-predict2.5'),
            '--output-dir', str(work / 'config-placeholder'),
            '--config-output', str(work / 'resolved.yaml'),
        ], check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    cfg = OmegaConf.load(work / 'resolved.yaml')
    if cfg.model.get('initial_checkpoint') is not None or cfg.max_steps != args.max_steps:
        raise ValueError('Fresh official initialization and the requested complete schedule required')
    print('REALMAN_NOACTION_DIFFUSION_CACHE_VERIFIED', flush=True)
    os.execv(sys.executable, [
        sys.executable, '-u', str(ROOT / 'scripts/cosmos/distributed_chunk.py'),
        '--config', str(work / 'resolved.yaml'), '--run-root', str(run),
        '--native-python', sys.executable, '--execute',
    ])


if __name__ == '__main__':
    main()
