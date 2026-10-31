"""Validate an existing RM71D78 cache, then preflight/restore/train on four H200s."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from fastwam.cosmos.backbone_experiments import (
    DATASET_SHA256, DATASET_EPISODES, DATASET_FRAMES, DATASET_WINDOWS,
    VARIANTS, default_variant, resolve_weights, validate_recipe,
)


def verify_cache(root, variant):
    """Check the assembled cache identity and immutable shard inventory once."""
    from fastwam.cache.latent import cache_identity_sha256
    from fastwam.cosmos.jepa_video_cache import SCHEMA, episode_splits
    from fastwam.cosmos.dense_encoder import teacher_spec
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    identity = manifest.get('identity', {})
    spec = VARIANTS[variant]
    teacher = identity.get('teacher', {})
    expected_teacher = teacher_spec(spec['teacher'])
    if (manifest.get('status') != 'complete'
            or manifest.get('identity_sha256') != cache_identity_sha256(identity)
            or identity.get('schema') != SCHEMA
            or identity.get('source_sha256') != DATASET_SHA256
            or teacher.get('name') != spec['teacher']
            or teacher.get('weight_sha256') != spec['weight_sha256']
            or teacher.get('upstream_commit') != expected_teacher['upstream_commit']
            or teacher.get('feature_dim') != spec['feature_dim']
            or teacher.get('temporal_aggregation') != expected_teacher['temporal_aggregation']
            or teacher.get('video_input_size') != 224
            or teacher.get('video_grid') != 14
            or identity.get('observation_offsets') != [-4, 0]
            or identity.get('target_offsets') != [4, 8, 12, 16, 20, 24, 28, 32]
            or identity.get('cameras') != ['cam_main', 'cam_wrist']
            or identity.get('feature_dtype') != 'bfloat16'
            or identity.get('observation_encoding') != 'separate_two_frame_forward_before_full_future_forward'
            or identity.get('feature_shape') != [5, 392, spec['feature_dim']]
            or len(identity.get('episode_lengths', [])) != DATASET_EPISODES
            or sum(identity.get('episode_lengths', [])) != DATASET_FRAMES
            or identity.get('splits') != episode_splits(DATASET_EPISODES)
            or manifest.get('completed_windows') != DATASET_WINDOWS
            or not manifest.get('shards')):
        raise ValueError('Complete RM71D78 cache for the selected backbone required before training')
    for entry in manifest['shards']:
        path = (root / entry['path']).resolve()
        if (not path.is_relative_to(root) or not path.is_file()
                or path.stat().st_size != entry['size_bytes']
                or path.stat().st_mtime_ns != entry['mtime_ns']):
            raise ValueError('Cache shard missing or changed: ' + entry['path'])
    return manifest


def stage_commands(config, run_root, dataset, cache_root, variant, teacher_source, weights):
    spec = VARIANTS[variant]
    common = ['--config', str(config), '--run-root', str(run_root),
        '--dataset', str(dataset), '--source-sha', DATASET_SHA256,
        '--cache-root', str(cache_root), '--teacher', spec['teacher'],
        '--teacher-source', str(teacher_source), '--weights', str(weights),
        '--weight-sha', spec['weight_sha256']]
    launch = [sys.executable, '-u', '-m', 'accelerate.commands.launch',
        '--config_file', str(ROOT / 'scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml'),
        '--num_processes', '4']
    return [(name, launch + [str(ROOT / 'scripts/dense' / script), *common])
            for name, script in (('preflight', 'preflight.py'),
                                  ('restore', 'restore.py'), ('formal', 'train.py'))]


def stage_environment():
    """Select all four container GPUs consistently for every training stage."""
    return dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', WANDB_MODE='offline',
        PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=os.pathsep.join((str(ROOT / 'src'), str(ROOT), os.environ.get('PYTHONPATH', ''))))


def validate_allocation(env):
    """Reject a wider or different physical allocation before creating output."""
    from fastwam.cosmos.job_identity import distributed_identity
    query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.total',
                            '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    return distributed_identity(env, socket.gethostname(), rows)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=tuple(VARIANTS))
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--teacher-source', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--check-only', action='store_true',
                        help='Validate prepared assets without allocating a model or launching training')
    args = parser.parse_args()
    variant = args.variant or default_variant(ROOT)
    spec = VARIANTS[variant]
    if spec['teacher'].startswith('dinov3') and os.environ.get('DINOV3_LICENSE_ACCEPTED') != '1':
        raise PermissionError('Set DINOV3_LICENSE_ACCEPTED=1 after accepting the model license')
    from omegaconf import OmegaConf
    from fastwam.cosmos.dense_encoder import _git_commit, teacher_spec, sha256_file
    config = ROOT / spec['config']
    cfg = OmegaConf.load(config)
    validate_recipe(cfg, variant)
    dataset, cache_root = args.dataset.resolve(), args.cache_root.resolve()
    info = json.loads((dataset / 'meta/info.json').read_text())
    if (info['total_episodes'], info['total_frames'], info['fps']) != (78, 86608, 30):
        raise ValueError('The RM71D78 20261029 dataset is required')
    teacher_source = (args.teacher_source or ROOT / 'third_party' / spec['vendor']).resolve()
    if _git_commit(teacher_source) != teacher_spec(spec['teacher'])['upstream_commit']:
        raise ValueError('Teacher source revision differs from pinned vendor')
    weights = resolve_weights(variant, args.weights)
    cache = verify_cache(cache_root, variant)
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    report = dict(status='prepared', variant=variant, teacher=spec['teacher'],
        code_revision=revision, config=str(config), config_sha256=sha256_file(config),
        dataset=str(dataset), dataset_source_sha256=DATASET_SHA256,
        cache_root=str(cache_root), cache_identity=cache['identity_sha256'],
        weights=str(weights), weight_sha256=spec['weight_sha256'],
        teacher_source=str(teacher_source), batch_per_rank=64, global_batch=256,
        max_steps=50000, precision='bf16', formal_training_started=False)
    if args.check_only:
        print(json.dumps(report), flush=True)
        return
    if args.run_root is None:
        parser.error('--run-root is required to start training')
    if not os.environ.get('TRAIN_JOB_ID'):
        raise RuntimeError('Launch from an allocated distributed training job')
    import torch
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 4
            or any('H200' not in torch.cuda.get_device_name(i) for i in range(4))):
        raise RuntimeError('Exactly four allocated H200 GPUs required')
    env = stage_environment()
    validate_allocation(env)
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / 'logs').mkdir()
    report.update(job_id=os.environ['TRAIN_JOB_ID'],
                  started_at_utc=datetime.now(timezone.utc).isoformat())
    for stage, command in stage_commands(config, run_root, dataset, cache_root,
                                          variant, teacher_source, weights):
        report.update(status='running_' + stage, command=command,
                      formal_training_started=stage == 'formal')
        write_json(run_root / 'claim.json', report)
        with (run_root / 'logs' / f'{stage}.log').open('x') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            report.update(status='failed_' + stage, exit_code=result.returncode)
            write_json(run_root / 'claim.json', report)
            raise RuntimeError(f'{stage} failed: see {run_root / "logs" / (stage + ".log")}')
        print(json.dumps(dict(stage=stage, status='finished')), flush=True)
    report.update(status='complete', completed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(run_root / 'claim.json', report)


if __name__ == '__main__':
    main()
