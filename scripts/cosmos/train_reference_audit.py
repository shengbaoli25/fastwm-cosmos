"""Resolve and train the observation-only reference auditor with the existing WAM.

Reuses the complete 78-episode delta cache, text bank and normalization. There is
no raw-video decode, VAE encode or GPT trajectory collection in this entrypoint.
Default is a non-executing plan; --resolve-only verifies assets on CPU and
--execute runs the four-rank update/save/restore gates before formal training.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/cosmos')]

SSD = '/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340'
HDD = '/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340'
GROUP = SSD+'/rm71d-reference-audit-20261105'
PREPARED = SSD+'/rm71d-cosmos-videoisolated-delta30k-20261103/online/job-83b69f78-eedb-40d3-9f6d-1fad8a22cba1/prepared'
CACHE_SHA = '324c730e617e157300f107bf39defa4ab5b949dde022f6d5043b36dacb107144'
REFERENCE_SHA = '7d0a0a108d8d338633439358548cd7e8a32b275c74420444c0ce652cf8ddd2b9'
UPSTREAM = HDD+'/realman-noaction-joint8/code-82cb9d2/third_party/cosmos-predict2.5'
CHECKPOINT = HDD+'/realman-noaction-joint8/assets/base/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='Print recipe and paths without opening assets (default)')
    mode.add_argument('--resolve-only', action='store_true', help='CPU asset verification and resolved config; no GPU training')
    mode.add_argument('--execute', action='store_true', help='On Qizhi: resolve, four-rank gates, then train')
    p.add_argument('--prepared', default=PREPARED)
    p.add_argument('--reference-bank', default=GROUP+'/reference-bank/rm71d78-v1')
    p.add_argument('--expected-cache-identity', default=CACHE_SHA)
    p.add_argument('--expected-reference-identity', default=REFERENCE_SHA)
    p.add_argument('--checkpoint', default=CHECKPOINT)
    p.add_argument('--upstream', default=UPSTREAM)
    p.add_argument('--work', help='New config/verification directory outside immutable input banks')
    p.add_argument('--run-root', help='New training run directory (defaults to training/$TRAIN_JOB_ID)')
    p.add_argument('--num-workers', type=int, default=8)
    # Intentionally fixed to the recipe explicitly authorized for this run.
    p.add_argument('--batch-size', type=int, choices=[64], default=64)
    p.add_argument('--max-steps', type=int, choices=[30000], default=30000)
    p.add_argument('--scheduler-total-steps', type=int, choices=[50000], default=50000)
    return p


def describe(args):
    job = os.environ.get('TRAIN_JOB_ID', '${TRAIN_JOB_ID}')
    return dict(
        mode='execute' if args.execute else 'resolve_only' if args.resolve_only else 'plan_only',
        ranks=4, gpu='H200', batch_per_rank=args.batch_size, accumulation=1,
        global_batch=4*args.batch_size, max_steps=args.max_steps,
        scheduler_total_steps=args.scheduler_total_steps, warmup_steps=2500,
        prepared=args.prepared, reference_bank=args.reference_bank,
        expected_cache_identity=args.expected_cache_identity,
        expected_reference_identity=args.expected_reference_identity,
        checkpoint=args.checkpoint, upstream=args.upstream,
        work=args.work or GROUP+'/resolved/'+job,
        run_root=args.run_root or GROUP+'/training/'+job,
        full_reference='variable M, all episode; padding only in collate',
        auditor_input='current first-frame latent, current state/TCP, causal force history, full reference',
        auditor_output='32 local tokens for Action Expert context only',
        model_output='unchanged 32xJoint8 delta action chunk',
        augmentation='CPU sparse/full plan + epoch-random length/noise/point-span-mask/reference-dropout',
        episode_selection='all78_no_holdout', vae_reencode=False, gpt_data_required=False,
        warm_start=False, optimizer_resume=False,
        gates=['complete cache/text/normalization/reference provenance',
               'four-rank two Adam updates and save', 'independent four-rank restore', 'formal30000'],
    )


def resolve(args, plan):
    from fastwam.cosmos.reference_bank import atomic_json
    from fastwam.cosmos.reference_binding import verify_binding
    from fastwam.cosmos.reference_trajectory import AugmentationConfig
    from fastwam.cosmos.preflight import validate_config
    from omegaconf import OmegaConf

    prepared, bank = Path(args.prepared).resolve(), Path(args.reference_bank).resolve()
    work, run = Path(plan['work']).resolve(), Path(plan['run_root']).resolve()
    if (work.exists() or run.exists() or work == run or work in run.parents or run in work.parents
            or any(out == protected or protected in out.parents or out in protected.parents
                   for out in (work,run) for protected in (prepared,bank))):
        raise ValueError('Use new, separate work/run directories outside both immutable input banks')
    if args.num_workers < 0:
        raise ValueError('num-workers must be nonnegative')
    receipt = verify_binding(bank, prepared/'features/manifest.json')
    if (receipt['status'] != 'bound' or receipt.get('episode_count') != 78
            or receipt.get('cache_identity_sha256') != args.expected_cache_identity
            or receipt.get('reference_identity_sha256') != args.expected_reference_identity):
        raise ValueError('Expected all78 reference/cache binding rejected: '+json.dumps(receipt))
    if not (prepared/'normalization.json').is_file():
        raise FileNotFoundError('Existing prepared/normalization.json is required')
    work.mkdir(parents=True, exist_ok=False)
    atomic_json(work/'reference_cache_binding.json', receipt)
    atomic_json(work/'plan.json', plan)
    # Existing resolver performs complete shard integrity, normalization binding,
    # text pairing, pinned official checkpoint SHA and upstream source checks.
    command = [sys.executable, '-u', str(ROOT/'scripts/cosmos/resolve_config.py'),
        '--task', 'cosmos_noaction_joint_4gpu_30k',
        '--batch-size', str(args.batch_size), '--max-steps', str(args.max_steps),
        '--scheduler-total-steps', str(args.scheduler_total_steps),
        '--action-representation', 'delta_current_joint7',
        '--normalization', str(prepared/'normalization.json'),
        '--text-cache', str(prepared/'text'), '--feature-cache', str(prepared/'features'),
        '--checkpoint', args.checkpoint, '--upstream', args.upstream,
        '--output-dir', str(work/'config-placeholder'),
        '--config-output', str(work/'baseline-resolved.yaml')]
    print('REFERENCE_AUDIT_VERIFYING_CACHE log='+str(work/'resolve.log'), flush=True)
    try:
        with (work/'resolve.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print((work/'resolve.log').read_text(encoding='utf-8',errors='replace')[-8000:],
              file=sys.stderr, flush=True)
        raise
    cfg = OmegaConf.to_container(OmegaConf.load(work/'baseline-resolved.yaml'), resolve=True)
    if cfg['data']['train']['expected_identity_sha256'] != args.expected_cache_identity:
        raise ValueError('Resolver selected a different cache identity')
    cfg['num_workers'] = args.num_workers
    cfg['model'].update(
        _target_='fastwam.cosmos.reference_model.create_reference_joint_cosmos',
        expected_reference_identity=args.expected_reference_identity,
        reference_hidden=128, reference_heads=4, reference_layers=2,
        reference_seed=46, alignment_weight=.1)
    cfg['data']['train'].update(
        _target_='fastwam.cosmos.reference_dataset.ReferenceCachedDataset',
        reference_bank=str(bank), expected_reference_identity=args.expected_reference_identity,
        reference_seed=42, reference_augmentation=asdict(AugmentationConfig()))
    validate_config(cfg)
    config = work/'resolved.yaml'
    with config.open('x', encoding='utf-8') as stream:
        OmegaConf.save(OmegaConf.create(cfg), stream)
    atomic_json(work/'cache_reuse.json', dict(
        status='verified', cache_identity=args.expected_cache_identity,
        reference_identity=args.expected_reference_identity,
        normalization=str(prepared/'normalization.json'), prepared=str(prepared),
        all_episodes=78, cache_tensor_payloads_verified=True,
        vae_reencoded=False, warm_start=False, optimizer_resume=False,
        gpu_preflight_passed=False, formal_training_started=False))
    print('REFERENCE_AUDIT_CONFIG_VERIFIED '+str(config), flush=True)
    return config, run


def main(argv=None):
    args = parser().parse_args(argv)
    plan = describe(args)
    if not (args.resolve_only or args.execute):
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    if (not args.work or not args.run_root) and not os.environ.get('TRAIN_JOB_ID'):
        raise ValueError('Outside a Qizhi job, pass explicit fresh --work and --run-root')
    if args.execute and not os.environ.get('TRAIN_JOB_ID'):
        raise ValueError('--execute requires a platform training job with TRAIN_JOB_ID')
    from distributed_fresh import clean_environment
    env = clean_environment(os.environ, ROOT)
    os.environ.clear()
    os.environ.update(env)
    config, run = resolve(args, plan)
    if args.resolve_only:
        return 0
    run.parent.mkdir(parents=True, exist_ok=True)
    os.execv(sys.executable, [sys.executable, '-u', str(ROOT/'scripts/cosmos/distributed_chunk.py'),
        '--config', str(config), '--run-root', str(run), '--native-python', sys.executable, '--execute'])


if __name__ == '__main__':
    raise SystemExit(main())
