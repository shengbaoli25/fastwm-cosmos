"""Matched ViT-B/L observation experiments from official initialization.

The all78 delta baseline and its cached targets are immutable. Both variants
share observed RGB prepared under a file lock; Force latent labels are computed
online with stopped gradients, without a pretrained robot-policy teacher.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SSD = Path('/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340')
HDD = Path('/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340')
GROUP = SSD / 'rm71d-jepa-force-latent-20261104'
PREPARED = SSD / 'rm71d-cosmos-videoisolated-delta30k-20261103/online/job-83b69f78-eedb-40d3-9f6d-1fad8a22cba1/prepared'
DATASET = SSD / 'rm71d-jepa-cosmos-50k-20261029/data/rm71d_plug_transfer_20261029'
RAW_SHA = 'b24b701239f54ad38d16c819b9ea9f6fad5f9374a683a84a68d2cd7ef27067a8'
RECIPE_PATH = ROOT / 'configs/experiments/jepa_force_variants.json'
JEPA_TARGET = 'fastwam.cosmos.jepa_force_latent.create_jepa_force_latent_cosmos'
JEPA_DATASET = 'fastwam.cosmos.jepa_training_data.JepaForceTrainingDataset'


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['vitb', 'vitl'], required=True)
    parser.add_argument('--jepa-checkpoint', type=Path, required=True)
    parser.add_argument('--jepa-sha256', required=True)
    parser.add_argument('--prepared', type=Path, default=PREPARED)
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--source-sha', default=RAW_SHA)
    parser.add_argument('--observation-cache', type=Path, default=GROUP / 'prepared/observations')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--query-redundancy-recipe', type=Path,
        help='Optional spatial-query regularizer recipe; omitted preserves the original experiment')
    parser.add_argument('--upstream', type=Path, default=HDD / 'realman-noaction-joint8/code-82cb9d2/third_party/cosmos-predict2.5')
    parser.add_argument('--checkpoint', '--cosmos-checkpoint', dest='checkpoint', type=Path,
        default=HDD / 'realman-noaction-joint8/assets/base/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--plan', action='store_true', help='Default: print the plan without filesystem or training changes')
    return parser.parse_args(argv)


def pinned_digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('An explicit lowercase SHA256 of the selected official JEPA checkpoint is required')
    return value


def recipe():
    result = json.loads(RECIPE_PATH.read_text())
    if result.get('schema') != 'rm71d78-jepa-force-latent-recipe-v1':
        raise ValueError('Unknown JEPA experiment recipe')
    return result


def query_redundancy_recipe(path):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    fields = {'query_redundancy_weight', 'query_redundancy_margin', 'query_redundancy_warmup_steps'}
    if (data.get('schema') != 'jepa-spatial-query-redundancy-v1'
            or set(data) != fields | {'schema'}):
        raise ValueError('Expected the explicit spatial query redundancy recipe')
    weight, margin, steps = (data[name] for name in
        ('query_redundancy_weight', 'query_redundancy_margin', 'query_redundancy_warmup_steps'))
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight <= 0
            or isinstance(margin, bool) or not isinstance(margin, (int, float))
            or not math.isfinite(margin) or not 0 <= margin < 1
            or isinstance(steps, bool) or not isinstance(steps, int) or steps < 0):
        raise ValueError('Positive finite query weight, margin in [0,1), and integer warmup required')
    return {name: data[name] for name in fields}


def plan_for(args):
    pinned_digest(args.jepa_sha256)
    if args.source_sha != RAW_SHA:
        raise ValueError('Use the same immutable all78 source as the delta baseline')
    data = recipe()
    additions = dict(data['model_additions'])
    additions.update(query_redundancy_recipe(args.query_redundancy_recipe))
    output = args.output_root or GROUP / args.variant / 'online' / os.environ.get('TRAIN_JOB_ID', 'PLAN_ONLY')
    return dict(mode='execute' if args.execute else 'plan_only', source=str(ROOT), variant=args.variant,
                recipe=data['shared'], encoder=data['variants'][args.variant], model_additions=additions,
                dataset=str(args.dataset), raw_source_sha256=RAW_SHA, prepared=str(args.prepared),
                observation_cache=str(args.observation_cache), output_root=str(output),
                cosmos_checkpoint=str(args.checkpoint), jepa_checkpoint=str(args.jepa_checkpoint),
                jepa_sha256=args.jepa_sha256, model_warm_start=False, optimizer_resume=False,
                baseline_mutation=False, verification_pending=True)


def add_jepa_config(baseline, *, variant, checkpoint, checkpoint_sha256,
                    observation_cache, observation_identity, output_dir, query_redundancy=None):
    """Pure config transform, retaining every original data/optimizer setting."""
    from fastwam.cosmos.preflight import validate_config
    validate_config(baseline)
    shared = recipe()['shared']
    expected = dict(batch_size=64, max_steps=30000, scheduler_total_steps=50000,
                    learning_rate=1e-4, weight_decay=.01, num_workers=8,
                    gradient_accumulation_steps=1, mixed_precision='bf16',
                    seed=42, save_every=5000, log_every=10, eval_every=0,
                    num_epochs=10, max_grad_norm=1., eval_num_inference_steps=16, resume=None)
    if any(baseline.get(key) != value for key, value in expected.items()):
        raise ValueError('The existing B64/30000-update/50000-scheduler recipe must be preserved')
    if (baseline['model']['_target_'] != 'fastwam.cosmos.assets.create_noaction_joint_cosmos'
            or baseline['model'].get('action_representation') != 'delta_current_joint7'
            or baseline['data']['train'].get('expected_action_representation') != 'delta_current_joint7'):
        raise ValueError('The unchanged all78 current-measured-joint delta baseline is required')
    if variant not in recipe()['variants']:
        raise ValueError('Explicit vitb or vitl variant required')
    pinned_digest(checkpoint_sha256)
    pinned_digest(observation_identity)
    cfg = copy.deepcopy(baseline)
    cfg['output_dir'] = str(output_dir)
    cfg['model'].update(recipe()['model_additions'])
    if query_redundancy:
        allowed = {'query_redundancy_weight', 'query_redundancy_margin', 'query_redundancy_warmup_steps'}
        if set(query_redundancy) != allowed:
            raise ValueError('Query regularization may only change its three dedicated loss settings')
        cfg['model'].update(query_redundancy)
    cfg['model']['_target_'] = JEPA_TARGET
    cfg['model']['observation_encoder'] = dict(variant=variant, checkpoint=str(checkpoint),
        expected_sha256=checkpoint_sha256, checkpoint_suffix=True)
    cfg['data']['train'] = dict(_target_=JEPA_DATASET, _recursive_=False,
        base_config=copy.deepcopy(baseline['data']['train']), observation_cache_root=str(observation_cache),
        expected_observation_identity=observation_identity)
    cfg['jepa_experiment'] = dict(variant=variant, external_teacher_required=False,
        appearance_consistency_weight=shared['appearance_consistency_weight'],
        target='online_current_force_clean_observed_only_stop_gradient',
        baseline_initialization='official_cosmos_fresh_action_force',
        raw_source_sha256=RAW_SHA, split_counts=shared['split_counts'])
    if query_redundancy:
        cfg['jepa_experiment']['query_regularization'] = dict(query_redundancy,
            target='within_view_reader_attention_cosine_hinge',
            claim='penalizes_near_duplicate_queries_not_contact_supervision')
    validate_config(cfg)
    return cfg


def verify_all78_source(source, cache_manifest, prepared_normalization, exported_normalization):
    if (len(source.lengths) != 78 or source.identity.get('source_episode_indices') != list(range(78))
            or source.identity.get('split_counts') != dict(train=78, val=0, test=0)
            or source.identity.get('action_representation') != 'delta_current_joint7'
            or cache_manifest['status'] != 'complete'
            or cache_manifest['completed_samples'] != len(source)
            or cache_manifest['identity'].get('realman_source') != source.identity):
        raise ValueError('Existing complete all78 delta cache must match the original raw source exactly')
    if json.loads(Path(prepared_normalization).read_text()) != json.loads(Path(exported_normalization).read_text()):
        raise ValueError('Reused baseline normalization differs from the unchanged source/window statistics')


def main():
    args = arguments()
    plan = plan_for(args)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    if not os.environ.get('TRAIN_JOB_ID'):
        raise ValueError('Run as the foreground command of a new four-H200 platform job')
    native = Path(sys.executable)
    if native.parent.name != 'bin' or native.parent.parent.name != '.venv':
        raise ValueError('Use the original native .venv/bin/python required by the GPU gates')
    output = Path(plan['output_root']).resolve()
    prepared = args.prepared.resolve()
    if (not Path(plan['output_root']).is_absolute() or output.exists()
            or output == prepared or output in prepared.parents or prepared in output.parents):
        raise ValueError('Use a fresh absolute output directory outside the baseline cache')
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT), str(ROOT / 'scripts/cosmos')]
    from distributed_fresh import clean_environment
    env = clean_environment(os.environ, ROOT)
    env['NCCL_NVLS_ENABLE'] = '0'
    os.environ.clear()
    os.environ.update(env)
    import torch
    from omegaconf import OmegaConf
    from fastwam.cache.latent import file_sha256
    from fastwam.cosmos.assets import checked, NONACTION_WEIGHT_SHA
    from fastwam.cosmos.cache import _atomic
    from fastwam.cosmos.realman_joint import open_realman_source
    from fastwam.cosmos.jepa_training_data import build_observation_cache
    from fastwam.cosmos.jepa_observation import variant_spec
    from fastwam.cosmos.persistence import resource_identity
    torch.set_num_threads(4)
    resource = resource_identity()
    checked(args.checkpoint, NONACTION_WEIGHT_SHA)
    checked(args.jepa_checkpoint, args.jepa_sha256)
    if variant_spec(args.variant)['filename'] != recipe()['variants'][args.variant]['filename']:
        raise ValueError('Encoder implementation and matched-variant recipe differ')
    output.mkdir(parents=True, exist_ok=False)
    work = output / 'resolve'
    work.mkdir()

    def status(phase, state='running', **extra):
        receipt = dict(phase=phase, status=state, variant=args.variant,
                       updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        _atomic(output / 'pipeline-status.json', receipt)
        print(json.dumps(receipt), flush=True)

    _atomic(output / 'experiment.json', dict(plan, resource=resource, verification_pending=True,
            recipe_sha256=file_sha256(RECIPE_PATH)))
    status('verify_all78_existing_cache')
    source = open_realman_source(args.dataset, RAW_SHA, action_representation='delta_current_joint7')
    source.export_normalization(work / 'source-normalization.json')
    cache_manifest = json.loads((prepared / 'features/manifest.json').read_text())
    verify_all78_source(source, cache_manifest, prepared / 'normalization.json', work / 'source-normalization.json')
    status('resolve_official_baseline_recipe')
    resolver = [str(native), '-u', str(ROOT / 'scripts/cosmos/resolve_config.py'),
        '--task', 'cosmos_noaction_joint_4gpu_30k', '--batch-size', '64', '--max-steps', '30000',
        '--scheduler-total-steps', '50000', '--action-representation', 'delta_current_joint7',
        '--normalization', str(prepared / 'normalization.json'), '--text-cache', str(prepared / 'text'),
        '--feature-cache', str(prepared / 'features'), '--checkpoint', str(args.checkpoint),
        '--upstream', str(args.upstream), '--output-dir', str(work / 'config-placeholder'),
        '--config-output', str(work / 'baseline-resolved.yaml')]
    with (work / 'resolve.log').open('w') as log:
        subprocess.run(resolver, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    # The platform filesystem is Linux. A kernel file lock is released if a job
    # exits; a second variant can resume only already verified episode payloads.
    import fcntl
    observation_root = args.observation_cache.resolve()
    observation_root.parent.mkdir(parents=True, exist_ok=True)
    status('waiting_for_shared_observed_rgb')
    with (observation_root.parent / (observation_root.name + '.lock')).open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        status('prepare_or_verify_observed_rgb')
        observed = build_observation_cache(source, cache_manifest, observation_root,
            resume=observation_root.exists(),
            on_episode=lambda done, total: status('prepare_or_verify_observed_rgb',
                                                  episodes=done, total_episodes=total))
    del source
    baseline = OmegaConf.to_container(OmegaConf.load(work / 'baseline-resolved.yaml'), resolve=True, throw_on_missing=True)
    final = add_jepa_config(baseline, variant=args.variant, checkpoint=args.jepa_checkpoint.resolve(),
        checkpoint_sha256=args.jepa_sha256, observation_cache=observation_root,
        observation_identity=observed['identity_sha256'], output_dir=work / 'jepa-placeholder',
        query_redundancy=query_redundancy_recipe(args.query_redundancy_recipe))
    resolved = work / 'resolved.yaml'
    with resolved.open('x') as stream:
        OmegaConf.save(OmegaConf.create(final), stream)
    _atomic(output / 'experiment.json', dict(plan, resource=resource, verification_pending=False,
        recipe_sha256=file_sha256(RECIPE_PATH), resolved_config_sha256=file_sha256(resolved),
        baseline_cache_identity=cache_manifest['identity_sha256'],
        observation_identity=observed['identity_sha256'],
        normalization_sha256=file_sha256(prepared / 'normalization.json')))
    status('gpu_update_restore_gates_then_fresh_training')
    os.execv(str(native), [str(native), '-u', str(ROOT / 'scripts/cosmos/distributed_chunk.py'),
        '--config', str(resolved), '--run-root', str(output / 'run'),
        '--native-python', str(native), '--execute'])


if __name__ == '__main__':
    main()
