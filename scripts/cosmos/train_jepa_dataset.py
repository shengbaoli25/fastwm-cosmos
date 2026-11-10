"""Fresh source-bound ViT-L run, optionally with an observation-only auditor.

Default mode prints a plan. Execution builds new source-bound caches and
statistics, then runs the existing four-H200 update/restore gates and training.
No learned robot checkpoint, old normalization, or force teacher is used.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/cosmos')]
from fastwam.cosmos.dataset_spec import load_spec, verify_source, verify_prepared, verify_prompt_manifest
from train_rm71d78_jepa import add_jepa_config, recipe

SSD = Path('/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340')
HDD = Path('/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340')
GROUP = SSD / 'rm71d-jepa-clean30-20261107'
REFERENCE_GROUP = SSD / 'rm71d-reference-jepa-wipe-20261111'
OLD_GROUP = SSD / 'rm71d-jepa-force-latent-20261104'
VITL_SHA = '963f92a54541f128cb1249c9d34a94a5be5767729e8edeac51b3bf76d3a2b0f3'
REFERENCE_TARGET = 'fastwam.cosmos.reference_model.create_reference_jepa_cosmos'
REFERENCE_DATASET = 'fastwam.cosmos.reference_dataset.ReferenceCachedDataset'
REFERENCE_MODEL = dict(reference_hidden=128, reference_heads=4, reference_layers=2,
                       reference_seed=46, alignment_weight=.1)
REFERENCE_COARSENING = dict(max_stride=12, position_tolerance_m=.005,
                            rotation_tolerance_rad=.05, gripper_tolerance=.05)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-audit', action='store_true',
                        help='Add the observation-only reference auditor to JEPA; defaults to wipe30 source')
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--prepared', type=Path, help='Optional already verified clean30 prepared directory')
    text = parser.add_mutually_exclusive_group()
    text.add_argument('--text-cache', type=Path, help='Existing official Reason1 bank with exactly the recorded prompts')
    text.add_argument('--reason1-checkpoint', type=Path, help='Encode recorded prompts if no matching text bank exists')
    parser.add_argument('--jepa-checkpoint', type=Path, default=OLD_GROUP / 'assets/vjepa2_1_vitl_ema_encoder.extracted.pt')
    parser.add_argument('--checkpoint', '--cosmos-checkpoint', dest='checkpoint', type=Path,
        default=HDD / 'realman-noaction-joint8/assets/base/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt')
    parser.add_argument('--tokenizer', type=Path, default=HDD / 'realman-dualwm/checkpoints/tokenizer.pth')
    parser.add_argument('--upstream', type=Path,
        default=HDD / 'realman-noaction-joint8/code-82cb9d2/third_party/cosmos-predict2.5')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--plan', action='store_true')
    args = parser.parse_args(argv)
    if args.spec is None:
        name = 'jepa_wipe30_dataset.json' if args.reference_audit else 'jepa_clean30_dataset.json'
        args.spec = ROOT / 'configs/experiments' / name
    if args.dataset is None:
        args.dataset = REFERENCE_GROUP / 'data/wipe30' if args.reference_audit else GROUP / 'data/clean'
    return args


def plan_for(args):
    spec = load_spec(args.spec)
    group = REFERENCE_GROUP if args.reference_audit else GROUP
    output = args.output_root or group / 'vitl/online' / os.environ.get('TRAIN_JOB_ID', 'PLAN_ONLY')
    inherited = recipe()
    training = dict(inherited['shared'], source_episodes=spec['source_episodes'],
                    split_counts=spec['split_counts'])
    return dict(mode='execute' if args.execute else 'plan_only', dataset_spec=spec,
        dataset=args.dataset.as_posix(), output_root=output.as_posix(),
        prepared=(args.prepared or output / 'prepared').as_posix(),
        training_recipe=training, encoder=inherited['variants']['vitl'],
        model_additions=inherited['model_additions'],
        model_target=REFERENCE_TARGET if args.reference_audit else 'fastwam.cosmos.jepa_force_latent.create_jepa_force_latent_cosmos',
        reference_audit_enabled=args.reference_audit,
        reference_bank=(output / 'reference-bank').as_posix() if args.reference_audit else None,
        reference_model_additions=REFERENCE_MODEL if args.reference_audit else None,
        reference_coarsening=REFERENCE_COARSENING if args.reference_audit else None,
        text_cache=args.text_cache.as_posix() if args.text_cache else None,
        reason1_checkpoint=args.reason1_checkpoint.as_posix() if args.reason1_checkpoint else None,
        jepa_checkpoint=args.jepa_checkpoint.as_posix(), jepa_sha256=VITL_SHA,
        cosmos_checkpoint=args.checkpoint.as_posix(), tokenizer=args.tokenizer.as_posix(), upstream=args.upstream.as_posix(),
        model_warm_start=False, optimizer_resume=False, external_teacher_required=False,
        verification_pending=True)


def dataset_jepa_config(baseline, spec, **kwargs):
    """Reuse the actual existing transform; replace only dataset provenance."""
    from fastwam.cosmos.dataset_spec import validate_spec
    validate_spec(spec)
    config = add_jepa_config(baseline, variant='vitl', **kwargs)
    config['jepa_experiment'].update(raw_source_sha256=spec['source_inventory_sha256'],
        split_counts=spec['split_counts'], dataset_id=spec['dataset_id'], dataset_spec=spec)
    return config


def reference_jepa_config(baseline, spec, *, reference_bank, reference_identity, **kwargs):
    """Compose both branches while preserving the complete existing JEPA recipe."""
    from fastwam.cosmos.preflight import validate_config
    from fastwam.cosmos.reference_trajectory import AugmentationConfig
    from train_rm71d78_jepa import pinned_digest
    pinned_digest(reference_identity)
    config = dataset_jepa_config(baseline, spec, **kwargs)
    config['model'].update(_target_=REFERENCE_TARGET,
        expected_reference_identity=reference_identity, **REFERENCE_MODEL)
    config['data']['train']['base_config'].update(_target_=REFERENCE_DATASET,
        reference_bank=str(reference_bank), expected_reference_identity=reference_identity,
        reference_seed=42, reference_augmentation=asdict(AugmentationConfig()))
    config['reference_experiment'] = dict(reference_identity=reference_identity,
        source_inventory_sha256=spec['source_inventory_sha256'],
        reference_source='measured_TCP_complete_valid_episode',
        auditor_inputs='current_first_frame_latent_current_state_TCP_causal_force_history_full_reference',
        injection='Action_Expert_only', alignment_weight=.1,
        augmentation=asdict(AugmentationConfig()), coarsening=REFERENCE_COARSENING)
    validate_config(config)
    return config


def prepare_bound_reference(source, spec, cache_manifest, output, *, progress=None):
    """Build a fresh bank from the exact raw source and reject any lineage drift."""
    from fastwam.cosmos.reference_bank import prepare_bank
    from fastwam.cosmos.reference_binding import verify_binding
    verify_source(spec, source)
    bank = prepare_bank(source.root, output, expected_episodes=spec['source_episodes'],
        expected_source_sha=spec['source_inventory_sha256'], progress=progress,
        **REFERENCE_COARSENING)
    receipt = verify_binding(output, cache_manifest)
    if (receipt['status'] != 'bound' or receipt.get('full_raw_inventory_equal') is not True
            or receipt.get('episode_count') != spec['source_episodes']
            or receipt.get('continuous_segment_count') != spec['continuous_segments']
            or receipt.get('retained_anchors') != spec['training_anchors']
            or receipt.get('reference_identity_sha256') != bank['identity_sha256']):
        raise ValueError('New reference bank must match this exact dataset/cache: '+json.dumps(receipt))
    return bank, receipt


def encode_four(native, encoder_args, prepared, env):
    workers, logs = [], []
    try:
        for rank in range(4):
            log = (prepared / f'encode_rank{rank}.log').open('w')
            logs.append(log)
            workers.append(subprocess.Popen([str(native), '-u', *map(str, encoder_args), '--rank', str(rank)],
                env=dict(env, CUDA_VISIBLE_DEVICES=str(rank)), stdout=log, stderr=subprocess.STDOUT, cwd=ROOT))
        while any(worker.poll() is None for worker in workers):
            if any(worker.poll() not in (None, 0) for worker in workers):
                raise RuntimeError('A source feature encoder failed; see encode_rank logs')
            time.sleep(2)
        if any(worker.returncode != 0 for worker in workers):
            raise RuntimeError('Source feature encoding did not complete')
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        for log in logs:
            log.close()


def main():
    args = arguments()
    plan = plan_for(args)
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    native = Path(sys.executable)
    if (not os.environ.get('TRAIN_JOB_ID') or native.parent.name != 'bin'
            or native.parent.parent.name != '.venv'):
        raise ValueError('Use native .venv/bin/python in a new four-H200 platform job')
    output = Path(plan['output_root']).resolve()
    dataset = args.dataset.resolve()
    prepared = args.prepared.resolve() if args.prepared else output / 'prepared'
    if (not Path(plan['output_root']).is_absolute() or output.exists()
            or output == dataset or dataset in output.parents or output in dataset.parents
            or args.prepared and (output == prepared or prepared in output.parents or output in prepared.parents)):
        raise ValueError('Fresh absolute output outside dataset and reused caches required')
    if not args.prepared and not (args.text_cache or args.reason1_checkpoint):
        raise ValueError('Supply matching --text-cache or official --reason1-checkpoint for new prompts')
    from distributed_fresh import clean_environment
    env = clean_environment(os.environ, ROOT)
    env['NCCL_NVLS_ENABLE'] = '0'
    os.environ.clear()
    os.environ.update(env)
    import torch
    from omegaconf import OmegaConf
    from fastwam.cache.latent import file_sha256
    from fastwam.cosmos.assets import checked, read_text_bank, NONACTION_WEIGHT_SHA, TOKENIZER_SHA
    from fastwam.cosmos.cache import _atomic
    from fastwam.cosmos.native import verify_upstream
    from fastwam.cosmos.persistence import resource_identity
    from fastwam.cosmos.realman_joint import open_realman_source
    from fastwam.cosmos.jepa_training_data import build_observation_cache
    torch.set_num_threads(4)
    resource = resource_identity()
    checked(args.checkpoint, NONACTION_WEIGHT_SHA)
    checked(args.jepa_checkpoint, VITL_SHA)
    verify_upstream(args.upstream)
    if not args.prepared:
        checked(args.tokenizer, TOKENIZER_SHA)
    spec = plan['dataset_spec']
    source = open_realman_source(dataset, spec['source_inventory_sha256'], action_representation='delta_current_joint7')
    verify_source(spec, source)
    if args.text_cache:
        bank, text_manifest, _ = read_text_bank(args.text_cache)
        del bank
        verify_prompt_manifest(spec, text_manifest)
    output.mkdir(parents=True, exist_ok=False)
    work = output / 'resolve'
    work.mkdir()

    def status(phase, **extra):
        value = dict(phase=phase, status='running', updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        _atomic(output / 'pipeline-status.json', value)
        print(json.dumps(value), flush=True)

    def run(phase, command, extra_env=None):
        status(phase)
        with (work / (phase + '.log')).open('w') as log:
            subprocess.run([str(native), '-u', *map(str, command)], cwd=ROOT,
                env=dict(env, **(extra_env or {})), stdout=log, stderr=subprocess.STDOUT, check=True)

    _atomic(output / 'experiment.json', dict(plan, resource=resource, spec_sha256=file_sha256(args.spec)))
    _atomic(work / 'source_audit.json', dict(identity=source.identity, timing=source.timing))
    source.export_normalization(work / 'source-normalization.json')
    if not args.prepared:
        prepared.mkdir()
        source.export_normalization(prepared / 'normalization.json')
        if args.text_cache:
            (prepared / 'text').symlink_to(args.text_cache.resolve(), target_is_directory=True)
        else:
            _atomic(prepared / 'prompts.json', spec['prompts'])
            run('encode_recorded_task', [ROOT / 'scripts/cosmos/precompute_text_sdpa.py',
                '--prompts', prepared / 'prompts.json', '--reason1-checkpoint', args.reason1_checkpoint,
                '--upstream', args.upstream, '--output', prepared / 'text', '--execute'], {'CUDA_VISIBLE_DEVICES': '0'})
        encoder_args = [ROOT / 'scripts/cosmos/precompute_realman.py', '--dataset', dataset,
            '--source-sha', spec['source_inventory_sha256'], '--text-cache', prepared / 'text',
            '--tokenizer', args.tokenizer, '--upstream', args.upstream, '--output', prepared / 'features',
            '--stage', prepared / 'encoder-stage', '--action-representation', 'delta_current_joint7',
            '--expected-episodes', str(spec['source_episodes'])]
        status('encode_source_on_four_gpus')
        encode_four(native, encoder_args, prepared, env)
        run('assemble_source_features', [*encoder_args, '--assemble'])
    bank, text_manifest, _ = read_text_bank(prepared / 'text')
    del bank
    verify_prompt_manifest(spec, text_manifest)
    manifest = json.loads((prepared / 'features/manifest.json').read_text())
    norm = json.loads((prepared / 'normalization.json').read_text())
    verify_prepared(spec, source, manifest, norm)
    if norm != json.loads((work / 'source-normalization.json').read_text()):
        raise ValueError('Prepared normalization differs from freshly exported source normalization')
    reference = None
    if args.reference_audit:
        status('prepare_and_bind_reference_bank')
        reference, binding = prepare_bound_reference(source, spec, prepared / 'features/manifest.json',
            output / 'reference-bank', progress=lambda done, total, points: status(
                'prepare_and_bind_reference_bank', episodes=done, total_episodes=total,
                last_reference_points=points))
        _atomic(work / 'reference_cache_binding.json', binding)
    run('resolve_baseline_recipe', [ROOT / 'scripts/cosmos/resolve_config.py',
        '--task', 'cosmos_noaction_joint_4gpu_30k', '--batch-size', '64', '--max-steps', '30000',
        '--scheduler-total-steps', '50000', '--action-representation', 'delta_current_joint7',
        '--normalization', prepared / 'normalization.json', '--text-cache', prepared / 'text',
        '--feature-cache', prepared / 'features', '--checkpoint', args.checkpoint,
        '--upstream', args.upstream, '--output-dir', work / 'baseline-placeholder',
        '--config-output', work / 'baseline-resolved.yaml'])
    # Own this attempt's observation cache: no mutation of reused prepared data.
    observations = output / 'observations'
    status('prepare_observed_rgb')
    observed = build_observation_cache(source, manifest, observations,
        on_episode=lambda done, total: status('prepare_observed_rgb', episodes=done, total_episodes=total))
    del source
    base = OmegaConf.to_container(OmegaConf.load(work / 'baseline-resolved.yaml'), resolve=True, throw_on_missing=True)
    reference_options = dict(reference_bank=output / 'reference-bank',
        reference_identity=reference['identity_sha256']) if reference is not None else {}
    transform = reference_jepa_config if args.reference_audit else dataset_jepa_config
    final = transform(base, spec, **reference_options, checkpoint=args.jepa_checkpoint.resolve(),
        checkpoint_sha256=VITL_SHA, observation_cache=observations,
        observation_identity=observed['identity_sha256'], output_dir=work / 'jepa-placeholder')
    resolved = work / 'resolved.yaml'
    with resolved.open('x') as stream:
        OmegaConf.save(OmegaConf.create(final), stream)
    _atomic(output / 'experiment.json', dict(plan, resource=resource, verification_pending=False,
        spec_sha256=file_sha256(args.spec), resolved_config_sha256=file_sha256(resolved),
        baseline_cache_identity=manifest['identity_sha256'], observation_identity=observed['identity_sha256'],
        reference_identity=reference['identity_sha256'] if reference is not None else None,
        normalization_sha256=file_sha256(prepared / 'normalization.json')))
    status('gpu_update_restore_gates_then_fresh_training')
    os.execv(str(native), [str(native), '-u', str(ROOT / 'scripts/cosmos/distributed_chunk.py'),
        '--config', str(resolved), '--run-root', str(output / 'run'), '--native-python', str(native), '--execute'])


if __name__ == '__main__':
    main()
