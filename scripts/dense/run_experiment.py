"""One-shot, immutable 20k run: encode teacher cache, gate, restore, train."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = '9846d4ac5fff6c3c0d96c306eeb5d5d8dc4aea0b176327c0f6bebc1baef980f2'
DATASET_DEFAULT = Path('/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-joint8-pluginsertion71-20261023/data/2026_10_23_lerobot_plug_insertion_100s')
RUNS_DEFAULT = Path('/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-joint8-densefeature-20261026')
TEACHERS = {
    'dinov3': dict(model='facebook/dinov3-vitb16-pretrain-lvd1689m',
        revision='23d0280ae6ee4ced592a3459674ad027d3c18906',
        filename='model.safetensors',
        sha256='9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b'),
    'vjepa21': dict(model='ChongZheng/V-JEPA-2.1-384',
        revision='b35cd7c431265ce6cad8f2af85863e201bc07115',
        filename='vjepa2_1_vitb_dist_vitG_384.pt',
        sha256='848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d'),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + '\n')
    temporary.replace(path)


def resolve_teacher_weights(teacher, explicit_path):
    """Resolve a pre-staged local teacher checkpoint without network fallback.

    The distributed job must never attempt a model download: a partially
    downloaded teacher can otherwise leave the cache looking usable while
    producing a late, non-reproducible failure during feature encoding.
    """
    if explicit_path is None:
        raise ValueError(
            f'Pass --weights with the pre-staged local {teacher} checkpoint; '
            'network downloads are disabled for distributed training')
    target = Path(explicit_path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f'Pre-staged teacher checkpoint is missing: {target}')
    pinned = TEACHERS[teacher]
    if target.name != pinned['filename']:
        raise ValueError(
            f'{teacher} requires a local file named {pinned["filename"]}; got {target.name}')
    if sha256(target) != pinned['sha256']:
        raise ValueError('Pre-staged teacher checkpoint SHA256 differs from the pinned artifact')
    return target


def stage_environment():
    """Use only pre-staged assets in every encoding and training subprocess."""
    return dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', WANDB_MODE='offline',
        PYTHONPATH=f"{ROOT / 'src'}:{ROOT}:{os.environ.get('PYTHONPATH','')}")


def source_revision():
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    expected = os.environ.get('EXPECTED_CODE_REVISION')
    if expected and revision != expected:
        raise ValueError(f'Checked-out source revision {revision} differs from expected {expected}')
    return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', choices=('vjepa21', 'dinov3'), required=True)
    parser.add_argument('--dataset', type=Path, default=DATASET_DEFAULT)
    parser.add_argument('--source-sha', default=SOURCE_SHA)
    parser.add_argument('--teacher-source', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--run-root', type=Path)
    args = parser.parse_args()

    if not os.environ.get('TRAIN_JOB_ID'):
        raise RuntimeError('Submit through the distributed-training platform')
    code_revision = source_revision()
    if args.source_sha != SOURCE_SHA or not args.dataset.is_dir():
        raise ValueError('The audited PlugInsertion71 source is required')
    if args.teacher == 'dinov3' and os.environ.get('DINOV3_LICENSE_ACCEPTED') != '1':
        raise PermissionError('Meta DINOv3 terms must be accepted before this job uses the materials')
    args.teacher_source = (args.teacher_source or ROOT / 'third_party' /
        ('dinov3' if args.teacher == 'dinov3' else 'vjepa21')).resolve()
    if not args.teacher_source.is_dir() or not (args.teacher_source / 'REVISION').is_file():
        raise FileNotFoundError('Pinned teacher source and revision must be present in this branch')
    args.weights = resolve_teacher_weights(args.teacher, args.weights)

    import torch
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 4
            or any('H200' not in torch.cuda.get_device_name(i) for i in range(4))):
        raise RuntimeError('This recipe requires exactly four H200 GPUs')
    if torch.cuda.get_device_properties(0).total_memory < 130 * 2**30:
        raise RuntimeError('Each allocated H200 must expose at least 130 GiB')

    job_id = os.environ['TRAIN_JOB_ID']
    run_root = (args.run_root or RUNS_DEFAULT / args.teacher / job_id).resolve()
    if run_root.exists():
        raise FileExistsError('Run path already exists; use a new platform job and output root')
    run_root.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(run_root.parent).free < 250 * 2**30:
        raise RuntimeError('At least 250 GiB free on the output filesystem is required')
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / 'logs').mkdir()
    claim = dict(status='claimed', job_id=job_id, teacher=args.teacher,
        code_revision=code_revision, modelscope_model=TEACHERS[args.teacher]['model'],
        modelscope_revision=TEACHERS[args.teacher]['revision'],
        checkpoint_filename=TEACHERS[args.teacher]['filename'],
        teacher_source_revision=('6876159a11b4df116f30f667f8c9888617df0751'
            if args.teacher == 'dinov3' else '204698b45b3712590f06245fbfba32d3be539812'),
        source_sha256=args.source_sha, dataset=str(args.dataset.resolve()),
        teacher_source=str(args.teacher_source.resolve()), weights=str(args.weights.resolve()),
        weight_sha256=sha256(args.weights), ranks=4, gpu='H200', batch_per_rank=64,
        global_batch=256, max_steps=20000, save_every=5000, learning_rate=1e-4,
        precision='bf16', accumulation=1, fresh_model=True, fresh_optimizer=True,
        started_at_utc=datetime.now(timezone.utc).isoformat(), formal_training_started=False)
    with (run_root / 'claim.json').open('x') as stream:
        json.dump(claim, stream, indent=2)
    cache_root = run_root / 'features'
    common = ['--dataset', str(args.dataset.resolve()), '--source-sha', args.source_sha,
        '--teacher', args.teacher, '--teacher-source', str(args.teacher_source.resolve()),
        '--weights', str(args.weights.resolve()), '--weight-sha', claim['weight_sha256']]
    encoder = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
        '--nnodes=1', '--nproc_per_node=4', str(ROOT / 'scripts/dense/precompute_features.py'),
        *common, '--output', str(cache_root)]
    assembler = [sys.executable, '-u', str(ROOT / 'scripts/dense/precompute_features.py'),
        *common, '--output', str(cache_root), '--assemble']
    task_config = ROOT / 'configs/experiments' / f'{args.teacher}_pluginsertion71_4h200_20k.yaml'
    training_args = ['--config', str(task_config), '--run-root', str(run_root),
        '--dataset', str(args.dataset.resolve()), '--source-sha', args.source_sha,
        '--cache-root', str(cache_root), '--teacher', args.teacher,
        '--teacher-source', str(args.teacher_source.resolve()), '--weights', str(args.weights.resolve()),
        '--weight-sha', claim['weight_sha256']]
    accelerate = [sys.executable, '-u', '-m', 'accelerate.commands.launch', '--config_file',
        str(ROOT / 'scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml'), '--num_processes', '4']
    stages = [
        ('encode', encoder),
        ('assemble', assembler),
        ('preflight', accelerate + [str(ROOT / 'scripts/dense/preflight.py'), *training_args]),
        ('restore', accelerate + [str(ROOT / 'scripts/dense/restore.py'), *training_args]),
        ('formal', accelerate + [str(ROOT / 'scripts/dense/train.py'), *training_args]),
    ]
    env = stage_environment()
    for name, command in stages:
        if name == 'formal':
            claim['formal_training_started'] = True
        claim.update(status='running_' + name, command=command,
                     updated_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(run_root / 'claim.json', claim)
        log_path = run_root / 'logs' / f'{name}.log'
        print(json.dumps(dict(stage=name, status='starting', command=command)), flush=True)
        with log_path.open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f'[{name}] {line}', end='', flush=True)
            returncode = process.wait()
        claim['stage_exit_codes'] = {**claim.get('stage_exit_codes', {}), name: returncode}
        write_json(run_root / 'claim.json', claim)
        print(json.dumps(dict(stage=name, status='finished', exit_code=returncode)), flush=True)
        if returncode != 0:
            claim.update(status='failed_' + name,
                error=f'Exit {returncode}; inspect {log_path}',
                updated_at_utc=datetime.now(timezone.utc).isoformat())
            write_json(run_root / 'claim.json', claim)
            raise RuntimeError(claim['error'])
        if name == 'preflight':
            gate = json.loads((run_root / 'preflight.json').read_text())
            if gate.get('status') != 'two_update_checkpoint_verified':
                claim.update(status='failed_preflight_gate', updated_at_utc=datetime.now(timezone.utc).isoformat())
                write_json(run_root / 'claim.json', claim)
                raise RuntimeError('Four-rank training update/save gate failed')
            claim['formal_training_started'] = False
        elif name == 'restore':
            restore = json.loads((run_root / 'restore/restore.json').read_text())
            if restore.get('status') != 'fresh_four_rank_restore_verified':
                claim.update(status='failed_restore_gate', updated_at_utc=datetime.now(timezone.utc).isoformat())
                write_json(run_root / 'claim.json', claim)
                raise RuntimeError('Four-rank full-state restore gate failed')
    claim.update(status='complete', formal_training_started=True,
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(run_root / 'claim.json', claim)


if __name__ == '__main__':
    main()
