"""Keep new branches tied to the same data, downstream geometry and optimizer."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from fastwam.cache.latent import cache_identity_sha256
from fastwam.cosmos.backbone_experiments import (
    DATASET_SHA256, DATASET_FRAMES, VARIANTS, validate_recipe, default_variant,
)
from fastwam.cosmos.jepa_video_cache import identity

ROOT = Path(__file__).resolve().parents[2]


def load_script(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('variant', tuple(VARIANTS))
def test_exact_configs_and_rejection_of_old_training_recipe(variant):
    cfg = OmegaConf.load(ROOT / VARIANTS[variant]['config'])
    validate_recipe(cfg, variant)
    cfg.model.depth = 12
    with pytest.raises(ValueError, match='matched RM71D78'):
        validate_recipe(cfg, variant)


def test_stage_plan_only_trains_from_already_complete_cache():
    runner = load_script('scripts/backbones/run.py')
    for variant, spec in VARIANTS.items():
        commands = runner.stage_commands('config', 'run', 'dataset', 'cache', variant, 'vendor', 'weights')
        assert [name for name, _ in commands] == ['preflight', 'restore', 'formal']
        for _, command in commands:
            assert command[command.index('--teacher') + 1] == spec['teacher']
            assert command[command.index('--weight-sha') + 1] == spec['weight_sha256']
            assert command[command.index('--num_processes') + 1] == '4'


@pytest.mark.parametrize('visible', [None, '4,5,6,7', 'GPU-a,GPU-b,GPU-c,GPU-d'])
def test_stages_explicitly_select_four_container_gpus(monkeypatch, visible):
    runner = load_script('scripts/backbones/run.py')
    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    monkeypatch.setenv('TRAIN_JOB_ID', 'job-test-1234')
    monkeypatch.setenv('PYTHONPATH', '/existing/dependencies')
    env = runner.stage_environment()
    assert env['CUDA_VISIBLE_DEVICES'] == '0,1,2,3'
    assert env['TRAIN_JOB_ID'] == 'job-test-1234'
    assert env['PYTHONPATH'].split(os.pathsep) == [str(ROOT / 'src'), str(ROOT), '/existing/dependencies']
    assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == env['HF_DATASETS_OFFLINE'] == '1'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == visible


@pytest.fixture
def allocated_launcher(tmp_path, monkeypatch):
    """Exercise launcher dispatch with real metadata guards and no GPU work."""
    import torch
    from fastwam.cosmos import dense_encoder
    runner = load_script('scripts/backbones/run.py')
    dataset, cache, run_root = (tmp_path / name for name in ('dataset', 'cache', 'run'))
    (dataset / 'meta').mkdir(parents=True)
    (dataset / 'meta/info.json').write_text(json.dumps(dict(
        total_episodes=78, total_frames=86608, fps=30)))
    cache.mkdir()
    make_manifest(cache)
    spec = VARIANTS['vjepa_vitb']
    monkeypatch.setattr(dense_encoder, '_git_commit',
                        lambda _: dense_encoder.teacher_spec(spec['teacher'])['upstream_commit'])
    monkeypatch.setattr(runner, 'resolve_weights', lambda _, path: path)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *args, **kwargs: 'test-revision\n')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'NVIDIA H200')
    monkeypatch.setenv('TRAIN_JOB_ID', 'job-test-1234')
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.delenv('COSMOS_ALLOCATION_ID', raising=False)
    monkeypatch.setenv('PET_NNODES', '1')
    monkeypatch.setenv('PET_NODE_RANK', '0')
    monkeypatch.setattr(sys, 'argv', ['run.py', '--variant', 'vjepa_vitb',
        '--dataset', str(dataset), '--cache-root', str(cache),
        '--weights', str(tmp_path / 'weights.pt'), '--run-root', str(run_root)])
    context = SimpleNamespace(runner=runner, run_root=run_root, calls=[],
        inventory=[f'GPU-{index}, NVIDIA H200, 143771' for index in range(4)])

    def execute(command, **kwargs):
        if command[0] == 'nvidia-smi':
            assert command == ['nvidia-smi', '--query-gpu=uuid,name,memory.total',
                               '--format=csv,noheader,nounits']
            assert kwargs == dict(check=True, capture_output=True, text=True)
            assert not run_root.exists()
            return SimpleNamespace(stdout='\n'.join(context.inventory) + '\n')
        context.calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, 'run', execute)
    return context


def test_every_stage_receives_the_same_validated_device_environment(allocated_launcher):
    context = allocated_launcher
    context.runner.main()
    assert [Path(command[command.index('--num_processes') + 2]).stem
            for command, _ in context.calls] == ['preflight', 'restore', 'train']
    environments = [kwargs['env'] for _, kwargs in context.calls]
    assert len(environments) == 3 and all(env is environments[0] for env in environments)
    assert environments[0]['CUDA_VISIBLE_DEVICES'] == '0,1,2,3'
    assert json.loads((context.run_root / 'claim.json').read_text())['status'] == 'complete'


@pytest.mark.parametrize('invalid', ['missing', 'extra', 'duplicate', 'wrong_gpu', 'low_memory'])
def test_physical_allocation_guard_rejects_hidden_or_wrong_cards_before_output(allocated_launcher, invalid):
    context = allocated_launcher
    if invalid == 'missing':
        context.inventory.pop()
    elif invalid == 'extra':
        context.inventory += [f'GPU-{index}, NVIDIA H200, 143771' for index in range(4, 8)]
    elif invalid == 'duplicate':
        context.inventory[-1] = context.inventory[0]
    elif invalid == 'wrong_gpu':
        context.inventory[-1] = context.inventory[-1].replace('H200', 'H100')
    else:
        context.inventory[-1] = context.inventory[-1].replace('143771', '80000')
    with pytest.raises(ValueError, match='Exactly four allocated H200'):
        context.runner.main()
    assert not context.run_root.exists() and not context.calls


def test_branch_default_must_match_its_config(tmp_path):
    (tmp_path / 'configs').mkdir()
    path = tmp_path / 'configs/backbone_variant.json'
    path.write_text(json.dumps(dict(variant='vjepa_vitb', config=VARIANTS['dinov3_vithplus']['config'])))
    with pytest.raises(ValueError, match='disagree'):
        default_variant(tmp_path)


def test_gate_contract_ignores_stage_directory_but_binds_optimizer():
    runtime = load_script('scripts/dense/runtime.py')
    cfg = OmegaConf.load(ROOT / VARIANTS['vjepa_vitb']['config'])
    cache = {'identity_sha256': 'same-cache'}
    before = runtime.run_contract(cfg, cache)
    cfg.output_dir = 'different-stage-output'
    assert runtime.run_contract(cfg, cache) == before
    cfg.learning_rate *= 2
    assert runtime.run_contract(cfg, cache) != before


def test_runtime_rejects_wrong_teacher_before_cache_or_model(tmp_path):
    runtime = load_script('scripts/dense/runtime.py')
    with pytest.raises(ValueError, match='Configuration teacher'):
        runtime.load_run(ROOT / VARIANTS['vjepa_vitb']['config'], tmp_path,
            'unused', DATASET_SHA256, tmp_path / 'missing-cache',
            'dinov3_vithplus', 'unused', 'unused', 'unused')


def make_manifest(tmp_path):
    from fastwam.cosmos.jepa_video_cache import cache_teacher_manifest
    spec = VARIANTS['vjepa_vitb']
    source = SimpleNamespace(source_sha=DATASET_SHA256, identity_sha256='numeric-test',
                             lengths=[1000] * 77 + [DATASET_FRAMES - 77000])
    cache_id = identity(source, cache_teacher_manifest(spec['teacher'], spec['weight_sha256']))
    shard = tmp_path / 'shard.safetensors'
    shard.write_bytes(b'fixture: launch validates metadata; dataset validates tensor contents')
    stat = shard.stat()
    manifest = dict(status='complete', identity=cache_id, identity_sha256=cache_identity_sha256(cache_id),
        completed_windows=66919, shards=[dict(path=shard.name, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)])
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize('mutate', ['teacher', 'temporal', 'split', 'shard'])
def test_launcher_rejects_mixed_or_changed_cache(tmp_path, mutate):
    runner = load_script('scripts/backbones/run.py')
    manifest = make_manifest(tmp_path)
    assert runner.verify_cache(tmp_path, 'vjepa_vitb')['completed_windows'] == 66919
    if mutate == 'teacher':
        manifest['identity']['teacher']['name'] = 'dinov3_vithplus'
    elif mutate == 'temporal':
        manifest['identity']['teacher']['temporal_aggregation'] = 'different-encoding'
    elif mutate == 'split':
        manifest['identity']['splits']['train'], manifest['identity']['splits']['test'] = (
            manifest['identity']['splits']['test'], manifest['identity']['splits']['train'])
    else:
        (tmp_path / 'shard.safetensors').write_bytes(b'changed')
    manifest['identity_sha256'] = cache_identity_sha256(manifest['identity'])
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        runner.verify_cache(tmp_path, 'vjepa_vitb')
