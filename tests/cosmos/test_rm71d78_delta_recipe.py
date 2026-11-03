"""CPU orchestration contracts; no platform launch, GPU or production asset load."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fastwam.cache.latent import cache_identity_sha256, file_sha256
from fastwam.cosmos.joint_action import joint_action_contract

ROOT = Path(__file__).resolve().parents[2]


def script(name):
    spec = importlib.util.spec_from_file_location('recipe_test_' + name,
                                                ROOT / 'scripts/cosmos' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_config():
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        cfg = compose(config_name='train', overrides=[
            'task=cosmos_noaction_joint_4gpu_30k', 'batch_size=64', 'max_steps=50000',
            'model.expected_text_identity=' + 'a' * 64,
            'model.expected_data_identity=' + 'b' * 64,
            'data.train.expected_identity_sha256=' + 'b' * 64])
    return OmegaConf.to_container(cfg, resolve=True)


def test_plan_requires_no_assets_or_job_and_does_not_create_output(tmp_path, monkeypatch, capsys):
    entry = script('train_rm71d78_delta')
    output = tmp_path / 'new-run'
    monkeypatch.setattr(entry.sys, 'argv', ['train_rm71d78_delta.py', '--output-root', str(output)])
    monkeypatch.delenv('TRAIN_JOB_ID', raising=False)
    entry.main()
    plan = json.loads(capsys.readouterr().out)
    assert not output.exists()
    assert plan['source_episodes'] == 78
    assert plan['split_counts'] == dict(train=78, val=0, test=0)
    assert plan['action_representation'] == 'delta_current_joint7'
    assert plan['branch_reads'] == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
    assert (plan['batch_per_rank'], plan['global_batch']) == (64, 256)
    assert (plan['max_steps'], plan['scheduler_total_steps']) == (30000, 50000)
    assert (plan['warmup_steps'], plan['cosine_steps']) == (2500, 47500)
    assert plan['loss_weights'] == dict(video=1., force=1., action=1.)
    assert plan['official_initialization'] and not plan['optimizer_resume']


@pytest.mark.parametrize('field,value', [('batch_size', 16), ('max_steps', 30000),
                                         ('model.force_ffn', 3072)])
def test_original_recipe_rejects_changed_reference_hyperparameters(field, value):
    entry = script('train_rm71d78_delta')
    cfg = original_config()
    entry.verify_original_recipe(cfg)
    if field.startswith('model.'):
        cfg['model'][field.split('.')[1]] = value
    else:
        cfg[field] = value
    with pytest.raises(ValueError, match='changed'):
        entry.verify_original_recipe(cfg)


def test_resolver_binds_real_task_hyperparameters_delta_stats_and_scheduler(tmp_path, monkeypatch):
    resolver = script('resolve_config')
    contract = joint_action_contract('delta_current_joint7')
    source = dict(contract, action_space='joint8', action_dim=8, statistics={'test': [1., 2.]},
                  force_preprocessing={'scope': 'test'}, state_columns=list(range(8)),
                  source_episode_indices=list(range(78)), split_counts=dict(train=78, val=0, test=0))
    norm = dict(source, source_identity_sha256=cache_identity_sha256(source))
    norm_path = tmp_path / 'normalization.json'
    norm_path.write_text(json.dumps(norm))
    digest, text_digest = 'b' * 64, 'a' * 64
    validator = Mock(return_value={'identity_sha256': digest,
                                 'identity': dict(text_identity=text_digest, realman_source=source)})
    monkeypatch.setattr(resolver, 'validate', validator)
    monkeypatch.setattr(resolver, 'read_text_bank', lambda _: (None, {}, text_digest))
    monkeypatch.setattr(resolver, 'checked', lambda *_: None)
    monkeypatch.setattr(resolver, 'verify_upstream', lambda _: {})
    output = tmp_path / 'resolved.yaml'
    import sys
    monkeypatch.setattr(sys, 'argv', ['resolve_config.py', '--task', 'cosmos_noaction_joint_4gpu_30k',
        '--text-cache', str(tmp_path / 'text'), '--feature-cache', str(tmp_path / 'features'),
        '--checkpoint', str(tmp_path / 'official.pt'), '--upstream', str(tmp_path / 'upstream'),
        '--output-dir', str(tmp_path / 'training'), '--config-output', str(output),
        '--normalization', str(norm_path), '--batch-size', '64', '--max-steps', '30000',
        '--scheduler-total-steps', '50000', '--action-representation', 'delta_current_joint7'])
    resolver.main()
    cfg = OmegaConf.to_container(OmegaConf.load(output), resolve=True)
    validator.assert_called_once_with(str(tmp_path / 'features'),
                                     expected_action_representation='delta_current_joint7')
    assert cfg['model']['training_recipe'] == dict(batch_per_rank=64, max_steps=30000,
                                                 scheduler_total_steps=50000)
    assert cfg['scheduler_total_steps'] == 50000
    assert cfg['model']['expected_normalization_sha256'] == file_sha256(norm_path)
    assert cfg['model']['expected_data_identity'] == cfg['data']['train']['expected_identity_sha256'] == digest
    assert cfg['model']['action_representation'] == cfg['data']['train']['expected_action_representation'] == 'delta_current_joint7'
    assert cfg['data'].get('val') is None
    assert cfg['model'].get('initial_checkpoint') is None
    # Apart from the requested stop, the real resolved training recipe must
    # retain all old optimizer, batch, worker, save and model hyperparameters.
    cfg['max_steps'] = 50000
    script('train_rm71d78_delta').verify_original_recipe(cfg)


def test_model_factory_keeps_stop_and_scheduler_horizon_in_checkpoint_recipe(monkeypatch):
    from fastwam.cosmos import assets, noaction
    model = Mock()
    model.to.return_value = model
    factory = Mock(return_value=model)
    monkeypatch.setattr(noaction, 'NoActionJointModel', factory)
    monkeypatch.setattr(assets, 'read_text_bank', lambda _: ('text-bank', {}, 'a' * 64))
    monkeypatch.setattr(assets, 'load_nonaction_video', lambda *_, **__: 'official-net')
    recipe = dict(batch_per_rank=64, max_steps=30000, scheduler_total_steps=50000)
    assert assets.create_noaction_joint_cosmos(upstream='upstream', checkpoint='official',
        text_cache='text', expected_text_identity='a' * 64, expected_data_identity='b' * 64,
        expected_normalization_sha256='c' * 64, training_recipe=recipe,
        action_representation='delta_current_joint7', action_space='joint8', device='cpu') is model
    kwargs = factory.call_args.kwargs
    assert kwargs['manifest']['recipe'] == dict(recipe, ranks=4, accum=1, global_batch=256)
    assert 'scheduler_total_steps' not in kwargs
    assert kwargs['action_representation'] == 'delta_current_joint7'


def test_nonrank_parent_resource_identity_uses_real_four_gpu_inventory(monkeypatch):
    from fastwam.cosmos import persistence
    env = script('distributed_fresh').clean_environment(dict(
        TRAIN_JOB_ID='job-regression-1234', PET_NNODES='1', PET_NODE_RANK='0',
        RANK='3', LOCAL_RANK='3', WORLD_SIZE='4', CUDA_VISIBLE_DEVICES='3'), ROOT)
    assert not {'RANK', 'LOCAL_RANK', 'WORLD_SIZE'} & env.keys()
    monkeypatch.setattr(os, 'environ', env)
    rows = [f'GPU-test-{i}, NVIDIA H200, 143771' for i in range(4)]
    query = Mock(return_value=SimpleNamespace(stdout='\n'.join(rows)))
    monkeypatch.setattr(persistence.subprocess, 'run', query)
    identity = persistence.resource_identity()
    assert identity['job'] == env['TRAIN_JOB_ID']
    assert identity['cuda_visible_devices'] == '0,1,2,3'
    assert identity['gpu_inventory'] == rows
    assert query.call_args.args[0][0] == 'nvidia-smi'
    query.return_value.stdout = '\n'.join(rows[:3])
    with pytest.raises(ValueError, match='four allocated H200'):
        persistence.resource_identity()
