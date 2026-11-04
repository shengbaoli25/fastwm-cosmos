import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('train_rm71d78_jepa', ROOT / 'scripts/cosmos/train_rm71d78_jepa.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def baseline():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        cfg = compose(config_name='train', overrides=['task=cosmos_noaction_joint_4gpu_30k',
            'batch_size=64', 'max_steps=30000', 'model.expected_text_identity='+'a'*64,
            'model.expected_data_identity='+'b'*64, 'data.train.expected_identity_sha256='+'b'*64])
    for key, value in [('scheduler_total_steps', 50000),
                       ('model.training_recipe.scheduler_total_steps', 50000),
                       ('model.action_representation', 'delta_current_joint7'),
                       ('data.train.expected_action_representation', 'delta_current_joint7')]:
        OmegaConf.update(cfg, key, value, force_add=True)
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def convert(config, variant):
    return launch.add_jepa_config(config, variant=variant, checkpoint=f'/assets/{variant}.pt',
        checkpoint_sha256=('c' if variant == 'vitb' else 'd') * 64,
        observation_cache='/cache/observations', observation_identity='e' * 64, output_dir='/new/run')


def test_b_l_configs_change_only_encoder_identity_and_variant_metadata():
    base = baseline()
    original = copy.deepcopy(base)
    b, l = convert(base, 'vitb'), convert(base, 'vitl')
    assert base == original
    assert b['data']['train']['base_config'] == base['data']['train']
    assert b['data']['train']['_recursive_'] is False
    assert b['batch_size'] == 64 and b['max_steps'] == 30000 and b['scheduler_total_steps'] == 50000
    assert b['model']['latent_warmup_steps'] == 2500
    assert b['model']['wrench_loss_weight'] == .1
    assert b['jepa_experiment']['appearance_consistency_weight'] == 0
    assert not b['jepa_experiment']['external_teacher_required']
    b['model'].pop('observation_encoder')
    l['model'].pop('observation_encoder')
    b['jepa_experiment'].pop('variant')
    l['jepa_experiment'].pop('variant')
    assert b == l


@pytest.mark.parametrize('key,value', [('batch_size', 32), ('num_workers', 4), ('scheduler_total_steps', 30000)])
def test_recipe_rejects_unmatched_training_settings(key, value):
    cfg = baseline()
    cfg[key] = value
    with pytest.raises(ValueError):
        convert(cfg, 'vitb')


def test_plan_requires_explicit_variant_and_pinned_checkpoint_but_creates_nothing(tmp_path):
    output = tmp_path / 'not_created'
    args = launch.arguments(['--variant', 'vitb', '--jepa-checkpoint', '/missing/not-loaded.pt',
                             '--jepa-sha256', 'f'*64, '--output-root', str(output), '--plan'])
    plan = launch.plan_for(args)
    assert plan['mode'] == 'plan_only' and plan['verification_pending']
    assert not output.exists()
    args.jepa_sha256 = 'invented'
    with pytest.raises(ValueError, match='SHA256'):
        launch.plan_for(args)


def test_source_guard_rejects_holdout_or_normalization_changes(tmp_path):
    class Source:
        lengths = [2] * 78
        identity = dict(source_episode_indices=list(range(78)), split_counts=dict(train=78, val=0, test=0),
                        action_representation='delta_current_joint7')

        def __len__(self):
            return 156

    source = Source()
    manifest = dict(status='complete', completed_samples=156, identity=dict(realman_source=copy.deepcopy(source.identity)))
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_text(json.dumps({'statistics': [1, 2]}))
    b.write_text(json.dumps({'statistics': [1, 2]}, indent=2))
    launch.verify_all78_source(source, manifest, a, b)
    b.write_text(json.dumps({'statistics': [1, 3]}))
    with pytest.raises(ValueError, match='normalization'):
        launch.verify_all78_source(source, manifest, a, b)
    source.identity = dict(source.identity, split_counts=dict(train=71, val=7, test=0))
    with pytest.raises(ValueError, match='all78'):
        launch.verify_all78_source(source, manifest, a, a)
