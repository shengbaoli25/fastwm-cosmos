import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/cosmos'))
from fastwam.cosmos.dataset_spec import load_spec, verify_source, verify_prepared, verify_prompt_manifest
from train_jepa_dataset import arguments, plan_for, dataset_jepa_config, VITL_SHA
from train_rm71d78_jepa import add_jepa_config


def spec():
    return load_spec(ROOT / 'configs/experiments/jepa_clean30_dataset.json')


class Source(SimpleNamespace):
    def __len__(self):
        return self.anchors


def source():
    s = spec()
    return Source(source_sha=s['source_inventory_sha256'], lengths=[1000] * 29 + [17870],
        anchors=s['training_anchors'], segments=[None] * 285, prompts=s['prompts'] * 30,
        identity=dict(action_representation='delta_current_joint7', source_episode_indices=list(range(30)),
            split_counts=dict(train=30, val=0, test=0), state_columns=[0,1,2,3,4,5,6,19],
            action_units=['rad'] * 7 + ['openness_0_1']),
        identity_sha256='a' * 64, stats={'force': {'mean': [1.0], 'std': [2.0]}},
        force_contract={'schema': 'test'})


def test_exact_source_rejects_old_data_relabeling_or_changed_retained_windows():
    original = source()
    verify_source(spec(), original)
    for key, value in [('source_sha', '0'*64), ('prompts', ['new task']), ('anchors', 45328),
                       ('lengths', [1000]*78), ('segments', [None]*284)]:
        changed = copy.deepcopy(original)
        setattr(changed, key, value)
        with pytest.raises(ValueError, match='Dataset bytes'):
            verify_source(spec(), changed)
    changed = copy.deepcopy(original)
    changed.identity['split_counts'] = dict(train=29, val=1, test=0)
    with pytest.raises(ValueError, match='Dataset bytes'):
        verify_source(spec(), changed)


def test_prepared_cache_binds_new_statistics_and_source_identity():
    src = source()
    manifest = dict(status='complete', completed_samples=len(src), identity=dict(realman_source=src.identity))
    norm = dict(source_identity_sha256=src.identity_sha256, statistics=src.stats,
        action_representation='delta_current_joint7', force_preprocessing=src.force_contract,
        state_columns=src.identity['state_columns'], action_units=src.identity['action_units'])
    verify_prepared(spec(), src, manifest, norm)
    stale = copy.deepcopy(norm)
    stale['statistics']['force']['mean'] = [99.0]
    with pytest.raises(ValueError, match='normalization'):
        verify_prepared(spec(), src, manifest, stale)
    stale = copy.deepcopy(manifest)
    stale['identity']['realman_source']['source_episode_indices'] = list(range(78))
    with pytest.raises(ValueError, match='Prepared features'):
        verify_prepared(spec(), src, stale, norm)


def test_text_bank_cannot_relabel_clean_task():
    verify_prompt_manifest(spec(), dict(texts=[dict(prompt='七轴相对遥操作演示')]))
    with pytest.raises(ValueError, match='exact prompt'):
        verify_prompt_manifest(spec(), dict(texts=[dict(prompt='insert the plug')]))


def test_clean30_recipe_matches_actual_existing_vitl_transform_exactly():
    fixture_spec = importlib.util.spec_from_file_location('old_recipe_fixture', ROOT / 'tests/cosmos/test_jepa_training_recipe.py')
    fixture = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture)
    baseline = fixture.baseline()
    original = copy.deepcopy(baseline)
    kwargs = dict(checkpoint='/assets/vitl.pt', checkpoint_sha256=VITL_SHA,
        observation_cache='/new/observations', observation_identity='e'*64, output_dir='/new/placeholder')
    expected = add_jepa_config(baseline, variant='vitl', **kwargs)
    actual = dataset_jepa_config(baseline, spec(), **kwargs)
    metadata = actual.pop('jepa_experiment')
    expected.pop('jepa_experiment')
    assert actual == expected
    assert baseline == original
    assert metadata['raw_source_sha256'] == spec()['source_inventory_sha256']
    assert metadata['split_counts'] == dict(train=30, val=0, test=0)
    assert metadata['external_teacher_required'] is False
    assert metadata['baseline_initialization'] == 'official_cosmos_fresh_action_force'


def test_plan_is_inert_and_does_not_inherit_all78_provenance(tmp_path):
    output = tmp_path / 'new-output'
    args = arguments(['--dataset', str(tmp_path/'missing-dataset'), '--output-root', str(output), '--plan'])
    plan = plan_for(args)
    assert not output.exists()
    assert plan['mode'] == 'plan_only' and plan['verification_pending']
    assert plan['dataset_spec']['source_episodes'] == 30
    assert plan['training_recipe']['split_counts'] == dict(train=30, val=0, test=0)
    assert plan['training_recipe']['global_batch'] == 256
    assert not plan['model_warm_start'] and not plan['optimizer_resume']
    assert not plan['external_teacher_required']


def test_new_data_still_rejects_recipe_changes():
    fixture_spec = importlib.util.spec_from_file_location('old_recipe_fixture', ROOT / 'tests/cosmos/test_jepa_training_recipe.py')
    fixture = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture)
    baseline = fixture.baseline()
    baseline['num_workers'] = 4
    with pytest.raises(ValueError, match='recipe'):
        dataset_jepa_config(baseline, spec(), checkpoint='/assets/vitl.pt', checkpoint_sha256=VITL_SHA,
            observation_cache='/new/observations', observation_identity='e'*64, output_dir='/new/placeholder')
