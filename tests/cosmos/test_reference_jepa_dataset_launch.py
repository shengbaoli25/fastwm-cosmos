"""Composition, isolation and real CPU bank binding for the wipe JEPA launcher."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/cosmos'))
from fastwam.cosmos.dataset_spec import load_spec
from fastwam.cosmos.reference_bank import digest, file_digest
from train_jepa_dataset import (arguments, plan_for, dataset_jepa_config,
    reference_jepa_config, prepare_bound_reference, REFERENCE_MODEL,
    REFERENCE_TARGET, REFERENCE_DATASET, VITL_SHA)
from test_jepa_training_recipe import baseline


def test_combined_configuration_preserves_every_original_jepa_setting():
    spec = load_spec(ROOT / 'configs/experiments/jepa_clean30_dataset.json')
    base = baseline()
    base['model']['expected_normalization_sha256'] = 'f' * 64
    original = copy.deepcopy(base)
    kwargs = dict(checkpoint='/assets/vitl.pt', checkpoint_sha256=VITL_SHA,
        observation_cache='/new/observations', observation_identity='e' * 64,
        output_dir='/new/config-placeholder')
    jepa = dataset_jepa_config(base, spec, **kwargs)
    combined = reference_jepa_config(base, spec, reference_bank='/new/reference-bank',
        reference_identity='d' * 64, **kwargs)
    assert base == original
    assert combined['model']['_target_'] == REFERENCE_TARGET
    assert combined['data']['train']['_target_'] == jepa['data']['train']['_target_']
    ref = combined['data']['train']['base_config']
    assert ref['_target_'] == REFERENCE_DATASET
    assert ref['expected_reference_identity'] == combined['model']['expected_reference_identity'] == 'd' * 64
    assert ref['reference_bank'] == '/new/reference-bank'
    assert ref['reference_seed'] == 42
    for key in ('reference_bank', 'expected_reference_identity', 'reference_seed', 'reference_augmentation'):
        ref.pop(key)
    ref['_target_'] = jepa['data']['train']['base_config']['_target_']
    for key in (*REFERENCE_MODEL, 'expected_reference_identity'):
        combined['model'].pop(key)
    combined['model']['_target_'] = jepa['model']['_target_']
    combined.pop('reference_experiment')
    assert combined == jepa


def test_wipe_plan_uses_independent_source_and_output_without_loading_assets(tmp_path):
    output = tmp_path / 'new-run'
    args = arguments(['--reference-audit', '--dataset', str(tmp_path/'missing-raw'),
                      '--output-root', str(output), '--plan'])
    plan = plan_for(args)
    assert not output.exists()
    assert args.spec.name == 'jepa_wipe30_dataset.json'
    assert plan['dataset_spec']['source_frames'] == 22843
    assert plan['dataset_spec']['source_episodes'] == 30
    assert plan['reference_audit_enabled'] and plan['model_target'] == REFERENCE_TARGET
    assert plan['reference_bank'] == (output/'reference-bank').as_posix()
    assert plan['training_recipe']['global_batch'] == 256
    assert plan['training_recipe']['max_steps'] == 30000
    assert plan['training_recipe']['scheduler_total_steps'] == 50000
    assert not plan['model_warm_start'] and not plan['optimizer_resume']
    assert not plan['external_teacher_required']
    default = plan_for(arguments(['--reference-audit', '--plan']))
    assert 'rm71d-reference-jepa-wipe-20261111' in default['output_root']
    assert default['dataset'].endswith('/data/wipe30')
    clean = plan_for(arguments(['--plan']))
    assert not clean['reference_audit_enabled'] and clean['reference_bank'] is None
    assert clean['dataset_spec']['source_frames'] == 46870
    assert 'rm71d-jepa-clean30-20261107' in clean['output_root']


class Source(SimpleNamespace):
    def __len__(self):
        return self.anchors


def test_real_reference_bank_binds_exact_source_and_rejects_other_file_drift():
    # Reuse real tiny Parquets and the production contract, without any VAE/GPU.
    from test_reference_cache_binding_cpu import CacheBindingTests
    fixture = CacheBindingTests()
    fixture.setUp()
    try:
        raw = fixture.root / 'raw'
        files = {p.relative_to(raw).as_posix(): file_digest(p)
                 for p in sorted(raw.rglob('*')) if p.is_file()}
        source_identity = fixture.cache['identity']['realman_source']
        source_identity['files'] = files
        fixture.write_cache()
        manifest = fixture.bank_manifest
        source = Source(root=raw, source_sha=digest(files),
            lengths=source_identity['episode_lengths'], identity=source_identity,
            segments=source_identity['segments'], prompts=['insert plug'] * 2,
            anchors=manifest['total_retained_anchors'])
        spec = dict(schema='cosmos-jepa-dataset-spec-v1', dataset_id='test-reference-jepa',
            source_inventory_sha256=source.source_sha, source_episodes=2,
            source_frames=manifest['total_source_rows'], training_anchors=len(source),
            continuous_segments=len(source.segments), prompts=['insert plug'],
            action_representation='delta_current_joint7', split_counts=dict(train=2,val=0,test=0),
            initialization='official_cosmos_and_jepa_fresh_action_force_optimizer', variant='vitl')
        bank, receipt = prepare_bound_reference(source, spec, fixture.path, fixture.root/'new-bank')
        assert bank['status'] == 'complete' and receipt['status'] == 'bound'
        assert receipt['full_raw_inventory_equal'] is True
        assert receipt['reference_identity_sha256'] == bank['identity_sha256']
        assert not receipt['cache_tensor_payloads_verified']
        # Numeric metadata remains identical; the stricter combined launcher also
        # rejects a cache whose claimed raw-video inventory contains another file.
        source_identity['files'] = dict(files, **{'videos/other.mp4': 'a' * 64})
        fixture.write_cache()
        with pytest.raises(ValueError, match='exact dataset/cache'):
            prepare_bound_reference(source, spec, fixture.path, fixture.root/'rejected-bank')
    finally:
        fixture.doCleanups()
