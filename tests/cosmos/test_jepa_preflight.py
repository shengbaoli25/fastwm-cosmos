from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from fastwam.cosmos import jepa_observation as jo
from fastwam.cosmos.preflight import JEPA_DATASET, JEPA_TARGET, JepaGradientProbe, validate_config

ROOT = Path(__file__).resolve().parents[2]


def config(variant='vitb'):
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        raw = compose(config_name='train', overrides=['task=cosmos_noaction_joint_4gpu_30k',
            'batch_size=64', 'model.expected_text_identity=' + 'a' * 64,
            'model.expected_data_identity=' + 'b' * 64,
            'data.train.expected_identity_sha256=' + 'b' * 64])
    cfg = OmegaConf.to_container(raw, resolve=True, throw_on_missing=True)
    cfg['scheduler_total_steps'] = 50000
    m = cfg['model']
    m['training_recipe']['scheduler_total_steps'] = 50000
    m.update(_target_=JEPA_TARGET, reader_width=256, query_count=8, reader_depth=2,
             reader_heads=4, injection_layers=[7, 14, 21, 28], latent_loss_weight=1.,
             wrench_loss_weight=.1, latent_warmup_steps=2500, jepa_seed=47,
             allow_preencoded_patches=False,
             observation_encoder=dict(variant=variant, checkpoint='assets/official.pt',
                                      expected_sha256='c' * 64, checkpoint_suffix=True))
    cfg['data']['train'] = dict(_target_=JEPA_DATASET, _recursive_=False,
        base_config=cfg['data']['train'], observation_cache_root='data/observed',
        expected_observation_identity='d' * 64)
    return cfg


@pytest.mark.parametrize('variant', ['vitb', 'vitl'])
def test_from_scratch_jepa_recipe_preserves_baseline_cache_and_schedule(variant):
    cfg = config(variant)
    validate_config(cfg)
    assert cfg['batch_size'] * 4 == 256
    assert cfg['resume'] is None
    assert cfg['model']['training_recipe'] == dict(batch_per_rank=64, max_steps=30000,
                                                  scheduler_total_steps=50000)


@pytest.mark.parametrize('field,value', [('reader_width', 512), ('query_count', 16),
    ('reader_depth', 4), ('latent_loss_weight', .5), ('wrench_loss_weight', 1.),
    ('latent_warmup_steps', 0), ('injection_layers', [7, 14]), ('allow_preencoded_patches', True),
    ('teacher_checkpoint', 'old-force.pt'), ('initial_checkpoint', 'old-joint.pt')])
def test_jepa_preflight_rejects_altered_architecture_or_prior_policy_dependency(field, value):
    cfg = config()
    cfg['model'][field] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize('field,value', [('batch_size', 16), ('max_steps', 20000),
    ('scheduler_total_steps', 30000), ('resume', 'previous-state')])
def test_jepa_preflight_rejects_different_training_recipe(field, value):
    cfg = config()
    cfg[field] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize('field,value', [('variant', 'vitg'), ('expected_sha256', 'unknown'),
    ('checkpoint_suffix', False), ('rank', 16)])
def test_jepa_preflight_requires_exact_official_lora_contract(field, value):
    cfg = config()
    cfg['model']['observation_encoder'][field] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_jepa_preflight_checks_both_baseline_and_observed_cache_identity():
    good = config()
    bad = deepcopy(good)
    bad['data']['train']['base_config']['expected_identity_sha256'] = 'f' * 64
    with pytest.raises(ValueError):
        validate_config(bad)
    bad = deepcopy(good)
    bad['data']['train']['base_config']['expected_text_identity'] = 'f' * 64
    with pytest.raises(ValueError, match='text identities'):
        validate_config(bad)
    bad = deepcopy(good)
    bad['data']['train']['expected_observation_identity'] = 'unknown'
    with pytest.raises(ValueError, match='observed-only'):
        validate_config(bad)


def probe_model(monkeypatch):
    monkeypatch.setitem(jo.VARIANT_SPECS, 'vitb', dict(width=24, depth=12,
        heads=3, factory='vit_base', filename='tiny-test.pt'))
    base = nn.Module()
    base.embed_dim, base.use_rope, base.tubelet_size, base.patch_size = 24, True, 2, 16
    base.blocks = nn.ModuleList()
    for _ in range(12):
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.qkv = nn.Linear(24, 72)
        base.blocks.append(block)
    encoder = jo.JEPAObservationEncoder(base, variant='vitb', weight_sha256='a' * 64)
    return SimpleNamespace(dit=nn.ModuleDict({'jepa_observation': encoder}))


def update_twice(model, zero=False):
    e = model.dit['jepa_observation']
    optimizer = torch.optim.AdamW(e.trainable_parameters(), lr=.01)
    x = torch.randn(2, 5, 24)
    for _ in range(2):
        optimizer.zero_grad()
        loss = sum(block.attn.qkv(x).square().mean() for block in e.encoder.blocks[-4:])
        (loss * (0 if zero else 1)).backward()
        optimizer.step()


def test_actual_lora_probe_allows_first_zero_a_gradient_then_requires_learning(monkeypatch):
    model = probe_model(monkeypatch)
    probe = JepaGradientProbe(model)
    update_twice(model)
    receipt = probe.finish()
    assert receipt['accepted']
    assert receipt['base_parameters_frozen'] and receipt['frozen_base_gradients_absent']
    assert receipt['lora_trainable_count'] == 4 * 4 * 8 * 24
    for name, records in receipt['lora_gradient_records'].items():
        if name.endswith('_a'):
            assert not records[0]['nonzero'] and records[1]['nonzero']
        else:
            assert all(record['nonzero'] for record in records)


def test_actual_lora_probe_rejects_silently_dead_side_path(monkeypatch):
    model = probe_model(monkeypatch)
    probe = JepaGradientProbe(model)
    update_twice(model, zero=True)
    receipt = probe.finish()
    assert not receipt['accepted']
    assert receipt['all_lora_seen_across_updates'] and receipt['all_lora_gradients_finite']
    assert not receipt['all_lora_received_nonzero_gradient']


def test_actual_lora_probe_rejects_accidental_base_unfreeze(monkeypatch):
    model = probe_model(monkeypatch)
    model.dit['jepa_observation'].requires_grad_(True)
    with pytest.raises(RuntimeError, match='frozen/eval'):
        JepaGradientProbe(model)


@pytest.mark.parametrize('field,value', [('query_redundancy_weight', float('inf')),
    ('query_redundancy_weight', -1.), ('query_redundancy_margin', 1.),
    ('query_redundancy_margin', float('nan')), ('query_redundancy_warmup_steps', -1),
    ('query_redundancy_warmup_steps', True)])
def test_preflight_rejects_invalid_query_regularizer(field, value):
    cfg = config()
    cfg['model'][field] = value
    with pytest.raises(ValueError, match='query redundancy'):
        validate_config(cfg)
