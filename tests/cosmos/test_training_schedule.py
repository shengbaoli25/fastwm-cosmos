"""CPU scheduler regressions; no model construction, GPU allocation or training."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fastwam.cosmos.preflight import validate_config, verify_trainer_schedule
from fastwam.trainer import Wan22Trainer
from fastwam.utils.training_schedule import resolve_scheduler_total_steps

ROOT = Path(__file__).resolve().parents[2]


def trainer(stop, horizon=None):
    instance = object.__new__(Wan22Trainer)
    instance.cfg = OmegaConf.create(dict(max_steps=stop, scheduler_total_steps=horizon,
                                        lr_scheduler_type='cosine'))
    instance.learning_rate = 1e-4
    instance.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=1e-4)
    instance._initialize_schedule(stop)
    return instance


def advance(instance):
    instance.optimizer.step()
    instance.scheduler.step()


def config():
    with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base=None):
        value = compose(config_name='train', overrides=[
            'task=cosmos_noaction_joint_4gpu_30k', 'batch_size=64',
            'model.expected_text_identity='+'a'*64,
            'model.expected_data_identity='+'b'*64,
            'data.train.expected_identity_sha256='+'b'*64])
    value = OmegaConf.to_container(value, resolve=True, throw_on_missing=True)
    value['scheduler_total_steps'] = 50000
    value['model']['training_recipe']['scheduler_total_steps'] = 50000
    return value


def test_30k_stop_preserves_every_learning_rate_of_the_original_50k_schedule():
    short = trainer(30000, 50000)
    original = trainer(50000)
    assert short.max_steps == 30000 and original.max_steps == 50000
    assert verify_trainer_schedule(short, short.cfg)['scheduler'] == dict(
        last_epoch=0, warmup_milestones=[2500], cosine_steps=47500)
    for step in range(1, 30001):
        advance(short)
        advance(original)
        assert short.scheduler.get_last_lr() == original.scheduler.get_last_lr()
        if step == 25000:
            saved = deepcopy((short.optimizer.state_dict(), short.scheduler.state_dict()))
    assert short.max_steps == 30000
    assert short.scheduler.get_last_lr()[0] > 1e-6
    verify_trainer_schedule(short, short.cfg, step=30000)

    restored = trainer(30000, 50000)
    restored.optimizer.load_state_dict(saved[0])
    restored.scheduler.load_state_dict(saved[1])
    for _ in range(5000):
        advance(restored)
    assert restored.scheduler.state_dict() == short.scheduler.state_dict()
    assert restored.scheduler.get_last_lr() == short.scheduler.get_last_lr()
    assert restored.max_steps == 30000


def test_missing_null_and_explicit_original_horizon_keep_legacy_geometry():
    for horizon in (None, 30000):
        instance = trainer(30000, horizon)
        assert instance.scheduler._milestones == [1500]
        assert instance.scheduler._schedulers[-1].T_max == 28500
    instance = trainer(30000)
    del instance.cfg['scheduler_total_steps']
    instance._initialize_schedule(30000)
    assert instance.scheduler_total_steps == 30000


@pytest.mark.parametrize('horizon', [0, -1, 29999, True, 50000.0, '50000'])
def test_invalid_or_shorter_horizon_is_rejected(horizon):
    with pytest.raises(ValueError, match='scheduler_total_steps'):
        resolve_scheduler_total_steps(30000, horizon)


def test_preflight_requires_checkpoint_recipe_to_bind_both_horizons():
    value = config()
    validate_config(value)
    assert value['max_steps'] == value['model']['training_recipe']['max_steps'] == 30000
    for key, change in [('max_steps', 50000), ('scheduler_total_steps', 30000), ('batch_per_rank', 16)]:
        wrong = deepcopy(value)
        wrong['model']['training_recipe'][key] = change
        with pytest.raises(ValueError, match='training_recipe'):
            validate_config(wrong)
    wrong = deepcopy(value)
    del wrong['model']['training_recipe']['scheduler_total_steps']
    with pytest.raises(ValueError, match='training_recipe'):
        validate_config(wrong)


def test_null_horizon_in_both_config_and_recipe_keeps_legacy_config_valid():
    value = config()
    value['scheduler_total_steps'] = None
    value['model']['training_recipe']['scheduler_total_steps'] = None
    validate_config(value)


def test_runtime_gate_rejects_wrong_stop_or_shortened_scheduler_and_handles_wrapper():
    instance = trainer(30000, 50000)
    cfg = instance.cfg
    instance.scheduler = SimpleNamespace(scheduler=instance.scheduler)
    verify_trainer_schedule(instance, cfg)
    instance.max_steps = 50000
    with pytest.raises(RuntimeError, match='stop limit'):
        verify_trainer_schedule(instance, cfg)
    instance = trainer(30000)
    with pytest.raises(RuntimeError, match='stop limit'):
        verify_trainer_schedule(instance, cfg)


def test_unapproved_extended_horizon_is_rejected():
    value = config()
    value['scheduler_total_steps'] = 60000
    value['model']['training_recipe']['scheduler_total_steps'] = 60000
    with pytest.raises(ValueError, match='50000'):
        validate_config(value)
