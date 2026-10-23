"""Offline replay plumbing checks. Synthetic fixtures do not measure policy quality."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('joint_steps', ROOT / 'scripts/cosmos/evaluate_joint8_steps.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def fixture_inputs():
    source = SimpleNamespace(rows=[(0, 1, 0, 4)], bounds=[(0, 4)], raw={
        'action': torch.arange(4.)[:, None].expand(4, 8),
        'state': torch.zeros(4, 8), 'force': torch.arange(4.)[:, None].expand(4, 6)})
    cached = dict(sample_index=torch.tensor(0), cosmos_text_index=torch.tensor(0),
        action=torch.zeros(32, 8), action_is_pad=torch.arange(32) >= 3,
        force_is_pad=torch.arange(32) >= 2, wrench_valid=torch.ones(32, 6, dtype=torch.bool),
        raw_wrench_future64=torch.cat([torch.tensor([[2.], [3.]]), torch.full((30, 1), 3.)]).expand(32, 6))
    cached['wrench_valid'][0, 2] = False
    normalizer = SimpleNamespace(encode=lambda _, x: x, decode=lambda _, x: x)
    history = ([[0.] * 14] * 32, [False] * 32, [[True] * 14] * 32)
    observation = SimpleNamespace(state=[0.] * 8)
    return source, cached, normalizer, history, observation


def test_current_action_and_next_force_targets_preserve_distinct_offsets():
    source, cached, normalizer, _, _ = fixture_inputs()
    target = evaluation.recorded_targets(source, cached, normalizer)
    assert target['action'][:3, 0].tolist() == [1., 2., 3.]
    assert target['force'][:2, 0].tolist() == [2., 3.]
    assert target['action_valid'].sum() == 3
    assert target['force_valid'].sum() == 11
    assert not target['force_valid'][0, 2]
    assert not target['force_valid'][2:].any()


def test_model_inputs_contain_current_state_and_full_horizon_without_gt():
    _, _, _, history, observation = fixture_inputs()
    sample = evaluation.model_input(torch.zeros(1, 16, 1, 2, 2), history, observation.state, 0, 'cpu')
    assert sample['proprio'].shape == (1, 1, 9)
    assert sample['action_is_pad'].shape == (1, 32)
    assert not sample['action_is_pad'].any()
    assert not {'action', 'wrench_future', 'raw_wrench_future64', 'force_is_pad', 'wrench_valid'} & sample.keys()


def test_replay_pairs_seeds_reports_gt_and_writes_reproducible_artifacts(tmp_path, monkeypatch):
    source, cached, normalizer, history, observation = fixture_inputs()
    (tmp_path / 'prepared').mkdir()
    norm = tmp_path / 'prepared/normalization.json'
    norm.write_text('{}')
    runtime = tmp_path / 'runtime.json'
    runtime.write_text(json.dumps(dict(joint_steps=16, normalization_file=str(norm))))
    output = tmp_path / 'output'; output.mkdir()
    calls = []

    class FakePolicy:
        device = 'cpu'
        text_index = 0
        runner = None
        diagnostics = dict(warmup=[])
        tokenizer = SimpleNamespace(encode=lambda x: x)

        def __init__(self, *args, **kwargs):
            self.model = SimpleNamespace(infer_joint=self.infer_joint)

        def _encode_image(self, observation):
            return torch.zeros(1, 16, 1, 2, 2)

        def infer_joint(self, sample, **kwargs):
            calls.append(kwargs.copy())
            # Different steps produce a deliberately known GT error change.
            return dict(action=torch.full((1, 32, 8), 1. if kwargs['num_inference_steps'] == 16 else 0.),
                        force=torch.zeros(1, 32, 6))

        def close(self):
            pass

    monkeypatch.setattr(evaluation, 'CosmosJointPolicy', FakePolicy)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    for name in ('synchronize', 'reset_peak_memory_stats', 'empty_cache'):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ('max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    args = SimpleNamespace(runtime=runtime, bundle=tmp_path, checkpoint=tmp_path / 'unused.pt',
        backend='joint_static_cache', steps=[16, 8, 4], seed_offsets=[0, 1], source=source, output=output, no_plots=True)
    report = {}
    evaluation.evaluate(args, normalizer, [(cached, observation, history)], report, lambda: None)
    assert len(report['inference']) == 6
    for step in args.steps:
        matching = [c for c in calls if c['num_inference_steps'] == step]
        assert [(c['seed'], c['video_seed'], c['force_seed']) for c in matching] == [(42, 44, 43), (42, 44, 43), (43, 45, 44)]
    assert report['aggregate']['16']['metrics']['action_vs_recorded']['gripper_openness']['pooled_mae'] == 1.
    assert report['aggregate']['4']['metrics']['action_vs_recorded']['gripper_openness']['pooled_mae'] == 2.
    assert report['aggregate']['4']['gt_mae_change_vs_16']['gripper_openness']['absolute'] == 1.
    assert (output / 'summary.csv').is_file()
    with np.load(output / 'sample_000000_seed1_steps4.npz') as saved:
        assert saved['action_time_s'][0] == 0.
        assert saved['force_time_s'][0] == pytest.approx(1 / 30)
        assert not saved['force_valid'][0, 2]
