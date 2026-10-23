"""Offline benchmark separates warmup and uses fixed inputs without a robot."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('deployment_benchmark', ROOT / 'deploy/benchmark.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_benchmark_uses_fresh_policy_fixed_seed_and_does_not_mutate_config():
    from realman_deploy.normalization import DatasetNormalizer
    normalizer = DatasetNormalizer.from_json(ROOT / 'deploy/normalization.example.json')
    records = []
    class FakePolicy:
        def __init__(self, checkpoint, config, config_path):
            self.normalizer = normalizer
            self.device = 'cuda:1'
            self.diagnostics = {'warmup': [{'total_ms': 1000}, {'total_ms': 900}]}
            self.inference_index = 99
            self.close = Mock()
            records.append((self, config))
        def infer(self, observation, reference, history):
            assert self.inference_index == 0
            assert reference is None
            assert len(observation.frames['global']) == 2 * 2 * 3
            self.inference_index += 1
            self.diagnostics['last_inference'] = {'total_ms': 10}
            return [[0.] * 7 for _ in range(32)]
    config = {'device': 'cuda:1', 'camera_width': 2, 'camera_height': 2,
              'inference_optimization': {'mode': 'auto'}}
    synchronize = Mock()
    report = benchmark.benchmark_profile(config, Path('/tmp/config.json'), Path('/tmp/checkpoint.pt'),
        '4416', 3, optimization='eager', policy_factory=FakePolicy, synchronize=synchronize)
    assert report['wall_ms']['count'] == 3
    assert len(report['warmup']) == 2
    assert records[0][1]['sampling_profile'] == '4416'
    assert records[0][1]['inference_optimization']['mode'] == 'eager'
    assert config['inference_optimization']['mode'] == 'auto'
    assert report['runs'][0]['action_shape'] == [32, 7]
    records[0][0].close.assert_called_once()
    assert all(call.args == ('cuda:1',) for call in synchronize.call_args_list)


def test_policy_is_closed_if_inference_fails():
    from realman_deploy.normalization import DatasetNormalizer
    policy = SimpleNamespace(device='cuda:0', diagnostics={},
        normalizer=DatasetNormalizer.from_json(ROOT / 'deploy/normalization.example.json'),
        infer=Mock(side_effect=RuntimeError('failed')), close=Mock())
    with pytest.raises(RuntimeError, match='failed'):
        benchmark.benchmark_profile({}, '/tmp/config.json', '/tmp/model.pt', '4416', 1,
                                    policy_factory=lambda *args, **kwargs: policy,
                                    synchronize=lambda device: None)
    policy.close.assert_called_once()


def test_percentiles_are_computed_from_samples_only():
    assert benchmark.summary([10, 20, 30]) == {'count': 3, 'p50': 20., 'p95': 29., 'mean': 20, 'min': 10, 'max': 30}
