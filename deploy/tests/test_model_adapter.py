"""Deployment backend selection and sampling contracts without a GPU or robot."""
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realman_deploy.contracts import Observation
from realman_deploy.model_adapter import CosmosJointPolicy
from realman_deploy.normalization import DatasetNormalizer, OnlineHistory


def policy_fixture(optimized):
    policy = CosmosJointPolicy.__new__(CosmosJointPolicy)
    policy.normalizer = DatasetNormalizer({
        key: {"mean": [0.] * width, "std": [1.] * width}
        for key, width in (("state", 7), ("action", 7), ("force", 6))
    })
    policy.device = "cpu"
    policy.config = {"action_steps": 16, "action_seed": 42, "video_seed": 44,
                     "camera_width": 1, "camera_height": 1}
    policy.inference_backend = "video_isolated_reuse" if optimized else "reference"
    policy.inference_index = 0
    policy.text_index = 0
    policy.diagnostics = {}
    policy.model = Mock()
    policy.runner = Mock() if optimized else None
    target = policy.runner if optimized else policy.model
    target.infer_forcewm.return_value = {"action": torch.zeros(1, 32, 7)}
    policy.tokenizer = Mock()
    policy.tokenizer.encode.return_value = torch.zeros(1, 16, 1, 2, 2)
    policy._encode_image = Mock(return_value=torch.zeros(1, 3, 1, 16, 32))
    return policy, target


@pytest.mark.parametrize("optimized", [False, True])
def test_selected_backend_preserves_steps_reference_and_seed_sequence(optimized):
    policy, target = policy_fixture(optimized)
    state = [0.1] * 7
    obs = Observation.now(1, tuple(state), (0.,) * 6,
                          {"global": bytes(3), "wrist": bytes(3)})
    history = OnlineHistory(policy.normalizer)
    history.append(state, (0.,) * 6)
    reference = [state] * 32
    for _ in range(2):
        result = policy.infer(obs, reference, history.tensors())
        assert len(result) == 32 and len(result[0]) == 7
    calls = target.infer_forcewm.call_args_list
    assert [call.kwargs["seed"] for call in calls] == [42, 43]
    assert all(call.kwargs["num_inference_steps"] == 16 for call in calls)
    assert all(call.kwargs["video_seed"] == 44 for call in calls)
    torch.testing.assert_close(calls[0].kwargs["action_condition"],
                               torch.tensor([reference], dtype=torch.bfloat16))
    if optimized:
        policy.model.infer_forcewm.assert_not_called()
    assert policy.diagnostics["last_inference"]["backend"] == policy.inference_backend


def test_warmup_does_not_consume_first_live_seed_and_close_releases_runner():
    policy, runner = policy_fixture(True)
    policy._warmup()
    assert policy.inference_index == 0
    assert [call.kwargs["seed"] for call in runner.infer_forcewm.call_args_list] == [42, 42]
    assert len(policy.diagnostics["warmup"]) == 2
    policy.close()
    runner.close.assert_called_once()
    assert policy.model is policy.tokenizer is policy.runner is None


@pytest.mark.parametrize('backend', ['joint_static_cache', 'joint_optimized'])
def test_no_reference_deployment_uses_same_joint_sampler_without_rolling_actions(backend):
    policy, _ = policy_fixture(False)
    policy.reference_mode = 'none'
    policy.inference_backend = backend
    if backend == 'joint_optimized':
        policy.runner = Mock()
    target = policy.runner if policy.runner is not None else policy.model
    policy.config['joint_steps'] = 16
    target.infer_joint.return_value = {'action': torch.zeros(1,32,7)}
    obs = Observation.now(1, (0.,)*7, (0.,)*6, {'global':bytes(3), 'wrist':bytes(3)})
    history = OnlineHistory(policy.normalizer);history.append(obs.state,obs.force)
    policy.infer(obs, None, history.tensors())
    call = target.infer_joint.call_args
    assert call.kwargs == dict(num_inference_steps=16, seed=42, video_seed=44, force_seed=43, use_cache=True)
    assert 'action_condition' not in call.args[0] and 'reference' not in call.args[0]
    policy.model.infer_forcewm.assert_not_called()
    if backend == 'joint_optimized':
        policy.model.infer_joint.assert_not_called()


@pytest.mark.parametrize('backend', ['reference', 'joint_static_cache', 'joint_optimized', 'joint_multirate'])
def test_merged_joint_backends_preserve_dispatch_seeds_and_sampling(backend):
    policy, _ = policy_fixture(False)
    policy.reference_mode = 'none'
    policy.inference_backend = backend
    policy.config.update(joint_steps=16, sampling_profile='4416')
    accelerated = backend in {'joint_optimized', 'joint_multirate'}
    policy.runner = Mock() if accelerated else None
    target = policy.runner if accelerated else policy.model
    method = target.infer if backend == 'joint_multirate' else target.infer_joint
    method.return_value = {'action': torch.zeros(1, 32, 7)}
    if backend == 'joint_multirate':
        target.diagnostics = {'effective_mode': 'eager'}
    obs = Observation.now(1, (0.,)*7, (0.,)*6, {'global':bytes(3), 'wrist':bytes(3)})
    history = OnlineHistory(policy.normalizer)
    history.append(obs.state, obs.force)
    for _ in range(2):
        assert len(policy.infer(obs, None, history.tensors())) == 32
    assert [call.kwargs['seed'] for call in method.call_args_list] == [42, 43]
    for call in method.call_args_list:
        assert call.kwargs['video_seed'] == 44 and call.kwargs['force_seed'] == 43
        assert 'reference' not in call.args[0] and 'action_condition' not in call.args[0]
        if backend == 'joint_multirate':
            assert call.kwargs['video_steps'] == call.kwargs['force_steps'] == 4
            assert call.kwargs['action_steps'] == 16
            assert 'num_inference_steps' not in call.kwargs
        else:
            assert call.kwargs['num_inference_steps'] == 16
            assert call.kwargs['use_cache'] is (backend != 'reference')
    if accelerated:
        policy.model.infer_joint.assert_not_called()
