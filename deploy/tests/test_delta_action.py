"""Relative model actions become absolute commands at the request boundary."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fastwam.cosmos.joint_action import joint_action_contract
from realman_deploy.contracts import Observation
from realman_deploy.normalization import DatasetNormalizer
from realman_deploy.runtime import DeploymentRuntime
from realman_sync.cosmos_policy import CosmosPolicy


def stats():
    return {name: dict(mean=[.5] * size, std=[2.] * size)
            for name, size in [('state', 8), ('action', 8), ('force', 6)]}


def normalizer(representation='delta_current_joint7'):
    return DatasetNormalizer(stats(), action_space='joint8', action_representation=representation)


def observation(state=None):
    return Observation(100., 4, tuple(state or [1., 2., 3., 4., 5., 6., 7., .2]),
                       (0.,) * 6, {'global': b'rgb', 'wrist': b'rgb'}, action_space='joint8')


def test_delta_unscales_before_common_request_anchor_without_cumsum():
    n = normalizer()
    values = [[1.] * 7 + [.1], [2.] * 7 + [.2]]
    state = observation().state
    physical = n.decode('action', values)
    result = n.decode_action(values, state)
    for actual, delta in zip(result, physical):
        assert actual[:7] == [state[i] + delta[i] for i in range(7)]
        assert actual[7] == delta[7]
    assert values == [[1.] * 7 + [.1], [2.] * 7 + [.2]]
    assert n.decode_action(values[0], state) == result[0]


def test_legacy_absolute_decode_ignores_anchor_and_keeps_exact_results():
    n = normalizer('absolute')
    values = [[1.] * 7 + [.1]] * 32
    assert n.decode_action(values) == n.decode('action', values)
    assert n.decode_action(values, [999.] * 8) == n.decode('action', values)


@pytest.mark.parametrize('anchor', [None, [1.] * 7, [float('nan')] * 8])
def test_delta_requires_valid_measured_anchor(anchor):
    with pytest.raises(ValueError, match='measured'):
        normalizer().decode_action([[0.] * 8], anchor)


def test_contract_mismatch_and_partial_metadata_are_rejected(tmp_path):
    contract = joint_action_contract('delta_current_joint7')
    payload = dict(statistics=stats(), action_space='joint8', state_columns=[0, 1, 2, 3, 4, 5, 6, 19], **contract)
    path = tmp_path / 'normalization.json'
    path.write_text(json.dumps(payload))
    loaded = DatasetNormalizer.from_json(path)
    loaded.validate_action_contract(contract)
    with pytest.raises(ValueError):
        loaded.validate_action_contract({})
    with pytest.raises(ValueError):
        normalizer('absolute').validate_action_contract(contract)
    for key in ('action_anchor', 'action_delta_mask', 'action_representation'):
        broken = copy.deepcopy(payload)
        broken.pop(key)
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError):
            DatasetNormalizer.from_json(path)


def test_sync_adapter_uses_observation_anchor_not_last_history_or_command():
    n = normalizer()
    values = [[0.] * 7 + [0.]] * 32
    wrapper = CosmosPolicy.__new__(CosmosPolicy)
    wrapper.policy = SimpleNamespace(normalizer=n, infer=Mock(return_value=values))
    obs = observation()
    raw = dict(stamp=obs.stamp, sequence=obs.sequence, state=list(obs.state), force=list(obs.force),
               frames={key: dict(data=value) for key, value in obs.frames.items()}, frame_meta={}, force_meta={})
    history = [dict(state=[50.] * 7 + [.9], force=[0.] * 6)]
    actual = wrapper.predict(raw, history, [raw])
    assert actual == n.decode_action(values, obs.state)
    assert actual[0] == actual[-1]


def test_runtime_uses_original_request_state_even_when_new_observation_arrives():
    n = normalizer()
    runtime = DeploymentRuntime(n, clock=lambda: 100.1, require_cuda=False)
    runtime.policy = SimpleNamespace(reference_mode='none')
    runtime.state = 'running'
    runtime.observations.put(observation([40.] * 7 + [.9]))
    runtime.timeline.replace = Mock(return_value=(0, 32))
    source = observation()
    values = [[0.] * 8] * 32
    assert runtime._publish(values, source, runtime.gate.current(), 100.)
    chunk = runtime.timeline.replace.call_args.args[0]
    assert chunk.actions == tuple(tuple(row) for row in n.decode_action(values, source.state))
    assert chunk.source_sequence == source.sequence


def test_runtime_rejects_matching_stats_with_different_action_semantics():
    runtime = DeploymentRuntime(normalizer(), require_cuda=False)
    with pytest.raises(ValueError):
        runtime._validate_policy_normalizer(SimpleNamespace(normalizer=normalizer('absolute')))


def test_legacy_eef_normalization_contract_is_preserved(tmp_path):
    path = tmp_path / 'eef.json'
    payload = dict(statistics={name: dict(mean=[0.] * size, std=[1.] * size)
                              for name, size in [('state', 7), ('action', 7), ('force', 6)]},
                   action_units='xyz metres + Euler radians + gripper openness')
    path.write_text(json.dumps(payload))
    n = DatasetNormalizer.from_json(path)
    n.validate_action_contract(payload)
    assert n.action_representation == 'absolute'
