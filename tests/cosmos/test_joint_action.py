"""The physical action contract is shared by training and deployment."""
import numpy as np
import pytest
import torch

from fastwam.cosmos.joint_action import (
    joint_action_contract, validate_joint_action_contract, encode_joint_actions, decode_joint_actions,
)


@pytest.mark.parametrize('tensor', [False, True])
def test_delta_uses_one_measured_anchor_and_round_trips_without_gripper_delta(tensor):
    actions = np.array([[[1.] * 7 + [.2], [3.] * 7 + [.7], [6.] * 7 + [.4]]], dtype=np.float32)
    state = np.array([[.5] * 7 + [.9]], dtype=np.float32)
    if tensor:
        actions, state = torch.from_numpy(actions), torch.from_numpy(state)
    before = actions.clone() if tensor else actions.copy()
    delta = encode_joint_actions(actions, state, 'delta_current_joint7')
    np.testing.assert_allclose(delta[0, :, 0], [.5, 2.5, 5.5])
    np.testing.assert_array_equal(delta[..., 7], before[..., 7])
    np.testing.assert_array_equal(actions, before)
    np.testing.assert_allclose(decode_joint_actions(delta, state, 'delta_current_joint7'), before)
    with pytest.raises(ValueError, match='shapes'):
        decode_joint_actions(delta, state[0], 'delta_current_joint7')
    with pytest.raises(ValueError, match='require'):
        decode_joint_actions(delta, None, 'delta_current_joint7')


def test_explicit_contract_cannot_confuse_legacy_absolute_with_delta():
    assert validate_joint_action_contract({}) == 'absolute'
    assert validate_joint_action_contract({'action_units': '7 absolute joint radians + gripper openness'}) == 'absolute'
    with pytest.raises(ValueError, match='differs'):
        validate_joint_action_contract({}, 'delta_current_joint7')
    valid = joint_action_contract('delta_current_joint7')
    assert validate_joint_action_contract(valid) == 'delta_current_joint7'
    for key in valid:
        broken = {k: v for k, v in valid.items() if k != key}
        with pytest.raises(ValueError):
            validate_joint_action_contract(broken)
    with pytest.raises(ValueError):
        validate_joint_action_contract({**valid, 'action_anchor': 'previous_action'})


def test_absolute_is_a_copy_and_bad_inputs_fail_closed():
    actions = np.zeros((32, 8), dtype=np.float32)
    assert encode_joint_actions(actions, None) is not actions
    np.testing.assert_array_equal(encode_joint_actions(actions, None), actions)
    with pytest.raises(ValueError, match='Unsupported'):
        joint_action_contract('adjacent_delta')
    with pytest.raises(ValueError, match='Finite'):
        encode_joint_actions(actions.astype(np.int64), None)
    with pytest.raises(ValueError, match='Nonfinite'):
        encode_joint_actions(actions, [float('nan')] * 8, 'delta_current_joint7')
