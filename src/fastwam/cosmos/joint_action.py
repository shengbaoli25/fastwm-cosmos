"""Explicit Joint8 action semantics shared by data preparation and deployment.

All functions operate in physical units, before normalization or after its
inverse. A delta chunk uses one measured observation as its common anchor;
it is never an adjacent-step difference and is never cumulatively summed.
"""
ACTION_REPRESENTATIONS = ('absolute', 'delta_current_joint7')
ACTION_UNITS = ['rad'] * 7 + ['openness_0_1']


def joint_action_contract(representation='absolute'):
    if representation not in ACTION_REPRESENTATIONS:
        raise ValueError('Unsupported joint action_representation: ' + str(representation))
    delta = representation == 'delta_current_joint7'
    return dict(action_representation=representation,
                action_anchor='current_measured_joint7' if delta else 'none',
                action_delta_mask=[delta] * 7 + [False], action_units=list(ACTION_UNITS))


def validate_joint_action_contract(metadata, expected=None, *, allow_legacy_absolute=True):
    """Accept old absolute metadata, but never infer or partially accept deltas."""
    representation = metadata.get('action_representation', 'absolute')
    contract = joint_action_contract(representation)
    if expected is not None and representation != expected:
        raise ValueError('Joint action representation differs from the requested contract')
    new_keys = ('action_representation', 'action_anchor', 'action_delta_mask')
    legacy = not any(key in metadata for key in new_keys)
    if legacy and allow_legacy_absolute:
        if ('action_units' in metadata and metadata['action_units'] != ACTION_UNITS
                and metadata['action_units'] != '7 absolute joint radians + gripper openness'):
            raise ValueError('Legacy joint action units differ')
        return representation
    if any(metadata.get(key) != value for key, value in contract.items()):
        raise ValueError('Incomplete or inconsistent joint action representation/anchor/units')
    return representation


def _convert(actions, current_state, representation, sign):
    import numpy as np
    import torch

    joint_action_contract(representation)
    is_torch = isinstance(actions, torch.Tensor)
    if not is_torch and not isinstance(actions, np.ndarray):
        actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim < 2 or actions.shape[-1] != 8:
        raise ValueError('Expected joint action chunk [..., horizon, 8]')
    if is_torch:
        if not actions.is_floating_point() or not bool(torch.isfinite(actions).all()):
            raise ValueError('Finite floating joint actions required')
        out = actions.clone()
    else:
        if not np.issubdtype(actions.dtype, np.floating) or not np.isfinite(actions).all():
            raise ValueError('Finite floating joint actions required')
        out = actions.copy()
    if representation == 'absolute':
        return out
    if current_state is None:
        raise ValueError('Delta actions require the request current measured joint state')
    state = (torch.as_tensor(current_state, dtype=out.dtype, device=out.device) if is_torch
             else np.asarray(current_state, dtype=out.dtype))
    if state.shape != (*out.shape[:-2], 8):
        raise ValueError('Action chunk and current measured joint8 anchor shapes differ')
    finite = bool(torch.isfinite(state).all()) if is_torch else bool(np.isfinite(state).all())
    if not finite:
        raise ValueError('Nonfinite current measured joint8 anchor')
    out[..., :7] += sign * state[..., None, :7]
    return out


def encode_joint_actions(actions, current_state, representation='absolute'):
    return _convert(actions, current_state, representation, -1)


def decode_joint_actions(actions, current_state, representation='absolute'):
    return _convert(actions, current_state, representation, 1)
