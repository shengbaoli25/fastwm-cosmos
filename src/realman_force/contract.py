"""Versioned model-input contract and causal timestamp selection."""
import copy
import hashlib
import json
from bisect import bisect_right

from .force_filter import make_manifest, validate_manifest

SELECTED_CONFIG = {
    'type': 'kalman', 'model': 'random_walk',
    'measurement_variance': [.25, .25, .25, .0004, .0004, .0004],
    'process_variance_per_s': [.000762939453125] * 3 + [.000001220703125] * 3,
    'sample_rate_hz': 200., 'warmup_s': .5, 'max_gap_s': .03,
}


def input_contract(calibration, manifest=None):
    manifest = manifest or make_manifest(SELECTED_CONFIG, calibration)
    validate_manifest(manifest, calibration)
    result = {
        'schema': 'realman_model_force_v1',
        'field': 'observation.force_filtered',
        'state_force': 'observation.force_filtered',
        'coordinates': 'World/base axes; controller wrench origin',
        'units': ['N', 'N', 'N', 'Nm', 'Nm', 'Nm'],
        'filter': copy.deepcopy(manifest),
        'sampling': {'fps': 30, 'method': 'latest_source_at_or_before_grid',
                     'max_source_age_s': .03, 'interpolation': 'none',
                     'filter_updates': 'all_native_packets_actual_dt',
                     'initialization': 'continuous_across_episodes; saved_state_for_replay'},
    }
    result['id'] = hashlib.sha256(json.dumps(result, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return result


def validate_contract(contract, calibration):
    if contract != input_contract(calibration, contract['filter']):
        raise ValueError('Model force preprocessing contract differs from calibration/filter/sampling')
    return contract


def validate_contract_identity(contract):
    data = {k: v for k, v in contract.items() if k != 'id'}
    digest = hashlib.sha256(json.dumps(data, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if (contract.get('id') != digest or contract.get('schema') != 'realman_model_force_v1'
            or contract.get('field') != 'observation.force_filtered'):
        raise ValueError('Invalid model force preprocessing identity')
    return contract


def source_at(history, anchor, max_age_s=.03):
    """No interpolation or future packets; caller owns an immutable snapshot."""
    times = [p['time'] for p in history]
    i = bisect_right(times, anchor) - 1
    if i < 0 or not 0 <= anchor-times[i] <= max_age_s:
        raise ValueError('No fresh force sample at or before observation grid time')
    return history[i], history[:i+1]
