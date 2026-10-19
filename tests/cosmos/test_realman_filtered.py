import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fastwam.cosmos.realman import RealManSource
from realman_force.contract import input_contract


def dataset(root):
    calibration = json.loads((Path(__file__).parents[2]/'deploy/force_calibration.json').read_text())
    contract = input_contract(calibration)
    n = 40
    for directory in ['meta/episodes', 'meta/force_sessions', 'data']:
        (root/directory).mkdir(parents=True)
    (root/'meta/info.json').write_text(json.dumps(dict(codebase_version='v3.0',
        total_episodes=1, total_frames=n, fps=30, data_path='data/rows.parquet')))
    (root/'meta/force.json').write_text(json.dumps(dict(schema='realman_force_v4', model_input=contract)))
    (root/'meta/force_sessions/episode_000000.json').write_text(json.dumps(calibration))
    pq.write_table(pa.Table.from_pylist([dict(episode_index=0, length=n,
        dataset_from_index=0, dataset_to_index=n, tasks=['cup'], **{'data/chunk_index':0, 'data/file_index':0})]), root/'meta/episodes/rows.parquet')
    rows = []
    for i in range(n):
        force = (np.arange(6)+i/10).tolist()
        state = [i/100]*14+force
        state[7] = .25+i/100
        action = state[:14]
        t = 1000+i/30
        rows.append({'episode_index':0, 'frame_index':i, 'index':i, 'timestamp':i/30,
            'observation.state':state, 'action':action, 'observation.force': [99.]*6,
            'observation.force_filtered':force, 'observation.force_timestamp_s':[t],
            'observation.monotonic_time_s':[t],
            'observation.force_filter_timestamp_s':[t], 'observation.robot_timestamp_s':[t],
            'action.timestamp_s':[t], 'observation.gripper_timestamp_s':[t],
            'action.gripper_timestamp_s':[t], 'observation.camera.arrival_time_s':[t,t],
            'observation.camera.age_s':[0.,0.], 'observation.camera.pair_skew_s':[0.],
            'observation.action_age_s':[0.]})
    pq.write_table(pa.Table.from_pylist(rows), root/'data/rows.parquet')
    return contract


def test_new_dataset_trains_on_selected_filtered_field_and_exports_contract(tmp_path):
    contract = dataset(tmp_path)
    source = RealManSource(tmp_path)
    np.testing.assert_allclose(source.raw['force'][5], np.arange(6)+.5)
    path = tmp_path/'normalization.json'
    source.export_normalization(path)
    saved = json.loads(path.read_text())
    assert saved['force_preprocessing'] == contract
    assert saved['statistics']['force']['mean'] == source.stats['force']['mean']
    assert source.identity['force_preprocessing'] == contract
    from realman_deploy.normalization import DatasetNormalizer, OnlineHistory
    online = DatasetNormalizer.from_json(path)
    history = OnlineHistory(online)
    for index in range(40):
        history.append(source.raw['state'][index].tolist(), source.raw['force'][index].tolist())
        rows, pad, valid = history.tensors()
        np.testing.assert_array_equal(rows, source[index]['history'].numpy())
        assert pad == source[index]['history_is_pad'].tolist()
        assert valid == source[index]['history_channel_valid'].tolist()


def test_tampered_filter_identity_is_rejected(tmp_path):
    dataset(tmp_path)
    path = tmp_path/'meta/force.json'
    meta = json.loads(path.read_text())
    meta['model_input']['filter']['config']['max_gap_s'] = .02
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        RealManSource(tmp_path)


def test_filtered_training_cache_accepts_new_scope_and_binds_preprocessing(tmp_path):
    from test_cache import build
    from fastwam.cache.latent import cache_identity_sha256
    from fastwam.cosmos.cache import validate
    calibration = json.loads((Path(__file__).parents[2]/'deploy/force_calibration.json').read_text())
    root = tmp_path/'features'
    manifest, _ = build(root)
    identity = manifest['identity']
    source = dict(profile='realman-filtered-force-v1', episode_lengths=[2,1], fps=30,
        files={}, source_inventory_sha256=cache_identity_sha256({}),
        eef_columns=[8,9,10,11,12,13,7],
        cameras=['observation.images.global','observation.images.wrist'],
        force_preprocessing=input_contract(calibration))
    identity.update(scope='production', dataset_profile=source['profile'],
        realman_source=source, numeric_identity=cache_identity_sha256(source))
    manifest['identity_sha256'] = cache_identity_sha256(identity)
    path = root/'manifest.json'
    path.write_text(json.dumps(manifest))
    validate(root)
    source['force_preprocessing']['sampling']['fps'] = 29
    identity['numeric_identity'] = cache_identity_sha256(source)
    manifest['identity_sha256'] = cache_identity_sha256(identity)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='preprocessing identity'):
        validate(root)
