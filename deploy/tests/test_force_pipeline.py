"""Native-rate preprocessing must not depend on model/transport batching."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from realman_force.contract import SELECTED_CONFIG, input_contract, source_at
from realman_force.force_calibration import compensation, design, fit_profile
from realman_force.force_filter import FilterWarmup, ForceFilter
from realman_deploy.hardware import _RobotFeedback
from realman_deploy.normalization import DatasetNormalizer, OnlineHistory
from realman_deploy.runtime import DeploymentRuntime


def profile():
    params = np.array([16., .3, -.2, .1, 1.2, -.6, .8, .02, -.03, .04])
    angles = [(0, 0, 0), (.8, 0, .2), (-.8, .2, -.2), (0, .8, .4),
              (.2, -.8, .3), (1.2, .5, .9), (-.5, -.5, -.8), (.6, .6, .6)]
    return fit_profile([dict(pose=[.2, .1, .3, *a],
        tool_wrench=(design([0, 0, 0, *a]) @ params).tolist(),
        contact_free_confirmed=True) for a in angles])


def sdk_state(point):
    p = point['pose']
    return SimpleNamespace(arm_ip=b'192.168.1.18', errCode=0,
        joint_status=SimpleNamespace(joint_err_code=[0]*7, joint_en_flag=[True]*7,
                                     joint_position=[0.]*7),
        waypoint=SimpleNamespace(position=SimpleNamespace(x=p[0], y=p[1], z=p[2]),
                                 euler=SimpleNamespace(rx=p[3], ry=p[4], rz=p[5])),
        force_sensor=SimpleNamespace(zero_force=point['force'], coordinate=1))


def test_200hz_callback_and_30hz_batched_collection_are_identical_with_jitter():
    calibration = profile()
    cfg = copy.deepcopy(SELECTED_CONFIG)
    collector = ForceFilter(cfg)
    hardware = _RobotFeedback(None, {'robot_ip': '192.168.1.18',
        '_force_calibration': calibration, '_force_filter': cfg})
    t = 1000.+np.arange(800)/200
    # 27 ms between two arrivals; a short gap must not erase a half-second.
    t[300:] += .022
    points = []
    for i, stamp in enumerate(t):
        pose = [.2, .1, .3, .4, .2, .3]
        signal = np.array([1, 2, 3, .1, .2, .3])*(i >= 250)
        signal += .2*np.sin(2*np.pi*50*i/200)
        points.append(dict(time=float(stamp), pose=pose,
                           force=(compensation(calibration, pose)+signal).tolist()))
    pending, cursor, rows, initial_epoch = [], 0, 0, None
    stats = {name: dict(mean=[0.]*width, std=[1.]*width)
             for name, width in [('state',7), ('action',7), ('force',6)]}
    normalization = DatasetNormalizer(stats, input_contract(calibration))
    for anchor in 1000.+np.arange(121)/30:
        while cursor < len(points) and points[cursor]['time'] <= anchor:
            point = points[cursor]
            hardware.ingest(sdk_state(point), point['time'])
            pending.append(point)
            cursor += 1
        if not pending:
            continue
        last = pending[-1]
        sample = dict(force_time=last['time'], force=last['force'],
                      pose=last['pose'], force_history=pending)
        pending = []
        try:
            got = collector.apply(sample, calibration)
        except FilterWarmup:
            continue
        _, _, deployed, source_stamp, meta = hardware.model_snapshot(float(anchor))
        assert source_stamp == last['time'] <= anchor
        assert meta['preprocessing_id'] == normalization.force_preprocessing['id']
        np.testing.assert_array_equal(deployed, got['force_filtered'])
        # Dataset float32 storage and deployment normalization produce identical inputs.
        stored = np.asarray(got['force_filtered'], dtype=np.float32).tolist()
        assert normalization.encode('force', stored) == normalization.encode('force', deployed)
        initial_epoch = got['force_filter_epoch'] if initial_epoch is None else initial_epoch
        assert got['force_filter_epoch'] == initial_epoch
        rows += 1
    assert rows > 90


def test_gap_and_no_future_selection_are_not_filled():
    values = [dict(time=t) for t in [1000., 1000.005, 1000.050]]
    assert source_at(values, 1000.010)[0]['time'] == 1000.005
    with pytest.raises(ValueError, match='fresh'):
        source_at(values, 1000.040)
    with pytest.raises(ValueError, match='fresh'):
        source_at(values, 999.)


def test_old_normalization_cannot_attach_filtered_hardware():
    stats = {name: dict(mean=[0.]*width, std=[1.]*width)
             for name, width in [('state',7), ('action',7), ('force',6)]}
    contract = input_contract(profile())
    hw = SimpleNamespace(status=lambda: dict(force_preprocessing=contract))
    runtime = DeploymentRuntime(DatasetNormalizer(stats), require_cuda=False)
    with pytest.raises(ValueError, match='预处理不一致'):
        runtime.attach(hardware=hw)
    good = DeploymentRuntime(DatasetNormalizer(stats, contract), require_cuda=False)
    good.attach(hardware=hw)


def test_policy_cannot_override_runtime_training_statistics():
    stats = {name: dict(mean=[0.]*width, std=[1.]*width)
             for name, width in [('state',7), ('action',7), ('force',6)]}
    contract = input_contract(profile())
    runtime = DeploymentRuntime(DatasetNormalizer(stats, contract), require_cuda=False)
    changed = copy.deepcopy(stats)
    changed['force']['mean'][0] = 1.
    policy = SimpleNamespace(normalizer=DatasetNormalizer(changed, contract))
    with pytest.raises(ValueError, match='normalization.json'):
        runtime.attach(policy=policy)


def test_native_processing_continues_without_inference_requests():
    p = profile()
    hw = _RobotFeedback(None, {'robot_ip': '192.168.1.18',
        '_force_calibration': p, '_force_filter': SELECTED_CONFIG})
    for i in range(400):
        pose = [.2, .1, .3, .4, .2, .3]
        point = dict(time=1000.+i/200, pose=pose, force=compensation(p, pose).tolist())
        hw.ingest(sdk_state(point), point['time'])
    assert hw.force_filter.count == 400 and hw.force_filter.ready
    assert hw.model_snapshot(1001.995)[4]['grid_time'] == 1001.995


def test_long_gap_still_requires_warmup():
    p = profile()
    f = ForceFilter(SELECTED_CONFIG)
    pose = [.2, .1, .3, .4, .2, .3]
    force = compensation(p, pose).tolist()
    def apply(t):
        return f.apply(dict(force_time=t, force=force, pose=pose,
                            force_history=[dict(time=t, force=force, pose=pose)]), p)
    for i in range(120):
        try: apply(1000.+i/200)
        except FilterWarmup: pass
    assert f.ready
    with pytest.raises(FilterWarmup): apply(1000.640)
    assert not f.ready and f.count == 1


def test_checkpoint_binding_rejects_changed_weights_or_statistics(tmp_path):
    import json
    from realman_deploy.preprocessing import binding, verify_binding
    checkpoint = tmp_path/'model.pt'
    checkpoint.write_bytes(b'checkpoint fixture')
    normalizer = tmp_path/'normalization.json'
    normalizer.write_text(json.dumps(dict(force_preprocessing=input_contract(profile()))))
    sidecar = Path(str(checkpoint)+'.force.json')
    sidecar.write_text(json.dumps(binding(checkpoint, normalizer)))
    verify_binding(checkpoint, normalizer)
    checkpoint.write_bytes(b'different checkpoint')
    with pytest.raises(ValueError, match='not bound'): verify_binding(checkpoint, normalizer)
