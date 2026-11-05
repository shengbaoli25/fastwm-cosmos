"""CPU-only reference sidecars for the immutable RealMan Joint8 source.

No torch, video decoding, normalization changes, or model imports. Source frame
indices are supervision metadata and must never become auditor input features.
"""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pyarrow.parquet as pq

from .reference_trajectory import build_reference, euler_xyz_to_quaternion, augment_reference

SCHEMA = 'realman-episode-reference-bank-v1'
SOURCE_SCHEMA = 'realman_leader_force_gripper_v2'
POSE_NAMES = ['tcp_x_m', 'tcp_y_m', 'tcp_z_m', 'tcp_rx_rad', 'tcp_ry_rad', 'tcp_rz_rad']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def check(ok, message):
    if not ok:
        raise ValueError(message)


def inside(root, relative):
    path = (root / relative).resolve()
    check(path.is_relative_to(root) and path.is_file(), f'Missing/escaping source: {relative}')
    return path


def continuous_segments_cpu(t):
    """Match realman_joint.continuous_segments without importing the GPU stack."""
    grid = np.asarray(t['observation.monotonic_time_s'], dtype=np.float64).reshape(-1)
    n = len(grid)
    check(n > 0 and np.isfinite(grid).all(), 'Invalid capture clock')
    ft = np.asarray(t['observation.force_timestamp_s'], dtype=np.float64).reshape(n)
    fresh = np.asarray(t['observation.capture_fresh'])
    check(fresh.shape == (n, 4), 'Missing four-source freshness flags')
    valid = (fresh == 1).all(1) & (np.asarray(t['action.applied']).reshape(n) == 1)
    valid &= np.isfinite(ft) & (grid-ft >= -1e-6) & (grid-ft <= .03)
    valid &= np.asarray(t['observation.force_filter_input_count']).reshape(n) >= 100
    breaks = np.zeros(n, dtype=bool)
    breaks[0] = True
    breaks[1:] = np.abs(np.diff(grid)-1/30) > 2e-5
    breaks |= np.asarray(t['observation.capture_discontinuity']).reshape(n) != 0
    breaks |= np.asarray(t['observation.capture_gap_s']).reshape(n) > 2e-5
    for key in ('observation.force_filter_epoch', 'action.follow_session_id',
                'action.leader_anchor', 'action.follower_anchor'):
        values = np.asarray(t[key])
        changes = values[1:] != values[:-1]
        breaks[1:] |= changes if changes.ndim == 1 else changes.any(1)
    runs, start = [], None
    for j in range(n):
        if start is not None and (breaks[j] or not valid[j]):
            runs.append([start, j]); start = None
        if valid[j] and start is None:
            start = j
    if start is not None:
        runs.append([start, n])
    return [r for r in runs if r[1]-r[0] >= 2], int((~valid).sum())


def prepare_bank(dataset, output, *, expected_episodes=78, expected_source_sha=None,
                 coordinate_frame='recorded_controller_tcp_frame', split_json=None,
                 progress=None, **coarsening):
    root, out = Path(dataset).resolve(), Path(output).resolve()
    check(root.is_dir(), 'Raw dataset directory does not exist')
    check(not out.exists(), 'Output already exists; use a fresh reference bank directory')
    check(not out.is_relative_to(root) and not root.is_relative_to(out),
          'Reference bank must be outside the immutable raw dataset')
    info = json.loads(inside(root, 'meta/info.json').read_text(encoding='utf-8'))
    force = json.loads(inside(root, 'meta/force.json').read_text(encoding='utf-8'))
    leader = json.loads(inside(root, 'meta/leader.json').read_text(encoding='utf-8'))
    check((info['codebase_version'], info['fps']) == ('v3.0', 30), 'Expected LeRobot v3 at 30 Hz')
    check(force['schema'] == SOURCE_SCHEMA, 'Expected measured TCP Joint8 source schema')
    check(leader.get('training_demonstration') is True and leader.get('mode') == 'execute',
          'Source must contain executed demonstrations')
    names = info['features']['observation.state']['names']
    check(len(names) == 20 and names[7:13] == POSE_NAMES and names[19] == 'gripper_open',
          'State layout differs; refuse an implicit TCP column guess')
    eps_files = sorted((root/'meta/episodes').rglob('*.parquet'))
    episodes = sorted([e for p in eps_files for e in pq.read_table(p).to_pylist()],
                      key=lambda e: e['episode_index'])
    check(len(episodes) == info['total_episodes'] == expected_episodes, 'Wrong episode count')
    check([e['episode_index'] for e in episodes] == list(range(len(episodes))), 'Episode gaps/duplicates')
    numeric_paths = sorted({info['data_path'].format(chunk_index=e['data/chunk_index'],
                            file_index=e['data/file_index']) for e in episodes})
    # Hash all bytes only when explicitly binding the pre-existing raw inventory.
    if expected_source_sha:
        files = {p.relative_to(root).as_posix(): file_digest(p)
                 for p in sorted(root.rglob('*')) if p.is_file()}
        actual_sha = digest(files)
        check(actual_sha == expected_source_sha,
              f'Raw inventory differs: expected {expected_source_sha}, actual {actual_sha}')
        identity_scope = 'complete_raw_inventory'
    else:
        paths = set(numeric_paths)
        paths.update(p.relative_to(root).as_posix() for p in (root/'meta').rglob('*') if p.is_file())
        files = {name: file_digest(inside(root, name)) for name in sorted(paths)}
        identity_scope = 'all_metadata_and_numeric_parquet_no_video_bytes'
    if split_json:
        splits = json.loads(Path(split_json).read_text(encoding='utf-8'))
        check(set(splits) == {'train', 'val', 'test'}, 'Split JSON requires train/val/test lists')
        ids = [i for part in splits.values() for i in part]
        check(all(type(i) is int for i in ids) and sorted(ids) == list(range(len(episodes))),
              'Splits must partition original episode IDs exactly once')
    else:
        # Do not pretend a sidecar can retroactively create an independent holdout.
        splits = {'train': list(range(len(episodes))), 'val': [], 'test': []}
    identity = dict(schema=SCHEMA, source_schema=SOURCE_SCHEMA, fps=30,
        source_inventory_scope=identity_scope, source_inventory_sha256=digest(files), source_files=files,
        coordinate_frame=coordinate_frame, pose_layout='xyz_m,quaternion_xyzw,measured_gripper_0_1',
        rotation_convention='recorded Euler extrinsic xyz, Rz(rz) Ry(ry) Rx(rx)',
        geometry_source='observation.state[7:13]', gripper_source='observation.state[19]',
        action_labels='unchanged: delta_current_joint7 plus absolute gripper in existing WAM cache',
        coarsening=coarsening, splits=splits, split_scope='membership_only_no_normalization_recomputed',
        source_lengths=[int(e['length']) for e in episodes],
        builder_sha256=file_digest(__file__),
        geometry_builder_sha256=file_digest(Path(__file__).with_name('reference_trajectory.py')))
    out.mkdir(parents=True)
    (out/'episodes').mkdir()
    manifest = dict(schema=SCHEMA, identity=identity, identity_sha256=digest(identity),
                    status='building', episodes=[], total_source_rows=0, total_retained_anchors=0,
                    total_reference_points=0)
    atomic_json(out/'manifest.json', manifest)
    cursor = 0
    for ep, e in enumerate(episodes):
        path = inside(root, info['data_path'].format(chunk_index=e['data/chunk_index'], file_index=e['data/file_index']))
        table = pq.read_table(path).to_pydict()
        ids = [j for j, value in enumerate(table['episode_index']) if value == ep]
        t = {k: [v[j] for j in ids] for k, v in table.items()}
        n = int(e['length'])
        check(len(ids) == n and t['frame_index'] == list(range(n))
              and t['index'] == list(range(cursor, cursor+n)), f'Episode {ep}: row order differs')
        check((e['dataset_from_index'], e['dataset_to_index']) == (cursor, cursor+n), 'Global source bounds differ')
        check(np.allclose(t['timestamp'], np.arange(n)/30, atol=2e-6), 'Video frame grid differs')
        state = np.asarray(t['observation.state'], dtype=np.float32)
        action = np.asarray(t['action'], dtype=np.float32)
        wrench = np.asarray(t['observation.force_filtered'], dtype=np.float32)
        check(state.shape == (n,20) and action.shape == (n,8) and wrench.shape == (n,6), 'Invalid numeric dimensions')
        check(all(np.isfinite(x).all() for x in (state,action,wrench)), 'Nonfinite numeric values')
        check(np.array_equal(state[:,13:19],wrench), 'Filtered force / state mismatch')
        check(all(((x >= 0)&(x <= 1)).all() for x in (state[:,19], action[:,7])), 'Gripper is not openness in [0,1]')
        ft = np.asarray(t['observation.force_timestamp_s']).reshape(n)
        for key in ('observation.force_filter_timestamp_s', 'observation.robot_timestamp_s'):
            check(np.array_equal(ft, np.asarray(t[key]).reshape(n)), 'Force/pose/filter timestamps differ')
        for key in ('action.timestamp_s', 'observation.leader_timestamp_s', 'observation.gripper_timestamp_s',
                    'action.gripper_timestamp_s', 'observation.camera.arrival_time_s'):
            value = np.asarray(t[key])
            bound = ft[:, None] if value.ndim == 2 else ft
            check(np.isfinite(value).all() and (value <= bound+1e-6).all(), 'Future input: '+key)
        check(len(e['tasks']) == 1 and bool(e['tasks'][0]), 'One task per episode required')
        from realman_force.contract import validate_contract
        calibration = json.loads(inside(root, f'meta/force_sessions/episode_{ep:06d}.json').read_text(encoding='utf-8'))
        validate_contract(force['model_input'], calibration)
        runs, invalid = continuous_segments_cpu(t)
        check(bool(runs), f'Episode {ep} has no valid segments')
        reference = build_reference(state, runs, **coarsening)
        frames = reference['source_frame']
        reference['source_timestamp_s'] = np.asarray(t['timestamp'],dtype=np.float64)[frames]
        reference['source_capture_time_s'] = np.asarray(t['observation.monotonic_time_s'],dtype=np.float64).reshape(n)[frames]
        reference['valid_anchor_frame'] = np.concatenate([np.arange(lo,hi,dtype=np.int64) for lo,hi in runs])
        anchors = reference['valid_anchor_frame']
        reference['observation_tcp'] = np.column_stack((state[anchors,7:10],
            euler_xyz_to_quaternion(state[anchors,10:13]),state[anchors,19])).astype(np.float32)
        rel = f'episodes/episode_{ep:06d}.npz'
        with (out/rel).open('wb') as f:
            np.savez_compressed(f, **reference)
        manifest['episodes'].append(dict(episode_index=ep, path=rel, sha256=file_digest(out/rel),
            source_rows=n, reference_points=len(frames), retained_anchors=len(reference['valid_anchor_frame']),
            invalid_rows=invalid, segment_bounds=runs, task=e['tasks']))
        manifest['total_source_rows'] += n
        manifest['total_retained_anchors'] += len(reference['valid_anchor_frame'])
        manifest['total_reference_points'] += len(frames)
        cursor += n
        atomic_json(out/'manifest.json',manifest)
        if progress:
            progress(ep+1,len(episodes),len(frames))
    check(cursor == info['total_frames'], 'Source total frame count differs')
    manifest['status'] = 'complete'
    atomic_json(out/'manifest.json',manifest)
    return verify_bank(out)


def verify_bank(root):
    root = Path(root).resolve()
    m = json.loads(inside(root,'manifest.json').read_text(encoding='utf-8'))
    check(m['schema'] == SCHEMA and m['status'] == 'complete', 'Incomplete/wrong reference bank')
    check(digest(m['identity']) == m['identity_sha256'], 'Reference identity mismatch')
    check([e['episode_index'] for e in m['episodes']] == list(range(len(m['identity']['source_lengths']))),
          'Missing or duplicate episode payloads')
    points, anchors, rows = 0,0,0
    for e in m['episodes']:
        path = inside(root,e['path'])
        check(file_digest(path) == e['sha256'], 'Reference payload checksum mismatch')
        with np.load(path,allow_pickle=False) as z:
            pose, frames, segments, bounds = (z[k] for k in ('reference_pose','source_frame','segment_id','segment_bounds'))
            check(pose.shape == (e['reference_points'],8) and np.isfinite(pose).all(), 'Invalid reference pose')
            check(frames.dtype == np.int64 and frames.shape == segments.shape == (len(pose),), 'Invalid reference indices')
            check(bounds.tolist() == e['segment_bounds'], 'Reference segment bounds mismatch')
            check(np.allclose(np.linalg.norm(pose[:,3:7],axis=1),1,atol=1e-5), 'Invalid reference rotations')
            check(((pose[:,7]>=0)&(pose[:,7]<=1)).all(), 'Invalid reference gripper')
            expected = []
            for sid,(lo,hi) in enumerate(bounds):
                f = frames[segments == sid]
                check(len(f)>=2 and f[0]==lo and f[-1]==hi-1 and (np.diff(f)>0).all(), 'Invalid sparse segment mapping')
                expected.extend(range(int(lo),int(hi)))
            check(z['valid_anchor_frame'].tolist() == expected, 'Anchor inventory differs from continuous segments')
            check(len(expected) == e['retained_anchors'], 'Anchor count mismatch')
            observed = z['observation_tcp']
            check(observed.shape == (len(expected),8) and np.isfinite(observed).all(), 'Invalid observed TCP inventory')
            check(np.allclose(np.linalg.norm(observed[:,3:7],axis=1),1,atol=1e-5), 'Invalid observed TCP rotations')
        points += e['reference_points']; anchors += e['retained_anchors']; rows += e['source_rows']
    check((points,anchors,rows)==(m['total_reference_points'],m['total_retained_anchors'],m['total_source_rows']),
          'Reference bank totals mismatch')
    expected_files={'manifest.json',*[e['path'] for e in m['episodes']]}
    check({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==expected_files,
          'Unexpected/partial files in reference bank')
    return m


class ReferenceBank:
    """CPU lookup contract for a future reference-aware training Dataset wrapper.

    Original sample_index is NOT a raw frame index. Join using episode_index and
    frame_index. The old NoActionJointModel cannot consume this result directly.
    """
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = verify_bank(self.root)
        self._episode = None
        self._payload = None

    def sample(self, episode_index, frame_index, *, epoch=0, seed=42, config=None):
        for name, value in (('episode_index', episode_index), ('frame_index', frame_index),
                            ('epoch', epoch), ('seed', seed)):
            check(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                  and value >= 0, f'{name} must be a nonnegative integer')
        ep, frame = int(episode_index), int(frame_index)
        check(0 <= ep < len(self.manifest['episodes']), 'Unknown reference episode')
        if self._episode != ep:
            path = inside(self.root,self.manifest['episodes'][ep]['path'])
            with np.load(path,allow_pickle=False) as z:
                self._payload = {k:z[k].copy() for k in z.files}
            self._episode = ep
        data = self._payload
        matches = np.flatnonzero(data['valid_anchor_frame'] == frame)
        check(len(matches) == 1, 'Frame is not an eligible source anchor')
        # No frame in the augmentation seed: one plan stays stable across an
        # episode/window. A new epoch generates another length/noise realization.
        rng = np.random.default_rng(np.random.SeedSequence([int(seed),int(epoch),ep]))
        inputs, targets = augment_reference(data,rng,config=config,anchor_frame=frame)
        inputs['current_tcp'] = data['observation_tcp'][matches[0]].copy()
        return {'inputs': inputs, 'supervision': targets}
