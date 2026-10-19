"""Joint-target demonstrations; preserve source rows and cut every discontinuity."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from fastwam.cache.latent import cache_identity_sha256, file_sha256
from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
from .realman import RealManSource, CAMERAS, require

PROFILE = 'realman-leader-joint8-filtered-v1'
STATE_COLUMNS = [0, 1, 2, 3, 4, 5, 6, 19]
ACTION_UNITS = ['rad'] * 7 + ['openness_0_1']


def continuous_segments(t):
    """Return [start, stop) valid runs in original episode/video row indices."""
    grid = np.asarray(t['observation.monotonic_time_s'], dtype=np.float64).reshape(-1)
    n = len(grid)
    require(np.isfinite(grid).all(), 'Invalid capture clock')
    ft = np.asarray(t['observation.force_timestamp_s'], dtype=np.float64).reshape(n)
    fresh = np.asarray(t['observation.capture_fresh'])
    require(fresh.shape == (n, 4), 'Missing four-source freshness flags')
    valid = (fresh == 1).all(1) & (np.asarray(t['action.applied']).reshape(n) == 1)
    valid &= np.isfinite(ft) & (grid-ft >= -1e-6) & (grid-ft <= .03)
    valid &= np.asarray(t['observation.force_filter_input_count']).reshape(n) >= 100
    # A row index is a video index, not proof of uninterrupted wall time.
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
    runs = []
    start = None
    for j in range(n):
        if start is not None and (breaks[j] or not valid[j]):
            runs.append([start, j]); start = None
        if valid[j] and start is None:
            start = j
    if start is not None:
        runs.append([start, n])
    # One-row runs cannot supervise any future video/force; don't train on them.
    return [r for r in runs if r[1]-r[0] >= 2], int((~valid).sum())


class RealManJointSource(RealManSource):
    def __init__(self, root, expected_source=None):
        self.root = Path(root).resolve()
        files = {p.relative_to(self.root).as_posix(): file_sha256(p)
                 for p in sorted(self.root.rglob('*')) if p.is_file()}
        self.source_sha = cache_identity_sha256(files)
        require(expected_source is None or expected_source == self.source_sha, 'Raw dataset inventory SHA differs')
        info = json.loads((self.root/'meta/info.json').read_text())
        force_meta = json.loads((self.root/'meta/force.json').read_text())
        leader = json.loads((self.root/'meta/leader.json').read_text())
        require(force_meta['schema'] == 'realman_leader_force_gripper_v2', 'Wrong joint source schema')
        require((info['codebase_version'], info['fps']) == ('v3.0', 30), 'Wrong joint source format/rate')
        require(leader.get('training_demonstration') is True and leader.get('mode') == 'execute',
                'Only executed demonstrations are training targets')
        from realman_force.contract import validate_contract
        self.force_contract = force_meta['model_input']
        self.episodes = sorted([e for p in sorted((self.root/'meta/episodes').rglob('*.parquet'))
                                for e in pq.read_table(p).to_pylist()], key=lambda e:e['episode_index'])
        require([e['episode_index'] for e in self.episodes] == list(range(info['total_episodes'])), 'Missing episodes')
        self.lengths = [int(e['length']) for e in self.episodes]
        require(sum(self.lengths) == info['total_frames'], 'Missing source rows')
        self.rows, self.segments, self.timing, self.prompts = [], [], [], []
        self.bounds = []
        arrays = []; cursor = 0
        for ep, e in enumerate(self.episodes):
            calibration = json.loads((self.root/'meta/force_sessions'/f'episode_{ep:06d}.json').read_text())
            validate_contract(self.force_contract, calibration)
            n = e['length']
            require((e['dataset_from_index'], e['dataset_to_index']) == (cursor, cursor+n), 'Global indices differ')
            p = self.root/info['data_path'].format(chunk_index=e['data/chunk_index'], file_index=e['data/file_index'])
            table = pq.read_table(p).to_pydict()
            ids = [j for j, value in enumerate(table['episode_index']) if value == ep]
            t = {k:[v[j] for j in ids] for k,v in table.items()}
            require(len(ids) == n and t['frame_index'] == list(range(n))
                    and t['index'] == list(range(cursor,cursor+n)), 'Source row order differs')
            require(np.allclose(t['timestamp'], np.arange(n)/30, atol=2e-6), 'Video index timeline differs')
            state = np.asarray(t['observation.state'], dtype=np.float32)
            action = np.asarray(t['action'], dtype=np.float32)
            force = np.asarray(t['observation.force_filtered'], dtype=np.float32)
            require(state.shape == (n,20) and action.shape == (n,8) and force.shape == (n,6), 'Joint field dimensions differ')
            require(all(np.isfinite(x).all() for x in (state,action,force)), 'Nonfinite source')
            require(np.array_equal(state[:,13:19],force), 'State/filtered wrench correspondence differs')
            require(all(((x >= 0)&(x <= 1)).all() for x in (state[:,19],action[:,7])), 'Wrong gripper units')
            ft = np.asarray(t['observation.force_timestamp_s']).reshape(n)
            for key in ('observation.force_filter_timestamp_s','observation.robot_timestamp_s'):
                require(np.array_equal(ft,np.asarray(t[key]).reshape(n)), 'Force/pose/filter timestamps differ')
            for key in ('action.timestamp_s','observation.leader_timestamp_s', 'observation.gripper_timestamp_s',
                        'action.gripper_timestamp_s','observation.camera.arrival_time_s'):
                value = np.asarray(t[key]); bound = ft[:,None] if value.ndim == 2 else ft
                require(np.isfinite(value).all() and (value <= bound+1e-6).all(), 'Future input: '+key)
            runs, invalid = continuous_segments(t)
            for lo,hi in runs:
                self.segments.append([ep,lo,hi])
                for frame in range(lo,hi):
                    self.rows.append((ep,frame,cursor,n));self.bounds.append((lo,hi))
            require(len(e['tasks']) == 1 and bool(e['tasks'][0]), 'One task per episode required')
            self.prompts.append(e['tasks'][0])
            arrays.append((state[:,STATE_COLUMNS],action,force))
            self.timing.append(dict(episode=ep, source_rows=n, invalid_rows=invalid,
                                    retained_rows=sum(hi-lo for lo,hi in runs),segments=runs))
            cursor += n
        require(bool(self.rows), 'No continuous valid training anchors')
        self.raw = {key:torch.from_numpy(np.concatenate([a[j] for a in arrays])).clone()
                    for j,key in enumerate(('state','action','force'))}
        selected = torch.tensor([start+f for _,f,start,_ in self.rows])
        self.stats, self.normalized = {}, {}
        for key,x in self.raw.items():
            mean = x[selected].double().mean(0).float()
            std = x[selected].double().std(0,correction=0).float()
            require(bool(torch.isfinite(std).all() & (std > 0).all()), 'Degenerate '+key+' statistics')
            self.stats[key] = dict(mean=mean.tolist(),std=std.tolist())
            self.normalized[key] = SingleFieldLinearNormalizer(dict(mean=mean,std=std),mode='z-score').forward(x)
        self.identity = dict(profile=PROFILE, source_schema=force_meta['schema'], action_space='joint8',
            state_columns=STATE_COLUMNS, action_columns=list(range(8)), action_units=ACTION_UNITS,
            state_dim=8, action_dim=8, history_dim=14, source_inventory_sha256=self.source_sha, files=files,
            episode_lengths=self.lengths, fps=30, cameras=CAMERAS, prompt_per_episode=self.prompts,
            statistics=self.stats, adapter_sha256=file_sha256(__file__),
            video_adapter_sha256=file_sha256(Path(__file__).with_name('realman.py')),
            normalization='stock_FP32_zscore_scale_offset_eps1e-8_clip[-5,5]',
            force_preprocessing=self.force_contract, segments=self.segments,
            temporal_policy='valid_capture_fresh4_applied_warm_force;split_gap20us_reset_reference;min_segment2;clamp_mask',
            actions='absolute_sent_joint_targets_rad_then_gripper_open',
            wrench='recorded_calibrated_causal_filtered_world_N_Nm_no_refilter',
            action_offsets=list(range(32)),force_offsets=list(range(1,33)),
            history_offsets=list(range(-31,1)),image_offsets=list(range(0,33,4)))
        self.identity_sha256 = cache_identity_sha256(self.identity)
        self.info = info; self._video_episode = None; self._frames = None

    def export_normalization(self,path):
        payload = dict(profile=PROFILE, statistics=self.stats, source_identity_sha256=self.identity_sha256,
                       force_preprocessing=self.force_contract,action_space='joint8',action_units=ACTION_UNITS,
                       state_columns=STATE_COLUMNS)
        Path(path).write_text(json.dumps(payload,indent=2)+'\n')

    def __getitem__(self,j):
        ep,f,start,_ = self.rows[j];lo,hi = self.bounds[j]
        def offsets(values):
            t = f+torch.tensor(values)
            return start+t.clamp(lo,hi-1),(t < lo)|(t >= hi)
        a,ap = offsets(range(32));h,hp = offsets(range(-31,1));w,wp = offsets(range(1,33));_,ip = offsets(range(0,33,4))
        return dict(action=self.normalized['action'][a].clone(),
            proprio=torch.cat((self.normalized['state'][a],torch.zeros(32,1)),-1),
            history=torch.cat((self.normalized['state'][h],self.normalized['force'][h]),-1).masked_fill(hp[:,None],0),
            wrench_future=self.normalized['force'][w].clone().masked_fill(wp[:,None],0),
            raw_wrench_future64=self.raw['force'][w].double(),
            action_is_pad=ap,image_is_pad=ip,history_is_pad=hp,force_is_pad=wp,
            history_channel_valid=(~hp[:,None]).expand(-1,14).clone(),wrench_valid=(~wp[:,None]).expand(-1,6).clone(),
            sample_index=torch.tensor(j),episode_index=torch.tensor(ep),frame_index=torch.tensor(f),
            cosmos_text_index=torch.tensor(sorted(set(self.prompts)).index(self.prompts[ep])))

    def window(self,j):
        ep,f,_,_ = self.rows[j];lo,hi = self.bounds[j]
        if self._video_episode != ep:
            self.decode_episode(ep)
        return self._frames[(f+torch.arange(0,33,4)).clamp(lo,hi-1)].permute(1,0,2,3).contiguous()


def open_realman_source(root,expected_source=None):
    metadata = json.loads((Path(root)/'meta/force.json').read_text()) if (Path(root)/'meta/force.json').is_file() else {}
    source = RealManJointSource if metadata.get('schema') == 'realman_leader_force_gripper_v2' else RealManSource
    return source(root,expected_source)
