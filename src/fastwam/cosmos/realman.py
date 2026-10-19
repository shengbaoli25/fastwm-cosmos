"""LeRobot v3 RealMan cup data, using the inherited two-view/window recipe.

No relabeling of the old ForceVLA cache: source bytes, statistics, adapter,
camera order, absolute EEF actions and temporal offsets are bound separately.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torchvision.transforms import Resize

from fastwam.cache.latent import cache_identity_sha256, file_sha256
from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer

PROFILE = 'realman-1018-test-cup-force-v1'
EEF_COLUMNS = [8, 9, 10, 11, 12, 13, 7]
CAMERAS = ['observation.images.global', 'observation.images.wrist']


def require(ok, message):
    if not ok:
        raise ValueError(message)


class RealManSource:
    def __init__(self, root, expected_source=None):
        self.root = Path(root).resolve()
        files = {p.relative_to(self.root).as_posix(): file_sha256(p)
                 for p in sorted(self.root.rglob('*')) if p.is_file()}
        self.source_sha = cache_identity_sha256(files)
        require(expected_source is None or self.source_sha == expected_source,
                'Raw dataset inventory SHA differs')
        info = json.loads((self.root/'meta/info.json').read_text())
        force_path = self.root/'meta/force.json'
        force_meta = json.loads(force_path.read_text()) if force_path.is_file() else {}
        self.force_contract = force_meta.get('model_input')
        filtered = self.force_contract is not None
        if filtered:
            require(force_meta.get('schema') == 'realman_force_v4', 'Unknown force preprocessing metadata')
            require((info['codebase_version'], info['fps']) == ('v3.0', 30), 'Wrong RealMan dataset format/rate')
            require(self.force_contract.get('field') == 'observation.force_filtered', 'Wrong model force field')
        else:
            require((info['codebase_version'], info['total_episodes'], info['total_frames'], info['fps'])
                    == ('v3.0', 73, 35611, 30), 'Wrong legacy RealMan dataset scope')
        force_field = 'observation.force_filtered' if filtered else 'observation.force'
        episodes = []
        for p in sorted((self.root/'meta/episodes').rglob('*.parquet')):
            episodes.extend(pq.read_table(p).to_pylist())
        self.episodes = sorted(episodes, key=lambda x: x['episode_index'])
        require([e['episode_index'] for e in self.episodes] == list(range(info['total_episodes'])),
                'Missing/duplicate episodes')
        self.lengths = [int(e['length']) for e in self.episodes]
        require(sum(self.lengths) == info['total_frames'], 'Missing source rows')
        self.rows, raw, prompts, timing = [], [], [], []
        cursor = 0
        for ep, e in enumerate(self.episodes):
            if filtered:
                from realman_force.contract import validate_contract
                calibration = json.loads((self.root/'meta/force_sessions'/f'episode_{ep:06d}.json').read_text())
                validate_contract(self.force_contract, calibration)
            n = e['length']
            require((e['dataset_from_index'], e['dataset_to_index']) == (cursor, cursor+n),
                    'Episode global indices differ')
            p = self.root / info['data_path'].format(chunk_index=e['data/chunk_index'],
                                                    file_index=e['data/file_index'])
            t = pq.read_table(p).to_pydict()
            ids = [j for j, value in enumerate(t['episode_index']) if value == ep]
            require(len(ids) == n, 'Episode numeric length differs')
            t = {k: [v[j] for j in ids] for k, v in t.items()}
            require(t['frame_index'] == list(range(n)) and t['index'] == list(range(cursor,cursor+n)),
                    'Source frame order differs')
            require(np.allclose(t['timestamp'], np.arange(n)/30, atol=2e-6),
                    'Source timeline differs')
            state = np.asarray(t['observation.state'], dtype=np.float32)
            action = np.asarray(t['action'], dtype=np.float32)
            force = np.asarray(t[force_field], dtype=np.float32)
            require(state.shape == (n,20) and action.shape == (n,14) and force.shape == (n,6),
                    'Source field dimensions differ')
            require(all(np.isfinite(x).all() for x in (state, action, force)), 'Nonfinite numeric source')
            require(np.array_equal(state[:,14:20], force), 'State/wrench correspondence differs')
            require(all(((x[:,7] >= 0)&(x[:,7] <= 1)).all() for x in (state, action)),
                    'Gripper must be in [0,1]')
            ft = np.asarray(t['observation.force_timestamp_s'], dtype=np.float64).reshape(n)
            if filtered:
                require(np.array_equal(ft, np.asarray(t['observation.force_filter_timestamp_s']).reshape(n)),
                        'Filtered force timestamp differs from source force')
                grid = np.asarray(t['observation.monotonic_time_s'], dtype=np.float64).reshape(n)
                require(np.isfinite(grid).all() and np.all(np.diff(grid) > 0)
                        and np.all((grid-ft >= 0) & (grid-ft <= .03)),
                        'Force sample missing/stale/future on the 30 Hz grid')
            require(np.array_equal(ft, np.asarray(t['observation.robot_timestamp_s']).reshape(n)), 'Force/robot callback mismatch')
            require(np.all(np.diff(ft) >= 0), 'Nonmonotonic force samples')
            for key in ('action.timestamp_s','observation.gripper_timestamp_s',
                        'action.gripper_timestamp_s','observation.camera.arrival_time_s'):
                x = np.asarray(t[key]); bound = ft[:,None] if x.ndim == 2 else ft
                require(np.isfinite(x).all() and np.all(x <= bound+1e-6), 'Future or missing observation: '+key)
            require(len(e['tasks']) == 1 and bool(e['tasks'][0]), 'One recorded task per episode required')
            prompts.append(e['tasks'][0])
            raw.append((state[:,EEF_COLUMNS], action[:,EEF_COLUMNS], force))
            timing.append({'episode':ep, 'max_camera_age_s':float(np.max(t['observation.camera.age_s'])),
                           'max_pair_skew_s':float(np.max(t['observation.camera.pair_skew_s'])),
                           'max_action_age_s':float(np.max(t['observation.action_age_s']))})
            self.rows.extend((ep,f,cursor,n) for f in range(n)); cursor += n
        self.prompts = prompts
        self.raw = {k:torch.from_numpy(np.concatenate([r[j] for r in raw])).clone()
                    for j,k in enumerate(('state','action','force'))}
        # Population moments, all recorded training rows, separately per field.
        self.stats, self.normalized = {}, {}
        for k,x in self.raw.items():
            mean = x.double().mean(0).float()
            std = x.double().std(0, correction=0).float()
            require(bool(torch.isfinite(std).all() & (std > 0).all()), 'Degenerate '+k+' statistics')
            norm = SingleFieldLinearNormalizer({'mean':mean,'std':std}, mode='z-score')
            self.stats[k] = {'mean':mean.tolist(),'std':std.tolist()}
            self.normalized[k] = norm.forward(x)
        self.identity = dict(profile=PROFILE, source_inventory_sha256=self.source_sha, files=files,
            episode_lengths=self.lengths, fps=30, eef_columns=EEF_COLUMNS, cameras=CAMERAS,
            prompt_per_episode=prompts, statistics=self.stats, adapter_sha256=file_sha256(__file__),
            normalization='stock_FP32_zscore_scale_offset_eps1e-8_clip[-5,5]',
            actions='absolute_eef_xyz_m_euler_xyz_rad_gripper_open',
            wrench='recorded_world_frame_N_Nm_no_added_gravity_or_tare_correction',
            image_transform='RGB_uint8_to_FP32_div255_Resize224_bilinear_antialias_concat_horizontal_norm0.5',
            action_offsets=list(range(32)), force_offsets=list(range(1,33)),
            history_offsets=list(range(-31,1)), image_offsets=list(range(0,33,4)))
        self.identity_sha256 = cache_identity_sha256(self.identity)
        if filtered:
            self.identity.update(profile='realman-filtered-force-v1',
                                 wrench='calibrated_causal_Kalman_World_N_Nm',
                                 force_preprocessing=self.force_contract)
            self.identity_sha256 = cache_identity_sha256(self.identity)
        self.timing = timing
        self._video_episode = None
        self._frames = None
        self.info = info

    def export_normalization(self, path):
        """Ship field/processing identity with the statistics used by training."""
        payload = dict(profile=self.identity['profile'], statistics=self.stats,
                       source_identity_sha256=self.identity_sha256)
        if self.force_contract is not None:
            payload['force_preprocessing'] = self.force_contract
        Path(path).write_text(json.dumps(payload, indent=2)+'\n')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, j):
        ep, f, start, n = self.rows[j]
        def offsets(values):
            t = f + torch.tensor(values)
            return start+t.clamp(0,n-1), (t < 0)|(t >= n)
        a, ap = offsets(list(range(32)))
        h, hp = offsets(list(range(-31,1)))
        w, wp = offsets(list(range(1,33)))
        _, ip = offsets(list(range(0,33,4)))
        proprio = torch.cat((self.normalized['state'][a],torch.zeros(32,1)), -1)
        history = torch.cat((self.normalized['state'][h],self.normalized['force'][h]),-1)
        return dict(action=self.normalized['action'][a].clone(), proprio=proprio,
            history=history.masked_fill(hp[:,None],0),
            wrench_future=self.normalized['force'][w].clone().masked_fill(wp[:,None],0),
            raw_wrench_future64=self.raw['force'][w].double(),
            action_is_pad=ap, image_is_pad=ip, history_is_pad=hp, force_is_pad=wp,
            history_channel_valid=(~hp[:,None]).expand(-1,13).clone(),
            wrench_valid=(~wp[:,None]).expand(-1,6).clone(),
            sample_index=torch.tensor(j), episode_index=torch.tensor(ep), frame_index=torch.tensor(f),
            cosmos_text_index=torch.tensor(sorted(set(self.prompts)).index(self.prompts[ep])))

    def decode_episode(self, ep):
        import av
        e = self.episodes[ep]; views = []; resize = Resize([224,224], antialias=True)
        for camera in CAMERAS:
            key = 'videos/'+camera+'/'
            path = self.root/self.info['video_path'].format(video_key=camera,
                chunk_index=e[key+'chunk_index'], file_index=e[key+'file_index'])
            first = round(e[key+'from_timestamp']*30); n=e['length']; frames=[]
            with av.open(str(path)) as container:
                stream=container.streams.video[0]
                require(float(stream.average_rate)==30, 'Wrong camera rate')
                for idx,frame in enumerate(container.decode(video=0)):
                    if first <= idx < first+n:
                        require((frame.height,frame.width)==(480,640), 'Wrong source RGB size')
                        rgb=torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2,0,1).float()/255
                        frames.append(resize(rgb))
                    if idx >= first+n: break
            require(len(frames)==n,'Missing camera frames: '+str(path))
            views.append(torch.stack(frames))
        self._frames = (torch.cat(views,dim=-1)-.5)/.5
        self._video_episode = ep
        return self._frames

    def window(self,j):
        ep,f,_,n = self.rows[j]
        if self._video_episode != ep:
            self.decode_episode(ep)
        return self._frames[(f+torch.arange(0,33,4)).clamp(0,n-1)].permute(1,0,2,3).contiguous()
