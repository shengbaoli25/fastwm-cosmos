"""CPU integration tests for source lineage, causal row joins, and reference banks.

Run with unittest, NumPy and PyArrow only. The baseline segmentation function is
loaded from its AST so comparison does not import torch or the training stack.
"""
import ast
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fastwam.cosmos.reference_bank import (
    POSE_NAMES, ReferenceBank, atomic_json, check, continuous_segments_cpu,
    digest, file_digest, prepare_bank, verify_bank,
)
from fastwam.cosmos.reference_trajectory import AugmentationConfig
from realman_force.contract import input_contract


REPO = Path(__file__).resolve().parents[2]


def baseline_segmenter():
    path = REPO/'src/fastwam/cosmos/realman_joint.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'continuous_segments')
    namespace = {'np': np, 'require': check}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['continuous_segments']


def timing_table(n=64):
    grid = 1000+np.arange(n)/30
    return {
        'observation.monotonic_time_s': grid,
        'observation.force_timestamp_s': grid-.001,
        'observation.capture_fresh': np.ones((n, 4), dtype=np.int64),
        'action.applied': np.ones(n, dtype=np.int64),
        'observation.force_filter_input_count': np.full(n, 200),
        'observation.capture_discontinuity': np.zeros(n, dtype=np.int64),
        'observation.capture_gap_s': np.zeros(n),
        'observation.force_filter_epoch': np.zeros(n, dtype=np.int64),
        'action.follow_session_id': np.zeros(n, dtype=np.int64),
        'action.leader_anchor': np.zeros((n, 7)),
        'action.follower_anchor': np.zeros((n, 7)),
    }


def source_fixture(root, lengths=(64, 72)):
    """Real metadata/contract + small Parquet episodes; no model, video or GPU."""
    calibration = json.loads((REPO/'deploy/force_calibration.json').read_text(encoding='utf-8'))
    for directory in ('meta/episodes', 'meta/force_sessions', 'data'):
        (root/directory).mkdir(parents=True)
    names = [f'joint_{i}_rad' for i in range(7)]+POSE_NAMES+[
        'fx_N', 'fy_N', 'fz_N', 'tx_Nm', 'ty_Nm', 'tz_Nm', 'gripper_open']
    atomic_json(root/'meta/info.json', dict(
        codebase_version='v3.0', fps=30, total_episodes=len(lengths),
        total_frames=sum(lengths), data_path='data/episode_{file_index:06d}.parquet',
        features={'observation.state': {'names': names}}))
    atomic_json(root/'meta/force.json', dict(
        schema='realman_leader_force_gripper_v2', model_input=input_contract(calibration)))
    atomic_json(root/'meta/leader.json', dict(mode='execute', training_demonstration=True))
    cursor = 0
    episodes, tables = [], []
    for ep, n in enumerate(lengths):
        atomic_json(root/f'meta/force_sessions/episode_{ep:06d}.json', calibration)
        episodes.append(dict(episode_index=ep, length=n, dataset_from_index=cursor,
            dataset_to_index=cursor+n, tasks=['insert plug'],
            **{'data/chunk_index': 0, 'data/file_index': ep}))
        t = timing_table(n)
        t['observation.capture_fresh'][12, 0] = 0
        t['observation.capture_discontinuity'][32] = 1
        t['observation.force_filter_epoch'][40:] = 1
        t['action.follow_session_id'][50:] = 1
        rows = []
        for i in range(n):
            ft = float(t['observation.force_timestamp_s'][i])
            force = [float(i+j)/100 for j in range(6)]
            # Episode and frame can be recovered independently from TCP x/y.
            state = [i/100]*7+[ep+i*.002, i*.003, .4, 0., 0., i*.004]+force+[.8]
            row = {key: value[i].tolist() for key, value in t.items()}
            row.update(episode_index=ep, frame_index=i, index=cursor+i, timestamp=i/30,
                action=[i*.001]*7+[.8], **{
                    'observation.state': state, 'observation.force_filtered': force,
                    'observation.force_filter_timestamp_s': ft,
                    'observation.robot_timestamp_s': ft,
                    'action.timestamp_s': ft-.002,
                    'observation.leader_timestamp_s': ft-.003,
                    'observation.gripper_timestamp_s': ft-.004,
                    'action.gripper_timestamp_s': ft-.005,
                    'observation.camera.arrival_time_s': [ft-.002, ft-.001],
                })
            rows.append(row)
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, root/f'data/episode_{ep:06d}.parquet')
        tables.append(table.to_pydict())
        cursor += n
    pq.write_table(pa.Table.from_pylist(episodes), root/'meta/episodes/rows.parquet')
    return tables


class ContinuousSegmentsParityTests(unittest.TestCase):
    def test_random_faults_match_original_training_adapter(self):
        baseline = baseline_segmenter()
        for seed in range(80):
            rng = np.random.default_rng(seed)
            t = timing_table(96)
            # Cover freshness, missed commands, force lag/future, filter warmup,
            # clock jumps, explicit gaps, session changes and anchor resets.
            for row in rng.choice(96, 4, replace=False):
                t['observation.capture_fresh'][row, rng.integers(4)] = 0
            t['action.applied'][rng.choice(96, 3, replace=False)] = 0
            t['observation.force_timestamp_s'][rng.integers(96)] -= .04
            t['observation.force_timestamp_s'][rng.integers(96)] += .01
            t['observation.force_timestamp_s'][rng.integers(96)] = np.nan
            t['observation.force_filter_input_count'][rng.choice(96, 3, replace=False)] = 99
            t['observation.capture_discontinuity'][rng.integers(96)] = 1
            t['observation.capture_gap_s'][rng.integers(96)] = .001
            cut = int(rng.integers(1, 95))
            t['observation.monotonic_time_s'][cut:] += .1
            t['observation.force_timestamp_s'][cut:] += .1
            for key in ('observation.force_filter_epoch', 'action.follow_session_id',
                        'action.leader_anchor', 'action.follower_anchor'):
                t[key][rng.integers(1, 95):] += 1
            with self.subTest(seed=seed):
                self.assertEqual(continuous_segments_cpu(t), baseline(t))

    def test_single_valid_rows_are_not_training_anchors(self):
        t = timing_table(8)
        t['observation.capture_fresh'][[1, 3], 0] = 0
        self.assertEqual(continuous_segments_cpu(t), ([[4, 8]], 2))


class ReferenceBankTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.raw, self.bank = self.base/'raw', self.base/'bank'
        self.tables = source_fixture(self.raw)
        self.original_files = self.inventory()
        self.manifest = prepare_bank(self.raw, self.bank, expected_episodes=2, max_stride=4)

    def inventory(self):
        return {p.relative_to(self.raw).as_posix(): file_digest(p)
                for p in self.raw.rglob('*') if p.is_file()}

    def test_prepare_preserves_source_and_exact_adapter_anchor_inventory(self):
        self.assertEqual(self.inventory(), self.original_files)
        self.assertEqual(verify_bank(self.bank), self.manifest)
        self.assertEqual(self.manifest['total_source_rows'], 136)
        self.assertEqual(self.manifest['identity']['splits'], {'train': [0, 1], 'val': [], 'test': []})
        self.assertEqual(self.manifest['identity']['source_inventory_sha256'], digest(self.original_files))
        baseline = baseline_segmenter()
        for ep, table in enumerate(self.tables):
            expected_runs, invalid = baseline(table)
            entry = self.manifest['episodes'][ep]
            self.assertEqual(entry['segment_bounds'], expected_runs)
            self.assertEqual(entry['invalid_rows'], invalid)
            with np.load(self.bank/entry['path'], allow_pickle=False) as data:
                expected_frames = [j for lo, hi in expected_runs for j in range(lo, hi)]
                self.assertEqual(data['valid_anchor_frame'].tolist(), expected_frames)
                np.testing.assert_allclose(data['observation_tcp'][:, :3],
                    np.asarray(table['observation.state'])[expected_frames, 7:10])
                np.testing.assert_allclose(data['source_timestamp_s'],
                    np.asarray(table['timestamp'])[data['source_frame']])

    def test_episode_raw_frame_join_excludes_gap_and_only_exposes_current_tcp(self):
        bank = ReferenceBank(self.bank)
        cfg = replace(AugmentationConfig(), dropout_probability=0)
        for ep, frame in ((0, 13), (1, 31), (0, 55)):
            sample = bank.sample(ep, frame, config=cfg)
            inputs, targets = sample['inputs'], sample['supervision']
            self.assertEqual(inputs['current_tcp'].shape, (8,))
            np.testing.assert_allclose(inputs['current_tcp'][:3], [ep+frame*.002, frame*.003, .4])
            self.assertEqual(set(inputs), {'reference_pose', 'reference_valid',
                'reference_channel_valid', 'reference_order', 'reference_break_before',
                'reference_present', 'current_tcp'})
            self.assertTrue({'source_frame', 'segment_id', 'segment_bounds',
                'alignment_target'}.issubset(targets))
            self.assertAlmostEqual(float(targets['alignment_target'].sum()), 1., places=6)
        # Filtered cache sample 12 is raw source frame 13. It must not silently
        # use retained-anchor position as though it were a raw frame index.
        with self.assertRaisesRegex(ValueError, 'eligible source anchor'):
            bank.sample(0, 12)
        with self.assertRaisesRegex(ValueError, 'Unknown reference episode'):
            bank.sample(2, 13)

    def test_plan_stable_across_anchors_but_varies_across_epoch(self):
        bank = ReferenceBank(self.bank)
        cfg = replace(AugmentationConfig(), dropout_probability=0)
        a = bank.sample(0, 13, epoch=3, seed=17, config=cfg)
        b = bank.sample(0, 18, epoch=3, seed=17, config=cfg)
        replay = ReferenceBank(self.bank).sample(0, 13, epoch=3, seed=17, config=cfg)
        for key in a['inputs']:
            np.testing.assert_array_equal(a['inputs'][key], replay['inputs'][key])
            if key != 'current_tcp':
                np.testing.assert_array_equal(a['inputs'][key], b['inputs'][key])
        self.assertFalse(np.array_equal(a['supervision']['alignment_target'],
                                       b['supervision']['alignment_target']))
        lengths = {len(bank.sample(0, 13, epoch=e, seed=17, config=cfg)['inputs']['reference_pose'])
                   for e in range(12)}
        self.assertGreater(len(lengths), 1)

    def test_lookup_rejects_noninteger_indices_and_negative_rng_inputs(self):
        bank = ReferenceBank(self.bank)
        for kwargs in ({'episode_index': .1}, {'episode_index': True},
                       {'frame_index': 13.7}, {'frame_index': '13'},
                       {'epoch': -1}, {'epoch': 2.5}, {'seed': -1}, {'seed': False}):
            args = dict(episode_index=0, frame_index=13)
            args.update(kwargs)
            with self.subTest(args=args), self.assertRaises((ValueError, TypeError)):
                bank.sample(**args)

    def test_preparation_rejects_future_and_misaligned_source_timestamps(self):
        path = self.raw/'data/episode_000000.parquet'
        pristine = pq.read_table(path).to_pydict()
        for key in ('observation.force_filter_timestamp_s', 'observation.robot_timestamp_s',
                    'action.timestamp_s', 'observation.leader_timestamp_s',
                    'observation.gripper_timestamp_s', 'action.gripper_timestamp_s',
                    'observation.camera.arrival_time_s'):
            data = {k: list(v) for k, v in pristine.items()}
            late = data['observation.force_timestamp_s'][17]+.1
            data[key][17] = [late, late] if key.endswith('arrival_time_s') else late
            pq.write_table(pa.Table.from_pydict(data), path)
            out = self.base/('bad-timestamps-'+key.replace('.', '-'))
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare_bank(self.raw, out, expected_episodes=2)

    def test_full_dropout_disables_alignment_without_hiding_current_state(self):
        sample = ReferenceBank(self.bank).sample(1, 17,
            config=replace(AugmentationConfig(), dropout_probability=1))
        self.assertFalse(sample['inputs']['reference_present'])
        self.assertFalse(sample['inputs']['reference_valid'].any())
        self.assertFalse(sample['supervision']['alignment_enabled'])
        np.testing.assert_allclose(sample['inputs']['current_tcp'][:3], [1.034, .051, .4])

    def test_corrupt_payload_is_detected_before_sampling(self):
        path = self.bank/self.manifest['episodes'][0]['path']
        with path.open('ab') as stream:
            stream.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            ReferenceBank(self.bank)

    def test_missing_payload_is_detected(self):
        (self.bank/self.manifest['episodes'][0]['path']).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing/escaping source'):
            verify_bank(self.bank)

    def test_manifest_identity_change_is_detected(self):
        self.manifest['identity']['coordinate_frame'] = 'unverified-new-frame'
        atomic_json(self.bank/'manifest.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            verify_bank(self.bank)

    def test_extra_files_rejected(self):
        (self.bank/'untracked-labels.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Unexpected/partial files'):
            verify_bank(self.bank)

    def test_raw_and_output_cannot_overlap_and_output_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, 'outside the immutable'):
            prepare_bank(self.raw, self.raw/'derived', expected_episodes=2)
        with self.assertRaisesRegex(ValueError, 'Output already exists'):
            prepare_bank(self.raw, self.bank, expected_episodes=2)
        self.assertEqual(self.inventory(), self.original_files)

    def test_expected_source_fails_before_creating_bank(self):
        out = self.base/'wrong-source'
        with self.assertRaisesRegex(ValueError, 'Raw inventory'):
            prepare_bank(self.raw, out, expected_episodes=2, expected_source_sha='0'*64)
        self.assertFalse(out.exists())

    def test_complete_source_inventory_and_explicit_split(self):
        split = self.base/'split.json'
        atomic_json(split, {'train': [0], 'val': [1], 'test': []})
        manifest = prepare_bank(self.raw, self.base/'bound-bank', expected_episodes=2,
            expected_source_sha=digest(self.original_files), split_json=split)
        self.assertEqual(manifest['identity']['source_inventory_scope'], 'complete_raw_inventory')
        self.assertEqual(manifest['identity']['splits']['val'], [1])
        self.assertEqual(manifest['identity']['split_scope'], 'membership_only_no_normalization_recomputed')


if __name__ == '__main__':
    unittest.main()
