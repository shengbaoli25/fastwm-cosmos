"""CPU-only regression tests; run with unittest and NumPy, no GPU packages."""
import unittest

import numpy as np

from fastwam.cosmos.reference_trajectory import (
    AugmentationConfig, alignment_target, augment_reference, build_reference,
    euler_xyz_to_quaternion, quaternion_distance, quaternion_slerp,
)


def trajectory(n=100):
    state = np.zeros((n, 20), dtype=np.float64)
    state[:, 7] = np.arange(n)*.002
    state[:, 19] = .8
    return state


class ReferenceTrajectoryTests(unittest.TestCase):
    def test_sparse_linear_curve_preserves_endpoints_and_stride(self):
        state = trajectory()
        ref = build_reference(state, [[0, 100]], max_stride=12)
        self.assertEqual(ref['source_frame'][0], 0)
        self.assertEqual(ref['source_frame'][-1], 99)
        self.assertLess(len(ref['source_frame']), 20)
        self.assertLessEqual(np.diff(ref['source_frame']).max(), 12)
        np.testing.assert_allclose(ref['reference_pose'][:, :3], state[ref['source_frame'], 7:10])

    def test_curved_path_has_bounded_interpolation_error(self):
        state = trajectory()
        state[:, 8] = np.sin(np.linspace(0, 6, 100))*.08
        state[:, 12] = np.sin(np.linspace(0, 4, 100))*.8
        ref = build_reference(state, [[0, 100]], max_stride=20,
                              position_tolerance_m=.003, rotation_tolerance_rad=.025)
        q = euler_xyz_to_quaternion(state[:, 10:13])
        for left, right in zip(ref['source_frame'][:-1], ref['source_frame'][1:]):
            idx = np.arange(left, right+1)
            frac = (idx-left)/(right-left)
            pos = state[left, 7:10]*(1-frac[:, None])+state[right, 7:10]*frac[:, None]
            self.assertLessEqual(np.linalg.norm(pos-state[idx, 7:10], axis=1).max(), .00300001)
            rot = quaternion_slerp(q[left], q[right], frac)
            self.assertLessEqual(quaternion_distance(rot, q[idx]).max(), .02500001)

    def test_no_interpolation_or_alignment_across_gap(self):
        state = trajectory(60)
        state[30:, 7] += 5
        ref = build_reference(state, [[0, 20], [30, 60]], max_stride=50)
        np.testing.assert_array_equal(ref['source_frame'], [0, 19, 30, 59])
        np.testing.assert_array_equal(ref['segment_id'], [0, 0, 1, 1])
        target = alignment_target(ref['source_frame'], ref['segment_id'], ref['segment_bounds'], 35)
        np.testing.assert_allclose(target, [0, 0, 24/29, 5/29])
        with self.assertRaisesRegex(ValueError, 'outside valid'):
            alignment_target(ref['source_frame'], ref['segment_id'], ref['segment_bounds'], 25)

    def test_adjacent_reset_segments_remain_separate(self):
        ref = build_reference(trajectory(40), [[0, 20], [20, 40]], max_stride=50)
        np.testing.assert_array_equal(ref['source_frame'], [0, 19, 20, 39])
        for frame, selected in [(19, 1), (20, 2)]:
            target = alignment_target(ref['source_frame'], ref['segment_id'], ref['segment_bounds'], frame)
            self.assertEqual(np.argmax(target), selected)
            self.assertEqual(target.sum(), 1)

    def test_euler_wrap_is_short_arc_and_unit_quaternion(self):
        state = trajectory(30)
        state[:, 12] = (np.linspace(3.05, 3.25, 30)+np.pi)%(2*np.pi)-np.pi
        ref = build_reference(state, [[0, 30]], max_stride=50)
        self.assertEqual(len(ref['source_frame']), 2)
        q = ref['reference_pose'][:, 3:7]
        np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=1e-6)
        self.assertGreater(np.dot(q[0], q[1]), 0)
        self.assertAlmostEqual(float(quaternion_distance(q[0], q[1])), .2, places=6)
        halfway = quaternion_slerp(q[0], q[1], [.5])[0]
        expected = euler_xyz_to_quaternion([[0, 0, 3.15]])[0]
        self.assertLess(float(quaternion_distance(halfway, expected)), 1e-6)

    def test_gripper_event_both_sides_retained(self):
        state = trajectory(60)
        state[25:, 19] = .1
        state[40:, 19] = .9
        ref = build_reference(state, [[0, 60]], max_stride=100)
        self.assertTrue({24, 25, 39, 40}.issubset(set(ref['source_frame'])))
        np.testing.assert_allclose(ref['reference_pose'][:, 7], state[ref['source_frame'], 19])

    def test_bad_segments_states_and_tolerances_rejected(self):
        for segments in ([[0, 1]], [[0, 20], [19, 30]], [[-1, 10]], [[0., 20.]]):
            with self.assertRaises(ValueError):
                build_reference(trajectory(30), segments)
        with self.assertRaises(ValueError):
            build_reference(trajectory(), [[0, 100]], position_tolerance_m=0)
        with self.assertRaises(ValueError):
            build_reference(trajectory(), [[0, 100]], max_stride=1.5)
        state = trajectory()
        state[0, 19] = 2
        with self.assertRaises(ValueError):
            build_reference(state, [[0, 100]])

    def test_augmentation_is_seeded_and_keeps_supervision_separate(self):
        ref = build_reference(trajectory(), [[0, 100]])
        pristine = {key:value.copy() for key, value in ref.items()}
        cfg = AugmentationConfig(point_mask_probability=0, span_mask_probability=0, dropout_probability=0,
                                 min_keep_fraction=1, max_keep_fraction=1, duplicate_probability=0)
        a, sup = augment_reference(ref, np.random.default_rng(42), config=cfg, anchor_frame=40)
        b, _ = augment_reference(ref, np.random.default_rng(42), config=cfg, anchor_frame=40)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        for key in ref:
            np.testing.assert_array_equal(ref[key], pristine[key])
        self.assertNotIn('source_frame', a)
        self.assertNotIn('alignment_target', a)
        self.assertNotIn('alignment_target_unmasked', a)
        self.assertNotIn('reference_position_error', a)
        self.assertGreater(sup['reference_position_error'].max(), 0)
        np.testing.assert_allclose(a['reference_pose'][[0, -1]], ref['reference_pose'][[0, -1]], atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(a['reference_pose'][:, 3:7], axis=1), 1, atol=1e-6)
        expected = alignment_target(ref['source_frame'], ref['segment_id'], ref['segment_bounds'], 40)
        np.testing.assert_array_equal(sup['alignment_target'], expected)

    def test_full_dropout_masks_values_and_disables_alignment(self):
        ref = build_reference(trajectory(), [[0, 100]])
        cfg = AugmentationConfig(dropout_probability=1)
        inp, sup = augment_reference(ref, np.random.default_rng(9), config=cfg, anchor_frame=20)
        self.assertFalse(inp['reference_present'])
        self.assertFalse(inp['reference_valid'].any())
        self.assertFalse(inp['reference_channel_valid'].any())
        self.assertFalse(sup['alignment_enabled'])
        self.assertFalse(sup['alignment_target'].any())
        self.assertAlmostEqual(float(sup['alignment_target_unmasked'].sum()), 1)
        np.testing.assert_array_equal(inp['reference_pose'][:, :3], 0)
        np.testing.assert_array_equal(inp['reference_pose'][:, 3:7],
                                      np.tile([0, 0, 0, 1], (len(inp['reference_pose']), 1)))

    def test_mask_slots_keep_provenance_and_endpoints(self):
        ref = build_reference(trajectory(), [[0, 40], [50, 100]])
        cfg = AugmentationConfig(point_mask_probability=1, span_mask_probability=1, dropout_probability=0,
                                 min_keep_fraction=1, max_keep_fraction=1, duplicate_probability=0)
        inp, sup = augment_reference(ref, np.random.default_rng(3), config=cfg, anchor_frame=20)
        endpoints = np.isin(ref['source_frame'], [0, 39, 50, 99])
        np.testing.assert_array_equal(inp['reference_valid'], endpoints)
        np.testing.assert_array_equal(sup['source_frame'], ref['source_frame'])
        self.assertEqual(inp['reference_break_before'].sum(), 2)
        self.assertEqual(len(sup['alignment_target']), len(ref['source_frame']))
        self.assertFalse(sup['alignment_target'][~inp['reference_valid']].any())
        target = sup['alignment_target']
        self.assertAlmostEqual(float(target[ref['source_frame'] == 0][0]), 19/39, places=6)
        self.assertAlmostEqual(float(target[ref['source_frame'] == 39][0]), 20/39, places=6)
        self.assertFalse(target[ref['segment_id'] == 1].any())

    def test_random_reference_length_and_sampling_pace(self):
        state = trajectory(300)
        state[150:, 19] = .1
        ref = build_reference(state, [[0, 120], [130, 300]], max_stride=6)
        lengths, intervals = [], set()
        cfg = AugmentationConfig(position_std_m=0, rotation_std_rad=0,
                                 point_mask_probability=0, span_mask_probability=0,
                                 dropout_probability=0)
        for seed in range(12):
            inp, sup = augment_reference(ref, np.random.default_rng(seed), config=cfg, anchor_frame=170)
            lengths.append(len(inp['reference_pose']))
            intervals.update(np.diff(sup['source_frame']).tolist())
            self.assertTrue({0, 119, 130, 149, 150, 299}.issubset(set(sup['source_frame'])))
            self.assertTrue((np.diff(sup['source_frame']) >= 0).all())
            np.testing.assert_allclose(sup['alignment_target'].sum(), 1, atol=1e-7)
            self.assertFalse(sup['alignment_target'][sup['segment_id'] == 0].any())
            self.assertNotIn('source_frame', inp)
        self.assertGreater(len(set(lengths)), 1)
        self.assertIn(0, intervals)  # repeated hold slots
        self.assertGreater(len(intervals), 2)  # nonuniform time spacing

    def test_repeated_reference_positions_share_alignment_mass(self):
        frames = np.array([0, 5, 5, 10, 19])
        ids = np.zeros(5, dtype=np.int64)
        bounds = np.array([[0, 20]])
        np.testing.assert_allclose(alignment_target(frames, ids, bounds, 5), [0,.5,.5,0,0])
        np.testing.assert_allclose(alignment_target(frames, ids, bounds, 7), [0,.3,.3,.4,0])

    def test_masked_alignment_skips_invisible_neighbours_and_duplicate_slots(self):
        frames = np.array([0, 5, 5, 10, 19])
        ids = np.zeros(5, dtype=np.int64)
        bounds = np.array([[0, 20]])
        valid = np.array([True, False, False, True, True])
        # The original exact match is hidden; interpolate between visible 0/10.
        np.testing.assert_allclose(alignment_target(frames, ids, bounds, 5, valid_mask=valid),
                                   [.5, 0, 0, .5, 0])
        valid = np.array([True, False, True, True, True])
        # Only one alias of source frame 5 is visible; it receives all its mass.
        np.testing.assert_allclose(alignment_target(frames, ids, bounds, 7, valid_mask=valid),
                                   [0, 0, .6, .4, 0])
        valid = np.array([True, True, True, False, True])
        np.testing.assert_allclose(alignment_target(frames, ids, bounds, 12, valid_mask=valid),
                                   [0, .25, .25, 0, .5])

    def test_masked_alignment_clamps_only_within_anchor_segment(self):
        frames = np.array([0, 5, 10, 19, 30, 39])
        ids = np.array([0, 0, 0, 0, 1, 1])
        bounds = np.array([[0, 20], [30, 40]])
        valid = np.array([False, True, True, False, True, True])
        np.testing.assert_array_equal(alignment_target(frames, ids, bounds, 0, valid_mask=valid),
                                      [0, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(alignment_target(frames, ids, bounds, 19, valid_mask=valid),
                                      [0, 0, 1, 0, 0, 0])
        valid[:4] = False
        # Other segments still have keys, but cannot receive this target.
        self.assertFalse(alignment_target(frames, ids, bounds, 10, valid_mask=valid).any())
        self.assertAlmostEqual(float(alignment_target(frames, ids, bounds, 35, valid_mask=valid).sum()), 1)
        with self.assertRaisesRegex(ValueError, 'Boolean'):
            alignment_target(frames, ids, bounds, 10, valid_mask=valid.astype(int))

    def test_augmented_alignment_only_has_mass_on_accessible_keys(self):
        ref = build_reference(trajectory(160), [[0, 60], [70, 160]], max_stride=4)
        cfg = AugmentationConfig(point_mask_probability=.7, span_mask_probability=.8,
                                 dropout_probability=.2, preserve_segment_endpoints=False)
        for seed in range(30):
            inp, sup = augment_reference(ref, np.random.default_rng(seed), config=cfg, anchor_frame=100)
            target = sup['alignment_target']
            self.assertFalse(target[~inp['reference_valid']].any())
            self.assertFalse(target[sup['segment_id'] == 0].any())
            self.assertNotIn('alignment_target_unmasked', inp)
            self.assertNotIn('source_frame', inp)
            self.assertEqual(bool(sup['alignment_enabled']), bool(target.any()))
            self.assertAlmostEqual(float(target.sum()), 1 if sup['alignment_enabled'] else 0, places=6)

    def test_length_change_does_not_change_unmasked_gripper_targets(self):
        state = trajectory(100)
        state[45:, 19] = .1
        ref = build_reference(state, [[0, 100]], max_stride=5)
        cfg = AugmentationConfig(position_std_m=0, rotation_std_rad=0,
                                 point_mask_probability=0, span_mask_probability=0,
                                 dropout_probability=0, min_keep_fraction=.1,
                                 max_keep_fraction=.1, duplicate_probability=.8)
        inp, sup = augment_reference(ref, np.random.default_rng(7), config=cfg)
        self.assertTrue({44, 45}.issubset(set(sup['source_frame'])))
        np.testing.assert_allclose(inp['reference_pose'][:, 7], state[sup['source_frame'], 19])


if __name__ == '__main__':
    unittest.main()
