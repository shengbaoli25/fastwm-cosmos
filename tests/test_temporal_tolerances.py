"""Runtime timing envelopes, independent from training/calibration identities."""
import copy
import math
import unittest

from realman_sync_client.wire import decode_observation, encode_observation, temporal_tolerances


RELAXED = dict(force_max_age_s=.10, camera_max_age_s=.25, camera_pair_skew_s=.15)


def fixture():
    metadata = dict(action_space='joint8', camera_names=['global', 'wrist'], width=2,
                    height=1, force_preprocessing={'id': 'unchanged-trained-calibration'})
    observation = encode_observation(dict(sequence=1, stamp=10., state=[0.]*7+[.5],
        force=[0.]*6, action_space='joint8',
        force_meta=dict(preprocessing_id='unchanged-trained-calibration', epoch=4, source_time=9.99),
        frames={name:dict(data=bytes(6), width=2, height=1) for name in metadata['camera_names']},
        frame_meta={name:dict(received=9.98) for name in metadata['camera_names']}))
    return metadata, observation


class TemporalToleranceTests(unittest.TestCase):
    def test_absent_declaration_keeps_strict_defaults(self):
        self.assertEqual(temporal_tolerances({}),
            dict(force_max_age_s=.03, camera_max_age_s=.12, camera_pair_skew_s=.05))
        meta, obs = fixture()
        self.assertEqual(decode_observation(obs, meta)['state'], obs['state'])

    def test_old_force_sample_only_passes_explicit_server_envelope(self):
        meta, obs = fixture()
        obs['force_meta']['source_time'] = 9.92
        for frame in obs['frame_meta'].values():
            frame['received'] = 9.91
        with self.assertRaisesRegex(ValueError, 'Force source is stale'):
            decode_observation(obs, meta)
        relaxed = dict(meta, temporal_tolerances=RELAXED)
        decoded = decode_observation(obs, relaxed)
        self.assertEqual(decoded['force_meta'], obs['force_meta'])
        self.assertEqual(relaxed['force_preprocessing'], meta['force_preprocessing'])

    def test_150ms_causal_camera_offset_passes_relaxed_only(self):
        meta, obs = fixture()
        obs['frame_meta']['global']['received'] = 9.99
        obs['frame_meta']['wrist']['received'] = 9.84
        with self.assertRaisesRegex(ValueError, 'misaligned'):
            decode_observation(obs, meta)
        decoded = decode_observation(obs, dict(meta, temporal_tolerances=RELAXED))
        self.assertEqual(decoded['frame_meta'], obs['frame_meta'])

    def test_relaxed_still_rejects_expired_and_future_data(self):
        for mutation in ('force_expired', 'force_future', 'camera_expired', 'camera_future'):
            with self.subTest(mutation=mutation):
                meta, obs = fixture()
                meta['temporal_tolerances'] = RELAXED
                if mutation.startswith('force'):
                    obs['force_meta']['source_time'] = 9.899 if mutation.endswith('expired') else 10.001
                else:
                    obs['frame_meta']['global']['received'] = 9.70 if mutation.endswith('expired') else 9.991
                with self.assertRaises(ValueError):
                    decode_observation(obs, meta)

    def test_request_cannot_raise_its_own_envelope_or_change_identity(self):
        meta, obs = fixture()
        obs['temporal_tolerances'] = RELAXED
        obs['frame_meta']['wrist']['received'] = 9.90
        with self.assertRaisesRegex(ValueError, 'misaligned'):
            decode_observation(obs, meta)
        obs['force_meta']['preprocessing_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'preprocessing differs'):
            decode_observation(obs, dict(meta, temporal_tolerances=RELAXED))

    def test_invalid_declarations_are_rejected(self):
        for value in (None, [], 'relaxed', {'unknown': 1}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                temporal_tolerances({'temporal_tolerances': value})
        for name, cap in (('force_max_age_s', .1), ('camera_max_age_s', .5), ('camera_pair_skew_s', .2)):
            for value in (0, -.1, True, '0.1', math.nan, math.inf, cap+.0001):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    temporal_tolerances({'temporal_tolerances': {name: value}})


if __name__ == '__main__':
    unittest.main()
