"""CPU-only bootstrap and exact v2 contract regression checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_runtime as bootstrap
sys.path[:0] = [str(bootstrap.ROOT / 'code/deploy'), str(bootstrap.ROOT / 'code/src')]


def training_fixture():
    teacher = dict(name='vjepa21', feature_dim=768, upstream_commit=bootstrap.EXPECTED_TEACHER_COMMIT,
        weight_sha256=bootstrap.EXPECTED_TEACHER_SHA, video_input_size=224, video_tubelet=2,
        video_grid=14, temporal_aggregation='joint_video_tubelet2')
    manifest = dict(teacher=teacher, dataset_source_sha256=bootstrap.EXPECTED_DATASET,
        cache_identity_sha256=bootstrap.EXPECTED_CACHE, source_revision='fastwam-jepa-video-window-v2',
        observation_offsets=[-4, 0], future_offsets=[4, 8, 12, 16, 20, 24, 28, 32],
        stride=4, normalization='training episodes only', splits={'train': [0, 1]})
    model = dict(_target_='fastwam.cosmos.dense_features.create_dense_feature_joint',
        manifest=manifest, **bootstrap.EXPECTED_ARCHITECTURE, force_shift=1., lambda_future=1.,
        action_seed=42, force_seed=43, proprio_seed=44, gradient_checkpointing=True, num_inference_steps=16)
    data = dict(source_sha256=bootstrap.EXPECTED_DATASET, expected_cache_identity=bootstrap.EXPECTED_CACHE,
        expected_teacher='vjepa21', expected_weight_sha256=bootstrap.EXPECTED_TEACHER_SHA)
    return dict(teacher='vjepa21', teacher_weight_sha256=bootstrap.EXPECTED_TEACHER_SHA,
                model=model, data={'train': data})


class BootstrapTests(unittest.TestCase):
    def test_runtime_paths_are_relative_and_initial_profile_is_dense16(self):
        config = bootstrap.runtime_config()
        for key in ('training_config', 'normalization_file', 'force_preprocessing_file',
                    'teacher_source', 'teacher_weights'):
            self.assertFalse(Path(config[key]).is_absolute())
        self.assertEqual(config['sampling_profile'], 'dense16')
        self.assertEqual(config['num_inference_steps'], 16)
        self.assertNotIn('model_kwargs', config)

    def test_exact_run_identity_and_geometry_reject_different_assets(self):
        original = training_fixture()
        bootstrap.check_training_identity(original)
        for mutate in ('cache', 'teacher', 'geometry'):
            trained = deepcopy(original)
            if mutate == 'cache':
                trained['model']['manifest']['cache_identity_sha256'] = 'f' * 64
            elif mutate == 'teacher':
                trained['model']['manifest']['teacher']['weight_sha256'] = 'e' * 64
            else:
                trained['model']['hidden_dim'] = 384
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                bootstrap.check_training_identity(trained)

    def test_v2_contract_uses_actual_teacher_geometry_without_fabricating_legacy_fields(self):
        import torch
        from fastwam.cosmos.dense_features import DenseFeatureJointModel
        from fastwam.cosmos.dense_encoder import _FactoryScalarsOnCPU
        from realman_sync.dense_policy import DensePolicy
        trained = training_fixture()
        kwargs = deepcopy(trained['model'])
        kwargs.pop('_target_')
        with torch.device('meta'), _FactoryScalarsOnCPU():
            model = DenseFeatureJointModel(**kwargs)
        policy = object.__new__(DensePolicy)
        before = deepcopy(model.manifest)
        validated = policy._model_configuration(trained, model.manifest)
        self.assertFalse(validated['gradient_checkpointing'])
        self.assertEqual(sum(value.numel() for value in model.parameters()), 1119958542)
        self.assertEqual(model.manifest, before)
        self.assertNotIn('temporal_pooling', model.manifest)
        for key, value in (('video_tubelet', 1), ('video_grid', 24), ('video_input_size', 384),
                           ('temporal_aggregation', 'adjacent_frame_mean')):
            candidate, manifest = deepcopy(trained), deepcopy(model.manifest)
            candidate['model']['manifest']['teacher'][key] = value
            manifest['teacher'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'temporal'):
                policy._model_configuration(candidate, manifest)
        self.assertFalse(torch.cuda.is_initialized())

    def test_wrong_step_rejected_before_tensor_loading(self):
        from fastwam.cosmos.dense_features import SCHEMA
        for step in (19999, 20001, True, '20000'):
            with self.subTest(step=step), self.assertRaisesRegex(ValueError, 'step_020000'):
                bootstrap.check_checkpoint(dict(schema=SCHEMA, step=step, model={'fake': 1},
                    manifest={}), training_fixture())

    def test_failed_receipt_replacement_leaves_previous_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'receipt.json'
            bootstrap.write_json(path, {'ready': False})
            before = path.read_bytes()
            with patch.object(bootstrap.os, 'replace', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    bootstrap.write_json(path, {'ready': True})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
