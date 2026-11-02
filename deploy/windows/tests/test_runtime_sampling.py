"""Offline launcher/asset-check parity; no models, network or hardware."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_assets
from server import validate_config


class RuntimeSamplingTests(unittest.TestCase):
    def config(self, profile=None):
        value = dict(execution_mode='synchronous', joint_steps=16,
            inference_backend={'4416': 'joint_multirate', '3310': 'joint_expert_schedule'}.get(
                profile, 'joint_static_cache'),
            inference_optimization=({'compile_layers': False, 'cuda_graph': False}
                if profile == '3310' else {'mode': 'eager'}),
            model_kwargs={'num_inference_steps': 16})
        if profile is not None:
            value['sampling_profile'] = profile
        return value

    def test_supported_profiles_do_not_modify_saved_model_inputs(self):
        for profile in (None, '161616', '4416', '3310'):
            with self.subTest(profile=profile):
                values = self.config(profile)
                before = deepcopy(values)
                validate_config(values)
                self.assertEqual(values, before)

    def test_mismatched_backends_and_unvalidated_profiles_are_rejected(self):
        for profile, backend in [('4416', 'joint_static_cache'),
                ('3310', 'joint_multirate'), ('3310', 'joint_static_cache'),
                ('4416', 'joint_expert_schedule'), ('161616', 'joint_expert_schedule'),
                ('161616', 'joint_multirate'), (None, 'joint_multirate'),
                ('448', 'joint_multirate'), (4416, 'joint_multirate'), ([], 'joint_multirate')]:
            with self.subTest(profile=profile, backend=backend):
                values = self.config()
                values.update(sampling_profile=profile, inference_backend=backend)
                with self.assertRaises(ValueError):
                    validate_config(values)

    def test_3310_requires_eager_schedule_and_preserves_trained_steps(self):
        replacements = [dict(execution_mode='async_relaxed'),
            dict(inference_optimization={}),
            dict(inference_optimization={'compile_layers': False}),
            dict(inference_optimization={'cuda_graph': False}),
            dict(inference_optimization={'mode': 'eager'}),
            dict(inference_optimization={'compile_layers': True, 'cuda_graph': False}),
            dict(inference_optimization={'compile_layers': False, 'cuda_graph': True}),
            dict(inference_optimization={'compile_layers': 0, 'cuda_graph': False}),
            dict(inference_optimization={'compile_layers': False, 'cuda_graph': 0}),
            dict(inference_optimization=None), dict(joint_steps=10),
            dict(model_kwargs={'num_inference_steps': 10})]
        for replacement in replacements:
            with self.subTest(replacement=replacement):
                values = self.config('3310')
                values.update(replacement)
                with self.assertRaises(ValueError):
                    validate_config(values)

    def test_synchronous_eager_and_original_step_counts_remain_required(self):
        replacements = [dict(execution_mode='asynchronous'), dict(execution_mode='eager'),
            dict(inference_optimization={'mode': 'compile'}),
            dict(inference_optimization={'mode': 'eager', 'compile': True}),
            dict(inference_optimization=None), dict(joint_steps=4), dict(joint_steps=True),
            dict(model_kwargs={'num_inference_steps': 4}),
            dict(model_kwargs={'num_inference_steps': 16.0}), dict(model_kwargs=[])]
        for replacement in replacements:
            with self.subTest(replacement=replacement):
                values = self.config('4416')
                values.update(replacement)
                with self.assertRaises(ValueError):
                    validate_config(values)

    def test_actual_persisted_20k_configuration_passes_launcher_guard(self):
        runtime = Path(__file__).resolve().parents[2] / 'cosmos/config/runtime.step_020000.json'
        if not runtime.is_file():
            self.skipTest('Local deployment integration config is not distributed with source')
        validate_config(json.loads(runtime.read_text(encoding='utf-8-sig')))

    def test_asset_check_uses_same_guard_and_bootstrap_does_not_require_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            normalization, tokenizer, context = [root/name for name in ('norm.json', 'token.pth', 'context.bin')]
            for path in (normalization, tokenizer, context):
                path.write_bytes(b'test-only')
            (root/'manifest.json').write_text(json.dumps({'texts': [{'path': context.name}]}))
            values = self.config('3310')
            values.update(normalization_file=str(normalization), tokenizer=str(tokenizer),
                          policy_backend='cosmos_checkpoint')
            values['model_kwargs'].update(upstream=str(root), text_cache=str(root))
            config_path = root/'runtime.json'
            experiment = dict(name='cosmos', supporting_assets=[dict(kind='generated_runtime_config',
                              target_path=str(config_path))])
            config_path.write_text(json.dumps(values))
            self.assertEqual(check_assets.inspect_runtime(experiment)['errors'], [])
            values['inference_backend'] = 'joint_static_cache'
            config_path.write_text(json.dumps(values))
            errors = check_assets.inspect_runtime(experiment)['errors']
            self.assertTrue(any('requires joint_expert_schedule' in error for error in errors))
            values['inference_backend'] = 'joint_expert_schedule'
            values['policy_backend'] = 'cosmos'
            config_path.write_text(json.dumps(values))
            errors = check_assets.inspect_runtime(experiment)['errors']
            self.assertTrue(any('model_kwargs.checkpoint' in error for error in errors))


if __name__ == '__main__':
    unittest.main()
