"""Production reference/baseline recipe guards, without loading model weights."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

from fastwam.cosmos.preflight import validate_config, REFERENCE_TARGET, REFERENCE_DATASET

ROOT = Path(__file__).resolve().parents[2]


def configuration(reference=True):
    model = dict(_target_=REFERENCE_TARGET if reference else 'fastwam.cosmos.assets.create_noaction_joint_cosmos',
        action_hidden=768, action_ffn=3072, force_hidden=768, force_ffn=2048,
        force_heads=16, force_shift=1., num_inference_steps=16,
        action_seed=42, force_seed=43, memory_seed=44, action_space='joint8',
        expected_data_identity='a'*64, expected_text_identity='b'*64,
        expected_normalization_sha256='c'*64, action_representation='delta_current_joint7',
        training_recipe=dict(batch_per_rank=64, max_steps=30000, scheduler_total_steps=50000))
    dataset = dict(_target_=REFERENCE_DATASET if reference else 'fastwam.cosmos.cache.CosmosCachedDataset',
        expected_identity_sha256='a'*64, expected_text_identity='b'*64, require_production=True,
        expected_action_representation='delta_current_joint7')
    if reference:
        model.update(reference_hidden=128, reference_heads=4, reference_layers=2,
                     reference_seed=46, alignment_weight=.1, expected_reference_identity='d'*64)
        dataset['expected_reference_identity'] = 'd'*64
    return dict(batch_size=64, gradient_accumulation_steps=1, max_steps=30000, save_every=5000,
        learning_rate=1e-4, weight_decay=.01, mixed_precision='bf16', seed=42, resume=None,
        lr_scheduler_type='cosine', eval_every=0, scheduler_total_steps=50000,
        model=model, data=dict(train=dataset,val=None))


class ReferencePreflightTests(unittest.TestCase):
    def test_reference_and_existing_baseline_accept_the_authorized_recipe(self):
        validate_config(configuration())
        validate_config(configuration(False))

    def test_reference_cannot_silently_change_wam_or_binding_or_training_size(self):
        edits = [
            ('batch_size',16), ('max_steps',20000), ('scheduler_total_steps',30000),
            ('mixed_precision','fp16'), ('gradient_accumulation_steps',2),
            ('model.action_hidden',1024), ('model.force_ffn',3072), ('model.reference_seed',44),
            ('model.action_representation','absolute'), ('model.initial_checkpoint','old-model.pt'),
            ('model.expected_data_identity','e'*64), ('model.expected_reference_identity','e'*64),
            ('model.expected_normalization_sha256',''), ('data.train.expected_text_identity','e'*64),
            ('data.train._target_','fastwam.cosmos.cache.CosmosCachedDataset'),
            ('data.train.expected_action_representation','absolute'), ('data.train.require_production',False),
            ('data.val',{}), ('model.training_recipe.scheduler_total_steps',30000),
        ]
        for path, value in edits:
            with self.subTest(path=path):
                cfg = deepcopy(configuration())
                target = cfg
                keys = path.split('.')
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with self.assertRaises(ValueError):
                    validate_config(cfg)

    def test_old_baseline_batch_and_short_schedules_remain_valid(self):
        for batch in (16,32,64):
            for steps in (1000,20000,30000):
                cfg = configuration(False)
                cfg.update(batch_size=batch, max_steps=steps, save_every=1000 if steps==1000 else 5000)
                cfg['model']['training_recipe'].update(batch_per_rank=batch,max_steps=steps)
                validate_config(cfg)

    def test_reference_dataset_cannot_enter_a_baseline_model(self):
        cfg = configuration(False)
        cfg['data']['train']['_target_'] = REFERENCE_DATASET
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_journal_describes_joint_refinement_and_auxiliary_loss(self):
        spec = importlib.util.spec_from_file_location('reference_distributed_chunk',
            ROOT/'scripts/cosmos/distributed_chunk.py')
        module = importlib.util.module_from_spec(spec)
        original_path = list(sys.path)
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = original_path
        name, semantics, losses = module.experiment_contract(configuration()['model'])
        self.assertEqual(name, 'cosmos-reference-audited-joint-diffusion')
        self.assertIn('AE only', semantics)
        self.assertNotIn('GT-conditioned reconstruction', semantics)
        self.assertEqual(losses,dict(video=1,action=1,force=1,reference_alignment=.1))
        baseline = module.experiment_contract(configuration(False)['model'])
        self.assertEqual(baseline[0], 'cosmos-noaction-joint-diffusion')
        self.assertEqual(baseline[2],dict(video=1,action=1,force=1))


if __name__ == '__main__':
    unittest.main()
