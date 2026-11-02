"""CPU contract tests for the scoped wrapper, including actual Accelerate padding."""
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cosmos_resume_wrapper", HERE / "resume_to_step.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


class ResumeTests(unittest.TestCase):
    def test_plan_without_torch_or_files_has_exact_stop_semantics(self):
        result = subprocess.run([sys.executable, "-S", str(HERE / "resume_to_step.py"),
                                 "--config", "/original/config.yaml", "--state", "/original/state/step_025000",
                                 "--output", "/new/training", "--source-root", "/source"],
                                check=True, capture_output=True, text=True)
        value = json.loads(result.stdout)
        self.assertFalse(value["training_started"])
        self.assertEqual((value["original_scheduler_and_model_recipe_steps"], value["restore_step"],
                          value["stop_at_total_step"], value["additional_updates"]), (50000, 25000, 30000, 5000))

    def test_original_config_accepts_but_rewritten_schedule_rejects(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(HERE / "original-config.yaml")
        self.assertEqual(wrapper.sha(HERE / "original-config.yaml"), wrapper.CONFIG_SHA256)
        wrapper.validate_config(cfg)
        cfg.model.training_recipe.max_steps = 30000
        with self.assertRaisesRegex(ValueError, "50k manifest"):
            wrapper.validate_config(cfg)
        cfg.model.training_recipe.max_steps = 50000
        cfg.max_steps = 30000
        with self.assertRaisesRegex(ValueError, "unchanged"):
            wrapper.validate_config(cfg)

    def test_four_rank_continuation_matches_original_order_and_end_padding(self):
        import torch
        from torch.utils.data import DataLoader
        from accelerate.data_loader import prepare_data_loader
        native = HERE.parents[1] / "src/fastwam/utils/samplers.py"
        sampler_class = runpy.run_path(str(native))["ResumableEpochSampler"]
        dataset = range(64462)

        def make_loader(rank):
            sampler = sampler_class(dataset, seed=42, batch_size=64, num_processes=4)
            loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=0)
            shard = prepare_data_loader(loader, device=torch.device("cpu"), num_processes=4,
                                        process_index=rank, even_batches=True, rng_types=[])
            return shard, sampler

        for rank in range(4):
            with self.subTest(rank=rank):
                original, _ = make_loader(rank)
                original.set_epoch(99)
                batches = [batch.tolist() for batch in original]
                original_next = [batch.tolist() for batch in original]
                self.assertEqual(len(batches), 252)
                resumed, sampler = make_loader(rank)
                sampler.set_epoch_offset(99)
                sampler.set_resume_batch_offset(52)
                continuation = wrapper.FirstEpochContinuation(resumed, sampler, 99, 52)
                snapshot = torch.get_rng_state().clone()
                with patch.object(wrapper, "rng_snapshot", side_effect=lambda: torch.get_rng_state().clone()), \
                        patch.object(wrapper, "restore_rng", side_effect=torch.set_rng_state):
                    iterator = iter(continuation)
                    first = next(iterator)
                    self.assertTrue(torch.equal(torch.get_rng_state(), snapshot))
                    actual = [first.tolist()] + [batch.tolist() for batch in iterator]
                self.assertEqual(actual, batches[52:])
                self.assertEqual([batch.tolist() for batch in continuation], original_next)

    def test_tensor_numpy_and_nested_checkpoint_equality(self):
        import numpy as np
        import torch
        a = {"torch": torch.tensor([1, 2]), "numpy": np.array([4, 5]), "nested": [1, (2, 3)]}
        b = {"torch": torch.tensor([1, 2]), "numpy": np.array([4, 5]), "nested": [1, (2, 3)]}
        self.assertTrue(wrapper.equal_state(a, b))
        b["torch"][1] = 9
        self.assertFalse(wrapper.equal_state(a, b))


if __name__ == "__main__":
    unittest.main()
