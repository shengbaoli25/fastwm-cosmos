"""Real Rich/Accelerate CPU regression; CUDA transport is mocked, not GPU-tested."""
import importlib.util
import io
import logging
from pathlib import Path
import random
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from accelerate import PartialState
from accelerate.checkpointing import load_accelerator_state
from rich.console import Console
from rich.logging import RichHandler

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cosmos_resume_rng_wrapper", HERE / "resume_to_step.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)
setup_logging = runpy.run_path(str(HERE.parents[1] / "src/fastwam/utils/logging_config.py"))["setup_logging"]


class RankZeroLogRngTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state = Path(self.temp.name)
        self.previous = random.getstate(), np.random.get_state(), torch.get_rng_state()
        self.root = logging.getLogger()
        self.handlers, self.level = self.root.handlers[:], self.root.level
        self.stream = io.StringIO()
        self.cuda = [torch.tensor([i, i + 1, i + 2], dtype=torch.uint8) for i in range(4)]
        self.getter = patch.object(torch.cuda, "get_rng_state_all", side_effect=lambda: [x.clone() for x in self.cuda])
        self.setter = patch.object(torch.cuda, "set_rng_state_all", side_effect=self.set_cuda)
        self.getter.start(); self.setter.start()
        PartialState(cpu=True)
        random.seed(177); np.random.seed(178); torch.manual_seed(179)
        py, npy, cpu, cuda = wrapper.rng_snapshot()
        self.saved = dict(random_state=py, numpy_random_seed=npy, torch_manual_seed=cpu,
                          torch_cuda_manual_seed=cuda, step=52)
        self.path = self.state / "random_states_0.pkl"
        torch.save(self.saved, self.path)
        self.digest = wrapper.sha(self.path)

    def tearDown(self):
        random.setstate(self.previous[0]); np.random.set_state(self.previous[1]); torch.set_rng_state(self.previous[2])
        self.root.handlers[:], self.root.level = self.handlers, self.level
        self.getter.stop(); self.setter.stop()
        self.temp.cleanup()

    def set_cuda(self, states):
        self.cuda = [item.clone() for item in states]

    def logging(self, fixed):
        # Exercise the actual audited setup function and actual Rich renderer.
        if fixed:
            wrapper.setup_resume_logging(setup_logging, True)
        else:
            setup_logging(log_level=logging.INFO, is_main_process=True)
        rich = next(handler for handler in self.root.handlers if isinstance(handler, RichHandler))
        rich.console = Console(file=self.stream, width=180)
        return rich

    def native_restore(self):
        random.random(); np.random.rand(); torch.rand(5)
        self.cuda = [state + 7 for state in self.cuda]
        return load_accelerator_state(self.state, [], [], [], [], process_index=0)["step"]

    def status(self, step=52):
        return wrapper.rank_rng_comparison(wrapper.rng_snapshot(), self.saved, step)

    def test_original_rank_zero_success_log_changes_only_python_rng(self):
        self.logging(fixed=False)
        step = self.native_restore()
        self.assertIn("All random states loaded successfully", self.stream.getvalue())
        self.assertEqual(self.status(step)["mismatches"], ["python"])
        self.assertEqual(wrapper.sha(self.path), self.digest)

    def test_fixed_handler_retains_native_restore_and_all_next_cpu_draws(self):
        handler = self.logging(fixed=True)
        step = self.native_restore()
        logging.info("Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d", 99, 52, 13312)
        logging.info("Train/val dataset size: %d/%d", 64462, 64462)
        self.assertEqual(self.status(step)["mismatches"], [])
        self.assertFalse(handler.enable_link_path)
        self.assertIn("test_rng_restore.py", self.stream.getvalue())
        self.assertIn("Restored dataloader progress", self.stream.getvalue())
        self.assertEqual(wrapper.sha(self.path), self.digest)
        expected_py = random.Random(); expected_py.setstate(self.saved["random_state"])
        expected_np = np.random.RandomState(); expected_np.set_state(self.saved["numpy_random_seed"])
        expected_cpu = torch.Generator(); expected_cpu.set_state(self.saved["torch_manual_seed"])
        self.assertEqual(random.random(), expected_py.random())
        np.testing.assert_array_equal(np.random.rand(7), expected_np.rand(7))
        self.assertTrue(torch.equal(torch.rand(7), torch.rand(7, generator=expected_cpu)))

    def test_silenced_nonmain_log_does_not_drift(self):
        self.logging(fixed=False)
        setup_logging(log_level=logging.INFO, is_main_process=False)
        step = self.native_restore()
        logging.info("Train/val dataset size: %d/%d", 64462, 64462)
        self.assertEqual(self.status(step)["mismatches"], [])

    def test_real_rng_drift_is_still_reported_and_never_repaired(self):
        self.logging(fixed=True)
        self.native_restore()
        for name, draw in (("python", random.random), ("numpy", np.random.rand), ("torch_cpu", lambda: torch.rand(1))):
            with self.subTest(domain=name):
                self.native_restore()
                draw()
                before = wrapper.rng_snapshot()
                self.assertEqual(self.status()["mismatches"], [name])
                self.assertTrue(wrapper.equal_state(before, wrapper.rng_snapshot()))
        self.assertIn("accelerate_step", self.status(25000)["mismatches"])
        self.cuda.append(self.cuda[0].clone())
        self.assertIn("torch_cuda", self.status()["mismatches"])
        self.assertEqual(self.status()["current_cuda_state_count"], 5)


if __name__ == "__main__":
    unittest.main()
