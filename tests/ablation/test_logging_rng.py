"""Actual Rich and Accelerate CPU save/restore; CUDA transport is mocked."""
import io
import logging
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from accelerate import PartialState
from accelerate import checkpointing
from rich.console import Console
from rich.logging import RichHandler

from fastwam.ablation.fastwam import rng_neutral_training_logging
from fastwam.ablation.training_audit import assert_restored, rng_fingerprint
from fastwam.utils.logging_config import setup_logging


@pytest.fixture
def real_logging(tmp_path, monkeypatch):
    root = logging.getLogger()
    previous = (root.handlers[:], root.level, random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
    monkeypatch.setattr(checkpointing, "is_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [value.clone() for value in cuda])
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda *_: cuda[0].clone())
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda values: cuda.__setitem__(slice(None), [value.clone() for value in values]))
    PartialState(cpu=True)
    runtime = SimpleNamespace(setup_logging=setup_logging)
    output = io.StringIO()

    def configure():
        runtime.setup_logging(log_level=logging.INFO, is_main_process=True)
        handler = next(item for item in root.handlers if isinstance(item, RichHandler))
        handler.console = Console(file=output, width=180)
        random.seed(101); np.random.seed(102); torch.manual_seed(103)
        return handler

    yield runtime, output, configure
    root.handlers[:], root.level = previous[0], previous[1]
    random.setstate(previous[2]); np.random.set_state(previous[3]); torch.set_rng_state(previous[4])


def native_save(path):
    checkpointing.save_accelerator_state(path, [], [], [], [], process_index=0, step=2)


def native_restore(path):
    random.random(); np.random.rand(); torch.rand(1)
    return checkpointing.load_accelerator_state(path, [], [], [], [], process_index=0)["step"]


def test_original_rich_path_links_drift_after_native_save_and_restore(tmp_path, real_logging):
    _, output, configure = real_logging
    configure()
    before = rng_fingerprint("cpu")
    native_save(tmp_path)
    assert rng_fingerprint("cpu") != before
    assert native_restore(tmp_path) == 2
    assert rng_fingerprint("cpu") != before
    assert "All random states loaded successfully" in output.getvalue()


def test_fixed_worker_preserves_native_save_restore_and_following_draws(tmp_path, real_logging):
    runtime, output, configure = real_logging
    original = runtime.setup_logging
    with rng_neutral_training_logging(runtime):
        handler = configure()
        assert handler.enable_link_path is False
        before = rng_fingerprint("cpu")
        expected_py = random.Random(); expected_py.setstate(random.getstate())
        expected_np = np.random.RandomState(); expected_np.set_state(np.random.get_state())
        expected_cpu = torch.Generator(); expected_cpu.set_state(torch.get_rng_state())
        native_save(tmp_path)
        logging.info("Saved state checkpoint at step=%d", 2)
        assert rng_fingerprint("cpu") == before
        assert native_restore(tmp_path) == 2
        logging.info("Restored dataloader progress: epoch=%d batch_in_epoch=%d", 0, 2)
        logging.info("Train/val dataset size: %d/%d", 100, 0)
        assert_restored({"rng_sha256": rng_fingerprint("cpu")}, {"rng_sha256": before})
        assert random.random() == expected_py.random()
        np.testing.assert_array_equal(np.random.rand(4), expected_np.rand(4))
        assert torch.equal(torch.rand(4), torch.rand(4, generator=expected_cpu))
        assert "test_logging_rng.py" in output.getvalue()
        assert "All random states loaded successfully" in output.getvalue()
    assert runtime.setup_logging is original


def test_real_rng_change_still_fails_strict_gate_without_repair(tmp_path, real_logging):
    runtime, _, configure = real_logging
    with rng_neutral_training_logging(runtime):
        configure()
        native_save(tmp_path)
        expected = rng_fingerprint("cpu")
        for draw in (random.random, np.random.rand, lambda: torch.rand(1)):
            native_restore(tmp_path)
            draw()
            changed = rng_fingerprint("cpu")
            with pytest.raises(ValueError, match="rng_sha256"):
                assert_restored({"rng_sha256": changed}, {"rng_sha256": expected})
            assert rng_fingerprint("cpu") == changed


def test_cli_wraps_entire_native_training_call_before_restore():
    source = (Path(__file__).resolve().parents[2] / "scripts/ablation/fastwam.py").read_text(encoding="utf-8")
    assert "with rng_neutral_training_logging(runtime), bounded_training_invocation(" in source
    assert "runtime.run_training(cfg)" in source
