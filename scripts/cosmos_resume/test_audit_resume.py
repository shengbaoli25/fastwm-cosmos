"""Small synthetic tests; no model weights, dependencies or training needed."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).with_name("audit_resume.py")
API = runpy.run_path(str(SCRIPT))


def fixture(root):
    run = root / "training"
    state = run / "checkpoints/state/step_025000"
    model = state / "pytorch_model"
    model.mkdir(parents=True)
    (run / "config.yaml").write_text("batch_size: 64\nmax_steps: 50000\n", encoding="utf-8")
    (state / "trainer_state.json").write_text(json.dumps({"global_step": 25000, "epoch": 99, "batch_in_epoch": 52}))
    (state / "latest").write_text("pytorch_model")
    files = [state / "scheduler.bin", state / "sampler.bin", model / "mp_rank_00_model_states.pt",
             run / "checkpoints/weights/step_025000.pt"]
    files += [state / f"random_states_{rank}.pkl" for rank in range(4)]
    files += [model / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt" for rank in range(4)]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("archive/data.pkl", b"synthetic: never unpickle")
            archive.writestr("archive/data/0", b"synthetic")
    return run, state


class AuditTests(unittest.TestCase):
    def test_stdlib_cli_is_read_only_and_does_not_claim_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            run, state = fixture(Path(directory))
            def contents():
                return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in run.rglob("*") if path.is_file()}
            before = contents()
            result = subprocess.run([sys.executable, "-S", str(SCRIPT), str(run), "--archive-index"],
                                    check=True, text=True, capture_output=True)
            report = json.loads(result.stdout)
            self.assertEqual(report["latest_metadata_candidate"], str(state.resolve()))
            self.assertFalse(report["ready_to_resume"])
            self.assertEqual(report["states"][0]["issues"], [])
            self.assertEqual(contents(), before)

    def test_missing_optimizer_shard_and_bad_progress_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run, state = fixture(Path(directory))
            shard = state / "pytorch_model/bf16_zero_pp_rank_3_mp_rank_00_optim_states.pt"
            shard.write_bytes(b"")
            (state / "trainer_state.json").write_text(json.dumps({"global_step": 24999, "epoch": 99, "batch_in_epoch": 52}))
            report = API["state_inventory"](run, state, 4, False)
            self.assertFalse(report["metadata_candidate"])
            self.assertTrue(any("progress" in issue for issue in report["issues"]))
            self.assertTrue(any("empty" in issue for issue in report["issues"]))

    def test_optional_archive_index_detects_corruption_without_unpickling(self):
        with tempfile.TemporaryDirectory() as directory:
            run, state = fixture(Path(directory))
            (state / "scheduler.bin").write_bytes(b"broken archive")
            self.assertTrue(API["state_inventory"](run, state, 4, False)["metadata_candidate"])
            self.assertFalse(API["state_inventory"](run, state, 4, True)["metadata_candidate"])

    def test_completed_log_matches_full_output_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            run, state = fixture(Path(directory))
            log = Path(directory) / "training.log"
            weight = run / "checkpoints/weights/step_025000.pt"
            log.write_text(f"epoch=99 step=25000/50000\nstep=25000 trainer.py:791 weights={weight}\n"
                           f"state={state}\nepoch=106 step=26920/50000\n")
            states = [API["state_inventory"](run, state, 4, False)]
            result = API["log_evidence"](log, run, states)
            self.assertEqual(result["completed_save_steps"], [25000])
            self.assertEqual(result["last_training_record"]["step"], 26920)
            wrong = API["log_evidence"](log, run / "different", states)
            self.assertEqual(wrong["completed_save_steps"], [])


if __name__ == "__main__":
    unittest.main()
