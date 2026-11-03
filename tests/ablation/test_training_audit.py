"""Offline audit regression fixtures; no baseline training or GPU execution."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
from fastwam.ablation import training_audit as audit

spec = importlib.util.spec_from_file_location("verify_smoke", Path(__file__).resolve().parents[2] / "scripts/ablation/verify_fastwam_smoke.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        action = torch.nn.Module()
        action.action_encoder = torch.nn.Linear(8, 4)
        action.head = torch.nn.Linear(4, 8)
        # Official FastWAM registers these aliases before mot/dit. Parent-level
        # named_parameters() therefore emits expert.* names and deduplicates mot.
        self.video_expert = torch.nn.Linear(4, 4)
        self.action_expert = action
        self.mot = torch.nn.Module()
        self.mot.mixtures = torch.nn.ModuleDict({"video": self.video_expert, "action": action})
        self.dit = self.mot
        self.proprio_encoder = torch.nn.Linear(8, 4)
        self.ablation_binding = {"dataset_identity": "fixture-only", "training_recipe": {"max_steps": 3}}


def test_shared_expert_aliases_are_audited_under_saved_checkpoint_names(tmp_path):
    model = ToyPolicy()
    names = dict(model.named_parameters())
    assert "action_expert.action_encoder.weight" in names
    assert "video_expert.weight" in names
    assert not any(name.startswith("mot.") for name in names)
    samples = audit.policy_samples(model)
    assert {key: value["name"] for key, value in samples.items()} == {
        "action_input": "mot.mixtures.action.action_encoder.weight",
        "action_output": "mot.mixtures.action.head.weight",
        "video_backbone": "mot.mixtures.video.weight",
        "proprio": "proprio_encoder.weight",
    }
    with torch.no_grad():
        model.action_expert.head.weight.add_(0.25)
    changed = audit.policy_samples(model)
    assert changed["action_output"]["sha256"] != samples["action_output"]["sha256"]
    path = tmp_path / "aliased_policy.pt"
    torch.save({"mot": model.mot.state_dict(), "proprio_encoder": model.proprio_encoder.state_dict(),
                "ablation": model.ablation_binding, "step": 2}, path)
    verify.verify_saved_weights(path, model.ablation_binding, 2, changed)
    with pytest.raises(ValueError, match="Actual saved weight bytes"):
        verify.verify_saved_weights(path, model.ablation_binding, 2, samples)


@pytest.mark.parametrize("category,path", [
    ("action_input", "action_expert.action_encoder"),
    ("action_output", "action_expert.head"),
    ("video_backbone", "video_expert"),
    ("proprio", "proprio_encoder"),
])
def test_frozen_audited_category_is_still_rejected(category, path):
    model = ToyPolicy()
    model.get_submodule(path).requires_grad_(False)
    with pytest.raises(ValueError, match=category):
        audit.policy_samples(model)


def trainer_fixture(tmp_path, *, change_parameters=True):
    model = ToyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    trainer = SimpleNamespace(global_step=0, epoch=0, batch_in_epoch=0)
    engine = SimpleNamespace(optimizer=optimizer, global_steps=0, get_global_grad_norm=lambda: 1.5,
                             train_micro_batch_size_per_gpu=lambda: 64, gradient_accumulation_steps=lambda: 1,
                             train_batch_size=lambda: 256)
    accelerator = SimpleNamespace(process_index=0, device="cpu", step=0, num_processes=4, optimizer_step_was_skipped=False,
                                  unwrap_model=lambda _engine: model, wait_for_everyone=lambda: None)
    def backward(_loss):
        # Deterministic test double, not a baseline optimizer or backward pass.
        with torch.no_grad():
            for parameter in model.parameters():
                old = optimizer.state.get(parameter, {})
                optimizer.state[parameter] = {"step": torch.tensor(float(engine.global_steps + 1)),
                                               "exp_avg": torch.ones_like(parameter) * (engine.global_steps + 1),
                                               "exp_avg_sq": torch.ones_like(parameter) * (engine.global_steps + 1) ** 2}
                if change_parameters:
                    parameter.add_(0.01)
        engine.global_steps += 1
    def save():
        state = tmp_path / f"checkpoints/state/step_{trainer.global_step:06d}"
        state.mkdir(parents=True, exist_ok=True)
        return {"state_path": str(state), "weights_path": None}
    accelerator.backward = backward
    trainer.model, trainer.accelerator, trainer.save_checkpoint = engine, accelerator, save
    trainer.scheduler = SimpleNamespace(state_dict=lambda: {"last_epoch": trainer.global_step})
    trainer.train_sampler = SimpleNamespace(epoch_offset=0, resume_batch_offset=0)
    return trainer, model


def run_fixture(trainer, cfg, limit):
    # execution_profile accepts DictConfig/mapping; retain attribute access for
    # the synthetic trainer configuration without loading any baseline assets.
    cfg.get = lambda name, default=None: getattr(cfg, name, default)
    with patch.object(audit, "rng_fingerprint", return_value="fixed-rng-fixture"), \
            patch.object(audit, "memory_record", return_value={"allocated_bytes": 1024, "reserved_bytes": 2048,
                                                               "peak_allocated_bytes": 1536, "peak_reserved_bytes": 2048}), \
            patch.object(torch.cuda, "reset_peak_memory_stats"):
        with audit.audit_prepared_training(trainer, cfg, limit):
            while trainer.global_step < limit:
                trainer.accelerator.backward(torch.tensor(2.0, requires_grad=True))
                trainer.global_step += 1
                trainer.batch_in_epoch += 1
                trainer.accelerator.step += 1
                trainer.save_checkpoint()


def test_audit_records_real_call_boundaries_and_matches_resumed_state(tmp_path):
    trainer, _ = trainer_fixture(tmp_path)
    cfg = SimpleNamespace(max_steps=3, output_dir=str(tmp_path), resume=None)
    run_fixture(trainer, cfg, 2)
    first = json.loads((tmp_path / "smoke-audit/from_000000-to_000002.rank_000.json").read_text())
    verify.validate_receipt(first, 0, 0, 2, first["binding_sha256"])
    cfg.resume = str(tmp_path / "checkpoints/state/step_000002")
    trainer.train_sampler.resume_batch_offset = 2
    run_fixture(trainer, cfg, 3)
    second = json.loads((tmp_path / "smoke-audit/from_000002-to_000003.rank_000.json").read_text())
    verify.validate_receipt(second, 0, 2, 3, first["binding_sha256"])
    assert second["restored_checkpoint_verified"] is True
    assert second["initial"] == first["final"]


def test_missing_actual_parameter_change_rejected(tmp_path):
    trainer, _ = trainer_fixture(tmp_path, change_parameters=False)
    cfg = SimpleNamespace(max_steps=3, output_dir=str(tmp_path), resume=None)
    with pytest.raises(ValueError, match="No sampled FP32 master"):
        run_fixture(trainer, cfg, 2)
    assert not (tmp_path / "smoke-audit/from_000000-to_000002.rank_000.json").exists()


@pytest.mark.parametrize("field", ["scheduler", "rng_sha256", "optimizer", "policy_samples"])
def test_restored_state_difference_rejected(field):
    original = {"scheduler": {"last_epoch": 2}, "rng_sha256": "saved", "optimizer": {"step": 2}, "policy_samples": {"x": "saved"}}
    changed = copy.deepcopy(original)
    changed[field] = None
    with pytest.raises(ValueError, match=field):
        audit.assert_restored(changed, original)


def test_verifier_rejects_nonfinite_gradient_even_with_completed_receipt(tmp_path):
    trainer, _ = trainer_fixture(tmp_path)
    run_fixture(trainer, SimpleNamespace(max_steps=3, output_dir=str(tmp_path), resume=None), 2)
    receipt = json.loads((tmp_path / "smoke-audit/from_000000-to_000002.rank_000.json").read_text())
    receipt["steps"][0]["global_grad_norm"] = float("nan")
    with pytest.raises(ValueError, match="gradient"):
        verify.validate_receipt(receipt, 0, 0, 2, receipt["binding_sha256"])


def test_actual_saved_weight_bytes_are_read_and_compared(tmp_path):
    _, model = trainer_fixture(tmp_path)
    samples = audit.policy_samples(model)
    path = tmp_path / "step_000002.pt"
    payload = {"mot": model.mot.state_dict(), "proprio_encoder": model.proprio_encoder.state_dict(),
               "ablation": model.ablation_binding, "step": 2}
    torch.save(payload, path)
    verify.verify_saved_weights(path, model.ablation_binding, 2, samples)
    payload["mot"]["mixtures.action.head.weight"].add_(1)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Actual saved weight bytes"):
        verify.verify_saved_weights(path, model.ablation_binding, 2, samples)


def test_complete_four_rank_gate_rejects_changed_resume_evidence(tmp_path):
    trainer, model = trainer_fixture(tmp_path)
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    model.ablation_binding["normalization_sha256"] = audit.hashlib.sha256(b"{}").hexdigest()
    cfg = SimpleNamespace(max_steps=3, output_dir=str(tmp_path), resume=None,
                          dataset_root=str(tmp_path), normalization_path=str(stats), asset_root=str(tmp_path),
                          text_embedding_cache_dir=str(tmp_path))
    (tmp_path / "fastwam_binding.json").write_text(json.dumps(model.ablation_binding))
    (tmp_path / "training-hardware.json").write_text(json.dumps({"num_processes": 4, "per_gpu_batch_size": 64,
        "global_batch_size": 256, "gradient_accumulation_steps": 1, "device_names": ["H200 test fixture"] * 4}))
    def saved_fixtures(start, end):
        receipt_path = tmp_path / f"smoke-audit/from_{start:06d}-to_{end:06d}.rank_000.json"
        sidecar_path = tmp_path / f"checkpoints/state/step_{end:06d}/smoke-audit-rank-000.json"
        for rank in range(1, 4):
            receipt = json.loads(receipt_path.read_text())
            receipt["rank"] = rank
            receipt_path.with_name(receipt_path.name.replace("rank_000", f"rank_{rank:03d}")).write_text(json.dumps(receipt))
            sidecar = json.loads(sidecar_path.read_text())
            sidecar["rank"] = rank
            sidecar_path.with_name(f"smoke-audit-rank-{rank:03d}.json").write_text(json.dumps(sidecar))
        (sidecar_path.parent / "trainer_state.json").write_text(json.dumps({"global_step": end, "epoch": 0, "batch_in_epoch": end}))
        weights = tmp_path / f"checkpoints/weights/step_{end:06d}.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mot": model.mot.state_dict(), "proprio_encoder": model.proprio_encoder.state_dict(),
                    "ablation": model.ablation_binding, "step": end}, weights)
    run_fixture(trainer, cfg, 2)
    saved_fixtures(0, 2)
    with patch.object(verify, "load_dataset_manifest", return_value={"identity_sha256": "fixture-only"}), \
            patch("fastwam.ablation.fastwam.semantic_recipe", return_value={"fixture": True}), \
            patch("fastwam.ablation.fastwam.build_binding", return_value=model.ablation_binding):
        assert verify.verify_run(cfg, "saved2")["formal_training_permitted"] is False
    cfg.resume = str(tmp_path / "checkpoints/state/step_000002")
    trainer.train_sampler.resume_batch_offset = 2
    run_fixture(trainer, cfg, 3)
    saved_fixtures(2, 3)
    with patch.object(verify, "load_dataset_manifest", return_value={"identity_sha256": "fixture-only"}), \
            patch("fastwam.ablation.fastwam.semantic_recipe", return_value={"fixture": True}), \
            patch("fastwam.ablation.fastwam.build_binding", return_value=model.ablation_binding):
        assert verify.verify_run(cfg, "resumed3")["formal_training_permitted"] is True
        corrupted_path = tmp_path / "smoke-audit/from_000002-to_000003.rank_003.json"
        corrupted = json.loads(corrupted_path.read_text())
        corrupted["initial"]["scheduler"]["last_epoch"] = 0
        corrupted_path.write_text(json.dumps(corrupted))
        with pytest.raises(ValueError, match="scheduler"):
            verify.verify_run(cfg, "resumed3")
