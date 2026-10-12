"""CPU semantic tests; no production model, CUDA, or distributed launch."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from functools import wraps
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fastwam.trainer import Wan22Trainer


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("cached_preflight", ROOT / "scripts/forcevla/preflight_cached_gpu.py")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def _arguments(tmp_path, *, identity="a" * 64):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["task=forcevla_libero_2cam_cached_4gpu_30k"])
    cfg.data.train.expected_identity_sha256 = identity
    path = tmp_path / "resolved.yaml"
    OmegaConf.save(cfg, path, resolve=True)
    return SimpleNamespace(config=str(path), plan=None, output=str(tmp_path / "unused-output"),
                           num_processes=None, io_benchmark_mib=0, remaining_budget_seconds=None)


def test_metadata_default_preserves_formal_horizon_and_never_constructs_hardware(tmp_path, monkeypatch, capsys):
    args = _arguments(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("The default entrypoint must not execute a GPU probe")

    monkeypatch.setattr(PROBE, "execute", forbidden)
    monkeypatch.setattr("sys.argv", ["preflight_cached_gpu.py", "--config", args.config, "--output", args.output])
    PROBE.main()
    report = json.loads(capsys.readouterr().out)
    assert report["execute_preflight"] is False and report["formal_training_started"] is False
    assert report["diagnostic_updates"] == 2 and report["planned_schedule_steps"] == 30000
    assert report["world_size"] == 4 and report["effective_batch"] == 64
    assert report["cache_identity_bound"] is True and report["checkpoint_saved"] is False
    assert not Path(args.output).exists()


def test_execution_requires_bound_cache_and_exact_distributed_world_before_cuda(tmp_path, monkeypatch):
    args = _arguments(tmp_path, identity=None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("no CUDA inspection")))
    with pytest.raises(ValueError, match="verified cache identity"):
        PROBE.execute(args)
    args = _arguments(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(RuntimeError, match="four/eight-rank"):
        PROBE.execute(args)


def test_plan_and_config_identity_or_world_size_cannot_diverge(tmp_path):
    args = _arguments(tmp_path)
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    plan = {"world_size": 4, "config": config, "cache_check": {"identity_sha256": "a" * 64}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    args.plan, args.config = str(plan_path), None
    assert PROBE.metadata(args)["world_size"] == 4
    args.num_processes = 8
    with pytest.raises(ValueError, match="world size"):
        PROBE.load_inputs(args)
    args.num_processes = None
    plan["cache_check"]["identity_sha256"] = "b" * 64
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="cache identity"):
        PROBE.load_inputs(args)


class FixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))

    def training_loss(self, sample):
        loss = (self.weight - 1).square()
        return loss, {"loss_fixture": float(loss.detach())}


class FixtureAccelerator:
    sync_gradients = True
    optimizer_step_was_skipped = False

    def accumulate(self, model):
        return nullcontext()

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, norm):
        return torch.nn.utils.clip_grad_norm_(parameters, norm)

    def gather(self, tensor):
        return tensor


def _fixture_trainer():
    trainer = object.__new__(Wan22Trainer)
    trainer.max_steps, trainer.global_step, trainer.batch_in_epoch = 30000, 0, 0
    trainer.model, trainer.accelerator = FixtureModel(), FixtureAccelerator()
    trainer.learning_rate, trainer.max_grad_norm = 1e-4, 1.
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-4, betas=(.9, .95), weight_decay=.01)
    trainer.scheduler = trainer._build_scheduler("cosine", 30000, warmup_steps=1500)
    sample = {"video_latents": torch.zeros(16, 48, 3, 14, 28, dtype=torch.bfloat16),
              "action": torch.zeros(16, 32, 7), "proprio": torch.zeros(16, 32, 8),
              "sample_index": torch.arange(16)}
    trainer.train_loader = [sample, sample]
    trainer.save_checkpoint = lambda: (_ for _ in ()).throw(AssertionError("no checkpoint writes"))
    return trainer


def test_bounded_loop_uses_existing_optimizer_and_keeps_actual_30000_scheduler():
    trainer = _fixture_trainer()
    optimizer, scheduler = trainer.optimizer, trainer.scheduler
    initial = trainer.model.weight.detach().clone()
    calls = []
    original = optimizer.step

    @wraps(original)
    def count_step(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    optimizer.step = count_step
    losses, updates = PROBE.run_two_updates(trainer, synchronize=lambda: None)
    assert trainer.optimizer is optimizer and trainer.scheduler is scheduler
    assert trainer.max_steps == 30000 and trainer.global_step == trainer.batch_in_epoch == 2
    assert scheduler._milestones == [1500] and scheduler._schedulers[-1].T_max == 28500
    assert scheduler.last_epoch == 2 and len(calls) == len(losses) == len(updates) == 2
    assert PROBE.inspect_scheduler(SimpleNamespace(scheduler=scheduler)) == {
        "last_epoch": 2, "warmup_milestones": [1500], "cosine_steps": 28500,
    }
    assert not torch.equal(trainer.model.weight.detach(), initial)
    inspected = PROBE.inspect_adam(optimizer)
    assert inspected["adam_state_steps"] == [2.]
    assert inspected["adam_master_and_moments_finite"] is True
    assert all(update["batch_size"] == 16 and update["microbatches"] == 1 for update in updates)
    assert all(not update["optimizer_step_was_skipped"] for update in updates)


@pytest.mark.parametrize("malformation", ["rgb", "short_batch", "accumulation"])
def test_loop_rejects_rgb_short_batches_and_unsynchronized_accumulation(malformation):
    trainer = _fixture_trainer()
    if malformation == "rgb":
        trainer.train_loader[0]["video"] = torch.zeros(1)
    elif malformation == "short_batch":
        trainer.train_loader[0]["video_latents"] = torch.zeros(8, 48, 3, 14, 28)
    else:
        trainer.accelerator.sync_gradients = False
    with pytest.raises((ValueError, RuntimeError)):
        PROBE.run_two_updates(trainer, synchronize=lambda: None)
    assert trainer.global_step == 0 and not trainer.optimizer.state


def test_adam_inspector_detects_nonfinite_moments_and_unwraps_actual_optimizer():
    trainer = _fixture_trainer()
    PROBE.run_two_updates(trainer, synchronize=lambda: None)
    wrapper = SimpleNamespace(optimizer=SimpleNamespace(optimizer=trainer.optimizer))
    assert PROBE.unwrap_optimizer(wrapper) is trainer.optimizer
    state = next(iter(trainer.optimizer.state.values()))
    state["exp_avg_sq"].fill_(float("nan"))
    assert PROBE.inspect_adam(trainer.optimizer)["adam_master_and_moments_finite"] is False


@pytest.mark.parametrize("bad_loss", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_loss_fails_before_backward_or_any_optimizer_update(bad_loss):
    trainer = _fixture_trainer()
    trainer.model.training_loss = lambda sample: (trainer.model.weight * bad_loss, {"loss_fixture": bad_loss})
    trainer.accelerator.backward = lambda loss: (_ for _ in ()).throw(AssertionError("must reject before backward"))
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        PROBE.run_two_updates(trainer, synchronize=lambda: None)
    assert trainer.global_step == 0 and trainer.scheduler.last_epoch == 0
    assert not trainer.optimizer.state and trainer.model.weight.grad is None


def test_skipped_optimizer_never_calls_native_adam_or_advances_scheduler():
    trainer = _fixture_trainer()
    adam = trainer.optimizer
    initial = trainer.model.weight.detach().clone()
    calls = []
    adam.step = lambda *args, **kwargs: calls.append(True)

    class SkippingOptimizer:
        optimizer = adam
        param_groups = adam.param_groups

        def step(self):
            trainer.accelerator.optimizer_step_was_skipped = True

        def zero_grad(self, **kwargs):
            adam.zero_grad(**kwargs)

    trainer.optimizer = SkippingOptimizer()
    losses, updates = PROBE.run_two_updates(trainer, synchronize=lambda: None)
    assert not calls and not adam.state and torch.equal(trainer.model.weight.detach(), initial)
    assert trainer.global_step == 2 and trainer.scheduler.last_epoch == 0
    assert len(losses) == len(updates) == 2 and all(update["optimizer_step_was_skipped"] for update in updates)
    receipt = _complete_receipt()
    receipt["ranks"][0]["optimizer_step_was_skipped"] = True
    receipt["ranks"][0]["update_timings"] = updates
    receipt["ranks"][0]["actual_adam_step_calls"] = 0
    receipt["ranks"][0]["adam_state_steps"] = []
    receipt["ranks"][0]["scheduler"] = PROBE.inspect_scheduler(trainer.scheduler)
    assert not PROBE.receipt_complete(receipt)


def _complete_receipt(world_size=4):
    launch = {"path": "/workspace/scripts/accelerate_configs/dedicated.yaml", "sha256": "b" * 64,
              "evidence": "ancestor_accelerate_command", "deepspeed_config_path": "/workspace/scripts/ds_configs/zero1.json",
              "deepspeed_config_sha256": "c" * 64}
    ranks = []
    for index in range(world_size):
        ranks.append({"rank": index, "world_size": world_size, "status": "complete", "cache_identity_sha256": "a" * 64,
                      "actual_engine_updates": 2, "trainer_updates": 2, "planned_schedule_steps": 30000,
                      "batch_per_rank": 16, "microbatches": 2, "optimizer_step_was_skipped": False,
                      "all_parameters_finite": True, "adam_master_and_moments_finite": True,
                      "adam_state_steps": [2.], "actual_adam_step_calls": 4, "expected_adam_step_calls": 4,
                      "scheduler": {"last_epoch": 2, "warmup_milestones": [1500], "cosine_steps": 28500},
                      "device": {"env_LOCAL_RANK": index, "accelerator_local_process_index": index,
                                 "current_cuda_device": index, "cuda_device_name": "NVIDIA H200",
                                 "chosen_factory_device": f"cuda:{index}", "CUDA_VISIBLE_DEVICES": "0,1,2,3"},
                      "launcher_config": deepcopy(launch), "losses": [{"loss": .2}, {"loss": .1}],
                      "update_timings": [{"update": update, "batch_size": 16, "microbatches": 1,
                                          "sample_indices": list(range(index * 16, (index + 1) * 16)),
                                          "optimizer_step_was_skipped": False, "global_loss": .1,
                                          "gradient_norm": 1., "seconds": 2., "global_components": {"fixture_loss": .1}}
                                         for update in (1, 2)]})
    return {"status": "complete", "world_size": world_size, "effective_batch": world_size * 16,
            "planned_schedule_steps": 30000, "actual_engine_updates": 2, "cache_identity_sha256": "a" * 64,
            "formal_training_started": False, "checkpoint_saved": False, "launcher_config": launch, "ranks": ranks}


@pytest.mark.parametrize("world_size", [4, 8])
def test_success_requires_exit_zero_complete_aggregate_and_no_failure_receipts(world_size):
    receipt = _complete_receipt(world_size)
    assert PROBE.receipt_complete(receipt) and PROBE.preflight_success(0, receipt)
    assert not PROBE.preflight_success(2, receipt)
    assert not PROBE.preflight_success(None, receipt)
    assert not PROBE.preflight_success(False, receipt)
    assert not PROBE.preflight_success(0, receipt, ["failure_rank_001.json"])
    assert not PROBE.receipt_complete(receipt, ["failure_rank_001.json"])
    receipt["status"] = "failed"
    assert not PROBE.preflight_success(0, receipt)


@pytest.mark.parametrize("world_size", [4, 8])
def test_each_global_update_has_distinct_samples_but_later_updates_may_reuse_them(world_size):
    receipt = _complete_receipt(world_size)
    assert PROBE.receipt_complete(receipt)
    for rank in receipt["ranks"]:
        rank["update_timings"][1]["sample_indices"] = list(range(
            ((rank["rank"] + 1) % world_size) * 16, (((rank["rank"] + 1) % world_size) + 1) * 16))
    receipt["ranks"][-1]["update_timings"][0]["sample_indices"][-1] = 34807
    assert PROBE.receipt_complete(receipt) and PROBE.preflight_success(0, receipt)
    assert all(len({sample for rank in receipt["ranks"]
                    for sample in rank["update_timings"][update_index]["sample_indices"]}) == world_size * 16
               for update_index in range(2))


@pytest.mark.parametrize("update_index", [0, 1])
@pytest.mark.parametrize("bad_indices", [
    None, 0, tuple(range(16)), list(range(15)), list(range(17)),
    [0] * 16, [0, *range(15)], [False, *range(1, 16)], [0., *range(1, 16)],
    ["0", *range(1, 16)], [-1, *range(1, 16)], [34808, *range(1, 16)],
    [float("nan"), *range(1, 16)],
])
def test_receipt_rejects_missing_malformed_out_of_range_or_locally_duplicate_sample_indices(update_index, bad_indices):
    receipt = _complete_receipt()
    receipt["ranks"][0]["update_timings"][update_index]["sample_indices"] = bad_indices
    assert not PROBE.receipt_complete(receipt)


@pytest.mark.parametrize("world_size", [4, 8])
@pytest.mark.parametrize("update_index", [0, 1])
def test_receipt_rejects_cross_rank_overlap_in_either_global_update(world_size, update_index):
    receipt = _complete_receipt(world_size)
    receipt["ranks"][1]["update_timings"][update_index]["sample_indices"][0] = 0
    assert len(set(receipt["ranks"][1]["update_timings"][update_index]["sample_indices"])) == 16
    assert not PROBE.receipt_complete(receipt) and not PROBE.preflight_success(0, receipt)


@pytest.mark.parametrize("world_size", [4., 8., "4", True, None])
def test_noninteger_world_size_rejects_without_raising(world_size):
    receipt = _complete_receipt()
    receipt["world_size"] = world_size
    assert not PROBE.receipt_complete(receipt)
    assert not PROBE.ranks_complete(receipt["ranks"], world_size, "a" * 64)


@pytest.mark.parametrize("mutation", [
    lambda receipt: receipt["ranks"].pop(),
    lambda receipt: receipt["ranks"].append(deepcopy(receipt["ranks"][0])),
    lambda receipt: receipt["ranks"][1].update(rank=0),
    lambda receipt: receipt["ranks"][1].update(rank="1"),
    lambda receipt: receipt["ranks"].__setitem__(0, None),
    lambda receipt: receipt.update(ranks=None),
    lambda receipt: receipt.update(cache_identity_sha256=None),
    lambda receipt: receipt.update(cache_identity_sha256="bad"),
    lambda receipt: receipt["ranks"][0].update(cache_identity_sha256="d" * 64),
    lambda receipt: receipt["ranks"][0].update(world_size=8),
    lambda receipt: receipt["ranks"][0].update(status="failed"),
    lambda receipt: receipt["ranks"][0].update(optimizer_step_was_skipped=True),
    lambda receipt: receipt["ranks"][0].pop("optimizer_step_was_skipped"),
    lambda receipt: receipt["ranks"][0].update(all_parameters_finite=False),
    lambda receipt: receipt["ranks"][0].update(adam_master_and_moments_finite=False),
    lambda receipt: receipt["ranks"][0].update(adam_state_steps=[]),
    lambda receipt: receipt["ranks"][0].update(adam_state_steps=[1.]),
    lambda receipt: receipt["ranks"][0].update(actual_adam_step_calls=0),
    lambda receipt: receipt["ranks"][0].update(actual_adam_step_calls=3),
    lambda receipt: receipt["ranks"][0].update(actual_engine_updates=1),
    lambda receipt: receipt["ranks"][0]["scheduler"].update(last_epoch=0),
    lambda receipt: receipt["ranks"][0]["losses"][0].update(loss=float("nan")),
    lambda receipt: receipt["ranks"][0]["update_timings"][0].update(global_loss=float("inf")),
    lambda receipt: receipt["ranks"][0]["update_timings"][0].update(optimizer_step_was_skipped=True),
    lambda receipt: receipt["ranks"][0]["device"].update(env_LOCAL_RANK=1),
    lambda receipt: receipt["ranks"][0]["device"].update(accelerator_local_process_index=1),
    lambda receipt: receipt["ranks"][0]["device"].update(current_cuda_device=1),
    lambda receipt: receipt["ranks"][0]["device"].update(chosen_factory_device="cuda:1"),
    lambda receipt: receipt["ranks"][0]["device"].update(CUDA_VISIBLE_DEVICES="4,5,6,7"),
    lambda receipt: receipt["ranks"][0]["launcher_config"].update(sha256="d" * 64),
    lambda receipt: receipt["ranks"][0]["launcher_config"].update(deepspeed_config_sha256="d" * 64),
    lambda receipt: receipt.pop("launcher_config"),
    lambda receipt: receipt.update(effective_batch=128),
    lambda receipt: receipt.update(planned_schedule_steps=2),
    lambda receipt: receipt.update(actual_engine_updates=1),
    lambda receipt: receipt.update(formal_training_started=True),
    lambda receipt: receipt.update(checkpoint_saved=True),
])
def test_incomplete_inconsistent_or_nonfinite_receipts_are_rejected(mutation):
    receipt = _complete_receipt()
    mutation(receipt)
    assert not PROBE.receipt_complete(receipt)


@pytest.mark.parametrize("receipt", [None, [], {}, {"status": "complete"}])
def test_malformed_receipts_reject_without_raising(receipt):
    assert not PROBE.receipt_complete(receipt)


def test_original_clipping_config_and_measured_zero_are_reported_without_mutation():
    config = json.loads((ROOT / "scripts/ds_configs/ds_zero1_config.json").read_text())
    original = deepcopy(config)
    report = PROBE.gradient_clipping_metadata(1., 0., config)
    assert report["configured_max_grad_norm"] == 1. and report["effective_gradient_clipping"] == 0.
    assert report["deepspeed_gradient_clipping_field_present"] is False
    assert report["deepspeed_gradient_clipping_field"] is None and report["recipe_change_applied"] is False
    assert config == original
    config["gradient_clipping"] = .5
    report = PROBE.gradient_clipping_metadata(1., .5, config)
    assert report["deepspeed_gradient_clipping_field_present"] is True
    assert report["deepspeed_gradient_clipping_field"] == .5


@pytest.mark.parametrize("world_size", [4, 8])
def test_actual_dedicated_launcher_yaml_and_deepspeed_json_hashes(world_size, monkeypatch):
    path = ROOT / f"scripts/accelerate_configs/accelerate_zero1_ds_{world_size}gpu.yaml"
    monkeypatch.setattr(PROBE, "ancestor_launcher_config", lambda: path)
    report = PROBE.launcher_identity(world_size, path)
    assert report["path"] == str(path) and report["sha256"] == PROBE._sha256(path)
    ds_path = ROOT / "scripts/ds_configs/ds_zero1_config.json"
    assert report["deepspeed_config_path"] == str(ds_path)
    assert report["deepspeed_config_sha256"] == PROBE._sha256(ds_path)
    assert report["evidence"] == "ancestor_accelerate_command"
    with pytest.raises(ValueError, match="Declared launcher YAML"):
        PROBE.launcher_identity(world_size, "/tmp/wrong.yaml")
    with pytest.raises(ValueError, match="dedicated launcher YAML"):
        PROBE.launcher_identity(8 if world_size == 4 else 4)
    monkeypatch.setattr(PROBE, "ancestor_launcher_config", lambda: None)
    assert PROBE.launcher_identity(world_size, path)["evidence"] == "explicit_cli"
    assert PROBE.launcher_identity(world_size)["evidence"] == "expected_dedicated_config"


@pytest.mark.parametrize("invocation", [
    ["accelerate", "launch", "--config_file", "dedicated.yaml"],
    ["python", "-m", "accelerate.commands.launch", "--config_file=dedicated.yaml"],
    ["python", "/venv/accelerate/commands/launch.py", "--config_file", "dedicated.yaml"],
])
def test_ancestor_detects_console_module_and_script_launch_without_exposing_argv(invocation, monkeypatch, tmp_path):
    fake_proc = tmp_path / "proc"
    process = fake_proc / "123"
    process.mkdir(parents=True)
    process.joinpath("cmdline").write_bytes(("\0".join(invocation + ["secret-token"]) + "\0").encode())
    process.joinpath("cwd").symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setattr(PROBE.os, "getppid", lambda: 123)
    monkeypatch.setattr(PROBE, "Path", lambda value: fake_proc if value == "/proc" else Path(value))
    assert PROBE.ancestor_launcher_config() == tmp_path / "dedicated.yaml"


@pytest.mark.parametrize("passed,exit_code", [(True, 0), (False, 2)])
def test_main_exit_code_reflects_completed_receipt_gate(tmp_path, monkeypatch, passed, exit_code):
    args = _arguments(tmp_path)
    monkeypatch.setattr(PROBE, "execute", lambda args: passed)
    monkeypatch.setattr("sys.argv", ["preflight_cached_gpu.py", "--config", args.config,
                                    "--output", args.output, "--execute-preflight"])
    with pytest.raises(SystemExit) as error:
        PROBE.main()
    assert error.value.code == exit_code


def test_main_writes_failed_rank_receipt_for_an_owned_probe_directory(tmp_path, monkeypatch):
    args = _arguments(tmp_path)

    def fail(execution_args):
        Path(execution_args.output).mkdir()
        execution_args._owns_output = True
        raise FloatingPointError("Nonfinite original FastWAM cached training loss")

    monkeypatch.setenv("RANK", "2")
    monkeypatch.setattr(PROBE, "execute", fail)
    monkeypatch.setattr("sys.argv", ["preflight_cached_gpu.py", "--config", args.config,
                                    "--output", args.output, "--execute-preflight"])
    with pytest.raises(FloatingPointError):
        PROBE.main()
    failures = tuple(Path(args.output).glob("failure_rank_*.json"))
    assert len(failures) == 1 and failures[0].name == "failure_rank_002.json"
    report = json.loads(failures[0].read_text())
    assert report["status"] == "failed" and report["rank"] == 2
    assert report["error_type"] == "FloatingPointError" and report["checkpoint_saved"] is False
    assert not PROBE.preflight_success(0, _complete_receipt(), failures)


def test_budget_counts_global_partition_states_slowest_rank_and_duplicate_final_write():
    ranks = [{"startup_seconds": startup, "update_timings": [{"seconds": 99}, {"seconds": warm}],
              "policy_checkpoint_bytes_estimate": 100, "state_module_bytes_estimate": 200,
              "local_adam_master_and_moments_bytes": 50}
             for startup, warm in [(10., 2.), (12., 3.), (11., 2.5), (9., 2.1)]]
    result = PROBE.budget_report(ranks, {"bytes_per_second": 100., "free_bytes": 4000},
                                 remaining_seconds=90200, probe_elapsed_seconds=20.)
    assert result["one_weights_and_state_checkpoint_bytes_estimate"] == 500
    assert result["retained_six_checkpoints_bytes_estimate"] == 3000
    assert result["checkpoint_write_calls"] == 7 and result["checkpoint_seconds_estimate"] == 35.
    assert result["training_seconds_estimate"] == 90096.
    assert result["startup_training_and_save_seconds_estimate"] == 90143.
    assert result["remaining_budget_after_preflight_seconds"] == 90180.
    assert result["retained_checkpoints_fit_free_disk_estimate"] is True
    assert result["fit_within_remaining_budget_estimate"] is True
    assert result["projection_is_verified"] is False
    assert PROBE.budget_report(ranks, {"bytes_per_second": None})["fit_within_remaining_budget_estimate"] is None


def test_bounded_io_measurement_removes_its_temporary_file(tmp_path):
    result = PROBE.io_probe(tmp_path, 1)
    assert result["bytes"] == 1024**2 and result["bytes_per_second"] > 0
    assert list(tmp_path.iterdir()) == []
