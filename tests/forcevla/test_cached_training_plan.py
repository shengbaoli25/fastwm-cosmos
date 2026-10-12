"""CPU-only checks of the prepared four/eight-rank cached training recipe.

Scheduler tests use one scalar with no gradients; checkpoint tests use tiny
stand-ins. No pretrained model, CUDA, distributed process, or training job is
constructed or launched.
"""
from __future__ import annotations

import json
from copy import deepcopy
import importlib.util
import inspect
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import pytest
import torch
from accelerate.data_loader import BatchSamplerShard
from accelerate.scheduler import AcceleratedScheduler
from torch.utils.data import BatchSampler
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.trainer import Wan22Trainer
from fastwam.utils.samplers import ResumableEpochSampler


ROOT = Path(__file__).resolve().parents[2]


def _plan_module():
    spec = importlib.util.spec_from_file_location(
        "forcevla_cached_training_plan", ROOT / "scripts/forcevla/prepare_cached_training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolved(task):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={task}", "output_dir=/tmp/test-only-output"])
    return cfg


@pytest.mark.parametrize("num_processes", [4, 8])
def test_cached_task_resolves_complete_baseline_model_without_processor_interpolation(num_processes):
    from fastwam.cache import LatentCacheDataset

    task = f"forcevla_libero_2cam_cached_{num_processes}gpu_30k"
    cfg = _resolved(task)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    _plan_module().validate_config(resolved)
    baseline = OmegaConf.to_container(_resolved("forcevla_baseline"), resolve=True)
    assert resolved["model"] == baseline["model"]
    assert resolved["model"]["video_dit_config"]["num_layers"] == 30
    assert resolved["model"]["action_dit_config"]["num_layers"] == 30
    assert resolved["model"]["video_dit_config"]["video_attention_mask_mode"] == "first_frame_causal"
    assert resolved["model"]["proprio_dim"] == 8
    assert resolved["model"]["action_dit_config"]["action_dim"] == 7
    assert (resolved["batch_size"], resolved["gradient_accumulation_steps"],
            resolved["max_steps"], resolved["save_every"], resolved["eval_every"]) == (16, 1, 30000, 5000, 0)
    data = resolved["data"]["train"]
    assert data["_target_"] == "fastwam.cache.LatentCacheDataset"
    assert data["require_production"] is True
    assert "processor" not in data and "dataset_dirs" not in data
    allowed = set(inspect.signature(LatentCacheDataset.__init__).parameters) - {"self"}
    assert set(data) - {"_target_"} <= allowed


def test_actual_hydra_model_factory_receives_explicit_dimensions_and_strict_release(monkeypatch):
    from fastwam.baseline import runtime as baseline_runtime
    from fastwam.runtime import create_fastwam

    cfg = _resolved("forcevla_libero_2cam_cached_8gpu_30k")
    calls, loads = [], []

    def no_weight_factory(**kwargs):
        inspect.signature(create_fastwam).bind(**kwargs)
        calls.append(kwargs)
        model = torch.nn.Module()
        model.dit = torch.nn.Linear(1, 1)
        return model

    monkeypatch.setattr(baseline_runtime, "create_fastwam", no_weight_factory)
    monkeypatch.setattr(baseline_runtime, "load_pretrained_fastwam",
                        lambda model, path, digest: loads.append((path, digest)))
    actual = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cpu")
    assert len(calls) == 1 and len(loads) == 1
    args = calls[0]
    assert args["proprio_dim"] == 8
    assert args["video_dit_config"]["action_dim"] == args["action_dit_config"]["action_dim"] == 7
    assert args["video_dit_config"]["num_layers"] == args["action_dit_config"]["num_layers"] == 30
    assert args["device"] == "cpu" and args["model_dtype"] == torch.bfloat16
    assert args["skip_dit_load_from_pretrain"] is True
    assert loads[0][1] == _plan_module().RELEASE_SHA
    assert callable(actual.optimizer_param_groups)


def test_actual_runtime_builds_only_cached_dataset_without_a_raw_processor(monkeypatch):
    from fastwam.cache import LatentCacheDataset
    from fastwam.runtime import build_datasets

    calls = []

    def fixture_init(self, cache_root, expected_identity_sha256=None,
                     max_cached_shards=2, require_production=True):
        calls.append((cache_root, expected_identity_sha256, max_cached_shards, require_production))

    monkeypatch.setattr(LatentCacheDataset, "__init__", fixture_init)
    cfg = _resolved("forcevla_libero_2cam_cached_8gpu_30k")
    train, val = build_datasets(cfg.data)
    assert isinstance(train, LatentCacheDataset) and val is train
    assert len(calls) == 1 and calls[0][3] is True


@pytest.mark.parametrize("change", ["batch", "accumulation", "horizon", "save", "rgb_reader", "mask", "layers"])
def test_cached_plan_rejects_recipe_and_architecture_regressions(change):
    cfg = deepcopy(OmegaConf.to_container(_resolved("forcevla_libero_2cam_cached_8gpu_30k"), resolve=True))
    if change in ("batch", "accumulation", "horizon", "save"):
        key, value = {"batch": ("batch_size", 8), "accumulation": ("gradient_accumulation_steps", 2),
                      "horizon": ("max_steps", 2176), "save": ("save_every", 500)}[change]
        cfg[key] = value
    elif change == "rgb_reader":
        cfg["data"]["train"]["processor"] = {"_target_": "must_not_decode_rgb"}
    elif change == "mask":
        cfg["model"]["video_dit_config"]["video_attention_mask_mode"] = "bidirectional"
    else:
        cfg["model"]["action_dit_config"]["num_layers"] = 3
    with pytest.raises(ValueError):
        _plan_module().validate_config(cfg)


@pytest.mark.parametrize("num_processes", [4, 8])
def test_exported_plan_pins_world_size_zero1_and_all_checkpoint_steps_without_launching(tmp_path, monkeypatch, num_processes):
    plan_module = _plan_module()

    def forbidden(*args, **kwargs):
        raise AssertionError("Preparing a plan must not construct a model or launch a subprocess.")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr("hydra.utils.instantiate", forbidden)
    output, run_output = tmp_path / "plan.json", tmp_path / "run"
    result = plan_module.prepare(tmp_path / "shared", tmp_path / "cache", tmp_path / "release.pt",
                                 output, run_output, num_processes=num_processes)
    assert result["status"] == "configuration_only" and result["training_started"] is False
    assert result["cache_check"]["status"] == "not_checked"
    assert result["cache_check"]["training_ready"] is False
    assert result["world_size"] == num_processes and result["per_gpu_batch_size"] == 16
    assert result["gradient_accumulation_steps"] == 1 and result["effective_batch"] == num_processes * 16
    assert result["max_steps"] == 30000 and result["warmup_updates"] == 1500
    assert result["nominal_max_sample_draws"] == num_processes * 16 * 30000
    assert result["periodic_checkpoint_steps"] == [5000, 10000, 15000, 20000, 25000, 30000]
    args = result["train_argv"]
    assert args[:2] == ["accelerate", "launch"]
    assert args[args.index("--num_processes") + 1] == str(num_processes)
    assert args[args.index("--num_machines") + 1] == "1"
    assert args[args.index("--machine_rank") + 1] == "0"
    assert shlex.split(result["train_shell"]) == args
    accelerate_config = OmegaConf.to_container(OmegaConf.load(ROOT / args[args.index("--config_file") + 1]))
    assert accelerate_config["distributed_type"] == "DEEPSPEED"
    assert accelerate_config["num_processes"] == num_processes and accelerate_config["num_machines"] == 1
    ds_path = ROOT / accelerate_config["deepspeed_config"]["deepspeed_config_file"]
    ds = json.loads(ds_path.read_text())
    assert ds["zero_optimization"]["stage"] == 1
    assert all(ds[key] == "auto" for key in ("train_batch_size", "train_micro_batch_size_per_gpu", "gradient_accumulation_steps"))
    assert output.is_file() and output.with_suffix(".yaml").is_file()
    assert result["resolved_yaml_sha256"] == plan_module.sha256(output.with_suffix(".yaml"))
    assert not run_output.exists()
    with pytest.raises(FileExistsError):
        plan_module.prepare(tmp_path / "shared", tmp_path / "cache", tmp_path / "release.pt",
                            output, run_output, num_processes=num_processes)


@pytest.mark.parametrize("changed_key", [None, "vae_weights_sha256", "dataset_stats_sha256"])
def test_verified_cache_is_bound_to_exact_manifest_and_pinned_component_provenance(tmp_path, monkeypatch, changed_key):
    import fastwam.cache as cache_module

    plan_module = _plan_module()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    manifest = {
        "identity": {"scope": "production", "sample_count": 34808,
                     "vae_weights_sha256": plan_module.VAE_SHA,
                     "dataset_stats_sha256": plan_module.STATS_SHA,
                     "numeric_dtype": "float32", "episode_lengths": [696] * 49 + [704]},
        "identity_sha256": "a" * 64,
    }
    if changed_key is not None:
        manifest["identity"][changed_key] = "b" * 64
    (cache_root / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(cache_module, "validate_cache", lambda root: manifest)
    output = tmp_path / "plan.json"
    if changed_key is not None:
        with pytest.raises(ValueError):
            plan_module.prepare(tmp_path / "shared", cache_root, tmp_path / "release.pt",
                                output, tmp_path / "run", check_cache=True)
        assert not output.exists() and not output.with_suffix(".yaml").exists()
    else:
        result = plan_module.prepare(tmp_path / "shared", cache_root, tmp_path / "release.pt",
                                     output, tmp_path / "run", check_cache=True)
        assert result["cache_check"]["identity_sha256"] == "a" * 64
        assert result["config"]["data"]["train"]["expected_identity_sha256"] == "a" * 64
        assert "data.train.expected_identity_sha256=" + "a" * 64 in result["train_argv"]
        assert result["cache_contract"]["target_dtype"] == "float32"


@pytest.mark.parametrize("num_processes", [4, 8])
def test_explicit_horizon_is_30000_optimizer_updates_for_distributed_processes(num_processes):
    trainer = object.__new__(Wan22Trainer)
    trainer.max_steps = 30000
    trainer.batch_size = 16
    trainer.gradient_accumulation_steps = 1
    trainer.num_epochs = 10
    trainer.train_dataset = range(34808)
    trainer.accelerator = SimpleNamespace(num_processes=num_processes)
    assert trainer._estimate_total_train_steps() == 30000
    # The explicit horizon must override the much smaller epoch estimate.
    trainer.max_steps = None
    assert trainer._estimate_total_train_steps() == (5440 if num_processes == 4 else 2720)


@pytest.mark.parametrize("num_processes,updates_per_epoch", [(4, 544), (8, 272)])
def test_actual_accelerate_sharding_retains_batch16_and_every_anchor(num_processes, updates_per_epoch):
    source = range(34808)
    sampler = ResumableEpochSampler(source, seed=42, batch_size=16, num_processes=num_processes)
    batches = BatchSampler(sampler, batch_size=16, drop_last=False)
    rank_batches = [list(BatchSamplerShard(batches, num_processes=num_processes, process_index=rank,
                                         split_batches=False, even_batches=True))
                    for rank in range(num_processes)]
    assert all(len(rank) == updates_per_epoch for rank in rank_batches)
    assert all(len(batch) == 16 for rank in rank_batches for batch in rank)
    drawn = [index for rank in rank_batches for batch in rank for index in batch]
    assert len(drawn) == 34816 and set(drawn) == set(source)
    assert len(drawn) - len(set(drawn)) == 8  # Stock even-batches final padding.
    for step in range(updates_per_epoch):
        assert sum(len(rank[step]) for rank in rank_batches) == num_processes * 16


def test_actual_scheduler_warms_up_1500_then_cosines_to_30000_without_world_size_multiplier(monkeypatch):
    import accelerate.scheduler as scheduler_module

    state_reads = []

    def eight_rank_state():
        state_reads.append(True)
        return SimpleNamespace(num_processes=8)

    monkeypatch.setattr(scheduler_module, "AcceleratorState", eight_rank_state)
    trainer = object.__new__(Wan22Trainer)
    scalar = torch.nn.Parameter(torch.zeros(1))
    trainer.learning_rate = 1e-4
    trainer.optimizer = torch.optim.AdamW([scalar], lr=trainer.learning_rate)
    scheduler = trainer._build_scheduler("cosine", total_train_steps=30000, warmup_steps=1500)
    wrapped = AcceleratedScheduler(scheduler, trainer.optimizer,
                                  step_with_optimizer=False, split_batches=False)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4 / 1500)
    rates = {}
    for update in range(1, 30001):
        trainer.optimizer.step()  # No gradient: the scalar never changes.
        wrapped.step()
        if update in (1, 1499, 1500, 1501, 30000):
            rates[update] = trainer.optimizer.param_groups[0]["lr"]
            assert scheduler.last_epoch == update
    assert rates[1] < rates[1499] < rates[1500]
    assert rates[1500] == pytest.approx(1e-4)
    assert rates[1501] < rates[1500]
    assert rates[30000] == pytest.approx(1e-6)
    assert not state_reads  # Step-with-optimizer=False avoids the x8 branch.
    assert torch.equal(scalar.detach(), torch.zeros(1)) and scalar.grad is None


@pytest.mark.parametrize("is_main_process", [True, False])
def test_stock_checkpoint_saves_main_rank_weights_and_every_rank_recoverable_state(tmp_path, is_main_process):
    events = []

    class FixtureModel:
        def save_checkpoint(self, path, optimizer, step):
            events.append(("weights", path, optimizer, step))
            Path(path).write_text(json.dumps({"step": step}))

    model = FixtureModel()

    class FixtureAccelerator:
        def wait_for_everyone(self):
            events.append(("barrier",))

        def unwrap_model(self, wrapped):
            assert wrapped is model
            return wrapped

        def save_state(self, output_dir):
            events.append(("state", output_dir))
            Path(output_dir, "fixture_optimizer_and_scheduler.txt").write_text("saved\n")

    accelerator = FixtureAccelerator()
    accelerator.is_main_process = is_main_process
    trainer = object.__new__(Wan22Trainer)
    trainer.model = model
    trainer.accelerator = accelerator
    trainer.global_step, trainer.epoch, trainer.batch_in_epoch = 5000, 18, 104
    trainer.weights_dir = str(tmp_path / "checkpoints" / "weights")
    trainer.state_dir = str(tmp_path / "checkpoints" / "state")
    Path(trainer.weights_dir).mkdir(parents=True)
    result = trainer.save_checkpoint()
    state = tmp_path / "checkpoints" / "state" / "step_005000"
    assert result["state_path"] == str(state)
    assert (state / "fixture_optimizer_and_scheduler.txt").is_file()
    assert sum(event[0] == "barrier" for event in events) == 3
    assert sum(event[0] == "state" for event in events) == 1
    if is_main_process:
        weights = tmp_path / "checkpoints" / "weights" / "step_005000.pt"
        assert result["weights_path"] == str(weights)
        assert json.loads(weights.read_text()) == {"step": 5000}
        assert json.loads((state / "trainer_state.json").read_text()) == {
            "global_step": 5000, "epoch": 18, "batch_in_epoch": 104,
        }
        assert next(event for event in events if event[0] == "weights")[2:] == (None, 5000)
    else:
        assert result["weights_path"] is None
        assert not (state / "trainer_state.json").exists()
        assert not any(event[0] == "weights" for event in events)
