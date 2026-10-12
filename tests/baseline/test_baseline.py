import importlib.util
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fastwam.baseline.analysis import action_metrics, select_five_windows
from fastwam.baseline.analysis import file_sha256
from fastwam.baseline.evaluation import infer_unconditioned
from fastwam.baseline.optimizer import bounded_optimizer_param_groups
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.trainer import Wan22Trainer

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/baseline" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(config_name="train", overrides=["task=forcevla_baseline"])


def test_original_architecture_and_matching_data_recipe_are_resolved():
    cfg = config()
    assert cfg.model.video_dit_config.num_layers == cfg.model.action_dit_config.num_layers == 30
    assert cfg.model.video_dit_config.action_conditioned is False
    assert cfg.model.mot_checkpoint_mixed_attn is False
    assert cfg.model.video_scheduler.train_shift == 5
    assert cfg.model.action_scheduler.train_shift == 1
    assert cfg.data.train._target_.endswith(".RobotVideoDataset")
    assert cfg.data.train.processor._target_.endswith(".FastWAMProcessor")
    assert cfg.data.train.action_video_freq_ratio == 4
    assert cfg.data.train.num_frames == 33
    assert list(cfg.data.train.video_size) == [224, 224]
    assert cfg.data.train.processor.norm_default_mode == "z-score"
    assert not any(cfg.data.train.processor.delta_action_dim_mask.default)
    assert cfg.data.train.val_set_proportion == 0
    assert cfg.batch_size * cfg.gradient_accumulation_steps == 16
    assert cfg.learning_rate == 1e-4 and cfg.weight_decay == 1e-2
    assert cfg.model.skip_dit_load_from_pretrain is True
    assert cfg.model.require_pretrained_fastwam is True
    assert cfg.data.train.processor.proprio_output_dim == 8
    assert cfg.data.train.processor.action_state_merger.state_target_dim == 8
    assert cfg.max_steps == 2176 and cfg.save_every == 0


def test_full_meta_model_preserves_parameters_and_bounds_all_zero_flats():
    cfg = config()
    video_cfg = OmegaConf.to_container(cfg.model.video_dit_config, resolve=True)
    action_cfg = OmegaConf.to_container(cfg.model.action_dit_config, resolve=True)
    with torch.device("meta"):
        model = torch.nn.Module()
        model.dit = MoT({"video": WanVideoDiT(**video_cfg), "action": ActionDiT(**action_cfg)})
        model.proprio_encoder = torch.nn.Linear(8, 4096)
    parameters = [*model.dit.parameters(), *model.proprio_encoder.parameters()]
    assert sum(p.numel() for p in parameters) == 6020710599
    groups = bounded_optimizer_param_groups(model, parameters)
    assert len(groups) == 7
    assert all(sum(p.numel() for p in group["params"]) <= 1_000_000_000 for group in groups)
    assert {id(p) for group in groups for p in group["params"]} == {id(p) for p in parameters}


def test_grouping_matches_official_adamw_and_warmup_cosine_bitwise():
    model = torch.nn.Module()
    model.dit = torch.nn.ParameterList(torch.nn.Parameter(torch.linspace(-.4, .7, n)) for n in (8, 13, 10, 4, 12))
    model.proprio_encoder = torch.nn.Linear(2, 2)
    reference = deepcopy(model)
    parameters = [*model.dit.parameters(), *model.proprio_encoder.parameters()]
    expected_parameters = [*reference.dit.parameters(), *reference.proprio_encoder.parameters()]
    optimizer = torch.optim.AdamW(bounded_optimizer_param_groups(model, parameters, max_group_numel=25),
                                lr=1e-4, weight_decay=1e-2, betas=(.9, .95), foreach=True)
    expected = torch.optim.AdamW(expected_parameters, lr=1e-4, weight_decay=1e-2, betas=(.9, .95), foreach=True)
    schedulers = []
    for opt in (optimizer, expected):
        trainer = object.__new__(Wan22Trainer)
        trainer.optimizer, trainer.learning_rate = opt, 1e-4
        schedulers.append(trainer._build_scheduler("cosine", 128, 6))
    groups = optimizer.param_groups
    for step in range(8):
        for actual, target in zip(parameters, expected_parameters):
            gradient = torch.sin(torch.arange(actual.numel()).reshape(actual.shape) + step)
            actual.grad, target.grad = gradient.clone(), gradient.clone()
        for group in groups:
            optimizer.param_groups = [group]
            optimizer.step()
        optimizer.param_groups = groups
        expected.step()
        for scheduler in schedulers:
            scheduler.step()
        for actual, target in zip(parameters, expected_parameters):
            torch.testing.assert_close(actual, target, atol=0, rtol=0)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(optimizer.state[actual][key], expected.state[target][key], atol=0, rtol=0)


def tiny_model():
    cfg = config()
    video_cfg = OmegaConf.to_container(cfg.model.video_dit_config, resolve=True)
    video_cfg.update(hidden_dim=16, ffn_dim=32, num_heads=2, attn_head_dim=8, num_layers=3, freq_dim=8, text_dim=8)
    action_cfg = dict(action_dim=7, hidden_dim=12, ffn_dim=24, num_heads=2, attn_head_dim=8,
                      num_layers=3, freq_dim=8, text_dim=8, eps=1e-6)
    model = object.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.video_expert = WanVideoDiT(**video_cfg)
    model.action_expert = ActionDiT(**action_cfg)
    model.mot = MoT({"video": model.video_expert, "action": model.action_expert})
    return model


def test_original_mask_prevents_future_video_leak_across_real_mot_layers():
    torch.manual_seed(42)
    model = tiny_model()
    video = torch.randn(1, 48, 3, 2, 2)
    action = torch.randn(1, 4, 7)
    mask = model._build_mot_attention_mask(3, 4, 1, torch.device("cpu"))
    assert mask[3:, 0].all() and not mask[3:, 1:3].any()
    assert not mask[:3, 3:].any() and not mask[0, 1:3].any()
    kwargs = dict(latents_action=action, timestep_video=torch.tensor([.3]), timestep_action=torch.tensor([.6]),
                  context=torch.randn(1, 2, 8), context_mask=torch.ones(1, 2, dtype=torch.bool),
                  attention_mask=mask, fuse_vae_embedding_in_latents=True)
    reference_video, reference_action = model._joint_denoise_core(latents_video=video, **kwargs)
    changed = video.clone()
    changed[:, :, 1:] = torch.randn_like(changed[:, :, 1:]) * 7
    _, new_action = model._joint_denoise_core(latents_video=changed, **kwargs)
    torch.testing.assert_close(reference_action, new_action, atol=0, rtol=0)
    _, observed_action = model._joint_denoise_core(latents_video=video + 2, **kwargs)
    assert not torch.allclose(reference_action, observed_action)
    changed_action = dict(kwargs, latents_action=action + 5)
    new_video, _ = model._joint_denoise_core(latents_video=video, **changed_action)
    torch.testing.assert_close(reference_video, new_video, atol=0, rtol=0)


def test_inference_adapter_never_passes_gt_future_action_video_or_proprio():
    calls = []
    class Model:
        def infer_joint(self, **kwargs):
            calls.append(kwargs)
            return {"action": None}
    sample = {"video": torch.randn(1, 3, 9, 16, 16), "action": torch.randn(1, 32, 7),
              "proprio": torch.randn(1, 32, 7), "context": torch.randn(1, 128, 4096),
              "context_mask": torch.ones(1, 128, dtype=torch.bool)}
    infer_unconditioned(Model(), sample, seed=0, steps=16)
    passed = calls[0]
    assert passed["action"] is None and passed["prompt"] is None
    assert passed["num_video_frames"] == 9 and passed["action_horizon"] == 32
    assert torch.equal(passed["input_image"], sample["video"][:, :, 0])
    assert torch.equal(passed["proprio"], sample["proprio"][:, 0])


def test_fixed_selection_is_complete_and_sequence_windows_do_not_overlap():
    starts = np.arange(50) * 700
    ends = starts + 700
    selected = select_five_windows(starts, ends)
    assert [row["episode_ordinal"] for row in selected] == [0, 12, 24, 36, 49]
    assert all(row["frame_index"] == 349 for row in selected)
    chunks = select_five_windows(starts, ends, mode="sequence")
    for episode in (0, 12, 24, 36, 49):
        frames = [row["frame_index"] for row in chunks if row["episode_ordinal"] == episode]
        assert all(b - a == 32 for a, b in zip(frames, frames[1:]))
        assert frames[-1] + 32 < 700
    with pytest.raises(ValueError):
        select_five_windows(starts[:49], ends[:49])


def test_raw_metrics_exclude_padding_and_do_not_clip_gt():
    target = np.zeros((32, 7))
    target[:31, 0] = 12
    target[-1] = 100
    padding = np.zeros(32, dtype=bool)
    padding[-1] = True
    metrics = action_metrics(np.zeros_like(target), target, padding)
    assert metrics["valid_steps"] == 31
    assert metrics["mae_source_units_per_axis"][0] == 12
    assert metrics["xyz_mean_axis_mae_mm"] == 4000
    with pytest.raises(ValueError):
        action_metrics(np.full_like(target, np.nan), target, padding)


def test_checkpoint_requires_exact_step_and_complete_original_mot_weights(tmp_path):
    evaluator = load_script("evaluate")
    model = SimpleNamespace(mot=torch.nn.Linear(2, 2), proprio_encoder=torch.nn.Linear(2, 3))
    payload = {"step": 512, "mot": model.mot.state_dict(), "proprio_encoder": model.proprio_encoder.state_dict()}
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    evaluator.load_exact_checkpoint(model, path, 512)
    with pytest.raises(ValueError):
        evaluator.load_exact_checkpoint(model, path, 256)
    payload["mot"].pop("weight")
    torch.save(payload, path)
    with pytest.raises(RuntimeError):
        evaluator.load_exact_checkpoint(model, path, 512)


def test_full_policy_initialization_is_strict_and_sha_pinned(tmp_path):
    from fastwam.baseline.runtime import load_pretrained_fastwam
    model = SimpleNamespace(mot=torch.nn.Linear(2, 2), proprio_encoder=torch.nn.Linear(8, 3))
    path = tmp_path / "release.pt"
    payload = {"step": 21700, "mot": model.mot.state_dict(), "proprio_encoder": model.proprio_encoder.state_dict()}
    torch.save(payload, path)
    digest = file_sha256(path)
    load_pretrained_fastwam(model, path, digest)
    assert model.pretrained_fastwam_metadata["source_step"] == 21700
    with pytest.raises(ValueError):
        load_pretrained_fastwam(model, path, "f" * 64)
    payload["proprio_encoder"]["weight"] = torch.ones(3, 7)
    torch.save(payload, path)
    with pytest.raises(RuntimeError):
        load_pretrained_fastwam(model, path, file_sha256(path))


def test_zero_pad_happens_after_source7_normalization_and_preserves_order():
    from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
    from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
    raw = torch.arange(21, dtype=torch.float32).reshape(3, 7)
    normalizer = SingleFieldLinearNormalizer({"mean": torch.arange(7, dtype=torch.float32), "std": torch.ones(7) * 2}, mode="z-score")
    normalized = normalizer.forward(raw)
    merger = ConcatLeftAlign(state_target_dim=8)
    merger.set_shape_meta({"state": [{"key": "default", "shape": 7}], "action": [{"key": "default", "shape": 7}]})
    merged = merger.forward({"state": {"default": normalized.clone()}})
    torch.testing.assert_close(merged["state"][:, :7], normalized, atol=0, rtol=0)
    assert torch.equal(merged["state"][:, 7], torch.zeros(3))
    assert merged["state_dim_is_pad"].tolist() == [False] * 7 + [True]


@pytest.mark.parametrize("loss,status", [("1.3", "complete"), ("nan", "failed")])
def test_runner_checks_last_log_even_if_child_exits_before_next_poll(tmp_path, loss, status):
    runner = load_script("run_fit")
    command = [sys.executable, "-c", f"print('[train] step=1/1 loss={loss} loss_action=1.0 lr=1e-4', flush=True)"]
    result = runner.run_process(command, tmp_path / "log", 10, dict(os.environ), guard_loss=True)
    assert result["status"] == status
    if status == "failed":
        assert result["reason"] == "nonfinite_training_log"


def test_runner_kills_its_process_group_when_budget_expires(tmp_path):
    runner = load_script("run_fit")
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    result = runner.run_process(command, tmp_path / "log", .1, dict(os.environ))
    assert result["status"] == "failed" and result["reason"] == "budget_timeout"
    assert result["exit_code"] < 0


@pytest.mark.parametrize("line", ["step=1/1 loss=nan lr=1e-4", "[train] step=1/1 loss=1.0 loss_action=inf lr=1e-4",
                                  "\x1b[31mstep=1/1 loss=nan\x1b[0m lr=1e-4"])
def test_nan_guard_recognizes_rich_tag_removal_and_ansi(line):
    assert load_script("run_fit").has_nonfinite_training_log(line)


@pytest.mark.parametrize("changed", ["scripts/train.py", "scripts/accelerate_configs/accelerate_zero1_ds.yaml",
                                     "scripts/ds_configs/ds_zero1_config.json"])
def test_source_identity_tracks_real_training_entrypoint_and_deepspeed_recipe(tmp_path, changed):
    from fastwam.baseline.analysis import source_identity
    for name in ("scripts/train.py", "scripts/accelerate_configs/accelerate_zero1_ds.yaml", "scripts/ds_configs/ds_zero1_config.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
    before = source_identity(tmp_path)
    (tmp_path / changed).write_text("modified")
    after = source_identity(tmp_path)
    assert before[changed] != after[changed]
