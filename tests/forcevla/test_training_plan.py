"""Resolve the actual task and reject image/control protocol regressions."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("forcevla_plan", ROOT / "scripts/forcevla/prepare_training.py")
PLAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAN)


def resolved(task):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={task}", "output_dir=/tmp/test-only-output"])
    return OmegaConf.to_container(cfg, resolve=True)


def test_two_view_task_reuses_official_libero_image_recipe_and_frozen_baseline_model():
    cfg = resolved(PLAN.TASK)
    PLAN.validate_config(cfg)
    baseline = resolved("forcevla_baseline")
    official = resolved("libero_uncond_2cam224_1e-4")
    assert cfg["model"] == baseline["model"]
    for key in ("num_frames", "action_video_freq_ratio", "video_size", "concat_multi_camera", "global_sample_stride"):
        assert cfg["data"]["train"][key] == official["data"]["train"][key]
    for key in ("train_transforms", "val_transforms", "num_output_cameras"):
        assert cfg["data"]["train"]["processor"][key] == official["data"]["train"]["processor"][key]
    for key in ("delta_action_dim_mask", "norm_default_mode", "action_state_merger", "proprio_output_dim", "action_output_dim"):
        assert cfg["data"]["train"]["processor"][key] == baseline["data"]["train"]["processor"][key]


@pytest.mark.parametrize("change", ["cropped_wrist", "swapped", "fake_camera", "libero_delta", "shorter_target"])
def test_invalid_camera_control_or_schedule_changes_are_rejected(change):
    cfg = deepcopy(resolved(PLAN.TASK))
    if change == "cropped_wrist":
        cfg["data"]["train"]["video_size"] = [224, 224]
    elif change == "swapped":
        cfg["data"]["train"]["shape_meta"]["images"].reverse()
    elif change == "fake_camera":
        cfg["data"]["train"]["shape_meta"]["images"][1]["key"] = "cam_main"
    elif change == "libero_delta":
        cfg["data"]["train"]["processor"]["delta_action_dim_mask"]["default"][0] = True
    else:
        cfg["max_steps"] = 2176
    with pytest.raises(ValueError):
        PLAN.validate_config(cfg)


def test_plan_is_configuration_only_and_has_no_asset_or_gpu_success_claim(tmp_path):
    output = tmp_path / "plan.json"
    result = PLAN.prepare(tmp_path / "shared", tmp_path / "dataset", tmp_path / "weights.pt",
                          output, tmp_path / "fresh-run")
    assert output.exists() and output.with_suffix(".yaml").exists()
    assert result["status"] == "configuration_only" and result["training_started"] is False
    assert result["data_check"]["status"] == "not_checked"
    assert result["max_steps"] == 30000 and result["warmup_updates"] == 1500
    assert result["video_shape_BCTHW"] == [1, 3, 9, 224, 448]
    assert "not a real VAE execution" in result["latent_shape_status"]
    assert result["train_argv"][0] == "accelerate"
    with pytest.raises(FileExistsError):
        PLAN.prepare(tmp_path / "shared", tmp_path / "dataset", tmp_path / "weights.pt",
                     output, tmp_path / "fresh-run")


def test_export_preserves_existing_companion_yaml_and_rejects_colliding_suffix(tmp_path):
    output = tmp_path / "plan.json"
    yaml = output.with_suffix(".yaml")
    yaml.write_text("previous experiment config\n")
    with pytest.raises(FileExistsError):
        PLAN.prepare(tmp_path / "shared", tmp_path / "dataset", tmp_path / "weights.pt", output, tmp_path / "run")
    assert yaml.read_text() == "previous experiment config\n" and not output.exists()
    with pytest.raises(ValueError):
        PLAN.prepare(tmp_path / "shared", tmp_path / "dataset", tmp_path / "weights.pt", tmp_path / "other.yaml", tmp_path / "run")
