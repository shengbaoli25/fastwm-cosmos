import numpy as np
import pytest

from fastwam.ablation.pi05 import Joint8Inputs, Joint8Outputs, UPSTREAM_REVISION, _verify_binding, plan, settings, wrap_policy


def sample():
    return {
        "state": np.arange(8, dtype=np.float32) / 10,
        "force": np.full(6, 999, dtype=np.float32),
        "images": {"global": np.full((24, 32, 3), 10, np.uint8), "wrist": np.full((24, 32, 3), 20, np.uint8)},
        "prompt": "拔下插头并插入另一插座",
    }


def test_camera_mapping_force_exclusion_and_missing_camera_mask():
    data = sample()
    actual = Joint8Inputs()(data)
    assert np.all(actual["image"]["base_0_rgb"] == 10)
    assert np.all(actual["image"]["left_wrist_0_rgb"] == 20)
    assert not actual["image_mask"]["right_wrist_0_rgb"]
    assert not actual["image"]["right_wrist_0_rgb"].any()
    assert "force" not in actual
    np.testing.assert_array_equal(actual["state"], data["state"])


def test_lerobot_chw_float_image_matches_wire_rgb():
    data = sample()
    expected = Joint8Inputs()(data)
    data["images"] = {key: image.transpose(2, 0, 1).astype(np.float32) / 255 for key, image in data["images"].items()}
    actual = Joint8Inputs()(data)
    for key in expected["image"]:
        np.testing.assert_array_equal(actual["image"][key], expected["image"][key])


def test_training_transforms_do_not_mutate_dataset_targets():
    data = sample()
    data["actions"] = np.ones((32, 8), dtype=np.float32)
    transformed = Joint8Inputs()(data)
    transformed["actions"][:, :7] -= transformed["state"][:7]
    np.testing.assert_array_equal(data["actions"], np.ones((32, 8)))


def test_output_keeps_eighth_gripper_dimension_and_discards_padding():
    raw = np.arange(32 * 32).reshape(32, 32)
    actual = Joint8Outputs()({"actions": raw})["actions"]
    np.testing.assert_array_equal(actual, raw[:, :8])


def test_predictor_does_not_send_force_or_rescale_actions():
    expected = np.tile(sample()["state"], (32, 1))

    class Policy:
        def infer(self, data):
            assert set(data) == {"state", "images", "prompt"}
            return {"actions": expected}

    np.testing.assert_array_equal(wrap_policy(Policy())(sample()), expected)


@pytest.mark.parametrize("value", [np.full((32, 8), np.nan), np.ones((32, 7))])
def test_invalid_model_outputs_fail_closed(value):
    class Policy:
        def infer(self, _):
            return {"actions": value}

    with pytest.raises(ValueError):
        wrap_policy(Policy())(sample())


def test_plan_never_requires_dataset_weights_or_imports_jax(tmp_path):
    config = {"dataset_root": str(tmp_path / "missing-data"), "run_root": str(tmp_path / "missing-run")}
    result = plan(config)
    assert result["training_started"] is False
    assert not (tmp_path / "missing-run").exists()
    assert result["action_horizon"] == 32
    assert result["model_action_dim"] == 32


@pytest.mark.parametrize("overrides", [{"finetune_mode": "fake"}, {"batch_size": 0}, {"num_train_steps": -1}])
def test_invalid_recipe_rejected(overrides):
    with pytest.raises(ValueError):
        settings(overrides)


def test_deployment_binding_does_not_need_dataset_or_run_directory():
    config = {"dataset_manifest_sha256": "abc"}
    binding = {"schema": "rm71d78-pi05-training-v1", "upstream_revision": UPSTREAM_REVISION, "run_id": "1" * 32,
               "settings": settings(config), "dataset_repo_id": "rm71d78", "dataset_identity_sha256": "abc"}
    _verify_binding(config, binding)
    with pytest.raises(ValueError, match="manifest identity"):
        _verify_binding(dict(config, dataset_manifest_sha256="other"), binding)


def test_predictor_loader_uses_checkpoint_assets_without_training_paths(tmp_path, monkeypatch):
    """Exercise deployment plumbing; the model factory is explicitly stubbed."""
    import dataclasses
    import json
    import sys
    import types
    from fastwam.ablation import pi05
    from fastwam.ablation.contracts import file_sha256

    checkpoint = tmp_path / "copied-checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    norm = checkpoint / "assets" / "rm71d78" / "norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}", encoding="utf-8")
    binding = {"schema": "rm71d78-pi05-training-v1", "upstream_revision": UPSTREAM_REVISION, "run_id": "1" * 32,
               "settings": settings({}), "dataset_repo_id": "rm71d78", "dataset_identity_sha256": "abc",
               "normalization_sha256": file_sha256(norm), "formal_training_started": True}
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    (checkpoint / "assets" / "ablation_training_binding.json").write_text(json.dumps(binding), encoding="utf-8")

    @dataclasses.dataclass
    class Assets:
        assets_dir: str
        asset_id: str

    @dataclasses.dataclass
    class Data:
        assets: object = None

    @dataclasses.dataclass
    class Train:
        data: Data

    expected = np.zeros((32, 8), dtype=np.float32)
    captured = {}

    def create(config, path, **kwargs):
        captured.update(config=config, path=path, kwargs=kwargs)
        return types.SimpleNamespace(infer=lambda _: {"actions": expected})

    def make(config):
        assert config["dataset_repo_id"] == "rm71d78"
        assert "dataset_root" not in config and "run_root" not in config
        return Train(Data())

    monkeypatch.setattr(pi05, "_activate", lambda _: None)
    monkeypatch.setattr(pi05, "make_train_config", make)
    monkeypatch.setitem(sys.modules, "openpi.policies.policy_config", types.SimpleNamespace(create_trained_policy=create))
    monkeypatch.setitem(sys.modules, "openpi.training.config", types.SimpleNamespace(AssetsConfig=Assets))
    predictor = pi05.load_predictor({"training_binding_path": str(binding_path), "checkpoint_path": str(checkpoint),
                                    "normalization_path": str(norm), "dataset_manifest_sha256": "abc"})
    np.testing.assert_array_equal(predictor(sample()), expected)
    assert captured["config"].data.assets.assets_dir == str(checkpoint / "assets")
    changed = dict(binding, settings=dict(binding["settings"], finetune_mode="full"))
    (checkpoint / "assets" / "ablation_training_binding.json").write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="different pi05 training run"):
        pi05.load_predictor({"training_binding_path": str(binding_path), "checkpoint_path": str(checkpoint),
                            "normalization_path": str(norm), "dataset_manifest_sha256": "abc"})


def test_resume_rejects_other_run_and_upstream_step_zero_restart(tmp_path):
    import json
    from fastwam.ablation.pi05 import verify_resume_checkpoint

    binding = {"run_id": "first"}
    for step in (0, 5):
        for child in ("params", "train_state", "assets"):
            (tmp_path / str(step) / child).mkdir(parents=True)
        (tmp_path / str(step) / "assets" / "ablation_training_binding.json").write_text(json.dumps(binding))
        if step == 0:
            with pytest.raises(FileNotFoundError, match="No resumable"):
                verify_resume_checkpoint(tmp_path, binding)
    assert verify_resume_checkpoint(tmp_path, binding).name == "5"
    with pytest.raises(ValueError, match="different pi05 training run"):
        verify_resume_checkpoint(tmp_path, {"run_id": "other"})
