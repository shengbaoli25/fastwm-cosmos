"""CPU contracts; these tests do not claim real-model/GPU validation."""
import ast
import dataclasses
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from fastwam.ablation.forcevla import Joint8Inputs, Joint8Outputs, patch_model_source, run


def sample():
    return {"state": np.array([.1, -.2, .3, -.4, .5, -.6, .7, .8], np.float32),
            "force": np.array([11, 12, 13, 14, 15, 16], np.float32),
            "image": np.zeros((4, 5, 3), np.uint8),
            "wrist_image": np.ones((3, 4, 5), np.float32),
            "actions": np.tile([.2, .3, .4, .5, .6, .7, .8, .9], (32, 1))}


def test_joint8_and_force_have_disjoint_slots_and_gripper_is_retained():
    original = sample()
    out = Joint8Inputs()(original)
    np.testing.assert_array_equal(out["state"][:8], original["state"])
    np.testing.assert_array_equal(out["state"][8:14], original["force"])
    assert not np.any(out["state"][14:])
    np.testing.assert_allclose(out["actions"][:, :8], original["actions"])
    assert out["state"][7] != out["state"][8]
    assert out["image"]["left_wrist_0_rgb"].shape == (4, 5, 3)
    assert np.all(out["image"]["left_wrist_0_rgb"] == 255)
    assert not out["image_mask"]["right_wrist_0_rgb"]


def test_raw_output_preserves_absolute_joint8():
    expected = sample()["actions"].astype(np.float32)
    padded = np.pad(expected, ((0, 0), (0, 24)))
    result = Joint8Outputs()({"actions": padded})["actions"]
    np.testing.assert_array_equal(result, expected)
    assert not np.shares_memory(result, padded)


@pytest.mark.parametrize("field,value", [
    ("force", [1, 2, 3, 4, 5]), ("force", [1, 2, 3, 4, 5, np.nan]),
    ("state", np.ones(7)), ("state", [0, 0, 0, 0, 0, 0, 0, 1.1]),
    ("actions", np.ones((50, 8))), ("wrist_image", np.full((3, 4, 5), 2.0)),
])
def test_reject_mismatched_training_or_sensor_contract(field, value):
    data = sample()
    data[field] = value
    with pytest.raises(ValueError):
        Joint8Inputs()(data)


def test_nonfinite_outputs_are_rejected():
    actions = np.zeros((32, 32))
    actions[4, 5] = np.nan
    with pytest.raises(ValueError):
        Joint8Outputs()({"actions": actions})


def test_plan_never_loads_a_model_or_starts_training():
    result = run({}, "plan")
    assert result["training_started"] is False
    assert result["runtime_validation"] == "not_run"
    assert result["force_input"].startswith("normalized")
    assert result["num_train_steps"] == 30000
    assert result["batch_size"] == 256
    assert result["execution_profile"]["per_gpu_batch_size"] == 64


def test_unknown_upstream_source_is_rejected_before_execution():
    with pytest.raises(ValueError, match="reviewed upstream"):
        patch_model_source("import os\nraise RuntimeError('must never execute')\n")


@pytest.mark.skipif(not os.environ.get("FORCEVLA_SOURCE_ROOT"), reason="optional pinned source audit")
def test_pinned_official_source_compiles_and_fixes_cached_prefix_and_joint8():
    root = Path(os.environ["FORCEVLA_SOURCE_ROOT"])
    source = (root / "src/openpi/models/pi0_force.py").read_text(encoding="utf-8")
    patched = patch_model_source(source)
    tree = ast.parse(patched)
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Pi0_Guidance")
    embed = ast.unparse(next(n for n in model.body if getattr(n, "name", None) == "embed_suffix"))
    inference = ast.unparse(next(n for n in model.body if getattr(n, "name", None) == "sample_actions"))
    assert "observations.at[:, :8].set(obs.state[:, :8])" in embed
    assert "force_in_proj(obs.state[:, 8:14])" in embed
    assert "concatenate([prefix_out_fix, force_tokens]" in inference
    assert "concatenate([prefix_out, force_tokens]" not in inference
    compile(patched, "reviewed_forcevla_joint8.py", "exec")
    with pytest.raises(ValueError):
        patch_model_source(source + "\n# drift\n")


@pytest.fixture
def fake_official_runtime(tmp_path, monkeypatch):
    """Stub only the unavailable JAX backend; keep filesystem/hash checks real."""
    from fastwam.ablation import forcevla
    from fastwam.ablation.contracts import file_sha256

    def install_module(name, **values):
        module = types.ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    @dataclasses.dataclass(frozen=True)
    class DataFactory:
        repo_id: str

    class ModelConfig:
        def __init__(self, **values):
            self.__dict__.update(values)

        def get_freeze_filter(self):
            return "stub-freeze-filter"

    model_module = types.SimpleNamespace(Pi0_GuidanceConfig=ModelConfig)
    monkeypatch.setattr(forcevla, "_activate_upstream", lambda root: model_module)
    # If serving attempts a filesystem lookup of the dataset, fail the test.
    monkeypatch.setattr(forcevla, "_manifest", lambda _: pytest.fail("Serving read the training dataset"))
    install_module("openpi")
    install_module("openpi.training")
    install_module("openpi.training.config", DataConfigFactory=DataFactory,
                   DataConfig=lambda **kw: types.SimpleNamespace(**kw),
                   ModelTransformFactory=lambda: lambda model: "stub-model-transforms",
                   TrainConfig=lambda **kw: types.SimpleNamespace(**kw))
    install_module("openpi.training.weight_loaders", Pi0GuidanceWeightLoader=lambda value: value)
    install_module("openpi.transforms", make_bool_mask=lambda a, b: [True] * a + [False] * -b,
                   Group=lambda **kw: types.SimpleNamespace(**kw), RepackTransform=lambda x: x,
                   DeltaActions=lambda x: ("delta", x), AbsoluteActions=lambda x: ("absolute", x))
    install_module("openpi.shared")

    def decode_stats(text):
        return {key: types.SimpleNamespace(**{name: np.asarray(value) for name, value in row.items()})
                for key, row in json.loads(text)["norm_stats"].items()}

    install_module("openpi.shared.normalize", deserialize_json=decode_stats)
    calls = []

    def create_policy(train_config, checkpoint, sample_kwargs):
        data_config = train_config.data.create(None, train_config.model)
        calls.append((train_config, checkpoint, data_config, sample_kwargs))

        def infer(observation):
            prepared = Joint8Inputs()(observation)
            np.testing.assert_array_equal(prepared["state"][8:14], observation["force"])
            return {"actions": np.zeros((32, 8), np.float32)}

        return types.SimpleNamespace(infer=infer)

    install_module("openpi.policies")
    install_module("openpi.policies.policy_config", create_trained_policy=create_policy)
    stats = tmp_path / "copied-stats.json"
    stats.write_text(json.dumps({"norm_stats": {
        key: {"mean": [0] * 32, "std": [1] * 32} for key in ("state", "actions")
    }}), encoding="utf-8")
    checkpoint = tmp_path / "copied-step"
    (checkpoint / "params").mkdir(parents=True)
    checkpoint_stats = checkpoint / "assets" / "deleted-dataset" / "norm_stats.json"
    checkpoint_stats.parent.mkdir(parents=True)
    checkpoint_stats.write_bytes(stats.read_bytes())
    binding = {"upstream_revision": forcevla.UPSTREAM_REVISION,
               "dataset_manifest_sha256": "a" * 64,
               "normalization_sha256": file_sha256(stats),
               "dataset_root": str(tmp_path / "deleted-dataset"),
               "run_root": str(tmp_path / "deleted-training-run"), "training_method": "lora"}
    binding_path = tmp_path / "copied-binding.json"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    (checkpoint / "assets/ablation_training_binding.json").write_text(json.dumps(binding), encoding="utf-8")
    config = {"upstream_root": str(tmp_path / "stub-upstream"),
              "training_binding_path": str(binding_path), "checkpoint_path": str(checkpoint),
              "normalization_path": str(stats), "normalization_sha256": file_sha256(stats),
              "dataset_manifest_sha256": "a" * 64}
    return config, checkpoint_stats, calls


def test_predictor_loads_copied_assets_without_dataset_root(fake_official_runtime):
    from fastwam.ablation.forcevla import load_predictor
    config, _, calls = fake_official_runtime
    assert "dataset_root" not in config
    predict = load_predictor(config)
    assert len(calls) == 1
    train_config, _, data_config, _ = calls[0]
    assert train_config.model.action_horizon == 32
    assert train_config.model.paligemma_variant == "gemma_2b_lora"
    assert data_config.norm_stats["state"].mean.shape == (32,)
    data = sample()
    action = predict({"state": data["state"], "force": data["force"],
                      "images": {"global": data["image"], "wrist": data["wrist_image"]}})
    assert action.shape == (32, 8)


def test_predictor_checks_real_checkpoint_statistics_before_stub_model_load(fake_official_runtime):
    from fastwam.ablation.forcevla import load_predictor
    config, checkpoint_stats, calls = fake_official_runtime
    checkpoint_stats.write_text("wrong-run statistics", encoding="utf-8")
    with pytest.raises(ValueError, match="Checkpoint normalization"):
        load_predictor(config)
    assert calls == []


def test_predictor_rejects_changed_copied_normalization_file(fake_official_runtime):
    from fastwam.ablation.forcevla import load_predictor
    config, _, calls = fake_official_runtime
    Path(config["normalization_path"]).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="normalization"):
        load_predictor(config)
    assert calls == []


def test_predictor_checks_bound_hash_even_if_both_statistics_copies_changed(fake_official_runtime):
    from fastwam.ablation.forcevla import load_predictor
    config, checkpoint_stats, calls = fake_official_runtime
    Path(config["normalization_path"]).write_text("same tampered bytes", encoding="utf-8")
    checkpoint_stats.write_text("same tampered bytes", encoding="utf-8")
    with pytest.raises(ValueError, match="normalization hash mismatch"):
        load_predictor(config)
    assert calls == []


def test_predictor_rejects_different_dataset_binding(fake_official_runtime):
    from fastwam.ablation.forcevla import load_predictor
    config, _, calls = fake_official_runtime
    config["dataset_manifest_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="training binding mismatch"):
        load_predictor(config)
    assert calls == []


def test_predictor_rejects_another_run_even_when_normalization_is_identical(fake_official_runtime):
    from fastwam.ablation.forcevla import load_predictor
    config, _, calls = fake_official_runtime
    path = Path(config["training_binding_path"])
    binding = json.loads(path.read_text(encoding="utf-8"))
    binding["training_method"] = "full"
    path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="embedded training binding"):
        load_predictor(config)
    assert calls == []


def test_replay_real_http_with_explicitly_fake_model_only(tmp_path, monkeypatch):
    """Exercise replay transport/report logic, NOT real checkpoint inference."""
    import importlib.util
    import io
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    from fastwam.ablation.contracts import read_json
    from realman_force.contract import input_contract

    root = Path(__file__).resolve().parents[2]
    path = root / "scripts/ablation/replay.py"
    spec = importlib.util.spec_from_file_location("_replay_fake_backend_test", path)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    contract = input_contract(read_json(root / "deploy/force_calibration.json"))
    image = io.BytesIO()
    Image.new("RGB", (640, 480), "red").save(image, format="PNG")
    rgb_cell = {"bytes": image.getvalue(), "path": None}
    frame_count = 40
    action = [0, 0, 0, 0, 0, 0, 0, .5]
    table = {"index": list(range(frame_count)), "observation.state": [action] * frame_count,
             "observation.force": [[0] * 6] * frame_count, "action": [action] * frame_count,
             "observation.images.global": [rgb_cell] * frame_count,
             "observation.images.wrist": [rgb_cell] * frame_count}
    episode = tmp_path / "data/chunk-000/episode_000000.parquet"
    episode.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pydict(table), episode)
    manifest = {"identity_sha256": "fake-dataset-identity-for-test-only",
                "episodes": [{"source_episode": 0}]}
    monkeypatch.setattr(replay, "load_dataset_manifest", lambda _: manifest)
    fake_calls = []

    class ExplicitlyFakePolicy:
        def __init__(self, deployment):
            self.config = {"backend": "forcevla", "dataset_identity_sha256": manifest["identity_sha256"],
                           "identity_sha256": "fake-deployment-identity-for-test-only"}
            self.metadata = dict(camera_names=["global", "wrist"], width=640, height=480,
                                 action_space="joint8", action_horizon=32, control_hz=30,
                                 history_length=32, force_preprocessing=contract)

        def predict(self, observation, history, video_context):
            assert len(history) == 32
            assert len(observation["frames"]["global"]["data"]) == 640 * 480 * 3
            fake_calls.append(observation["sequence"])
            return [action] * 32

        def close(self):
            pass

    monkeypatch.setattr(replay, "AblationPolicy", ExplicitlyFakePolicy)
    monkeypatch.setattr(replay, "memory_report", lambda: {"provider": None, "reason": "FAKE MODEL UNIT TEST"})
    output = tmp_path / "fake-model-unit-test-only.json"
    result = replay.replay("unused-fake-deployment.json", tmp_path, 2, 1, output)
    assert len(fake_calls) == 3
    assert result["measured_samples"] == 2
    assert result["joint_mae_rad"] == 0
    assert result["gripper_mae"] == 0
    assert result["gpu_memory"]["reason"] == "FAKE MODEL UNIT TEST"
    assert result["http_ms_p50"] > 0
    assert read_json(output)["deployment_identity"] == "fake-deployment-identity-for-test-only"


def test_resume_rejects_step_zero_and_wrong_embedded_binding(tmp_path):
    from fastwam.ablation.forcevla import _check_resume_checkpoint
    contract = {"seed": 42, "dataset_manifest_sha256": "test-only"}

    def checkpoint(step, binding):
        path = tmp_path / str(step)
        (path / "train_state").mkdir(parents=True)
        (path / "assets").mkdir()
        (path / "assets/ablation_training_binding.json").write_text(json.dumps(binding), encoding="utf-8")

    checkpoint(0, contract)
    with pytest.raises(ValueError, match="step-0-only"):
        _check_resume_checkpoint(tmp_path, contract)
    checkpoint(1, contract)
    _check_resume_checkpoint(tmp_path, contract)
    checkpoint(2, dict(contract, seed=43))
    with pytest.raises(ValueError, match="embedded training binding"):
        _check_resume_checkpoint(tmp_path, contract)


def test_four_h200_profile_is_physical_batch64_without_accumulation():
    from fastwam.ablation.forcevla import execution_profile, verify_training_devices
    devices = [types.SimpleNamespace(platform="gpu", device_kind="NVIDIA H200") for _ in range(4)]
    profile = verify_training_devices({}, devices, 1)
    assert profile["global_batch_size"] == 256
    assert profile["per_gpu_batch_size"] == 64
    assert profile["gradient_accumulation_steps"] == 1
    assert profile["mesh_shape"] == [1, 4]
    assert profile["data_axes"] == ["batch", "fsdp"]
    assert profile["num_train_steps"] == 30000
    with pytest.raises(ValueError):
        execution_profile({"batch_size": 64})
    with pytest.raises(ValueError):
        execution_profile({"fsdp_devices": 3})
    with pytest.raises(RuntimeError, match="exactly 4"):
        verify_training_devices({}, devices[:2], 1)
    with pytest.raises(RuntimeError, match="one process"):
        verify_training_devices({}, devices, 4)
    with pytest.raises(RuntimeError, match="H200"):
        verify_training_devices({}, [types.SimpleNamespace(platform="gpu", device_kind="NVIDIA A100")] * 4, 1)


@pytest.mark.skipif(not os.environ.get("FORCEVLA_SOURCE_ROOT"), reason="optional pinned source audit")
def test_official_fsdp_mesh_shards_batch_over_all_four_devices():
    source = (Path(os.environ["FORCEVLA_SOURCE_ROOT"]) / "src/openpi/training/sharding.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    constants = [node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id in {"BATCH_AXIS", "FSDP_AXIS", "DATA_AXIS"}
                         for target in node.targets)]
    make_mesh = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_mesh")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             *constants, make_mesh], type_ignores=[])
    namespace = {"jax": types.SimpleNamespace(device_count=lambda: 4,
                                              make_mesh=lambda shape, axes: (shape, axes))}
    exec(compile(ast.fix_missing_locations(module), "official_sharding_source_test", "exec"), namespace)
    assert namespace["make_mesh"](4) == ((1, 4), ("batch", "fsdp"))
    assert namespace["DATA_AXIS"] == ("batch", "fsdp")
    assert 256 // np.prod(namespace["make_mesh"](4)[0]) == 64


def test_bounded_smoke_keeps_same_frozen_target_and_learning_rate_schedule():
    from fastwam.ablation.forcevla import _bounded_training_config

    @dataclasses.dataclass(frozen=True)
    class Config:
        num_train_steps: int
        lr_schedule: object
        checkpoint_dir: str

    schedule = object()
    original = Config(3, schedule, "separate-smoke-run")
    bounded = _bounded_training_config(original, 2)
    assert bounded.num_train_steps == 2 and original.num_train_steps == 3
    assert bounded.lr_schedule is schedule
    assert bounded.checkpoint_dir == original.checkpoint_dir
    assert _bounded_training_config(original, None) is original
    for bad in (0, 1, 4):
        with pytest.raises(ValueError, match="stop_after_steps"):
            _bounded_training_config(original, bad)
    with pytest.raises(ValueError, match="only supported"):
        run({}, "plan", stop_after_steps=2)
