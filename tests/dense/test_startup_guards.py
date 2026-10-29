"""CPU-only regression tests for the dense-teacher distributed launcher."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_launcher():
    path = ROOT / 'scripts/dense/run_experiment.py'
    spec = importlib.util.spec_from_file_location('dense_run_experiment', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_runtime():
    spec = importlib.util.spec_from_file_location('dense_runtime', ROOT / 'scripts/dense/runtime.py')
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_teacher_weights_require_a_pre_staged_local_file():
    launcher = load_launcher()
    with pytest.raises(ValueError, match='network downloads are disabled'):
        launcher.resolve_teacher_weights('vjepa21', None)


def test_teacher_weights_are_verified_before_any_stage_starts(tmp_path, monkeypatch):
    launcher = load_launcher()
    payload = b'pre-staged V-JEPA test checkpoint'
    checkpoint = tmp_path / 'vjepa2_1_vitb_dist_vitG_384.pt'
    checkpoint.write_bytes(payload)
    pinned = dict(launcher.TEACHERS['vjepa21'])
    pinned['sha256'] = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(launcher.TEACHERS, 'vjepa21', pinned)
    assert launcher.resolve_teacher_weights('vjepa21', checkpoint) == checkpoint.resolve()

    checkpoint.write_bytes(b'mutated')
    with pytest.raises(ValueError, match='SHA256'):
        launcher.resolve_teacher_weights('vjepa21', checkpoint)


def test_dense_cache_progress_log_does_not_pass_flush_to_json_dumps():
    for relative in ('src/fastwam/cosmos/dense_cache.py', 'scripts/dense/preflight.py'):
        tree = ast.parse((ROOT / relative).read_text())
        for call in [node for node in ast.walk(tree) if isinstance(node, ast.Call)]:
            if isinstance(call.func, ast.Attribute) and call.func.attr == 'dumps':
                assert all(keyword.arg != 'flush' for keyword in call.keywords)


def test_all_stages_override_online_environment_settings(monkeypatch):
    launcher = load_launcher()
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        monkeypatch.setenv(name, '0')
    monkeypatch.setenv('WANDB_MODE', 'online')
    environment = launcher.stage_environment()
    assert all(environment[name] == '1'
               for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'))
    assert environment['WANDB_MODE'] == 'offline'


def test_vjepa_teacher_forward_uses_bf16_autocast_for_sdpa():
    source = (ROOT / 'src/fastwam/cosmos/dense_encoder.py').read_text()
    assert 'with torch.autocast(device_type=device.type, dtype=torch.bfloat16):' in source


def test_gradient_probe_observes_gradients_after_engine_zeroes_them():
    torch = pytest.importorskip('torch')
    runtime = load_runtime()
    modules = torch.nn.ModuleDict({name: torch.nn.Linear(2, 1, bias=False)
                                  for name in ('video', 'force', 'action')})
    model = SimpleNamespace(dit=modules)
    probe = runtime.ExpertGradientProbe(model)
    try:
        loss = sum(module(torch.ones(1, 2)).sum() for module in modules.values())
        loss.backward()
        modules.zero_grad(set_to_none=True)  # DeepSpeed clears grads before backward returns.
        assert all(parameter.grad is None for parameter in modules.parameters())
        assert probe.norms() == pytest.approx({name: 2 ** .5 for name in modules})
    finally:
        probe.close()
    assert not probe.handles
    assert all(not parameter._backward_hooks for parameter in modules.parameters())


def test_gradient_probe_rejects_missing_expert_gradients():
    torch = pytest.importorskip('torch')
    runtime = load_runtime()
    modules = torch.nn.ModuleDict({name: torch.nn.Linear(1, 1)
                                  for name in ('video', 'force', 'action')})
    probe = runtime.ExpertGradientProbe(SimpleNamespace(dit=modules))
    try:
        modules['video'](torch.ones(1, 1)).sum().backward()
        with pytest.raises(RuntimeError, match='force expert received no gradients'):
            probe.norms()
    finally:
        probe.close()


def test_hplus_configs_change_only_future_weight_and_output_names():
    from omegaconf import OmegaConf
    main = OmegaConf.load(ROOT / 'configs/experiments/dinov3_vith16plus_pluginsertion71_4h200_20k.yaml')
    control = OmegaConf.load(ROOT / 'configs/experiments/dinov3_vith16plus_pluginsertion71_no_future_4h200_20k.yaml')
    assert main.model.feature_dim == 1280
    assert main.teacher == main.data.train.expected_teacher == 'dinov3_vith16plus'
    assert main.model.lambda_future == 1 and control.model.lambda_future == 0
    control.model.lambda_future = main.model.lambda_future
    control.output_dir, control.wandb.name = main.output_dir, main.wandb.name
    assert OmegaConf.to_container(main) == OmegaConf.to_container(control)
    data = OmegaConf.load(ROOT / 'configs/data/dense_feature_joint8.yaml')
    model = OmegaConf.load(ROOT / 'configs/model/dense_feature_joint8.yaml')
    assert data.train.expected_teacher == main.teacher and model.feature_dim == 1280


def test_hplus_cache_reuse_and_runtime_bind_temporal_representation(tmp_path, monkeypatch):
    import hydra.utils
    from fastwam.cache.latent import cache_identity_sha256
    from fastwam.cosmos.jepa_video_cache import identity
    from fastwam.cosmos.dense_encoder import DINOV3_SOURCE_COMMIT

    launcher, runtime = load_launcher(), load_runtime()
    teacher = 'dinov3_vith16plus'
    pinned = launcher.TEACHERS[teacher]
    spec = identity(SimpleNamespace(source_sha='source', identity_sha256='numeric', lengths=[40] * 71),
        dict(name=teacher, weight_sha256=pinned['sha256'], upstream_commit=DINOV3_SOURCE_COMMIT,
             feature_dim=1280, video_input_size=224, video_grid=14))
    manifest = dict(status='complete', identity=spec, identity_sha256=cache_identity_sha256(spec), shards=[])
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    assert launcher.verify_reused_cache(tmp_path, teacher, 'source', pinned['sha256']) == manifest['identity_sha256']
    monkeypatch.setenv('DINOV3_LICENSE_ACCEPTED', '1')
    instantiated = []

    def instantiate(config, **kwargs):
        instantiated.append(config._target_)
        return SimpleNamespace(normalization={})

    monkeypatch.setattr(hydra.utils, 'instantiate', instantiate)
    config = ROOT / 'configs/experiments/dinov3_vith16plus_pluginsertion71_4h200_20k.yaml'
    args = (config, tmp_path / 'run', tmp_path, 'source', tmp_path, teacher, tmp_path, tmp_path, pinned['sha256'])
    cfg, _, _, _ = runtime.load_run(*args)
    assert cfg.model.manifest.temporal_pooling == 'adjacent_frame_mean_v1'
    assert cfg.model.manifest.temporal_slots[0] == [-4, 0]
    assert len(instantiated) == 2
    spec['temporal_pooling'] = 'official_video_tubelet2'
    manifest['identity_sha256'] = cache_identity_sha256(spec)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='cache identity'):
        launcher.verify_reused_cache(tmp_path, teacher, 'source', pinned['sha256'])
    with pytest.raises(ValueError, match='cache identity'):
        runtime.load_run(*args)
    assert len(instantiated) == 2
