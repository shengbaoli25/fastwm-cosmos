"""CPU-only regression tests for the dense-teacher distributed launcher."""
import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_launcher():
    path = ROOT / 'scripts/dense/run_experiment.py'
    spec = importlib.util.spec_from_file_location('dense_run_experiment', path)
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
    path = ROOT / 'src/fastwam/cosmos/dense_cache.py'
    tree = ast.parse(path.read_text())
    for call in [node for node in ast.walk(tree) if isinstance(node, ast.Call)]:
        if isinstance(call.func, ast.Attribute) and call.func.attr == 'dumps':
            assert all(keyword.arg != 'flush' for keyword in call.keywords)


def test_dense_preflight_uses_hooks_before_deepspeed_clears_gradients():
    runtime = (ROOT / 'scripts/dense/runtime.py').read_text()
    path = ROOT / 'scripts/dense/preflight.py'
    tree = ast.parse(path.read_text())
    backward_calls = [node for node in ast.walk(tree)
                      if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute)
                      and node.func.attr == 'backward']
    assert 'ExpertGradientProbe' in path.read_text()
    assert 'register_hook' in runtime
    assert any(isinstance(node.func.value, ast.Attribute)
               and node.func.value.attr == 'accelerator'
               for node in backward_calls)
