"""CPU CLI default/explicit-action boundaries; never queries hardware."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/forcevla/prepare_pixel_cache.py"


def test_default_plan_imports_no_torch_and_creates_no_output(tmp_path):
    output = tmp_path / "missing-output"
    script = """import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('plan imported torch')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = [sys.argv[1], '--raw-config', '/absent/TEST-only.yaml', '--pixel-cache-root', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", script, str(SCRIPT), str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["cuda_queried"] is plan["vae_constructed"] is plan["training_started"] is False
    assert plan["pixel_bytes"] == 41916628992 and not output.exists()


@pytest.mark.parametrize("flags", [["--execute-pix"], ["--audit-pix"], ["--execute-pixels", "--audit-pixels"]])
def test_execution_abbreviations_and_conflicting_modes_rejected(tmp_path, flags):
    output = tmp_path / "untouched"
    result = subprocess.run([sys.executable, str(SCRIPT), "--pixel-cache-root", str(output), *flags], capture_output=True, text=True)
    assert result.returncode == 2 and not output.exists()
