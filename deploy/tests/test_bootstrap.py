"""Installer dry-run is side-effect-free and architecture options are explicit."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def run(*args):
    return subprocess.run(['bash', str(ROOT / 'deploy/bootstrap.sh'), *args], text=True,
                         capture_output=True, check=False)


def test_keep_dry_run_preserves_vendor_environment_plan():
    result = run('--cuda', 'keep', '--python', sys.executable, '--in-place', '--dry-run')
    assert result.returncode == 0
    assert 'no Torch package replacement' in result.stdout
    assert sys.executable in result.stdout


def test_wheel_target_dry_run_and_invalid_combinations():
    result = run('--cuda', 'cu130', '--python', 'python3', '--dry-run')
    assert result.returncode == 0
    assert '2.9.1 / torchvision 0.24.1' in result.stdout
    assert run('--cuda', 'keep', '--torch-version', '2.9', '--vision-version', '0.24', '--dry-run').returncode == 2
    assert run('--torch-version', '2.9', '--dry-run').returncode == 2
    assert run('--cuda', 'cu999', '--dry-run').returncode == 2
