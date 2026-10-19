"""Check replay metrics and artifact verification against silent misdiagnosis."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/cosmos' / (name + '.py'))
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


replay = module('validate_joint8_local')
finish = module('finish_joint8_local')


def test_fit_uses_channel_validity_instead_of_counting_padded_targets():
    prediction = [[1., 10.], [3., 200.], [100., 300.]]
    target = torch.zeros(3, 2)
    result = replay.fit_metrics(prediction, target, [[True, True], [True, False], [False, False]])
    assert result['mae_per_channel'] == [2., 10.]
    assert result['valid_per_channel'] == [2, 1]
    assert result['rmse_per_channel'][0] == pytest.approx(5.**.5)


def test_nonfinite_outputs_and_misaligned_shapes_cannot_pass_parity():
    with pytest.raises(ValueError):
        replay.comparison(torch.zeros(2, 1), torch.zeros(1, 2))
    with pytest.raises(ValueError):
        replay.comparison([0., float('nan')], [0., 0.])
    with pytest.raises(ValueError):
        replay.require_equal([0., .01], [0., 0.], 'history')
    with pytest.raises(ValueError):
        replay.fit_metrics(torch.zeros(3, 2), torch.zeros(3, 1), torch.ones(3).bool())
    with pytest.raises(ValueError):
        replay.require_equal(torch.tensor([1.], dtype=torch.float64),
                             torch.tensor([1. + 1e-10], dtype=torch.float64), 'precise source')


def test_checkpoint_install_verifies_remote_hash_and_preserves_existing_file(tmp_path):
    source = tmp_path / 'source.pt'; source.write_bytes(b'new checkpoint')
    dest = tmp_path / 'weights/step_001000.pt'
    sha = finish.file_sha256(source)
    with pytest.raises(ValueError):
        finish.verified_copy(source, dest, '0' * 64, source.stat().st_size)
    assert not dest.exists()
    finish.verified_copy(source, dest, sha, source.stat().st_size)
    assert dest.read_bytes() == source.read_bytes()
    finish.verified_copy(source, dest, sha, source.stat().st_size)
    dest.write_bytes(b'different installed checkpoint')
    with pytest.raises(ValueError):
        finish.verified_copy(source, dest, sha, source.stat().st_size)
    assert dest.read_bytes() == b'different installed checkpoint'
