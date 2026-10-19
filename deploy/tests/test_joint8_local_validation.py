"""Check replay metrics and artifact verification against silent misdiagnosis."""
import importlib.util
import json
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
receipt_capture = module('capture_joint8_download_receipt')


@pytest.mark.parametrize('marker', [
    '[done] max_steps reached step=1000 weights=/remote/step_001000.pt',
    '                 INFO     | >>  max_steps reached step=1000       trainer.py:801',
])
def test_final_receipt_accepts_raw_and_rich_rendered_completion(tmp_path, marker):
    weight = tmp_path / 'short/training/checkpoints/weights/step_001000.pt'
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b'final saved weights')
    log = tmp_path / 'short/formal-watch/training.log'
    log.parent.mkdir(parents=True)
    log.write_text(marker + '\n')
    (tmp_path / 'short_exit.json').write_text('{"returncode": 0}')
    output = tmp_path / 'receipt.json'
    result = receipt_capture.capture(tmp_path, output, timeout=1)
    assert json.loads(output.read_text()) == result
    assert result['training_complete'] is True
    assert result['sha256'] == finish.file_sha256(weight)
    assert result['size_bytes'] == weight.stat().st_size


def test_progress_or_different_step_does_not_prove_final_save():
    for text in ['step=1000 loss=0.0676', 'max_steps reached step=10000',
                 'max_steps reached step=999', 'saving step_001000.pt']:
        assert not receipt_capture.final_save_completed(text)


def test_failed_exit_prevents_final_receipt_even_with_completion_marker(tmp_path):
    weight = tmp_path / 'short/training/checkpoints/weights/step_001000.pt'
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b'weights')
    log = tmp_path / 'short/formal-watch/training.log'
    log.parent.mkdir(parents=True)
    log.write_text('max_steps reached step=1000\n')
    (tmp_path / 'short_exit.json').write_text('{"returncode": 1}')
    output = tmp_path / 'receipt.json'
    with pytest.raises(RuntimeError, match='exited unsuccessfully'):
        receipt_capture.capture(tmp_path, output, timeout=1)
    assert not output.exists()


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
