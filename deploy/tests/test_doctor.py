"""Migration diagnostics are read-only and never instantiate hardware or Torch."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'deploy'), str(ROOT / 'src')]
spec = importlib.util.spec_from_file_location('deployment_doctor', ROOT / 'deploy/doctor.py')
doctor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doctor)


def fixture_config(tmp_path):
    text = tmp_path / 'texts'
    text.mkdir()
    (text / 'context.safetensors').touch()
    pinned = json.loads((ROOT / 'src/fastwam/cosmos/reason1_weights.json').read_text())
    manifest = {'schema': 'cosmos-reason1-full-concat-v1', 'status': 'complete',
                'encoder': 'nvidia/Cosmos-Reason1-7B', 'feature_dim': 100352,
                'encoder_revision': pinned['revision'], 'encoder_files': pinned['weights'],
                'texts': [{'prompt': 'test task', 'path': 'context.safetensors',
                           'sha256': hashlib.sha256(b'').hexdigest()}]}
    (text / 'manifest.json').write_text(json.dumps(manifest))
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    (tmp_path / 'normalization.json').write_text((ROOT / 'deploy/normalization.example.json').read_text())
    (tmp_path / 'tokenizer.pth').touch()
    (tmp_path / 'base.pt').touch()
    (tmp_path / 'models').mkdir()
    (tmp_path / 'models/model.pt').touch()
    hardware = dict(robot_ip='192.168.1.18', robot_port=8080, expected_model='RM_75',
                    expected_force_type='6FB', cameras=[{'name': 'global', 'serial': '111'},
                    {'name': 'wrist', 'serial': '222'}])
    (tmp_path / 'hardware.json').write_text(json.dumps(hardware))
    config = dict(device='cuda:1', normalization_file='normalization.json', tokenizer='tokenizer.pth',
                  hardware_config='hardware.json', checkpoint_directory='models', task_prompt='test task',
                  sampling_profile='4416', execution_mode='async_relaxed', model_kwargs={
                  'upstream': str(ROOT / 'third_party/cosmos-predict2.5'), 'checkpoint': 'base.pt',
                  'text_cache': 'texts', 'expected_text_identity': digest})
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps(config))
    return path, config


def test_static_checks_resolve_config_paths_and_do_not_open_devices(tmp_path, monkeypatch):
    path, config = fixture_config(tmp_path)
    from realman_deploy.hardware import RealManHardwareAdapter
    def forbidden(*args, **kwargs):
        raise AssertionError('doctor must not open hardware or import torch')
    monkeypatch.setattr(RealManHardwareAdapter, '__init__', forbidden)
    monkeypatch.setattr(doctor.importlib.metadata, 'version', lambda name: 'test')
    monkeypatch.setattr(doctor, 'sha256_file', lambda path: {
        'base.pt': 'b2e993cea3e242bc72288612488fe91770a1c073c98c3fec6d610411ffc30646',
        'tokenizer.pth': '38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981',
        'context.safetensors': hashlib.sha256(b'').hexdigest()}[Path(path).name])
    monkeypatch.setitem(sys.modules, 'torch', None)
    result = doctor.inspect_config(path)
    assert all(check['status'] != 'error' for check in result['checks']), result
    assert result['requested_device'] == 'cuda:1'
    checks = {c['name']: c for c in result['checks']}
    assert checks['model_kwargs.text_cache']['detail'] == str(tmp_path / 'texts')
    assert checks['hardware_config_only']['detail']['connected'] is False
    assert checks['sampling']['detail']['video_steps'] == 4


def test_static_checks_catch_mismatched_text_and_missing_asset(tmp_path, monkeypatch):
    path, config = fixture_config(tmp_path)
    config['model_kwargs']['expected_text_identity'] = '0' * 64
    path.write_text(json.dumps(config))
    (tmp_path / 'tokenizer.pth').unlink()
    monkeypatch.setattr(doctor.importlib.metadata, 'version', lambda name: 'test')
    monkeypatch.setattr(doctor, 'sha256_file', lambda path: {
        'base.pt': 'b2e993cea3e242bc72288612488fe91770a1c073c98c3fec6d610411ffc30646',
        'tokenizer.pth': '38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981',
        'context.safetensors': hashlib.sha256(b'').hexdigest()}[Path(path).name])
    result = doctor.inspect_config(path)
    failures = {c['name'] for c in result['checks'] if c['status'] == 'error'}
    assert {'text_manifest', 'tokenizer'} <= failures


def test_camera_mapping_and_sampling_modes_are_checked(tmp_path, monkeypatch):
    path, config = fixture_config(tmp_path)
    config['sampling_profile'] = '161616'
    config['camera_order'] = ['wrist', 'global']
    path.write_text(json.dumps(config))
    monkeypatch.setattr(doctor.importlib.metadata, 'version', lambda name: 'test')
    monkeypatch.setattr(doctor, 'sha256_file', lambda path: {
        'base.pt': 'b2e993cea3e242bc72288612488fe91770a1c073c98c3fec6d610411ffc30646',
        'tokenizer.pth': '38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981',
        'context.safetensors': hashlib.sha256(b'').hexdigest()}[Path(path).name])
    failures = {c['name'] for c in doctor.inspect_config(path)['checks'] if c['status'] == 'error'}
    assert {'hardware_config_only', 'sampling'} <= failures


def test_missing_config_returns_failure_json_without_exception(tmp_path, capsys):
    assert doctor.main(['--config', str(tmp_path / 'absent.json')]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result['ok'] is False
    assert result['hardware_connected'] is False
    assert result['model_loaded'] is False


def test_explicitly_offline_deployment_reports_hardware_unverified(tmp_path):
    path, config = fixture_config(tmp_path)
    config['hardware_factory'] = None
    path.write_text(json.dumps(config))
    (tmp_path / 'hardware.json').write_text('{"action_space": "joint8"}')
    checks = {c['name']: c for c in doctor.inspect_config(path)['checks']}
    assert checks['hardware_config_only']['status'] == 'warning'
    assert 'hardware_factory' in checks['hardware_config_only']['detail']


@pytest.mark.parametrize('base_digest,valid', [
    ('565bbb2c9645737327983f4461e4d32627bba465b0a8dc26447edea144e1ff47', True),
    ('0' * 64, False),
])
def test_nonaction_official_base_is_supported_without_accepting_unknown_weights(tmp_path, monkeypatch, base_digest, valid):
    path, _ = fixture_config(tmp_path)
    monkeypatch.setattr(doctor.importlib.metadata, 'version', lambda name: 'test')
    monkeypatch.setattr(doctor, 'sha256_file', lambda path: {
        'base.pt': base_digest,
        'tokenizer.pth': '38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981',
        'context.safetensors': hashlib.sha256(b'').hexdigest()}[Path(path).name])
    monkeypatch.setitem(sys.modules, 'torch', None)
    checks = {c['name']: c for c in doctor.inspect_config(path)['checks']}
    assert (checks['official_cosmos_assets']['status'] == 'ok') is valid
    if valid:
        assert checks['official_cosmos_assets']['detail']['base_variant'] == 'base/post-trained'
