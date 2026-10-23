"""Selected CUDA ordinals and unsupported GPU architectures must fail explicitly."""
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'deploy'))
from realman_deploy.device import validate_device


class Device:
    def __init__(self, kind, index=None):
        self.type, _, ordinal = kind.partition(':')
        self.index = int(ordinal) if ordinal else index

    def __str__(self):
        return self.type + (f':{self.index}' if self.index is not None else '')


class Tensor:
    def __matmul__(self, other):
        return self

    def __getitem__(self, key):
        return SimpleNamespace(item=lambda: 16)


def fake_torch():
    props = SimpleNamespace(name='GPU 1', major=9, minor=0, total_memory=1000)
    cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 0, device_count=lambda: 2,
        device=lambda device: nullcontext(), get_device_properties=Mock(return_value=props),
        is_bf16_supported=lambda: True, synchronize=Mock(), mem_get_info=Mock(return_value=(800, 1000)),
        get_arch_list=lambda: ['sm_90'])
    return SimpleNamespace(device=Device, cuda=cuda, bfloat16='bf16', ones=Mock(return_value=Tensor()),
                           __version__='test', version=SimpleNamespace(cuda='12.8'))


def test_checks_requested_gpu_not_always_device_zero(monkeypatch):
    torch = fake_torch()
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4,7')
    result = validate_device('cuda:1')
    assert result['device'] == result['selected_device'] == 'cuda:1'
    assert result['free_bytes'] == 800
    assert result['cuda_visible_devices'] == '4,7'
    assert str(torch.cuda.get_device_properties.call_args.args[0]) == 'cuda:1'
    assert str(torch.ones.call_args.kwargs['device']) == 'cuda:1'


def test_fails_before_allocation_for_invalid_ordinal_or_old_gpu(monkeypatch):
    torch = fake_torch()
    monkeypatch.setitem(sys.modules, 'torch', torch)
    with pytest.raises(ValueError, match='索引'):
        validate_device('cuda:2')
    torch.cuda.get_device_properties.return_value.major = 7
    with pytest.raises(RuntimeError, match='BF16'):
        validate_device('cuda:1')
    torch.ones.assert_not_called()


def test_cuda_kernel_failure_is_reported_as_wheel_compatibility(monkeypatch):
    torch = fake_torch()
    torch.ones.side_effect = RuntimeError('no kernel image')
    monkeypatch.setitem(sys.modules, 'torch', torch)
    with pytest.raises(RuntimeError, match='BF16 CUDA 自检失败'):
        validate_device('cuda:1')


def test_compile_caches_are_private_per_host_device_and_repository(tmp_path, monkeypatch):
    from realman_deploy.device import configure_compile_cache
    monkeypatch.setenv('FASTWAM_CACHE_ROOT', str(tmp_path / 'cache'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-a,GPU-b')
    monkeypatch.setattr('realman_deploy.device.platform.node', lambda: 'host-a')
    monkeypatch.setattr('realman_deploy.device.subprocess.run',
        lambda *args, **kwargs: SimpleNamespace(stdout='0, GPU-a\n1, GPU-b\n'))
    first = configure_compile_cache('cuda:0', tmp_path / 'repo-a')
    second = configure_compile_cache('cuda:1', tmp_path / 'repo-a')
    third = configure_compile_cache('cuda:0', tmp_path / 'repo-b')
    assert first['CUDA_CACHE_PATH'] != second['CUDA_CACHE_PATH']
    assert first['CUDA_CACHE_PATH'] != third['CUDA_CACHE_PATH']
    assert all(Path(path).is_dir() for path in first.values())
    assert sys.modules['os'].environ['TORCHINDUCTOR_CACHE_DIR'] == first['TORCHINDUCTOR_CACHE_DIR'] or \
        sys.modules['os'].environ['TORCHINDUCTOR_CACHE_DIR'] == third['TORCHINDUCTOR_CACHE_DIR']
