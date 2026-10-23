"""Driver stalls must not block HTTP status or accumulate probe processes."""
import io
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'deploy'))
from realman_deploy.cuda_status import CudaStatusProbe


def test_driver_timeout_is_reported_without_waiting_or_spawning_more_probes():
    now = [0.]
    probe = CudaStatusProbe(timeout=8, refresh_interval=30, clock=lambda: now[0])
    process = Mock()
    process.poll.return_value = None
    with patch('realman_deploy.cuda_status.subprocess.Popen', return_value=process) as spawn:
        assert probe.status()['checking'] is True
        now[0] = 9
        result = probe.status()
        assert result['available'] is False
        assert result['checking'] is False
        assert '超时' in result['error']
        now[0] = 1000
        assert probe.status() == result
        assert spawn.call_count == 1
        process.kill.assert_called_once()
        process.wait.assert_not_called()
        process.communicate.assert_not_called()
    probe.output.close()


def test_successful_probe_is_cached_without_importing_torch_in_web_process():
    now = [0.]
    probe = CudaStatusProbe(clock=lambda: now[0])
    process = Mock()
    process.poll.return_value = 0
    result = dict(available=True, device='test GPU', total_bytes=1234)
    output = io.BytesIO(json.dumps(result).encode())
    with patch('realman_deploy.cuda_status.subprocess.Popen', return_value=process) as spawn, \
         patch('realman_deploy.cuda_status.tempfile.TemporaryFile', return_value=output):
        assert probe.status()['checking'] is True
        assert probe.status() == {**result, 'checking': False}
        assert probe.status()['available'] is True
        assert spawn.call_count == 1
        assert output.closed


def test_failed_probe_reports_error_and_cannot_claim_cuda_is_available():
    probe = CudaStatusProbe(clock=lambda: 0.)
    process = Mock()
    process.poll.return_value = 1
    with patch('realman_deploy.cuda_status.subprocess.Popen', return_value=process), \
         patch('realman_deploy.cuda_status.tempfile.TemporaryFile', return_value=io.BytesIO(b'bad output')):
        probe.status()
        result = probe.status()
        assert result['available'] is False
        assert result['error']


def test_requested_device_is_forwarded_and_close_never_waits_for_driver():
    probe = CudaStatusProbe(device='cuda:2')
    process = Mock()
    process.poll.return_value = None
    with patch('realman_deploy.cuda_status.subprocess.Popen', return_value=process) as spawn:
        probe.status()
        command = spawn.call_args.args[0]
        assert command[3] == 'cuda:2'
        probe.close()
        probe.status()
        assert spawn.call_count == 1
        process.kill.assert_called_once()
        process.wait.assert_not_called()
