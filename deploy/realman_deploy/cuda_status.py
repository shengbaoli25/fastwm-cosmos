"""Probe CUDA outside the web process: a broken driver can hold Python's GIL."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time


PROBE = '''
import json, sys
try:
    sys.path.insert(0, sys.argv[2])
    from realman_deploy.device import validate_device
    result = validate_device(sys.argv[1])
    result = dict(result, available=True, device=result['name'])
except Exception as exc:
    result = dict(available=False, selected_device=sys.argv[1], error=str(exc))
print(json.dumps(result), flush=True)
'''


class CudaStatusProbe:
    def __init__(self, *, device="cuda", timeout=8.0, refresh_interval=30.0, clock=time.monotonic):
        self.device = str(device)
        self.timeout = timeout
        self.refresh_interval = refresh_interval
        self.clock = clock
        self.lock = threading.Lock()
        self.process = None
        self.output = None
        self.started_at = 0.0
        self.next_check = 0.0
        self.timed_out = False
        self.closed = False
        self.result = {"available": False, "checking": True}

    def status(self) -> dict:
        with self.lock:
            if self.closed:
                return dict(self.result)
            now = self.clock()
            if self.process is not None:
                code = self.process.poll()
                if code is None:
                    if not self.timed_out and now - self.started_at >= self.timeout:
                        self.timed_out = True
                        self.result = {"available": False, "checking": False,
                            "error": "CUDA 检查超时：NVIDIA 驱动无响应；硬件连接可单独进行，推理暂不可用"}
                        self.next_check = now + self.refresh_interval
                        # SIGKILL cannot immediately reap a driver call in D state.
                        # Never wait for it here or spawn another probe while it lives.
                        try:
                            self.process.kill()
                        except OSError:
                            pass
                    return dict(self.result)
                if not self.timed_out:
                    try:
                        self.output.seek(0)
                        result = json.loads(self.output.read(65536))
                        if code or not isinstance(result, dict) or type(result.get("available")) is not bool:
                            raise ValueError("CUDA 检查进程异常退出")
                        self.result = {**result, "checking": False}
                    except (ValueError, OSError) as exc:
                        self.result = {"available": False, "checking": False, "error": str(exc)}
                    self.next_check = now + self.refresh_interval
                self.output.close()
                self.output = self.process = None
            if self.process is None and now >= self.next_check:
                try:
                    self.output = tempfile.TemporaryFile()
                    self.process = subprocess.Popen([sys.executable, "-c", PROBE, self.device,
                        str(Path(__file__).resolve().parents[1])],
                        stdout=self.output, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
                    self.started_at = now
                    self.timed_out = False
                    self.result = {"available": False, "checking": True}
                except OSError as exc:
                    if self.output is not None:
                        self.output.close()
                    self.output = self.process = None
                    self.result = {"available": False, "checking": False, "error": str(exc)}
                    self.next_check = now + self.refresh_interval
            return dict(self.result)

    def close(self) -> None:
        """Stop the probe without waiting on a potentially wedged driver call."""
        with self.lock:
            self.closed = True
            if self.process is not None and self.process.poll() is None:
                try:
                    self.process.kill()
                except OSError:
                    pass
            if self.output is not None:
                self.output.close()
                self.output = None


CUDA_STATUS = CudaStatusProbe()
