"""RealSense acquisition isolated from model CUDA Graph capture.

CUDA-enabled librealsense wheels allocate/free device buffers in native
capture threads. Those calls invalidate process-global CUDA Graph captures.
Use spawn (never fork a CUDA process) and a bounded RGB queue per camera.
"""
from collections import deque
import math
import multiprocessing as mp
import os
from queue import Empty, Full
import threading
import time


def _capture_frames(spec, width, height, fps, stop, output):
    # Never wait for the feeder at exit if the consumer has already stopped.
    output.cancel_join_thread()
    def publish(meta, rgb=None, error=None):
        try:
            output.put_nowait((meta, rgb, error))
        except Full:
            pass  # bounded buffering; timestamps still expose any old frames

    serial = str(spec['serial'])
    while not stop.is_set():
        pipeline = None
        try:
            import numpy as np
            import pyrealsense2 as rs
            context = rs.context()
            devices = [d for d in context.query_devices()
                       if d.get_info(rs.camera_info.serial_number) == serial]
            if len(devices) != 1:
                raise RuntimeError(f"未找到相机 {spec.get('name', serial)} ({serial})")
            pipeline = rs.pipeline(context)
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
            pipeline.start(config)
            warmup = int(spec.get('warmup_frames', 15))
            while not stop.is_set():
                frames = pipeline.wait_for_frames(1000)
                received = time.monotonic()
                color = frames.get_color_frame()
                if not color:
                    continue
                image = np.asanyarray(color.get_data())
                if image.shape != (height, width, 3) or image.dtype != np.uint8:
                    raise RuntimeError(f"相机 {serial} 返回尺寸/dtype 不匹配")
                if warmup:
                    warmup -= 1
                    continue
                meta = dict(name=spec['name'], serial=serial, received=received,
                    device_timestamp_ms=float(color.get_timestamp()),
                    frame_number=int(color.get_frame_number()), color_order='RGB',
                    width=width, height=height, acquisition_pid=os.getpid())
                publish(meta, image.tobytes())
        except Exception as exc:
            publish({}, error=str(exc))
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
        stop.wait(float(spec.get('retry_delay_s', 2.0)))


class RealSenseCamera:
    """Serial-pinned RGB camera; only bytes/metadata cross the process boundary."""

    def __init__(self, spec, *, width, height, fps, max_age_s, worker=_capture_frames):
        self.spec = dict(spec)
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.max_age_s = float(max_age_s)
        self._worker = worker
        self._context = mp.get_context('spawn')
        self.stop = self._context.Event()
        self.lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self.thread = self.process = self.output = None
        self.rgb = None
        self.received = 0.0
        self.meta = {}
        self.frames = deque(maxlen=8)
        self.error = None

    def start(self):
        with self._lifecycle_lock:
            if self.process is not None:
                return
            self.stop.clear()
            with self.lock:
                self.rgb, self.received, self.meta = None, 0.0, {}
                self.frames.clear()
                self.error = None
            self.output = self._context.Queue(maxsize=2)
            self.process = self._context.Process(target=self._worker,
                args=(self.spec, self.width, self.height, self.fps, self.stop, self.output),
                name=f"realman-camera-{self.spec['name']}", daemon=True)
            try:
                self.process.start()
                self.thread = threading.Thread(target=self._receive,
                    args=(self.process, self.output),
                    name=f"realman-camera-ipc-{self.spec['name']}", daemon=True)
                self.thread.start()
            except Exception:
                self.close()
                raise

    def _accept(self, message):
        meta, rgb, error = message
        if error:
            with self.lock:
                self.error = str(error)
            return
        received = float(meta['received'])
        if (meta.get('name') != self.spec['name'] or str(meta.get('serial')) != str(self.spec['serial'])
                or meta.get('color_order') != 'RGB' or meta.get('width') != self.width
                or meta.get('height') != self.height or not math.isfinite(received)
                or received <= 0 or received > time.monotonic()
                or (self.process is not None and meta.get('acquisition_pid') != self.process.pid)
                or not isinstance(rgb, bytes) or len(rgb) != self.width * self.height * 3):
            raise ValueError('相机进程返回的 RGB/身份/时间戳不匹配')
        with self.lock:
            if received <= self.received:
                raise ValueError('相机时间戳未递增')
            self.rgb, self.received, self.meta = rgb, received, dict(meta)
            self.frames.append((received, rgb, dict(meta)))
            self.error = None

    def _receive(self, process, output):
        while not self.stop.is_set():
            try:
                self._accept(output.get(timeout=.1))
            except Empty:
                if not process.is_alive():
                    with self.lock:
                        self.error = f'相机进程退出 ({process.exitcode})'
                    return
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)

    def snapshot(self, anchor, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            eligible = [(stamp, rgb, dict(meta)) for stamp, rgb, meta in self.frames if stamp <= anchor]
            error = self.error
        if error:
            raise RuntimeError(error)
        if not eligible:
            raise RuntimeError(f"相机 {self.spec['name']} 缺少状态锚点之前的画面")
        received, rgb, meta = eligible[-1]
        age = anchor - received
        if not 0 <= age <= self.max_age_s or now - received > self.max_age_s:
            raise RuntimeError(f"相机 {self.spec['name']} 画面过期 ({max(age, 0) * 1000:.1f} ms)")
        meta['age_ms'] = round(age * 1000, 2)
        return rgb, meta

    def status(self):
        with self.lock:
            received, error = self.received, self.error
            frame_number = self.meta.get('frame_number')
        age = time.monotonic() - received if received else None
        with self._lifecycle_lock:
            alive = self.process is not None and self.process.is_alive()
            pid = self.process.pid if self.process else None
        ready = alive and age is not None and 0 <= age <= self.max_age_s and error is None
        return dict(name=self.spec.get('name'), serial=self.spec.get('serial'), ready=ready,
            age_ms=round(age * 1000, 2) if age is not None else None, frame_number=frame_number,
            error=None if ready else (error or '等待相机新帧'),
            acquisition='spawn_process', acquisition_pid=pid,
            source_format='RGB8', conversion='librealsense')

    def close(self):
        with self._lifecycle_lock:
            self.stop.set()
            if self.process is not None and self.process.pid is not None:
                self.process.join(timeout=2.0)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=1.0)
            if self.thread and self.thread.ident is not None and self.thread is not threading.current_thread():
                self.thread.join(timeout=.5)
            if self.output is not None:
                self.output.close()
                self.output.cancel_join_thread()
            self.thread = self.process = self.output = None
            with self.lock:
                self.rgb, self.received, self.meta = None, 0.0, {}
                self.frames.clear()
                self.error = '相机已关闭'
