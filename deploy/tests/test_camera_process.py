"""Camera IPC/lifecycle regressions without opening hardware or running a model."""
import multiprocessing as mp
import os
import sys
import time
from unittest.mock import patch

import pytest

from realman_deploy.camera import RealSenseCamera


def _message(spec, width, height, *, received=None):
    return (dict(name=spec['name'], serial=spec['serial'],
        received=time.monotonic() if received is None else received,
        width=width, height=height, color_order='RGB', frame_number=17,
        device_timestamp_ms=1234.5, acquisition_pid=os.getpid(),
        start_method=mp.get_start_method(), driver_imported='pyrealsense2' in sys.modules),
        bytes(range(width * height * 3)), None)


def _fake_capture(spec, width, height, fps, stop, output):
    output.cancel_join_thread()
    output.put(_message(spec, width, height))
    stop.wait(10)


def _exit_capture(spec, width, height, fps, stop, output):
    output.cancel_join_thread()


def _stuck_capture(spec, width, height, fps, stop, output):
    output.cancel_join_thread()
    output.put(_message(spec, width, height))
    time.sleep(30)


def _camera(worker=_fake_capture):
    return RealSenseCamera(dict(name='global', serial='test-serial'),
        width=2, height=1, fps=30, max_age_s=.12, worker=worker)


def _wait(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('camera worker timed out')


def test_spawn_preserves_rgb_and_capture_timestamp_and_can_restart():
    camera = _camera()
    try:
        camera.start()
        _wait(lambda: camera.received > 0)
        process = camera.process
        assert camera.status()['acquisition_pid'] == process.pid != os.getpid()
        received, rgb, meta = camera.frames[-1]
        assert meta['start_method'] == 'spawn'
        assert meta['driver_imported'] is False
        assert meta['received'] == received
        assert meta['device_timestamp_ms'] == 1234.5
        assert rgb == bytes(range(6))
        assert camera.snapshot(received + .01, received + .02)[0] == rgb
        camera.start()
        assert camera.process is process
        camera.close()
        assert not process.is_alive()
        assert not camera.frames
        assert camera.status()['ready'] is False
        camera.start()
        _wait(lambda: camera.received > 0)
        assert camera.process.pid != process.pid
        assert camera.received > received
    finally:
        camera.close()
        camera.close()


def test_worker_exit_is_reported_and_close_reaps_process():
    camera = _camera(_exit_capture)
    try:
        camera.start()
        process = camera.process
        _wait(lambda: camera.error is not None)
        assert '进程退出' in camera.status()['error']
        assert not camera.status()['ready']
        with pytest.raises(RuntimeError, match='进程退出'):
            camera.snapshot(time.monotonic())
    finally:
        camera.close()
    assert not process.is_alive()


def test_close_terminates_unresponsive_capture_worker():
    camera = _camera(_stuck_capture)
    camera.start()
    process, thread = camera.process, camera.thread
    try:
        _wait(lambda: camera.received > 0)
    finally:
        camera.close()
    assert not process.is_alive()
    assert not thread.is_alive()


def test_start_failure_cleans_queue_and_allows_retry():
    camera = _camera()
    with patch('multiprocessing.process.BaseProcess.start', side_effect=RuntimeError('spawn failed')):
        with pytest.raises(RuntimeError, match='spawn failed'):
            camera.start()
    assert camera.process is camera.thread is camera.output is None
    try:
        camera.start()
        _wait(lambda: camera.received > 0)
    finally:
        camera.close()


@pytest.mark.parametrize('change', [dict(serial='wrong'), dict(name='wrist'),
    dict(width=3), dict(height=2), dict(color_order='BGR'),
    dict(received=float('nan')), dict(received=float('inf')),
    dict(received=-1), dict(received=time.monotonic() + 3600)])
def test_rejects_bad_identity_size_color_and_timestamp(change):
    camera = _camera()
    meta, rgb, _ = _message(camera.spec, 2, 1)
    meta.update(change)
    with pytest.raises(ValueError):
        camera._accept((meta, rgb, None))
    assert not camera.frames


def test_rejects_bad_bytes_wrong_process_and_nonmonotonic_frames():
    camera = _camera()
    meta, rgb, _ = _message(camera.spec, 2, 1)
    for bad in (rgb[:-1], bytearray(rgb)):
        with pytest.raises(ValueError):
            camera._accept((meta, bad, None))
    camera.process = type('Process', (), dict(pid=os.getpid() + 1))()
    with pytest.raises(ValueError):
        camera._accept((meta, rgb, None))
    camera.process = None
    camera._accept((meta, rgb, None))
    with pytest.raises(ValueError, match='未递增'):
        camera._accept((meta, rgb, None))


def test_history_is_bounded_selects_before_anchor_and_rejects_stale_data():
    camera = _camera()
    base = time.monotonic() - 1
    for index in range(20):
        camera._accept(_message(camera.spec, 2, 1, received=base + index * .01))
    assert len(camera.frames) == 8
    rgb, meta = camera.snapshot(base + .185, base + .20)
    assert rgb == bytes(range(6))
    assert meta['received'] == base + .18
    with pytest.raises(RuntimeError, match='锚点之前'):
        camera.snapshot(base + .1, base + .20)
    with pytest.raises(RuntimeError, match='过期'):
        camera.snapshot(base + .5, base + .5)
    camera._accept(({}, None, 'USB disconnected'))
    with pytest.raises(RuntimeError, match='USB disconnected'):
        camera.snapshot(base + .19, base + .20)
