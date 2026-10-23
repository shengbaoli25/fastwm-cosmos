"""A failed gripper shutdown must not skip arm/camera cleanup."""
import sys
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realman_deploy.hardware import RealManHardwareAdapter


def test_gripper_close_failure_still_cleans_other_components_and_can_retry():
    hardware = object.__new__(RealManHardwareAdapter)
    hardware.lock = threading.RLock()
    hardware.closed = False
    hardware.armed = hardware.connected = True
    hardware.cameras = [Mock(), Mock()]
    hardware.feedback = Mock()
    hardware.sdk = Mock()
    hardware.gripper = Mock()
    hardware.gripper.close.side_effect = [RuntimeError('worker still alive'), None]
    hardware._stop_pool = Mock()
    with pytest.raises(RuntimeError, match='worker still alive'):
        hardware.close()
    assert hardware.closed and not hardware.armed and not hardware.connected
    for camera in hardware.cameras:
        camera.close.assert_called_once()
    hardware.feedback.restore.assert_called_once()
    hardware.sdk.rm_delete_robot_arm.assert_called_once()
    hardware._stop_pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
    hardware.close()
    assert hardware.gripper.close.call_count == 2
    hardware.sdk.rm_delete_robot_arm.assert_called_once()
