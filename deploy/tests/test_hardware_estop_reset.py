"""Recovery tests never construct a real controller, camera, or gripper."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy"))
from realman_deploy.hardware import RealManHardwareAdapter, _RobotFeedback


@pytest.fixture
def hardware():
    config = json.loads((ROOT / "deploy/realman_hardware.example.json").read_text())
    config["force_preprocessing_required"] = False
    config["reset_timeout_s"] = .25
    with patch.object(RealManHardwareAdapter, "_connect"):
        adapter = RealManHardwareAdapter(config)
    adapter.connected = True
    adapter.estop_latched = True
    adapter.last_command = (1.,) * 8
    adapter.last_gripper = .7
    adapter.sdk = Mock()
    adapter.sdk.rm_get_current_arm_state.return_value = (0, {
        "err": {"err_len": 0, "err": []}, "joint": [0.] * 7, "pose": [0.] * 6})
    adapter.sdk.rm_get_arm_power_state.return_value = (0, 1)
    adapter.sdk.rm_get_joint_en_state.return_value = (0, [1] * 7)
    adapter.sdk.rm_get_joint_err_flag.return_value = dict(return_code=0, err_flag=[0] * 7, brake_state=[1] * 7)
    adapter.sdk.rm_get_arm_current_trajectory.return_value = dict(return_code=0, trajectory_type=0, data=[0.] * 7)
    adapter.sdk.rm_set_arm_stop.return_value = 0
    adapter.sdk.rm_set_arm_slow_stop.return_value = 0
    adapter.feedback = Mock()
    adapter.feedback.snapshot.side_effect = lambda _: ([0.] * 7, [0.] * 6, [0.] * 6, time.monotonic())
    adapter.feedback.status.return_value = dict(ready=True, error=None)
    adapter.feedback.force_contract = None
    adapter.gripper = Mock()
    adapter.gripper.status.return_value = dict(ready=False, error="夹爪报警32", alarm=32)
    adapter.cameras = [Mock(), Mock()]
    for i, camera in enumerate(adapter.cameras):
        camera.status.return_value = dict(ready=True, error=None, name=["global", "wrist"][i])
    yield adapter
    adapter._reset_pool.shutdown(wait=True, cancel_futures=True)
    adapter._stop_pool.shutdown(wait=True, cancel_futures=True)


def test_reset_releases_only_software_latch_and_keeps_gripper_alarm(hardware):
    result = hardware.reset_emergency_stop()
    assert not result["estop_latched"] and not result["armed"]
    assert not result["reset_busy"] and not result["stop_pending"]
    assert not result["gripper_ready"] and result["gripper"]["alarm"] == 32
    assert hardware.last_command is None and hardware.last_gripper is None
    hardware.gripper.hold.assert_called_once()
    hardware.gripper.command.assert_not_called()
    names = {call[0] for call in hardware.sdk.mock_calls}
    assert names == {"rm_get_current_arm_state", "rm_get_arm_power_state",
                     "rm_get_joint_en_state", "rm_get_joint_err_flag",
                     "rm_get_arm_current_trajectory"}
    assert hardware.sdk.rm_get_current_arm_state.call_count == 2
    with pytest.raises(RuntimeError, match="夹爪"):
        hardware.arm()


@pytest.mark.parametrize("method,value,message", [
    ("rm_get_current_arm_state", (-2, {}), "读取失败"),
    ("rm_get_current_arm_state", (0, {}), "缺失或格式未知"),
    ("rm_get_current_arm_state", (0, {"err": {"err_len": 1, "err": ["4107"]}}), "仍有报警"),
    ("rm_get_current_arm_state", (0, {"err": {"err_len": 0, "err": []}}), "位姿/关节"),
    ("rm_get_arm_power_state", (0, 0), "未上电"),
    ("rm_get_arm_power_state", (0, None), "状态未知"),
    ("rm_get_joint_en_state", (0, [1] * 6), "关节未全部使能"),
    ("rm_get_joint_en_state", (0, [1] * 6 + [0]), "关节未全部使能"),
    ("rm_get_joint_err_flag", dict(return_code=0, err_flag=[0] * 6 + [32]), "关节仍有报警"),
    ("rm_get_joint_err_flag", {}, "错误状态缺失"),
    ("rm_get_arm_current_trajectory", dict(return_code=0, trajectory_type=1), "仍有规划轨迹"),
    ("rm_get_arm_current_trajectory", dict(return_code=-2, trajectory_type=0), "仍有规划轨迹"),
    ("rm_get_arm_current_trajectory", {}, "仍有规划轨迹"),
])
def test_alarm_unknown_and_active_trajectory_keep_latch(hardware, method, value, message):
    getattr(hardware.sdk, method).return_value = value
    with pytest.raises(RuntimeError, match=message):
        hardware.reset_emergency_stop()
    assert hardware.estop_latched and not hardware.armed
    assert hardware.last_command is not None
    hardware.sdk.rm_set_arm_continue.assert_not_called()
    hardware.sdk.rm_set_arm_delete_trajectory.assert_not_called()


def test_controller_fault_appearing_on_second_read_keeps_latch(hardware):
    good = hardware.sdk.rm_get_current_arm_state.return_value
    hardware.sdk.rm_get_current_arm_state.side_effect = [good, (0, {"err": {"err_len": 1, "err": ["0x100B"]}})]
    with pytest.raises(RuntimeError, match="仍有报警"):
        hardware.reset_emergency_stop()
    assert hardware.estop_latched


def test_feedback_fault_after_final_controller_check_keeps_latch(hardware):
    hardware.feedback.snapshot.side_effect = [
        ([0.] * 7, [0.] * 6, [0.] * 6, time.monotonic() + .001),
        RuntimeError("反馈过期"),
    ]
    with pytest.raises(RuntimeError, match="反馈过期"):
        hardware.reset_emergency_stop()
    assert hardware.estop_latched and not hardware.armed


@pytest.mark.parametrize("stamp", [0., float("nan")])
def test_reset_requires_new_feedback_from_after_request(hardware, stamp):
    hardware.config["reset_timeout_s"] = .03
    hardware.feedback.snapshot.side_effect = None
    hardware.feedback.snapshot.return_value = ([0.] * 7, [0.] * 6, [0.] * 6, stamp)
    with pytest.raises(RuntimeError, match="超时|新鲜"):
        hardware.reset_emergency_stop()
    assert hardware.estop_latched


def test_estop_during_slow_status_check_remains_prompt_and_wins(hardware):
    entered, release = threading.Event(), threading.Event()
    good = hardware.sdk.rm_get_current_arm_state.return_value
    def blocked():
        entered.set()
        assert release.wait(2.)
        return good
    hardware.sdk.rm_get_current_arm_state.side_effect = blocked
    with ThreadPoolExecutor(max_workers=2) as pool:
        resetting = pool.submit(hardware.reset_emergency_stop)
        assert entered.wait(1.)
        stopping = pool.submit(hardware.emergency_stop)
        try:
            stopping.result(timeout=.2)
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="恢复已取消"):
            resetting.result(timeout=1.)
    hardware.sdk.rm_set_arm_stop.assert_called_once()
    assert hardware.estop_latched and not hardware.armed


def test_timeout_late_result_never_unlocks_and_duplicate_is_blocked(hardware):
    entered, release = threading.Event(), threading.Event()
    good = hardware.sdk.rm_get_current_arm_state.return_value
    def blocked():
        entered.set()
        assert release.wait(2.)
        return good
    hardware.sdk.rm_get_current_arm_state.side_effect = blocked
    hardware.config["reset_timeout_s"] = .03
    try:
        with pytest.raises(RuntimeError, match="超时"):
            hardware.reset_emergency_stop()
        assert entered.is_set()
        with pytest.raises(RuntimeError, match="仍在进行"):
            hardware.reset_emergency_stop()
    finally:
        release.set()
    hardware._reset_future.result(timeout=1.)
    assert hardware.estop_latched and not hardware.armed


def test_underlying_stop_blocks_reset_even_after_caller_timeout(hardware):
    release = threading.Event()
    def blocked_stop():
        assert release.wait(2.)
        return 0
    hardware.sdk.rm_set_arm_stop.side_effect = blocked_stop
    hardware.config["stop_timeout_s"] = .03
    try:
        with pytest.raises(RuntimeError, match="急停.*超时"):
            hardware.emergency_stop()
        assert hardware.status()["stop_pending"]
        with pytest.raises(RuntimeError, match="停止尚未完成"):
            hardware.reset_emergency_stop()
    finally:
        release.set()
    hardware._stop_pool.submit(lambda: None).result(timeout=1.)
    assert hardware._stop_pending == 0
    assert not hardware.reset_emergency_stop()["estop_latched"]


def test_gripper_hold_failure_cannot_prevent_arm_stop(hardware):
    hardware.gripper.hold.side_effect = RuntimeError("fake hold failure")
    with pytest.raises(RuntimeError, match="夹爪待发指令取消失败"):
        hardware.emergency_stop()
    hardware.sdk.rm_set_arm_stop.assert_called_once()
    assert hardware.estop_latched and hardware._stop_pending == 0


def test_missing_sdk_check_fails_closed(hardware):
    hardware.sdk.rm_get_joint_en_state = None
    with pytest.raises(RuntimeError, match="SDK 不支持"):
        hardware.reset_emergency_stop()
    assert hardware.estop_latched and not hardware.armed


@pytest.mark.parametrize("attribute,value", [("closed", True), ("connected", False),
                                           ("sdk", None), ("feedback", None)])
def test_disconnected_adapter_cannot_be_reset(hardware, attribute, value):
    setattr(hardware, attribute, value)
    with pytest.raises(RuntimeError, match="硬件未连接"):
        hardware.reset_emergency_stop()
    assert hardware.estop_latched and not hardware.armed


def test_estop_during_arm_readiness_check_cannot_rearm(hardware):
    hardware.estop_latched = False
    ready = dict(robot_ready=True, cameras_ready=True, force_ready=True, gripper_ready=True)
    def interrupted_status():
        hardware.emergency_stop()
        return ready
    with patch.object(hardware, "status", side_effect=interrupted_status):
        with pytest.raises(RuntimeError, match="拒绝 arm"):
            hardware.arm()
    assert hardware.estop_latched and not hardware.armed


def test_close_waits_for_recovery_read_before_deleting_handle(hardware):
    entered, release, closing = threading.Event(), threading.Event(), threading.Event()
    good = hardware.sdk.rm_get_current_arm_state.return_value
    def blocked():
        entered.set()
        assert release.wait(2.)
        return good
    def signal_closing():
        closing.set()
    hardware.sdk.rm_get_current_arm_state.side_effect = blocked
    hardware.cameras[0].close.side_effect = signal_closing
    with ThreadPoolExecutor(max_workers=2) as pool:
        reset = pool.submit(hardware.reset_emergency_stop)
        assert entered.wait(1.)
        close = pool.submit(hardware.close)
        try:
            assert closing.wait(1.)
            hardware.sdk.rm_delete_robot_arm.assert_not_called()
        finally:
            release.set()
        close.result(timeout=1.)
        with pytest.raises(RuntimeError, match="恢复已取消"):
            reset.result(timeout=1.)
    hardware.sdk.rm_delete_robot_arm.assert_called_once()
    assert hardware.closed and hardware.estop_latched and not hardware.armed


def test_close_timeout_retains_handle_for_retry(hardware):
    hardware._reset_io_lock = Mock()
    hardware._reset_io_lock.acquire.side_effect = [False, True]
    with pytest.raises(RuntimeError, match="保留机械臂连接句柄"):
        hardware.close()
    hardware.sdk.rm_delete_robot_arm.assert_not_called()
    assert hardware._arm_close_pending
    hardware.close()
    hardware.sdk.rm_delete_robot_arm.assert_called_once()
    assert not hardware._arm_close_pending


def test_udp_controller_error_is_distinct_from_parser_status():
    feedback = _RobotFeedback(None, {"robot_ip": "192.168.1.18"})
    state = SimpleNamespace(arm_ip=b"192.168.1.18", errCode=0,
        err=SimpleNamespace(err_len=1, err=[4107]),
        joint_status=SimpleNamespace(joint_err_code=[0] * 7, joint_en_flag=[True] * 7))
    feedback.ingest(state, time.monotonic())
    assert "4107" in feedback.error and not feedback.status(.1)["ready"]
