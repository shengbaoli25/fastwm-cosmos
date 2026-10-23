"""Gripper enable changes no target; manual controls cannot race a policy."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from realman_deploy.gripper import Ctag2F120Worker
from realman_deploy.hardware import _RobotFeedback
from realman_deploy.normalization import DatasetNormalizer
from realman_deploy.runtime import DeploymentRuntime


def test_enable_only_writes_enable_and_rejects_alarm_or_bad_readback():
    worker = Ctag2F120Worker({})
    sdk = Mock()
    sdk.rm_get_current_arm_state.return_value = (0, {"err": {"err": ["0"]}})
    sdk.rm_write_single_register.return_value = 0
    factory = lambda *args: args
    worker.desired = .2
    with patch.object(worker, '_read', side_effect=[[0, 0, 0], [1]]):
        worker._set_enabled(sdk, factory, True)
    sdk.rm_write_single_register.assert_called_once_with((1, 0x0100, 1, 1), 1)
    sdk.rm_write_registers.assert_not_called()
    assert worker.desired is None
    sdk.rm_write_single_register.reset_mock()
    with patch.object(worker, '_read', return_value=[0, 0, 9]), pytest.raises(RuntimeError, match='报警'):
        worker._set_enabled(sdk, factory, True)
    sdk.rm_write_single_register.assert_not_called()
    with patch.object(worker, '_read', return_value=[1]), pytest.raises(RuntimeError, match='回读'):
        worker._set_enabled(sdk, factory, False)


def test_disabled_initialized_gripper_can_connect_but_cannot_move():
    worker = Ctag2F120Worker({})
    import time
    worker.state = dict(initialized=True, enabled=False, ready=False, received=time.monotonic())
    assert worker.wait_initialized(.1)['initialized']
    with pytest.raises(RuntimeError, match='未就绪'):
        worker.command(.5)


def test_manual_operation_blocks_start_and_settings_but_not_emergency_stop():
    normalizer = DatasetNormalizer({name: dict(mean=[0.] * width, std=[1.] * width)
        for name, width in [('state', 8), ('action', 8), ('force', 6)]}, action_space='joint8')
    runtime = DeploymentRuntime(normalizer, require_cuda=False, execution_mode='synchronous')
    entered, release = threading.Event(), threading.Event()
    hardware = Mock()
    hardware.status.return_value = dict(action_space='joint8', force_preprocessing=None)
    def operate(operation):
        entered.set()
        assert release.wait(2)
        return dict(enabled=True)
    hardware.control_gripper.side_effect = operate
    runtime.attach(hardware=hardware)
    thread = threading.Thread(target=runtime.control_gripper, args=('enable',))
    thread.start()
    try:
        assert entered.wait(1)
        with pytest.raises(RuntimeError, match='夹爪操作'):
            runtime.start()
        with pytest.raises(RuntimeError, match='夹爪操作'):
            runtime.configure_sampling('4416', 'synchronous')
        runtime.emergency_stop()
        hardware.emergency_stop.assert_called_once()
    finally:
        release.set()
        thread.join(2)
        runtime.close()
    with pytest.raises(RuntimeError, match='急停'):
        runtime.control_gripper('open')


def test_runtime_rejects_running_manual_controls_and_unknown_operations():
    normalizer = DatasetNormalizer({name: dict(mean=[0.] * width, std=[1.] * width)
        for name, width in [('state', 8), ('action', 8), ('force', 6)]}, action_space='joint8')
    runtime = DeploymentRuntime(normalizer, require_cuda=False)
    runtime.hardware = Mock()
    runtime.state = 'running'
    with pytest.raises(RuntimeError, match='暂停'):
        runtime.control_gripper('open')
    with pytest.raises(ValueError, match='未知'):
        runtime.control_gripper('calibrate')
    runtime.hardware.control_gripper.assert_not_called()
    runtime.state = 'ready'
    runtime.close()


def test_failed_connect_detaches_callback_even_before_push_was_changed():
    feedback = _RobotFeedback(Mock(), {})
    feedback.callback_registered = True
    with patch.dict('sys.modules', {'Robotic_Arm.rm_robot_interface': SimpleNamespace(
            rm_realtime_arm_state_callback_ptr=lambda: 'null-callback')}):
        feedback.restore()
    feedback.sdk.rm_realtime_arm_state_call_back.assert_called_once_with('null-callback')
    assert feedback.callback_registered is False


def test_native_sdk_is_initialized_once_for_multiple_handles():
    from realman_deploy import sdk
    module = SimpleNamespace(RoboticArm=Mock(),
        rm_thread_mode_e=SimpleNamespace(RM_TRIPLE_MODE_E='udp'))
    with patch.dict('sys.modules', {'Robotic_Arm.rm_robot_interface': module}), \
            patch.object(sdk, '_initialized', False):
        sdk.create_sdk()
        sdk.create_sdk()
    assert module.RoboticArm.call_args_list[0].args == ('udp',)
    assert module.RoboticArm.call_args_list[1].args == ()


def test_target_trace_uses_last_sent_target_not_contact_position(caplog):
    worker = Ctag2F120Worker({})
    worker.mapping_min, worker.mapping_max = 0, 12000
    # Cup contact prevents the fingers from reaching the closing target.
    worker.state = dict(ready=True, opening=.6, received=time.monotonic())
    worker.command(.1)
    sdk = Mock()
    sdk.rm_write_registers.return_value = 0
    with caplog.at_level('INFO', logger='realman_deploy.gripper'):
        worker._send_target(sdk, lambda *args: args, .1, .25, .4,
                            worker.command_sequence, 'policy')
    params, payload = sdk.rm_write_registers.call_args.args
    assert params == (1, 0x0102, 1, 7)
    assert int.from_bytes(bytes(payload[:4]), 'big', signed=True) == 9000
    assert len(payload) == 14 and payload[-2:] == [0, 1]
    status = worker.status()
    assert status['requested_opening'] == .1
    assert status['opening'] == .6
    assert status['last_sent']['opening'] == .25
    assert status['last_sent']['direction'] == 'close'
    assert status['last_sent']['source'] == 'policy'
    assert 'Gripper target' in caplog.text
    worker.hold()
    assert worker.status()['requested_opening'] is None
    assert worker.status()['last_sent']['opening'] == .25
    assert sdk.rm_write_registers.call_count == 1
    # A new policy target can release contact despite being below measured width.
    worker._send_target(sdk, lambda *args: args, .4, .4, .25, 2, 'policy')
    assert worker.status()['last_sent']['direction'] == 'open'


def test_failed_target_write_is_not_reported_as_acknowledged():
    worker = Ctag2F120Worker({})
    worker.mapping_min, worker.mapping_max = 0, 12000
    sdk = Mock()
    sdk.rm_write_registers.return_value = 1
    with pytest.raises(RuntimeError, match='下发夹爪位置'):
        worker._send_target(sdk, lambda *args: args, .5, .5, .4, 1, 'manual')
    assert worker.sent_target is None
