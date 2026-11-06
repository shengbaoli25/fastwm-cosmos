"""Offline web-adapter checks: no SDK handles, cameras, or network sockets."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import unittest
from unittest.mock import Mock, patch

from realman_sync_client import web_hardware as web
from realman_force.contract import input_contract
from realman_force.force_calibration import validate_profile
from realman_force.force_filter import make_manifest, validate_config


EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


def config():
    value = json.loads((EXAMPLES / 'realman_hardware.joint8.example.json').read_text())
    for name in ('force_calibration_file', 'force_filter_file'):
        value[name] = str(EXAMPLES / value[name])
    return value


def force_config():
    value = config()
    value['_force_calibration'] = validate_profile(json.loads(Path(value['force_calibration_file']).read_text()))
    value['_force_filter'] = validate_config(json.loads(Path(value['force_filter_file']).read_text()))
    return value


def native_state(coordinate=1):
    return SimpleNamespace(arm_ip=b'192.168.1.18', errCode=0,
        err=SimpleNamespace(err_len=0, err=[0]*24),
        joint_status=SimpleNamespace(joint_err_code=[0]*7, joint_en_flag=[1]*7,
                                     joint_position=list(range(7))),
        waypoint=SimpleNamespace(position=SimpleNamespace(x=.261785, y=.012017, z=.420512),
                                 euler=SimpleNamespace(rx=3.056, ry=.684, rz=-3.072)),
        force_sensor=SimpleNamespace(zero_force=[1., 2., 3., .1, .2, .3], coordinate=coordinate))


class FakeCamera:
    failed = set()

    def __init__(self, spec, width, height, **_):
        self.spec, self.width, self.height = spec, width, height
        self.lock = threading.Lock()
        self.error = None
        self.started = self.closed = False

    def start(self):
        if self.spec['name'] in self.failed:
            raise RuntimeError('camera unavailable')
        self.started = True

    def close(self):
        self.closed = True

    def status(self):
        return dict(name=self.spec['name'], ready=self.started and not self.closed,
                    error=self.error, serial=self.spec['serial'])

    def diagnostics(self):
        return dict(self.status(), fps=0., width=self.width, height=self.height)

    def frame(self):
        return None


class FakeGripperSDK:
    def __init__(self, voltage=3, bus=None):
        self.voltage = voltage
        self.bus = bus or dict(mode=1, baudrate=115200, timeout=2)

    def rm_get_current_arm_state(self):
        return 0, {}

    def rm_get_tool_voltage(self):
        return 0, self.voltage

    def rm_get_tool_rs485_mode(self):
        return 0, self.bus

    def __getattr__(self, name):
        raise AssertionError(f'Unexpected SDK operation: {name}')


class RecoverySDK(FakeGripperSDK):
    def __init__(self, alarm=32, fail_write=None, disable_read=0, reset_changes_alarm=True):
        super().__init__()
        self.alarm, self.fail_write, self.disable_read = alarm, fail_write, disable_read
        self.reset_changes_alarm = reset_changes_alarm
        self.writes, self.bus_calls, self.reads = [], [], []
        self.parameters = [30, 50, 100, 100]
        self.block_writes = []
        self.fail_block_write = False

    def rm_set_modbus_mode(self, *args):
        self.bus_calls.append(args)
        return 0

    def rm_write_single_register(self, params, value):
        self.writes.append((params[1], value))
        if self.fail_write == params[1]:
            return 1
        if params[1] == 0x0403 and self.reset_changes_alarm:
            self.alarm = 0
        return 0

    def rm_read_holding_registers(self, params):
        self.reads.append(params[1])
        return 0, (self.alarm if params[1] == 0x0612 else
                   self.disable_read.pop(0) if isinstance(self.disable_read, list) else self.disable_read)

    def rm_read_multiple_holding_registers(self, params):
        assert params == (1, 0x0104, 1, 4)
        return 0, [byte for value in self.parameters for byte in value.to_bytes(2, 'big')]

    def rm_write_registers(self, params, payload):
        self.block_writes.append((params, payload))
        if self.fail_block_write:
            return 1
        self.parameters = [payload[i]*256 + payload[i+1] for i in range(0, len(payload), 2)]
        return 0


class GeometrySDK:
    def __init__(self):
        self.tool = dict(name='Arm_Tip', pose=[0.]*6)
        self.work = dict(name='World', pose=[0.]*6)
        self.controller = dict(d=[.2405, 0., .256, 0., .21, 0., .184],
                               a=[0.]*7, alpha=[0.]*7, offset=[0.]*7)
        self.algorithm = copy.deepcopy(self.controller)

    def rm_get_current_tool_frame(self):
        return 0, self.tool

    def rm_get_current_work_frame(self):
        return 0, self.work

    def rm_get_DH_data(self):
        return 0, self.controller

    def rm_algo_get_dh(self):
        return self.algorithm

    def __getattr__(self, name):
        raise AssertionError(f'Unexpected hardware operation: {name}')


class RecoveredWorker:
    def __init__(self, _):
        self.thread = None
        self.started = False

    def start(self):
        self.started = True

    def wait_initialized(self, _):
        return self.status()

    def status(self, *_):
        return dict(initialized=True, enabled=False, ready=False, measurement_ready=True,
                    alarm=0, opening=.42, age_ms=1., error='夹爪未使能')

    def close(self):
        pass


class WebHardwareTests(unittest.TestCase):
    def test_extended_observation_budget_never_relaxes_motion_or_hides_expiry(self):
        hardware, worker = self.manual_hardware(enabled=True)
        hardware.config['gripper_feedback_max_age_s'] = 1.5
        worker.state.update(received=10., measurement_started=9.94)
        worker.history.clear()
        worker.history.append((10., .42))
        worker.history_started.append((10., 9.94))
        # A recent enable response cannot refresh the older actual position.
        worker.enabled_received = 11.14
        hardware.feedback = Mock()
        hardware.feedback.diagnostics.side_effect = lambda: (
            dict(values=[0.]*7, ready=True, error=None), dict(values=[0.]*6, ready=True, error=None))
        hardware.cameras = [Mock(), Mock()]
        status = dict(connected=True, joint_limits_ready=True, robot_ready=True,
            force_ready=True, cameras_ready=True, camera_order=['global', 'wrist'],
            force_preprocessing={'id': 'test-contract'})
        with patch.object(web.clock, 'now', return_value=11.14), \
                patch.object(web.RealManHardwareAdapter, 'command') as joint_command:
            state = worker.status(1.5)
            self.assertEqual(state['age_ms'], 1200.)
            self.assertTrue(state['measurement_ready'])
            self.assertIsNone(state['measurement_error'])
            self.assertFalse(state['ready'])
            self.assertFalse(state['control_feedback_ready'])
            self.assertEqual(worker.snapshot(11.1, 1.5), .42)
            self.assertEqual(hardware._local_ready(dict(status, gripper=state), for_execution=False),
                             (True, None))
            with self.assertRaisesRegex(RuntimeError, '运动反馈未就绪'):
                hardware.command([0.]*7+[.4])
            with self.assertRaises(RuntimeError):
                worker.command(.8)
            joint_command.assert_not_called()
            self.assertTrue(hardware.diagnostics()['state']['ready'])
            self.assertIsNone(hardware.gripper_diagnostics()['measurement_error'])
        with patch.object(web.clock, 'now', return_value=11.51):
            state = worker.status(1.5)
            self.assertFalse(state['measurement_ready'])
            self.assertIn('观测反馈过期', state['measurement_error'])
            self.assertIn('1500 ms', state['measurement_error'])
            self.assertEqual(hardware._local_ready(dict(status, gripper=state), for_execution=False),
                             (False, state['measurement_error']))
            self.assertEqual(hardware.diagnostics()['state']['error'], state['measurement_error'])
            self.assertEqual(hardware.gripper_diagnostics()['measurement_error'], state['measurement_error'])
            with self.assertRaisesRegex(RuntimeError, '观测反馈过期'):
                worker.snapshot(11.5, 1.5)

    def test_extended_observation_budget_still_rejects_alarm_and_future_measurement(self):
        worker = web._CheckedGripper(config())
        worker.state = dict(initialized=True, enabled=True, ready=True, alarm=32,
                            opening=.42, received=10., measurement_started=9.94, error=None)
        worker.history.append((10., .42))
        worker.history_started.append((10., 9.94))
        with patch.object(web.clock, 'now', return_value=11.14):
            self.assertEqual(worker.status(1.5)['measurement_error'], '夹爪报警 0x0020')
            with self.assertRaisesRegex(RuntimeError, '夹爪报警'):
                worker.snapshot(11.1, 1.5)
            worker.state['alarm'] = 0
            # Even the longer observation budget cannot make a response known
            # before its actual arrival at 10.0.
            with self.assertRaisesRegex(RuntimeError, '锚点之前'):
                worker.snapshot(9.98, 1.5)

    def test_fresh_reply_cannot_renew_older_selected_observation(self):
        worker = web._CheckedGripper(config())
        worker.state = dict(initialized=True, enabled=True, ready=True, alarm=0,
                            opening=.8, received=11.6, measurement_started=11.5, error=None)
        worker.history.extend([(10., .42), (11.6, .8)])
        worker.history_started.extend([(10., 9.94), (11.6, 11.5)])
        worker.enabled_received = 11.5
        with patch.object(web.clock, 'now', return_value=11.7):
            self.assertTrue(worker.status(1.5)['measurement_ready'])
            with self.assertRaisesRegex(RuntimeError, '观测反馈过期'):
                worker.snapshot(11.4, 1.5)
            self.assertEqual(worker.snapshot(11.6, 1.5), .8)

    def test_recent_measurement_is_observable_during_transport_loss_but_motion_is_blocked(self):
        worker = web._CheckedGripper(config())
        worker.state = dict(initialized=True, enabled=True, ready=True, alarm=0,
                            opening=.42, received=10., measurement_started=9.94, error=None)
        worker.history.append((10., .42))
        worker.history_started.append((10., 9.94))
        worker.enabled_received, worker.enabled_value = 9.9, True
        with patch.object(web.clock, 'now', return_value=10.3):
            worker._feedback_failed(TimeoutError('read timeout'))
            self.assertTrue(worker.status(1.)['measurement_ready'])
            self.assertFalse(worker.status(1.)['ready'])
            self.assertFalse(worker.status(1.)['control_feedback_ready'])
            self.assertEqual(worker.snapshot(10.2, 1.), .42)
            with self.assertRaises(RuntimeError):
                worker.command(.8)
        with patch.object(web.clock, 'now', return_value=11.01):
            self.assertFalse(worker.status(1.)['measurement_ready'])
            with self.assertRaises(RuntimeError):
                worker.snapshot(10.9, 1.)

    def test_known_stale_gripper_rejects_before_any_joint_write(self):
        hardware, worker = self.manual_hardware(enabled=True)
        worker.history.clear()
        worker.history.append((10., .42))
        worker.state['received'] = 10.
        with patch.object(web.clock, 'now', return_value=10.75), \
                patch.object(hardware, 'armed', True), \
                patch.object(web.RealManHardwareAdapter, 'command') as joint_command:
            with self.assertRaisesRegex(RuntimeError, '关节目标未写入.*750.0'):
                hardware.command([0.]*7+[.4])
            joint_command.assert_not_called()

    def test_alarm_remains_primary_with_stale_and_communication_evidence(self):
        hardware, worker = self.manual_hardware(enabled=True)
        worker.state.update(alarm=32, received=10., measurement_started=9.94,
                            error='夹爪报警 0x0020', ready=False)
        worker.history.clear()
        worker.history.append((10., .42))
        worker.history_started.append((10., 9.94))
        worker.enabled_received = 9.9
        with patch.object(web.clock, 'now', return_value=10.75):
            worker._feedback_failed(TimeoutError('Controller response timeout'))
            state = worker.status(1.)
            self.assertEqual(state['error'], '夹爪报警 0x0020')
            self.assertEqual(state['communication_error'], 'Controller response timeout')
            self.assertIn('运动反馈过期', state['freshness_error'])
            self.assertIn('使能反馈过期', state['freshness_error'])
            for key in ('ready', 'measurement_ready', 'control_feedback_ready', 'monitoring_ready'):
                self.assertFalse(state[key])
            diagnostic = hardware.gripper_diagnostics()
            self.assertEqual(diagnostic['fault_code'], 32)
            self.assertEqual(diagnostic['communication_error'], state['communication_error'])
            self.assertEqual(diagnostic['freshness_error'], state['freshness_error'])
            worker.set_enabled.assert_not_called()

    def test_stale_alarm_is_not_misreported_as_a_communication_exception(self):
        worker = web._CheckedGripper(config())
        worker.state = dict(initialized=True, enabled=True, ready=False, alarm=32,
            opening=.42, received=10., measurement_started=9.4, communication_fault=True,
            error='夹爪报警 0x0020')
        worker.history.append((10., .42))
        worker.history_started.append((10., 9.4))
        worker.enabled_received = 9.4
        with patch.object(web.clock, 'now', return_value=10.1):
            state = worker.status()
            self.assertEqual(state['error'], '夹爪报警 0x0020')
            self.assertIsNone(state['communication_error'])
            self.assertIn('运动反馈过期', state['freshness_error'])
            self.assertIn('使能反馈过期', state['freshness_error'])
            worker.transport = SimpleNamespace(status=lambda: dict(needs_sync=True,
                failed=False, last_event={'error': 'Controller response timeout'}))
            state = worker.status()
            self.assertEqual(state['error'], '夹爪报警 0x0020')
            self.assertEqual(state['communication_error'], 'Controller response timeout')

    def test_command_failure_keeps_before_after_feedback_and_duration_evidence(self):
        hardware, worker = self.manual_hardware(enabled=True)
        with patch.object(web.RealManHardwareAdapter, 'command', side_effect=RuntimeError('after joint write')):
            with self.assertRaisesRegex(RuntimeError, 'after joint write; command_ms=.*gripper_before=.*gripper_after='):
                hardware.command([0.]*7+[.4])

    def manual_hardware(self, *, enabled=False):
        hardware = self.recovery_hardware(RecoverySDK())
        worker = web._CheckedGripper(config())
        worker.state = dict(initialized=True, enabled=enabled, ready=enabled, alarm=0,
                            opening=.42, received=web.clock.now(), error=None if enabled else '夹爪未使能')
        worker.enabled_received, worker.enabled_value = worker.state['received'], enabled
        worker.history.append((worker.state['received'], .42))
        worker.thread = SimpleNamespace(is_alive=lambda: True, join=lambda **_: None)
        hardware.gripper = worker
        def confirmed(value):
            worker.state.update(enabled=value, ready=value, error=None if value else '夹爪未使能')
            worker.hold()
            return worker.status()
        worker.set_enabled = Mock(side_effect=confirmed)
        return hardware, worker

    def test_manual_enable_disable_requires_explicit_call_and_confirmed_readback(self):
        hardware, worker = self.manual_hardware()
        diagnostic = hardware.gripper_diagnostics()
        self.assertTrue(diagnostic['enable_available'])
        self.assertFalse(diagnostic['disable_available'])
        self.assertFalse(diagnostic['motion_available'])
        self.assertEqual(diagnostic['opening'], .42)
        self.assertIsNone(diagnostic['requested_opening'])
        worker.set_enabled.assert_not_called()
        enabled = hardware.manual_gripper('enable')
        worker.set_enabled.assert_called_once_with(True)
        self.assertTrue(enabled['confirmed'])
        self.assertIsNone(worker.desired)
        self.assertTrue(hardware.gripper_diagnostics()['motion_available'])
        disabled = hardware.manual_gripper('disable')
        self.assertEqual(worker.set_enabled.call_args_list[-1].args, (False,))
        self.assertTrue(disabled['confirmed'])
        self.assertFalse(disabled['enabled'])
        self.assertTrue(worker.status()['measurement_ready'])
        self.assertFalse(hardware.gripper_diagnostics()['motion_available'])

    def test_manual_open_close_submit_once_without_claiming_completion(self):
        for operation, target in (('open', 1.), ('close', 0.)):
            with self.subTest(operation=operation):
                hardware, worker = self.manual_hardware(enabled=True)
                with patch.object(worker, 'command', wraps=worker.command) as command:
                    result = hardware.manual_gripper(operation)
                command.assert_called_once_with(target, source='manual')
                self.assertTrue(result['ok'])
                self.assertFalse(result['completed'])
                self.assertFalse(result['confirmed'])
                self.assertIn('尚未确认完成', result['message'])
                self.assertEqual(worker.desired, target)
                self.assertEqual(worker.command_sequence, 1)
                worker.set_enabled.assert_not_called()

    def test_opt_in_manual_open_routes_to_worker_preparation_and_keeps_completion_unknown(self):
        hardware, worker = self.manual_hardware(enabled=True)
        hardware.config['gripper_manual_release'] = True
        worker.manual_open = Mock(return_value=dict(worker.status(), requested_opening=1.,
            manual_open_prepared=True, manual_release_performed=True))
        with patch.object(worker, 'command', wraps=worker.command) as command:
            result = hardware.manual_gripper('open')
            worker.manual_open.assert_called_once_with(timeout_s=6.0)
            command.assert_not_called()
            self.assertTrue(result['ok'])
            self.assertTrue(result['queued'])
            self.assertTrue(result['manual_open_prepared'])
            self.assertTrue(result['manual_release_performed'])
            self.assertFalse(result['acknowledged'])
            self.assertIsNone(result['moving'])
            self.assertFalse(result['confirmed'])
            self.assertFalse(result['completed'])
            self.assertEqual(result['manual_open_result'], dict(queued=True, acknowledged=False,
                moving=None, completed=False, source='manual', hold_released=True))
            hardware.manual_gripper('close')
            command.assert_called_once_with(0., source='manual')
            # A policy target keeps the ordinary queue; it cannot opt into the
            # manual force-hold release simply because this config flag exists.
            worker.command(.4, source='policy')
            self.assertEqual(worker.command_source, 'policy')
            worker.manual_open.assert_called_once_with(timeout_s=6.0)
        worker.set_enabled.assert_not_called()

    def test_manual_release_route_requires_explicit_boolean_opt_in(self):
        for opt_in in (False, None, 'true', 1):
            with self.subTest(opt_in=opt_in):
                hardware, worker = self.manual_hardware(enabled=True)
                hardware.config['gripper_manual_release'] = opt_in
                worker.manual_open = Mock()
                with patch.object(worker, 'command', wraps=worker.command) as command:
                    hardware.manual_gripper('open')
                    command.assert_called_once_with(1., source='manual')
                worker.manual_open.assert_not_called()

    def test_manual_open_timeout_is_unconfirmed_and_not_replayed(self):
        hardware, worker = self.manual_hardware(enabled=True)
        hardware.config['gripper_manual_release'] = True
        worker.manual_open = Mock(side_effect=TimeoutError('manual opening cancelled after timeout'))
        with patch.object(worker, 'command') as command:
            with self.assertRaisesRegex(RuntimeError, 'cancelled after timeout'):
                hardware.manual_gripper('open')
            worker.manual_open.assert_called_once_with(timeout_s=6.0)
            command.assert_not_called()
        worker.set_enabled.assert_not_called()
        result = hardware.gripper_diagnostics()['last_manual_result']
        for key in ('ok', 'queued', 'acknowledged', 'confirmed', 'completed'):
            self.assertFalse(result[key])
        self.assertIsNone(result['moving'])
        self.assertFalse(hardware._diagnostic_active)
        self.assertIsNone(hardware._manual_gripper_operation)

    def test_manual_rejects_dangerous_or_unready_states_and_shared_lock(self):
        hardware, worker = self.manual_hardware(enabled=True)
        with patch.object(worker, 'command') as command:
            for key in ('armed', 'estop_latched', '_closing', 'closed', '_diagnostic_active'):
                with self.subTest(flag=key), patch.object(hardware, key, True):
                    with self.assertRaisesRegex(RuntimeError, '条件未满足'):
                        hardware.manual_gripper('open')
            for update in ({'alarm':32}, {'initialized':False}, {'received':web.clock.now()-2},
                           {'enabled':False, 'ready':False}):
                with self.subTest(state=update), patch.dict(worker.state, update):
                    self.assertFalse(hardware.gripper_diagnostics()['motion_available'])
                    with self.assertRaises(RuntimeError):
                        hardware.manual_gripper('close')
            with hardware._diagnostic_lock:
                with self.assertRaisesRegex(RuntimeError, '正在进行'):
                    hardware.manual_gripper('open')
            command.assert_not_called()
        worker.set_enabled.assert_not_called()

    def test_manual_failure_is_not_replayed_and_false_enable_is_not_confirmed(self):
        hardware, worker = self.manual_hardware()
        worker.set_enabled = Mock(return_value=dict(enabled=False))
        with self.assertRaisesRegex(RuntimeError, '读回未确认'):
            hardware.manual_gripper('enable')
        worker.set_enabled.assert_called_once_with(True)
        self.assertFalse(hardware._diagnostic_active)
        self.assertFalse(hardware.gripper_diagnostics()['last_manual_result']['ok'])
        worker.state.update(enabled=True, ready=True)
        with patch.object(worker, 'command', side_effect=RuntimeError('uncertain reply')) as command:
            with self.assertRaisesRegex(RuntimeError, 'uncertain reply'):
                hardware.manual_gripper('open')
            command.assert_called_once_with(1., source='manual')
        self.assertFalse(hardware._diagnostic_active)

    def test_gripper_metrics_count_source_feedback_not_ui_or_enable_ack(self):
        worker = web._CheckedGripper(config())
        worker.state = dict(initialized=True, enabled=True, alarm=0, ready=True,
                            opening=.42, received=10.6, error=None)
        worker.history.extend([(10., .42), (10.25, .42), (10.5, .42)])
        with patch.object(web.clock, 'now', return_value=10.6):
            for _ in range(5):
                state = worker.status(1.)
                self.assertEqual(state['samples'], 3)
                self.assertEqual(state['hz'], 4.)
                self.assertEqual(state['age_ms'], 100.)
                self.assertEqual(state['last_interval_ms'], 250.)
        # A new enable ACK alone must not refresh the position or its rate.
        worker.state['received'] = 11.2
        with patch.object(web.clock, 'now', return_value=11.2):
            state = worker.status(1.)
            self.assertEqual(state['age_ms'], 700.)
            self.assertEqual(state['samples'], 3)
            self.assertTrue(state['measurement_ready'])
            self.assertFalse(state['ready'])
            self.assertFalse(state['control_feedback_ready'])
            self.assertEqual(worker.snapshot(11.1, 1.), .42)
            with self.assertRaises(RuntimeError):
                worker.command(.2)
        with patch.object(web.clock, 'now', return_value=11.51):
            self.assertFalse(worker.status(1.)['measurement_ready'])
            self.assertEqual(worker.status(1.)['hz'], 0.)
            with self.assertRaises(RuntimeError):
                worker.snapshot(11.5, 1.)

    def test_stale_position_enable_ack_cannot_continue_an_accepted_ramp(self):
        worker = web._CheckedGripper(config())
        worker.state = dict(initialized=True, enabled=True, alarm=0, ready=True,
                            opening=.42, received=11.2, error=None)
        worker.enabled_received, worker.enabled_value = 11.19, True
        worker.history.append((10.5, .42))
        worker.desired = worker.ramp_target = .2
        args = (object(), object(), .2, .3, .42, 1, 'manual')
        with patch.object(web.clock, 'now', return_value=11.2), \
                patch.object(web.Ctag2F120Worker, '_send_target', return_value=11.2) as write:
            with self.assertRaisesRegex(RuntimeError, '真实位置反馈过期'):
                worker._send_target(*args)
            write.assert_not_called()
            self.assertIsNone(worker.desired)
            self.assertIsNone(worker.ramp_target)
            # A real new position permits the original one-shot writer again.
            worker.history.append((11.18, .42))
            worker.desired = worker.ramp_target = .2
            self.assertEqual(worker._send_target(*args), 11.2)
            write.assert_called_once_with(*args)

    def test_frames_and_dh_match_bound_calibration_with_read_queries_only(self):
        hardware = self.recovery_hardware(GeometrySDK())
        result = hardware._verify_frames_and_dh()
        self.assertTrue(result['verified'])
        self.assertEqual(result['tool']['name'], 'Arm_Tip')
        self.assertEqual(result['work']['name'], 'World')
        self.assertEqual(result['controller_dh']['d'], result['expected_dh_d_m'])
        reference = hardware.diagnostics()['reference_checks']
        self.assertTrue(reference['passed'])
        self.assertEqual(reference['controller_dh']['d'], result['expected_dh_d_m'])

    def test_frames_reject_wrong_name_nonidentity_or_config_override(self):
        for mutation in ('tool_name', 'work_pose', 'override'):
            with self.subTest(mutation=mutation):
                sdk = GeometrySDK()
                hardware = self.recovery_hardware(sdk)
                if mutation == 'tool_name':
                    sdk.tool['name'] = 'Another_tool'
                elif mutation == 'work_pose':
                    sdk.work['pose'][0] = .001
                else:
                    hardware.config['expected_tool_frame'] = 'Another_tool'
                with self.assertRaises(RuntimeError):
                    hardware._verify_frames_and_dh()
                self.assertFalse(hardware.frame_geometry['verified'])

    def test_dh_rejects_sdk_mismatch_wrong_physical_geometry_and_nonfinite(self):
        for mutation in ('sdk', 'physical', 'nonfinite'):
            with self.subTest(mutation=mutation):
                sdk = GeometrySDK()
                hardware = self.recovery_hardware(sdk)
                if mutation == 'sdk':
                    sdk.algorithm['alpha'][0] = .01
                elif mutation == 'physical':
                    sdk.controller['d'][0] = sdk.algorithm['d'][0] = .2415
                else:
                    sdk.controller['offset'][0] = float('nan')
                with self.assertRaises((RuntimeError, ValueError)):
                    hardware._verify_frames_and_dh()
                self.assertFalse(hardware.frame_geometry['verified'])

    def recovery_hardware(self, sdk):
        with patch.object(web.WebHardware, '_connect'):
            hardware = web.WebHardware(config())
        hardware.connected = hardware._robot_identity_valid = True
        hardware.sdk = sdk
        self.addCleanup(hardware.close)
        return hardware

    def invoke_recovery(self, hardware):
        fake_sdk = SimpleNamespace(rm_peripheral_read_write_params_t=lambda *args: args)
        with patch.dict(sys.modules, {'Robotic_Arm': SimpleNamespace(),
                                     'Robotic_Arm.rm_robot_interface': fake_sdk}), \
                patch.object(web, '_CheckedGripper', RecoveredWorker), patch.object(web.time, 'sleep'):
            return hardware.clear_gripper_error()

    def test_recovery_reinitializes_bus_and_resets_once_while_disabled(self):
        sdk = RecoverySDK()
        hardware = self.recovery_hardware(sdk)
        self.assertTrue(hardware.gripper_diagnostics()['clear_available'])
        result = self.invoke_recovery(hardware)
        self.assertTrue(result['ok'])
        self.assertEqual((result['before_alarm'], result['after_alarm'], result['enabled']), (32, 0, False))
        self.assertTrue(result['reset_written'])
        self.assertEqual(sdk.bus_calls, [(1, 115200, 2)])
        self.assertEqual(sdk.writes, [(0x0100, 0), (0x0403, 1)])
        self.assertTrue(hardware.gripper.started)
        self.assertFalse(hardware.gripper_diagnostics()['enabled'])
        self.assertTrue(hardware.gripper_diagnostics()['measurement_ready'])
        self.assertEqual(hardware.gripper_diagnostics()['last_result'], result)

    def test_recovery_with_zero_alarm_never_writes_reset(self):
        sdk = RecoverySDK(alarm=0)
        result = self.invoke_recovery(self.recovery_hardware(sdk))
        self.assertTrue(result['ok'])
        self.assertFalse(result['reset_written'])
        self.assertEqual(sdk.writes, [(0x0100, 0)])
        self.assertIn('原报警为 0', result['message'])

    def test_recovery_aligns_only_volatile_parameter_block_and_confirms_disabled(self):
        sdk = RecoverySDK(alarm=0)
        sdk.parameters = [100, 100, 100, 100]
        hardware = self.recovery_hardware(sdk)
        hardware.config['gripper_speed'] = 100
        result = self.invoke_recovery(hardware)
        self.assertTrue(result['ok'])
        self.assertEqual(result['params_before'], [100, 100, 100, 100])
        self.assertEqual(result['params_after'], [100, 50, 100, 100])
        self.assertTrue(result['params_written'])
        self.assertEqual(sdk.block_writes, [((1, 0x0104, 1, 4), [0, 100, 0, 50, 0, 100, 0, 100])])
        self.assertEqual(sdk.writes, [(0x0100, 0)])
        self.assertEqual([row['address'] for row in result['writes']], [0x0100, 0x0104])
        self.assertFalse(result['enabled'])

    def test_recovery_parameter_write_failure_is_not_replayed(self):
        sdk = RecoverySDK(alarm=0)
        sdk.parameters, sdk.fail_block_write = [100, 100, 100, 100], True
        hardware = self.recovery_hardware(sdk)
        hardware.config['gripper_speed'] = 100
        with self.assertRaisesRegex(RuntimeError, 'align_volatile_motion_parameters'):
            self.invoke_recovery(hardware)
        self.assertEqual(len(sdk.block_writes), 1)
        result = hardware.gripper_diagnostics()['last_result']
        self.assertTrue(result['recovery_verified'])
        self.assertFalse(result['params_written'])
        self.assertFalse(result['writes'][-1]['acknowledged'])
        self.assertFalse(result['ok'])

    def test_recovery_does_not_retry_failed_disable_or_reset_writes(self):
        for address, expected in ((0x0100, [(0x0100, 0)]), (0x0403, [(0x0100, 0), (0x0403, 1)])):
            with self.subTest(address=address):
                sdk = RecoverySDK(fail_write=address)
                hardware = self.recovery_hardware(sdk)
                with self.assertRaisesRegex(RuntimeError, '失败不重发'):
                    self.invoke_recovery(hardware)
                self.assertEqual(sdk.writes, expected)
                self.assertFalse(hardware.gripper_diagnostics()['last_result']['ok'])
                self.assertFalse(hardware._clearing_gripper)

    def test_recovery_requires_actual_disable_readback_before_reset(self):
        sdk = RecoverySDK(disable_read=1)
        hardware = self.recovery_hardware(sdk)
        with self.assertRaisesRegex(RuntimeError, '尚非 0'):
            self.invoke_recovery(hardware)
        self.assertEqual(sdk.writes, [(0x0100, 0)])
        self.assertNotIn(0x0612, sdk.reads)

    def test_unexpected_enable_after_reset_disables_once_and_still_fails(self):
        sdk = RecoverySDK(disable_read=[0, 1, 0])
        hardware = self.recovery_hardware(sdk)
        with self.assertRaisesRegex(RuntimeError, '意外出现使能'):
            self.invoke_recovery(hardware)
        self.assertEqual(sdk.writes, [(0x0100, 0), (0x0403, 1), (0x0100, 0)])
        result = hardware.gripper_diagnostics()['last_result']
        self.assertFalse(result['ok'])
        self.assertFalse(result['enabled'])
        self.assertTrue(result['failsafe_disable_confirmed'])
        self.assertEqual(result['stage'], 'unexpected_enable_fail_stop')

    def test_recovery_requires_old_worker_exit_and_no_estop(self):
        sdk = RecoverySDK()
        hardware = self.recovery_hardware(sdk)
        hardware.gripper = Mock()
        hardware.gripper.thread.is_alive.return_value = True
        hardware.gripper.status.return_value = {}
        with self.assertRaisesRegex(RuntimeError, '旧夹爪线程未退出'):
            self.invoke_recovery(hardware)
        self.assertEqual(sdk.bus_calls, [])
        self.assertEqual(sdk.writes, [])
        hardware.estop_latched = True
        with self.assertRaisesRegex(RuntimeError, '无急停'):
            self.invoke_recovery(hardware)
        self.assertEqual(sdk.writes, [])

    def test_disabled_fresh_gripper_can_be_observed_but_not_commanded(self):
        worker = web._CheckedGripper(config())
        worker.state = dict(initialized=True, enabled=False, alarm=0, ready=False, opening=.42,
                            received=10., error='夹爪未使能')
        worker.history.append((10., .42))
        with patch.object(web.clock, 'now', return_value=10.02):
            self.assertTrue(worker.status()['measurement_ready'])
            self.assertFalse(worker.status()['ready'])
            self.assertEqual(worker.snapshot(10.01), .42)
            with self.assertRaisesRegex(RuntimeError, '尚未就绪'):
                worker.command(.1)
            worker.state['alarm'] = 32
            self.assertFalse(worker.status()['measurement_ready'])
            with self.assertRaises(RuntimeError):
                worker.snapshot(10.01)
            worker.state['alarm'] = 0
        with patch.object(web.clock, 'now', return_value=11.):
            with self.assertRaises(RuntimeError):
                worker.snapshot(11.)

    def test_disabled_observation_ready_does_not_allow_arm(self):
        hardware = self.recovery_hardware(RecoverySDK())
        contract = input_contract(hardware.config['_force_calibration'],
            make_manifest(hardware.config['_force_filter'], hardware.config['_force_calibration']))
        hardware.feedback = Mock()
        hardware.feedback.model_snapshot.return_value = (None, None, None, 10., {})
        hardware.gripper = Mock()
        hardware.cameras = [Mock(), Mock()]
        for camera in hardware.cameras:
            camera.snapshot.return_value = (b'frame', dict(received=9.99))
        status = dict(connected=True, joint_limits_ready=True, robot_ready=True, force_ready=True,
            gripper_ready=False, gripper=dict(ready=False, enabled=False, measurement_ready=True,
                                              error='夹爪未使能'), cameras_ready=True,
            camera_order=['global', 'wrist'], cameras=[dict(ready=True)]*2, force_preprocessing=contract)
        metadata = dict(action_space='joint8', width=640, height=480, action_horizon=32,
            history_length=32, control_hz=30, camera_names=['global', 'wrist'],
            video_context_frames=1, video_context_stride=1, force_preprocessing=contract)
        with patch.object(hardware, 'status', return_value=status), patch.object(web.clock, 'now', return_value=10.001):
            self.assertEqual(hardware.can_infer(metadata), (True, None))
            self.assertFalse(hardware._local_ready(status)[0])
            with self.assertRaisesRegex(RuntimeError, '夹爪未使能'):
                hardware.arm()
            self.assertFalse(hardware.armed)

    def test_diagnose_uses_existing_handle_only_and_rejects_failed_zero_buffer(self):
        with patch.object(web.WebHardware, '_connect'):
            hardware = web.WebHardware(config())
        hardware.connected = True
        hardware.sdk = Mock(spec=['rm_get_tool_voltage', 'rm_get_tool_rs485_mode', 'rm_read_holding_registers'])
        hardware.sdk.rm_get_tool_voltage.return_value = (0, 3)
        hardware.sdk.rm_get_tool_rs485_mode.return_value = dict(return_code=0, mode=1, baudrate=115200, timeout=2)
        hardware.sdk.rm_read_holding_registers.side_effect = lambda p: (1, 0) if p[1] == 0x0612 else (0, 1)
        fake_sdk = SimpleNamespace(rm_peripheral_read_write_params_t=lambda *args: args)
        try:
            with patch.dict(sys.modules, {'Robotic_Arm': SimpleNamespace(),
                                         'Robotic_Arm.rm_robot_interface': fake_sdk}):
                result = hardware.diagnose_gripper()
            self.assertFalse(result['ok'])
            self.assertFalse(result['writes_performed'])
            self.assertEqual(result['settings']['tool_rs485_mode'],
                             dict(ok=True, return_code=0, value=dict(mode=1, baudrate=115200, timeout=2),
                                  elapsed_ms=result['settings']['tool_rs485_mode']['elapsed_ms']))
            self.assertIsNone(result['registers'][0]['word'])
            self.assertIsNone(hardware.gripper_diagnostics()['fault_code'])
            self.assertFalse(hardware._diagnostic_active)
            self.assertEqual([call.args[0] for call in hardware.sdk.rm_read_holding_registers.call_args_list],
                             [(1, address, 1, 1) for address in (0x0612, 0x0100, 0x0104, 0x0105)])
            with self.assertRaisesRegex(RuntimeError, '需已连接'):
                hardware.clear_gripper_error()
            hardware._last_gripper_diagnosis = dict(registers=[dict(address=0x0612, ok=True, word=2)])
            self.assertEqual(hardware.gripper_diagnostics()['fault_code'], 2)
            with self.assertRaisesRegex(RuntimeError, '需已连接'):
                hardware.clear_gripper_error()
        finally:
            hardware.close()

    def test_diagnose_refuses_motion_live_gripper_worker_and_concurrent_request(self):
        with patch.object(web.WebHardware, '_connect'):
            hardware = web.WebHardware(config())
        hardware.connected = True
        hardware.sdk = Mock(spec=['rm_read_holding_registers'])
        try:
            hardware.armed = True
            with self.assertRaisesRegex(RuntimeError, '先停止动作执行'):
                hardware.diagnose_gripper()
            hardware.armed = False
            hardware.gripper = Mock()
            hardware.gripper.thread.is_alive.return_value = True
            with self.assertRaisesRegex(RuntimeError, '不能并发访问'):
                hardware.diagnose_gripper()
            with hardware._diagnostic_lock:
                with self.assertRaisesRegex(RuntimeError, '诊断正在进行'):
                    hardware.diagnose_gripper()
            hardware.sdk.rm_read_holding_registers.assert_not_called()
        finally:
            hardware.close()

    def test_original_force_contract_matches_deployed_cosmos_identity(self):
        value = force_config()
        contract = input_contract(value['_force_calibration'],
                                  make_manifest(value['_force_filter'], value['_force_calibration']))
        self.assertEqual(contract['id'], '41dff495c3ac878829234b43004c450720f8fd45751b0136e39dca5853af5236')

    def test_rate_counts_sources_and_becomes_zero_when_stale(self):
        rate = web._Rate()
        for stamp in (10., 10.005, 10.01):
            rate.add(stamp)
        rate.add(10.01)
        self.assertEqual(rate.snapshot(10.011, .03)['samples'], 3)
        self.assertEqual(rate.snapshot(10.011, .03)['hz'], 200.)
        self.assertEqual(rate.snapshot(10.1, .03)['hz'], 0.)

    def test_camera_fps_uses_accepted_frames_not_preview_reads(self):
        camera = web._MonitoredCamera(dict(name='global', serial='test'), width=2, height=2,
                                       fps=30, max_age_s=.12)
        try:
            with patch.object(web.clock, 'now', return_value=11.):
                for n, stamp in enumerate((10.9, 10.933333333, 10.966666666)):
                    camera._accept((dict(name='global', serial='test', received=stamp,
                        color_order='RGB', width=2, height=2, frame_number=n), bytes(12), None))
                for _ in range(5):
                    self.assertEqual(camera.frame()[3], 2)
                self.assertEqual(camera.rate.snapshot(11., .12)['samples'], 3)
                self.assertEqual(camera.rate.snapshot(11., .12)['hz'], 30.)
        finally:
            camera.close()

    def test_gripper_setup_reads_existing_state_without_any_write(self):
        worker = web._CheckedGripper(config())
        values = {0x0610:[0, 0, 0], 0x0404:[0, 0, 0, 1000], 0x0301:[0, 0, 0, 1000],
                  0x0104:[30, 50, 100, 100], 0x060D:[0, 250, 0]}
        with patch.object(worker, '_read', side_effect=lambda sdk, factory, address, count: values[address]), \
                patch.object(worker, '_write', side_effect=AssertionError('unexpected write')):
            self.assertEqual(worker._setup(FakeGripperSDK(), object()), (0, 1000, .75))
            worker.mapping_min, worker.mapping_max = 0, 1000
            self.assertEqual(worker._verify_readonly_reconnect(FakeGripperSDK(), object()), (0, 1000, .75))
            worker.mapping_max = 1200
            with self.assertRaisesRegex(RuntimeError, '映射不匹配'):
                worker._verify_readonly_reconnect(FakeGripperSDK(), object())

    def test_gripper_rejects_missing_power_bus_or_parameters_without_repair(self):
        worker = web._CheckedGripper(config())
        for sdk in (FakeGripperSDK(voltage=0), FakeGripperSDK(bus=dict(mode=0))):
            with self.subTest(sdk=sdk), self.assertRaisesRegex(RuntimeError, '监测模式未改变'):
                worker._setup(sdk, object())
        values = {0x0610:[0, 0, 0], 0x0404:[0, 0, 0, 1000], 0x0301:[0, 0, 0, 1000],
                  0x0104:[90, 90, 100, 100]}
        with patch.object(worker, '_read', side_effect=lambda sdk, factory, address, count: values[address]), \
                patch.object(worker, '_write', side_effect=AssertionError('unexpected write')), \
                self.assertRaisesRegex(RuntimeError, '未改写夹爪参数'):
            worker._setup(FakeGripperSDK(), object())

    def test_failed_gripper_setup_error_is_preserved_when_stale(self):
        worker = web._CheckedGripper(config())
        worker.state = dict(ready=False, received=1., error='existing voltage is not 24V')
        self.assertEqual(worker.status()['error'], 'existing voltage is not 24V')

    def test_robot_failure_keeps_cameras_and_monitor_close_never_stops(self):
        with patch.object(web, '_MonitoredCamera', FakeCamera), \
                patch.object(web.WebHardware, '_connect_robot', side_effect=RuntimeError('robot offline')):
            hardware = web.WebHardware(config())
        self.assertTrue(all(camera.started for camera in hardware.cameras))
        self.assertFalse(hardware.execution_available)
        self.assertEqual(hardware.diagnostics()['state']['values'], None)
        self.assertEqual(hardware.diagnostics()['state']['error'], 'robot offline')
        with patch.object(hardware, 'slow_stop') as stop:
            hardware.close()
            hardware.close()
            stop.assert_not_called()
        self.assertTrue(all(camera.closed for camera in hardware.cameras))

    def test_one_camera_failure_does_not_prevent_other_camera(self):
        with patch.object(web, '_MonitoredCamera', FakeCamera), patch.object(FakeCamera, 'failed', {'global'}), \
                patch.object(web.WebHardware, '_connect_robot', side_effect=RuntimeError('robot offline')):
            hardware = web.WebHardware(config())
        try:
            self.assertFalse(hardware.cameras[0].started)
            self.assertEqual(hardware.cameras[0].error, 'camera unavailable')
            self.assertTrue(hardware.cameras[1].started)
        finally:
            hardware.close()

    def test_execution_close_stops_and_surfaces_failure_after_cleanup(self):
        with patch.object(web, '_MonitoredCamera', FakeCamera), \
                patch.object(web.WebHardware, '_connect_robot', side_effect=RuntimeError('robot offline')):
            hardware = web.WebHardware(config())
        hardware._ever_armed = True
        with patch.object(hardware, 'slow_stop', side_effect=RuntimeError('stop failed')) as stop:
            with self.assertRaisesRegex(RuntimeError, '缓停失败'):
                hardware.close()
            stop.assert_called_once()
        self.assertTrue(hardware.closed)
        self.assertTrue(all(camera.closed for camera in hardware.cameras))

    def test_native_frequency_independent_of_force_validity_and_filter_warmup(self):
        feedback = web._MonitoredFeedback(None, force_config())
        for n in range(105):
            feedback.ingest(native_state(), 10 + n * .005)
        with patch.object(web.clock, 'now', return_value=10.521):
            state, force = feedback.diagnostics()
        self.assertEqual(state['hz'], 200.)
        self.assertEqual(force['hz'], 200.)
        self.assertEqual(force['kind'], 'calibrated_filtered')
        self.assertTrue(force['ready'])
        feedback.ingest(native_state(coordinate=0), 10.525)
        with patch.object(web.clock, 'now', return_value=10.526):
            state, force = feedback.diagnostics()
        self.assertEqual(state['samples'], 106)
        self.assertEqual(force['samples'], 105)
        self.assertIsNone(state['error'])
        self.assertIsNotNone(force['error'])
        self.assertEqual(force['kind'], 'controller_zero_force')
        self.assertEqual(list(feedback.processed), [])

    def test_udp_system_fault_invalidates_state_force_history_and_filter(self):
        feedback = web._MonitoredFeedback(None, force_config())
        for n in range(105):
            feedback.ingest(native_state(), 10+n*.005)
        self.assertGreater(len(feedback.processed), 0)
        bad = native_state()
        bad.err = SimpleNamespace(err_len=1, err=[0x1001]+[0]*23)
        feedback.ingest(bad, 10.525)
        self.assertIsNone(feedback.q)
        self.assertEqual(list(feedback.processed), [])
        self.assertIn('控制器系统错误', feedback.error)
        with patch.object(web.clock, 'now', return_value=10.526):
            state, force = feedback.diagnostics()
            self.assertFalse(state['ready'])
            self.assertFalse(force['ready'])
            with self.assertRaises(RuntimeError):
                feedback.model_snapshot(10.526)
        feedback.ingest(native_state(), 10.53)
        self.assertEqual(list(feedback.processed), [])
        self.assertIsNotNone(feedback.force_error)

    def test_udp_malformed_system_error_length_is_rejected(self):
        feedback = web._MonitoredFeedback(None, force_config())
        bad = native_state()
        bad.err = SimpleNamespace(err_len=25, err=[0]*24)
        feedback.ingest(bad, 10.)
        self.assertIn('长度无效', feedback.error)
        self.assertEqual(feedback.state_rate.samples, 0)

    def test_malformed_native_packet_does_not_escape_callback(self):
        feedback = web._MonitoredFeedback(None, force_config())
        feedback.ingest(object(), 10.)
        self.assertIsNotNone(feedback.error)
        self.assertEqual(feedback.state_rate.samples, 0)

    def test_dashboard_joint8_units_and_missing_gripper_are_explicit(self):
        with patch.object(web.WebHardware, '_connect'):
            hardware = web.WebHardware(config())
        hardware.feedback = Mock()
        state = dict(hz=200., samples=100, age_ms=1., values=[180.] * 7, ready=True, error=None)
        force = dict(hz=200., samples=100, age_ms=1., values=[0.] * 6, ready=True, error=None)
        hardware.feedback.diagnostics.side_effect = lambda: (dict(state), dict(force))
        try:
            result = hardware.diagnostics()['state']
            self.assertAlmostEqual(result['values'][0], 3.141592653589793)
            self.assertIsNone(result['values'][7])
            self.assertFalse(result['ready'])
            hardware.gripper = Mock()
            hardware.gripper.status.return_value = dict(initialized=True, opening=.42,
                                                         ready=True, age_ms=20., error=None)
            result = hardware.diagnostics()['state']
            self.assertEqual(result['values'][7], .42)
            self.assertEqual(result['units'], ['rad']*7 + ['0..1'])
            self.assertTrue(result['ready'])
            hardware.gripper.status.return_value['age_ms'] = 700.
            self.assertIsNone(hardware.diagnostics()['state']['values'][7])
        finally:
            hardware.close()

    def test_can_infer_never_creates_observation_and_rejects_contract_mismatch(self):
        with patch.object(web.WebHardware, '_connect'):
            hardware = web.WebHardware(config())
        contract = input_contract(hardware.config['_force_calibration'],
            make_manifest(hardware.config['_force_filter'], hardware.config['_force_calibration']))
        metadata = dict(action_space='joint8', width=640, height=480, action_horizon=32,
            history_length=32, control_hz=30, camera_names=['global', 'wrist'],
            video_context_frames=1, video_context_stride=1, force_preprocessing=contract)
        hardware.feedback = Mock()
        hardware.feedback.model_snapshot.return_value = (None, None, None, 10., {})
        hardware.feedback.restore = Mock()
        hardware.gripper = Mock()
        hardware.cameras = [Mock(), Mock()]
        for camera in hardware.cameras:
            camera.snapshot.return_value = (b'frame', dict(received=9.99))
        try:
            with patch.object(hardware, '_local_ready', return_value=(True, None)), \
                    patch.object(hardware, 'status', return_value={'force_preprocessing':contract}), \
                    patch.object(hardware, 'observe_at', side_effect=AssertionError('UI must not sample')), \
                    patch.object(web.clock, 'now', return_value=10.001):
                self.assertEqual(hardware.can_infer(metadata), (True, None))
                self.assertEqual(hardware.sequence, 0)
                wrong = copy.deepcopy(metadata)
                wrong['force_preprocessing']['id'] = 'mismatch'
                self.assertFalse(hardware.can_infer(wrong)[0])
                hardware.cameras[0].snapshot.return_value = (b'frame', dict(received=9.8))
                self.assertFalse(hardware.can_infer(metadata)[0])
        finally:
            hardware.close()


if __name__ == '__main__':
    unittest.main()
