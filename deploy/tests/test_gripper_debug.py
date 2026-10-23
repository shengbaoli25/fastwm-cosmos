"""Gripper debug protocol checks use a fake SDK and never connect hardware."""
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'deploy'))
from realman_deploy.gripper import Ctag2F120Worker


class Params:
    def __init__(self, port, address, device, num):
        self.port, self.address, self.device, self.num = port, address, device, num


class FakeSDK:
    def __init__(self, *, enabled=1, alarm=0, position=6000):
        self.regs = {0x0100: enabled, 0x0102: 0, 0x0103: 10000,
                     0x0104: 30, 0x0105: 50, 0x0106: 100, 0x0107: 100, 0x0108: 1,
                     0x0404: 0, 0x0405: 0, 0x0406: 0, 0x0407: 12000,
                     0x0301: 0, 0x0302: 0, 0x0303: 0, 0x0304: 3000,
                     0x0601: 0, 0x0602: 1, 0x060D: 0, 0x060E: position, 0x0612: alarm}
        self.writes = []
        self.threads = []
        self.auto_arrive = True
        self.enable_applies = True
        self.reset_applies = True
        self.disable_ignore = 0
        self.on_write = None
        self.on_read = None
        self.deleted = False

    def note(self):
        self.threads.append(threading.get_ident())

    def rm_set_timeout(self, timeout):
        self.note()

    def rm_create_robot_arm(self, *args):
        self.note()
        return types.SimpleNamespace(id=1)

    def rm_delete_robot_arm(self):
        self.note()
        self.deleted = True
        return 0

    def rm_read_holding_registers(self, params):
        self.note()
        if self.on_read:
            self.on_read(params)
        return 0, self.regs.get(params.address, 0)

    def rm_read_multiple_holding_registers(self, params):
        self.note()
        if self.on_read:
            self.on_read(params)
        values = [self.regs.get(params.address+i, 0) for i in range(params.num)]
        return 0, [byte for value in values for byte in value.to_bytes(2, 'big')]

    def rm_write_registers(self, params, raw):
        self.note()
        values = [raw[i]*256+raw[i+1] for i in range(0, len(raw), 2)]
        self.writes.append((params.address, values))
        if self.on_write:
            self.on_write(params, values)
        if params.address == 0x0100:
            if values[0] == 1 and not self.enable_applies:
                return 0
            if values[0] == 0 and self.disable_ignore:
                self.disable_ignore -= 1
                return 0
        for i, value in enumerate(values):
            self.regs[params.address+i] = value
        if params.address == 0x0403 and self.reset_applies:
            self.regs[0x0612] = 0
        if params.address == 0x0108 and values == [1] and self.auto_arrive:
            self.regs[0x060D], self.regs[0x060E] = self.regs[0x0102], self.regs[0x0103]
            self.regs[0x0602] = 1
        return 0


class GripperDebugTests(unittest.TestCase):
    def setUp(self):
        self.workers = []
        self.patches = []

    def tearDown(self):
        for worker in self.workers:
            worker.close()
            if worker.thread is not None:
                worker.thread.join(3)
                self.assertFalse(worker.thread.is_alive())
        for item in reversed(self.patches):
            item.stop()

    def worker(self, sdk=None, **config):
        sdk = sdk or FakeSDK()
        package = types.ModuleType('Robotic_Arm')
        api = types.ModuleType('Robotic_Arm.rm_robot_interface')
        api.RoboticArm = lambda mode: sdk
        api.rm_peripheral_read_write_params_t = Params
        api.rm_thread_mode_e = types.SimpleNamespace(RM_DUAL_MODE_E=2)
        item = patch.dict(sys.modules, {'Robotic_Arm': package, 'Robotic_Arm.rm_robot_interface': api})
        item.start()
        self.patches.append(item)
        worker = Ctag2F120Worker(dict(robot_ip='fake', robot_port=8080,
                                    gripper_poll_s=.01, gripper_command_interval_s=.01,
                                    **config), diagnostic=True)
        worker._configure_bus = lambda actual: None
        self.workers.append(worker)
        worker.start()
        worker.wait_connected(.5)
        return worker, sdk

    @staticmethod
    def wait_idle(worker, timeout=1):
        deadline = time.monotonic()+timeout
        while worker.status()['busy'] and time.monotonic() < deadline:
            time.sleep(.01)
        if worker.status()['busy']:
            raise AssertionError('debug worker did not finish cleanup')

    def test_diagnostic_connection_tolerates_alarm_and_disabled_without_actuator_writes(self):
        worker, sdk = self.worker(FakeSDK(enabled=0, alarm=3))
        state = worker.wait_connected(.1)
        self.assertTrue(state['connected'])
        self.assertFalse(state['ready'])
        self.assertFalse(state['enabled'])
        self.assertEqual(state['alarm_hex'], '0x0003')
        self.assertEqual(state['alarm_labels'], ['过温', '堵转'])
        self.assertEqual(sdk.writes, [])

    def test_enable_replaces_retained_target_clears_trigger_then_confirms(self):
        worker, sdk = self.worker(FakeSDK(enabled=0, position=2401))
        state = worker.debug_command('enable')
        self.assertTrue(state['ready'])
        self.assertTrue(state['command_completed'])
        self.assertEqual(sdk.writes, [(0x0108, [0]), (0x0102, [0, 2401]), (0x0100, [1])])
        self.assertEqual(sdk.regs[0x060E], 2401)
        self.assertEqual(set(sdk.threads), {worker.thread.ident})

    def test_clear_error_disables_before_single_reset_and_never_enables(self):
        worker, sdk = self.worker(FakeSDK(alarm=2))
        result = worker.debug_command('clear_error')
        self.assertEqual(sdk.writes, [(0x0100, [0]), (0x0403, [1])])
        self.assertFalse(result['enabled'])
        self.assertEqual(result['alarm'], 0)
        self.assertTrue(result['operation_result']['reset_written'])

    def test_disable_requires_actual_register_readback(self):
        sdk = FakeSDK()
        sdk.disable_ignore = 1
        worker, sdk = self.worker(sdk)
        result = worker.debug_command('disable')
        self.assertFalse(result['enabled'])
        self.assertEqual(sdk.writes, [(0x0100, [0]), (0x0100, [0])])

    def test_enable_is_not_retried_when_ack_does_not_change_register(self):
        sdk = FakeSDK(enabled=0)
        sdk.enable_applies = False
        worker, sdk = self.worker(sdk)
        with self.assertRaisesRegex(TimeoutError, '超时'):
            worker.debug_command('enable', timeout_s=.07)
        self.wait_idle(worker)
        self.assertEqual(sum(a == 0x0100 and v == [1] for a, v in sdk.writes), 1)
        self.assertFalse(worker.status()['enabled'])

    def test_move_open_and_close_use_measured_bounded_steps_and_arrival(self):
        worker, sdk = self.worker(max_gripper_step=.15)
        result = worker.debug_command('open')
        self.assertEqual(result['opening'], 1)
        self.assertEqual(result['operation_result']['steps'], 4)
        targets = [values[0]*65536+values[1] for address, values in sdk.writes if address == 0x0102]
        self.assertEqual(targets, [4200, 2400, 600, 0])
        self.assertTrue(all(abs(a-b) <= 1800 for a, b in zip([6000]+targets, targets)))
        result = worker.debug_command('close')
        self.assertEqual(result['opening'], 0)
        result = worker.debug_command('move', opening=.5, speed=10, torque=25)
        self.assertEqual(result['opening'], .5)
        self.assertEqual(sdk.writes[-2][1][2:4], [10, 25])

    def test_stale_arrival_flag_never_claims_success_and_timeout_discards_steps(self):
        sdk = FakeSDK()
        sdk.auto_arrive = False
        worker, sdk = self.worker(sdk)
        with self.assertRaisesRegex(TimeoutError, '超时'):
            worker.debug_command('open', timeout_s=.08)
        self.wait_idle(worker)
        self.assertEqual(sum(a == 0x0108 and v == [1] for a, v in sdk.writes), 1)
        self.assertEqual(sdk.regs[0x0100], 0)
        self.assertFalse(worker.status()['command_completed'])

    def test_alarm_or_invalid_mapping_prevents_all_motion_writes(self):
        for alarm, invalid in [(2, False), (0, True)]:
            with self.subTest(alarm=alarm, invalid=invalid):
                sdk = FakeSDK(alarm=alarm)
                if invalid:
                    sdk.regs[0x0407] = 0
                worker, sdk = self.worker(sdk)
                with self.assertRaises(RuntimeError):
                    worker.debug_command('open')
                self.assertEqual(sdk.writes, [])

    def test_alarm_during_poll_clears_desired_but_explicit_clear_can_recover(self):
        worker, sdk = self.worker()
        with worker.lock:
            worker.desired = .9
            sdk.regs[0x0612] = 2
        deadline = time.monotonic()+.5
        while worker.status()['alarm'] != 2 and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(worker.status()['ready'])
        self.assertIsNone(worker.desired)
        self.assertTrue(worker.thread.is_alive())
        worker.debug_command('clear_error')
        self.assertFalse(worker.status()['ready'])
        worker.debug_command('enable')
        self.assertTrue(worker.status()['ready'])

    def test_cancel_event_set_before_enqueue_rejects_with_no_writes(self):
        worker, sdk = self.worker(FakeSDK(enabled=0))
        cancel = threading.Event()
        cancel.set()
        with self.assertRaisesRegex(RuntimeError, '取消'):
            worker.debug_command('enable', cancel_event=cancel)
        self.assertEqual(sdk.writes, [])

    def test_hold_between_target_and_trigger_cancels_old_action_without_enable(self):
        worker, sdk = self.worker()
        sdk.on_write = lambda params, values: worker.hold() if params.address == 0x0102 else None
        with self.assertRaisesRegex(RuntimeError, '取消'):
            worker.debug_command('open')
        self.assertFalse(any(a == 0x0108 and v == [1] for a, v in sdk.writes))
        self.assertEqual(sdk.writes[-1], (0x0100, [0]))

    def test_external_cancel_between_enable_preparation_and_enable(self):
        worker, sdk = self.worker(FakeSDK(enabled=0))
        cancel = threading.Event()
        sdk.on_write = lambda params, values: cancel.set() if params.address == 0x0102 else None
        with self.assertRaisesRegex(RuntimeError, '取消'):
            worker.debug_command('enable', cancel_event=cancel)
        self.assertFalse(any(a == 0x0100 and v == [1] for a, v in sdk.writes))

    def test_hold_never_waits_for_sdk_and_debug_busy_excludes_stream_command(self):
        worker, sdk = self.worker()
        entered, released = threading.Event(), threading.Event()
        def block(params, values):
            if params.address == 0x0102:
                entered.set()
                released.wait(1)
        sdk.on_write = block
        errors = []
        def command():
            try:
                worker.debug_command('open')
            except Exception as exc:
                errors.append(str(exc))
        caller = threading.Thread(target=command)
        caller.start()
        self.assertTrue(entered.wait(.5))
        with self.assertRaisesRegex(RuntimeError, '调试'):
            worker.command(.1)
        started = time.monotonic()
        worker.hold()
        self.assertLess(time.monotonic()-started, .05)
        released.set()
        caller.join(1)
        self.assertFalse(caller.is_alive())
        self.assertTrue(errors)
        self.assertFalse(any(a == 0x0108 and v == [1] for a, v in sdk.writes))

    def test_close_keeps_live_thread_reference(self):
        worker = Ctag2F120Worker({})
        class StillAlive:
            def is_alive(self): return True
            def join(self, timeout): pass
        thread = StillAlive()
        worker.thread = thread
        with self.assertRaisesRegex(RuntimeError, '后台线程尚未退出'):
            worker.close()
        self.assertIs(worker.thread, thread)
        self.assertTrue(worker.stop.is_set())
        self.assertIn('尚未退出', worker.status()['error'])

    def test_status_staleness_preserves_original_failure(self):
        worker = Ctag2F120Worker({})
        worker.state.update(error='原始 RTU 失败', received=time.monotonic()-10)
        state = worker.status()
        self.assertEqual(state['error'], '原始 RTU 失败')
        self.assertTrue(state['feedback_stale'])

    def test_poll_caches_mapping_and_refreshes_controls_at_bounded_interval(self):
        worker, sdk = self.worker()
        reads = []
        sdk.on_read = lambda params: reads.append(params.address)
        time.sleep(.06)
        self.assertTrue(reads)
        self.assertEqual(set(reads), {0x060D})
        sdk.regs[0x0100] = 0
        time.sleep(.4)
        self.assertIn(0x0100, reads)
        self.assertNotIn(0x0404, reads)
        self.assertFalse(worker.status()['ready'])

    def test_invalid_enable_and_arrival_feedback_prevent_motion(self):
        for address in (0x0100, 0x0602):
            with self.subTest(address=address):
                sdk = FakeSDK()
                sdk.regs[address] = 2
                worker, sdk = self.worker(sdk)
                with self.assertRaises(RuntimeError):
                    worker.debug_command('move', opening=.7)
                self.assertEqual(sdk.writes, [])

    def test_open_releases_force_hold_then_enables_at_measured_position_once(self):
        sdk = FakeSDK()
        sdk.regs[0x0601] = 1
        worker, sdk = self.worker(sdk)
        result = worker.debug_command('open')
        self.assertTrue(result['operation_result']['force_hold_released'])
        self.assertEqual(sdk.writes[:4], [
            (0x0100, [0]), (0x0108, [0]), (0x0102, [0, 6000]), (0x0100, [1])])
        self.assertEqual(sum(a == 0x0100 and v == [1] for a, v in sdk.writes), 1)
        self.assertEqual(result['opening'], 1)

    def test_force_release_cancellation_never_reenables(self):
        sdk = FakeSDK()
        sdk.regs[0x0601] = 1
        worker, sdk = self.worker(sdk)
        cancel = threading.Event()
        sdk.on_write = lambda params, values: cancel.set() if params.address == 0x0100 and values == [0] else None
        with self.assertRaisesRegex(RuntimeError, '取消'):
            worker.debug_command('open', cancel_event=cancel)
        self.assertFalse(any(a == 0x0100 and v == [1] for a, v in sdk.writes))
        self.assertFalse(any(a == 0x0108 and v == [1] for a, v in sdk.writes))

    def test_existing_motion_requires_explicit_stop_before_debug_target(self):
        sdk = FakeSDK()
        sdk.regs[0x0601] = sdk.regs[0x0602] = 0
        worker, sdk = self.worker(sdk)
        with self.assertRaisesRegex(RuntimeError, '仍在运动'):
            worker.debug_command('open')
        self.assertEqual(sdk.writes, [])

    def test_final_motion_read_rechecks_alarm(self):
        worker, sdk = self.worker()
        reads = 0
        def fail_final(params):
            nonlocal reads
            if params.address == 0x060D:
                reads += 1
                if reads == 3:
                    sdk.regs[0x0612] = 2
        sdk.on_read = fail_final
        with self.assertRaisesRegex(RuntimeError, '报警'):
            worker.debug_command('move', opening=.5)
        self.assertFalse(worker.status()['command_completed'])

    def test_failed_stop_stays_latched_across_healthy_poll_until_explicit_disable(self):
        sdk = FakeSDK()
        sdk.auto_arrive = False
        worker, sdk = self.worker(sdk)
        with patch.object(worker, '_confirm_disabled', side_effect=RuntimeError('mock stop failure')):
            with self.assertRaises(TimeoutError):
                worker.debug_command('open', timeout_s=.07)
            self.wait_idle(worker)
        time.sleep(.04)
        state = worker.status()
        self.assertFalse(state['ready'])
        self.assertTrue(state['recovery_required'])
        self.assertIn('停止未确认', state['error'])
        with self.assertRaises(RuntimeError):
            worker.command(.4)
        with self.assertRaisesRegex(RuntimeError, '停止未确认'):
            worker.debug_command('enable')
        worker.debug_command('disable')
        self.assertFalse(worker.status()['recovery_required'])
        worker.debug_command('enable')
        self.assertTrue(worker.status()['ready'])

    def test_stream_lost_ack_confirms_disable_and_does_not_resend_target(self):
        worker, sdk = self.worker()
        original_write = sdk.rm_write_registers
        def lose_ack(params, raw):
            result = original_write(params, raw)
            return -1 if params.address == 0x0102 and params.num == 7 else result
        sdk.rm_write_registers = lose_ack
        worker.command(.8)
        deadline = time.monotonic()+.5
        while not worker.status().get('last_command_error') and time.monotonic() < deadline:
            time.sleep(.01)
        time.sleep(.05)
        self.assertEqual(sum(a == 0x0102 and len(v) == 7 for a, v in sdk.writes), 1)
        self.assertIn((0x0100, [0]), sdk.writes)
        self.assertIsNone(worker.desired)
        state = worker.status()
        self.assertFalse(state['ready'])
        self.assertFalse(state['enabled'])
        self.assertTrue(state['stop_confirmed'])
        self.assertIn('下发夹爪位置', state['last_command_error'])

    def test_stream_lost_ack_and_failed_stop_latches_across_healthy_poll(self):
        worker, sdk = self.worker()
        original_write = sdk.rm_write_registers
        def lose_ack(params, raw):
            result = original_write(params, raw)
            return -1 if params.address == 0x0102 and params.num == 7 else result
        sdk.rm_write_registers = lose_ack
        with patch.object(worker, '_confirm_disabled', side_effect=RuntimeError('mock stop failure')):
            worker.command(.8)
            deadline = time.monotonic()+.5
            while not worker.status().get('recovery_required') and time.monotonic() < deadline:
                time.sleep(.01)
        time.sleep(.05)
        state = worker.status()
        self.assertFalse(state['ready'])
        self.assertFalse(state['stop_confirmed'])
        self.assertTrue(state['recovery_required'])
        self.assertIn('停止未确认', state['last_command_error'])
        self.assertEqual(sum(a == 0x0102 and len(v) == 7 for a, v in sdk.writes), 1)
        self.assertIsNone(worker.desired)
        with self.assertRaises(RuntimeError):
            worker.command(.9)
        worker.debug_command('disable')
        self.assertFalse(worker.status()['recovery_required'])

    def test_hold_allows_new_normal_command(self):
        worker, sdk = self.worker()
        worker.hold()
        worker.command(.75)
        deadline = time.monotonic()+.5
        while not sdk.writes and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(sdk.writes)


if __name__ == '__main__':
    unittest.main()
