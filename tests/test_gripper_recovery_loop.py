"""Exercise the actual worker loop with timed RTU failures, never hardware."""
from concurrent.futures import Future
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from realman_sync_client import gripper
from realman_sync_client.gripper_transport import ModbusTransportError, MotionCancelled


class Timer:
    def __init__(self):
        self.value = 10.

    def now(self):
        return self.value

    def advance(self, delay):
        self.value += max(0., delay)
        if self.value > 20:
            raise AssertionError('Worker loop exceeded virtual test deadline')


class Event:
    def __init__(self, timer):
        self.timer, self.flag, self.on_wait = timer, False, None

    def set(self):
        self.flag = True

    def clear(self):
        self.flag = False

    def is_set(self):
        return self.flag

    def wait(self, timeout):
        if self.on_wait:
            self.on_wait()
        if not self.flag:
            self.timer.advance(timeout)
        return self.flag


class SetupSDK:
    def __init__(self):
        self.deleted = False

    def rm_create_robot_arm(self, *_):
        return SimpleNamespace(id=1)

    def rm_delete_robot_arm(self):
        self.deleted = True


class JsonIO:
    """SDK-shaped fake, including real-time cost and transport guard hooks."""
    def __init__(self, worker, timer):
        self.worker, self.timer = worker, timer
        self.motion_guard = self.control_guard = None
        self.failed, self.closed = False, False
        self.positions = self.enabled_reads = 0
        self.writes = []
        self.on_position = self.on_enable_read = self.on_alarm = None
        self.position = 0

    def status(self):
        return dict(backend='json', failed=self.failed)

    def close(self):
        self.closed = True

    def rm_get_current_arm_state(self):
        return 0, {'err': {'err': []}}

    def rm_read_holding_registers(self, args):
        assert args == (1, 0x0100, 1, 1)
        self.enabled_reads += 1
        self.timer.advance(.06)
        if self.on_enable_read:
            self.on_enable_read()
        return 0, 1

    def rm_read_multiple_holding_registers(self, args):
        if args == (1, 0x0610, 1, 3):
            self.timer.advance(.06)
            if self.on_alarm:
                self.on_alarm()
            return 0, [0]*6
        assert args == (1, 0x060D, 1, 6)
        self.positions += 1
        self.timer.advance(.06)
        if self.on_position:
            self.on_position()
        words = [self.position >> 16, self.position & 65535, 0, 0, 0, 0]
        return 0, [byte for word in words for byte in word.to_bytes(2, 'big')]

    def rm_write_registers(self, args, data):
        assert args == (1, 0x0102, 1, 7)
        if self.motion_guard is not None and not self.motion_guard():
            raise MotionCancelled('cancelled')
        self.writes.append((args[1], int.from_bytes(bytes(data[:4]), 'big', signed=True)))
        self.timer.advance(.06)
        self.worker.stop.set()
        return 0

    def rm_write_single_register(self, args, data):
        assert args == (1, 0x0100, 1, 1)
        if data == 1 and self.control_guard is not None and not self.control_guard():
            raise MotionCancelled('enable cancelled')
        self.writes.append((args[1], data))
        self.timer.advance(.06)
        return 0


class RecoveryLoopTests(unittest.TestCase):
    def make(self, **config):
        timer = Timer()
        worker = gripper.Ctag2F120Worker(dict(robot_ip='not-connected', robot_port=8080,
            gripper_transport='json', gripper_poll_s=.1, gripper_enable_poll_s=.25,
            gripper_command_interval_s=.1, max_gripper_step=.15, **config))
        worker.stop, worker.wake = Event(timer), Event(timer)
        return worker, timer, SetupSDK(), JsonIO(worker, timer)

    def run_loop(self, worker, timer, sdk, io):
        module = SimpleNamespace(rm_peripheral_read_write_params_t=lambda *args: args)
        with patch.dict('sys.modules', {'Robotic_Arm.rm_robot_interface': module}), \
                patch('realman_sync_client.sdk.create_sdk', return_value=sdk), \
                patch('realman_sync_client.gripper_transport.JsonModbusArm', return_value=io), \
                patch.object(worker, '_setup', return_value=(0, 12000, 1.)), \
                patch.object(gripper.clock, 'now', timer.now):
            worker._run()
        self.assertTrue(sdk.deleted)
        self.assertTrue(io.closed)

    def test_read_timeout_cancels_old_target_and_rebases_new_ramp_on_measurement(self):
        worker, timer, sdk, io = self.make()
        accepted = []
        def position():
            if io.positions == 2:
                timer.advance(.24)
                raise ModbusTransportError('late position')
            if io.positions >= 3:
                io.position = 9600  # actual opening .20 after interrupted movement
        io.on_position = position
        def intent():
            if io.positions == 1 and not accepted:
                worker.command(0.)
                accepted.append('old')
            elif io.positions == 2:
                self.assertIsNone(worker.desired)
                self.assertFalse(worker.status()['ready'])
            elif io.positions == 3 and accepted == ['old']:
                self.assertIsNone(worker.desired)  # recovery never resumes old target
                self.assertTrue(worker.status()['ready'])
                self.assertAlmostEqual(worker.status()['opening'], .2)
                worker.command(.9)
                accepted.append('new')
        worker.wake.on_wait = intent
        self.run_loop(worker, timer, sdk, io)
        self.assertEqual(accepted, ['old', 'new'])
        self.assertEqual(io.writes, [(0x0102, 7800)])  # .20 + .15 = .35 opening
        self.assertAlmostEqual(worker.sent_target['previous_opening'], .2)
        self.assertAlmostEqual(worker.sent_target['opening'], .35)
        self.assertEqual(worker.fault_generation, 1)
        self.assertEqual(len(worker.history), 3)  # failed read supplied no history sample

    def test_fatal_transport_stops_after_one_failed_read_without_busy_retry(self):
        worker, timer, sdk, io = self.make()
        def fail():
            if io.positions == 2:
                io.failed = True
                raise ModbusTransportError('closed stream')
        io.on_position = fail
        worker.wake.on_wait = lambda: worker.command(0.) if io.positions == 1 else None
        self.run_loop(worker, timer, sdk, io)
        self.assertEqual(io.positions, 2)
        self.assertEqual(io.writes, [])
        self.assertFalse(worker.state['ready'])
        self.assertIn('closed stream', worker.state['error'])

    def test_startup_timeout_recovers_once_or_exits_at_finite_deadline(self):
        for persistent in (False, True):
            with self.subTest(persistent=persistent):
                worker, timer, sdk, io = self.make(gripper_start_timeout_s=.7)
                def failure():
                    if persistent or io.enabled_reads == 1:
                        timer.advance(.24)
                        raise ModbusTransportError('initial read timeout')
                io.on_enable_read = failure
                worker.wake.on_wait = worker.stop.set
                self.run_loop(worker, timer, sdk, io)
                self.assertEqual(io.writes, [])
                if persistent:
                    self.assertEqual(io.enabled_reads, 2)
                    self.assertFalse(worker.state['ready'])
                    self.assertEqual(len(worker.history), 0)
                    self.assertLess(timer.value, 11.)
                else:
                    self.assertEqual(io.enabled_reads, 2)
                    self.assertTrue(worker.state['ready'])
                    self.assertEqual(len(worker.history), 1)

    def test_stop_during_enable_alarm_read_cancels_enable_before_socket_dispatch(self):
        worker, timer, sdk, io = self.make()
        future = Future()
        def enqueue():
            if io.positions == 1 and worker.requests.empty():
                worker.requests.put((True, future))
        worker.wake.on_wait = enqueue
        io.on_alarm = worker.stop.set
        self.run_loop(worker, timer, sdk, io)
        self.assertIsInstance(future.exception(), MotionCancelled)
        self.assertEqual(io.writes, [])
        self.assertIsNone(io.control_guard)


if __name__ == '__main__':
    unittest.main()
