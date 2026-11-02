"""Request quiet-time fault injection; no robot, sockets, or motion."""
from copy import deepcopy
import threading
import time
import unittest
from unittest.mock import patch

from realman_sync_client.gripper import Ctag2F120Worker
from realman_sync_client.gripper_transport import JsonModbusArm, ModbusTransportError, MotionCancelled
from test_gripper_transport import FakeLink, params, single, multiple


class Timer:
    def __init__(self):
        self.value = 10.

    def now(self):
        return self.value


class WaitEvent:
    def __init__(self, timer):
        self.timer, self.flag = timer, False
        self.waits = []
        self.on_wait = None

    def set(self):
        self.flag = True

    def wait(self, duration):
        self.waits.append(duration)
        if self.on_wait:
            self.on_wait()
        if not self.flag:
            self.timer.value += duration
        return self.flag


class TimedLink(FakeLink):
    def __init__(self, timer):
        super().__init__({})
        self.timer = timer
        self.send_times, self.reply_times = [], []

    def send(self, payload, deadline):
        self.send_times.append(self.timer.now())
        super().send(payload, deadline)

    def receive(self, deadline):
        delay, reply = self.replies.pop(0)
        self.deadlines.append(deadline)
        if self.timer.now()+delay > deadline+1e-9:
            self.timer.value = deadline
            raise TimeoutError('deadline')
        self.timer.value += delay
        if isinstance(reply, Exception):
            raise reply
        self.reply_times.append(self.timer.now())
        return deepcopy(reply)


class PacingTests(unittest.TestCase):
    def make(self, gap=.01):
        timer = Timer()
        link = TimedLink(timer)
        arm = JsonModbusArm('unused', 8080, request_gap_s=gap, link_factory=lambda cfg: link)
        arm._close_event = WaitEvent(timer)
        patcher = patch('realman_sync_client.gripper_transport.time.perf_counter', timer.now)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(arm.close)
        return arm, link, timer

    def test_normal_replies_enforce_actual_send_gap_and_trace(self):
        arm, link, timer = self.make()
        link.replies = [(.06, single()), (.06, multiple()), (.06, single())]
        arm.rm_read_holding_registers(params(0x100, 1))
        arm.rm_read_multiple_holding_registers(params())
        timer.value += .004
        arm.rm_read_holding_registers(params(0x100, 1))
        for index in (1, 2):
            self.assertAlmostEqual(link.send_times[index]-link.reply_times[index-1], .01)
        self.assertEqual(arm.status()['request_gap_ms'], 10.)
        event = arm.last_event['send_pacing'][0]
        self.assertAlmostEqual(event['wait_ms'], 6.)
        self.assertAlmostEqual(event['reply_gap_ms'], 10.)

    def test_existing_idle_time_and_default_zero_do_not_add_wait(self):
        for gap in (0., .01):
            with self.subTest(gap=gap):
                arm, link, timer = self.make(gap)
                link.replies = [(.06, single()), (.06, multiple())]
                arm.rm_read_holding_registers(params(0x100, 1))
                timer.value += .02
                arm.rm_read_multiple_holding_registers(params())
                self.assertEqual(arm._close_event.waits, [])

    def test_late_reply_fence_and_recovered_request_each_have_quiet_time(self):
        arm, link, _ = self.make(.05)
        link.replies = [(.3, TimeoutError()), (.25, multiple(99)), (.06, single()), (.06, multiple(3))]
        with self.assertRaises(ModbusTransportError):
            arm.rm_read_multiple_holding_registers(params())
        self.assertEqual(arm.rm_read_multiple_holding_registers(params())[1][1], 3)
        self.assertEqual(len(link.sent), 3)
        self.assertAlmostEqual(link.send_times[1]-link.reply_times[0], .05)
        self.assertAlmostEqual(link.send_times[2]-link.reply_times[1], .05)
        self.assertEqual([row['stage'] for row in arm.last_event['send_pacing']], ['fence', 'request'])
        self.assertFalse(arm.status()['needs_sync'])

    def test_late_fence_is_not_resent_and_still_paces_new_request(self):
        arm, link, _ = self.make()
        link.replies = [(.3, TimeoutError()), (.1, multiple()), (.3, TimeoutError()),
                        (.06, single()), (.06, multiple(4))]
        for _ in range(2):
            with self.assertRaises(ModbusTransportError):
                arm.rm_read_multiple_holding_registers(params())
        self.assertEqual(arm.rm_read_multiple_holding_registers(params())[1][1], 4)
        self.assertEqual(len(link.sent), 3)
        self.assertAlmostEqual(link.send_times[2]-link.reply_times[1], .01)

    def test_request_gets_full_timeout_after_pacing(self):
        arm, link, _ = self.make(.05)
        link.replies = [(.06, single()), (.299, multiple(5))]
        arm.rm_read_holding_registers(params(0x100, 1))
        self.assertEqual(arm.rm_read_multiple_holding_registers(params())[1][1], 5)
        self.assertAlmostEqual(link.deadlines[-1]-link.send_times[-1], .3)
        self.assertAlmostEqual(arm.last_event['request_duration_ms'], 299.)
        self.assertAlmostEqual(arm.last_event['duration_ms'], 349.)

    def test_recovery_deadline_includes_fence_pacing_and_never_sends_over_budget(self):
        arm, link, timer = self.make(.05)
        link.replies = [(.3, TimeoutError()), (.29, multiple())]
        with self.assertRaises(ModbusTransportError):
            arm.rm_read_multiple_holding_registers(params())
        expires = arm._recovery_deadline
        timer.value = expires-.3
        with self.assertRaisesRegex(ModbusTransportError, '2-second recovery budget'):
            arm.rm_read_multiple_holding_registers(params())
        self.assertAlmostEqual(timer.now(), expires)
        self.assertEqual(len(link.sent), 1)
        self.assertTrue(link.closed)

    def test_motion_and_control_guards_are_checked_after_wait(self):
        for kind in ('target', 'trigger', 'enable', 'neutral'):
            with self.subTest(kind=kind):
                arm, link, _ = self.make()
                intent = [True]
                arm.motion_guard = arm.control_guard = lambda: intent[0]
                arm._close_event.on_wait = lambda: intent.__setitem__(0, False)
                link.replies = [(.06, single())]
                arm.rm_read_holding_registers(params(0x100, 1))
                with self.assertRaises(MotionCancelled):
                    if kind == 'target':
                        arm.rm_write_registers(params(0x102, 7), [0]*14)
                    elif kind == 'trigger':
                        arm.rm_write_single_register(params(0x108, 1), 1)
                    elif kind == 'enable':
                        arm.rm_write_single_register(params(0x100, 1), 1)
                    else:
                        arm.rm_write_registers(params(0x102, 2), [0]*4)
                self.assertEqual(len(link.sent), 1)
                self.assertEqual(arm.status()['cancellations'], 1)
                self.assertFalse(arm.status()['failed'])

    def test_uncertain_write_is_not_retried_with_pacing(self):
        arm, link, _ = self.make()
        ack = {'command': 'write_registers', 'write_state': True}
        link.replies = [(.3, TimeoutError()), (.1, ack), (.06, single()), (.06, multiple())]
        with self.assertRaises(ModbusTransportError):
            arm.rm_write_registers(params(0x102, 7), [0]*14)
        with self.assertRaises(ModbusTransportError):
            arm.rm_write_registers(params(0x102, 7), [0]*14)
        arm.rm_read_multiple_holding_registers(params())
        self.assertEqual(sum(row['command'] == 'write_registers' for row in link.sent), 1)

    def test_close_interrupts_pacing_without_another_send(self):
        link = FakeLink({})
        arm = JsonModbusArm('unused', 8080, request_gap_s=.05, link_factory=lambda cfg: link)
        self.addCleanup(arm.close)
        link.replies = [single()]
        arm.rm_read_holding_registers(params(0x100, 1))
        waiting = threading.Event()
        original_wait = arm._close_event.wait
        def wait(duration):
            waiting.set()
            return original_wait(duration)
        errors = []
        def call():
            try:
                arm.rm_read_multiple_holding_registers(params())
            except Exception as exc:
                errors.append(exc)
        with patch.object(arm._close_event, 'wait', wait):
            worker = threading.Thread(target=call)
            worker.start()
            self.assertTrue(waiting.wait(1.))
            started = time.monotonic()
            arm.close()
            worker.join(.1)
        self.assertFalse(worker.is_alive())
        self.assertLess(time.monotonic()-started, .1)
        self.assertEqual(len(link.sent), 1)
        self.assertIsInstance(errors[0], ModbusTransportError)

    def test_invalid_configuration_rejected_without_connecting(self):
        for value in (True, False, -.001, .050001, float('nan'), float('inf'), None, '0.01'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'gripper_request_gap_s'):
                    JsonModbusArm('unused', 8080, request_gap_s=value)
                with self.assertRaisesRegex(ValueError, 'gripper_request_gap_s'):
                    Ctag2F120Worker({'gripper_request_gap_s': value})
        self.assertEqual(JsonModbusArm('unused', 8080).status()['request_gap_ms'], 0.)

    def test_worker_budget_includes_gap_without_double_counting_measurements(self):
        worker = Ctag2F120Worker({'gripper_transport': 'json', 'gripper_request_gap_s': .01})
        self.assertAlmostEqual(worker._read_budget(), .085)
        self.assertAlmostEqual(worker._enable_refresh_interval(), .085)
        worker._feedback_read_costs.append(.1)
        self.assertAlmostEqual(worker._read_budget(), .1)
        self.assertEqual(worker.status()['feedback_schedule']['request_gap_ms'], 10.)


if __name__ == '__main__':
    unittest.main()
