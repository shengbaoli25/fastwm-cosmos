"""Clock-driven transport recovery tests; no SDK or network connection."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from realman_sync_client.gripper_transport import JsonModbusArm, ModbusTransportError


def params(address=0x060D, count=6):
    return SimpleNamespace(port=1, device=1, address=address, num=count)


class TimedLink:
    def __init__(self, timer, delays):
        self.timer, self.delays = timer, iter(delays)
        self.sent = []
        self.outstanding = None
        self.closed = False

    def pending(self):
        return self.outstanding is not None and self.outstanding[0] <= self.timer[0]

    def send(self, request, deadline):
        if self.outstanding is not None:
            raise AssertionError('A second transaction overlapped its predecessor')
        if deadline <= self.timer[0]:
            raise TimeoutError('Expired send')
        self.sent.append(deepcopy(request))
        command = request['command']
        if command.startswith('write_'):
            reply = dict(command=command, write_state=True)
        elif command == 'read_holding_registers':
            reply = dict(command=command, data=1)
        else:
            reply = dict(command=command, data=[0, len(self.sent)]+[0]*10)
        self.outstanding = self.timer[0]+next(self.delays), reply

    def receive(self, deadline):
        due, reply = self.outstanding
        if due >= deadline:
            self.timer[0] = deadline
            raise TimeoutError('Controller response timeout')
        self.timer[0] = max(self.timer[0], due)
        self.outstanding = None
        return reply

    def close(self):
        self.closed = True


class FenceDeadlineTests(unittest.TestCase):
    def make(self, delays):
        timer = [10.]
        link = TimedLink(timer, delays)
        arm = JsonModbusArm('not-connected', 8080, .3, link_factory=lambda cfg: link)
        self.addCleanup(arm.close)
        clock = patch('realman_sync_client.gripper_transport.time.perf_counter',
                      side_effect=lambda: timer[0])
        clock.start()
        self.addCleanup(clock.stop)
        return arm, link, timer

    def test_late_original_250ms_then_healthy_70ms_fence_succeeds(self):
        arm, link, timer = self.make([.55, .07, .06])
        with self.assertRaises(ModbusTransportError):
            arm.rm_read_multiple_holding_registers(params())
        self.assertAlmostEqual(timer[0], 10.3)
        self.assertEqual(arm.rm_read_multiple_holding_registers(params())[1][1], 3)
        self.assertAlmostEqual(timer[0], 10.68)
        self.assertEqual([r['command'] for r in link.sent],
            ['read_multiple_holding_registers', 'read_holding_registers',
             'read_multiple_holding_registers'])
        self.assertFalse(arm.needs_sync)
        self.assertEqual(arm.timeouts, 1)
        self.assertEqual(arm.last_event['synchronization']['discarded'][0]['data'][1], 1)

    def test_new_fence_deadline_never_extends_total_two_second_budget(self):
        arm, link, timer = self.make([2.24, .07])
        with self.assertRaises(ModbusTransportError):
            arm.rm_read_multiple_holding_registers(params())
        original_deadline = arm._recovery_deadline
        timer[0] = 12.21
        with self.assertRaisesRegex(ModbusTransportError, '2-second recovery budget'):
            arm.rm_read_multiple_holding_registers(params())
        self.assertAlmostEqual(timer[0], original_deadline)
        self.assertAlmostEqual(timer[0], 12.3)
        self.assertTrue(link.closed)
        self.assertTrue(arm.status()['failed'])
        self.assertEqual(len(link.sent), 2)
        with self.assertRaisesRegex(ModbusTransportError, 'explicit reconnect'):
            arm.rm_read_multiple_holding_registers(params())
        self.assertEqual(len(link.sent), 2)

    def test_delayed_fence_is_consumed_without_resending_another(self):
        arm, link, timer = self.make([.55, .40, .06])
        for _ in range(2):
            with self.assertRaises(ModbusTransportError):
                arm.rm_read_multiple_holding_registers(params())
        self.assertAlmostEqual(timer[0], 10.85)
        self.assertEqual(len(link.sent), 2)
        self.assertEqual(arm.rm_read_multiple_holding_registers(params())[1][1], 3)
        self.assertAlmostEqual(timer[0], 11.01)
        self.assertEqual(len(link.sent), 3)
        self.assertEqual(sum(r['command']=='read_holding_registers' for r in link.sent), 1)

    def test_late_write_ack_is_not_replayed_or_counted_as_new_measurement(self):
        arm, link, timer = self.make([.55, .07, .06])
        with self.assertRaises(ModbusTransportError):
            arm.rm_write_registers(params(0x0102, 7), [0]*14)
        with self.assertRaisesRegex(ModbusTransportError, 'read must resynchronize'):
            arm.rm_write_registers(params(0x0102, 7), [0]*14)
        self.assertEqual(arm.rm_read_multiple_holding_registers(params())[1][1], 3)
        self.assertEqual(sum(r['command']=='write_registers' for r in link.sent), 1)
        self.assertTrue(arm.last_event['synchronization']['discarded'][0]['write_state'])
        self.assertEqual(arm.successes, 1)


if __name__ == '__main__':
    unittest.main()
