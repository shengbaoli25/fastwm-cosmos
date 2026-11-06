"""Documented controller read failures are complete transactions, not data."""
import unittest
from unittest.mock import Mock, patch

from realman_sync_client.gripper_transport import JsonModbusArm, ModbusTransportError, PeripheralReadTimeout
from test_gripper_fence_deadline import TimedLink
import test_gripper_transport as fixture
from test_gripper_transport import params, single, multiple


def negative(command='read_multiple_holding_registers'):
    return dict(command=command, read_state=False)


class NegativeReadTests(unittest.TestCase):
    def make(self, **kwargs):
        return fixture.TransportTests.make(self, **kwargs)

    def test_failed_peripheral_read_keeps_tcp_and_next_read_is_fresh(self):
        for method, p, reply in (
            ('rm_read_multiple_holding_registers', params(), multiple(23)),
            ('rm_read_holding_registers', params(0x0100, 1), single(1)),
        ):
            with self.subTest(method=method):
                arm, link = self.make()
                call = getattr(arm, method)
                link.replies = [negative(arm.METHODS[method]), reply]
                with self.assertRaises(PeripheralReadTimeout):
                    call(p)
                status = arm.status()
                self.assertEqual(status['failure_kind'], 'peripheral_timeout')
                self.assertEqual(status['peripheral_read_timeouts'], 1)
                self.assertEqual(status['timeouts'], 0)
                self.assertFalse(status['failed'])
                self.assertFalse(status['needs_sync'])
                self.assertFalse(status['reconnect_allowed'])
                self.assertIsNone(arm._pending)
                self.assertFalse(link.closed)
                self.assertEqual(status['successes'], 0)
                actual = call(p)
                self.assertEqual(actual[1], reply['data'])
                self.assertEqual(len(link.sent), 2)  # no hidden read or fence
                self.assertIsNone(arm.status()['failure_kind'])

    def test_retired_negative_read_is_consumed_before_distinct_fence(self):
        arm, link = self.make()
        link.replies = [TimeoutError('late'), negative(), single(), multiple(17)]
        with self.assertRaises(ModbusTransportError):
            arm.rm_read_multiple_holding_registers(params())
        result = arm.rm_read_multiple_holding_registers(params())
        self.assertEqual(result[1][1], 17)
        self.assertEqual([p['command'] for p in link.sent],
            ['read_multiple_holding_registers', 'read_holding_registers', 'read_multiple_holding_registers'])
        self.assertEqual(arm.status()['peripheral_read_timeouts'], 1)
        self.assertFalse(arm.status()['failed'])

    def test_negative_fence_completes_sync_but_returns_no_sample(self):
        arm, link = self.make()
        link.replies = [TimeoutError('late'), multiple(99), negative('read_holding_registers'), multiple(17)]
        with self.assertRaises(ModbusTransportError):
            arm.rm_read_multiple_holding_registers(params())
        with self.assertRaises(PeripheralReadTimeout):
            arm.rm_read_multiple_holding_registers(params())
        self.assertFalse(arm.status()['failed'])
        self.assertFalse(arm.status()['needs_sync'])
        self.assertEqual(len(link.sent), 2)
        self.assertEqual(arm.rm_read_multiple_holding_registers(params())[1][1], 17)
        self.assertEqual(len(link.sent), 3)

    def test_negative_read_must_match_request_and_have_no_conflicting_data(self):
        for reply in (negative('read_holding_registers'), dict(negative(), data=[0]*12),
                      dict(negative(), read_state=0), dict(negative(), err=[32])):
            with self.subTest(reply=reply):
                arm, link = self.make()
                link.replies = [reply]
                with self.assertRaises(RuntimeError):
                    arm.rm_read_multiple_holding_registers(params())
                self.assertTrue(link.closed)
                self.assertEqual(arm.status()['failure_kind'], 'protocol_error')
                self.assertFalse(arm.status()['reconnect_allowed'])

    def test_negative_write_remains_hard_and_is_never_retried(self):
        arm, link = self.make()
        link.replies = [dict(command='write_registers', write_state=False)]
        with self.assertRaises(RuntimeError):
            arm.rm_write_registers(params(0x0102, 7), [0]*14)
        self.assertEqual(len(link.sent), 1)
        self.assertEqual(arm.status()['failure_kind'], 'protocol_error')
        self.assertFalse(arm.status()['reconnect_allowed'])

    def test_connection_loss_and_uncertain_send_allow_readonly_reopen_only(self):
        for send_failure in (False, True):
            with self.subTest(send_failure=send_failure):
                arm, link = self.make()
                if send_failure:
                    link.send_callback = Mock(side_effect=TimeoutError('partial send'))
                else:
                    link.replies = [ConnectionError('TCP closed')]
                with self.assertRaises(ModbusTransportError):
                    arm.rm_read_multiple_holding_registers(params())
                expected = 'send_uncertain' if send_failure else 'connection_lost'
                self.assertTrue(arm.status()['reconnect_allowed'])
                self.assertEqual(arm.status()['failure_kind'], expected)
                with self.assertRaises(ModbusTransportError):
                    arm.rm_read_multiple_holding_registers(params())
                self.assertEqual(len(link.sent), 1)
                self.assertEqual(arm.status()['failure_kind'], expected)
                arm.close()
                self.assertFalse(arm.status()['reconnect_allowed'])

    def test_recovery_budget_exhaustion_allows_only_external_readonly_reopen(self):
        timer = [10.]
        link = TimedLink(timer, [20.])
        arm = JsonModbusArm('unused', 8080, .3, read_timeout_s=.7, link_factory=lambda _: link)
        self.addCleanup(arm.close)
        with patch('realman_sync_client.gripper_transport.time.perf_counter', side_effect=lambda: timer[0]):
            for _ in range(4):
                with self.assertRaises(ModbusTransportError):
                    arm.rm_read_multiple_holding_registers(params())
        self.assertTrue(arm.status()['failed'])
        self.assertTrue(arm.status()['reconnect_allowed'])
        self.assertEqual(arm.status()['failure_kind'], 'recovery_timeout')
        self.assertEqual(len(link.sent), 1)  # transport did not reopen/replay

    def test_connect_error_is_classified_without_any_write(self):
        factory = Mock(side_effect=ConnectionRefusedError('refused'))
        arm = JsonModbusArm('unused', 8080, link_factory=factory)
        self.addCleanup(arm.close)
        with self.assertRaises(ModbusTransportError):
            arm.rm_read_multiple_holding_registers(params())
        self.assertEqual(factory.call_count, 1)
        self.assertTrue(arm.status()['reconnect_allowed'])
        self.assertEqual(arm.status()['failure_kind'], 'connect_error')


if __name__ == '__main__':
    unittest.main()
