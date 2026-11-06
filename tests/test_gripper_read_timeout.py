"""Read tolerance is independent of write ACKs and motion freshness."""
import unittest
from unittest.mock import patch
from realman_sync_client.gripper import Ctag2F120Worker
from realman_sync_client.gripper_transport import JsonModbusArm, ModbusTransportError
from test_gripper_fence_deadline import TimedLink, params


class ReadTimeoutTests(unittest.TestCase):
    def make(self, delays):
        timer = [10.]
        link = TimedLink(timer, delays)
        arm = JsonModbusArm('unused', 8080, .3, read_timeout_s=.7, link_factory=lambda _: link)
        self.addCleanup(arm.close)
        clock = patch('realman_sync_client.gripper_transport.time.perf_counter', side_effect=lambda: timer[0])
        clock.start()
        self.addCleanup(clock.stop)
        return arm, link, timer

    def test_531_ms_read_is_accepted_without_fence_or_retry(self):
        arm, link, timer = self.make([.531])
        self.assertEqual(arm.rm_read_multiple_holding_registers(params())[0], 0)
        self.assertEqual(len(link.sent), 1)
        self.assertEqual(arm.timeouts, 0)
        self.assertAlmostEqual(timer[0], 10.531)
        self.assertEqual(arm.status()['read_timeout_ms'], 700.)

    def test_821_ms_read_still_times_out_at_700_ms(self):
        arm, link, timer = self.make([.821])
        with self.assertRaises(ModbusTransportError):
            arm.rm_read_multiple_holding_registers(params())
        self.assertAlmostEqual(timer[0], 10.7)
        self.assertEqual(len(link.sent), 1)

    def test_write_ack_timeout_is_still_300_ms_without_retry(self):
        arm, link, timer = self.make([.531])
        with self.assertRaises(ModbusTransportError):
            arm.rm_write_single_register(params(0x0100, 1), 0)
        self.assertAlmostEqual(timer[0], 10.3)
        self.assertEqual(len(link.sent), 1)
        self.assertEqual(arm.status()['write_timeout_ms'], 300.)

    def test_slow_valid_read_does_not_extend_500_ms_motion_freshness(self):
        arm, link, timer = self.make([.06, .531])
        worker = Ctag2F120Worker(dict(gripper_transport='json', gripper_read_timeout_s=.7))
        factory = lambda port, address, device, count: params(address, count)
        _, ready, _ = worker._poll_feedback(arm, factory, 0, 1000000)
        self.assertFalse(ready)
        self.assertFalse(worker.status()['ready'])
        self.assertEqual(worker.status()['feedback_schedule']['control_max_age_ms'], 500)
        with self.assertRaises(RuntimeError):
            worker.command(.5)
        self.assertEqual(len(link.sent), 2)

    def test_read_timeout_bounds(self):
        for value in (True, 0, -.1, .701, float('nan'), float('inf'), '0.7'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                JsonModbusArm('unused', 8080, read_timeout_s=value)

    def test_longer_read_wait_preserves_two_second_recovery_budget(self):
        arm, link, timer = self.make([20.])
        with self.assertRaises(ModbusTransportError):
            arm.rm_read_multiple_holding_registers(params())
        deadline = arm._recovery_deadline
        for _ in range(3):
            with self.assertRaises(ModbusTransportError):
                arm.rm_read_multiple_holding_registers(params())
        self.assertAlmostEqual(timer[0], deadline)
        self.assertTrue(arm.status()['failed'])
        self.assertEqual(len(link.sent), 1)
