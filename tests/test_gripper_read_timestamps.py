"""Fresh request causality across late replies/fences; no real sockets."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from realman_sync_client import gripper
from realman_sync_client.gripper_transport import JsonModbusArm, ModbusTransportError
from test_gripper_fence_deadline import TimedLink


def params(port, address, device, num):
    return SimpleNamespace(port=port, address=address, device=device, num=num)


class ReadTimestampTests(unittest.TestCase):
    def make(self, delays):
        timer = [10.]
        link = TimedLink(timer, delays)
        transport = JsonModbusArm('unused', 8080, .3, link_factory=lambda _: link)
        self.addCleanup(transport.close)
        patcher = patch('realman_sync_client.gripper_transport.time.perf_counter',
                        side_effect=lambda: timer[0])
        patcher.start()
        self.addCleanup(patcher.stop)
        worker = gripper.Ctag2F120Worker(dict(gripper_transport='json'))
        worker.mapping_min, worker.mapping_max = 0, 1000000
        return worker, transport, link, timer

    def test_poll_uses_actual_fresh_read_send_after_long_recovery(self):
        worker, transport, link, timer = self.make([.55, .25, .06, .06])
        with self.assertRaises(ModbusTransportError):
            worker._poll_feedback(transport, params, 0, 1000000)
        self.assertEqual(len(worker.history), 0)
        # This call spends .50 s retiring the old enable and reading a fence
        # before its own enable read is sent. Those waits are not sensor age.
        _, ready, call_start = worker._poll_feedback(transport, params, 0, 1000000)
        self.assertAlmostEqual(call_start, 10.3)
        self.assertTrue(ready)
        self.assertAlmostEqual(worker.enabled_received, 10.8)
        self.assertAlmostEqual(worker.state['measurement_started'], 10.86)
        self.assertAlmostEqual(worker.state['response_received'], 10.92)
        self.assertAlmostEqual(worker.state['read_duration_ms'], 60.)
        self.assertEqual(len(worker.history), 1)  # original/fence never samples
        self.assertEqual(worker.state['position'], 4 << 16)  # fresh fourth request
        self.assertAlmostEqual(worker.history_started[0][1], 10.86)

    def test_transport_returns_metadata_only_for_final_validated_request(self):
        _, transport, link, timer = self.make([.55, .25, .06])
        p = params(1, 0x060D, 1, 6)
        with self.assertRaises(ModbusTransportError):
            transport.read_registers_timed(p)
        result, sent = transport.read_registers_timed(p)
        self.assertAlmostEqual(sent, 10.8)
        self.assertEqual(result[1][1], 3)  # distinct from original reply 1
        self.assertAlmostEqual(timer[0]-sent, .06)
        self.assertEqual(len(link.sent), 3)

    def test_fence_failure_cannot_publish_new_feedback_or_refresh_old_stamp(self):
        worker, transport, link, timer = self.make([.55, .4, .06, .06])
        worker.state = dict(received=9.9, measurement_started=9.8, ready=False)
        worker.history.append((9.9, .5))
        worker.history_started.append((9.9, 9.8))
        for _ in range(2):
            with self.assertRaises(ModbusTransportError):
                worker._poll_feedback(transport, params, 0, 1000000)
            self.assertEqual(list(worker.history_started), [(9.9, 9.8)])
            self.assertEqual(worker.state['measurement_started'], 9.8)
        self.assertEqual(len(link.sent), 2)  # one original, one outstanding fence
        worker._poll_feedback(transport, params, 0, 1000000)
        self.assertEqual(worker.history_started[0], (9.9, 9.8))
        self.assertGreater(timer[0]-worker.history_started[0][1], .5)
        self.assertEqual(len(worker.history), 2)  # fresh sample appended only

    def test_manual_enable_read_uses_actual_send_after_recovery(self):
        worker, transport, _, timer = self.make([.55, .25, .06])
        with self.assertRaises(ModbusTransportError):
            worker._transition_enabled_read(transport, params)
        self.assertTrue(worker._transition_enabled_read(transport, params))
        self.assertAlmostEqual(worker.enabled_received, 10.8)
        self.assertAlmostEqual(timer[0]-worker.enabled_received, .06)

    def test_manual_position_read_uses_actual_send_after_recovery(self):
        worker, transport, _, timer = self.make([.55, .25, .06])
        with self.assertRaises(ModbusTransportError):
            worker._transition_position_read(transport, params)
        raw, _ = worker._transition_position_read(transport, params)
        self.assertEqual(raw, 3 << 16)
        self.assertAlmostEqual(worker.state['measurement_started'], 10.8)
        self.assertAlmostEqual(worker.state['read_duration_ms'], 60.)
        # A fresh position alone cannot refresh an old/missing enable sample.
        self.assertTrue(worker.state['communication_fault'])

    def test_sdk_without_metadata_retains_call_entry_as_conservative_stamp(self):
        worker, _, _, timer = self.make([])
        class SDK:
            def rm_read_holding_registers(self, _):
                timer[0] += .55
                return 0, 1
        self.assertTrue(worker._transition_enabled_read(SDK(), params))
        self.assertAlmostEqual(worker.enabled_received, 10.)
        self.assertGreater(timer[0]-worker.enabled_received, .5)

    def test_invalid_metadata_is_rejected_instead_of_fabricating_freshness(self):
        worker, _, _, timer = self.make([])
        for invalid in (9.9, 10.1, float('nan'), True, None):
            with self.subTest(stamp=invalid):
                class BadMetadata:
                    def read_registers_timed(self, _):
                        return (0, 1), invalid
                with self.assertRaisesRegex(RuntimeError, '时间戳无效'):
                    worker._transition_enabled_read(BadMetadata(), params)
                self.assertIsNone(worker.enabled_received)


if __name__ == '__main__':
    unittest.main()
