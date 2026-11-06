"""Experimental freshness budgets; fake register reads and no hardware writes."""
import unittest
from unittest.mock import patch

from realman_sync_client import gripper
from test_gripper_multirate import Clock, RegisterDevice


class GripperToleranceTests(unittest.TestCase):
    def setUp(self):
        self.timer = Clock()
        active = patch.object(gripper.clock, 'now', self.timer.now)
        active.start()
        self.addCleanup(active.stop)

    def make(self, limit=None, **config):
        if limit is not None:
            config['gripper_control_max_age_s'] = limit
        worker = gripper.Ctag2F120Worker(config)
        worker.mapping_min, worker.mapping_max = 0, 12000
        device = RegisterDevice(self.timer)
        worker._poll_feedback(device, lambda *args: args, 0, 12000)
        return worker, device

    def test_default_rejects_1200_ms_but_experimental_budget_can_queue(self):
        strict, strict_device = self.make()
        relaxed, relaxed_device = self.make(1.5)
        self.timer.advance(1.2)
        self.assertFalse(strict.status()['ready'])
        with self.assertRaises(RuntimeError):
            strict.command(.5)
        self.assertTrue(relaxed.status()['ready'])
        self.assertEqual(relaxed.status()['control_max_age_ms'], 1500)
        self.assertEqual(relaxed.status()['feedback_schedule']['control_max_age_ms'], 1500)
        before = list(relaxed_device.calls)
        relaxed.command(.5)
        self.assertEqual(relaxed.desired, .5)
        self.assertEqual(before, relaxed_device.calls)  # command admission only
        self.assertTrue(all('write' not in name for name, _ in strict_device.calls + relaxed_device.calls))

    def test_experimental_limit_is_bounded_and_cannot_renew_old_enable(self):
        worker, device = self.make(1.5)
        enable_stamp = worker.enabled_received
        self.timer.advance(1.49)
        self.assertFalse(worker.status()['ready'])  # enable read is older than position
        self.assertEqual(worker.enabled_received, enable_stamp)
        with self.assertRaises(RuntimeError):
            worker.command(.5)
        self.assertIsNone(worker.desired)

    def test_explicit_age_check_can_tighten_but_not_bypass_control_limit(self):
        worker, _ = self.make(1.5)
        self.timer.advance(.6)
        self.assertTrue(worker.status()['ready'])
        self.assertFalse(worker.status(.5)['ready'])
        self.timer.advance(1.)
        self.assertFalse(worker.status(2.)['ready'])

    def test_manual_ticket_and_position_transition_use_same_budget(self):
        worker, device = self.make(1.5)
        ticket = worker._control_ticket_locked()
        self.timer.advance(1.2)
        self.assertTrue(worker._control_fresh(enabled=True))
        worker._require_control(ticket, fresh=True, enabled=True)
        worker._transition_position_read(device, lambda *args: args)
        self.assertFalse(worker.state['communication_fault'])
        self.timer.advance(.3)
        self.assertFalse(worker._control_fresh(enabled=True))
        with self.assertRaisesRegex(RuntimeError, '1500ms'):
            worker._require_control(ticket, fresh=True, enabled=True)

    def test_delayed_read_keeps_original_request_timestamp(self):
        worker = gripper.Ctag2F120Worker({'gripper_control_max_age_s': 1.5})
        worker.mapping_min, worker.mapping_max = 0, 12000
        device = RegisterDevice(self.timer)
        device.position_delay = 1.2
        _, ready, _ = worker._poll_feedback(device, lambda *args: args, 0, 12000)
        self.assertTrue(ready)
        self.assertAlmostEqual(worker.status()['age_ms'], 1200)
        self.assertAlmostEqual(worker.state['measurement_started'], 10.01)
        self.assertAlmostEqual(worker.history[-1][0], 11.21)
        self.timer.advance(.31)
        self.assertFalse(worker.status()['ready'])

    def test_alarm_and_disabled_feedback_remain_blocking_in_relaxed_mode(self):
        for enabled, alarm in ((1, 32), (0, 0)):
            with self.subTest(enabled=enabled, alarm=alarm):
                worker, device = self.make(1.5)
                device.enabled, device.alarm = enabled, alarm
                self.timer.advance(.26)
                worker._poll_feedback(device, lambda *args: args, 0, 12000)
                self.assertFalse(worker.status()['ready'])
                with self.assertRaises(RuntimeError):
                    worker.command(.5)
                self.assertIsNone(worker.desired)

    def test_dispatch_reserve_uses_configured_deadline(self):
        worker, _ = self.make(1.5, gripper_transport='json')
        self.timer.advance(1.)
        self.assertTrue(worker._motion_has_feedback_budget('policy'))
        self.assertLessEqual(worker._enable_refresh_interval(), worker.enable_poll_s)
        self.timer.advance(.3)
        self.assertFalse(worker._motion_has_feedback_budget('policy'))

    def test_invalid_and_unbounded_control_budgets_are_rejected(self):
        for limit in (True, False, 0, -1, 2.001, float('nan'), float('inf'), '1.5'):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                gripper.Ctag2F120Worker({'gripper_control_max_age_s': limit})
        self.assertEqual(gripper.Ctag2F120Worker({}).control_max_age_s, .5)
        self.assertEqual(gripper.Ctag2F120Worker({'gripper_control_max_age_s': 2}).control_max_age_s, 2.)


if __name__ == '__main__':
    unittest.main()
