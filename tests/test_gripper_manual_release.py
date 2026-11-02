"""Explicit manual CTAG release/safe enable, using only a fake register device."""
from concurrent.futures import Future, TimeoutError as FutureTimeout
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from realman_sync_client import gripper
from realman_sync_client.gripper_transport import MotionCancelled


class Clock:
    def __init__(self):
        self.value = 10.

    def now(self):
        return self.value


class Device:
    def __init__(self, worker, timer):
        self.worker, self.timer = worker, timer
        self.enabled, self.position, self.alarm, self.torque = 1, 12000, 0, 1
        self.target = 0  # Previously ACKed open goal did not leave force hold.
        self.events, self.moves = [], []
        self.after = self.before = None
        self.duration = .02
        self.fail_after = None
        self.ignore_disable = False
        self.bad_neutral_readback = False

    def done(self, event):
        self.events.append(event)
        self.timer.value += self.duration
        if self.after:
            self.after(event)
        if event == self.fail_after:
            raise TimeoutError('uncertain ACK injected')

    def guard(self, event):
        if self.before:
            self.before(event)
        control = self.worker.transport.control_guard if self.worker.transport else None
        if control is not None and not control():
            raise MotionCancelled('cancelled after transport drain')

    def rm_get_current_arm_state(self):
        self.done(('arm',))
        return 0, {'err': {'err': []}}

    def rm_create_robot_arm(self, *_):
        return SimpleNamespace(id=1)

    def rm_delete_robot_arm(self):
        pass

    def rm_read_holding_registers(self, params):
        _, address, _, count = params
        assert count == 1
        if address == 0x0100:
            value, event = self.enabled, ('read_enable', self.enabled)
        else:
            assert address == 0x0601
            value, event = self.torque, ('torque', self.torque)
        self.done(event)
        return 0, value

    def rm_read_multiple_holding_registers(self, params):
        _, address, _, count = params
        if address == 0x060D:
            assert count == 6
            raw = self.position & 0xffffffff
            words = [raw >> 16, raw & 65535, 0, 500, 378, self.alarm]
            event = ('position', self.position)
        else:
            assert address == 0x0102 and count == 3
            raw = self.target + int(self.bad_neutral_readback)
            words, event = [raw >> 16, raw & 65535, 100], ('target_readback', raw)
        self.done(event)
        return 0, [byte for word in words for byte in word.to_bytes(2, 'big')]

    def rm_write_single_register(self, params, value):
        assert params == (1, 0x0100, 1, 1)
        event = ('enable', value)
        self.guard(event)
        if value and not self.enabled:
            # Enabling can execute a retained trigger. Neutralization must have
            # replaced that retained goal while the drive was confirmed disabled.
            assert self.target == self.position, 'old goal would move on enable'
        if value or not self.ignore_disable:
            self.enabled = value
        self.done(event)
        return 0

    def rm_write_registers(self, params, payload):
        assert params[1] == 0x0102 and params[3] in (2, 7)
        raw = int.from_bytes(bytes(payload[:4]), 'big', signed=True)
        event = ('neutral' if params[3] == 2 else 'target', raw)
        self.guard(event)
        if params[3] == 2:
            assert self.enabled == 0
        else:
            self.moves.append(raw)
        self.target = raw
        self.done(event)
        return 0


class ManualReleaseTests(unittest.TestCase):
    def setUp(self):
        self.timer = Clock()
        self.worker = gripper.Ctag2F120Worker(dict(robot_ip='unused', robot_port=8080,
            gripper_manual_release=True, max_gripper_step=.15))
        self.worker.mapping_min, self.worker.mapping_max = 0, 12000
        self.worker.transport = SimpleNamespace(control_guard=None,
            status=lambda: {'backend': 'fake'})
        self.device = Device(self.worker, self.timer)
        self.factory = lambda *args: args
        self.clock_patch = patch.object(gripper.clock, 'now', self.timer.now)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.worker._poll_feedback(self.device, self.factory, 0, 12000)
        self.device.events.clear()

    def ticket(self):
        self.worker.hold()
        with self.worker.lock:
            return self.worker._control_ticket_locked()

    def prepare(self):
        return self.worker._manual_open(self.device, self.factory, self.ticket())

    def test_force_hold_release_neutralizes_before_enable_then_queues_from_actual(self):
        result = self.prepare()
        self.assertEqual(self.device.events, [
            ('arm',), ('read_enable', 1), ('position', 12000), ('torque', 1),
            ('enable', 0), ('read_enable', 0), ('position', 12000),
            ('neutral', 12000), ('target_readback', 12000),
            ('enable', 1), ('read_enable', 1), ('read_enable', 1), ('position', 12000)])
        self.assertTrue(result['manual_open_prepared'])
        self.assertTrue(result['manual_release_performed'])
        self.assertTrue(result['queued'])
        self.assertFalse(result['acknowledged'])
        self.assertFalse(result['completed'])
        self.assertEqual(self.device.moves, [])
        self.assertEqual(self.worker.desired, 1.)
        self.assertIsNone(self.worker.sent_target)
        self.assertAlmostEqual(self.worker._next_target(result['opening'])[1], .15)
        self.assertTrue(self.worker.status()['ready'])
        self.assertIsNone(self.worker.transport.control_guard)

    def test_free_opening_does_not_toggle_enable_or_neutralize(self):
        self.device.torque = 0
        result = self.prepare()
        self.assertFalse(result['manual_release_performed'])
        self.assertTrue(result['queued'])
        self.assertFalse(any(e[0] in ('enable', 'neutral', 'target') for e in self.device.events))

    def test_ramp_rebases_at_contact_without_waiting_for_unsent_intermediate(self):
        self.device.position = 9962
        result = self.prepare()
        measured = 1-9962/12000
        self.assertAlmostEqual(result['opening'], measured)
        _, bounded, _, _ = self.worker._next_target(result['opening'])
        self.assertAlmostEqual(bounded, measured+.15)
        self.assertLess(self.worker._raw_target(bounded), 9962)

    def test_safe_enable_replaces_retained_close_goal_without_trigger(self):
        self.device.enabled, self.device.position, self.device.target = 0, 6000, 12000
        self.worker._set_enabled(self.device, self.factory, True)
        events = self.device.events
        self.assertLess(events.index(('neutral', 6000)), events.index(('enable', 1)))
        self.assertLess(events.index(('target_readback', 6000)), events.index(('enable', 1)))
        self.assertTrue(self.worker.status()['ready'])
        self.assertEqual(self.device.target, 6000)
        self.assertEqual(self.device.moves, [])
        self.assertIsNone(self.worker.desired)

    def test_invalid_or_slow_torque_cannot_disable_or_trigger(self):
        for kind in ('invalid', 'slow'):
            with self.subTest(kind=kind):
                self.setUp()
                if kind == 'invalid':
                    self.device.torque = 2
                else:
                    self.device.after = lambda event: setattr(self.timer, 'value', self.timer.value+.51) \
                        if event[0] == 'torque' else None
                with self.assertRaises(RuntimeError):
                    self.prepare()
                self.assertFalse(any(e[0] == 'enable' for e in self.device.events))
                self.assertIsNone(self.worker.desired)

    def test_cancellation_at_each_preparation_boundary_never_triggers(self):
        points = [('torque', 1), ('enable', 0), ('read_enable', 0),
                  ('neutral', 12000), ('target_readback', 12000), ('enable', 1)]
        for point in points:
            with self.subTest(point=point):
                self.setUp()
                def cancel(event):
                    if event == point:
                        self.device.after = None
                        self.worker.hold()
                self.device.after = cancel
                with self.assertRaises(MotionCancelled):
                    self.prepare()
                self.assertEqual(self.device.moves, [])
                self.assertIsNone(self.worker.desired)
                if point != ('torque', 1):
                    self.assertEqual(self.device.enabled, 0)
                self.assertIsNone(self.worker.transport.control_guard)

    def test_pause_resume_epoch_is_not_erased_between_transactions(self):
        def cancel_then_resume(event):
            if event == ('neutral', 12000):
                self.device.after = None
                self.worker.hold()
                with self.worker.lock:
                    self.worker.desired = .3
                    self.worker.command_sequence += 1
        self.device.after = cancel_then_resume
        with self.assertRaises(MotionCancelled):
            self.prepare()
        self.assertFalse(any(e == ('enable', 1) for e in self.device.events))
        self.assertEqual(self.worker.desired, .3)  # old helper cannot erase a replacement
        self.assertEqual(self.device.enabled, 0)

    def test_transport_guard_rechecks_after_drain_for_neutral_and_enable(self):
        for point in [('neutral', 12000), ('enable', 1), ('enable', 0)]:
            with self.subTest(point=point):
                self.setUp()
                def cancel(event):
                    if event == point:
                        self.device.before = None
                        self.worker.hold()
                self.device.before = cancel
                with self.assertRaises(MotionCancelled):
                    self.prepare()
                if point != ('enable', 0):
                    self.assertNotIn(point, self.device.events)
                self.assertNotIn(('enable', 1), self.device.events)
                self.assertEqual(self.device.moves, [])

    def test_uncertain_enable_ack_is_not_retried_and_cleanup_confirms_disable(self):
        self.device.fail_after = ('enable', 1)
        with self.assertRaises(TimeoutError):
            self.prepare()
        self.assertEqual(self.device.events.count(('enable', 1)), 1)
        self.assertEqual(self.device.enabled, 0)
        self.assertEqual(self.device.moves, [])
        self.assertIsNone(self.worker.desired)

    def test_disable_ack_without_zero_readback_never_neutralizes_or_enables(self):
        self.device.ignore_disable = True
        with self.assertRaisesRegex(RuntimeError, '停止未确认'):
            self.prepare()
        self.assertFalse(any(e[0] in ('neutral', 'target') or e == ('enable', 1)
                             for e in self.device.events))

    def test_neutral_readback_mismatch_stays_disabled(self):
        self.device.bad_neutral_readback = True
        with self.assertRaisesRegex(RuntimeError, '读回不一致'):
            self.prepare()
        self.assertEqual(self.device.enabled, 0)
        self.assertNotIn(('enable', 1), self.device.events)

    def test_alarm_after_disable_never_enables(self):
        def alarm(event):
            if event == ('read_enable', 0):
                self.device.alarm = 32
        self.device.after = alarm
        with self.assertRaisesRegex(RuntimeError, '报警'):
            self.prepare()
        self.assertEqual(self.device.enabled, 0)
        self.assertNotIn(('enable', 1), self.device.events)

    def test_feedback_history_is_causal_and_write_acks_add_no_measurement(self):
        before = len(self.worker.history)
        self.prepare()
        self.assertEqual(len(self.worker.history)-before, 3)
        for (completed, _), (same, started) in zip(self.worker.history, self.worker.history_started):
            self.assertEqual(completed, same)
            self.assertAlmostEqual(completed-started, .02)

    def test_feature_is_opt_in_and_normal_policy_dispatch_has_no_release(self):
        self.worker.config['gripper_manual_release'] = False
        with self.assertRaisesRegex(RuntimeError, '未启用'):
            self.worker.manual_open()
        self.worker.config['gripper_manual_release'] = True
        self.worker.command(1., source='policy')
        requested, bounded, sequence, source = self.worker._next_target(0.)
        self.worker.transport = None
        self.worker._send_target(self.device, self.factory, requested, bounded, 0., sequence, source)
        self.assertEqual([e[0] for e in self.device.events], ['target'])

    def test_queue_timeout_invalidates_own_ticket_without_replaying_old_target(self):
        self.worker.thread = SimpleNamespace(is_alive=lambda: True)
        self.worker.desired, self.worker.ramp_target = .1, .1
        with self.assertRaisesRegex(RuntimeError, '超时并已取消'):
            self.worker.manual_open(timeout_s=0)
        _, future, ticket = self.worker.requests.get_nowait()
        self.assertTrue(future.cancelled())
        self.assertFalse(self.worker._control_valid(ticket))
        self.assertIsNone(self.worker.desired)
        self.assertIsNone(self.worker.ramp_target)
        self.assertEqual(self.device.events, [])

    def test_queued_disable_during_neutralization_prevents_reenable(self):
        def enqueue(event):
            if event == ('neutral', 12000):
                self.device.after = None
                self.worker.requests.put((False, Future()))
        self.device.after = enqueue
        with self.assertRaises(MotionCancelled):
            self.prepare()
        self.assertEqual(self.device.enabled, 0)
        self.assertNotIn(('enable', 1), self.device.events)
        self.assertEqual(self.device.moves, [])

    def test_shutdown_during_enable_ack_uses_unguarded_disable_cleanup(self):
        def stop(event):
            if event == ('enable', 1):
                self.device.after = None
                self.worker.stop.set()
        self.device.after = stop
        with self.assertRaises(MotionCancelled):
            self.prepare()
        self.assertEqual(self.device.enabled, 0)
        self.assertEqual(self.device.events.count(('enable', 1)), 1)
        self.assertEqual(self.device.moves, [])

    def test_slow_neutral_ack_cannot_enable_with_expired_position(self):
        def slow(event):
            if event == ('neutral', 12000):
                self.timer.value += .51
        self.device.after = slow
        with self.assertRaisesRegex(RuntimeError, '500ms'):
            self.prepare()
        self.assertEqual(self.device.enabled, 0)
        self.assertNotIn(('enable', 1), self.device.events)

    def test_timeout_after_preparation_cancels_only_its_queued_target(self):
        self.worker.thread = SimpleNamespace(is_alive=lambda: True)
        worker = self.worker
        class CompletingTimeoutFuture(Future):
            def result(inner, timeout=None):
                _, _, ticket = worker.requests.get_nowait()
                self.assertTrue(inner.set_running_or_notify_cancel())
                worker._manual_open(self.device, self.factory, ticket)
                raise FutureTimeout()
        with patch.object(gripper, 'Future', CompletingTimeoutFuture):
            with self.assertRaisesRegex(RuntimeError, '超时并已取消'):
                self.worker.manual_open()
        self.assertIsNone(self.worker.desired)
        self.assertEqual(self.device.moves, [])

    def test_timeout_of_old_operation_preserves_newer_accepted_target(self):
        self.worker.thread = SimpleNamespace(is_alive=lambda: True)
        worker = self.worker
        class CompletingTimeoutFuture(Future):
            def result(inner, timeout=None):
                _, _, ticket = worker.requests.get_nowait()
                self.assertTrue(inner.set_running_or_notify_cancel())
                worker._manual_open(self.device, self.factory, ticket)
                worker.command(.5, source='new_manual')
                raise FutureTimeout()
        with patch.object(gripper, 'Future', CompletingTimeoutFuture):
            with self.assertRaisesRegex(RuntimeError, '超时并已取消'):
                self.worker.manual_open()
        self.assertEqual(self.worker.desired, .5)
        self.assertEqual(self.device.moves, [])

    def test_actual_worker_dispatches_prepared_manual_intent_from_measurement(self):
        self.worker.transport = None
        future = Future()
        ticket = self.ticket()
        self.worker.requests.put(('manual_open', future, ticket))
        def stop_after_first_target(event):
            if event[0] == 'target':
                self.worker.stop.set()
        self.device.after = stop_after_first_target
        module = SimpleNamespace(rm_peripheral_read_write_params_t=self.factory)
        with patch.dict('sys.modules', {'Robotic_Arm.rm_robot_interface': module}), \
                patch('realman_sync_client.sdk.create_sdk', return_value=self.device), \
                patch.object(self.worker, '_setup', return_value=(0, 12000, 0.)):
            self.worker._run()
        result = future.result()
        self.assertTrue(result['manual_open_prepared'])
        self.assertTrue(result['queued'])
        self.assertFalse(result['acknowledged'])
        self.assertEqual(self.device.moves, [10200])
        self.assertAlmostEqual(self.worker.sent_target['previous_opening'], 0.)


if __name__ == '__main__':
    unittest.main()
