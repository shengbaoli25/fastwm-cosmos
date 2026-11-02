"""Stop/manual admission races, with no sockets, SDK, or hardware threads."""
from concurrent.futures import Future
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

from realman_sync_client import gripper
from realman_sync_client.gripper_transport import MotionCancelled
from realman_sync_client.web_hardware import _CheckedGripper
import test_hardware_feedback_recovery as recovery


class EnableBus:
    def __init__(self, worker):
        self.worker, self.enabled = worker, 0
        self.writes = []
        self.on_alarm = self.before_dispatch = None

    def rm_get_current_arm_state(self):
        return 0, {}

    def rm_read_multiple_holding_registers(self, args):
        if self.on_alarm:
            self.on_alarm()
        return 0, [0]*6

    def rm_read_holding_registers(self, args):
        return 0, self.enabled

    def rm_write_single_register(self, args, value):
        if self.before_dispatch:
            self.before_dispatch()
        guard = self.worker.transport.control_guard
        if guard is not None and not guard():
            raise MotionCancelled('last socket guard cancelled enable')
        self.writes.append(value)
        self.enabled = value
        return 0


class ManualTicketTests(unittest.TestCase):
    def setUp(self):
        fixture = recovery.RecoveryPermitTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.worker, self.hardware = fixture.worker, fixture.hardware
        self.hardware.connected, self.hardware.armed = True, False
        self.hardware.config = self.worker.config
        self.hardware._diagnostic_lock = threading.Lock()
        self.hardware.gripper_diagnostics = Mock(return_value=dict(
            motion_available=True, enable_available=True, disable_available=True))
        self.hardware._manual_gripper_operation = None
        self.worker.transport = SimpleNamespace(control_guard=None,
            status=lambda: dict(backend='json', failed=False))
        self.bus = EnableBus(self.worker)

    def queued_future(self, *, hold=False):
        future = Future()
        def finish(timeout):
            operation, queued, ticket = self.worker.requests.get_nowait()
            self.assertIs(queued, future)
            # Future waits must not block the hardware lock needed by Stop.
            self.assertFalse(self.hardware.lock._is_owned())
            if hold:
                self.worker.hold()
            self.worker._set_enabled(self.bus, lambda *args: args, operation, ticket=ticket)
            return self.worker.status()
        future.result = finish
        return future

    def test_every_manual_entry_rejects_token_captured_before_stop(self):
        for release, operation in ((False, 'open'), (False, 'close'), (False, 'enable'),
                                   (False, 'disable'), (True, 'open'), (True, 'enable')):
            with self.subTest(release=release, operation=operation):
                self.worker.config['gripper_manual_release'] = release
                token = self.hardware.capture_manual_control_token()
                self.worker.hold()
                with self.assertRaisesRegex(RuntimeError, '取消'):
                    self.hardware.manual_gripper(operation, expected_token=token)
                self.assertTrue(self.worker.requests.empty())
                self.assertIsNone(self.worker.desired)
                self.assertEqual(self.bus.writes, [])

    def test_stop_between_status_admission_and_final_queue_rejects_old_open(self):
        for release in (False, True):
            with self.subTest(release=release):
                self.worker.config['gripper_manual_release'] = release
                token = self.hardware.capture_manual_control_token()
                original = self.worker.status
                called = [False]
                def status(*args, **kwargs):
                    result = original(*args, **kwargs)
                    if not called[0]:
                        called[0] = True
                        self.worker.hold()
                    return result
                with patch.object(self.worker, 'status', side_effect=status):
                    with self.assertRaisesRegex(RuntimeError, '取消|入队前'):
                        self.hardware.manual_gripper('open', expected_token=token)
                self.assertTrue(self.worker.requests.empty())
                self.assertIsNone(self.worker.desired)

    def test_replaced_worker_cannot_receive_old_manual_token(self):
        token = self.hardware.capture_manual_control_token()
        self.hardware.gripper = _CheckedGripper({})
        with self.assertRaisesRegex(RuntimeError, '所属连接已改变'):
            self.hardware.manual_gripper('close', expected_token=token)
        self.assertIsNone(self.hardware.gripper.desired)

    def test_stop_after_enable_queue_cancels_default_and_release_tickets(self):
        for release in (False, True):
            with self.subTest(release=release):
                self.worker.config['gripper_manual_release'] = release
                token = self.hardware.capture_manual_control_token()
                with patch.object(gripper, 'Future', side_effect=lambda: self.queued_future(hold=True)):
                    with self.assertRaisesRegex(RuntimeError, '取消'):
                        self.hardware.manual_gripper('enable', expected_token=token)
                self.assertEqual(self.bus.writes, [])
                self.assertIsNone(self.worker.desired)

    def test_default_enable_rechecks_stop_after_alarm_read(self):
        self.bus.on_alarm = self.worker.hold
        token = self.hardware.capture_manual_control_token()
        with patch.object(gripper, 'Future', side_effect=self.queued_future):
            with self.assertRaisesRegex(RuntimeError, '取消'):
                self.hardware.manual_gripper('enable', expected_token=token)
        self.assertEqual(self.bus.writes, [])
        self.assertIsNone(self.worker.transport.control_guard)

    def test_default_enable_final_transport_guard_checks_ticket_again(self):
        self.bus.before_dispatch = self.worker.hold
        token = self.hardware.capture_manual_control_token()
        with patch.object(gripper, 'Future', side_effect=self.queued_future):
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                self.hardware.manual_gripper('enable', expected_token=token)
        self.assertEqual(self.bus.writes, [])
        self.assertIsNone(self.worker.transport.control_guard)

    def test_valid_default_enable_keeps_own_ticket_and_writes_once(self):
        token = self.hardware.capture_manual_control_token()
        with patch.object(gripper, 'Future', side_effect=self.queued_future):
            result = self.hardware.manual_gripper('enable', expected_token=token)
        self.assertTrue(result['confirmed'])
        self.assertEqual(self.bus.writes, [1])
        self.assertIsNone(self.worker.desired)
        self.assertIsNone(self.worker.transport.control_guard)

    def test_target_admitted_before_stop_is_cancelled_without_new_intent(self):
        token = self.hardware.capture_manual_control_token()
        self.hardware.manual_gripper('close', expected_token=token)
        self.assertEqual(self.worker.desired, 0.)
        self.worker.hold()
        self.assertIsNone(self.worker.desired)
        self.assertIsNone(self.worker.ramp_target)

    def test_legacy_optional_parameter_omission_still_accepts_new_manual_target(self):
        self.hardware.manual_gripper('close')
        self.assertEqual(self.worker.desired, 0.)


if __name__ == '__main__':
    unittest.main()
