"""Deterministic worker scheduling tests: no SDK initialization or hardware."""
from concurrent.futures import Future
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from realman_sync_client import gripper


class VirtualClock:
    def __init__(self):
        self.value = 10.

    def now(self):
        return self.value

    def advance(self, seconds):
        self.value += max(0., seconds)
        if self.value > 30:
            raise AssertionError('virtual worker failed to terminate')


class Wake:
    def __init__(self, clock, callback=None):
        self.clock, self.callback, self.flag = clock, callback, False

    def set(self):
        self.flag = True

    def clear(self):
        self.flag = False

    def wait(self, timeout):
        if self.callback:
            callback, self.callback = self.callback, None
            callback()
        if not self.flag:
            self.clock.advance(timeout)
        return self.flag


class FakeSDK:
    def __init__(self, worker, clock):
        self.worker, self.clock = worker, clock
        self.enabled, self.alarm, self.position = 1, 0, 0
        self.calls, self.writes = [], []
        self.block_count, self.enable_count = 0, 0
        self.on_block = self.on_enabled = self.on_write = None
        self.fail_block = None
        self.deleted = False

    def rm_set_timeout(self, value):
        pass

    def rm_create_robot_arm(self, *_):
        return SimpleNamespace(id=1)

    def rm_delete_robot_arm(self):
        self.deleted = True

    def rm_read_holding_registers(self, params):
        assert params == (1, 0x0100, 1, 1)
        self.clock.advance(.09)
        self.enable_count += 1
        self.calls.append(('enabled', self.clock.now()))
        if self.on_enabled:
            self.on_enabled()
        return 0, self.enabled

    def rm_read_multiple_holding_registers(self, params):
        assert params == (1, 0x060D, 1, 6)
        self.clock.advance(.09)
        self.block_count += 1
        self.calls.append(('position_alarm', self.clock.now()))
        if self.on_block:
            self.on_block()
        raw = self.position & 0xffffffff
        # Other status words are deliberately nonzero to check alarm offset 5.
        words = [raw >> 16, raw & 65535, 17, 23, 31, self.alarm]
        return (1 if self.fail_block == self.block_count else 0,
                [byte for value in words for byte in value.to_bytes(2, 'big')])

    def rm_write_registers(self, params, payload):
        assert params == (1, 0x0102, 1, 7)
        self.clock.advance(.09)
        self.calls.append(('target', self.clock.now()))
        self.writes.append((params[1], payload))
        self.position = int.from_bytes(bytes(payload[:4]), 'big', signed=True)
        if self.on_write:
            self.on_write()
        return 0

    def rm_write_single_register(self, params, value):
        assert params == (1, 0x0100, 1, 1)
        self.clock.advance(.09)
        self.calls.append(('enable_write', self.clock.now()))
        self.writes.append((params[1], value))
        self.enabled = value
        if self.on_write:
            self.on_write()
        return 0


class GripperSchedulingTests(unittest.TestCase):
    def make_worker(self):
        worker = gripper.Ctag2F120Worker(dict(robot_ip='not-used', robot_port=8080,
            gripper_poll_s=.1, gripper_command_interval_s=.1, max_gripper_step=.15))
        timer = VirtualClock()
        worker.wake = Wake(timer, lambda: worker.command(0.))
        sdk = FakeSDK(worker, timer)
        return worker, timer, sdk

    def run_worker(self, worker, timer, sdk):
        # Importing the struct is replaced; create_sdk itself is never run.
        module = SimpleNamespace(rm_peripheral_read_write_params_t=lambda *args: args)
        with patch.dict('sys.modules', {'Robotic_Arm.rm_robot_interface': module}), \
                patch('realman_sync_client.sdk.create_sdk', return_value=sdk), \
                patch.object(worker, '_setup', return_value=(0, 12000, 1.)), \
                patch.object(gripper.clock, 'now', timer.now):
            worker._run()
        self.assertTrue(sdk.deleted)

    def test_combined_feedback_has_correct_position_and_alarm_offsets(self):
        worker, timer, sdk = self.make_worker()
        sdk.position, sdk.alarm = 3000, 32
        with patch.object(gripper.clock, 'now', timer.now):
            opening, ready, started = worker._poll_feedback(sdk, lambda *args: args, 0, 12000)
        self.assertEqual(opening, .75)
        self.assertFalse(ready)
        self.assertEqual(worker.state['alarm'], 32)
        self.assertEqual(worker.state['position'], 3000)
        self.assertAlmostEqual(worker.history[-1][0], started+.18)
        self.assertAlmostEqual(worker.state['measurement_started'], started+.09)
        self.assertEqual([call[0] for call in sdk.calls], ['enabled', 'position_alarm'])

    def test_repeated_targets_cannot_starve_feedback_or_add_post_read_sleep(self):
        worker, timer, sdk = self.make_worker()
        def newest():
            if sdk.block_count > 1:
                worker.command(0.)
        sdk.on_block = newest
        sdk.on_write = lambda: worker.stop.set() if len(sdk.writes) == 5 else None
        self.run_worker(worker, timer, sdk)
        self.assertEqual(len(sdk.writes), 5)
        self.assertEqual(len(worker.history), 6)
        names = [call[0] for call in sdk.calls]
        self.assertEqual(names[:2], ['enabled', 'position_alarm'])
        self.assertEqual(names.count('position_alarm'), 6)
        self.assertLess(names.count('enabled'), names.count('position_alarm'))
        self.assertTrue(all(names[index-1] == 'position_alarm'
                            for index, name in enumerate(names) if name == 'target'))
        intervals = [b[0]-a[0] for a, b in zip(worker.history, tuple(worker.history)[1:])]
        self.assertLessEqual(max(intervals), .270001)
        self.assertTrue(all(0 < interval for interval in intervals))

    def test_hold_during_feedback_read_cancels_target_without_write(self):
        worker, timer, sdk = self.make_worker()
        def hold_or_finish():
            if sdk.block_count == 2:
                worker.hold()
            elif sdk.block_count == 3:
                worker.stop.set()
        sdk.on_block = hold_or_finish
        self.run_worker(worker, timer, sdk)
        self.assertEqual(sdk.writes, [])
        self.assertIsNone(worker.desired)
        self.assertIsNone(worker.ramp_target)
        self.assertEqual(len(worker.history), 2)

    def test_stop_during_enable_read_prevents_position_read_and_target(self):
        worker, timer, sdk = self.make_worker()
        sdk.on_enabled = lambda: worker.stop.set() if sdk.enable_count == 2 else None
        self.run_worker(worker, timer, sdk)
        self.assertEqual(sdk.block_count, 2)
        # One earlier target preceded the second (slow-rate) enable read.
        # Stopping inside that read must issue neither another read nor target.
        self.assertEqual(len(sdk.writes), 1)
        self.assertEqual(sdk.calls[-1][0], 'enabled')

    def test_queued_disable_outranks_target_and_ack_does_not_refresh_position(self):
        worker, timer, sdk = self.make_worker()
        future = Future()
        def request_disable():
            if sdk.block_count == 2:
                worker.requests.put((False, future))
                worker.wake.set()
        sdk.on_block = request_disable
        sdk.on_write = worker.stop.set
        self.run_worker(worker, timer, sdk)
        self.assertEqual(sdk.writes, [(0x0100, 0)])
        state = future.result()
        self.assertFalse(state['enabled'])
        self.assertEqual(state['received'], worker.history[-1][0])
        self.assertGreaterEqual(state['age_ms'], 179.9)
        self.assertEqual(len(worker.history), 2)

    def test_failed_feedback_read_cannot_publish_or_write_target(self):
        worker, timer, sdk = self.make_worker()
        sdk.fail_block = 2
        self.run_worker(worker, timer, sdk)
        self.assertEqual(sdk.writes, [])
        self.assertEqual(len(worker.history), 1)
        self.assertFalse(worker.status()['ready'])
        self.assertIn('0x060d', worker.state['error'])

    def test_selected_target_is_rechecked_against_hold_stop_disable_and_replacement(self):
        for cancellation in ('hold', 'stop', 'disable', 'replacement'):
            with self.subTest(cancellation=cancellation):
                worker, timer, sdk = self.make_worker()
                worker.mapping_min, worker.mapping_max = 0, 12000
                worker.desired, worker.command_sequence = 0., 1
                args = (sdk, lambda *args: args, 0., .85, 1., 1, 'policy')
                if cancellation == 'hold':
                    worker.hold()
                elif cancellation == 'stop':
                    worker.close()
                elif cancellation == 'disable':
                    with worker.lock:
                        worker.requests.put((False, Future()))
                else:
                    with worker.lock:
                        worker.desired, worker.command_sequence = .4, 2
                with patch.object(gripper.clock, 'now', timer.now):
                    self.assertIsNone(worker._send_target(*args))
                self.assertEqual(sdk.writes, [])
                self.assertIsNone(worker.sent_target)
                self.assertIsNone(worker._target_inflight)

    def test_hold_during_inflight_sdk_write_never_revives_ramp(self):
        worker, timer, sdk = self.make_worker()
        worker.mapping_min, worker.mapping_max = 0, 12000
        worker.desired, worker.command_sequence = 0., 1
        def cancel_inflight():
            self.assertEqual(worker._target_inflight['sequence'], 1)
            # This would deadlock if the SDK write held worker.lock.
            worker.hold()
        sdk.on_write = cancel_inflight
        with patch.object(gripper.clock, 'now', timer.now):
            completed = worker._send_target(sdk, lambda *args: args, 0., .85, 1., 1, 'policy')
        self.assertIsNotNone(completed)
        self.assertEqual(len(sdk.writes), 1)
        self.assertIsNone(worker.desired)
        self.assertIsNone(worker.ramp_target)
        self.assertIsNone(worker._target_inflight)

    def test_cancelled_dispatch_does_not_advance_commanded_position(self):
        worker, timer, sdk = self.make_worker()
        original = worker._send_target
        previous_positions = []
        def cancel_first(*args):
            previous_positions.append(args[4])
            if len(previous_positions) == 1:
                worker.command(.4)
                return None
            return original(*args)
        sdk.on_write = worker.stop.set
        with patch.object(worker, '_send_target', side_effect=cancel_first):
            self.run_worker(worker, timer, sdk)
        self.assertEqual(previous_positions, [1., 1.])
        self.assertEqual(len(sdk.writes), 1)


if __name__ == '__main__':
    unittest.main()
