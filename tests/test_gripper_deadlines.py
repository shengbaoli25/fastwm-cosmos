"""Deadline scheduler replay with the actual worker loop, never hardware."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from realman_sync_client import gripper
from realman_sync_client.gripper_transport import ModbusTransportError
from test_gripper_recovery_loop import Timer, Event, SetupSDK


class TimedBus:
    def __init__(self, worker, timer, *, slow=False, fail=False):
        self.worker, self.timer = worker, timer
        self.slow, self.fail = slow, fail
        self.positions = self.enables = self.writes = 0
        self.motion_guard = self.control_guard = None
        self.position = 0
        self.max_position_age = self.max_enable_age = 0.
        self.write_times = []
        self.failure_seen = False

    def status(self):
        return dict(backend='json', failed=self.failure_seen)

    def close(self):
        pass

    def elapsed(self, duration):
        self.timer.advance(duration)
        if self.worker.state.get('initialized'):
            state = self.worker.status()
            self.max_position_age = max(self.max_position_age, state['age_ms'])
            if state['enabled_age_ms'] is not None:
                self.max_enable_age = max(self.max_enable_age, state['enabled_age_ms'])

    def rm_read_holding_registers(self, params):
        assert params == (1, 0x0100, 1, 1)
        self.enables += 1
        # Isolated latency spikes, never two 200 ms reads in the same poll.
        self.elapsed(.2 if self.slow and self.enables % 9 == 0
                     and (self.positions+1) % 6 != 0 else .06)
        return 0, 1

    def rm_read_multiple_holding_registers(self, params):
        assert params == (1, 0x060D, 1, 6)
        self.positions += 1
        if self.fail and self.positions == 3:
            self.elapsed(.301)
            self.failure_seen = True
            raise ModbusTransportError('read exceeded bounded transport timeout')
        self.elapsed(.2 if self.slow and self.positions % 6 == 0 else .06)
        words = [self.position >> 16, self.position & 65535, 0, 0, 0, 0]
        return 0, [byte for word in words for byte in word.to_bytes(2, 'big')]

    def rm_write_registers(self, params, data):
        assert params == (1, 0x0102, 1, 7)
        if self.motion_guard is not None:
            assert self.motion_guard(), 'scheduler dispatched an unready target'
        assert not self.failure_seen
        self.writes += 1
        self.write_times.append(self.timer.now())
        self.elapsed(.07)
        self.position = int.from_bytes(bytes(data[:4]), 'big', signed=True)
        return 0


def replay(*, old_schedule=False, slow=False, fail=False, duration=6.):
    timer = Timer()
    worker = gripper.Ctag2F120Worker(dict(robot_ip='unused', robot_port=8080,
        gripper_transport='json', gripper_timeout_s=.3, gripper_poll_s=.05,
        gripper_enable_poll_s=.25, gripper_command_interval_s=.1))
    worker.stop, worker.wake = Event(timer), Event(timer)
    io = TimedBus(worker, timer, slow=slow, fail=fail)
    sdk = SetupSDK()
    last = [None]
    def control():
        if timer.now() >= 10.+duration:
            worker.stop.set()
            return
        marker = (io.positions, io.writes)
        if marker != last[0]:
            last[0] = marker
            if worker.status().get('ready'):
                worker.command(float(io.writes % 2))
    worker.wake.on_wait = control
    module = SimpleNamespace(rm_peripheral_read_write_params_t=lambda *args: args)
    original_poll = worker._poll_feedback
    def old_poll(sdk, factory, lower, upper, motion_due=False):
        return original_poll(sdk, factory, lower, upper, False)
    with patch.dict('sys.modules', {'Robotic_Arm.rm_robot_interface': module}), \
            patch('realman_sync_client.sdk.create_sdk', return_value=sdk), \
            patch('realman_sync_client.gripper_transport.JsonModbusArm', return_value=io), \
            patch.object(worker, '_setup', return_value=(0, 12000, 1.)), \
            patch.object(gripper.clock, 'now', timer.now):
        if old_schedule:
            with patch.object(worker, '_enable_refresh_interval', return_value=.25), \
                    patch.object(worker, '_poll_feedback', side_effect=old_poll), \
                    patch.object(worker, '_motion_has_feedback_budget', return_value=True):
                worker._run()
        else:
            worker._run()
    return worker, io, timer


class DeadlineSchedulerTests(unittest.TestCase):
    def test_healthy_continuous_targets_do_not_starve_motion_or_feedback(self):
        worker, io, timer = replay()
        self.assertGreaterEqual(io.writes, 25)
        self.assertLess(io.max_position_age, 500)
        self.assertLess(io.max_enable_age, 500)
        self.assertGreaterEqual(min(b-a for a,b in zip(io.write_times, io.write_times[1:])), .1)
        self.assertEqual(worker.fault_generation, 0)
        self.assertIsNotNone(worker.sent_target)

    def test_occasional_200ms_reads_still_allow_motion_with_fresh_measurements(self):
        worker, io, _ = replay(slow=True)
        self.assertGreaterEqual(io.writes, 15)
        self.assertLess(io.max_position_age, 500)
        self.assertLess(io.max_enable_age, 500)
        self.assertEqual(worker.fault_generation, 0)

    def test_old_quarter_second_schedule_reproduces_age_boundary(self):
        _, old, _ = replay(old_schedule=True, slow=True)
        _, new, _ = replay(slow=True)
        self.assertGreaterEqual(max(old.max_position_age, old.max_enable_age), 500)
        self.assertLess(max(new.max_position_age, new.max_enable_age), 500)

    def test_transport_timeout_cancels_target_without_retrying_write(self):
        worker, io, _ = replay(fail=True)
        self.assertTrue(io.failure_seen)
        self.assertLessEqual(io.writes, 1)
        self.assertIsNone(worker.desired)
        self.assertFalse(worker.state.get('ready'))
        self.assertEqual(worker.fault_generation, 1)

    def test_budget_defers_without_faking_measurement_or_dropping_target(self):
        worker = gripper.Ctag2F120Worker(dict(gripper_transport='json'))
        worker.enabled_received = 9.7
        worker.state = dict(received=9.8, measurement_started=9.7, ready=True)
        worker.desired = .5
        with patch.object(gripper.clock, 'now', return_value=10.):
            self.assertFalse(worker._motion_has_feedback_budget('policy'))
        self.assertEqual(worker.enabled_received, 9.7)
        self.assertEqual(worker.state['measurement_started'], 9.7)
        self.assertEqual(worker.desired, .5)
        self.assertEqual(worker._deadline_deferrals, 1)
        self.assertIsNotNone(worker._deadline_reason)


if __name__ == '__main__':
    unittest.main()
