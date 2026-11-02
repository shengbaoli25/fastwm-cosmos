"""Safety regressions for multirate gripper feedback; no real SDK or device."""
import unittest
from unittest.mock import patch

from realman_sync_client import gripper


class Clock:
    def __init__(self):
        self.value = 10.0

    def now(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class RegisterDevice:
    """One complete RTU transaction per call, including optional latency."""

    def __init__(self, timer):
        self.timer = timer
        self.enabled, self.position, self.alarm = 1, 3000, 0
        self.enable_delay = self.position_delay = self.write_delay = .01
        self.calls = []
        self.fail_position = False
        self.after_write = None

    def rm_read_holding_registers(self, params):
        assert params == (1, 0x0100, 1, 1)
        self.calls.append(('enable', self.timer.now()))
        self.timer.advance(self.enable_delay)
        return 0, self.enabled

    def rm_read_multiple_holding_registers(self, params):
        assert params == (1, 0x060D, 1, 6)
        self.calls.append(('position_alarm', self.timer.now()))
        self.timer.advance(self.position_delay)
        if self.fail_position:
            raise TimeoutError('injected bounded RTU read timeout')
        position = self.position & 0xffffffff
        words = [position >> 16, position & 65535, 17, 23, 31, self.alarm]
        return 0, [byte for word in words for byte in word.to_bytes(2, 'big')]

    def rm_write_single_register(self, params, value):
        assert params == (1, 0x0100, 1, 1)
        self.calls.append(('enable_write', self.timer.now()))
        self.timer.advance(self.write_delay)
        self.enabled = value
        if self.after_write:
            self.after_write()
        return 0


class MultirateGripperTests(unittest.TestCase):
    def setUp(self):
        self.timer = Clock()
        self.device = RegisterDevice(self.timer)
        self.worker = gripper.Ctag2F120Worker(dict(robot_ip='unused', robot_port=8080,
            gripper_poll_s=.05, gripper_enable_poll_s=.25))
        self.worker.mapping_min, self.worker.mapping_max = 0, 12000
        self.factory = lambda *args: args
        self.clock_patch = patch.object(gripper.clock, 'now', self.timer.now)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)

    def poll(self):
        return self.worker._poll_feedback(self.device, self.factory, 0, 12000)

    def test_position_and_enable_keep_separate_conservative_timestamps(self):
        self.poll()
        self.assertAlmostEqual(self.worker.state['enabled_received'], 10.)
        self.assertAlmostEqual(self.worker.state['measurement_started'], 10.01)
        self.assertAlmostEqual(self.worker.state['received'], 10.02)
        self.assertAlmostEqual(self.worker.state['response_received'], 10.02)
        self.assertAlmostEqual(self.worker.history[-1][0], 10.02)
        self.assertAlmostEqual(self.worker.history_started[-1][0], 10.02)
        self.assertAlmostEqual(self.worker.history_started[-1][1], 10.01)
        self.assertAlmostEqual(self.worker.status()['age_ms'], 10.)
        self.assertTrue(self.worker.status()['ready'])

    def test_cached_enable_skips_transactions_without_skipping_position_or_alarm(self):
        for raw in (3000, 6000, 9000):
            self.device.position = raw
            self.poll()
            self.timer.advance(.03)
        names = [name for name, _ in self.device.calls]
        self.assertEqual(names, ['enable'] + ['position_alarm']*3)
        self.assertEqual(len(self.worker.history), 3)
        self.assertAlmostEqual(self.worker.state['opening'], .25)
        self.assertAlmostEqual(self.worker.state['enabled_received'], 10.)

    def test_enable_is_refreshed_after_its_own_quarter_second_deadline(self):
        self.poll()
        self.timer.advance(.231)
        self.poll()
        reads = [stamp for name, stamp in self.device.calls if name == 'enable']
        self.assertEqual(len(reads), 2)
        self.assertAlmostEqual(self.worker.state['enabled_received'], reads[-1])

    def test_fresh_position_never_renews_cached_enable_for_motion(self):
        self.poll()
        self.timer.advance(.22)
        self.poll()
        enable_stamp = self.worker.state['enabled_received']
        position_stamp = self.worker.state['received']
        self.assertGreater(position_stamp, enable_stamp)
        self.timer.advance(.261)
        status = self.worker.status(max_age_s=1.)
        self.assertLess(status['age_ms'], 500.)
        self.assertGreater(status['enabled_age_ms'], 500.)
        self.assertFalse(status['ready'])
        with self.assertRaises(RuntimeError):
            self.worker.command(0.)

    def test_slow_position_reply_is_not_born_fresh_at_receive_completion(self):
        self.device.position_delay = .55
        _, ready, _ = self.poll()
        status = self.worker.status()
        self.assertFalse(ready)
        self.assertFalse(status['ready'])
        self.assertGreaterEqual(status['age_ms'], 550.)
        self.assertAlmostEqual(self.worker.state['response_received'] -
                               self.worker.state['measurement_started'], .55)
        self.assertAlmostEqual(self.worker.history[-1][0], self.timer.now())
        with self.assertRaises(RuntimeError):
            self.worker.command(0.)

    def test_slow_enable_reply_blocks_motion_even_with_fast_position_read(self):
        self.device.enable_delay = .51
        _, ready, _ = self.poll()
        status = self.worker.status()
        self.assertFalse(ready)
        self.assertLess(status['age_ms'], 20.)
        self.assertGreater(status['enabled_age_ms'], 500.)
        self.assertFalse(status['ready'])

    def test_alarm_is_checked_each_position_poll_despite_cached_enable(self):
        self.poll()
        self.worker.command(0.)
        self.worker.ramp_target = 0.
        self.device.alarm = 32
        self.poll()
        self.assertEqual([name for name, _ in self.device.calls].count('enable'), 1)
        self.assertEqual(self.worker.state['alarm'], 32)
        self.assertFalse(self.worker.status()['ready'])
        self.assertIsNone(self.worker.desired)
        self.assertIsNone(self.worker.ramp_target)

    def test_failed_read_preserves_measurements_and_recovery_cannot_replay_target(self):
        self.poll()
        self.worker.command(0.)
        self.worker.ramp_target = 0.
        history_before = tuple(self.worker.history)
        started_before = tuple(self.worker.history_started)
        position_stamp = self.worker.state['received']
        generation_before = self.worker.status()['fault_generation']
        self.device.fail_position = True
        try:
            self.poll()
        except TimeoutError as exc:
            self.worker._feedback_failed(exc)
        else:
            self.fail('fault injection must reject the poll')
        self.assertEqual(tuple(self.worker.history), history_before)
        self.assertEqual(tuple(self.worker.history_started), started_before)
        self.assertEqual(self.worker.state['received'], position_stamp)
        self.assertTrue(self.worker.state['communication_fault'])
        fault_generation = self.worker.status()['fault_generation']
        self.assertGreater(fault_generation, generation_before)
        self.assertFalse(self.worker.status()['ready'])
        self.assertIsNone(self.worker.desired)
        self.assertIsNone(self.worker.ramp_target)
        self.device.fail_position = False
        self.poll()
        self.assertFalse(self.worker.state['communication_fault'])
        self.assertEqual(self.worker.status()['fault_generation'], fault_generation)
        self.assertTrue(self.worker.status()['ready'])
        self.assertIsNone(self.worker.desired)
        self.assertIsNone(self.worker._next_target(.75))
        self.assertEqual(len(self.worker.history), len(history_before)+1)

    def test_measured_disable_cancels_pending_target_on_next_enable_read(self):
        self.poll()
        self.worker.command(0.)
        self.worker.ramp_target = 0.
        self.timer.advance(.26)
        self.device.enabled = 0
        self.poll()
        self.assertFalse(self.worker.state['enabled'])
        self.assertFalse(self.worker.status()['ready'])
        self.assertIsNone(self.worker.desired)
        self.assertIsNone(self.worker.ramp_target)

    def test_cancel_after_transport_wait_does_not_replay_expired_motion(self):
        from realman_sync_client.gripper_transport import MotionCancelled
        self.poll()
        self.worker.command(0.)
        self.worker.ramp_target = 0.
        sequence = self.worker.command_sequence
        owner = self
        class DelayedTransport:
            motion_guard = None

            def status(self):
                return dict(backend='test_transport')

            def rm_write_registers(self, params, payload):
                # Delayed reply isolation may consume the entire fresh-state
                # budget before the transport is allowed to send target bytes.
                owner.timer.advance(.51)
                owner.assertFalse(self.motion_guard())
                raise MotionCancelled('feedback expired before dispatch')
        transport = DelayedTransport()
        self.worker.transport = transport
        result = self.worker._send_target(transport, self.factory, 0., .60, .75,
                                          sequence, 'policy')
        self.assertIsNone(result)
        self.assertIsNone(self.worker.sent_target)
        self.assertIsNone(self.worker._target_inflight)
        self.assertIsNone(transport.motion_guard)
        self.assertIsNone(self.worker.desired)
        self.assertIsNone(self.worker.ramp_target)
        self.poll()
        self.assertTrue(self.worker.status()['ready'])
        self.assertIsNone(self.worker._next_target(.75))

    def test_enable_write_ack_does_not_fake_measurement_but_readback_refreshes(self):
        self.poll()
        old_enable = self.worker.state['enabled_received']
        old_position = self.worker.state['received']
        history_before = tuple(self.worker.history)
        self.timer.advance(.05)
        ack_observed = []
        def at_ack():
            ack_observed.append(True)
            self.assertEqual(self.worker.state['enabled_received'], old_enable)
            self.assertEqual(self.worker.state['received'], old_position)
        self.device.after_write = at_ack
        self.worker._set_enabled(self.device, self.factory, False)
        readback = [stamp for name, stamp in self.device.calls if name == 'enable'][-1]
        self.assertEqual(ack_observed, [True])
        self.assertAlmostEqual(self.worker.state['enabled_received'], readback)
        self.assertEqual(self.worker.state['received'], old_position)
        self.assertEqual(tuple(self.worker.history), history_before)
        self.assertFalse(self.worker.status()['ready'])

    def test_transport_cancellation_of_old_generation_preserves_new_operator_target(self):
        from realman_sync_client.gripper_transport import MotionCancelled
        self.poll()
        self.worker.command(0.)
        sequence = self.worker.command_sequence
        owner = self
        class ReplacedTransport:
            motion_guard = None

            def status(self):
                return dict(backend='test_transport')

            def rm_write_registers(self, params, payload):
                owner.worker.command(.5, source='manual')
                owner.assertFalse(self.motion_guard())
                raise MotionCancelled('operator replaced the selected target')
        transport = ReplacedTransport()
        self.worker.transport = transport
        result = self.worker._send_target(transport, self.factory, 0., .60, .75,
                                          sequence, 'policy')
        self.assertIsNone(result)
        self.assertIsNone(self.worker.sent_target)
        self.assertEqual(self.worker.desired, .5)
        self.assertGreater(self.worker.command_sequence, sequence)
        self.assertEqual(self.worker._next_target(.75)[0], .5)
        self.assertEqual(self.worker.command_source, 'manual')

    def snapshot_worker_types(self):
        from realman_sync_client.web_hardware import _CheckedGripper
        return gripper.Ctag2F120Worker, _CheckedGripper

    def reset_snapshot_worker(self, worker_type):
        self.timer.value = 10.
        self.device = RegisterDevice(self.timer)
        self.worker = worker_type(dict(robot_ip='unused', robot_port=8080,
            gripper_poll_s=.05, gripper_enable_poll_s=.25))

    def test_first_reply_cannot_be_selected_for_anchor_before_it_arrived(self):
        for worker_type in self.snapshot_worker_types():
            with self.subTest(worker=worker_type.__name__):
                self.reset_snapshot_worker(worker_type)
                self.device.position_delay = .1
                self.poll()
                request_start = self.worker.state['measurement_started']
                arrived = self.worker.state['received']
                anchor_during_request = (request_start+arrived)/2
                with self.assertRaises(RuntimeError):
                    self.worker.snapshot(anchor_during_request)
                self.assertEqual(self.worker.snapshot(arrived), .75)

    def test_anchor_inside_new_request_uses_previous_completed_sample(self):
        for worker_type in self.snapshot_worker_types():
            with self.subTest(worker=worker_type.__name__):
                self.reset_snapshot_worker(worker_type)
                self.poll()
                self.timer.advance(.04)
                self.device.position = 9000
                self.device.position_delay = .1
                self.poll()
                started = self.worker.state['measurement_started']
                arrived = self.worker.state['received']
                self.assertEqual(self.worker.snapshot((started+arrived)/2), .75)
                self.assertEqual(self.worker.snapshot(arrived), .25)
                self.assertEqual([name for name, _ in self.device.calls],
                                 ['enable', 'position_alarm', 'position_alarm'])

    def test_selected_historical_sample_age_uses_its_own_request_start(self):
        for worker_type in self.snapshot_worker_types():
            with self.subTest(worker=worker_type.__name__):
                self.reset_snapshot_worker(worker_type)
                self.device.position_delay = .4
                self.poll()
                old_arrival = self.worker.state['received']
                self.timer.advance(.11)
                self.device.position_delay = .01
                self.device.position = 9000
                self.poll()
                self.assertTrue(self.worker.status()['ready'])
                self.assertLess(self.timer.now()-old_arrival, .5)
                # The historical observation arrived recently, but its own
                # conservative acquisition bound is already older than 500 ms.
                with self.assertRaises(RuntimeError):
                    self.worker.snapshot(old_arrival+.01)
                self.assertEqual(self.worker.snapshot(self.timer.now()), .25)


if __name__ == '__main__':
    unittest.main()
