"""Replay timeout recovery scheduling with the real worker loop, no hardware."""
import unittest

from realman_sync_client.gripper_transport import ModbusTransportError
import test_gripper_recovery_loop as loop_fixture


class PromptRecoveryTests(unittest.TestCase):
    def replay_read_timeout(self, *, needs_sync=True, stop_before_recovery=False):
        fixture = loop_fixture.RecoveryLoopTests()
        worker, timer, sdk, io = fixture.make()
        timing = {}
        io.status = lambda: dict(backend='json', failed=False,
                                 needs_sync=needs_sync and 'failed_at' in timing)
        original_enable = io.rm_read_holding_registers
        def enable(args):
            if 'failed_at' in timing:
                timing.setdefault('recovery_at', timer.now())
            return original_enable(args)
        io.rm_read_holding_registers = enable
        def position():
            if io.positions == 2:
                timer.advance(.24)
                timing['failed_at'] = timer.now()
                raise ModbusTransportError('late position reply')
        io.on_position = position
        def intent():
            if io.positions == 1:
                worker.command(0.)
            elif io.positions == 2:
                self.assertIsNone(worker.desired)
                if stop_before_recovery:
                    worker.stop.set()
            elif io.positions >= 3:
                worker.stop.set()
        worker.wake.on_wait = intent
        fixture.run_loop(worker, timer, sdk, io)
        self.assertEqual(io.writes, [])  # recovery cannot replay the old target
        return worker, io, timing

    def test_known_pending_json_read_recovery_has_no_added_poll_sleep(self):
        worker, io, timing = self.replay_read_timeout()
        self.assertEqual(timing['recovery_at'], timing['failed_at'])
        self.assertEqual(io.positions, 3)
        self.assertEqual(worker.fault_generation, 1)

    def test_unclassified_transport_failure_retains_retry_pacing(self):
        _, _, timing = self.replay_read_timeout(needs_sync=False)
        self.assertAlmostEqual(timing['recovery_at']-timing['failed_at'], .1)

    def test_stop_before_prompt_recovery_prevents_another_request(self):
        _, io, timing = self.replay_read_timeout(stop_before_recovery=True)
        self.assertNotIn('recovery_at', timing)
        self.assertEqual(io.positions, 2)

    def test_startup_retires_pending_read_without_extra_sleep(self):
        fixture = loop_fixture.RecoveryLoopTests()
        worker, timer, sdk, io = fixture.make()
        timing = {}
        io.status = lambda: dict(backend='json', failed=False, needs_sync='failed_at' in timing)
        original_enable = io.rm_read_holding_registers
        def enable(args):
            if 'failed_at' in timing:
                timing.setdefault('recovery_at', timer.now())
            return original_enable(args)
        io.rm_read_holding_registers = enable
        def fail_once():
            if io.enabled_reads == 1:
                timer.advance(.24)
                timing['failed_at'] = timer.now()
                raise ModbusTransportError('startup read timeout')
        io.on_enable_read = fail_once
        worker.wake.on_wait = worker.stop.set
        fixture.run_loop(worker, timer, sdk, io)
        self.assertEqual(timing['recovery_at'], timing['failed_at'])
        self.assertEqual(io.writes, [])

    def test_uncertain_write_ack_is_never_replayed_during_prompt_recovery(self):
        fixture = loop_fixture.RecoveryLoopTests()
        worker, timer, sdk, io = fixture.make()
        timing = {}
        io.status = lambda: dict(backend='json', failed=False, needs_sync='failed_at' in timing)
        original_enable = io.rm_read_holding_registers
        def enable(args):
            if 'failed_at' in timing:
                timing.setdefault('recovery_at', timer.now())
            return original_enable(args)
        io.rm_read_holding_registers = enable
        def write(args, data):
            io.writes.append((args, data))
            timer.advance(.3)
            timing['failed_at'] = timer.now()
            raise ModbusTransportError('uncertain target ACK')
        io.rm_write_registers = write
        def intent():
            if io.positions == 1:
                worker.command(0.)
            elif 'failed_at' in timing:
                self.assertIsNone(worker.desired)
                if 'recovery_at' in timing:
                    worker.stop.set()
        worker.wake.on_wait = intent
        fixture.run_loop(worker, timer, sdk, io)
        self.assertEqual(len(io.writes), 1)
        self.assertEqual(timing['recovery_at'], timing['failed_at'])


if __name__ == '__main__':
    unittest.main()
