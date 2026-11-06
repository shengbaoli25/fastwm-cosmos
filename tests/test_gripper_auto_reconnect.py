"""Bounded read-only worker reopening; fake SDK and deterministic virtual clock."""
from concurrent.futures import Future
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from realman_sync_client import gripper
from realman_sync_client.gripper_transport import ModbusTransportError, MotionCancelled
from realman_sync_client.recovery import transient_failure
from test_gripper_recovery_loop import Event, JsonIO, SetupSDK, Timer


class ReconnectIO(JsonIO):
    reconnectable = True

    def status(self):
        return dict(backend='json', failed=self.failed, closed=self.closed,
            reconnect_allowed=self.failed and not self.closed and self.reconnectable)


class AutoReconnectTests(unittest.TestCase):
    def test_reconnect_state_is_transient_before_and_during_backoff(self):
        worker, timer, sdk, old, _ = self.make()
        old.failed = True
        worker.transport = old
        worker._feedback_failed(ModbusTransportError('sync budget exceeded; explicit reconnect required'))
        self.assertTrue(transient_failure(RuntimeError(worker.state['error'])))
        seen = []
        def stop_after_reading_status():
            seen.append(worker.state['error'])
            worker.stop.set()
        worker.stop.on_wait = stop_after_reading_status
        with patch.object(gripper.clock, 'now', timer.now), self.assertRaises(MotionCancelled):
            worker._recover_readonly(sdk, None)
        self.assertTrue(seen)
        self.assertTrue(all(transient_failure(RuntimeError(error)) for error in seen))

    def make(self, **config):
        timer = Timer()
        worker = gripper.Ctag2F120Worker(dict(robot_ip='unused', robot_port=8080,
            gripper_transport='json', gripper_auto_reconnect=True,
            gripper_poll_s=.1, gripper_enable_poll_s=.25, gripper_command_interval_s=.1,
            **config))
        worker.stop, worker.wake = Event(timer), Event(timer)
        old, new = ReconnectIO(worker, timer), ReconnectIO(worker, timer)
        return worker, timer, SetupSDK(), old, new

    def run_loop(self, worker, timer, sdk, buses, verifier):
        module = SimpleNamespace(rm_peripheral_read_write_params_t=lambda *args: args)
        with patch.dict('sys.modules', {'Robotic_Arm.rm_robot_interface': module}), \
                patch('realman_sync_client.sdk.create_sdk', return_value=sdk), \
                patch.object(worker, '_new_json_transport', side_effect=buses) as factory, \
                patch.object(worker, '_setup', return_value=(0, 12000, 1.)) as initial_setup, \
                patch.object(worker, '_verify_readonly_reconnect', side_effect=verifier) as verify, \
                patch.object(gripper.clock, 'now', timer.now):
            worker._run()
        self.assertTrue(sdk.deleted)
        self.assertEqual(initial_setup.call_count, 1)  # mutating base setup is never reused
        self.assertFalse(worker._reconnecting)
        return factory, verify

    def test_failed_stream_reopens_after_old_owner_closes_without_replaying_intent(self):
        worker, timer, sdk, old, new = self.make()
        queued = Future()
        def fail_position():
            if old.positions == 2:
                worker.requests.put((True, queued))
                old.failed = True
                raise ModbusTransportError('closed stream')
        old.on_position = fail_position
        def intent():
            if new.positions:
                worker.stop.set()
            elif old.positions == 1:
                worker.command(0.)
        worker.wake.on_wait = intent
        verified_at = []
        def verify(*_):
            self.assertTrue(old.closed)
            self.assertEqual(worker.connection_generation, 1)
            self.assertEqual(len(worker.history), 0)
            self.assertIsNone(worker.enabled_received)
            self.assertIsNone(worker.desired)
            self.assertIsInstance(queued.exception(), MotionCancelled)
            self.assertFalse(worker.status()['ready'])
            verified_at.append(timer.now())
            return 0, 12000, 1.
        factory, _ = self.run_loop(worker, timer, sdk, [old, new], verify)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(old.writes + new.writes, [])
        self.assertTrue(worker.state['ready'])
        self.assertEqual(len(worker.history), 1)
        self.assertGreaterEqual(worker.state['measurement_started'], verified_at[0])
        self.assertEqual(worker.hard_fault_generation, 0)
        self.assertEqual(worker.auto_reconnect['phase'], 'monitoring')

    def test_uncertain_write_ack_is_never_retried_on_new_connection(self):
        worker, timer, sdk, old, new = self.make()
        def uncertain_write(args, data):
            old.writes.append((args, data))
            old.failed = True
            raise ModbusTransportError('uncertain write ACK')
        old.rm_write_registers = uncertain_write
        worker.wake.on_wait = lambda: worker.stop.set() if new.positions else worker.command(0.)
        self.run_loop(worker, timer, sdk, [old, new], lambda *_: (0, 12000, 1.))
        self.assertEqual(len(old.writes), 1)
        self.assertEqual(new.writes, [])
        self.assertIsNone(worker.desired)
        self.assertEqual(worker.connection_generation, 1)

    def test_protocol_failure_does_not_reconnect(self):
        worker, timer, sdk, old, new = self.make()
        old.reconnectable = False
        def fail():
            if old.positions == 2:
                old.failed = True
                raise ModbusTransportError('bad protocol frame')
        old.on_position = fail
        factory, verify = self.run_loop(worker, timer, sdk, [old, new], lambda *_: (0, 12000, 1.))
        self.assertEqual(factory.call_count, 1)
        verify.assert_not_called()
        self.assertEqual(worker.connection_generation, 0)
        self.assertFalse(worker.state['ready'])

    def test_changed_configuration_is_terminal_and_never_repaired(self):
        worker, timer, sdk, old, new = self.make()
        def fail():
            if old.positions == 2:
                old.failed = True
                raise ModbusTransportError('EOF')
        old.on_position = fail
        factory, verify = self.run_loop(worker, timer, sdk, [old, new],
            lambda *_: (_ for _ in ()).throw(RuntimeError('bus mode changed')))
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(verify.call_count, 1)
        self.assertIn('bus mode changed', worker.state['error'])
        self.assertEqual(worker.auto_reconnect['phase'], 'failed')
        self.assertEqual(old.writes + new.writes, [])

    def test_stop_interrupts_backoff_without_setup_or_new_connection(self):
        worker, timer, sdk, old, new = self.make()
        old.failed = True
        worker.transport = old
        worker.stop.on_wait = worker.stop.set
        with patch.object(gripper.clock, 'now', timer.now), \
                patch.object(worker, '_verify_readonly_reconnect') as verify, \
                patch.object(worker, '_new_json_transport') as factory:
            with self.assertRaises(MotionCancelled):
                worker._recover_readonly(sdk, None)
        verify.assert_not_called()
        factory.assert_not_called()
        self.assertTrue(old.closed)
        self.assertEqual(worker.auto_reconnect['phase'], 'stopped')

    def test_three_attempts_are_bounded_and_no_writes_are_sent(self):
        worker, timer, sdk, old, _ = self.make()
        old.failed = True
        worker.transport = old
        worker.mapping_min, worker.mapping_max = 0, 12000
        buses = [ReconnectIO(worker, timer) for _ in range(3)]
        for bus in buses:
            def fail(bus=bus):
                bus.failed = True
                raise ModbusTransportError('connection lost')
            bus.on_enable_read = fail
        with patch.object(gripper.clock, 'now', timer.now), \
                patch.object(worker, '_verify_readonly_reconnect', return_value=(0, 12000, 1.)), \
                patch.object(worker, '_new_json_transport', side_effect=buses) as factory:
            with self.assertRaisesRegex(RuntimeError, '三次失败'):
                worker._recover_readonly(sdk, lambda *args: args)
            self.assertEqual(factory.call_count, 3)
        self.assertAlmostEqual(timer.now(), 10+.5+1+2+3*.06)
        self.assertEqual(worker.auto_reconnect['count'], 3)
        self.assertTrue(all(bus.closed and not bus.writes for bus in buses))
        self.assertFalse(worker.state['ready'])
        # A different break inside the same minute cannot evade the global cap.
        replacement = ReconnectIO(worker, timer)
        replacement.failed = True
        worker.transport = replacement
        with patch.object(gripper.clock, 'now', timer.now), \
                patch.object(worker, '_verify_readonly_reconnect') as verify:
            with self.assertRaisesRegex(RuntimeError, '次数或时间上限'):
                worker._recover_readonly(sdk, None)
        verify.assert_not_called()

    def test_base_worker_hook_refuses_reusing_mutating_setup(self):
        worker, timer, sdk, old, _ = self.make()
        worker.mapping_min, worker.mapping_max = 0, 12000
        old.failed = True
        worker.transport = old
        with patch.object(gripper.clock, 'now', timer.now), patch.object(worker, '_setup') as setup:
            with self.assertRaisesRegex(RuntimeError, '缺少只读配置核验'):
                worker._recover_readonly(sdk, None)
        setup.assert_not_called()

    def test_manual_intent_cannot_queue_during_readonly_recovery(self):
        worker, _, _, _, _ = self.make()
        worker._reconnecting = True
        worker.thread = Mock(is_alive=Mock(return_value=True))
        with self.assertRaisesRegex(RuntimeError, '手动请求未入队'):
            worker.set_enabled(True)
        self.assertTrue(worker.requests.empty())


if __name__ == '__main__':
    unittest.main()
