"""Offline checks for the opt-in target/readback/separate-trigger path."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from realman_sync_client import gripper
from realman_sync_client.gripper_transport import MotionCancelled


class FakeIO:
    def __init__(self, worker):
        self.worker, self.now = worker, 10.
        self.calls, self.payload = [], []
        self.after = {}
        self.fail = None
        self.mismatch = None
        self.before_dispatch = None

    def stage(self, name, params, payload=None):
        if self.before_dispatch:
            self.before_dispatch(name)
        if name in ('prepare', 'trigger') and self.worker.transport is not None:
            if not self.worker.transport.motion_guard():
                raise MotionCancelled('cancelled at socket dispatch')
        self.calls.append((name, params.address, params.num, payload))
        self.now += .06
        if name in self.after:
            self.after[name]()
        if self.fail == name:
            raise TimeoutError('uncertain ' + name + ' acknowledgement')

    def rm_write_registers(self, params, payload):
        self.stage('prepare', params, list(payload))
        self.payload = list(payload)
        return 0

    def rm_read_multiple_holding_registers(self, params):
        self.stage('readback', params)
        data = list(self.payload)
        if self.mismatch is not None:
            data[self.mismatch] ^= 1
        return 0, data

    def rm_write_single_register(self, params, value):
        self.stage('trigger', params, value)
        return 0


def parameters(port, address, device, num):
    return SimpleNamespace(port=port, address=address, device=device, num=num)


class SplitTriggerTests(unittest.TestCase):
    def make_worker(self, mode='separate'):
        cfg = dict(gripper_speed=100, gripper_torque=50)
        if mode is not None:
            cfg['gripper_trigger_mode'] = mode
        worker = gripper.Ctag2F120Worker(cfg)
        worker.mapping_min, worker.mapping_max = 0, 12000
        worker.desired, worker.command_sequence, worker.command_source = 1., 7, 'manual'
        worker.state = dict(initialized=True, ready=True, enabled=True, alarm=0,
                            opening=0., received=10., measurement_started=10.)
        worker.enabled_received, worker.enabled_value = 10., True
        io = FakeIO(worker)
        return worker, io

    def send(self, worker, io, source='manual'):
        with patch.object(gripper.clock, 'now', lambda: io.now):
            return worker._send_target(io, parameters, 1., .15, 0., 7, source)

    def test_default_and_explicit_batch_keep_one_original_write(self):
        for mode in (None, 'batch'):
            with self.subTest(mode=mode):
                worker, io = self.make_worker(mode)
                self.assertIsNotNone(self.send(worker, io))
                self.assertEqual(len(io.calls), 1)
                self.assertEqual(io.calls[0][:3], ('prepare', 0x0102, 7))
                self.assertEqual(io.calls[0][3][-2:], [0, 1])
                self.assertEqual(worker.sent_target['trigger_mode'], 'batch')

    def test_separate_writes_six_words_verifies_all_then_triggers_once(self):
        worker, io = self.make_worker()
        worker.transport = SimpleNamespace(motion_guard=None, status=lambda: {})
        previous = {'sequence': 6}
        worker.sent_target = previous
        for stage in ('prepare', 'readback', 'trigger'):
            io.after[stage] = lambda: self.assertIs(worker.sent_target, previous)
        completed = self.send(worker, io)
        self.assertEqual([call[:3] for call in io.calls], [
            ('prepare', 0x0102, 6), ('readback', 0x0102, 6), ('trigger', 0x0108, 1)])
        self.assertEqual(io.calls[0][3], [0, 0, 39, 216, 0, 100, 0, 50, 0, 100, 0, 100])
        self.assertEqual(io.calls[-1][3], 1)
        self.assertEqual(worker.sent_target['acknowledged_at'], completed)
        self.assertEqual(worker.sent_target['trigger_mode'], 'separate')
        self.assertEqual(worker.sent_target['raw_target'], 10200)
        self.assertIsNone(worker.transport.motion_guard)
        self.assertIsNone(worker._target_inflight)

    def test_any_readback_word_mismatch_prevents_trigger_and_ack_publication(self):
        for word in range(6):
            with self.subTest(word=word):
                worker, io = self.make_worker()
                io.mismatch = word * 2
                with self.assertRaisesRegex(RuntimeError, '回读不一致'):
                    self.send(worker, io)
                self.assertEqual([call[0] for call in io.calls], ['prepare', 'readback'])
                self.assertIsNone(worker.sent_target)

    def test_uncertain_stage_never_retries_falls_back_or_publishes_target_ack(self):
        for stage, count in (('prepare', 1), ('readback', 2), ('trigger', 3)):
            with self.subTest(stage=stage):
                worker, io = self.make_worker()
                io.fail = stage
                with self.assertRaisesRegex(TimeoutError, 'uncertain ' + stage):
                    self.send(worker, io)
                self.assertEqual(len(io.calls), count)
                self.assertIsNone(worker.sent_target)
                self.assertIsNone(worker._target_inflight)

    def test_cancellation_at_each_boundary_prevents_next_stage(self):
        for stage, expected in (('before', 0), ('prepare', 1), ('readback', 2)):
            for cause in ('hold', 'stop', 'queued_control', 'replacement', 'stale'):
                with self.subTest(stage=stage, cause=cause):
                    worker, io = self.make_worker()
                    def cancel():
                        if cause == 'hold':
                            worker.hold()
                        elif cause == 'stop':
                            worker.stop.set()
                        elif cause == 'queued_control':
                            worker.requests.put(('disable', None))
                        elif cause == 'replacement':
                            worker.desired, worker.command_sequence = .8, 8
                        else:
                            io.now += .51
                    if stage == 'before':
                        cancel()
                    else:
                        io.after[stage] = cancel
                    self.assertIsNone(self.send(worker, io))
                    self.assertEqual(len(io.calls), expected)
                    self.assertIsNone(worker.sent_target)
                    if cause == 'replacement':
                        self.assertEqual((worker.desired, worker.command_sequence), (.8, 8))

    def test_transport_guard_stays_installed_for_both_write_dispatches(self):
        worker, io = self.make_worker()
        worker.transport = SimpleNamespace(motion_guard=None, status=lambda: {})
        io.before_dispatch = lambda name: worker.hold() if name == 'trigger' else None
        self.assertIsNone(self.send(worker, io))
        self.assertEqual([call[0] for call in io.calls], ['prepare', 'readback'])
        self.assertIsNone(worker.sent_target)
        self.assertIsNone(worker.transport.motion_guard)

    def test_new_target_during_inflight_trigger_ack_is_not_overwritten(self):
        worker, io = self.make_worker()
        def replace_target():
            worker.desired, worker.command_sequence = .8, 8
        io.after['trigger'] = replace_target
        self.assertIsNotNone(self.send(worker, io))
        self.assertEqual((worker.desired, worker.command_sequence), (.8, 8))
        self.assertEqual(worker.sent_target['sequence'], 7)
        self.assertIsNone(worker.ramp_target)

    def test_manual_override_does_not_change_policy_or_other_sources(self):
        for source, count in (('manual', 3), ('manual_open', 3), ('policy', 1), ('other', 1)):
            with self.subTest(source=source):
                worker, io = self.make_worker('batch')
                worker.config['gripper_manual_trigger_mode'] = 'separate'
                self.assertIsNotNone(self.send(worker, io, source))
                self.assertEqual(len(io.calls), count)
                self.assertEqual(worker.sent_target['trigger_mode'], 'separate' if count == 3 else 'batch')

    def test_missing_manual_override_inherits_global_and_can_explicitly_use_batch(self):
        for override, count in ((None, 3), ('batch', 1)):
            with self.subTest(override=override):
                worker, io = self.make_worker('separate')
                if override is not None:
                    worker.config['gripper_manual_trigger_mode'] = override
                self.assertIsNotNone(self.send(worker, io, 'manual_open'))
                self.assertEqual(len(io.calls), count)

    def test_invalid_modes_rejected_before_worker_start_or_sdk_call(self):
        for key in ('gripper_trigger_mode', 'gripper_manual_trigger_mode'):
            for value in ('automatic-fallback', '', None, True, 1, []):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, key):
                        gripper.Ctag2F120Worker({key:value})
                    worker, io = self.make_worker('batch')
                    worker.config[key] = value
                    with self.assertRaisesRegex(ValueError, key):
                        self.send(worker, io)
                    self.assertEqual(io.calls, [])


if __name__ == '__main__':
    unittest.main()
