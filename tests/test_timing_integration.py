"""Timing experiments at real gates and the async dashboard, never hardware I/O."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from realman_sync_client import clock, webapp
from realman_sync_client.gripper import Ctag2F120Worker, FeedbackUnavailable
from realman_sync_client.hardware import RealManHardwareAdapter
from realman_sync_client.mock import MockHardware
from realman_sync_client.timing_profiles import hardware_settings
from realman_sync_client.web_hardware import WebHardware, _CheckedGripper
from test_hardware_feedback_recovery import Bus
from test_web_hardware import config
from test_webapp import GateTransport, META, until


CAPS = dict(force_max_age_s=.10, camera_max_age_s=.25, camera_pair_skew_s=.15)


class TimingHardwareIntegrationTests(unittest.TestCase):
    def test_can_infer_rejects_larger_local_limits_before_sampling_sensors(self):
        hardware = object.__new__(WebHardware)
        hardware.config = dict(hardware_settings('relaxed'), camera_width=8,
                               camera_height=6, command_period_s=1/30)
        hardware.status = Mock(return_value={'force_preprocessing': META['force_preprocessing']})
        hardware._local_ready = Mock(return_value=(True, None))
        hardware.feedback = Mock()
        hardware.feedback.model_snapshot.return_value = (None, None, None, 10., {})
        hardware.gripper = Mock()
        hardware._aligned_camera_frames = Mock(return_value=[])
        hardware.observe_at = Mock(side_effect=AssertionError('Preflight cannot collect a new observation'))
        ready, reason = hardware.can_infer(META)
        self.assertFalse(ready)
        self.assertIn('Server', reason)
        hardware.status.assert_not_called()
        hardware.feedback.model_snapshot.assert_not_called()
        meta = dict(META, temporal_tolerances=CAPS)
        self.assertEqual(hardware.can_infer(meta), (True, None))
        hardware.observe_at.assert_not_called()
        hardware.config.update(hardware_settings('strict'))
        self.assertEqual(hardware.can_infer(meta), (True, None))

    def test_checked_gripper_uses_profile_at_both_joint_and_target_gates(self):
        timer = SimpleNamespace(value=10.)
        with patch.object(clock, 'now', lambda: timer.value):
            workers = {}
            for name in ('strict', 'relaxed'):
                timer.value = 10.
                worker = _CheckedGripper(hardware_settings(name))
                worker.mapping_min, worker.mapping_max = 0, 12000
                worker._poll_feedback(Bus(timer), lambda *args: args, 0, 12000)
                workers[name] = worker
            timer.value = 11.2
            for name, worker in workers.items():
                hardware = object.__new__(WebHardware)
                hardware.gripper, hardware._closing = worker, False
                hardware._acknowledged_gripper_faults = 0
                with patch.object(Ctag2F120Worker, '_send_target', return_value=11.2) as actual_send:
                    if name == 'strict':
                        with self.assertRaisesRegex(RuntimeError, '反馈未就绪'):
                            hardware._check_command_dispatch()
                        with self.assertRaises(FeedbackUnavailable):
                            worker._send_target(None, None, .4, .4, .5, 0, 'policy')
                        actual_send.assert_not_called()
                    else:
                        hardware._check_command_dispatch()
                        self.assertEqual(worker._send_target(None, None, .4, .4, .5, 0, 'policy'), 11.2)
                        actual_send.assert_called_once()

    def test_relaxed_profile_never_overrides_alarm_or_disabled_feedback(self):
        timer = SimpleNamespace(value=10.)
        with patch.object(clock, 'now', lambda: timer.value):
            worker = _CheckedGripper(hardware_settings('relaxed'))
            bus = Bus(timer)
            for enabled, alarm in ((True, 32), (False, 0)):
                with self.subTest(enabled=enabled, alarm=alarm):
                    bus.enabled, bus.alarm = int(enabled), alarm
                    worker.enabled_received = None
                    worker._poll_feedback(bus, lambda *args: args, 0, 12000)
                    self.assertFalse(worker.status()['ready'])
                    with patch.object(Ctag2F120Worker, '_send_target') as actual_send:
                        with self.assertRaises(FeedbackUnavailable):
                            worker._send_target(None, None, .4, .4, .5, 0, 'policy')
                        actual_send.assert_not_called()

    def test_profile_hardware_overrides_are_bounded_and_leave_motion_parameters_unchanged(self):
        original = config()
        for name in ('strict', 'relaxed'):
            source = dict(original, timing_profile=name, camera_max_age_s=999,
                          gripper_control_max_age_s=999, feedback_max_age_s=999)
            actual = RealManHardwareAdapter._validate_config(source)
            for key, expected in hardware_settings(name).items():
                self.assertEqual(actual[key], expected)
            for key in ('command_period_s', 'joint_limit_margin_deg', 'gripper_speed', 'gripper_torque'):
                self.assertEqual(actual[key], original[key])
            self.assertEqual(source['camera_max_age_s'], 999, 'Validation must not mutate its caller')
        for value in ('disabled', True, None):
            with self.assertRaises(ValueError):
                RealManHardwareAdapter._validate_config(dict(original, timing_profile=value))


class TimingDashboardIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'hardware.json'
        self.path.write_text(json.dumps(dict(timing_profile='strict', preserved_value=42)), encoding='utf-8')
        self.transport = GateTransport()
        self.transport.metadata['temporal_tolerances'] = dict(CAPS)
        self.created = []
        def factory(metadata):
            item = MockHardware(width=8, height=6, force_preprocessing=META['force_preprocessing'],
                                force_preprocessing_id=META['force_preprocessing']['id'])
            self.created.append(item)
            return item
        self.dashboard = webapp.Dashboard(self.transport, mode='mock', hardware_config=self.path,
                                         hardware_factory=factory, auto_infer=False)
        self.dashboard.metadata = dict(self.transport.metadata)
        self.addCleanup(self.finish)

    def finish(self):
        self.dashboard.stop_event.set()
        for gate in self.transport.gates:
            gate.set()
        self.dashboard.close()

    def saved(self):
        return json.loads(self.path.read_text(encoding='utf-8'))

    def test_queued_switch_persists_then_reconnects_without_starting_inference(self):
        self.dashboard.start()
        until(lambda: self.dashboard.hardware is not None)
        old = self.dashboard.hardware
        self.dashboard.control('timing_set', timing_profile='relaxed')
        until(lambda: self.dashboard.timing_profile == 'relaxed')
        until(lambda: self.dashboard.hardware is not None and self.dashboard.hardware is not old)
        self.assertEqual(self.saved(), dict(timing_profile='relaxed', preserved_value=42))
        self.assertTrue(old.closed)
        self.assertEqual(self.dashboard.client_options.max_observation_age_s, .3)
        self.assertFalse(self.dashboard.auto_infer)
        self.assertFalse(self.dashboard.execute)
        self.assertEqual(self.transport.calls, [])
        self.assertTrue(all(not hw.commands for hw in self.created))

    def test_active_execution_refuses_switch_before_changing_or_queueing_anything(self):
        self.dashboard.execute = True
        with self.assertRaisesRegex(ValueError, '停止动作执行'):
            self.dashboard.control('timing_set', timing_profile='relaxed')
        self.assertIsNone(self.dashboard._timing_action)
        self.assertEqual(self.saved()['timing_profile'], 'strict')
        self.dashboard.execute = False

    def test_strict_server_does_not_advertise_or_accept_relaxed(self):
        self.dashboard.metadata.pop('temporal_tolerances')
        self.assertEqual([p['id'] for p in self.dashboard.timing_status()['profiles']], ['strict'])
        with self.assertRaisesRegex(ValueError, '不支持'):
            self.dashboard.control('timing_set', timing_profile='relaxed')
        self.assertIsNone(self.dashboard._timing_action)

    def test_stop_cancels_queued_switch_before_worker_can_take_it(self):
        self.dashboard.control('timing_set', timing_profile='relaxed')
        self.dashboard.control('inference_stop')
        self.assertIsNone(self.dashboard._timing_action)
        self.dashboard.start()
        until(lambda: self.dashboard.hardware is not None)
        self.assertEqual(self.saved()['timing_profile'], 'strict')
        self.assertEqual(self.dashboard.timing_profile, 'strict')

    def test_stop_during_close_prevents_profile_file_commit(self):
        entered, release = threading.Event(), threading.Event()
        real_close = self.dashboard._close_hardware
        def gated_close():
            entered.set()
            release.wait(2)
            real_close()
        with patch.object(self.dashboard, '_close_hardware', side_effect=gated_close):
            self.dashboard.control('timing_set', timing_profile='relaxed')
            self.dashboard.start()
            self.assertTrue(entered.wait(1))
            try:
                self.dashboard.control('inference_stop')
            finally:
                release.set()
            until(lambda: not self.dashboard._timing_busy)
            self.assertEqual(self.saved()['timing_profile'], 'strict')
            self.assertEqual(self.dashboard.timing_profile, 'strict')
            self.assertFalse(self.dashboard.auto_infer)


if __name__ == '__main__':
    unittest.main()
