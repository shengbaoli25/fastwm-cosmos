"""Reset button scheduling tests; no robot SDK, HTTP socket, camera or GPU."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from realman_sync_client import webapp
from realman_sync_client.mock import MockHardware
from test_webapp import Client, GateTransport, META, until


SAVED_POSE = {
    'schema': 'realman-reset-pose-v1', 'robot_ip': '192.168.1.18', 'model': 'RM_75',
    'recorded_at': '2025-01-01T00:00:00+00:00',
    'joint_degrees': [0., 10., -5., 20., 0., 30., -10.],
    'speed_percent': 10, 'gripper_action': 'unchanged',
}
SAVED_POSE['joint_radians'] = [math.radians(value) for value in SAVED_POSE['joint_degrees']]


class ResetPoseFileTests(unittest.TestCase):
    def load_dashboard(self, folder, pose):
        config = Path(folder) / 'client.json'
        config.write_text('{}', encoding='utf8')
        (Path(folder) / 'reset_pose.json').write_text(json.dumps(pose), encoding='utf-8-sig')
        return webapp.Dashboard(GateTransport(), mode='mock', hardware_config=config)

    def test_saved_file_next_to_hardware_config_is_loaded_and_exposed(self):
        with tempfile.TemporaryDirectory(prefix='reset-pose-test-') as folder:
            dashboard = self.load_dashboard(folder, SAVED_POSE)
            self.assertEqual(dashboard.saved_reset_pose, SAVED_POSE)
            state = dashboard.status()['reset_pose']
            self.assertTrue(state['saved'])
            self.assertFalse(state['available'], 'A saved file alone does not connect hardware')
            self.assertEqual(state['joint_degrees'], SAVED_POSE['joint_degrees'])
            self.assertEqual(state['recorded_at'], SAVED_POSE['recorded_at'])
            self.assertEqual(state['speed_percent'], 10)
            self.assertIsNone(dashboard.hardware)

    def test_invalid_file_is_not_admitted_as_a_saved_reset_target(self):
        variants = {
            'schema': {'schema': 'other'},
            'wrong_count': {'joint_degrees': [0] * 6},
            'boolean': {'joint_degrees': [True] + SAVED_POSE['joint_degrees'][1:]},
            'non_finite': {'joint_radians': [float('nan')] * 7},
            'units': {'joint_radians': [0] * 7},
            'speed': {'speed_percent': 11},
            'gripper_action': {'gripper_action': 'open'},
            'model': {'model': 'other'},
            'recorded_at': {'recorded_at': ''},
        }
        with tempfile.TemporaryDirectory(prefix='reset-pose-invalid-test-') as folder:
            for label, changes in variants.items():
                with self.subTest(label=label):
                    dashboard = self.load_dashboard(folder, dict(deepcopy(SAVED_POSE), **changes))
                    state = dashboard.status()['reset_pose']
                    self.assertFalse(state['saved'])
                    self.assertFalse(state['available'])
                    self.assertTrue(state['reason'])
                    with self.assertRaises(ValueError):
                        dashboard.control('reset_pose')


class ResetHardware(MockHardware):
    def __init__(self):
        super().__init__(width=8, height=6,
                         force_preprocessing_id=META['force_preprocessing']['id'],
                         force_preprocessing=META['force_preprocessing'])
        self.execution_available = True
        self.reset_token = None
        self.reset_calls = []
        self.reset_completions = []
        self.capture_calls = 0
        self.invalidate_calls = 0
        self.reset_entered = threading.Event()
        self.reset_release = threading.Event()
        self.reset_release.set()
        self.prepare_entered = threading.Event()
        self.prepare_release = threading.Event()
        self.prepare_release.set()
        self.block_next_stop = False
        self.before_reset = None

    def reset_available(self):
        return not self.closed, None if not self.closed else '设备未连接'

    def capture_reset_token(self):
        self.capture_calls += 1
        self.reset_token = object()
        return self.reset_token

    def invalidate_feedback_recovery(self):
        self.invalidate_calls += 1
        self.reset_token = None

    def reset_to_pose(self, pose, *, expected_token):
        if expected_token is None or expected_token is not self.reset_token:
            raise InterruptedError('复位已被停止取消')
        if self.before_reset:
            self.before_reset()
        self.reset_calls.append((deepcopy(pose), expected_token, threading.get_ident()))
        # Planned movej does not set the normal action-stream armed flag.
        self.armed = False
        self.reset_entered.set()
        while not self.reset_release.wait(.005):
            if expected_token is not self.reset_token:
                raise InterruptedError('复位已被停止取消')
        if expected_token is not self.reset_token:
            raise InterruptedError('复位已被停止取消')
        self.reset_completions.append(deepcopy(pose))
        return {'ok': True, 'completed': True, 'message': '记录位置已到达',
                'speed_percent': 10, 'gripper_action': 'unchanged'}

    def slow_stop(self):
        super().slow_stop()
        if self.block_next_stop:
            self.block_next_stop = False
            self.prepare_entered.set()
            if not self.prepare_release.wait(2):
                raise RuntimeError('test preparation gate timed out')

    def emergency_stop(self):
        self.slow_stop()


class DashboardResetPoseTests(unittest.TestCase):
    def setUp(self):
        Client.instances = []
        Client.collector_hook = None
        self.fixtures = []
        self.control_threads = []
        self.client_patch = patch.object(webapp, 'SynchronousClient', Client)
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.transport, self.hardware, self.dashboard = self.make_fixture()

    def make_fixture(self):
        transport, hardware = GateTransport(), ResetHardware()
        dashboard = webapp.Dashboard(transport, mode='mock', hardware_factory=lambda _meta: hardware)
        dashboard.saved_reset_pose = deepcopy(SAVED_POSE)
        self.fixtures.append((transport, hardware, dashboard))
        return transport, hardware, dashboard

    def close_fixture(self, fixture):
        transport, hardware, dashboard = fixture
        dashboard.stop_event.set()
        hardware.prepare_release.set()
        hardware.reset_release.set()
        for gate in transport.gates:
            gate.set()
        dashboard.close()
        self.assertFalse(dashboard.thread and dashboard.thread.is_alive())

    def tearDown(self):
        for _transport, hardware, dashboard in self.fixtures:
            dashboard.stop_event.set()
            hardware.prepare_release.set()
            hardware.reset_release.set()
        for thread in self.control_threads:
            thread.join(2)
            self.assertFalse(thread.is_alive())
        for fixture in self.fixtures:
            self.close_fixture(fixture)
        Client.collector_hook = None

    def start_request(self, fixture=None):
        transport, hardware, dashboard = fixture or (self.transport, self.hardware, self.dashboard)
        dashboard.start()
        until(lambda: len(transport.calls) == 1)
        return dashboard.client

    def control_in_thread(self, dashboard, action):
        done, errors = threading.Event(), []
        def control():
            try:
                dashboard.control(action)
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()
        thread = threading.Thread(target=control, daemon=True)
        self.control_threads.append(thread)
        thread.start()
        return thread, done, errors

    def test_missing_saved_pose_rejects_reset_without_changing_control_state(self):
        self.start_request()
        self.dashboard.saved_reset_pose = None
        generation = self.dashboard.generation
        state = self.dashboard.status()['reset_pose']
        self.assertFalse(state['saved'])
        self.assertFalse(state['available'])
        with self.assertRaises(ValueError):
            self.dashboard.control('reset_pose')
        self.assertEqual(self.dashboard.generation, generation)
        self.assertTrue(self.dashboard.auto_infer)
        self.assertEqual(self.hardware.capture_calls, 0)
        self.assertEqual(self.hardware.reset_calls, [])

    def test_reset_pauses_immediately_then_runs_on_worker_and_discards_old_http_action(self):
        old_client = self.start_request()
        self.dashboard.control('execute_start')
        self.hardware.armed = True
        self.dashboard.recovery.update(active=True, resume_execution=True, reason='old transient error')
        def before_reset():
            self.assertTrue(old_client.collector.closed)
            self.assertIn(old_client.session_id, self.transport.released)
            self.assertIsNone(self.dashboard.client)
        self.hardware.before_reset = before_reset
        result = self.dashboard.control('reset_pose')
        self.assertIn('message', result)
        self.assertFalse(self.dashboard.auto_infer)
        self.assertFalse(self.dashboard.execute)
        self.assertFalse(self.dashboard.status()['recovery']['active'])
        self.assertTrue(old_client.stop_event.is_set())
        self.assertFalse(self.hardware.armed)
        self.assertGreaterEqual(self.hardware.stop_count, 1)
        self.assertTrue(self.dashboard.status()['reset_pose']['busy'])
        self.assertEqual(self.hardware.reset_calls, [])
        self.transport.gates[0].set()
        until(lambda: bool(self.hardware.reset_completions))
        until(lambda: not self.dashboard.status()['reset_pose']['busy'])
        state = self.dashboard.status()
        self.assertTrue(state['reset_pose']['last_result']['completed'])
        self.assertFalse(state['inference']['enabled'])
        self.assertFalse(state['execution']['enabled'])
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(self.hardware.commands, [], 'The old inference result must never execute')
        self.assertEqual(len(self.hardware.reset_calls), 1)
        pose, token, thread_id = self.hardware.reset_calls[0]
        self.assertEqual(pose, SAVED_POSE)
        self.assertIs(token, self.hardware.reset_token)
        self.assertEqual(thread_id, self.dashboard.thread.ident)

    def test_stop_cancels_queued_reset_before_old_http_returns(self):
        for action in ('execute_stop', 'emergency_stop', 'disconnect'):
            with self.subTest(action=action):
                fixture = self.make_fixture()
                transport, hardware, dashboard = fixture
                self.start_request(fixture)
                dashboard.control('reset_pose')
                dashboard.control(action)
                transport.gates[0].set()
                until(lambda: not dashboard.status()['reset_pose']['busy'])
                if action == 'disconnect':
                    until(lambda: dashboard.phase == 'disconnected')
                    self.assertTrue(hardware.closed)
                else:
                    until(lambda: dashboard.inference['phase'] == 'paused')
                self.assertFalse(dashboard.auto_infer)
                self.assertFalse(dashboard.execute)
                self.assertEqual(hardware.reset_calls, [])
                self.assertEqual(hardware.commands, [])
                self.close_fixture(fixture)

    def test_user_stop_during_initial_slow_stop_prevents_later_reset_queue(self):
        self.start_request()
        self.hardware.armed = True
        self.hardware.block_next_stop = True
        self.hardware.prepare_release.clear()
        reset_thread, _reset_done, _reset_errors = self.control_in_thread(self.dashboard, 'reset_pose')
        self.assertTrue(self.hardware.prepare_entered.wait(1))
        self.assertTrue(self.dashboard.status()['reset_pose']['busy'])
        stop_thread, stop_done, stop_errors = self.control_in_thread(self.dashboard, 'execute_stop')
        stopped_promptly = stop_done.wait(.5)
        self.hardware.prepare_release.set()
        reset_thread.join(1)
        stop_thread.join(1)
        self.assertTrue(stopped_promptly, 'User stop must not wait behind reset preparation')
        self.assertEqual(stop_errors, [])
        self.assertFalse(reset_thread.is_alive())
        self.transport.gates[0].set()
        until(lambda: self.dashboard.inference['phase'] == 'paused')
        self.assertFalse(self.dashboard.status()['reset_pose']['busy'])
        self.assertEqual(self.hardware.reset_calls, [])
        self.assertEqual(self.hardware.commands, [])

    def test_busy_reset_blocks_other_controls_but_stop_cancels_running_reset_even_unarmed(self):
        self.start_request()
        self.hardware.reset_release.clear()
        self.dashboard.control('reset_pose')
        self.transport.gates[0].set()
        self.assertTrue(self.hardware.reset_entered.wait(1))
        self.assertFalse(self.hardware.armed)
        for action in ('reset_pose', 'connect', 'inference_start', 'execute_start',
                       'sampling_set', 'gripper_enable', 'gripper_open'):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.dashboard.control(action)
        stops = self.hardware.stop_count
        self.dashboard.control('execute_stop')
        self.assertGreater(self.hardware.stop_count, stops,
                           'Reset movej must stop even when the ordinary action stream is unarmed')
        until(lambda: not self.dashboard.status()['reset_pose']['busy'])
        self.assertFalse(self.dashboard.auto_infer)
        self.assertFalse(self.dashboard.execute)
        self.assertEqual(self.hardware.reset_completions, [])
        self.assertEqual(self.hardware.commands, [])

    def test_disconnect_cancels_running_reset_stops_then_closes_hardware(self):
        self.start_request()
        self.hardware.reset_release.clear()
        self.dashboard.control('reset_pose')
        self.transport.gates[0].set()
        self.assertTrue(self.hardware.reset_entered.wait(1))
        self.assertFalse(self.hardware.armed)
        stops = self.hardware.stop_count
        self.dashboard.control('disconnect')
        self.assertGreater(self.hardware.stop_count, stops)
        until(lambda: self.dashboard.phase == 'disconnected')
        self.assertTrue(self.hardware.closed)
        self.assertFalse(self.dashboard.status()['reset_pose']['busy'])
        self.assertFalse(self.dashboard.auto_infer)
        self.assertFalse(self.dashboard.execute)
        self.assertEqual(self.hardware.reset_completions, [])
        self.assertEqual(self.hardware.commands, [])


if __name__ == '__main__':
    unittest.main()
