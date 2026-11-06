"""Deterministic IPC-lateness tests; no SDK, camera process, or robot socket."""
import threading
import unittest
from unittest.mock import Mock, patch

from realman_sync_client.camera import RealSenseCamera
from realman_sync_client.hardware import RealManHardwareAdapter


class CameraAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.001
        self.clock = patch('realman_sync_client.clock.now', side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.cameras = [RealSenseCamera(dict(name=name, serial=name), width=2,
            height=2, fps=30, max_age_s=.12) for name in ('global', 'wrist')]
        for camera in self.cameras:
            self.addCleanup(camera.close)
        self.hardware = object.__new__(RealManHardwareAdapter)
        self.hardware.cameras = self.cameras
        self.hardware.config = dict(camera_pair_skew_s=.05, camera_max_age_s=.12,
                                    cameras=[dict(name='global'), dict(name='wrist')])
        for camera in self.cameras:
            self.publish(camera, 9.94)

    def publish(self, camera, stamp):
        meta = dict(name=camera.spec['name'], serial=camera.spec['serial'],
            received=stamp, frame_number=camera._revision + 1,
            color_order='RGB', width=2, height=2)
        camera._accept((meta, bytes(12), None))

    def select(self, *, wait=.02):
        return self.hardware._aligned_camera_frames(10., 10., wait_s=wait)

    def test_late_past_frames_use_one_shared_budget_and_fixed_anchor(self):
        budgets = []
        def deliver(camera, delay, stamp):
            def wait(revision, timeout):
                budgets.append(timeout)
                self.assertLessEqual(delay, timeout)
                self.now += delay
                self.publish(camera, stamp)
                return True
            return wait
        with patch.object(self.cameras[0], 'wait_for_frame_update',
                          side_effect=deliver(self.cameras[0], .013, 9.975)), \
             patch.object(self.cameras[1], 'wait_for_frame_update',
                          side_effect=deliver(self.cameras[1], .006, 9.98)):
            selected = self.select()
        self.assertEqual([meta['received'] for _, meta in selected], [9.975, 9.98])
        self.assertAlmostEqual(budgets[0], .020)
        self.assertAlmostEqual(budgets[1], .007)
        self.assertAlmostEqual(self.now, 10.020)

    def test_future_delivery_does_not_replace_old_causal_frame(self):
        self.publish(self.cameras[1], 9.98)
        def future(revision, timeout):
            self.now += timeout
            self.publish(self.cameras[0], 10.002)
        with patch.object(self.cameras[0], 'wait_for_frame_update', side_effect=future):
            with self.assertRaisesRegex(RuntimeError, '时间差超限'):
                self.select()
        self.assertEqual(self.cameras[0].snapshot(10., self.now)[1]['received'], 9.94)

    def test_observe_keeps_one_force_snapshot_and_original_grid_stamp(self):
        hardware = self.hardware
        hardware.connected = True
        hardware.lock = threading.RLock()
        hardware.sequence = 17
        hardware.config.update(action_space='joint8', gripper_feedback_max_age_s=1.)
        hardware.feedback = Mock()
        hardware.feedback.force_filter = object()
        force_meta = dict(source_time=9.998, grid_time=10., source_age_s=.002)
        hardware.feedback.model_snapshot.return_value = (
            (0.,) * 7, (0.,) * 6, (0.,) * 6, 9.998, force_meta)
        hardware.gripper = Mock()
        hardware.gripper.snapshot.return_value = .5
        def deliver(revision, timeout):
            self.assertAlmostEqual(timeout, .020)
            self.now += .013
            for camera in self.cameras:
                self.publish(camera, 9.98)
        with patch.object(self.cameras[0], 'wait_for_frame_update', side_effect=deliver):
            observation = hardware.observe_at(10.)
        hardware.feedback.model_snapshot.assert_called_once_with(10.)
        hardware.gripper.snapshot.assert_called_once_with(9.998, 1.)
        self.assertEqual(observation.stamp, 10.)
        self.assertEqual(observation.sequence, 17)
        self.assertEqual(observation.force_meta, force_meta)
        self.assertEqual(hardware.sequence, 18)

    def test_first_camera_cannot_spend_second_camera_budget(self):
        calls = []
        def first(revision, timeout):
            calls.append(timeout)
            self.now += timeout
            self.publish(self.cameras[0], 9.98)
        with patch.object(self.cameras[0], 'wait_for_frame_update', side_effect=first), \
             patch.object(self.cameras[1], 'wait_for_frame_update') as second:
            with self.assertRaisesRegex(RuntimeError, '时间差超限'):
                self.select()
            second.assert_not_called()
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(self.now, 10.021)

    def test_requested_wait_cannot_exceed_shared_twenty_ms_cap(self):
        def timeout(revision, seconds):
            self.assertAlmostEqual(seconds, .020)
            self.now += seconds
            return False
        with patch.object(self.cameras[0], 'wait_for_frame_update', side_effect=timeout) as wait:
            with self.assertRaisesRegex(RuntimeError, '相机与机械臂状态时间差超限'):
                self.select(wait=.5)
        wait.assert_called_once()
        self.assertAlmostEqual(self.now, 10.021)

    def test_matching_old_pair_still_fails_state_skew_after_wait(self):
        # Pair skew is zero, but both timestamps remain 60 ms before the fixed
        # state anchor. A larger arrival budget must not relax the 50 ms gate.
        def timeout(revision, seconds):
            self.now += seconds
            return False
        with patch.object(self.cameras[0], 'wait_for_frame_update', side_effect=timeout):
            with self.assertRaisesRegex(RuntimeError, '相机与机械臂状态时间差超限'):
                self.select()
        self.assertEqual([camera.snapshot(10., self.now)[1]['received']
                          for camera in self.cameras], [9.94, 9.94])

    def test_late_collector_is_capped_by_next_30hz_grid(self):
        self.now = 10.031
        def timeout(revision, seconds):
            self.assertAlmostEqual(seconds, 10. + 1/30 - 10.031)
            self.now += seconds
            return False
        with patch.object(self.cameras[0], 'wait_for_frame_update', side_effect=timeout):
            with self.assertRaisesRegex(RuntimeError, '时间差超限'):
                self.select()
        self.assertAlmostEqual(self.now, 10. + 1/30)

    def test_nonblocking_preflight_does_not_wait_or_allocate_sequence(self):
        self.hardware.sequence = 99
        with patch.object(self.cameras[0], 'wait_for_frame_update') as wait:
            with self.assertRaisesRegex(RuntimeError, '时间差超限'):
                self.select(wait=0.)
            wait.assert_not_called()
        self.assertEqual(self.hardware.sequence, 99)

    def test_real_age_not_relaxed_by_small_anchor_age(self):
        self.publish(self.cameras[0], 9.98)
        self.publish(self.cameras[1], 9.98)
        self.now = 10.2
        with patch.object(self.cameras[0], 'wait_for_frame_update') as wait:
            with self.assertRaisesRegex(RuntimeError, '画面过期'):
                self.select()
            wait.assert_not_called()  # already beyond the next grid

    def test_camera_fault_is_not_retried_as_late_ipc(self):
        self.cameras[0]._accept(({}, None, '相机连接断开'))
        with patch.object(self.cameras[0], 'wait_for_frame_update') as wait:
            with self.assertRaisesRegex(RuntimeError, '相机连接断开'):
                self.select()
            wait.assert_not_called()

    def test_notification_between_selection_and_wait_is_not_lost(self):
        camera = self.cameras[0]
        revision = camera.frame_revision()
        self.publish(camera, 9.98)
        self.assertTrue(camera.wait_for_frame_update(revision, 0.))

    def test_close_wakes_waiter_without_starting_capture(self):
        camera = self.cameras[0]
        entered = threading.Event()
        returned = threading.Event()
        revision = camera.frame_revision()
        def waiting():
            entered.set()
            camera.wait_for_frame_update(revision, 1.)
            returned.set()
        thread = threading.Thread(target=waiting)
        thread.start()
        self.assertTrue(entered.wait(.5))
        camera.close()
        self.assertTrue(returned.wait(.2))
        thread.join(.2)
        self.assertFalse(thread.is_alive())


if __name__ == '__main__':
    unittest.main()
