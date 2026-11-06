"""Bounded experiment profiles cannot be mutated through returned UI/config data."""
import unittest

from realman_sync_client.timing_profiles import (
    hardware_settings, motion_settings, profile_status, validate_profile)


class TimingProfilesTests(unittest.TestCase):
    def test_explicit_hardware_thresholds(self):
        self.assertEqual(hardware_settings('strict'), dict(gripper_control_max_age_s=.5,
            gripper_feedback_max_age_s=1.5, feedback_max_age_s=.03,
            force_source_max_age_s=.03, camera_max_age_s=.12, camera_pair_skew_s=.05))
        self.assertEqual(hardware_settings('relaxed'), dict(gripper_control_max_age_s=1.5,
            gripper_feedback_max_age_s=2., feedback_max_age_s=.1,
            force_source_max_age_s=.1, camera_max_age_s=.25, camera_pair_skew_s=.15))

    def test_profiles_leave_device_and_training_parameters_out(self):
        forbidden = {'gripper_speed', 'gripper_torque', 'joint_limit_margin_deg',
                     'gripper_read_timeout_s', 'command_period_s', 'force_filter', 'num_inference_steps'}
        for profile in ('strict', 'relaxed'):
            status = profile_status(profile)
            self.assertFalse(forbidden.intersection(status['limits']))
            self.assertGreater(motion_settings(profile)['settle_window_s'], 0)
        self.assertEqual(motion_settings('relaxed')['max_inference_drift_deg'], 2.)
        self.assertEqual(motion_settings('strict')['max_inference_drift_deg'], .5)

    def test_unknown_profiles_fail_without_coercion(self):
        for invalid in (None, True, 1, '', 'RELAXED', 'strict ', {}, []):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_profile(invalid)

    def test_returned_config_and_status_cannot_change_future_defaults(self):
        settings = hardware_settings('relaxed')
        settings['gripper_control_max_age_s'] = 999
        status = profile_status('relaxed')
        status['limits']['feedback_max_age_s'] = 999
        status['profiles'][0]['label'] = 'mutated'
        status['profiles'].clear()
        fresh = profile_status('relaxed')
        self.assertEqual(fresh['limits']['gripper_control_max_age_s'], 1.5)
        self.assertEqual(fresh['limits']['feedback_max_age_s'], .1)
        self.assertEqual([p['id'] for p in fresh['profiles']], ['strict', 'relaxed'])


if __name__ == '__main__':
    unittest.main()
