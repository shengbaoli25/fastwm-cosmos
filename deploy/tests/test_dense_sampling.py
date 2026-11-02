"""Dense dashboard sampling transactions; CPU only, no official weights or hardware."""
from copy import deepcopy
from http.client import HTTPConnection
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'deploy'), str(ROOT / 'src')]

from realman_sync.dense_policy import DensePolicy, _sampling_config, _persist_sampling
from realman_sync.mock_policy import MockPolicy
from realman_sync.server import Conflict, InferenceService, PolicyHTTPServer
from realman_sync.wire import PROTOCOL


def request(profile='dense10'):
    return dict(protocol=PROTOCOL, sampling_profile=profile)


def bare_policy(path):
    policy = object.__new__(DensePolicy)
    policy.config = _sampling_config(dict(training_config='exact-trained.yaml',
        normalization_file='exact-normalization.json', custom={'keep': True}))
    policy.config_path = path
    policy.num_inference_steps = 16
    policy.inference_index = 37
    policy.model, policy.teacher = object(), object()
    policy.metadata = MockPolicy(width=2, height=2).metadata
    policy.diagnostics = {}
    policy._warmup_sampling = Mock()
    path.write_text(json.dumps(policy.config), encoding='utf-8')
    return policy


class DenseSamplingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'runtime.json'
        self.policy = bare_policy(self.path)

    def test_settings_truthfully_describe_single_visual_forward(self):
        sampling = self.policy.sampling_settings()
        self.assertEqual(sampling['sampling_profile'], 'dense16')
        self.assertIsNone(sampling['video_steps'])
        self.assertEqual(sampling['video_mode'], 'single_forward')
        self.assertEqual((sampling['force_steps'], sampling['action_steps']), (16, 16))
        self.assertEqual([entry['id'] for entry in sampling['profiles']], ['dense16', 'dense10'])
        sampling['profiles'][0]['force_steps'] = 99
        self.assertEqual(self.policy.sampling_settings()['profiles'][0]['force_steps'], 16)

    def test_configuration_does_not_touch_training_and_rejects_false_profile_labels(self):
        source = dict(model_kwargs={'num_inference_steps': 16}, num_inference_steps=16)
        candidate = _sampling_config(source, 'dense10')
        self.assertEqual(source['num_inference_steps'], 16)
        self.assertEqual(candidate['num_inference_steps'], 10)
        self.assertEqual(candidate['model_kwargs']['num_inference_steps'], 16)
        self.assertEqual(_sampling_config(candidate), candidate)
        for changes in ({'sampling_profile': '3310'}, {'sampling_profile': 10},
                        {'num_inference_steps': True}, {'num_inference_steps': 0},
                        {'execution_mode': 'async_relaxed'},
                        {'sampling_profile': 'dense10', 'num_inference_steps': 16}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                _sampling_config(changes)
        # Existing callers can still exercise a one-step tiny model outside UI.
        self.assertEqual(_sampling_config({'num_inference_steps': 1})['num_inference_steps'], 1)

    def test_switch_warms_then_persists_without_changing_model_or_rng(self):
        model, teacher = self.policy.model, self.policy.teacher
        def warm(steps):
            self.assertEqual(steps, 10)
            self.assertEqual(self.policy.num_inference_steps, 16)
            self.assertEqual(json.loads(self.path.read_text())['num_inference_steps'], 16)
        self.policy._warmup_sampling.side_effect = warm
        result = self.policy.configure_sampling('dense10')
        self.assertEqual(result['action_steps'], 10)
        self.assertIs(self.policy.model, model)
        self.assertIs(self.policy.teacher, teacher)
        self.assertEqual(self.policy.inference_index, 37)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved['num_inference_steps'], 10)
        self.assertEqual(saved['training_config'], 'exact-trained.yaml')
        self.assertEqual(saved['custom'], {'keep': True})
        self.policy.configure_sampling('dense10')
        self.policy._warmup_sampling.assert_called_once_with(10)
        self.policy._warmup_sampling.side_effect = None
        self.assertEqual(self.policy.configure_sampling('dense16')['action_steps'], 16)

    def test_failed_warmup_or_atomic_write_preserves_published_settings(self):
        for stage in ('warmup', 'persist'):
            with self.subTest(stage=stage):
                policy = bare_policy(self.path)
                original = self.path.read_bytes()
                before = policy.config
                policy._warmup_sampling.side_effect = RuntimeError('warm failed') if stage == 'warmup' else None
                with patch('realman_sync.dense_policy.os.replace', side_effect=OSError('disk failed')) as replace:
                    with self.assertRaises((RuntimeError, OSError)):
                        policy.configure_sampling('dense10')
                    if stage == 'warmup':
                        replace.assert_not_called()
                self.assertIs(policy.config, before)
                self.assertEqual(policy.num_inference_steps, 16)
                self.assertEqual(policy.inference_index, 37)
                self.assertEqual(self.path.read_bytes(), original)
                self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_service_rejects_unadvertised_profiles_active_sessions_and_inflight_calls(self):
        service = InferenceService(self.policy)
        for profile in ('4416', '3310', '161616', 'unknown', None):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                service.configure_sampling(request(profile))
        service.session_id = 'active'
        with self.assertRaises(Conflict):
            service.configure_sampling(request())
        service.release(dict(protocol=PROTOCOL, session_id='active'))
        service.lock.acquire()
        try:
            with self.assertRaises(Conflict):
                service.configure_sampling(request())
        finally:
            service.lock.release()
        original = deepcopy(service.metadata)
        result = service.configure_sampling(request())
        self.assertEqual(result['sampling']['sampling_profile'], 'dense10')
        self.assertEqual(service.metadata, original)
        self.assertEqual(service.metadata['action_horizon'], 32)

    def test_native_dense_model_runs_one_visual_forward_and_selected_force_action_counts(self):
        import torch
        from fastwam.cosmos.dense_features import DenseFeatureJointModel
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            policy = self.policy
            del policy._warmup_sampling  # Exercise the real deployment implementation.
            policy.device, policy.dtype = 'cpu', torch.float32
            policy.model = DenseFeatureJointModel(manifest={'scope': 'deployment-cpu-test'},
                feature_dim=8, hidden_dim=24, depth=1, heads=2, view_grid=2,
                action_hidden=24, action_ffn=48, force_hidden=24, force_ffn=48,
                joint_heads=2, joint_head_dim=12, context_dim=32,
                gradient_checkpointing=False, num_inference_steps=16).eval()
            manifest = deepcopy(policy.model.manifest)
            for steps in (16, 10):
                with patch.object(policy.model.dit['video'], 'forward',
                        wraps=policy.model.dit['video'].forward) as video, \
                     patch.object(policy.model, 'joint_denoise_core',
                        wraps=policy.model.joint_denoise_core) as denoise:
                    policy._warmup_sampling(steps)
                self.assertEqual(video.call_count, 1)
                self.assertEqual(denoise.call_count, steps)
                self.assertEqual(policy.model.num_inference_steps, 16)
                self.assertEqual(policy.model.manifest, manifest)
                self.assertEqual(policy.inference_index, 37)
            self.assertFalse(torch.cuda.is_initialized())
        finally:
            torch.set_num_threads(previous_threads)


class DenseSamplingHTTPTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.policy = bare_policy(Path(self.directory.name) / 'runtime.json')
        self.service = InferenceService(self.policy)
        self.server = PolicyHTTPServer(('127.0.0.1', 0), self.service, token='local-test')
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.directory.cleanup()

    def http(self, method, path, data=None, token='local-test'):
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=2)
        try:
            connection.request(method, path, None if data is None else json.dumps(data),
                {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_auth_dynamic_profiles_and_health_contract(self):
        self.assertEqual(self.http('GET', '/sampling', token='wrong')[0], 401)
        self.assertEqual(self.http('POST', '/sampling', request(), token='wrong')[0], 401)
        original = self.http('GET', '/health')[1]
        self.assertIsNone(original['sampling']['video_steps'])
        self.assertEqual(self.http('POST', '/sampling', request('3310'))[0], 400)
        status, result = self.http('POST', '/sampling', request())
        self.assertEqual(status, 200)
        self.assertEqual(result['sampling']['sampling_profile'], 'dense10')
        self.assertEqual(self.http('GET', '/health')[1]['metadata'], original['metadata'])

    def test_warmup_keeps_old_published_settings_and_rejects_concurrent_change(self):
        entered, release = threading.Event(), threading.Event()
        def warm(_steps):
            entered.set()
            if not release.wait(2):
                raise RuntimeError('test gate timeout')
        self.policy._warmup_sampling.side_effect = warm
        replies = []
        worker = threading.Thread(target=lambda: replies.append(self.http('POST', '/sampling', request())))
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            health = self.http('GET', '/health')[1]
            self.assertFalse(health['ready'])
            self.assertTrue(health['sampling']['updating'])
            self.assertEqual(health['sampling']['sampling_profile'], 'dense16')
            self.assertEqual(self.http('POST', '/sampling', request())[0], 409)
        finally:
            release.set()
            worker.join(2)
        self.assertEqual(replies[0][0], 200)
        self.assertTrue(self.http('GET', '/health')[1]['ready'])


if __name__ == '__main__':
    unittest.main()
