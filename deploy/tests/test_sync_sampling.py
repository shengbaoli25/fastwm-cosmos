"""Offline sampler transactions and authenticated HTTP; never uses CUDA/hardware."""
from copy import deepcopy
from http.client import HTTPConnection
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'deploy'), str(ROOT / 'src')]

from realman_deploy.model_adapter import CosmosJointPolicy
from realman_deploy.sampling import profile_steps, updated_config
from realman_sync.cosmos_checkpoint_policy import (
    CheckpointJointPolicy, checkpoint_sampling_config, _persist_sampling)
from realman_sync.mock_policy import MockPolicy
from realman_sync.server import Conflict, InferenceService, PolicyHTTPServer
from realman_sync.wire import PROTOCOL


def payload(profile='4416'):
    return dict(protocol=PROTOCOL, sampling_profile=profile)


class SamplingPolicy(MockPolicy):
    def __init__(self):
        super().__init__(width=2, height=2)
        self.profile = '161616'
        self.configure = Mock()

    def sampling_settings(self):
        return dict(supported=True, sampling_profile=self.profile,
                    **profile_steps(self.profile),
                    execution_mode='synchronous', profiles=[])

    def configure_sampling(self, profile):
        self.configure(profile)
        self.profile = profile
        return self.sampling_settings()


class ServiceSamplingTests(unittest.TestCase):
    def setUp(self):
        self.policy = SamplingPolicy()
        self.service = InferenceService(self.policy)

    def test_switch_keeps_metadata_and_action_contract(self):
        original = deepcopy(self.service.metadata)
        result = self.service.configure_sampling(payload())
        self.assertTrue(result['updated'])
        self.assertFalse(result['sampling']['updating'])
        self.assertEqual(result['sampling']['sampling_profile'], '4416')
        self.assertEqual(self.service.metadata, original)
        self.assertEqual(self.service.metadata['action_horizon'], 32)
        self.assertEqual(self.service.metadata['control_hz'], 30)
        rapid = self.service.configure_sampling(payload('3310'))['sampling']
        self.assertEqual(tuple(rapid[key] for key in ('video_steps', 'force_steps', 'action_steps')), (3, 3, 10))
        self.assertEqual(self.service.metadata, original)
        self.assertEqual(self.service.configure_sampling(payload('161616'))['sampling']['video_steps'], 16)

    def test_unsupported_policy_and_invalid_schema_rejected_without_update(self):
        with self.assertRaises(ValueError):
            InferenceService(MockPolicy(width=2, height=2)).configure_sampling(payload())
        for value in (None, [], {}, payload('unknown'), payload(4416),
                      dict(payload(), execution_mode='async_relaxed'),
                      dict(payload(), model_kwargs={'num_inference_steps': 4}),
                      dict(payload(), protocol='other')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.service.configure_sampling(value)
        self.policy.configure.assert_not_called()

    def test_inflight_session_and_shutdown_reject_changes(self):
        self.service.lock.acquire()
        try:
            with self.assertRaises(Conflict):
                self.service.configure_sampling(payload())
        finally:
            self.service.lock.release()
        self.service.session_id = 'active'
        with self.assertRaisesRegex(Conflict, 'Release'):
            self.service.configure_sampling(payload())
        self.service.release(dict(protocol=PROTOCOL, session_id='active'))
        self.service.closing.set()
        with self.assertRaisesRegex(Conflict, 'shutting down'):
            self.service.configure_sampling(payload())
        self.policy.configure.assert_not_called()

    def test_failed_switch_preserves_published_settings_and_releases_lock(self):
        before = self.service.sampling_status()
        self.policy.configure.side_effect = RuntimeError('warmup failed')
        with self.assertRaisesRegex(RuntimeError, 'warmup failed'):
            self.service.configure_sampling(payload())
        self.assertEqual(self.service.sampling_status(), before)
        self.assertFalse(self.service.reconfiguring.is_set())
        self.assertFalse(self.service.lock.locked())
        self.assertFalse(self.service.failed)


class CheckpointSamplingTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(inference_backend='joint_static_cache', joint_steps=16,
            execution_mode='synchronous', normalization_file='exact-normalization.json',
            model_kwargs={'num_inference_steps': 16, 'action_space': 'joint8',
                          'expected_normalization_sha256': 'a' * 64},
            tokenizer='verified.pth', policy_backend='cosmos_checkpoint', custom={'keep': True})
        self.policy = object.__new__(CheckpointJointPolicy)
        self.policy.config = checkpoint_sampling_config(self.config)
        self.policy.inference_backend = 'joint_static_cache'
        self.policy.model, self.policy.tokenizer = object(), object()
        self.policy.normalizer = object()
        self.policy.runner = None
        self.policy.inference_index = 37
        self.policy.diagnostics = {'checkpoint_bootstrap': 'strict', 'old': 'kept'}

    def test_sampler_config_preserves_all_training_inputs_and_rejects_modes(self):
        original = deepcopy(self.config)
        balanced = checkpoint_sampling_config(self.config, '4416')
        self.assertEqual(self.config, original)
        self.assertEqual(balanced['model_kwargs'], original['model_kwargs'])
        self.assertEqual(balanced['joint_steps'], 16)
        self.assertEqual(balanced['inference_backend'], 'joint_multirate')
        self.assertEqual(balanced['inference_optimization'], {'mode': 'eager'})
        standard = checkpoint_sampling_config(balanced, '161616')
        self.assertEqual(standard['inference_backend'], 'joint_static_cache')
        self.assertNotIn('sampling_profile', standard)
        for change in ({'execution_mode': 'async_relaxed'}, {'joint_steps': 4}, {'joint_steps': True},
                       {'sampling_profile': 'unknown'}, {'inference_backend': 'joint_expert_schedule'},
                       {'inference_optimization': {'mode': 'auto'}},
                       {'inference_optimization': {'cuda_graph': True}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                checkpoint_sampling_config(dict(self.config, **change))

    def test_constructor_selects_runtime_sampler_without_changing_model_kwargs(self):
        with patch.object(CosmosJointPolicy, '__init__', return_value=None) as base:
            CheckpointJointPolicy('model.pt', dict(self.config, sampling_profile='4416'), config_path='runtime.json')
        supplied = base.call_args.args[1]
        self.assertEqual(supplied['sampling_profile'], '4416')
        self.assertEqual(supplied['model_kwargs'], self.config['model_kwargs'])

    def test_3310_uses_independent_eager_grids_and_round_trips(self):
        rapid = checkpoint_sampling_config(self.config, '3310')
        self.assertEqual(rapid['inference_backend'], 'joint_expert_schedule')
        self.assertEqual(rapid['inference_optimization'], {'compile_layers': False, 'cuda_graph': False})
        self.assertEqual(rapid['joint_steps'], 16)
        self.assertEqual(rapid['model_kwargs'], self.config['model_kwargs'])
        self.assertEqual(checkpoint_sampling_config(rapid), rapid)
        for profile, backend in (('4416', 'joint_multirate'), ('161616', 'joint_static_cache')):
            restored = checkpoint_sampling_config(rapid, profile)
            self.assertEqual(restored['inference_backend'], backend)
            self.assertEqual(restored['inference_optimization'], {'mode': 'eager'})
            self.assertEqual(restored['model_kwargs'], self.config['model_kwargs'])
        for options in ({'compile_layers': True}, {'cuda_graph': True}, {'compile_layers': 0}, {'mode': 'eager'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                checkpoint_sampling_config(dict(rapid, inference_optimization=options))
        self.assertEqual(profile_steps('3310'), dict(video_steps=3, force_steps=3, action_steps=10))
        shared = updated_config(self.config, '3310', 'synchronous')
        self.assertEqual(shared['inference_backend'], 'joint_expert_schedule')
        self.assertEqual(shared['inference_optimization'], rapid['inference_optimization'])
        with self.assertRaises(ValueError):
            updated_config(self.config, '3310', 'async_relaxed')

    def test_3310_hot_switch_commits_expert_binding_only_after_warmup_and_persist(self):
        runner, persist = Mock(), Mock()
        steps = profile_steps('3310')
        def configure(candidate):
            candidate.runner = runner
            candidate._expert_steps = steps
        with patch.object(CheckpointJointPolicy, '_configure_inference', configure), \
             patch.object(CheckpointJointPolicy, '_warmup') as warm:
            result = self.policy.configure_sampling('3310', persist=persist)
            self.assertEqual(self.policy._expert_steps, steps)
            self.assertEqual(result['action_steps'], 10)
            self.assertEqual(self.policy.inference_index, 37)
            self.assertIs(self.policy.runner, runner)
            warm.assert_called_once_with()
            persist.assert_called_once_with(self.policy.config)
        with patch.object(CheckpointJointPolicy, '_configure_inference'), \
             patch.object(CheckpointJointPolicy, '_warmup'):
            self.policy.configure_sampling('4416')
        self.assertFalse(hasattr(self.policy, '_expert_steps'))
        runner.close.assert_called_once_with()

    def test_3310_failed_persist_keeps_existing_expert_binding(self):
        self.policy._expert_steps = {'existing': 'binding'}
        candidate_runner = Mock()
        def configure(candidate):
            candidate.runner = candidate_runner
            candidate._expert_steps = profile_steps('3310')
        before = self.policy.config
        with patch.object(CheckpointJointPolicy, '_configure_inference', configure), \
             patch.object(CheckpointJointPolicy, '_warmup'), \
             self.assertRaisesRegex(OSError, 'disk failed'):
            self.policy.configure_sampling('3310', persist=Mock(side_effect=OSError('disk failed')))
        self.assertIs(self.policy.config, before)
        self.assertEqual(self.policy._expert_steps, {'existing': 'binding'})
        candidate_runner.close.assert_called_once_with()

    def test_stage_reuses_model_and_tokenizer_and_switches_back_to_static(self):
        model, tokenizer, normalizer = self.policy.model, self.policy.tokenizer, self.policy.normalizer
        runners, warmed = [], []
        def configure(candidate):
            if candidate.inference_backend == 'joint_multirate':
                candidate.runner = Mock()
                runners.append(candidate.runner)
        def warm(candidate):
            self.assertIs(candidate.model, model)
            self.assertIs(candidate.tokenizer, tokenizer)
            self.assertIs(candidate.normalizer, normalizer)
            self.assertIsNot(candidate, self.policy)
            warmed.append(candidate.inference_backend)
            candidate.inference_index = 0
            candidate.diagnostics['warmup'] = ['first', 'second']
        persist = Mock()
        with patch.object(CheckpointJointPolicy, '_configure_inference', configure), \
             patch.object(CheckpointJointPolicy, '_warmup', warm):
            self.policy.configure_sampling('4416', persist=persist)
            self.assertEqual(self.policy.sampling_settings()['video_steps'], 4)
            self.assertEqual(self.policy.inference_index, 37)
            self.policy.configure_sampling('161616', persist=persist)
        self.assertEqual(warmed, ['joint_multirate', 'joint_static_cache'])
        self.assertIsNone(self.policy.runner)
        runners[0].close.assert_called_once_with()
        self.assertIs(self.policy.model, model)
        self.assertEqual(self.policy.inference_index, 37)
        self.assertEqual(persist.call_count, 2)

    def test_warmup_or_persistence_failure_rolls_back_and_closes_only_candidate(self):
        for fail_at in ('warmup', 'persist'):
            with self.subTest(fail_at=fail_at):
                self.setUp()
                original_config, original_diag = self.policy.config, self.policy.diagnostics
                old_runner, candidate_runner = Mock(), Mock()
                self.policy.runner = old_runner
                persist = Mock(side_effect=OSError('disk write failed') if fail_at == 'persist' else None)
                def configure(candidate):
                    candidate.runner = candidate_runner
                def warm(candidate):
                    candidate.diagnostics['partial'] = 'must not publish'
                    if fail_at == 'warmup':
                        raise RuntimeError('warmup failed')
                with patch.object(CheckpointJointPolicy, '_configure_inference', configure), \
                     patch.object(CheckpointJointPolicy, '_warmup', warm), \
                     self.assertRaises((RuntimeError, OSError)):
                    self.policy.configure_sampling('4416', persist=persist)
                self.assertIs(self.policy.config, original_config)
                self.assertIs(self.policy.diagnostics, original_diag)
                self.assertNotIn('partial', original_diag)
                self.assertIs(self.policy.runner, old_runner)
                self.assertEqual(self.policy.inference_backend, 'joint_static_cache')
                old_runner.close.assert_not_called()
                candidate_runner.close.assert_called_once_with()
                if fail_at == 'warmup':
                    persist.assert_not_called()

    def test_atomic_persistence_preserves_non_sampler_fields_and_failure_keeps_original(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'runtime.json'
            path.write_text(json.dumps(self.config), encoding='utf-8')
            updated = checkpoint_sampling_config(self.config, '3310')
            _persist_sampling(path, updated)
            saved = json.loads(path.read_text(encoding='utf-8'))
            for key in ('model_kwargs', 'normalization_file', 'tokenizer', 'policy_backend', 'custom'):
                self.assertEqual(saved[key], self.config[key])
            self.assertEqual(saved['sampling_profile'], '3310')
            self.assertEqual(checkpoint_sampling_config(saved), saved)
            original = path.read_bytes()
            with patch('realman_sync.cosmos_checkpoint_policy.os.replace', side_effect=OSError('read only')):
                with self.assertRaises(OSError):
                    _persist_sampling(path, checkpoint_sampling_config(updated, '161616'))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(root).iterdir()), [path])


class SamplingHTTPTests(unittest.TestCase):
    def setUp(self):
        self.policy = SamplingPolicy()
        self.service = InferenceService(self.policy)
        self.server = PolicyHTTPServer(('127.0.0.1', 0), self.service, token='private-test-token')
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(self, method, path, data=None, token='private-test-token'):
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=2)
        try:
            connection.request(method, path, None if data is None else json.dumps(data),
                               {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
            reply = connection.getresponse()
            return reply.status, json.loads(reply.read())
        finally:
            connection.close()

    def test_auth_status_and_switch_keep_health_metadata_unchanged(self):
        self.assertEqual(self.request('GET', '/sampling', token='wrong')[0], 401)
        self.assertEqual(self.request('POST', '/sampling', payload(), token='wrong')[0], 401)
        _, original = self.request('GET', '/health')
        status, result = self.request('POST', '/sampling', payload())
        self.assertEqual(status, 200)
        self.assertEqual(result['sampling']['sampling_profile'], '4416')
        self.assertFalse(result['sampling']['updating'])
        _, current = self.request('GET', '/health')
        self.assertTrue(current['ready'])
        self.assertEqual(current['metadata'], original['metadata'])
        self.assertNotIn('sampling', current['metadata'])
        status, result = self.request('POST', '/sampling', payload('3310'))
        self.assertEqual(status, 200)
        self.assertEqual(result['sampling']['action_steps'], 10)
        self.assertEqual(self.request('GET', '/health')[1]['metadata'], original['metadata'])

    def test_busy_session_and_invalid_profile_http_errors(self):
        self.assertEqual(self.request('POST', '/sampling', payload('invalid'))[0], 400)
        self.service.session_id = 'active'
        self.assertEqual(self.request('POST', '/sampling', payload())[0], 409)
        self.service.session_id = None
        self.service.lock.acquire()
        try:
            self.assertEqual(self.request('POST', '/sampling', payload())[0], 409)
        finally:
            self.service.lock.release()

    def test_health_not_ready_during_warmup_and_old_settings_stay_visible(self):
        entered, release = threading.Event(), threading.Event()
        def warm(_profile):
            entered.set()
            if not release.wait(2):
                raise RuntimeError('test warmup gate timed out')
        self.policy.configure.side_effect = warm
        results = []
        worker = threading.Thread(target=lambda: results.append(self.request('POST', '/sampling', payload())))
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            _, health = self.request('GET', '/health')
            self.assertFalse(health['ready'])
            self.assertTrue(health['sampling']['updating'])
            self.assertEqual(health['sampling']['sampling_profile'], '161616')
            self.assertEqual(self.request('POST', '/sampling', payload())[0], 409)
        finally:
            release.set()
            worker.join(2)
        self.assertEqual(results[0][0], 200)
        self.assertTrue(self.request('GET', '/health')[1]['ready'])


if __name__ == '__main__':
    unittest.main()
