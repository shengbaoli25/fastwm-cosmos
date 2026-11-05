"""Real reference payloads with synthetic cache manifests; no torch required."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fastwam.cosmos.joint_action import joint_action_contract
from fastwam.cosmos.reference_bank import atomic_json, digest, file_digest, prepare_bank
from test_reference_bank_cpu import source_fixture

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'scripts/cosmos/verify_reference_cache_binding.py'
spec = importlib.util.spec_from_file_location('reference_cache_binding', SCRIPT)
binding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(binding)


class CacheBindingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        raw, self.bank = self.root/'raw', self.root/'bank'
        source_fixture(raw)
        self.bank_manifest = prepare_bank(raw, self.bank, expected_episodes=2, max_stride=4)
        bm = self.bank_manifest
        files = dict(bm['identity']['source_files'])
        files['README.md'] = 'a'*64
        files['videos/camera/file.mp4'] = 'b'*64
        contract = joint_action_contract('delta_current_joint7')
        source = dict(profile=binding.PROFILE, source_schema='realman_leader_force_gripper_v2',
            files=files, source_inventory_sha256=digest(files), action_space='joint8',
            state_columns=[0,1,2,3,4,5,6,19], action_columns=list(range(8)),
            state_dim=8, action_dim=8, history_dim=14,
            actions='sent_joint_targets_minus_current_measured_joint7_then_absolute_gripper',
            action_statistics_scope='all_retained_anchor_windows_valid_action_positions_excluding_padding',
            action_transform_sha256='c'*64, fps=30,
            episode_lengths=bm['identity']['source_lengths'],source_episode_indices=[0,1],
            training_selection='all_episodes_no_holdout',split_counts=dict(train=2,val=0,test=0),
            segments=[[e['episode_index'],lo,hi] for e in bm['episodes'] for lo,hi in e['segment_bounds']],
            action_offsets=list(range(32)),force_offsets=list(range(1,33)),
            history_offsets=list(range(-31,1)),image_offsets=list(range(0,33,4)),**contract)
        identity = dict(schema='cosmos-force28-full-window-cache-v1', scope='production',
                        dataset_profile=binding.PROFILE, realman_source=source,
                        episode_lengths=list(source['episode_lengths']), **contract)
        self.cache = dict(identity=identity, status='complete', completed_samples=bm['total_retained_anchors'])
        self.path = self.root/'cache/manifest.json'
        self.path.parent.mkdir()
        self.write_cache()

    def write_cache(self):
        i = self.cache['identity']
        s = i['realman_source']
        s['source_inventory_sha256'] = digest(s['files'])
        for key in ('numeric_identity','numeric_sidecar_identity','pixel_identity'):
            i[key] = digest(s)
        self.cache['identity_sha256'] = digest(i)
        atomic_json(self.path, self.cache)

    def verify(self):
        return binding.verify_binding(self.bank, self.path)

    def test_matching_manifest_binds_but_does_not_claim_training_readiness(self):
        receipt = self.verify()
        self.assertEqual(receipt['status'],'bound')
        self.assertTrue(receipt['numeric_metadata_binding_verified'])
        self.assertIsNone(receipt['full_raw_inventory_equal'])
        self.assertIsNone(receipt['other_files_diff'])
        self.assertFalse(receipt['cache_tensor_payloads_verified'])
        self.assertFalse(receipt['training_readiness_evaluated'])
        self.assertNotIn('training_ready',receipt)

    def test_other_file_change_is_diagnosed_without_rejecting_numeric_binding(self):
        files = copy.deepcopy(self.cache['identity']['realman_source']['files'])
        files['README.md'] = 'd'*64
        inventory = self.root/'local-full-inventory.json'
        atomic_json(inventory,files)
        receipt = binding.verify_binding(self.bank,self.path,local_raw_inventory=inventory)
        self.assertEqual(receipt['status'],'bound')
        self.assertFalse(receipt['full_raw_inventory_equal'])
        self.assertEqual([x['path'] for x in receipt['other_files_diff']['changed']], ['README.md'])
        files['README.md'] = 'a'*64
        atomic_json(inventory,files)
        self.assertTrue(binding.verify_binding(self.bank,self.path,local_raw_inventory=inventory)['full_raw_inventory_equal'])

    def test_changed_missing_and_extra_relevant_files_are_rejected(self):
        original = copy.deepcopy(self.cache)
        for operation in ('change','missing','extra'):
            with self.subTest(operation=operation):
                self.cache = copy.deepcopy(original)
                files = self.cache['identity']['realman_source']['files']
                if operation == 'change': files['meta/info.json'] = 'd'*64
                elif operation == 'missing': files.pop('data/episode_000000.parquet')
                else: files['meta/extra.json'] = 'e'*64
                self.write_cache()
                receipt = self.verify()
                self.assertEqual(receipt['status'],'rejected')
                self.assertFalse(receipt['numeric_metadata_binding_verified'])
                self.assertFalse(receipt['numeric_metadata_files_equal'])

    def test_lengths_segments_and_delta_anchor_cannot_be_silently_changed(self):
        original = copy.deepcopy(self.cache)
        for field in ('length','segment','anchor','membership','absolute','completed'):
            with self.subTest(field=field):
                self.cache = copy.deepcopy(original)
                s = self.cache['identity']['realman_source']
                if field == 'length': s['episode_lengths'][0] += 1
                elif field == 'segment': s['segments'][0][2] -= 1
                elif field == 'anchor': s['action_anchor'] = 'previous_action'
                elif field == 'membership': s['source_episode_indices'] = [1,0]
                elif field == 'absolute': s.update(joint_action_contract('absolute'))
                else: self.cache['completed_samples'] -= 1
                self.write_cache()
                receipt = self.verify()
                self.assertEqual(receipt['status'],'rejected')
                self.assertFalse(receipt['numeric_metadata_binding_verified'])

    def test_tampered_digest_or_corrupted_bank_rejects(self):
        self.cache['identity_sha256'] = '0'*64
        atomic_json(self.path,self.cache)
        self.assertEqual(self.verify()['status'],'rejected')
        self.write_cache()
        path = self.bank/self.bank_manifest['episodes'][0]['path']
        path.write_bytes(b'invalid payload')
        self.assertEqual(self.verify()['status'],'rejected')

    def test_optional_full_inventory_must_match_bank(self):
        inventory = self.root/'bad-inventory.json'
        files = dict(self.cache['identity']['realman_source']['files'])
        files['meta/info.json'] = 'e'*64
        atomic_json(inventory,files)
        receipt = binding.verify_binding(self.bank,self.path,local_raw_inventory=inventory)
        self.assertEqual(receipt['status'],'rejected')

    def test_cli_writes_failure_receipt_without_modifying_bank(self):
        before = {p.relative_to(self.bank).as_posix():file_digest(p)
                  for p in self.bank.rglob('*') if p.is_file()}
        self.cache['identity']['realman_source']['files']['meta/info.json'] = 'd'*64
        self.write_cache()
        receipt = self.root/'receipts/binding.json'
        p = subprocess.run([sys.executable,str(SCRIPT),'--bank',str(self.bank),
                            '--cache-manifest',str(self.path),'--output',str(receipt)],
                           capture_output=True,text=True)
        self.assertEqual(p.returncode,2,msg=p.stderr)
        self.assertEqual(json.loads(receipt.read_text())['status'],'rejected')
        after = {p.relative_to(self.bank).as_posix():file_digest(p)
                 for p in self.bank.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        isolated = subprocess.run([sys.executable,'-c',
            "import runpy,sys; runpy.run_path(sys.argv[1]); assert 'torch' not in sys.modules",str(SCRIPT)],
            capture_output=True,text=True)
        self.assertEqual(isolated.returncode,0,msg=isolated.stderr)


if __name__ == '__main__':
    unittest.main()
