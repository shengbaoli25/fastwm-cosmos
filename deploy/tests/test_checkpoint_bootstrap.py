"""Focused CPU checks; no full policy, checkpoint downloads, CUDA or hardware."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "deploy")]

import torch
from torch import nn

from fastwam.cosmos import assets
from fastwam.cosmos.noaction import SCHEMA
from fastwam.cosmos.state import ForceRuntimeState
from realman_sync.cosmos_checkpoint_policy import (
    checkpoint_contract, strict_materialize, move_materialized, checkpoint_sampling_config)


def contract_fixture():
    manifest = dict(schema=SCHEMA, joint_protocol=SCHEMA, reference_mode="none",
        action_space="joint8", action_dim=8, state_dim=8, history_dim=14,
        upstream=assets.UPSTREAM_COMMIT, model=assets.MODEL_ID, revision=assets.REVISION,
        video_variant="base/post-trained", video_sha256=assets.NONACTION_WEIGHT_SHA,
        tokenizer_sha256=assets.TOKENIZER_SHA, text_identity="a" * 64,
        dataset_identity="b" * 64, normalization_sha256="c" * 64,
        extra_config=dict(action_space="joint8", force_shift=1.0, gradient_checkpointing=True),
        recipe=dict(batch_per_rank=16, max_steps=50000, ranks=4, accum=1, global_batch=64))
    config = dict(expected_text_identity="a" * 64, expected_data_identity="b" * 64,
        expected_normalization_sha256="c" * 64, training_recipe=dict(max_steps=50000),
        gradient_checkpointing=False)
    payload = dict(schema=SCHEMA, manifest=manifest, model={"dit.video.net.weight": torch.ones(1)})
    return payload, config


class TinyState(nn.Module):
    checkpoint_schema = "unit-complete-state"

    def __init__(self, *, meta=False):
        super().__init__()
        self.manifest = {"protocol": "unit-state", "source": "pinned"}
        with torch.device("meta" if meta else "cpu"):
            self.video = nn.Linear(3, 2)
        # Exercise the actual unchanged manifest/RNG load hook on CPU.
        self.runtime = ForceRuntimeState(44, self.manifest)


class CheckpointBootstrapTests(unittest.TestCase):
    def test_all_runtime_profiles_preserve_strict_checkpoint_contract(self):
        for profile in ('161616', '4416', '3310'):
            with self.subTest(profile=profile):
                payload, configured = contract_fixture()
                original = deepcopy(payload['manifest'])
                runtime = checkpoint_sampling_config(dict(model_kwargs=configured), profile)
                checkpoint_contract(payload, runtime['model_kwargs'])
                self.assertEqual(payload['manifest'], original)
                self.assertEqual(runtime['model_kwargs'], configured)
                # Sampling settings cannot smuggle changed training steps into
                # the constructor accepted by the complete-checkpoint loader.
                runtime['model_kwargs']['num_inference_steps'] = 4
                with self.assertRaisesRegex(ValueError, 'num_inference_steps'):
                    checkpoint_contract(payload, runtime['model_kwargs'])

    def test_3310_native_joint8_matches_recomputation_and_real_forward_counts(self):
        from fastwam.cosmos.expert_schedule_inference import ExpertScheduleRunner, infer_expert_reference, expert_plan
        from fastwam.cosmos.multirate import MultirateSteps
        from fastwam.cosmos.native import nonaction_native_imports, nonaction_net_config
        from fastwam.cosmos.noaction import NoActionJointModel
        counts, ratios, plan = expert_plan(3, 3, 10)
        self.assertEqual((counts, ratios, len(plan)), ((3, 3, 10), (10, 10, 3), 30))
        self.assertEqual(sum(tick['force_features'] for tick in plan), 3)
        with self.assertRaisesRegex(ValueError, 'divisible'):
            MultirateSteps(3, 3, 10)
        upstream = Path(os.environ.get('COSMOS_UPSTREAM', ROOT / 'third_party/cosmos-predict2.5'))
        if not upstream.is_dir():
            self.skipTest('Set COSMOS_UPSTREAM to the verified Cosmos upstream checkout for native CPU parity')
        _, module, _, Net, _ = nonaction_native_imports(upstream)
        config = nonaction_net_config()
        config.update(max_img_h=8, max_img_w=8, max_frames=4, model_channels=32, num_blocks=2,
            num_heads=2, adaln_lora_dim=8, crossattn_proj_in_channels=24, crossattn_emb_channels=16,
            use_wan_fp32_strategy=True, sac_config=module.SACConfig(mode='none'))
        original_threads = torch.get_num_threads()
        runner = None
        try:
            torch.set_num_threads(2)
            torch.manual_seed(7)
            model = NoActionJointModel(Net(**config), manifest={'scope': 'deployment-cpu-test'},
                action_hidden=24, action_ffn=48, force_hidden=24, force_ffn=48, force_heads=2,
                gradient_checkpointing=False, action_space='joint8', num_inference_steps=16).eval()
            generator = torch.Generator().manual_seed(98)
            normal = lambda *shape: torch.randn(*shape, generator=generator)
            sample = dict(first_frame_latents=normal(1, 16, 1, 4, 4), cosmos_context=normal(1, 3, 24),
                proprio=normal(1, 32, 9), history=normal(1, 32, 14),
                history_is_pad=torch.zeros(1, 32, dtype=torch.bool),
                history_channel_valid=torch.ones(1, 32, 14, dtype=torch.bool),
                action_is_pad=torch.zeros(1, 32, dtype=torch.bool))
            settings = dict(video_steps=3, force_steps=3, action_steps=10)
            runner = ExpertScheduleRunner(model, **settings, compile_layers=False, cuda_graph=False)
            with torch.no_grad(), \
                 patch.object(runner.layers[0], 'video_forward', wraps=runner.layers[0].video_forward) as video, \
                 patch.object(runner.force_layers[0], 'forward', wraps=runner.force_layers[0].forward) as force, \
                 patch.object(runner.action_layers[0], 'forward', wraps=runner.action_layers[0].forward) as action:
                actual = runner.infer_joint(sample)
                self.assertEqual((video.call_count, force.call_count, action.call_count), (3, 3, 10))
                expected = infer_expert_reference(model, sample, **settings)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], atol=4e-6, rtol=4e-5)
            self.assertEqual(tuple(actual['action'].shape), (1, 32, 8))
            self.assertEqual(runner.diagnostics['effective_mode'], 'eager')
            self.assertEqual(runner.stats['merged_ticks'], 30)
            self.assertEqual(runner.stats['force_forward_evaluations'], 3)
            self.assertEqual(model.num_inference_steps, 16)
        finally:
            if runner is not None:
                runner.close()
            torch.set_num_threads(original_threads)

    def test_contract_keeps_training_manifest_and_accepts_50k_explicit_recipe(self):
        payload, config = contract_fixture()
        original = deepcopy(payload["manifest"])
        initial, kwargs = checkpoint_contract(payload, config)
        self.assertEqual(initial["recipe"]["max_steps"], 50000)
        self.assertTrue(kwargs["gradient_checkpointing"])
        self.assertEqual(payload["manifest"], original)

    def test_pinned_asset_changes_are_rejected(self):
        for key in ("upstream", "revision", "video_sha256", "tokenizer_sha256"):
            with self.subTest(key=key):
                payload, config = contract_fixture()
                payload["manifest"][key] = "d" * 64
                with self.assertRaisesRegex(ValueError, "identities"):
                    checkpoint_contract(payload, config)

    def test_external_identity_mismatch_or_absence_is_rejected(self):
        for key in ("expected_text_identity", "expected_data_identity", "expected_normalization_sha256"):
            with self.subTest(key=key):
                payload, config = contract_fixture()
                config.pop(key)
                with self.assertRaisesRegex(ValueError, key):
                    checkpoint_contract(payload, config)

    def test_changed_training_architecture_and_recipe_are_rejected(self):
        for key, value in (("force_shift", 2.0), ("action_space", "eef_pose7"),
                           ("training_recipe", {"max_steps": 30000}), ("unrecognized", 1)):
            with self.subTest(key=key):
                payload, config = contract_fixture()
                config[key] = value
                with self.assertRaises(ValueError):
                    checkpoint_contract(payload, config)

    def test_missing_video_and_wrong_schema_are_rejected(self):
        payload, config = contract_fixture()
        payload["model"] = {"action.weight": torch.ones(1)}
        with self.assertRaisesRegex(ValueError, "video"):
            checkpoint_contract(payload, config)
        payload["schema"] = "old-model"
        with self.assertRaisesRegex(ValueError, "complete"):
            checkpoint_contract(payload, config)

    def fixture(self):
        source, target = TinyState(), TinyState(meta=True)
        payload = dict(schema=source.checkpoint_schema, manifest=deepcopy(source.manifest),
                       model=source.state_dict())
        return source, target, payload

    def test_full_state_materializes_meta_and_preserves_values(self):
        source, target, payload = self.fixture()
        self.assertTrue(target.video.weight.is_meta)
        strict_materialize(target, payload)
        torch.testing.assert_close(target.video.weight, source.video.weight)
        torch.testing.assert_close(target.runtime.manifest_bytes, source.runtime.manifest_bytes)
        self.assertFalse(any(x.is_meta for x in target.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_device_move_preserves_trained_fp32_precision(self):
        source, target, payload = self.fixture()
        with torch.no_grad():
            source.video.weight.fill_(1.0001)
        strict_materialize(target, payload)
        moved = move_materialized(target, "cpu")
        self.assertEqual(moved.video.weight.dtype, torch.float32)
        self.assertTrue(torch.equal(moved.video.weight, source.video.weight))
        self.assertFalse(torch.equal(moved.video.weight, source.video.weight.bfloat16().float()))

    def test_checkpoint_precision_mismatch_is_rejected(self):
        _, target, payload = self.fixture()
        payload["model"]["video.weight"] = payload["model"]["video.weight"].bfloat16()
        with self.assertRaisesRegex(ValueError, "dtype"):
            strict_materialize(target, payload)

    def test_missing_unexpected_and_wrong_shape_states_are_rejected(self):
        for mutation in ("missing", "unexpected", "shape"):
            with self.subTest(mutation=mutation):
                _, target, payload = self.fixture()
                if mutation == "missing":
                    payload["model"].pop("video.weight")
                elif mutation == "unexpected":
                    payload["model"]["foreign.weight"] = torch.ones(1)
                else:
                    payload["model"]["video.weight"] = torch.ones(2, 4)
                with self.assertRaises(RuntimeError):
                    strict_materialize(target, payload)

    def test_original_runtime_manifest_and_rng_hooks_still_reject_tampering(self):
        for name, value in (("runtime.manifest_bytes", torch.zeros(2, dtype=torch.uint8)),
                            ("runtime.rng_state", torch.zeros(1, dtype=torch.uint8))):
            with self.subTest(name=name):
                _, target, payload = self.fixture()
                payload["model"][name] = value
                with self.assertRaises(ValueError):
                    strict_materialize(target, payload)

    def test_nonfinite_and_unmaterialized_weights_are_rejected(self):
        for value in (torch.full((2, 3), float("nan")), torch.full((2, 3), float("inf")),
                      torch.empty((2, 3), device="meta")):
            _, target, payload = self.fixture()
            payload["model"]["video.weight"] = value
            with self.assertRaises(ValueError):
                strict_materialize(target, payload)

    def test_whole_manifest_must_match_before_loading(self):
        _, target, payload = self.fixture()
        payload["manifest"]["source"] = "another"
        with self.assertRaisesRegex(ValueError, "protocol"):
            strict_materialize(target, payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
