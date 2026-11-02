"""Small contract regressions; no network, model download or training run."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fastwam.ablation.fastwam import (
    HEAD_KEYS, adaptation_state, convert_stats, text_cache_path, prepare_observation,
    load_joint8_checkpoint, make_training_config, normalization_parameters, semantic_recipe,
    build_binding, VAE_RELATIVE_PATH,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_OMEGA = importlib.util.find_spec("omegaconf") is not None


class FastWAMContracts(unittest.TestCase):
    def tensors(self):
        target = {"mixtures.action.action_encoder.weight": np.ones((4, 8)),
                  "mixtures.action.action_encoder.bias": np.ones(4),
                  "mixtures.action.head.weight": np.ones((8, 4)),
                  "mixtures.action.head.bias": np.ones(8),
                  "mixtures.video.blocks.0.weight": np.ones((4, 4)),
                  "mixtures.action.blocks.0.weight": np.ones((4, 4))}
        source = {k: np.full_like(v, 17) for k, v in target.items()}
        source["mixtures.action.action_encoder.weight"] = np.full((4, 7), 17)
        source["mixtures.action.head.weight"] = np.full((7, 4), 17)
        source["mixtures.action.head.bias"] = np.full(7, 17)
        return target, source

    def test_only_robot_heads_reset_including_shape_compatible_bias(self):
        target, source = self.tensors()
        loaded = adaptation_state(target, source)
        for key in target:
            np.testing.assert_array_equal(loaded[key], target[key] if key in HEAD_KEYS else source[key])

    def test_backbone_shape_mismatch_and_missing_keys_fail(self):
        target, source = self.tensors()
        source["mixtures.video.blocks.0.weight"] = np.ones((5, 4))
        with self.assertRaisesRegex(ValueError, "Backbone"):
            adaptation_state(target, source)
        target, source = self.tensors()
        del source["mixtures.action.blocks.0.weight"]
        with self.assertRaisesRegex(ValueError, "keys differ"):
            adaptation_state(target, source)

    def test_joint8_checkpoint_cannot_masquerade_as_7d_release(self):
        target, _ = self.tensors()
        with self.assertRaisesRegex(ValueError, "7D"):
            adaptation_state(target, target)

    def test_stats_are_raw_joint8_not_padded_7d(self):
        raw = {key: {name: [0.25] * 8 for name in ("mean", "std", "min", "max", "q01", "q99")}
               for key in ("observation.state", "action")}
        result = convert_stats(raw, "identity")
        self.assertEqual(result["action"]["default"]["global_std"], [0.25] * 8)
        self.assertEqual(result["ablation"]["dataset_identity"], "identity")
        raw["action"]["mean"] = [0] * 7
        with self.assertRaises(ValueError):
            convert_stats(raw, "identity")

    def test_text_cache_names_follow_official_wrapped_task(self):
        self.assertNotEqual(text_cache_path("cache", "one"), text_cache_path("cache", "two"))
        self.assertTrue(str(text_cache_path("cache", "one")).endswith(".t5_len128.wan22ti2v5b.pt"))

    def test_deployment_rejects_semantic_recipe_change_with_same_assets(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vae = root / VAE_RELATIVE_PATH
            vae.parent.mkdir(parents=True)
            vae.write_bytes(b"synthetic VAE identity only")
            cache = root / "text"
            cache.mkdir()
            text_cache_path(cache, "task").write_bytes(b"synthetic T5 identity only")
            raw = {key: {name: [0.25] * 8 for name in ("mean", "std", "min", "max", "q01", "q99")}
                   for key in ("observation.state", "action")}
            norm = root / "normalization.json"
            norm.write_text(json.dumps(convert_stats(raw, "dataset")), encoding="utf-8")
            manifest = {"identity_sha256": "dataset", "prompt": "task"}
            with patch("fastwam.ablation.fastwam.load_dataset_manifest", return_value=manifest):
                original = build_binding("unused", norm, root, cache, training_recipe={"seed": 42})
            with self.assertRaisesRegex(ValueError, "Deployment binding differs"):
                build_binding(None, norm, root, cache, training_recipe={"seed": 43}, deployment_binding=original)

    @unittest.skipUnless(HAS_OMEGA, "OmegaConf required")
    def test_training_config_is_8d_absolute_with_two_cameras(self):
        root = Path(__file__).resolve().parents[2]
        values = [f"{k}=placeholder" for k in ("dataset_root", "normalization_path", "asset_root",
                  "text_embedding_cache_dir", "pretrained_fastwam_checkpoint")]
        cfg = make_training_config(root / "configs/ablation/fastwam.yaml", values)
        self.assertEqual(cfg.model.action_dit_config.action_dim, 8)
        self.assertEqual(cfg.max_steps, 50000)
        self.assertEqual(cfg.save_every, 5000)
        self.assertTrue(Path(cfg.output_dir).is_absolute())
        self.assertEqual(cfg.model.dataset_root, cfg.dataset_root)
        with self.assertRaises(ValueError):
            make_training_config(root / "configs/ablation/fastwam.yaml", values + ["data.train.video_size=[224,224]"])
        recipe = semantic_recipe(cfg)
        cfg.model.action_scheduler.train_shift = 2
        cfg.model.action_scheduler.infer_shift = 2
        self.assertNotEqual(semantic_recipe(cfg), recipe)
        cfg = make_training_config(root / "configs/ablation/fastwam.yaml", values + ["seed=43"])
        self.assertNotEqual(semantic_recipe(cfg), recipe)

    @unittest.skipUnless(HAS_TORCH and HAS_OMEGA and importlib.util.find_spec("transformers") is not None,
                         "Needs official model imports")
    def test_factory_forwards_native_precision_and_scheduler_arguments(self):
        import inspect
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        from hydra.utils import instantiate
        from fastwam.models.wan22.fastwam import FastWAM
        root = Path(__file__).resolve().parents[2]
        values = [f"{k}=placeholder" for k in ("dataset_root", "normalization_path", "asset_root",
                  "text_embedding_cache_dir", "pretrained_fastwam_checkpoint")]
        cfg = make_training_config(root / "configs/ablation/fastwam.yaml", values)
        signature = inspect.signature(FastWAM.from_wan22_pretrained)
        captured = {}

        def constructor(**kwargs):
            signature.bind(**kwargs)  # Real upstream constructor signature, no allocation.
            captured.update(kwargs)
            return SimpleNamespace()

        with patch("fastwam.ablation.fastwam.build_binding", return_value={}), \
                patch.object(FastWAM, "from_wan22_pretrained", side_effect=constructor):
            instantiate(cfg.model, initialize_from_official=False, model_dtype=torch.bfloat16, device="cpu")
        self.assertEqual(captured["torch_dtype"], torch.bfloat16)
        self.assertEqual(captured["video_train_shift"], 5.0)
        self.assertEqual(captured["action_train_shift"], 1.0)
        self.assertEqual(captured["action_infer_shift"], 1.0)
        self.assertEqual(captured["action_num_train_timesteps"], 1000)
        self.assertEqual(captured["action_dit_config"]["action_dim"], 8)
        self.assertNotIn("model_dtype", captured)
        cfg.model.action_scheduler.train_shift = 2
        cfg.model.action_scheduler.infer_shift = 2
        with patch("fastwam.ablation.fastwam.build_binding", return_value={}), \
                patch.object(FastWAM, "from_wan22_pretrained", side_effect=constructor):
            with self.assertRaisesRegex(Exception, "semantic training recipe"):
                instantiate(cfg.model, initialize_from_official=False, model_dtype=torch.bfloat16, device="cpu")

    @unittest.skipUnless(HAS_TORCH, "Torch required")
    def test_preprocessing_preserves_camera_order_and_joint_units(self):
        image, state = prepare_observation({"state": np.arange(8, dtype=np.float32) / 10,
                                           "images": {"global": np.zeros((480, 640, 3), np.uint8),
                                                      "wrist": np.full((480, 640, 3), 255, np.uint8)}})
        self.assertEqual(tuple(image.shape), (3, 224, 448))
        self.assertTrue((image[:, :, :224] == -1).all())
        np.testing.assert_allclose(image[:, :, 224:].numpy(), 1, atol=1e-6)
        np.testing.assert_allclose(state.numpy(), np.arange(8) / 10)

    @unittest.skipUnless(HAS_TORCH, "Torch required")
    def test_normalization_roundtrip_keeps_eighth_action_component(self):
        import torch
        stats = {"action": {"default": {"global_mean": [0.1] * 8, "global_std": [0.3] * 8}}}
        scale, offset = normalization_parameters(stats, "action")
        raw = torch.arange(8).float().repeat(32, 1) / 10
        normalized = raw * scale + offset
        recovered = (normalized - offset) / scale
        torch.testing.assert_close(recovered, raw)

    @unittest.skipUnless(HAS_TORCH, "Torch required")
    def test_deployment_rejects_unbound_checkpoint_before_loading(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            torch.save({"mot": {}, "proprio_encoder": {}}, path)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_joint8_checkpoint(object(), path, {"schema": "fastwam-realman-joint8-v1"})

    @unittest.skipUnless(HAS_TORCH, "Torch required")
    def test_joint8_checkpoint_load_is_strict_for_both_components(self):
        import torch
        from types import SimpleNamespace
        model = SimpleNamespace(mot=torch.nn.Linear(8, 8), proprio_encoder=torch.nn.Linear(8, 4))
        binding = {"schema": "fastwam-realman-joint8-v1", "dataset_identity": "test"}
        payload = {"mot": model.mot.state_dict(), "proprio_encoder": model.proprio_encoder.state_dict(),
                   "ablation": binding}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            torch.save(payload, path)
            load_joint8_checkpoint(model, path, binding)
            del payload["mot"]["bias"]
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                load_joint8_checkpoint(model, path, binding)


if __name__ == "__main__":
    unittest.main()
