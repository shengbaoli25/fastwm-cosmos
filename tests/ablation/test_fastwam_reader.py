"""Exercise the unchanged official reader with tiny embedded PNG episodes."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

AVAILABLE = all(importlib.util.find_spec(name) is not None
                for name in ("torch", "datasets", "pyarrow", "accelerate", "omegaconf", "git", "boto3"))


@unittest.skipUnless(AVAILABLE, "Needs the training reader's optional dependencies")
class FastWAMReader(unittest.TestCase):
    def test_embedded_png_two_cameras_and_no_cross_episode_action(self):
        import datasets
        import torch
        import pyarrow.parquet as pq
        from PIL import Image
        from hydra.utils import instantiate
        from fastwam.ablation.fastwam import make_training_config, convert_stats, text_cache_path
        from fastwam.ablation.data import numeric_stats
        from fastwam.ablation.contracts import write_json
        from fastwam.utils.misc import register_work_dir
        from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "dataset"
            (root / "meta").mkdir(parents=True)
            (root / "data/chunk-000").mkdir(parents=True)
            task = "拔下插头并插入另一插座"
            raw_stats, episode_stats = {}, []
            all_actions = []
            metadata = {key: {"dtype": "float32", "shape": [8], "names": None}
                        for key in ("observation.state", "action")}
            metadata.update({"observation.images." + camera: {"dtype": "image", "shape": [480, 640, 3],
                              "names": ["height", "width", "channels"]} for camera in ("global", "wrist")})
            metadata.update({key: {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1], "names": None}
                             for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")})
            hf_features = datasets.Features({key: datasets.Sequence(datasets.Value("float32"), length=8)
                                             for key in ("observation.state", "action")})
            hf_features.update({"observation.images." + camera: datasets.Image() for camera in ("global", "wrist")})
            hf_features.update({key: datasets.Value("float32" if key == "timestamp" else "int64")
                                for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")})
            pngs = {}
            for camera, value in (("global", 0), ("wrist", 255)):
                stream = io.BytesIO()
                Image.fromarray(np.full((480, 640, 3), value, np.uint8)).save(stream, format="PNG")
                pngs[camera] = {"bytes": stream.getvalue(), "path": None}
            for ep in range(2):
                action = np.full((3, 8), 0.1 + ep * 0.7, dtype=np.float32)
                all_actions.append(action)
                rows = {"observation.state": action.tolist(), "action": action.tolist(),
                        "timestamp": [0., 1 / 30, 2 / 30], "frame_index": [0, 1, 2],
                        "episode_index": [ep] * 3, "index": list(range(ep * 3, ep * 3 + 3)), "task_index": [0] * 3}
                rows.update({"observation.images." + camera: [pngs[camera]] * 3 for camera in pngs})
                ds = datasets.Dataset.from_dict(rows, features=hf_features)
                ds.to_parquet(str(root / f"data/chunk-000/episode_{ep:06d}.parquet"))
                episode_stats.append({"episode_index": ep, "stats": {key: numeric_stats(action)
                                      for key in ("observation.state", "action")}})
            info = dict(codebase_version="v2.1", robot_type="rm71d", fps=30, total_episodes=2,
                        total_frames=6, total_tasks=1, total_videos=0, total_chunks=1, chunks_size=1000,
                        splits={"train": "0:2"}, data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                        video_path=None, features=metadata)
            write_json(root / "meta/info.json", info)
            for filename, rows in (("tasks.jsonl", [{"task_index": 0, "task": task}]),
                                   ("episodes.jsonl", [{"episode_index": ep, "tasks": [task], "length": 3} for ep in range(2)]),
                                   ("episodes_stats.jsonl", episode_stats)):
                (root / "meta" / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            raw_stats = {key: numeric_stats(np.concatenate(all_actions)) for key in ("observation.state", "action")}
            stats = convert_stats(raw_stats, "synthetic-only")
            write_json(base / "stats.json", stats)
            cache = base / "text"
            cache.mkdir()
            torch.save({"context": torch.zeros((128, 4096)), "mask": torch.ones(128, dtype=torch.bool)}, text_cache_path(cache, task))
            cfg = make_training_config(Path(__file__).resolve().parents[2] / "configs/ablation/fastwam.yaml",
                                       [f"dataset_root={root.as_posix()}", f"normalization_path={(base / 'stats.json').as_posix()}",
                                        f"text_embedding_cache_dir={cache.as_posix()}", "asset_root=unused", "pretrained_fastwam_checkpoint=unused"])
            register_work_dir(base / "work")
            reader = instantiate(cfg.data.train)
            sample = reader._get(2)  # Last row of episode zero must never cross into episode one.
            self.assertEqual(tuple(sample["video"].shape), (3, 9, 224, 448))
            self.assertEqual(tuple(sample["action"].shape), (32, 8))
            self.assertEqual(tuple(sample["proprio"].shape), (32, 8))
            torch.testing.assert_close(sample["action"], sample["action"][0].repeat(32, 1))
            torch.testing.assert_close(sample["video"][:, :, :, :224], torch.full((3, 9, 224, 224), -1.))
            torch.testing.assert_close(sample["video"][:, :, :, 224:], torch.ones((3, 9, 224, 224)))
            # Verify lightweight deployment normalization matches the actual native implementation.
            from fastwam.ablation.fastwam import normalization_parameters
            mean = torch.tensor(stats["state"]["default"]["global_mean"], dtype=torch.float32)
            std = torch.tensor(stats["state"]["default"]["global_std"], dtype=torch.float32)
            official = SingleFieldLinearNormalizer({"mean": mean, "std": std}, mode="z-score")
            raw = torch.linspace(-4, 4, 8)
            scale, offset = normalization_parameters(stats, "state")
            torch.testing.assert_close(torch.clamp(raw * scale + offset, -5, 5), official.forward(raw), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
