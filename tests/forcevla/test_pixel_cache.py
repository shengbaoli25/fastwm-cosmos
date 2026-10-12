"""CPU stock-equivalence checks on marked synthetic recorded RGB, never a VAE."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

from fastwam.cache.latent import CacheIntegrityError, cache_identity_sha256, file_sha256
from fastwam.cache.pixels import PixelCacheDataset, make_pixel_identity, validate_pixel_cache, write_pixel_cache
from fastwam.cache.deduplicated_source import (
    RecordedFrameSource, build_deduplicated_source, pixel_identity_from_source, validate_raw_config,
)
from fastwam.cache.source import build_index_rows, build_strict_source
from fastwam.cache.pixel_audit import audit_pixel_cache, validate_pixel_audit_receipt
from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _png(camera, episode, frame):
    random = np.random.default_rng(913 + camera * 10000 + episode * 1000 + frame)
    rgb = random.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    rgb[100:190, 300:401, camera] = (19 * frame + 31 * camera) % 256
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    temporary = tmp_path_factory.mktemp("stock-pixels")
    conversion = _module("pixel_conversion_fixture", "tests/forcevla/test_conversion.py")
    original, converted = temporary / "TEST-only-source", temporary / "converted"
    conversion._write_source(original)
    info = json.loads((original / "meta/info.json").read_text())
    lengths, offset = [35, 6], 0
    info["total_frames"] = sum(lengths)
    for camera in conversion.CAMERAS:
        info["features"][camera]["shape"] = [480, 640, 3]
    (original / "meta/info.json").write_text(json.dumps(info) + "\n")
    for episode, length in enumerate(lengths):
        path = original / f"data/chunk-000/episode_{episode:06d}.parquet"
        old = pq.read_table(path)
        table = old.take(pa.array([frame % len(old) for frame in range(length)], type=pa.int64()))
        values = {"index": range(offset, offset + length), "episode_index": [episode] * length,
                  "frame_index": range(length), "timestamp": np.arange(length, dtype=np.float32) / 30,
                  "observation.state": (np.arange(length * 13).reshape(length, 13) * .013123456789 + episode).tolist(),
                  "action": (np.arange(length * 7).reshape(length, 7) * .011987654321 - episode).tolist()}
        values.update({key: [{"bytes": _png(camera, episode, frame), "path": None} for frame in range(length)]
                       for camera, key in enumerate(conversion.CAMERAS)})
        for key, value in values.items():
            table = table.set_column(table.column_names.index(key), key, pa.array(value, type=table[key].type))
        pq.write_table(table, path)
        offset += length
    conversion.converter.convert_dataset(original, converted, test_fixture=True)
    stats = temporary / "stats.json"
    save_dataset_stats_to_json({group: {"default": {"global_mean": torch.linspace(-.2, .3, 7),
                                                    "global_std": torch.linspace(.1, .8, 7)}}
                               for group in ("action", "state")}, stats)
    wrapper = OmegaConf.create({"data": OmegaConf.to_container(OmegaConf.load(ROOT / "configs/data/forcevla_libero_2cam.yaml"), resolve=False)})
    raw = wrapper.data.train
    raw.dataset_dirs = [str(converted)]
    raw.pretrained_norm_stats = str(stats)
    raw.text_embedding_cache_dir = str(temporary / "texts")
    Path(raw.text_embedding_cache_dir).mkdir()
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    task = json.loads((converted / "meta/tasks.jsonl").read_text())["task"]
    prompt = DEFAULT_PROMPT.format(task=task)
    context = torch.arange(128, dtype=torch.bfloat16)[:, None].expand(128, 4096).clone()
    mask = torch.arange(128) < 11
    torch.save({"context": context, "mask": mask}, Path(raw.text_embedding_cache_dir) / (hashlib.sha256(prompt.encode()).hexdigest() + ".t5_len128.wan22ti2v5b.pt"))
    identity = pixel_identity_from_source(raw, test_fixture=True)
    root = temporary / "pixels"
    calls = []
    recorded = RecordedFrameSource(converted, identity)
    def process(index):
        calls.append(index)
        return recorded(index)
    write_pixel_cache(root, identity, process, shard_size=8)
    assert calls == list(range(sum(lengths)))
    rows = build_index_rows(converted)
    stock = build_strict_source(raw, rows, temporary / "runtime")
    deduplicated = build_deduplicated_source(raw, root, rows, test_fixture=True)
    helper = _module("pixel_original_precompute", "scripts/forcevla/precompute_vae_cache.py")
    texts, text_hashes = helper._task_texts(raw)
    return {"root": root, "raw": raw, "identity": identity, "rows": rows,
            "stock": stock, "deduplicated": deduplicated, "texts": texts, "text_hashes": text_hashes}


def _numpy_state_equal(left, right):
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def test_all_stock_anchors_bits_strides_labels_masks_prompts_rng_and_t5(fixture):
    source = fixture["deduplicated"]
    for index in range(len(source)):
        np.random.seed(178 + index)
        np_before, torch_before = np.random.get_state(), torch.get_rng_state()
        stock = fixture["stock"](index)
        np_after, torch_after = np.random.get_state(), torch.get_rng_state()
        np.random.set_state(np_before)
        torch.set_rng_state(torch_before)
        cached = source(index)
        assert _numpy_state_equal(np_after, np.random.get_state())
        assert torch.equal(torch_after, torch.get_rng_state())
        assert stock["video"].stride() == cached["video"].stride() == (100352, 301056, 448, 1)
        assert torch.equal(stock["video"].view(torch.uint8), cached["video"].view(torch.uint8)), index
        assert torch.equal(stock["video"].to(torch.bfloat16), cached["video"].to(torch.bfloat16)), index
        for key in ("action", "proprio", "image_is_pad", "action_is_pad", "proprio_is_pad", "raw_current_state", "raw_current_wrench"):
            assert stock[key].dtype == cached[key].dtype
            assert torch.equal(stock[key], cached[key]), (index, key)
        assert stock["sample_index"] == cached["sample_index"] == index
        assert stock["prompt"] == cached["prompt"]
        text = fixture["texts"][cached["prompt"]]
        assert text["context_mask"].all() and not text["context"][11:].any()
        # Independently verify future raw values against converted parquet.
        row = fixture["rows"][index]
        table = pq.read_table(Path(fixture["raw"].dataset_dirs[0]) / f"data/chunk-000/episode_{row['episode_index']:06d}.parquet")
        local = torch.minimum(row["frame_index"] + torch.arange(32), torch.tensor(len(table) - 1))
        for target, column in (("raw_action", "action"), ("raw_wrench_future", "observation.wrench")):
            expected = torch.tensor(table[column].to_pylist(), dtype=torch.float64)[local]
            assert torch.equal(cached[target].view(torch.uint8), expected.view(torch.uint8)), (index, target)
    assert source(0)["video"].shape == (3, 9, 224, 448)
    assert source(34)["image_is_pad"].tolist() == [False] + [True] * 8
    assert source(35)["image_is_pad"].tolist() == [False, False] + [True] * 7


def test_resume_never_reprocesses_committed_frames(fixture, tmp_path):
    root, calls = tmp_path / "resume", []
    source = fixture["deduplicated"].pixels
    def frame(index):
        calls.append(index)
        return source[index]
    first = write_pixel_cache(root, fixture["identity"], frame, shard_size=8, max_new_frames=11)
    assert first["status"] == "partial" and calls == list(range(11))
    with pytest.raises(CacheIntegrityError, match="Incomplete"):
        PixelCacheDataset(root)
    calls.clear()
    final = write_pixel_cache(root, fixture["identity"], frame, shard_size=8, resume=True)
    assert final["status"] == "complete" and calls == list(range(11, 41))
    calls.clear()
    assert write_pixel_cache(root, fixture["identity"], frame, shard_size=8, resume=True) == final
    assert calls == []
    with pytest.raises(FileExistsError):
        write_pixel_cache(root, fixture["identity"], frame)


@pytest.mark.parametrize("change", ["sha", "index", "nonfinite", "untracked", "identity"])
def test_rejects_immutable_tamper_even_with_rehashed_bad_tensor(fixture, tmp_path, change):
    root = tmp_path / "pixels"
    shutil.copytree(fixture["root"], root)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    entry = manifest["shards"][0]
    payload_path = root / entry["path"]
    if change == "sha":
        with payload_path.open("ab") as stream:
            stream.write(b"changed")
    elif change in ("index", "nonfinite"):
        payload = load_file(str(payload_path))
        if change == "index":
            payload["frame_index"][1] = 0
        else:
            payload["pixel_frame"][0, 0, 0, 0] = float("nan")
        save_file(payload, str(payload_path))
        entry["sha256"] = file_sha256(payload_path)
    elif change == "untracked":
        (root / "interrupted.tmp").write_text("untracked")
    else:
        manifest["identity"]["source_code_sha256"]["src/fastwam/cache/pixels.py"] = "f" * 64
        manifest["identity_sha256"] = cache_identity_sha256(manifest["identity"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(CacheIntegrityError):
        validate_pixel_cache(root, cache_identity_sha256(fixture["identity"]))


def test_rejects_source_rows_length_and_changed_original_recipe(fixture):
    for rows in (fixture["rows"][:-1], fixture["rows"] + [fixture["rows"][-1]]):
        with pytest.raises(CacheIntegrityError, match="index inventory"):
            build_deduplicated_source(fixture["raw"], fixture["root"], rows, test_fixture=True)
    raw = copy.deepcopy(fixture["raw"])
    raw.processor.action_state_merger.state_target_dim = 7
    with pytest.raises(CacheIntegrityError, match="exact original"):
        validate_raw_config(raw)


def test_rejects_non_dictionary_pixel_source_hashes():
    with pytest.raises(CacheIntegrityError, match="provenance"):
        make_pixel_identity(data_conversion_sha256="a" * 64, source_code_sha256=["b" * 64],
                            runtime_versions={"torch": "TEST-only"}, episode_lengths=[2], preprocessing={}, test_fixture=True)


def test_exhaustive_independent_stock_operator_proof_and_receipt_binding(fixture, tmp_path, monkeypatch):
    def no_cuda(*args, **kwargs):
        raise AssertionError("Exhaustive pixel CPU proof must not query CUDA")
    monkeypatch.setattr(torch.cuda, "is_available", no_cuda)
    monkeypatch.setattr(torch.cuda, "get_device_properties", no_cuda)
    receipt_path = tmp_path / "proof.json"
    receipt = audit_pixel_cache(fixture["raw"], fixture["root"], receipt_path, test_fixture=True)
    digest = cache_identity_sha256(fixture["identity"])
    assert receipt["unique_frames_checked"] == receipt["anchors_checked"] == 41
    assert receipt["hardware_vae_parity_checked"] is receipt["cuda_queried"] is False
    assert validate_pixel_audit_receipt(receipt_path, fixture["root"], digest) == receipt
    receipt["anchors_checked"] -= 1
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(CacheIntegrityError, match="missing/incomplete"):
        validate_pixel_audit_receipt(receipt_path, fixture["root"], digest)
    with pytest.raises(CacheIntegrityError, match="outside"):
        audit_pixel_cache(fixture["raw"], fixture["root"], fixture["root"] / "proof.json", test_fixture=True)


def test_pixel_writer_lock_blocks_concurrent_resume(fixture, tmp_path):
    root = tmp_path / "locked"
    lock = tmp_path / "locked_pixel_writer.lock"
    lock.write_text("existing writer")
    with pytest.raises(CacheIntegrityError, match="lock already exists"):
        write_pixel_cache(root, fixture["identity"], fixture["deduplicated"].pixels.__getitem__)
    assert not root.exists() and lock.read_text() == "existing writer"


def test_audit_rejects_finite_wrong_pixels_with_valid_updated_storage_hash(fixture, tmp_path):
    root = tmp_path / "rehashed-wrong-pixels"
    shutil.copytree(fixture["root"], root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["shards"][0]
    path = root / entry["path"]
    payload = load_file(str(path))
    payload["pixel_frame"][0, 1, 2, 3] += .001
    save_file(payload, str(path))
    entry["sha256"] = file_sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    # Storage consistency succeeds; independently recomputed stock bits must fail.
    validate_pixel_cache(root, cache_identity_sha256(fixture["identity"]))
    with pytest.raises(CacheIntegrityError, match="unique recorded frame.*mismatch"):
        audit_pixel_cache(fixture["raw"], root, tmp_path / "bad-proof.json", test_fixture=True)
    assert not (tmp_path / "bad-proof.json").exists()
