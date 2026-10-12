"""CPU-only, explicitly synthetic TEST fixtures for the ForceVLA converter.

These tiny RGB PNG/parquet fixtures are never presented as ForceVLA episodes.
They test payload and row preservation and the production safety boundaries.
"""
from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/forcevla/prepare_forcevla.py"
SPEC = importlib.util.spec_from_file_location("prepare_forcevla_libero2cam", SCRIPT)
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)

IMAGE_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
CAMERAS = {"observation.image": "cam_main", "observation.wrist_image": "cam_wrist"}


def _json(path, value):
    path.write_text(json.dumps(value) + "\n")


def _png(camera, episode, frame):
    pixels = np.zeros((12, 16, 3), dtype=np.uint8)
    pixels[:, :, camera] = 120 + frame
    pixels[episode, frame, :] = [3 + camera, 11 + frame, 21 + episode]
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return stream.getvalue()


def _write_source(root, dtype="float64"):
    """Produce two tiny TEST-only episodes with distinct numeric/image values."""
    (root / "meta").mkdir(parents=True)
    features = {
        "observation.state": {"dtype": dtype, "shape": [13],
            "names": ["current_eef_pose", "gripper_width", "f_ext_base_frame"]},
        "action": {"dtype": dtype, "shape": [7], "names": ["target_eef_pose", "target_gripper_width"]},
        **{key: {"dtype": "image", "shape": [12, 16, 3], "names": ["height", "width", "channels"]}
           for key in CAMERAS},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        **{key: {"dtype": "int64", "shape": [1], "names": None}
           for key in ("frame_index", "episode_index", "index", "task_index")},
    }
    _json(root / "meta/info.json", {"forcevla_test_fixture": True,
        "codebase_version": "v2.1", "fps": 30, "robot_type": "flexiv",
        "total_episodes": 2, "total_frames": 9, "total_tasks": 1, "features": features})
    _json(root / "meta/tasks.jsonl", {"task_index": 0, "task": "TEST-only synthetic wipe instruction."})
    numeric_type = pa.float64() if dtype == "float64" else pa.float32()
    tables, payloads = [], []
    offset = 0
    for episode, n in enumerate((4, 5)):
        # Sub-float32 differences catch accidental quantization of raw float64.
        state = (episode + np.arange(n * 13).reshape(n, 13) * 0.000000123456789 + 0.1).astype(dtype)
        action = (episode + np.arange(n * 7).reshape(n, 7) * 0.000000987654321 + 0.2).astype(dtype)
        images = {key: [{"bytes": _png(camera, episode, frame), "path": None} for frame in range(n)]
                  for camera, key in enumerate(CAMERAS)}
        table = pa.table({
            "observation.state": pa.array(state.tolist(), type=pa.list_(numeric_type, 13)),
            "action": pa.array(action.tolist(), type=pa.list_(numeric_type, 7)),
            **{key: pa.array(values, type=IMAGE_TYPE) for key, values in images.items()},
            "timestamp": pa.array(np.arange(n, dtype=np.float32) / 30, type=pa.float32()),
            "frame_index": pa.array(range(n), type=pa.int64()),
            "episode_index": pa.array([episode] * n, type=pa.int64()),
            "index": pa.array(range(offset, offset + n), type=pa.int64()),
            "task_index": pa.array([0] * n, type=pa.int64()),
        })
        path = root / f"data/chunk-000/episode_{episode:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        tables.append(table)
        payloads.append(images)
        offset += n
    return tables, payloads


def _read_info(source):
    return json.loads((source / "meta/info.json").read_text())


def _write_manifest(source, target):
    files = []
    for path in sorted(source.rglob("*")):
        if path.is_file():
            files.append({"path": f"data_lerobot/{source.name}/{path.relative_to(source).as_posix()}",
                          "size_bytes": path.stat().st_size, "sha256": converter.sha256(path)})
    _json(target, {"repo_id": converter.SOURCE_REPOSITORY, "revision": converter.SOURCE_REVISION,
                  "subset": source.name, "files": files})


def _replace_column(source, name, values, episode=1, dtype=None):
    path = source / f"data/chunk-000/episode_{episode:06d}.parquet"
    table = pq.read_table(path)
    column = pa.array(values, type=dtype or table[name].type)
    pq.write_table(table.set_column(table.column_names.index(name), name, column), path)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "TEST-only-source"
    _write_source(root)
    return root


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_preserves_exact_numeric_values_encoded_images_and_all_rows(tmp_path, dtype):
    source = tmp_path / "TEST-only-source"
    tables, images = _write_source(source, dtype)
    output = tmp_path / "output"
    result = converter.convert_dataset(source, output, test_fixture=True)
    assert result["episodes"] == 2 and result["frames"] == 9
    assert result["profile"] == "forcevla-libero2cam-v1"
    assert result["test_fixture"] is True and result["usage"] == "TEST-only synthetic fixture"
    assert result["cameras"] == CAMERAS
    assert result["camera_output_order"] == ["cam_main", "cam_wrist"]
    assert result["camera_packing"] == {"cam_main": "left", "cam_wrist": "right"}
    assert result["sampling"] == "all_original_frames"
    assert result["image_processing"] == "original_encoded_bytes"
    assert result["converter_provenance"]["source_sha256"] == converter.ORIGINAL_CONVERTER_SHA256
    info = json.loads((output / "meta/info.json").read_text())
    assert info["total_frames"] == 9 and info["total_episodes"] == 2
    assert info["features"]["action"]["dtype"] == dtype
    assert info["features"]["observation.state"]["dtype"] == dtype
    assert info["features"]["observation.wrench"]["shape"] == [6]
    assert (output / "meta/tasks.jsonl").read_bytes() == (source / "meta/tasks.jsonl").read_bytes()
    for episode, original in enumerate(tables):
        destination = pq.read_table(output / f"data/chunk-000/episode_{episode:06d}.parquet")
        state = np.asarray(original["observation.state"].to_pylist())
        np.testing.assert_array_equal(destination["observation.state"].to_pylist(), state[:, :7])
        np.testing.assert_array_equal(destination["observation.history_state"].to_pylist(), state[:, :7])
        np.testing.assert_array_equal(destination["observation.wrench"].to_pylist(), state[:, 7:])
        assert destination["action"].equals(original["action"])
        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            assert destination[key].equals(original[key])
        for key, output_name in CAMERAS.items():
            assert destination["observation.images." + output_name].to_pylist() == images[episode][key]
    assert len(set(result["camera_payload_sha256"].values())) == 2
    for entry in result["input_files"]:
        assert converter.sha256(source / entry["path"]) == entry["sha256"]
    for entry in result["output_files"]:
        assert converter.sha256(output / entry["path"]) == entry["sha256"]


def test_explicit_reversed_argument_order_still_emits_libero_order(source, tmp_path):
    result = converter.convert_dataset(source, tmp_path / "out",
        cameras=list(reversed(list(CAMERAS.items()))), test_fixture=True)
    assert list(result["cameras"].items()) == list(CAMERAS.items())
    table = pq.read_table(tmp_path / "out/data/chunk-000/episode_000000.parquet")
    assert [name for name in table.column_names if name.startswith("observation.images.")] == [
        "observation.images.cam_main", "observation.images.cam_wrist"]


def test_path_images_materialized_as_identical_original_encoded_bytes(source, tmp_path):
    table = pq.read_table(source / "data/chunk-000/episode_000001.parquet")
    values = table["observation.wrist_image"].to_pylist()
    payload = values[2]["bytes"]
    (source / "assets").mkdir()
    (source / "assets/view.png").write_bytes(payload)
    values[2] = {"bytes": None, "path": "assets/view.png"}
    _replace_column(source, "observation.wrist_image", values)
    converter.convert_dataset(source, tmp_path / "out", test_fixture=True)
    output = pq.read_table(tmp_path / "out/data/chunk-000/episode_000001.parquet")
    assert output["observation.images.cam_wrist"][2].as_py() == {"bytes": payload, "path": None}


@pytest.mark.parametrize("mapping", [
    [("observation.image", "cam_main")],
    [("observation.image", "cam_main"), ("observation.image", "cam_wrist")],
    [("observation.image", "cam_main"), ("observation.wrist_image", "cam_main")],
    [("observation.image", "cam_wrist"), ("observation.wrist_image", "cam_main")],
    [("observation.image", "cam_main"), ("observation.fake_camera", "cam_wrist")],
])
def test_rejects_missing_wrong_or_double_camera_mapping(source, tmp_path, mapping):
    with pytest.raises(ValueError, match="camera|mapping"):
        converter.convert_dataset(source, tmp_path / "out", cameras=mapping, test_fixture=True)
    assert not (tmp_path / "out").exists()


def test_missing_camera_metadata_rejected(source, tmp_path):
    info = _read_info(source)
    del info["features"]["observation.wrist_image"]
    _json(source / "meta/info.json", info)
    with pytest.raises(ValueError, match="Missing real source camera"):
        converter.convert_dataset(source, tmp_path / "out", test_fixture=True)


def test_missing_camera_parquet_column_rejected_and_cleanup(source, tmp_path):
    path = source / "data/chunk-000/episode_000001.parquet"
    pq.write_table(pq.read_table(path).drop(["observation.wrist_image"]), path)
    with pytest.raises(ValueError, match="Missing synchronized source columns"):
        converter.convert_dataset(source, tmp_path / "out", test_fixture=True)
    assert not (tmp_path / "out").exists() and not list(tmp_path.glob("out.tmp-*"))


@pytest.mark.parametrize("cell", [{"bytes": None, "path": None}, {"bytes": b"invalid", "path": None}])
def test_rejects_absent_or_invalid_camera_payload(source, tmp_path, cell):
    table = pq.read_table(source / "data/chunk-000/episode_000001.parquet")
    values = table["observation.wrist_image"].to_pylist()
    values[2] = cell
    _replace_column(source, "observation.wrist_image", values)
    with pytest.raises(ValueError, match="image"):
        converter.convert_dataset(source, tmp_path / "out", test_fixture=True)
    assert not list(tmp_path.glob("out.tmp-*"))


def test_rejects_camera_payload_wrong_shape(source, tmp_path):
    table = pq.read_table(source / "data/chunk-000/episode_000001.parquet")
    values = table["observation.wrist_image"].to_pylist()
    stream = io.BytesIO()
    Image.new("RGB", (9, 9), "red").save(stream, format="PNG")
    values[2] = {"bytes": stream.getvalue(), "path": None}
    _replace_column(source, "observation.wrist_image", values)
    with pytest.raises(ValueError, match="metadata shape"):
        converter.convert_dataset(source, tmp_path / "out", test_fixture=True)


def test_rejects_duplicated_camera_streams_with_distinct_keys(source, tmp_path):
    for episode in range(2):
        table = pq.read_table(source / f"data/chunk-000/episode_{episode:06d}.parquet")
        _replace_column(source, "observation.wrist_image", table["observation.image"].to_pylist(), episode)
    with pytest.raises(ValueError, match="duplicate complete camera payload streams"):
        converter.convert_dataset(source, tmp_path / "out", test_fixture=True)
    assert not (tmp_path / "out").exists() and not list(tmp_path.glob("out.tmp-*"))


@pytest.mark.parametrize("column,values,pattern", [
    ("timestamp", [0.0, 1 / 30, 1 / 30, 3 / 30, 4 / 30], "strictly increasing"),
    ("timestamp", [0.0, 1 / 30, 3 / 30, 4 / 30, 5 / 30], "contiguous 30 Hz"),
    ("frame_index", [0, 1, 3, 4, 5], "frame indices"),
    ("episode_index", [0, 0, 0, 0, 0], "episode_index disagree"),
    ("index", [4, 5, 7, 8, 9], "global index"),
    ("task_index", [0, 0, 1, 0, 0], "exactly one known task"),
])
def test_rejects_changed_sync_or_frame_order(source, tmp_path, column, values, pattern):
    _replace_column(source, column, values)
    with pytest.raises(ValueError, match=pattern):
        converter.convert_dataset(source, tmp_path / "out", test_fixture=True)
    assert not (tmp_path / "out").exists() and not list(tmp_path.glob("out.tmp-*"))


def test_rejects_missing_episode_and_nonfinite_numeric_values(source, tmp_path):
    path = source / "data/chunk-000/episode_000001.parquet"
    table = pq.read_table(path)
    states = table["observation.state"].to_pylist()
    states[3][8] = float("nan")
    _replace_column(source, "observation.state", states)
    with pytest.raises(ValueError, match="Nonfinite ForceVLA"):
        converter.convert_dataset(source, tmp_path / "out", test_fixture=True)
    path.unlink()
    with pytest.raises(ValueError, match="every published episode"):
        converter.convert_dataset(source, tmp_path / "out", test_fixture=True)


def test_manifest_integrity_verified_and_tampering_rejected(source, tmp_path):
    manifest = tmp_path / "manifest.json"
    _write_manifest(source, manifest)
    result = converter.convert_dataset(source, tmp_path / "out", test_fixture=True, manifest=manifest)
    assert result["source_revision_provenance"] == "verified_manifest"
    task_path = source / "meta/tasks.jsonl"
    task_path.write_text(task_path.read_text().replace("synthetic", "synthEtic"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        converter.convert_dataset(source, tmp_path / "tampered", test_fixture=True, manifest=manifest)


def test_production_requires_all50_and_test_flag_cannot_relax_real_data(source, tmp_path):
    with pytest.raises(ValueError, match="TEST-only"):
        converter.convert_dataset(source, tmp_path / "out")
    info = _read_info(source)
    del info["forcevla_test_fixture"]
    _json(source / "meta/info.json", info)
    with pytest.raises(ValueError, match="TEST-only"):
        converter.convert_dataset(source, tmp_path / "out", test_fixture=True)
    with pytest.raises(ValueError, match="all 50 episodes"):
        converter.convert_dataset(source, tmp_path / "out")


def test_rejects_existing_or_nested_output_paths(source, tmp_path):
    (tmp_path / "existing").mkdir()
    with pytest.raises(FileExistsError):
        converter.convert_dataset(source, tmp_path / "existing", test_fixture=True)
    with pytest.raises(ValueError, match="must be separate"):
        converter.convert_dataset(source, source / "out", test_fixture=True)


def test_production_validator_rejects_explicit_synthetic_fixture(source, tmp_path):
    converter.convert_dataset(source, tmp_path / "out", test_fixture=True)
    with pytest.raises(ValueError, match="test_fixture"):
        converter.validate_converted_libero2cam(tmp_path / "out")


def test_rejects_changed_scalar_dtypes(source, tmp_path):
    _replace_column(source, "frame_index", [0.0, 1.0, 2.0, 3.0, 4.0], dtype=pa.float64())
    with pytest.raises(ValueError, match="pinned source scalar dtype"):
        converter.convert_dataset(source, tmp_path / "out", test_fixture=True)


def test_cli_test_only_default_profile_and_duplicate_arguments(source, tmp_path):
    common = [sys.executable, str(SCRIPT), "--source-root", str(source), "--test-fixture"]
    result = subprocess.run([*common, "--output-root", str(tmp_path / "out")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["frames"] == 9
    duplicate = subprocess.run([*common, "--output-root", str(tmp_path / "bad"),
        "--camera", "observation.image=cam_main", "--camera", "observation.image=cam_wrist"],
        capture_output=True, text=True)
    assert duplicate.returncode != 0 and "duplicate mappings" in duplicate.stderr
    assert not (tmp_path / "bad").exists()
