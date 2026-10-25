"""Local catalog installation keeps the exported preprocessing binding."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from realman_deploy.checkpoints import CheckpointCatalog, CheckpointStore


def fixture_store(tmp_path, *, with_receipt=True, file_uri=False):
    source = tmp_path / "export" / "step_001000.pt"
    source.parent.mkdir()
    source.write_bytes(b"a small verified checkpoint fixture")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    receipt = {"schema": "realman_checkpoint_preprocessing_v1",
               "checkpoint_sha256": digest, "normalization_sha256": "a" * 64,
               "force_preprocessing_id": "b" * 64}
    receipt_bytes = (json.dumps(receipt, indent=2) + "\n").encode()
    if with_receipt:
        Path(str(source) + ".force.json").write_bytes(receipt_bytes)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"checkpoints": [{
        "id": "pluginsertion71_step001000", "label": "PlugInsertion71",
        "source": source.as_uri() if file_uri else str(source),
        "sha256": digest, "size_bytes": source.stat().st_size}]}))
    store = CheckpointStore(CheckpointCatalog(catalog), tmp_path / "installed")
    return store, source, receipt_bytes


@pytest.mark.parametrize("file_uri", [False, True])
def test_local_install_preserves_receipt_with_catalog_name(tmp_path, file_uri):
    store, source, receipt = fixture_store(tmp_path, file_uri=file_uri)
    installed = store.download("pluginsertion71_step001000")
    assert installed.name == "pluginsertion71_step001000.pt"
    assert installed.read_bytes() == source.read_bytes()
    assert Path(str(installed) + ".force.json").read_bytes() == receipt
    assert not list(store.root.glob(".*"))


def test_existing_verified_checkpoint_gets_missing_receipt_without_recopy(tmp_path):
    store, source, receipt = fixture_store(tmp_path)
    target = store.path_for("pluginsertion71_step001000")
    target.write_bytes(source.read_bytes())
    before = target.stat()
    assert store.download("pluginsertion71_step001000") == target
    assert target.stat().st_ino == before.st_ino
    assert target.stat().st_mtime_ns == before.st_mtime_ns
    assert Path(str(target) + ".force.json").read_bytes() == receipt


@pytest.mark.parametrize("already_installed", [False, True])
def test_receipt_bound_to_wrong_checkpoint_is_rejected(tmp_path, already_installed):
    store, source, raw = fixture_store(tmp_path)
    target = store.path_for("pluginsertion71_step001000")
    if already_installed:
        target.write_bytes(source.read_bytes())
    receipt = json.loads(raw)
    receipt["checkpoint_sha256"] = "0" * 64
    Path(str(source) + ".force.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="differs from catalog SHA256"):
        store.download("pluginsertion71_step001000")
    assert target.exists() == already_installed
    assert not Path(str(target) + ".force.json").exists()


def test_conflicting_installed_receipt_is_not_overwritten(tmp_path):
    store, source, raw = fixture_store(tmp_path)
    target = store.path_for("pluginsertion71_step001000")
    target.write_bytes(source.read_bytes())
    receipt = json.loads(raw)
    receipt["normalization_sha256"] = "c" * 64
    destination = Path(str(target) + ".force.json")
    destination.write_text(json.dumps(receipt))
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        store.download("pluginsertion71_step001000")
    assert destination.read_bytes() == before


def test_matching_installed_receipt_is_idempotent(tmp_path):
    store, _, raw = fixture_store(tmp_path)
    target = store.download("pluginsertion71_step001000")
    destination = Path(str(target) + ".force.json")
    before = destination.stat()
    assert store.download("pluginsertion71_step001000") == target
    assert destination.read_bytes() == raw
    assert destination.stat().st_mtime_ns == before.st_mtime_ns


def test_legacy_local_checkpoint_without_receipt_still_installs(tmp_path):
    store, source, _ = fixture_store(tmp_path, with_receipt=False)
    installed = store.download("pluginsertion71_step001000")
    assert installed.read_bytes() == source.read_bytes()
    assert not Path(str(installed) + ".force.json").exists()


def test_corrupt_weight_never_installs_receipt(tmp_path):
    store, source, _ = fixture_store(tmp_path)
    source.write_bytes(b"z" * source.stat().st_size)
    with pytest.raises(ValueError, match="checkpoint SHA256 differs"):
        store.download("pluginsertion71_step001000")
    assert not store.path_for("pluginsertion71_step001000").exists()
    assert not list(store.root.glob("*.force.json"))


@pytest.mark.parametrize("key,value", [("schema", "unknown"),
    ("normalization_sha256", "not-a-hash"), ("force_preprocessing_id", None)])
def test_malformed_source_receipt_is_rejected(tmp_path, key, value):
    store, source, raw = fixture_store(tmp_path)
    receipt = json.loads(raw)
    receipt[key] = value
    Path(str(source) + ".force.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="preprocessing binding"):
        store.download("pluginsertion71_step001000")
    assert not store.path_for("pluginsertion71_step001000").exists()
