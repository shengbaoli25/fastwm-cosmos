"""Read-only checks of the manifest, local files, and runtime JSON dependencies.

No downloads, package installs, checkpoint deserialization, model imports,
hardware connections, or edits to either asset manifest or download receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from server import validate_config

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inspect_file(item, calculate_hash):
    result = {key: item.get(key) for key in ("id", "kind", "status", "target_path")}
    result["required"] = item.get("required", True)
    value = item.get("target_path")
    if not value:
        return dict(result, local_state="path_unresolved", present=False, integrity_verified=False)
    path = Path(value)
    if not path.is_file():
        return dict(result, local_state="missing", present=False, integrity_verified=False)
    size = path.stat().st_size
    result.update(local_state="present_unverified" if size else "empty_file", present=size > 0,
                  size_bytes=size, integrity_verified=False)
    if size and calculate_hash:
        actual = digest(path)
        expected = item.get("expected_sha256")
        result["sha256"] = actual
        if expected:
            result.update(integrity_verified=actual == expected,
                          local_state="hash_verified" if actual == expected else "hash_mismatch")
        else:
            result["local_state"] = "hash_computed_without_reference"
    return result


def inspect_runtime(experiment):
    configured = next(item for item in experiment["supporting_assets"] if item["kind"] == "generated_runtime_config")
    config_path = Path(configured["target_path"])
    if not config_path.is_file():
        return dict(present=False, dependencies=[], errors=["Runtime JSON is missing"])
    result = dict(present=True, dependencies=[], errors=[])
    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict):
            raise ValueError("Runtime JSON must contain an object")
        def dependency(key, value, directory=False):
            if not isinstance(value, str) or not value:
                result["errors"].append(f"Unresolved runtime path: {key}")
                return None
            if value.startswith("/inspire/"):
                result["errors"].append(f"Remote training path is not a local Windows asset: {key}")
                return None
            path = Path(value).expanduser()
            path = (path if path.is_absolute() else config_path.parent / path).resolve()
            exists = path.is_dir() if directory else path.is_file()
            result["dependencies"].append(dict(key=key, path=str(path), exists=exists,
                within_infer=path.is_relative_to(ROOT)))
            if not exists:
                result["errors"].append(f"Missing runtime dependency: {key}")
            return path if exists else None
        dependency("normalization_file", config.get("normalization_file"))
        if experiment["name"] == "cosmos":
            dependency("tokenizer", config.get("tokenizer"))
            kwargs = config.get("model_kwargs", {})
            if not isinstance(kwargs, dict):
                raise ValueError("model_kwargs must be an object")
            dependency("model_kwargs.upstream", kwargs.get("upstream"), True)
            if config.get("policy_backend") != "cosmos_checkpoint" or kwargs.get("checkpoint"):
                dependency("model_kwargs.checkpoint", kwargs.get("checkpoint"))
            cache = dependency("model_kwargs.text_cache", kwargs.get("text_cache"), True)
            if cache:
                manifest = cache / "manifest.json"
                if not manifest.is_file():
                    result["errors"].append("Reason1 text cache manifest is missing")
                else:
                    text = json.loads(manifest.read_text(encoding="utf-8-sig"))
                    entries = text.get("texts", [])
                    if not entries:
                        result["errors"].append("Reason1 manifest contains no text context entries")
                    for index, item in enumerate(entries):
                        path = (cache / item["path"]).resolve()
                        if not path.is_relative_to(cache):
                            raise ValueError("Text context path escapes its cache root")
                        dependency(f"text_context_{index}", str(path))
            validate_config(config)
        else:
            for key in ("training_config", "teacher_weights"):
                dependency(key, config.get(key))
            dependency("teacher_source", config.get("teacher_source"), True)
            if config.get("force_preprocessing_file"):
                dependency("force_preprocessing_file", config["force_preprocessing_file"])
            elif not isinstance(config.get("force_preprocessing"), dict):
                result["errors"].append("Matching force preprocessing contract is missing")
    except (ValueError, OSError, KeyError, TypeError) as error:
        result["errors"].append(str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash", action="store_true", help="Read potentially large local files to calculate SHA256")
    parser.add_argument("--json", action="store_true", help="Print complete JSON report to stdout")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "assets-manifest.json").read_text(encoding="utf-8"))
    checkpoints = [inspect_file(item, args.hash) for item in manifest["checkpoints"]]
    experiments = []
    for item in manifest["experiments"]:
        assets = [inspect_file(asset, args.hash) for asset in item["supporting_assets"]]
        code = Path(item["prepared_code_dependency"]["path"])
        experiments.append(dict(name=item["name"], training_source_commit=item["training_source_commit"],
            deployment_code_commit=item["deployment_code_commit"], adapter_status=item["adapter_status"],
            source_dependency_exists=code.is_dir(), supporting_assets=assets,
            runtime=inspect_runtime(item), model_inference_verified=item["model_inference_verified"]))
    inputs = [asset for item in experiments for asset in item["supporting_assets"] if asset["required"]]
    summary = dict(checkpoint_files_present=sum(item["present"] for item in checkpoints),
        checkpoint_files_required=len(checkpoints), supporting_files_present=sum(item["present"] for item in inputs),
        supporting_assets_required=len(inputs), unresolved_target_paths=sum(item["local_state"] == "path_unresolved" for item in inputs),
        source_dependencies_present=sum(item["source_dependency_exists"] for item in experiments),
        adapters_validated=sum(item["adapter_status"] == "validated" for item in experiments),
        full_model_inference_verified=sum(item["model_inference_verified"] for item in experiments),
        checkpoint_deserialization_performed=False, hardware_access_performed=False,
        manifest_modified=False, download_receipts_modified=False)
    ready = (all(item["present"] and item["local_state"] != "hash_mismatch" for item in checkpoints + inputs)
             and all(item["adapter_status"] == "validated" and item["model_inference_verified"] and
                     not item["runtime"]["errors"] for item in experiments))
    report = dict(summary=summary, ready_for_inference=ready, checkpoints=checkpoints, experiments=experiments,
        limitation="File presence or a calculated hash alone does not prove transfer completeness, checkpoint compatibility, model quality, or robot readiness")
    print(json.dumps(report if args.json else dict(summary=summary, ready_for_inference=ready), indent=2))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
