"""Native Windows entry point for the five exact source snapshots in ../code-layout.json."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_config(values):
    """Validate the supported native-Windows inference combinations only.

    This deliberately avoids model imports and does not normalize or mutate
    trained model constructor inputs. Runtime samplers still validate their
    own checkpoint, manifest and actual tensor state when loading the model.
    """
    if not isinstance(values, dict):
        raise ValueError("Cosmos runtime configuration must be an object")
    profile = values.get("sampling_profile")
    if profile is not None and (not isinstance(profile, str) or profile not in {"161616", "4416", "3310"}):
        raise ValueError("Native Windows Cosmos supports sampling_profile 161616, 4416 or 3310 only")
    expected_backend = {"4416": "joint_multirate", "3310": "joint_expert_schedule"}.get(
        profile, "joint_static_cache")
    if values.get("inference_backend", "joint_static_cache") != expected_backend:
        raise ValueError(f"Sampling profile {profile or '161616'} requires {expected_backend}")
    if values.get("execution_mode", "synchronous") != "synchronous":
        raise ValueError("Native Windows Cosmos requires synchronous execution_mode")
    options = values.get("inference_optimization", {})
    if profile == "3310":
        eager = (isinstance(options, dict)
                 and not set(options) - {"compile_layers", "cuda_graph"}
                 and options.get("compile_layers") is False
                 and options.get("cuda_graph") is False)
    else:
        eager = (isinstance(options, dict) and not set(options) - {"mode"}
                 and options.get("mode", "eager") == "eager")
    if not eager:
        raise ValueError("Native Windows Cosmos supports eager inference_optimization only; compile is disabled")
    if type(values.get("joint_steps", 16)) is not int or values.get("joint_steps", 16) != 16:
        raise ValueError("Native Windows Cosmos requires joint_steps=16")
    kwargs = values.get("model_kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("Cosmos model_kwargs must be an object")
    if type(kwargs.get("num_inference_steps", 16)) is not int or kwargs.get("num_inference_steps", 16) != 16:
        raise ValueError("Trained model_kwargs.num_inference_steps must remain 16")


def env_python(path):
    return path.expanduser().resolve() / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def existing_venv():
    """Reuse this host's installed environment without copying or installing it."""
    if os.environ.get("REALMAN_SERVER_VENV"):
        return Path(os.environ["REALMAN_SERVER_VENV"]).expanduser().resolve()
    local = ROOT / ".venv-server"
    if env_python(local).is_file():
        return local
    layout = read_json(ROOT / "code-layout.json")
    cosmos = next(item for item in layout["experiments"] if item["name"] == "cosmos")
    workspace = Path(cosmos["source_path"]).parents[1] / ".venv-server"
    return workspace if env_python(workspace).is_file() else local


def runtime_environment(source):
    """Pin source imports to the copied experiment and pinned local upstream."""
    upstream = ROOT / "runtime/upstream/cosmos-predict2.5"
    paths = [source / "src", source / "deploy", source, upstream,
             upstream / "packages/cosmos-oss"]
    environment = os.environ.copy()
    environment.update(PYTHONPATH=os.pathsep.join(map(str, paths)),
                       PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                       COSMOS_TRAINING="false")
    return environment


def execute(command, dry_run=False, cwd=None, env=None):
    command = list(map(str, command))
    print(subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("install", help="Create a separate inference venv; no training install")
    install.add_argument("--python", default=sys.executable)
    install.add_argument("--venv", type=Path, default=ROOT / ".venv-server")
    install.add_argument("--dry-run", action="store_true")
    check = commands.add_parser("check", help="Inspect layout without imports, models or hardware")
    check.add_argument("--verify", action="store_true", help="Verify the copied code file SHA256 inventory")
    imports = commands.add_parser("imports", help="Check Cosmos CPU imports without loading models or hardware")
    imports.add_argument("--venv", type=Path, default=existing_venv())
    imports.add_argument("--dry-run", action="store_true")
    serve = commands.add_parser("serve", help="Start one synchronous policy with real assets")
    serve.add_argument("--experiment", required=True)
    serve.add_argument("--checkpoint-bootstrap", action="store_true",
                       help="Explicit Cosmos-only bootstrap from the complete trained state, without the base file")
    serve.add_argument("--config", type=Path, required=True)
    serve.add_argument("--checkpoint", type=Path, required=True)
    serve.add_argument("--venv", type=Path, default=existing_venv())
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--token-env", default="REALMAN_SERVER_TOKEN")
    serve.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    layout = read_json(ROOT / "code-layout.json")
    entries = layout["experiments"]
    if args.command == "check":
        report = dict(root=str(ROOT), source_snapshots=[dict(name=e["name"], commit=e["commit"],
            code_exists=(ROOT / e["name"] / "code/deploy/serve_policy.py").is_file(),
            adapter_note=e.get("adapter_note")) for e in entries], environments_copied=False,
            model_assets_checked=False, models_started=False)
        if args.verify:
            inventory = read_json(ROOT / "code-inventory.sha256.json")
            failed = []
            for name, expected in inventory.items():
                path = ROOT / name
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    failed.append(name)
            report.update(verified_files=len(inventory), mismatches=failed)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return int(bool(report.get("mismatches")) or not all(e["code_exists"] for e in report["source_snapshots"]))
    if args.command == "install":
        if os.name != "nt":
            raise ValueError("This layout installer targets native Windows")
        python = env_python(args.venv)
        execute([args.python, "-c", "import sys; assert sys.version_info >= (3, 10)"], args.dry_run)
        if not python.is_file():
            execute([args.python, "-m", "venv", args.venv], args.dry_run)
        execute([python, "-m", "pip", "install", "--upgrade", "pip"], args.dry_run)
        execute([python, "-m", "pip", "install", "--only-binary=:all:", "torch==2.7.1", "torchvision==0.22.1",
                 "--index-url", "https://download.pytorch.org/whl/cu128"], args.dry_run)
        execute([python, "-m", "pip", "install", "-r", ROOT / "runtime/requirements.server.txt"], args.dry_run)
        execute([python, "-m", "pip", "check"], args.dry_run)
        execute([python, "-X", "utf8", ROOT / "runtime/verify_cuda.py"], args.dry_run)
        return 0
    if args.command == "imports":
        source = ROOT / "cosmos/code"
        python = env_python(args.venv)
        if not python.is_file():
            raise FileNotFoundError(f"Existing inference Python is missing: {python}")
        execute([python, "-X", "utf8", ROOT / "runtime/verify_cosmos_imports.py"],
                args.dry_run, source, runtime_environment(source))
        execute([python, "-X", "utf8", source / "deploy/serve_policy.py", "--help"],
                args.dry_run, source, runtime_environment(source))
        return 0
    matches = [entry for entry in entries if entry["name"] == args.experiment]
    if len(matches) != 1:
        raise ValueError("Unknown experiment; run check for the five exact names")
    entry = matches[0]
    if args.checkpoint_bootstrap and entry["name"] != "cosmos":
        raise ValueError("--checkpoint-bootstrap is supported only for the cosmos experiment")
    backend = "cosmos_checkpoint" if args.checkpoint_bootstrap else entry["policy_backend"]
    source = ROOT / entry["name"] / "code"
    config, checkpoint = args.config.expanduser().resolve(), args.checkpoint.expanduser().resolve()
    if not config.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Both an actual runtime JSON and an actual checkpoint file are required")
    if not 1 <= args.port <= 65535:
        raise ValueError("Port must be in 1..65535")
    if entry["policy_backend"] == "cosmos":
        values = read_json(config)
        validate_config(values)
        configured_upstream = Path(values.get("model_kwargs", {}).get("upstream", ""))
        if not configured_upstream.is_absolute():
            configured_upstream = config.parent / configured_upstream
        if configured_upstream.resolve() != (ROOT / "runtime/upstream/cosmos-predict2.5").resolve():
            raise ValueError("Cosmos model_kwargs.upstream must point to infer/runtime/upstream/cosmos-predict2.5")
    python = env_python(args.venv)
    if not python.is_file():
        raise FileNotFoundError(f"Existing inference Python is missing: {python}")
    if not args.dry_run and not os.environ.get(args.token_env):
        raise ValueError(f"Set {args.token_env} for both server and client; its value is not logged")
    execute([python, "-X", "utf8", source / "deploy/serve_policy.py",
             "--backend", backend, "--config", config, "--checkpoint", checkpoint,
             "--host", args.host, "--port", args.port, "--token-env", args.token_env],
            args.dry_run, source, runtime_environment(source))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
