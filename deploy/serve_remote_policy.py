#!/usr/bin/env python3
"""Serve one warmed WAM policy over an existing SSH/启智 port-forward channel.

Defaults to loopback. This process does not import/open any robot or camera.
Set REALMAN_REMOTE_TOKEN to a random secret (at least 16 characters) on both ends.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "deploy"), str(ROOT / "src")]

from realman_deploy.model_adapter import CosmosJointPolicy
from realman_deploy.remote_transport import RemotePolicyService, checkpoint_identity, make_server
from realman_deploy.sampling import updated_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--sampling-profile", choices=("4416", "161616"), default="4416")
    parser.add_argument("--token-env", default="REALMAN_REMOTE_TOKEN")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--allow-optimization-fallback", action="store_true",
                        help="Allow readiness when full sampler compile+graph is unavailable")
    parser.add_argument("--ready-file", help="Optional JSON readiness metadata; excludes secret")
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env, "")
    if len(token) < 16:
        parser.error(f"set {args.token_env} to a random secret with at least 16 characters")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path = Path(args.config).expanduser().resolve()
    config = updated_config(json.loads(path.read_text()), args.sampling_profile, "synchronous")
    config["inference_device"] = "local"
    config["inference_optimization"] = {**config.get("inference_optimization", {}), "mode": "compile_cuda_graph"}
    if not 1 <= args.cpu_threads <= 256:
        parser.error("cpu-threads must be in 1..256")
    import torch
    torch.set_num_threads(args.cpu_threads)
    started = time.monotonic()
    policy = CosmosJointPolicy(args.checkpoint, config, config_path=path)
    server = None
    try:
        optimization = policy.diagnostics.get("optimization", {})
        if not args.allow_optimization_fallback and (
                optimization.get("effective_mode") != "compile_cuda_graph" or
                optimization.get("graph_scope") != "sampler" or optimization.get("fallback_reason")):
            raise RuntimeError("Remote sampler compile/CUDA Graph is not fully active: " +
                               json.dumps(optimization, ensure_ascii=False))
        service = RemotePolicyService(policy, config, checkpoint_sha256=checkpoint_identity(args.checkpoint))
        service.metadata["initialization_s"] = time.monotonic() - started
        service.metadata["cpu_threads"] = args.cpu_threads
        server = make_server(service, host=args.host, port=args.port, token=token)
        if args.ready_file:
            target = Path(args.ready_file).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_text(json.dumps(service.metadata, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(target)
        print(f"READY: WAM {args.sampling_profile} at {args.host}:{server.server_port}; "
              f"initialization {service.metadata['initialization_s']:.1f}s", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        policy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
