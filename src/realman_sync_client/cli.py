"""CLI defaults to mock hardware; real connection requires explicit execution."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import signal

from .client import ClientOptions, SynchronousClient, validate_metadata
from .http import InferenceHTTPClient
from .mock import MockHardware


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--server", default="http://127.0.0.1:8766")
    result.add_argument("--hardware", choices=("mock", "realman"), default="mock")
    result.add_argument("--hardware-config", type=Path)
    result.add_argument("--execute", action="store_true", help="explicitly permit real hardware connection and action execution")
    result.add_argument("--enable-gripper", action="store_true", help="enable the CTAG2F120 before sensing (requires --execute)")
    result.add_argument("--cycles", type=int, default=1, help="cycles to run; 0 runs continuously (default: 1)")
    result.add_argument("--execute-steps", type=int, default=32, help="execute this many rows of each 32-row chunk")
    result.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds; no retries")
    result.add_argument("--max-result-age", type=float, default=120.0, help="reject result older than this local observation age")
    result.add_argument("--sensor-timeout", type=float, default=10.0)
    result.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    return result


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.hardware == "realman" and (not args.execute or args.hardware_config is None):
        arguments.error("realman requires --hardware-config FILE and --execute")
    if args.enable_gripper and (args.hardware != "realman" or not args.execute):
        arguments.error("--enable-gripper requires --hardware realman --execute")
    if args.cycles < 0:
        arguments.error("--cycles must be >= 0")
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    hardware = None
    client = None
    previous_signals = {}
    try:
        transport = InferenceHTTPClient(args.server, timeout_s=args.timeout)
        metadata = validate_metadata(transport.health())
        options = ClientOptions(args.execute_steps, args.max_result_age, args.sensor_timeout)
        if args.hardware == "mock":
            expected = metadata.get("force_preprocessing")
            force_id = metadata.get("force_preprocessing_id") or (
                expected.get("id") if isinstance(expected, dict) else None)
            hardware = MockHardware(width=metadata["width"], height=metadata["height"],
                                    camera_names=metadata["camera_names"], force_preprocessing_id=force_id,
                                    force_preprocessing=expected)
        else:
            config = json.loads(args.hardware_config.read_text())
            if config.get("action_space") != metadata["action_space"]:
                raise ValueError("hardware config action_space must match server joint8")
            if config.get("camera_width") != metadata["width"] or config.get("camera_height") != metadata["height"]:
                raise ValueError("hardware camera resolution must match server metadata")
            if abs(float(config.get("command_period_s", 1 / 30)) - 1 / metadata["control_hz"]) > 1e-9:
                raise ValueError("hardware command period differs from trained control rate")
            from .hardware import RealManHardwareAdapter
            hardware = RealManHardwareAdapter(args.hardware_config)
            if args.enable_gripper:
                hardware.control_gripper("enable")
        client = SynchronousClient(transport, hardware, metadata, options=options)
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_signals[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda *_: client.stop_event.set())
        completed = client.run(cycles=args.cycles)
        print(json.dumps(dict(completed_cycles=completed, hardware=args.hardware, protocol="realman-sync-v1")))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        logging.error("client stopped: %s", exc)
        return 1
    finally:
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
        if hardware is not None and client is None:
            try:
                hardware.slow_stop()
            finally:
                hardware.close()
