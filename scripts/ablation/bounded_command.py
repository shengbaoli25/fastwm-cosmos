"""Run one preparation stage with a total deadline and periodic elapsed-time logs."""
import argparse
import os
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", required=True, type=int)
    parser.add_argument("--label", required=True)
    parser.add_argument("--grace-seconds", type=int, default=10, help="Allow a controller to clean up its child processes after SIGTERM")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.seconds <= 0 or args.grace_seconds <= 0 or not command or os.name != "posix":
        parser.error("A positive deadline, a command and a Linux/POSIX worker are required")
    start = time.monotonic()
    print(f"[{args.label}] starting, total deadline {args.seconds}s", flush=True)
    process = subprocess.Popen(command, start_new_session=True)

    def stop():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=args.grace_seconds)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()

    def interrupted(signum, _frame):
        stop()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    while True:
        remaining = args.seconds - (time.monotonic() - start)
        if remaining <= 0:
            print(f"[{args.label}] deadline exceeded; terminating stage. Retained cache files can be reused on retry.", flush=True)
            stop()
            return 124
        try:
            result = process.wait(timeout=min(30, remaining))
            print(f"[{args.label}] exit={result}, elapsed={time.monotonic()-start:.0f}s", flush=True)
            return result
        except subprocess.TimeoutExpired:
            print(f"[{args.label}] still running, elapsed={time.monotonic()-start:.0f}s, remaining={max(0, args.seconds-(time.monotonic()-start)):.0f}s (heartbeat, not transfer progress)", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
