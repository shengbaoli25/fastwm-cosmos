"""Offline comparison of gripper trace windows; standard library, no device access.

Example (all times are Windows perf_counter seconds, not Unix timestamps)::

    python compare_gripper_pacing.py gripper-transactions.jsonl \
        --window baseline 42000 42120 --window paced 42300 42420 \
        --bucket-seconds 10 --output pacing-comparison.json

The base JSONL and numeric rotations are included automatically. Events are
assigned by attempt start in [start, end); completed events crossing the end are
counted and identified. Fences are internal operations, not separately logged
new-request attempts. Counter resets across transport instances are irrelevant.
All windows must be from the same host boot/perf_counter clock.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys


RECOVERY_STAGES = {"pending_receive", "sync_send", "sync_receive", "recovery_expired"}


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def distribution(values):
    """Nearest-rank percentiles expose rare slow attempts in short windows."""
    values = sorted(float(value) for value in values if finite_number(value))
    if not values:
        return {"n": 0, "min": None, "mean": None, "p50": None,
                "p95": None, "p99": None, "max": None}
    at = lambda q: round(values[max(0, math.ceil(q * len(values)) - 1)], 4)
    return {"n": len(values), "min": round(values[0], 4),
            "mean": round(statistics.fmean(values), 4),
            "p50": at(.5), "p95": at(.95), "p99": at(.99),
            "max": round(values[-1], 4)}


def discover_files(inputs):
    files = set()
    for argument in inputs:
        requested = Path(argument).expanduser().resolve()
        match = re.match(r"^(.*\.jsonl)(?:\.\d+)?$", requested.name)
        base = requested.with_name(match.group(1)) if match else requested
        candidates = [base]
        candidates.extend(path for path in base.parent.glob(base.name + ".*")
                          if path.name[len(base.name) + 1:].isdigit())
        existing = [path for path in candidates if path.is_file()]
        if not existing:
            raise ValueError(f"No trace or rotations found for {argument}")
        files.update(existing)
    return sorted(files, key=str)


def valid_event(row):
    if not isinstance(row, dict) or not isinstance(row.get("request"), dict):
        return False
    if not all(finite_number(row.get(key)) for key in ("started", "completed")):
        return False
    if row["completed"] < row["started"]:
        return False
    if "request_started" in row and (not finite_number(row["request_started"])
            or not row["started"] <= row["request_started"] <= row["completed"]):
        return False
    return isinstance(row["request"].get("command"), str)


def load_events(inputs):
    """Snapshot each file once; never interpret a partial final line as failure."""
    unique, metadata = {}, []
    for path in discover_files(inputs):
        counts = Counter()
        try:
            # read(size) avoids chasing a writer appending during analysis.
            with path.open("rb") as stream:
                size = os.fstat(stream.fileno()).st_size
                raw = stream.read(size)
        except FileNotFoundError:
            metadata.append({"path": str(path), "warning": "rotated away during snapshot"})
            continue
        for line in raw.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                counts["incomplete_final_lines"] += 1
                continue
            if not line.strip():
                counts["blank_lines"] += 1
                continue
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError):
                counts["invalid_json_lines"] += 1
                continue
            if not valid_event(row):
                counts["invalid_event_lines"] += 1
                continue
            counts["valid_events"] += 1
            # Do not merge distinct events merely because a process counter
            # restarted. Duplicate complete lines across rotations are copies.
            identity = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            if identity in unique:
                counts["duplicate_events"] += 1
            else:
                unique[identity] = row
        metadata.append({"path": str(path), "snapshot_bytes": len(raw), **dict(counts)})
    return sorted(unique.values(), key=lambda row: (row["started"], row["completed"])), metadata


def is_timeout(row):
    return row.get("timeout") is True


def timeout_kind(row):
    if not is_timeout(row):
        return None
    if row.get("stage") in RECOVERY_STAGES:
        return "recovery"
    if "request_started" in row:
        return "new_request"
    return "other"


def counts_and_rates(rows, duration):
    counts = {
        "transactions": len(rows),
        "successes": sum(row.get("success") is True for row in rows),
        "errors": sum("error" in row for row in rows),
        "cancellations": sum(row.get("cancelled") is True for row in rows),
        "new_request_attempts": sum("request_started" in row for row in rows),
        "recovery_attempts": sum("synchronization" in row for row in rows),
        "timeouts": sum(is_timeout(row) for row in rows),
        "new_request_timeouts": sum(timeout_kind(row) == "new_request" for row in rows),
        "recovery_timeouts": sum(timeout_kind(row) == "recovery" for row in rows),
        "other_timeouts": sum(timeout_kind(row) == "other" for row in rows),
        "read_attempts": sum("request_started" in row and row["request"]["command"].startswith("read_") for row in rows),
        "write_attempts": sum("request_started" in row and row["request"]["command"].startswith("write_") for row in rows),
    }
    def ratio(numerator, denominator, factor=1):
        return round(numerator / denominator * factor, 6) if denominator else None
    rates = {
        "transactions_per_s": ratio(counts["transactions"], duration),
        "new_requests_per_s": ratio(counts["new_request_attempts"], duration),
        "new_request_timeouts_per_min": ratio(counts["new_request_timeouts"], duration, 60),
        "recovery_timeouts_per_min": ratio(counts["recovery_timeouts"], duration, 60),
        "all_timeouts_per_1000_transactions": ratio(counts["timeouts"], counts["transactions"], 1000),
        "new_timeouts_per_1000_new_requests": ratio(counts["new_request_timeouts"], counts["new_request_attempts"], 1000),
        "errors_per_1000_transactions": ratio(counts["errors"], counts["transactions"], 1000),
    }
    return {"counts": counts, "rates": rates}


def summarize_window(events, label, start, end, bucket_seconds=10, position_address=0x060D, position_count=6):
    if not all(finite_number(value) for value in (start, end, bucket_seconds)) or end <= start or bucket_seconds <= 0:
        raise ValueError("Window requires finite start < end and positive bucket seconds")
    rows = [row for row in events if start <= row["started"] < end]
    duration = end - start
    result = {"label": label, "start_perf": start, "end_perf": end, "duration_s": duration,
              **counts_and_rates(rows, duration)}
    result["coverage"] = {
        "first_event_started": rows[0]["started"] if rows else None,
        "last_event_completed": max((row["completed"] for row in rows), default=None),
        "completions_after_window_end": sum(row["completed"] > end for row in rows),
        "events_started_before_window_still_in_progress": sum(row["started"] < start < row["completed"] for row in events),
    }
    successful = [row for row in rows if row.get("success") is True]
    request_ms = lambda row: 1000 * (row["completed"] - row["request_started"])
    attempt_ms = lambda row: 1000 * (row["completed"] - row["started"])
    result["latency_ms"] = {
        "successful_request": distribution(request_ms(row) for row in successful if "request_started" in row),
        "all_new_request_attempts": distribution(request_ms(row) for row in rows if "request_started" in row),
        "all_attempts_including_recovery": distribution(attempt_ms(row) for row in rows),
        "recovery_attempts": distribution(attempt_ms(row) for row in rows if "synchronization" in row),
        "before_new_request_dispatch": distribution(1000 * (row["request_started"] - row["started"])
                                                    for row in rows if "request_started" in row),
    }
    positions = [row for row in successful if "request_started" in row
                 and row["request"]["command"] == "read_multiple_holding_registers"
                 and row["request"].get("address") == position_address
                 and row["request"].get("num") == position_count]
    position_gaps = [1000 * (right["request_started"] - left["request_started"])
                     for left, right in zip(positions, positions[1:])]
    result["successful_position"] = {
        "count": len(positions), "per_s": round(len(positions) / duration, 6),
        "request_start_interval_ms": distribution(position_gaps),
        "intervals_over_500ms": sum(value > 500 for value in position_gaps),
        "intervals_over_1000ms": sum(value > 1000 for value in position_gaps),
    }
    # Only adjacent trace events can establish an ACK -> next request gap.
    # Exclude recovery/drain events with hidden internal requests. A cancelled
    # or failed attempt between successes must not silently disappear.
    ack_gaps, read_ack_gaps, overlaps = [], [], 0
    for left, right in zip(rows, rows[1:]):
        if right["started"] < left["completed"]:
            overlaps += 1
        if (left.get("success") is not True or "request_started" not in right
                or "synchronization" in right or right.get("discarded_before_send")):
            continue
        gap = 1000 * (right["request_started"] - left["completed"])
        if gap < 0:
            continue
        ack_gaps.append(gap)
        if left["request"]["command"].startswith("read_") and right["request"]["command"].startswith("read_"):
            read_ack_gaps.append(gap)
    result["ack_to_next_request_ms"] = distribution(ack_gaps)
    result["read_ack_to_next_read_ms"] = distribution(read_ack_gaps)
    result["read_ack_gaps_below_1ms"] = sum(value < 1 for value in read_ack_gaps)
    result["overlapping_adjacent_attempts"] = overlaps
    result["failed_stages"] = dict(Counter(row.get("stage", "unknown") for row in rows if "error" in row))
    result["error_messages"] = dict(Counter(str(row["error"]) for row in rows if "error" in row))
    groups = defaultdict(list)
    for row in rows:
        request = row["request"]
        groups[(request["command"], request.get("address"), request.get("num"))].append(row)
    result["requests"] = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        command, address, count = key
        result["requests"].append({
            "command": command, "address": address, "address_hex": hex(address) if isinstance(address, int) else None,
            "register_count": count, **counts_and_rates(group, duration),
            "successful_request_latency_ms": distribution(request_ms(row) for row in group
                                                          if row.get("success") is True and "request_started" in row),
        })
    buckets = defaultdict(list)
    for row in rows:
        buckets[int((row["started"] - start) // bucket_seconds)].append(row)
    result["time_buckets"] = []
    for index in range(math.ceil(duration / bucket_seconds)):
        left, right = start + index * bucket_seconds, min(end, start + (index + 1) * bucket_seconds)
        result["time_buckets"].append({"start_perf": left, "end_perf": right,
                                       **counts_and_rates(buckets[index], right - left)})
    return result


def self_test():
    """Synthetic replay only. Temp fixtures never contain real device settings."""
    import tempfile

    def row(start, completed, *, success=True, stage="receive", dispatched=None, **extra):
        event = {"started": start, "completed": completed,
                 "request": {"command": "read_multiple_holding_registers", "address": 0x060D, "num": 6},
                 "stage": stage, **extra}
        if success:
            event["success"] = True
        if dispatched is not None:
            event["request_started"] = dispatched
        return event

    events = [row(1, 1.06, dispatched=1),
              row(1.061, 1.131, dispatched=1.071),
              row(1.2, 1.5, success=False, dispatched=1.2, timeout=True, error="timeout"),
              row(1.6, 1.9, success=False, stage="sync_receive", timeout=True,
                  error="timeout", synchronization={"request": {}}),
              row(2, 2.2, dispatched=2.14, synchronization={"response": {}}),
              row(2.21, 2.27, dispatched=2.22),
              row(2.9, 3.2, success=False, dispatched=2.9, timeout=True, error="timeout")]
    result = summarize_window(events, "fixture", 1, 3, bucket_seconds=.75)
    assert result["counts"]["transactions"] == 7
    assert result["counts"]["new_request_timeouts"] == 2
    assert result["counts"]["recovery_timeouts"] == 1
    assert result["counts"]["new_request_attempts"] == 6
    assert result["coverage"]["completions_after_window_end"] == 1
    assert result["successful_position"]["intervals_over_1000ms"] == 1
    assert result["successful_position"]["request_start_interval_ms"]["n"] == 3
    assert result["ack_to_next_request_ms"]["n"] == 4
    assert result["ack_to_next_request_ms"]["min"] == 11
    assert result["latency_ms"]["successful_request"]["max"] == 60
    assert sum(bucket["counts"]["transactions"] for bucket in result["time_buckets"]) == 7
    assert result["time_buckets"][-1]["end_perf"] == 3
    assert result["rates"]["new_timeouts_per_1000_new_requests"] == 333.333333
    assert summarize_window(events, "empty", 10, 11)["latency_ms"]["successful_request"]["n"] == 0
    # Simulated per-instance counter reset must not erase valid trace events.
    events[0]["transactions"], events[1]["transactions"] = 999, 1
    with tempfile.TemporaryDirectory(prefix="gripper-pacing-test-") as temp:
        base = Path(temp) / "trace.jsonl"
        base.with_name(base.name + ".1").write_text(json.dumps(events[0]) + "\n", encoding="utf8")
        base.write_bytes((json.dumps(events[0]) + "\n" + json.dumps(events[1]) + "\n"
                          + "bad json\n{}\n" + json.dumps(events[2])).encode("utf8"))
        loaded, metadata = load_events([base, base.with_name(base.name + ".1")])
        assert len(loaded) == 2
        assert sum(item.get("duplicate_events", 0) for item in metadata) == 1
        assert sum(item.get("incomplete_final_lines", 0) for item in metadata) == 1
        assert sum(item.get("invalid_json_lines", 0) for item in metadata) == 1
        assert sum(item.get("invalid_event_lines", 0) for item in metadata) == 1
    print("PASS: timeout classification, rates, latency, gaps, windows, rotations, duplicate/reset counters, partial tails")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="*", help="Base JSONL or rotation paths; numeric rotations included automatically")
    parser.add_argument("--window", action="append", nargs=3, metavar=("LABEL", "START_PERF", "END_PERF"))
    parser.add_argument("--bucket-seconds", type=float, default=10)
    parser.add_argument("--position-address", type=lambda value: int(value, 0), default=0x060D)
    parser.add_argument("--position-count", type=int, default=6)
    parser.add_argument("--output", type=Path, help="Create a new JSON report; refuse overwriting an existing file")
    parser.add_argument("--self-test", action="store_true", help="Run offline synthetic checks and exit")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.traces or not args.window:
        parser.error("Supply at least one trace and --window LABEL START_PERF END_PERF")
    try:
        events, files = load_events(args.traces)
        windows = [summarize_window(events, label, float(start), float(end), args.bucket_seconds,
                                    args.position_address, args.position_count)
                   for label, start, end in args.window]
        report = {
            "schema": "gripper-pacing-comparison-v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
            "hardware_accessed": False, "files": files, "unique_completed_trace_events": len(events),
            "percentile_method": "nearest rank", "windows": windows,
            "notes": [
                "Rates use requested wall duration in the perf_counter clock, including quiet periods; compare equivalent work and observation duration.",
                "Transactions are logged API attempts. Internal fence sends are not separately counted as new requests.",
                "New-request timeouts and repeated recovery waits are distinct; use both time-normalized and request-normalized rates.",
                "Successful-position gaps use fresh returned request starts; late discarded replies and fences are excluded.",
                "ACK gaps are host receive-completion to next dispatch, not measured RS485 wire idle time; recovery internals are excluded.",
                "Rows are selected by attempt start in [start, end). Window-crossing completed rows are identified in coverage.",
                "Trace spans multiple transport instances; cumulative status counters are ignored. Do not combine different host boots.",
                "Rotation during snapshot or finite retention can omit events. Inspect files and coverage; missing trace is not proof of no traffic.",
                "Short A/B observations cannot establish that a rare timeout is solved; lower rates do not establish causality on their own.",
            ],
        }
        if args.output:
            with args.output.open("x", encoding="utf8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            print(json.dumps({"report": str(args.output.resolve()), "windows": [
                {key: window[key] for key in ("label", "duration_s", "counts", "rates", "successful_position",
                                               "ack_to_next_request_ms", "read_ack_to_next_read_ms")}
                for window in windows]}, ensure_ascii=False, indent=2, allow_nan=False))
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
