"""Linux-only process ownership for separate-session Torch elastic workers.

Stdlib only; importing is inert. A per-phase ProcessOwnership must enable its
subreaper BEFORE the sole authorized Popen launch, register that actual Popen,
observe while it runs, cleanup under the caller's existing monotonic deadline,
then restore the previous subreaper setting. No preexisting direct children are
allowed. The caller must create no other children while the tracker is enabled.

Linux PR_SET_CHILD_SUBREAPER reparents orphaned descendants here even if a short
lived intermediate ancestor was never sampled. Combined with an initially
empty, exclusively managed child tree, this proves adoption ownership without
guessing PGIDs. PID/start_ticks binds identities; PGID/session/PPID may change
legitimately during setsid or reparenting and are updated from actual /proc.

Every signal uses a mandatory Linux pidfd opened before an exact start_ticks
recheck; there is no os.kill/killpg fallback. A pidfd pins the process identity
across concurrent PID reuse. PGIDs are evidence only and never signal targets.
Ownership errors always prevent verified closure.
Registered Popen leaders are reaped only via their real poll(); adopted direct
children use waitpid(WNOHANG). No waitpid(-1) or unowned child reap is used.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SCHEMA = "force30-roundtrip-process-ownership-v1"
PR_SET_CHILD_SUBREAPER = 36
PR_GET_CHILD_SUBREAPER = 37


class OwnershipError(RuntimeError):
    pass


def read_proc(pid, *, _confirm_terminal=True):
    """Read Linux process identity, or None after disappearance; no signalling."""
    if type(pid) is not int or pid <= 0:
        raise OwnershipError("Process PID must be a positive integer")
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    end = raw.rfind(")")
    if end < 0:
        raise OwnershipError("Malformed Linux process stat")
    fields = raw[end + 2:].split()
    if len(fields) < 20:
        raise OwnershipError("Truncated Linux process stat")
    result = {"pid": int(raw.split(" ", 1)[0]), "state": fields[0],
              "ppid": int(fields[1]), "pgid": int(fields[2]), "session": int(fields[3]),
              "start_ticks": int(fields[19])}
    # Unrelated kernel threads legitimately have PGID/session 0. Actual owned
    # launchers are separately required to establish positive isolated IDs.
    if result["pid"] != pid or result["pgid"] < 0 or result["session"] < 0 or result["start_ticks"] < 0:
        # Actual target evidence includes both X and Z with PGID/session -1
        # during exit. A Z row is not by itself proof of disappearance: only
        # skip after a second read proves the PID is gone. A valid later-start
        # identity also proves the old process is gone, and must be discovered
        # independently. Persistent invalid rows fail closed; zombies are tracked
        # for reaping. Never use invalid group/session values as identities.
        if (_confirm_terminal and result["pid"] == pid and result["state"] in {"X", "Z"}
                and result["start_ticks"] >= 0):
            replacement = read_proc(pid, _confirm_terminal=False)
            if replacement is None:
                return None
            if replacement["start_ticks"] > result["start_ticks"]:
                return replacement
        raise OwnershipError("Invalid Linux process identity: " + json.dumps(result, sort_keys=True))
    return result


def scan_processes():
    result = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                row = read_proc(int(path.name))
            except (PermissionError, ProcessLookupError, FileNotFoundError):
                continue
            if row is not None:
                result[row["pid"]] = row
    return result


def _prctl():
    if not sys.platform.startswith("linux"):
        raise OwnershipError("Linux subreaper and /proc are required")
    value = ctypes.CDLL(None, use_errno=True)
    value.prctl.restype = ctypes.c_int
    return value


def get_subreaper():
    result = ctypes.c_int()
    if _prctl().prctl(PR_GET_CHILD_SUBREAPER, ctypes.byref(result), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER failed")
    return result.value


def set_subreaper(enabled):
    if type(enabled) is not int or enabled not in (0, 1):
        raise OwnershipError("Subreaper setting must be integer 0 or 1")
    if _prctl().prctl(PR_SET_CHILD_SUBREAPER, enabled, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")


def _identity(row):
    return {"pid": row["pid"], "start_ticks": row["start_ticks"]}


class ProcessOwnership:
    """One exclusive launcher tree, including rank workers with their own PGIDs."""

    def __init__(self):
        row = read_proc(os.getpid())
        if row is None:
            raise OwnershipError("Caller process identity disappeared")
        self.owner = {key: row[key] for key in ("pid", "pgid", "session", "start_ticks")}
        self.phase = None
        self.enabled = False
        self.prior_enabled = None
        self.launcher = None
        self.registration_complete = False
        self.records = {}
        self.retired_records = {}
        self.current = {}
        self.signals = []
        self.reaped = []
        self.errors = []
        self.closed = False
        self.cleanup_finished = None
        self._latest_scan = {}

    def _retire_replaced(self, row, replacement):
        """A later start at the same PID proves the old identity has exited.

        Retain its full ancestry as evidence, but never transfer that ownership
        to the replacement. observe() must independently discover any new child.
        """
        if replacement["pid"] != row["pid"] or replacement["start_ticks"] <= row["start_ticks"]:
            self._error("Invalid replacement process identity: " + str(row["pid"]))
            raise OwnershipError(self.errors[-1])
        known = self.records.get(row["pid"])
        identity_record = known if known is not None and known["start_ticks"] == row["start_ticks"] else row
        if (self.launcher is not None and identity_record.get("discovered_by") == "registered_launcher"
                and self.launcher.returncode is None):
            self._error("Unreaped launcher identity changed")
            raise OwnershipError(self.errors[-1])
        if known is not None and known["start_ticks"] == row["start_ticks"]:
            self.retired_records[(row["pid"], row["start_ticks"])] = {
                **known, "retired_reason": "later_start_at_same_pid",
                "replacement_identity": _identity(replacement)}
            del self.records[row["pid"]]
        current = self.current.get(row["pid"])
        if current is not None and current["start_ticks"] == row["start_ticks"]:
            del self.current[row["pid"]]

    def _error(self, message):
        if message not in self.errors:
            self.errors.append(message)

    def enable_subreaper(self):
        if self.enabled:
            raise OwnershipError("Subreaper is already enabled by this tracker")
        existing = [row for row in scan_processes().values() if row["ppid"] == self.owner["pid"]]
        if existing:
            raise OwnershipError("No preexisting direct children are allowed")
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise OwnershipError("Linux pidfd_open and pidfd_send_signal are required; no unsafe fallback")
        # Verify kernel support before any child is launched. Do not send a
        # signal during capability detection (not even signal 0).
        try:
            descriptor = os.pidfd_open(self.owner["pid"], 0)
        except OSError as error:
            raise OwnershipError("Kernel pidfd support is required before launch") from error
        os.close(descriptor)
        self.prior_enabled = get_subreaper()
        # Own the setting before the syscall: an interrupt immediately after
        # prctl succeeds must still leave a restorable tracker for the caller.
        self.enabled = True
        try:
            set_subreaper(1)
        except BaseException:
            set_subreaper(self.prior_enabled)
            self.enabled = False  # Retain ownership if rollback itself is interrupted.
            raise
        return self.prior_enabled

    def attach_launcher(self, process, phase="save"):
        """Minimal immediate Popen ownership: no /proc, scans, poll or I/O.

        Call while launch signals are briefly deferred, immediately after Popen
        returns. Restore the active alarm before heavy register_launcher().
        """
        if not self.enabled or self.launcher is not None:
            raise OwnershipError("Enable a fresh per-phase tracker before registering one launcher")
        if not isinstance(process, subprocess.Popen) or type(process.pid) is not int or process.pid <= 1:
            raise OwnershipError("Register the actual owned subprocess.Popen")
        if type(phase) is not str or not phase:
            raise OwnershipError("Phase must be a nonempty string")
        self.launcher, self.phase = process, phase

    def register_launcher(self, process, phase="save"):
        if self.launcher is None:
            self.attach_launcher(process, phase)
        elif self.launcher is not process or self.phase != phase or self.registration_complete:
            raise OwnershipError("Register exactly the attached launcher once")
        row = read_proc(process.pid)
        if row is None or row["ppid"] != self.owner["pid"]:
            self._error("Launcher identity is unavailable or is not our direct child")
            raise OwnershipError(self.errors[-1])
        if row["pid"] != row["pgid"] or row["pid"] != row["session"] or row["pgid"] == self.owner["pgid"]:
            self._error("Launcher must create a separate process group and session")
            raise OwnershipError(self.errors[-1])
        self.records[row["pid"]] = {**row, "initial_ppid": row["ppid"],
                                     "initial_pgid": row["pgid"], "initial_session": row["session"],
                                     "ancestry": [_identity(self.owner)],
                                     "discovered_by": "registered_launcher", "adopted": False}
        self.current[row["pid"]] = row
        self.registration_complete = True
        self.observe()
        return dict(self.records[row["pid"]])

    def observe(self):
        """Discover actual identities/ancestry. Return True on record-set changes."""
        if not self.enabled:
            raise OwnershipError("Subreaper must be enabled before observing")
        if self.launcher is not None:
            self.launcher.poll()  # actual registered Popen owns its wait status
        scan = scan_processes()
        self._latest_scan = scan
        actual_owner = scan.get(self.owner["pid"])
        if actual_owner is None or actual_owner["start_ticks"] != self.owner["start_ticks"]:
            self._error("Caller PID identity changed")
            raise OwnershipError(self.errors[-1])
        changed = False
        current = {}
        for pid, known in list(self.records.items()):
            row = scan.get(pid)
            if row is not None and row["start_ticks"] != known["start_ticks"]:
                self._retire_replaced(known, row)
                changed = True
                continue
            if row is not None:
                changed = changed or any(known[key] != row[key] for key in ("ppid", "pgid", "session"))
                if row["ppid"] == self.owner["pid"] and known["discovered_by"] != "registered_launcher":
                    changed = changed or not known["adopted"]
                    known["adopted"] = True
                known.update(row)
                current[pid] = row
        # Repeated passes discover deep descendants even when /proc iteration
        # listed the child before its parent. No ancestry is inferred from PGID.
        progress = True
        while progress:
            progress = False
            for pid, row in scan.items():
                if pid == self.owner["pid"] or pid in self.records:
                    continue
                parent = current.get(row["ppid"])
                adopted = row["ppid"] == self.owner["pid"] and self.launcher is not None
                if parent is None and not adopted:
                    continue
                if parent is not None and parent["start_ticks"] > row["start_ticks"]:
                    continue  # inconsistent non-atomic /proc sample; retry later
                if parent is not None:
                    actual_parent = read_proc(parent["pid"])
                    if actual_parent is None or actual_parent["start_ticks"] != parent["start_ticks"]:
                        continue  # never inherit ownership from a reused parent PID
                if adopted and row["start_ticks"] < self.owner["start_ticks"]:
                    continue
                if parent is not None:
                    chain = [*self.records[parent["pid"]]["ancestry"], _identity(parent)]
                    reason = "observed_descendant"
                else:
                    chain, reason = [_identity(self.owner)], "subreaper_adoption"
                self.records[pid] = {**row, "initial_ppid": row["ppid"],
                                     "initial_pgid": row["pgid"], "initial_session": row["session"],
                                     "ancestry": chain, "discovered_by": reason, "adopted": adopted}
                current[pid] = row
                changed = progress = True
        self.current = current
        return changed

    def _signal_pid(self, row, signum, reason):
        if row["pid"] <= 1 or row["pid"] == self.owner["pid"]:
            self._error("Refusing caller/system PID signal")
            return
        try:
            descriptor = os.pidfd_open(row["pid"], 0)
        except ProcessLookupError:
            return
        except OSError as error:
            self._error("Cannot pin owned process for signal: " + str(error))
            return
        try:
            actual = read_proc(row["pid"])
            if actual is None:
                return
            if actual["start_ticks"] != row["start_ticks"]:
                self._retire_replaced(row, actual)
                return
            try:
                signal.pidfd_send_signal(descriptor, signum, None, 0)
            except ProcessLookupError:
                return
            except OSError as error:
                self._error("Pinned owned process signal failed: " + str(error))
                return
            self.signals.append({"method": "pidfd", "pid": actual["pid"],
                                 "start_ticks": actual["start_ticks"], "pgid": actual["pgid"],
                                 "signal": signum, "reason": reason})
        finally:
            os.close(descriptor)

    def _reap_adopted(self):
        for row in list(self.current.values()):
            if row["ppid"] != self.owner["pid"] or (self.launcher is not None and row["pid"] == self.launcher.pid):
                continue
            actual = read_proc(row["pid"])
            if actual is None:
                continue
            if actual["start_ticks"] != row["start_ticks"]:
                self._retire_replaced(row, actual)
                continue
            try:
                pid, status = os.waitpid(row["pid"], os.WNOHANG)
            except ChildProcessError:
                continue
            if pid:
                self.reaped.append({**_identity(row), "wait_status": status,
                                    "returncode": os.waitstatus_to_exitcode(status)})

    def cleanup(self, deadline_monotonic, *, term_grace_seconds=2.0, kill_reserve_seconds=1.0):
        """TERM -> KILL all owned sessions, reap under one existing deadline."""
        for value in (deadline_monotonic, term_grace_seconds, kill_reserve_seconds):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise OwnershipError("Cleanup bounds must be finite numbers")
        if term_grace_seconds < 0 or kill_reserve_seconds <= 0 or not self.enabled:
            raise OwnershipError("Enabled tracker and nonnegative grace/positive KILL reserve required")
        began = time.monotonic()
        if self.launcher is not None and not self.registration_complete:
            self._error("Launcher registration did not complete")
        kill_at = min(began + term_grace_seconds, max(began, deadline_monotonic - kill_reserve_seconds))
        signalled = set()
        while True:
            try:
                self.observe()
            except OwnershipError as error:
                self._error(str(error))
            self._reap_adopted()
            try:
                self.observe()
            except OwnershipError as error:
                self._error(str(error))
            if not self.current:
                break
            now = time.monotonic()
            signum = signal.SIGKILL if now >= kill_at else signal.SIGTERM
            for row in list(self.current.values()):
                signature = (row["pid"], row["start_ticks"], signum)
                if row["state"] != "Z" and signature not in signalled:
                    self._signal_pid(row, signum, "bounded owned-tree cleanup")
                    signalled.add(signature)
            if now >= deadline_monotonic:
                self._error("Owned cleanup exceeded the supplied monotonic deadline")
                break
            time.sleep(min(.02, max(0.0, deadline_monotonic - time.monotonic())))
        # Recheck once after the final signals/reaps without extending the bound.
        try:
            self.observe()
        except OwnershipError as error:
            self._error(str(error))
        self._reap_adopted()
        try:
            self.observe()
        except OwnershipError as error:
            self._error(str(error))
        unknown_children = [row for row in self._latest_scan.values()
                            if row["ppid"] == self.owner["pid"] and row["pid"] not in self.current]
        if unknown_children:
            self._error("Unproven direct children remain after cleanup")
        self.cleanup_finished = time.monotonic()
        if self.cleanup_finished > deadline_monotonic:
            self._error("Owned cleanup exceeded the supplied monotonic deadline")
        self.closed = not self.current and not unknown_children and not self.errors and self.cleanup_finished <= deadline_monotonic
        return self.evidence()

    def evidence(self):
        records = [*self.retired_records.values(), *self.records.values()]
        value = {"schema": SCHEMA, "owner": dict(self.owner), "phase": self.phase,
                 "pidfd_signalling_required": True,
                 "subreaper_enabled": self.enabled, "prior_subreaper_enabled": self.prior_enabled,
                 "tracked_processes": [dict(row) for row in sorted(records, key=lambda row: (row["pid"], row["start_ticks"]))],
                 "retired_identity_count": len(self.retired_records),
                 "owned_pgids": sorted(set(row["pgid"] for row in records)),
                 "signals": list(self.signals), "reaped": list(self.reaped),
                 "live_after_cleanup": [dict(row) for row in sorted(self.current.values(), key=lambda row: row["pid"])],
                 "closure_verified": self.closed,
                 "launcher_returncode": None if self.launcher is None else self.launcher.returncode,
                 "launcher_registration_complete": self.registration_complete,
                 "cleanup_finished_monotonic": self.cleanup_finished, "errors": list(self.errors)}
        return json.loads(json.dumps(value, allow_nan=False))

    def restore_subreaper(self):
        if not self.enabled:
            return
        try:
            self.observe()
        except OwnershipError:
            pass
        direct = [row for row in scan_processes().values() if row["ppid"] == self.owner["pid"]]
        if self.current or direct:
            raise OwnershipError("Cannot restore subreaper while owned/unproven children remain")
        set_subreaper(self.prior_enabled)
        self.enabled = False
