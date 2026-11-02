"""Windows-native, single-outstanding RealMan JSON/Modbus transport.

The controller has no transaction IDs. A timed-out read/ACK therefore leaves
the stream uncertain. A subsequent READ first retires the known late response,
then uses a different read command as a fence; no second RTU request overlaps
the unresolved original. Recovery has a two-second total deadline and a bounded
number of waits. Fence results and discarded replies are diagnostic
only; they are never returned as a fresh device measurement. Writes are sent
once, never retried. A broken or ambiguous stream requires explicit reopening.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
import math
from pathlib import Path
import select
import socket
import threading
import time


class ModbusTransportError(RuntimeError):
    """Transport lost or unsynchronized; no confirmed device response."""


class MotionCancelled(RuntimeError):
    """Intent changed before any target/enable bytes were sent."""


class _ProtocolError(RuntimeError):
    pass


def transport_interruption(error):
    while error is not None:
        if isinstance(error, ModbusTransportError):
            return True
        error = error.__cause__
    return False


class JsonLink:
    """A bounded CRLF JSON stream, with deadlines including partial receives."""

    def __init__(self, cfg):
        self.timeout = cfg['socket_timeout_s']
        self.socket = socket.create_connection((cfg['robot_ip'], cfg['robot_port']), self.timeout)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.buffer = bytearray()

    def close(self):
        self.socket.close()

    def pending(self):
        return bool(self.buffer) or bool(select.select([self.socket], [], [], 0)[0])

    def send(self, payload, deadline):
        remaining = deadline-time.perf_counter()
        if remaining <= 0:
            raise TimeoutError('Controller send deadline exceeded')
        self.socket.settimeout(remaining)
        self.socket.sendall((json.dumps(payload, separators=(',', ':'), allow_nan=False)+'\r\n').encode('ascii'))

    def receive(self, deadline):
        if time.perf_counter() >= deadline:
            raise TimeoutError('Controller response timeout')
        while b'\n' not in self.buffer:
            remaining = deadline-time.perf_counter()
            if remaining <= 0 or not select.select([self.socket], [], [], remaining)[0]:
                raise TimeoutError('Controller response timeout')
            self.socket.settimeout(max(.001, deadline-time.perf_counter()))
            data = self.socket.recv(8192)
            if not data:
                raise ConnectionError('Controller TCP disconnected')
            self.buffer.extend(data)
            if len(self.buffer) > 65536:
                raise _ProtocolError('Controller message exceeds 64 KiB')
        line, _, rest = self.buffer.partition(b'\n')
        self.buffer = bytearray(rest)
        if not line.endswith(b'\r'):
            raise _ProtocolError('Controller response is not CRLF terminated')
        try:
            result = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f'Non-finite JSON constant {value}')))
        except (ValueError, UnicodeError) as exc:
            raise _ProtocolError('Invalid controller JSON') from exc
        if not isinstance(result, dict):
            raise _ProtocolError('Controller JSON must be an object')
        return result


class JsonModbusArm:
    METHODS = {'rm_write_single_register': 'write_single_register',
               'rm_write_registers': 'write_registers',
               'rm_read_holding_registers': 'read_holding_registers',
               'rm_read_multiple_holding_registers': 'read_multiple_holding_registers'}

    def __init__(self, ip, port, timeout_s=.3, *, setup_sdk=None, trace_path=None, request_gap_s=0.,
                 link_factory=JsonLink):
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or not 0 < timeout_s <= .5:
            raise ValueError('JSON timeout must be finite and in (0, 0.5] seconds')
        if (isinstance(request_gap_s, bool) or not isinstance(request_gap_s, (int, float))
                or not math.isfinite(request_gap_s) or not 0 <= request_gap_s <= .05):
            raise ValueError('gripper_request_gap_s must be finite and in [0, 0.05] seconds')
        self.cfg = dict(robot_ip=ip, robot_port=port, socket_timeout_s=float(timeout_s))
        self.request_gap_s = float(request_gap_s)
        self._last_reply_at = None
        self._close_event = threading.Event()
        self.setup_sdk, self.factory, self.link = setup_sdk, link_factory, None
        self.motion_guard = self.control_guard = None
        self.needs_sync = False
        self._pending = self._fence = None
        self._pending_drained = False
        self._recovery_deadline = None
        self._sync_waits = 0
        self.recovery_timeout_s = 2.
        self.max_sync_waits = math.ceil(self.recovery_timeout_s/timeout_s)+1
        self._failed = self._closed = False
        self._call_lock, self._metrics_lock = threading.Lock(), threading.Lock()
        self.last_event = None
        self.events, self.durations = deque(maxlen=64), deque(maxlen=512)
        self.all_attempt_durations = deque(maxlen=512)
        self.transactions = self.successes = self.errors = self.timeouts = self.cancellations = 0
        self.trace_error = None
        self.trace_path = str(trace_path) if trace_path else None
        self._trace = None
        if trace_path:
            path = Path(trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._trace = path.open('a', encoding='utf-8', buffering=1)

    def rm_get_current_arm_state(self):
        if self.setup_sdk is None:
            raise RuntimeError('No setup SDK for read-only arm health verification')
        return self.setup_sdk.rm_get_current_arm_state()

    def status(self, *, include_events=False):
        with self._metrics_lock:
            values = sorted(self.durations)
            attempts = sorted(self.all_attempt_durations)
            def percentile(fraction):
                return round(values[min(len(values)-1, int((len(values)-1)*fraction))], 2) if values else None
            def attempt_percentile(fraction):
                # Nearest rank includes rare failures even in a short window.
                return round(attempts[max(0, math.ceil(len(attempts)*fraction)-1)], 2) if attempts else None
            result = dict(backend='json', transactions=self.transactions, successes=self.successes,
                errors=self.errors, timeouts=self.timeouts, cancellations=self.cancellations,
                transaction_p50_ms=percentile(.5), transaction_p95_ms=percentile(.95),
                transaction_p99_ms=percentile(.99), transaction_max_ms=max(values) if values else None,
                p50_ms=percentile(.5), p95_ms=percentile(.95), max_ms=max(values) if values else None,
                all_attempt_p50_ms=attempt_percentile(.5),
                all_attempt_p95_ms=attempt_percentile(.95),
                all_attempt_max_ms=max(attempts) if attempts else None,
                all_attempt_samples=len(attempts),
                timeout_ms=self.cfg['socket_timeout_s']*1000, needs_sync=self.needs_sync,
                request_gap_ms=self.request_gap_s*1000,
                recovery_timeout_ms=self.recovery_timeout_s*1000,
                recovery_deadline=self._recovery_deadline, sync_waits=self._sync_waits,
                max_sync_waits=self.max_sync_waits,
                pending_drained=self._pending_drained,
                recovery_phase=('waiting_fence' if self._pending_drained else 'waiting_original') if self.needs_sync else None,
                failed=self._failed, closed=self._closed, last_event=deepcopy(self.last_event),
                trace_path=self.trace_path,
                trace_error=self.trace_error)
            if include_events:
                result['recent_events'] = deepcopy(list(self.events))
            return result

    def _write_trace(self, event):
        """At most three 8 MiB files per transport trace path."""
        if self._trace is None:
            return
        line = json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n'
        if self._trace.tell()+len(line.encode('utf-8')) > 8*1024*1024:
            self._trace.close()
            self._trace = None
            path = Path(self.trace_path)
            previous = path.with_name(path.name+'.1')
            if previous.exists():
                previous.replace(path.with_name(path.name+'.2'))
            if path.exists():
                path.replace(previous)
            self._trace = path.open('a', encoding='utf-8', buffering=1)
        self._trace.write(line)

    def close(self):
        # Socket close also interrupts an outstanding read. No reconnect occurs.
        self._closed = True
        self._close_event.set()
        self._drop_link()
        if self._trace:
            self._trace.close()
            self._trace = None

    def _drop_link(self):
        link, self.link = self.link, None
        if link:
            try:
                link.close()
            except OSError:
                pass

    def _receive(self, deadline):
        reply = self.link.receive(deadline)
        # Includes retired late replies and fences, not just usable feedback.
        self._last_reply_at = time.perf_counter()
        return reply

    def _before_send(self, event, request, *, recovery=False):
        """Quiet time belongs before dispatch, outside its request timeout."""
        wait_started = time.perf_counter()
        while True:
            if self._closed or self._failed:
                raise ModbusTransportError('Gripper transport closed; explicit reconnect is required')
            if recovery:
                self._check_recovery_budget(event)
            now = time.perf_counter()
            remaining = (self._last_reply_at+self.request_gap_s-now
                         if self._last_reply_at is not None else 0.)
            if remaining <= 0:
                break
            if recovery:
                remaining = min(remaining, self._recovery_deadline-now)
            self._close_event.wait(remaining)
        # Intent may have changed during quiet time. Never dispatch a queued
        # motion/control transition using a guard checked before the wait.
        guarded_motion = ((request['command'] == 'write_registers' and request['address'] == 0x0102
                           and request['num'] in (6, 7))
                          or (request['command'] == 'write_single_register' and request['address'] == 0x0108))
        if guarded_motion and self.motion_guard is not None and not self.motion_guard():
            event['cancelled'] = True
            raise MotionCancelled('Gripper target cancelled before socket dispatch')
        guarded_control = ((request['command'] == 'write_single_register' and request['address'] == 0x0100)
                           or (request['command'] == 'write_registers' and request['address'] == 0x0102
                               and request['num'] == 2))
        if guarded_control and self.control_guard is not None and not self.control_guard():
            event['cancelled'] = True
            raise MotionCancelled('Gripper control transition cancelled before socket dispatch')
        started = time.perf_counter()
        event.setdefault('send_pacing', []).append(dict(
            stage='fence' if recovery else 'request', sent_at=started,
            wait_ms=(started-wait_started)*1000,
            reply_gap_ms=((started-self._last_reply_at)*1000 if self._last_reply_at is not None else None)))
        return started

    @staticmethod
    def _payload(method, params, data):
        command = JsonModbusArm.METHODS[method]
        port, device, address = params.port, params.device, params.address
        if (type(port) is not int or port != 1 or type(device) is not int or not 1 <= device <= 247
                or type(address) is not int or not 0 <= address <= 65535):
            raise ValueError('Invalid end-effector Modbus address/port/device')
        payload = dict(command=command, port=port, device=device, address=address)
        if method in ('rm_read_multiple_holding_registers', 'rm_write_registers'):
            count = params.num
            lower, upper = (3, 12) if command.startswith('read_') else (1, 10)
            if type(count) is not int or not lower <= count <= upper or address+count > 65536:
                raise ValueError('Invalid multiple-register count')
            payload['num'] = count
        if command == 'write_single_register':
            if type(data) is not int or not 0 <= data <= 65535:
                raise ValueError('Single register must be an unsigned 16-bit integer')
            payload['data'] = data
        elif command == 'write_registers':
            if not isinstance(data, list) or len(data) != payload['num']*2 or any(
                    type(value) is not int or not 0 <= value <= 255 for value in data):
                raise ValueError('Multiple-register payload must contain exact unsigned bytes')
            payload['data'] = list(data)
        return payload

    @staticmethod
    def _validate(reply, request):
        command = request['command']
        if not isinstance(reply, dict) or reply.get('command') != command:
            raise _ProtocolError('Controller response command mismatch')
        for key in ('err', 'err_flag', 'arm_err'):
            if key in reply:
                values = reply[key] if isinstance(reply[key], list) else [reply[key]]
                if not values or any(type(value) is not int or value != 0 for value in values):
                    raise _ProtocolError(f'Controller reports {key}: {reply[key]!r}')
        if command.startswith('write_'):
            if reply.get('write_state') is not True:
                raise _ProtocolError('Controller did not acknowledge the write')
            return None
        if 'read_state' in reply and reply['read_state'] is not True:
            raise _ProtocolError('Controller rejected the read')
        value = reply.get('data')
        if command == 'read_holding_registers':
            if type(value) is not int or not -32768 <= value <= 65535:
                raise _ProtocolError('Invalid single-register response')
            return value & 65535
        if not isinstance(value, list) or len(value) != request['num']*2 or any(
                type(item) is not int or not -128 <= item <= 255 for item in value):
            raise _ProtocolError('Invalid multiple-register response length/bytes')
        return [item & 255 for item in value]

    def _drain(self, event):
        discarded = []
        deadline = time.perf_counter()+self.cfg['socket_timeout_s']
        while self.link.pending():
            reply = self._receive(deadline)
            if reply.get('command') not in self.METHODS.values():
                raise _ProtocolError('Unrecognized unsolicited controller response')
            discarded.append(reply)
            if len(discarded) > 20:
                raise _ProtocolError('Too many queued controller responses')
        if discarded:
            event['discarded_before_send'] = discarded
            self.needs_sync = True
            # On a healthy stream, an unsolicited packet has no unique owner.
            # Reopening alone cannot prove it belongs to us: fail closed.
            if self._pending is None:
                raise _ProtocolError('Unsolicited RTU response on an idle stream')

    def _synchronize(self, event, request):
        self._check_recovery_budget(event)
        if self._pending is None:
            raise _ProtocolError('No known pending transaction to fence')
        self._sync_waits += 1
        deadline = min(self._recovery_deadline, time.perf_counter()+self.cfg['socket_timeout_s'])
        event['synchronization'] = dict(request=dict(self._fence) if self._fence else None,
                                        discarded=[])
        if not self._pending_drained:
            # Do not assume that the controller queues RTU commands. Waiting
            # for the known old response retires the sole outstanding request
            # before we send even a read-only fence. Its value is diagnostic
            # only, never published as current measurement or write success.
            event['stage'] = 'pending_receive'
            reply = self._receive(deadline)
            event['synchronization']['discarded'].append(reply)
            if reply.get('command') != self._pending['command']:
                raise _ProtocolError('Unknown response while retiring the pending transaction')
            self._validate(reply, self._pending)
            self._pending_drained = True
            if time.perf_counter() >= self._recovery_deadline:
                self._check_recovery_budget(event)
        if self._fence is None:
            # The two documented READ command types have different response
            # tags; thus the timed-out transaction cannot impersonate this fence.
            if self._pending['command'] == 'read_holding_registers':
                fence = dict(command='read_multiple_holding_registers', port=request['port'],
                             device=request['device'], address=0x060D, num=6)
            else:
                fence = dict(command='read_holding_registers', port=request['port'],
                             device=request['device'], address=0x0100)
            event['stage'] = 'sync_send'
            # Retiring the original reply may consume most of its wait budget.
            # The newly sent fence gets its own send/receive deadline, still
            # capped by the SAME total recovery deadline. Reusing the original
            # wait's deadline here caused a healthy 50-70 ms fence to time out
            # when the late original response arrived after about 250 ms.
            started = self._before_send(event, fence, recovery=True)
            deadline = min(self._recovery_deadline, started+self.cfg['socket_timeout_s'])
            self.link.send(fence, deadline)
            self._fence = fence
        event['synchronization']['request'] = dict(self._fence)
        event['stage'] = 'sync_receive'
        reply = self._receive(deadline)
        if reply.get('command') != self._fence['command']:
            raise _ProtocolError('Unknown response while fencing the retired transaction')
        self._validate(reply, self._fence)
        event['synchronization']['response'] = reply
        self._pending = self._fence = None
        self._pending_drained = False
        self.needs_sync = False
        self._recovery_deadline = None
        self._sync_waits = 0

    def _check_recovery_budget(self, event):
        # A fence itself can be lost. Never turn one lost packet into an
        # indefinitely "recovering" stream with no fresh requests or feedback.
        if (self._recovery_deadline is not None and
                (time.perf_counter() >= self._recovery_deadline or self._sync_waits >= self.max_sync_waits)):
            message = ('Gripper synchronization failed within the 2-second recovery budget; '
                       'transport closed, explicit reconnect is required')
            event.update(stage='recovery_expired', error=message)
            self._failed = True
            self._drop_link()
            raise ModbusTransportError(message)

    def _call(self, method, params, data=None):
        request = self._payload(method, params, data)
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError('Only one outstanding gripper transaction is permitted')
        event = dict(started=time.perf_counter(), request=request, stage='connect')
        try:
            if self._closed or self._failed:
                raise ModbusTransportError('Gripper transport closed; explicit reconnect is required')
            if self.link is None:
                self.link = self.factory(self.cfg)
            if self.needs_sync:
                self._check_recovery_budget(event)
                if request['command'].startswith('write_'):
                    event['stage'] = 'write_blocked'
                    raise ModbusTransportError('A read must resynchronize before any new write')
                # Do not drain a pending fence: its identity must be consumed
                # here, and a fence timeout must not create additional fences.
                self._synchronize(event, request)
                if self.link.pending():
                    raise _ProtocolError('Unexpected queued data after the synchronization fence')
            else:
                event['stage'] = 'drain'
                self._drain(event)
            event['request_started'] = self._before_send(event, request)
            deadline = event['request_started']+self.cfg['socket_timeout_s']
            event['stage'] = 'send'
            self.link.send(request, deadline)
            self._pending = request
            self._pending_drained = False
            event['stage'] = 'receive'
            response = self._receive(deadline)
            event['response'] = response
            value = self._validate(response, request)
            self._pending = None
            event['success'] = True
            return 0 if request['command'].startswith('write_') else (0, value)
        except MotionCancelled:
            raise
        except Exception as exc:
            event['error'] = str(exc)
            if isinstance(exc, TimeoutError) and event['stage'] in ('receive', 'pending_receive', 'sync_receive'):
                event['timeout'] = True
                self.needs_sync = True
                if self._recovery_deadline is None:
                    self._recovery_deadline = time.perf_counter()+self.recovery_timeout_s
                    self._sync_waits = 0
                self._check_recovery_budget(event)
                raise ModbusTransportError(str(exc)) from exc
            if event['stage'] == 'write_blocked':
                raise
            # Sending can time out after a partial JSON frame. Unknown frames,
            # invalid payloads and TCP loss must not reopen behind the caller.
            self._failed = True
            self._drop_link()
            if isinstance(exc, (OSError, ModbusTransportError)):
                event['timeout'] = isinstance(exc, TimeoutError)
                raise ModbusTransportError(str(exc)) from exc
            raise
        finally:
            event['completed'] = time.perf_counter()
            event['duration_ms'] = (event['completed']-event['started'])*1000
            if 'request_started' in event:
                event['request_duration_ms'] = (event['completed']-event['request_started'])*1000
            with self._metrics_lock:
                self.transactions += 1
                self.successes += int(event.get('success', False))
                self.errors += int('error' in event)
                self.timeouts += int(event.get('timeout', False))
                self.cancellations += int(event.get('cancelled', False))
                if event.get('success'):
                    self.durations.append(event['request_duration_ms'])
                # Include failures, cancellations and fence/drain waits. This
                # measures caller-observed attempt time, not just successful RTT.
                self.all_attempt_durations.append(event['duration_ms'])
                self.last_event = deepcopy(event)
                self.events.append(deepcopy(event))
            if self._trace:
                try:
                    self._write_trace(event)
                except (OSError, ValueError) as exc:
                    self.trace_error = str(exc)
            self._call_lock.release()

    def rm_read_holding_registers(self, params):
        return self._call('rm_read_holding_registers', params)

    def rm_read_multiple_holding_registers(self, params):
        return self._call('rm_read_multiple_holding_registers', params)

    def rm_write_single_register(self, params, data):
        return self._call('rm_write_single_register', params, data)

    def rm_write_registers(self, params, data):
        return self._call('rm_write_registers', params, data)
