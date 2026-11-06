"""Socket-level timing telemetry with a controlled clock and no live network."""
from collections import deque
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from realman_sync_client.gripper_transport import JsonLink, JsonModbusArm, ModbusTransportError


class FakeSocket:
    def __init__(self, timer, chunks):
        self.timer, self.chunks = timer, deque(chunks)
        self.sent, self.timeouts = [], []
        self.closed = False
        self.send_failure = False

    def setsockopt(self, *_):
        pass

    def getsockname(self):
        return ('192.168.1.100', 51000)

    def getpeername(self):
        return ('192.168.1.18', 8080)

    def settimeout(self, value):
        self.timeouts.append(value)

    def sendall(self, value):
        self.sent.append(value)
        self.timer[0] += .001
        if self.send_failure:
            raise TimeoutError('uncertain send')

    def recv(self, size):
        self.timer[0] += .001
        assert size == 8192
        return self.chunks.popleft()

    def close(self):
        self.closed = True


def params():
    return SimpleNamespace(port=1, device=1, address=0x0100, num=1)


class SocketTimingTests(unittest.TestCase):
    def make(self, chunks, waits=(), *, trace_path=None):
        timer = [10.]
        sock = FakeSocket(timer, chunks)
        waits = deque(waits)
        def select(read, write, error, timeout):
            self.assertEqual(read, [sock])
            if timeout == 0:
                return [], [], []
            delay, ready = waits.popleft() if waits else (.001, True)
            timer[0] += timeout+delay if not ready else delay
            return ([sock] if ready else []), [], []
        for target, value in (
                ('time.perf_counter', lambda: timer[0]),
                ('socket.create_connection', lambda *_: sock),
                ('select.select', select)):
            patcher = patch('realman_sync_client.gripper_transport.'+target, side_effect=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        transport = JsonModbusArm('unused', 8080, .3, trace_path=trace_path)
        self.addCleanup(transport.close)
        return transport, sock, timer

    def test_success_records_send_wait_recv_parse_and_socket_endpoints(self):
        arm, sock, timer = self.make([b'{"command":"read_holding_registers","data":1}\r\n'],
                                    [(.06, True)])
        loads = json.loads
        def slow_parse(*args, **kwargs):
            timer[0] += .002
            return loads(*args, **kwargs)
        with patch('realman_sync_client.gripper_transport.json.loads', side_effect=slow_parse):
            self.assertEqual(arm.rm_read_holding_registers(params()), (0, 1))
        trace = arm.last_event['socket_timing']
        self.assertEqual(trace['endpoints']['local']['port'], 51000)
        self.assertEqual(trace['endpoints']['remote']['ip'], '192.168.1.18')
        items = {item['operation']: item for item in trace['events']}
        self.assertAlmostEqual(items['send']['duration_ms'], 1.)
        self.assertAlmostEqual(items['select']['duration_ms'], 60.)
        self.assertTrue(items['select']['readable'])
        self.assertAlmostEqual(items['recv']['duration_ms'], 1.)
        self.assertAlmostEqual(items['parse']['duration_ms'], 2.)
        self.assertEqual(items['recv']['received_bytes'], 47)
        self.assertEqual(items['select']['deadline_overrun_ms'], 0.)
        self.assertEqual(trace['dropped'], 0)
        self.assertTrue(all(row['phase'] == 'request' for row in trace['events']))
        self.assertFalse(any('data' in row or 'payload' in row for row in trace['events']))

    def test_timeout_overrun_then_retired_fence_fresh_timing_is_saved_to_jsonl(self):
        frames = [b'{"command":"read_holding_registers","data":1}\r\n',
                  b'{"command":"read_multiple_holding_registers","data":[0,0,0,0,0,0,0,0,0,0,0,0]}\r\n',
                  b'{"command":"read_holding_registers","data":1}\r\n']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'timing.jsonl'
            arm, sock, timer = self.make(frames,
                [(.013, False), (.1, True), (.06, True), (.06, True)], trace_path=path)
            with self.assertRaises(ModbusTransportError):
                arm.rm_read_holding_registers(params())
            initial = arm.last_event['socket_timing']['events']
            wait = next(row for row in initial if row['operation'] == 'select')
            self.assertFalse(wait['readable'])
            self.assertAlmostEqual(wait['deadline_overrun_ms'], 13.)
            self.assertFalse(any(row['operation'] == 'recv' for row in initial))
            self.assertEqual(arm.rm_read_holding_registers(params()), (0, 1))
            recovered = arm.last_event['socket_timing']['events']
            self.assertEqual([row['phase'] for row in recovered if row['operation']=='recv'],
                             ['retired', 'fence', 'request'])
            self.assertEqual([row['phase'] for row in recovered if row['operation']=='send'],
                             ['fence', 'request'])
            self.assertEqual(len(sock.sent), 3)  # no retry of original request
            arm.close()
            saved = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(saved), 2)
            self.assertEqual(saved[-1]['socket_timing']['events'], recovered)

    def test_failed_send_still_emits_timing_after_link_is_closed(self):
        arm, sock, _ = self.make([])
        sock.send_failure = True
        with self.assertRaises(ModbusTransportError):
            arm.rm_read_holding_registers(params())
        self.assertTrue(sock.closed)
        send = next(row for row in arm.last_event['socket_timing']['events']
                    if row['operation']=='send')
        self.assertFalse(send['success'])
        self.assertAlmostEqual(send['duration_ms'], 1.)

    def test_partial_frame_telemetry_is_bounded_and_drain_resets_it(self):
        frame = b'{"data":"'+b'x'*80+b'"}\r\n'
        arm, sock, timer = self.make([bytes([byte]) for byte in frame])
        link = JsonLink(dict(robot_ip='unused', robot_port=8080, socket_timeout_s=.3))
        self.assertEqual(link.receive(20.), {'data':'x'*80})
        trace = link.drain_timing()
        self.assertEqual(len(trace['events']), 128)
        self.assertGreater(trace['dropped'], 0)
        self.assertEqual(link.drain_timing()['events'], [])
        self.assertEqual(link.drain_timing()['dropped'], 0)
        link.close()


if __name__ == '__main__':
    unittest.main()
