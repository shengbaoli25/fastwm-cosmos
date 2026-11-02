"""Protocol fault injection only; no hardware or live sockets."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from realman_sync_client.gripper_transport import (
    JsonLink, JsonModbusArm, ModbusTransportError, MotionCancelled,
)


def params(address=0x060D, count=6):
    return SimpleNamespace(port=1, address=address, device=1, num=count)


def single(value=1):
    return dict(command='read_holding_registers', data=value)


def multiple(value=0):
    return dict(command='read_multiple_holding_registers', data=[0, value]+[0]*10)


class FakeLink:
    def __init__(self, cfg):
        self.cfg = cfg
        self.replies, self.sent, self.deadlines = [], [], []
        self.closed = False
        self.pending_callback = None
        self.send_callback = None
        self.pending_count = 0

    def close(self):
        self.closed = True

    def pending(self):
        if self.pending_callback:
            self.pending_callback()
        return self.pending_count > 0

    def send(self, payload, deadline):
        self.sent.append(deepcopy(payload))
        self.deadlines.append(deadline)
        if self.send_callback:
            self.send_callback(payload)

    def receive(self, deadline):
        self.deadlines.append(deadline)
        if self.pending_count:
            self.pending_count -= 1
        value = self.replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class TransportTests(unittest.TestCase):
    def make(self, **kwargs):
        link = FakeLink({})
        transport = JsonModbusArm('not-connected', 8080, link_factory=lambda cfg: link, **kwargs)
        self.addCleanup(transport.close)
        return transport, link

    def test_success_shapes_and_true_ack(self):
        arm, link = self.make()
        link.replies = [single(), multiple(12), {'command':'write_single_register','write_state':True},
                        {'command':'write_registers','write_state':True}]
        self.assertEqual(arm.rm_read_holding_registers(params(0x100, 1)), (0, 1))
        self.assertEqual(arm.rm_read_multiple_holding_registers(params())[1][1], 12)
        self.assertEqual(arm.rm_write_single_register(params(0x100, 1), 0), 0)
        self.assertEqual(arm.rm_write_registers(params(0x102, 7), [0]*14), 0)
        self.assertEqual(arm.status()['successes'], 4)
        self.assertEqual(link.sent[-1]['num'], 7)
        self.assertEqual(len(link.sent[-1]['data']), 14)

    def test_late_same_command_reply_is_never_returned_as_current_measurement(self):
        arm, link = self.make()
        link.replies = [TimeoutError('late'), multiple(99), single(), multiple(17)]
        with self.assertRaises(ModbusTransportError):
            arm.rm_read_multiple_holding_registers(params())
        self.assertTrue(arm.status()['needs_sync'])
        self.assertEqual(arm.rm_read_multiple_holding_registers(params())[1][1], 17)
        self.assertEqual([p['command'] for p in link.sent],
            ['read_multiple_holding_registers', 'read_holding_registers', 'read_multiple_holding_registers'])
        self.assertEqual(arm.last_event['synchronization']['discarded'][0]['data'][1], 99)
        self.assertEqual(arm.status()['successes'], 1)
        self.assertEqual(arm.status()['timeouts'], 1)

    def test_fence_timeout_resumes_original_fence_without_sending_another(self):
        arm, link = self.make()
        link.replies = [TimeoutError(), multiple(99), TimeoutError(), single(), multiple(3)]
        for _ in range(2):
            with self.assertRaises(ModbusTransportError):
                arm.rm_read_multiple_holding_registers(params())
        self.assertEqual(len(link.sent), 2)
        self.assertEqual(arm.rm_read_multiple_holding_registers(params())[1][1], 3)
        self.assertEqual(len(link.sent), 3)
        self.assertFalse(link.closed)

    def test_lost_write_ack_is_never_replayed_and_requires_read_recovery(self):
        arm, link = self.make()
        link.replies = [TimeoutError(), {'command':'write_registers','write_state':True}, single(), multiple(2)]
        with self.assertRaises(ModbusTransportError):
            arm.rm_write_registers(params(0x102, 7), [0]*14)
        with self.assertRaisesRegex(ModbusTransportError, 'read must resynchronize'):
            arm.rm_write_registers(params(0x102, 7), [0]*14)
        self.assertEqual(len(link.sent), 1)
        arm.rm_read_multiple_holding_registers(params())
        self.assertEqual(sum(p['command']=='write_registers' for p in link.sent), 1)
        self.assertFalse(arm.status()['failed'])

    def test_motion_guard_after_read_resynchronization_cancels_target(self):
        arm, link = self.make()
        link.replies = [TimeoutError(), multiple(8), single(), multiple(2)]
        with self.assertRaises(ModbusTransportError):
            arm.rm_read_multiple_holding_registers(params())
        arm.rm_read_multiple_holding_registers(params())
        arm.motion_guard = lambda: False
        with self.assertRaises(MotionCancelled):
            arm.rm_write_registers(params(0x102, 7), [0]*14)
        self.assertEqual(len(link.sent), 3)
        self.assertEqual(arm.status()['cancellations'], 1)

    def test_guard_runs_after_drain_observes_new_stop_intent(self):
        arm, link = self.make()
        intent = {'run': True}
        link.pending_callback = lambda: intent.update(run=False)
        arm.motion_guard = lambda: intent['run']
        with self.assertRaises(MotionCancelled):
            arm.rm_write_registers(params(0x102, 7), [0]*14)
        self.assertEqual(link.sent, [])

    def test_separate_target_and_trigger_recheck_motion_after_drain(self):
        for stage in ('target', 'trigger'):
            with self.subTest(stage=stage):
                arm, link = self.make()
                intent = {'run': True}
                link.pending_callback = lambda: intent.update(run=False)
                arm.motion_guard = lambda: intent['run']
                with self.assertRaises(MotionCancelled):
                    if stage == 'target':
                        arm.rm_write_registers(params(0x102, 6), [0]*12)
                    else:
                        arm.rm_write_single_register(params(0x108, 1), 1)
                self.assertEqual(link.sent, [])
                self.assertFalse(arm.status()['failed'])

    def test_transition_guard_blocks_old_control_but_explicit_stop_can_clear_guard(self):
        arm, link = self.make()
        arm.control_guard = lambda: False
        for value in (1, 0):
            with self.assertRaises(MotionCancelled):
                arm.rm_write_single_register(params(0x100, 1), value)
        self.assertEqual(link.sent, [])
        # Worker cleanup deliberately removes the old transition ticket; a
        # fresh explicit stop must remain possible after that ticket expires.
        arm.control_guard = None
        link.replies = [{'command':'write_single_register','write_state':True}]
        self.assertEqual(arm.rm_write_single_register(params(0x100, 1), 0), 0)
        self.assertEqual(link.sent[0]['data'], 0)

    def test_neutral_target_guard_checks_intent_after_drain(self):
        arm, link = self.make()
        intent = {'run': True}
        arm.control_guard = lambda: intent['run']
        link.pending_callback = lambda: intent.update(run=False)
        with self.assertRaises(MotionCancelled):
            arm.rm_write_registers(params(0x102, 2), [0, 0, 0, 100])
        self.assertEqual(link.sent, [])
        self.assertFalse(arm.status()['failed'])

    def test_partial_send_timeout_is_fatal_and_never_reconnects(self):
        arm, link = self.make()
        link.send_callback = Mock(side_effect=TimeoutError('partial write'))
        with self.assertRaises(ModbusTransportError):
            arm.rm_read_multiple_holding_registers(params())
        self.assertTrue(link.closed)
        self.assertTrue(arm.status()['failed'])
        with self.assertRaisesRegex(ModbusTransportError, 'explicit reconnect'):
            arm.rm_read_multiple_holding_registers(params())
        self.assertEqual(len(link.sent), 1)

    def test_malformed_response_or_negative_ack_fail_closed(self):
        bad = [dict(command='unknown', data=[]), dict(command='read_multiple_holding_registers',data=[0]*11),
               dict(command='read_multiple_holding_registers',data=[True]*12),
               dict(command='read_multiple_holding_registers',data=[256]*12),
               dict(command='read_multiple_holding_registers',data=[0]*12, read_state=False),
               dict(command='read_multiple_holding_registers',data=[0]*12, err=[32])]
        for reply in bad:
            with self.subTest(reply=reply):
                arm, link = self.make()
                link.replies = [reply]
                with self.assertRaises(RuntimeError):
                    arm.rm_read_multiple_holding_registers(params())
                self.assertTrue(link.closed)
                self.assertEqual(arm.status()['successes'], 0)
        for ack in (False, 1, None):
            arm, link = self.make()
            link.replies = [dict(command='write_registers',write_state=ack)]
            with self.assertRaises(RuntimeError):
                arm.rm_write_registers(params(0x102, 7), [0]*14)

    def test_unknown_fence_reply_is_fatal(self):
        arm, link = self.make()
        link.replies = [TimeoutError(), dict(command='write_single_register',write_state=True)]
        with self.assertRaises(ModbusTransportError):
            arm.rm_read_multiple_holding_registers(params())
        with self.assertRaisesRegex(RuntimeError, 'Unknown response'):
            arm.rm_read_multiple_holding_registers(params())
        self.assertTrue(link.closed)
        self.assertEqual(len(link.sent), 1)  # unknown original response cannot permit a fence

    def test_original_must_retire_before_fence_is_sent(self):
        arm, link = self.make()
        link.replies = [TimeoutError(), TimeoutError(), multiple(99), single(), multiple(4)]
        for _ in range(2):
            with self.assertRaises(ModbusTransportError):
                arm.rm_read_multiple_holding_registers(params())
            self.assertEqual(len(link.sent), 1)
        self.assertEqual(arm.status()['recovery_phase'], 'waiting_original')
        def sending_fence(request):
            if request['command'] == 'read_holding_registers':
                self.assertTrue(arm._pending_drained)
        link.send_callback = sending_fence
        self.assertEqual(arm.rm_read_multiple_holding_registers(params())[1][1], 4)
        self.assertEqual([p['command'] for p in link.sent],
            ['read_multiple_holding_registers', 'read_holding_registers', 'read_multiple_holding_registers'])

    def test_lost_original_or_lost_fence_closes_within_total_recovery_budget(self):
        for lost in ('original', 'fence'):
            with self.subTest(lost=lost):
                arm, link = self.make()
                timer, first = [10.], [True]
                original_arrived = [False]
                def receive(deadline):
                    if first[0]:
                        first[0] = False
                        timer[0] = deadline
                        raise TimeoutError('first transaction timeout')
                    if lost == 'fence' and not original_arrived[0]:
                        original_arrived[0] = True
                        timer[0] += .01
                        return multiple(99)
                    timer[0] = deadline
                    raise TimeoutError(f'lost {lost}')
                link.receive = receive
                with patch('realman_sync_client.gripper_transport.time.perf_counter', side_effect=lambda: timer[0]):
                    with self.assertRaises(ModbusTransportError):
                        arm.rm_read_multiple_holding_registers(params())
                    recovery_started = timer[0]
                    for _ in range(arm.max_sync_waits+1):
                        with self.assertRaises(ModbusTransportError):
                            arm.rm_read_multiple_holding_registers(params())
                        if arm.status()['failed']:
                            break
                    self.assertTrue(arm.status()['failed'])
                    self.assertTrue(link.closed)
                    self.assertLessEqual(timer[0]-recovery_started, 2.000001)
                    self.assertEqual(arm.last_event['stage'], 'recovery_expired')
                    self.assertIn('explicit reconnect', arm.last_event['error'])
                    self.assertEqual(arm.successes, 0)
                    self.assertEqual(len(link.sent), 1 if lost == 'original' else 2)
                    sent = len(link.sent)
                    with self.assertRaisesRegex(ModbusTransportError, 'explicit reconnect'):
                        arm.rm_read_multiple_holding_registers(params())
                    self.assertEqual(len(link.sent), sent)

    def test_no_clock_progress_is_still_bounded_by_sync_wait_count(self):
        arm, link = self.make()
        link.replies = [TimeoutError() for _ in range(arm.max_sync_waits+3)]
        with patch('realman_sync_client.gripper_transport.time.perf_counter', return_value=10.):
            for _ in range(arm.max_sync_waits+1):
                with self.assertRaises(ModbusTransportError):
                    arm.rm_read_multiple_holding_registers(params())
        self.assertTrue(arm.status()['failed'])
        self.assertTrue(link.closed)
        self.assertEqual(len(link.sent), 1)

    def test_late_negative_write_ack_is_fatal_without_replay_or_fence(self):
        arm, link = self.make()
        link.replies = [TimeoutError(), {'command':'write_registers','write_state':False}]
        with self.assertRaises(ModbusTransportError):
            arm.rm_write_registers(params(0x102, 7), [0]*14)
        with self.assertRaisesRegex(RuntimeError, 'did not acknowledge'):
            arm.rm_read_multiple_holding_registers(params())
        self.assertTrue(link.closed)
        self.assertEqual(len(link.sent), 1)
        self.assertFalse(arm.last_event.get('success', False))

    def test_idle_unsolicited_reply_is_not_used_as_fresh_data(self):
        arm, link = self.make()
        link.pending_count = 1
        link.replies = [multiple(19)]
        with self.assertRaisesRegex(RuntimeError, 'Unsolicited'):
            arm.rm_read_multiple_holding_registers(params())
        self.assertEqual(link.sent, [])
        self.assertTrue(link.closed)

    def test_concurrent_calls_rejected_without_touching_socket(self):
        arm, link = self.make()
        arm._call_lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, 'one outstanding'):
                arm.rm_read_multiple_holding_registers(params())
        finally:
            arm._call_lock.release()
        self.assertEqual(link.sent, [])

    def test_invalid_requests_never_open_socket(self):
        arm, link = self.make()
        for count in (0, 1, 2, 13, True):
            with self.assertRaises(ValueError):
                arm.rm_read_multiple_holding_registers(params(count=count))
        with self.assertRaises(ValueError):
            arm.rm_write_registers(params(0x102, 7), [0]*13)
        self.assertIsNone(arm.link)
        for timeout in (0, -.1, .51, float('nan'), True):
            with self.assertRaises(ValueError):
                JsonModbusArm('not-used', 8080, timeout)

    def test_signed_read_bytes_are_normalized_and_metrics_are_copies(self):
        arm, link = self.make()
        reply = multiple()
        reply['data'][0] = -1
        link.replies = [reply]
        self.assertEqual(arm.rm_read_multiple_holding_registers(params())[1][0], 255)
        result = arm.status(include_events=True)
        result['recent_events'][0]['response']['data'][0] = 8
        self.assertEqual(arm.status(include_events=True)['recent_events'][0]['response']['data'][0], -1)
        self.assertNotIn('recent_events', arm.status())

    def test_trace_contains_one_completed_event_for_timeout_and_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'transactions.jsonl'
            arm, link = self.make(trace_path=path)
            link.replies = [single(), TimeoutError()]
            arm.rm_read_holding_registers(params(0x100, 1))
            with self.assertRaises(ModbusTransportError):
                arm.rm_read_multiple_holding_registers(params())
            arm.close()
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[0]['success'])
            self.assertTrue(rows[1]['timeout'])

    def test_all_attempt_distribution_includes_timeout_and_fence_wait(self):
        arm, link = self.make()
        timer = [10.]
        # Healthy request, a 300 ms timeout, then late reply + fence + new read.
        link.replies = [single(), TimeoutError('300 ms timeout'), multiple(99), single(), multiple(3)]
        delays = iter([.06, .30, .10, .10, .06])
        receive = link.receive
        def timed_receive(deadline):
            timer[0] += next(delays)
            return receive(deadline)
        link.receive = timed_receive
        with patch('realman_sync_client.gripper_transport.time.perf_counter', side_effect=lambda: timer[0]):
            arm.rm_read_holding_registers(params(0x100, 1))
            with self.assertRaises(ModbusTransportError):
                arm.rm_read_multiple_holding_registers(params())
            arm.rm_read_multiple_holding_registers(params())
        metrics = arm.status()
        self.assertEqual(metrics['all_attempt_samples'], 3)
        self.assertAlmostEqual(metrics['all_attempt_max_ms'], 300.)
        self.assertAlmostEqual(metrics['all_attempt_p95_ms'], 300.)
        self.assertAlmostEqual(metrics['all_attempt_p50_ms'], 260.)
        self.assertAlmostEqual(metrics['max_ms'], 60.)
        self.assertAlmostEqual(metrics['p50_ms'], 60.)
        self.assertEqual(metrics['successes'], 2)
        self.assertEqual(metrics['timeouts'], 1)
        self.assertEqual(arm.all_attempt_durations.maxlen, 512)

    def test_trace_rotation_retains_only_two_backups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'transactions.jsonl'
            arm, link = self.make(trace_path=path)
            for index in range(4):
                # Pre-fill to the boundary without producing any hardware I/O.
                arm._trace.write('x'*(8*1024*1024))
                link.replies = [single()]
                arm.rm_read_holding_registers(params(0x100, 1))
            arm.close()
            self.assertEqual(sorted(item.name for item in Path(directory).iterdir()),
                             ['transactions.jsonl','transactions.jsonl.1','transactions.jsonl.2'])
            self.assertEqual(len(path.read_text(encoding='utf-8').splitlines()), 1)


class LinkTests(unittest.TestCase):
    def make(self, chunks):
        sock = Mock()
        sock.recv.side_effect = chunks
        with patch('realman_sync_client.gripper_transport.socket.create_connection', return_value=sock):
            link = JsonLink(dict(robot_ip='unused', robot_port=8080, socket_timeout_s=.3))
        return link, sock

    def test_nodelay_split_frame_and_multiple_buffered_lines(self):
        link, sock = self.make([b'{"command":"read_', b'holding_registers","data":1}\r\n{"data":2}\r\n'])
        with patch('realman_sync_client.gripper_transport.select.select', return_value=([sock], [], [])), \
                patch('realman_sync_client.gripper_transport.time.perf_counter', return_value=1):
            self.assertEqual(link.receive(1.3), single())
            self.assertEqual(link.receive(1.3), {'data':2})
            link.send({'command':'test'}, 1.3)
        sock.setsockopt.assert_called_once()
        self.assertEqual(sock.sendall.call_args.args[0], b'{"command":"test"}\r\n')

    def test_bounded_frames_strict_crlf_and_no_nonfinite(self):
        for chunks in ([b'0'*65537], [b'{}\n'], [b'[]\r\n'], [b'{"data":NaN}\r\n'], [b'']):
            with self.subTest(chunks=str(chunks)[:50]):
                link, sock = self.make(chunks)
                with patch('realman_sync_client.gripper_transport.select.select', return_value=([sock], [], [])), \
                        patch('realman_sync_client.gripper_transport.time.perf_counter', return_value=1):
                    with self.assertRaises((RuntimeError, ConnectionError)):
                        link.receive(1.3)

    def test_expired_deadline_does_not_accept_buffered_data(self):
        link, sock = self.make([])
        link.buffer = bytearray(b'{"data":1}\r\n')
        with patch('realman_sync_client.gripper_transport.time.perf_counter', return_value=2):
            with self.assertRaises(TimeoutError):
                link.receive(1.3)
        sock.recv.assert_not_called()


if __name__ == '__main__':
    unittest.main()
