"""Wire equivalence and admission checks for cooperative request encoding."""
from dataclasses import replace
import io
import json
import unittest
from unittest.mock import patch

from realman_sync_client import client as client_module
from realman_sync_client.client import SynchronousClient
from realman_sync_client.contracts import Observation
from realman_sync_client.http import InferenceHTTPClient, _encode_json_body
from realman_sync_client.wire import PROTOCOL, decode_request


class JsonSerializationTests(unittest.TestCase):
    def test_exact_existing_json_bytes_for_nested_values_and_unicode(self):
        values = {'text': '\u76f8\u673a\\\"\n\u0000\U0001f4f7', 'v': [None, True, False, -0.0, 1.5, 1e-90],
                  'nested': [{'1': (1, 2)}, {2: 'integer key'}]}
        expected = json.dumps(values, separators=(',', ':'), allow_nan=False).encode('utf-8')
        self.assertEqual(_encode_json_body(values), expected)

    def test_large_camera_strings_yield_between_parts_and_keep_every_byte(self):
        values = {'frames': [{'data': 'ABCD' * (320 * 1024)} for _ in range(6)]}
        with patch('realman_sync_client.http.time.sleep') as sleep:
            body = _encode_json_body(values)
        self.assertEqual(body, json.dumps(values, separators=(',', ':'), allow_nan=False).encode())
        self.assertEqual(sleep.call_count, 7)  # One before encoding, one per large frame.
        self.assertTrue(all(call.args == (.001,) for call in sleep.call_args_list))

    def test_nan_infinity_and_circular_values_are_rejected(self):
        cycle = {}; cycle['self'] = cycle
        for value in [float('nan'), float('inf'), -float('inf'), cycle]:
            with self.subTest(value=type(value).__name__), self.assertRaises(ValueError):
                _encode_json_body({'v': value})

    def test_byte_limit_accepts_exact_boundary_rejects_next_byte(self):
        data = {'value': '\u4e2d' * 20}
        length = len(json.dumps(data, separators=(',', ':'), allow_nan=False).encode())
        with patch('realman_sync_client.http.MAX_BODY_BYTES', length):
            self.assertEqual(len(_encode_json_body(data)), length)
        with patch('realman_sync_client.http.MAX_BODY_BYTES', length - 1):
            with self.assertRaisesRegex(ValueError, 'request exceeds 64 MiB'):
                _encode_json_body(data)

    def test_invalid_body_never_reaches_network(self):
        transport = InferenceHTTPClient('http://127.0.0.1:8766', token='test-token')
        with patch.object(transport.opener, 'open') as opened:
            with self.assertRaises(ValueError):
                transport.infer({'v': float('nan')})
            with patch('realman_sync_client.http.MAX_BODY_BYTES', 32):
                with self.assertRaises(ValueError):
                    transport.infer({'v': 'x' * 100})
            opened.assert_not_called()

    def test_transport_keeps_one_authenticated_bounded_post_and_timeout(self):
        transport = InferenceHTTPClient('http://127.0.0.1:8766', timeout_s=7, token='test-token')
        payload = {'protocol': PROTOCOL, 'data': 'z' * 200000}
        response = io.BytesIO(json.dumps({'protocol': PROTOCOL}).encode())
        with patch.object(transport.opener, 'open', return_value=response) as opened:
            self.assertEqual(transport.infer(payload)['protocol'], PROTOCOL)
        opened.assert_called_once()
        request = opened.call_args.args[0]
        self.assertEqual(request.data, json.dumps(payload, separators=(',', ':'), allow_nan=False).encode())
        self.assertEqual(request.get_method(), 'POST')
        self.assertEqual(request.get_header('Authorization'), 'Bearer test-token')
        self.assertEqual(opened.call_args.kwargs['timeout'], 7)


class RequestEncodingReuseTests(unittest.TestCase):
    def make_data(self):
        meta = dict(camera_names=['global', 'wrist'], width=4, height=3, action_space='joint8',
                    action_horizon=32, history_length=32, control_hz=30,
                    video_context_frames=2, video_context_stride=4)
        class Hardware:
            def status(self): return {'force_preprocessing': None}
        client = SynchronousClient(None, Hardware(), meta)
        history = [Observation(100+i/30, i, (0.,)*7+(.5,), (0.,)*6,
                        {'global': bytes(36), 'wrist': bytes(36)}, action_space='joint8') for i in range(32)]
        return client, history

    def test_actual_current_observation_is_encoded_once_without_changing_wire(self):
        client, history = self.make_data()
        current = history[-1]
        with patch.object(client_module, 'observation_row', wraps=client_module.observation_row) as encode:
            request = client._request(current, history, [history[-5], current])
        self.assertEqual(encode.call_count, 2)
        self.assertIs(request['video_context'][-1], request['observation'])
        decoded = decode_request(json.loads(_encode_json_body(request)), client.metadata)
        self.assertEqual(len(decoded['video_context']), 2)
        self.assertEqual(decoded['video_context'][-1]['frames'], decoded['observation']['frames'])

    def test_equal_but_distinct_observation_is_not_assumed_to_be_current(self):
        client, history = self.make_data()
        current = history[-1]
        copy = replace(current)
        with patch.object(client_module, 'observation_row', wraps=client_module.observation_row) as encode:
            request = client._request(current, history, [history[-5], copy])
        self.assertEqual(encode.call_count, 3)
        self.assertIsNot(request['video_context'][-1], request['observation'])
        self.assertEqual(request['video_context'][-1], request['observation'])


if __name__ == '__main__':
    unittest.main()
