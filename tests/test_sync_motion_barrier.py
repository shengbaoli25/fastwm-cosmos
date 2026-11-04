"""Exercise production CLI/web motion gates without threads, SDKs or wall time.

The collector, wire validation, client and dashboard are real. Only hardware,
HTTP and clock waits are replaced; each wait feeds the real collector on its
normal 30 Hz grid. This makes a stop ACK followed by residual motion repeatable.
"""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import math
import unittest
from unittest.mock import patch

from realman_sync_client import cli, clock, webapp
from realman_sync_client.client import SynchronousClient
from realman_sync_client.mock import MockHardware
from realman_sync_client.wire import PROTOCOL, decode_request


META = dict(camera_names=['global', 'wrist'], width=4, height=3,
            action_space='joint8', action_horizon=4, history_length=32,
            control_hz=30, video_context_frames=1, video_context_stride=1,
            force_preprocessing=None)


class VirtualGrid:
    """Advance a collector at exact sample times instead of sleeping."""
    period = 1 / 30

    def __init__(self):
        self.now = 100.0
        self.collector = None
        self.next_tick = None

    def attach(self, stack, client):
        self.collector = client.collector
        stack.enter_context(patch.object(self.collector, 'start', self.start))
        stack.enter_context(patch.object(self.collector.condition, 'wait', self.wait))

    def start(self):
        # The history is complete before inference, but only samples acquired
        # after slow_stop may count toward its separate stationary window.
        for index in range(32):
            stamp = self.now - (31 - index) * self.period
            self.collector._append(self.collector.hardware.observe_at(stamp), stamp)
        self.next_tick = self.now + self.period

    def advance(self, duration):
        end = self.now + duration
        while self.next_tick is not None and self.next_tick <= end + 1e-10:
            self.now = self.next_tick
            observation = self.collector.hardware.observe_at(self.now)
            with self.collector.condition:
                self.collector._append(observation, self.now)
                self.collector.condition.notify_all()
            self.next_tick += self.period
        self.now = max(self.now, end)

    def wait(self, timeout=None):
        # Condition.wait is only used by this test's synchronous motion gate.
        self.advance(self.period if timeout is None else min(timeout, self.period))

    def wait_until(self, deadline, stop_event, *, check=None):
        if stop_event.is_set():
            return False
        if check is not None:
            check()
        self.advance(max(0.0, deadline - self.now))
        if check is not None:
            check()
        return not stop_event.is_set()


class ResidualMotionHardware(MockHardware):
    def __init__(self, grid, *, residual_after_chunk=False, move_on_arm=False):
        super().__init__(width=META['width'], height=META['height'])
        self.grid = grid
        self.residual_after_chunk = residual_after_chunk
        self.move_on_arm = move_on_arm
        self.arm_count = 0
        self.stop_acks = []
        self.tail = None
        self.tail_end = None

    def measured(self, stamp):
        values = list(self.state)
        if self.tail is not None:
            began, base = self.tail
            fraction = min(1.0, max(0.0, (stamp - began) / .3))
            values[0] = base + math.radians(fraction)
        return tuple(values)

    def observe_at(self, anchor):
        return replace(super().observe_at(anchor), state=self.measured(anchor))

    def joint_feedback(self):
        return self.measured(self.grid.now)[:7], self.grid.now

    def displace(self):
        values = list(self.measured(self.grid.now))
        values[0] += math.radians(1.0)
        self.state, self.tail = tuple(values), None

    def arm(self):
        super().arm()
        self.arm_count += 1
        if self.move_on_arm:
            self.displace()

    def command(self, action):
        self.tail = None
        super().command(action)

    def slow_stop(self):
        super().slow_stop()
        self.stop_acks.append(self.grid.now)
        if self.residual_after_chunk and len(self.stop_acks) == 2:
            # Acknowledgement is immediate; the feedback still moves for .3 s.
            self.tail = (self.grid.now, self.state[0])
            self.tail_end = self.grid.now + .3


class LocalTransport:
    server_url = 'http://127.0.0.1:1'

    def __init__(self, grid, hardware, *, displace_during_inference=False):
        self.grid, self.hardware = grid, hardware
        self.displace_during_inference = displace_during_inference
        self.requests = []
        self.released = []

    def health(self):
        return dict(protocol=PROTOCOL, ready=True, metadata=deepcopy(META))

    def infer(self, request):
        decode_request(request, META)
        self.requests.append(deepcopy(request))
        if self.displace_during_inference:
            self.hardware.displace()
        self.grid.advance(.1)
        return dict(protocol=PROTOCOL, session_id=request['session_id'],
                    request_id=request['request_id'],
                    source_sequence=request['observation']['sequence'],
                    action_space='joint8', control_hz=30, inference_ms=100.,
                    actions=[list(request['observation']['state'])
                             for _ in range(META['action_horizon'])])

    def release(self, session_id):
        self.released.append(session_id)


class MotionBarrierIntegrationTests(unittest.TestCase):
    def fixture(self, stack, **hardware_options):
        grid = VirtualGrid()
        hardware = ResidualMotionHardware(grid, **hardware_options)
        transport = LocalTransport(grid, hardware)
        clients = []

        def real_client(*args, **kwargs):
            client = SynchronousClient(*args, **kwargs)
            clients.append(client)
            grid.attach(stack, client)
            return client

        stack.enter_context(patch.object(clock, 'now', lambda: grid.now))
        stack.enter_context(patch.object(clock, 'wait_until', grid.wait_until))
        stack.enter_context(patch('realman_sync_client.client.logger'))
        stack.enter_context(patch('realman_sync_client.cli.logging'))
        stack.enter_context(patch('realman_sync_client.webapp.LOG'))
        return grid, hardware, transport, clients, real_client

    def invoke_cli(self, stack, hardware, transport, factory, *, cycles=1):
        stack.enter_context(patch.object(cli, 'InferenceHTTPClient', return_value=transport))
        stack.enter_context(patch.object(cli, 'MockHardware', return_value=hardware))
        stack.enter_context(patch.object(cli, 'SynchronousClient', side_effect=factory))
        with redirect_stdout(io.StringIO()):
            return cli.main(['--cycles', str(cycles), '--execute-steps', '1'])

    def test_cli_second_request_waits_for_residual_motion_and_full_stable_window(self):
        with ExitStack() as stack:
            _, hardware, transport, clients, factory = self.fixture(
                stack, residual_after_chunk=True)
            self.assertEqual(self.invoke_cli(stack, hardware, transport, factory, cycles=2), 0)

        self.assertEqual(len(transport.requests), 2)
        self.assertEqual(len(hardware.commands), 2)
        self.assertEqual(hardware.arm_count, 2)
        second = transport.requests[1]['observation']
        self.assertGreater(hardware.tail_end, hardware.stop_acks[1])
        self.assertGreaterEqual(second['stamp'] + 1e-8,
                                hardware.tail_end + clients[0].options.settle_window_s)
        self.assertAlmostEqual(math.degrees(second['state'][0]), 1.0, places=7)
        self.assertEqual(len(transport.released), 1)
        self.assertTrue(hardware.closed)
        self.assertFalse(hardware.armed)

    def test_cli_discards_result_when_joints_move_during_inference_before_arm(self):
        with ExitStack() as stack:
            _, hardware, transport, _, factory = self.fixture(stack)
            transport.displace_during_inference = True
            self.assertEqual(self.invoke_cli(stack, hardware, transport, factory), 1)

        self.assertEqual(len(transport.requests), 1)
        self.assertEqual(hardware.arm_count, 0)
        self.assertEqual(hardware.commands, [])
        self.assertTrue(hardware.closed)
        self.assertFalse(hardware.armed)
        self.assertEqual(len(transport.released), 1)

    def test_cli_rechecks_feedback_after_arm_before_first_command(self):
        with ExitStack() as stack:
            _, hardware, transport, _, factory = self.fixture(stack, move_on_arm=True)
            self.assertEqual(self.invoke_cli(stack, hardware, transport, factory), 1)

        self.assertEqual(len(transport.requests), 1)
        self.assertEqual(hardware.arm_count, 1)
        self.assertEqual(hardware.commands, [])
        self.assertFalse(hardware.armed)

    def test_dashboard_real_client_rejects_motion_during_inference_and_pauses(self):
        with ExitStack() as stack:
            _, hardware, transport, clients, factory = self.fixture(stack)
            transport.displace_during_inference = True
            dashboard = webapp.Dashboard(transport, mode='mock', execute_steps=1)
            dashboard.hardware = hardware
            dashboard.execute = True
            original_pause = dashboard._pause_failure

            def pause_and_finish(error):
                original_pause(error)
                dashboard.stop_event.set()

            stack.enter_context(patch.object(webapp, 'SynchronousClient', side_effect=factory))
            stack.enter_context(patch.object(dashboard, '_pause_failure', pause_and_finish))
            dashboard._run()

        self.assertEqual(len(clients), 1)
        self.assertEqual(len(transport.requests), 1)
        self.assertEqual(hardware.arm_count, 0)
        self.assertEqual(hardware.commands, [])
        self.assertFalse(dashboard.execute)
        self.assertFalse(dashboard.auto_infer)
        self.assertTrue(dashboard.inference['last_error'])
        self.assertTrue(hardware.closed)
        self.assertFalse(hardware.armed)
        self.assertEqual(len(transport.released), 1)

    def test_dashboard_second_request_also_waits_for_physical_settling(self):
        with ExitStack() as stack:
            _, hardware, transport, clients, factory = self.fixture(
                stack, residual_after_chunk=True)
            dashboard = webapp.Dashboard(transport, mode='mock', execute_steps=1)
            dashboard.hardware = hardware
            dashboard.execute = True
            original_stop = hardware.slow_stop

            def finish_after_two_chunks():
                original_stop()
                if len(hardware.commands) == 2:
                    dashboard.stop_event.set()

            stack.enter_context(patch.object(hardware, 'slow_stop', finish_after_two_chunks))
            stack.enter_context(patch.object(webapp, 'SynchronousClient', side_effect=factory))
            dashboard._run()

        self.assertEqual(len(clients), 1)
        self.assertEqual(len(transport.requests), 2)
        self.assertEqual(len(hardware.commands), 2)
        self.assertEqual(hardware.arm_count, 2)
        self.assertGreaterEqual(transport.requests[1]['observation']['stamp'] + 1e-8,
                                hardware.tail_end + clients[0].options.settle_window_s)
        self.assertAlmostEqual(math.degrees(transport.requests[1]['observation']['state'][0]),
                               1.0, places=7)
        self.assertIsNone(dashboard.inference['last_error'])
        self.assertTrue(hardware.closed)


if __name__ == '__main__':
    unittest.main()
