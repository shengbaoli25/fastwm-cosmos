from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy"))

from realman_deploy.contracts import ActionChunk
from realman_deploy.queues import ActionTimeline, ChunkActionQueue


class ActionTimelineTests(unittest.TestCase):
    def chunk(self, sequence, stamp, epoch, value):
        rows = tuple((value + step / 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5)
                     for step in range(32))
        return ActionChunk(sequence, stamp, epoch, rows, stamp)

    def test_new_ae_prediction_keeps_only_unexpired_prefix_until_its_start(self):
        queue = ActionTimeline(30.0, max_actions=32)
        queue.replace(self.chunk(0, 10.0, 1, 1.0), 10.0)
        self.assertAlmostEqual(queue.take(10.0, 1).action[0], 1.0)
        queue.replace(self.chunk(1, 10.02, 1, 2.0), 10.03)
        prefix = queue.take(10.034, 1)
        self.assertIsNotNone(prefix)
        self.assertLess(prefix.action[0], 1.01)
        later = queue.take(10.054, 1)
        self.assertIsNotNone(later)
        self.assertGreaterEqual(later.action[0], 2.0)

    def test_epoch_invalidation_cannot_execute_old_action(self):
        queue = ActionTimeline(30.0)
        queue.replace(self.chunk(0, 10.0, 4, 1.0), 10.0)
        self.assertIsNone(queue.take(10.0, 5))
        queue.clear()
        self.assertEqual(len(queue), 0)

    def test_slow_result_drops_elapsed_actions_and_never_replays_them(self):
        queue = ActionTimeline(30.0)
        skipped, kept = queue.replace(self.chunk(0, 10.0, 1, 1.0), 10.2)
        self.assertEqual(skipped, 6)
        self.assertEqual(kept, 26)
        self.assertGreaterEqual(queue.take(10.2, 1).action[0], 1.006)


class RelaxedQueueTests(unittest.TestCase):
    chunk = ActionTimelineTests.chunk

    def test_delayed_result_keeps_all_actions_without_rewriting_source_time(self):
        for delay in (1.5, 5.0):
            with self.subTest(delay=delay):
                chunk = self.chunk(3, 10.0, 1, 1.0)
                strict = ActionTimeline()
                self.assertEqual(strict.replace(chunk, 10.0 + delay), (32, 0))
                queue = ChunkActionQueue()
                self.assertEqual(queue.replace(chunk, 10.0 + delay), (0, 32))
                self.assertEqual(chunk.source_stamp, 10.0)
                self.assertEqual(queue.start_stamp, 10.0 + delay)
                item = queue.take(10.0 + delay, 1)
                self.assertEqual(item.source_sequence, 3)
                self.assertEqual(item.action, chunk.actions[0])

    def test_paces_through_small_jitter_but_never_bursts_after_stall(self):
        queue = ChunkActionQueue()
        chunk = self.chunk(0, 10.0, 1, 1.0)
        queue.replace(chunk, 11.5)
        for i in range(8):
            now = 11.5 + i * queue.dt + (0.0004 if i % 2 else 0)
            self.assertEqual(queue.take(now, 1).action, chunk.actions[i])
            self.assertIsNone(queue.take(now, 1))
        # A long scheduling stall consumes only the next action; no catch-up.
        self.assertEqual(queue.take(15.0, 1).action, chunk.actions[8])
        self.assertIsNone(queue.take(15.0, 1))
        self.assertEqual(len(queue), 23)

    def test_replacement_is_bounded_and_fast_updates_do_not_starve_control(self):
        queue = ChunkActionQueue()
        queue.replace(self.chunk(0, 10.0, 1, 1.0), 11.5)
        queue.take(11.5, 1)
        queue.replace(self.chunk(1, 11.5, 1, 2.0), 11.51)
        queue.replace(self.chunk(2, 11.51, 1, 3.0), 11.52)
        self.assertEqual(len(queue), 32)
        self.assertIsNone(queue.take(11.52, 1))
        item = queue.take(11.5 + queue.dt, 1)
        self.assertEqual(item.source_sequence, 2)
        self.assertEqual(item.action[0], 3.0)

    def test_invalidated_epoch_clears_every_delayed_action(self):
        queue = ChunkActionQueue()
        queue.replace(self.chunk(0, 10.0, 1, 1.0), 11.5)
        self.assertIsNone(queue.take(11.5, 2))
        self.assertEqual(len(queue), 0)
        self.assertIsNone(queue.take(11.6, 1))


if __name__ == "__main__":
    unittest.main()
