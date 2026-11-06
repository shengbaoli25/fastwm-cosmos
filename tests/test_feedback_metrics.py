"""Diagnostic-only callback timing, with no native SDK or networking."""
import threading
import time
import unittest
from unittest.mock import patch

from realman_sync_client.feedback_metrics import FeedbackMetrics, HostSchedulerProbe


class FeedbackMetricsTests(unittest.TestCase):
    def test_gap_and_processing_are_independent(self):
        metrics = FeedbackMetrics(max_gap_s=.02)
        metrics.record(10., .1)
        metrics.record(10.005, 8.)
        metrics.record(10.055, .2)
        state = metrics.snapshot()
        self.assertEqual((state['callbacks'], state['gaps'], state['gaps_over_budget']), (3, 2, 1))
        self.assertAlmostEqual(state['arrival_gap_ms']['maximum'], 50.)
        self.assertEqual(state['processing_ms']['maximum'], 8.)
        self.assertAlmostEqual(state['last_gap_ms'], 50.)
        self.assertEqual(state['last_processing_ms'], .2)
        self.assertEqual(state['max_gap_ms'], 20.)

    def test_empty_and_first_sample_have_no_invented_arrival_gap(self):
        metrics = FeedbackMetrics()
        self.assertEqual(metrics.snapshot()['arrival_gap_ms'], dict(n=0, p50=None, p95=None, maximum=None))
        metrics.record(0., 0.)
        state = metrics.snapshot()
        self.assertEqual(state['callbacks'], 1)
        self.assertEqual(state['gaps'], 0)
        self.assertEqual(state['arrival_gap_ms']['n'], 0)
        self.assertEqual(state['processing_ms']['n'], 1)
        self.assertIsNone(state['last_gap_ms'])

    def test_invalid_record_preserves_every_existing_metric(self):
        metrics = FeedbackMetrics()
        metrics.record(10., .1)
        before = metrics.snapshot()
        for value in (-1, float('nan'), float('inf'), -float('inf'), True, '10', None):
            for arrival, duration in ((value, .1), (11., value)):
                with self.subTest(arrival=arrival, duration=duration), self.assertRaises(ValueError):
                    metrics.record(arrival, duration)
                self.assertEqual(metrics.snapshot(), before)
        for duplicate in (10., 9.9):
            with self.assertRaises(ValueError):
                metrics.record(duplicate, .1)
            self.assertEqual(metrics.snapshot(), before)
        metrics.record(10.005, .1)
        self.assertEqual(metrics.snapshot()['gaps_over_budget'], 0)

    def test_window_bounds_quantiles_but_preserves_total_counts(self):
        metrics = FeedbackMetrics(max_gap_s=.02, window=3)
        for arrival, duration in ((1., 99.), (1.1, 10.), (1.105, 1.), (1.110, 2.), (1.115, 3.)):
            metrics.record(arrival, duration)
        state = metrics.snapshot()
        self.assertEqual((state['callbacks'], state['gaps'], state['gaps_over_budget']), (5, 4, 1))
        self.assertEqual(state['processing_ms'], dict(n=3, p50=2., p95=3., maximum=3.))
        self.assertEqual(state['arrival_gap_ms']['n'], 3)
        self.assertAlmostEqual(state['arrival_gap_ms']['maximum'], 5.)
        # Returned values cannot mutate the producer's bounded history.
        state['processing_ms']['maximum'] = -1
        self.assertEqual(metrics.snapshot()['processing_ms']['maximum'], 3.)

    def test_configuration_is_finite_positive_and_bounded(self):
        for gap in (0, -.1, float('nan'), float('inf'), True, '.02'):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                FeedbackMetrics(max_gap_s=gap)
        for window in (0, -1, 65537, 1.5, True):
            with self.subTest(window=window), self.assertRaises(ValueError):
                FeedbackMetrics(window=window)

    def test_snapshot_is_consistent_with_concurrent_producer(self):
        metrics = FeedbackMetrics(window=8)
        done = threading.Event()
        def producer():
            for index in range(2000):
                metrics.record(1.+index*.005, .1)
            done.set()
        thread = threading.Thread(target=producer)
        thread.start()
        while not done.is_set():
            state = metrics.snapshot()
            self.assertEqual(state['gaps'], max(0, state['callbacks']-1))
            self.assertLessEqual(state['processing_ms']['n'], 8)
            self.assertLessEqual(state['arrival_gap_ms']['n'], 8)
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(metrics.snapshot()['callbacks'], 2000)


class HostSchedulerProbeTests(unittest.TestCase):
    def test_probe_measures_actual_wait_gaps_without_catchup(self):
        probe = HostSchedulerProbe(period_s=.005, max_gap_s=.03, window=2)
        timer = [10.]
        waits = []
        class StopEvent:
            delays = iter((.005, .050, .005))
            def wait(self, timeout):
                waits.append(timeout)
                try:
                    timer[0] += next(self.delays)
                    return False
                except StopIteration:
                    return True
        probe._stop_event = StopEvent()
        with patch('realman_sync_client.feedback_metrics.clock.now', side_effect=lambda: timer[0]):
            probe._run()
        result = probe.snapshot()
        self.assertEqual(waits, [.005]*4)
        self.assertEqual((result['ticks'], result['intervals'], result['gaps_over_budget']), (4, 3, 1))
        self.assertEqual(result['scheduled_gap_ms']['n'], 2)
        self.assertAlmostEqual(result['scheduled_gap_ms']['maximum'], 50.)
        self.assertAlmostEqual(result['last_gap_ms'], 5.)
        self.assertFalse(result['running'])

    def test_start_is_idempotent_and_close_interrupts_long_wait(self):
        probe = HostSchedulerProbe(period_s=10.)
        self.addCleanup(probe.close)
        probe.start()
        first_thread = probe._thread
        probe.start()
        self.assertIs(probe._thread, first_thread)
        self.assertTrue(probe.snapshot()['running'])
        started = time.perf_counter()
        self.assertTrue(probe.close())
        self.assertLess(time.perf_counter()-started, 1.)
        self.assertFalse(probe.snapshot()['running'])
        self.assertTrue(probe.close())
        # A fresh run must not count the deliberate shutdown gap as host lag.
        previous_metrics = probe._metrics
        probe.start()
        self.assertIsNot(probe._metrics, previous_metrics)
        self.assertIsNot(probe._thread, first_thread)
        self.assertTrue(probe.close())

    def test_unstarted_close_and_invalid_periods(self):
        probe = HostSchedulerProbe()
        self.assertTrue(probe.close())
        self.assertEqual(probe.snapshot()['ticks'], 0)
        for period in (0, -.1, float('nan'), float('inf'), True, '.005'):
            with self.subTest(period=period), self.assertRaises(ValueError):
                HostSchedulerProbe(period_s=period)

    def test_repeated_clock_does_not_create_a_zero_or_negative_gap(self):
        probe = HostSchedulerProbe()
        class StopEvent:
            calls = 0
            def wait(self, _timeout):
                self.calls += 1
                return self.calls > 2
        probe._stop_event = StopEvent()
        with patch('realman_sync_client.feedback_metrics.clock.now', side_effect=(10., 10., 9.)):
            probe._run()
        self.assertEqual(probe.snapshot()['intervals'], 0)
        self.assertEqual(probe.snapshot()['ticks'], 1)


if __name__ == '__main__':
    unittest.main()
