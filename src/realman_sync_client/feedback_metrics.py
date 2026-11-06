"""Bounded callback delivery/processing telemetry, never a sensor clock source.

Arrival gaps are measured at the Python callback boundary. They do not establish
when a packet reached the network adapter or whether UDP packets were lost.
"""
from collections import deque
import math
from numbers import Real
import threading

from . import clock


def _nonnegative(value, name):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f'{name} must be a finite nonnegative number')
    return float(value)


def _summary(values):
    values = sorted(values)
    if not values:
        return dict(n=0, p50=None, p95=None, maximum=None)
    middle = len(values)//2
    median = values[middle] if len(values)%2 else (values[middle-1]+values[middle])/2
    return dict(n=len(values), p50=median,
        p95=values[math.ceil(.95*len(values))-1], maximum=values[-1])


class FeedbackMetrics:
    """One ordered producer, concurrent readers; all durations are milliseconds."""

    def __init__(self, max_gap_s=.02, window=512):
        max_gap_s = _nonnegative(max_gap_s, 'max_gap_s')
        if max_gap_s == 0:
            raise ValueError('max_gap_s must be positive')
        if isinstance(window, bool) or not isinstance(window, int) or not 1 <= window <= 65536:
            raise ValueError('window must be an integer in 1..65536')
        self._lock = threading.Lock()
        self._window = window
        self._max_gap_s = max_gap_s
        self._arrival_gaps = deque(maxlen=window)
        self._processing = deque(maxlen=window)
        self._callbacks = self._gaps = self._gaps_over_budget = 0
        self._last_arrival = self._last_gap_ms = self._last_processing_ms = None

    def record(self, arrival, processing_ms):
        """Record supplied callback time, without rewriting it or filter state.

        Invalid or non-increasing times are rejected atomically. Callers can
        isolate ValueError from their sensor pipeline; metrics never authorize
        accepting an old sensor sample.
        """
        arrival = _nonnegative(arrival, 'arrival')
        processing_ms = _nonnegative(processing_ms, 'processing_ms')
        with self._lock:
            if self._last_arrival is not None:
                if arrival <= self._last_arrival:
                    raise ValueError('arrival must strictly increase')
                gap = arrival-self._last_arrival
                self._last_gap_ms = gap*1000
                self._arrival_gaps.append(self._last_gap_ms)
                self._gaps += 1
                self._gaps_over_budget += int(gap > self._max_gap_s)
            self._callbacks += 1
            self._last_arrival, self._last_processing_ms = arrival, processing_ms
            self._processing.append(processing_ms)

    def snapshot(self):
        with self._lock:
            gaps, processing = tuple(self._arrival_gaps), tuple(self._processing)
            result = dict(callbacks=self._callbacks, gaps=self._gaps,
                gaps_over_budget=self._gaps_over_budget, max_gap_ms=self._max_gap_s*1000,
                window=self._window, last_arrival=self._last_arrival,
                last_gap_ms=self._last_gap_ms, last_processing_ms=self._last_processing_ms)
        result.update(arrival_gap_ms=_summary(gaps), processing_ms=_summary(processing),
            measurement_boundary='Python callback arrival; includes upstream delivery and host scheduling')
        return result


class HostSchedulerProbe:
    """Measure a sleeping Python thread, independent of hardware callbacks.

    Event.wait scheduling, Python/GIL availability and system load all affect
    this signal. It is not a packet timestamp or proof of network loss. The
    thread only waits and records timing; it never accesses a device.
    """

    def __init__(self, period_s=.005, max_gap_s=.03, window=512):
        self._period_s = _nonnegative(period_s, 'period_s')
        if self._period_s == 0:
            raise ValueError('period_s must be positive')
        self._metrics_args = dict(max_gap_s=max_gap_s, window=window)
        self._metrics = FeedbackMetrics(**self._metrics_args)
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def _run(self):
        self._metrics.record(clock.now(), 0.)
        while not self._stop_event.wait(self._period_s):
            # Same system-wide QPC clock as sensor callbacks. No inferred tick,
            # deadline catch-up, busy polling or manipulation of sensor time.
            now = clock.now()
            try:
                self._metrics.record(now, 0.)
            except ValueError:
                # A repeated/nonmonotonic injected clock cannot crash hardware.
                # Production perf_counter is monotonic; leave metrics unchanged.
                pass

    def start(self):
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._metrics = FeedbackMetrics(**self._metrics_args)
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run,
                name='realman-host-scheduler-probe', daemon=True)
            self._thread.start()

    def close(self, timeout_s=1.):
        timeout_s = _nonnegative(timeout_s, 'timeout_s')
        with self._lifecycle_lock:
            self._stop_event.set()
            thread = self._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=timeout_s)
            return thread is None or not thread.is_alive()

    def snapshot(self):
        with self._lifecycle_lock:
            metrics = self._metrics
            running = self._thread is not None and self._thread.is_alive()
        values = metrics.snapshot()
        return dict(running=running, period_ms=self._period_s*1000,
            ticks=values['callbacks'], intervals=values['gaps'],
            window=values['window'], max_gap_ms=values['max_gap_ms'],
            gaps_over_budget=values['gaps_over_budget'],
            last_tick_qpc=values['last_arrival'], last_gap_ms=values['last_gap_ms'],
            scheduled_gap_ms=values['arrival_gap_ms'],
            measurement_boundary='Sleeping Python thread; includes timer granularity, GIL and host scheduling; not NIC arrival')
