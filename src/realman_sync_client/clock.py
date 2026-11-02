"""One high-resolution monotonic clock for all local observations and control.

On Windows Python 3.10+, perf_counter uses the system-wide QPC clock, including
spawned camera workers. Python 3.12's monotonic can still use 15.625 ms ticks,
which cannot timestamp the 200 Hz force stream reliably. No clock is changed
globally; every client timestamp explicitly uses this module.
"""
import time


def now():
    return time.perf_counter()


def wait_until(deadline, stop_event, *, check=None):
    """Wait without early acquisition or a busy loop; check stop each <=5 ms slice.

    Windows Python 3.11+ sleep uses a high-resolution waitable timer. Event.wait
    can have coarse scheduling, so it must not determine a 30 Hz control period.
    Sleep is still a soft timer; callers retain lateness and freshness checks.
    """
    while not stop_event.is_set():
        if check is not None:
            check()
        remaining = deadline - now()
        if remaining <= 0:
            return True
        time.sleep(min(.005, remaining))
    return False
