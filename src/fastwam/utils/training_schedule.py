"""Resolve the optimizer schedule horizon independently of the stop limit."""
from numbers import Integral


def resolve_scheduler_total_steps(max_steps, scheduler_total_steps=None):
    """Missing/null keeps the historical schedule ending at ``max_steps``."""
    horizon = max_steps if scheduler_total_steps is None else scheduler_total_steps
    if (isinstance(horizon, bool) or not isinstance(horizon, Integral)
            or horizon < 1 or horizon < max_steps):
        raise ValueError('scheduler_total_steps must be an integer at least max_steps')
    return int(horizon)
