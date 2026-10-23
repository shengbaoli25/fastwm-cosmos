"""Physical-unit errors with explicit masks and count-weighted aggregation."""
import numpy as np


def _nullable(values, count):
    return [float(v) if n else None for v, n in zip(values, count)]


def _finish(count, absolute, squared, maximum):
    denominator = np.maximum(count, 1)
    total = int(count.sum())
    return dict(
        valid_per_channel=count.tolist(),
        absolute_error_sum=absolute.tolist(), squared_error_sum=squared.tolist(),
        mae_per_channel=_nullable(absolute / denominator, count),
        rmse_per_channel=_nullable(np.sqrt(squared / denominator), count),
        max_abs_per_channel=_nullable(maximum, count),
        pooled_mae=float(absolute.sum() / total) if total else None,
        pooled_rmse=float(np.sqrt(squared.sum() / total)) if total else None,
        pooled_max_abs=float(maximum[count > 0].max()) if total else None,
    )


def error_stats(prediction, target, valid):
    """Missing GT is allowed only outside valid; predictions must all be finite.

    A wholly masked channel is reported as null, never as zero error. Pool only
    channels with the same physical unit (e.g. joints, not joints + gripper).
    """
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if p.ndim != 2 or p.shape != y.shape or mask.shape not in (p.shape, p.shape[:1]):
        raise ValueError('Prediction, target and validity shapes differ')
    if mask.ndim == 1:
        mask = np.broadcast_to(mask[:, None], p.shape)
    if not np.isfinite(p).all() or not np.isfinite(y[mask]).all():
        raise ValueError('Nonfinite prediction or valid GT')
    delta = np.zeros_like(p)
    np.subtract(p, y, out=delta, where=mask)
    return _finish(mask.sum(0), np.abs(delta).sum(0), np.square(delta).sum(0),
                   np.abs(delta).max(0, initial=0.))


def merge_stats(records):
    """Combine sufficient statistics, not per-window means/RMSEs."""
    if not records:
        raise ValueError('No metric records')
    widths = {len(r['valid_per_channel']) for r in records}
    if len(widths) != 1:
        raise ValueError('Metric channel counts differ')
    count = np.sum([r['valid_per_channel'] for r in records], axis=0)
    absolute = np.sum([r['absolute_error_sum'] for r in records], axis=0)
    squared = np.sum([r['squared_error_sum'] for r in records], axis=0)
    maximum = np.max([[0. if x is None else x for x in r['max_abs_per_channel']]
                      for r in records], axis=0)
    return _finish(count, absolute, squared, maximum)


def action_errors(prediction, target, valid):
    p, y = np.asarray(prediction), np.asarray(target)
    if p.ndim != 2 or p.shape[1] != 8 or p.shape != y.shape:
        raise ValueError('Joint8 predictions and targets required')
    mask = np.asarray(valid, dtype=bool)
    if mask.shape == p.shape[:1]:
        mask = np.broadcast_to(mask[:, None], p.shape)
    if mask.shape != p.shape:
        raise ValueError('Wrong action validity shape')
    # Both adjacent frames must be valid; never differentiate across padding.
    adjacent = mask[1:] & mask[:-1]
    return dict(
        joint_deg=error_stats(np.rad2deg(p[:, :7]), np.rad2deg(y[:, :7]), mask[:, :7]),
        gripper_openness=error_stats(p[:, 7:], y[:, 7:], mask[:, 7:]),
        joint_velocity_deg_s=error_stats(np.rad2deg(np.diff(p[:, :7], axis=0)) * 30,
            np.rad2deg(np.diff(y[:, :7], axis=0)) * 30, adjacent[:, :7]),
    )


def force_errors(prediction, target, valid):
    p, y, mask = np.asarray(prediction), np.asarray(target), np.asarray(valid, dtype=bool)
    if p.ndim != 2 or p.shape[1] != 6 or p.shape != y.shape or mask.shape != p.shape:
        raise ValueError('Force6 predictions, targets and channel masks required')
    return dict(force_N=error_stats(p[:, :3], y[:, :3], mask[:, :3]),
                torque_Nm=error_stats(p[:, 3:], y[:, 3:], mask[:, 3:]))
