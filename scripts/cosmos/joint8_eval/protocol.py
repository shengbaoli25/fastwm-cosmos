"""Evaluation scheduling and physical metrics; never join across capture gaps."""
import numpy as np


def anchors(segments, lookup, stride=32):
    return [lookup[ep, f] for ep, lo, hi in segments for f in range(lo, hi, stride)]


def channel_metrics(pred, target, valid):
    p, y, mask = np.asarray(pred), np.asarray(target), np.asarray(valid, bool)
    if p.shape != y.shape or mask.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Finite, aligned predictions, targets and channel masks required')
    n = mask.sum(tuple(range(mask.ndim - 1)))
    d = np.where(mask, p-y, 0)
    axes = tuple(range(d.ndim - 1))
    return dict(mae=(abs(d).sum(axes)/np.maximum(n, 1)).tolist(),
                rmse=np.sqrt((d*d).sum(axes)/np.maximum(n, 1)).tolist(), count=n.tolist())


def choose_episode(source):
    # Deterministic before loading either checkpoint: median length among episodes
    # with at least 90% contiguous valid training rows, otherwise maximum coverage.
    candidates = []
    for ep, n in enumerate(source.lengths):
        longest = max((hi-lo for e, lo, hi in source.segments if e == ep), default=0)
        candidates.append((ep, n, longest/n))
    good = [x for x in candidates if x[2] >= .9] or sorted(candidates, key=lambda x:-x[2])[:5]
    good.sort(key=lambda x:(x[1], x[0]))
    return good[len(good)//2][0]
