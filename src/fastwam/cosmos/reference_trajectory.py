"""CPU-only full-episode reference construction and augmentation.

References describe measured TCP motion, never the Joint8 command target.
Geometry is in the recorded TCP frame (metres, quaternion xyzw, gripper 0..1).
The caller supplies the exact valid runs from the production source adapter.
No operation interpolates across capture/reset discontinuities.
"""
from dataclasses import dataclass

import numpy as np


def euler_xyz_to_quaternion(angles):
    """Extrinsic xyz radians, Rz(z) @ Ry(y) @ Rx(x), as unit xyzw."""
    a = np.asarray(angles, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError('Expected finite Euler angles [N,3]')
    sx, sy, sz = np.sin(a / 2).T
    cx, cy, cz = np.cos(a / 2).T
    q = np.column_stack((sx*cy*cz-cx*sy*sz, cx*sy*cz+sx*cy*sz,
                         cx*cy*sz-sx*sy*cz, cx*cy*cz+sx*sy*sz))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _continuous_sign(q):
    q = q.copy()
    for j in range(1, len(q)):
        if np.dot(q[j-1], q[j]) < 0:
            q[j] *= -1
    return q


def quaternion_distance(a, b):
    """Shortest SO(3) angle; antipodal quaternions represent the same pose."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return 2*np.arccos(np.clip(np.abs(np.sum(a*b, axis=-1)), 0, 1))


def quaternion_slerp(a, b, fraction):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    t = np.asarray(fraction, dtype=np.float64)[..., None]
    if dot > .9995:
        out = (1-t)*a+t*b
    else:
        theta = np.arccos(np.clip(dot, -1, 1))
        out = (np.sin((1-t)*theta)*a+np.sin(t*theta)*b)/np.sin(theta)
    return out/np.linalg.norm(out, axis=-1, keepdims=True)


def _bounds(segments, n):
    raw = np.asarray(segments)
    if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) == 0:
        raise ValueError('At least one [lo, hi) continuous segment is required')
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError('Segment boundaries must be integers')
    bounds = raw.astype(np.int64, copy=True)
    if (bounds[:, 0] < 0).any() or (bounds[:, 1] > n).any():
        raise ValueError('Segment outside episode')
    if (bounds[:, 1]-bounds[:, 0] < 2).any():
        raise ValueError('Continuous segments must have at least two rows')
    if len(bounds) > 1 and (bounds[1:, 0] < bounds[:-1, 1]).any():
        raise ValueError('Segments must be ordered and nonoverlapping')
    return bounds


def build_reference(state20, segments, *, max_stride=12,
                    position_tolerance_m=.005, rotation_tolerance_rad=.05,
                    gripper_tolerance=.05):
    """Coarsen valid measured motion with bounded interpolation error.

    Each run keeps its first/last row and both sides of gripper events.
    Remaining samples are retained as necessary to bound position, rotation,
    gripper interpolation error and source-frame stride. No smoothing changes
    the retained samples. Output provenance is for supervision, not inputs.
    """
    state = np.asarray(state20, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 20 or not np.isfinite(state).all():
        raise ValueError('Expected finite measured Joint8 state [N,20]')
    if not isinstance(max_stride, (int, np.integer)) or isinstance(max_stride, bool) or max_stride < 1:
        raise ValueError('max_stride must be a positive integer')
    tolerances = np.asarray([position_tolerance_m, rotation_tolerance_rad, gripper_tolerance])
    if not np.isfinite(tolerances).all() or (tolerances <= 0).any():
        raise ValueError('All simplification tolerances must be positive and finite')
    grip = state[:, 19]
    if ((grip < 0) | (grip > 1)).any():
        raise ValueError('Measured gripper openness must be in [0,1]')
    bounds = _bounds(segments, len(state))
    xyz = state[:, 7:10]
    quat = euler_xyz_to_quaternion(state[:, 10:13])
    frames, segment_ids = [], []
    for sid, (lo, hi) in enumerate(bounds):
        quat[lo:hi] = _continuous_sign(quat[lo:hi])
        keep = {int(lo), int(hi-1)}
        changes = np.flatnonzero((np.abs(np.diff(grip[lo:hi])) >= gripper_tolerance)
                                 | (np.diff((grip[lo:hi] >= .5).astype(np.int8)) != 0)) + lo
        for j in changes:
            keep.update((int(j), int(j+1)))
        mandatory = sorted(keep)
        pending = list(zip(mandatory[:-1], mandatory[1:]))
        while pending:
            left, right = pending.pop()
            if right-left <= 1:
                continue
            idx = np.arange(left+1, right)
            t = (idx-left)/(right-left)
            interpolated = xyz[left]*(1-t[:, None])+xyz[right]*t[:, None]
            pos_error = np.linalg.norm(xyz[idx]-interpolated, axis=1)/position_tolerance_m
            rotation = quaternion_slerp(quat[left], quat[right], t)
            rot_error = quaternion_distance(quat[idx], rotation)/rotation_tolerance_rad
            g = grip[left]*(1-t)+grip[right]*t
            grip_error = np.abs(grip[idx]-g)/gripper_tolerance
            error = np.maximum.reduce((pos_error, rot_error, grip_error))
            if error.max() > 1:
                split = int(idx[np.argmax(error)])
            elif right-left > max_stride:
                split = (left+right)//2
            else:
                continue
            keep.add(split)
            pending.extend(((left, split), (split, right)))
        selected = sorted(keep)
        frames.extend(selected)
        segment_ids.extend([sid]*len(selected))
    frames = np.asarray(frames, dtype=np.int64)
    return dict(reference_pose=np.column_stack((xyz[frames], quat[frames], grip[frames])).astype(np.float32),
                source_frame=frames, segment_id=np.asarray(segment_ids, dtype=np.int64),
                segment_bounds=bounds)


def _validate_provenance(source_frame, segment_id, segment_bounds, *, allow_repeated=False):
    frames = np.asarray(source_frame)
    ids = np.asarray(segment_id)
    bounds = np.asarray(segment_bounds)
    if frames.ndim != 1 or ids.shape != frames.shape or not len(frames):
        raise ValueError('Reference frame/id arrays must be nonempty 1D arrays')
    if not np.issubdtype(frames.dtype, np.integer) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('Reference frame/id arrays must be integers')
    if bounds.ndim != 2 or bounds.shape[1] != 2 or not len(bounds):
        raise ValueError('Missing segment boundaries')
    bounds = _bounds(bounds, int(bounds[-1, 1]))
    invalid_order = np.diff(frames) < 0 if allow_repeated else np.diff(frames) <= 0
    if invalid_order.any() or (np.diff(ids) < 0).any():
        raise ValueError('Reference source rows must be ordered')
    if not np.array_equal(np.unique(ids), np.arange(len(bounds))):
        raise ValueError('Reference segment ids must cover all continuous segments')
    for sid, (lo, hi) in enumerate(bounds):
        chosen = frames[ids == sid]
        if len(chosen) < 2 or chosen[0] != lo or chosen[-1] != hi-1:
            raise ValueError('Reference must retain each segment endpoint')
        if ((chosen < lo) | (chosen >= hi)).any():
            raise ValueError('Reference source row outside its segment')
    return frames, ids, bounds


def alignment_target(source_frame, segment_id, segment_bounds, anchor_frame, *, valid_mask=None):
    """Soft correspondence over usable keys; duplicate hold slots share mass.

    Points need not have a one-to-one correspondence with source rows or action
    slots. After random deletion, neighbours interpolate the original anchor;
    after repetition, equivalent usable source coordinates share the mass.

    With valid_mask, masked keys receive exactly zero probability. Interpolate
    between the nearest usable source coordinates in the anchor's continuous
    segment. If only one side remains, clamp to its nearest usable coordinate;
    this is a reference-reading target, not a claim that progress has moved to
    that coordinate. If that entire segment is masked, return all zeros and
    disable the alignment loss. Never borrow keys across segment boundaries.
    """
    frames, ids, bounds = _validate_provenance(source_frame, segment_id, segment_bounds, allow_repeated=True)
    if valid_mask is None:
        valid = np.ones(len(frames), dtype=bool)
    else:
        valid = np.asarray(valid_mask)
        if valid.shape != frames.shape or valid.dtype != np.dtype(bool):
            raise ValueError('valid_mask must be a Boolean array with one entry per reference slot')
    if not isinstance(anchor_frame, (int, np.integer)) or isinstance(anchor_frame, bool):
        raise ValueError('anchor_frame must be an original integer episode row')
    matching = np.flatnonzero((bounds[:, 0] <= anchor_frame) & (anchor_frame < bounds[:, 1]))
    if len(matching) != 1:
        raise ValueError('Anchor is outside valid continuous segments')
    slots = np.flatnonzero((ids == matching[0]) & valid)
    target = np.zeros(len(frames), dtype=np.float32)
    if not len(slots):
        return target
    local_frames = frames[slots]
    read_frame = np.clip(anchor_frame, local_frames[0], local_frames[-1])
    right = int(np.searchsorted(local_frames, read_frame))
    if local_frames[right] == read_frame:
        same = slots[local_frames == read_frame]
        target[same] = 1/len(same)
    else:
        left = right-1
        weight = (read_frame-local_frames[left])/(local_frames[right]-local_frames[left])
        left_slots = slots[local_frames == local_frames[left]]
        right_slots = slots[local_frames == local_frames[right]]
        target[left_slots] = (1-weight)/len(left_slots)
        target[right_slots] = weight/len(right_slots)
    return target


@dataclass(frozen=True)
class AugmentationConfig:
    """Starting settings, not task-calibrated accuracy or safety thresholds."""
    position_std_m: float = .005
    rotation_std_rad: float = .03
    point_mask_probability: float = .05
    span_mask_probability: float = .25
    max_span_fraction: float = .2
    dropout_probability: float = .1
    control_points: int = 4
    preserve_segment_endpoints: bool = True
    min_keep_fraction: float = .45
    max_keep_fraction: float = 1.0
    duplicate_probability: float = .15
    max_extra_repeats: int = 2
    gripper_event_threshold: float = .05


def _multiply_quaternions(a, b):
    v = a[:, 3, None]*b[:, :3]+b[:, 3, None]*a[:, :3]+np.cross(a[:, :3], b[:, :3])
    w = a[:, 3]*b[:, 3]-np.sum(a[:, :3]*b[:, :3], axis=1)
    return np.column_stack((v, w))


def augment_reference(reference, rng=None, *, config=None, anchor_frame=None):
    """Return (model_inputs, supervision), keeping provenance out of inputs.

    Random point retention and repeated interior points vary reference length
    and local sampling pace. Segment endpoints and gripper-event neighbours
    remain present before optional geometry masking. Smooth perturbations are
    independent within each continuous segment.
    The default retains segment endpoints and gripper values. Local point/span
    masks retain reference slots/order, with explicit validity flags. Alignment
    supervision uses only unmasked keys in the anchor's segment. A dropped
    segment/reference disables alignment loss; task/goal context must be supplied
    separately by the caller. Reuse one sampled augmentation over a temporal
    training window instead of changing it at every observed frame.
    """
    cfg = AugmentationConfig() if config is None else config
    rng = np.random.default_rng() if rng is None else rng
    if not isinstance(rng, np.random.Generator):
        raise TypeError('rng must be a numpy.random.Generator')
    if not all(np.isfinite(x) and x >= 0 for x in (cfg.position_std_m, cfg.rotation_std_rad)):
        raise ValueError('Noise scales must be finite and nonnegative')
    if not all(np.isfinite(x) and 0 <= x <= 1 for x in
               (cfg.point_mask_probability, cfg.span_mask_probability, cfg.dropout_probability,
                cfg.duplicate_probability)):
        raise ValueError('Mask/dropout probabilities must be in [0,1]')
    if not np.isfinite(cfg.max_span_fraction) or not 0 < cfg.max_span_fraction <= 1:
        raise ValueError('max_span_fraction must be in (0,1]')
    if not isinstance(cfg.control_points, int) or cfg.control_points < 2:
        raise ValueError('At least two noise control points are required')
    if not (0 < cfg.min_keep_fraction <= cfg.max_keep_fraction <= 1):
        raise ValueError('Require 0 < min_keep_fraction <= max_keep_fraction <= 1')
    if not isinstance(cfg.max_extra_repeats, int) or cfg.max_extra_repeats < 1:
        raise ValueError('max_extra_repeats must be a positive integer')
    if not np.isfinite(cfg.gripper_event_threshold) or cfg.gripper_event_threshold <= 0:
        raise ValueError('gripper_event_threshold must be positive and finite')
    frames, ids, bounds = _validate_provenance(reference['source_frame'], reference['segment_id'],
                                             reference['segment_bounds'])
    original = np.asarray(reference['reference_pose'], dtype=np.float64)
    if original.shape != (len(frames), 8) or not np.isfinite(original).all():
        raise ValueError('Expected finite reference_pose [M,8]')
    if not np.allclose(np.linalg.norm(original[:, 3:7], axis=1), 1, atol=1e-5):
        raise ValueError('Reference quaternions must be normalized')
    if ((original[:, 7] < 0) | (original[:, 7] > 1)).any():
        raise ValueError('Reference gripper must be in [0,1]')
    # Nonuniform source-coordinate retention: no fixed count, no fixed ratio
    # to either the dense GT episode or the WAM's 32-action chunk.
    selected = []
    for sid in range(len(bounds)):
        slots = np.flatnonzero(ids == sid)
        g = original[slots, 7]
        changes = np.flatnonzero((np.abs(np.diff(g)) >= cfg.gripper_event_threshold)
                                 | (np.diff((g >= .5).astype(np.int8)) != 0))
        required = {0, len(slots)-1}
        for j in changes:
            required.update((int(j), int(j+1)))
        fraction = rng.uniform(cfg.min_keep_fraction, cfg.max_keep_fraction)
        for local, index in enumerate(slots):
            if local not in required and rng.random() >= fraction:
                continue
            selected.append(int(index))
            if 0 < local < len(slots)-1 and rng.random() < cfg.duplicate_probability:
                selected.extend([int(index)]*int(rng.integers(1, cfg.max_extra_repeats+1)))
    selected = np.asarray(selected, dtype=np.int64)
    original = original[selected]
    frames, ids = frames[selected], ids[selected]
    pose = original.copy()
    valid = np.ones(len(frames), dtype=bool)
    for sid, (lo, hi) in enumerate(bounds):
        slots = np.flatnonzero(ids == sid)
        phase = (frames[slots]-lo)/(hi-1-lo)
        knots = np.linspace(0, 1, cfg.control_points)
        noise = rng.normal(size=(cfg.control_points, 6))
        noise[:, :3] *= cfg.position_std_m
        noise[:, 3:] *= cfg.rotation_std_rad
        if cfg.preserve_segment_endpoints:
            noise[[0, -1]] = 0
        smooth = np.column_stack([np.interp(phase, knots, noise[:, c]) for c in range(6)])
        pose[slots, :3] += smooth[:, :3]
        rv = smooth[:, 3:]
        angle = np.linalg.norm(rv, axis=1)
        scale = .5*np.sinc(angle/(2*np.pi))
        delta_q = np.column_stack((rv*scale[:, None], np.cos(angle/2)))
        pose[slots, 3:7] = _continuous_sign(_multiply_quaternions(delta_q, pose[slots, 3:7]))
        valid[slots] &= rng.random(len(slots)) >= cfg.point_mask_probability
        if rng.random() < cfg.span_mask_probability:
            length = int(rng.integers(1, max(1, int(np.ceil(len(slots)*cfg.max_span_fraction)))+1))
            first = int(rng.integers(0, len(slots)-length+1))
            valid[slots[first:first+length]] = False
        if cfg.preserve_segment_endpoints:
            valid[slots[[0, -1]]] = True
    dropped = bool(rng.random() < cfg.dropout_probability)
    if dropped:
        valid[:] = False
    position_error = np.linalg.norm(pose[:, :3]-original[:, :3], axis=1).astype(np.float32)
    rotation_error = quaternion_distance(pose[:, 3:7], original[:, 3:7]).astype(np.float32)
    # Explicit mask is mandatory. Placeholder quaternion is a valid identity,
    # not a zero rotation vector misread as a genuine recorded orientation.
    pose[~valid] = 0
    pose[~valid, 6] = 1
    inputs = dict(reference_pose=pose.astype(np.float32), reference_valid=valid,
                  reference_channel_valid=np.repeat(valid[:, None], 3, axis=1),
                  reference_order=np.linspace(0, 1, len(frames), dtype=np.float32),
                  reference_break_before=np.r_[True, ids[1:] != ids[:-1]],
                  reference_present=np.asarray(valid.any(), dtype=bool))
    supervision = dict(source_frame=frames.copy(), segment_id=ids.copy(), segment_bounds=bounds.copy(),
                       reference_position_error=position_error, reference_rotation_error=rotation_error,
                       alignment_enabled=np.asarray(valid.any(), dtype=bool))
    if anchor_frame is not None:
        target = alignment_target(frames, ids, bounds, anchor_frame, valid_mask=valid)
        # The unmasked target is diagnostic provenance only. It must not be
        # used as the cross-attention target when masked keys are inaccessible.
        supervision.update(alignment_target=target,
                           alignment_target_unmasked=alignment_target(frames, ids, bounds, anchor_frame),
                           alignment_enabled=np.asarray(target.sum() > 0, dtype=bool))
    return inputs, supervision
