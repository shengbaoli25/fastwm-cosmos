"""Passive, evaluation-only diagnostics for the action-conditioned JEPA reader.

These are reader cross-attention weights, not encoder self-attention, gradients,
or causal explanations. The two observed frames are one temporal tubelet: a
spatial heatmap must not be presented as independent attention to either frame.
No semantic contact region is inferred here. ROI labels must be independently
provided, and training samples support descriptive diagnostics only.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import torch


class AttentionCapture:
    """Capture each visual reader layer without changing its computation.

    ``with AttentionCapture(model.dit['jepa_reader']) as capture: ...``
    produces dictionaries containing CPU tensors ``attention [B,H,Q,N]``,
    ``valid [B,N]``, and ``action_clock [B]``, plus zero-based ``call_index``
    and ``layer_index``. Query projection outputs are captured after the actual
    normalization/linear operation, including autocast. QK softmax is recomputed
    in FP32 using SDPA's default 1/sqrt(head_dim) scale and boolean key mask.
    This is the mathematical weight matrix; fused SDPA rounding can differ.

    The horizon reader attends to eight *query features*, not image patches,
    and is intentionally excluded. Hooks retain no computational graph or K/V.
    Use one capture instance per model forward stream (not concurrent forwards).
    """

    def __init__(self, reader, *, max_records=4096):
        if not hasattr(reader, 'readers') or not len(reader.readers):
            raise ValueError('Expected an ActionGuidedVisualReader with visual layers')
        if isinstance(max_records, bool) or int(max_records) != max_records or max_records < 1:
            raise ValueError('max_records must be a positive integer')
        self.reader = reader
        self.max_records = int(max_records)
        self.records = []
        self._handles = []
        self._projected = {}
        self._call_index = -1
        self._clock = None

    def _begin(self, module, args):
        if len(args) < 2:
            raise ValueError('Capture expects reader(action, action_clock, observation)')
        self._call_index += 1
        self._clock = args[1].detach().float().cpu().clone()

    def _query_hook(self, index):
        def capture(module, args, output):
            self._projected[index] = output.detach()
        return capture

    def _reader_hook(self, index):
        def capture(module, args, output):
            if len(self.records) >= self.max_records:
                raise RuntimeError('AttentionCapture record limit reached; evaluate smaller batches')
            if len(args) != 3 or index not in self._projected or self._clock is None:
                raise RuntimeError('Unexpected VisualValueRead execution order or arguments')
            _, (key, _), valid = args
            projected = self._projected.pop(index)
            query = projected.unflatten(-1, (module.heads, projected.shape[-1] // module.heads))
            query = query.transpose(1, 2)
            if valid.dtype != torch.bool or valid.shape != (query.shape[0], key.shape[2]):
                raise ValueError('Expected a boolean per-example visual key validity mask')
            if not bool(valid.any(-1).all()):
                raise ValueError('Cannot capture attention with no valid visual keys')
            with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
                logits = torch.matmul(query.float(), key.detach().float().transpose(-2, -1))
                logits = logits / math.sqrt(query.shape[-1])
                logits = logits.masked_fill(~valid[:, None, None, :], -torch.inf)
                attention = logits.softmax(-1)
                if not bool(torch.isfinite(attention).all()):
                    raise ValueError('Nonfinite visual attention')
            self.records.append(dict(call_index=self._call_index, layer_index=index,
                attention=attention.cpu(), valid=valid.detach().cpu().clone(),
                action_clock=self._clock.clone()))
        return capture

    def __enter__(self):
        if self._handles:
            raise RuntimeError('AttentionCapture is already active')
        if self.reader.training:
            raise ValueError('AttentionCapture is evaluation-only; call model.eval() first')
        self.records.clear()
        self._projected.clear()
        self._call_index = -1
        self._clock = None
        self._handles.append(self.reader.register_forward_pre_hook(self._begin))
        for index, layer in enumerate(self.reader.readers):
            self._handles.append(layer.query.register_forward_hook(self._query_hook(index)))
            self._handles.append(layer.register_forward_hook(self._reader_hook(index)))
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._projected.clear()
        self._clock = None
        return False


def token_layout(token_count=392, observed_frame_indices=None, *, image_size=224):
    """Return the encoder's exact view/grid order and explicit temporal caveat.

    ``observed_frame_indices`` is an optional pair of actual source frame IDs,
    after segment-boundary clamping. The caller must get them from the dataset;
    simply subtracting four at a segment boundary gives incorrect provenance.
    Pixel boxes use half-open coordinates in the resized observation image.
    """
    if isinstance(token_count, bool) or int(token_count) != token_count or token_count < 2:
        raise ValueError('token_count must be a positive two-view square-grid count')
    side = math.isqrt(int(token_count) // 2)
    if token_count != 2 * side * side or image_size < 1 or image_size % side:
        raise ValueError('Expected two equal square grids dividing image_size')
    if observed_frame_indices is not None:
        if len(observed_frame_indices) != 2 or any(int(v) != v or v < 0 for v in observed_frame_indices):
            raise ValueError('Provide exactly two nonnegative observed frame indices')
        observed_frame_indices = [int(v) for v in observed_frame_indices]
        if observed_frame_indices[0] > observed_frame_indices[1]:
            raise ValueError('Observed frame pair must be chronological')
    patch = image_size // side
    tokens = []
    for camera_index, camera in enumerate(('cam_main', 'cam_wrist')):
        for row in range(side):
            for col in range(side):
                tokens.append(dict(token_index=len(tokens), camera_index=camera_index,
                    camera=camera, row=row, col=col,
                    pixel_box_xyxy=[col * patch, row * patch, (col + 1) * patch, (row + 1) * patch]))
    return dict(schema='jepa-reader-token-layout-v1', grid_shape=[2, side, side],
        image_size=[image_size, image_size], order='camera_main_then_wrist_row_major',
        temporal_aggregation='observed_two_frame_tubelet2',
        observed_frame_indices=observed_frame_indices,
        frame_specific_attribution_available=False, tokens=tokens)


def _broadcast_keys(mask, target, name):
    mask = torch.as_tensor(mask, device=target.device)
    if mask.ndim < 1 or mask.shape[-1] != target.shape[-1]:
        raise ValueError(f'{name} must have the same final token dimension as attention')
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(-2)
    try:
        return torch.broadcast_to(mask, target.shape)
    except RuntimeError as error:
        raise ValueError(f'{name} cannot broadcast to attention') from error


def attention_roi_metrics(attention, roi, valid=None):
    """Per-head/per-query statistics on a supplied binary patch contact mask.

    Inputs broadcast as [N], [B,N], or [B,H,Q,N]. Values are normalized over
    *all valid tokens across both cameras*, so equal-size view grids do not
    hide camera preference. Enrichment is ROI mass / valid ROI area fraction;
    a uniform map gives one. Empty ROIs have undefined enrichment (NaN), not
    evidence of failure. ROI validity must be reported separately by callers.
    """
    attention = torch.as_tensor(attention).detach().float()
    if attention.ndim < 1 or attention.shape[-1] < 1:
        raise ValueError('Expected attention with a nonempty final token axis')
    if not bool(torch.isfinite(attention).all()) or bool((attention < 0).any()):
        raise ValueError('Attention must be finite and nonnegative')
    valid = torch.ones_like(attention, dtype=torch.bool) if valid is None else _broadcast_keys(valid, attention, 'valid')
    roi = _broadcast_keys(roi, attention, 'roi')
    if valid.dtype != torch.bool or roi.dtype != torch.bool:
        raise ValueError('valid and roi must be boolean patch masks')
    if not bool(valid.any(-1).all()) or bool((attention.masked_select(~valid) > 1e-7).any()):
        raise ValueError('Attention must be zero on invalid keys with at least one valid key')
    if not torch.allclose(attention.sum(-1), torch.ones_like(attention.sum(-1)), atol=2e-5, rtol=2e-5):
        raise ValueError('Attention must sum to one over the token axis')
    count = valid.sum(-1)
    roi_count = (roi & valid).sum(-1)
    area = roi_count / count
    mass = (attention * (roi & valid)).sum(-1)
    enrichment = torch.where(roi_count > 0, mass / area.clamp_min(1e-12), torch.nan)
    peak = attention.argmax(-1, keepdim=True)
    pointing = (roi & valid).gather(-1, peak).squeeze(-1).float()
    entropy = -(attention * attention.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1)
    normalized_entropy = torch.where(count > 1, entropy / count.float().log().clamp_min(1e-12), 0.)
    return dict(roi_mass=mass, roi_area_fraction=area, roi_enrichment=enrichment,
        pointing_hit=pointing, entropy=entropy, normalized_entropy=normalized_entropy,
        valid_token_count=count, roi_token_count=roi_count, roi_available=roi_count > 0)


def paired_perturbation_metrics(baseline, perturbed, target=None, valid=None):
    """Normalized-action change and optional error change for one matched pair.

    Positive delta MSE means perturbation worsened prediction against a target.
    Output movement alone does not establish worse task performance. Inference
    noise and all other conditions must be identical across the pair.
    """
    baseline = torch.as_tensor(baseline).detach().float()
    perturbed = torch.as_tensor(perturbed, device=baseline.device).detach().float()
    if baseline.shape != perturbed.shape:
        raise ValueError('Paired predictions must have identical shape')
    mask = torch.ones_like(baseline, dtype=torch.bool) if valid is None else torch.as_tensor(valid, device=baseline.device)
    if mask.dtype != torch.bool:
        raise ValueError('Prediction validity must be boolean')
    while mask.ndim < baseline.ndim:
        mask = mask.unsqueeze(-1)
    mask = torch.broadcast_to(mask, baseline.shape)
    if not bool(mask.any()):
        raise ValueError('At least one valid predicted component is required')
    left, right = baseline[mask], perturbed[mask]
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ValueError('Predictions contain nonfinite valid components')
    difference = right - left
    result = dict(prediction_change_mse=float(difference.square().mean()),
        prediction_change_mae=float(difference.abs().mean()), valid_component_count=int(mask.sum()))
    if target is not None:
        target = torch.as_tensor(target, device=baseline.device).detach().float()
        if target.shape != baseline.shape or not bool(torch.isfinite(target[mask]).all()):
            raise ValueError('Target shape or valid values differ from prediction contract')
        expected = target[mask]
        base_error = (left - expected).square().mean()
        altered_error = (right - expected).square().mean()
        result.update(baseline_mse=float(base_error), perturbed_mse=float(altered_error),
            delta_mse=float(altered_error - base_error))
    return result


def cluster_bootstrap(records: Sequence[Mapping], value_key, *, cluster_key='episode_index',
                      n_bootstrap=2000, seed=20261113, confidence=.95):
    """Equal-weight episode mean and percentile bootstrap over whole episodes.

    Repeated frames/queries/seeds are averaged *within* each episode first.
    This interval describes the sampled trajectories; it does not turn training
    observations into a held-out evaluation. Missing/nonfinite values are
    excluded and explicitly counted. One cluster has no estimable interval.
    """
    if isinstance(n_bootstrap, bool) or int(n_bootstrap) != n_bootstrap or n_bootstrap < 1:
        raise ValueError('n_bootstrap must be a positive integer')
    if not 0 < confidence < 1:
        raise ValueError('confidence must lie between zero and one')
    grouped = defaultdict(list)
    excluded = 0
    for row in records:
        if cluster_key not in row:
            raise ValueError(f'Missing independent cluster ID: {cluster_key}')
        value = row.get(value_key)
        if value is None or not math.isfinite(float(value)):
            excluded += 1
            continue
        grouped[row[cluster_key]].append(float(value))
    means = np.asarray([np.mean(v) for v in grouped.values()], dtype=np.float64)
    result = dict(estimand='equal-weight mean of per-episode means', value_key=value_key,
        n_clusters=len(means), n_records=sum(map(len, grouped.values())), excluded_records=excluded,
        mean=float(means.mean()) if len(means) else None, ci_low=None, ci_high=None,
        confidence=confidence, bootstrap_seed=int(seed), n_bootstrap=int(n_bootstrap),
        interpretation='descriptive unless episodes are independently held out from training')
    if len(means) >= 2:
        generator = np.random.default_rng(seed)
        draws = means[generator.integers(0, len(means), size=(int(n_bootstrap), len(means)))].mean(1)
        alpha = (1 - confidence) / 2
        result['ci_low'], result['ci_high'] = map(float, np.quantile(draws, [alpha, 1 - alpha]))
    return result
