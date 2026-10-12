"""The original PDF's Force [H, Uc, W] visibility, independent of its loss.

``True`` means allowed in PyTorch SDPA. H reads causal H, Uc reads only
bidirectional Uc, and W reads H/Uc and causal W. Only W may cross-attend generated video;
the video mask consequently has W query rows, never H or Uc rows. Under the
current contract, Uc is encoded from the current noisy Action sample using its
clock. Its Force-internal hidden stream remains separate from the Action Expert
and cannot absorb W or generated video at later layers. The mask itself has no
timestep or objective and never introduces a hold/reference/clean-GT input.

Padding is per sample. Unavailable tokens are excluded as KV for every valid
query. An unavailable query gets only its own diagonal as a numerical sentinel,
so no self-attention row is fully masked. Its output must be discarded (or
zeroed) and must never be used as a valid token, target, or cached observation.
Channel validity must also be applied before projection with
``mask_invalid_channels``; a token mask cannot hide part of a projected vector.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _padding(name: str, value: Tensor) -> Tensor:
    if (not isinstance(value, Tensor) or value.dtype != torch.bool
            or value.ndim != 2 or min(value.shape) < 1):
        raise ValueError(f"{name} must be nonempty bool [B,T]")
    return value


def _same_batch_device(reference: Tensor, name: str, value: Tensor) -> None:
    if value.shape[0] != reference.shape[0] or value.device != reference.device:
        raise ValueError(f"{name} batch/device differs")


def _channel_validity(name: str, padding: Tensor, channels: Tensor | None) -> Tensor | None:
    if channels is None:
        return None
    if (not isinstance(channels, Tensor) or channels.dtype != torch.bool
            or channels.ndim != 3 or tuple(channels.shape[:2]) != tuple(padding.shape)
            or channels.shape[-1] < 1 or channels.device != padding.device):
        raise ValueError(f"{name} must be bool [B,T,D] on the padding device")
    return channels & ~padding[..., None]


def _token_validity(name: str, padding: Tensor, channels: Tensor | None) -> Tensor:
    channel_valid = _channel_validity(name, padding, channels)
    return ~padding if channel_valid is None else channel_valid.any(dim=-1)


def mask_invalid_channels(
    values: Tensor, is_pad: Tensor, channel_valid: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return zeroed unavailable inputs and their explicit channel validity.

    Apply before history/candidate/future projection. Pass returned validity to
    the corresponding mask argument and encode it separately when missing
    channels occur; zero is a placeholder, not a measured zero. Valid values
    must be finite; NaN/Inf in unavailable channels are removed with ``where``.
    """
    padding = _padding("is_pad", is_pad)
    if (not isinstance(values, Tensor) or not values.is_floating_point()
            or values.ndim != 3 or tuple(values.shape[:2]) != tuple(padding.shape)
            or values.shape[-1] < 1 or values.device != padding.device):
        raise ValueError("values must be floating [B,T,D] on the padding device")
    valid = _channel_validity("channel_valid", padding, channel_valid)
    if valid is None:
        valid = (~padding[..., None]).expand_as(values)
    elif valid.shape != values.shape:
        raise ValueError("channel_valid must match values")
    cleaned = torch.where(valid, values, torch.zeros_like(values))
    if not bool(torch.isfinite(cleaned).all()):
        raise ValueError("Available input channels must be finite")
    return cleaned, valid.clone()


def build_force_attention_mask(
    history_is_pad: Tensor,
    candidate_is_pad: Tensor,
    future_is_pad: Tensor,
    *,
    history_channel_valid: Tensor | None = None,
    candidate_channel_valid: Tensor | None = None,
    future_channel_valid: Tensor | None = None,
) -> Tensor:
    """Build bool [B,1,H+Uc+W,H+Uc+W] with PDF history/action isolation plus causal W visibility.

    Candidate and future horizons must match. Partial-channel tokens remain
    available; a token with no available channels is treated as unavailable.
    Padding tensors and validity are read without mutation. There is no Action
    Expert stream or timestep in this function.

    For shifted autoregressive W tokens, ``future_is_pad`` / ``future_channel_valid`` describe
    only a deployment-known requested query horizon, normally all available.
    Never derive them from future target availability or GT ``wrench_valid``;
    those masks belong exclusively to the Force loss/evaluation path.
    """
    h_pad = _padding("history_is_pad", history_is_pad)
    u_pad = _padding("candidate_is_pad", candidate_is_pad)
    w_pad = _padding("future_is_pad", future_is_pad)
    _same_batch_device(h_pad, "candidate_is_pad", u_pad)
    _same_batch_device(h_pad, "future_is_pad", w_pad)
    if u_pad.shape[1] != w_pad.shape[1]:
        raise ValueError("Candidate and future horizons must match")
    h_valid = _token_validity("history_channel_valid", h_pad, history_channel_valid)
    u_valid = _token_validity("candidate_channel_valid", u_pad, candidate_channel_valid)
    w_valid = _token_validity("future_channel_valid", w_pad, future_channel_valid)
    valid = torch.cat((h_valid, u_valid, w_valid), dim=1)
    h = h_pad.shape[1]
    u, w = u_pad.shape[1], w_pad.shape[1]
    size = h + u + w
    topology = torch.zeros((size, size), dtype=torch.bool, device=h_pad.device)
    topology[:h, :h] = torch.ones((h, h), dtype=torch.bool, device=h_pad.device).tril()
    topology[h:h + u, h:h + u] = True
    topology[h + u:, :h + u] = True
    topology[h + u:, h + u:] = torch.ones((w, w), dtype=torch.bool, device=h_pad.device).tril()
    allowed = topology[None] & valid[:, None, :] & valid[:, :, None]
    # Each available H query can read itself. Uc/W have the same property.
    # Unavailable query outputs are never visible to available queries.
    diagonal = torch.eye(size, dtype=torch.bool, device=h_pad.device)
    allowed |= diagonal[None] & ~valid[:, :, None]
    return allowed[:, None]


def build_force_video_attention_mask(
    future_is_pad: Tensor,
    video_is_pad: Tensor,
    *,
    future_channel_valid: Tensor | None = None,
) -> Tensor:
    """Build bool [B,1,W,V] for W-only generated-video cross-attention.

    Call cross-attention on the W slice and splice its residual back. Supplying
    H/Uc rows, or applying video cross-attention to the whole Force sequence,
    violates the PDF. Every sample must have at least one available video KV.
    Unavailable W queries use those valid KV for numerical safety; discard their
    outputs and exclude them from self-attention KV/loss with their validity.
    W query validity is deployment-known, never inferred from Force GT validity.
    """
    w_pad = _padding("future_is_pad", future_is_pad)
    v_pad = _padding("video_is_pad", video_is_pad)
    _same_batch_device(w_pad, "video_is_pad", v_pad)
    _token_validity("future_channel_valid", w_pad, future_channel_valid)
    if not bool((~v_pad).any(dim=-1).all()):
        raise ValueError("Each sample requires an available video key/value")
    return (~v_pad)[:, None, None, :].expand(-1, 1, w_pad.shape[1], -1).clone()
