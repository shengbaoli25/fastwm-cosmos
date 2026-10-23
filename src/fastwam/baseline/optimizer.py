"""Preserve native dtype and bound ZeRO1 flats for Torch 2.7.1 foreach.

ForeachFunctors.cuh computes ``chunk_idx * chunk_size`` in int32 before
subtracting from its int64 remaining length. A large flat can therefore run
past its final partial chunk. Keep each group well below 2**31 elements;
single-GPU ZeRO1's alignment padding also stays within this even-valued cap.
"""
from __future__ import annotations


MAX_OPTIMIZER_GROUP_NUMEL = 1_000_000_000


def bounded_optimizer_param_groups(model, parameters, *, max_group_numel=MAX_OPTIMIZER_GROUP_NUMEL):
    """Group the exact requires-grad DiT/proprio list by dtype, then stable size.

    Every group contains only ``params`` and inherits all AdamW defaults. No
    parameter is split, duplicated, dropped or moved to another dtype/device.
    Full optimizer-state checkpoints require this same deterministic grouping.
    """
    if isinstance(max_group_numel, bool) or not isinstance(max_group_numel, int) or not 0 < max_group_numel < 2**31:
        raise ValueError("Optimizer group cap must be a positive integer below 2**31")
    parameters = list(parameters)
    expected = [parameter for parameter in model.dit.parameters() if parameter.requires_grad]
    if getattr(model, "proprio_encoder", None) is not None:
        expected.extend(parameter for parameter in model.proprio_encoder.parameters() if parameter.requires_grad)
    identifiers = [id(parameter) for parameter in parameters]
    expected_identifiers = [id(parameter) for parameter in expected]
    if (not identifiers or len(set(identifiers)) != len(identifiers) or
            len(set(expected_identifiers)) != len(expected_identifiers) or
            set(identifiers) != set(expected_identifiers)):
        raise ValueError("Optimizer groups must cover each official DiT/proprio parameter exactly once")
    by_dtype = {}
    for parameter in parameters:
        if parameter.numel() > max_group_numel:
            raise ValueError(f"Single parameter has {parameter.numel()} elements, exceeding optimizer group cap {max_group_numel}")
        by_dtype.setdefault(parameter.dtype, []).append(parameter)
    groups = []
    for same_dtype in by_dtype.values():
        current, count = [], 0
        for parameter in same_dtype:
            if current and count + parameter.numel() > max_group_numel:
                groups.append({"params": current})
                current, count = [], 0
            current.append(parameter)
            count += parameter.numel()
        if current:
            groups.append({"params": current})
    return groups
