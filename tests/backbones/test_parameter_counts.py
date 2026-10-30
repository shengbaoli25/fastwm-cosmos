"""Check failure-prone count semantics without constructing large real tensors."""
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "backbone_parameter_counter", ROOT / "scripts/backbones/count_parameters.py")
counter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(counter)


def test_shared_weights_count_once_and_buffers_are_excluded():
    model = nn.Module()
    model.first = nn.Linear(3, 2)
    model.second = nn.Linear(3, 2)
    model.second.weight = model.first.weight
    model.first.bias.requires_grad_(False)
    model.register_buffer("cache", torch.zeros(100))
    result = counter.parameter_summary(model)
    assert result["parameters"] == 10
    assert result["trainable_parameters"] == 8
    assert result["frozen_parameters"] == 2
    assert result["registered_buffer_elements"] == 100
    assert result["duplicate_parameter_aliases"] == [["first.weight", "second.weight"]]


class TinyTrainingModel(nn.Module):
    def __init__(self, bad_grouping=None):
        super().__init__()
        self.dit = nn.Linear(3, 2)
        self.teacher = nn.Linear(4, 3)
        self.bad_grouping = bad_grouping

    def optimizer_param_groups(self, parameters):
        if self.bad_grouping == "missing":
            parameters = parameters[:-1]
        elif self.bad_grouping == "duplicate":
            parameters = parameters + parameters[:1]
        elif self.bad_grouping == "frozen":
            parameters = parameters + list(self.teacher.parameters())
        return [{"params": parameters}]


def test_exact_trainer_freeze_excludes_teacher_from_optimizer():
    model = TinyTrainingModel()
    counter.apply_training_freeze(model)
    result = counter.verify_optimizer_coverage(model)
    assert result["parameters"] == 8
    assert not any(parameter.requires_grad for parameter in model.teacher.parameters())
    assert all(parameter.requires_grad for parameter in model.dit.parameters())


@pytest.mark.parametrize("mode", ["missing", "duplicate", "frozen"])
def test_incomplete_duplicate_or_frozen_optimizer_groups_fail(mode):
    model = TinyTrainingModel(mode)
    counter.apply_training_freeze(model)
    with pytest.raises(ValueError, match="exactly once"):
        counter.verify_optimizer_coverage(model)
