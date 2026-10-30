"""CPU tests of timing semantics and observation-only input selection."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "backbone_benchmark", ROOT / "scripts/backbones/benchmark.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_statistics_have_explicit_interpolated_percentiles_and_call_hz():
    result = benchmark.latency_statistics([40., 10., 30., 20.])
    assert result["p50_ms"] == 25.
    assert result["p95_ms"] == 38.5
    assert result["mean_ms"] == 25.
    assert result["hz"] == 40.


@pytest.mark.parametrize("values", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_invalid_latency_samples_fail(values):
    with pytest.raises(ValueError, match="positive and finite"):
        benchmark.latency_statistics(values)


@pytest.mark.parametrize("args", [(0, 1, 0), (1, 0, 0), (1, 1, -1), (True, 1, 0)])
def test_invalid_measurement_counts_fail(args):
    with pytest.raises(ValueError):
        benchmark.validate_measurement(*args)


def test_warmup_is_not_counted_and_each_timed_call_is_synchronized():
    events = []
    clock_values = iter([1., 1.01, 2., 2.02])

    def run():
        events.append("run")
        return {"result": 1}

    timings, output = benchmark.measure(run, lambda: events.append("sync"),
        warmup=1, iterations=2, before_measure=lambda: events.append("reset"),
        clock=lambda: next(clock_values))
    assert timings == pytest.approx([10., 20.])
    assert output == {"result": 1}
    assert events == ["run", "sync", "sync", "reset", "sync", "run", "sync",
                      "sync", "run", "sync"]


def fake_dataset():
    # Each full frame carries its temporal index; future indexes are detectable.
    frames = torch.arange(12.).reshape(12, 1, 1, 1).expand(-1, 3, 2, 4).clone()
    source = SimpleNamespace(rows=[(7, 6, 100, 12)], bounds=[(0, 12)])
    source.decode_episode = lambda episode: frames if episode == 7 else None
    return SimpleNamespace(source=source, windows=[(0, 7, 6)]), frames


def test_rgb_context_contains_only_minus_four_and_current_frame():
    dataset, frames = fake_dataset()
    clip = benchmark.observed_clip(dataset, 0)
    assert clip.shape == (1, 3, 2, 2, 4)
    assert clip[0, :, 0].eq(2).all()
    assert clip[0, :, 1].eq(6).all()
    frames[7:] = 999
    assert torch.equal(clip, benchmark.observed_clip(dataset, 0))


def test_rgb_context_rejects_segment_boundary_and_invalid_indexes():
    dataset, _ = fake_dataset()
    dataset.source.bounds = [(3, 12)]
    with pytest.raises(ValueError, match="valid source segment"):
        benchmark.observed_clip(dataset, 0)
    for index in [-1, 1]:
        with pytest.raises(IndexError):
            benchmark.observed_clip(dataset, index)


def test_saved_run_identity_binding_rejects_template_and_changed_source():
    cfg = dict(teacher="vjepa21", data=dict(train=dict(source_sha256="a" * 64,
        expected_cache_identity="b" * 64, expected_teacher="vjepa21",
        expected_weight_sha256="c" * 64)), model=dict(manifest=dict(
        dataset_source_sha256="a" * 64, cache_identity_sha256="b" * 64,
        teacher=dict(name="vjepa21", weight_sha256="c" * 64), observation_offsets=[-4, 0])))
    benchmark.validate_bound_config(cfg, "a" * 64)
    with pytest.raises(ValueError, match="saved training config"):
        benchmark.validate_bound_config(cfg, "d" * 64)
    cfg["model"]["manifest"] = {}
    with pytest.raises(ValueError, match="saved training config"):
        benchmark.validate_bound_config(cfg, "a" * 64)
