"""CPU-only cache bridge checks using the real small DiT, MoT, and FM loss.

Only the VAE is synthetic: it deterministically returns scaled 48-channel
latents. No pretrained weights, video files, CUDA, or training jobs are used.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


@pytest.fixture(scope="module", autouse=True)
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class _FixtureEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def encode(self, video, scale):
        self.calls.append((video.detach().clone(), scale))
        # CPU average pooling lacks a BF16 kernel; the fixture returns the
        # same latent dtype as its input after deterministic pooling.
        pooled = F.avg_pool3d(video[:, :, ::4].float(), kernel_size=(1, 16, 16)).to(video)
        mu = pooled.repeat(1, 16, 1, 1, 1)
        mean, inverse_std = [part.to(mu).reshape(1, 48, 1, 1, 1) for part in scale]
        return (mu - mean) * inverse_std


class _FixtureVAE(torch.nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 16
    z_dim = 48

    def __init__(self):
        super().__init__()
        self.model = _FixtureEncoder()
        self.register_buffer("mean", torch.linspace(-.2, .3, 48))
        self.register_buffer("inverse_std", torch.linspace(.8, 1.7, 48))

    @property
    def scale(self):
        return [self.mean, self.inverse_std]


@pytest.fixture
def model(monkeypatch):
    # Keep the original raw-video bridge active, but avoid its CUDA-only
    # torch.compile backend when exercising this CPU fixture encoder.
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    torch.manual_seed(42)
    video_expert = WanVideoDiT(
        hidden_dim=16, in_dim=48, ffn_dim=32, out_dim=48, text_dim=8,
        freq_dim=8, eps=1e-6, patch_size=(1, 2, 2), num_heads=2,
        attn_head_dim=8, num_layers=3, has_image_input=False,
        seperated_timestep=True, fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal", action_conditioned=False,
    )
    action_expert = ActionDiT(
        action_dim=7, hidden_dim=12, ffn_dim=24, num_heads=2,
        attn_head_dim=8, num_layers=3, freq_dim=8, text_dim=8, eps=1e-6,
    )
    return FastWAM(
        video_expert=video_expert, action_expert=action_expert,
        mot=MoT({"video": video_expert, "action": action_expert}),
        vae=_FixtureVAE(), text_dim=8, proprio_dim=8, device="cpu",
        torch_dtype=torch.float32, action_train_shift=1.,
    )


@pytest.fixture
def raw_sample():
    generator = torch.Generator().manual_seed(16)
    return {
        "video": torch.randn(2, 3, 9, 32, 32, generator=generator),
        "action": torch.randn(2, 32, 7, generator=generator),
        "proprio": torch.randn(2, 32, 8, generator=generator),
        "context": torch.randn(2, 3, 8, generator=generator),
        "context_mask": torch.tensor([[True, True, False], [True, False, False]]),
        # The second tail latent is fully padded in row 0. Row 1 has only
        # partial frame padding, which the stock all-of-four rule retains.
        "image_is_pad": torch.tensor([[False] * 5 + [True] * 4,
                                      [False, False, False, False, True,
                                       False, True, True, True]]),
        "action_is_pad": torch.tensor([[False] * 16 + [True] * 16,
                                       [False] * 28 + [True] * 4]),
    }


def _cached_sample(model, raw_sample):
    inputs = model.build_inputs(raw_sample)
    cached = {key: value for key, value in raw_sample.items() if key != "video"}
    cached.update(video_latents=inputs["input_latents"].detach().clone(),
                  video_num_frames=torch.tensor([9, 9]))
    return cached, inputs


def _forbid_encoding(monkeypatch, model):
    def forbidden(*args, **kwargs):
        raise AssertionError("Cached training must not call any VAE encoder.")
    monkeypatch.setattr(model, "_encode_video_latents", forbidden)
    monkeypatch.setattr(model.vae.model, "encode", forbidden)


def test_cached_inputs_match_raw_inputs_without_rgb_or_vae_calls(model, raw_sample, monkeypatch):
    cached, expected = _cached_sample(model, raw_sample)
    _forbid_encoding(monkeypatch, model)
    actual = model.build_inputs(cached)
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, atol=0, rtol=0)
        else:
            assert actual[key] == value
    assert actual["first_frame_latents"].shape == (2, 48, 1, 2, 2)
    assert actual["first_frame_latents"].data_ptr() == actual["input_latents"].data_ptr()
    assert actual["context"].shape == (2, 4, 8)  # Stock initial-proprio token.
    assert actual["context_mask"][:, -1].all()
    assert actual["image_is_pad"].shape == (2, 9)


def test_raw_video_path_still_encodes_scaled_input_and_preserves_tiled_error(model, raw_sample):
    actual = model.build_inputs(raw_sample)
    assert len(model.vae.model.calls) == 1
    encoded_video, scale = model.vae.model.calls[0]
    torch.testing.assert_close(encoded_video, raw_sample["video"], atol=0, rtol=0)
    assert encoded_video.device.type == "cpu" and encoded_video.dtype == model.torch_dtype
    assert scale[0] is model.vae.mean and scale[1] is model.vae.inverse_std
    assert actual["input_latents"].shape == (2, 48, 3, 2, 2)
    with pytest.raises(NotImplementedError, match="tiled"):
        model.build_inputs(raw_sample, tiled=True)


def test_cached_libero_shape_is_used_without_rescaling_or_input_gradients(model, raw_sample, monkeypatch):
    cached, _ = _cached_sample(model, raw_sample)
    latents = torch.linspace(-2., 3., 2 * 48 * 3 * 14 * 28).reshape(2, 48, 3, 14, 28)
    cached["video_latents"] = latents.requires_grad_()
    _forbid_encoding(monkeypatch, model)
    actual = model.build_inputs(cached)
    assert actual["input_latents"].shape == (2, 48, 3, 14, 28)
    torch.testing.assert_close(actual["input_latents"], latents, atol=0, rtol=0)
    assert not actual["input_latents"].requires_grad


@pytest.mark.parametrize("num_frames", [9, torch.tensor(9), torch.tensor([9, 9])])
def test_cached_frame_metadata_accepts_scalar_and_collated_integer_forms(model, raw_sample, num_frames):
    cached, _ = _cached_sample(model, raw_sample)
    cached["video_num_frames"] = num_frames
    assert model.build_inputs(cached)["input_latents"].shape[2] == 3


@pytest.mark.parametrize("value,match", [
    (None, "integer"), (9., "integer"), (True, "integer"),
    (torch.tensor(9.), "integer"), (torch.tensor([9, 5]), "equal"),
    (torch.tensor([[9, 9]]), "integer"), (torch.tensor([9, 9, 9]), "integer"),
    (8, "align"), (5, "temporal dimension"), (1, "temporal dimension"),
])
def test_cached_frame_metadata_rejects_missing_inconsistent_or_misaligned_values(model, raw_sample, value, match):
    cached, _ = _cached_sample(model, raw_sample)
    cached["video_num_frames"] = value
    with pytest.raises(ValueError, match=match):
        model.build_inputs(cached)


@pytest.mark.parametrize("replacement,match", [
    (torch.zeros(2, 48, 3, 2), "5D"),
    (torch.zeros(2, 47, 3, 2, 2), "48 channels"),
    (torch.zeros(0, 48, 3, 2, 2), "positive dimensions"),
    (torch.zeros(2, 48, 2, 2, 2), "temporal dimension"),
    (torch.zeros(2, 48, 3, 3, 2), "patch sizes"),
    (torch.zeros(2, 48, 3, 2, 2, dtype=torch.int64), "floating dtype"),
    (torch.full((2, 48, 3, 2, 2), float("nan")), "finite"),
    (torch.full((2, 48, 3, 2, 2), float("inf")), "finite"),
    (torch.full((2, 48, 3, 2, 2), 1e100, dtype=torch.float64), "remain finite"),
])
def test_malformed_cached_latents_are_rejected(model, raw_sample, replacement, match):
    cached, _ = _cached_sample(model, raw_sample)
    cached["video_latents"] = replacement
    with pytest.raises(ValueError, match=match):
        model.build_inputs(cached)


def test_cached_path_retains_stock_prompt_fallback_and_required_action(model, raw_sample, monkeypatch):
    cached, expected = _cached_sample(model, raw_sample)
    del cached["context"], cached["context_mask"]
    _forbid_encoding(monkeypatch, model)
    with pytest.raises(ValueError, match="context/context_mask.*prompt"):
        model.build_inputs(cached)
    prompts = ["First fixture instruction", "Second fixture instruction"]
    calls = []

    def encode_prompt(prompt):
        calls.append(prompt)
        return raw_sample["context"], raw_sample["context_mask"]

    monkeypatch.setattr(model, "encode_prompt", encode_prompt)
    cached["prompt"] = prompts
    actual = model.build_inputs(cached)
    assert calls == [prompts]
    torch.testing.assert_close(actual["context"], expected["context"], atol=0, rtol=0)
    torch.testing.assert_close(actual["context_mask"], expected["context_mask"], atol=0, rtol=0)
    del cached["action"]
    with pytest.raises(ValueError, match="action.*required"):
        model.build_inputs(cached)


@pytest.mark.parametrize("key,replacement,match", [
    ("action", torch.zeros(2, 31, 7), "divisible"),
    ("action_is_pad", torch.zeros(2, 31), "shape mismatch"),
    ("image_is_pad", torch.zeros(2, 3), "shape mismatch"),
    ("context_mask", None, "must both exist"),
    ("proprio", None, "required"),
])
def test_cached_path_preserves_stock_action_context_proprio_and_padding_validation(model, raw_sample, key, replacement, match):
    cached, _ = _cached_sample(model, raw_sample)
    cached[key] = replacement
    with pytest.raises(ValueError, match=match):
        model.build_inputs(cached)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_training_preserves_rng_noise_timesteps_outputs_losses_and_all_gradients(model, raw_sample, monkeypatch, dtype):
    # Native CPU kernels support BF16 backward on machines where oneDNN's
    # convolution backward does not; this does not affect model semantics.
    monkeypatch.setattr(torch.backends.mkldnn, "enabled", False)
    model.torch_dtype = dtype
    model.to(dtype=dtype)
    cached, _ = _cached_sample(model, raw_sample)
    snapshots = []
    original = model._joint_denoise_core

    def record_core(**kwargs):
        outputs = original(**kwargs)
        snapshots.append(({
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }, tuple(value.detach().clone() for value in outputs)))
        return outputs

    monkeypatch.setattr(model, "_joint_denoise_core", record_core)
    torch.manual_seed(739)
    expected_loss, expected_metrics = model.training_loss(raw_sample)
    expected_loss.backward()
    expected_gradients = {name: parameter.grad.detach().clone()
                          for name, parameter in model.named_parameters()
                          if parameter.grad is not None}
    expected_rng = torch.get_rng_state().clone()
    model.zero_grad(set_to_none=True)
    _forbid_encoding(monkeypatch, model)
    torch.manual_seed(739)
    actual_loss, actual_metrics = model.training_loss(cached)
    actual_loss.backward()

    torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
    assert actual_metrics == expected_metrics
    assert torch.equal(torch.get_rng_state(), expected_rng)
    actual_gradients = {name: parameter.grad for name, parameter in model.named_parameters()
                        if parameter.grad is not None}
    assert actual_gradients.keys() == expected_gradients.keys()
    assert expected_gradients and any(gradient.abs().sum() > 0 for gradient in expected_gradients.values())
    assert "proprio_encoder.weight" in expected_gradients
    for name, expected in expected_gradients.items():
        torch.testing.assert_close(actual_gradients[name], expected, atol=0, rtol=0)
    expected_core, expected_outputs = snapshots[0]
    actual_core, actual_outputs = snapshots[1]
    for key, value in expected_core.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual_core[key], value, atol=0, rtol=0)
        else:
            assert actual_core[key] == value
    for actual, expected in zip(actual_outputs, expected_outputs):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # The observed first frame remains clean; action tokens can only attend
    # that frame, and the first video frame cannot attend future frames.
    torch.testing.assert_close(actual_core["latents_video"][:, :, :1],
                               cached["video_latents"][:, :, :1], atol=0, rtol=0)
    mask = actual_core["attention_mask"]
    assert mask[3:, 0].all() and not mask[3:, 1:3].any()
    assert not mask[:3, 3:].any() and not mask[0, 1:3].any()


def test_cached_future_video_cannot_change_actions_through_real_mot_layers(model, raw_sample):
    cached, _ = _cached_sample(model, raw_sample)
    inputs = model.build_inputs(cached)
    latents = inputs["input_latents"]
    kwargs = dict(
        latents_action=inputs["action"], timestep_video=torch.tensor([.3, .5]),
        timestep_action=torch.tensor([.6, .2]), context=inputs["context"],
        context_mask=inputs["context_mask"],
        attention_mask=model._build_mot_attention_mask(3, 32, 1, torch.device("cpu")),
        fuse_vae_embedding_in_latents=True, action_condition=inputs["action"],
    )
    expected_video, expected_action = model._joint_denoise_core(latents_video=latents, **kwargs)
    changed = latents.clone()
    changed[:, :, 1:] = changed[:, :, 1:] * 7 + 5
    actual_video, actual_action = model._joint_denoise_core(latents_video=changed, **kwargs)
    torch.testing.assert_close(actual_action, expected_action, atol=0, rtol=0)
    torch.testing.assert_close(actual_video[:, :, :1], expected_video[:, :, :1], atol=0, rtol=0)
    _, observed_action = model._joint_denoise_core(latents_video=latents + 2, **kwargs)
    assert not torch.allclose(observed_action, expected_action)


def test_stock_video_loss_excludes_initial_frame_and_pools_raw_padding_by_four(model, raw_sample):
    # Excluding the clean first frame gives two latent loss steps. Fully
    # padded groups contribute zero, while a partially padded group stays.
    prediction = torch.empty(2, 48, 2, 2, 2)
    prediction[:, :, 0] = 2.
    prediction[:, :, 1] = 3.
    actual = model._compute_video_loss_per_sample(
        prediction, torch.zeros_like(prediction), raw_sample["image_is_pad"],
        include_initial_video_step=False,
    )
    torch.testing.assert_close(actual, torch.tensor([4., 6.5]), atol=0, rtol=0)
