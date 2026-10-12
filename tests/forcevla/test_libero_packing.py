"""Exercise upstream preprocessing with independent, synthetic RGB cameras.

These CPU tests use the actual FastWAMProcessor and RobotVideoDataset._get.
They do not load data or weights, reproduce the packing implementation, or
invoke model training. VAE shape checks run its actual encoder on meta tensors.
"""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from omegaconf import OmegaConf
from torchvision.transforms import Resize

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from fastwam.datasets.lerobot.transforms.image import ToTensor


@pytest.fixture(scope="module", autouse=True)
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _raw_rgb_cameras():
    # Different recorded-camera stand-ins, with frame-dependent colors so
    # temporal decimation and swapped/duplicated cameras are observable.
    main_colors = torch.stack((torch.arange(33) * 4, torch.zeros(33), torch.full((33,), 255)), dim=1)
    wrist_colors = torch.stack((torch.full((33,), 255), 200 - torch.arange(33) * 4, torch.zeros(33)), dim=1)
    images = {name: colors.to(torch.uint8)[:, :, None, None].expand(-1, -1, 480, 640).clone()
              for name, colors in (("cam_main", main_colors), ("cam_wrist", wrist_colors))}
    return images, (main_colors, wrist_colors)


def _preprocess(images, *, image_steps=33):
    # Preserve Hydra's DictConfig/ListConfig types used by the real processor.
    shape_meta = OmegaConf.create({
        "images": [{"key": key, "raw_shape": [3, 480, 640], "shape": [3, 224, 224]}
                   for key in ("cam_main", "cam_wrist")],
        "action": [{"key": "default", "raw_shape": 7, "shape": 7}],
        "state": [{"key": "default", "raw_shape": 7, "shape": 7}],
    })
    transforms = [ToTensor(), Resize([224, 224])]
    processor = FastWAMProcessor(
        shape_meta=shape_meta, num_obs_steps=33, num_image_steps=image_steps,
        num_output_cameras=2, action_output_dim=7, proprio_output_dim=8,
        action_state_transforms=None, use_stepwise_action_norm=False,
        norm_default_mode="z-score", norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(state_target_dim=8),
        train_transforms=transforms, val_transforms=transforms,
        delta_action_dim_mask={"default": [False] * 7},
    ).train()
    processor.set_normalizer_from_stats({group: {"default": {
        "global_mean": torch.zeros(7), "global_std": torch.ones(7)}} for group in ("action", "state")})
    raw_action = torch.arange(32 * 7, dtype=torch.float32).reshape(32, 7) / 100
    raw_state = torch.arange(33 * 7, dtype=torch.float32).reshape(33, 7) / 100
    image_is_pad = torch.zeros(image_steps, dtype=torch.bool)
    image_is_pad[-1] = True
    raw = {"images": images, "action": {"default": raw_action.clone()},
           "state": {"default": raw_state.clone()}, "image_is_pad": image_is_pad,
           "action_is_pad": torch.zeros(32, dtype=torch.bool),
           "state_is_pad": torch.zeros(33, dtype=torch.bool), "task": "Synthetic test fixture", "idx": 0}
    return processor.preprocess(raw), raw_action, raw_state, shape_meta


def _official_dataset(monkeypatch, processed, shape_meta, *, presampled=False, layout="horizontal"):
    class FixtureReader:
        presample_images = presampled

        def __init__(self, **kwargs):
            pass

        def _set_return_images(self, flag):
            assert flag

        def __getitem__(self, index):
            assert index == 0
            return processed

    monkeypatch.setattr(RobotVideoDataset, "base_dataset_cls", FixtureReader)
    return RobotVideoDataset(
        dataset_dirs=["synthetic-test-only"], shape_meta=OmegaConf.create(shape_meta),
        num_frames=33, action_video_freq_ratio=4, video_size=[224, 448],
        concat_multi_camera=layout, processor=None, skip_padding_as_possible=False,
        use_text_embed_cache=False,
    )


@pytest.mark.parametrize("presampled", [False, True])
def test_official_libero_packing_has_correct_camera_order_time_and_normalization(monkeypatch, presampled):
    images, colors = _raw_rgb_cameras()
    if presampled:
        images = {key: value[0:33:4] for key, value in images.items()}
    processed, raw_action, raw_state, shape_meta = _preprocess(images, image_steps=9 if presampled else 33)
    dataset = _official_dataset(monkeypatch, processed, shape_meta, presampled=presampled)
    sample = dataset._get(0)

    assert sample["video"].shape == (3, 9, 224, 448)
    assert sample["video"].device.type == "cpu"
    # Float tensor resize has sub-ulp interpolation roundoff at RGB255.
    torch.testing.assert_close(torch.stack((sample["video"].min(), sample["video"].max())),
                               torch.tensor([-1., 1.]), atol=1e-6, rtol=0)
    for color, tile in zip(colors, (sample["video"][..., :224], sample["video"][..., 224:])):
        expected = (color[0:33:4].to(torch.float32) / 255 - 0.5) / 0.5
        # Both entire tiles and their shared seam must retain the right camera.
        torch.testing.assert_close(tile, expected.T[:, :, None, None].expand_as(tile), atol=1e-6, rtol=0)
    assert sample["image_is_pad"].tolist() == [False] * 8 + [True]
    assert sample["action"].shape == (32, 7)
    assert sample["proprio"].shape == (32, 8)
    torch.testing.assert_close(sample["action"], raw_action, atol=0, rtol=0)
    torch.testing.assert_close(sample["proprio"][:, :7], raw_state[:-1], atol=0, rtol=0)
    assert torch.equal(sample["proprio"][:, 7], torch.zeros(32))


@pytest.mark.parametrize("camera,affected,unaffected", [(0, slice(None, 224), slice(224, None)),
                                                       (1, slice(224, None), slice(None, 224))])
def test_replacing_one_independent_camera_only_changes_its_own_tile(monkeypatch, camera, affected, unaffected):
    images, _ = _raw_rgb_cameras()
    processed, _, _, shape_meta = _preprocess(images)
    dataset = _official_dataset(monkeypatch, processed, shape_meta)
    reference = dataset._get(0)
    changed = deepcopy(processed)
    changed["pixel_values"][camera] = 0.25
    output = _official_dataset(monkeypatch, changed, shape_meta)._get(0)
    torch.testing.assert_close(output["video"][..., unaffected], reference["video"][..., unaffected], atol=0, rtol=0)
    assert not torch.equal(output["video"][..., affected], reference["video"][..., affected])
    torch.testing.assert_close(output["action"], reference["action"], atol=0, rtol=0)
    torch.testing.assert_close(output["proprio"], reference["proprio"], atol=0, rtol=0)


def test_a_missing_wrist_camera_is_rejected_by_actual_processor(monkeypatch):
    images, _ = _raw_rgb_cameras()
    del images["cam_wrist"]
    with pytest.raises(KeyError, match="cam_wrist"):
        _preprocess(images)


def test_actual_wan22_vae_meta_encoder_matches_libero_spatial_and_temporal_shape():
    from fastwam.models.wan22.wan_video_vae import WanVideoVAE38

    # Meta executes the actual encoder's shape operations without allocating
    # parameter/data storage, loading trained weights, or touching a GPU.
    with torch.device("meta"):
        vae = WanVideoVAE38()
        video = torch.empty(1, 3, 9, 224, 448)
        latents = vae.model.encode(video, vae.scale)
        observed = vae.model.encode(video[:, :, :1], vae.scale)
    assert vae.z_dim == 48
    assert vae.upsampling_factor == 16 and vae.temporal_downsample_factor == 4
    assert latents.shape == (1, 48, 3, 14, 28)
    assert observed.shape == (1, 48, 1, 14, 28)
    assert all(parameter.device.type == "meta" for parameter in vae.parameters())
