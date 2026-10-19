import copy

import pytest
import torch

from conftest import sample
from test_joint import joint_model
from fastwam.cosmos.inference import JointInferenceRunner


@pytest.mark.parametrize("padded", [False, True])
def test_reused_inference_matches_reference_and_refreshes_conditions(padded):
    model = joint_model(False)
    model.generated_video_steps = 3
    data = sample(2)
    if padded:
        data["history_is_pad"][:, :20] = True
        data["history_channel_valid"][:, :20] = False
        data["history_channel_valid"][:, 21:25, :7] = False
        data["action_is_pad"][:, -3:] = True
    runner = JointInferenceRunner(model)
    rng = model.dit["runtime"].get_state()
    manifest = copy.deepcopy(model.manifest)
    for changed in [False, True]:
        if changed:
            data["first_frame_latents"] += 0.2
            data["history"] += 0.5
            data["proprio"] -= 0.3
            data["cosmos_context"] += 0.1
            data["action"] -= 0.2
        expected = model.infer_forcewm(data, action_condition=data["action"],
                                      num_inference_steps=3, seed=42, video_seed=44)
        actual = runner.infer_forcewm(data, action_condition=data["action"],
                                     num_inference_steps=3, seed=42, video_seed=44)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=2e-6)
    assert torch.equal(rng, model.dit["runtime"].get_state())
    assert manifest == model.manifest
    assert model.training


def test_invalid_inputs_still_fail_before_generating_actions():
    model = joint_model(False)
    data = sample(1)
    data["first_frame_latents"][0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite|Nonfinite"):
        JointInferenceRunner(model).infer_forcewm(data, action_condition=data["action"])
