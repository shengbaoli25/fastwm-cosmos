"""Known-error examples for GT masks, physical units, and pooled statistics."""
import numpy as np
import pytest

from fastwam.cosmos.offline_metrics import action_errors, error_stats, force_errors, merge_stats


def test_masked_nan_and_missing_channel_do_not_dilute_error():
    result = error_stats([[1., 10., 1.], [3., 200., 2.]],
                         [[0., 0., np.nan], [0., np.nan, np.nan]],
                         [[True, True, False], [True, False, False]])
    assert result['valid_per_channel'] == [2, 1, 0]
    assert result['mae_per_channel'] == [2., 10., None]
    assert result['rmse_per_channel'][:2] == pytest.approx([np.sqrt(5), 10.])
    assert result['pooled_mae'] == pytest.approx(14 / 3)
    assert result['pooled_rmse'] == pytest.approx(np.sqrt(110 / 3))


def test_pooled_error_weights_frames_and_combines_squared_errors():
    short = error_stats([[10.]], [[0.]], [True])
    long = error_stats([[0.], [0.], [0.]], [[0.], [0.], [0.]], [True] * 3)
    result = merge_stats([short, long])
    assert result['pooled_mae'] == 2.5  # A mean of window MAEs would be 5.
    assert result['pooled_rmse'] == 5.
    assert result['pooled_max_abs'] == 10.
    assert result['valid_per_channel'] == [4]


def test_wholly_padded_targets_are_missing_not_perfect():
    result = error_stats(np.ones((2, 3)), np.full((2, 3), np.nan), [False, False])
    assert result['pooled_mae'] is None
    assert result['mae_per_channel'] == [None] * 3
    merged = merge_stats([result, result])
    assert merged['pooled_rmse'] is None


@pytest.mark.parametrize('prediction,target,mask', [
    ([[np.nan]], [[0.]], [False]),
    ([[0.]], [[np.nan]], [True]),
    ([[0., 1.]], [[0.]], [True]),
    ([[0.]], [[0.]], [[True, False]]),
])
def test_invalid_predictions_and_alignment_fail(prediction, target, mask):
    with pytest.raises(ValueError):
        error_stats(prediction, target, mask)


def test_joint_degrees_gripper_and_adjacent_velocity_remain_separate():
    prediction = np.zeros((3, 8))
    prediction[1, :7] = np.deg2rad(1.)
    prediction[2, :7] = 100.  # Padded: must not contribute a giant velocity.
    prediction[:2, 7] = .2
    result = action_errors(prediction, np.zeros((3, 8)), [True, True, False])
    assert result['joint_deg']['pooled_mae'] == pytest.approx(.5)
    assert result['gripper_openness']['pooled_mae'] == pytest.approx(.2)
    assert result['joint_velocity_deg_s']['pooled_mae'] == pytest.approx(30.)
    assert result['joint_velocity_deg_s']['valid_per_channel'] == [1] * 7


def test_force_and_torque_units_and_channel_masks_stay_separate():
    prediction = np.array([[1., 2., 999., .1, .2, .3]])
    result = force_errors(prediction, np.zeros((1, 6)), [[True, True, False, True, True, True]])
    assert result['force_N']['pooled_mae'] == 1.5
    assert result['torque_Nm']['pooled_mae'] == pytest.approx(.2)
    assert result['force_N']['mae_per_channel'][2] is None
