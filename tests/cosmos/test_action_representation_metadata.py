"""Checkpoint semantics cannot silently reinterpret the same eight output channels."""
from unittest.mock import patch

import pytest
import torch
from torch import nn

from fastwam.cosmos.assets import create_noaction_joint_cosmos
from fastwam.cosmos.joint_action import joint_action_contract
from fastwam.cosmos.noaction import NoActionJointModel


class TinyVideo(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.num_heads = 2
        self.num_layers = 1
        self.attn_head_dim = 4
        self.hidden_dim = 8


def model(representation='absolute', action_space='joint8'):
    net = nn.Module()
    block = nn.Module()
    block.cross_attn = nn.Module()
    block.cross_attn.k_proj = nn.Linear(4, 8)
    net.blocks = nn.ModuleList([block])
    with patch('fastwam.cosmos.noaction.CosmosVideoExpert', TinyVideo), \
         patch('fastwam.cosmos.noaction.ActionDiT', side_effect=lambda **kw: nn.Linear(8, 8)), \
         patch('fastwam.cosmos.noaction.ForceDiffusionExpert', side_effect=lambda **kw: nn.Linear(8, 6)):
        return NoActionJointModel(net, manifest={'scope': 'metadata-test'}, action_hidden=8,
            action_ffn=16, force_hidden=8, force_ffn=16, force_heads=2,
            action_space=action_space, action_representation=representation)


def test_legacy_absolute_manifest_bytes_and_restore_stay_compatible(tmp_path):
    original = model()
    assert original.action_representation == 'absolute'
    assert original.action_anchor == 'none'
    assert 'action_representation' not in original.manifest
    assert original.manifest['action_units'] == '7 absolute joint radians + gripper openness'
    path = tmp_path / 'absolute.pt'
    original.save_checkpoint(path, step=12)
    restored = model()
    assert restored.load_checkpoint(path) == 12
    for key, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, atol=0, rtol=0)


def test_delta_metadata_is_complete_and_cannot_load_as_absolute(tmp_path):
    delta = model('delta_current_joint7')
    for key, value in joint_action_contract('delta_current_joint7').items():
        assert delta.manifest[key] == value
    path = tmp_path / 'delta.pt'
    delta.save_checkpoint(path, step=30)
    assert model('delta_current_joint7').load_checkpoint(path) == 30
    with pytest.raises(ValueError, match='differ'):
        model().load_checkpoint(path)
    absolute = tmp_path / 'absolute.pt'
    model().save_checkpoint(absolute)
    with pytest.raises(ValueError, match='differ'):
        delta.load_checkpoint(absolute)


def test_full_state_buffer_also_rejects_mixed_action_representation():
    with pytest.raises(ValueError, match='manifest differs'):
        model().load_state_dict(model('delta_current_joint7').state_dict())


@pytest.mark.parametrize('representation,space', [('unknown', 'joint8'), ('delta_current_joint7', 'eef_pose7')])
def test_invalid_contract_rejected_by_model_and_factory_before_asset_loading(representation, space):
    with pytest.raises(ValueError):
        model(representation, space)
    with patch('fastwam.cosmos.assets.read_text_bank', side_effect=AssertionError('No asset load')):
        with pytest.raises(ValueError):
            create_noaction_joint_cosmos(upstream='unused', checkpoint='unused', text_cache='unused',
                expected_text_identity='a' * 64, expected_data_identity='b' * 64,
                action_space=space, action_representation=representation)
