"""Native tiny-network contracts for Action-only JEPA/Force memory."""
import copy
from unittest.mock import patch

import pytest
import torch
from torch import nn

from conftest import sample
from test_noaction import model as baseline_model, core_inputs
from fastwam.cosmos.jepa_force_latent import JepaForceLatentModel, SCHEMA


def make_model(*, checkpointing=False, encoder=None, latent_weight=1., wrench_weight=.1):
    baseline = baseline_model(checkpointing)
    model = JepaForceLatentModel(copy.deepcopy(baseline.video.net), manifest={'scope': 'test'},
        action_hidden=24, action_ffn=48, force_hidden=24, force_ffn=48, force_heads=2,
        gradient_checkpointing=checkpointing, action_space='eef_pose7',
        observation_encoder=encoder, allow_preencoded_patches=encoder is None,
        jepa_feature_dim=12, reader_width=16, reader_heads=2, query_count=4,
        reader_depth=2, injection_layers=(1, 2), latent_loss_weight=latent_weight,
        wrench_loss_weight=wrench_weight)
    return model, baseline


def observed_sample():
    data = sample(1)
    generator = torch.Generator().manual_seed(815)
    data['jepa_observed_patches'] = torch.randn(1, 8, 12, generator=generator)
    return data


def inputs(model, data):
    _, args = core_inputs(model)
    return args, model._conditions(data)


def test_zero_gate_is_exact_baseline_and_teacher_features_are_final_state():
    model, baseline = make_model()
    data = observed_sample()
    args, conditions = inputs(model, data)
    result = model.joint_denoise_core(*args, conditions, return_auxiliary=True)
    expected = baseline.joint_denoise_core(*args, baseline._conditions(data))
    for actual, reference in zip(result[:3], expected):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    final_features = result[3]['force_final_features']
    torch.testing.assert_close(model.dit['force'].head(final_features), result[1], atol=0, rtol=0)
    assert final_features.shape == (1, 32, 24)
    assert model.manifest['schema'] == SCHEMA


def test_action_conditioned_memory_does_not_change_video_or_force():
    model, _ = make_model()
    model.dit['jepa_adapter'].gates.data.fill_(.4)
    data = observed_sample()
    args, conditions = inputs(model, data)
    before = model.joint_denoise_core(*args, conditions, return_auxiliary=True)
    changed = list(args)
    changed[2] = changed[2] + 4
    changed[5] = changed[5] + 200
    after = model.joint_denoise_core(*changed, conditions, return_auxiliary=True)
    for a, b in zip(before[:2], after[:2]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not torch.allclose(before[2], after[2])
    assert not torch.allclose(before[3]['action_force_context']['features'],
                              after[3]['action_force_context']['features'])
    other = copy.deepcopy(data)
    other['jepa_observed_patches'] = torch.randn_like(other['jepa_observed_patches'])
    alternate = model.joint_denoise_core(*args, model._conditions(other))
    for a, b in zip(before[:2], alternate[:2]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not torch.allclose(before[2], alternate[2])


def test_teacher_and_future_targets_never_enter_inference_or_query():
    model, _ = make_model()
    model.dit['jepa_adapter'].gates.data.fill_(.5)
    data = observed_sample()
    other = copy.deepcopy(data)
    for key in ('action', 'video_latents', 'wrench_future'):
        other[key].fill_(float('nan'))
    other['force_is_pad'][:] = True
    other['wrench_valid'][:] = False
    first = model.infer_joint(data, num_inference_steps=2)
    second = model.infer_joint(other, num_inference_steps=2)
    for key in first:
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
    observation_only = {key: value for key, value in data.items() if key not in (
        'action', 'video_latents', 'wrench_future', 'force_is_pad', 'wrench_valid',
        'action_is_pad', 'image_is_pad')}
    third = model.infer_joint(observation_only, num_inference_steps=2)
    for key in first:
        torch.testing.assert_close(first[key], third[key], atol=0, rtol=0)


def test_no_raw_action_query_residual_can_bypass_visual_values():
    model, _ = make_model()
    data = observed_sample()
    data['jepa_observed_patches'].zero_()
    _, conditions = inputs(model, data)
    reader = model.dit['jepa_reader']
    a = reader(data['action'], torch.tensor([250.]), conditions.observation)
    b = reader(data['action'] + 40, torch.tensor([850.]), conditions.observation)
    # Identical image values at every position cannot carry an action-dependent
    # output through an implicit Q residual, despite position-dependent keys.
    torch.testing.assert_close(a['features'], b['features'], atol=1e-6, rtol=1e-6)


def test_gate_opens_then_action_loss_trains_reader_and_residual():
    model, _ = make_model(latent_weight=0., wrench_weight=0.)
    data = observed_sample()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    args, conditions = inputs(model, data)
    model.joint_denoise_core(*args, conditions)[2].square().mean().backward()
    assert model.dit['jepa_adapter'].gates.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    args, conditions = inputs(model, data)
    model.joint_denoise_core(*args, conditions)[2].square().mean().backward()
    assert model.dit['jepa_adapter'].output[-1].weight.grad.abs().sum() > 0
    assert model.dit['jepa_reader'].patch[-1].weight.grad.abs().sum() > 0
    assert torch.isfinite(model.dit['jepa_reader'].patch[-1].weight.grad).all()


def test_online_force_label_is_detached_and_masks_invalid_channels():
    model, _ = make_model()
    model.set_auxiliary_step(2500)
    data = observed_sample()
    data['wrench_future'].requires_grad_()
    data['force_is_pad'][:, 0] = True
    with torch.no_grad():
        data['wrench_future'][:, 0] = float('nan')
    args, conditions = inputs(model, data)
    auxiliary = model.joint_denoise_core(*args, conditions, return_auxiliary=True)[3]
    target = model.clean_force_target(data, conditions)
    assert not target.requires_grad and torch.isfinite(target).all()
    loss, metrics = model._auxiliary_training_loss(data, auxiliary)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics['loss_force_feature'] > 0 and metrics['loss_query_wrench'] > 0
    assert data['wrench_future'].grad is None
    assert model.dit['jepa_reader'].feature_head.weight.grad.abs().sum() > 0


def test_online_target_depends_on_true_force_but_not_future_video_or_action():
    model, _ = make_model()
    data = observed_sample()
    args, conditions = inputs(model, data)
    before = model.clean_force_target(data, conditions)
    changed = copy.deepcopy(data)
    changed['video_latents'].fill_(float('nan'))
    changed['action'].fill_(float('nan'))
    target = model.clean_force_target(changed, model._conditions(changed))
    torch.testing.assert_close(before, target, atol=0, rtol=0)
    changed['wrench_future'] += 3
    target = model.clean_force_target(changed, model._conditions(changed))
    assert not torch.allclose(before, target)
    # Holding the current noisy states fixed, changed labels cannot enter U/A.
    output = model.joint_denoise_core(*args, conditions, return_auxiliary=True)
    alternate = model.joint_denoise_core(*args, model._conditions(changed), return_auxiliary=True)
    for a, b in zip(output[:3], alternate[:3]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(output[3]['action_force_context']['features'],
                               alternate[3]['action_force_context']['features'], atol=0, rtol=0)


def test_auxiliary_ramp_and_external_teacher_rejection():
    model, _ = make_model()
    data = observed_sample()
    args, conditions = inputs(model, data)
    auxiliary = model.joint_denoise_core(*args, conditions, return_auxiliary=True)[3]
    for step, expected in ((0, 0.), (1, .0004), (1250, .5), (2500, 1.), (9000, 1.)):
        model.set_auxiliary_step(step)
        _, metrics = model._auxiliary_training_loss(data, auxiliary)
        assert metrics['force_feature_weight'] == expected
    with pytest.raises(ValueError, match='External teacher'):
        model._auxiliary_training_loss(dict(data, force_teacher_features=torch.zeros(1, 32, 24)), auxiliary)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA smoke needs GPU'))])
def test_real_jepa_lora_receives_joint_training_gradients(device, monkeypatch):
    from test_jepa_observation import tiny_official
    from fastwam.cosmos import jepa_observation as jo
    monkeypatch.setitem(jo.VARIANT_SPECS, 'vitb', dict(
        width=24, depth=12, heads=3, factory='vit_base', filename='tiny-test.pt'))
    encoder = jo.JEPAObservationEncoder(tiny_official(), variant='vitb', weight_sha256='a' * 64)
    model, _ = make_model(encoder=encoder, checkpointing=True)
    from fastwam.trainer import Wan22Trainer
    Wan22Trainer._apply_dit_only_train_mode(model)
    model.set_auxiliary_step(1000)
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    model.to(device=device, dtype=dtype)
    data = observed_sample()
    del data['jepa_observed_patches']
    data['jepa_observation'] = torch.rand(1, 2, 3, 2, 224, 224) * 2 - 1
    seen_rgb = []
    hook = encoder.register_forward_pre_hook(lambda _, args: seen_rgb.append(args[0].detach().cpu()))
    loss, _ = model.training_loss(data)
    hook.remove()
    torch.testing.assert_close(seen_rgb[0], data['jepa_observation'], atol=0, rtol=0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.dit['force'].force_encoder.weight.grad.abs().sum() > 0
    for name, parameter in encoder.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if name.endswith(('_q_b', '_v_b')):
                assert parameter.grad.abs().sum() > 0, name
        else:
            assert parameter.grad is None, name


def test_checkpointing_matches_gradients_and_resume_identity(tmp_path):
    plain, _ = make_model()
    recomputed, _ = make_model(checkpointing=True)
    plain.dit['jepa_adapter'].gates.data.fill_(.1)
    recomputed.load_state_dict(plain.state_dict())
    data = observed_sample()
    torch.manual_seed(123)
    first, _ = plain.training_loss(data)
    first.backward()
    torch.manual_seed(123)
    second, _ = recomputed.training_loss(data)
    second.backward()
    torch.testing.assert_close(first, second)
    for (name, a), (_, b) in zip(plain.named_parameters(), recomputed.named_parameters()):
        if a.grad is None:
            assert b.grad is None, name
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=3e-5, msg=name)
    path = tmp_path / 'jepa.pt'
    plain.save_checkpoint(path, step=3)
    assert recomputed.load_checkpoint(path) == 3
    mismatched, _ = make_model(wrench_weight=.2)
    with pytest.raises(ValueError, match='differ'):
        mismatched.load_checkpoint(path)


def test_optimizer_resume_matches_next_step_with_restored_trainer_step(tmp_path):
    model, _ = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    data = observed_sample()

    def update(current, opt, step):
        current.set_auxiliary_step(step)
        opt.zero_grad(set_to_none=True)
        loss, _ = current.training_loss(data)
        loss.backward()
        opt.step()
        return loss.detach()

    update(model, optimizer, 0)
    path = tmp_path / 'resume.pt'
    model.save_checkpoint(path, optimizer, 1)
    expected = update(model, optimizer, 1)
    restored, _ = make_model()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    step = restored.load_checkpoint(path, restored_optimizer)
    actual = update(restored, restored_optimizer, step)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_factory_reuses_official_asset_validation_with_direct_subclass_construction(monkeypatch):
    from fastwam.cosmos import assets
    from fastwam.cosmos.jepa_force_latent import create_jepa_force_latent_cosmos
    from test_noaction import net
    monkeypatch.setattr(assets, 'read_text_bank', lambda _: (torch.randn(1, 3, 24), {}, 'a' * 64))
    monkeypatch.setattr(assets, 'load_nonaction_video', lambda *args, **kwargs: net())
    model = create_jepa_force_latent_cosmos(upstream='test', checkpoint='test', text_cache='test',
        expected_text_identity='a' * 64, expected_data_identity='b' * 64, device='cpu',
        allow_preencoded_patches=True, jepa_feature_dim=12, reader_width=16, reader_heads=2,
        query_count=4, injection_layers=(1, 2), action_hidden=24, action_ffn=48,
        force_hidden=24, force_ffn=48, force_heads=2, action_space='eef_pose7')
    assert isinstance(model, JepaForceLatentModel)
    assert model.manifest['schema'] == SCHEMA
    assert model.manifest['jepa_force_latent']['latent_warmup_steps'] == 2500
    assert model.torch_dtype == torch.bfloat16


def test_encoder_variant_construction_does_not_shift_main_rng(monkeypatch):
    from fastwam.cosmos import assets, jepa_observation
    from fastwam.cosmos.jepa_force_latent import create_jepa_force_latent_cosmos
    from test_noaction import net
    monkeypatch.setattr(assets, 'read_text_bank', lambda _: (torch.zeros(1, 3, 24), {}, 'a' * 64))
    monkeypatch.setattr(assets, 'load_nonaction_video', lambda *args, **kwargs: net())

    def encoder_factory(variant, **kwargs):
        # Emulate variants consuming different initialization RNG amounts.
        torch.randn(10 if variant == 'vitb' else 10000)
        return CountingEncoder()

    monkeypatch.setattr(jepa_observation, 'create_jepa_observation', encoder_factory)
    states = []
    for variant in ('vitb', 'vitl'):
        torch.manual_seed(735)
        create_jepa_force_latent_cosmos(upstream='test', checkpoint='test', text_cache='test',
            expected_text_identity='a' * 64, expected_data_identity='b' * 64, device='cpu',
            observation_encoder=dict(variant=variant, checkpoint='test'),
            reader_width=16, reader_heads=2, query_count=4, injection_layers=(1, 2),
            action_hidden=24, action_ffn=48, force_hidden=24, force_ffn=48,
            force_heads=2, action_space='eef_pose7')
        states.append(torch.get_rng_state())
    assert torch.equal(*states)


class CountingEncoder(nn.Module):
    feature_dim = 12
    identity = {'scope': 'synthetic-observation-encoder'}

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 12)
        self.calls = 0

    def forward(self, observation):
        self.calls += 1
        return self.projection(observation)

    def configure_trainable_parameters(self):
        self.projection.requires_grad_(True)


def test_online_encoder_once_per_observation_query_refreshes_each_step_and_caches_match():
    encoder = CountingEncoder()
    model, _ = make_model(encoder=encoder)
    model.dit['jepa_adapter'].gates.data.fill_(.2)
    data = observed_sample()
    del data['jepa_observed_patches']
    data['jepa_observation'] = torch.randn(1, 8, 3)
    reader = model.dit['jepa_reader']
    with patch.object(reader, 'forward', wraps=reader.forward) as read:
        cached = model.infer_joint(data, num_inference_steps=3, use_cache=True)
        assert encoder.calls == 1 and read.call_count == 3
        clocks = [call.args[1] for call in read.call_args_list]
        assert not torch.equal(clocks[0], clocks[-1])
    direct = model.infer_joint(data, num_inference_steps=3, use_cache=False)
    assert encoder.calls == 2
    for key in cached:
        torch.testing.assert_close(cached[key], direct[key], atol=3e-6, rtol=3e-5)
    loss, _ = model.training_loss(data)
    loss.backward()
    assert encoder.projection.weight.grad.abs().sum() > 0


def test_bf16_auxiliary_and_sampling_are_finite():
    model, _ = make_model(checkpointing=True)
    model.to(dtype=torch.bfloat16)
    data = observed_sample()
    with torch.backends.mkldnn.flags(enabled=False):
        loss, _ = model.training_loss(data)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        result = model.infer_joint(data, num_inference_steps=2)
        assert all(torch.isfinite(value).all() for value in result.values())


def test_real_probe_loop_sets_auxiliary_step_without_shortening_scheduler():
    from contextlib import nullcontext
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from fastwam.cosmos.probe_updates import run_two_updates
    from fastwam.trainer import Wan22Trainer

    class ProbeModel(nn.Module):
        action_dim = 8
        set_auxiliary_step = JepaForceLatentModel.set_auxiliary_step

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.steps = []

        def training_loss(self, sample):
            self.steps.append(self._auxiliary_step)
            loss = self.weight.square()
            return loss, {'auxiliary_step': float(self._auxiliary_step)}

    model = ProbeModel()
    trainer = object.__new__(Wan22Trainer)
    trainer.model = model
    trainer.cfg = OmegaConf.create(dict(max_steps=30000, scheduler_total_steps=50000,
                                        lr_scheduler_type='cosine'))
    trainer.learning_rate = 1e-4
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer._initialize_schedule(30000)
    trainer.global_step = trainer.batch_in_epoch = 0
    trainer.batch_size, trainer.max_grad_norm = 1, 1.
    trainer.accelerator = SimpleNamespace(accumulate=lambda _: nullcontext(),
        autocast=lambda: nullcontext(), backward=lambda loss: loss.backward(),
        sync_gradients=True, optimizer_step_was_skipped=False, gather=lambda value: value,
        clip_grad_norm_=torch.nn.utils.clip_grad_norm_)
    data = dict(video_latents=torch.zeros(1, 16, 3, 28, 56),
                first_frame_latents=torch.zeros(1, 16, 1, 28, 56),
                action=torch.zeros(1, 32, 8), proprio=torch.zeros(1, 32, 9),
                sample_index=torch.zeros(1, dtype=torch.long))
    trainer.train_loader = [data, data]
    _, updates = run_two_updates(trainer, synchronize=lambda: None)
    assert model.steps == [0, 1]
    assert [row['global_components']['auxiliary_step'] for row in updates] == [0., 1.]
    assert trainer.max_steps == 30000 and trainer.scheduler_total_steps == 50000
    assert trainer.scheduler._milestones == [2500] and trainer.scheduler.last_epoch == 2
