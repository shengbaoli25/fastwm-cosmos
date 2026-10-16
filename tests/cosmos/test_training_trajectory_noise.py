"""CPU contracts for the training change; no GPU quality claims."""
import copy
import pytest
import torch
from conftest import tiny_model, sample


def observed_training(model, value):
    records={}
    original=model.action_velocity
    def record(action,clock,*args,**kwargs):
        records['ae_input']=action.detach().clone();records['clock']=clock.detach().clone()
        return original(action,clock,*args,**kwargs)
    model.action_velocity=record
    controls=[]
    hook=model.dit['controls'].register_forward_pre_hook(lambda m,args: controls.append(args[0].detach().clone()))
    try:
        loss,metrics=model.training_loss(value)
    finally:
        hook.remove();model.action_velocity=original
    return loss,metrics,records,controls


def test_small_noise_clock_input_and_world_models_unchanged():
    a=tiny_model();b=copy.deepcopy(a);b.ae_training_noise_max=.1
    value=sample();rng=torch.get_rng_state();memory=a.dit['runtime'].get_state()
    _,ma,ra,ca=observed_training(a,value)
    torch.set_rng_state(rng);b.dit['runtime'].set_state(memory)
    loss,mb,rb,cb=observed_training(b,value)
    torch.testing.assert_close(rb['clock'],ra['clock']*.1,atol=0,rtol=0)
    assert bool((rb['clock']>=0).all() and (rb['clock']<=100).all())
    # Same Gaussian draw and original interpolation at the scaled clock.
    sigma=ra['clock'].view(-1,1,1)/1000
    noise=(ra['ae_input']-(1-sigma)*value['action'])/sigma
    expected=(1-sigma*.1)*value['action']+sigma*.1*noise
    torch.testing.assert_close(rb['ae_input'],expected,atol=3e-7,rtol=1e-5)
    assert bool((rb['ae_input']!=value['action']).all())
    assert len(ca)==len(cb) and all(torch.equal(x,y) and torch.equal(x,value['action']) for x,y in zip(ca,cb))
    assert ma['loss_video']==mb['loss_video'] and ma['loss_force']==mb['loss_force']
    loss.backward()
    for branch in ['video','action','force']:
        gradients=[p.grad for p in b.dit[branch].parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().max()>0 for g in gradients)


def test_default_and_explicit_one_preserve_loss_grad_rng():
    a=tiny_model();b=copy.deepcopy(a);b.ae_training_noise_max=1.0
    s=sample();rng=torch.get_rng_state();state=a.dit['runtime'].get_state()
    x,ma=a.training_loss(s);x.backward();end=torch.get_rng_state()
    torch.set_rng_state(rng);b.dit['runtime'].set_state(state)
    y,mb=b.training_loss(s);y.backward()
    assert torch.equal(x,y) and ma==mb and torch.equal(end,torch.get_rng_state())
    for p,q in zip(a.parameters(),b.parameters()):
        if p.grad is None:assert q.grad is None
        else:assert torch.equal(p.grad,q.grad)


@pytest.mark.parametrize('maximum',[0,-.1,1.1,float('nan'),float('inf'),True])
def test_invalid_training_max_rejected(maximum):
    from fastwam.cosmos.model import CosmosForceWM
    from conftest import tiny_net
    with pytest.raises(ValueError,match='AE training noise'):
        CosmosForceWM(tiny_net(),manifest={},ae_training_noise_max=maximum)


def test_manifest_binds_noise_and_rejects_different_resume(tmp_path):
    from fastwam.cosmos.model import CosmosForceWM
    from conftest import tiny_net
    kwargs=dict(action_hidden=32,action_ffn=64,force_hidden=32,force_ffn=64,force_heads=2,memory_width=16,memory_heads=2,generated_video_steps=1)
    m=CosmosForceWM(tiny_net(),manifest={'TEST':'only'},ae_training_noise_max=.1,**kwargs)
    assert m.manifest['ae_training_noise']['sigma_max']==.1
    p=tmp_path/'state.pt';m.save_checkpoint(p)
    n=CosmosForceWM(tiny_net(),manifest={'TEST':'only'},ae_training_noise_max=.1,**kwargs);n.load_checkpoint(p)
    n=CosmosForceWM(tiny_net(),manifest={'TEST':'only'},**kwargs)
    with pytest.raises(ValueError,match='differ'):n.load_checkpoint(p)


def test_task_changes_only_noise_and_output():
    from pathlib import Path
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from fastwam.cosmos.preflight import validate_config
    root=Path(__file__).resolve().parents[2]
    overrides=['model.expected_text_identity='+'a'*64,'data.train.expected_identity_sha256='+'b'*64]
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        old=compose(config_name='train',overrides=['task=cosmos_forcewm_4gpu_30k',*overrides])
        new=compose(config_name='train',overrides=['task=cosmos_ae_noisy_trajectory_4gpu_30k',*overrides])
    a=OmegaConf.to_container(old,resolve=True);b=OmegaConf.to_container(new,resolve=True)
    validate_config(b)
    assert b['model'].pop('ae_training_noise_max')==.1
    b['output_dir']=a['output_dir']
    assert a==b
