"""CPU sampling contracts, not trained-model quality measurements."""
import copy
import pytest
import torch
from conftest import tiny_model, sample
from fastwam.cosmos.action_sampling import ActionSampling
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler


@pytest.mark.parametrize('strength',[0.,.1,.25,.5,.75,1.])
def test_whole_trajectory_noise_clock_and_oracle(strength):
    reference=sample()['action']; original=reference.clone()
    scheduler=WanContinuousFlowMatchScheduler(shift=1)
    spec=ActionSampling(noise_strength=strength,seed=8)
    rng=torch.get_rng_state()
    action,times,deltas=spec.prepare(scheduler,batch_size=2,device='cpu',dtype=torch.float32,reference=reference)
    noise=torch.randn(reference.shape,generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(action,(1-strength)*reference+strength*noise,rtol=0,atol=0)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(reference,original)
    if strength:
        assert len(times)==16 and float(times[0])==pytest.approx(strength*1000)
        assert float(deltas.sum())==pytest.approx(-strength)
        assert bool((action!=reference).all())
        for dt in deltas: action=scheduler.step(noise-reference,dt,action)
    else: assert len(times)==len(deltas)==0
    torch.testing.assert_close(action,reference,atol=8e-7,rtol=1e-6)


def test_real_ae_first_input_and_clean_world_models(monkeypatch):
    m=tiny_model();s=sample(1);reference=s['action'].clone()
    observed=[]; controls=[]; forces=[]
    original=m.action_velocity
    def velocity(action,clock,*args):
        observed.append((action.clone(),clock.clone()))
        return original(action,clock,*args)
    monkeypatch.setattr(m,'action_velocity',velocity)
    h=m.dit['controls'].register_forward_pre_hook(lambda _,args:controls.append(args[0].clone()))
    f=m.dit['force'].register_forward_pre_hook(lambda _,args,kwargs:forces.append(kwargs['action_condition'].clone()),with_kwargs=True)
    before=torch.get_rng_state(); runtime=m.dit['runtime'].get_state(); modes={x:x.training for x in m.modules()}
    result=m.infer_trajectory_denoising(s,action_condition=s['action'],action_reference=reference,num_inference_steps=3,seed=8)
    h.remove();f.remove()
    noise=torch.randn(reference.shape,generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(observed[0][0],.9*reference+.1*noise,rtol=0,atol=0)
    assert [float(t[1][0]) for t in observed]==pytest.approx([100,100*2/3,100/3])
    assert len(controls)==len(forces)==1
    assert torch.equal(controls[0],s['action']) and torch.equal(forces[0],s['action'])
    assert torch.equal(before,torch.get_rng_state()) and torch.equal(runtime,m.dit['runtime'].get_state())
    assert modes=={x:x.training for x in m.modules()}
    baseline=m.infer_forcewm(s,action_condition=s['action'],num_inference_steps=3,seed=8)
    for key in ('force','video_latents'): torch.testing.assert_close(result[key],baseline[key],rtol=0,atol=0)
    endpoint=m.infer_trajectory_denoising(s,action_condition=s['action'],action_reference=reference,noise_strength=1,num_inference_steps=3,seed=8)
    for key in baseline: torch.testing.assert_close(baseline[key],endpoint[key],rtol=0,atol=0)


def test_reference_explicit_and_no_future_label_leak():
    m=tiny_model();s=sample(1);other=copy.deepcopy(s)
    for key in ('action','wrench_future','video_latents'): other[key].fill_(999)
    a=m.infer_trajectory_denoising(s,action_condition=s['action'],action_reference=s['action'],num_inference_steps=1)
    b=m.infer_trajectory_denoising(other,action_condition=s['action'],action_reference=s['action'],num_inference_steps=1)
    for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    with pytest.raises(ValueError,match='reference'):
        m.infer_forcewm(s,action_condition=s['action'],action_sampling=ActionSampling())
    with pytest.raises(TypeError):m.infer_trajectory_denoising(s,action_condition=s['action'])


@pytest.mark.parametrize('strength',[-.1,1.1,float('nan'),float('inf'),True])
def test_invalid_strength(strength):
    with pytest.raises(ValueError):ActionSampling(noise_strength=strength)


@pytest.mark.parametrize('reference',[None,torch.zeros(1,31,7),torch.zeros(1,32,7,dtype=torch.int64),torch.full((1,32,7),float('nan'))])
def test_invalid_reference_restores_modes_and_rng(reference):
    m=tiny_model();s=sample(1);m.dit['action'].eval();rng=m.dit['runtime'].get_state()
    with pytest.raises(ValueError):m.infer_forcewm(s,action_condition=s['action'],action_reference=reference,action_sampling=ActionSampling())
    assert not m.dit['action'].training and torch.equal(rng,m.dit['runtime'].get_state())


def test_zero_noise_passthrough_and_no_ae_forward(monkeypatch):
    m=tiny_model();s=sample(1)
    def forbidden(*args):raise AssertionError('No denoising needed at sigma0')
    monkeypatch.setattr(m,'action_velocity',forbidden)
    out=m.infer_trajectory_denoising(s,action_condition=s['action'],action_reference=s['action'],noise_strength=0)
    assert torch.equal(out['action'],s['action'])
    with pytest.raises(ValueError):m.infer_forcewm(s,action_condition=s['action'],generate_action=False,action_sampling=ActionSampling())


def test_sampling_adds_no_checkpoint_parameters(tmp_path):
    m=tiny_model();s=sample(1);before=copy.deepcopy(m.state_dict())
    m.infer_trajectory_denoising(s,action_condition=s['action'],action_reference=s['action'],num_inference_steps=1)
    for key,value in before.items():
        if torch.is_tensor(value):assert torch.equal(value,m.state_dict()[key]),key
    p=tmp_path/'v3.pt';m.save_checkpoint(p,step=30000)
    payload=torch.load(p,weights_only=True)
    assert payload['schema']=='fastwam-cosmos-clean-action-force-chunk28-v3'
    restored=tiny_model();assert restored.load_checkpoint(p)==30000
