"""Synthetic CPU action-conditioning checks; no official GPU evidence."""
import copy
import pytest
import torch
from conftest import tiny_model, sample, tiny_net
from fastwam.cosmos.controls import VideoControlAdapter
from fastwam.cosmos.expert import CosmosExpert


def test_all_32_actions_map_to_their_own_sampled_interval():
    adapter = VideoControlAdapter()
    with torch.no_grad():
        adapter.projection.weight.fill_(1)
        adapter.projection.bias.zero_()
    actions = torch.zeros(1,32,7)
    pad = torch.zeros(1,32,dtype=torch.bool)
    original = adapter(actions,pad)
    for t in range(32):
        changed = actions.clone(); changed[:,t,0] = 1
        delta = adapter(changed,pad) - original
        expected = torch.zeros_like(delta); expected[:,t//4] = 1
        torch.testing.assert_close(delta,expected,rtol=0,atol=0)


def test_masked_controls_and_empty_intervals_have_no_padding_leak():
    adapter = VideoControlAdapter(); x = torch.randn(2,32,7)
    pad = torch.zeros(2,32,dtype=torch.bool); pad[:,6:] = True
    a = adapter(x,pad); changed=x.clone(); changed[:,6:]=1e6
    torch.testing.assert_close(a,adapter(changed,pad),rtol=0,atol=0)
    assert torch.count_nonzero(a[:,2:]) == 0
    a.sum().backward()
    assert adapter.projection.weight.grad.abs().max()>0


@pytest.mark.parametrize('bad',['length','nan','mask_dtype','mask_shape'])
def test_invalid_controls_rejected(bad):
    x=torch.randn(2,32,7);pad=torch.zeros(2,32,dtype=torch.bool)
    if bad=='length':x=x[:,:8]
    if bad=='nan':x[0,0,0]=float('nan')
    if bad=='mask_dtype':pad=pad.float()
    if bad=='mask_shape':pad=pad[:,:8]
    with pytest.raises(ValueError):VideoControlAdapter()(x,pad)


def test_native_controls_required_and_condition_future_video():
    e=CosmosExpert(tiny_net());x=torch.randn(1,16,3,4,4);ctx=torch.randn(1,3,24);t=torch.tensor([.5])
    with pytest.raises(ValueError,match='controls'):e(x,t,ctx)
    a=e(x,t,ctx,candidate_actions=torch.zeros(1,8,7))
    b=e(x,t,ctx,candidate_actions=torch.ones(1,8,7))
    assert not torch.equal(a,b)
    # Observed image is always an independent pass with no control input.
    obs=e.prepare(x[:,:,:1],torch.zeros(1),ctx,observed_only=True,observed_time='clean_zero')
    assert obs['state']['time'].shape[1]==1


def test_fixed_clean_memory_independent_of_action_noise_and_steps():
    m=tiny_model();s=sample();actions=[];forces=[]
    original=m.generate_video_memory
    def capture(first,raw,current,pad):
        actions.append(current.clone());return original(first,raw,current,pad)
    m.generate_video_memory=capture
    hook=m.dit['force'].register_forward_pre_hook(lambda _,args,kw:forces.append(kw),with_kwargs=True)
    first=m.infer_forcewm(s,action_condition=s['action'],num_inference_steps=2,seed=14)
    second=m.infer_forcewm(s,action_condition=s['action'],num_inference_steps=3,seed=29)
    hook.remove()
    assert len(actions)==len(forces)==2
    for a,c in zip(actions,forces):
        torch.testing.assert_close(a,s['action'],rtol=0,atol=0)
        torch.testing.assert_close(c['action_condition'],s['action'],rtol=0,atol=0)
        assert not {'noisy_action','action_clock','previous_wrench'} & c.keys()
    for key in ('force','video_latents'):torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
    assert not torch.equal(first['action'],second['action'])
    wm=m.infer_world_model(s,action_condition=s['action'])
    assert set(wm)=={'force','video_latents'}
    for key in wm:torch.testing.assert_close(wm[key],first[key],rtol=0,atol=0)


def test_train_and_infer_use_identical_clean_wm_contract():
    m=tiny_model();s=sample();controls=[];forces=[];predictions=[]
    m.a_scheduler.add_noise=lambda clean,noise,clock:torch.full_like(clean,999.)
    hook=m.dit['controls'].register_forward_pre_hook(lambda _,args:controls.append(args[0].clone()))
    hook2=m.dit['force'].register_forward_hook(lambda _,args,kw,out:(forces.append(kw),predictions.append(out.wrench.detach().clone())) and None,with_kwargs=True)
    m.dit['runtime'].manual_seed(44);m.training_loss(s)
    m.infer_world_model(s,action_condition=s['action'],video_seed=44)
    hook.remove();hook2.remove()
    assert len(controls)==3 and len(forces)==2
    for x in controls:torch.testing.assert_close(x,s['action'],rtol=0,atol=0)
    for c in forces:torch.testing.assert_close(c['action_condition'],s['action'],rtol=0,atol=0)
    torch.testing.assert_close(predictions[0],predictions[1],rtol=0,atol=0)


def test_clean_action_intervention_changes_both_world_models():
    m=tiny_model();s=sample(1)
    a=m.infer_world_model(s,action_condition=s['action'])
    b=m.infer_world_model(s,action_condition=s['action']+1)
    assert not torch.equal(a['video_latents'],b['video_latents'])
    assert not torch.equal(a['force'],b['force'])
