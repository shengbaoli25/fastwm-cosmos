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


def test_refresh_video_with_current_action_every_step_and_final_force():
    m=tiny_model();s=sample();actions=[];memories=[];forces=[]
    original=m.generate_video_memory;force_original=m.dit['force'].rollout
    def capture(first,raw,current,pad):
        actions.append(current.clone());memory=original(first,raw,current,pad)
        memories.append(memory);return memory
    def capture_force(**condition):
        forces.append(condition);return force_original(**condition)
    m.generate_video_memory=capture;m.dit['force'].rollout=capture_force
    result=m.infer_forcewm(s,num_inference_steps=2,seed=14)
    assert len(actions)==len(forces)==3
    assert not torch.equal(actions[0],actions[1])
    torch.testing.assert_close(actions[-1],result['action'],rtol=0,atol=0)
    torch.testing.assert_close(memories[-1][0],result['video_latents'],rtol=0,atol=0)
    for a,mem,cond in zip(actions,memories,forces):
        torch.testing.assert_close(a,cond['noisy_action'],rtol=0,atol=0)
        assert cond['video_features'] is mem[1]
        assert all(not v.requires_grad for v in mem[1])
    assert torch.count_nonzero(forces[-1]['action_clock'])==0


def test_clean_teacher_actions_never_reach_free_memory():
    m=tiny_model();s=sample();changed=copy.deepcopy(s)
    changed['action'].fill_(50);changed['video_latents'][:,:,1:].fill_(60);changed['wrench_future'].fill_(70)
    fixed=torch.randn_like(s['action'])
    # Hold the current noisy Action fixed to isolate clean teacher paths.
    m.a_scheduler.add_noise=lambda clean,noise,clock:fixed
    controls=[];memories=[]
    original=m.generate_video_memory
    def capture(first,raw,current,pad):
        torch.testing.assert_close(current,fixed)
        out=original(first,raw,current,pad);memories.append(out);return out
    m.generate_video_memory=capture
    hook=m.dit['controls'].register_forward_pre_hook(lambda _,args:controls.append(args[0].clone()))
    rng=torch.get_rng_state();state=m.dit['runtime'].get_state()
    m.training_loss(s)
    torch.set_rng_state(rng);m.dit['runtime'].set_state(state);m.training_loss(changed)
    hook.remove()
    assert len(controls)==4
    torch.testing.assert_close(controls[0],s['action']);torch.testing.assert_close(controls[2],changed['action'])
    torch.testing.assert_close(controls[1],fixed);torch.testing.assert_close(controls[3],fixed)
    torch.testing.assert_close(memories[0][0],memories[1][0],rtol=0,atol=0)
    for a,b in zip(memories[0][1],memories[1][1]):torch.testing.assert_close(a,b,rtol=0,atol=0)
