import torch
import pytest
from conftest import tiny_model,sample
from fastwam.forcewm.model import shifted_wrench_inputs
from fastwam.forcewm.masks import build_force_attention_mask

def setup():
    m=tiny_model();s=sample(1);raw,c,mask=m._contexts(s)
    memory=m.generate_video_memory(s['first_frame_latents'],raw)
    cond=m._condition(s,s['action'],torch.tensor([250.]),c,mask,memory)
    return m,s,cond

def test_force_mask_isolation_and_causality():
    z=torch.zeros(1,32,dtype=torch.bool);mask=build_force_attention_mask(z,z,z)[0,0]
    assert not mask[:32,32:].any() and torch.equal(mask[:32,:32],torch.ones(32,32,dtype=torch.bool).tril())
    assert mask[32:64,32:64].all() and not mask[32:64,:32].any() and not mask[32:64,64:].any()
    assert mask[64:,:64].all() and torch.equal(mask[64:,64:],torch.ones(32,32,dtype=torch.bool).tril())

def test_force_teacher_cannot_see_current_or_later_labels():
    m,s,condition=setup();f=m.dit['force'];prev=shifted_wrench_inputs(s['history'],s['wrench_future'],s['force_is_pad'],s['wrench_valid'])
    target=s['wrench_future'].clone();target[:,15:]+=200
    prev2=shifted_wrench_inputs(s['history'],target,s['force_is_pad'],s['wrench_valid'])
    a=f.forward_teacher(previous_wrench=prev,**condition).wrench
    b=f.forward_teacher(previous_wrench=prev2,**condition).wrench
    torch.testing.assert_close(a[:,:16],b[:,:16],atol=0,rtol=0)
    assert not torch.equal(a[:,16:],b[:,16:])

def test_cached_free_force_matches_shifted_teacher_of_own_predictions():
    m,s,condition=setup();f=m.dit['force'];a=f.rollout(**condition)
    previous=torch.cat((s['history'][:,-1:,7:],a.wrench[:,:-1]),1)
    b=f.forward_teacher(previous_wrench=previous,**condition)
    torch.testing.assert_close(a.wrench,b.wrench,atol=3e-6,rtol=3e-5)
    for x,y in zip(a.layers,b.layers):torch.testing.assert_close(x,y,atol=3e-6,rtol=3e-5)
    with pytest.raises(ValueError,match='labels/noise'):f.rollout(noise=torch.zeros(1),**condition)
