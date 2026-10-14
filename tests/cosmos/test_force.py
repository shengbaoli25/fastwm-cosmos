import torch
import pytest
from conftest import tiny_model,sample
from fastwam.forcewm.model import masked_force_mse
from fastwam.forcewm.masks import build_force_attention_mask

def setup():
    m=tiny_model();s=sample(1);raw,c,mask=m._contexts(s)
    memory=m.generate_video_memory(s['first_frame_latents'],raw,s['action'],s['action_is_pad'])
    cond=m._condition(s,s['action'],c,mask,memory)
    return m,s,cond

def test_force_mask_isolation_and_causality():
    z=torch.zeros(1,32,dtype=torch.bool);mask=build_force_attention_mask(z,z,z)[0,0]
    assert not mask[:32,32:].any() and torch.equal(mask[:32,:32],torch.ones(32,32,dtype=torch.bool).tril())
    assert mask[32:64,32:64].all() and not mask[32:64,:32].any() and not mask[32:64,64:].any()
    assert mask[64:,:64].all() and torch.equal(mask[64:,64:],torch.ones(32,32,dtype=torch.bool).tril())

def test_chunk_is_single_pass_and_all_queries_receive_loss_gradients():
    m,s,c=setup();f=m.dit['force'];calls=[]
    hooks=[b.register_forward_hook(lambda *args:calls.append(1)) for b in f.blocks]
    pred=f(**c)
    assert pred.wrench.shape==(1,32,6) and len(calls)==len(f.blocks)
    loss=masked_force_mse(pred.wrench,s['wrench_future'],s['force_is_pad'],s['wrench_valid'])
    loss.backward()
    for h in hooks:h.remove()
    assert f.queries.grad.abs().sum(-1).gt(0).all()
    assert all(any(p.grad is not None and p.grad.abs().max()>0 for p in b.parameters()) for b in f.blocks)
    assert not hasattr(f,'wrench_encoder') and not hasattr(f,'action_clock')


@pytest.mark.parametrize('key',['previous_wrench','wrench_future','noise','noisy_action','action_clock'])
def test_chunk_rejects_teacher_and_diffusion_inputs(key):
    m,s,c=setup()
    with pytest.raises(TypeError):m.dit['force'](**c,**{key:torch.zeros(1)})


def test_padding_and_invalid_history_cannot_change_prediction():
    m,s,c=setup();c['history_is_pad'][:,:5]=True;c['candidate_is_pad'][:,25:]=True
    c['history_channel_valid'][:,8,9]=False
    a=m.dit['force'](**c).wrench
    c['history'][:,:5]=float('nan');c['history'][:,8,9]=float('nan')
    c['action_condition'][:,25:]=float('nan')
    b=m.dit['force'](**c).wrench
    torch.testing.assert_close(a,b,rtol=0,atol=0)
