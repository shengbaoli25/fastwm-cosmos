"""Synthetic CPU contracts, never GPU throughput/model quality evidence."""
import copy
import pytest
import torch
from torch.nn import functional as F
from conftest import tiny_net,tiny_model,sample
from fastwam.cosmos.expert import CosmosExpert
from fastwam.cosmos.native import target_net_config


def test_native_forward_and_gradients():
    torch.manual_seed(6)
    net=tiny_net();adapter=CosmosExpert(copy.deepcopy(net));x=torch.randn(2,16,3,4,4,requires_grad=True)
    y=x.detach().clone().requires_grad_();context=torch.randn(2,3,24);sigma=torch.tensor([.4,.7])
    condition=torch.zeros(2,1,3,4,4);condition[:,:,0]=1
    native=net(x,sigma[:,None]*1000,context,condition,fps=torch.full((2,),7.5),padding_mask=torch.zeros(2,1,4,4))
    actual=adapter(y,sigma,context,fps=7.5)
    torch.testing.assert_close(native,actual,atol=2e-6,rtol=2e-5)
    native.square().mean().backward();actual.square().mean().backward()
    torch.testing.assert_close(x.grad,y.grad,atol=2e-6,rtol=2e-5)
    for (n,p),(m,q) in zip(net.named_parameters(),adapter.net.named_parameters()):
        assert n==m
        if p.grad is None: assert q.grad is None
        else: torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=5e-5)


def test_projected_context_matches_native():
    net=tiny_net();e=CosmosExpert(net);x=torch.randn(2,16,3,4,4);raw=torch.randn(2,3,24);sigma=torch.tensor([.1,.6])
    expected=e(x,sigma,raw,fps=7.5)
    p=e.prepare(x,sigma,net.crossattn_proj(raw),fps=7.5,context_is_projected=True);tokens=p['tokens']
    for i in range(e.num_layers):
        q,k,v,state=e.block_qkv(i,tokens,p);tokens=e.block_finish(i,F.scaled_dot_product_attention(q,k,v),state,p)
    torch.testing.assert_close(e.finish(tokens,p),expected)


def test_depth28_and_official_geometry():
    cfg=target_net_config();assert (cfg['num_blocks'],cfg['model_channels'],cfg['num_heads'],cfg['in_channels'])==(28,2048,16,16)
    model=tiny_model(depth=28);assert model.depth==28
    assert len(model.dit['action'].blocks)==len(model.dit['force'].blocks)==len(model.dit['memories'])==28
    loss,_=model.training_loss(sample(1));loss.backward();assert torch.isfinite(loss)
    assert all(b.force_gate.grad is not None for b in model.dit['memories'])
    assert all(p.grad is not None for p in model.dit['force'].head.parameters())


def test_two_adam_updates_and_full_checkpoint(tmp_path):
    model=tiny_model();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.95));s=sample()
    for _ in range(2):
        opt.zero_grad();loss,metrics=model.training_loss(s);loss.backward()
        assert set(metrics)=={'loss_video','loss_action','loss_force'}
        for branch in ('video','action','force'):
            assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.dit[branch].parameters())
        opt.step()
    assert {int(v['step']) for v in opt.state.values()}=={2}
    assert all(torch.isfinite(p).all() for p in model.parameters())
    path=tmp_path/'state.pt';model.save_checkpoint(path,opt,2)
    restored=tiny_model();other=torch.optim.AdamW(restored.parameters(),lr=1e-4,betas=(.9,.95))
    assert restored.load_checkpoint(path,other)==2
    for k,v in model.state_dict().items():
        w=restored.state_dict()[k]
        if torch.is_tensor(v): assert torch.equal(v,w),k
    assert {int(v['step']) for v in other.state.values()}=={2}
    rng=torch.get_rng_state()
    opt.zero_grad();a,_=model.training_loss(s);a.backward();opt.step()
    torch.set_rng_state(rng);other.zero_grad();b,_=restored.training_loss(s);b.backward();other.step()
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    restored.manifest={'wrong':True}
    with pytest.raises(ValueError,match='differ'):restored.load_checkpoint(path)


def test_no_teacher_future_leak_and_final_clock_refresh():
    m=tiny_model();s=sample();changed={k:v.clone() for k,v in s.items()}
    for key in ['video_latents','wrench_future','action']:changed[key].fill_(999)
    seen=[];force=m.dit['force'];original=force.rollout
    def observe(**condition):
        assert not {'wrench_future','previous_wrench','target','noise'}&condition.keys()
        seen.append((condition['noisy_action'].clone(),condition['action_clock'].clone()))
        return original(**condition)
    force.rollout=observe
    modes={mod:mod.training for mod in m.modules()};m.dit['action'].eval()
    original_mode=m.dit['action'].training;rng=m.dit['runtime'].get_state()
    first=m.infer_forcewm(s,num_inference_steps=2,seed=8)
    assert len(seen)==3 and torch.count_nonzero(seen[-1][1])==0
    torch.testing.assert_close(seen[-1][0],first['action'])
    assert m.dit['action'].training==original_mode and torch.equal(rng,m.dit['runtime'].get_state())
    second=m.infer_forcewm(changed,num_inference_steps=2,seed=8)
    for key in first:torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)


def test_training_action_loss_isolated_from_future_ground_truth():
    m=tiny_model();s=sample();t=copy.deepcopy(s);t['video_latents'][:,:,1:]*=10;t['wrench_future']*=20
    rng=torch.get_rng_state();memory=m.dit['runtime'].get_state()
    _,a=m.training_loss(s)
    torch.set_rng_state(rng);m.dit['runtime'].set_state(memory);_,b=m.training_loss(t)
    assert a['loss_action']==b['loss_action']
    assert a['loss_video']!=b['loss_video'] and a['loss_force']!=b['loss_force']


def test_checkpoint_gradient_equivalence():
    a=tiny_model(checkpoint=True);b=tiny_model(checkpoint=False);b.load_state_dict(a.state_dict());s=sample()
    rng=torch.get_rng_state();la,_=a.training_loss(s);la.backward()
    torch.set_rng_state(rng);lb,_=b.training_loss(s);lb.backward()
    torch.testing.assert_close(la,lb)
    for (name,p),(name2,q) in zip(a.named_parameters(),b.named_parameters()):
        if p.grad is None:assert q.grad is None,name
        else:torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5,msg=name)


@pytest.mark.parametrize('case',['rgb','wan48','action_condition'])
def test_invalid_input_rejected(case):
    m=tiny_model();s=sample()
    with pytest.raises(ValueError):
        if case=='rgb':s['image']=torch.zeros(1);m.training_loss(s)
        elif case=='wan48':s['video_latents']=torch.zeros(2,48,3,4,4);m.training_loss(s)
        else:m.video.prepare(s['first_frame_latents'],torch.zeros(2),s['cosmos_context'],candidate_actions=s['action'])

def test_bf16_native_fp32_strategy_forward_and_backward():
    from fastwam.cosmos.native import preserve_fp32_strategy
    net=tiny_net();net.use_wan_fp32_strategy=True
    for b in net.blocks:b.use_wan_fp32_strategy=True
    net.final_layer.use_wan_fp32_strategy=True
    net=net.bfloat16();preserve_fp32_strategy(net)
    adapter=CosmosExpert(copy.deepcopy(net))
    x=torch.randn(1,16,3,4,4,dtype=torch.bfloat16,requires_grad=True)
    y=x.detach().clone().requires_grad_();c=torch.randn(1,3,24,dtype=torch.bfloat16);t=torch.tensor([.5])
    mask=torch.zeros(1,1,3,4,4,dtype=torch.bfloat16);mask[:,:,0]=1
    a=net(x,t[:,None]*1000,c,mask,fps=torch.full((1,),7.5),padding_mask=torch.zeros(1,1,4,4,dtype=torch.bfloat16))
    b=adapter(y,t,c,fps=7.5)
    torch.testing.assert_close(a,b,rtol=.02,atol=.03)
    with torch.backends.mkldnn.flags(enabled=False):
        a.square().mean().backward();b.square().mean().backward()
    torch.testing.assert_close(x.grad,y.grad,rtol=.03,atol=.005)

def test_state_fingerprint_covers_values_dtypes_and_rng():
    from fastwam.cosmos.persistence import state_digest
    value={'a':torch.tensor([1.,2.]),'meta':None,'rng':torch.Generator().get_state()}
    original=state_digest(value)
    assert original==state_digest(copy.deepcopy(value))
    value['a'][1]+=1
    assert state_digest(value)!=original
    value['a']=torch.tensor([1.,2.],dtype=torch.float64)
    assert state_digest(value)!=original
