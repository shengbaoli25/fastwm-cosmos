"""Actual native tiny networks test math and isolation; not trained-model quality."""
import copy
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from conftest import sample
from fastwam.cosmos.native import nonaction_native_imports, nonaction_net_config
from fastwam.cosmos.expert import CosmosVideoExpert
from fastwam.cosmos.noaction import NoActionJointModel, SCHEMA
from fastwam.cosmos.preflight import validate_config

ROOT = Path(__file__).resolve().parents[2]


def net(depth=2):
    _, module, _, Net, _ = nonaction_native_imports(ROOT / 'third_party/cosmos-predict2.5')
    cfg = nonaction_net_config()
    cfg.update(max_img_h=8, max_img_w=8, max_frames=4, model_channels=32, num_blocks=depth,
        num_heads=2, adaln_lora_dim=8, crossattn_proj_in_channels=24, crossattn_emb_channels=16,
        use_wan_fp32_strategy=True, sac_config=module.SACConfig(mode='none'))
    return Net(**cfg)


def model(checkpointing=False):
    torch.manual_seed(7)
    return NoActionJointModel(net(), manifest={'scope':'test'}, action_hidden=24, action_ffn=48,
        force_hidden=24, force_ffn=48, force_heads=2, gradient_checkpointing=checkpointing, action_space='eef_pose7')


def core_inputs(m):
    s = sample(1); v = s['video_latents'].clone(); v[:, :, :1] = s['first_frame_latents']
    return s, (v, s['wrench_future'], s['action'], torch.tensor([700.]), torch.tensor([450.]), torch.tensor([200.]))


def test_official_nonaction_forward_and_gradients_match():
    n = net(); a = CosmosVideoExpert(copy.deepcopy(n))
    x = torch.randn(2,16,3,4,4,requires_grad=True); y = x.detach().clone().requires_grad_()
    ctx = torch.randn(2,3,24); sigma = torch.tensor([.2,.7])
    mask = torch.zeros(2,1,3,4,4); mask[:,:,0] = 1
    expected = n(x, sigma[:,None]*1000, ctx, mask, fps=torch.full((2,),7.5), padding_mask=torch.zeros(2,1,4,4))
    actual = a(y, sigma, ctx, fps=7.5)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    expected.square().mean().backward();actual.square().mean().backward()
    torch.testing.assert_close(x.grad,y.grad,atol=2e-6,rtol=2e-5)
    assert not any('action_embedder' in k for k in a.state_dict())
    with pytest.raises(ValueError,match='rejects'):
        a(y,sigma,ctx,candidate_actions=torch.zeros(2,8,7))


def test_forward_isolation_and_dynamic_force_input():
    m = model();s,args = core_inputs(m);c=m._conditions(s)
    v,f,a = m.joint_denoise_core(*args,c)
    changed=list(args);changed[2]=changed[2]+10;changed[5]=changed[5]+100
    v2,f2,a2=m.joint_denoise_core(*changed,c)
    torch.testing.assert_close(v,v2,rtol=0,atol=0);torch.testing.assert_close(f,f2,rtol=0,atol=0)
    assert not torch.allclose(a,a2)
    changed=list(args);changed[1]=changed[1]+5
    v3,f3,a3=m.joint_denoise_core(*changed,c)
    torch.testing.assert_close(v,v3,rtol=0,atol=0)
    assert not torch.allclose(f,f3) and not torch.allclose(a,a3)
    a.square().mean().backward()
    for part in (m.video,m.dit['force'],m.dit['action']):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in part.parameters())
    assert m.dit['force'].force_encoder.weight.grad.abs().sum()>0


@pytest.mark.parametrize('cache',[False,True])
def test_inference_has_no_future_target_or_reference_dependency(cache):
    m=model();s=sample(1);other=copy.deepcopy(s)
    for key in ('action','video_latents','wrench_future'):
        other[key]=torch.randn_like(other[key])*100
    other['proprio'][:,1:]=999
    other['force_is_pad'][:]=True;other['wrench_valid'][:]=False;other['action_is_pad'][:]=True
    a=m.infer_joint(s,num_inference_steps=2,use_cache=cache)
    b=m.infer_joint(other,num_inference_steps=2,use_cache=cache)
    for key in a:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    with pytest.raises(TypeError):m.infer_forcewm(s,action_condition=s['action'])
    with pytest.raises(ValueError,match='no future'):
        m.infer_joint(dict(s,action_reference=s['action']),num_inference_steps=1)


def test_cache_matches_reference_and_refreshes_between_requests():
    m=model();s=sample(1)
    for _ in range(2):
        a=m.infer_joint(s,num_inference_steps=3,use_cache=False)
        b=m.infer_joint(s,num_inference_steps=3,use_cache=True)
        for key in a:torch.testing.assert_close(a[key],b[key],atol=3e-6,rtol=3e-5)
        s['history']+=.3;s['first_frame_latents']+=.5
    with pytest.raises(ValueError,match='gradients'):
        m.prefill_static(s['first_frame_latents'],m._conditions(s))


def test_observed_history_does_not_condition_video():
    m=model();s,args=core_inputs(m)
    first=m.joint_denoise_core(*args,m._conditions(s))
    other=copy.deepcopy(s);other['history']+=10
    second=m.joint_denoise_core(*args,m._conditions(other))
    torch.testing.assert_close(first[0],second[0],atol=0,rtol=0)
    assert not torch.allclose(first[1],second[1])
    assert not torch.allclose(first[2],second[2])


def test_one_training_forward_checkpointing_and_masked_force():
    m=model();s=sample(1);s['wrench_valid'][:,0]=False;s['wrench_future'][:,0]=float('nan')
    with patch.object(m,'joint_denoise_core',wraps=m.joint_denoise_core) as core, \
         patch.object(m,'infer_joint',side_effect=AssertionError('No training rollout')):
        torch.manual_seed(99);loss,metrics=m.training_loss(s);loss.backward()
        assert core.call_count==1 and torch.isfinite(loss)
        assert set(metrics)=={'loss_video_fm','loss_force_fm','loss_action_fm'}
    other=model(True);other.load_state_dict(m.state_dict());torch.manual_seed(99)
    loss2,_=other.training_loss(s);loss2.backward()
    torch.testing.assert_close(loss,loss2)
    for (name,p),(_,q) in zip(m.named_parameters(),other.named_parameters()):
        if p.grad is None: assert q.grad is None,name
        else:torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=3e-5,msg=name)


def test_resume_reproduces_next_optimizer_step(tmp_path):
    m=model();opt=torch.optim.AdamW(m.parameters(),lr=1e-4);s=sample(1)
    def update(m,opt):
        opt.zero_grad();loss,_=m.training_loss(s);loss.backward();opt.step();return loss.detach()
    update(m,opt);p=tmp_path/'checkpoint.pt';m.save_checkpoint(p,opt,1)
    expected=update(m,opt)
    n=model();no=torch.optim.AdamW(n.parameters(),lr=1e-4)
    assert n.load_checkpoint(p,no)==1
    torch.testing.assert_close(update(n,no),expected,rtol=0,atol=0)
    for a,b in zip(m.parameters(),n.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    payload=torch.load(p,weights_only=True);payload['schema']='old';torch.save(payload,p)
    with pytest.raises(ValueError,match='differ'):n.load_checkpoint(p)


def test_noaction_training_config_and_model_sizes():
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):
        cfg=compose(config_name='train',overrides=['task=cosmos_noaction_joint_4gpu_30k',
            'model.expected_text_identity='+'a'*64,'model.expected_data_identity='+'b'*64,'data.train.expected_identity_sha256='+'b'*64])
    c=OmegaConf.to_container(cfg,resolve=True,throw_on_missing=True);validate_config(c)
    assert c['model']['action_hidden']==768 and c['model']['force_ffn']==2048
    c['model']['control_seed']=45
    with pytest.raises(ValueError):validate_config(c)


def test_bf16_training_and_inference_have_finite_outputs():
    m=model().to(dtype=torch.bfloat16);s=sample(1)
    # oneDNN on AVX2 does not implement BF16 backward; use native CPU kernels.
    with torch.backends.mkldnn.flags(enabled=False):
        loss,_=m.training_loss(s);loss.backward();assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        for v in m.infer_joint(s,num_inference_steps=2).values():assert torch.isfinite(v).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA smoke check needs a GPU')
def test_cuda_bf16_checkpointing_and_cached_sampling():
    m=model(True).to('cuda',torch.bfloat16);s=sample(1)
    loss,_=m.training_loss(s);loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    a=m.infer_joint(s,num_inference_steps=3,use_cache=False)
    b=m.infer_joint(s,num_inference_steps=3,use_cache=True)
    for key in a:
        assert torch.isfinite(a[key]).all() and torch.isfinite(b[key]).all()
        torch.testing.assert_close(a[key],b[key],atol=.04,rtol=.04)
    print({'gpu':torch.cuda.get_device_name(),'bf16_loss':float(loss),
           'cache_max_abs':{k:float((a[k]-b[k]).abs().max()) for k in a}})


def test_joint8_training_sampling_and_checkpoint_reject_eef(tmp_path):
    m=NoActionJointModel(net(),manifest={'scope':'test'},action_hidden=24,action_ffn=48,
        force_hidden=24,force_ffn=48,force_heads=2,gradient_checkpointing=True,action_space='joint8')
    s=sample(1)
    s.update(action=torch.randn(1,32,8),proprio=torch.randn(1,32,9),
        history=torch.randn(1,32,14),history_channel_valid=torch.ones(1,32,14,dtype=torch.bool))
    loss,_=m.training_loss(s);loss.backward();assert torch.isfinite(loss)
    a=m.infer_joint(s,num_inference_steps=2,use_cache=True)
    b=m.infer_joint(s,num_inference_steps=2,use_cache=False)
    assert a['action'].shape==(1,32,8) and a['force'].shape==(1,32,6)
    for key in a:torch.testing.assert_close(a[key],b[key],atol=3e-6,rtol=3e-5)
    path=tmp_path/'joint.pt';m.save_checkpoint(path,None,0)
    with pytest.raises(ValueError):model().load_checkpoint(path)
