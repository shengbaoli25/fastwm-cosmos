"""Synthetic CPU checks of routing, gradients, time mapping and persistence."""
import copy
import pytest
import torch
from conftest import tiny_net, sample
from fastwam.cosmos.paired import PairedCosmosForceWM
from fastwam.cosmos.regions import PairedRegionReader, merge_views, view_grid
from fastwam.forcewm.model import masked_force_mse


def model(checkpoint=False, pairs=32):
    torch.manual_seed(7)
    return PairedCosmosForceWM(tiny_net(), manifest={'scope':'TEST-paired'},
        action_hidden=32, action_ffn=64, force_hidden=32, force_ffn=64,
        force_heads=2, memory_width=16, memory_heads=2, generated_video_steps=1,
        gradient_checkpointing=checkpoint, region_layer=0, region_pairs=pairs,
        region_width=16, region_heads=2)


def world(m, s):
    raw, ctx, cmask = m._contexts(s)
    return m._world_models(s, s['first_frame_latents'], raw, ctx, cmask,
                           s['action'], s['action_is_pad'])


def test_production_grid_preserves_camera_and_time_indices():
    x = torch.arange(3*14*28).reshape(1, 3*14*28, 1).float()
    v = view_grid(x, (3,14,28))
    assert v.shape == (1,2,2,14,14,1)
    assert v[0,0,0,0,0,0] == 392
    assert v[0,0,1,0,0,0] == 406
    torch.testing.assert_close(merge_views(v), x[:,392:])


def test_32_pairs_count_and_normalized_per_view_maps():
    reader = PairedRegionReader(24, pairs=32, width=16, heads=2)
    future = torch.randn(1,2,2,14,14,24)
    tokens, maps, xy = reader(future, torch.randn(1,32,7), torch.zeros(1,32,dtype=torch.bool))
    assert reader.queries.shape == (2,32,2,16)  # 128 shared-in-time query vectors
    assert tokens.flatten(1,4).shape == (1,256,16)
    assert maps.shape == (1,2,2,32,2,14,14)
    torch.testing.assert_close(maps.sum((-1,-2)), torch.ones(1,2,2,32,2))
    assert ((xy > 0) & (xy < 1)).all()
    assert len(reader.anchors[:,0].unique()) == 8
    assert len(reader.anchors[:,1].unique()) == 4


def test_reader_padded_actions_ignored_and_target_labels_not_accepted():
    reader = PairedRegionReader(24,pairs=4,width=16,heads=2)
    v = torch.randn(1,2,2,3,4,24); a = torch.randn(1,32,7)
    pad = torch.zeros(1,32,dtype=torch.bool);pad[:,20:] = True
    expected = reader(v,a,pad)
    a[:,20:] = float('nan')
    for x,y in zip(expected,reader(v,a,pad)):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    with pytest.raises(TypeError): reader(v,a,pad,wrench_future=torch.zeros(1))


def test_force_and_aux_losses_reach_queries_without_ae_gradient():
    m=model();s=sample(1);_,force=world(m,s)
    loss=masked_force_mse(force.wrench,s['wrench_future'],s['force_is_pad'],s['wrench_valid'])
    loss.backward()
    assert (m.dit['regions'].queries.grad.abs().sum(-1)>0).all()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.video.parameters())
    assert all(p.grad is None for p in m.dit['action'].parameters())
    m.zero_grad();loss,metrics=m.training_loss(s);loss.backward()
    assert {'loss_region_reconstruction','loss_region_diversity'} <= metrics.keys()
    assert m.dit['regions'].decode.weight.grad.abs().sum()>0


def test_future_labels_and_ae_seed_cannot_change_regions_or_world_models():
    m=model();s=sample(1);changed=copy.deepcopy(s)
    for k in ('video_latents','wrench_future','action'):changed[k].fill_(float('nan'))
    for k in ('image_is_pad','force_is_pad','wrench_valid'):changed[k].logical_not_()
    before=m.dit['runtime'].get_state()
    a=m.infer_with_regions(s,action_condition=s['action'],seed=1,num_inference_steps=1)
    b=m.infer_with_regions(changed,action_condition=s['action'],seed=8,num_inference_steps=1)
    for key in ('force','video_latents'):
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    torch.testing.assert_close(a['regions']['attention'],b['regions']['attention'],rtol=0,atol=0)
    assert torch.equal(before,m.dit['runtime'].get_state())
    assert not torch.equal(a['action'],b['action'])


def test_force_feedback_changes_generated_video_and_visual_read_exists():
    m=model();s=sample(1);changed=copy.deepcopy(s)
    changed['history'][:,-1,7:] += 20
    a=m.infer_with_regions(s,action_condition=s['action'],generate_action=False)
    b=m.infer_with_regions(changed,action_condition=s['action'],generate_action=False)
    assert not torch.equal(a['video_latents'],b['video_latents'])
    assert a['regions']['attention'].shape == (1,2,2,32,2,2,1)


def test_gradient_checkpointing_matches_all_parameters():
    a,b=model(True),model(False);b.load_state_dict(a.state_dict());s=sample(1)
    rng=torch.get_rng_state();la,_=a.training_loss(s);la.backward()
    torch.set_rng_state(rng);lb,_=b.training_loss(s);lb.backward()
    torch.testing.assert_close(la,lb)
    for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other
        assert p.grad is not None,name
        torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=2e-4,msg=name)


def test_save_restore_and_reject_baseline_schema(tmp_path):
    a=model();s=sample(1);opt=torch.optim.AdamW(a.parameters(),lr=1e-4)
    for _ in range(2):
        opt.zero_grad();loss,_=a.training_loss(s);loss.backward();opt.step()
    path=tmp_path/'paired.pt';a.save_checkpoint(path,opt,2)
    b=model();other=torch.optim.AdamW(b.parameters(),lr=1e-4)
    assert b.load_checkpoint(path,other)==2
    rng=torch.get_rng_state()
    opt.zero_grad();la,_=a.training_loss(s);la.backward();opt.step()
    torch.set_rng_state(rng)
    other.zero_grad();lb,_=b.training_loss(s);lb.backward();other.step()
    torch.testing.assert_close(la,lb,rtol=0,atol=0)
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    payload=torch.load(path,weights_only=True);payload['schema']='fastwam-cosmos-clean-action-force-chunk28-v3'
    torch.save(payload,path)
    with pytest.raises(ValueError,match='differ'):b.load_checkpoint(path)


def test_bf16_native_fp32_and_clean_action_hooks():
    m=model();m.video.net.use_wan_fp32_strategy=True
    for b in m.video.blocks:b.use_wan_fp32_strategy=True
    m.video.net.final_layer.use_wan_fp32_strategy=True;m.bfloat16()
    s=sample(1);seen=[]
    hook=m.dit['regions'].register_forward_pre_hook(lambda _,args:seen.append(args[1].clone()))
    with torch.backends.mkldnn.flags(enabled=False):
        loss,_=m.training_loss(s);loss.backward()
    hook.remove()
    assert len(seen)>=3
    for a in seen:torch.testing.assert_close(a,s['action'].bfloat16(),rtol=0,atol=0)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_region_force_mask_and_padding_are_safe():
    from fastwam.cosmos.regions import region_force_mask, RegionForceBridge
    from fastwam.forcewm.masks import build_force_attention_mask
    hp=torch.tensor([[True,False]]);ap=torch.tensor([[False,True]]);wp=torch.zeros_like(ap)
    direct=build_force_attention_mask(hp,ap,wp)
    valid=~torch.cat((hp,ap,wp),1)
    mask=region_force_mask(8,direct,valid)
    assert torch.equal(mask[:,:,8:,8:],direct)
    assert not mask[0,0,:8,8].any() and not mask[0,0,:8,11].any()
    bridge=RegionForceBridge(12,10,8,2)
    visual=torch.randn(1,2,2,1,2,8);force=torch.randn(1,6,10)
    maps=torch.ones(1,2,2,1,2,2,2)/4
    a=bridge(visual,force,maps,direct,valid)
    force[:,~valid[0]]=1000
    b=bridge(visual,force,maps,direct,valid)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_production_config_keeps_recipe_and_32_pairs():
    from pathlib import Path
    from hydra import initialize_config_dir,compose
    from omegaconf import OmegaConf
    from fastwam.cosmos.preflight import validate_config
    root=Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        cfg=compose(config_name='train',overrides=['task=cosmos_forcewm_paired_4gpu_30k',
            'model.expected_text_identity='+'a'*64,'data.train.expected_identity_sha256='+'b'*64])
    cfg=OmegaConf.to_container(cfg,resolve=True,throw_on_missing=True);validate_config(cfg)
    assert cfg['model']['region_pairs']==32 and cfg['model']['region_layer']==13
    assert cfg['batch_size']==16 and cfg['max_steps']==30000 and cfg['save_every']==5000


def test_validation_loss_in_eval_mode_still_has_auxiliary_metrics():
    m=model();m.eval()
    with torch.no_grad():loss,metrics=m.training_loss(sample(1))
    assert torch.isfinite(loss) and 'loss_region_reconstruction' in metrics
    assert not m._region_loss_requested
