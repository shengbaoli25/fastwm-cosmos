import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

BASE=Path(__file__).resolve().parents[2]/'scripts/cosmos/joint8_eval'
def load(name):
    spec=importlib.util.spec_from_file_location(name,BASE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_attention_probability_has_actual_mask_and_key_mass():
    probe=load('attention')
    q=torch.zeros(1,2,4,3);k=torch.zeros(1,2,5,3)
    valid=torch.tensor([[True,False,True,False,True]])
    probabilities=probe.mean_key_probabilities(q,k,valid)
    np.testing.assert_allclose(probabilities.sum(-1),1)
    np.testing.assert_array_equal(probabilities[...,~valid[0]],0)
    v=torch.randn(1,2,5,3)
    expected=np.einsum('bhk,bhkd->bhd',probabilities,v.numpy())
    actual=F.scaled_dot_product_attention(q,k,v,attn_mask=valid[:,None,None,:]).mean(-2)
    np.testing.assert_allclose(expected,actual.numpy(),rtol=1e-6,atol=1e-6)


def test_attention_probability_includes_additive_observation_bias():
    probe=load('attention')
    q=torch.zeros(1,2,3,4);k=torch.zeros(1,2,5,4)
    valid=torch.tensor([[True,False,True,True,False]])
    bias=torch.zeros(1,1,3,5);bias[...,2]=torch.log(torch.tensor(3.))
    probabilities=probe.mean_key_probabilities(q,k,valid,bias=bias)
    expected=np.array([1/5,0,3/5,1/5,0],dtype=np.float32)
    np.testing.assert_allclose(probabilities.mean(1)[0],expected,atol=1e-6)
    additive=bias.masked_fill(~valid[:,None,None,:],-torch.inf)
    actual=F.scaled_dot_product_attention(q,k,torch.eye(5).reshape(1,1,5,5).expand(1,2,5,5),
        attn_mask=additive).mean(-2).mean(1)
    np.testing.assert_allclose(probabilities.mean(1),actual.numpy(),atol=1e-6)


def test_probe_saves_raw_per_query_maps_only_at_selected_layer_and_step(tmp_path):
    probe=load('attention').AttentionProbe(object(),layers=(0,6,13,20,27),steps=(0,7,15))
    maps=torch.softmax(torch.randn(1,32,4,392),dim=-1)
    probe.active=6;probe.step=7
    probe._record_queries(None,(),(maps,torch.zeros(1,32,392),torch.tensor(0.)))
    probe.active=5
    probe._record_queries(None,(),(maps,torch.zeros(1,32,392),torch.tensor(0.)))
    path=tmp_path/'query_maps.npz';probe.save(path)
    saved=np.load(path)
    assert saved.files==['s07_l06_query_maps']
    np.testing.assert_allclose(saved[saved.files[0]],maps.numpy(),atol=0,rtol=0)


def test_anchors_cover_every_valid_action_once_without_crossing_gap():
    protocol=load('protocol')
    segments=[(0,3,70),(0,80,90),(1,0,35)]
    rows=[(ep,f) for ep,lo,hi in segments for f in range(lo,hi)]
    ids=protocol.anchors(segments,{r:i for i,r in enumerate(rows)})
    covered=[]
    for j in ids:
        ep,f=rows[j];hi=next(hi for e,lo,hi in segments if e==ep and lo<=f<hi)
        covered.extend((ep,k) for k in range(f,min(f+32,hi)))
    assert covered==rows


def test_physical_metrics_exclude_padding_and_reject_nan():
    import pytest
    protocol=load('protocol')
    pred=np.array([[[1.,2.],[100.,2.]]]);target=np.zeros_like(pred)
    valid=np.array([[[True,True],[False,True]]])
    assert protocol.channel_metrics(pred,target,valid)['mae']==[1.,2.]
    pred[0,0,0]=np.nan
    with pytest.raises(ValueError):protocol.channel_metrics(pred,target,valid)


def test_real_joint_sampler_is_unchanged_with_attention_probe():
    from test_noaction import model
    from conftest import sample
    m=model();s=sample(1)
    probe=load('attention').AttentionProbe(m,layers=(0,1),steps=(0,1))
    with torch.inference_mode():
        expected=m.infer_joint(s,num_inference_steps=2)
        with probe.recording():actual=m.infer_joint(s,num_inference_steps=2)
    for k in expected:torch.testing.assert_close(expected[k],actual[k],atol=0,rtol=0)
    assert len(probe.records)==12
    for probabilities in probe.records.values():
        np.testing.assert_allclose(probabilities.sum(-1),1,atol=1e-6)


def test_roi_area_is_corrected_and_overlapping_boxes_are_not_counted_twice():
    roi=load('roi')
    heat=np.ones((14,28))/392
    stat=roi.region_statistics(heat,[[0,0,112,224],[0,0,56,224]],0)
    assert stat['area_fraction']==.5
    np.testing.assert_allclose(stat['view_attention_mass'],.5)
    np.testing.assert_allclose(stat['within_view_fraction'],.5)
    np.testing.assert_allclose(stat['enrichment'],1)


def test_cpu_recovery_rejects_missing_predicted_windows(tmp_path,monkeypatch):
    import json,pytest
    monkeypatch.syspath_prepend(str(BASE))
    recover=load('finalize_cpu')
    (tmp_path/'protocol.json').write_text(json.dumps(dict(status='running',limited_smoke=False,windows=2,episodes=1)))
    np.savez_compressed(tmp_path/'predictions.npz',sample=np.array([0]))
    with pytest.raises(ValueError,match='Full unique predicted windows required'):
        recover.validate(tmp_path)


def test_attention_renderer_accepts_20k_25k_pair_and_rejects_reversed_order(tmp_path,monkeypatch):
    import json,pytest
    monkeypatch.syspath_prepend(str(BASE))
    renderer=load('render_attention_video')
    shared=dict(status='complete',source_sha256='source',source_identity='source-id',
        dataset_identity='dataset-id',video_episode=15,windows=974,seeds=dict(action=42,force=43,video=44),
        probe_indices=[1,2])
    paths=[]
    for name,step in [('baseline',20000),('posttrain',25000)]:
        path=tmp_path/name;path.mkdir()
        (path/'protocol.json').write_text(json.dumps(dict(shared,step=step)))
        paths.append(path)
    assert [p['step'] for p in renderer.load_runs(paths)]==[20000,25000]
    with pytest.raises(ValueError,match='ordered baseline then later checkpoint'):
        renderer.load_runs(paths[::-1])
