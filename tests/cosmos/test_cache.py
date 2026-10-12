"""Synthetic cache tests: complete windows, exact +1 labels and fail-closed reads."""
from types import SimpleNamespace
import json
import torch
import pytest
from safetensors.torch import load_file,save_file
from fastwam.cosmos.cache import build_cache,validate,CosmosCachedDataset,VIDEO_SHAPE,FIRST_SHAPE
from fastwam.cache.latent import cache_identity_sha256,file_sha256

class Numeric:
    identity_sha256='n'*64;sidecar_identity_sha256='s'*64
    manifest={'identity':{'episode_lengths':[2,1],'data_conversion_sha256':'c'*64}}
    def __len__(self):return 3
    def __getitem__(self,j):
        ep,anchor,length=(0,j,2) if j<2 else (1,0,1)
        out=dict(sample_index=torch.tensor(j),episode_index=torch.tensor(ep),frame_index=torch.tensor(anchor),
            action=torch.ones(32,7)*j,proprio=torch.ones(32,8),history=torch.ones(32,13),
            wrench_future=torch.ones(32,6),raw_wrench_future64=torch.ones(32,6,dtype=torch.float64),
            history_channel_valid=torch.ones(32,13,dtype=torch.bool),wrench_valid=torch.ones(32,6,dtype=torch.bool),
            context=torch.ones(1),video_latents=torch.zeros(48,3,14,28),prompt='wipe')
        for k,off in [('action_is_pad',torch.arange(32)),('history_is_pad',torch.arange(-31,1)),('force_is_pad',torch.arange(1,33)),('image_is_pad',torch.arange(0,33,4))]:
            t=anchor+off;out[k]=(t<0)|(t>=length)
        return out

class Pixels:
    manifest=Numeric.manifest;identity_sha256='p'*64
    def window(self,j):return torch.ones(3,9,224,448)*j


def build(root,**kwargs):
    calls=[]
    def encode(x):
        calls.append(tuple(x.shape));sh=VIDEO_SHAPE if x.shape[2]==9 else FIRST_SHAPE
        return torch.full((1,*sh),float(x.flatten()[0]))
    m=build_cache(root,pixels=Pixels(),numeric=Numeric(),encoder=encode,
        text_manifest={'texts':[{'prompt':'wipe'}]},text_identity='t'*64,upstream_hashes={},
        shard_size=2,production=False,**kwargs)
    return m,calls


def test_offline_whole_windows_and_cached_training(tmp_path,monkeypatch):
    root=tmp_path/'features';m,calls=build(root)
    assert calls==[(1,3,9,224,448),(1,3,1,224,448)]*3
    ds=CosmosCachedDataset(root,m['identity_sha256'],'t'*64,require_production=False)
    monkeypatch.setattr(Pixels,'window',lambda *a:(_ for _ in ()).throw(AssertionError('RGB training read')))
    assert len(ds)==3
    assert ds[2]['video_latents'].shape==VIDEO_SHAPE
    assert ds[2]['first_frame_latents'].shape==FIRST_SHAPE
    assert ds[2]['force_is_pad'].all() and not ds[2]['action_is_pad'][0]
    assert not {'image','context','context_mask'}&ds[1].keys()
    with pytest.raises(ValueError,match='Production'):CosmosCachedDataset(root,m['identity_sha256'],'t'*64)


def test_resume_partial_requires_explicit_opt_in(tmp_path):
    root=tmp_path/'features';m,calls=build(root,limit=1)
    assert m['status']=='partial'
    with pytest.raises(ValueError,match='Partial'):validate(root,production=False)
    with pytest.raises(FileExistsError):build(root)
    done,calls=build(root,resume=True);assert done['completed_samples']==3 and len(calls)==4


@pytest.mark.parametrize('mutation',['wan48','wrong_force_time','wrong_sample','nonfinite','bad_prompt_shape','missing_fp64','rgb','checksum'])
def test_corrupt_payloads_rejected(tmp_path,mutation):
    root=tmp_path/'features';m,_=build(root);e=m['shards'][0];p=root/e['path'];q=load_file(str(p))
    if mutation=='wan48':q['video_latents']=torch.zeros(2,48,3,14,28,dtype=torch.bfloat16)
    elif mutation=='wrong_force_time':q['force_is_pad']=q['action_is_pad'].clone()
    elif mutation=='wrong_sample':q['sample_index']=torch.tensor([1,0])
    elif mutation=='nonfinite':q['action'][0,0,0]=float('nan')
    elif mutation=='bad_prompt_shape':q['cosmos_text_index']=torch.zeros(2,1,dtype=torch.int64)
    elif mutation=='missing_fp64':q['raw_wrench_future64']=q['raw_wrench_future64'].float()
    elif mutation=='rgb':q['image']=torch.zeros(2,3,9,224,448)
    else:q['action'].add_(1)
    save_file(q,str(p))
    if mutation!='checksum':
        e['sha256']=file_sha256(p);(root/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):validate(root,production=False)


def test_mutation_after_open_rejected(tmp_path):
    root=tmp_path/'features';m,_=build(root);ds=CosmosCachedDataset(root,m['identity_sha256'],'t'*64,False)
    (root/'manifest.json').write_text(json.dumps(m)+' ')
    with pytest.raises(ValueError,match='changed'):ds[0]


def test_staged_assembly_does_not_decode_or_encode(monkeypatch,tmp_path):
    class Stage:
        pixel_identity=Pixels.identity_sha256;identity_sha256='s'*64
        def __getitem__(self,j):
            return (torch.full((1,*VIDEO_SHAPE),j,dtype=torch.bfloat16),
                    torch.full((1,*FIRST_SHAPE),j,dtype=torch.bfloat16))
    monkeypatch.setattr(Pixels,'window',lambda *a:(_ for _ in ()).throw(AssertionError('No RGB read for assembly')))
    m=build_cache(tmp_path/'cache',pixels=Pixels(),numeric=Numeric(),encoder=None,
        text_manifest={'texts':[{'prompt':'wipe'}]},text_identity='t'*64,upstream_hashes={},
        shard_size=2,production=False,staged=Stage())
    assert m['status']=='complete' and m['identity']['latent_stage_identity']=='s'*64
    ds=CosmosCachedDataset(tmp_path/'cache',m['identity_sha256'],'t'*64,False)
    assert torch.equal(ds[2]['video_latents'],torch.full(VIDEO_SHAPE,2,dtype=torch.bfloat16))
