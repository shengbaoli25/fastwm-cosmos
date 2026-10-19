"""Cosmos feature cache; immutable RGB/numeric sources, no decoder in training."""
from pathlib import Path
from collections import OrderedDict
import json
import torch
from safetensors.torch import load_file,save_file
from fastwam.cache.latent import file_sha256,cache_identity_sha256
from .assets import TOKENIZER_SHA

SCHEMA='cosmos-force28-full-window-cache-v1'
VIDEO_SHAPE=(16,3,28,56)
FIRST_SHAPE=(16,1,28,56)


def _path(root,relative):
    p=(root/relative).resolve()
    if not p.is_relative_to(root) or not p.is_file(): raise ValueError('Cache path missing/escapes root')
    return p


def _atomic(path,obj):
    p=Path(path); t=p.with_suffix('.tmp'); t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n'); t.replace(p)


def validate(root,expected=None,*,complete=True,production=True):
    root=Path(root).resolve(); m=json.loads((root/'manifest.json').read_text()); i=m['identity']
    if i.get('schema')!=SCHEMA or i.get('tokenizer_sha256')!=TOKENIZER_SHA or i.get('video_shape')!=list(VIDEO_SHAPE):
        raise ValueError('Wrong tokenizer/latent contract; Wan48 cache cannot be used')
    if i.get('camera_order')!=['cam_main','cam_wrist'] or i.get('rgb_shape')!=[3,9,224,448] or i.get('encoding')!='independent9frame_and_independent1frame':
        raise ValueError('Camera/window encoding differs')
    if m['identity_sha256']!=cache_identity_sha256(i) or expected is not None and expected!=m['identity_sha256']:
        raise ValueError('Cache identity differs')
    if production:
        profile=i.get('dataset_profile','forcevla-wipe50')
        scopes={'forcevla-wipe50':(50,34808),'realman-1018-test-cup-force-v1':(73,35611)}
        joint = profile == 'realman-leader-joint8-filtered-v1'
        filtered = profile == 'realman-filtered-force-v1' or joint
        lengths = i['episode_lengths']
        if (i.get('scope')!='production' or (not filtered and (profile not in scopes
                or (len(lengths),sum(lengths))!=scopes[profile]))
                or not lengths or any(type(n) is not int or n <= 0 for n in lengths)):
            raise ValueError('Production dataset profile/scope differs')
        if filtered or profile=='realman-1018-test-cup-force-v1':
            source=i.get('realman_source',{})
            if (source.get('profile')!=profile or source.get('episode_lengths')!=i['episode_lengths']
                    or cache_identity_sha256(source)!=i.get('numeric_identity')
                    or cache_identity_sha256(source.get('files',{}))!=source.get('source_inventory_sha256')
                    or (not joint and source.get('eef_columns')!=[8,9,10,11,12,13,7])
                    or source.get('cameras')!=['observation.images.global','observation.images.wrist']):
                raise ValueError('RealMan source lineage differs')
            if filtered:
                from realman_force.contract import validate_contract_identity
                validate_contract_identity(source.get('force_preprocessing', {}))
                if source.get('fps') != 30:
                    raise ValueError('RealMan force grid differs')
            if joint and (source.get('action_space') != 'joint8' or source.get('state_columns') != [0,1,2,3,4,5,6,19]
                          or source.get('action_units') != ['rad']*7+['openness_0_1']
                          or (source.get('action_dim'),source.get('state_dim'),source.get('history_dim')) != (8,8,14)):
                raise ValueError('Joint action/state semantics differ')
    if complete and m['status']!='complete': raise ValueError('Partial cache is not trainable')
    if m['status'] not in ['complete','partial']: raise ValueError('Invalid cache status')
    joint = i.get('dataset_profile') == 'realman-leader-joint8-filtered-v1'
    rows=[(ep,f,0,n) for ep,n in enumerate(i['episode_lengths']) for f in range(n)]
    if joint:
        segments=i['realman_source'].get('segments',[]);rows=[];previous=(-1,0)
        for ep,lo,hi in segments:
            if (not all(type(x) is int for x in (ep,lo,hi)) or not 0 <= ep < len(i['episode_lengths'])
                    or not 0 <= lo < hi <= i['episode_lengths'][ep] or hi-lo < 2
                    or ep < previous[0] or ep == previous[0] and lo < previous[1]):
                raise ValueError('Invalid continuous source segments')
            rows.extend((ep,f,lo,hi) for f in range(lo,hi));previous=(ep,hi)
        if not rows:raise ValueError('Missing joint training anchors')
    stop=0
    for e in m['shards']:
        if e['start']!=stop or e['stop']<=stop or e['stop']>len(rows): raise ValueError('Cache gaps/overlap')
        p=_path(root,e['path'])
        if file_sha256(p)!=e['sha256']: raise ValueError('Cache payload SHA differs')
        q=load_file(str(p)); n=e['stop']-e['start']
        for k,sh in [('video_latents',VIDEO_SHAPE),('first_frame_latents',FIRST_SHAPE)]:
            if q[k].shape!=(n,*sh) or q[k].dtype!=torch.bfloat16: raise ValueError('Invalid Cosmos features')
        if 'context' in q or 'image' in q: raise ValueError('No Wan text/RGB in Cosmos cache')
        for k,v in q.items():
            if len(v)!=n or v.is_floating_point() and not bool(torch.isfinite(v).all()): raise ValueError('Invalid feature rows')
        for k,col in [('episode_index',0),('frame_index',1)]:
            if q[k].tolist()!=[r[col] for r in rows[e['start']:e['stop']]]: raise ValueError('Episode/anchor mismatch')
        if q['sample_index'].tolist()!=list(range(e['start'],e['stop'])): raise ValueError('Anchor permutation')
        ids=q['cosmos_text_index']
        if ids.shape!=(n,) or ids.dtype!=torch.int64 or bool((ids<0).any() | (ids>=len(i['prompts'])).any()): raise ValueError('Bad Cosmos prompt index')
        width=8 if joint else 7;history_width=width+6
        required={'action':(32,width),'proprio':(32,width+1),'history':(32,history_width),'wrench_future':(32,6),
            'action_is_pad':(32,),'image_is_pad':(9,),'history_is_pad':(32,),'force_is_pad':(32,),
            'history_channel_valid':(32,history_width),'wrench_valid':(32,6),'raw_wrench_future64':(32,6)}
        for k,shape in required.items():
            if k not in q or q[k].shape!=(n,*shape): raise ValueError('Missing/wrong numeric feature '+k)
        for k in ['action_is_pad','image_is_pad','history_is_pad','force_is_pad','history_channel_valid','wrench_valid']:
            if q[k].dtype!=torch.bool: raise ValueError('Invalid Boolean mask '+k)
        if q['raw_wrench_future64'].dtype!=torch.float64: raise ValueError('FP64 future force required')
        for local,(ep,anchor,lo,hi) in enumerate(rows[e['start']:e['stop']]):
            for key,offsets in [('force_is_pad',torch.arange(1,33)),('history_is_pad',torch.arange(-31,1)),('action_is_pad',torch.arange(32)),('image_is_pad',torch.arange(0,33,4))]:
                t=anchor+offsets
                if not torch.equal(q[key][local],(t<lo)|(t>=hi)): raise ValueError('Temporal alignment/padding differs')
            if joint and (not torch.equal(q['history_channel_valid'][local],(~q['history_is_pad'][local,None].T).expand(-1,14))
                          or not torch.equal(q['wrench_valid'][local],(~q['force_is_pad'][local,None].T).expand(-1,6))):
                raise ValueError('Joint temporal/channel validity differs')
        stop=e['stop']
    if stop!=m['completed_samples'] or m['status']=='complete' and stop!=len(rows): raise ValueError('Incomplete inventory')
    expected_files={'manifest.json',*[e['path'] for e in m['shards']]}
    if {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}!=expected_files:
        raise ValueError('Untracked/partial files in feature cache')
    return m


def build_cache(root,*,pixels,numeric,encoder,text_manifest,text_identity,upstream_hashes,
                shard_size=32,resume=False,limit=None,production=True,staged=None):
    if pixels.manifest['identity']['episode_lengths']!=numeric.manifest['identity']['episode_lengths'] or pixels.manifest['identity']['data_conversion_sha256']!=numeric.manifest['identity']['data_conversion_sha256']:
        raise ValueError('RGB/numeric source lineage differs')
    if shard_size<1 or limit is not None and limit<1: raise ValueError('Positive shard/limit required')
    root=Path(root).resolve(); prompts=[x['prompt'] for x in text_manifest['texts']]
    identity=dict(schema=SCHEMA,scope='production' if production else 'TEST-only synthetic fixture',
        tokenizer_sha256=TOKENIZER_SHA,video_shape=list(VIDEO_SHAPE),rgb_shape=[3,9,224,448],
        camera_order=['cam_main','cam_wrist'],encoding='independent9frame_and_independent1frame',
        pixel_identity=pixels.identity_sha256,numeric_identity=numeric.identity_sha256,
        numeric_sidecar_identity=numeric.sidecar_identity_sha256,text_identity=text_identity,
        upstream_hashes=upstream_hashes,episode_lengths=numeric.manifest['identity']['episode_lengths'],prompts=prompts,
        exporter_sha256=file_sha256(__file__))
    if staged is not None:
        if staged.pixel_identity!=pixels.identity_sha256:raise ValueError('Stage/RGB pairing differs')
        identity['latent_stage_identity']=staged.identity_sha256
    digest=cache_identity_sha256(identity)
    if root.exists():
        if not resume: raise FileExistsError('Existing cache requires explicit resume')
        m=validate(root,digest,complete=False,production=production)
    else:
        root.mkdir(parents=True); (root/'shards').mkdir()
        m=dict(identity=identity,identity_sha256=digest,status='partial',completed_samples=0,shards=[])
        _atomic(root/'manifest.json',m)
    end=len(numeric) if limit is None else min(len(numeric),m['completed_samples']+limit)
    for start in range(m['completed_samples'],end,shard_size):
        samples=[]; stop=min(end,start+shard_size)
        for j in range(start,stop):
            sample=numeric[j]
            if int(sample['sample_index'])!=j: raise ValueError('Numeric reader substituted anchor')
            if staged is None:
                window=pixels.window(j).unsqueeze(0)
                with torch.inference_mode():
                    z=encoder(window); first=encoder(window[:,:,:1])
            else:
                z,first=staged[j]
            if not bool(torch.isfinite(z).all() & torch.isfinite(first).all()) or z.shape!=(1,*VIDEO_SHAPE) or first.shape!=(1,*FIRST_SHAPE): raise ValueError('Wrong official tokenizer output')
            item={k:v.detach().cpu().contiguous().clone() for k,v in sample.items() if isinstance(v,torch.Tensor) and k not in ['video_latents','context','context_mask','text_index']}
            item.update(video_latents=z[0].to('cpu',torch.bfloat16),first_frame_latents=first[0].to('cpu',torch.bfloat16),cosmos_text_index=torch.tensor(prompts.index(sample['prompt']),dtype=torch.int64))
            samples.append(item)
        payload={k:torch.stack([s[k] for s in samples]) for k in samples[0]}
        rel=f'shards/{start:06d}-{stop:06d}.safetensors'; p=root/rel; tmp=p.with_suffix('.tmp')
        save_file(payload,str(tmp)); tmp.replace(p)
        m['shards'].append(dict(start=start,stop=stop,path=rel,sha256=file_sha256(p)))
        m['completed_samples']=stop;m['status']='complete' if stop==len(numeric) else 'partial'
        _atomic(root/'manifest.json',m)
    return validate(root,digest,complete=m['status']=='complete',production=production)


class CosmosCachedDataset(torch.utils.data.Dataset):
    def __init__(self,cache_root,expected_identity_sha256,expected_text_identity,require_production=True,max_cached_shards=2):
        self.root=Path(cache_root).resolve(); self.manifest=validate(self.root,expected_identity_sha256,production=require_production)
        if self.manifest['identity']['text_identity']!=expected_text_identity: raise ValueError('Text/cache pairing differs')
        if max_cached_shards<1: raise ValueError('Positive shard capacity required')
        self.identity_sha256=self.manifest['identity_sha256'];self.capacity=max_cached_shards;self.loaded=OrderedDict()
        self.manifest_sha=file_sha256(self.root/'manifest.json')
        self.lookup=[(e,j-e['start']) for e in self.manifest['shards'] for j in range(e['start'],e['stop'])]
    def __len__(self):return len(self.lookup)
    def __getitem__(self,j):
        if type(j)!=int or not 0<=j<len(self):raise IndexError(j)
        if file_sha256(self.root/'manifest.json')!=self.manifest_sha:raise ValueError('Manifest changed during training')
        e,offset=self.lookup[j];key=e['path']
        if key not in self.loaded:
            p=_path(self.root,key)
            if file_sha256(p)!=e['sha256']:raise ValueError('Feature shard changed')
            self.loaded[key]=load_file(str(p))
            if len(self.loaded)>self.capacity:self.loaded.popitem(last=False)
        self.loaded.move_to_end(key)
        return {k:v[offset].clone() for k,v in self.loaded[key].items()}
