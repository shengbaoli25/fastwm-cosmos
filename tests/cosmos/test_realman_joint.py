"""Joint labels, causal force, discontinuity masks and online input parity."""
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from safetensors.torch import save_file

from fastwam.cosmos.realman_joint import RealManJointSource,open_realman_source
from fastwam.cosmos.cache import SCHEMA,VIDEO_SHAPE,FIRST_SHAPE,validate
from fastwam.cosmos.assets import TOKENIZER_SHA
from fastwam.cache.latent import file_sha256,cache_identity_sha256
from realman_force.contract import input_contract
from realman_deploy.normalization import DatasetNormalizer,OnlineHistory


def dataset(root):
    calibration=json.loads((Path(__file__).parents[2]/'deploy/force_calibration.json').read_text())
    for directory in ('meta/episodes','meta/force_sessions','data'):(root/directory).mkdir(parents=True)
    n=45
    (root/'meta/info.json').write_text(json.dumps(dict(codebase_version='v3.0',fps=30,total_frames=n,
        total_episodes=1,data_path='data/rows.parquet')))
    (root/'meta/force.json').write_text(json.dumps(dict(schema='realman_leader_force_gripper_v2',model_input=input_contract(calibration))))
    (root/'meta/leader.json').write_text('{"mode":"execute","training_demonstration":true}')
    (root/'meta/force_sessions/episode_000000.json').write_text(json.dumps(calibration))
    pq.write_table(pa.Table.from_pylist([dict(episode_index=0,length=n,dataset_from_index=0,dataset_to_index=n,
        tasks=['cup'],**{'data/chunk_index':0,'data/file_index':0})]),root/'meta/episodes/rows.parquet')
    rows=[]
    for i in range(n):
        t=1000+i/30+(1 if i >= 20 else 0)
        force=(np.arange(6)+i/10).tolist()
        state=[i/100+j/10 for j in range(7)]+[99.]*6+force+[.2+i/100]
        rows.append(dict(episode_index=0,frame_index=i,index=i,timestamp=i/30,action=[i/50+j/10 for j in range(7)]+[.1+i/100],**{
            'observation.state':state,'observation.force_filtered':force,
            'observation.force_timestamp_s':t,'observation.force_filter_timestamp_s':t,'observation.robot_timestamp_s':t,
            'observation.monotonic_time_s':t+.001,'action.timestamp_s':t-.002,'observation.leader_timestamp_s':t-.003,
            'observation.gripper_timestamp_s':t-.005,'action.gripper_timestamp_s':t-10,
            'observation.camera.arrival_time_s':[t-.002,t-.001],
            'observation.capture_fresh':[1,1,1,1] if i != 12 else [0,1,1,1],
            'action.applied':1,'observation.force_filter_input_count':200+i,
            'observation.force_filter_epoch':1 if i < 30 else 2,
            'observation.capture_discontinuity':int(i==20),'observation.capture_gap_s':1. if i==20 else 0.,
            'action.follow_session_id':1 if i < 35 else 2,
            'action.leader_anchor':[0.]*7,'action.follower_anchor':[0.]*7}))
    pq.write_table(pa.Table.from_pylist(rows),root/'data/rows.parquet')


def test_joint_targets_and_discontinuities_match_online_inputs(tmp_path):
    dataset(tmp_path);s=open_realman_source(tmp_path)
    assert isinstance(s,RealManJointSource)
    assert s.segments==[[0,0,12],[0,13,20],[0,20,30],[0,30,35],[0,35,45]]
    assert len(s)==44 and s.identity['state_columns']==[0,1,2,3,4,5,6,19]
    np.testing.assert_allclose(s.raw['state'][8,:7],np.arange(7)/10+.08)
    np.testing.assert_allclose(s.raw['action'][8,:7],np.arange(7)/10+.16)
    path=tmp_path/'normalization.json';s.export_normalization(path)
    norm=DatasetNormalizer.from_json(path);history=OnlineHistory(norm)
    last_segment=None
    for j,(ep,f,start,_) in enumerate(s.rows):
        bounds=s.bounds[j]
        if bounds!=last_segment:history=OnlineHistory(norm);last_segment=bounds
        history.append(s.raw['state'][start+f].tolist(),s.raw['force'][start+f].tolist())
        values,pad,valid=history.tensors();item=s[j]
        np.testing.assert_array_equal(values,item['history'])
        assert pad==item['history_is_pad'].tolist() and valid==item['history_channel_valid'].tolist()
        assert item['proprio'].shape==(32,9) and item['action'].shape==(32,8)
    assert s[11]['force_is_pad'].all() and s[12]['frame_index']==13
    s._video_episode=0;s._frames=torch.arange(45)[:,None,None,None].expand(-1,3,2,2)
    assert s.window(11).flatten().unique().tolist()==[11]


def test_cache_tracks_retained_original_indices_and_segment_masks(tmp_path):
    raw=tmp_path/'raw';dataset(raw);s=RealManJointSource(raw)
    root=tmp_path/'cache';root.mkdir()
    samples=[]
    for j in range(len(s)):
        item=s[j];item.update(video_latents=torch.zeros(VIDEO_SHAPE,dtype=torch.bfloat16),
            first_frame_latents=torch.zeros(FIRST_SHAPE,dtype=torch.bfloat16));samples.append(item)
    q={k:torch.stack([item[k] for item in samples]) for k in samples[0]}
    save_file(q,str(root/'data.safetensors'))
    identity=dict(schema=SCHEMA,tokenizer_sha256=TOKENIZER_SHA,video_shape=list(VIDEO_SHAPE),
        rgb_shape=[3,9,224,448],camera_order=['cam_main','cam_wrist'],encoding='independent9frame_and_independent1frame',
        scope='production',dataset_profile=s.identity['profile'],realman_source=s.identity,
        episode_lengths=s.lengths,numeric_identity=s.identity_sha256,prompts=['cup'])
    manifest=dict(identity=identity,identity_sha256=cache_identity_sha256(identity),status='complete',
        completed_samples=len(s),shards=[dict(start=0,stop=len(s),path='data.safetensors',sha256=file_sha256(root/'data.safetensors'))])
    (root/'manifest.json').write_text(json.dumps(manifest));validate(root)
    q['force_is_pad'][11,0]=False;save_file(q,str(root/'data.safetensors'))
    manifest['shards'][0]['sha256']=file_sha256(root/'data.safetensors');(root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Temporal'):validate(root)
