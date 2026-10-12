import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from safetensors.torch import save_file
from fastwam.cache.latent import cache_identity_sha256,file_sha256
from fastwam.cosmos.assets import TOKENIZER_SHA
from fastwam.cosmos.latent_stage import StagedLatents,EXPORTER_SHA


def fixture(root):
    pixel=root/'pixels';pixel.mkdir();(pixel/'manifest.json').write_text('pixel fixture')
    pixels=SimpleNamespace(root=pixel,identity_sha256='p'*64,manifest={'identity':{'episode_lengths':[8]}})
    stage=root/'stage';stage.mkdir();(stage/'run.json').write_text(json.dumps({'status':'complete','trainable':False}))
    for rank in range(4):
        folder=stage/f'rank{rank}';folder.mkdir();start=rank*2;stop=start+2
        q={k:torch.full((2,*shape),rank,dtype=torch.bfloat16) for k,shape in [('video_latents',(16,3,28,56)),('first_frame_latents',(16,1,28,56))]}
        q.update(sample_index=torch.arange(start,stop),episode_index=torch.zeros(2,dtype=torch.int64),frame_index=torch.arange(start,stop))
        save_file(q,str(folder/'s.safetensors'))
        i=dict(schema='cosmos-image-only-stage-v1',pixel_identity='p'*64,pixel_manifest_sha=file_sha256(pixel/'manifest.json'),
            tokenizer_sha=TOKENIZER_SHA,upstream={},exporter_sha=EXPORTER_SHA,offsets=list(range(0,33,4)),camera_order=['cam_main','cam_wrist'],
            video_shape=[16,3,28,56],first_shape=[16,1,28,56],encoding='independent9frame_and_independent1frame',start=start,stop=stop,rank=rank,workers=4)
        m=dict(identity=i,identity_sha256=cache_identity_sha256(i),status='complete',trainable=False,completed_samples=2,
            shards=[dict(path='s.safetensors',start=start,stop=stop,sha256=file_sha256(folder/'s.safetensors'))])
        (folder/'manifest.json').write_text(json.dumps(m))
    return stage,pixels


def test_stage_covers_all_anchors_once(tmp_path):
    root,pixels=fixture(tmp_path);stage=StagedLatents(root,pixels,{})
    assert len(stage.lookup)==8
    for j in range(8):assert (stage[j][0]==j//2).all()


@pytest.mark.parametrize('failure',['partial','wrong_camera','gap','changed_payload'])
def test_stage_rejects_partial_wrong_source_or_corruption(tmp_path,failure):
    root,pixels=fixture(tmp_path);path=root/'rank2/manifest.json';m=json.loads(path.read_text())
    if failure=='partial':m['status']='partial'
    elif failure=='wrong_camera':m['identity']['camera_order'].reverse();m['identity_sha256']=cache_identity_sha256(m['identity'])
    elif failure=='gap':m['shards'][0]['start']+=1
    else:(root/'rank2/s.safetensors').write_bytes(b'changed')
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):StagedLatents(root,pixels,{})
