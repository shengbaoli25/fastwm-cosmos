"""Read complete image-only Cosmos shards for audited text/numeric assembly."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from fastwam.cache.latent import cache_identity_sha256,file_sha256
from fastwam.cache.pixels import pixel_rows
from .assets import TOKENIZER_SHA

EXPORTER_SHA='99ad39f777c515c9fb7f7a5e9867db1c24748a0b539663c1484c77da2290e2e1'


class StagedLatents:
    def __init__(self,root,pixels,upstream_hashes):
        self.root=Path(root).resolve();self.lookup=[];bindings=[];self.loaded=None
        run=json.loads((self.root/'run.json').read_text())
        if run.get('status')!='complete' or run.get('trainable') is not False:
            raise ValueError('Complete image-only stage required')
        self.rows=pixel_rows(pixels.manifest['identity'])
        total=len(self.rows);cursor=0
        for rank in range(4):
            directory=self.root/f'rank{rank}';manifest=directory/'manifest.json'
            m=json.loads(manifest.read_text());i=m['identity']
            expected=dict(schema='cosmos-image-only-stage-v1',pixel_identity=pixels.identity_sha256,
                pixel_manifest_sha=file_sha256(pixels.root/'manifest.json'),tokenizer_sha=TOKENIZER_SHA,
                upstream=upstream_hashes,exporter_sha=EXPORTER_SHA,offsets=list(range(0,33,4)),
                camera_order=['cam_main','cam_wrist'],video_shape=[16,3,28,56],first_shape=[16,1,28,56],
                encoding='independent9frame_and_independent1frame',start=total*rank//4,
                stop=total*(rank+1)//4,rank=rank,workers=4)
            if i!=expected or m['identity_sha256']!=cache_identity_sha256(i):
                raise ValueError('Image stage source/window identity differs')
            if m['status']!='complete' or m.get('trainable') is not False or m['completed_samples']!=i['stop']-i['start']:
                raise ValueError('Image stage rank incomplete')
            for e in m['shards']:
                path=(directory/e['path']).resolve()
                if not path.is_relative_to(directory) or e['start']!=cursor or not cursor<e['stop']<=i['stop']:
                    raise ValueError('Image stage overlap/gap/path')
                if file_sha256(path)!=e['sha256']:raise ValueError('Image stage payload SHA differs')
                self.lookup.extend((path,e,j-e['start']) for j in range(e['start'],e['stop']))
                cursor=e['stop']
            if cursor!=i['stop']:raise ValueError('Image stage missing rank anchors')
            bindings.append(dict(rank=rank,manifest_sha=file_sha256(manifest),identity=m['identity_sha256']))
        if cursor!=total:raise ValueError('Incomplete image stage')
        self.pixel_identity=pixels.identity_sha256
        self.identity_sha256=cache_identity_sha256(dict(run_sha=file_sha256(self.root/'run.json'),ranks=bindings))

    def __getitem__(self,j):
        path,e,offset=self.lookup[j]
        if self.loaded is None or self.loaded[0]!=path:
            if file_sha256(path)!=e['sha256']:raise ValueError('Image stage changed')
            q=load_file(str(path));n=e['stop']-e['start']
            for k,shape in [('video_latents',(n,16,3,28,56)),('first_frame_latents',(n,16,1,28,56))]:
                if q[k].shape!=shape or q[k].dtype!=torch.bfloat16 or not bool(torch.isfinite(q[k]).all()):
                    raise ValueError('Invalid Cosmos stage shape/dtype/finite')
            for col,key in enumerate(['sample_index','episode_index','frame_index']):
                if q[key].dtype!=torch.int64 or q[key].tolist()!=[r[col] for r in self.rows[e['start']:e['stop']]]:
                    raise ValueError('Image stage anchor correspondence differs')
            self.loaded=(path,q)
        q=self.loaded[1]
        return q['video_latents'][offset].unsqueeze(0),q['first_frame_latents'][offset].unsqueeze(0)
