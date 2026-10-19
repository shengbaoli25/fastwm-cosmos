"""Make an independent, byte-preserving video copy of complete joint episodes."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from fastwam.cache.latent import file_sha256, cache_identity_sha256
from fastwam.cosmos.realman_joint import RealManJointSource


def inventory(root):
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob('*')) if p.is_file()}


def merge(sources, output, expected_episodes):
    sources = [Path(x).resolve() for x in sources]; output = Path(output).resolve()
    if output.exists() or any(output == r or r in output.parents for r in sources):
        raise ValueError('Fresh output outside the source directories required')
    snapshots = [inventory(r) for r in sources]
    infos = [json.loads((r/'meta/info.json').read_text()) for r in sources]
    for info in infos[1:]:
        if any(info[k] != infos[0][k] for k in ('features','fps','codebase_version','robot_type')):
            raise ValueError('Source formats differ')
    if sum(i['total_episodes'] for i in infos) != expected_episodes:
        raise ValueError('Source episode count changed')
    for name in ('force.json','leader.json','cameras.json','tasks.parquet'):
        if len({s['meta/'+name] for s in snapshots}) != 1:
            raise ValueError('Source metadata differ: '+name)
    output.mkdir(parents=True)
    for name in ('force.json','leader.json','cameras.json','tasks.parquet'):
        (output/'meta').mkdir(exist_ok=True)
        shutil.copy2(sources[0]/'meta'/name,output/'meta'/name)
    episodes=[]; mappings=[]; cursor=0; video_copies=[]
    for root,info in zip(sources,infos):
        rows=sorted([e for p in sorted((root/'meta/episodes').rglob('*.parquet'))
                     for e in pq.read_table(p).to_pylist()],key=lambda e:e['episode_index'])
        for old in rows:
            ep=len(episodes);n=old['length'];row=copy.deepcopy(old)
            table=pq.read_table(root/info['data_path'].format(chunk_index=old['data/chunk_index'],file_index=old['data/file_index']))
            if set(table['episode_index'].to_pylist()) != {old['episode_index']} or len(table) != n:
                raise ValueError('This merger requires one complete episode per source file')
            for key,values in [('episode_index',[ep]*n),('index',list(range(cursor,cursor+n)))]:
                table=table.set_column(table.schema.get_field_index(key),table.schema.field(key),pa.array(values,type=pa.int64()))
                vals=np.asarray(values)
                stats=dict(min=vals.min(),max=vals.max(),mean=vals.mean(),std=vals.std())
                stats.update({q:np.quantile(vals,float(q[1:])/100) for q in ('q01','q10','q50','q90','q99')})
                for stat,value in stats.items():
                    if f'stats/{key}/{stat}' in row:row[f'stats/{key}/{stat}']=[float(value)]
            data=output/f'data/chunk-000/file-{ep:03d}.parquet';data.parent.mkdir(parents=True,exist_ok=True)
            pq.write_table(table,data)
            row.update(episode_index=ep,dataset_from_index=cursor,dataset_to_index=cursor+n)
            row.update({'data/chunk_index':0,'data/file_index':ep,'meta/episodes/chunk_index':0,'meta/episodes/file_index':0})
            for camera in ('observation.images.global','observation.images.wrist'):
                prefix='videos/'+camera+'/'
                source=root/info['video_path'].format(video_key=camera,chunk_index=old[prefix+'chunk_index'],file_index=old[prefix+'file_index'])
                dest=output/f'videos/{camera}/chunk-000/file-{ep:03d}.mp4';dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,dest)
                if file_sha256(dest) != snapshots[sources.index(root)][str(source.relative_to(root))]:
                    raise ValueError('Video copy differs')
                video_copies.append(str(dest.relative_to(output)))
                row[prefix+'chunk_index']=0;row[prefix+'file_index']=ep
            calibration=root/'meta/force_sessions'/f'episode_{old["episode_index"]:06d}.json'
            dest=output/'meta/force_sessions'/f'episode_{ep:06d}.json';dest.parent.mkdir(exist_ok=True)
            shutil.copy2(calibration,dest)
            mappings.append(dict(source=str(root),source_episode=old['episode_index'],episode=ep,length=n))
            episodes.append(row);cursor+=n
    info=copy.deepcopy(infos[0]);info.update(total_episodes=len(episodes),total_frames=cursor,splits={'train':f'0:{len(episodes)}'})
    (output/'meta/info.json').write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n')
    target=output/'meta/episodes/chunk-000/file-000.parquet';target.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episodes),target)
    # LeRobot's population-moment pooling, with per-episode image statistics retained.
    grouped={}
    for e in episodes:
        values={}
        for key,value in e.items():
            if key.startswith('stats/'):
                feature,stat=key[6:].rsplit('/',1);values.setdefault(feature,{})[stat]=np.asarray(value)
        for feature,stats in values.items():grouped.setdefault(feature,[]).append(stats)
    pooled={}
    for feature,items in grouped.items():
        total=sum(s['count'] for s in items);mean=sum(s['mean']*s['count'] for s in items)/total
        var=sum((s['std']**2+(s['mean']-mean)**2)*s['count'] for s in items)/total
        pooled[feature]={k:v.tolist() for k,v in dict(min=np.minimum.reduce([s['min'] for s in items]),
            max=np.maximum.reduce([s['max'] for s in items]),mean=mean,std=np.sqrt(var),count=total).items()}
    (output/'meta/stats.json').write_text(json.dumps(pooled,indent=2)+'\n')
    if [inventory(r) for r in sources] != snapshots:raise ValueError('A source changed during snapshot')
    receipt=dict(source_inventories=[dict(path=str(r),sha256=cache_identity_sha256(s)) for r,s in zip(sources,snapshots)],
        episodes=mappings,total_episodes=len(episodes),total_frames=cursor,video_files=len(video_copies),videos_reencoded=False)
    (output/'meta/merge.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    source=RealManJointSource(output)
    return dict(output=str(output),episodes=len(episodes),frames=cursor,retained_anchors=len(source),
                source_sha256=source.source_sha,numeric_identity=source.identity_sha256)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--expected-episodes',type=int,required=True);a=p.parse_args()
    print(json.dumps(merge(a.sources,a.output,a.expected_episodes),ensure_ascii=False),flush=True)
