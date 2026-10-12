"""Offline official Reason1 features with pinned NVIDIA single-device SDPA."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import save_file
from fastwam.cache.latent import file_sha256
from fastwam.cosmos import assets
from fastwam.cosmos.reason1_native import load_reason1,prompt_ids,full_concat,NETWORK_SHA


def main():
    p=argparse.ArgumentParser()
    for k in ['prompts','reason1-checkpoint','upstream','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:p.error('Explicit --execute required')
    torch.set_num_threads(8)
    prompts=json.loads(Path(a.prompts).read_text())
    if not isinstance(prompts,list) or not prompts or any(not isinstance(x,str) or not x for x in prompts) or len(set(prompts))!=len(prompts):
        raise ValueError('Unique recorded prompts required')
    source=Path(a.reason1_checkpoint).resolve();root=Path(a.output)
    pinned=json.loads(Path(assets.__file__).with_name('reason1_weights.json').read_text())
    index=json.loads((source/'model.safetensors.index.json').read_text())
    if set(index['weight_map'].values())!=set(pinned['weights']):raise ValueError('Wrong shard index')
    files={name:file_sha256(source/name) for name in [*pinned['weights'],'model.safetensors.index.json','config.json','tokenizer.json','tokenizer_config.json','preprocessor_config.json','chat_template.json']}
    if any(files.get(k)!=v for k,v in pinned['weights'].items()):raise ValueError('Official Reason1 SHA differs')
    root.mkdir(parents=True,exist_ok=False)
    model,processor,receipt=load_reason1(a.upstream,source)
    print(json.dumps({'status':'strict_official_reason1_loaded',**receipt}),flush=True)
    entries=[]
    for j,prompt in enumerate(prompts):
        ids=prompt_ids(processor,prompt,'cuda')
        with torch.inference_mode():
            states=model.model(ids,attention_mask=None,use_cache=False,output_hidden_states=True,return_dict=True).hidden_states
            if len(states)!=29:raise ValueError('Expected embedding plus 28 Reason1 layers')
            x=full_concat(states)[0].to('cpu',torch.bfloat16).contiguous()
        if x.shape!=(512,100352) or not bool(torch.isfinite(x).all()):raise ValueError('Invalid full text context')
        name=f'text_{j:03d}.safetensors';save_file({'context':x},str(root/name))
        entries.append(dict(prompt=prompt,path=name,sha256=file_sha256(root/name),input_ids=ids[0].cpu().tolist()))
    m=dict(schema='cosmos-reason1-full-concat-v1',status='complete',encoder=pinned['model'],encoder_revision=pinned['revision'],
           encoder_files=files,feature_dim=100352,strict_load=receipt,source_sha256=NETWORK_SHA,
           exporter_sha256=file_sha256(__file__),texts=entries)
    (root/'manifest.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':'complete','prompts':len(entries),'output':str(root)}),flush=True)


if __name__=='__main__':main()
