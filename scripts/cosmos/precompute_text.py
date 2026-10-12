"""Use the pinned official Cosmos TextEncoder in its supported environment.

Run in a fresh process with the upstream Cosmos dependencies installed; do not
substitute Wan T5, an empty prompt, random tensors, or a width projection.
"""
import argparse,json,sys
from pathlib import Path
import torch
from safetensors.torch import save_file
from fastwam.cache.latent import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--prompts',required=True,help='JSON list of exact dataset prompts')
    p.add_argument('--reason1-checkpoint',required=True,help='Local official Cosmos-Reason1 checkpoint directory')
    p.add_argument('--upstream',default='third_party/cosmos-predict2.5');p.add_argument('--output',required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:p.error('Text encoding requires explicit --execute')
    prompts=json.loads(Path(a.prompts).read_text())
    if not isinstance(prompts,list) or not prompts or any(not isinstance(x,str) or not x for x in prompts) or len(set(prompts))!=len(prompts):
        raise ValueError('Unique real dataset task strings required')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    source=Path(a.reason1_checkpoint).resolve()
    files={p.relative_to(source).as_posix():file_sha256(p) for p in source.rglob('*') if p.is_file() and not p.relative_to(source).as_posix().startswith('.cache/')}
    from fastwam.cosmos import assets
    pinned=json.loads(Path(assets.__file__).with_name('reason1_weights.json').read_text())
    if any(files.get(k)!=v for k,v in pinned['weights'].items()):
        raise ValueError('All official pinned Reason1 weight shard hashes must match')
    sys.path.insert(0,str(Path(a.upstream).resolve()))
    from cosmos_predict2._src.predict2.text_encoders.text_encoder import TextEncoder,TextEncoderConfig
    from cosmos_predict2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy
    cfg=TextEncoderConfig(ckpt_path=str(source),embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT))
    # Upstream uses strict=False; refuse any missing actual parameter locally.
    from cosmos_predict2._src.predict2.text_encoders.reason1 import QwenVLBaseModel
    original_load=QwenVLBaseModel.load_state_dict
    load_receipt={}
    def checked_load(model,state_dict,strict=True,assign=False):
        required={k for k,_ in model.named_parameters()}
        missing=sorted(required-set(state_dict))
        if missing:raise ValueError('Incomplete Reason1 parameter initialization: '+str(missing[:20]))
        incompatible=original_load(model,state_dict,strict=False,assign=assign)
        bad_missing=[k for k in incompatible.missing_keys if not k.endswith('._extra_state')]
        bad_extra=[k for k in incompatible.unexpected_keys if not k.endswith('._extra_state')]
        if bad_missing or bad_extra:raise ValueError('Reason1 strict parameter/key mismatch: '+str((bad_missing[:20],bad_extra[:20])))
        load_receipt.update(parameters=len(required),missing_parameters=missing,unexpected=bad_extra)
        return incompatible
    QwenVLBaseModel.load_state_dict=checked_load
    try:encoder=TextEncoder(cfg,device='cuda')
    finally:QwenVLBaseModel.load_state_dict=original_load
    if not load_receipt:raise ValueError('No official Reason1 load observed')
    entries=[]
    for j,prompt in enumerate(prompts):
        with torch.no_grad():x=encoder.compute_text_embeddings_online({'prompt':[prompt]},'prompt')[0].to('cpu',torch.bfloat16)
        if x.shape!=(512,100352) or not bool(torch.isfinite(x).all()):raise ValueError('Unexpected official text features')
        name=f'text_{j:03d}.safetensors';save_file({'context':x.contiguous()},str(root/name))
        entries.append(dict(prompt=prompt,path=name,sha256=file_sha256(root/name)))
    m=dict(schema='cosmos-reason1-full-concat-v1',status='complete',encoder='nvidia/Cosmos-Reason1-7B',
        encoder_revision='3210bec0495fdc7a8d3dbb8d58da5711eab4b423',encoder_files=files,feature_dim=100352,
        strict_load=load_receipt,source_sha256=file_sha256(Path(a.upstream)/'cosmos_predict2/_src/predict2/text_encoders/text_encoder.py'),texts=entries)
    (root/'manifest.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
