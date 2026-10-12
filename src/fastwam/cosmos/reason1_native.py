"""Pinned NVIDIA Reason1 network, using its existing single-device SDPA path.

The upstream network contains an unconditional flash-attn import assertion even
when its SDPA implementation is selected. Only that assertion is omitted during
module loading. The network, weights, RoPE, padding and FULL_CONCAT math remain
upstream. No CUDA extension installation or online tokenizer download is needed.
"""
import ast
import importlib.util
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen2_5_VLConfig
from fastwam.cache.latent import file_sha256

NETWORK='cosmos_predict2/_src/reason1/networks/qwen2_5_vl.py'
NETWORK_SHA='ec6925e5c06fd3881141fa993e34bc841757c22a2e01766f3ff0a6d1295c5726'


def native_network(upstream):
    path=Path(upstream)/NETWORK
    if file_sha256(path)!=NETWORK_SHA: raise ValueError('Pinned Reason1 network differs')
    tree=ast.parse(path.read_text()); removed=0
    for i,node in enumerate(tree.body):
        if isinstance(node,ast.Assert) and ast.unparse(node.test)=='is_flash_attn_2_available()':
            tree.body[i]=ast.copy_location(ast.Pass(),node); removed+=1
    if removed!=1: raise ValueError('Expected one optional flash-attn assertion')
    name='_cosmos_native_reason1_sdpa'
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),mod.__dict__)
    return mod


def full_concat(hidden_states):
    # Exact upstream TextEncoder.mean_normalize and FULL_CONCAT ordering.
    return torch.cat([(x-x.mean(dim=-1,keepdim=True))/x.std(dim=-1,keepdim=True).add(1e-8)
                      for x in hidden_states[1:]],dim=-1)


def prompt_ids(processor,prompt,device):
    messages=[{'role':'system','content':[{'type':'text','text':'You are a helpful assistant who will provide prompts to an image generator.'}]},
              {'role':'user','content':[{'type':'text','text':prompt}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,add_vision_id=False)
    ids=processor(text=[text],images=None,videos=None,padding=False,return_tensors='pt')['input_ids'][0].tolist()
    pad=processor.tokenizer.pad_token_id
    ids=(ids+[pad]*max(0,512-len(ids)))[:512]
    return torch.tensor([ids],dtype=torch.long,device=device)


class Reason1SDPA(torch.nn.Module):
    def __init__(self,network,config):
        super().__init__()
        text_config=getattr(config,'text_config',config)
        text_config._attn_implementation='sdpa';config.vision_config._attn_implementation='sdpa'
        self.visual=network.Qwen2_5_VisionTransformerPretrainedModel(config.vision_config)
        self.model=network.Qwen2_5_VLModel(text_config)
        self.lm_head=torch.nn.Linear(text_config.hidden_size,text_config.vocab_size,bias=False)

    def materialize_rotary(self,network):
        for part in self.modules():
            if isinstance(part,(network.Qwen2_5_VLRotaryEmbedding,network.Qwen2_5_VisionRotaryEmbedding)):
                part.init_weights(buffer_device=torch.device('cpu'))
                if hasattr(part,'original_inv_freq'):part.original_inv_freq=part.inv_freq


def load_reason1(upstream,checkpoint,device='cuda'):
    import json
    root=Path(checkpoint); network=native_network(upstream)
    config=Qwen2_5_VLConfig.from_pretrained(root,local_files_only=True)
    config.output_hidden_states=True;config.use_cache=False
    with torch.device('meta'): model=Reason1SDPA(network,config)
    state={}
    for path in sorted(root.glob('model-*-of-*.safetensors')):
        values=load_file(str(path))
        if state.keys() & values.keys(): raise ValueError('Duplicated Reason1 weights')
        state.update(values)
    model.load_state_dict(state,strict=True,assign=True)
    parameters=len(list(model.named_parameters()))
    # Rotary frequencies are non-persistent meta buffers, initialized by the
    # unchanged upstream helper after parameter assignment.
    model.materialize_rotary(network)
    if any(x.is_meta for x in list(model.parameters())+list(model.buffers())):
        raise ValueError('Unmaterialized Reason1 parameter/buffer')
    for k,x in model.state_dict().items():
        if x.is_floating_point() and not bool(torch.isfinite(x).all()):
            raise ValueError('Nonfinite official Reason1 parameter '+k)
    model=model.to(device=device,dtype=torch.bfloat16).eval().requires_grad_(False)
    processor=AutoProcessor.from_pretrained(root,local_files_only=True)
    receipt=dict(parameters=parameters,missing_parameters=[],unexpected=[],
                 backend='pinned NVIDIA Qwen network existing SDPA; text-only native FULL_CONCAT',
                 network_sha256=NETWORK_SHA)
    return model,processor,receipt
