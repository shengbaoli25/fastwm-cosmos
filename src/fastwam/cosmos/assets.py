"""Pinned official Cosmos robot/action-cond assets. No Wan policy/AE checkpoint is loaded."""
from pathlib import Path
import json
import torch
from safetensors.torch import load_file
from fastwam.cache.latent import file_sha256, cache_identity_sha256
from .native import native_imports, target_net_config, preserve_fp32_strategy
from .controls import CONTROL_PROTOCOL

UPSTREAM_COMMIT='a2c298b0a3df3778b973fe65e9e58877b292d8a7'
MODEL_ID='nvidia/Cosmos-Predict2.5-2B'
REVISION='15a82a2ec231bc318692aa0456a36537c806e7d4'
WEIGHT_FILE='robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt'
WEIGHT_SHA='b2e993cea3e242bc72288612488fe91770a1c073c98c3fec6d610411ffc30646'
TOKENIZER_FILE='tokenizer.pth'
TOKENIZER_SHA='38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981'


def checked(path, digest):
    p=Path(path).resolve()
    if not p.is_file() or file_sha256(p)!=digest:
        raise ValueError(f'Official asset SHA256 differs: {p}')
    return p


def load_video(upstream, checkpoint, *, device='cuda', dtype=torch.bfloat16):
    _,module,_,Net,_=native_imports(Path(upstream).resolve())
    p=checked(checkpoint,WEIGHT_SHA)
    config=target_net_config(); config['sac_config']=module.SACConfig(mode='none')
    with torch.device('meta'): net=Net(**config)
    payload=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
    if not isinstance(payload,dict) or not payload or any(not k.startswith('net.') for k in payload):
        raise ValueError('Expected official flat net.* Cosmos checkpoint')
    net.load_state_dict({k.removeprefix('net.'):v for k,v in payload.items()},strict=True,assign=True)
    if any(x.is_meta for x in (*net.parameters(),*net.buffers())):
        raise ValueError('Incomplete Cosmos checkpoint materialization')
    for k,x in net.state_dict().items():
        if isinstance(x,torch.Tensor) and x.is_floating_point() and not bool(torch.isfinite(x).all()):
            raise ValueError('Nonfinite official tensor '+k)
    net.to(device=device,dtype=dtype); preserve_fp32_strategy(net)
    return net


def load_tokenizer(upstream,checkpoint,*,device='cuda'):
    _,_,vae,_,_=native_imports(Path(upstream).resolve())
    p=checked(checkpoint,TOKENIZER_SHA)
    return vae.WanVAE(vae_pth=str(p),device=torch.device(device),dtype=torch.bfloat16,is_amp=False,temporal_window=16)


def read_text_bank(path):
    root=Path(path).resolve(); manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('schema')!='cosmos-reason1-full-concat-v1' or manifest.get('status')!='complete':
        raise ValueError('Complete Cosmos Reason1 text manifest required')
    if manifest.get('encoder')!='nvidia/Cosmos-Reason1-7B' or manifest.get('feature_dim')!=100352:
        raise ValueError('Wan T5 / random text cannot substitute for Cosmos Reason1')
    pinned=json.loads(Path(__file__).with_name('reason1_weights.json').read_text())
    if manifest.get('encoder_revision')!=pinned['revision'] or any(manifest.get('encoder_files',{}).get(k)!=v for k,v in pinned['weights'].items()):
        raise ValueError('Text provenance does not bind all official Reason1 weight shards')
    items=manifest['texts']; bank=[]
    if not items or len({x['prompt'] for x in items})!=len(items): raise ValueError('Unique nonempty task prompts required')
    for item in items:
        p=(root/item['path']).resolve()
        if not p.is_relative_to(root): raise ValueError('Text path escapes root')
        checked(p,item['sha256']); x=load_file(str(p))['context']
        if x.shape!=(512,100352) or x.dtype!=torch.bfloat16 or not bool(torch.isfinite(x).all()):
            raise ValueError('Cosmos text must be finite BF16 [512,100352]')
        bank.append(x)
    return torch.stack(bank),manifest,cache_identity_sha256(manifest)


def create_cosmos_forcewm(*,upstream,checkpoint,text_cache,expected_text_identity,
                         action_seed=42,force_seed=43,memory_seed=44,control_seed=45,
                         generated_video_steps=16,gradient_checkpointing=True,device='cuda',
                         _model_type=None,_constructor_options=None,**kwargs):
    from .model import CosmosForceWM
    bank,texts,digest=read_text_bank(text_cache)
    if digest!=expected_text_identity: raise ValueError('Pinned text cache identity differs')
    net=load_video(upstream,checkpoint,device='cpu')
    manifest=dict(schema='cosmos-clean-action-force-chunk28-v3',upstream=UPSTREAM_COMMIT,model=MODEL_ID,
        revision=REVISION,video_sha256=WEIGHT_SHA,tokenizer_sha256=TOKENIZER_SHA,
        depth=28,action_seed=action_seed,force_seed=force_seed,memory_seed=memory_seed,control_seed=control_seed,
        generated_video_steps=generated_video_steps,text_identity=digest,
        video='official robot/action-cond bidirectional/global-time; isolated sigma0 observed KV',
        control_protocol=CONTROL_PROTOCOL,
        memory='detached generated video from explicit clean action; shared fixed memory across AE diffusion',
        force='direct learned-query chunk masked MSE; H[-31..0], U[0..31] -> W[1..32]; no previous future wrench or Action clock',
        action_semantics='conditional reconstruction under GT-action-conditioned WM; not an observation-only policy',
        initialization='official Cosmos only; AE/Force/proprio/dual memory/control projection random',
        loss='FastWAM shifted-uniform FM v5/a1; Force MSE; lambda 1/1/1',
        recipe=dict(ranks=4,batch16=16,accum=1,global_batch=64,max_steps=30000),
        extra_config=kwargs)
    model_type=CosmosForceWM if _model_type is None else _model_type
    model=model_type(net,manifest=manifest,text_bank=bank,action_seed=action_seed,
        force_seed=force_seed,memory_seed=memory_seed,control_seed=control_seed,generated_video_steps=generated_video_steps,
        gradient_checkpointing=gradient_checkpointing,**(_constructor_options or {}),**kwargs)
    return model.to(device=device,dtype=torch.bfloat16)


def create_cosmos_paired_regions(*, region_pairs=32, region_width=512, region_heads=8,
                                region_layer=13, region_seed=46,
                                region_reconstruction_weight=.05,
                                region_diversity_weight=.001, region_writeback=True, **kwargs):
    from .paired import PairedCosmosForceWM
    options = dict(region_pairs=region_pairs, region_width=region_width, region_heads=region_heads,
        region_layer=region_layer, region_seed=region_seed,
        region_reconstruction_weight=region_reconstruction_weight,
        region_diversity_weight=region_diversity_weight, region_writeback=region_writeback)
    return create_cosmos_forcewm(_model_type=PairedCosmosForceWM,
                                _constructor_options=options, **kwargs)
