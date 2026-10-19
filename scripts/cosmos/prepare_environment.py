"""CPU environment/assets audit only; never creates a trainer or starts training."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','tokenizer','reason1','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    from fastwam.cache.latent import file_sha256
    from fastwam.cosmos.assets import NONACTION_WEIGHT_SHA,TOKENIZER_SHA,load_nonaction_video
    from fastwam.cosmos.native import verify_upstream
    from fastwam.cosmos.noaction import NoActionJointModel
    import torch
    torch.set_num_threads(4)
    versions={}
    for package,module in [('torch','torch'),('torchvision','torchvision'),('accelerate','accelerate'),
            ('deepspeed','deepspeed'),('transformers','transformers'),('pyarrow','pyarrow'),
            ('av','av'),('safetensors','safetensors'),('hydra-core','hydra'),('omegaconf','omegaconf'),
            ('einops','einops'),('GitPython','git'),('loguru','loguru'),('attrs','attrs'),
            ('nvidia-ml-py','pynvml')]:
        importlib.import_module(module)
        versions[package]=importlib.metadata.version(package)
    print('DEPENDENCIES_OK',json.dumps(versions),flush=True)
    from fastwam.cosmos import realman_joint,reason1_native
    from fastwam.trainer import Wan22Trainer
    files={}
    for path,sha in ((args.checkpoint,NONACTION_WEIGHT_SHA),(args.tokenizer,TOKENIZER_SHA)):
        actual=file_sha256(path)
        if actual != sha:raise ValueError('Asset SHA mismatch: '+str(path))
        files[str(path.resolve())]=dict(sha256=actual,bytes=path.stat().st_size)
        print('ASSET_SHA_OK',str(path),flush=True)
    pinned=json.loads((ROOT/'src/fastwam/cosmos/reason1_weights.json').read_text())
    for name,sha in pinned['weights'].items():
        path=args.reason1/name;actual=file_sha256(path)
        if actual != sha:raise ValueError('Reason1 shard SHA mismatch: '+name)
        files[str(path.resolve())]=dict(sha256=actual,bytes=path.stat().st_size)
        print('ASSET_SHA_OK',str(path),flush=True)
    from transformers import AutoProcessor,Qwen2_5_VLConfig
    processor=AutoProcessor.from_pretrained(args.reason1,local_files_only=True)
    config=Qwen2_5_VLConfig.from_pretrained(args.reason1,local_files_only=True)
    ids=reason1_native.prompt_ids(processor,'Pick up the cup.',device='cpu')
    if ids.shape != (1,512):raise ValueError('Wrong Reason1 tokenizer output')
    print('REASON1_PROCESSOR_OK',flush=True)
    upstream=ROOT/'third_party/cosmos-predict2.5'
    hashes=verify_upstream(upstream)
    net=load_nonaction_video(upstream,args.checkpoint,device='cpu')
    print('BASE_STRICT_LOAD_OK',flush=True)
    model=NoActionJointModel(net,manifest={'scope':'preparation'},action_space='joint8')
    receipt=dict(status='cpu_environment_assets_ready',python=sys.version,executable=sys.executable,
        platform=platform.platform(),versions=versions,assets=files,upstream=hashes,
        model_manifest=model.manifest,parameters={key:sum(p.numel() for p in value.parameters())
                                               for key,value in model.dit.items()},
        formal_training_started=False,gpu_preflight_completed=False,data_frozen=False,
        reason1_tokenizer_local_only=True,base_strict_load=True)
    receipt['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    receipt['reason1_config']=dict(hidden_size=config.hidden_size,num_hidden_layers=config.num_hidden_layers)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],output=str(args.output),parameters=receipt['parameters'])),flush=True)


if __name__=='__main__':main()
