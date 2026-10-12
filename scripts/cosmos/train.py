"""Foreground formal entry. Exact four-rank recipe and actual GPU preflight gate."""
import argparse,json,logging,os
from pathlib import Path
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import OmegaConf
from fastwam.trainer import Wan22Trainer
from fastwam.utils import misc
from fastwam.utils.logging_config import setup_logging
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.cosmos.preflight import validate_config,build_probe_datasets
from fastwam.cosmos.binding import verify_preflight


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--preflight',required=True);p.add_argument('--preflight-outer',required=True)
    p.add_argument('--restore',required=True);p.add_argument('--restore-outer',required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    cfg=OmegaConf.load(a.config);validate_config(OmegaConf.to_container(cfg,resolve=True,throw_on_missing=True))
    gate=verify_preflight(a.config,a.preflight,a.preflight_outer)
    if not a.execute:print(json.dumps({'mode':'plan_only','gate':gate}));return
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required')
    from fastwam.cosmos.persistence import resource_identity,verify_restore
    resource=resource_identity()
    preflight=json.loads(Path(a.preflight).read_text())
    if preflight['resource']!=resource:raise ValueError('Preflight belongs to another resource')
    gate['restore']=verify_restore(a.config,a.restore,a.restore_outer,gate['preflight_sha256'],resource)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    acc=Accelerator(gradient_accumulation_steps=1,mixed_precision='bf16',step_scheduler_with_optimizer=False)
    if acc.num_processes!=4:raise ValueError('Use four ranks even on an eight-GPU allocation')
    out=Path(cfg.output_dir)
    if acc.is_main_process:out.mkdir(parents=True,exist_ok=False)
    acc.wait_for_everyone();misc.register_work_dir(str(out));set_global_seed(42)
    setup_logging(log_level=logging.INFO,is_main_process=acc.is_main_process)
    origins=out/'origins';origins.mkdir(exist_ok=True)
    origin=origins/f'rank_{acc.process_index:03d}.json'
    receipt={'status':'starting','rank':acc.process_index,'world_size':4,'gate':gate,
        'resource':resource,'pid':os.getpid(),'device':torch.cuda.current_device(),'gpu':torch.cuda.get_device_name()}
    origin.write_text(json.dumps(receipt,indent=2))
    model=instantiate(cfg.model,device=f'cuda:{torch.cuda.current_device()}')
    ds,val=build_probe_datasets(cfg.data);trainer=Wan22Trainer(model,ds,val,cfg=cfg)
    engine=trainer.model
    if (engine.train_micro_batch_size_per_gpu(),engine.gradient_accumulation_steps(),engine.train_batch_size())!=(16,1,64):raise ValueError('Actual engine batch differs')
    if acc.is_main_process:OmegaConf.save(cfg,str(out/'config.yaml'))
    receipt['status']='running';origin.write_text(json.dumps(receipt,indent=2))
    trainer.train()
    if trainer.global_step!=30000:raise RuntimeError('Formal run did not complete30000')
    receipt.update(status='complete',global_step=trainer.global_step)
    origin.write_text(json.dumps(receipt,indent=2));acc.wait_for_everyone()
    torch.distributed.destroy_process_group()
if __name__=='__main__':main()
