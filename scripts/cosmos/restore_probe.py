"""Fresh four-rank full-state reload; checks tensors, moments and RNG."""
import argparse,json,os
from pathlib import Path
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import OmegaConf
from fastwam.trainer import Wan22Trainer
from fastwam.utils import misc
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.cosmos.preflight import validate_config,build_probe_datasets
from fastwam.cosmos.binding import binding,verify_preflight
from fastwam.cosmos.persistence import resource_identity,training_state_fingerprints

def main():
    p=argparse.ArgumentParser()
    for k in ('config','preflight','preflight-outer','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    cfg=OmegaConf.load(a.config);validate_config(OmegaConf.to_container(cfg,resolve=True))
    gate=verify_preflight(a.config,a.preflight,a.preflight_outer)
    if not a.execute:print(json.dumps({'mode':'plan_only','gate':gate}));return
    source=json.loads(Path(a.preflight).read_text());resource=resource_identity()
    if source['resource']!=resource:raise ValueError('Restore must use the same allocated GPUs')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    acc=Accelerator(gradient_accumulation_steps=1,mixed_precision='bf16',step_scheduler_with_optimizer=False)
    if acc.num_processes!=4:raise ValueError('Four actual ranks required')
    output=Path(a.output)
    if acc.is_main_process:output.mkdir(parents=True,exist_ok=False)
    acc.wait_for_everyone();rank=acc.process_index;cfg.output_dir=str(output/f'rank_{rank:03d}')
    misc.register_work_dir(cfg.output_dir);set_global_seed(42)
    model=instantiate(cfg.model,device=f'cuda:{torch.cuda.current_device()}')
    ds,val=build_probe_datasets(cfg.data);trainer=Wan22Trainer(model,ds,val,cfg=cfg)
    prior=next(r for r in source['ranks'] if r['rank']==rank)
    if not prior.get('checkpoint_saved'):raise ValueError('Actual checkpoint missing')
    trainer.load_training_state(prior['state_path'])
    actual=training_state_fingerprints(trainer,model)
    report={'rank':rank,'accepted':actual==prior['state_fingerprints'],'actual':actual,'expected':prior['state_fingerprints']}
    (output/f'rank_{rank:03d}.json').write_text(json.dumps(report,indent=2))
    ranks=[None]*4;torch.distributed.all_gather_object(ranks,report)
    ok=all(r['accepted'] for r in ranks)
    if acc.is_main_process:
        result={'status':'fresh_four_rank_restore_verified' if ok else 'failed','binding':binding(a.config),
            'preflight_sha256':gate['preflight_sha256'],'resource':resource,'ranks':ranks}
        (output/'restore.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    acc.wait_for_everyone();torch.distributed.destroy_process_group()
    raise SystemExit(0 if ok else 2)
if __name__=='__main__':main()
