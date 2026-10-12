"""Foreground launch with native Python, current receipts and no user time cap.

Only for this explicitly authorized unlimited Cosmos experiment. Disposable
preflight/restore use separate bounded_probe commands. No daemon or detach.
"""
import argparse,json,os,shutil
from pathlib import Path
from omegaconf import OmegaConf
from bounded_probe import execute
from fastwam.cosmos.binding import verify_preflight
from fastwam.cosmos.persistence import resource_identity,verify_restore
from fastwam.cosmos.preflight import validate_config

def main():
    p=argparse.ArgumentParser()
    for key in ('config','preflight','preflight-outer','restore','restore-outer','supervision','native-python'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    cfg=OmegaConf.load(a.config);validate_config(OmegaConf.to_container(cfg,resolve=True))
    gate=verify_preflight(a.config,a.preflight,a.preflight_outer)
    resource=resource_identity();r=json.loads(Path(a.preflight).read_text())
    if r['resource']!=resource:raise ValueError('Hardware differs from the actual probe')
    gate['restore']=verify_restore(a.config,a.restore,a.restore_outer,gate['preflight_sha256'],resource)
    native=Path(a.native_python)  # Preserve .venv invocation; never resolve its symlink.
    if not native.is_absolute() or not native.is_file() or native.parent.name!='bin' or native.parent.parent.name!='.venv':
        raise ValueError('Explicit original absolute .venv/bin/python invocation required')
    if Path(cfg.output_dir).exists():raise FileExistsError('Fresh formal output required')
    free=shutil.disk_usage(Path(cfg.output_dir).parent).free
    required=int(r['checkpoint_bytes']*6*1.1)
    if free<required:raise ValueError('Insufficient disk for six full checkpoints plus10% margin')
    command=[str(native),'-m','accelerate.commands.launch','--config_file',str(root/'scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml'),
        str(root/'scripts/cosmos/train.py'),'--config',str(Path(a.config).absolute()),'--preflight',str(Path(a.preflight).absolute()),
        '--preflight-outer',str(Path(a.preflight_outer).absolute()),'--restore',str(Path(a.restore).absolute()),
        '--restore-outer',str(Path(a.restore_outer).absolute()),'--execute']
    plan={'argv':command,'resource':resource,'gate':gate,'user_time_cap':None,'free_disk_bytes':free,
        'six_checkpoint_bytes_with_margin':required,'training_seconds_estimate':r['warm_seconds_per_step']*30000,
        'six_saves_seconds_estimate':r['max_save_seconds']*6,'evaluation_seconds_measured':None}
    print(json.dumps(plan,indent=2),flush=True)
    if not a.execute:return
    os.environ['PYTHONPATH']=str(root/'src')+':'+str(root)+(':'+os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
    result=execute(command,str(root),a.supervision,None,maximum_seconds=None,log_name='training.log',phase='cosmos-force28-formal')
    print(json.dumps(result),flush=True)
    raise SystemExit(0 if result['status']=='process_exited_zero' else 2)
if __name__=='__main__':main()
