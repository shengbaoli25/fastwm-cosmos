from copy import deepcopy
from pathlib import Path
import json
import pytest
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from fastwam.cosmos.preflight import validate_config
from fastwam.cosmos.binding import verify_preflight,binding

ROOT=Path(__file__).resolve().parents[2]

def config():
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base=None):
        cfg=compose(config_name='train',overrides=['task=cosmos_forcewm_4gpu_30k','model.expected_text_identity='+'a'*64,'data.train.expected_identity_sha256='+'b'*64])
    return OmegaConf.to_container(cfg,resolve=True,throw_on_missing=True)

def test_exact_recipe():
    cfg=config();validate_config(cfg)
    assert cfg['batch_size']*4*cfg['gradient_accumulation_steps']==64
    assert cfg['max_steps']==30000 and cfg['save_every']==5000

@pytest.mark.parametrize('field,value',[('batch_size',8),('max_steps',5000),('resume','some.pt'),('learning_rate',1e-5),('gradient_accumulation_steps',2)])
def test_recipe_changes_fail(field,value):
    cfg=config();cfg[field]=value
    with pytest.raises(ValueError):validate_config(cfg)

def test_gpu_gate_binds_source_recipe_and_unforced_exit(tmp_path):
    p=tmp_path/'config.yaml';OmegaConf.save(config(),p);b=binding(p)
    r={'status':'two_adam_verified','binding':b,'ranks':[{'rank':i,'accepted':True,'binding':b} for i in range(4)]}
    o={'status':'process_exited_zero','returncode':0,'subreaper_restored':True,'ownership':{'closure_verified':True,'signals':[],'errors':[],'live_after_cleanup':[]}}
    rp=tmp_path/'preflight.json';op=tmp_path/'outer.json';rp.write_text(json.dumps(r));op.write_text(json.dumps(o))
    assert verify_preflight(p,rp,op)['binding']==b
    for mutate in ['signals','code','ranks']:
        rr=deepcopy(r);oo=deepcopy(o)
        if mutate=='signals':oo['ownership']['signals']=[{'signal':15}]
        elif mutate=='code':rr['binding']={'code':'changed'}
        else:rr['ranks'].pop()
        rp.write_text(json.dumps(rr));op.write_text(json.dumps(oo))
        with pytest.raises(ValueError):verify_preflight(p,rp,op)
