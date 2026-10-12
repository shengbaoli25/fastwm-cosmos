"""CPU dispatch and identity contracts; not model/GPU training evidence."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(path, name):
    spec=importlib.util.spec_from_file_location(name, ROOT/path)
    value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


driver=module('scripts/cosmos/distributed_fresh.py', 'fresh_driver')
identity=module('src/fastwam/cosmos/job_identity.py', 'job_identity')


def env():
    return {'TRAIN_JOB_ID': 'job-test-1234', 'CUDA_VISIBLE_DEVICES': '0,1,2,3'}


def rows():
    return [f'GPU-{i}, NVIDIA H200, 143771' for i in range(4)]


def success():
    return {'status':'process_exited_zero','returncode':0,'subreaper_restored':True,
            'ownership':{'closure_verified':True,'signals':[],'errors':[],'live_after_cleanup':[]}}


def test_actual_job_identity():
    value=identity.distributed_identity(env(), 'node-1', rows())
    assert value['job']=='job-test-1234' and value['execution_kind']=='distributed_training'
    assert 'allocation' not in value


@pytest.mark.parametrize('change', [
    {'TRAIN_JOB_ID':''}, {'TRAIN_JOB_ID':'../job'}, {'PET_NNODES':'2'},
    {'PET_NODE_RANK':'1'}, {'COSMOS_ALLOCATION_ID':'old-notebook'},
    {'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'}])
def test_wrong_execution_identity_rejected(change):
    value=env(); value.update(change)
    with pytest.raises(ValueError): identity.distributed_identity(value, 'node-1', rows())


@pytest.mark.parametrize('inventory', [[], rows()[:3], rows()*2, [rows()[0]]*4,
                                     [r.replace('H200','H100') for r in rows()],
                                     [r.replace('143771','80000') for r in rows()]])
def test_wrong_allocation_rejected(inventory):
    with pytest.raises(ValueError): identity.distributed_identity(env(),'node-1',inventory)


def test_outer_rank_environment_does_not_leak():
    old={**env(),'RANK':'3','WORLD_SIZE':'8','MASTER_PORT':'30000','PATH':'/usr/bin'}
    new=driver.clean_environment(old, ROOT)
    assert all(k not in new for k in ('RANK','WORLD_SIZE','MASTER_PORT'))
    assert new['TRAIN_JOB_ID']==old['TRAIN_JOB_ID'] and new['PATH']==old['PATH']
    assert old['RANK']=='3'


def test_native_invocation_and_30000_formal_are_separate(tmp_path):
    native=Path('/shared/realman-fastwam/.venv/bin/python')
    commands=driver.phase_commands(native, ROOT, tmp_path)
    assert [x[0] for x in commands]==['preflight','restore','formal']
    assert [x[2] for x in commands]==[3600,3600,None]
    for _,command,_ in commands:
        assert command[0]==str(native) and command[-1]=='--execute'
        assert '--resume' not in command and not any('step_000002' in x for x in command)


@pytest.mark.parametrize('failure_index',[0,1,2])
def test_failed_phase_stops_dispatch(tmp_path,failure_index):
    calls=[]
    def execute(*args,**kwargs):
        calls.append(args)
        value=success()
        if len(calls)-1==failure_index:value['returncode']=1
        return value
    with pytest.raises(RuntimeError):
        driver.run_phases(driver.phase_commands(Path('/x/.venv/bin/python'),ROOT,tmp_path),
                          ROOT,tmp_path,execute,lambda *x:None)
    assert len(calls)==failure_index+1


@pytest.mark.parametrize('field,value',[('signals',['SIGTERM']),('errors',['lost child']),
                                      ('live_after_cleanup',[123]),('closure_verified',False)])
def test_forced_or_unclosed_success_blocks_next_phase(tmp_path,field,value):
    calls=[]
    def execute(*args,**kwargs):
        calls.append(args); result=success(); result['ownership'][field]=value; return result
    with pytest.raises(RuntimeError):
        driver.run_phases(driver.phase_commands(Path('/x/.venv/bin/python'),ROOT,tmp_path),
                          ROOT,tmp_path,execute,lambda *x:None)
    assert len(calls)==1


def test_three_phases_are_foreground_and_formal_unlimited(tmp_path):
    calls=[]
    def execute(*args,**kwargs):calls.append((args,kwargs)); return success()
    driver.run_phases(driver.phase_commands(Path('/x/.venv/bin/python'),ROOT,tmp_path),
                      ROOT,tmp_path,execute,lambda *x:None)
    assert len(calls)==3 and calls[-1][0][3] is None
    assert calls[-1][1]['maximum_seconds'] is None
    assert calls[-1][0][2]==tmp_path/'formal-controller-watch'


def stop_fixture(tmp_path):
    prior=tmp_path/'prior';prior.mkdir()
    config=prior/'config.yaml';config.write_text('original config')
    screenshot=tmp_path/'stopped.png';screenshot.write_bytes(b'TEST-only screen')
    evidence=tmp_path/'stop.json'
    data={'resource_id':driver.PRIOR_ID,'status':'STOPPED','source':'platform_ui',
          'prior_output':str(prior),'observed_at_utc':'2026-10-12T15:09:00Z',
          'evidence_file':screenshot.name,'evidence_sha256':driver.sha(screenshot),
          'config_sha256':driver.sha(config)}
    evidence.write_text(json.dumps(data))
    return prior,config,evidence,screenshot


def test_stop_evidence_and_empty_formal_state(tmp_path):
    prior,config,evidence,_=stop_fixture(tmp_path)
    assert driver.check_stopped(evidence,config,prior)['status']=='STOPPED'
    state=prior/'checkpoints/state/step_005000';state.mkdir(parents=True)
    (state/'trainer_state.json').write_text('{}')
    with pytest.raises(ValueError,match='formal checkpoint'):driver.check_stopped(evidence,config,prior)


@pytest.mark.parametrize('corrupt',['screenshot','config','identity','status'])
def test_invalid_stop_evidence_rejected(tmp_path,corrupt):
    prior,config,evidence,screenshot=stop_fixture(tmp_path)
    if corrupt=='screenshot':screenshot.write_bytes(b'changed')
    elif corrupt=='config':config.write_text('changed')
    else:
        data=json.loads(evidence.read_text())
        data['resource_id' if corrupt=='identity' else 'status']='wrong'
        evidence.write_text(json.dumps(data))
    with pytest.raises(ValueError):driver.check_stopped(evidence,config,prior)


def test_default_is_inert_plan_without_gpu_or_files(tmp_path):
    run=tmp_path/'run'
    r=subprocess.run([sys.executable,str(ROOT/'scripts/cosmos/distributed_fresh.py'),
                      '--config',str(tmp_path/'missing.yaml'),'--stop-evidence',str(tmp_path/'missing.json'),
                      '--run-root',str(run),'--native-python','/shared/.venv/bin/python'],
                     text=True,capture_output=True)
    assert r.returncode==0,r.stderr
    assert json.loads(r.stdout)['mode']=='plan_only' and not run.exists()
