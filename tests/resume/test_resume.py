"""CPU contracts only; these do not certify a fresh production GPU restore."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/resume'))
from checks import (complete_state, force_deadline, platform_identity, launch_environment,
                    stopped_evidence, verify_code, prior_resource, sha, budget_estimate)


def archive(path, tensors=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('archive/data.pkl', b'fixture')
        if tensors:
            z.writestr('archive/data/0', b'fixture')


def checkpoint(out, step=5000):
    s = out / f'checkpoints/state/step_{step:06d}'
    s.mkdir(parents=True)
    (s/'trainer_state.json').write_text(json.dumps(dict(global_step=step,epoch=9,batch_in_epoch=104)))
    (s/'latest').write_text('pytorch_model')
    archive(s/'scheduler.bin',False)
    archive(s/'pytorch_model/mp_rank_00_model_states.pt')
    archive(out/f'checkpoints/weights/step_{step:06d}.pt')
    for rank in range(4):
        archive(s/f'random_states_{rank}.pkl')
        archive(s/f'pytorch_model/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt')
    return s


def save_line(out, step=5000, tag=''):
    return f'{tag} step={step} trainer.py:791\n weights={out}/checkpoints/weights/step_{step:06d}.pt\n state={out}/checkpoints/state/step_{step:06d}\n'


@pytest.mark.parametrize('tag', ['', '[ckpt]', '\x1b[32m[ckpt]\x1b[0m'])
def test_completed_log_with_rich_wrapping(tmp_path,tag):
    out=tmp_path/'out';s=checkpoint(out);log=tmp_path/'log'
    log.write_text(save_line(out,tag=tag).replace('checkpoints','check\n points'))
    assert complete_state(out,log)==(s,5000)


@pytest.mark.parametrize('damage',['optimizer','rng','scheduler','weights','trainer','tag','empty','truncated','extra_rank','unsaved'])
def test_partial_or_wrong_checkpoint_rejected(tmp_path,damage):
    out=tmp_path/'out';s=checkpoint(out);log=tmp_path/'log';log.write_text(save_line(out))
    targets={'optimizer':s/'pytorch_model/bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt',
             'rng':s/'random_states_1.pkl','scheduler':s/'scheduler.bin',
             'weights':out/'checkpoints/weights/step_005000.pt'}
    if damage in targets: targets[damage].unlink()
    elif damage=='trainer': (s/'trainer_state.json').write_text('{"global_step": 5000}')
    elif damage=='tag': (s/'latest').write_text('../../other')
    elif damage=='empty': (s/'scheduler.bin').write_bytes(b'')
    elif damage=='truncated': (s/'pytorch_model/mp_rank_00_model_states.pt').write_bytes(b'PKfixture')
    elif damage=='extra_rank': archive(s/'pytorch_model/bf16_zero_pp_rank_4_mp_rank_00_optim_states.pt')
    elif damage=='unsaved': log.write_text('epoch=9 step=5000/30000 loss=0.1')
    with pytest.raises(ValueError): complete_state(out,log)


def test_fallback_to_last_completed_save(tmp_path):
    out=tmp_path/'out';s=checkpoint(out);checkpoint(out,10000)
    log=tmp_path/'log';log.write_text(save_line(out))
    assert complete_state(out,log)==(s,5000)


def test_final_needs_second_save_and_done(tmp_path):
    out=tmp_path/'out';s=checkpoint(out,30000);log=tmp_path/'log'
    log.write_text(save_line(out,30000))
    with pytest.raises(ValueError): complete_state(out,log,final=True)
    log.write_text(save_line(out,30000,'[done] max_steps reached'))
    assert complete_state(out,log,final=True)==(s,30000)
    with pytest.raises(ValueError): complete_state(out,log)


@pytest.mark.parametrize('steps,save',[(1000,1000),(20000,5000)])
def test_configured_joint8_final_checkpoint_requires_completed_full_save(tmp_path,steps,save):
    out=tmp_path/'out';state=checkpoint(out,steps);log=tmp_path/'log'
    log.write_text(save_line(out,steps))
    with pytest.raises(ValueError):complete_state(out,log,final=True,max_steps=steps,save_every=save)
    log.write_text(save_line(out,steps,'[done] max_steps reached'))
    assert complete_state(out,log,final=True,max_steps=steps,save_every=save)==(state,steps)


def test_platform_and_environment():
    with pytest.raises(ValueError): platform_identity({})
    with pytest.raises(ValueError): platform_identity({'TRAIN_JOB_ID':'job-1234','PET_NNODES':'2'})
    assert platform_identity({'TRAIN_JOB_ID':'job-1234'})=='job-1234'
    env=launch_environment({'RANK':'0','WORLD_SIZE':'1','MASTER_ADDR':'host','TRAIN_JOB_ID':'job-1234'},ROOT)
    assert 'RANK' not in env and 'MASTER_ADDR' not in env and env['CUDA_VISIBLE_DEVICES']=='0,1,2,3'
    assert env['TRAIN_JOB_ID']=='job-1234'


def test_stopped_and_repeated_lineage(tmp_path):
    evidence=tmp_path/'screenshot';evidence.write_bytes(b'test-only')
    p=tmp_path/'stop.json';prior=tmp_path/'prior';prior.mkdir()
    d=dict(prior_output=str(prior),source='platform_ui',status='STOPPED',resource_id='job-new',
           observed_at_utc='2026-10-13T00:00:00Z',evidence_file=evidence.name,evidence_sha256=sha(evidence))
    p.write_text(json.dumps(d));assert stopped_evidence(p,prior)==d
    assert prior_resource(prior,'initial')=='initial'
    (prior/'resume_origin.json').write_text('{"job":"job-new"}')
    assert prior_resource(prior,'initial')=='job-new'
    d['status']='RUNNING';p.write_text(json.dumps(d))
    with pytest.raises(ValueError): stopped_evidence(p,prior)
    d['status']='STOPPED';p.write_text(json.dumps(d));evidence.write_bytes(b'changed')
    with pytest.raises(ValueError): stopped_evidence(p,prior)


def test_absolute_budget_cannot_reset(tmp_path):
    p=tmp_path/'budget';d=dict(launcher_started_unix=1791795479.8201158,
       absolute_deadline_unix=1791975479.8201158,duration_seconds=180000,budget_anchor='formal_launcher_start',
       execution_identity={'kind':'interactive_modeling','allocation_id':'ab5d6993-795b-4bf7-9084-e59f089e7788'})
    p.write_text(json.dumps(d));assert force_deadline(p,1791800000)==d['absolute_deadline_unix']
    with pytest.raises(ValueError): force_deadline(p,d['absolute_deadline_unix'])
    d['absolute_deadline_unix']+=100;p.write_text(json.dumps(d))
    with pytest.raises(ValueError): force_deadline(p,1791800000)


def test_frozen_code_and_budget(tmp_path):
    assert verify_code(ROOT)['commit']
    log=tmp_path/'log';log.write_text('speed=0.28 step/s')
    assert budget_estimate(log,5000,1791975479.8201158,1791814000)['available_seconds']>0
    with pytest.raises(ValueError): budget_estimate(log,5000,100,99)


@pytest.mark.parametrize('epoch,offset',[(0,0),(9,104),(9,543),(9,544)])
def test_four_shards_resume_exact_padding_and_next_epoch(epoch,offset):
    import torch
    from torch.utils.data import DataLoader
    from accelerate.data_loader import prepare_data_loader
    from continuation import FirstEpochContinuation
    spec=importlib.util.spec_from_file_location('frozen_sampler',ROOT/'src/fastwam/utils/samplers.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    ds=list(range(34808))
    def make(rank):
        s=m.ResumableEpochSampler(ds,42,16,4)
        loader=prepare_data_loader(DataLoader(ds,batch_size=16,sampler=s),num_processes=4,
            process_index=rank,put_on_device=False,rng_types=[])
        return loader,s
    for rank in range(4):
        full,s=make(rank);full.set_epoch(epoch);expected=[x.tolist() for x in full]
        next_expected=[x.tolist() for x in full]
        loader,sampler=make(rank);sampler.set_epoch_offset(epoch);sampler.set_resume_batch_offset(offset)
        resumed=FirstEpochContinuation(loader,sampler,epoch,offset)
        torch.manual_seed(1234);before=torch.get_rng_state().clone()
        iterator=iter(resumed)
        actual=[]
        try:
            actual.append(next(iterator).tolist())
            assert torch.equal(torch.get_rng_state(),before)
            actual += [x.tolist() for x in iterator]
        except StopIteration:
            assert torch.equal(torch.get_rng_state(),before)
        assert actual == expected[offset:]
        assert [x.tolist() for x in resumed] == next_expected


@pytest.mark.parametrize('damage',[None,'cpu_rng','python_rng','memory_rng','offset','acc_step'])
def test_native_rng_and_sampler_restore(tmp_path,damage,monkeypatch):
    import random
    import numpy as np
    import torch
    from continuation import verify_rng_and_sampler, rng_snapshot, restore_rng
    initial=rng_snapshot()
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    try:
        random.seed(7);np.random.seed(8);torch.manual_seed(9)
        model=torch.nn.Module();model.register_buffer('rng_state',torch.Generator().manual_seed(44).get_state())
        model.register_buffer('manifest_bytes',torch.tensor([1,2],dtype=torch.uint8))
        acc=SimpleNamespace(process_index=0,step=5000,unwrap_model=lambda _:model)
        trainer=SimpleNamespace(accelerator=acc,model=model,train_loader=list(range(544)),
          train_sampler=SimpleNamespace(epoch_offset=9,resume_batch_offset=104,seed=42,batch_size=16,num_processes=4))
        state=tmp_path; (state/'pytorch_model').mkdir()
        torch.save({'module':model.state_dict()},state/'pytorch_model/mp_rank_00_model_states.pt')
        torch.save(dict(random_state=random.getstate(),numpy_random_seed=np.random.get_state(),
                   torch_manual_seed=torch.get_rng_state(),step=5000),state/'random_states_0.pkl')
        if damage=='cpu_rng':torch.rand(1)
        if damage=='python_rng':random.random()
        if damage=='memory_rng':model.rng_state.copy_(torch.Generator().manual_seed(3).get_state())
        if damage=='offset':trainer.train_sampler.resume_batch_offset=0
        if damage=='acc_step':acc.step=0
        if damage:
            with pytest.raises(ValueError):verify_rng_and_sampler(trainer,state,dict(epoch=9,batch_in_epoch=104))
        else:
            assert verify_rng_and_sampler(trainer,state,dict(epoch=9,batch_in_epoch=104))['python_numpy_torch_cuda']
    finally:restore_rng(initial)


def test_finished_run_cannot_fall_back_to_25000(tmp_path):
    out=tmp_path/'out';checkpoint(out,25000);checkpoint(out,30000)
    log=tmp_path/'log';log.write_text(save_line(out,25000)+save_line(out,30000,'max_steps reached'))
    with pytest.raises(ValueError,match='already logged'):
        complete_state(out,log)
