"""Training-bound runtime export must reject mismatched statistics or architecture."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('noaction_export', ROOT/'scripts/cosmos/export_noaction_deployment.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def fixture(tmp_path):
    normalization=tmp_path/'normalization.json'
    normalization.write_text(json.dumps(dict(statistics={k:dict(mean=[0.]*n,std=[1.]*n) for k,n in [('state',8),('action',8),('force',6)]},
        action_space='joint8',action_units=['rad']*7+['openness_0_1'],state_columns=[0,1,2,3,4,5,6,19])))
    (tmp_path/'hardware.json').write_text('{"action_space":"joint8"}')
    text=tmp_path/'text';text.mkdir()
    (text/'manifest.json').write_text(json.dumps({'texts':[{'prompt':'new-cup-task'}]}))
    sha=export.file_sha256(normalization)
    model=dict(_target_='fastwam.cosmos.assets.create_noaction_joint_cosmos',
        expected_data_identity='a'*64,expected_text_identity='b'*64,expected_normalization_sha256=sha,
        upstream=str(tmp_path/'upstream'),checkpoint=str(tmp_path/'base.pt'),text_cache=str(text),
        action_hidden=768,force_hidden=768,num_inference_steps=16,action_space='joint8')
    manifest=dict(dataset_identity='a'*64,text_identity='b'*64,normalization_sha256=sha,
        extra_config=dict(action_hidden=768,force_hidden=768,num_inference_steps=16,action_space='joint8'),
        num_inference_steps=16,action_space='joint8')
    paths=dict(checkpoint=tmp_path/'trained.pt',normalization=normalization,
        tokenizer=tmp_path/'tokenizer.pth',hardware_config=tmp_path/'hardware.json')
    return {'model':model},manifest,paths


def test_export_uses_training_steps_and_model_identity(tmp_path):
    config,manifest,paths=fixture(tmp_path)
    runtime=export.runtime_config(config,manifest,**paths)
    assert runtime['joint_steps']==16 and runtime['inference_backend']=='joint_static_cache'
    assert runtime['execution_mode']=='synchronous' and runtime['task_prompt']=='new-cup-task'
    assert runtime['model_kwargs']['expected_data_identity']==manifest['dataset_identity']
    assert 'generated_video_steps' not in runtime['model_kwargs']
    assert 'sampling_profile' not in runtime
    assert runtime['device'] == 'cuda'


@pytest.mark.parametrize('optimization', ['cuda_graph', 'compile_cuda_graph'])
def test_export_selects_ui_consistent_4416_profile_and_visible_cuda_device(tmp_path, optimization):
    from realman_deploy.sampling import settings
    config,manifest,paths=fixture(tmp_path)
    runtime=export.runtime_config(config,manifest,**paths,sampling_profile='4416',
        execution_mode='async_relaxed',device='cuda:1',optimization=optimization)
    ui=settings(runtime)
    assert runtime['sampling_profile']=='4416'
    assert runtime['inference_backend']=='joint_multirate'
    assert runtime['device']=='cuda:1'
    assert runtime['inference_optimization']=={'mode':optimization}
    assert ui['sampling_profile']=='4416' and ui['execution_mode']=='async_relaxed'


@pytest.mark.parametrize('profile,mode,device',[
    ('161616','async_relaxed','cuda'), ('4416','timestamped','cuda'),
    ('unknown','synchronous','cuda'), ('4416','synchronous','cpu'),
])
def test_export_rejects_invalid_profile_mode_or_device(tmp_path,profile,mode,device):
    config,manifest,paths=fixture(tmp_path)
    with pytest.raises(ValueError):
        export.runtime_config(config,manifest,**paths,sampling_profile=profile,
            execution_mode=mode,device=device)


@pytest.mark.parametrize('mismatch',['normalization','architecture','dataset'])
def test_export_rejects_mismatched_training_artifacts(tmp_path,mismatch):
    config,manifest,paths=fixture(tmp_path)
    if mismatch=='normalization':paths['normalization'].write_text('{"statistics":{"force":"changed"}}')
    elif mismatch=='architecture':config['model']['force_hidden']=1280
    else:config['model']['expected_data_identity']='c'*64
    with pytest.raises(ValueError):export.runtime_config(config,manifest,**paths)
