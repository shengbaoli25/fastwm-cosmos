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
    normalization=tmp_path/'normalization.json';normalization.write_text('{"statistics": {}}')
    text=tmp_path/'text';text.mkdir()
    (text/'manifest.json').write_text(json.dumps({'texts':[{'prompt':'new-cup-task'}]}))
    sha=export.file_sha256(normalization)
    model=dict(_target_='fastwam.cosmos.assets.create_noaction_joint_cosmos',
        expected_data_identity='a'*64,expected_text_identity='b'*64,expected_normalization_sha256=sha,
        upstream=str(tmp_path/'upstream'),checkpoint=str(tmp_path/'base.pt'),text_cache=str(text),
        action_hidden=768,force_hidden=768,num_inference_steps=16)
    manifest=dict(dataset_identity='a'*64,text_identity='b'*64,normalization_sha256=sha,
        extra_config=dict(action_hidden=768,force_hidden=768,num_inference_steps=16),num_inference_steps=16)
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


@pytest.mark.parametrize('mismatch',['normalization','architecture','dataset'])
def test_export_rejects_mismatched_training_artifacts(tmp_path,mismatch):
    config,manifest,paths=fixture(tmp_path)
    if mismatch=='normalization':paths['normalization'].write_text('{"statistics":{"force":"changed"}}')
    elif mismatch=='architecture':config['model']['force_hidden']=1280
    else:config['model']['expected_data_identity']='c'*64
    with pytest.raises(ValueError):export.runtime_config(config,manifest,**paths)
