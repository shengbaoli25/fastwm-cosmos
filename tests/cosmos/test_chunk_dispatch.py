"""Independent experiment preserves the recipe and cannot reuse existing output."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest
from test_config import config

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('chunk_driver',ROOT/'scripts/cosmos/distributed_chunk.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def test_new_recipe_changes_only_output_and_keeps_input_immutable(tmp_path):
    old=config();old['output_dir']=str(tmp_path/'old');before=copy.deepcopy(old)
    actual=driver.new_recipe(old,tmp_path/'new')
    assert old==before
    assert actual.pop('output_dir')==str(tmp_path/'new/training')
    before.pop('output_dir');assert actual==before


@pytest.mark.parametrize('path',['old','old/child','.'])
def test_existing_run_or_ancestor_not_reused(tmp_path,path):
    old=config();old['output_dir']=str(tmp_path/'old')
    with pytest.raises(ValueError):driver.new_recipe(old,tmp_path/path)


def test_default_plan_does_not_create_files(tmp_path):
    run=tmp_path/'new'
    r=subprocess.run([sys.executable,str(ROOT/'scripts/cosmos/distributed_chunk.py'),
        '--config',str(tmp_path/'absent.yaml'),'--run-root',str(run),
        '--native-python','/shared/.venv/bin/python'],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    data=json.loads(r.stdout)
    assert data['mode']=='plan_only' and data['ranks']==4 and data['max_steps']==30000
    assert not run.exists()
