"""Real CPU child/session cleanup. No GPU, train or save result is implied."""
from pathlib import Path
import json,subprocess,sys
import pytest
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('mode',['unlimited_success','nonzero_orphans','bounded_timeout'])
def test_supervisor_owns_four_sessions(tmp_path,mode):
    duration='.15' if mode=='unlimited_success' else '20'
    workers=tmp_path/'workers.py'
    workers.write_text('import subprocess,sys,time\n'
        +f"children=[subprocess.Popen([sys.executable,'-c','import time;time.sleep({duration})'],start_new_session=True) for _ in range(4)]\n"
        +('time.sleep(.2);raise SystemExit(7)\n' if mode=='nonzero_orphans' else '[p.wait() for p in children]\n'))
    report=tmp_path/'result.json';out=tmp_path/'watch'
    seconds='1' if mode=='bounded_timeout' else 'None'
    driver=tmp_path/'driver.py'
    driver.write_text('import sys,json\nfrom pathlib import Path\n'
        +f'sys.path.insert(0,{str(ROOT/"scripts/cosmos")!r})\n'
        +'from bounded_probe import execute\n'
        +f'r=execute([{sys.executable!r},{str(workers)!r}],{str(tmp_path)!r},{str(out)!r},{seconds},maximum_seconds={"3" if mode=="bounded_timeout" else "None"})\n'
        +f'Path({str(report)!r}).write_text(json.dumps(r))\n')
    proc=subprocess.run(["/usr/bin/python3",str(driver)],capture_output=True,text=True,timeout=30)
    assert proc.returncode==0,proc.stderr
    result=json.loads(report.read_text());tree=result['ownership']
    assert result['subreaper_restored'] and tree['closure_verified'] and not tree['live_after_cleanup'] and not tree['errors']
    if mode=='unlimited_success':assert result['status']=='process_exited_zero' and not tree['signals']
    else:assert result['status']=='failed' and tree['signals']
