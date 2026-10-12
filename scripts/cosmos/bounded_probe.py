"""Bound a disposable probe and all descendants with the reviewed pidfd tracker."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime,timezone
from process_ownership import ProcessOwnership


def execute(command,cwd,output,seconds,*,maximum_seconds=3600,deadline_unix=None,log_name='probe.log',phase='cosmos-force28'):
    unlimited=seconds is None and maximum_seconds is None and deadline_unix is None
    if not unlimited and (not isinstance(seconds,(int,float)) or not math.isfinite(seconds)
            or not isinstance(maximum_seconds,(int,float)) or not 1<=seconds<=maximum_seconds<=172800):
        raise ValueError('Execution exceeds its bounded time allowance')
    if deadline_unix is not None:
        seconds=min(seconds,deadline_unix-time.time()-60)
        if seconds<1:raise ValueError('Absolute deadline has no execution and cleanup time remaining')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    end=float('inf') if unlimited else time.monotonic()+seconds
    tracker=ProcessOwnership();child=None;pending=[]
    handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    result={'status':'starting','started_at':datetime.now(timezone.utc).isoformat(),'command':command,'limit_seconds':seconds}
    def save():
        temp=output/'status.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(output/'status.json')
    def cleanup_deadline():
        cap=time.monotonic()+60
        return cap if deadline_unix is None else min(cap,time.monotonic()+max(0,deadline_unix-time.time()))
    result['absolute_deadline_unix']=deadline_unix
    try:
        for s in handlers:signal.signal(s,lambda s,f:pending.append(s))
        tracker.enable_subreaper()
        with (output/log_name).open('xb') as log:
            child=subprocess.Popen(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            tracker.attach_launcher(child,phase=phase)
        tracker.register_launcher(child,phase=phase);result['status']='running';save()
        while child.poll() is None:
            tracker.observe()
            if pending:raise RuntimeError('Probe interrupted')
            if time.monotonic()>=end or (deadline_unix is not None and time.time()>=deadline_unix-60):
                raise TimeoutError('Execution time limit reached')
            try:child.wait(timeout=.2)
            except subprocess.TimeoutExpired:pass
        result['returncode']=child.returncode
        closure=tracker.cleanup(cleanup_deadline())
        if child.returncode!=0:raise RuntimeError('Native probe exited nonzero')
        if not closure['closure_verified'] or closure['signals'] or closure['errors']:
            raise RuntimeError('Successful probe requires unforced full descendant closure')
        result['status']='process_exited_zero'
    except BaseException as error:
        result.update(status='failed',error=str(error))
    finally:
        if tracker.enabled:
            result['ownership']=tracker.cleanup(cleanup_deadline())
            try:tracker.restore_subreaper();result['subreaper_restored']=True
            except Exception as e:result.update(status='failed',cleanup_error=str(e))
        if child is not None:result['returncode']=child.poll()
        for s,h in handlers.items():signal.signal(s,h)
        result['finished_at']=datetime.now(timezone.utc).isoformat();save()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--command-json',required=True);p.add_argument('--cwd',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seconds',type=float,default=1800)
    p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    command=json.loads(Path(a.command_json).read_text())
    if not isinstance(command,list) or not command or not all(isinstance(x,str) for x in command):p.error('JSON argv list required')
    if not a.execute:print(json.dumps({'mode':'plan_only','argv':command,'seconds':a.seconds}))
    else:
        result=execute(command,a.cwd,a.output,a.seconds,deadline_unix=a.deadline_unix)
        print(json.dumps(result));raise SystemExit(0 if result['status']=='process_exited_zero' else 2)
