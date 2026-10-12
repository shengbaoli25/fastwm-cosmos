"""Fresh foreground platform job after the unsaved interactive run was stopped.

This is not resume: the old run has no formal checkpoint. It reuses its complete
feature cache and recipe, then creates new preflight/restore evidence on the
allocated node before launching 30000 steps from official initialization.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PRIOR_ID = 'dd805094-b321-4f0a-9793-8c5b585ff7a9'
PRIOR = Path('/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/cosmos-dd805094-4rank-30k-v2')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_stopped(evidence, config, prior=PRIOR):
    data = json.loads(Path(evidence).read_text())
    screenshot = Path(evidence).parent / data.get('evidence_file', '')
    if (data.get('resource_id') != PRIOR_ID or data.get('status') != 'STOPPED'
            or data.get('source') != 'platform_ui' or data.get('prior_output') != str(prior)
            or not data.get('observed_at_utc') or not screenshot.is_file()
            or sha(screenshot) != data.get('evidence_sha256')
            or sha(config) != data.get('config_sha256')):
        raise ValueError('Matching actual stopped-platform and original config evidence required')
    if not prior.is_dir():
        raise ValueError('Original output is not mounted')
    if list(prior.rglob('trainer_state.json')) or list((prior/'checkpoints/weights').glob('*.pt')):
        raise ValueError('A formal checkpoint exists: inspect it for resume instead of restarting')
    return data


def clean_environment(env, source):
    result = dict(env)
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                 'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                 'TORCHELASTIC_RUN_ID', 'TORCHELASTIC_RESTART_COUNT'):
        result.pop(name, None)
    if result.get('COSMOS_ALLOCATION_ID'):
        raise ValueError('Notebook allocation environment is not valid for this job')
    result['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    result['PYTHONPATH'] = ':'.join((str(source/'src'), str(source), result.get('PYTHONPATH', '')))
    return result


def phase_commands(native, source, run):
    config = str(run/'resolved.yaml')
    preflight = str(run/'preflight/preflight.json')
    outer = str(run/'preflight-watch/status.json')
    restore = str(run/'restore/restore.json')
    restore_outer = str(run/'restore-watch/status.json')
    launcher = [str(native), '-u', '-m', 'accelerate.commands.launch', '--config_file',
                str(source/'scripts/accelerate_configs/accelerate_zero1_ds_4gpu.yaml')]
    probe = launcher + ['-m', 'fastwam.cosmos.preflight', '--config', config,
                        '--output', str(run/'preflight'), '--execute']
    reload = launcher + [str(source/'scripts/cosmos/restore_probe.py'), '--config', config,
                         '--preflight', preflight, '--preflight-outer', outer,
                         '--output', str(run/'restore'), '--execute']
    formal = [str(native), '-u', str(source/'scripts/cosmos/formal_run.py'), '--config', config,
              '--preflight', preflight, '--preflight-outer', outer, '--restore', restore,
              '--restore-outer', restore_outer, '--supervision', str(run/'formal-watch'),
              '--native-python', str(native), '--execute']
    return [('preflight', probe, 3600), ('restore', reload, 3600), ('formal', formal, None)]


def run_phases(commands, source, run, execute, update):
    for phase, command, seconds in commands:
        update(phase, 'running')
        watch = run / (phase+'-watch' if phase != 'formal' else 'formal-controller-watch')
        result = execute(command, str(source), watch, seconds,
                         maximum_seconds=3600 if seconds is not None else None,
                         phase='cosmos-distributed-'+phase)
        tree = result.get('ownership', {})
        if (result.get('status') != 'process_exited_zero' or result.get('returncode') != 0
                or not result.get('subreaper_restored') or not tree.get('closure_verified')
                or tree.get('signals') or tree.get('errors') or tree.get('live_after_cleanup')):
            update(phase, 'failed')
            raise RuntimeError(phase+' failed: formal training will not be launched or retried')
        update(phase, 'process_exited_zero')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'stop-evidence', 'run-root', 'native-python'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    native, run = args.native_python, args.run_root
    if (not native.is_absolute() or native.parent.name != 'bin'
            or native.parent.parent.name != '.venv' or not run.is_absolute()):
        raise ValueError('Absolute run path and original .venv/bin/python invocation required')
    commands = phase_commands(native, ROOT, run)
    if not args.execute:
        print(json.dumps({'mode': 'plan_only', 'commands': commands, 'resume': False,
                          'ranks': 4, 'global_batch': 64, 'max_steps': 30000}))
        return
    if not native.is_file() or Path(sys.executable) != native:
        raise ValueError('Invoke this foreground command with the original native Python')
    sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
    from omegaconf import OmegaConf
    from fastwam.cosmos.preflight import validate_config
    from fastwam.cosmos.persistence import resource_identity
    from bounded_probe import execute
    if not os.environ.get('TRAIN_JOB_ID'):
        raise ValueError('Submit through the platform distributed-training job UI')
    env = clean_environment(os.environ, ROOT)
    os.environ.clear(); os.environ.update(env)
    resource = resource_identity()
    old = check_stopped(args.stop_evidence, args.config)
    cfg = OmegaConf.load(args.config)
    validate_config(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    if Path(cfg.output_dir) != PRIOR:
        raise ValueError('Original stopped run recipe required')
    cfg.output_dir = str(run/'training')
    run.mkdir(parents=False, exist_ok=False)
    OmegaConf.save(cfg, str(run/'resolved.yaml'))
    journal = {'status': 'starting', 'resource': resource, 'source': str(ROOT), 'resume': False,
               'prior_stop_evidence': old, 'original_config_sha256': sha(args.config),
               'config_sha256': sha(run/'resolved.yaml'), 'commands': commands,
               'user_time_cap': None, 'started_at': datetime.now(timezone.utc).isoformat()}
    def update(phase, status):
        journal.update(phase=phase, status=status, updated_at=datetime.now(timezone.utc).isoformat())
        temp=run/'dispatch.tmp'; temp.write_text(json.dumps(journal, indent=2)); temp.replace(run/'dispatch.json')
        print(json.dumps({'phase': phase, 'status': status, 'run_root': str(run)}), flush=True)
    try:
        run_phases(commands, ROOT, run, execute, update)
        # Native ranks and supervisor verify completion; also require final full-state save.
        sys.path.insert(0, str(ROOT/'scripts/resume'))
        from checks import complete_state
        state, step = complete_state(run/'training', run/'formal-watch/training.log', final=True)
        if step != 30000:
            raise ValueError('Final complete 30000-step save is absent')
        for rank in range(4):
            origin=json.loads((run/'training/origins'/f'rank_{rank:03d}.json').read_text())
            if (origin.get('rank') != rank or origin.get('global_step') != 30000
                    or origin.get('status') != 'complete' or origin.get('resource') != resource):
                raise ValueError('All four completed original training ranks required')
        journal['final_state'] = str(state)
        update('complete', 'completed_30000')
    except BaseException as error:
        journal['error'] = str(error); update(journal.get('phase', 'dispatch'), 'failed'); raise


if __name__ == '__main__':
    main()
