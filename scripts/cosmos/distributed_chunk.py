"""Independent clean-action/chunk experiment with the unchanged four-H200 recipe.

Existing runs are never stopped, reused as outputs, or used as initialization.
Fresh two-Adam/save and independent four-rank restore must pass before formal.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/cosmos'))
from distributed_fresh import sha, clean_environment, phase_commands, run_phases


def new_recipe(original, run):
    from fastwam.cosmos.preflight import validate_config
    validate_config(original)
    run = Path(run)
    old = Path(original['output_dir'])
    if not run.is_absolute() or run == old or run in old.parents or old in run.parents:
        raise ValueError('New independent absolute output directory required')
    cfg = copy.deepcopy(original)
    cfg['output_dir'] = str(run/'training')
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'run-root', 'native-python'):
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
    cfg = OmegaConf.load(args.config)
    validate_config(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    original = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    cfg = OmegaConf.create(new_recipe(original, run))
    run.mkdir(parents=False, exist_ok=False)
    OmegaConf.save(cfg, str(run/'resolved.yaml'))
    journal = {'status': 'starting', 'resource': resource, 'source': str(ROOT), 'resume': False,
               'experiment': 'clean-action-force-chunk', 'action_semantics': 'GT-conditioned reconstruction',
               'loss_weights': {'video': 1, 'action': 1, 'force': 1}, 'original_config_sha256': sha(args.config),
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
