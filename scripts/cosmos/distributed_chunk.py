"""Independent clean-action/chunk experiment with the unchanged four-H200 recipe.

Existing outputs are never reused. An explicit SHA-bound model-only warm start
may initialize a changed architecture; optimizer/scheduler start fresh.
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


def experiment_contract(model):
    """Keep journal semantics faithful to the actual conditioning interface."""
    target = model['_target_']
    if target.endswith(('.create_reference_joint_cosmos', '.create_reference_jepa_cosmos')):
        experiment = ('cosmos-reference-audited-jepa-force-latent'
                      if target.endswith('.create_reference_jepa_cosmos')
                      else 'cosmos-reference-audited-joint-diffusion')
        return (experiment,
                'observation-audited full-plan condition to AE only; unchanged Joint8 FM targets',
                {'video': 1, 'action': 1, 'force': 1,
                 'reference_alignment': model['alignment_weight']})
    if target.endswith('.create_jepa_force_latent_cosmos'):
        return 'cosmos-jepa-force-latent', 'no-reference joint diffusion', {'video': 1, 'action': 1, 'force': 1}
    if target.endswith('.create_noaction_joint_cosmos'):
        return 'cosmos-noaction-joint-diffusion', 'no-reference joint diffusion', {'video': 1, 'action': 1, 'force': 1}
    if target.endswith('.create_joint_cosmos_forcewm'):
        return 'cosmos-force-joint-self-attention', 'GT-conditioned reconstruction', {'video': 1, 'action': 1, 'force': 1}
    return 'clean-action-force-chunk', 'GT-conditioned reconstruction', {'video': 1, 'action': 1, 'force': 1}


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
        batch, steps = None, 30000
        if args.config.is_file():
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(args.config)
            batch, steps = 4*int(cfg.batch_size), int(cfg.max_steps)
        print(json.dumps({'mode': 'plan_only', 'commands': commands, 'resume': False,
                          'ranks': 4, 'global_batch': batch, 'max_steps': steps,
                          'configuration_resolved': args.config.is_file()}))
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
    experiment, action_semantics, loss_weights = experiment_contract(original['model'])
    journal = {'status': 'starting', 'resource': resource, 'source': str(ROOT), 'resume': False,
               'experiment': experiment, 'action_semantics': action_semantics,
               'loss_weights': loss_weights, 'original_config_sha256': sha(args.config),
               'config_sha256': sha(run/'resolved.yaml'), 'commands': commands,
               'user_time_cap': None, 'started_at': datetime.now(timezone.utc).isoformat()}
    if experiment in ('cosmos-jepa-force-latent', 'cosmos-reference-audited-jepa-force-latent'):
        journal['auxiliary'] = dict(variant=original['model']['observation_encoder']['variant'],
            force_latent_weight=original['model']['latent_loss_weight'],
            latent_warmup_steps=original['model']['latent_warmup_steps'],
            wrench_weight=original['model']['wrench_loss_weight'],
            target='online_current_force_clean_observed_only_stop_gradient', external_teacher_required=False,
            appearance_consistency_weight=original['jepa_experiment']['appearance_consistency_weight'])
    def update(phase, status):
        journal.update(phase=phase, status=status, updated_at=datetime.now(timezone.utc).isoformat())
        temp=run/'dispatch.tmp'; temp.write_text(json.dumps(journal, indent=2)); temp.replace(run/'dispatch.json')
        print(json.dumps({'phase': phase, 'status': status, 'run_root': str(run)}), flush=True)
    try:
        run_phases(commands, ROOT, run, execute, update)
        # Native ranks and supervisor verify completion; also require final full-state save.
        sys.path.insert(0, str(ROOT/'scripts/resume'))
        from checks import complete_state
        state, step = complete_state(run/'training', run/'formal-watch/training.log', final=True,
                                    max_steps=int(cfg.max_steps),save_every=int(cfg.save_every))
        if step != cfg.max_steps:
            raise ValueError('Final complete configured-step save is absent')
        for rank in range(4):
            origin=json.loads((run/'training/origins'/f'rank_{rank:03d}.json').read_text())
            if (origin.get('rank') != rank or origin.get('global_step') != cfg.max_steps
                    or origin.get('status') != 'complete' or origin.get('resource') != resource):
                raise ValueError('All four completed original training ranks required')
        journal['final_state'] = str(state)
        update('complete', 'completed_'+str(cfg.max_steps))
    except BaseException as error:
        journal['error'] = str(error); update(journal.get('phase', 'dispatch'), 'failed'); raise


if __name__ == '__main__':
    main()
