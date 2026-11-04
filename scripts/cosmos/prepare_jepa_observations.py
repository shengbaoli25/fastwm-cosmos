"""Cache observed RGB for online JEPA encoding; no trained teacher is needed."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('training-config', 'dataset', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--features', type=Path, help='Relocated existing baseline feature cache')
    args = parser.parse_args()
    import torch
    from omegaconf import OmegaConf
    from fastwam.cosmos.cache import CosmosCachedDataset
    from fastwam.cosmos.realman_joint import open_realman_source
    from fastwam.cosmos.jepa_training_data import build_observation_cache
    torch.set_num_threads(4)
    config = OmegaConf.to_container(OmegaConf.load(args.training_config), resolve=True, throw_on_missing=True)
    base_config = dict(config['data']['train'])
    if base_config.pop('_target_') != 'fastwam.cosmos.cache.CosmosCachedDataset':
        raise ValueError('Resolved baseline training data configuration required')
    if args.features:
        base_config['cache_root'] = str(args.features.resolve())
    base_config['expected_action_representation'] = 'delta_current_joint7'
    dataset = CosmosCachedDataset(**base_config)
    identity = dataset.manifest['identity']['realman_source']
    if (len(dataset.manifest['identity']['episode_lengths']) != 78
            or identity['source_episode_indices'] != list(range(78))
            or identity['split_counts'] != dict(train=78, val=0, test=0)):
        raise ValueError('Exactly the existing all78 delta windows are required')
    source = open_realman_source(args.dataset, args.source_sha, action_representation='delta_current_joint7')
    manifest = build_observation_cache(source, dataset.manifest, args.output,
        on_episode=lambda done, total: print(json.dumps(dict(phase='observed_rgb', episodes=done,
                                                              total_episodes=total)), flush=True))
    print(json.dumps(dict(status='observed_rgb_complete', identity=manifest['identity_sha256'],
                          samples=len(dataset), teacher_required=False)), flush=True)


if __name__ == '__main__':
    main()
