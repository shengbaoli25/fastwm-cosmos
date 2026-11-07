"""Bind a fresh JEPA dataset without changing the all78 experiment identity."""
import json
from pathlib import Path
import re


def load_spec(path):
    return validate_spec(json.loads(Path(path).read_text(encoding='utf-8')))


def validate_spec(spec):
    digest = spec.get('source_inventory_sha256', '')
    count = spec.get('source_episodes')
    prompts = spec.get('prompts')
    if (spec.get('schema') != 'cosmos-jepa-dataset-spec-v1'
            or not re.fullmatch(r'[a-z0-9][a-z0-9-]+', spec.get('dataset_id', ''))
            or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)
            or type(count) is not int or count <= 0
            or any(type(spec.get(k)) is not int or spec[k] <= 0
                   for k in ('source_frames', 'training_anchors', 'continuous_segments'))
            or spec['training_anchors'] > spec['source_frames']
            or not isinstance(prompts, list) or not prompts
            or any(not isinstance(p, str) or not p for p in prompts)
            or prompts != sorted(set(prompts))
            or spec.get('split_counts') != dict(train=count, val=0, test=0)
            or spec.get('action_representation') != 'delta_current_joint7'
            or spec.get('variant') != 'vitl'
            or spec.get('initialization') != 'official_cosmos_and_jepa_fresh_action_force_optimizer'):
        raise ValueError('An exact all-episode Joint8 dataset specification is required')
    return spec


def verify_source(spec, source):
    validate_spec(spec)
    count = spec['source_episodes']
    if (source.source_sha != spec['source_inventory_sha256']
            or len(source.lengths) != count or sum(source.lengths) != spec['source_frames']
            or len(source) != spec['training_anchors']
            or len(source.segments) != spec['continuous_segments']
            or sorted(set(source.prompts)) != spec['prompts']
            or source.identity.get('action_representation') != spec['action_representation']
            or source.identity.get('source_episode_indices') != list(range(count))
            or source.identity.get('split_counts') != spec['split_counts']):
        raise ValueError('Dataset bytes, tasks, retained windows or training selection differ from spec')


def verify_prompt_manifest(spec, manifest):
    if [entry['prompt'] for entry in manifest['texts']] != spec['prompts']:
        raise ValueError('Text bank differs from the recorded dataset task; encode its exact prompt')


def verify_prepared(spec, source, manifest, normalization):
    verify_source(spec, source)
    if (manifest.get('status') != 'complete'
            or manifest.get('completed_samples') != len(source)
            or manifest.get('identity', {}).get('realman_source') != source.identity):
        raise ValueError('Prepared features must come from this exact new dataset and adapter')
    # Source identity includes every input hash, statistics and action semantics.
    if (normalization.get('source_identity_sha256') != source.identity_sha256
            or normalization.get('statistics') != source.stats
            or normalization.get('action_representation') != spec['action_representation']
            or normalization.get('force_preprocessing') != source.force_contract
            or normalization.get('state_columns') != source.identity['state_columns']
            or normalization.get('action_units') != source.identity['action_units']):
        raise ValueError('New dataset normalization must be recomputed from its retained windows')
