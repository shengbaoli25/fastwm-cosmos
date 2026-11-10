"""CPU-only, fail-closed provenance binding shared by dataset and CLI."""
import json
from pathlib import Path, PurePosixPath
from .reference_bank import check, digest, file_digest, verify_bank
from .joint_action import validate_joint_action_contract

PROFILE = 'realman-leader-joint8-filtered-v1'
REPRESENTATION = 'delta_current_joint7'


def relevant(path):
    return path.startswith(('data/', 'meta/'))


def valid_inventory(files):
    check(isinstance(files, dict) and bool(files), 'Missing complete source files mapping')
    for key, value in files.items():
        check(isinstance(key, str) and not PurePosixPath(key).is_absolute()
              and '..' not in PurePosixPath(key).parts and '\\' not in key
              and PurePosixPath(key).as_posix() == key, 'Invalid source inventory path')
        check(isinstance(value, str) and len(value) == 64
              and all(c in '0123456789abcdef' for c in value), 'Invalid source file SHA256')


def differences(local, cached):
    return dict(
        only_local=sorted(local.keys()-cached.keys()),
        only_cache=sorted(cached.keys()-local.keys()),
        changed=[dict(path=p, local_sha256=local[p], cache_sha256=cached[p])
                 for p in sorted(local.keys() & cached.keys()) if local[p] != cached[p]],
    )


def has_differences(diff):
    return any(diff.values())


def verify_binding(bank, cache_manifest, *, local_raw_inventory=None):
    """Return a diagnostic receipt; rejected bindings retain reasons and diffs."""
    bank, cache_manifest = Path(bank).resolve(), Path(cache_manifest).resolve()
    receipt = dict(
        schema='realman-reference-cache-binding-v1', status='rejected',
        reference_bank=str(bank), cache_manifest=str(cache_manifest),
        numeric_metadata_binding_verified=False, full_raw_inventory_equal=None,
        cache_tensor_payloads_verified=False, model_integration_verified=False,
        training_readiness_evaluated=False, errors=[],
    )
    try:
        m = verify_bank(bank)
        receipt['reference_identity_sha256'] = m['identity_sha256']
        receipt['reference_manifest_sha256'] = file_digest(bank/'manifest.json')
        receipt['cache_manifest_sha256'] = file_digest(cache_manifest)
        cache = json.loads(cache_manifest.read_text(encoding='utf-8'))
        identity, b = cache['identity'], m['identity']
        check(cache.get('identity_sha256') == digest(identity), 'Cache identity digest differs')
        receipt['cache_identity_sha256'] = cache['identity_sha256']
        check(cache.get('status') == 'complete', 'Incomplete cache manifest')
        check(identity.get('schema') == 'cosmos-force28-full-window-cache-v1'
              and identity.get('scope') == 'production'
              and identity.get('dataset_profile') == PROFILE, 'Wrong Cosmos Joint8 cache schema/profile')
        source = identity['realman_source']
        check(source.get('profile') == PROFILE
              and source.get('source_schema') == 'realman_leader_force_gripper_v2',
              'Wrong measured Joint8 source schema/profile')
        local, cached = b['source_files'], source['files']
        valid_inventory(local)
        valid_inventory(cached)
        check(b['source_inventory_sha256'] == digest(local), 'Bank source inventory digest differs')
        check(source.get('source_inventory_sha256') == digest(cached), 'Cache raw inventory digest differs')
        receipt['bank_source_inventory_scope'] = b['source_inventory_scope']
        receipt['bank_source_inventory_sha256'] = b['source_inventory_sha256']
        receipt['cache_raw_inventory_sha256'] = source['source_inventory_sha256']
        for key in ('numeric_identity', 'numeric_sidecar_identity', 'pixel_identity'):
            check(identity.get(key) == digest(source), 'Cache source identity differs: '+key)

        # Compare both key sets: an omitted numeric/metadata file cannot be hidden
        # by comparing only the common subset of filenames.
        local_numeric = {k:v for k,v in local.items() if relevant(k)}
        cached_numeric = {k:v for k,v in cached.items() if relevant(k)}
        check(any(k.startswith('data/') for k in local_numeric)
              and 'meta/info.json' in local_numeric, 'Bank lacks numeric/metadata lineage')
        delta = differences(local_numeric, cached_numeric)
        receipt['numeric_metadata_diff'] = delta
        receipt['numeric_metadata_file_count'] = len(local_numeric)
        receipt['numeric_metadata_files_equal'] = not has_differences(delta)

        full_local = local if b['source_inventory_scope'] == 'complete_raw_inventory' else None
        if local_raw_inventory is not None:
            inventory_path = Path(local_raw_inventory).resolve()
            full_local = json.loads(inventory_path.read_text(encoding='utf-8'))
            valid_inventory(full_local)
            check(all(full_local.get(k) == v for k,v in local.items()),
                  'Optional full raw inventory does not contain the exact bank source files')
            check({k:v for k,v in full_local.items() if relevant(k)} == local_numeric,
                  'Optional full inventory has additional numeric/metadata files')
            receipt['local_raw_inventory_file'] = str(inventory_path)
            receipt['local_raw_inventory_file_sha256'] = file_digest(inventory_path)
        if full_local is not None:
            receipt['local_raw_inventory_sha256'] = digest(full_local)
            receipt['full_raw_inventory_equal'] = full_local == cached
            receipt['other_files_diff'] = differences(
                {k:v for k,v in full_local.items() if not relevant(k)},
                {k:v for k,v in cached.items() if not relevant(k)})
            receipt['other_files_comparison'] = 'complete_local_inventory_vs_cached_inventory'
        else:
            receipt['other_files_comparison'] = 'not_available_in_metadata_numeric_only_bank'
            receipt['other_files_diff'] = None
            receipt['cache_other_files_uncompared'] = sorted(k for k in cached if not relevant(k))

        check(not has_differences(delta), 'Numeric/metadata source files differ; refuse cache binding')
        check(source.get('action_space') == 'joint8'
              and source.get('state_columns') == [0,1,2,3,4,5,6,19]
              and source.get('action_columns') == list(range(8))
              and (source.get('state_dim'),source.get('action_dim'),source.get('history_dim')) == (8,8,14),
              'Joint8 observation/action dimensions or columns differ')
        validate_joint_action_contract(source, REPRESENTATION, allow_legacy_absolute=False)
        validate_joint_action_contract(identity, REPRESENTATION, allow_legacy_absolute=False)
        check(source.get('actions') == 'sent_joint_targets_minus_current_measured_joint7_then_absolute_gripper'
              and source.get('action_statistics_scope') == 'all_retained_anchor_windows_valid_action_positions_excluding_padding'
              and isinstance(source.get('action_transform_sha256'), str)
              and len(source['action_transform_sha256']) == 64, 'Delta target/statistics contract differs')
        check(source.get('fps') == b.get('fps') == 30, 'Source frame rate differs')
        lengths = b['source_lengths']
        ids = list(range(len(lengths)))
        check(source.get('episode_lengths') == identity.get('episode_lengths') == lengths,
              'Cache and bank episode lengths differ')
        check(source.get('source_episode_indices') == ids
              and source.get('training_selection') == 'all_episodes_no_holdout'
              and source.get('split_counts') == dict(train=len(ids),val=0,test=0)
              and b.get('splits') == dict(train=ids,val=[],test=[]),
              'This binding requires the original all-episode training membership')
        expected_segments = [[e['episode_index'],lo,hi] for e in m['episodes']
                             for lo,hi in e['segment_bounds']]
        check(source.get('segments') == expected_segments, 'Cache and bank continuous segments differ')
        check(cache.get('completed_samples') == m['total_retained_anchors'], 'Cached anchor count differs')
        for key, expected in dict(action_offsets=list(range(32)), force_offsets=list(range(1,33)),
                                  history_offsets=list(range(-31,1)), image_offsets=list(range(0,33,4))).items():
            check(source.get(key) == expected, 'Temporal target contract differs: '+key)
        receipt.update(status='bound', numeric_metadata_binding_verified=True,
                       episode_count=len(ids), continuous_segment_count=len(expected_segments),
                       retained_anchors=m['total_retained_anchors'],
                       action_representation=REPRESENTATION,
                       join_keys=['episode_index','frame_index'])
    except (ValueError, KeyError, TypeError, OSError, EOFError) as exc:
        receipt['errors'].append(f'{type(exc).__name__}: {exc}')
    return receipt
