"""CPU-only verification of this exact ViT-B run; never starts a server or robot."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
EXPECTED_STEP = 20000
EXPECTED_TEACHER_SHA = '848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d'
EXPECTED_TEACHER_COMMIT = '204698b45b3712590f06245fbfba32d3be539812'
EXPECTED_CACHE = '7019254c3a99f3859e0cf742a211a7fbd05201797c7e17066d799c41174fe0ce'
EXPECTED_DATASET = 'b24b701239f54ad38d16c819b9ea9f6fad5f9374a683a84a68d2cd7ef27067a8'
EXPECTED_ARCHITECTURE = dict(feature_dim=768, hidden_dim=768, depth=28, heads=12,
    view_grid=14, action_hidden=768, action_ffn=3072, force_hidden=768, force_ffn=2048,
    joint_heads=16, joint_head_dim=128, context_dim=1024)
SOURCE_HASHES = {
    'src/fastwam/cosmos/dense_features.py': '89b91d5735c3f1dc522ee0e290724fc588f35be166fb736d696ef12f23544cb2',
    'src/fastwam/cosmos/dense_encoder.py': 'b5c5d4a7859460def5f13584895f76647290adab14e7c5d442bbf0741ad0b96f',
}
ASSETS = {
    'training_config': 'config/training.config.yaml',
    'normalization_file': 'config/normalization.json',
    'force_preprocessing_file': 'config/force.json',
    'teacher_weights': 'assets/vjepa2_1_vitb_dist_vitG_384.pt',
    'checkpoint': 'checkpoints/step_020000.pt',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def runtime_config():
    """All artifact paths resolve relative to the runtime JSON's config folder."""
    return dict(training_config='training.config.yaml', normalization_file='normalization.json',
        force_preprocessing_file='force.json', teacher_source='../code/third_party/vjepa21',
        teacher_weights='../assets/vjepa2_1_vitb_dist_vitG_384.pt', device='cuda:0',
        model_dtype='bfloat16', camera_order=['global', 'wrist'], camera_width=640,
        camera_height=480, action_horizon=32, sampling_profile='dense16',
        execution_mode='synchronous', num_inference_steps=16, action_seed=42, force_seed=43)


def check_training_identity(trained):
    model = trained.get('model', {})
    manifest = model.get('manifest', {})
    teacher = manifest.get('teacher', {})
    if (trained.get('teacher') != 'vjepa21' or teacher.get('name') != 'vjepa21'
            or teacher.get('upstream_commit') != EXPECTED_TEACHER_COMMIT
            or teacher.get('weight_sha256') != EXPECTED_TEACHER_SHA
            or manifest.get('cache_identity_sha256') != EXPECTED_CACHE
            or manifest.get('dataset_source_sha256') != EXPECTED_DATASET
            or manifest.get('source_revision') != 'fastwam-jepa-video-window-v2'
            or model.get('depth') != 28 or model.get('feature_dim') != 768
            or model.get('num_inference_steps') != 16):
        raise ValueError('Training config is not the pinned RM71D78 ViT-B 28-layer run')
    train = trained.get('data', {}).get('train', {})
    for field, expected in (('source_sha256', EXPECTED_DATASET),
            ('expected_cache_identity', EXPECTED_CACHE), ('expected_teacher', 'vjepa21'),
            ('expected_weight_sha256', EXPECTED_TEACHER_SHA)):
        if train.get(field) != expected:
            raise ValueError(f'Training data identity differs: {field}')
    if trained.get('teacher_weight_sha256') != EXPECTED_TEACHER_SHA:
        raise ValueError('Training teacher weight identity differs')
    if any(type(model.get(key)) is not int or model[key] != expected
           for key, expected in EXPECTED_ARCHITECTURE.items()):
        raise ValueError('Training model geometry differs from the pinned ViT-B 28-layer experiment')


def check_checkpoint(payload, trained, *, scan_tensors=True):
    import torch
    from fastwam.cosmos.dense_features import DenseFeatureJointModel, SCHEMA
    from fastwam.cosmos.dense_encoder import _FactoryScalarsOnCPU
    from realman_sync.dense_policy import DensePolicy

    if (not isinstance(payload, dict) or payload.get('schema') != SCHEMA
            or type(payload.get('step')) is not int or payload['step'] != EXPECTED_STEP
            or not isinstance(payload.get('model'), dict) or not payload['model']
            or not isinstance(payload.get('manifest'), dict)):
        raise ValueError('Expected the complete dense Joint8 step_020000 weights checkpoint')
    validator = object.__new__(DensePolicy)
    kwargs = validator._model_configuration(trained, payload['manifest'])
    # The original source constructor is used unchanged. Only its parameter-free
    # scheduler linspace is on CPU, so no multi-GB random initialization occurs.
    with torch.device('meta'), _FactoryScalarsOnCPU():
        model = DenseFeatureJointModel(**kwargs)
    if model.manifest != payload['manifest']:
        raise ValueError('Full reconstructed source model manifest differs from checkpoint')
    expected = model.state_dict()
    if set(expected) != set(payload['model']):
        raise ValueError('Checkpoint state keys differ from the exact source architecture')
    tensor_count = elements = 0
    for name, value in payload['model'].items():
        reference = expected[name]
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                or value.is_meta or tuple(value.shape) != tuple(reference.shape)):
            raise ValueError(f'Checkpoint tensor type/device/shape differs: {name}')
        if not value.is_floating_point():
            raise ValueError(f'Unexpected non-floating dense policy state: {name}')
        if scan_tensors:
            for block in value.reshape(-1).split(1 << 20):
                if not bool(torch.isfinite(block).all()):
                    raise ValueError(f'Nonfinite checkpoint tensor: {name}')
        tensor_count += 1
        elements += value.numel()
    model.load_state_dict(payload['model'], strict=True, assign=True)
    return dict(step=payload['step'], schema=SCHEMA, tensor_count=tensor_count,
        elements=elements, strict_state_loaded=True, finite_tensors_scanned=scan_tensors,
        trained_num_inference_steps=model.num_inference_steps, manifest=deepcopy(model.manifest))


def verify(root, *, scan_tensors=True):
    root = Path(root).resolve()
    report = dict(root=str(root), ready=False, gpu_accessed=False, server_started=False,
                  runtime_written=False, missing=[], checks={}, errors=[])
    for key, relative in ASSETS.items():
        if not (root / relative).is_file():
            report['missing'].append(dict(asset=key, path=str(root / relative)))
    source = root / 'code'
    teacher_source = source / 'third_party/vjepa21'
    required_source = ('REVISION', 'app/vjepa_2_1/models/vision_transformer.py',
                       'src/utils/tensors.py')
    for relative in required_source:
        if not (teacher_source / relative).is_file():
            report['missing'].append(dict(asset='teacher_source', path=str(teacher_source / relative)))
    try:
        for relative, expected in SOURCE_HASHES.items():
            if sha256(source / relative) != expected:
                raise ValueError(f'Original model source SHA256 differs: {relative}')
        report['checks']['original_model_source_sha256'] = dict(SOURCE_HASHES)
        if (teacher_source / 'REVISION').is_file():
            revision = (teacher_source / 'REVISION').read_text().strip()
            if revision != EXPECTED_TEACHER_COMMIT:
                raise ValueError('Official teacher source revision differs')
            report['checks']['teacher_source_revision'] = revision
        if report['missing']:
            return report
        sys.path[:0] = [str(source / 'deploy'), str(source / 'src')]
        import torch
        from omegaconf import OmegaConf
        from realman_sync.dense_policy import DensePolicy
        from realman_force.contract import validate_contract_identity
        if torch.cuda.is_initialized():
            raise RuntimeError('Bootstrap must run in a fresh CPU-only process')
        trained = OmegaConf.to_container(OmegaConf.load(root / ASSETS['training_config']), resolve=True)
        check_training_identity(trained)
        report['checks']['training_config_sha256'] = sha256(root / ASSETS['training_config'])
        teacher_sha = sha256(root / ASSETS['teacher_weights'])
        if teacher_sha != EXPECTED_TEACHER_SHA:
            raise ValueError('Teacher weight SHA256 differs from the saved training run')
        report['checks']['teacher_weight_sha256'] = teacher_sha
        payload = torch.load(root / ASSETS['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
        report['checks']['checkpoint'] = check_checkpoint(payload, trained, scan_tensors=scan_tensors)
        validator = object.__new__(DensePolicy)
        validator.config_dir = root / 'config'
        validator.config = runtime_config()
        validator._load_normalizer(payload['manifest'])
        force = validate_contract_identity(validator.force_preprocessing)
        if (force.get('units') != ['N'] * 3 + ['Nm'] * 3
                or force.get('sampling', {}).get('fps') != 30):
            raise ValueError('Force preprocessing units/rate do not match the trained 30 Hz Joint8 protocol')
        report['checks']['normalization_sha256'] = sha256(root / ASSETS['normalization_file'])
        report['checks']['force_contract_id'] = force['id']
        report['checks']['force_file_sha256'] = sha256(root / ASSETS['force_preprocessing_file'])
        report['checks']['force_dataset_binding'] = 'Validate downloaded dataset provenance separately; checkpoint has no force-contract hash'
        report['checks']['checkpoint_sha256'] = sha256(root / ASSETS['checkpoint'])
        report['checks']['cuda_initialized'] = torch.cuda.is_initialized()
        if report['checks']['cuda_initialized']:
            raise RuntimeError('Unexpected CUDA initialization during CPU-only bootstrap')
        report['ready'] = True
    except Exception as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--write-runtime', action='store_true', help='Write runtime only after successful verification')
    parser.add_argument('--metadata-only', action='store_true', help='Skip full finite-tensor scan; strict state checks still run')
    args = parser.parse_args()
    report = verify(args.root, scan_tensors=not args.metadata_only)
    if report['ready'] and args.write_runtime:
        target = args.root / 'config/runtime.step_020000.json'
        config = runtime_config()
        if target.is_file():
            # Preserve an explicitly selected dashboard profile across rechecks.
            from realman_sync.dense_policy import _sampling_config
            previous = _sampling_config(json.loads(target.read_text(encoding='utf-8')))
            config.update({key: previous[key] for key in
                ('sampling_profile', 'num_inference_steps', 'execution_mode')})
        write_json(target, config)
        report.update(runtime_written=True, runtime_path=str(target), sampling_profile=config['sampling_profile'])
    write_json(args.root / 'config/bootstrap.check.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
