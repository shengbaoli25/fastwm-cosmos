"""Shared setup for the dense-feature GPU probe, restore check and formal run."""
import json
import os
from pathlib import Path


def load_run(config_path, run_root, dataset, source_sha, cache_root, teacher,
             teacher_source, weights, weight_sha):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config_path = Path(config_path).resolve()
    run_root = Path(run_root).resolve()
    root = Path(cache_root).resolve()
    if not config_path.is_file():
        raise FileNotFoundError('Training configuration is missing')
    manifest_path = root / 'manifest.json'
    cache = json.loads(manifest_path.read_text())
    identity = cache.get('identity', {})
    from fastwam.cache.latent import cache_identity_sha256
    digest = cache_identity_sha256(identity)
    expected_revision = ('204698b45b3712590f06245fbfba32d3be539812'
                         if teacher == 'vjepa21' else
                         '6876159a11b4df116f30f667f8c9888617df0751')
    if (cache.get('status') != 'complete' or digest != cache.get('identity_sha256')
            or identity.get('dataset_source_sha256') != source_sha
            or identity.get('teacher', {}).get('name') != teacher
            or identity.get('teacher', {}).get('upstream_commit') != expected_revision
            or identity.get('teacher', {}).get('weight_sha256') != weight_sha
            or identity.get('episodes') != 71 or identity.get('feature_shape') != [392, 768]):
        raise ValueError('Completed feature cache identity is not bound to this dataset and teacher')
    if teacher == 'dinov3' and os.environ.get('DINOV3_LICENSE_ACCEPTED') != '1':
        raise PermissionError('DINOv3 terms must be accepted before the materials are used')
    cfg = OmegaConf.load(config_path)
    cfg.output_dir = str(run_root / 'training')
    cfg.data.train.dataset_root = str(Path(dataset).resolve())
    cfg.data.train.source_sha256 = source_sha
    cfg.data.train.feature_root = str(root)
    cfg.data.train.expected_cache_identity = digest
    cfg.data.train.expected_teacher = teacher
    cfg.data.train.expected_weight_sha256 = weight_sha
    cfg.model.manifest = dict(dataset_source_sha256=source_sha,
        cache_identity_sha256=digest, teacher=identity['teacher'],
        observation_offsets=[0], future_offsets=[16, 32], stride=4,
        source_revision=identity.get('exporter_schema'))
    cfg.teacher = teacher
    cfg.teacher_source = str(Path(teacher_source).resolve())
    cfg.teacher_weights = str(Path(weights).resolve())
    cfg.teacher_weight_sha256 = weight_sha
    model = instantiate(cfg.model, device=f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}",
                        model_dtype=torch.bfloat16)
    train_dataset = instantiate(cfg.data.train)
    return cfg, model, train_dataset, cache


def fingerprint(trainer, model):
    from fastwam.cosmos.persistence import state_digest
    return dict(model=state_digest(model.state_dict()),
        optimizer=state_digest(trainer.optimizer.state_dict()),
        scheduler=state_digest(trainer.scheduler.state_dict()),
        cpu_rng=state_digest(__import__('torch').get_rng_state()),
        cuda_rng=state_digest(__import__('torch').cuda.get_rng_state()),
        trainer=dict(global_step=trainer.global_step, epoch=trainer.epoch,
                     batch_in_epoch=trainer.batch_in_epoch))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + '\n')
    temp.replace(path)
