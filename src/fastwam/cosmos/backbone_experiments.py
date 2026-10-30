"""Pinned assets and equal downstream recipes for the RM71D78 comparison."""
from pathlib import Path
import json


DATASET_SHA256 = 'b24b701239f54ad38d16c819b9ea9f6fad5f9374a683a84a68d2cd7ef27067a8'
DATASET_EPISODES = 78
DATASET_FRAMES = 86608
DATASET_WINDOWS = 66919
VARIANTS = {
    'vjepa_vitb': dict(
        teacher='vjepa21', feature_dim=768, vendor='vjepa21',
        config='configs/experiments/rm71d78_vjepa_vitb_28l_50k.yaml',
        model_id='ChongZheng/V-JEPA-2.1-384',
        asset_revision='b35cd7c431265ce6cad8f2af85863e201bc07115',
        filename='vjepa2_1_vitb_dist_vitG_384.pt',
        weight_sha256='848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d'),
    'dinov3_vithplus': dict(
        teacher='dinov3_vithplus', feature_dim=1280, vendor='dinov3',
        config='configs/experiments/rm71d78_dinov3_vithplus_28l_50k.yaml',
        model_id='facebook/dinov3-vith16plus-pretrain-lvd1689m',
        asset_revision='a7e110a46be5aab90d0d369a986728826fec06ee',
        filename='model.safetensors', size_bytes=3362432800,
        weight_sha256='3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d'),
}


def default_variant(root):
    selection = json.loads((Path(root) / 'configs/backbone_variant.json').read_text())
    name = selection['variant']
    if name not in VARIANTS or selection.get('config') != VARIANTS[name]['config']:
        raise ValueError('Branch default variant and config disagree')
    return name


def validate_recipe(cfg, variant):
    """Reject accidental old 12-layer models, old data or different optimization."""
    spec = VARIANTS[variant]
    expected_model = dict(feature_dim=spec['feature_dim'], hidden_dim=768, depth=28,
        heads=12, view_grid=14, action_hidden=768, action_ffn=3072,
        force_hidden=768, force_ffn=2048, joint_heads=16, joint_head_dim=128,
        context_dim=1024, force_shift=1.0, lambda_future=1.0,
        action_seed=42, force_seed=43, proprio_seed=44,
        gradient_checkpointing=True, num_inference_steps=16)
    expected_train = dict(dataset_episode_count=78, batch_size=64,
        gradient_accumulation_steps=1, learning_rate=1e-4, weight_decay=1e-2,
        max_steps=50000, mixed_precision='bf16', seed=42, eval_every=0,
        lr_scheduler_type='cosine', max_grad_norm=1.0,
        save_every=1000, validation_every=1000,
        eval_num_inference_steps=16, teacher=spec['teacher'], resume=None)
    if (any(cfg.model.get(key) != value for key, value in expected_model.items())
            or any(cfg.get(key) != value for key, value in expected_train.items())
            or cfg.model.get('_target_') != 'fastwam.cosmos.dense_features.create_dense_feature_joint'
            or cfg.data.train.get('_target_') != 'fastwam.cosmos.jepa_video_cache.JepaVideoCachedDataset'
            or cfg.data.train.get('expected_teacher') != spec['teacher']
            or cfg.data.train.get('split') != 'train'):
        raise ValueError('Configuration differs from the matched RM71D78 28-layer 50k recipe')


def resolve_weights(variant, path):
    from .dense_encoder import sha256_file
    spec = VARIANTS[variant]
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.name != spec['filename']:
        raise FileNotFoundError(f'Pre-staged {spec["filename"]} required; no download fallback')
    if 'size_bytes' in spec and path.stat().st_size != spec['size_bytes']:
        raise ValueError('Teacher checkpoint size differs from pinned asset')
    if sha256_file(path) != spec['weight_sha256']:
        raise ValueError('Teacher checkpoint SHA256 differs from pinned asset')
    return path
