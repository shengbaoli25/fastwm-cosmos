"""Observed RGB for the exact baseline cache rows, with no teacher dependency.

The existing Cosmos numeric/latent cache is the authority for every training
window. JEPA pixels contain only recorded observations. The model constructs its
own stopped-gradient Force target online; this module supplies no target cache.
"""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch

from fastwam.cache.latent import cache_identity_sha256, file_sha256
from .cache import CosmosCachedDataset, _atomic, _path

OBSERVATION_SCHEMA = 'cosmos-jepa-observed-rgb-uint8-v1'
OFFSETS = (-4, 0)


def anchor_rows(identity):
    source = identity['realman_source']
    return [(ep, frame, lo, hi) for ep, lo, hi in source['segments'] for frame in range(lo, hi)]


def observation_indices(frame, lo, hi):
    if not lo <= frame < hi:
        raise ValueError('Observation anchor is outside its continuous segment')
    return [max(lo, frame + offset) for offset in OFFSETS]


def _read_manifest(root, schema, expected):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    digest = cache_identity_sha256(manifest['identity'])
    if (manifest['identity'].get('schema') != schema or manifest.get('status') != 'complete'
            or manifest.get('identity_sha256') != digest or expected != digest):
        raise ValueError('Incomplete or mismatched JEPA cache identity')
    return root, manifest


def observation_identity(source, base_manifest):
    base_identity = base_manifest['identity']
    if (base_identity['realman_source'] != source.identity
            or anchor_rows(base_identity) != [(ep, f, *source.bounds[j])
                                             for j, (ep, f, _, _) in enumerate(source.rows)]):
        raise ValueError('RGB source and exact cached windows differ')
    return dict(schema=OBSERVATION_SCHEMA, base_identity_sha256=base_manifest['identity_sha256'],
                    source_identity_sha256=source.identity_sha256, camera_order=['cam_main', 'cam_wrist'],
                    offsets=list(OFFSETS), boundary='clamp_to_same_continuous_segment',
                    source_decoder='RealManSource.decode_episode', resize=[224, 224],
                    decoder_sha256=file_sha256(Path(__file__).with_name('realman.py')),
                    preparation_sha256=file_sha256(__file__),
                    storage='uint8_round_after_baseline_resize', output_range=[-1, 1],
                    episode_lengths=source.lengths)


def build_observation_cache(source, base_manifest, output, *, on_episode=None, resume=False):
    """Decode each source episode once, preserving the baseline resize operation.

    Pixels are rounded to uint8 *after* the baseline's bilinear 224 resize. The
    additional quantization error is at most 1/255 in normalized [-1,1] RGB.
    Resume only appends episodes after checking all already committed payloads.
    """
    root = Path(output).resolve()
    identity = observation_identity(source, base_manifest)
    digest = cache_identity_sha256(identity)
    if root.exists():
        if not resume:
            raise FileExistsError('Existing observed-RGB cache requires explicit resume')
        manifest = json.loads((root / 'manifest.json').read_text())
        if (manifest['identity'] != identity or manifest['identity_sha256'] != digest
                or manifest['status'] not in ('partial', 'complete')
                or len(manifest['episodes']) > len(source.lengths)):
            raise ValueError('Cannot reuse a different observed-RGB cache')
        for ep, entry in enumerate(manifest['episodes']):
            if (entry['episode_index'] != ep or entry['shape'] != [source.lengths[ep], 2, 3, 224, 224]
                    or file_sha256(_path(root, entry['path'])) != entry['sha256']):
                raise ValueError('Previously committed observed RGB changed')
        if manifest['status'] == 'complete' and len(manifest['episodes']) != len(source.lengths):
            raise ValueError('Incomplete observed RGB inventory')
    else:
        root.mkdir(parents=True)
        (root / 'episodes').mkdir()
        manifest = dict(identity=identity, identity_sha256=digest, status='partial', episodes=[])
    _atomic(root / 'manifest.json', manifest)
    for ep in range(len(manifest['episodes']), len(source.lengths)):
        count = source.lengths[ep]
        frames = source.decode_episode(ep)  # [N,3,224,448], [-1,1]
        if frames.shape != (count, 3, 224, 448) or not torch.isfinite(frames).all():
            raise ValueError('Baseline camera decoder returned invalid frames')
        cameras = torch.stack(frames.split(224, dim=-1), 1)
        pixels = ((cameras.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu().numpy()
        relative = f'episodes/{ep:06d}.npy'
        path = root / relative
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, pixels, allow_pickle=False)
        temporary.replace(path)
        manifest['episodes'].append(dict(episode_index=ep, path=relative,
                                         shape=list(pixels.shape), sha256=file_sha256(path)))
        _atomic(root / 'manifest.json', manifest)
        if on_episode is not None:
            on_episode(ep + 1, len(source.lengths))
    manifest['status'] = 'complete'
    _atomic(root / 'manifest.json', manifest)
    return manifest


class JepaForceTrainingDataset(torch.utils.data.Dataset):
    """Augment exact baseline samples without changing their tensors or masks."""

    def __init__(self, base_config, observation_cache_root, expected_observation_identity):
        config = dict(base_config)
        target = config.pop('_target_', 'fastwam.cosmos.cache.CosmosCachedDataset')
        if target != 'fastwam.cosmos.cache.CosmosCachedDataset':
            raise ValueError('An existing verified CosmosCachedDataset config is required')
        config.pop('_recursive_', None)
        self.base = CosmosCachedDataset(**config)
        self.root, self.identity_sha256 = self.base.root, self.base.identity_sha256
        self.manifest = self.base.manifest
        self.obs_root, self.obs_manifest = _read_manifest(observation_cache_root, OBSERVATION_SCHEMA,
                                                         expected_observation_identity)
        self.observation_identity_sha256 = expected_observation_identity
        self.frames = OrderedDict()
        self.rows = anchor_rows(self.manifest['identity'])
        if self.obs_manifest['identity']['base_identity_sha256'] != self.identity_sha256:
            raise ValueError('JEPA data was prepared for a different baseline cache')
        oi = self.obs_manifest['identity']
        if (oi['offsets'] != list(OFFSETS) or oi['camera_order'] != ['cam_main', 'cam_wrist']
                or oi['output_range'] != [-1, 1]):
            raise ValueError('Observed RGB geometry differs')
        if len(self.rows) != len(self.base):
            raise ValueError('Baseline anchor inventory differs')
        lengths = self.manifest['identity']['episode_lengths']
        if len(self.obs_manifest['episodes']) != len(lengths):
            raise ValueError('Missing RGB episodes')
        for ep, entry in enumerate(self.obs_manifest['episodes']):
            if entry['episode_index'] != ep or entry['shape'] != [lengths[ep], 2, 3, 224, 224]:
                raise ValueError('RGB episode layout differs')
            if file_sha256(_path(self.obs_root, entry['path'])) != entry['sha256']:
                raise ValueError('Observed RGB payload changed')
        self.observation_manifest_sha256 = file_sha256(self.obs_root / 'manifest.json')

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        if any(key in sample for key in ('force_teacher_features', 'force_teacher_valid')):
            raise ValueError('External Force-teacher labels are not part of this online-target experiment')
        ep, frame, lo, hi = self.rows[index]
        if (int(sample['sample_index']), int(sample['episode_index']), int(sample['frame_index'])) != (index, ep, frame):
            raise ValueError('Baseline substituted a training anchor')
        if file_sha256(self.obs_root / 'manifest.json') != self.observation_manifest_sha256:
            raise ValueError('JEPA cache manifest changed during training')
        if ep not in self.frames:
            entry = self.obs_manifest['episodes'][ep]
            value = np.load(_path(self.obs_root, entry['path']), mmap_mode='r', allow_pickle=False)
            if value.dtype != np.uint8 or list(value.shape) != entry['shape']:
                raise ValueError('Invalid observed RGB array')
            self.frames[ep] = value
            if len(self.frames) > 4:
                self.frames.popitem(last=False)
        self.frames.move_to_end(ep)
        frames = np.array(self.frames[ep][observation_indices(frame, lo, hi)], copy=True)
        sample['jepa_observation'] = torch.from_numpy(frames).permute(1, 2, 0, 3, 4).float().div_(127.5).sub_(1).contiguous()
        return sample
