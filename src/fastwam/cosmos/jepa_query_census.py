"""Streaming, evaluation-only attention census helpers; no training mutation."""
from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile

import numpy as np
import torch


class MeanAttentionCapture(AbstractContextManager):
    """Accumulate true reader QK softmax on device, retaining every head/query.

    Only the denoising-step axis is averaged. Does not retain 16 full CPU copies
    or alter module outputs. The last dimension is two cameras ×14×14 patches.
    """
    def __init__(self, reader, expected_steps=16):
        self.reader = reader
        self.expected_steps = expected_steps
        self.handles = []
        self.projected = {}
        self.sums = {}
        self.counts = {}

    def __enter__(self):
        if self.reader.training or self.handles:
            raise ValueError('An inactive evaluation reader is required')
        for index, layer in enumerate(self.reader.readers):
            def q_hook(module, args, output, index=index):
                self.projected[index] = output.detach()
            def reader_hook(module, args, output, index=index):
                _, (key, _), valid = args
                projected = self.projected.pop(index)
                query = projected.unflatten(-1, (module.heads, projected.shape[-1] // module.heads)).transpose(1, 2)
                if index not in self.sums and not bool(valid.all()):
                    raise ValueError('Census requires every observed patch valid')
                with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
                    logits = query.float() @ key.detach().float().transpose(-2, -1)
                    attention = (logits / math.sqrt(query.shape[-1])).masked_fill(~valid[:, None, None], -torch.inf).softmax(-1)
                if index not in self.sums:
                    self.sums[index] = attention
                    self.counts[index] = 1
                else:
                    self.sums[index].add_(attention)
                    self.counts[index] += 1
            self.handles.append(layer.query.register_forward_hook(q_hook))
            self.handles.append(layer.register_forward_hook(reader_hook))
        return self

    def mean(self):
        expected = set(range(len(self.reader.readers)))
        if set(self.sums) != expected or any(n != self.expected_steps for n in self.counts.values()):
            raise ValueError('Incomplete denoising-step attention census')
        result = torch.stack([self.sums[index] / self.counts[index] for index in sorted(expected)], dim=1)
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Nonfinite captured attention')
        return result.detach().cpu().numpy()

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.projected.clear()
        return False


def observed_batch(samples):
    """Only past/current observations; future proprio and all targets excluded."""
    keys = ('first_frame_latents', 'history', 'history_is_pad', 'history_channel_valid',
            'cosmos_text_index', 'jepa_observation')
    batch = {key: torch.stack([sample[key] for sample in samples]) for key in keys}
    batch['proprio'] = torch.stack([sample['proprio'][0] for sample in samples])
    if batch['proprio'].shape != (len(samples), 9):
        raise ValueError('Current-only Joint8 proprio required')
    return batch


@torch.no_grad()
def infer_repeated_seed(model, sample, *, steps=16, seed=113):
    """Original joint sampler with identical per-example noise irrespective of B.

    Original infer_joint draws a single [B,...] RNG stream, changing noise by batch
    position. This adapter repeats the exact [1,...] draws of seed/seed+1/seed+2.
    Conditions, static cache, clocks, 16 denoising cores and updates are unchanged.
    B1 must agree with original infer_joint before any full census is allowed.
    """
    if model.training or steps < 1:
        raise ValueError('Evaluation mode and positive inference steps required')
    first = sample['first_frame_latents'].to(model.device, model.torch_dtype)
    conditions = model._conditions(sample)
    def noise(shape, noise_seed):
        one = torch.randn((1, *shape[1:]), generator=torch.Generator(device='cpu').manual_seed(noise_seed))
        return one.expand(shape).clone().to(model.device, model.torch_dtype)
    video = noise((*first.shape[:2], 3, *first.shape[3:]), seed + 2)
    video[:, :, :1] = first
    force = noise((len(first), 32, 6), seed + 1)
    action = noise((len(first), 32, model.action_dim), seed)
    cache = model.prefill_static(first, conditions)
    schedules = [scheduler.build_inference_schedule(steps, model.device, model.torch_dtype)
                 for scheduler in (model.v_scheduler, model.f_scheduler, model.a_scheduler)]
    for index in range(steps):
        tv, tf, ta = (pair[0][index].expand(len(first)) for pair in schedules)
        pv, pf, pa = model.joint_denoise_core(video, force, action, tv, tf, ta, conditions, cache)
        video = model.v_scheduler.step(pv, schedules[0][1][index], video)
        video[:, :, :1] = first
        force = model.f_scheduler.step(pf, schedules[1][1][index], force)
        action = model.a_scheduler.step(pa, schedules[2][1][index], action)
    return dict(action=action, force=force, video_latents=video)


def attention_metrics(weights):
    """Descriptive concentration/redundancy, never semantic contact quality."""
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != (2, 4, 8, 392) or not np.isfinite(weights).all():
        raise ValueError('Expected finite [2,4,8,392] step-mean reader attention')
    entropy = -(weights * np.log(np.maximum(weights, 1e-30))).sum(-1) / np.log(392)
    peak = weights.max(-1)
    main_mass = weights[..., :196].sum(-1)
    normed = weights / np.maximum(np.linalg.norm(weights, axis=-1, keepdims=True), 1e-30)
    cosine = normed @ np.swapaxes(normed, -1, -2)
    pairwise = cosine[..., np.triu_indices(8, 1)[0], np.triu_indices(8, 1)[1]].mean(-1)
    return dict(entropy_normalized=float(entropy.mean()), peak_probability=float(peak.mean()),
        query_cosine_similarity=float(pairwise.mean()), camera_main_mass=float(main_mass.mean()),
        entropy_by_layer_head_query=entropy.tolist(), peak_probability_by_layer_head_query=peak.tolist(),
        peak_token_by_layer_head_query=weights.argmax(-1).tolist(),
        camera_main_mass_by_layer_head_query=main_mass.tolist(),
        query_cosine_by_layer_head=pairwise.tolist())


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def write_shard(directory, start, rows, attention, metrics, thumbnails, action, force, run_identity):
    """Atomic archive publication. Incomplete .partial files are never receipts."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stop = start + len(rows)
    if [row['sample_index'] for row in rows] != list(range(start, stop)):
        raise ValueError('A shard must contain exactly one ordered contiguous index range')
    basename = f'shard_{start:06d}_{stop:06d}'
    path = directory / f'{basename}.tar.gz'
    if path.exists():
        raise FileExistsError(f'Committed shard already exists: {path}')
    attention = np.asarray(attention, dtype=np.float16)
    if attention.shape != (len(rows), 2, 4, 8, 392):
        raise ValueError('Attention shard geometry differs')
    action, force = np.asarray(action, dtype=np.float32), np.asarray(force, dtype=np.float32)
    if action.shape != (len(rows), 32, 8) or force.shape != (len(rows), 32, 6):
        raise ValueError('Prediction shard geometry differs')
    if any(not np.isfinite(value).all() for value in (attention, action, force)):
        raise ValueError('Nonfinite shard arrays')
    members = dict(thumbnails)
    members['attention.npy'] = npy_bytes(attention)
    members['action_predictions.npy'] = npy_bytes(np.asarray(action, dtype=np.float32))
    members['force_predictions.npy'] = npy_bytes(np.asarray(force, dtype=np.float32))
    members['samples.jsonl'] = ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows).encode()
    members['metrics.jsonl'] = ''.join(json.dumps(row, allow_nan=False) + '\n' for row in metrics).encode()
    manifest = dict(schema='jepa-query-census-shard-v1', start=start, stop=stop, count=len(rows),
        sample_ids=list(range(start, stop)), run_identity=run_identity,
        array=dict(name='attention.npy', shape=list(attention.shape), dtype='float16',
                   aggregation='mean_over_16_denoising_steps'),
        prediction_arrays={'action_predictions.npy': [len(rows), 32, 8], 'force_predictions.npy': [len(rows), 32, 6]},
        camera_order=['cam_main', 'cam_wrist'], thumbnails='current anchor RGB 224x224 JPEG; no frame-specific attribution',
        members={name: dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest()) for name, content in members.items()})
    members['shard-manifest.json'] = json.dumps(manifest, allow_nan=False).encode()
    temporary = path.with_suffix(path.suffix + '.partial')
    with tarfile.open(temporary, 'w:gz', compresslevel=3) as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(content))
    temporary.replace(path)
    receipt = dict(schema='jepa-query-census-shard-receipt-v1', start=start, stop=stop, count=len(rows),
        path=path.name, sha256=digest_file(path), bytes=path.stat().st_size, run_identity=run_identity)
    receipt_path = directory / f'{basename}.json'
    temp_receipt = receipt_path.with_suffix('.tmp')
    temp_receipt.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    temp_receipt.replace(receipt_path)
    return receipt


def verify_shard(path, start, stop, run_identity):
    """Verify every committed member plus lineage, including crash-before-receipt."""
    path = Path(path)
    with tarfile.open(path, 'r:gz') as archive:
        manifest = json.loads(archive.extractfile('shard-manifest.json').read())
        if (manifest.get('schema') != 'jepa-query-census-shard-v1'
                or manifest['start'] != start or manifest['stop'] != stop
                or manifest['sample_ids'] != list(range(start, stop)) or manifest['run_identity'] != run_identity):
            raise ValueError('Existing shard lineage or range differs')
        if set(archive.getnames()) != set(manifest['members']) | {'shard-manifest.json'}:
            raise ValueError('Unexpected or missing archive member')
        for name, expected in manifest['members'].items():
            content = archive.extractfile(name).read()
            if len(content) != expected['bytes'] or hashlib.sha256(content).hexdigest() != expected['sha256']:
                raise ValueError(f'Shard member changed: {name}')
    return dict(schema='jepa-query-census-shard-receipt-v1', start=start, stop=stop, count=stop-start, path=path.name,
                bytes=path.stat().st_size, sha256=digest_file(path), run_identity=run_identity)
