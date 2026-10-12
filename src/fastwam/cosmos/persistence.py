"""Actual-device identity and full-state fingerprints for fresh-rank restore."""
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
import torch


def resource_identity():
    allocation = os.environ.get('COSMOS_ALLOCATION_ID')
    if not allocation:
        raise ValueError('Set COSMOS_ALLOCATION_ID to the observed platform allocation UUID')
    query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.total',
                            '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    rows = [x.strip() for x in query.stdout.splitlines() if x.strip()]
    if len(rows) < 4:
        raise RuntimeError('At least four actual GPUs required')
    return {'allocation': allocation, 'hostname': socket.gethostname(),
            'gpu_inventory': rows, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


def state_digest(value):
    digest = hashlib.sha256()
    def add(x):
        if isinstance(x, torch.Tensor):
            digest.update(str((str(x.dtype), tuple(x.shape))).encode())
            flat = x.detach().reshape(-1)
            for start in range(0, flat.numel(), 1048576):
                digest.update(flat[start:start+1048576].contiguous().cpu().view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                add(str(k)); add(x[k])
        elif isinstance(x, (list, tuple)):
            digest.update(str(len(x)).encode())
            for y in x: add(y)
        else:
            digest.update(json.dumps(x, sort_keys=True, allow_nan=False).encode())
    add(value)
    return digest.hexdigest()


def training_state_fingerprints(trainer, model):
    from .preflight import helpers
    h = helpers()
    return {'model': state_digest(model.state_dict()),
            'optimizer': state_digest(h.unwrap_optimizer(trainer.optimizer).state_dict()),
            'scheduler': state_digest(h.inspect_scheduler(trainer.scheduler)),
            'memory_rng': state_digest(model.dit['runtime'].get_state()),
            'cpu_rng': state_digest(torch.get_rng_state()),
            'cuda_rng': state_digest(torch.cuda.get_rng_state()),
            'trainer': {'global_step': trainer.global_step, 'epoch': trainer.epoch,
                        'batch_in_epoch': trainer.batch_in_epoch}}


def verify_restore(config, receipt, outer, preflight_sha256, resource):
    from .binding import binding, verify_outer
    from fastwam.cache.latent import file_sha256
    r = json.loads(Path(receipt).read_text())
    verify_outer(json.loads(Path(outer).read_text()))
    if (r.get('status') != 'fresh_four_rank_restore_verified' or r.get('binding') != binding(config)
            or r.get('preflight_sha256') != preflight_sha256 or r.get('resource') != resource):
        raise ValueError('Current fresh four-rank persistence verification required')
    ranks = r.get('ranks', [])
    if len(ranks) != 4 or {x['rank'] for x in ranks} != {0,1,2,3} or any(not x.get('accepted') for x in ranks):
        raise ValueError('Incomplete fresh-rank restore evidence')
    return {'receipt_sha256': file_sha256(receipt), 'outer_sha256': file_sha256(outer)}
