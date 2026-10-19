"""Bind a trained checkpoint to its exact normalization/preprocessing file."""
import hashlib
import json
from pathlib import Path

from realman_force.contract import validate_contract_identity


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binding(checkpoint, normalization):
    metadata = json.loads(Path(normalization).read_text())
    contract = validate_contract_identity(metadata['force_preprocessing'])
    return dict(schema='realman_checkpoint_preprocessing_v1',
                checkpoint_sha256=file_sha256(checkpoint),
                normalization_sha256=file_sha256(normalization),
                force_preprocessing_id=contract['id'])


def verify_binding(checkpoint, normalization):
    path = Path(str(checkpoint)+'.force.json')
    if not path.is_file() or json.loads(path.read_text()) != binding(checkpoint, normalization):
        raise ValueError('Checkpoint is not bound to this force preprocessing/normalization; export the matching training artifacts')
