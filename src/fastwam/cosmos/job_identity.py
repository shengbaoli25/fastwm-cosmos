"""Platform identity for a single-node, four-GPU distributed Cosmos job."""
import re


def distributed_identity(env, hostname, gpu_rows):
    job = env.get('TRAIN_JOB_ID', '')
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{3,127}', job)
            or env.get('PET_NNODES', '1') != '1'
            or env.get('PET_NODE_RANK', '0') != '0'):
        raise ValueError('Actual single-node TRAIN_JOB_ID required')
    if env.get('COSMOS_ALLOCATION_ID'):
        raise ValueError('Distributed job must not impersonate a notebook allocation')
    parsed = [row.split(',') for row in gpu_rows]
    if (len(parsed) != 4 or any(len(row) != 3 for row in parsed)
            or len({row[0].strip() for row in parsed}) != 4
            or any(not row[0].strip().startswith('GPU-') or 'H200' not in row[1]
                   or int(row[2].strip()) < 140000 for row in parsed)):
        raise ValueError('Exactly four allocated H200 GPUs required')
    if env.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3':
        raise ValueError('The four allocated devices must be explicitly selected')
    return {'execution_kind': 'distributed_training', 'job': job,
            'hostname': hostname, 'gpu_inventory': gpu_rows,
            'cuda_visible_devices': env['CUDA_VISIBLE_DEVICES']}
