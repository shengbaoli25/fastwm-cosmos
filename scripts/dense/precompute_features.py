"""Build one immutable teacher-feature cache on exactly four allocated GPUs."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'source-sha', 'teacher', 'teacher-source', 'weights', 'weight-sha', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--rank', type=int, default=int(os.environ.get('LOCAL_RANK', '-1')))
    parser.add_argument('--assemble', action='store_true')
    parser.add_argument('--frame-batch', type=int, default=16)
    args = parser.parse_args()
    import torch
    from fastwam.cosmos.dense_cache import write_rank_cache, assemble_cache
    torch.set_num_threads(4)
    if args.assemble:
        result = assemble_cache(dataset=args.dataset, source_sha=args.source_sha,
            teacher_name=args.teacher, teacher_repo=args.teacher_source, weights=args.weights,
            weight_sha256=args.weight_sha, root=args.output)
        print(json.dumps({'status': result['status'], 'identity_sha256': result['identity_sha256'],
                          'frames': result['completed_frames'], 'shards': len(result['shards'])}), flush=True)
        return
    if args.rank not in range(4) or not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError('Feature encoding requires exactly four visible allocated GPUs')
    torch.cuda.set_device(args.rank)
    receipt = write_rank_cache(dataset=args.dataset, source_sha=args.source_sha,
        teacher_name=args.teacher, teacher_repo=args.teacher_source, weights=args.weights,
        weight_sha256=args.weight_sha, root=args.output, rank=args.rank, frame_batch=args.frame_batch)
    print(json.dumps({'status': receipt['status'], 'rank': args.rank,
                      'episodes': len(receipt['episodes']), 'shards': len(receipt['shards'])}), flush=True)


if __name__ == '__main__':
    main()
