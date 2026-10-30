"""Prepare pinned V-JEPA/DINO window features offline on CPU or allocated GPUs."""
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
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--rank', type=int, default=int(os.environ.get('RANK', '0')))
    parser.add_argument('--world-size', type=int, default=int(os.environ.get('WORLD_SIZE', '1')))
    parser.add_argument('--local-rank', type=int, default=int(os.environ.get('LOCAL_RANK', '0')))
    parser.add_argument('--frame-batch', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--assemble', action='store_true', help='Validate completed rank shards without building an encoder')
    args = parser.parse_args()
    if args.world_size < 1 or args.rank not in range(args.world_size) or args.threads < 1:
        parser.error('positive world size/threads and rank in [0, world-size) required')
    import torch
    from fastwam.cosmos.jepa_video_cache import write_rank_cache, assemble_cache
    torch.set_num_threads(args.threads)
    common = dict(dataset=args.dataset, source_sha=args.source_sha,
                  teacher_name=args.teacher, teacher_repo=args.teacher_source,
                  weights=args.weights, weight_sha256=args.weight_sha,
                  root=args.output, world_size=args.world_size)
    if args.assemble:
        result = assemble_cache(**common)
        print(json.dumps(dict(status=result['status'], identity_sha256=result['identity_sha256'],
                              windows=result['completed_windows'], shards=len(result['shards']))), flush=True)
        return
    if args.device == 'cuda':
        if not torch.cuda.is_available() or args.local_rank not in range(torch.cuda.device_count()):
            raise RuntimeError('Requested CUDA device is unavailable; choose an allocated GPU or --device cpu')
        torch.cuda.set_device(args.local_rank)
    result = write_rank_cache(**common, rank=args.rank, frame_batch=args.frame_batch,
                              device=args.device, resume=args.resume)
    print(json.dumps(dict(status=result['status'], rank=args.rank,
                          identity_sha256=result['identity_sha256'],
                          episodes=len(result['episodes']), shards=len(result['shards']))), flush=True)


if __name__ == '__main__':
    main()
