"""Prepare/verify GT-derived reference banks on CPU; never launch training."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare')
    prepare.add_argument('--dataset',type=Path,required=True)
    prepare.add_argument('--output',type=Path,required=True)
    prepare.add_argument('--expected-episodes',type=int,default=78)
    prepare.add_argument('--expected-source-sha')
    prepare.add_argument('--coordinate-frame',default='recorded_controller_tcp_frame')
    prepare.add_argument('--split-json',type=Path)
    prepare.add_argument('--max-stride',type=int,default=12)
    prepare.add_argument('--position-tolerance-m',type=float,default=.005)
    prepare.add_argument('--rotation-tolerance-rad',type=float,default=.05)
    prepare.add_argument('--gripper-tolerance',type=float,default=.05)
    verify=sub.add_parser('verify')
    verify.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    from fastwam.cosmos.reference_bank import prepare_bank,verify_bank
    if a.command=='verify':
        result=verify_bank(a.root)
    else:
        def progress(done,total,points):
            if done==1 or done%10==0 or done==total:
                print(json.dumps(dict(episodes_done=done,total=total,last_reference_points=points)),flush=True)
        options=vars(a).copy();options.pop('command')
        result=prepare_bank(**options,progress=progress)
    print(json.dumps({k:result[k] for k in ('status','identity_sha256','total_source_rows',
                                          'total_retained_anchors','total_reference_points')},indent=2))


if __name__=='__main__':
    main()
