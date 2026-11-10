"""EXP-019 order entry; no cloud rental/start/controller behavior."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.common.video.order_training import MODES, train


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--mode',choices=MODES,required=True)
    parser.add_argument('--expected-initial-state-sha256')
    parser.add_argument('--cache-acceptance',type=Path)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only:
        from experiments.common.video.order_plan import load_order_plan
        plan,bank,epochs,validation,dense,sources=load_order_plan(ROOT,args.plan)
        print(json.dumps(dict(status='metadata_contract_only',clips=len(bank),epochs=len(epochs),
            validation=len(validation),dense_windows=len(dense),pixels_ready=plan['pixels_ready'],training_ready=plan['training_ready'])))
        return
    if args.out is None:parser.error('--out is required')
    if args.cache_acceptance is None:parser.error('--cache-acceptance is required for an actual fit')
    args.out.resolve().relative_to(ROOT/'data/results')
    print(json.dumps(train(ROOT,args.plan,args.out,args.mode,device=args.device,
        expected_initial_state_sha256=args.expected_initial_state_sha256,cache_acceptance=args.cache_acceptance)))


if __name__=='__main__':main()
