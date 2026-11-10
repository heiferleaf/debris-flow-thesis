"""Common EXP-018 A/B/C entry; does not rent/start cloud instances."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.paired_plan import load_pair_plan
from experiments.common.video.paired_training import train
from experiments.common.video.supervised_plan import rooted
from experiments.common.video.supervised_training import load_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, default=ROOT / 'data/metadata/EXP-018/paired-plan-v1')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--mode', choices=['bce', 'cross', 'same'], default='bce')
    parser.add_argument('--seed', type=int, choices=[7], default=7)
    parser.add_argument('--epochs', type=int, choices=[1, 20], default=20)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cache-mib', type=int, default=128)
    parser.add_argument('--microbatch', type=int, choices=[8], default=8)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        plan, epochs = load_pair_plan(ROOT, args.plan)
        _, rows, schedule, val = load_inputs(ROOT, rooted(ROOT, plan['base_plan_path']), check_files=True)
        print(json.dumps(dict(status='paired_input_contract_passed', clips=len(rows),
                              epochs=len(epochs), train_per_epoch=len(schedule[0]), validation=len(val))))
        return
    if args.out is None:
        parser.error('--out is required for an actual run')
    args.out.resolve().relative_to(ROOT / 'data/results')
    print(json.dumps(train(ROOT, args.plan, args.out, args.mode, args.seed, args.epochs,
                           args.device, args.workers, args.cache_mib, args.microbatch, args.smoke)))


if __name__ == '__main__':
    main()
