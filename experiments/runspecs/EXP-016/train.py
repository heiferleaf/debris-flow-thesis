"""Registered EXP-016 supervised run; never silently rent or start an instance."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.supervised_training import load_inputs, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, default=ROOT / 'data/metadata/EXP-016/training-plan-v2')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--mode', choices=['rgb_zero', 'rgb_diff'], default='rgb_zero')
    ap.add_argument('--seed', type=int, choices=[7, 8], default=7)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--cache-mib', type=int, default=128)
    ap.add_argument('--microbatch', type=int, default=8)
    ap.add_argument('--checkpoint', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args()
    if a.check_only:
        p, rows, schedule, val = load_inputs(ROOT, a.plan, check_files=True)
        print(json.dumps(dict(status='local_input_contract_passed', clips=len(rows),
                              train_per_epoch=len(schedule[0]), validation=len(val), plan=p)))
        return
    if a.out is None:
        ap.error('--out is required for an actual run')
    a.out.resolve().relative_to(ROOT / 'data/results')
    print(json.dumps(train(ROOT, a.plan, a.out, a.mode, a.seed, a.epochs, a.device,
                           a.workers, a.cache_mib, a.microbatch, a.checkpoint, a.smoke)))


if __name__ == '__main__':
    main()
