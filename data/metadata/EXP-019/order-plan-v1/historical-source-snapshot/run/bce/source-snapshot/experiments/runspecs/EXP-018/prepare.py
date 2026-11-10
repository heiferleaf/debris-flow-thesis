"""Freeze new EXP-018 metadata without modifying any EXP-016 input."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.paired_plan import freeze_pair_plan, load_pair_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-plan', type=Path, default=ROOT / 'data/metadata/EXP-016/training-plan-v2')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/metadata/EXP-018/paired-plan-v1')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if not args.check_only:
        freeze_pair_plan(ROOT, args.base_plan, args.out)
    plan, epochs = load_pair_plan(ROOT, args.out)
    print(json.dumps(dict(status='paired_plan_verified', epochs=len(epochs), plan=plan), ensure_ascii=False))


if __name__ == '__main__':
    main()
