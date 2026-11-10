"""Freeze shared video/class quotas; reuse verified JPEGs without decoding."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.supervised_plan import freeze_plan


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--resource', type=Path, default=ROOT / 'data/metadata/EXP-016/metadata-v4')
    ap.add_argument('--out', type=Path, default=ROOT / 'data/metadata/EXP-016/training-plan-v2')
    args = ap.parse_args()
    print(json.dumps(freeze_plan(ROOT, args.resource, args.out), ensure_ascii=False))
