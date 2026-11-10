"""Metadata-only EXP-019 import; actual JPEG pixel acceptance is a separate stage."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.order_plan import freeze_order_plan, load_order_plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'data/metadata/EXP-019/order-plan-v1')
    ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args()
    if a.check_only:
        plan, bank, epochs, validation, dense, sources = load_order_plan(ROOT, a.out)
        print(json.dumps(dict(status=plan['status'], supervised_clips=len(bank), epochs=len(epochs),
            fixed_validation=len(validation), dense_windows=len(dense), sources=len(sources), pixels_ready=False)))
    else:
        plan, audit = freeze_order_plan(ROOT, a.out)
        print(json.dumps(dict(status=plan['status'], seconds=plan['seconds'], audit=audit)))


if __name__ == '__main__': main()
