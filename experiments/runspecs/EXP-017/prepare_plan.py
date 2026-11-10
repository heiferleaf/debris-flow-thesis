"""EXP-017 metadata only: no raw labels, media decoding, model or paid compute."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.cv_supervised_plan import freeze_cv_plan, load_cv_plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'data/metadata/EXP-017/role-free-plan-v1')
    ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args()
    if a.check_only:
        plan, bank, epochs, fixed, folds = load_cv_plan(ROOT, a.out)
        print(json.dumps(dict(status=plan['status'], clips=len(bank), epochs=len(epochs),
            fixed_validation=len(fixed['clip_ids']), folds=len(folds), pixels_ready=False)))
    else:
        plan, audit = freeze_cv_plan(ROOT, a.out)
        print(json.dumps(dict(status=plan['status'], seconds=plan['seconds'], audit=audit), ensure_ascii=False))


if __name__ == '__main__':
    main()
