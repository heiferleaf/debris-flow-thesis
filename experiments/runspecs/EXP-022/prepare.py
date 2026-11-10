"""Freeze small EXP022 role/reuse/source metadata; no model construction or pixel scan."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.linear_cv_execution import prepare
from experiments.common.video.cv_supervised_training import _sha

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'data/metadata/EXP-022/linear-plan-v1')
    args = parser.parse_args()
    plan = prepare(ROOT, args.out)
    print(dict(plan_sha256=_sha(args.out / 'plan.json'), new_inner_folds=plan['new_inner_folds'],
               max_new_epochs=plan['maximum_new_epochs'], max_new_updates=plan['maximum_new_updates']))
