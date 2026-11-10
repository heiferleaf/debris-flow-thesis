"""Local small EXP021 plan; reuses actual prior acceptance and fixed kernels."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.oriented_energy_plan import prepare
from experiments.common.video.cv_supervised_training import _sha

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'data/metadata/EXP-021/energy-plan-v1')
    args = parser.parse_args()
    result = prepare(ROOT, args.out)
    print(dict(plan_sha256=_sha(args.out / 'plan.json'), fit_order=result['fit_order'], unique_role_videos=result['unique_role_videos']))
