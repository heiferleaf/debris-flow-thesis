"""Freeze one EXP020 head comparison plan and CPU-only initialization fingerprints."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from plan import prepare
from experiments.common.video.cv_supervised_training import _sha

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'data/metadata/EXP-020/head-plan-v1')
    p.add_argument('--initialization-proof', type=Path, required=True)
    a = p.parse_args()
    plan = prepare(ROOT, a.out, a.initialization_proof)
    print(dict(plan_sha256=_sha(a.out / 'plan.json'), initializations=plan['initializations'], new_fits=len(plan['fit_order'])))
