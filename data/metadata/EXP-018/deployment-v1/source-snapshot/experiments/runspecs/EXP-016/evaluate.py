"""Check dense input pixels or score selected models on complete videos."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.dense_evaluation import evaluate
from experiments.common.video.r2plus1d_classifier import configure_fp32


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', type=Path, default=ROOT / 'data/metadata/EXP-016/training-plan-v2')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ffmpeg', type=Path, required=True)
    ap.add_argument('--rgb', type=Path)
    ap.add_argument('--diff', type=Path)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--pixel-reference', type=Path)
    a = ap.parse_args()
    a.out.resolve().relative_to(ROOT / 'data')
    configure_fp32(7)
    paths = {mode: path for mode, path in [('rgb_zero', a.rgb), ('rgb_diff', a.diff)] if path}
    r = evaluate(ROOT, a.plan, a.out, a.ffmpeg, paths, a.device, a.pixel_reference)
    print(json.dumps({k: v for k, v in r.items() if k not in ('videos', 'scores')}))
