"""Verify existing EXP-019 JPEG pixels; never extract/decode source video."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.order_cache_acceptance import accept_order_cache


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, default=ROOT / 'data/metadata/EXP-019/order-plan-v1')
    ap.add_argument('--out', type=Path, default=ROOT / 'data/metadata/EXP-019/order-cache-acceptance-v1')
    ap.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    args = ap.parse_args()
    result = accept_order_cache(ROOT, args.plan, args.out, args.workers)
    print(json.dumps(dict(status=result['status'], verified=result['verified_required_frame_count'],
        required=result['required_frame_count'], failed=result['failed_required_frames'], seconds=result['seconds'])))


if __name__ == '__main__': main()
