"""Independently read and verify all191812 required existing/new JPEG pixels."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.cv_cache_preparation import accept_cache, record_cli_failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'data/metadata/EXP-017/role-free-plan-v3')
    parser.add_argument('--producer', type=Path, default=ROOT/'data/process/EXP-017/cv-inputs-v1')
    parser.add_argument('--out', type=Path, default=ROOT/'data/metadata/EXP-017/cv-cache-acceptance-v1')
    parser.add_argument('--workers', type=int, default=4)
    a = parser.parse_args()
    try: result = accept_cache(ROOT, a.plan, a.producer, a.out, a.workers)
    except Exception as error:
        record_cli_failure(ROOT, a.out, 'actual_JPEG_acceptance', error)
        raise
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
