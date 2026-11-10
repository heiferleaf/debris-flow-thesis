"""Prepare or execute one bounded read-only EXP-017 CPU training-panel replay."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.training_replay_diagnostic import prepare, replay, worker, DEFAULT_AUDIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    prep = commands.add_parser('prepare', help='Metadata only; no model forward or JPEG decoding')
    prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--audit', default=DEFAULT_AUDIT)
    run = commands.add_parser('replay', help='One CPU run, hard 900-second cap; refuses an earlier launch')
    run.add_argument('--prepared', type=Path, required=True)
    run.add_argument('--manifest-sha256', required=True, help='Manifest SHA frozen by independent review')
    private = commands.add_parser('_worker', help=argparse.SUPPRESS)
    private.add_argument('--prepared', type=Path, required=True)
    private.add_argument('--nonce', required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(ROOT, args.out, args.audit)
        print(json.dumps(dict(status=result['status'], folds=[dict(fold=f['fold'],
            windows=len(f['windows']), complete=f['panel_quota_complete'], coverage=f['coverage'])
            for f in result['folds']]), ensure_ascii=False))
    elif args.action == 'replay':
        result = replay(ROOT, args.prepared, Path(__file__).resolve(),
                        expected_manifest_sha256=args.manifest_sha256)
        print(json.dumps(dict(status=result['status'], routing=result['routing'], seconds=result['seconds'])))
        return 0 if result['status'] == 'complete' else 1
    else:
        worker(ROOT, args.prepared.resolve(), args.nonce)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
