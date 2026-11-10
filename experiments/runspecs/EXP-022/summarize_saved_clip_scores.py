"""Summarize existing OOF pure-window counts, without model scoring or fitting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'data/results/exp022/error-diagnosis-20261105-v2/all-video-statistics.csv'


def summarize(rows):
    tp = sum(int(r['fully_pos_positive_windows']) for r in rows)
    p = sum(int(r['fully_pos_windows']) for r in rows)
    fp = sum(int(r['fully_neg_positive_windows']) for r in rows)
    n = sum(int(r['fully_neg_windows']) for r in rows)
    ignored = sum(int(r['boundary_or_unannotated_windows']) for r in rows)
    total = sum(int(r['windows']) for r in rows)
    assert p + n + ignored == total
    return dict(videos=len(rows), saved_windows=total,
                pure_P=p, pure_N=n, boundary_or_unannotated=ignored,
                TP=tp, FN=p-tp, FP=fp, TN=n-fp,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/p if p else None,
                FPR=fp/n if n else None,
                F1=2*tp/(p+tp+fp) if p+tp+fp else None)


def build():
    with SOURCE.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 150 and len({r['unit_id'] for r in rows}) == 150
    assert sum(int(r['windows']) for r in rows) == 143470
    return dict(
        scope='Post hoc description of saved EXP-022 outer-fold scores; no new fit, forward pass or threshold tuning.',
        threshold=.5,
        pure_window_policy='First and last sampled PTS wholly inside one frozen half-open P/N annotation interval; uses existing v2 diagnostic counts.',
        evaluation_domain='Existing dense grid already excludes QC plus all manual cut ranges; semantic-transition cut was not scored.',
        all=summarize(rows),
        folds={str(f): summarize([r for r in rows if int(r['fold']) == f]) for f in range(1, 6)},
        video_label_negative_only=summarize([r for r in rows if int(r['label']) == 0]),
        limits=[
            'Pure-window performance conditional on frozen labels and availability domain; not all-time temporal detection performance.',
            'Boundary/unannotated counts are not guaranteed mixed P/N windows or equivalent to excluded cut duration.',
            'Overlapping windows are correlated; pooled counts weight long videos more heavily and are not independent samples.',
            'Window metrics and video-max metrics have different denominators and meanings; differences are not model improvement.',
            'Some outer inputs/results were used in prior development; these are development diagnostics, not untouched independent-test estimates.'
        ],
        sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [SOURCE, Path(__file__).resolve()]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = build()
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
