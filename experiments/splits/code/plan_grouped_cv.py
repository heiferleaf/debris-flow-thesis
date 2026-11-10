"""Data-only window-balanced outer folds and grouped inner holdouts.

Produces a reviewable proposal, never changes historical folds or trains.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random
import sys

SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[3]


def aggregate(rows):
    return (sum(int(r['pos_quota']) for r in rows), sum(int(r['neg_quota']) for r in rows),
            sum(r['label'] == 'pos' for r in rows), sum(r['label'] == 'neg' for r in rows))


def counts(rows):
    wp, wn, p, n = aggregate(rows)
    return dict(videos=len(rows), P=p, N=n, pos_windows=wp, neg_windows=wn,
                groups=len({r['session_group'] for r in rows}),
                negative_groups=len({r['session_group'] for r in rows if r['label'] == 'neg'}))


def propose(rows, n_folds=5, seed=42, restarts=500):
    if len({r['unit_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate video identity')
    by_group = defaultdict(list)
    for row in rows:
        if row['label'] not in ('pos', 'neg'):
            raise ValueError('Unknown video label')
        by_group[row['session_group']].append(row)
    vectors = {g: aggregate(v) for g, v in by_group.items()}
    totals = aggregate(rows)
    target = [v/n_folds for v in totals]
    if min(target) <= 0:
        raise ValueError('Both video/window classes required')
    negative = {g for g, v in vectors.items() if v[3]}
    rng, winner = random.Random(seed), None
    for attempt in range(restarts):
        priority = {g: (vectors[g][0]/target[0]+vectors[g][1]/target[1])
                    * (1 if attempt == 0 else rng.uniform(.8, 1.2)) for g in sorted(vectors)}
        order = sorted(vectors, key=lambda g: (-priority[g], g))
        sums, assignment = [[0]*4 for _ in range(n_folds)], {}
        for group in order:
            vector = vectors[group]
            def delta(f):
                return sum(((sums[f][j]+vector[j]-target[j])/target[j])**2
                           - ((sums[f][j]-target[j])/target[j])**2 for j in (0, 1))
            f = min(range(n_folds), key=lambda f: (delta(f), f))
            assignment[group] = f
            sums[f] = [a+b for a, b in zip(sums[f], vector)]
        if (any(min(v) <= 0 for v in sums)
                or any(sum(assignment[g] == f for g in negative) not in (1, 2, 3) for f in range(n_folds))):
            continue
        primary = sum(((v[j]-target[j])/target[j])**2 for v in sums for j in (0, 1))
        secondary = sum(((v[j]-target[j])/target[j])**2 for v in sums for j in (2, 3))
        key = primary, secondary, tuple(sorted(assignment.items()))
        if winner is None or key < winner[0]:
            winner = key, assignment
    if winner is None:
        raise ValueError('No outer proposal meets class/group constraints')
    assignment = winner[1]
    folds = []
    for f in range(n_folds):
        outer_groups = sorted(g for g in vectors if assignment[g] == f)
        train_groups = sorted(set(vectors)-set(outer_groups))
        train_negative = negative.intersection(train_groups)
        full_vector = tuple(sum(vectors[g][j] for g in train_groups) for j in range(4))
        inner_target = [v*.2 for v in full_vector]
        target_groups = max(2, round(len(train_groups)*.2))
        best = None
        for size in range(max(2, target_groups-1), min(len(train_groups), target_groups+1)+1):
            for subset in combinations(train_groups, size):
                n_groups = len(train_negative.intersection(subset))
                if n_groups < 2 or len(train_negative)-n_groups < 4:
                    continue
                vector = tuple(sum(vectors[g][j] for g in subset) for j in range(4))
                fraction = (vector[0]+vector[1])/(full_vector[0]+full_vector[1])
                if min(vector[2:]) < 4 or not .15 <= fraction <= .3:
                    continue
                key = (sum(((vector[j]-inner_target[j])/inner_target[j])**2 for j in (0, 1)),
                       sum(((vector[j]-inner_target[j])/inner_target[j])**2 for j in (2, 3)), subset)
                if best is None or key < best[0]:
                    best = key, subset
        if best is None:
            raise ValueError(f'No grouped inner holdout for outer fold {f+1}; proposal withheld')
        inner_groups = set(best[1])
        memberships = {name: [r for r in rows if r['session_group'] in groups] for name, groups in
                       [('outer_evaluation', set(outer_groups)), ('outer_refit_train', set(train_groups)),
                        ('inner_validation', inner_groups), ('inner_train', set(train_groups)-inner_groups)]}
        fold = dict(fold=f+1)
        for name, subset in memberships.items():
            fold[name] = dict(video_ids=sorted(r['unit_id'] for r in subset),
                              group_ids=sorted({r['session_group'] for r in subset}), counts=counts(subset))
        folds.append(fold)
    return dict(status='proposed_not_training_authorized', n_folds=n_folds, split_seed=seed,
                search_restarts=restarts, primary_objective='squared relative deviations of pos/neg supervised window quotas',
                inner_rule='whole groups near 20% of outer-train windows; 15-30% bounds; >=2 negative validation groups and >=4 negative training groups; >=4 videos each class in inner validation',
                inputs_used='identities, groups and label/window counts only; no predictions or model metrics',
                total=counts(rows), folds=folds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resources', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    source, out = a.resources.resolve(), a.out.resolve()
    out.relative_to(ROOT/'data/metadata/splits')
    if out.exists():
        raise FileExistsError(out)
    with source.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    result = propose(rows)
    result.update(generator_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  source=source.relative_to(ROOT).as_posix(), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  command=sys.argv)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], folds=[dict(fold=f['fold'], counts={k:v['counts'] for k,v in f.items() if k!='fold'}) for f in result['folds']]), ensure_ascii=True))


if __name__ == '__main__':
    main()
