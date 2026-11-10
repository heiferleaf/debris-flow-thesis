"""EXP-024: fixed two-window confirmation of closed EXP-021 Linear A scores.

Standard library only. No models, media, frames, external inputs, or new tuning.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'data/results/exp021/energy-s7-20261103-v2'
PLAN = ROOT / 'data/metadata/EXP-021/energy-plan-v2/plan.json'
DEFAULT_OUT = ROOT / 'data/results/exp024/confirmation-s7-dev-v1'
STEP = 1.5
TOLERANCE = 1e-6
THRESHOLD = .5
EXPECTED_CONFUSION = {1: dict(TP=18, FP=5, FN=0, TN=0), 4: dict(TP=18, FP=2, FN=0, TN=4)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def tuple_digest(indices):
    """Exactly the original numpy dtype='<i4' 16-frame tuple byte convention."""
    require(len(indices) == 16 and all(type(i) is int for i in indices), 'Require 16 original integer frame indices')
    return hashlib.sha256(struct.pack('<16i', *indices)).hexdigest()


def clean_run_index(indices, runs):
    matches = [i for i, (start, stop) in enumerate(runs) if all(start <= frame < stop for frame in indices)]
    require(len(matches) <= 1, 'Overlapping clean runs')
    return matches[0] if matches else None


def describe(windows):
    """One video's mapped saved windows. No legal pair means unknown B, never 0."""
    rows = sorted(windows, key=lambda row: row['target_start_seconds'])
    require(rows, 'Empty video')
    require(all(math.isfinite(row[key]) for row in rows for key in ('target_start_seconds', 'start', 'end', 'score')), 'Nonfinite window')
    require(all(0 <= row['score'] <= 1 and row['start'] <= row['end'] for row in rows), 'Invalid saved window')
    require(all(a['target_start_seconds'] < b['target_start_seconds'] for a, b in zip(rows, rows[1:])), 'Repeated window target')
    excluded = Counter(quality_or_cut=0, short_tail_interval=0, missing_window_gap=0, repeated_physical_tuple=0)
    pairs = []
    for left, right in zip(rows, rows[1:]):
        delta = right['target_start_seconds'] - left['target_start_seconds']
        if left['clean_run_index'] is None or right['clean_run_index'] is None or left['clean_run_index'] != right['clean_run_index']:
            excluded['quality_or_cut'] += 1
        elif left['frame_tuple_sha256'] == right['frame_tuple_sha256']:
            excluded['repeated_physical_tuple'] += 1
        elif delta < STEP - TOLERANCE:
            excluded['short_tail_interval'] += 1
        elif delta > STEP + TOLERANCE:
            excluded['missing_window_gap'] += 1
        else:
            pairs.append((left, right))
    a_score = max(row['score'] for row in rows)
    b_score = max((min(a['score'], b['score']) for a, b in pairs), default=None)
    a_trigger = min((row['end'] for row in rows if row['score'] >= THRESHOLD), default=None)
    b_trigger = min((b['end'] for a, b in pairs if min(a['score'], b['score']) >= THRESHOLD), default=None)
    a_pred = int(a_score >= THRESHOLD)
    b_pred = None if b_score is None else int(b_score >= THRESHOLD)
    require(b_score is None or b_score <= a_score, 'B score exceeds A maximum')
    require(b_pred != 1 or a_pred == 1, 'B positive must be A positive')
    wait = None if a_trigger is None or b_trigger is None else b_trigger - a_trigger
    require(wait is None or wait >= -TOLERANCE, 'Confirmation precedes earliest A output')
    return dict(windows=len(rows), A_score=a_score, B_score=b_score, A_prediction=a_pred, B_prediction=b_pred,
        A_first_trigger_end_seconds=a_trigger, B_first_trigger_end_seconds=b_trigger,
        relative_output_wait_seconds=wait, legal_pairs=len(pairs), covered=bool(pairs),
        quality_excluded_windows=sum(row['clean_run_index'] is None for row in rows),
        adjacent_candidate_pairs=max(0, len(rows) - 1),
        **{'excluded_pairs_' + key: value for key, value in excluded.items()})


def confusion(rows, arm):
    counts = Counter()
    unknown = []
    for row in rows:
        pred = row[arm + '_prediction']
        if pred is None:
            unknown.append(row['unit_id'])
        else:
            counts[('TP' if pred else 'FN') if row['label'] else ('FP' if pred else 'TN')] += 1
    known = {key: counts[key] for key in ('TP', 'FP', 'FN', 'TN')}
    return dict(status='incomplete_coverage' if unknown else 'complete',
        **{key: None if unknown else value for key, value in known.items()},
        known_only_counts=known, unknown_videos=unknown, videos=len(rows))


def summarize(rows, sources):
    by_uid = defaultdict(list)
    for row in rows:
        by_uid[row['unit_id']].append(row)
    negatives = sorted(uid for uid in by_uid if sources[uid]['video_label'] == 'neg')
    groups = sorted({sources[uid]['group_id'] for uid in negatives})
    require(len(by_uid) == 42 and len(negatives) == 11 and len(groups) == 4, 'Expected 42 unique videos, 11 negatives, four negative groups')
    role_means = {arm: {} for arm in ('A', 'B')}
    for arm in role_means:
        for uid in negatives:
            values = [row[arm + '_prediction'] for row in by_uid[uid]]
            role_means[arm][uid] = None if None in values else math.fsum(values) / len(values)
    group_rows = []
    for group in groups:
        ids = [uid for uid in negatives if sources[uid]['group_id'] == group]
        item = dict(group_id=group, unique_negative_videos=len(ids))
        for arm in role_means:
            values = [role_means[arm][uid] for uid in ids]
            item[arm + '_FPR'] = None if None in values else math.fsum(values) / len(values)
        item['FPR_change_B_minus_A'] = None if item['B_FPR'] is None else item['B_FPR'] - item['A_FPR']
        group_rows.append(item)
    macro = {arm: None if any(row[arm + '_FPR'] is None for row in group_rows)
        else math.fsum(row[arm + '_FPR'] for row in group_rows) / len(group_rows) for arm in ('A', 'B')}
    baseline_fp = [uid for uid in negatives if any(row['A_prediction'] == 1 for row in by_uid[uid])]
    repaired = [uid for uid in baseline_fp if all(row['B_prediction'] == 0 for row in by_uid[uid])]
    remaining = [uid for uid in negatives if any(row['B_prediction'] == 1 for row in by_uid[uid])]
    uncovered = [dict(fold=row['fold'], unit_id=row['unit_id']) for row in rows if not row['covered']]
    new_fn = [dict(fold=row['fold'], unit_id=row['unit_id']) for row in rows if row['label'] == 1 and row['A_prediction'] == 1 and row['B_prediction'] == 0]
    new_errors = [dict(fold=row['fold'], unit_id=row['unit_id']) for row in rows
        if row['A_prediction'] == row['label'] and row['B_prediction'] is not None and row['B_prediction'] != row['label']]
    improved = [row['group_id'] for row in group_rows if row['B_FPR'] is not None and row['B_FPR'] < row['A_FPR']]
    timing = [{key: row[key] for key in ('fold', 'unit_id', 'group_id', 'A_first_trigger_end_seconds',
        'B_first_trigger_end_seconds', 'relative_output_wait_seconds')} for row in rows if row['label'] == 1]
    waits = sorted(row['relative_output_wait_seconds'] for row in timing if row['relative_output_wait_seconds'] is not None)
    wait_summary = dict(positive_role_appearances=len(timing), unique_positive_videos=sum(sources[u]['video_label'] == 'pos' for u in by_uid),
        with_both_triggers=len(waits), missing_relative_wait=len(timing) - len(waits),
        sorted_seconds=waits, minimum_seconds=min(waits) if waits else None,
        median_seconds=statistics.median(waits) if waits else None,
        mean_seconds=statistics.mean(waits) if waits else None, maximum_seconds=max(waits) if waits else None)
    wait_summary['maximum_cases'] = [row for row in timing if waits and row['relative_output_wait_seconds'] == max(waits)]
    checks = dict(all_47_role_appearances_covered=len(rows) == 47 and not uncovered,
        no_new_FN=not new_fn, no_new_video_errors=not new_errors,
        at_least_two_completely_repaired_unique_FP=len(repaired) >= 2,
        at_least_two_negative_groups_improved=len(improved) >= 2,
        macro_negative_group_FPR_decreased=macro['B'] is not None and macro['B'] < macro['A'])
    return dict(per_role={str(fold): {arm: confusion([row for row in rows if row['fold'] == fold], arm) for arm in ('A', 'B')} for fold in (1, 4)},
        negative_video_role_mean_FPR=role_means, negative_group_macro_FPR=macro,
        baseline_unique_FP=baseline_fp, completely_repaired_unique_FP=repaired, remaining_unique_FP=remaining,
        unique_FP_role_details=[dict(unit_id=uid, group_id=sources[uid]['group_id'],
            roles=[{key: row[key] for key in ('fold', 'A_score', 'B_score', 'A_prediction', 'B_prediction')} for row in by_uid[uid]]) for uid in baseline_fp],
        uncovered_role_appearances=uncovered, new_FN=new_fn, new_video_errors=new_errors,
        improved_negative_groups=improved, gate_checks=checks, worth_further_full_validation=all(checks.values()),
        positive_video_output_timing=timing, relative_output_wait_distribution=wait_summary), group_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT, help='New direct child of data/results/exp024; existing directories are refused')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    require(out.parent == DEFAULT_OUT.parent.resolve() and not out.exists(), 'Output must be a new direct EXP024 result directory; no overwrite')
    inputs = {}

    def bind(path, expected=None, size=None):
        path = Path(path).resolve()
        require(path.is_file(), 'Missing input: ' + str(path))
        value = sha(path)
        require(expected is None or value == expected, 'Input SHA differs: ' + str(path))
        require(size is None or path.stat().st_size == size, 'Input size differs: ' + str(path))
        inputs[path.relative_to(ROOT).as_posix()] = value
        return path

    def read(path, expected=None, size=None):
        return json.loads(bind(path, expected, size).read_text(encoding='utf-8'))

    closed = read(RUN / 'closed-run.json')
    require(closed['status'] == 'run_closed', 'Original run is not closed')
    closed_files = {row['path']: row for row in closed['files']}
    require(len(closed_files) == len(closed['files']), 'Duplicate closed-run file record')
    plan = read(PLAN)
    require(plan['spec_id'] == 'EXP-021' and plan['seed'] == 7 and plan['independent_test'] is False, 'Wrong original plan')
    plan_sha = inputs[PLAN.relative_to(ROOT).as_posix()]
    fingerprints = plan['source_fingerprints']
    sources_path = ROOT / plan['base17_path'] / 'sources.csv'
    with bind(sources_path, fingerprints[sources_path.relative_to(ROOT).as_posix()]).open(encoding='utf-8-sig', newline='') as stream:
        source_rows = list(csv.DictReader(stream))
    sources = {row['unit_id']: row for row in source_rows}
    require(len(sources) == len(source_rows), 'Duplicate source video')
    membership = {fold: plan['roles'][str(fold)]['video_ids'] for fold in (1, 4)}
    ids = set(membership[1]) | set(membership[4])
    require((len(membership[1]), len(membership[4]), len(ids), len(set(membership[1]) & set(membership[4]))) == (23, 24, 42, 5), 'Frozen role coverage differs')
    require(ids <= sources.keys(), 'Role missing from frozen sources')
    runs = {}
    for uid in ids:
        require(sources[uid]['video_label'] in ('pos', 'neg'), 'Unknown source video label')
        runs[uid] = json.loads(sources[uid]['clean_runs_json'])
        require(runs[uid] and all(len(pair) == 2 and all(type(v) is int for v in pair) and 0 <= pair[0] < pair[1] for pair in runs[uid]), 'Invalid clean runs')
        require(all(a[1] <= b[0] for a, b in zip(runs[uid], runs[uid][1:])), 'Unordered/overlapping clean runs')
    reference_path = ROOT / plan['dense_reference']
    reference = read(reference_path, fingerprints[reference_path.relative_to(ROOT).as_posix()])
    require(reference['status'] == 'cv_dense_pixels_verified' and reference['spec_id'] == 'EXP-017', 'Wrong inherited dense-pixel proof')
    grid_path = ROOT / reference['grid_path']
    require(reference['grid_sha256'] == fingerprints[grid_path.relative_to(ROOT).as_posix()], 'Dense-grid SHA bindings differ')
    grid = {uid: {} for uid in ids}
    with bind(grid_path, reference['grid_sha256']).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            uid = row['unit_id']
            if uid not in ids:
                continue
            indices, pts = row['frame_indices'], row['source_pts_seconds']
            require(len(pts) == 16 and all(math.isfinite(v) for v in pts) and all(a < b for a, b in zip(pts, pts[1:])), 'Invalid grid source timestamps')
            require(all(a < b for a, b in zip(indices, indices[1:])), 'Invalid grid frame sequence')
            digest = tuple_digest(indices)
            require(digest not in grid[uid], 'Repeated frozen physical window')
            grid[uid][digest] = dict(target_start_seconds=float(row['target_start_seconds']), start=pts[0], end=pts[-1],
                frame_tuple_sha256=digest, clean_run_index=clean_run_index(indices, runs[uid]))
    rows = []
    for fold in (1, 4):
        role = 'dense-f%02d-A' % fold
        expected = set(membership[fold])
        require(len(expected) == len(membership[fold]), 'Duplicate role video')
        receipt = closed_files[role + '/result.json']
        result = read(RUN / receipt['path'], receipt['sha256'], receipt['size_bytes'])
        require(result['status'] == 'inner_role_dense_complete' and result['arm'] == 'A' and result['fold'] == fold
            and result['plan_sha256'] == plan_sha and result['threshold'] == THRESHOLD and result['aggregation'] == 'max'
            and result['independent_test'] is False, 'Wrong saved Linear A result contract')
        saved = {row['unit_id']: row for row in result['scores']}
        require(set(saved) == expected and len(saved) == len(result['scores']), 'Saved role coverage differs')
        receipt = closed_files[role + '/window-scores.csv']
        require(receipt['sha256'] == result['window_scores_sha256'], 'Closed CSV and result SHA bindings differ')
        grouped = defaultdict(list)
        seen = set()
        with bind(RUN / receipt['path'], receipt['sha256'], receipt['size_bytes']).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames == ['unit_id', 'target_start_seconds', 'start', 'end', 'frame_tuple_sha256', 'logit', 'score'], 'Saved CSV schema differs')
            for score in reader:
                uid, digest = score['unit_id'], score['frame_tuple_sha256']
                require(uid in expected and digest in grid[uid] and (uid, digest) not in seen, 'Unknown/repeated physical saved window')
                seen.add((uid, digest))
                mapped = grid[uid][digest]
                require(all(float(score[key]) == mapped[key] for key in ('target_start_seconds', 'start', 'end')), 'Saved target/actual timestamps differ from frozen physical tuple')
                require(math.isfinite(float(score['logit'])), 'Nonfinite saved logit')
                grouped[uid].append(dict(mapped, score=float(score['score'])))
        require(set(grouped) == expected, 'Saved window video coverage differs')
        current = []
        for uid in sorted(expected):
            stat = describe(grouped[uid])
            label, group = int(sources[uid]['video_label'] == 'pos'), sources[uid]['group_id']
            require(len(grouped[uid]) == len(grid[uid]) == saved[uid]['windows'] and saved[uid]['score'] == stat['A_score']
                and saved[uid]['label'] == label and saved[uid]['group_id'] == group and saved[uid]['fold'] == fold, 'A max/count/label/group/fold reproduction differs')
            current.append(dict(fold=fold, unit_id=uid, group_id=group, label=label, **stat))
        actual = confusion(current, 'A')
        require(all(actual[key] == result['metrics'][key] == value for key, value in EXPECTED_CONFUSION[fold].items()), 'Original A confusion reproduction differs')
        rows.extend(current)
    summary, group_rows = summarize(rows, sources)
    require(all(sha(ROOT / path) == value for path, value in inputs.items()), 'Input changed during replay')
    source = Path(__file__).read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    record = dict(status='saved_score_development_comparison_pending_result_review', spec_id='EXP-024',
        run_id=out.name, original_run_id=RUN.name, created_utc=datetime.now(timezone.utc).isoformat(),
        threshold=THRESHOLD, target_step_seconds=STEP, target_step_absolute_tolerance_seconds=TOLERANCE,
        full_design_boundary=dict(role_appearances=47, unique_videos=42, shared_videos=5, unique_negative_videos=11, negative_groups=4),
        input_sha256=inputs, source_sha256=source_sha,
        argv=[sys.executable, *sys.argv] if argv is None else [sys.executable, str(Path(__file__)), *argv],
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
            implementation=platform.python_implementation(), libraries='Python standard library only'),
        inherited_evidence='Original media/pixel validation and model selection inherited; no media/model/weight read or full-library acceptance',
        training_updates=0, model_forwards=0, rental_cost_yuan=0, independent_test=False,
        limitations=['Adaptive development observation from previously inspected scores; does not revise EXP-021 decisions',
            'Overlapping windows are not independent evidence; no new physical-motion or early-detection claim',
            'Relative output wait is B trigger end minus A trigger end, not delay from a true event onset',
            'Full-validation gate does not establish acceptable application delay; two normal windows need about 4.5 seconds of clean support'],
        **summary)
    out.mkdir(parents=True, exist_ok=False)
    try:
        for fold in (1, 4):
            write_csv(out / ('videos-f%02d.csv' % fold), [row for row in rows if row['fold'] == fold])
        write_csv(out / 'negative-groups.csv', group_rows)
        (out / 'replay.py').write_bytes(source)
        (out / 'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except Exception as error:
        (out / 'failure.json').write_text(json.dumps(dict(status='output_incomplete', error=str(error)), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        raise
    print(json.dumps(dict(output=str(out), worth_further_full_validation=record['worth_further_full_validation'],
        completely_repaired_unique_FP=record['completely_repaired_unique_FP'], negative_group_macro_FPR=record['negative_group_macro_FPR']), ensure_ascii=True))
    return record


def write_csv(path, rows):
    # None is the literal null marker in CSV, preserving unknown/non-trigger values.
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: 'null' if value is None else value for key, value in row.items()} for row in rows)


if __name__ == '__main__':
    main()
