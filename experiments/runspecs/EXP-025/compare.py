"""EXP-025 metadata preparation and once-complete five-fold saved-score comparison.

Only Python's standard library; EXP-024's byte-frozen single-video rule is reused.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve()
RULE = ROOT / 'experiments/runspecs/EXP-024/replay.py'
RULE_SHA = 'b2ea1dc33ab9d1865b23a55e3ecb05d921cedcefca600de8e626d2016e4f0442'
CARD = ROOT / 'docs/process/experiments/EXP-025.md'
LINEAR_PLAN = ROOT / 'data/metadata/EXP-022/linear-plan-v1/plan.json'
PLAN_DIR = ROOT / 'data/metadata/EXP-025/comparison-plan-v1'
MODEL_RUN_ID = 'linear-cv-s7-20261104-v1'
MODEL_RUN = ROOT / 'data/results/exp022' / MODEL_RUN_ID
RESULTS = ROOT / 'data/results/exp025'
EXPECTED_COUNTS = dict(videos=150, P=113, N=37, groups=31, negative_groups=9, windows=143470)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: 'null' if value is None else value for key, value in row.items()} for row in rows)


def environment(argv):
    return dict(argv=[sys.executable, str(SOURCE), *argv], python=sys.version,
        executable=sys.executable, platform=platform.platform(), libraries='Python standard library only')


class Inputs:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None, size=None):
        path = Path(path).resolve()
        relative = path.relative_to(ROOT).as_posix()
        require(path.is_file(), 'Missing input: ' + relative)
        value = sha(path)
        require(expected is None or value == expected, 'Input SHA differs: ' + relative)
        require(size is None or path.stat().st_size == size, 'Input size differs: ' + relative)
        require(relative not in self.hashes or value == self.hashes[relative], 'Input changed: ' + relative)
        self.hashes[relative] = value
        return path

    def read(self, path, expected=None, size=None):
        return read_json(self.bind(path, expected, size))

    def unchanged(self):
        require(all(sha(ROOT / path) == value for path, value in self.hashes.items()), 'Input changed during execution')


def load_rule(inputs):
    path = inputs.bind(RULE, RULE_SHA)
    spec = importlib.util.spec_from_file_location('exp025_frozen_confirmation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def outer_membership(folds, sources):
    """Each video and each event group belongs to exactly one of all five folds."""
    require(len(folds) == 5 and {row['fold'] for row in folds} == set(range(1, 6)), 'Require exactly five distinct folds')
    membership, group_folds = {}, {}
    for row in folds:
        fold = row['fold']
        require(type(fold) is int, 'Fold must be an integer')
        for uid in row['outer_eval']['video_ids']:
            require(uid in sources and uid not in membership, 'Unknown or repeated outer video: ' + uid)
            membership[uid] = fold
            group = sources[uid]['group_id']
            require(group not in group_folds or group_folds[group] == fold, 'Outer event group split across folds: ' + group)
            group_folds[group] = fold
    require(set(membership) == set(sources), 'Missing outer evaluation video')
    return membership


def load_metadata(inputs):
    rule = load_rule(inputs)
    linear = inputs.read(LINEAR_PLAN)
    require(linear['spec_id'] == 'EXP-022' and linear['seed'] == 7 and linear['independent_test'] is False, 'Wrong Linear plan')
    frozen = linear['source_fingerprints']
    base_dir = ROOT / linear['base_plan_path']
    inputs.read(base_dir / 'plan.json', linear['base_plan_sha256'])
    source_path = base_dir / 'sources.csv'
    with inputs.bind(source_path, frozen[source_path.relative_to(ROOT).as_posix()]).open(encoding='utf-8-sig', newline='') as stream:
        source_rows = list(csv.DictReader(stream))
    sources = {row['unit_id']: row for row in source_rows}
    require(len(sources) == len(source_rows) == EXPECTED_COUNTS['videos'], 'Source video count/duplicates differ')
    labels = Counter(row['video_label'] for row in source_rows)
    require(labels == Counter(pos=113, neg=37), 'Expected 113P/37N')
    require(len({row['group_id'] for row in source_rows}) == 31 and len({row['group_id'] for row in source_rows if row['video_label'] == 'neg'}) == 9, 'Expected 31 event groups/nine negative groups')
    fold_path = base_dir / 'folds.json'
    folds = inputs.read(fold_path, frozen[fold_path.relative_to(ROOT).as_posix()])
    membership = outer_membership(folds, sources)
    reference_path = ROOT / linear['dense_reference']
    reference = inputs.read(reference_path, frozen[reference_path.relative_to(ROOT).as_posix()])
    require(reference['status'] == 'cv_dense_pixels_verified' and reference['video_count'] == 150
        and reference['windows'] == 143470 and set(reference['videos']) == set(sources), 'Wrong inherited dense grid proof')
    runs = {}
    for uid, source in sources.items():
        runs[uid] = json.loads(source['clean_runs_json'])
        require(runs[uid] and all(len(pair) == 2 and all(type(v) is int for v in pair)
            and 0 <= pair[0] < pair[1] for pair in runs[uid]), 'Invalid clean runs: ' + uid)
        require(all(a[1] <= b[0] for a, b in zip(runs[uid], runs[uid][1:])), 'Unordered/overlapping clean runs: ' + uid)
    grid_path = ROOT / reference['grid_path']
    require(reference['grid_sha256'] == frozen[grid_path.relative_to(ROOT).as_posix()], 'Dense grid SHA bindings differ')
    grid = {uid: {} for uid in sources}
    with inputs.bind(grid_path, reference['grid_sha256']).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            uid = row['unit_id']
            require(uid in sources, 'Unknown grid video')
            indices, pts = row['frame_indices'], row['source_pts_seconds']
            require(len(pts) == 16 and all(math.isfinite(t) for t in pts) and all(a < b for a, b in zip(pts, pts[1:])), 'Invalid original source PTS')
            require(all(a < b for a, b in zip(indices, indices[1:])), 'Invalid original frame sequence')
            digest = rule.tuple_digest(indices)
            require(digest not in grid[uid], 'Repeated grid physical tuple')
            grid[uid][digest] = dict(target_start_seconds=float(row['target_start_seconds']), start=pts[0], end=pts[-1],
                frame_tuple_sha256=digest, clean_run_index=rule.clean_run_index(indices, runs[uid]))
    require(sum(map(len, grid.values())) == 143470 and all(len(grid[uid]) == reference['videos'][uid]['windows'] for uid in sources), 'Inherited dense physical window coverage differs')
    return dict(rule=rule, linear=linear, sources=sources, membership=membership, grid=grid)


def prepare(argv):
    require(not PLAN_DIR.exists(), 'Comparison plan already exists; no overwrite')
    inputs = Inputs()
    inputs.bind(SOURCE)
    inputs.bind(CARD)
    context = load_metadata(inputs)
    coverage = []
    for uid in sorted(context['sources']):
        # Constant placeholders permit reuse of the reviewed geometry rule. No model scores are read or reported.
        stat = context['rule'].describe([dict(row, score=0.) for row in context['grid'][uid].values()])
        source = context['sources'][uid]
        coverage.append(dict(unit_id=uid, fold=context['membership'][uid], group_id=source['group_id'],
            label=int(source['video_label'] == 'pos'), **{key: value for key, value in stat.items()
            if key in ('windows', 'legal_pairs', 'covered', 'quality_excluded_windows', 'adjacent_candidate_pairs') or key.startswith('excluded_pairs_')}))
    missing = [row['unit_id'] for row in coverage if not row['covered']]
    inputs.unchanged()
    PLAN_DIR.mkdir(parents=True, exist_ok=False)
    write_csv(PLAN_DIR / 'coverage.csv', coverage)
    (PLAN_DIR / 'compare.py').write_bytes(SOURCE.read_bytes())
    (PLAN_DIR / 'exp024-replay.py').write_bytes(RULE.read_bytes())
    plan = dict(spec_id='EXP-025', version=1, status='metadata_coverage_complete_analysis_frozen' if not missing else 'metadata_coverage_incomplete_do_not_start_comparison',
        created_utc=datetime.now(timezone.utc).isoformat(), metadata_only=True, coverage_complete=not missing,
        missing_coverage_videos=missing, counts=EXPECTED_COUNTS, minimum_legal_pairs=min(row['legal_pairs'] for row in coverage),
        coverage_csv_sha256=sha(PLAN_DIR / 'coverage.csv'), source_sha256=inputs.hashes[SOURCE.relative_to(ROOT).as_posix()],
        confirmation_source_sha256=RULE_SHA, input_sha256=inputs.hashes, environment=environment(argv),
        outer_video_fold=context['membership'], linear_plan_path=LINEAR_PLAN.relative_to(ROOT).as_posix(),
        expected_model_run_id=MODEL_RUN_ID, model_run_locations=dict(local=MODEL_RUN.relative_to(ROOT).as_posix(),
            cloud='/root/exp022-results/' + MODEL_RUN_ID),
        threshold=.5, target_step_seconds=1.5, target_step_absolute_tolerance_seconds=1e-6,
        score_input_policy='All five closed outer folds once complete; never feed comparison scores back into fit or E selection',
        scientific_gate='Full150 coverage; total FP and nine-negative-group macro FPR decrease; total FN nonincrease; at least two negative groups improve',
        application_suitability='undetermined', new_training_updates=0, new_model_forwards=0, incremental_cost_yuan=0,
        scope='Metadata feasibility only; inherited media/pixel/weight acceptance not rerun; constant placeholders are not predictions')
    write_json(PLAN_DIR / 'plan.json', plan)
    print(json.dumps(dict(plan=str(PLAN_DIR), coverage_complete=not missing, missing=missing,
        videos=len(coverage), minimum_legal_pairs=plan['minimum_legal_pairs']), ensure_ascii=True))
    require(not missing, 'Coverage incomplete; saved all videos and blocked full comparison')
    return plan


def metrics(rows, arm, include_auc=False):
    counts = Counter()
    missing = []
    for row in rows:
        pred = row[arm + '_prediction']
        if pred is None:
            missing.append(row['unit_id'])
        else:
            counts[('TP' if pred else 'FN') if row['label'] else ('FP' if pred else 'TN')] += 1
    p = sum(row['label'] == 1 for row in rows)
    n = len(rows) - p
    values = {key: None if missing else counts[key] for key in ('TP', 'FP', 'FN', 'TN')}
    tp, fp, fn = (counts[key] for key in ('TP', 'FP', 'FN'))
    result = dict(status='incomplete_coverage_metrics_withheld' if missing else 'complete', P=p, N=n, **values,
        Recall=None if missing or not p else tp / p, FPR=None if missing or not n else fp / n,
        Precision=None if missing else tp / (tp + fp) if tp + fp else 0.,
        F1=None if missing else 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        AUC=None, missing_coverage_videos=missing)
    if include_auc and not missing and p and n:
        pos = [row[arm + '_score'] for row in rows if row['label'] == 1]
        neg = [row[arm + '_score'] for row in rows if row['label'] == 0]
        result['AUC'] = math.fsum((a > b) + .5 * (a == b) for a in pos for b in neg) / (p * n)
    return result


def receipt_index(receipt):
    indexed = {row['path']: row for row in receipt['files']}
    require(len(indexed) == len(receipt['files']), 'Duplicate closed file receipt')
    return indexed


def compare(plan_dir, out, argv):
    require(out.parent == RESULTS.resolve() and not out.exists(), 'Choose a new direct EXP025 result directory; no overwrite')
    inputs = Inputs()
    plan = inputs.read(plan_dir / 'plan.json')
    require(plan['spec_id'] == 'EXP-025' and plan['coverage_complete'] is True and plan['counts'] == EXPECTED_COUNTS,
        'Complete150 metadata plan required')
    require(plan['expected_model_run_id'] == MODEL_RUN_ID and plan['confirmation_source_sha256'] == RULE_SHA, 'Frozen run/rule differs')
    for relative, value in plan['input_sha256'].items():
        inputs.bind(ROOT / relative, value)
    inputs.bind(plan_dir / 'coverage.csv', plan['coverage_csv_sha256'])
    require(sha(SOURCE) == plan['source_sha256'], 'Analysis source changed after metadata freeze')
    context = load_metadata(inputs)
    require(context['membership'] == plan['outer_video_fold'], 'Frozen outer mapping differs')
    closed = inputs.read(MODEL_RUN / 'closed-run.json')
    require(closed['status'] == 'run_closed', 'Wait for the entire successful five-fold model run to close')
    run_files = receipt_index(closed)

    def closed_read(relative):
        item = run_files[relative]
        return inputs.read(MODEL_RUN / relative, item['sha256'], item['size_bytes'])

    pipeline = closed_read('pipeline-result.json')
    require(pipeline['status'] == 'five_fold_linear_cv_complete_pending_independent_audit', 'Incomplete five-fold pipeline')
    stages = {}
    # Check the complete five-fold inventory before reading any probability CSV.
    for fold in range(1, 6):
        name = 'f%02d-dense' % fold
        receipt = closed_read('closed-' + name + '.json')
        require(receipt['status'] == 'dense_closed' and receipt['item'] == name, 'Missing/failed dense fold')
        stage_files = receipt_index(receipt)
        for suffix in ('result.json', 'window-scores.csv'):
            relative = name + '/' + suffix
            require(stage_files[relative] == run_files[relative], 'Per-stage/whole-run closure differs')
        stages[fold] = name
    rows = []
    for fold, name in stages.items():
        result = closed_read(name + '/result.json')
        linear_sha = inputs.hashes[LINEAR_PLAN.relative_to(ROOT).as_posix()]
        require(result['status'] == 'outer_fold_complete' and result['spec_id'] == 'EXP-022' and result['head'] == 'linear'
            and result['fold'] == fold and result['linear_plan_sha256'] == linear_sha
            and result['plan_sha256'] == context['linear']['base_plan_sha256'] and result['threshold'] == .5
            and result['aggregation'] == 'max' and result['outer_selection_use'] is False
            and result['independent_test_used'] is False, 'Wrong fixed outer Linear result contract')
        require(result['pixel_reference_sha256'] == inputs.hashes[context['linear']['dense_reference']], 'Inherited dense reference differs')
        expected = {uid for uid, assigned in context['membership'].items() if assigned == fold}
        saved = {row['unit_id']: row for row in result['scores']}
        require(set(saved) == expected and len(saved) == len(result['scores']), 'Wrong saved outer video coverage')
        item = run_files[name + '/window-scores.csv']
        require(item['sha256'] == result['window_scores_sha256'], 'Closed CSV/result SHA differs')
        grouped, seen = defaultdict(list), set()
        with inputs.bind(MODEL_RUN / item['path'], item['sha256'], item['size_bytes']).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            required = {'unit_id', 'target_start_seconds', 'start', 'end', 'frame_tuple_sha256', 'score'}
            require(reader.fieldnames is not None and set(reader.fieldnames) in (required, required | {'logit'})
                and len(reader.fieldnames) == len(set(reader.fieldnames)), 'Wrong outer score schema')
            for row in reader:
                uid, digest = row['unit_id'], row['frame_tuple_sha256']
                require(uid in expected and digest in context['grid'][uid] and (uid, digest) not in seen, 'Unknown/repeated/wrong-fold saved physical window')
                seen.add((uid, digest))
                mapped = context['grid'][uid][digest]
                require(all(float(row[key]) == mapped[key] for key in ('target_start_seconds', 'start', 'end')), 'Physical tuple/target/actual timestamps differ')
                require('logit' not in row or math.isfinite(float(row['logit'])), 'Nonfinite optional saved logit')
                grouped[uid].append(dict(mapped, score=float(row['score'])))
        require(set(grouped) == expected, 'Incomplete outer score video coverage')
        current = []
        for uid in sorted(expected):
            stat = context['rule'].describe(grouped[uid])
            source = context['sources'][uid]
            label = int(source['video_label'] == 'pos')
            original = saved[uid]
            require(stat['windows'] == len(context['grid'][uid]) == original['windows'] and original['score'] == stat['A_score']
                and original['fold'] == fold and original['label'] == label and original['group_id'] == source['group_id'], 'Original A max/count/label/group/fold reproduction differs')
            current.append(dict(fold=fold, unit_id=uid, group_id=source['group_id'], label=label, **stat))
        restored = metrics(current, 'A', include_auc=True)
        require(all(restored[key] == result['metrics'][key] for key in ('P', 'N', 'TP', 'FP', 'FN', 'TN', 'Recall', 'FPR', 'AUC')), 'New Linear A baseline metrics reproduction differs')
        rows.extend(current)
    require(len(rows) == len({row['unit_id'] for row in rows}) == 150
        and sum(row['windows'] for row in rows) == 143470, 'Require all150 outer videos/143470 windows exactly once')
    summary, fold_csv, group_csv = summarize(rows)
    inputs.unchanged()
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'videos.csv', rows)
    write_csv(out / 'folds.csv', fold_csv)
    write_csv(out / 'negative-groups.csv', group_csv)
    write_csv(out / 'positive-output-timing.csv', summary['positive_output_timing'])
    (out / 'compare.py').write_bytes(SOURCE.read_bytes())
    (out / 'exp024-replay.py').write_bytes(RULE.read_bytes())
    record = dict(spec_id='EXP-025', run_id=out.name, original_model_run_id=MODEL_RUN_ID,
        status='complete_saved_score_comparison_pending_result_review' if summary['coverage_complete'] else 'coverage_incomplete_no_complete_performance_claim',
        created_utc=datetime.now(timezone.utc).isoformat(), environment=environment(argv), input_sha256=inputs.hashes,
        analysis_source_sha256=sha(SOURCE), confirmation_source_sha256=RULE_SHA, counts=EXPECTED_COUNTS,
        threshold=.5, target_step_seconds=1.5, target_step_absolute_tolerance_seconds=1e-6,
        pooled_AUC=None, pooled_AUC_reason='Different fold models; only within-fold AUC is calculated, preserving saved probability ties',
        application_suitability='undetermined', application_reason='No accepted output-delay limit or independent application validation; scientific count gate cannot establish suitability',
        new_training_updates=0, new_model_forwards=0, incremental_cost_yuan=0, independent_test=False,
        limits=['Previously inspected adaptive development pool, event-grouped CV; no unseen-camera or independent-generalization claim',
            'Relative output wait is not latency from true event onset', 'Decision-layer persistence, not improved network temporal features',
            'Overlapping windows are not independent evidence; no threshold/k/stride search or feedback to fitting'], **summary)
    write_json(out / 'result.json', record)
    print(json.dumps(dict(output=str(out), scientific_gate_passed=record['scientific_gate_passed'],
        application_suitability='undetermined'), ensure_ascii=True))
    return record


def summarize(rows):
    folds, fold_csv = {}, []
    for fold in range(1, 6):
        selected = [row for row in rows if row['fold'] == fold]
        require(selected, 'Missing output fold')
        values = {arm: metrics(selected, arm, include_auc=True) for arm in ('A', 'B')}
        values['repaired_FP'] = [row['unit_id'] for row in selected if row['label'] == 0 and row['A_prediction'] == 1 and row['B_prediction'] == 0]
        values['new_FN'] = [row['unit_id'] for row in selected if row['label'] == 1 and row['A_prediction'] == 1 and row['B_prediction'] == 0]
        values['score_distributions'] = {arm: sorted(row[arm + '_score'] for row in selected if row[arm + '_score'] is not None) for arm in ('A', 'B')}
        folds[str(fold)] = values
        flat = dict(fold=fold, videos=len(selected), repaired_FP=json.dumps(values['repaired_FP']), new_FN=json.dumps(values['new_FN']))
        flat.update({arm + '_' + key: value for arm in ('A', 'B') for key, value in values[arm].items() if key not in ('status', 'missing_coverage_videos')})
        fold_csv.append(flat)
    negative_groups = sorted({row['group_id'] for row in rows if row['label'] == 0})
    require(len(negative_groups) == 9, 'Use all nine negative event groups')
    group_details, group_csv = {}, []
    for group in negative_groups:
        selected = [row for row in rows if row['group_id'] == group]
        values = {arm: metrics(selected, arm) for arm in ('A', 'B')}
        group_details[group] = values
        flat = dict(group_id=group, fold=selected[0]['fold'], videos=len(selected))
        flat.update({arm + '_' + key: value for arm in ('A', 'B') for key, value in values[arm].items() if key not in ('status', 'missing_coverage_videos', 'AUC')})
        group_csv.append(flat)
    macro = {arm: None if any(group_details[group][arm]['FPR'] is None for group in negative_groups)
        else math.fsum(group_details[group][arm]['FPR'] for group in negative_groups) / 9 for arm in ('A', 'B')}
    total = {arm: metrics(rows, arm) for arm in ('A', 'B')}
    improved = [group for group in negative_groups if group_details[group]['B']['FPR'] is not None
        and group_details[group]['B']['FPR'] < group_details[group]['A']['FPR']]
    coverage = all(row['covered'] for row in rows)
    checks = dict(full150_coverage=coverage and len(rows) == 150,
        total_FP_decreased=total['B']['FP'] is not None and total['B']['FP'] < total['A']['FP'],
        total_FN_nonincrease=total['B']['FN'] is not None and total['B']['FN'] <= total['A']['FN'],
        nine_negative_group_macro_FPR_decreased=macro['B'] is not None and macro['B'] < macro['A'],
        at_least_two_negative_groups_improved=len(improved) >= 2)
    timing = [dict(fold=row['fold'], unit_id=row['unit_id'], group_id=row['group_id'],
        A_first_trigger_end_seconds=row['A_first_trigger_end_seconds'], B_first_trigger_end_seconds=row['B_first_trigger_end_seconds'],
        relative_output_wait_seconds=row['relative_output_wait_seconds'], B_covered=row['covered'],
        A_triggered=row['A_first_trigger_end_seconds'] is not None, B_triggered=row['B_first_trigger_end_seconds'] is not None)
        for row in rows if row['label'] == 1]
    waits = sorted(row['relative_output_wait_seconds'] for row in timing if row['relative_output_wait_seconds'] is not None)
    wait_summary = dict(positive_videos=len(timing), with_both_triggers=len(waits), missing_relative_wait=len(timing) - len(waits),
        A_not_triggered=sum(not row['A_triggered'] for row in timing), B_not_triggered=sum(not row['B_triggered'] for row in timing),
        B_without_coverage=sum(not row['B_covered'] for row in timing), sorted_seconds=waits,
        median_seconds=statistics.median(waits) if waits else None, mean_seconds=statistics.mean(waits) if waits else None,
        maximum_seconds=max(waits) if waits else None,
        maximum_cases=[row for row in timing if waits and row['relative_output_wait_seconds'] == max(waits)])
    return dict(coverage_complete=coverage, uncovered_videos=[row['unit_id'] for row in rows if not row['covered']],
        per_fold=folds, total_counts=total, negative_groups=group_details, negative_group_macro_FPR=macro,
        improved_negative_groups=improved, regressed_negative_groups=[group for group in negative_groups
            if group_details[group]['B']['FPR'] is not None and group_details[group]['B']['FPR'] > group_details[group]['A']['FPR']],
        repaired_FP=[row['unit_id'] for row in rows if row['label'] == 0 and row['A_prediction'] == 1 and row['B_prediction'] == 0],
        remaining_FP=[row['unit_id'] for row in rows if row['label'] == 0 and row['B_prediction'] == 1],
        new_FN=[row['unit_id'] for row in rows if row['label'] == 1 and row['A_prediction'] == 1 and row['B_prediction'] == 0],
        all_B_FN=[row['unit_id'] for row in rows if row['label'] == 1 and row['B_prediction'] == 0],
        positive_output_timing=timing, relative_output_wait_distribution=wait_summary,
        scientific_gate_checks=checks, scientific_gate_passed=all(checks.values())), fold_csv, group_csv


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare', help='Metadata-only150-video coverage check and analysis freeze; no probability inputs')
    comparison = commands.add_parser('compare', help='Read all five closed outer score CSVs once the complete model run exists')
    comparison.add_argument('--plan', type=Path, default=PLAN_DIR)
    comparison.add_argument('--out', type=Path, default=RESULTS / 'confirmation-linear-cv-s7-20261104-v1')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        return prepare(argv)
    return compare(args.plan.resolve(), args.out.resolve(), argv)


if __name__ == '__main__':
    main()
