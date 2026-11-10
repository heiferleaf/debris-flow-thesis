"""Derive EXP022's closed five-fold comparison from saved JSON/CSV only.

No training modules, model loading, inference, decoding or checkpoint hashing.
An incomplete/uncollected run raises Incomplete before any output is created.
Checkpoint bytes remain the collector's responsibility; manifest links and
collected sizes are checked here. Every consumed metadata/CSV byte is hashed.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import statistics
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
PLAN_SHA = 'acb698ccf8e32eec8f60af873f7bd6c428067f55577324057376e7e630284443'
RUN_ID = 'linear-cv-s7-20261104-v1'


class Incomplete(ValueError):
    """A completed result or its local collection is unavailable."""


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def rooted(root, relative):
    name = PurePosixPath(relative)
    require(not name.is_absolute() and '..' not in name.parts and '\\' not in relative
            and ':' not in relative, 'Expected safe relative artifact path')
    path = root / relative
    require(not path.is_symlink(), 'Symlink artifacts are not accepted')
    path.resolve().relative_to(root.resolve())
    return path


class Inputs:
    def __init__(self):
        self.hashes = {}
        self.checkpoints = {}

    def bind(self, path, expected=None, size=None):
        path = Path(path).resolve()
        if not path.is_file():
            raise Incomplete('Required closed artifact not collected: ' + str(path))
        require(size is None or path.stat().st_size == size, 'Collected file size differs: ' + str(path))
        require(path.suffix != '.pt', 'Checkpoint hashing is deliberately disabled')
        actual = self.hashes.get(str(path))
        if actual is None:
            actual = sha(path)
            self.hashes[str(path)] = actual
        require(expected is None or actual == expected, 'Source SHA differs: ' + str(path))
        return path

    def read(self, path, expected=None):
        return json.loads(self.bind(path, expected).read_text(encoding='utf-8'))

    def jsonl(self, path):
        with self.bind(path).open(encoding='utf-8') as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def rows(self, path, expected=None):
        with self.bind(path, expected).open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))

    def inventory(self, root, records):
        indexed = {}
        for record in records:
            name = record['path']
            require(name not in indexed, 'Duplicate closure artifact: ' + name)
            path = rooted(root, name)
            require(isinstance(record['size_bytes'], int) and record['size_bytes'] >= 0
                    and len(record['sha256']) == 64, 'Invalid closure record')
            if path.suffix == '.pt':
                if not path.is_file():
                    raise Incomplete('Checkpoint not fully collected: ' + str(path))
                require(path.stat().st_size == record['size_bytes'], 'Checkpoint collection size differs')
                self.checkpoints[str(path.resolve())] = record
            else:
                self.bind(path, record['sha256'], record['size_bytes'])
            indexed[name] = record
        return indexed

    def unchanged(self):
        require(all(sha(path) == value for path, value in self.hashes.items()), 'Report input changed during derivation')
        require(all(Path(p).is_file() and Path(p).stat().st_size == r['size_bytes']
                    for p, r in self.checkpoints.items()), 'Checkpoint collection changed during derivation')


def metrics(rows, *, auc=True):
    require(all(type(r['label']) is int and r['label'] in (0, 1) and math.isfinite(r['score'])
                and 0 <= r['score'] <= 1 for r in rows), 'Invalid probability or label')
    pos = [r['score'] for r in rows if r['label'] == 1]
    neg = [r['score'] for r in rows if r['label'] == 0]
    tp, fp = sum(p >= .5 for p in pos), sum(p >= .5 for p in neg)
    fn, tn = len(pos) - tp, len(neg) - fp
    result = dict(status='complete', P=len(pos), N=len(neg), TP=tp, FP=fp, FN=fn, TN=tn,
        Recall=tp / len(pos) if pos else None, FPR=fp / len(neg) if neg else None,
        Precision=tp / (tp + fp) if tp + fp else 0.,
        F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    if auc:
        # Saved probability values, exact comparison, and half credit for ties.
        result['AUC'] = (math.fsum((p > n) + .5 * (p == n) for p in pos for n in neg)
                         / (len(pos) * len(neg)) if pos and neg else None)
    return result


def same_metrics(actual, saved):
    require(actual.keys() == saved.keys(), 'Saved metric fields differ')
    for key, value in actual.items():
        other = saved[key]
        require((type(other) in (int, float) and math.isclose(value, other, rel_tol=0, abs_tol=1e-15))
                if type(value) in (int, float) else value == other, 'Saved metric differs: ' + key)


def paired_comparison(deep_rows, linear_rows):
    """Identity-safe counts/gates; works with tiny fixtures as well as full CV."""
    def index(rows):
        found = {}
        for row in rows:
            require(row['unit_id'] not in found, 'Duplicate paired video identity')
            found[row['unit_id']] = row
        return found
    deep, linear = index(deep_rows), index(linear_rows)
    require(deep.keys() == linear.keys() and bool(deep), 'Paired video coverage differs')
    pairs = []
    for uid, old in sorted(deep.items()):
        new = linear[uid]
        require(all(old[k] == new[k] for k in ('fold', 'group_id', 'label', 'windows')),
                'Paired video label/group/fold/window identity differs: ' + uid)
        a, b, y = int(old['score'] >= .5), int(new['score'] >= .5), old['label']
        change = ('new_false_negative' if y else 'new_false_positive') if a == y and b != y else (
            'repaired_false_negative' if y else 'repaired_false_positive') if a != y and b == y else 'unchanged'
        pairs.append(dict(unit_id=uid, fold=old['fold'], group_id=old['group_id'], label=y, windows=old['windows'],
            deep_score=old['score'], linear_score=new['score'], deep_prediction=a, linear_prediction=b,
            deep_correct=a == y, linear_correct=b == y, change=change))
    old_total, new_total = metrics(deep_rows, auc=False), metrics(linear_rows, auc=False)
    groups = []
    for group in sorted({r['group_id'] for r in deep_rows if r['label'] == 0}):
        selected = [r for r in pairs if r['group_id'] == group and r['label'] == 0]
        n = len(selected)
        fp_old, fp_new = sum(r['deep_prediction'] for r in selected), sum(r['linear_prediction'] for r in selected)
        groups.append(dict(group_id=group, N=n, deep_FP=fp_old, linear_FP=fp_new,
            deep_FPR=fp_old / n, linear_FPR=fp_new / n, delta_FPR=(fp_new - fp_old) / n))
    require(bool(groups), 'At least one negative group is needed')
    macro_old, macro_new = [statistics.mean(r[arm + '_FPR'] for r in groups) for arm in ('deep', 'linear')]
    improved = [r['group_id'] for r in groups if r['linear_FP'] < r['deep_FP']]
    gate = dict(total_FP_strictly_lower=new_total['FP'] < old_total['FP'],
        macro_negative_group_FPR_strictly_lower=macro_new < macro_old,
        total_FN_not_increased=new_total['FN'] <= old_total['FN'],
        at_least_two_negative_groups_improved=len(improved) >= 2)
    return dict(deep=old_total, linear=new_total, deep_macro_negative_group_FPR=macro_old,
        linear_macro_negative_group_FPR=macro_new, negative_groups=groups, improved_negative_groups=improved,
        worsened_negative_groups=[r['group_id'] for r in groups if r['linear_FP'] > r['deep_FP']],
        gate_components=gate, supports_overall_head_simplification_video_improvement=all(gate.values()),
        regression_cases=[r for r in pairs if r['deep_correct'] and not r['linear_correct']]), pairs


def probability_stats(values):
    require(bool(values), 'Empty probability distribution')
    ordered = sorted(values)
    def q(fraction):
        position = (len(ordered) - 1) * fraction
        lo = math.floor(position); hi = math.ceil(position)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    return dict(n=len(values), minimum=ordered[0], q25=q(.25), median=q(.5), q75=q(.75), maximum=ordered[-1],
        mean=statistics.mean(values), population_sd=statistics.pstdev(values), below_0_5=sum(v < .5 for v in values))


def window_review(inputs, path, result, reference, paired_windows=None):
    rows = result['scores']
    by_id = {r['unit_id']: r for r in rows}
    require(len(rows) == len(by_id) and set(result['videos']) == set(by_id), 'Duplicate/missing dense video identity')
    windows, counts, maxima, values = {}, Counter(), {}, []
    for row in inputs.rows(path, result['window_scores_sha256']):
        uid, value = row['unit_id'], float(row['score'])
        require(uid in by_id and math.isfinite(value) and 0 <= value <= 1
                and struct.unpack('f', struct.pack('f', value))[0] == value, 'Invalid saved FP32 window probability')
        key = uid, row['frame_tuple_sha256']
        require(key not in windows and len(key[1]) == 64, 'Duplicate/invalid physical window identity')
        times = tuple(float(row[k]) for k in ('target_start_seconds', 'start', 'end'))
        require(all(math.isfinite(v) for v in times) and times[2] > times[1], 'Invalid saved window time')
        windows[key] = times
        counts[uid] += 1; maxima[uid] = max(maxima.get(uid, 0.), value); values.append(value)
    require(set(counts) == set(by_id), 'Missing video windows')
    if paired_windows is not None:
        require(windows == paired_windows, 'Deep/Linear physical window identity or time differs')
    for uid, score in by_id.items():
        observed, expected = result['videos'][uid], reference['videos'][uid]
        require(score['score'] == maxima[uid] and score['windows'] == counts[uid] == observed['windows'] == expected['windows']
                and observed['score'] == maxima[uid] and observed['canonical_reference_matched'] is True
                and observed['rgb112_digest'] == expected['rgb112_digest'], 'Video/window/pixel reference closure differs: ' + uid)
    same_metrics(metrics(rows), result['metrics'])
    return windows, probability_stats(values)


def stage_order():
    return [item for f in range(1, 6) for item in
            (['f%02d-inner' % f] if f in (2, 3, 5) else []) + ['f%02d-outer' % f, 'f%02d-dense' % f]]


def closed_stage(inputs, run, item, status, master=None, expected_sha=None):
    if master is not None:
        require('closed-' + item + '.json' in master, 'Stage receipt absent from closed-run inventory')
    receipt = inputs.read(run / ('closed-' + item + '.json'), expected_sha)
    require(receipt['status'] == status and receipt['item'] == item and receipt['path'] == item,
            'Wrong/unclosed stage: ' + item)
    records = receipt['files']
    require(all(r['path'].startswith(item + '/') or r['path'] == item + '.log' for r in records), 'Stage closure path differs')
    inventory = inputs.inventory(run, records)
    if master is not None:
        require(all(name in master and master[name] == record for name, record in inventory.items()),
                'Stage/master closure manifests differ: ' + item)
    return inventory


def fit_review(inputs, directory, fold, stage, role, selected_E, files, initial_sha, fingerprints=None, validation_ids=None):
    def load(name, lines=False):
        require(directory.name + '/' + name in files, 'Fit metadata absent from stage closure')
        return inputs.jsonl(directory / name) if lines else inputs.read(directory / name)
    result, profile, history = load('training-result.json'), load('profile.json'), load('epochs.jsonl', True)
    epochs = 20 if stage == 'inner_select' else selected_E
    require(result['status'] == 'fit_complete' and result['stage'] == stage and result['fold'] == fold
            and result['completed_epochs'] == epochs and result['selected_epochs'] == selected_E
            and result['initial_state_sha256'] == initial_sha and result['fixed_outer_validation_evaluated'] is False,
            'Fit identity/completion/selection differs')
    require(profile['execution_kind'] == 'registered_fit' and profile['head'] == 'linear'
            and profile['head_parameters'] == 513 and profile['seed'] == 7 and profile['mode'] == 'rgb_zero'
            and profile['stage'] == stage and profile['fold'] == fold and profile['requested_epochs'] == epochs
            and profile['initial_state_sha256'] == initial_sha and profile['optimizer']['initial_state_entries'] == 0,
            'Unregistered Linear fit or fresh initialization differs')
    if fingerprints is not None:
        require(result['input_fingerprints'] == profile['input_fingerprints'] == fingerprints,
                'Fit frozen source fingerprints differ')
    require(result['profile_sha256'] == inputs.hashes[str((directory / 'profile.json').resolve())]
            and len(history) == epochs, 'Fit profile/history closure differs')
    require(type(selected_E) is int and 1 <= selected_E <= 20, 'Invalid selected E')
    total, curves = 0, []
    orders = role['epoch_clip_ids']
    for epoch, row in enumerate(history, 1):
        n = len(orders[epoch - 1]); steps = math.ceil(n / 24); q, r = divmod(n, steps)
        total += steps
        require(row['completed_epochs'] == epoch and row['train_clip_ids'] == orders[epoch - 1]
                and row['samples'] == n and row['optimizer_steps'] == steps and row['total_updates'] == total
                and row['actual_effective_batch_sizes'] == [q + 1] * r + [q] * (steps - r)
                and math.isfinite(row['train_weighted_bce']), 'Fit epoch exposure/update/loss differs')
        curve = dict(fold=fold, stage=stage, completed_epochs=epoch, optimizer_steps=steps, cumulative_updates=total,
            train_weighted_bce=row['train_weighted_bce'], validation_unweighted_bce=row.get('validation_unweighted_bce'))
        if stage == 'inner_select':
            require(math.isfinite(row['validation_unweighted_bce'])
                    and (validation_ids is None or row['validation_clip_ids'] == validation_ids),
                    'Nonfinite inner selection BCE or changed fixed validation identities')
        curves.append(curve)
    if stage == 'inner_select':
        require(selected_E == min(range(1, 21), key=lambda e: history[e - 1]['validation_unweighted_bce']),
                'Selected E is not recorded earliest minimum inner BCE')
    else:
        require(all('validation_unweighted_bce' not in row for row in history), 'Outer fit used validation')
    require(result['optimizer_steps'] == profile['planned_optimizer_steps'] == total, 'Fit total optimizer updates differ')
    checkpoint = result['selected_checkpoint']
    expected_name = 'best.pt' if stage == 'inner_select' else 'selected.pt'
    require(checkpoint['filename'] == expected_name and checkpoint['completed_epochs'] == selected_E
            and checkpoint['sha256'] == files[directory.name + '/' + expected_name]['sha256'], 'Selected checkpoint provenance differs')
    return dict(selected_E=selected_E, completed_epochs=epochs, optimizer_updates=total,
                artifact_directory=str(directory)), curves, result


def _csv(path, rows, columns):
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def generate(root, plan_dir, run, out, argv=None):
    root, plan_dir, run, out = [Path(p).resolve() for p in (root, plan_dir, run, out)]
    require(not out.exists(), 'Refuse derived output overwrite')
    require(not any(out == parent or parent in out.parents for parent in (run, plan_dir)),
            'Derived output must be outside run and frozen plan')
    inputs = Inputs()
    terminal = inputs.read(run / 'closed-run.json')
    if terminal['status'] != 'run_closed':
        raise Incomplete('Run is not fully closed; five-fold metrics withheld')
    for failure in ('pipeline-failure.json', 'hard-time-limit.json'):
        require(not (run / failure).exists(), 'Failed/interrupted run is not a full result')
    master = inputs.inventory(run, terminal['files'])
    plan = inputs.read(plan_dir / 'plan.json', PLAN_SHA)
    baseline = inputs.read(plan_dir / 'deep-baseline.json', plan['deep_baseline_sha256'])
    require(plan['spec_id'] == 'EXP-022' and plan['seed'] == 7 and plan['dense_videos'] == 150
            and plan['dense_physical_windows'] == 143470 and plan['independent_test'] is False, 'Unexpected EXP022 plan')
    fingerprint = dict(plan['source_fingerprints'])
    for name in ('plan.json', 'deep-baseline.json'):
        path = plan_dir / name
        fingerprint[path.relative_to(root).as_posix()] = inputs.hashes[str(path)]
    contract = inputs.read(run / 'execution-inputs.json')
    require(contract['plan_sha256'] == PLAN_SHA and contract['input_fingerprints'] == fingerprint
            and contract['stage_order'] == stage_order() and contract['local_archive_mapping'] == 'data/results/exp022/' + run.name,
            'Controller plan/source/run mapping differs')
    for name, expected in contract['code_sha256'].items():
        require(fingerprint.get(name) == expected, 'Controller code source differs from plan')
        inputs.bind(rooted(run / 'source-snapshot', name), expected)
    pipeline = inputs.read(run / 'pipeline-result.json')
    require(pipeline['status'] == 'five_fold_linear_cv_complete_pending_independent_audit', 'Pipeline is not five-fold complete')
    controller_comparison = inputs.read(run / 'comparison.json', pipeline['comparison_sha256'])
    reuse_receipt = inputs.read(run / 'inner-reuse.json', pipeline['source_inner_reuse_sha256'])
    require(reuse_receipt['status'] == 'two_closed_inner_stages_reused_no_retraining', 'Missing registered inner reuse receipt')
    require(set(reuse_receipt['items']) == set(plan['reused_inner']) == {'1', '4'}, 'Registered reused inner identities differ')
    for name in ('execution-inputs.json', 'pipeline-result.json', 'comparison.json', 'inner-reuse.json'):
        require(name in master, 'Controller result missing from closed-run inventory')
    base_dir = rooted(root, plan['base_plan_path'])
    base = inputs.read(base_dir / 'plan.json', plan['base_plan_sha256'])
    folds = inputs.read(base_dir / 'folds.json', base['files']['folds.json'])
    source_rows = inputs.rows(base_dir / 'sources.csv', base['files']['sources.csv'])
    sources = {r['unit_id']: r for r in source_rows}
    require(len(source_rows) == len(sources) == 150 and Counter(r['video_label'] for r in source_rows) == Counter(pos=113, neg=37),
            'Expected 150 unique videos, 113P/37N')
    reference = inputs.read(rooted(root, plan['dense_reference']), fingerprint[plan['dense_reference']])
    require(reference['status'] == 'cv_dense_pixels_verified' and reference['video_count'] == 150
            and reference['windows'] == 143470 and set(reference['videos']) == set(sources), 'Incomplete canonical dense reference')
    deep_audit = inputs.read(rooted(root, baseline['audit_path']), baseline['audit_sha256'])
    require(deep_audit['status'] == 'audit_passed', 'Deep baseline audit not passed')
    audited_fits = {r['fold']: r for r in deep_audit['fits']}
    require([f['fold'] for f in folds] == [f['fold'] for f in baseline['folds']] == list(range(1, 6)), 'Five fixed fold identities required')
    origins = pipeline['fold_artifact_origins']
    require([r['fold'] for r in origins] == list(range(1, 6)), 'Five pipeline origins required')
    deep_all, linear_all, fold_rows, curves = [], [], [], []
    reference_root = rooted(root, plan['reference_roots']['local'])
    for fold, old, origin in zip(folds, baseline['folds'], origins):
        f = fold['fold']; inner_name = 'f%02d-inner' % f; outer_name = 'f%02d-outer' % f; dense_name = 'f%02d-dense' % f
        require(origin['outer'] == outer_name and origin['dense'] == dense_name, 'Pipeline fold source mapping differs')
        if str(f) in plan['reused_inner']:
            reused = plan['reused_inner'][str(f)]; inner_dir = reference_root / reused['item']
            files = closed_stage(inputs, reference_root, reused['item'], 'fit_closed', expected_sha=reused['closure_sha256'])
            require(origin['inner'] == plan['reference_roots']['cloud'] + '/' + reused['item']
                    and reuse_receipt['items'][str(f)]['item'] == reused['item']
                    and reuse_receipt['items'][str(f)]['selected_epochs'] == reused['selected_epochs']
                    and reuse_receipt['items'][str(f)]['closure_sha256'] == reused['closure_sha256']
                    and reuse_receipt['items'][str(f)]['source_profile_sha256'] ==
                        files[reused['item'] + '/profile.json']['sha256'], 'Reused inner mapping differs')
            E = reused['selected_epochs']; reused_fit = True
        else:
            inner_dir = run / inner_name; files = closed_stage(inputs, run, inner_name, 'fit_closed', master)
            require(origin['inner'] == inner_name, 'Fresh inner mapping differs')
            E = inputs.read(inner_dir / 'training-result.json')['selected_epochs']; reused_fit = False
        inner, epoch_rows, _ = fit_review(inputs, inner_dir, f, 'inner_select', fold['inner_train'], E, files,
                                        plan['expected_initial_state_sha256'], None if reused_fit else fingerprint,
                                        fold['inner_val']['clip_ids'])
        curves.extend(dict(row, reused_fit=reused_fit) for row in epoch_rows)
        files = closed_stage(inputs, run, outer_name, 'fit_closed', master)
        outer, epoch_rows, fit = fit_review(inputs, run / outer_name, f, 'fixed_outer', fold['outer_train'], E, files,
                                          plan['expected_initial_state_sha256'], fingerprint)
        curves.extend(dict(row, reused_fit=False) for row in epoch_rows)
        files = closed_stage(inputs, run, dense_name, 'dense_closed', master)
        require(dense_name + '/result.json' in files and dense_name + '/window-scores.csv' in files, 'Dense result absent from closure')
        new = inputs.read(run / dense_name / 'result.json')
        require(new['status'] == 'outer_fold_complete' and new['fold'] == f and new['selected_epochs'] == E
                and new['spec_id'] == 'EXP-022' and new['head'] == 'linear'
                and new['score_kind'] == 'max_saved_FP32_window_probability_ties_retained'
                and new['linear_plan_sha256'] == PLAN_SHA and new['plan_sha256'] == plan['base_plan_sha256']
                and new['input_fingerprints'] == fingerprint and new['checkpoint_sha256'] == fit['selected_checkpoint']['sha256'],
                'Dense model/plan/checkpoint provenance differs')
        raw = inputs.read(rooted(root, old['result_path']), old['result_sha256'])
        require(deep_audit['source_sha256'][old['result_path']] == old['result_sha256']
                and audited_fits[f]['artifact_directory'] == old['artifact_directory']
                and audited_fits[f]['selected_epochs'] == old['selected_epochs']
                and raw['status'] == 'outer_fold_complete' and raw['fold'] == f
                and raw['selected_epochs'] == old['selected_epochs'] and raw['scores'] == old['scores']
                and raw['metrics'] == old['metrics'] and raw['window_scores_sha256'] == old['window_scores_sha256'],
                'Audited original deep predictions differ from frozen baseline')
        for result in (new, raw):
            require(result['threshold'] == .5 and result['aggregation'] == 'max'
                    and result['outer_selection_use'] is False and result['independent_test_used'] is False
                    and result['pixel_reference_sha256'] == fingerprint[plan['dense_reference']], 'Dense evaluation rule/reference differs')
            ids = [r['unit_id'] for r in result['scores']]
            require(len(ids) == len(set(ids)) and set(ids) == set(fold['outer_eval']['video_ids']), 'Fold-held-out identity differs')
            for row in result['scores']:
                source = sources[row['unit_id']]
                require(row['fold'] == f and row['label'] == int(source['video_label'] == 'pos')
                        and row['group_id'] == source['group_id'], 'Dense video label/group/fold identity differs')
        old_windows_path = rooted(root, old['result_path']).parent / 'window-scores.csv'
        require(deep_audit['source_sha256'][old_windows_path.relative_to(root).as_posix()] == old['window_scores_sha256'],
                'Original deep window predictions are outside the passed audit')
        window_ids, old_stats = window_review(inputs, old_windows_path, raw, reference)
        _, new_stats = window_review(inputs, run / dense_name / 'window-scores.csv', new, reference, window_ids)
        deep_all.extend(raw['scores']); linear_all.extend(new['scores'])
        row = dict(fold=f, linear_selected_E=E, deep_selected_E=old['selected_epochs'], inner_reused=reused_fit,
            deep_inner_completed_epochs=20, deep_inner_optimizer_updates=audited_fits[f]['inner_updates'],
            deep_outer_completed_epochs=old['selected_epochs'], deep_outer_optimizer_updates=audited_fits[f]['outer_updates'],
            inner_completed_epochs=inner['completed_epochs'], inner_optimizer_updates=inner['optimizer_updates'],
            outer_completed_epochs=outer['completed_epochs'], outer_optimizer_updates=outer['optimizer_updates'],
            windows=new_stats['n'], P=new['metrics']['P'], N=new['metrics']['N'])
        for arm, result, window_stats in [('deep', raw, old_stats), ('linear', new, new_stats)]:
            row.update({arm + '_' + k: result['metrics'][k] for k in ('TP', 'FP', 'FN', 'TN', 'Recall', 'FPR', 'AUC')})
            row[arm + '_video_probabilities'] = probability_stats([r['score'] for r in result['scores']])
            row[arm + '_window_probabilities'] = window_stats
        fold_rows.append(row)
    require(len(linear_all) == len({r['unit_id'] for r in linear_all}) == 150
            and {r['unit_id'] for r in linear_all} == set(sources)
            and Counter(r['label'] for r in linear_all) == Counter({1:113, 0:37})
            and sum(r['windows'] for r in linear_all) == 143470, 'Incomplete full five-fold video/window coverage')
    result, pairs = paired_comparison(deep_all, linear_all)
    require(len(result['negative_groups']) == 9, 'Expected nine negative session groups')
    for arm in ('deep', 'linear'):
        require(result[arm + '_macro_negative_group_FPR'] == controller_comparison[arm + '_macro_negative_group_FPR'],
                'Controller negative-session macro FPR differs')
    require(result['improved_negative_groups'] == controller_comparison['improved_negative_groups'], 'Controller improved negative groups differ')
    for arm in ('deep', 'linear'):
        for key in ('TP', 'FP', 'FN', 'TN', 'P', 'N'):
            require(result[arm][key] == controller_comparison[arm][key], 'Controller confusion differs')
        aucs = [row[arm + '_AUC'] for row in fold_rows]
        require(aucs == controller_comparison[arm]['per_fold_auc'], 'Controller per-fold saved-probability AUC differs')
        result[arm].update(per_fold_auc=aucs, mean_fold_auc=statistics.mean(aucs), sample_std_fold_auc=statistics.stdev(aucs),
                           pooled_auc=None, pooled_auc_reason='Different outer models; no pooled cross-fold AUC.')
    require(result['supports_overall_head_simplification_video_improvement'] == pipeline['scientific_gate_passed']
            == controller_comparison['supports_overall_head_simplification_video_improvement'], 'Controller scientific gate differs')
    regressions = []
    for row in fold_rows:
        for key, higher_worse in [('FP', True), ('FN', True), ('FPR', True), ('Recall', False), ('AUC', False)]:
            old, new = row['deep_' + key], row['linear_' + key]
            if (new > old) if higher_worse else (new < old):
                regressions.append(dict(fold=row['fold'], metric=key, deep=old, linear=new, delta=new-old))
    result.update(status='full', spec_id='EXP-022', run_id=run.name, independent_test=False, no_automatic_followup=True,
        scope='Single-seed session-group development CV on historically reused data; not unseen-camera or external testing.',
        dataset=dict(videos=150, P=113, N=37, physical_windows=143470, negative_session_groups=9), folds=fold_rows,
        fold_regressions=regressions, selected_E=[r['linear_selected_E'] for r in fold_rows],
        new_fit_completed_epochs=sum(r['outer_completed_epochs'] + (0 if r['inner_reused'] else r['inner_completed_epochs']) for r in fold_rows),
        new_fit_optimizer_updates=sum(r['outer_optimizer_updates'] + (0 if r['inner_reused'] else r['inner_optimizer_updates']) for r in fold_rows),
        metric_definition='Per-fold AUC of max saved FP32 window probability; exact ties receive half credit. Fixed max/0.5 confusion.',
        regression_definition='All newly wrong video decisions; every fold with increased FP/FN/FPR or decreased Recall/AUC; all worsened negative groups.',
        checkpoint_verification='Closure SHA relationships and local sizes only; collector byte verification inherited. No model load or checkpoint rehash.',
        provenance=dict(plan_sha256=PLAN_SHA, deep_baseline_sha256=plan['deep_baseline_sha256'], original_deep_audit_sha256=baseline['audit_sha256'],
            input_sha256=inputs.hashes, checkpoint_manifest_records=inputs.checkpoints,
            command=argv or [], report_source_sha256=sha(Path(__file__)), environment=dict(python=sys.version, platform=platform.platform())))
    inputs.unchanged()
    require(result['new_fit_completed_epochs'] <= plan['maximum_new_epochs']
            and result['new_fit_optimizer_updates'] <= plan['maximum_new_updates'], 'Completed work exceeds the frozen fit schedule')
    # Only accepted full coverage reaches this point. All outputs are separate.
    out.mkdir(parents=True, exist_ok=False)
    _csv(out / 'video-pairs.csv', pairs, list(pairs[0]))
    _csv(out / 'negative-groups.csv', result['negative_groups'], list(result['negative_groups'][0]))
    _csv(out / 'regressions.csv', result['regression_cases'], list(pairs[0]))
    _csv(out / 'fold-regressions.csv', regressions, ['fold', 'metric', 'deep', 'linear', 'delta'])
    _csv(out / 'learning-curves.csv', curves, list(curves[0]))
    flat_folds = []
    for fold in fold_rows:
        flat = {k: v for k, v in fold.items() if not isinstance(v, dict)}
        for k, distribution in fold.items():
            if isinstance(distribution, dict):
                flat.update({k + '_' + field: value for field, value in distribution.items()})
        flat_folds.append(flat)
    _csv(out / 'folds.csv', flat_folds, list(flat_folds[0]))
    (out / 'report_exp022.py').write_bytes(Path(__file__).read_bytes())
    result['outputs_sha256'] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    with (out / 'metrics.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / 'data/metadata/EXP-022/linear-plan-v1')
    parser.add_argument('--run', type=Path, default=ROOT / 'data/results/exp022' / RUN_ID)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = generate(ROOT, args.plan, args.run, args.out, sys.argv if argv is None else argv)
    except Incomplete as error:
        print(json.dumps(dict(status='incomplete_metrics_withheld', error=str(error)), ensure_ascii=True))
        return 2
    print(json.dumps(dict(status=result['status'], linear=result['linear'], deep=result['deep'],
        gate=result['gate_components'], output=str(args.out.resolve())), ensure_ascii=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
