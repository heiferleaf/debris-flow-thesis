"""Post hoc fixed-validation score drift; no model, video, labels or selection changes."""
from __future__ import annotations
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import shutil
import sys

import numpy as np

SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[3]
MODES = ('bce', 'cross', 'same')
BIN_NAMES = ('[0,3)s', '[3,10)s', '>=10s', 'unknown')
LARGE_DRIFT = .25  # Fixed descriptive magnitude in probability units; never a model threshold.
DIAGNOSTIC_VERSION = 2


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def finite(value):
    if isinstance(value, dict):
        for item in value.values(): finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value: finite(item)
    elif isinstance(value, Real) and not np.isfinite(value):
        raise ValueError('Nonfinite numeric input')


class Inputs:
    def __init__(self, root):
        self.root, self.records = root, {}

    def read(self, path, expected=None):
        path = Path(path).resolve()
        relative = path.relative_to(self.root).as_posix()
        before = sha(path)
        if expected is not None and before != expected:
            raise ValueError('Frozen input SHA mismatch: ' + relative)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != before:
            raise ValueError('Input changed while reading: ' + relative)
        self.records[relative] = dict(path=relative, sha256=before, size_bytes=len(raw))
        return raw.decode('utf-8-sig')

    def json(self, path, expected=None):
        value = json.loads(self.read(path, expected))
        finite(value)
        return value

    def jsonl(self, path, expected=None):
        rows = [json.loads(line) for line in self.read(path, expected).splitlines() if line.strip()]
        finite(rows)
        return rows

    def csv(self, path, expected=None):
        import io
        return list(csv.DictReader(io.StringIO(self.read(path, expected))))

    def unchanged(self):
        for row in self.records.values():
            path = self.root / row['path']
            if path.stat().st_size != row['size_bytes'] or sha(path) != row['sha256']:
                raise ValueError('Input changed during diagnosis: ' + row['path'])


def write_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refuse silently empty diagnostic table')
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def stats(logits, labels):
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    if len(z) != len(y) or not np.isfinite(z).all() or not np.isin(y, (0, 1)).all():
        raise ValueError('Invalid logits/labels for diagnostic statistics')
    p = probability(z)
    pred = z >= 0  # Exactly the registered sigmoid(logit)>=0.5 decision.
    result = dict(n=len(y), P=int((y == 1).sum()), N=int((y == 0).sum()),
                  TP=int(((y == 1) & pred).sum()), FN=int(((y == 1) & ~pred).sum()),
                  FP=int(((y == 0) & pred).sum()), TN=int(((y == 0) & ~pred).sum()),
                  bce_fp64=float(np.logaddexp(0., (1 - 2 * y) * z).mean()) if len(y) else None,
                  mean_score=float(p.mean()) if len(y) else None,
                  mean_logit=float(z.mean()) if len(y) else None)
    result.update({'score_q%02d' % q: float(np.quantile(p, q / 100)) if len(y) else None
                   for q in (0, 10, 25, 50, 75, 90, 100)})
    return result


def probability(logits):
    z = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(z).all():
        raise ValueError('Probability requires finite logits')
    q = np.exp(-np.abs(z))
    # Avoid 1-sigmoid(abs(z)): subtraction rounds small negative-tail values
    # to zero long before exp(z) underflows in FP64.
    return np.where(z >= 0, 1 / (1 + q), q / (1 + q))


def probability_numerical_check():
    z = np.array([-100., -43., -1., 0., 1., 43., 100.], dtype=np.float64)
    actual = probability(z)
    reference = np.array([math.exp(x) / (1 + math.exp(x)) if x < 0
                          else 1 / (1 + math.exp(-x)) for x in z])
    np.testing.assert_allclose(actual, reference, rtol=1e-15, atol=0.)
    if not np.all(actual > 0) or not np.all(np.diff(actual) >= 0):
        raise AssertionError('FP64 sigmoid tail positivity or monotonicity failed')
    if not np.all(np.diff(actual[z < 0]) > 0) or actual[z == 0][0] != .5:
        raise AssertionError('Negative sigmoid tail ordering or zero-logit midpoint failed')
    return dict(status='passed', logits=z.tolist(), stable_fp64_probabilities=actual.tolist(),
                math_reference_probabilities=reference.tolist(), rtol=1e-15, atol=0.,
                note='Positive logits43 and100 both round to1 in FP64; monotonic means nondecreasing.')


def compare_v1(inputs, directory, names, tables):
    """Audit observed output differences without changing or replacing v1 files."""
    manifest = inputs.json(directory / 'output-manifest.json')
    refs = {r['path']: r['sha256'] for r in manifest['artifacts']}
    reports, invariant_mismatches = {}, []
    for name, current in zip(names, tables):
        previous = inputs.csv(directory / name, refs[name])
        if len(previous) != len(current) or set(previous[0]) != set(current[0]):
            raise ValueError('v1/v2 diagnostic row/schema mismatch: ' + name)
        changed = Counter()
        maximum_absolute_difference = defaultdict(float)
        for i, (old, new) in enumerate(zip(previous, current)):
            for field in ('mode', 'completed_epochs', 'clip_id', 'dimension', 'subset'):
                if field in new and old[field] != str(new[field]):
                    raise ValueError('v1/v2 row identity mismatch')
            for field, value in new.items():
                rendered = '' if value is None else str(value)
                if old[field] == rendered:
                    continue
                changed[field] += 1
                if isinstance(value, Real) and not isinstance(value, bool) and old[field]:
                    difference = abs(float(old[field]) - float(value))
                    maximum_absolute_difference[field] = max(maximum_absolute_difference[field], difference)
                is_invariant = ('bce' in field.lower() or field in ('TP', 'FP', 'FN', 'TN')
                    or field.endswith(('_TP', '_FP', '_FN', '_TN')) or 'crossings' in field
                    or ('large' in field and 'ge_0_25' in field) or field == 'crossed_0_5_from_previous')
                if is_invariant:
                    invariant_mismatches.append(dict(table=name, row_zero_based=i, field=field,
                                                      v1=old[field], v2=rendered))
        reports[name] = dict(rows=len(current), changed_cells_by_column=dict(changed),
            maximum_absolute_difference_by_numeric_column=dict(maximum_absolute_difference))
    old_scores = inputs.csv(directory / names[0], refs[names[0]])
    new_scores = tables[0]
    restored = [(old, new) for old, new in zip(old_scores, new_scores)
                if float(old['score']) == 0. and new['score'] > 0.]
    example = min(restored, key=lambda pair: abs(pair[1]['logit'] + 43)) if restored else None
    return dict(v1_directory=directory.relative_to(ROOT).as_posix(),
        policy='v1 preserved byte-for-byte; v2 replaces its probability-tail diagnostics for future use.',
        FP_FN_BCE_crossing_and_large_drift_counts_unchanged=not invariant_mismatches,
        invariant_mismatches=invariant_mismatches, tables=reports,
        negative_tail_clip_epoch_scores_restored_from_zero=len(restored),
        representative_restored_score=({key: example[1][key] for key in ('mode', 'completed_epochs', 'clip_id', 'logit', 'score')}
            if example else None),
        representative_v1_score=float(example[0]['score']) if example else None)


def load_metadata(inputs, pair_dir):
    pair = inputs.json(pair_dir / 'plan.json')
    if (pair.get('spec_id') != 'EXP-018' or pair['epochs'] != 20 or pair['seed'] != 7
            or pair['validation_count'] != 136 or pair['base_plan_path'] != 'data/metadata/EXP-016/training-plan-v2'):
        raise ValueError('Unexpected registered paired plan')
    schedule = inputs.jsonl(pair_dir / 'epochs.jsonl', pair['files']['epochs.jsonl'])
    if len(schedule) != 20 or [r['epoch'] for r in schedule] != list(range(20)):
        raise ValueError('Missing paired-plan epoch')
    base_dir = ROOT / pair['base_plan_path']
    base = inputs.json(base_dir / 'plan.json', pair['base_plan_sha256'])
    assignments = inputs.csv(base_dir / 'assignments.csv', base['files']['assignments.csv'])
    resource = ROOT / base['resource']
    refs = {r['path']: r['sha256'] for r in base['sources']}
    def source(name, method):
        path = resource / name
        return method(path, refs[path.relative_to(ROOT).as_posix()])
    windows = source('windows.jsonl', inputs.jsonl)
    segments = source('segments.csv', inputs.csv)
    videos = source('video-inputs.csv', inputs.csv)
    bank = {r['clip_id']: r for r in windows}
    seg = {(r['unit_id'], r['segment_id']): r for r in segments}
    video = {r['unit_id']: r for r in videos}
    if len(bank) != len(windows) or len(seg) != len(segments) or len(video) != len(videos):
        raise ValueError('Duplicate metadata identity')
    val_assign = [r for r in assignments if r['split'] == 'validation']
    ids = [r['clip_id'] for r in val_assign]
    if ids != pair['validation_clip_ids'] or len(ids) != 136 or len(set(ids)) != 136:
        raise ValueError('Fixed validation identity/order differs from base and pair plans')
    train_groups = {v['group_id'] for v in videos if v['split'] == 'train'}
    if train_groups & {v['group_id'] for v in videos if v['split'] == 'validation'}:
        raise ValueError('Frozen group leakage')
    metadata = []
    for assigned, cid in zip(val_assign, ids):
        w = bank[cid]; s = seg[(w['unit_id'], w['segment_id'])]; v = video[w['unit_id']]
        if (w['split'] != 'validation' or assigned['epoch'] != '-1'
                or any(w[k] != assigned[k] for k in ('unit_id', 'kind', 'split'))
                or any(w[k] != s[k] for k in ('unit_id', 'segment_id', 'kind', 'split', 'group_id'))
                or any(w[k] != v[k] for k in ('unit_id', 'split', 'group_id'))):
            raise ValueError('Metadata label/group/source identity mismatch')
        pts = np.asarray(w['source_pts_seconds'], dtype=np.float64)
        ix = np.asarray(w['frame_indices'])
        if (len(pts) != 16 or len(ix) != 16 or not np.isfinite(pts).all()
                or not np.issubdtype(ix.dtype, np.integer) or np.any(np.diff(ix) <= 0)
                or np.any(np.diff(pts) <= 0) or not 2.9 - 1e-9 <= pts[-1] - pts[0] <= 3.1 + 1e-9
                or pts[0] < float(s['sampling_start_sec']) - 1e-9
                or pts[-1] >= float(s['sampling_end_sec'])
                or not any(ix[0] >= a and ix[-1] < b for a, b in json.loads(v['clean_runs_json']))):
            raise ValueError('Invalid supervised window/quality closure')
        start, end, reason = None, None, ''
        try:
            a, b = float(s['annotation_start_sec']), float(s['annotation_end_sec'])
            if not np.isfinite([a, b]).all() or not a < b or pts[0] < a - 1e-9 or pts[-1] >= b:
                raise ValueError()
            start, end = a, b
        except (ValueError, TypeError, KeyError):
            reason = 'Original annotation boundary absent, nonfinite or incompatible; no quality-boundary fallback'
        elapsed = float(pts[0] - start) if start is not None else None
        remaining = float(end - pts[-1]) if end is not None else None
        bucket = ('not_positive' if w['kind'] == 'neg' else 'unknown' if elapsed is None
                  else BIN_NAMES[0] if elapsed < 3 else BIN_NAMES[1] if elapsed < 10 else BIN_NAMES[2])
        metadata.append(dict(clip_id=cid, unit_id=w['unit_id'], group_id=w['group_id'],
            segment_id=w['segment_id'], label=int(w['kind'] == 'pos'),
            first_frame_index=int(ix[0]), last_frame_index=int(ix[-1]),
            first_frame_pts_sec=float(pts[0]), last_frame_pts_sec=float(pts[-1]),
            annotation_start_sec=start, annotation_end_sec=end,
            from_annotation_start_sec=elapsed, to_annotation_end_sec=remaining,
            nearest_annotation_boundary_sec=min(elapsed, remaining) if elapsed is not None else None,
            positive_start_bin=bucket, boundary_unknown_reason=reason,
            source_origin=s['origin'], supervision_timebase_status=v['supervision_timebase_status']))
    labels = np.array([r['label'] for r in metadata], dtype=np.int64)
    if Counter(labels) != Counter({1: 96, 0: 40}) or len({r['unit_id'] for r in metadata}) != 31:
        raise ValueError('Expected 31 validation videos and 96P/40N fixed clips')
    for ep, row in enumerate(schedule):
        original = [r['clip_id'] for r in assignments if r['split'] == 'train' and int(r['epoch']) == ep]
        ordered = [cid for batch in row['batches'] for cid in batch['clip_ids']]
        if len(row['batches']) != 21 or any(len(b['clip_ids']) != 24 for b in row['batches']):
            raise ValueError('Effective batch schedule changed')
        if len(ordered) != len(set(ordered)) or len(ordered) != 504 or set(ordered) != set(original):
            raise ValueError('Paired training endpoint coverage mismatch')
    return pair, schedule, metadata, labels


def load_runs(inputs, run, pair_dir, pair, schedule, metadata, labels):
    result, histories, matrices = {}, {}, {}
    ids = [r['clip_id'] for r in metadata]
    pair_sha = sha(pair_dir / 'plan.json')
    for mode in MODES:
        directory = run / mode
        r = inputs.json(directory / 'training-result.json')
        p = inputs.json(directory / 'profile.json')
        if r['profile'] != p or r['initial_state_sha256'] != p['initial_state_sha256']:
            raise ValueError('Training result/profile initialization mismatch')
        if (r['status'] != 'supervised_complete_dense_evaluation_pending' or r['completed_epochs'] != 20
                or r['optimizer_steps'] != 420 or p['smoke'] or p['mode'] != 'rgb_zero'
                or p['objective_mode'] != mode or p['seed'] != 7 or p['requested_epochs'] != 20
                or p['requested_optimizer_updates'] != 420 or p['pair_plan_sha256'] != pair_sha
                or p['plan_sha256'] != pair['base_plan_sha256']
                or p['pair_schedule_sha256'] != pair['files']['epochs.jsonl']):
            raise ValueError('Run identity/budget/plan profile mismatch')
        inputs.json(directory / 'paired-plan.json', pair_sha)
        inputs.jsonl(directory / 'paired-epochs.jsonl', pair['files']['epochs.jsonl'])
        history = inputs.jsonl(directory / 'epochs.jsonl')
        if len(history) != 20 or [h['completed_epochs'] for h in history] != list(range(1, 21)):
            raise ValueError('Incomplete twenty-epoch training history')
        logits = []
        for ep, h in enumerate(history, 1):
            expected_order = [c for b in schedule[ep - 1]['batches'] for c in b['clip_ids']]
            if (h['train_clip_ids'] != expected_order or h['samples'] != 504 or h['optimizer_steps'] != 21
                    or h['total_updates'] != 21 * ep or h['train_pair_count'] != 80
                    or h['validation_P'] != 96 or h['validation_N'] != 40 or h['objective_mode'] != mode):
                raise ValueError('Training history input/order/count mismatch')
            predictions = inputs.json(directory / ('validation-epoch-%02d.json' % ep))
            if len(predictions) != 136 or [row['clip_id'] for row in predictions] != ids:
                raise ValueError('Prediction identity/order/coverage mismatch')
            if any(type(row['label']) is not int or row['label'] != int(y)
                   or not isinstance(row['logit'], Real) or isinstance(row['logit'], bool)
                   for row, y in zip(predictions, labels)):
                raise ValueError('Prediction label or scalar logit mismatch')
            z = np.array([row['logit'] for row in predictions], dtype=np.float64)
            recomputed = stats(z, labels)
            # Training used FP32 reductions. This is a consistency gate, not the
            # 1e-8 FP64 between-condition comparison rule from the experiment card.
            if abs(recomputed['bce_fp64'] - h['validation_unweighted_bce']) > 1e-5 * max(1., recomputed['bce_fp64']):
                raise ValueError('Prediction FP64 BCE incompatible with FP32 training log')
            for y, key in ((0, 'validation_neg_bce'), (1, 'validation_pos_bce')):
                value = stats(z[labels == y], labels[labels == y])['bce_fp64']
                if abs(value - h[key]) > 1e-5 * max(1., value):
                    raise ValueError('Per-class prediction BCE incompatible with training log')
            logits.append(z)
        best = min(range(20), key=lambda i: history[i]['validation_unweighted_bce']) + 1
        if (r['best_epoch'] != best or r['selected_checkpoint']['completed_epochs'] != best
                or r['best_validation_bce'] != history[best - 1]['validation_unweighted_bce']):
            raise ValueError('Recorded selected epoch does not follow frozen early-tie rule')
        running_best = 0
        for i, h in enumerate(history):
            if h['validation_unweighted_bce'] < history[running_best]['validation_unweighted_bce']:
                running_best = i
            if h['best_epoch'] != running_best + 1:
                raise ValueError('History best-epoch trace mismatch')
        result[mode], histories[mode], matrices[mode] = r, history, np.stack(logits)
    if len({r['initial_state_sha256'] for r in result.values()}) != 1:
        raise ValueError('Three conditions did not share initialization')
    return result, histories, matrices


def masks(metadata, labels):
    result = [('all', 'all', np.ones(len(labels), dtype=bool))]
    result += [('class', name, labels == y) for y, name in ((0, 'negative'), (1, 'positive'))]
    for group in sorted({r['group_id'] for r in metadata}):
        mask = np.array([r['group_id'] == group for r in metadata])
        result.append(('group', group, mask))
        for y, name in ((0, 'negative'), (1, 'positive')):
            result.append(('group_class', group + '|' + name, mask & (labels == y)))
    for bucket in BIN_NAMES:
        result.append(('positive_annotation_start_bin', bucket,
                       np.array([r['positive_start_bin'] == bucket for r in metadata])))
    return result


def drift_stats(previous, current, labels):
    delta = current - previous
    down, up = (previous >= .5) & (current < .5), (previous < .5) & (current >= .5)
    return dict(n=len(delta), mean_delta_score=float(delta.mean()) if len(delta) else None,
                abs_delta_q50=float(np.quantile(np.abs(delta), .5)) if len(delta) else None,
                abs_delta_q90=float(np.quantile(np.abs(delta), .9)) if len(delta) else None,
                abs_delta_max=float(np.abs(delta).max()) if len(delta) else None,
                large_abs_delta_ge_0_25=int((np.abs(delta) >= LARGE_DRIFT).sum()),
                crossings_0_5=int((down | up).sum()), downward_crossings=int(down.sum()),
                upward_crossings=int(up.sum()), new_FN=int((down & (labels == 1)).sum()),
                recovered_FN=int((up & (labels == 1)).sum()), new_FP=int((up & (labels == 0)).sum()),
                recovered_FP=int((down & (labels == 0)).sum()))


def tabulate(results, histories, matrices, metadata, labels):
    clip_rows, aggregates, transitions, clip_summary, selected_stats, summaries = [], [], [], [], [], {}
    subsets = masks(metadata, labels)
    for mode in MODES:
        z = matrices[mode]; p = probability(z); best = results[mode]['best_epoch']
        for ep in range(20):
            for i, meta in enumerate(metadata):
                previous = float(p[ep - 1, i]) if ep else None
                row = dict(mode=mode, completed_epochs=ep + 1, selected_epoch=best, **meta)
                row.update(logit=float(z[ep, i]), score=float(p[ep, i]),
                    unweighted_bce_fp64=float(np.logaddexp(0., (1 - 2 * labels[i]) * z[ep, i])),
                    predicted_positive=int(z[ep, i] >= 0), error=int((z[ep, i] >= 0) != labels[i]),
                    delta_score_previous_epoch=float(p[ep, i] - previous) if ep else None,
                    crossed_0_5_from_previous=int((p[ep, i] >= .5) != (previous >= .5)) if ep else None,
                    large_abs_delta_ge_0_25=int(abs(p[ep, i] - previous) >= LARGE_DRIFT) if ep else None)
                clip_rows.append(row)
            for dimension, subset, mask in subsets:
                row = dict(mode=mode, completed_epochs=ep + 1, dimension=dimension, subset=subset)
                row.update(stats(z[ep, mask], labels[mask]))
                row.update(train_weighted_bce=histories[mode][ep]['train_weighted_bce'],
                           train_pair_mean=histories[mode][ep]['train_pair_mean'],
                           is_selected_epoch=ep + 1 == best)
                aggregates.append(row)
                if ep:
                    transitions.append(dict(mode=mode, from_epoch=ep, to_epoch=ep + 1,
                        dimension=dimension, subset=subset, **drift_stats(p[ep - 1, mask], p[ep, mask], labels[mask])))
        for dimension, subset, mask in subsets:
            a, b = stats(z[best - 1, mask], labels[mask]), stats(z[-1, mask], labels[mask])
            row = dict(mode=mode, selected_epoch=best, final_epoch=20, dimension=dimension, subset=subset)
            row.update({'selected_' + k: v for k, v in a.items()})
            row.update({'final_' + k: v for k, v in b.items()})
            row.update(drift_stats(p[best - 1, mask], p[-1, mask], labels[mask]))
            selected_stats.append(row)
        for i, meta in enumerate(metadata):
            adj = np.diff(p[:, i]); crosses = (p[1:, i] >= .5) != (p[:-1, i] >= .5)
            clip_summary.append(dict(mode=mode, **meta, selected_epoch=best,
                selected_score=float(p[best - 1, i]), final_score=float(p[-1, i]),
                final_minus_selected_score=float(p[-1, i] - p[best - 1, i]),
                selected_FN=int(labels[i] == 1 and z[best - 1, i] < 0),
                final_FN=int(labels[i] == 1 and z[-1, i] < 0),
                selected_FP=int(labels[i] == 0 and z[best - 1, i] >= 0),
                final_FP=int(labels[i] == 0 and z[-1, i] >= 0),
                adjacent_crossings_0_5=int(crosses.sum()), large_adjacent_abs_delta_ge_0_25=int((np.abs(adj) >= LARGE_DRIFT).sum()),
                largest_adjacent_abs_delta=float(np.abs(adj).max()),
                largest_adjacent_delta_to_epoch=int(np.argmax(np.abs(adj))) + 2,
                min_score=float(p[:, i].min()), max_score=float(p[:, i].max())))
        fn_counts = [stats(row, labels)['FN'] for row in z]
        most_fn = int(np.argmax(fn_counts))
        summaries[mode] = dict(selected_epoch=best, selected=stats(z[best - 1], labels),
            final=stats(z[-1], labels), selected_to_final=drift_stats(p[best - 1], p[-1], labels),
            largest_FN_epoch=most_fn + 1, largest_FN_count=fn_counts[most_fn],
            largest_FN_group_decomposition={subset: stats(z[most_fn, mask], labels[mask])
                for dimension, subset, mask in subsets if dimension == 'group'},
            largest_FN_positive_annotation_bin_decomposition={subset: stats(z[most_fn, mask], labels[mask])
                for dimension, subset, mask in subsets if dimension == 'positive_annotation_start_bin'},
            total_adjacent_crossings_0_5=int(((p[1:] >= .5) != (p[:-1] >= .5)).sum()),
            clips_ever_crossing_0_5=int(np.any((p[1:] >= .5) != (p[:-1] >= .5), axis=0).sum()),
            total_large_adjacent_abs_delta_ge_0_25=int((np.abs(np.diff(p, axis=0)) >= LARGE_DRIFT).sum()))
    return clip_rows, aggregates, transitions, clip_summary, selected_stats, summaries


def plot(matrices, metadata, labels, results, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'savefig.dpi': 200})
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.5), sharex=True)
    x = np.arange(1, 21)
    titles = {'bce': 'A 普通BCE / BCE', 'cross': 'B 跨场次排序 / Cross-group', 'same': 'C 同场次排序 / Same-group'}
    colors = {'[0,3)s': '#c66e27', '[3,10)s': '#8663aa', '>=10s': '#2f7e9e', 'unknown': '#999999'}
    for col, mode in enumerate(MODES):
        p = probability(matrices[mode]); top, bottom = axes[:, col]
        for y, color, name in ((1, '#207c64', '正类 Positive (n=96)'), (0, '#c15855', '负类 Negative (n=40)')):
            values = p[:, labels == y]
            top.plot(x, np.median(values, axis=1), color=color, label=name, lw=1.8)
            top.fill_between(x, np.quantile(values, .1, axis=1), np.quantile(values, .9, axis=1), color=color, alpha=.15)
        top.axhline(.5, color='#777777', ls=':', lw=1, label='固定阈值 / Fixed 0.5')
        top.set_title(titles[mode]); top.set_ylim(-.03, 1.03)
        top.set_ylabel('预测概率 / Score (0–1)')
        top.legend(fontsize=8, loc='upper right', framealpha=.85)
        bottom.plot(x, ((p < .5) & (labels == 1)).sum(axis=1), color='#252525', lw=2, label='全部正类 / All positive (n=96)')
        for bucket in BIN_NAMES:
            mask = np.array([m['positive_start_bin'] == bucket for m in metadata])
            if mask.any():
                bottom.plot(x, (p[:, mask] < .5).sum(axis=1), color=colors[bucket], lw=1.5,
                            label='%s (n=%d)' % (bucket, mask.sum()))
        best = results[mode]['best_epoch']
        for axis in (top, bottom):
            axis.axvline(best, color='#4f555e', ls='--', alpha=.65, lw=1)
            axis.set_xlim(.5, 20.5); axis.set_xticks([1, 5, 10, 15, 20]); axis.grid(alpha=.16)
        bottom.set_ylim(-2, 99); bottom.set_ylabel('正类漏报 / False negatives (clips)')
        bottom.set_xlabel('完成训练轮次 / Completed epochs')
        bottom.legend(fontsize=8, loc='upper right', framealpha=.85)
        top.text(.02, .05, '预定选中轮 / Selected: %d' % best, transform=top.transAxes, fontsize=9)
    fig.suptitle('EXP-018：固定开发验证片段的分数漂移 / Score drift on fixed development validation', fontsize=14, y=.98)
    fig.text(.02, .055, '上：分数中位数及10%–90%分位带；下：按原正类标注起点距离分解漏报。虚线标示已选中轮次。', fontsize=10)
    fig.text(.02, .025, 's = first sampled frame minus original positive annotation start; not visual onset. Post hoc diagnosis; fixed 0.5; no independent test.', fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, .955))
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(out / ('fixed-validation-score-drift.' + suffix))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, default=ROOT / 'data/results/exp018/paired-s7-20261101-v1')
    ap.add_argument('--plan', type=Path, default=ROOT / 'data/metadata/EXP-018/paired-plan-v1')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--no-plot', action='store_true')
    a = ap.parse_args()
    run, pair_dir = a.run.resolve(), a.plan.resolve()
    out = a.out.resolve() if a.out else run / 'score-drift-v2'
    out.relative_to(run)
    if out.exists(): raise FileExistsError('Refuse to overwrite existing score-drift output')
    inputs = Inputs(ROOT)
    source_sha = sha(SOURCE)
    numerical_check = probability_numerical_check()
    pair, schedule, metadata, labels = load_metadata(inputs, pair_dir)
    results, histories, matrices = load_runs(inputs, run, pair_dir, pair, schedule, metadata, labels)
    tables = tabulate(results, histories, matrices, metadata, labels)
    names = ('clip-epoch-scores.csv', 'epoch-subset-statistics.csv', 'adjacent-epoch-drift.csv',
             'clip-drift-summary.csv', 'selected-versus-final-subsets.csv')
    previous = run / 'score-drift-v1'
    comparison = compare_v1(inputs, previous, names, tables[:-1]) if previous.is_dir() else None
    inputs.unchanged()
    if sha(SOURCE) != source_sha: raise ValueError('Diagnostic source changed before output')
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in zip(names, tables[:-1]): write_csv(out / name, rows)
    write_csv(out / 'validation-clip-metadata.csv', metadata)
    unknown = [r['clip_id'] for r in metadata if r['boundary_unknown_reason']]
    summary = dict(status='post_hoc_score_drift_complete', diagnostic_version=DIAGNOSTIC_VERSION,
        probability_formula='q=exp(-abs(z)); where(z>=0,1/(1+q),q/(1+q))',
        independent_test=False, model_executed=False,
        fixed_score_threshold=.5, descriptive_large_absolute_probability_change=LARGE_DRIFT,
        modes=tables[-1], fixed_validation=dict(clips=136, P=96, N=40, videos=31,
            groups=len({m['group_id'] for m in metadata}),
            positive_annotation_start_bins=dict(Counter(m['positive_start_bin'] for m in metadata if m['label'] == 1)),
            annotation_boundary_unknown_clip_ids=unknown),
        output_row_counts={name: len(rows) for name, rows in zip(names, tables[:-1])},
        limitations=['All twenty epochs are post hoc development-set diagnostics, not independent testing.',
            'Original annotation boundaries are used; no inference of actual visual onset or label correction.',
            'Probability changes and group association cannot establish LR, BN, background or other causal explanations.',
            'No new threshold/epoch/seed is selected. FP64 BCE recomputation is separate from FP32 log consistency tolerance.',
            'Prediction hashes freeze the observed local artifacts; metadata hashes are checked against the frozen base plan.'])
    write_json(out / 'summary.json', summary)
    write_json(out / 'probability-numerical-check.json', numerical_check)
    if comparison is not None: write_json(out / 'v1-v2-comparison.json', comparison)
    write_json(out / 'input-manifest.json', dict(argv=sys.argv, source_sha256=source_sha,
        numpy_version=np.__version__, inputs=sorted(inputs.records.values(), key=lambda r: r['path']),
        checks=['metadata SHA against frozen base plan', 'paired plan and per-run snapshot SHA',
            'shared initialization/run/budget/profile identity', '20 epochs x136 ordered binary labels and finite logits',
            'each epoch504 original training IDs in frozen order', 'FP64 per-class/overall BCE versus FP32 log (relative tolerance1e-5)',
            'unchanged inputs before and after generation'], prediction_hash_policy='Observed local SHA, checked unchanged; no fabricated historical checksum'))
    shutil.copyfile(SOURCE, out / SOURCE.name)
    (out / 'README.md').write_text(
        '# EXP-018 事后分数漂移诊断\n\n'
        '三组同一136片段、20轮预测的完整描述统计；不运行模型，不读原视频，不改选轮、阈值或标签。'
        '该验证集用于过开发选择，全部结果不是独立测试。\n\n'
        '`clip-epoch-scores.csv`逐片段逐轮分数；`epoch-subset-statistics.csv`按类别、场次及原正标注起点距离分组的BCE/FP/FN/分位数；'
        '`adjacent-epoch-drift.csv`相邻轮分数变化、0.5上下跨越和绝对变化≥0.25计数；'
        '`clip-drift-summary.csv`每片段的全部轮次漂移及选中/末轮对照；`selected-versus-final-subsets.csv`完整分组对照。\n\n'
        '时间单位秒，距离按窗口首帧PTS减原`annotation_start_sec`计算，末端距离用原`annotation_end_sec`减末帧PTS；'
        '不以质量排除或cut边界替代。正片段固定分桶[0,3)、[3,10)、≥10秒；缺失/不兼容留空并记原因。'
        '这些距离不等于真实可见起点。0.25仅为描述性概率变化幅度，不是新分类阈值。\n\n'
        'BCE以FP64稳定公式重算；与原FP32日志一致性检查容差为1e-5×max(1,BCE)，不与卡片的组间FP64比较1e-8混用。'
        '各场次/片段都纳入输出，无按已知异常数字挑选样本。最大FN轮仅为完整轨迹的确定性描述，不作为新选轮。\n\n'
        '元数据核对冻结计划SHA；预测记录观察到的本地SHA，并在生成前后复核不变；未声称拥有训练时未保存的预测历史校验和。'
        '源码快照、命令及所有读取文件SHA见`input-manifest.json`。\n\n'
        'v2替代v1的概率尾部诊断：v1对负logit采用`1-sigmoid(abs(z))`，会因FP64消减把约-43的非零概率误记为0；'
        'v2统一使用`q=exp(-abs(z)); where(z>=0,1/(1+q),q/(1+q))`，统计函数复用同一实现。'
        'v1原件保留，不覆盖；后续引用应使用v2。`probability-numerical-check.json`保存[-100,-43,-1,0,1,43,100]'
        '非零尾部、单调性及math稳定参考核对。正logit43和100在FP64均舍入为1，单调性指非递减。\n\n'
        '与v1的全部表格逐单元比较见`v1-v2-comparison.json`，其中单独核对FP/FN、BCE、跨0.5次数及绝对漂移≥0.25计数；'
        '概率均值/分位数和连续漂移量允许出现已记录的浮点精度差异，原始logit与预测文件不变。\n\n'
        '复算：`.venv/Scripts/python.exe experiments/runspecs/EXP-018/diagnose_score_drift.py --out '
        'data/results/exp018/paired-s7-20261101-v1/score-drift-v3`；目标已存在则拒绝覆盖。\n', encoding='utf-8')
    if not a.no_plot: plot(matrices, metadata, labels, results, out)
    inputs.unchanged()
    if sha(SOURCE) != source_sha or sha(out / SOURCE.name) != source_sha:
        raise ValueError('Diagnostic source changed during generation')
    artifacts = [dict(path=p.relative_to(out).as_posix(), sha256=sha(p), size_bytes=p.stat().st_size)
                 for p in sorted(out.iterdir()) if p.is_file()]
    write_json(out / 'output-manifest.json', dict(source_sha256=source_sha, artifacts=artifacts))
    print(json.dumps(dict(status=summary['status'], rows=summary['output_row_counts'],
                         modes={m: dict(selected_epoch=results[m]['best_epoch'], selected_FN=tables[-1][m]['selected']['FN'],
                             final_FN=tables[-1][m]['final']['FN'], max_FN=tables[-1][m]['largest_FN_count'],
                             max_FN_epoch=tables[-1][m]['largest_FN_epoch']) for m in MODES})))


if __name__ == '__main__':
    main()
