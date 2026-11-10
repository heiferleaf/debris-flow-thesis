"""Offline EXP-019 evidence audit. No torch, model execution or media decoding.

Selection uses the saved FP32 validation reduction, earliest exact minimum.
Stable Python float64 BCE is an independent consistency check at absolute2e-6,
not a CUDA reproduction claim or a between-branch1e-8 comparison gate.
Exit0 means evidence integrity passed, not that the scientific gate was positive.
Exit2 is incomplete; exit1 is invalid. All three preserve a new audit directory.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import re
import sys
import time

SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.temporal_order import MODES, window_order_record

PLAN = 'data/metadata/EXP-019/order-plan-v1'
PLAN_SHA = '0e1e697aa4959e69fa40b4a33166370f57a1298af94883e50301cb4e6eee944a'
DEPLOYMENT = 'data/metadata/EXP-019/deployment-v2'
ACCEPTANCE = 'data/metadata/EXP-019/order-cache-acceptance-v1/acceptance.json'
REFERENCE = 'data/process/EXP-016/dense-pixel-verification-v3/result.json'
PREFLIGHT = 'data/process/EXP-019/cloud-preflight-v1/preflight.json'
BCE_ATOL = 2e-6
SCOPE = ('Reused development validation, not independent testing or generalization. '
         'The32567 overlapping windows are not independent samples. '
         'Permutation also changes local adjacency and K400 pretraining compatibility; '
         'this audit does not isolate intrinsic debris-flow dynamics or span effects.')


class Incomplete(ValueError):
    def __init__(self, missing):
        self.missing = list(map(str, missing))
        super().__init__('Required evidence unavailable: '+', '.join(self.missing))


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): result.update(block)
    return result.hexdigest()


def _object(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, 'Duplicate JSON key: '+k)
        result[k] = v
    return result


def _json(text):
    def bad(value): raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(text, object_pairs_hook=_object, parse_constant=bad)


class Evidence:
    def __init__(self, root): self.root, self.inputs = Path(root).resolve(), {}

    def path(self, path):
        path = Path(path)
        path = (path if path.is_absolute() else self.root/path).resolve()
        path.relative_to(self.root)
        return path

    def bind(self, path, expected=None):
        path = self.path(path)
        if not path.is_file(): raise Incomplete([path.relative_to(self.root).as_posix()])
        value = sha(path)
        name = path.relative_to(self.root).as_posix()
        require(expected is None or value == expected, 'SHA256 differs: '+name)
        require(name not in self.inputs or self.inputs[name] == value, 'Input changed during audit: '+name)
        self.inputs[name] = value
        return path

    def read(self, path): return _json(self.bind(path).read_text(encoding='utf-8-sig'))

    def lines(self, path):
        path = self.bind(path)
        with path.open(encoding='utf-8-sig') as f:
            for line in f:
                require(bool(line.strip()), 'Blank JSONL row: '+str(path))
                yield _json(line)

    def rows(self, path, columns=None):
        path = self.bind(path)
        with path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), 'Invalid CSV header')
            require(columns is None or reader.fieldnames == columns, 'Unexpected CSV columns: '+str(path))
            for row in reader:
                require(None not in row and all(v is not None for v in row.values()), 'Malformed CSV row')
                yield row

    def close(self):
        for name, value in self.inputs.items():
            require(sha(self.root/name) == value, 'Input changed before audit closure: '+name)


def finite(value): return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def near(a, b, message, tolerance=1e-10):
    require(finite(a) and finite(b) and abs(a-b) <= tolerance, message)


def bce(z, y):
    require(finite(z) and type(y) is int and y in (0, 1), 'Invalid BCE logit/label')
    return max(z, 0)-z*y+math.log1p(math.exp(-abs(z)))


def sigmoid(z):
    require(finite(z), 'Invalid probability logit')
    q = math.exp(-abs(z))
    return 1/(1+q) if z >= 0 else q/(1+q)


def metrics(rows):
    require(rows and len({r['unit_id'] for r in rows}) == len(rows), 'Empty/duplicate metric identity')
    require(all(type(r['label']) is int and r['label'] in (0, 1) and finite(r['score'])
                and 0 <= r['score'] <= 1 for r in rows), 'Invalid metric label/score')
    p = sum(r['label'] for r in rows); n = len(rows)-p
    tp = sum(r['label'] == 1 and r['score'] >= .5 for r in rows)
    fp = sum(r['label'] == 0 and r['score'] >= .5 for r in rows)
    fn, tn = p-tp, n-fp
    # Independent rank-sum AUC, average ranks for exact ties; no sklearn dependency.
    ranked = sorted(rows, key=lambda r: r['score']); rank_sum, first = 0., 0
    while first < len(ranked):
        end = first+1
        while end < len(ranked) and ranked[end]['score'] == ranked[first]['score']: end += 1
        rank_sum += (first+1+end)/2*sum(r['label'] for r in ranked[first:end]); first = end
    return dict(status='complete', P=p, N=n, TP=tp, FP=fp, FN=fn, TN=tn,
        Recall=tp/p if p else None, FPR=fp/n if n else None,
        Precision=tp/(tp+fp) if tp+fp else 0., F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        AUC=(rank_sum-p*(p+1)/2)/(p*n) if p and n else None)


def same_metrics(computed, stored, message):
    require(set(computed) == set(stored), message+' fields')
    for key, value in computed.items():
        if value is None or isinstance(value, str): require(value == stored[key], message+'/'+key)
        else: near(value, stored[key], message+'/'+key)


def validation_summary(predictions, ids, truth):
    require(isinstance(predictions, list) and [p['clip_id'] for p in predictions] == ids
            and len(ids) == len(set(ids)), 'Validation identity/order/duplicates changed')
    require(all(type(p['label']) is int and p['label'] == truth[p['clip_id']] and finite(p['logit'])
                for p in predictions), 'Validation truth/nonfinite logit')
    loss = math.fsum(bce(p['logit'], p['label']) for p in predictions)/len(ids)
    scores = [dict(unit_id=p['clip_id'], label=p['label'], score=sigmoid(p['logit'])) for p in predictions]
    values = [r['score'] for r in scores]
    return loss, metrics(scores), max(values)-min(values)


def selected_epoch(history):
    require(history and all(finite(r['validation_unweighted_bce']) and r['validation_unweighted_bce'] >= 0 for r in history), 'Invalid selection BCE')
    return min(range(len(history)), key=lambda i: history[i]['validation_unweighted_bce'])+1


def rng_contract(a, b):
    require(len(a) == len(b) and bool(a), 'Missing RNG histories')
    for history in (a, b):
        for i, row in enumerate(history):
            for key in ('rng_before', 'rng_after'):
                state = row[key]
                require(set(state) == {'cpu', 'cuda'} and isinstance(state['cuda'], list) and len(state['cuda']) == 1
                        and all(isinstance(s, str) and re.fullmatch('[0-9a-f]{64}', s) for s in [state['cpu']]+state['cuda']), 'Invalid actual single-GPU RNG digest')
            if i: require(history[i-1]['rng_after'] == row['rng_before'], 'RNG trace is not continuous across epochs')
    require(all(x[k] == y[k] for x, y in zip(a, b) for k in ('rng_before', 'rng_after')), 'Branches consumed different Torch RNG streams')


def supervised_mapping(rows, bank, mode):
    expected = sorted(bank); seen = []
    for row in rows:
        cid = row['clip_id']; require(cid in bank, 'Unknown supervised mapping clip')
        w = bank[cid]
        record = window_order_record(w['media_sha256'], w['frame_indices'])
        require(row == dict(clip_id=cid, order_mode=mode, **record), 'Incorrect supervised mode/physical input mapping')
        seen.append(cid)
    require(seen == expected, 'Missing/duplicated/reordered supervised mapping')


def dense_windows(score_rows, order_rows, windows):
    expected = {(w['unit_id'], w['order_record']['physical_window_sha256']): w for w in windows}
    require(len(expected) == len(windows), 'Duplicate planned physical window within video')
    seen, counts = set(), defaultdict(int)
    maxima = {m: {} for m in MODES}
    for row, order in zip_longest(score_rows, order_rows):
        require(row is not None and order is not None, 'Score/mapping row counts differ')
        key = (row['unit_id'], row['physical_window_sha256'])
        require(key in expected and key not in seen, 'Unexpected/duplicate dense physical window')
        w = expected[key]; record = window_order_record(w['media_sha256'], w['frame_indices'])
        require(order == dict(unit_id=w['unit_id'], frame_indices=w['frame_indices'],
                source_pts_seconds=w['source_pts_seconds'], **record), 'Dense canonical frames/PTS/permutation differs')
        near(float(row['start']), w['source_pts_seconds'][0], 'Dense start differs', 1e-8)
        near(float(row['end']), w['source_pts_seconds'][-1], 'Dense end differs', 1e-8)
        uid = w['unit_id']; seen.add(key); counts[uid] += 1
        for mode in MODES:
            value = float(row[mode]); require(math.isfinite(value) and 0 <= value <= 1, 'Invalid dense probability')
            maxima[mode][uid] = max(maxima[mode].get(uid, value), value)
    require(seen == set(expected), 'Missing dense physical windows')
    return maxima, dict(counts)


def video_closure(dense, reference, sources, maxima, counts):
    require(set(dense['videos']) == set(reference['videos']) == set(sources) == set(counts), 'Canonical video coverage differs')
    recomputed, grouped, rows_by_mode = {}, {}, {}
    for uid in sources:
        v, ref = dense['videos'][uid], reference['videos'][uid]
        require(v['canonical_reference_matched'] is True and v['rgb112_digest'] == ref['rgb112_digest']
                and v['windows'] == v['accepted_windows'] == v['input_order_windows'] == ref['windows'] == counts[uid], 'Canonical digest/count closure failed: '+uid)
        require(set(v['maxima']) == set(MODES), 'Unexpected dense branch maxima')
        for mode in MODES: near(v['maxima'][mode], maxima[mode][uid], 'Accepted video max differs', 1e-12)
    require(set(dense['scores']) == set(MODES), 'Unexpected dense scored branches')
    for mode in MODES:
        stored = dense['scores'][mode]
        require(len(stored) == len(sources) and len({v['unit_id'] for v in stored}) == len(stored), 'Duplicate/missing accepted video')
        stored = {v['unit_id']: v for v in stored}; require(set(stored) == set(sources), 'Unknown accepted video')
        rows = []
        for uid, source in sources.items():
            require(source['video_label'] in ('pos', 'neg'), 'Unknown frozen video truth')
            truth = int(source['video_label'] == 'pos'); item = stored[uid]
            require(type(item['label']) is int and item['label'] == truth and item['group_id'] == source['group_id']
                    and item['windows'] == counts[uid], 'Prediction self-reported truth/group/coverage differs from sources')
            near(item['score'], maxima[mode][uid], 'Saved video score differs from max', 1e-12)
            rows.append(dict(unit_id=uid, label=truth, group_id=source['group_id'], score=maxima[mode][uid]))
        rows_by_mode[mode] = rows; recomputed[mode] = metrics(rows)
        grouped[mode] = {g: metrics([r for r in rows if r['group_id'] == g]) for g in sorted({r['group_id'] for r in rows})}
    return recomputed, grouped, rows_by_mode


def interpret(summaries):
    a, b = [summaries[m] for m in MODES]
    checks = {m: summaries[m]['clip_metrics']['AUC'] is not None and summaries[m]['clip_metrics']['AUC'] > .5
              and summaries[m]['probability_range'] >= .05 for m in MODES}
    av, bv = a['video_metrics'], b['video_metrics']
    dominates = all(av[k] <= bv[k] for k in ('FP', 'FN')) and any(av[k] < bv[k] for k in ('FP', 'FN'))
    if not all(checks.values()): classification = 'optimization_stability_only'
    elif dominates: classification = 'positive_development_signal'
    elif all(av[k] == bv[k] for k in ('FP', 'FN')): classification = 'same_error_counts_scoring_comparison_only'
    elif all(bv[k] <= av[k] for k in ('FP', 'FN')): classification = 'permuted_has_fewer_errors_no_automatic_promotion'
    else: classification = 'mixed_error_tradeoff'
    return dict(classification=classification, both_selected_clips_nondegenerate=all(checks.values()),
        nondegenerate_by_branch=checks, chronological_FP_FN_dominate=dominates,
        positive_development_signal=all(checks.values()) and dominates,
        another_run_automatically_authorized=False, scope=SCOPE)


def _fingerprints(values, expected, required):
    require(set(values) == set(required), 'Incomplete/extra execution fingerprint keys')
    require(all(expected.get(k) == v for k, v in values.items()), 'Execution fingerprints differ from frozen transfer sources')


def pixel_reference_contract(reference, base, base_sha, original, original_sha):
    require(reference['status'] == 'dense_pixels_verified', 'Canonical reference not label-free pixel verification')
    if reference['plan_sha256'] == base_sha: return
    require(reference['plan_sha256'] == original_sha and original['resource'] == base['resource']
            and original['sources'] == base['sources'], 'Pixel reference original resources/source SHA differ')


def _load_plan(e, directory):
    e.bind(directory/'plan.json', PLAN_SHA); plan = e.read(directory/'plan.json')
    for name, value in plan['files'].items(): e.bind(directory/name, value)
    require(plan['spec_id'] == 'EXP-019' and plan['modes'] == list(MODES) and plan['seed'] == 7
            and plan['permutation_seed'] == 1901 and plan['epochs'] == 20 and plan['optimizer_updates'] == 420
            and plan['pixels_ready'] is False and plan['training_ready'] is False, 'Unexpected frozen metadata contract')
    bank_rows = list(e.lines(directory/'windows.jsonl')); bank = {w['clip_id']: w for w in bank_rows}
    sources_rows = list(e.rows(directory/'sources.csv')); sources = {s['unit_id']: s for s in sources_rows}
    require(len(bank) == len(bank_rows) == 10180 and len(sources) == len(sources_rows) == 150, 'Frozen bank/source coverage changed')
    require(sum(s['split'] == 'train' for s in sources_rows) == 119 and sum(s['split'] == 'validation' for s in sources_rows) == 31, 'Video split coverage differs')
    old_bank = {w['clip_id']: w for w in e.lines(Path(plan['resource'])/'windows.jsonl')}
    e.bind(Path(plan['base_plan_path'])/'plan.json', plan['base_plan_sha256'])
    base = e.read(Path(plan['base_plan_path'])/'plan.json')
    assignment_path = Path(plan['base_plan_path'])/'assignments.csv'
    e.bind(assignment_path, base['files']['assignments.csv'])
    assignments = list(e.rows(assignment_path))
    require(set(bank) == {r['clip_id'] for r in assignments} and set(bank) <= set(old_bank), 'Assigned canonical supervised bank differs from original')
    for cid, w in bank.items():
        s = sources[w['unit_id']]
        require(w['kind'] in ('pos', 'neg') and all(w[k] == old_bank[cid][k] for k in
            ('unit_id', 'group_id', 'split', 'kind', 'frame_indices', 'source_pts_seconds'))
            and w['media_sha256'] == s['middleware_sha256'] and w['group_id'] == s['group_id']
            and w['split'] == s['split'] and len(w['source_pts_seconds']) == 16
            and all(finite(v) for v in w['source_pts_seconds'])
            and all(a < b for a, b in zip(w['source_pts_seconds'], w['source_pts_seconds'][1:]))
            and w['order_record'] == window_order_record(w['media_sha256'], w['frame_indices']), 'Changed canonical supervised identity/label/order')
    epochs = list(e.lines(directory/'epochs.jsonl')); ids = e.read(directory/'validation.json')['clip_ids']
    e.bind(Path(plan['paired_plan_path'])/'plan.json', plan['paired_plan_sha256'])
    e.bind(Path(plan['paired_plan_path'])/'epochs.jsonl', plan['paired_epochs_sha256'])
    paired = e.read(Path(plan['paired_plan_path'])/'plan.json')
    old_epochs = list(e.lines(Path(plan['paired_plan_path'])/'epochs.jsonl'))
    old_history = list(e.lines(directory/'historical-source-snapshot/run/bce/epochs.jsonl'))
    require(len(epochs) == len(old_epochs) == len(old_history) == 20 and ids == paired['validation_clip_ids']
            and ids == [r['clip_id'] for r in assignments if r['split'] == 'validation']
            and len(ids) == len(set(ids)) == 136 and sum(bank[c]['kind'] == 'pos' for c in ids) == 96
            and all(bank[c]['split'] == 'validation' for c in ids), 'Original fixed validation differs')
    for i, (row, old, hist) in enumerate(zip(epochs, old_epochs, old_history)):
        flat = [c for batch in row['batches'] for c in batch['clip_ids']]
        require(row['epoch'] == old['epoch'] == i and row['batches'] == old['batches']
                and row['clip_ids'] == flat == hist['train_clip_ids'] and len(row['batches']) == 21
                and set(flat) == {r['clip_id'] for r in assignments if r['split'] == 'train' and int(r['epoch']) == i}
                and all(len(b['clip_ids']) == 24 for b in row['batches']) and len(flat) == len(set(flat)) == 504
                and all(bank[c]['split'] == 'train' for c in flat)
                and sum(bank[c]['kind'] == 'pos' for c in flat) == 356, 'Actual old A batch order/exposure differs')
    dense = list(e.lines(directory/'dense-windows.jsonl'))
    original = list(e.lines(Path(plan['resource'])/'dense-eval-windows.jsonl'))
    require(len(dense) == len(original) == 32567, 'Dense grid count changed')
    for w, old in zip(dense, original):
        require(all(w[k] == old[k] for k in ('unit_id', 'frame_indices', 'source_pts_seconds'))
            and w['media_sha256'] == sources[w['unit_id']]['middleware_sha256']
            and w['order_record'] == window_order_record(w['media_sha256'], w['frame_indices']), 'Dense physical grid/order differs')
    require({w['unit_id'] for w in dense} == {bank[c]['unit_id'] for c in ids}, 'Dense videos differ from fixed validation')
    # Bind the original authoritative metadata by the registered data SHA, not live documentation.
    refs = {r['path']: r['sha256'] for r in plan['data_fingerprints']}
    for path in (Path(plan['resource'])/'windows.jsonl', Path(plan['resource'])/'dense-eval-windows.jsonl'):
        e.bind(path, refs[path.as_posix()])
    return plan, bank, epochs, ids, dense, sources


def _sources(e, run, plan_dir, deployment, execution, acceptance_path, reference_path, preflight_path):
    index = e.read(deployment/'snapshot-index.json'); manifest = e.bind(deployment/'files.csv', execution['transfer_manifest_sha256'])
    require(index['status'] == 'execution_and_transfer_closure_frozen' and index['manifest_sha256'] == execution['transfer_manifest_sha256']
            and index['order_plan_sha256'] == PLAN_SHA and index['cache_acceptance_sha256'] == execution['cache_acceptance_sha256'], 'Execution deployment identity differs')
    rows = list(e.rows(manifest)); indexed = {r['path']: r for r in rows}
    require(len(rows) == len(indexed) == index['files'] and sum(int(r['size_bytes']) for r in rows) == index['total_bytes'], 'Invalid deployment manifest')
    require([{k: (int(r[k]) if k == 'size_bytes' else r[k]) for k in ('path','size_bytes','sha256','kind')} for r in rows] == index['sources'], 'Deployment CSV/index differ')
    expected = {r['path']: r['sha256'] for r in rows}
    for row in rows:
        path = deployment/'source-snapshot'/row['path'] if row['kind'] == 'execution_source' else e.root/row['path']
        p = e.bind(path, row['sha256']); require(p.stat().st_size == int(row['size_bytes']), 'Deployment size differs')
    code = execution['code_sha256']
    source_keys = {r['path'] for r in rows if r['kind'] == 'execution_source'}
    require(set(code) == source_keys-{'experiments/runspecs/EXP-019/prepare_transfer.py'}, 'Execution code closure differs')
    _fingerprints(code, expected, code)
    # The pure audit dependency must be the very function used by the actual execution.
    pure = 'experiments/common/video/temporal_order.py'; e.bind(pure, code[pure])
    for name, value in code.items(): e.bind(run/'source-snapshot'/name, value)
    small = {p.relative_to(e.root).as_posix(): sha(p) for p in plan_dir.iterdir() if p.is_file()}
    small[acceptance_path.relative_to(e.root).as_posix()] = execution['cache_acceptance_sha256']
    for name, value in small.items():
        require(expected.get(name) == value, 'Run plan/acceptance differs from transfer')
        e.bind(run/'source-snapshot'/name, value)
    e.bind(reference_path, execution['pixel_reference_sha256']); e.bind(preflight_path, execution['preflight_sha256'])
    preflight = e.read(preflight_path)
    require(preflight['status'] == 'exp019_static_preflight_passed' and preflight['decoded_frames'] == 0
            and preflight['actual_cache_acceptance_bound'] is True, 'Static preflight not accepted')
    for key in ('code_sha256','order_plan_sha256','cache_acceptance_sha256','pixel_reference_sha256','ffmpeg_sha256','transfer_manifest_sha256'):
        require(preflight[key] == execution[key], 'Execution differs from preflight: '+key)
    return expected, set(small), source_keys


def _profile(profile, mode, plan, execution):
    required = dict(spec_id='EXP-019', order_mode=mode, mode='rgb_zero', objective_mode='bce',
        seed=7, permutation_seed=1901, initial_state_sha256=plan['expected_initial_state_sha256'],
        frozen_plan_initial_state_sha256=plan['expected_initial_state_sha256'], old_initialization_reference=plan['expected_initial_state_sha256'],
        plan_sha256=PLAN_SHA, base_plan_sha256=plan['base_plan_sha256'], paired_plan_sha256=plan['paired_plan_sha256'],
        paired_schedule_sha256=plan['paired_epochs_sha256'], cache_acceptance_sha256=execution['cache_acceptance_sha256'],
        pretrained='KINETICS400_V1', fixture=False, smoke=False, architecture='r2plus1d_18_six_channel',
        effective_batch=24, microbatch=8, workers=2, persistent_workers=True, prefetch_factor=2,
        loader_generator_seed=100007, cpu_threads=4, cache_mib_per_worker=128, checkpoint_backbone=False,
        supervised_pos_weight=148/356, rank_lambda=0., rank_scale=21/80, requested_epochs=20,
        requested_optimizer_updates=420, precision='FP32', aggregation='max', score_threshold=.5,
        clip_len=16, span_seconds=3., inference_stride_seconds=1.5,
        input_pipeline='canonical JPEG95/RGB112 -> T-axis order -> MEAN_STD/RGB+zeroD')
    require(all(profile.get(k) == v for k, v in required.items()), 'Unregistered branch training recipe/profile')
    require(profile['expected_paired_initial_state_sha256'] == plan['expected_initial_state_sha256'], 'Measured paired initialization missing')
    env = profile['environment']
    require(env['device'] == 'cuda' and env['precision'] == 'FP32' and env['amp'] is False
            and env['tf32_matmul'] is False and env['tf32_cudnn'] is False and env['cudnn_benchmark'] is False
            and env['cudnn_deterministic'] is True, 'FP32 deterministic execution environment differs')


def _branch(e, run, mode, plan, bank, epochs, ids, execution, expected, small, source_keys):
    directory = run/mode; result = e.read(directory/'training-result.json'); profile = e.read(directory/'profile.json')
    require(result['status'] == 'supervised_complete_dense_evaluation_pending' and result['completed_epochs'] == 20
            and result['optimizer_steps'] == 420 and result['independent_test_used'] is False, 'Formal branch incomplete/invalid')
    require(profile == result['profile'], 'Result/profile differ'); _profile(profile, mode, plan, execution)
    train_names = ('order_training.py','temporal_order.py','order_plan.py','paired_plan.py','paired_runtime.py',
                   'supervised_runtime.py','supervised_training.py','supervised_plan.py','r2plus1d_classifier.py','pts_clip_sampling.py')
    train_paths = {'experiments/common/video/'+n for n in train_names}
    train_paths |= {p for p in source_keys if not p.startswith('experiments/common/video/')}
    _fingerprints(profile['execution_fingerprints'], expected, train_paths | small)
    history = list(e.lines(directory/'epochs.jsonl')); require(len(history) == 20, 'Formal history not twenty epochs')
    truth = {cid: int(bank[cid]['kind'] == 'pos') for cid in ids}; curve, selected_predictions = [], {}
    for i, row in enumerate(history):
        epoch = i+1
        require(row['completed_epochs'] == epoch and row['samples'] == 504 and row['optimizer_steps'] == 21
                and row['total_updates'] == 21*epoch and row['train_clip_ids'] == epochs[i]['clip_ids']
                and row['validation_clip_ids'] == ids and row['validation_P'] == 96 and row['validation_N'] == 40
                and row['order_mode'] == mode and row['rank_lambda'] == 0.
                and row['initial_state_sha256'] == plan['expected_initial_state_sha256'], 'Epoch identity/update/recipe differs')
        for key in ('train_loss','train_weighted_bce','train_bce_sum','validation_pos_bce','validation_neg_bce','grad_norm_mean','grad_norm_max','seconds'):
            require(finite(row[key]) and row[key] >= 0, 'Invalid training trace: '+key)
        require(row['seconds'] > 0 and row['grad_norm_max'] >= row['grad_norm_mean'] > 0, 'Invalid gradient/time trace')
        near(row['train_weighted_bce'], row['train_bce_sum']/504, 'Train BCE reduction differs')
        near(row['train_loss'], row['train_weighted_bce'], 'Unexpected non-BCE objective', 2e-6)
        require(isinstance(row['model_state_sha256'], str) and re.fullmatch('[0-9a-f]{64}', row['model_state_sha256']), 'Invalid epoch model state SHA')
        predictions = e.read(directory/('validation-epoch-%02d.json'%epoch))
        loss, clip_metrics, span = validation_summary(predictions, ids, truth)
        near(loss, row['validation_unweighted_bce'], 'Saved logits do not reproduce recorded BCE', BCE_ATOL)
        for label, field in ((1, 'validation_pos_bce'), (0, 'validation_neg_bce')):
            subset = [p for p in predictions if p['label'] == label]
            near(math.fsum(bce(p['logit'], label) for p in subset)/len(subset), row[field], 'Class BCE differs', BCE_ATOL)
        require(row['best_epoch'] == selected_epoch(history[:epoch]), 'Running selection is not earliest recorded minimum')
        curve.append(dict(epoch=epoch, recorded_bce=row['validation_unweighted_bce'], bce_fp64=loss,
            absolute_bce_difference=abs(loss-row['validation_unweighted_bce']), clip_metrics=clip_metrics, probability_range=span))
        selected_predictions[epoch] = predictions
    selected = selected_epoch(history); require(result['best_epoch'] == selected, 'Selected epoch differs')
    near(result['best_validation_bce'], history[selected-1]['validation_unweighted_bce'], 'Selected recorded BCE differs', 0.)
    require(result['initial_state_sha256'] == plan['expected_initial_state_sha256']
            and result['final_state_sha256'] == history[-1]['model_state_sha256'], 'Model state provenance differs')
    checkpoint = result['selected_checkpoint']
    require(checkpoint['filename'] == 'best.pt' and checkpoint['completed_epochs'] == selected, 'Selected checkpoint identity differs')
    e.bind(directory/'best.pt', checkpoint['sha256']); e.bind(directory/'last.pt')
    mapping = directory/'input-order-mapping.jsonl'; e.bind(mapping, result['input_order_mapping_sha256'])
    supervised_mapping(e.lines(mapping), bank, mode)
    chosen = curve[selected-1]
    return dict(best_epoch=selected, clip_bce_fp64=chosen['bce_fp64'], clip_metrics=chosen['clip_metrics'],
        probability_range=chosen['probability_range'], max_bce_absolute_difference=max(x['absolute_bce_difference'] for x in curve),
        epochs=curve, selected_predictions=selected_predictions[selected]), profile, history, checkpoint['sha256']


def _complete(e, run, plan_dir, deployment, acceptance_path, reference_path, preflight_path):
    pipeline = e.read(run/'pipeline-result.json'); execution = e.read(run/'execution-inputs.json')
    require(pipeline['status'] == 'completed_pending_independent_audit' and pipeline['order_plan_sha256'] == PLAN_SHA
            and execution['order_plan_sha256'] == PLAN_SHA, 'Pipeline not completely accepted')
    expected, small, source_keys = _sources(e, run, plan_dir, deployment, execution, acceptance_path, reference_path, preflight_path)
    plan, bank, epochs, ids, grid, sources = _load_plan(e, plan_dir)
    require(execution['canonical_validation_ids'] == ids, 'Execution validation IDs differ')
    acceptance = e.read(acceptance_path)
    require(acceptance['status'] == 'order_supervised_cache_ready' and acceptance['order_plan_sha256'] == PLAN_SHA
            and acceptance['required_frame_count'] == acceptance['verified_required_frame_count'] == 150784
            and acceptance['failed_required_frames'] == acceptance['unverified_required_frames'] == 0
            and acceptance['actual_verified_JPEG_reads'] == 150784 and acceptance['actual_JPEG_byte_SHA_checked'] is True
            and acceptance['actual_RGB112_pixel_SHA_checked'] is True, 'Actual cache acceptance incomplete')
    refs = {r['path']: r['sha256'] for r in plan['data_fingerprints']}
    require(refs[acceptance['verified_cache_index']] == acceptance['verified_cache_index_sha256'], 'Cache index provenance differs')
    e.bind(acceptance['verified_cache_index'], acceptance['verified_cache_index_sha256'])
    e.bind(acceptance_path.parent/acceptance['verified_required_frames_index'], acceptance['verified_required_frames_index_sha256'])
    summaries, profiles, histories, selected = {}, [], [], {}
    for mode in MODES:
        summary, profile, history, selected[mode] = _branch(e, run, mode, plan, bank, epochs, ids, execution, expected, small, source_keys)
        summaries[mode] = summary; profiles.append(profile); histories.append(history)
    ignore = {'order_mode'}
    require({k: v for k, v in profiles[0].items() if k not in ignore} == {k: v for k, v in profiles[1].items() if k not in ignore}, 'Branch profiles/environment differ beyond input order')
    rng_contract(*histories)
    dense_dir = run/'two-dense'; e.bind(dense_dir/'result.json', pipeline['dense_result_sha256']); dense = e.read(dense_dir/'result.json')
    require(dense['status'] == 'dense_evaluation_complete' and dense['spec_id'] == 'EXP-019' and dense['plan_sha256'] == PLAN_SHA
            and dense['checkpoint_sha256'] == selected and dense['independent_test_used'] is False
            and dense['pixel_reference_sha256'] == execution['pixel_reference_sha256'], 'Dense checkpoint/plan/source identity differs')
    dense_names = ('order_dense_evaluation.py','order_training.py','temporal_order.py','order_plan.py',
                   'dense_evaluation.py','supervised_runtime.py','pts_clip_sampling.py','r2plus1d_classifier.py')
    required = {'experiments/common/video/'+n for n in dense_names} | small | {reference_path.relative_to(e.root).as_posix()}
    expected[reference_path.relative_to(e.root).as_posix()] = execution['pixel_reference_sha256']
    _fingerprints(dense['code_and_input_sha256'], expected, required)
    score_path, order_path = dense_dir/'window-scores.csv', dense_dir/'window-orders.jsonl'
    e.bind(score_path, dense['window_scores_sha256']); e.bind(order_path, dense['input_order_mapping_sha256'])
    maxima, counts = dense_windows(e.rows(score_path, ['unit_id','start','end','physical_window_sha256',*MODES]), e.lines(order_path), grid)
    val_sources = {uid: sources[uid] for uid in counts}; require(len(counts) == 31 and sum(counts.values()) == 32567, 'Incomplete full dense coverage')
    reference = e.read(reference_path)
    base = e.read(Path(plan['base_plan_path'])/'plan.json')
    origin_path = e.bind(reference_path.parent/'source-plan.json', reference['plan_sha256'])
    original = e.read(origin_path)
    pixel_reference_contract(reference, base, plan['base_plan_sha256'], original, e.inputs[origin_path.relative_to(e.root).as_posix()])
    computed, grouped, video_rows = video_closure(dense, reference, val_sources, maxima, counts)
    require(set(dense['metrics']) == set(dense['metrics_by_group']) == set(pipeline['metrics']) == set(MODES), 'Unexpected result metric branches')
    for mode in MODES:
        same_metrics(computed[mode], dense['metrics'][mode], 'Dense metric '+mode)
        same_metrics(computed[mode], pipeline['metrics'][mode], 'Pipeline metric '+mode)
        require(set(grouped[mode]) == set(dense['metrics_by_group'][mode]), 'Missing group metrics')
        for group, value in grouped[mode].items(): same_metrics(value, dense['metrics_by_group'][mode][group], 'Group metric '+group)
        summaries[mode].update(video_metrics=computed[mode], metrics_by_group=grouped[mode], videos=video_rows[mode])
    return dict(status='audit_passed', metrics_accepted=True, summaries=summaries, registered_gate=interpret(summaries),
        paired_video_differences=[dict(unit_id=uid, group_id=sources[uid]['group_id'], label=int(sources[uid]['video_label']=='pos'),
            chronological=maxima[MODES[0]][uid], fixed_permutation=maxima[MODES[1]][uid],
            chronological_minus_permuted=maxima[MODES[0]][uid]-maxima[MODES[1]][uid]) for uid in sorted(counts)],
        checks=dict(epochs_per_branch=20, updates_per_branch=420, same_initialization=True, same_rng=True,
            old_A_batch_template=True, fixed_validation=136, supervised_mappings=len(bank), dense_videos=31,
            dense_physical_windows=32567, canonical_reference_closed=True, source_snapshots_closed=True),
        limitations=['Recorded source/environment/RNG audit, not numerical training replay.',
            'Checkpoint bytes are SHA-verified without deserializing pickle; checkpoint state/optimizer are not independently executed.',
            'Existing per-JPEG acceptance proof is checked; audit does not reread JPEGs or decode videos.'])


def audit(run, out, *, root=ROOT, plan_dir=None, deployment=None, acceptance=None, pixel_reference=None, preflight=None, argv=None):
    e = Evidence(root); run, out = e.path(run), e.path(out)
    run.relative_to(e.root/'data/results/exp019'); out.relative_to(e.root/'data/results/exp019')
    require(out != run and out not in run.parents, 'Audit output must not replace/contain the run')
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter(); started = datetime.now(timezone.utc).isoformat()
    source_bytes = SOURCE.read_bytes(); (out/'generator-source.py').write_bytes(source_bytes)
    launch = dict(argv=list(sys.argv if argv is None else argv), started_utc=started,
        generator_sha256=hashlib.sha256(source_bytes).hexdigest(), audit_python=sys.version, no_model_execution=True, no_media_decode=True)
    (out/'launch.json').write_text(json.dumps(launch, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    try:
        required = [run/'pipeline-result.json', run/'execution-inputs.json', run/'two-dense/result.json',
                    run/'two-dense/window-scores.csv', run/'two-dense/window-orders.jsonl']
        for mode in MODES:
            required.extend(run/mode/n for n in ('profile.json','training-result.json','epochs.jsonl','best.pt','last.pt','input-order-mapping.jsonl'))
            required.extend(run/mode/('validation-epoch-%02d.json'%i) for i in range(1, 21))
        failures = [run/name for name in ('pipeline-failure.json', 'hard-time-limit.json',
                    'hard-stop-shutdown.json', 'hard-stop-shutdown.log')]
        failures += [run/'two-dense/failure.json']+[run/m/'failure.json' for m in MODES]
        require(not any(p.exists() for p in failures), 'Run contains a failure receipt; complete metrics withheld')
        missing = [p.relative_to(e.root).as_posix() for p in required if not p.is_file()]
        if missing: raise Incomplete(missing)
        report = _complete(e, run, e.path(plan_dir or PLAN), e.path(deployment or DEPLOYMENT), e.path(acceptance or ACCEPTANCE),
                           e.path(pixel_reference or REFERENCE), e.path(preflight or PREFLIGHT))
        e.close()
    except Exception as error:
        report = dict(status='incomplete' if isinstance(error, Incomplete) else 'failed', metrics_accepted=False,
            error_type=type(error).__name__, error=str(error), missing_inputs=getattr(error, 'missing', []),
            registered_gate=None, complete_metrics_withheld=True)
    report.update(spec_id='EXP-019', scope=SCOPE, bce_logit_recompute_absolute_tolerance=BCE_ATOL,
        selection='earliest exact minimum of recorded FP32 unweighted136-clip BCE; FP64 consistency check only',
        another_run_automatically_authorized=False, independent_test_used=False, command=launch['argv'],
        generator_sha256=launch['generator_sha256'], started_utc=started, seconds=time.perf_counter()-began,
        source_sha256=e.inputs, no_model_execution=True, no_media_decode=True)
    (out/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (out/'README.md').write_text('EXP-019 offline audit\n\nStatus: '+report['status']+'\n\n'+SCOPE+
        '\n\nSee audit.json for exact input SHA256, recomputed metrics and the registered gate. '
        'BCE selection uses recorded FP32 values, with stable FP64 consistency tolerance2e-6. '
        'No model inference, training or media decode was performed. This directory never overwrites another audit.\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], metrics_accepted=report['metrics_accepted'], seconds=report['seconds'])), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    for key, default in (('plan', PLAN), ('deployment', DEPLOYMENT), ('cache-acceptance', ACCEPTANCE), ('pixel-reference', REFERENCE), ('preflight', PREFLIGHT)):
        parser.add_argument('--'+key, type=Path, default=ROOT/default)
    args = parser.parse_args(argv)
    result = audit(args.run, args.out, plan_dir=args.plan, deployment=args.deployment, acceptance=args.cache_acceptance,
                   pixel_reference=args.pixel_reference, preflight=args.preflight, argv=sys.argv if argv is None else argv)
    return {'audit_passed': 0, 'incomplete': 2, 'failed': 1}[result['status']]


if __name__ == '__main__': sys.exit(main())
