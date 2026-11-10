"""Independently audit completed EXP-016 logs and recompute video metrics.

No model is trained or evaluated here. Refuse incomplete runs; retain every
source hash used by this derived report. Does not authorize another seed/CV.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODES = {'rgb': 'rgb_zero', 'diff': 'rgb_diff'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(a, b, message, tolerance=2e-6):
    require(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance, message)


def metrics(rows):
    require(len({r['unit_id'] for r in rows}) == len(rows), 'Duplicate scored video')
    require(all(r['score'] is not None and math.isfinite(r['score']) and 0 <= r['score'] <= 1
                and r['label'] in (0, 1) for r in rows), 'Invalid/unknown video prediction')
    p = sum(r['label'] for r in rows)
    n = len(rows)-p
    require(p > 0 and n > 0, 'Two-class video metrics required')
    tp = sum(r['label'] == 1 and r['score'] >= .5 for r in rows)
    fp = sum(r['label'] == 0 and r['score'] >= .5 for r in rows)
    fn, tn = p-tp, n-fp
    # Independent rank-sum AUC, with average ranks for tied scores.
    ranked = sorted(rows, key=lambda r: r['score'])
    rank_sum, first = 0., 0
    while first < len(ranked):
        end = first+1
        while end < len(ranked) and ranked[end]['score'] == ranked[first]['score']:
            end += 1
        rank_sum += ((first+1+end)/2) * sum(r['label'] for r in ranked[first:end])
        first = end
    return dict(P=p, N=n, TP=tp, FP=fp, FN=fn, TN=tn, Recall=tp/p, FPR=fp/n,
                Precision=tp/(tp+fp) if tp+fp else 0., F1=2*tp/(2*tp+fp+fn),
                AUC=(rank_sum-p*(p+1)/2)/(p*n))


def audit(run, out, seed=7):
    require(seed in (7, 8), 'Unregistered seed')
    run, out = Path(run).resolve(), Path(out).resolve()
    run.relative_to(ROOT/'data/results/exp016')
    out.relative_to(ROOT/'data/results/exp016')
    sources = {}

    def read(path):
        path = Path(path)
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))

    pipeline = read(run/'pipeline-result.json')
    require(pipeline['status'] == 'completed', 'Pipeline is not complete')
    require(pipeline.get('seed', 7) == seed, 'Pipeline seed mismatch')
    plan_dir = ROOT/'data/metadata/EXP-016/training-plan-v2'
    plan = read(plan_dir/'plan.json')
    sources[(plan_dir/'assignments.csv').relative_to(ROOT).as_posix()] = sha(plan_dir/'assignments.csv')
    with (plan_dir/'assignments.csv').open(encoding='utf-8-sig', newline='') as f:
        assignments = list(csv.DictReader(f))
    require(sha(plan_dir/'assignments.csv') == plan['files']['assignments.csv'], 'Frozen assignments changed')
    train_ids = {ep: {r['clip_id'] for r in assignments if r['split'] == 'train' and int(r['epoch']) == ep}
                 for ep in range(20)}
    val_ids = {r['clip_id'] for r in assignments if r['split'] == 'validation'}
    val_labels = {r['clip_id']: int(r['kind'] == 'pos') for r in assignments if r['split'] == 'validation'}
    require(len(val_ids) == 136 and all(len(ids) == 504 for ids in train_ids.values()), 'Unexpected plan counts')
    results, histories = {}, {}
    for folder, mode in MODES.items():
        result = read(run/folder/'training-result.json')
        profile = read(run/folder/'profile.json')
        require(result['status'] == 'supervised_complete_dense_evaluation_pending'
                and result['completed_epochs'] == 20 and result['optimizer_steps'] == 420,
                'Incomplete training budget')
        require(profile == result['profile'] and profile['mode'] == mode and not profile['smoke']
                and profile['seed'] == seed and profile['requested_epochs'] == 20
                and profile['requested_optimizer_updates'] == 420 and profile['precision'] == 'FP32'
                and profile['plan_sha256'] == sha(plan_dir/'plan.json'), 'Training profile mismatch')
        history_path = run/folder/'epochs.jsonl'
        sources[history_path.relative_to(ROOT).as_posix()] = sha(history_path)
        history = [json.loads(line) for line in history_path.read_text(encoding='utf-8').splitlines()]
        require(len(history) == 20, 'Incomplete epoch log')
        for ep, record in enumerate(history):
            require(record['completed_epochs'] == ep+1 and record['samples'] == 504
                    and record['optimizer_steps'] == 21 and record['total_updates'] == 21*(ep+1)
                    and (record['validation_P'], record['validation_N']) == (96, 40)
                    and record['learning_rates'] == [1e-4, 1e-3], 'Epoch budget/config changed')
            ids = record['train_clip_ids']
            require(len(ids) == 504 and len(set(ids)) == 504 and set(ids) == train_ids[ep]
                    and not set(ids).intersection(val_ids), 'Epoch assignments mismatch')
            predictions = read(run/folder/f'validation-epoch-{ep+1:02d}.json')
            require(len(predictions) == 136 and {r['clip_id'] for r in predictions} == val_ids,
                    'Validation clip coverage mismatch')
            losses = {0: [], 1: []}
            for row in predictions:
                z, y = row['logit'], row['label']
                require(y == val_labels[row['clip_id']] and math.isfinite(z), 'Invalid validation logit/label')
                losses[y].append(max(z, 0)-z*y+math.log1p(math.exp(-abs(z))))
            require((len(losses[1]), len(losses[0])) == (96, 40), 'Validation class mismatch')
            near(sum(losses[0]+losses[1])/136, record['validation_unweighted_bce'], 'BCE recomputation mismatch')
            for y, key in [(0, 'validation_neg_bce'), (1, 'validation_pos_bce')]:
                near(sum(losses[y])/len(losses[y]), record[key], 'Class BCE mismatch')
            require(math.isfinite(record['train_loss']) and record['grad_norm_mean'] > 0
                    and math.isfinite(record['grad_norm_mean'])
                    and math.isfinite(record['grad_norm_max']), 'Invalid training trace')
            prefix_best = min(range(ep+1), key=lambda i: history[i]['validation_unweighted_bce'])+1
            require(record['best_epoch'] == prefix_best, 'Epoch best record is not prefix minimum')
        best = min(range(20), key=lambda i: history[i]['validation_unweighted_bce'])+1
        require(result['best_epoch'] == best and result['selected_checkpoint']['completed_epochs'] == best,
                'Selected epoch is not first minimum of fixed validation BCE')
        near(result['best_validation_bce'], history[best-1]['validation_unweighted_bce'],
             'Best validation BCE summary mismatch', 1e-12)
        checkpoint = run/folder/'best.pt'
        sources[checkpoint.relative_to(ROOT).as_posix()] = sha(checkpoint)
        require(sources[checkpoint.relative_to(ROOT).as_posix()] == result['selected_checkpoint']['sha256'],
                'Downloaded selected checkpoint SHA mismatch')
        results[folder], histories[folder] = result, history
    a, b = results['rgb']['profile'], results['diff']['profile']
    for key in set(a) | set(b):
        if key != 'mode':
            require(a[key] == b[key], f'Unpaired profile field: {key}')
    for ra, rb in zip(histories['rgb'], histories['diff']):
        require(ra['train_clip_ids'] == rb['train_clip_ids'], 'A/B training orders differ')
    dense = read(run/'paired-dense/result.json')
    require(dense['status'] == 'dense_evaluation_complete' and not dense['independent_test_used']
            and dense['plan_sha256'] == sha(plan_dir/'plan.json')
            and len(dense['videos']) == 31 and sum(v['windows'] for v in dense['videos'].values()) == 32567,
            'Incomplete or wrong-scope video evaluation')
    video_path = ROOT/plan['resource']/'video-inputs.csv'
    video_sha = sha(video_path)
    require(any(r['path'] == video_path.relative_to(ROOT).as_posix() and r['sha256'] == video_sha
                for r in plan['sources']), 'Frozen video labels changed')
    sources[video_path.relative_to(ROOT).as_posix()] = video_sha
    with video_path.open(encoding='utf-8-sig', newline='') as f:
        truth = {r['unit_id']: (int(r['video_label'] == 'pos'), r['group_id'])
                 for r in csv.DictReader(f) if r['split'] == 'validation'}
    grid_path = ROOT/plan['resource']/'dense-eval-windows.jsonl'
    grid_sha = sha(grid_path)
    require(any(r['path'] == grid_path.relative_to(ROOT).as_posix() and r['sha256'] == grid_sha
                for r in plan['sources']), 'Frozen dense window grid changed')
    sources[grid_path.relative_to(ROOT).as_posix()] = grid_sha
    windows = defaultdict(list)
    with grid_path.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            windows[row['unit_id']].append(row)
    expected = (w for uid in truth for w in sorted(windows[uid], key=lambda r: (r['frame_indices'][-1], r['frame_indices'][0])))
    window_path = run/'paired-dense/window-scores.csv'
    sources[window_path.relative_to(ROOT).as_posix()] = sha(window_path)
    maxima = {mode: {uid: -math.inf for uid in truth} for mode in MODES.values()}
    counts = defaultdict(int)
    with window_path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == ['unit_id', 'start', 'end', *MODES.values()], 'Unexpected window score columns')
        for window, row in zip_longest(expected, reader):
            require(window is not None and row is not None, 'Missing/extra scored window')
            uid = row['unit_id']
            require(uid == window['unit_id'], 'Scored window video/order mismatch')
            near(float(row['start']), window['source_pts_seconds'][0], 'Window start mismatch', 1e-9)
            near(float(row['end']), window['source_pts_seconds'][-1], 'Window end mismatch', 1e-9)
            for mode in MODES.values():
                value = float(row[mode])
                require(math.isfinite(value) and 0 <= value <= 1, 'Invalid window score')
                maxima[mode][uid] = max(maxima[mode][uid], value)
            counts[uid] += 1
    require(sum(counts.values()) == 32567 and dict(counts) == {uid: dense['videos'][uid]['windows'] for uid in truth},
            'Dense window coverage mismatch')
    require(set(dense['scores']) == set(MODES.values()), 'Missing comparison arm')
    recomputed = {}
    for folder, mode in MODES.items():
        require(dense['checkpoint_sha256'][mode] == results[folder]['selected_checkpoint']['sha256'],
                'Evaluation did not use selected checkpoint')
        rows = dense['scores'][mode]
        require(len(rows) == 31 and {r['unit_id'] for r in rows} == set(dense['videos']), 'Scored video mismatch')
        require({r['unit_id']: (r['label'], r['group_id']) for r in rows} == truth, 'Video truth/group mismatch')
        for row in rows:
            require(row['windows'] == counts[row['unit_id']], 'Scored video window count mismatch')
            near(row['score'], maxima[mode][row['unit_id']], 'Video score is not complete-window maximum', 1e-12)
        recomputed[mode] = metrics(rows)
        for key, value in recomputed[mode].items():
            near(value, dense['metrics'][mode][key], f'Video metric mismatch: {mode}/{key}', 1e-12)
        require(pipeline['metrics'][mode] == dense['metrics'][mode], 'Pipeline summary mismatch')
    require([(r['unit_id'], r['label'], r['group_id']) for r in dense['scores']['rgb_zero']]
            == [(r['unit_id'], r['label'], r['group_id']) for r in dense['scores']['rgb_diff']],
            'A/B video ground truths differ')
    base, fused = recomputed['rgb_zero'], recomputed['rgb_diff']
    report = dict(status='paired_artifacts_audited', run_id=run.name, seed=seed,
                  scope='single development split; not full CV or independent test',
                  metrics=recomputed, best_epochs={k: v['best_epoch'] for k, v in results.items()},
                  differences_B_minus_A={k: fused[k]-base[k] for k in ('FP', 'FN', 'AUC', 'F1')},
                  registered_replication_condition=(fused['FP'] < base['FP'] and fused['FN'] <= base['FN']
                                                     and fused['AUC'] >= base['AUC']),
                  another_seed_or_cv_authorized=False, sources_sha256=sources,
                  audit_code_sha256=sha(__file__))
    require(not out.exists(), 'Audit destination already exists')
    out.mkdir(parents=True)
    (out/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(7, 8), default=7)
    args = parser.parse_args()
    args.run.resolve().relative_to(ROOT/'data/results/exp016')
    args.out.resolve().relative_to(ROOT/'data/results/exp016')
    print(json.dumps(audit(args.run, args.out, args.seed), ensure_ascii=True))
