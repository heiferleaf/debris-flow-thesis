"""Read-only independent EXP021 result derivation; never a training gate.

No weights/media are loaded or hashed. Complete mode requires four closed fits,
six closed dense stages and a completed run. Partial mode reviews exactly one
closed fit, writes a separate directory, and never computes an overall gate.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN_ID = 'energy-s7-20261103-v2'
BCE_ATOL = 2e-6  # Original FP32 mean compared with stable FP64 logit recomputation.


class Incomplete(ValueError):
    pass


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        if not path.is_file():
            raise Incomplete('Required closed artifact not collected: ' + str(path))
        actual = sha(path)
        require(expected is None or actual == expected, 'Source SHA differs: ' + str(path))
        previous = self.hashes.setdefault(str(path), actual)
        require(previous == actual, 'Source changed during report: ' + str(path))
        return path

    def read(self, path, expected=None):
        return json.loads(self.bind(path, expected).read_text(encoding='utf-8'))

    def jsonl(self, path, expected=None):
        with self.bind(path, expected).open(encoding='utf-8') as f:
            return [json.loads(s) for s in f]

    def unchanged(self):
        require(all(sha(Path(p)) == value for p, value in self.hashes.items()), 'Report input changed')


def probability(z):
    q = math.exp(-abs(z))
    return 1 / (1 + q) if z >= 0 else q / (1 + q)


def bce(z, label):
    return max(z, 0.) - label * z + math.log1p(math.exp(-abs(z)))


def metrics(labels, scores):
    p = [s for y, s in zip(labels, scores) if y == 1]
    n = [s for y, s in zip(labels, scores) if y == 0]
    require(len(labels) == len(scores) and set(labels) <= {0, 1} and all(math.isfinite(s) for s in scores), 'Invalid metric rows')
    tp, fp = sum(s >= .5 for s in p), sum(s >= .5 for s in n)
    return dict(P=len(p), N=len(n), TP=tp, FP=fp, FN=len(p) - tp, TN=len(n) - fp,
        positive_Recall=tp / len(p) if p else None, negative_Recall=(len(n) - fp) / len(n) if n else None,
        AUC=math.fsum((a > b) + .5 * (a == b) for a in p for b in n) / (len(p) * len(n)) if p and n else None)


def closure(inputs, run, stage, status):
    receipt = inputs.read(run / ('closed-' + stage + '.json'))
    require(receipt['status'] == status and receipt.get('item', receipt.get('path')) == stage, 'Stage not closed: ' + stage)
    rows = {r['path']: r for r in receipt['files']}
    require(len(rows) == len(receipt['files']), 'Duplicate closure file')
    return rows


def fit_review(inputs, directory, fold, arm, schedule, labels, expected_files=None):
    def load(name, lines=False):
        expected = None
        if expected_files is not None:
            key = directory.name + '/' + name
            require(key in expected_files, 'Consumed artifact absent from closure: ' + key)
            expected = expected_files[key]['sha256']
        return (inputs.jsonl if lines else inputs.read)(directory / name, expected)
    result, profile, history = load('training-result.json'), load('profile.json'), load('epochs.jsonl', True)
    require(result['status'] == 'fit_complete' and result['stage'] == 'inner_select' and result['completed_epochs'] == 20
        and result['fold'] == profile['fold'] == fold and profile['seed'] == 7 and profile['execution_kind'] == 'registered_fit'
        and len(history) == 20, 'Require complete registered twenty-epoch inner fit')
    require((profile['head'] == 'linear' if arm == 'A' else profile.get('energy_mode') == arm), 'Wrong arm profile')
    require(profile['mode'] == 'rgb_zero' and profile['requested_epochs'] == 20, 'Wrong RGB/full-epoch recipe')
    order = schedule['epoch_clip_ids']; val = schedule['validation_clip_ids']
    require(len(order) == 20 and len(val) == len(set(val)), 'Frozen schedule identity invalid')
    actual_p = sum(labels[c] for c in order[0]); actual_n = len(order[0]) - actual_p
    require((actual_p, actual_n) == ((288, 120) if fold == 1 else (292, 120)), 'Training quota differs')
    recorded, replayed, predictions, epoch_rows = [], [], [], []
    updates = 0
    for epoch, (row, ids) in enumerate(zip(history, order), 1):
        steps = math.ceil(len(ids) / 24); q, r = divmod(len(ids), steps)
        updates += steps
        require(row['completed_epochs'] == epoch and row['train_clip_ids'] == ids and row['samples'] == len(ids)
            and row['optimizer_steps'] == steps and row['total_updates'] == updates
            and row['actual_effective_batch_sizes'] == [q + 1] * r + [q] * (steps - r)
            and (row['train_P'], row['train_N']) == (actual_p, actual_n), 'Epoch exposure/balanced batches/updates differ')
        require(math.isfinite(row['seconds']) and row['seconds'] > 0 and math.isfinite(row['train_weighted_bce']), 'Invalid epoch timing/loss')
        pred = load('validation-epoch-%02d.json' % epoch)
        require([p['clip_id'] for p in pred] == val and all(p['label'] == labels[p['clip_id']] and math.isfinite(p['logit']) for p in pred),
            'Fixed validation identity/labels/logits differ')
        loss = math.fsum(bce(p['logit'], p['label']) for p in pred) / len(pred)
        original = row['validation_unweighted_bce']
        require(math.isfinite(original) and abs(original - loss) <= BCE_ATOL, 'FP64 BCE replay differs from original FP32 record')
        recorded.append(original); replayed.append(loss); predictions.append(pred)
        epoch_rows.append(dict(fold=fold, arm=arm, source='reused_EXP020_inner' if arm == 'A' else 'new_EXP021_fit', epoch=epoch,
            samples=len(ids), optimizer_steps=steps, total_updates=updates, seconds=row['seconds'],
            online_train_weighted_BCE=row['train_weighted_bce'], original_validation_BCE=original, FP64_validation_BCE=loss))
    selected = min(range(20), key=recorded.__getitem__)
    fp64_selected = min(range(20), key=replayed.__getitem__)
    require(result['selected_epochs'] == selected + 1 and result['optimizer_steps'] == updates
        and replayed[selected] - replayed[fp64_selected] <= BCE_ATOL, 'Selected E inconsistent with earliest recorded BCE minimum')
    require(result['selected_checkpoint']['filename'] == 'best.pt' and result['selected_checkpoint']['completed_epochs'] == selected + 1
        and math.isfinite(result['seconds']) and result['seconds'] > 0, 'Selected checkpoint/timing provenance differs')
    pred = predictions[selected]; y = [p['label'] for p in pred]; z = [p['logit'] for p in pred]
    require((sum(y), len(y) - sum(y)) == ((72, 28) if fold == 1 else (72, 32)), 'Fixed validation class counts differ')
    values = metrics(y, [probability(v) for v in z])
    pos, neg = [p['logit'] for p in pred if p['label']], [p['logit'] for p in pred if not p['label']]
    auc = math.fsum((a > b) + .5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
    values.update(fold=fold, arm=arm, source='reused_EXP020_inner' if arm == 'A' else 'new_EXP021_fit', artifact_directory=str(directory),
        selected_E=selected + 1, FP64_earliest_min_E=fp64_selected + 1, FP64_near_tie_selection=(selected != fp64_selected), AUC_logit=auc,
        selected_FP64_BCE=replayed[selected], selected_positive_BCE=math.fsum(bce(v, 1) for v in pos) / len(pos),
        selected_negative_BCE=math.fsum(bce(v, 0) for v in neg) / len(neg), completed_epochs=20, optimizer_updates=updates,
        fit_seconds=result['seconds'], sum_epoch_seconds=math.fsum(r['seconds'] for r in epoch_rows),
        fragment_floor_pass=auc >= .75 and values['positive_Recall'] >= .5 and values['negative_Recall'] >= .5,
        selected_checkpoint_sha256=result['selected_checkpoint']['sha256'],
        initial_state_sha256=result['initial_state_sha256'], checkpoint_SHA_scope='recorded provenance only; weights not read or rehashed')
    return values, epoch_rows


def dense_review(inputs, run, fold, arm, roles, sources, grid, pixel_reference, plan_sha, selected_fit):
    stage = 'dense-f%02d-%s' % (fold, arm); directory = run / stage
    closed = closure(inputs, run, stage, 'dense_closed')
    def bound(name):
        require(stage + '/' + name in closed, 'Dense consumed file absent from closure')
        return inputs.bind(directory / name, closed[stage + '/' + name]['sha256'])
    result = json.loads(bound('result.json').read_text(encoding='utf-8'))
    score_path = bound('window-scores.csv')
    require(result['status'] == 'inner_role_dense_complete' and result['fold'] == fold and result['arm'] == arm
        and result['plan_sha256'] == plan_sha and result['threshold'] == .5 and result['aggregation'] == 'max'
        and result['selection_used_dense'] is False and result['window_scores_sha256'] == sha(score_path), 'Dense result contract differs')
    require(result['selected_checkpoint']['selected_epochs'] == selected_fit['selected_E']
        and result['selected_checkpoint']['checkpoint_sha256'] == selected_fit['selected_checkpoint_sha256'], 'Dense model differs from BCE-selected checkpoint')
    uids = set(roles[str(fold)]['video_ids']); observed = {u: {} for u in uids}
    with score_path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            uid, key = row['unit_id'], row['frame_tuple_sha256']
            require(uid in uids and key not in observed[uid] and key in grid[uid], 'Unknown/duplicate physical dense window')
            z, score = float(row['logit']), float(row['score'])
            require(math.isfinite(z) and math.isfinite(score) and 0 <= score <= 1 and abs(score - probability(z)) <= 1e-7,
                    'Nonfinite or inconsistent dense logit/probability')
            expected = grid[uid][key]
            require(all(abs(float(row[k]) - expected[k]) <= 1e-8 for k in ('start', 'end', 'target_start_seconds')), 'Physical grid timing differs')
            observed[uid][key] = score
    saved = {r['unit_id']: r for r in result['scores']}
    require(len(saved) == len(result['scores']) == len(uids) and set(saved) == uids, 'Duplicate/incomplete saved video coverage')
    rows = []
    for uid in sorted(uids):
        require(set(observed[uid]) == set(grid[uid]), 'Missing/extra physical dense windows')
        info = result['videos'][uid]
        require(info['canonical_reference_matched'] is True and info['rgb112_digest'] == pixel_reference['videos'][uid]['rgb112_digest']
            and info['windows'] == len(grid[uid]) == pixel_reference['videos'][uid]['windows'], 'Canonical EOF pixel/grid reference differs')
        label = int(sources[uid]['video_label'] == 'pos'); group = sources[uid]['group_id']; maximum = max(observed[uid].values())
        require(saved[uid]['label'] == label and saved[uid]['group_id'] == group and saved[uid]['windows'] == len(grid[uid])
            and abs(saved[uid]['score'] - maximum) <= 1e-12, 'Video label/group/max arithmetic differs')
        rows.append(dict(fold=fold, arm=arm, unit_id=uid, group_id=group, label=label, score=maximum,
            prediction=int(maximum >= .5), windows=len(grid[uid])))
    summary = metrics([r['label'] for r in rows], [r['score'] for r in rows])
    require(all(summary[k] == result['metrics'][k] for k in ('P', 'N', 'TP', 'FP', 'FN', 'TN', 'AUC')), 'Saved per-role metrics differ')
    summary.update(fold=fold, arm=arm, videos=len(rows), windows=sum(r['windows'] for r in rows))
    return rows, summary


def unique_gates(videos, fragment):
    """Independent role→video→session arithmetic; never pools model AUC."""
    indexed = {(r['fold'], r['arm'], r['unit_id']): r for r in videos}
    require(len(indexed) == len(videos), 'Duplicate model/role/video')
    negative = sorted({r['unit_id'] for r in videos if not r['label']})
    group_of = {r['unit_id']: r['group_id'] for r in videos}; arms = ('A', 'B', 'C')
    means = {a: {u: math.fsum(r['prediction'] for k, r in indexed.items() if k[1:] == (a, u))
        / sum(k[1:] == (a, u) for k in indexed) for u in negative} for a in arms}
    groups = sorted({group_of[u] for u in negative})
    fpr = {a: {g: math.fsum(means[a][u] for u in negative if group_of[u] == g) / sum(group_of[u] == g for u in negative)
               for g in groups} for a in arms}
    macro = {a: math.fsum(fpr[a].values()) / len(groups) for a in arms}
    roles = sorted({r['fold'] for r in videos}); references, error_rows = {}, []
    for ref in ('A', 'B'):
        repaired, worsened, mixed = set(), set(), set()
        for uid in sorted(group_of):
            changes = []
            for f in roles:
                if (f, 'C', uid) not in indexed: continue
                x, c = indexed[(f, ref, uid)], indexed[(f, 'C', uid)]
                better = x['prediction'] != x['label'] and c['prediction'] == c['label']
                worse = x['prediction'] == x['label'] and c['prediction'] != c['label']
                changes.append(dict(fold=f, reference_prediction=x['prediction'], C_prediction=c['prediction'], better=better, worse=worse))
            repair, regress = any(r['better'] for r in changes), any(r['worse'] for r in changes)
            label = next(r['label'] for r in videos if r['unit_id'] == uid)
            if not label:
                if repair and not regress: repaired.add(uid)
                if regress: worsened.add(uid)
                if repair and regress: mixed.add(uid)
            if repair or regress:
                error_rows.append(dict(reference=ref, unit_id=uid, group_id=group_of[uid], label=label,
                    in_R=uid in repaired, in_W=uid in worsened, role_inconsistent=repair and regress,
                    roles=json.dumps(changes, ensure_ascii=False, separators=(',', ':'))))
        role_guard = {}
        for fold in roles:
            x = [r for r in videos if r['fold'] == fold and r['arm'] == ref]
            c = [r for r in videos if r['fold'] == fold and r['arm'] == 'C']
            mx, mc = metrics([r['label'] for r in x], [r['score'] for r in x]), metrics([r['label'] for r in c], [r['score'] for r in c])
            role_guard[str(fold)] = mc['FP'] <= mx['FP'] and mc['FN'] <= mx['FN']
        improved = [g for g in groups if fpr['C'][g] < fpr[ref][g]]; regressed = [g for g in groups if fpr['C'][g] > fpr[ref][g]]
        references[ref] = dict(R=sorted(repaired), W=sorted(worsened), role_inconsistent=sorted(mixed), unique_net_repair=len(repaired) - len(worsened),
            per_role_FP_FN_nonincrease=role_guard, improved_negative_groups=improved, regressed_negative_groups=regressed,
            macro_FPR_reference=macro[ref], macro_FPR_C=macro['C'],
            passed=all(role_guard.values()) and len(repaired) - len(worsened) >= 2 and len(improved) >= 2 and macro['C'] < macro[ref])
    floor = all(r['fragment_floor_pass'] for r in fragment if r['arm'] in ('B', 'C'))
    require(sum(r['arm'] in ('B', 'C') for r in fragment) == 4, 'Four new fit floors required')
    result = dict(fragment_floor_all_four=floor, references=references, negative_video_role_mean_FPR=means,
        unique_group_FPR=fpr, unique_group_macro_FPR=macro,
        promote_to_matched_formal_design_only=floor and all(r['passed'] for r in references.values()))
    session_rows = [dict(session_alias='S%d' % (groups.index(g) + 1), group_id=g, arm=a, FPR=fpr[a][g],
        unique_negative_videos=sum(group_of[u] == g for u in negative), macro_FPR=macro[a]) for g in groups for a in arms]
    return result, error_rows, session_rows


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8'); return
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def figures(fragment, role_results, sessions, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = dict(A='#666666', B='#D98B23', C='#2878B5')
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.5), sharey=True)
    fig.subplots_adjust(left=.065, right=.99, bottom=.22, top=.80, wspace=.25)
    for ax, (key, label, cut) in zip(axes, [('AUC_logit', 'Logit AUC', .75), ('positive_Recall', 'Positive Recall', .5), ('negative_Recall', 'Negative Recall', .5)]):
        for j, arm in enumerate(('A', 'B', 'C')):
            rows = sorted((r for r in fragment if r['arm'] == arm), key=lambda r: r['fold'])
            ax.bar(np.arange(2) + (j - 1) * .24, [r[key] for r in rows], width=.22, color=colors[arm], label=arm)
        ax.axhline(cut, ls='--', c='black', lw=.8); ax.set_xticks([0, 1], ['Fold1 inner', 'Fold4 inner'])
        ax.set_ylim(0, 1.055); ax.set_title(label, fontsize=10); ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Selected validation metric')
    fig.legend(handles=[Patch(color=colors[a], label=a + (' (reused RGB)' if a == 'A' else ' (fresh fusion)')) for a in colors], ncol=3, loc='upper center', bbox_to_anchor=(.55, .94), frameon=False)
    fig.text(.065, .985, 'EXP-021: selected clip validation (known development splits)', va='top', weight='bold', fontsize=11)
    fig.text(.065, .055, 'Dashed lines: registered floors; all four B/C fits must pass. E uses BCE, never dense results.', fontsize=8)
    save_figure(fig, out / 'clip-floors', 'EXP021 selected clip floors'); plt.close(fig)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 3.7)); fig.subplots_adjust(left=.065, right=.99, bottom=.24, top=.80, wspace=.28)
    for j, arm in enumerate(('A', 'B', 'C')):
        rows = sorted((r for r in role_results if r['arm'] == arm), key=lambda r: r['fold'])
        x = np.arange(2) + (j - 1) * .24
        a.bar(x, [r['FP'] for r in rows], .22, color=colors[arm])
        a.bar(x, [r['FN'] for r in rows], .22, bottom=[r['FP'] for r in rows], facecolor='white', edgecolor=colors[arm], hatch='///')
        for xx, row in zip(x, rows): a.text(xx, row['FP'] + row['FN'] + .10, '%d/%d' % (row['FP'], row['FN']), ha='center', fontsize=7)
        group_rows = [r for r in sessions if r['arm'] == arm]
        b.bar(np.arange(len(group_rows)) + (j - 1) * .24, [r['FPR'] for r in group_rows], .22, color=colors[arm], label='%s macro %.3f' % (arm, group_rows[0]['macro_FPR']))
    a.set_title('(a) Dense errors: label = FP / FN', fontsize=10); a.set_xticks([0, 1], ['Fold1 (23 roles)', 'Fold4 (24 roles)']); a.set_ylabel('Videos (count)')
    a.set_ylim(0, max(r['FP'] + r['FN'] for r in role_results) + 2)
    a.legend(handles=[Patch(color='#888888', label='FP (solid)'), Patch(facecolor='white', edgecolor='#888888', hatch='///', label='FN (hatched)')], loc='upper left', frameon=False, fontsize=8)
    aliases = sorted({r['session_alias'] for r in sessions})
    b.set_xticks(np.arange(len(aliases)), aliases); b.set_ylim(0, 1.06); b.set_ylabel('Unique negative-session FPR'); b.set_title('(b) Deduplicated session errors', fontsize=10); b.legend(frameon=False, fontsize=8)
    fig.text(.065, .985, 'EXP-021: dense max / 0.5 and unique-session consequences', va='top', weight='bold', fontsize=11)
    fig.text(.065, .07, '47 video roles share five videos: 42 unique videos. No pooled cross-model AUC.\nSession FPR: average roles within video, then videos within session; aliases map to negative-sessions.csv.', fontsize=8)
    save_figure(fig, out / 'dense-errors-and-sessions', 'EXP021 dense errors and unique-session FPR'); plt.close(fig)


def save_figure(fig, path, title):
    fig.savefig(str(path) + '.png', dpi=240, metadata={'Software': 'Matplotlib', 'Title': title})
    fig.savefig(str(path) + '.pdf', metadata={'Creator': 'Matplotlib', 'Author': '', 'Title': title})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=ROOT / 'data/results/exp021' / RUN_ID)
    p.add_argument('--plan', type=Path, default=ROOT / 'data/metadata/EXP-021/energy-plan-v2')
    p.add_argument('--reference-root', type=Path, default=ROOT / 'data/results/exp020/head-s7s8-20261103-v2')
    p.add_argument('--partial-fit-review', choices=['f01-B-s7', 'f01-C-s7', 'f04-B-s7', 'f04-C-s7'])
    p.add_argument('--out', type=Path)
    a = p.parse_args(argv)
    run, plan_dir, reference = a.run.resolve(), a.plan.resolve(), a.reference_root.resolve()
    default_name = ('partial-' + run.name + '-' + a.partial_fit_review + '-v1') if a.partial_fit_review else ('report-' + run.name + '-v1')
    out = (a.out if a.out is not None else ROOT / 'data/results/exp021' / default_name).resolve()
    require(not out.exists(), 'Choose a new output directory; no overwrite')
    require(all(out != parent and parent not in out.parents for parent in (run, plan_dir, reference)), 'Derived output must stay outside original runs and frozen plan')
    inputs = Inputs(); plan = inputs.read(plan_dir / 'plan.json'); plan_sha = sha(plan_dir / 'plan.json')
    require(plan['spec_id'] == 'EXP-021' and plan['epochs'] == 20 and plan['seed'] == 7, 'Wrong frozen plan')
    schedule = inputs.read(plan_dir / 'schedules.json', plan['schedules_sha256'])
    base = ROOT / plan['base17_path']; label_file = base / 'windows.jsonl'
    labels = {r['clip_id']: r['label'] for r in inputs.jsonl(label_file, plan['source_fingerprints'][label_file.relative_to(ROOT).as_posix()])}
    fragment, epoch_rows = [], []
    if a.partial_fit_review:
        item = a.partial_fit_review; fold = int(item[1:3]); arm = item.split('-')[1]
        closed = closure(inputs, run, item, 'fit_closed')
        review, epochs = fit_review(inputs, run / item, fold, arm, schedule[str(fold)], labels, closed)
        fragment.append(review); epoch_rows.extend(epochs)
        status = 'partial_one_closed_fit_only_no_overall_gate'
        payload = dict(fit=review, overall_gate=None, all_results_complete=False)
    else:
        terminal = inputs.read(run / 'closed-run.json'); finished = inputs.read(run / 'pipeline-result.json')
        require(terminal['status'] == 'run_closed' and finished['status'] == 'four_fits_six_dense_complete_pending_independent_audit'
            and not any((run / n).exists() for n in ('pipeline-failure.json', 'hard-time-limit.json')), 'Run is incomplete/failed; no final derivation')
        execution = inputs.read(run / 'execution-inputs.json')
        require(execution['plan_sha256'] == plan_sha, 'Execution/plan SHA differs')
        for fold in (1, 4):
            for arm in ('A', 'B', 'C'):
                item = 'f%02d-%s-s7' % (fold, arm)
                directory = reference / plan['A_references'][str(fold)]['item'] if arm == 'A' else run / item
                closed = plan['A_references'][str(fold)]['files'] if arm == 'A' else closure(inputs, run, item, 'fit_closed')
                if arm != 'A':
                    for name in ('best.pt', 'last.pt'):
                        path = directory / name; key = item + '/' + name
                        if not path.is_file(): raise Incomplete('Complete-mode checkpoint not yet collected: ' + str(path))
                        require(key in closed and path.stat().st_size == closed[key]['size_bytes'], 'Checkpoint size differs from collected closure')
                review, epochs = fit_review(inputs, directory, fold, arm, schedule[str(fold)], labels, closed)
                if arm == 'A': require(review['selected_E'] == plan['A_references'][str(fold)]['selected_epochs'], 'Reused A selected E differs')
                fragment.append(review); epoch_rows.extend(epochs)
        require(sum(r['optimizer_updates'] for r in fragment if r['arm'] != 'A') == 1400, 'New-fit budget must be80 epochs/1400 updates')
        source_path = base / 'sources.csv'; inputs.bind(source_path, plan['source_fingerprints'][source_path.relative_to(ROOT).as_posix()])
        with source_path.open(encoding='utf-8-sig', newline='') as f: sources = {r['unit_id']: r for r in csv.DictReader(f)}
        roles = plan['roles']; uid1, uid4 = set(roles['1']['video_ids']), set(roles['4']['video_ids'])
        require((len(uid1), len(uid4), len(uid1 & uid4), len(uid1 | uid4)) == (23, 24, 5, 42), 'Role overlap boundary differs')
        pixel_ref = inputs.read(ROOT / plan['dense_reference'], plan['source_fingerprints'][plan['dense_reference']])
        grid_path = ROOT / pixel_ref['grid_path']; inputs.bind(grid_path, pixel_ref['grid_sha256'])
        grid = {u: {} for u in uid1 | uid4}
        with grid_path.open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line); uid = row['unit_id']
                if uid not in grid: continue
                key = hashlib.sha256(np.asarray(row['frame_indices'], dtype='<i4').tobytes()).hexdigest()
                require(key not in grid[uid], 'Frozen grid contains duplicate physical tuple')
                grid[uid][key] = dict(start=row['source_pts_seconds'][0], end=row['source_pts_seconds'][-1], target_start_seconds=row['target_start_seconds'])
        videos, role_results = [], []
        for fold in (1, 4):
            for arm in ('A', 'B', 'C'):
                selected_fit = next(r for r in fragment if r['fold'] == fold and r['arm'] == arm)
                rows, summary = dense_review(inputs, run, fold, arm, roles, sources, grid, pixel_ref, plan_sha, selected_fit)
                videos.extend(rows); role_results.append(summary)
        verdict, errors, sessions = unique_gates(videos, fragment)
        producer = inputs.read(run / 'comparison.json', finished['comparison_sha256'])
        require(producer['gate']['promote_to_matched_formal_design_only'] == verdict['promote_to_matched_formal_design_only'], 'Producer/independent gate differs')
        for ref in ('A', 'B'):
            for key in ('R', 'W', 'role_inconsistent', 'improved_negative_groups', 'regressed_negative_groups'):
                require(set(producer['gate']['references'][ref][key]) == set(verdict['references'][ref][key]), 'Producer/independent unique-error arithmetic differs')
        status = 'independently_recomputed_development_results_pending_readonly_review'
        payload = dict(fragment=fragment, dense_roles=role_results, gate=verdict, role_videos=47, unique_videos=42, shared_videos=5,
            new_fit_epochs=80, new_optimizer_updates=1400, reused_A_inner_fits=2, no_pooled_AUC=True)
    inputs.unchanged()
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'clip-results.csv', fragment); write_csv(out / 'fit-epochs.csv', epoch_rows)
    if not a.partial_fit_review:
        write_csv(out / 'dense-video-results.csv', videos); write_csv(out / 'dense-role-results.csv', role_results)
        write_csv(out / 'negative-sessions.csv', sessions); write_csv(out / 'error-changes.csv', errors)
        figures(fragment, role_results, sessions, out)
    metadata = dict(status=status, run_id=run.name, original_run=str(run), reference_root=str(reference), plan=str(plan_dir),
        input_sha256=inputs.hashes, source_path=str(Path(__file__).resolve()), source_sha256=sha(Path(__file__).resolve()),
        argv=sys.argv if argv is None else argv, environment=dict(python=sys.version, numpy=np.__version__, platform=platform.platform()),
        BCE_replay_absolute_tolerance=BCE_ATOL, selection_semantics='earliest recorded FP32 minimum is actual E; independent FP64 earliest E and near-tie status explicitly retained',
        limits=['Known development splits; not independent repetitions or unseen-camera/external testing',
                'No model/weight/media execution or weight SHA pass', 'Online train loss includes the actual train path; not fixed-eval train fitting',
                'No training/threshold/seed change; no automatic followup', 'Figures require actual visual QA after full generation'],
        result=payload, output_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    with (out / 'result.json').open('x', encoding='utf-8') as f: json.dump(metadata, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(dict(status=status, output=str(out)), ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Incomplete as e:
        print(json.dumps(dict(status='incomplete_no_final_report', reason=str(e)), ensure_ascii=False))
        sys.exit(2)
