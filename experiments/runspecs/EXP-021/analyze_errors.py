"""Closed dense-score diagnostics only: fixed0.5, no model/threshold/gate changes."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
RUN = 'energy-s7-20261103-v2'


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def describe(rows, label):
    """Rows are one source video. Gaps split runs; short tail alignment does not."""
    rows = sorted(rows, key=lambda r: float(r['target_start_seconds']))
    require(rows and label in (0, 1) and len({r['unit_id'] for r in rows}) == 1, 'Require one nonempty source video')
    values = [float(r['score']) for r in rows]
    require(all(math.isfinite(s) and 0 <= s <= 1 for s in values), 'Invalid saved probability')
    longest = current = 0; previous = None
    for row, value in zip(rows, values):
        start, end, target = [float(row[k]) for k in ('start', 'end', 'target_start_seconds')]
        require(all(math.isfinite(x) for x in (start, end, target)) and start <= end, 'Invalid window times')
        consecutive = previous is not None and target - previous[0] <= 1.500001 and start <= previous[1] + 1e-8
        current = (current + 1 if consecutive else 1) if value >= .5 else 0
        longest = max(longest, current); previous = target, end
    positive = sum(s >= .5 for s in values); prediction = int(max(values) >= .5)
    case = ('TP' if prediction else 'FN') if label else ('FP' if prediction else 'TN')
    return dict(windows=len(values), positive_windows=positive, positive_fraction=positive / len(values),
        score_min=min(values), score_median=statistics.median(values), score_max=max(values),
        longest_positive_run_windows=longest, longest_positive_run_fraction=longest / len(values),
        false_positive_windows=positive if label == 0 else None, video_prediction=prediction, video_case=case)


def plot(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    colors = dict(A='#666666', B='#D98B23', C='#2878B5')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    fig.subplots_adjust(left=.07, right=.99, top=.79, bottom=.26, wspace=.18)
    for ax, fold in zip(axes, (1, 4)):
        negative = [r for r in rows if r['fold'] == fold and r['label'] == 0]
        ids = sorted({r['unit_id'] for r in negative})
        for j, arm in enumerate(colors):
            series = {r['unit_id']: r for r in negative if r['arm'] == arm}
            x = [i + (j - 1) * .25 for i in range(len(ids))]
            ax.bar(x, [series[u]['positive_fraction'] for u in ids], .23, color=colors[arm])
            ax.scatter(x, [series[u]['longest_positive_run_fraction'] for u in ids], marker='D', s=18, c='black', edgecolors='white', linewidths=.4, zorder=3)
        ax.set_xticks(list(range(len(ids))), ids, rotation=38, ha='right'); ax.set_ylim(-.02, 1.05)
        ax.set_title('Fold%d inner: %d negative videos' % (fold, len(ids)), fontsize=10); ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Fraction of saved windows')
    fig.text(.07, .98, 'EXP-021: negative-video score persistence at the fixed0.5 threshold', va='top', weight='bold', fontsize=11)
    fig.legend(handles=[Patch(color=colors[a], label=a) for a in colors] + [Line2D([], [], marker='D', color='black', ls='', label='Longest positive run / all windows')],
        loc='upper center', bbox_to_anchor=(.55, .91), ncol=4, frameon=False, fontsize=8)
    fig.text(.07, .045, 'Bars: positive-window fraction. Gaps >1.500001s break runs; tail alignment is retained.\nRoles use different fitted models; shared videos are not independent repeats. This tests no replacement aggregation.', fontsize=8)
    fig.savefig(out / 'negative-window-persistence.png', dpi=240, metadata={'Software': 'Matplotlib', 'Title': 'EXP021 negative-window persistence'})
    fig.savefig(out / 'negative-window-persistence.pdf', metadata={'Creator': 'Matplotlib', 'Author': '', 'Title': 'EXP021 negative-window persistence'})
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=ROOT / 'data/results/exp021' / RUN)
    p.add_argument('--plan', type=Path, default=ROOT / 'data/metadata/EXP-021/energy-plan-v2')
    p.add_argument('--partial-role', type=int, choices=(1, 4))
    p.add_argument('--out', type=Path)
    a = p.parse_args(argv); run, plan_dir = a.run.resolve(), a.plan.resolve()
    name = 'window-errors-' + run.name + ('-partial-role%d' % a.partial_role if a.partial_role else '') + '-v1'
    out = (a.out if a.out else ROOT / 'data/results/exp021' / name).resolve()
    require(not out.exists() and out.parent == (ROOT / 'data/results/exp021').resolve(), 'Choose a new direct EXP021 derived directory, no overwrite')
    inputs = {}
    def bind(path, expected=None):
        path = Path(path).resolve(); require(path.is_file(), 'Incomplete closed input: ' + str(path))
        value = sha(path); require(expected is None or value == expected, 'Input SHA mismatch: ' + str(path)); inputs[str(path)] = value; return path
    def read(path, expected=None): return json.loads(bind(path, expected).read_text(encoding='utf-8'))
    plan = read(plan_dir / 'plan.json'); require(plan['spec_id'] == 'EXP-021', 'Wrong experiment plan')
    source_path = ROOT / plan['base17_path'] / 'sources.csv'
    with bind(source_path, plan['source_fingerprints'][source_path.relative_to(ROOT).as_posix()]).open(encoding='utf-8-sig', newline='') as f:
        sources = {r['unit_id']: r for r in csv.DictReader(f)}
    membership = {u: [f for f in (1, 4) if u in plan['roles'][str(f)]['video_ids']] for u in {u for r in plan['roles'].values() for u in r['video_ids']}}
    require((len(membership), sum(len(r) == 2 for r in membership.values())) == (42, 5), 'Frozen role overlap differs')
    rows, summaries = [], []
    for fold in ((a.partial_role,) if a.partial_role else (1, 4)):
        expected = set(plan['roles'][str(fold)]['video_ids'])
        for arm in ('A', 'B', 'C'):
            stage = 'dense-f%02d-%s' % (fold, arm); directory = run / stage
            receipt = read(run / ('closed-' + stage + '.json')); require(receipt['status'] == 'dense_closed', 'Dense stage not closed')
            listed = {r['path']: r for r in receipt['files']}; require(len(listed) == len(receipt['files']), 'Duplicate receipt entry')
            result = read(directory / 'result.json', listed[stage + '/result.json']['sha256'])
            require(result['status'] == 'inner_role_dense_complete' and result['fold'] == fold and result['arm'] == arm
                and result['plan_sha256'] == inputs[str((plan_dir / 'plan.json').resolve())], 'Wrong closed result/role/plan')
            csv_path = bind(directory / 'window-scores.csv', listed[stage + '/window-scores.csv']['sha256'])
            require(sha(csv_path) == result['window_scores_sha256'], 'Dense score reference differs')
            grouped = defaultdict(list)
            with csv_path.open(encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f): grouped[row['unit_id']].append(row)
            require(set(grouped) == expected, 'Incomplete/extra video coverage')
            saved = {r['unit_id']: r for r in result['scores']}; require(len(saved) == len(result['scores']) == len(expected), 'Invalid saved video coverage')
            current = []
            for uid in sorted(expected):
                require(len({r['frame_tuple_sha256'] for r in grouped[uid]}) == len(grouped[uid]), 'Repeated physical window')
                label, group = int(sources[uid]['video_label'] == 'pos'), sources[uid]['group_id']
                stat = describe(grouped[uid], label)
                require(saved[uid]['label'] == label and saved[uid]['group_id'] == group and saved[uid]['windows'] == stat['windows']
                    and saved[uid]['score'] == stat['score_max'], 'Frozen label/group or max/count differs')
                current.append(dict(fold=fold, arm=arm, unit_id=uid, group_id=group, label=label, **stat))
            counts = Counter(r['video_case'] for r in current)
            require(all(counts[k] == result['metrics'][k] for k in ('TP', 'FP', 'FN', 'TN')), 'Video confusion differs')
            negative = [r for r in current if not r['label']]
            summaries.append(dict(fold=fold, arm=arm, videos=len(current), TP=counts['TP'], FP=counts['FP'], FN=counts['FN'], TN=counts['TN'],
                mean_negative_video_positive_fraction=statistics.mean(r['positive_fraction'] for r in negative),
                median_negative_video_positive_fraction=statistics.median(r['positive_fraction'] for r in negative),
                video_level_FN_cases=[r['unit_id'] for r in current if r['video_case'] == 'FN']))
            rows.extend(current)
    require(all(sha(Path(path)) == value for path, value in inputs.items()), 'Closed inputs changed during diagnostic')
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'video-window-diagnostics.csv').open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if not a.partial_role: plot(rows, out)
    (out / 'analyze_exp021_errors.py').write_bytes(Path(__file__).read_bytes())
    record = dict(status='partial_one_role_only_no_overall_conclusion' if a.partial_role else 'all_six_closed_dense_descriptive_diagnostic_pending_review',
        run_id=run.name, roles_analyzed=[a.partial_role] if a.partial_role else [1, 4], complete_six_dense=not bool(a.partial_role),
        source_sha256=sha(Path(__file__)), input_sha256=inputs, threshold=.5, maximum_contiguous_target_step_seconds=1.500001,
        per_role_summaries=summaries, role_membership=membership, negative_role_membership={u: r for u, r in membership.items() if sources[u]['video_label'] == 'neg'},
        full_design_boundary=dict(video_roles=47, unique_videos=42, shared_videos=5), argv=sys.argv if argv is None else argv,
        environment=dict(python=sys.version, platform=platform.platform()),
        limits=['Descriptive saved-score analysis, no new gate/training/threshold scan or aggregation replacement',
            'Negative-video positive windows are false alarms; negative windows within positive videos are not automatically FN',
            'FN means positive VIDEO with max saved score<0.5 only; no per-stage truth inferred',
            'Continuity is sampled-window target order within one video; gaps/source boundaries are never joined',
            'Per-role summaries primary; windows correlated, no pooled AUC, significance or cross-role ensemble', 'Full figure requires actual visual QA'],
        output_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    with (out / 'result.json').open('x', encoding='utf-8') as f: json.dump(record, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(dict(status=record['status'], output=str(out)), ensure_ascii=False))


if __name__ == '__main__': main()
