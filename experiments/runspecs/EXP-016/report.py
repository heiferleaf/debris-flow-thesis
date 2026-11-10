"""Render thesis-reference figures from the audited completed paired run."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT/'data/results/exp016/paired-s7-20261030-v2'
OUT = RUN/'analysis-v1'
FIG = ROOT/'docs/process/experiments/figures/EXP-016'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 10, 'pdf.fonttype': 42})
    audit = json.loads((RUN/'audit-v1/audit.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'paired_artifacts_audited'
    dense = json.loads((RUN/'paired-dense/result.json').read_text(encoding='utf-8'))
    modes = [('rgb', 'rgb_zero', 'RGB', '#2f64a2'), ('diff', 'rgb_diff', 'RGB＋帧差', '#cc7c32')]
    OUT.mkdir(exist_ok=False)
    FIG.mkdir(parents=True, exist_ok=False)
    outputs = []

    def save(fig, name):
        for extension in ('png', 'pdf', 'svg'):
            p = FIG/(name+'.'+extension)
            fig.savefig(p, dpi=180, bbox_inches='tight')
            outputs.append(p)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for folder, mode, label, color in modes:
        history = [json.loads(x) for x in (RUN/folder/'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
        epochs = [r['completed_epochs'] for r in history]
        axes[0].plot(epochs, [r['train_loss'] for r in history], marker='.', color=color, label=label)
        axes[1].plot(epochs, [r['validation_unweighted_bce'] for r in history], marker='.', color=color, label=label)
        best = audit['best_epochs'][folder]
        axes[1].scatter([best], [history[best-1]['validation_unweighted_bce']], marker='*', s=110, color=color)
    constant = ((37/89)*356+148)/504*math.log(2)
    axes[0].axhline(constant, color='gray', linestyle='--', label='恒定q=0.5：0.4071')
    for ax, title, ylabel in [(axes[0], '训练损失（正类加权）', '加权BCE／片段'),
                              (axes[1], '固定片段验证损失与选中轮次', '未加权BCE／片段')]:
        ax.set(title=title, xlabel='训练轮次（轮）', ylabel=ylabel, xticks=[1, 5, 10, 15, 20])
        ax.grid(alpha=.15)
        ax.legend(fontsize=9)
    fig.suptitle('EXP-016：两组均完成20轮；固定开发划分、seed7')
    fig.text(.5, -.04, '训练504窗／轮；固定验证136窗。两图权重和运行状态不同，不直接比较绝对高低。星号为各组选中best。', ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, 'training-curves')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    positions = np.arange(4)
    for offset, (_, mode, label, color) in zip([-.18, .18], modes):
        metric = audit['metrics'][mode]
        values = [metric[k] for k in ('TP', 'FP', 'FN', 'TN')]
        bars = axes[0].bar(positions+offset, values, .36, label=label, color=color)
        axes[0].bar_label(bars, padding=3)
        rows = sorted(dense['scores'][mode], key=lambda r: -r['score'])
        fpr, tpr, tp, fp = [0.], [0.], 0, 0
        for i, row in enumerate(rows):
            tp += row['label']; fp += 1-row['label']
            if i+1 == len(rows) or rows[i+1]['score'] != row['score']:
                fpr.append(fp/metric['N']); tpr.append(tp/metric['P'])
        axes[1].plot(fpr, tpr, color=color, label=f"{label}（AUC={metric['AUC']:.3f}）")
    axes[0].set(title='固定0.5判定线下的视频计数', xlabel='实际分类结果', ylabel='视频数量（条）',
                xticks=positions, xticklabels=['TP', 'FP', 'FN', 'TN'], ylim=(0, 27))
    axes[0].legend()
    axes[1].plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    axes[1].set(title='开发验证视频ROC', xlabel='FPR（负视频误报比例）', ylabel='Recall（正视频召回比例）',
                xlim=(0, 1), ylim=(0, 1.03))
    axes[1].legend(loc='lower right')
    fig.suptitle('31条开发验证视频：24正／7负，负视频仅来自3个场次组')
    fig.text(.5, -.03, '32567窗完整max聚合。该验证资料也用于选轮；图示为开发结果，不能替代独立测试或正式交叉验证。', ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, 'video-results')

    cases = {uid: [] for uid in ('v1_120', 'v1_62', 'v1_121')}
    with (RUN/'paired-dense/window-scores.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['unit_id'] in cases:
                cases[row['unit_id']].append(row)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    case_summary = []
    for ax, (uid, rows) in zip(axes, cases.items()):
        scores = [float(r['rgb_diff']) for r in rows]
        peak = max(rows, key=lambda r: float(r['rgb_diff']))
        above = sum(s >= .5 for s in scores)
        case_summary.append(dict(unit_id=uid, windows=len(rows), above_threshold=above,
                                 peak_start=float(peak['start']), peak_end=float(peak['end']), peak_score=float(peak['rgb_diff'])))
        for _, mode, label, color in modes:
            ax.scatter([(float(r['start'])+float(r['end']))/120 for r in rows],
                       [float(r[mode]) for r in rows], s=4, color=color, label=label)
        ax.axhline(.5, color='gray', linestyle='--', linewidth=.8)
        ax.set(title=f'{uid}（整视频标注neg）：融合{above}/{len(rows)}窗达到0.5',
               xlabel='窗口中点时间（分）', ylabel='窗口预测分数', ylim=(-.03, 1.03))
        ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('剩余两条误报与同场次成功纠正对照；仅作错误分析，不改阈值或标签')
    fig.tight_layout()
    save(fig, 'negative-video-timelines')

    with (OUT/'video-predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['unit_id', 'group_id', 'label', 'windows', 'rgb_score', 'diff_score', 'rgb_prediction', 'diff_prediction'])
        writer.writeheader()
        rgb = {r['unit_id']: r for r in dense['scores']['rgb_zero']}
        for row in dense['scores']['rgb_diff']:
            a = rgb[row['unit_id']]
            writer.writerow(dict(unit_id=row['unit_id'], group_id=row['group_id'], label=row['label'], windows=row['windows'],
                                 rgb_score=a['score'], diff_score=row['score'], rgb_prediction=int(a['score'] >= .5),
                                 diff_prediction=int(row['score'] >= .5)))
    (OUT/'error-summary.json').write_text(json.dumps(case_summary, indent=2), encoding='utf-8')
    index = dict(status='derived_from_audited_completed_run', audit_sha256=sha(RUN/'audit-v1/audit.json'),
                 score_source_sha256=sha(RUN/'paired-dense/window-scores.csv'), generator_sha256=sha(__file__),
                 files={p.relative_to(ROOT).as_posix(): sha(p) for p in outputs})
    (OUT/'figure-index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    print(json.dumps(dict(figures=len(outputs), analysis=str(OUT))))


if __name__ == '__main__':
    main()
