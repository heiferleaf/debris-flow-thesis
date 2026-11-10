"""Summarize audited paired runs without selecting seeds, epochs or thresholds."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARMS = [('rgb', 'rgb_zero', 'RGB'), ('diff', 'rgb_diff', 'RGB+frame difference')]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, action='append', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--figures', type=Path, required=True)
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.figures = a.figures.resolve()
    a.out.relative_to(ROOT/'data/results/exp016')
    a.figures.relative_to(ROOT/'docs/process/experiments/figures')
    if len(a.run) != 2 or a.out.exists() or a.figures.exists():
        raise ValueError('Exactly two audited runs and new output directories required')
    sources, rows, histories, predictions = {}, [], {}, []
    for run in a.run:
        run = run.resolve()
        run.relative_to(ROOT/'data/results/exp016')
        audit_path = run/'audit-v1/audit.json'
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        if audit['status'] != 'paired_artifacts_audited':
            raise ValueError('Unaudited run')
        sources[audit_path.relative_to(ROOT).as_posix()] = sha(audit_path)
        dense_path = run/'paired-dense/result.json'
        dense = json.loads(dense_path.read_text(encoding='utf-8'))
        if sha(dense_path) != audit['sources_sha256'][dense_path.relative_to(ROOT).as_posix()]:
            raise ValueError('Dense result changed after audit')
        for folder, mode, label in ARMS:
            profile_path = run/folder/'profile.json'
            profile = json.loads(profile_path.read_text(encoding='utf-8'))
            seed = profile['seed']
            if sha(profile_path) != audit['sources_sha256'][profile_path.relative_to(ROOT).as_posix()]:
                raise ValueError('Profile changed after audit')
            history_path = run/folder/'epochs.jsonl'
            if sha(history_path) != audit['sources_sha256'][history_path.relative_to(ROOT).as_posix()]:
                raise ValueError('History changed after audit')
            history = [json.loads(line) for line in history_path.read_text(encoding='utf-8').splitlines()]
            if len(history) != 20 or (seed, mode) in histories:
                raise ValueError('Missing epochs or duplicate arm')
            histories[seed, mode] = history
            scores = dense['scores'][mode]
            best = audit['best_epochs'][folder]
            row = dict(seed=seed, mode=mode, run_id=run.name, **audit['metrics'][mode],
                       best_epoch=best, best_validation_bce=history[best-1]['validation_unweighted_bce'],
                       score_min=min(r['score'] for r in scores), score_max=max(r['score'] for r in scores),
                       distinct_predicted_classes=len({int(r['score'] >= .5) for r in scores}),
                       epoch_seconds_sum=sum(r['seconds'] for r in history))
            rows.append(row)
            predictions.extend(dict(seed=seed, mode=mode, **r, prediction=int(r['score'] >= .5)) for r in scores)
    if {r['seed'] for r in rows} != {7, 8} or len(rows) != 4:
        raise ValueError('This comparison requires both complete pairs: seed7 and seed8')
    truth = {(r['unit_id'], r['group_id'], r['label'], r['windows'])
             for r in predictions if r['seed'] == 7 and r['mode'] == 'rgb_zero'}
    if any({(r['unit_id'], r['group_id'], r['label'], r['windows']) for r in predictions
            if r['seed'] == seed and r['mode'] == mode} != truth for seed, mode in histories):
        raise ValueError('Compared videos/truth/coverage differ')
    a.out.mkdir(parents=True)
    a.figures.mkdir(parents=True)
    for filename, records in [('metrics.csv', rows), ('video-predictions.csv', predictions)]:
        with (a.out/filename).open('x', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'pdf.fonttype': 42})
    colors, labels = {'rgb_zero': '#2f64a2', 'rgb_diff': '#cc7c32'}, {'rgb_zero': 'RGB', 'rgb_diff': 'RGB＋帧差'}
    outputs = []

    def save(fig, name):
        for extension in ('png', 'pdf', 'svg'):
            path = a.figures/(name+'.'+extension)
            fig.savefig(path, dpi=180, bbox_inches='tight')
            outputs.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for (seed, mode), history in sorted(histories.items()):
        style = '-' if seed == 7 else '--'
        epochs = list(range(1, 21))
        for ax, key in zip(axes, ('train_loss', 'validation_unweighted_bce')):
            ax.plot(epochs, [r[key] for r in history], style, color=colors[mode],
                    label=f'{labels[mode]}，seed{seed}')
        best = next(r['best_epoch'] for r in rows if (r['seed'], r['mode']) == (seed, mode))
        axes[1].scatter(best, history[best-1]['validation_unweighted_bce'], marker='*', s=90, color=colors[mode])
    axes[0].axhline(2*148/504*math.log(2), linestyle=':', color='gray', label='恒定q=0.5：0.4071')
    for ax, title, ylabel in zip(axes, ('正类加权训练损失', '固定片段验证损失与最佳轮次'),
                                 ('加权BCE／片段', '未加权BCE／片段')):
        ax.set(title=title, ylabel=ylabel, xlabel='训练轮次（轮）', xticks=[1, 5, 10, 15, 20])
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.suptitle('EXP-016：两个随机种子，均为同一个开发划分')
    fig.text(.5, -.025, '实线seed7，虚线seed8；两组各20轮。训练与验证损失权重不同，不能直接比较两图绝对高低。', ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, 'paired-seed-curves')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in zip(axes, ('FPR', 'Recall', 'AUC'),
                              ('负视频误报比例 FPR', '正视频召回比例 Recall', '视频排序指标 AUC')):
        for _, mode, _ in ARMS:
            values = [next(r[key] for r in rows if r['seed'] == seed and r['mode'] == mode) for seed in (7, 8)]
            ax.plot([7, 8], values, 'o-', color=colors[mode], label=labels[mode])
            for seed, value in zip((7, 8), values):
                ax.annotate(f'{value:.3f}', (seed, value), xytext=(0, 7 if mode == 'rgb_zero' else -14),
                            textcoords='offset points', ha='center', fontsize=9)
        ax.set(title=title, xlabel='随机种子', ylabel=f'{key}（0–1）', xticks=[7, 8], xlim=(6.7, 8.3), ylim=(-.08, 1.13))
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    fig.suptitle('同一批31条开发验证视频：24正／7负；固定0.5与完整max聚合')
    fig.text(.5, -.035, '同一验证资料也用于选轮；负视频来自3个场次组。连线仅连接两次运行，不代表样本增加或趋势预测。', ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, 'paired-seed-video-metrics')
    summary = dict(status='derived_from_two_audited_pairs',
                   scope='two random seeds on the same development split; not independent test or formal CV',
                   metrics=rows, sources_sha256=sources, generator_sha256=sha(Path(__file__)),
                   outputs_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in outputs},
                   automatically_authorizes_training=False)
    (a.out/'comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(status=summary['status'], figures=len(outputs), out=str(a.out))))


if __name__ == '__main__':
    main()
