"""Plot the completed D-zero diagnostic, never new predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    run = args.run.resolve()
    run.relative_to(ROOT/'data/results/exp016')
    result = json.loads((run/'result.json').read_text(encoding='utf-8'))
    if (result['status'] != 'difference_dependency_diagnostic_complete'
            or sha(run/'paired-clips.csv') != result['paired_predictions_sha256']):
        raise ValueError('Completed immutable diagnostic required')
    with (run/'paired-clips.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = [{k: (int(v) if k == 'label' else float(v) if k not in ('clip_id', 'unit_id') else v)
                 for k, v in r.items()} for r in csv.DictReader(f)]
    out = run/'analysis-v1'
    out.mkdir(exist_ok=False)
    fig_dir = ROOT/'docs/process/experiments/figures/EXP-016-D-dependency'
    fig_dir.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    import numpy as np
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'pdf.fonttype': 42,
                         'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for label, name, color in [(0, '负片段（40）', '#2f64a2'), (1, '正片段（96）', '#cc7c32')]:
        selected = [r for r in rows if r['label'] == label]
        axes[0].scatter([r['zero_probability'] for r in selected], [r['normal_probability'] for r in selected],
                        s=24, alpha=.7, color=color, label=name)
        axes[1].scatter(label+np.linspace(-.12, .12, len(selected)),
                        [100*r['delta_probability'] for r in selected], s=20, alpha=.65, color=color)
        mean = 100*result['by_class'][str(label)]['delta_probability']['mean']
        axes[1].plot([label-.2, label+.2], [mean, mean], color=color, linewidth=2)
        axes[1].annotate(f'均值 {mean:.4f}', (label+.2, mean), xytext=(3, 5), textcoords='offset points', fontsize=9)
    axes[0].plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='分数相同')
    axes[0].set(title='同一融合权重：136个配对分数', xlabel='D置零时的片段概率（0–1）',
                ylabel='正常D时的片段概率（0–1）', xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, linestyle='--', color='gray', linewidth=.8)
    axes[1].set(title='D引起的概率变化：正常减置零', xlabel='固定片段真值（横向点位仅为展开）',
                ylabel='概率差（百分点）', xticks=[0, 1], xticklabels=['负片段', '正片段'], xlim=(-.4, 1.65))
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle('seed8融合best9：D置零后136个固定片段的0.5判定全部保持一致')
    fig.text(.5, -.035, '仅衡量当前权重的输入依赖；非另训RGB对照、非完整视频评价，也不证明运动理解或训练期D无贡献。', ha='center', fontsize=9)
    fig.tight_layout()
    outputs = []
    for ext in ('png', 'pdf', 'svg'):
        path = fig_dir/('difference-dependency.'+ext)
        fig.savefig(path, dpi=180, bbox_inches='tight')
        outputs.append(path)
    plt.close(fig)
    (out/'generator-source.py').write_bytes(SOURCE.read_bytes())
    (out/'figure-index.json').write_text(json.dumps(dict(source_result_sha256=sha(run/'result.json'),
        source_predictions_sha256=sha(run/'paired-clips.csv'), generator_sha256=sha(SOURCE),
        outputs_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in outputs}), indent=2), encoding='utf-8')
    print(json.dumps(dict(figures=len(outputs), analysis=str(out))))


if __name__ == '__main__':
    main()
