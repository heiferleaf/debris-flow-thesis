"""Recompute current 150-video P/N folds and render an exportable overview."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import numpy as np

    current = ROOT / 'data/metadata/EXP-016/metadata-v4/video-inputs.csv'
    parent = ROOT / 'data/process/EXP-008-units-v2/unit-manifest.csv'
    split = ROOT / 'data/metadata/splits/EXP-008-units-folds.json'
    with current.open(encoding='utf-8-sig', newline='') as f:
        videos = list(csv.DictReader(f))
    with parent.open(encoding='utf-8-sig', newline='') as f:
        original = {r['unit_id']: r for r in csv.DictReader(f)}
    definition = json.loads(split.read_text(encoding='utf-8'))
    group_to_fold = {}
    for entry in definition['folds']:
        for group in entry['val_groups']:
            if group in group_to_fold:
                raise ValueError('Group assigned to multiple validation folds')
            group_to_fold[group] = entry['fold']
    if len({v['unit_id'] for v in videos}) != 150:
        raise ValueError('Expected 150 unique current videos')
    for row in videos:
        previous = original[row['unit_id']]
        if (row['video_label'] != previous['label'] or row['group_id'] != previous['group_id']
                or group_to_fold[row['group_id']] != int(previous['fold'])):
            raise ValueError('Current/parent labels or folds disagree')

    def count(rows):
        labels = Counter(r['video_label'] for r in rows)
        return dict(n=len(rows), P=labels['pos'], N=labels['neg'],
                    positive_percent=100 * labels['pos'] / len(rows),
                    groups=len({r['group_id'] for r in rows}),
                    negative_groups=len({r['group_id'] for r in rows if r['video_label'] == 'neg'}))
    folds = []
    for fold in range(1, 6):
        val = [r for r in videos if group_to_fold[r['group_id']] == fold]
        train = [r for r in videos if group_to_fold[r['group_id']] != fold]
        folds.append(dict(fold=fold, train=count(train), validation=count(val), current_screen=fold == 2))
    data = dict(total=count(videos), folds=folds, current_screen_fold=2,
                current_clip_counts=dict(train_P=356, train_N=148, validation_P=96, validation_N=40),
                scope='existing frozen folds for development videos; EXP-016 currently runs fold2 screening only',
                external_test_included=False,
                source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (current, parent, split)},
                note='Counts recomputed from current video rows, not historical JSON prose; negative-containing groups=9, not old meta 10.',
                command=sys.argv)
    output = ROOT / 'data/metadata/EXP-016/fold-overview-v1'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'distribution.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / 'distribution.csv').open('x', encoding='utf-8-sig', newline='') as f:
        fields = ['fold', 'side', 'videos', 'P', 'N', 'positive_percent', 'groups', 'negative_groups']
        writer = csv.DictWriter(f, fieldnames=fields);writer.writeheader()
        for row in folds:
            for side in ('train', 'validation'):
                stats = row[side]
                writer.writerow(dict(fold=row['fold'], side=side, videos=stats['n'], P=stats['P'], N=stats['N'],
                                     positive_percent=stats['positive_percent'], groups=stats['groups'],
                                     negative_groups=stats['negative_groups']))
    figures = ROOT / '.codex/docs/visualizations/fold-overview-v1'
    figures.mkdir(parents=True, exist_ok=False)
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    blue, orange, ink, muted = '#2C66A8', '#D17B35', '#202A35', '#647080'
    fig = plt.figure(figsize=(13, 10), facecolor='white')
    top = fig.add_axes([.15, .81, .77, .065])
    ax = fig.add_axes([.15, .14, .77, .55])
    fig.text(.05, .95, '开发视频的正负分布与冻结五折划分', fontproperties=font, fontsize=21, color=ink, weight='bold')
    fig.text(.05, .913, '本轮只运行第2折作方案初筛；正式五折尚未开始，外部测试候选不在图中。', fontproperties=font, fontsize=12, color=muted)
    top.barh([0], [113], color=blue, height=.6, label='P：视频出现泥石流')
    top.barh([0], [37], left=[113], color=orange, height=.6, label='N：视频未出现泥石流')
    top.text(56.5, 0, 'P 113条 · 75.3%', ha='center', va='center', color='white', fontsize=12, fontproperties=font)
    top.text(131.5, 0, 'N 37条 · 24.7%', ha='center', va='center', color='white', fontsize=11, fontproperties=font)
    top.set_xlim(0, 150);top.set_yticks([0]);top.set_yticklabels(['总体 150条'], fontproperties=font)
    top.set_xticks([]);top.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    fig.text(.15, .753, '每折训练侧与验证侧（按完整视频计数）', fontproperties=font, fontsize=14, color=ink, weight='bold')
    ax.axhspan(2 - .48, 3 + .48, color='#EFF5FB', zorder=0)
    positions, labels, values = [], [], []
    for row in folds:
        for side, chinese in [('train', '训练'), ('validation', '验证')]:
            positions.append(len(positions))
            labels.append(f"第{row['fold']}折 · {chinese}" + (' ← 本轮' if row['fold'] == 2 else ''))
            values.append(row[side])
    for y, stats in zip(positions, values):
        p, n = stats['P'], stats['N']
        ax.barh(y, p, color=blue, height=.57)
        ax.barh(y, n, left=p, color=orange, height=.57)
        ax.text(p / 2, y, str(p), ha='center', va='center', color='white', fontsize=11)
        ax.text(p + n / 2, y, str(n), ha='center', va='center', color='white', fontsize=11)
        ax.text(p+n+2, y, f"共{p+n}条", va='center', fontsize=10, color=muted, fontproperties=font)
    ax.set_yticks(positions);ax.set_yticklabels(labels, fontproperties=font, fontsize=10)
    ax.invert_yaxis();ax.set_xlim(0, 150);ax.set_xticks(np.arange(0,151,30))
    ax.set_xlabel('视频数量（条）；蓝色 P，橙色 N', fontproperties=font, fontsize=11)
    ax.set_ylabel('折号与用途', fontproperties=font, fontsize=11)
    ax.spines[['top','right','left']].set_visible(False)
    ax.grid(axis='x', color='#E7EBEF', linewidth=.7);ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
    fig.text(.15, .067, '本轮模型输入另计：训练504窗（pos 356 / neg 148）；固定验证136窗（pos 96 / neg 40）。', fontproperties=font, fontsize=10, color=ink)
    fig.text(.15, .041, '同一机位、同一天的视频不跨折；图示比例不要求完全相同，以保持组隔离。第2折验证含3个负视频场次组。', fontproperties=font, fontsize=9, color=muted)
    fig.text(.15, .017, '来源：当前150视频清单 + 冻结group_id折映射；2026-10-30复算。CSV/JSON含输入SHA与生成命令。', fontproperties=font, fontsize=9, color=muted)
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(figures / f'fold-distribution.{suffix}', dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(dict(metadata=str(output), figures=str(figures), data=data), ensure_ascii=False))


if __name__ == '__main__':
    main()
