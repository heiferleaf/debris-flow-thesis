"""Chinese EXP022 figures from the reviewed derived metrics/CSV, without models.

Real mode requires --input <report directory> and an explicit new --out directory.
--demo renders visibly simulated layout fixtures only under .codex/docs/.temp.
The plotting tool never selects an epoch or recomputes the scientific gate.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEEP = '#4C6E91'
LINEAR = '#2B8C7F'
INK = '#243342'
MUTED = '#566575'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def configure_style():
    available = {f.name for f in font_manager.fontManager.ttflist}
    candidates = ['Microsoft YaHei', 'Noto Sans CJK SC', 'Source Han Sans SC', 'SimHei', 'SimSun']
    selected = next((name for name in candidates if name in available), None)
    require(selected is not None, 'A local Chinese font is required; no font download will be attempted')
    plt.rcParams.update({'font.family': selected, 'font.size': 10, 'axes.unicode_minus': False,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': INK,
        'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.titleweight': 'bold', 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'svg.fonttype': 'path', 'savefig.facecolor': 'white'})
    return selected


def load_report(directory):
    """Read only the accepted derivation, and check its recorded CSV hashes."""
    path = directory / 'metrics.json'
    with path.open(encoding='utf-8') as stream:
        metrics = json.load(stream)
    require(metrics['status'] == 'full' and metrics['spec_id'] == 'EXP-022'
            and metrics['independent_test'] is False, 'Only a complete EXP022 development derivation can be plotted')
    require(metrics['dataset'] == dict(videos=150, P=113, N=37, physical_windows=143470, negative_session_groups=9),
            'Unexpected full-report coverage')
    inputs = {str(path.resolve()): sha(path)}
    def rows(name):
        item = directory / name
        require(item.is_file() and name in metrics['outputs_sha256'], 'Required report CSV is unavailable: ' + name)
        expected = metrics['outputs_sha256'][name]
        require(sha(item) == expected, 'Reviewed CSV SHA differs: ' + name)
        inputs[str(item.resolve())] = expected
        with item.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    groups = rows('negative-groups.csv')
    require(len(groups) == 9 and len({r['group_id'] for r in groups}) == 9, 'Nine unique negative groups required')
    typed_groups = []
    for row in groups:
        typed = dict(group_id=row['group_id'], N=int(row['N']), deep_FP=int(row['deep_FP']), linear_FP=int(row['linear_FP']),
                     deep_FPR=float(row['deep_FPR']), linear_FPR=float(row['linear_FPR']), delta_FPR=float(row['delta_FPR']))
        require(typed in metrics['negative_groups'] and typed['N'] > 0
                and all(0 <= typed[arm + '_FP'] <= typed['N'] and 0 <= typed[arm + '_FPR'] <= 1 for arm in ('deep', 'linear')),
                'Negative-group CSV differs from reviewed metrics')
        typed_groups.append(typed)
    pairs = rows('video-pairs.csv')
    require(len(pairs) == len({r['unit_id'] for r in pairs}) == 150, 'Full paired-video CSV required')
    require(Counter(int(r['label']) for r in pairs) == Counter({1:113, 0:37}), 'Paired video class coverage differs')
    # Check the already-recorded decisions, without applying any new threshold.
    for arm in ('deep', 'linear'):
        cells = Counter(('TP' if int(r['label']) else 'FP') if int(r[arm + '_prediction']) else
                        ('FN' if int(r['label']) else 'TN') for r in pairs)
        require(all(cells[key] == metrics[arm][key] for key in ('TP', 'FP', 'FN', 'TN')), 'Paired decisions differ from reviewed totals')
    curves = rows('learning-curves.csv')
    return metrics, typed_groups, pairs, curves, inputs


def inner_curves(metrics, rows):
    result = {}
    for fold in metrics['folds']:
        f = fold['fold']
        selected = sorted((r for r in rows if r['stage'] == 'inner_select' and int(r['fold']) == f),
                          key=lambda r: int(r['completed_epochs']))
        require([int(r['completed_epochs']) for r in selected] == list(range(1, 21)), 'Each inner curve needs its completed 20 epochs')
        values = [float(r['validation_unweighted_bce']) for r in selected]
        require(all(math.isfinite(value) and value >= 0 for value in values), 'Invalid inner unweighted validation BCE')
        E = fold['linear_selected_E']
        require(type(E) is int and 1 <= E <= 20, 'Invalid recorded selected E')
        require(fold['inner_reused'] == (f in (1, 4)), 'Reused-inner labels differ from the EXP022 mapping')
        result[f] = dict(values=values, E=E, reused=fold['inner_reused'])
    require(sorted(result) == list(range(1, 6)), 'Five registered inner curves required')
    return result


def decorate(fig, title, subtitle, footnote, simulated=False):
    prefix = '模拟数据 · ' if simulated else ''
    fig.text(.045, .965, prefix + title, fontsize=15, fontweight='bold', va='top')
    fig.text(.045, .915, subtitle, fontsize=10, color=MUTED, va='top')
    if simulated:
        footnote = '仅用于版式验证，不是 EXP-022 实测。\n' + footnote
    fig.text(.045, .035, footnote, fontsize=9, color=MUTED, va='bottom', linespacing=1.6)


def save(fig, out, name, description, simulated):
    title = ('模拟数据 · ' if simulated else '') + description
    fig.savefig(out / (name + '.png'), dpi=220,
                metadata={'Software': 'Matplotlib', 'Title': title, 'Description': description})
    fig.savefig(out / (name + '.svg'), metadata={'Creator': 'Matplotlib', 'Title': title,
                'Description': description, 'Date': None})
    plt.close(fig)


def group_figure(metrics, groups, out, prefix, simulated):
    fig, ax = plt.subplots(figsize=(11.8, 8.0))
    fig.subplots_adjust(left=.30, right=.965, bottom=.16, top=.83)
    y = np.arange(len(groups)); width = .31
    for arm, offset, color, label in [('deep', -width/2, DEEP, '旧深 MLP 头'), ('linear', width/2, LINEAR, '新 Linear 头')]:
        values = [r[arm + '_FPR'] for r in groups]
        ax.barh(y + offset, values, height=width, color=color, label=label, zorder=3)
        for position, value, row in zip(y + offset, values, groups):
            ax.text(value + .012, position, '%.0f%%（%d/%d）' % (100*value, row[arm + '_FP'], row['N']),
                    va='center', fontsize=8.5, color=INK)
    labels = []
    for row in groups:
        parts = row['group_id'].split('|')
        identity = ' / '.join(parts[:-1]) + '\n' + parts[-1] if len(parts) > 1 else parts[0]
        labels.append(identity + '（N=%d）' % row['N'])
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 1.28)
    ax.set_xticks([0, .25, .5, .75, 1]); ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('负视频中被报正的比例 FPR')
    ax.grid(axis='x', color='#E4E9ED', linewidth=.8, zorder=0)
    ax.legend(loc='lower right', bbox_to_anchor=(1., 1.01), ncol=2, frameon=False, fontsize=10)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0, pad=10)
    foot = ('场次宏 FPR：旧 %.2f%% → 新 %.2f%%；视频微 FPR：旧 %.2f%% → 新 %.2f%%。\n'
            '宏平均让每个负场次等权；微平均按负视频计数。N 为该场次的负视频数，柱旁为 FP/N。') % (
        100*metrics['deep_macro_negative_group_FPR'], 100*metrics['linear_macro_negative_group_FPR'],
        100*metrics['deep']['FPR'], 100*metrics['linear']['FPR'])
    decorate(fig, '九个负场次：新旧分类头的误报率', '同一视频配对；视频分数取窗口最大值，固定阈值 0.5', foot, simulated)
    save(fig, out, prefix + 'negative-session-fpr', '负场次逐组 FPR 与各组负视频数', simulated)


def confusion_figure(metrics, pairs, out, prefix, simulated):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.7))
    fig.subplots_adjust(left=.075, right=.975, bottom=.22, top=.79, wspace=.27)
    for ax, cells, names, total in [(axes[0], ['TP', 'FN'], ['TP：正确报正', 'FN：漏报'], metrics['dataset']['P']),
                                   (axes[1], ['FP', 'TN'], ['FP：误报', 'TN：正确判负'], metrics['dataset']['N'])]:
        x = np.arange(2); width = .30
        for arm, offset, color, label in [('deep', -width/2, DEEP, '旧深 MLP 头'), ('linear', width/2, LINEAR, '新 Linear 头')]:
            heights = [metrics[arm][cell] for cell in cells]
            ax.bar(x + offset, heights, width=width, color=color, label=label, zorder=3)
            for position, value in zip(x + offset, heights):
                ax.text(position, value + max(1, total*.022), str(value), ha='center', fontsize=11, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylim(0, max(total*1.15, 3)); ax.set_ylabel('视频条数')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.grid(axis='y', color='#E4E9ED', linewidth=.8, zorder=0)
        ax.set_title(('正视频' if cells[0] == 'TP' else '负视频') + '（%d 条）' % total, fontsize=11, pad=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.98, .895), ncol=2, frameon=False)
    changes = Counter(row['change'] for row in pairs)
    foot = ('同一视频判决变化：修复 FP %d 条，新增 FP %d 条，修复 FN %d 条，新增 FN %d 条。\n'
            '两图按各自类别总数缩放纵轴；本图汇总判决计数，不合并不同外折模型的 AUC。') % (
        changes['repaired_false_positive'], changes['new_false_positive'],
        changes['repaired_false_negative'], changes['new_false_negative'])
    decorate(fig, '整视频判决：新旧分类头的配对概况',
             '%d 条视频；片段分数经 max 聚合，以 0.5 判正' % metrics['dataset']['videos'], foot, simulated)
    save(fig, out, prefix + 'paired-video-confusion', '同一批视频的 TP、FP、FN、TN 与新增错误', simulated)


def curve_figure(curves, out, prefix, simulated):
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.4))
    fig.subplots_adjust(left=.075, right=.975, bottom=.145, top=.825, wspace=.30, hspace=.42)
    for ax, (fold, row) in zip(axes.flat, sorted(curves.items())):
        epochs = list(range(1, 21)); E = row['E']; values = row['values']
        ax.plot(epochs, values, color=LINEAR, linewidth=1.6)
        ax.axvline(E, color=DEEP, linewidth=.9, linestyle='--', alpha=.75)
        ax.scatter([E], [values[E-1]], marker='*', s=115, color=DEEP, edgecolor='white', linewidth=.6, zorder=5)
        ax.annotate('E=%d' % E, (E, values[E-1]), xytext=(-5 if E >= 16 else 5, 10),
                    textcoords='offset points', ha='right' if E >= 16 else 'left', fontsize=9, color=DEEP)
        ax.set_title('第 %d 折（%s）' % (fold, '复用 EXP-020' if row['reused'] else '本轮新增'), fontsize=10)
        ax.set_xlim(.5, 20.5); ax.set_xticks([1, 5, 10, 15, 20]); ax.set_xlabel('已完成 epoch')
        ax.set_ylabel('未加权验证 BCE'); ax.set_ylim(bottom=0)
        ax.grid(color='#E4E9ED', linewidth=.65)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.03, .94, '怎样读这张图', transform=axes.flat[-1].transAxes, fontsize=12, fontweight='bold', va='top')
    axes.flat[-1].text(.03, .74, '线条：固定内层片段的未加权验证 BCE。\n\n星号：已审汇总记录的所选 E。\n\n第 1、4 折复用 EXP-020 的完整训练。\n\n各折纵轴独立缩放，便于观察波动。',
                        transform=axes.flat[-1].transAxes, fontsize=10, va='top', linespacing=1.5)
    decorate(fig, '内层选轮：五折的未加权验证 BCE', '仅绘制同一个验证目标；星号沿用已记录的 E，不重新选轮',
             '这些曲线来自内层验证，用于解释训练侧选轮；整视频成绩来自另行 fresh 重训的外折模型。', simulated)
    save(fig, out, prefix + 'inner-validation-bce', '五个内层角色的未加权验证 BCE 与已记录选轮', simulated)


def demo_data():
    """Intentionally artificial counts/curves, never usable as experiment data."""
    groups, pairs = [], []
    for i in range(9):
        gid = '模拟|机位%02d|2026-01-%02d' % (i+1, i+1)
        old_fp, new_fp = 2, i % 3
        groups.append(dict(group_id=gid, N=2, deep_FP=old_fp, linear_FP=new_fp,
                           deep_FPR=1., linear_FPR=new_fp/2, delta_FPR=new_fp/2-1))
        for j in range(2):
            pairs.append(dict(change='unchanged' if j < new_fp else 'repaired_false_positive'))
    pairs.extend([dict(change='unchanged') for _ in range(12)])
    metrics = dict(dataset=dict(videos=30, P=12, N=18),
        deep=dict(TP=12, FP=18, FN=0, TN=0, FPR=1.), linear=dict(TP=12, FP=9, FN=0, TN=9, FPR=.5),
        deep_macro_negative_group_FPR=1., linear_macro_negative_group_FPR=.5,
        folds=[dict(fold=i, linear_selected_E=e, inner_reused=i in (1, 4)) for i,e in enumerate([10, 4, 8, 6, 12], 1)])
    rows = []
    for fold in metrics['folds']:
        f, E = fold['fold'], fold['linear_selected_E']
        for e in range(1, 21):
            rows.append(dict(fold=f, stage='inner_select', completed_epochs=e,
                validation_unweighted_bce=.16 + .004*(e-E)**2 + .016*abs(math.sin(e*1.3 + f))))
    return metrics, groups, pairs, rows


def generate(directory, out, simulated=False, argv=None):
    out = Path(out).resolve()
    require(not out.exists(), 'Refuse figure directory overwrite')
    if simulated:
        allowed = ROOT / '.codex/docs/.temp'
        require(allowed.resolve() in out.parents, 'Simulated plots must stay under .codex/docs/.temp')
        metrics, groups, pairs, rows = demo_data(); inputs = {}
    else:
        require(directory is not None, '--input is required for real figures')
        directory = Path(directory).resolve()
        require(out != directory and directory not in out.parents, 'Use a separate derived figure directory')
        metrics, groups, pairs, rows, inputs = load_report(directory)
    curves = inner_curves(metrics, rows)
    font = configure_style()
    out.mkdir(parents=True, exist_ok=False)
    prefix = 'demo-' if simulated else ''
    group_figure(metrics, groups, out, prefix, simulated)
    confusion_figure(metrics, pairs, out, prefix, simulated)
    curve_figure(curves, out, prefix, simulated)
    require(all(sha(Path(path)) == expected for path, expected in inputs.items()), 'Reviewed input changed while plotting')
    manifest = dict(status='simulated_layout_fixture_not_experimental_results' if simulated else 'figures_from_full_reviewed_derivation',
        input_sha256=inputs, source_sha256=sha(Path(__file__)), command=argv or [], chinese_font=font,
        matplotlib=matplotlib.__version__, python=sys.version, platform=platform.platform(),
        no_training_or_model_loading=True, no_gate_or_epoch_selection=True,
        outputs_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    with (out / 'figure-manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Directory containing reviewed metrics.json and its CSV outputs')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--demo', action='store_true', help='Simulated layout fixture, restricted to .codex/docs/.temp')
    args = parser.parse_args(argv)
    require(not (args.demo and args.input), '--demo and --input are mutually exclusive')
    result = generate(args.input, args.out, args.demo, sys.argv if argv is None else argv)
    print(json.dumps(dict(status=result['status'], out=str(args.out.resolve()), outputs=list(result['outputs_sha256'])), ensure_ascii=True))


if __name__ == '__main__':
    main()
