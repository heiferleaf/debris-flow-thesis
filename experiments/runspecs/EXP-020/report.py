"""Render a complete EXP020 comparison into a Chinese brief, CSV and one figure.

No model execution, media access, old-input re-audit or automatic training. Run
only after collecting the complete child run; output directories never overwrite.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = ROOT / 'data/results/exp020/head-s7s8-20261103-v2'
DEFAULT_PLAN = ROOT / 'data/metadata/EXP-020/head-plan-v2'
EXPECTED = {(fold, head, seed) for fold in (1, 4) for head in ('deep', 'linear') for seed in (7, 8)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-12


def validate(comparison, plan):
    """Only new report risks: identity, completeness, arithmetic and fixed gates."""
    require(comparison['status'] == 'all_eight_items_complete_pending_independent_audit', 'Incomplete comparison')
    require(plan['spec_id'] == 'EXP-020' and plan['epochs'] == 20, 'Wrong comparison plan')
    planned = {(r['fold'], r['head'], r['seed']): r for r in plan['items']}
    rows = comparison['items']
    actual = {(r['fold'], r['head'], r['seed']): r for r in rows}
    require(len(plan['items']) == len(planned) == len(rows) == len(actual) == 8 and set(actual) == set(planned) == EXPECTED,
            'Eight unique registered items required')
    ordered = []
    for key in sorted(EXPECTED):
        r, p = actual[key], planned[key]
        fold, head, seed = key
        action = 'reuse' if head == 'deep' and seed == 7 else 'fit'
        require(r['item'] == p['item'] and r['action'] == p['action'] == action, 'Incorrect reuse/fit identity')
        if action == 'reuse':
            require(r['artifact_directory'] == p['artifact_directory'] and '/inner' in p['artifact_directory'],
                    'Reused reference must retain original inner directory')
        require(type(r['selected_epochs']) is int and 1 <= r['selected_epochs'] <= 20, 'Invalid selected E')
        require(r['score_kind'] == 'logit_AUC_and_stable_FP64_sigmoid_for_fixed_0.5_recalls', 'Unsupported score semantics')
        for name in ('AUC_logit', 'positive_Recall', 'negative_Recall', 'probability_min', 'probability_max', 'probability_sd',
                     'derived_FP32_probability_AUC', 'derived_FP64_probability_AUC'):
            require(type(r[name]) in (int, float) and math.isfinite(r[name]) and 0 <= r[name] <= 1, 'Invalid finite metric: ' + name)
        require(r['probability_min'] <= r['probability_max'], 'Reversed probability range')
        require(all(type(r[k]) is int and r[k] >= 0 for k in ('TP', 'FP', 'FN', 'TN')), 'Invalid confusion counts')
        p_count, n_count = r['TP'] + r['FN'], r['TN'] + r['FP']
        require((p_count, n_count) == ((72, 28) if fold == 1 else (72, 32)), 'Fixed inner validation class counts changed')
        require(close(r['positive_Recall'], r['TP'] / p_count) and close(r['negative_Recall'], r['TN'] / n_count),
                'Recall/count arithmetic differs')
        passed = r['AUC_logit'] >= .75 and r['positive_Recall'] >= .5 and r['negative_Recall'] >= .5
        require(type(r['gate_pass']) is bool and r['gate_pass'] == passed, 'Per-item gate differs')
        ordered.append(r)
    pairs = comparison['pairs']
    indexed = {(r['fold'], r['seed']): r for r in pairs}
    require(len(pairs) == len(indexed) == 4 and set(indexed) == {(f, s) for f in (1, 4) for s in (7, 8)},
            'Four unique pairs required')
    for (fold, seed), pair in indexed.items():
        deep, linear = actual[(fold, 'deep', seed)], actual[(fold, 'linear', seed)]
        require(pair['deep_item'] == deep['item'] and pair['linear_item'] == linear['item'], 'Pair identity differs')
        for name in ('AUC_logit', 'positive_Recall', 'negative_Recall'):
            require(close(pair['delta_' + name], linear[name] - deep[name]), 'Paired difference arithmetic differs')
    counts = {h: sum(r['gate_pass'] for r in rows if r['head'] == h) for h in ('deep', 'linear')}
    require(comparison['passed_by_head'] == counts, 'Head pass counts differ')
    require(type(comparison['linear_all_four_pass']) is bool and comparison['linear_all_four_pass'] == (counts['linear'] == 4),
            'All-four linear decision differs')
    return ordered, [indexed[k] for k in sorted(indexed)], counts


def csv_file(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    pairs = [(1, 7), (1, 8), (4, 7), (4, 8)]
    lookup = {(r['fold'], r['head'], r['seed']): r for r in rows}
    panels = [('AUC_logit', 'Logit AUC', .75), ('positive_Recall', 'Positive Recall', .5), ('negative_Recall', 'Negative Recall', .5)]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.55), sharey=True)
    fig.subplots_adjust(left=.065, right=.99, bottom=.24, top=.79, wspace=.25)
    for ax, (name, title, threshold) in zip(axes, panels):
        for x, (fold, seed) in enumerate(pairs, 1):
            a, b = lookup[(fold, 'deep', seed)], lookup[(fold, 'linear', seed)]
            ax.plot([x - .10, x + .10], [a[name], b[name]], c='#888888', lw=1, zorder=1)
            ax.scatter(x - .10, a[name], c='#D95319', marker='D', s=32, zorder=3)
            ax.scatter(x + .10, b[name], c='#2878B5', marker='o', s=34, zorder=3)
            if a['action'] == 'reuse':
                ax.scatter(x - .10, a[name], c='black', marker='x', s=18, linewidths=.8, zorder=4)
        ax.axhline(threshold, c='#555555', ls='--', lw=1)
        ax.text(.98, threshold + .025, 'Gate %.2f' % threshold, transform=ax.get_yaxis_transform(), ha='right', fontsize=8)
        ax.set_xticks(np.arange(1, 5), ['F1 / s7', 'F1 / s8', 'F4 / s7', 'F4 / s8'], rotation=18)
        ax.set_ylim(-.025, 1.055)
        ax.set_yticks(np.arange(0, 1.01, .25))
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Selected inner validation metric')
    handles = [Line2D([], [], c='#D95319', marker='D', ls='', label='MLP head'),
               Line2D([], [], c='#2878B5', marker='o', ls='', label='Linear head'),
               Line2D([], [], c='black', marker='x', ls='', label='Reused MLP / seed7')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55, .94), ncol=3, frameon=False, fontsize=8)
    fig.text(.065, .985, 'EXP-020: paired overall-head development trial', va='top', weight='bold', fontsize=11)
    fig.text(.065, .055, 'E selected by minimum validation BCE; fixed probability threshold = 0.5.\nKnown inner splits only; each of the four linear items must pass all three gates.', fontsize=8)
    fig.savefig(str(path) + '.png', dpi=240, metadata={'Software': 'Matplotlib', 'Title': 'EXP-020 paired head development metrics'})
    fig.savefig(str(path) + '.pdf', metadata={'Creator': 'Matplotlib', 'Author': '', 'Title': 'EXP-020 paired head development metrics'})
    plt.close(fig)


def brief(rows, pairs, counts, run, plan_path, recovered):
    lines = ['# EXP-020 分类头配对开发小试简报', '',
        '使用了 data-usage-protocol skill。第6条沿用共享配额例外，第10条按已看过两划分的开发小试解释；本简报是派生交付，仍待独立结果复核。', '',
        '本轮比较整体 MLP 头与 Linear(512,1)，固定 fold1／4、模型种子7／8。8项中，MLP seed7两项引用原 EXP-017 inner；其余6项各跑满20轮。选轮始终按固定 inner 验证未加权 BCE 最低，完全相同取最早；AUC统一使用保存 logit 计算。', '',
        '四个线性项有%d/4同时达到 logit AUC≥0.75、正类Recall≥0.50、负类Recall≥0.50；MLP对应通过数为%d/4。%s' %
        (counts['linear'], counts['deep'], '按预定门槛，线性头仅获得下一项研究设计资格。' if counts['linear'] == 4 else '按预定门槛，线性头不晋级；不能用均值掩盖未通过项。'), '',
        '## 8项完整结果', '',
        '| 划分 | 头 | seed | 来源 | E | logit AUC | 正类Recall | 负类Recall | TP/FP/FN/TN | 门槛 |',
        '|---|---|---:|---|---:|---:|---:|---:|---|---|']
    for r in rows:
        lines.append('| F%d | %s | %d | %s | %d | %.4f | %.4f | %.4f | %d/%d/%d/%d | %s |' %
            (r['fold'], 'Linear' if r['head'] == 'linear' else 'MLP', r['seed'], '旧inner复用' if r['action'] == 'reuse' else '本轮新fit',
             r['selected_epochs'], r['AUC_logit'], r['positive_Recall'], r['negative_Recall'], r['TP'], r['FP'], r['FN'], r['TN'], '通过' if r['gate_pass'] else '未过'))
    lines += ['', '## 4对变化', '', '差值均为 Linear−MLP，保留改善和退步，不跨不同模型计算 pooled AUC。', '',
        '| 划分 | seed | Δlogit AUC | Δ正类Recall | Δ负类Recall |', '|---|---:|---:|---:|---:|']
    for r in pairs:
        lines.append('| F%d | %d | %+.4f | %+.4f | %+.4f |' % (r['fold'], r['seed'], r['delta_AUC_logit'], r['delta_positive_Recall'], r['delta_negative_Recall']))
    lines += ['', '![整体分类头配对结果](paired-head-gates.png)', '',
        '图：每个位置是一组相同划分／种子的配对；连线仅连接该配对的两种头。虚线为预定门槛，黑色×标出旧MLP seed7复用项。图中均为选中inner轮的片段验证指标；不是整视频、outer或外部测试。', '',
        '## 证据与边界', '',
        '- [8项全精度CSV](items.csv)、[4对全精度CSV](pairs.csv)、[来源及生成记录](report.json)、[矢量图](paired-head-gates.pdf)。表中四位小数只用于显示，门槛判断使用原始全精度。',
        '- MLP seed7保留原run及inner路径，详见items.csv；未复制为新训练项。FP32／FP64概率AUC另列为派生指标，不回写历史文件，不替代logit AUC。',
        '- 比较整体头同时改变容量、非线性和Dropout，不能把差异归因于其中单项，也不能据此证明其为EXP-017失败根因。',
        '- 两个划分已看过，种子也仅7／8；不作独立确认、未见机位或跨域泛化主张。不对相关片段做独立样本显著性检验。',
        '- 即使四项全过，也不等于整视频误报改善或时序贡献；本轮没有outer fresh、dense或外部评分。脚本不会追加训练、搜索阈值或更换种子。', '',
        '结果归档：`%s`。计划来源：`%s`。' % (run, plan_path), '']
    if recovered:
        lines += ['原run的比较聚合失败回执保持原样；本简报读取另立文件中的离线派生比较，按原日志恢复，不重训、不替换原run，也不把收尾异常隐去。其来源与原failure SHA均见report.json；仍待独立结果复核。', '']
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    p.add_argument('--comparison', type=Path, help='Optional separate offline-derived comparison; never replaces the original run')
    p.add_argument('--out', type=Path, default=ROOT / 'data/results/exp020/report-head-s7s8-20261103-v2-v1')
    a = p.parse_args(argv)
    run, plan_dir, out = a.run.resolve(), a.plan.resolve(), a.out.resolve()
    require(not out.exists(), 'Output already exists; choose a new report directory')
    require(out != run and run not in out.parents, 'Report must be outside the immutable original run')
    require(out != plan_dir and plan_dir not in out.parents, 'Report must be outside the frozen plan directory')
    comparison_path = (a.comparison if a.comparison is not None else run / 'comparison.json').resolve()
    recovered = comparison_path != (run / 'comparison.json').resolve()
    paths = [comparison_path, run / 'execution-inputs.json', plan_dir / 'plan.json', plan_dir / 'schedules.json']
    require(all(path.is_file() for path in paths), 'Complete collected comparison/run/plan required; no report generated')
    require(not (run / 'hard-time-limit.json').exists(), 'Actual fee-deadline failure cannot receive complete report')
    inputs = {str(path): sha(path) for path in paths}
    comparison, execution, plan = [read(path) for path in paths[:3]]
    if recovered:
        derivation = comparison['derivation']
        failure = run / 'pipeline-failure.json'
        require(derivation['kind'] == 'offline_recovery_comparison_only' and derivation['source_sha256'], 'Missing separate recovery provenance')
        require((ROOT / derivation['original_run_path']).resolve() == run and failure.is_file()
                and sha(failure) == derivation['original_failure_sha256'], 'Original failure/run provenance differs')
        require(derivation['plan_sha256'] == inputs[str(plan_dir / 'plan.json')], 'Recovery plan differs')
        inputs[str(failure)] = sha(failure)
        for name, expected in derivation['source_sha256'].items():
            source = (ROOT / name).resolve()
            source.relative_to(ROOT)
            require(source.is_file() and sha(source) == expected, 'Recovery metadata/log SHA differs: ' + name)
            inputs[str(source)] = expected
        for item in plan['items']:
            if item['action'] != 'fit':
                continue
            path = run / item['item'] / 'training-result.json'
            result = read(path)
            require(result['status'] == 'fit_complete' and result['completed_epochs'] == 20
                    and result['stage'] == 'inner_select' and result['fold'] == item['fold']
                    and result['optimizer_steps'] == (340 if item['fold'] == 1 else 360),
                    'Offline comparison cannot conceal an incomplete new fit')
            inputs[str(path)] = sha(path)
    else:
        path = run / 'pipeline-result.json'
        require(path.is_file() and not (run / 'pipeline-failure.json').exists(), 'Incomplete or failed original pipeline')
        result = read(path)
        require(result['status'] == 'six_new_fits_complete_pending_independent_audit', 'Incomplete pipeline')
        require(result['comparison_sha256'] == inputs[str(comparison_path)], 'Comparison/result SHA differs')
        inputs[str(path)] = sha(path)
    require(execution['plan_sha256'] == inputs[str(plan_dir / 'plan.json')], 'Execution/plan SHA differs')
    require(plan['schedules_sha256'] == inputs[str(plan_dir / 'schedules.json')], 'Schedule/plan SHA differs')
    rows, pairs, counts = validate(comparison, plan)
    fields = ('item', 'fold', 'head', 'seed', 'action', 'selected_epochs', 'AUC_logit', 'positive_Recall', 'negative_Recall',
              'TP', 'FP', 'FN', 'TN', 'gate_pass', 'probability_min', 'probability_max', 'probability_sd',
              'derived_FP32_probability_AUC', 'derived_FP64_probability_AUC', 'logit_minus_derived_FP32_probability_AUC',
              'actual_artifact_directory')
    table = [dict({k: r[k] for k in fields}, original_reused_inner_directory=r.get('artifact_directory', '')) for r in rows]
    out.mkdir(parents=True, exist_ok=False)
    csv_file(out / 'items.csv', table)
    csv_file(out / 'pairs.csv', pairs)
    figure(rows, out / 'paired-head-gates')
    (out / 'brief.md').write_text(brief(rows, pairs, counts, run, plan_dir, recovered), encoding='utf-8')
    require(all(sha(Path(path)) == value for path, value in inputs.items()), 'Report inputs changed while reading')
    metadata = dict(status='derived_report_complete_pending_independent_result_review', input_sha256=inputs,
        report_source_path=str(Path(__file__).resolve()), report_source_sha256=sha(Path(__file__).resolve()),
        argv=sys.argv if argv is None else argv, environment=dict(python=sys.version, platform=platform.platform()),
        score_kind='logit_AUC', fixed_probability_threshold=.5,
        gate=dict(AUC_logit_min=.75, positive_Recall_min=.5, negative_Recall_min=.5, all_four_linear_required=True),
        passed_by_head=counts, linear_all_four_pass=counts['linear'] == 4, new_fits=6, reused_inner_fits=2,
        comparison_source=str(comparison_path), separate_offline_derivation=recovered,
        limitations=['Known developmental inner splits only', 'No outer/dense/external evaluation',
                     'No pooled cross-model AUC or independent-window significance claim', 'No model/media execution or automatic training',
                     'Printed rounding does not affect gate decisions', 'Figure still requires actual visual QA after generation'],
        outputs_sha256={path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()})
    with (out / 'report.json').open('x', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(status=metadata['status'], output=str(out), passed_by_head=counts), ensure_ascii=False))


if __name__ == '__main__':
    main()
