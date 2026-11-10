"""Describe saved negative-video alarm runs at the existing 0.5 threshold.

No decoding, inference, threshold selection or change to published run artifacts.
An episode contains adjacent accepted grid windows, not independent detections.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alarm_runs(rows, mode):
    result, current = [], []
    for row in rows:
        if row[mode] >= 0.5:
            if current and not math.isclose(row['target']-current[-1]['target'], 1.5,
                                           rel_tol=0, abs_tol=1e-8):
                result.append(current)
                current = []
            current.append(row)
        elif current:
            result.append(current)
            current = []
    if current:
        result.append(current)
    return [dict(windows=len(run), start=run[0]['start'], end=run[-1]['end'],
                 source_span_seconds=run[-1]['end']-run[0]['start']) for run in result]


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, default=root/'data/results/exp019/order-s7-20261101-v1')
    ap.add_argument('--plan', type=Path, default=root/'data/metadata/EXP-019/order-plan-v1')
    ap.add_argument('--out', type=Path, default=root/'data/results/exp019/negative-alarm-analysis-s7-v1')
    args = ap.parse_args()
    score_path, plan_path, sources_path = (args.run/'two-dense/window-scores.csv',
                                         args.plan/'dense-windows.jsonl', args.plan/'sources.csv')
    with sources_path.open(encoding='utf-8-sig', newline='') as handle:
        sources = {r['unit_id']: r for r in csv.DictReader(handle)}
    physical = {}
    for line in plan_path.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        key = r['order_record']['physical_window_sha256']
        if key in physical:
            raise ValueError('Duplicate physical window in plan')
        physical[key] = r
    negative, seen, all_units = defaultdict(list), set(), set()
    modes = ('chronological', 'fixed_permutation')
    with score_path.open(encoding='utf-8-sig', newline='') as handle:
        for raw in csv.DictReader(handle):
            key, unit = raw['physical_window_sha256'], raw['unit_id']
            if key in seen:
                raise ValueError('Duplicate scored physical window')
            seen.add(key)
            original = physical[key]
            if unit != original['unit_id'] or sources[unit]['video_label'] not in ('pos','neg'):
                raise ValueError('Window/video/label identity mismatch')
            row = dict(start=float(raw['start']), end=float(raw['end']),
                       target=float(original['target_start_seconds']))
            for mode in modes:
                row[mode] = float(raw[mode])
                if not math.isfinite(row[mode]) or not 0 <= row[mode] <= 1:
                    raise ValueError('Invalid saved probability')
            if (not math.isclose(row['start'], original['source_pts_seconds'][0], abs_tol=1e-9)
                    or not math.isclose(row['end'], original['source_pts_seconds'][-1], abs_tol=1e-9)):
                raise ValueError('Scored window PTS differ from physical input plan')
            all_units.add(unit)
            if sources[unit]['video_label'] == 'neg':
                negative[unit].append(row)
    if seen != set(physical):
        raise ValueError('Scored/physical coverage mismatch')
    summaries = []
    for unit, rows in sorted(negative.items()):
        rows.sort(key=lambda r:r['target'])
        for mode in modes:
            episodes = alarm_runs(rows, mode)
            longest = max(episodes, key=lambda r:r['windows'], default=None)
            summaries.append(dict(unit_id=unit, mode=mode, accepted_windows=len(rows),
                above_threshold_windows=sum(r[mode] >= 0.5 for r in rows),
                max_probability=max(r[mode] for r in rows), episodes=len(episodes),
                longest_windows=longest['windows'] if longest else 0,
                longest_span_seconds=longest['source_span_seconds'] if longest else 0,
                longest_start=longest['start'] if longest else None,
                longest_end=longest['end'] if longest else None,
                episode_intervals=episodes))
    args.out.mkdir(parents=True, exist_ok=False)
    source_copy = args.out/'generator-source.py'
    source_copy.write_bytes(Path(__file__).read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42
    false_positives = [r['unit_id'] for r in summaries
                      if r['mode'] == 'chronological' and r['above_threshold_windows']]
    if not false_positives:
        raise ValueError('No chronological false positives to plot')
    fig, axes = plt.subplots(len(false_positives), 1,
                             figsize=(13, 3.5*len(false_positives)), squeeze=False)
    for ax, unit in zip(axes[:,0], false_positives):
        rows = negative[unit]
        for mode, label, color in [('chronological','A 真实顺序','#28669e'),
                                   ('fixed_permutation','B 固定置换','#c07825')]:
            xx, yy, previous = [], [], None
            for r in rows:
                if previous is not None and not math.isclose(r['target']-previous, 1.5,
                                                           rel_tol=0, abs_tol=1e-8):
                    xx.append(float('nan')); yy.append(float('nan'))
                xx.append((r['start']+r['end'])/120); yy.append(r[mode]); previous=r['target']
            ax.plot(xx, yy, color=color, linewidth=1, alpha=.85, label=label)
        entry = next(r for r in summaries if r['unit_id']==unit and r['mode']=='chronological')
        ax.axhline(.5, color='#555555', linestyle='--', linewidth=1, label='既定阈值 0.5')
        ax.axvspan(entry['longest_start']/60, entry['longest_end']/60,
                   color='#28669e', alpha=.12, label='A 最长连续越阈值窗覆盖')
        ax.set_title('{}：A {}/{}窗越阈值；最长{}窗，覆盖{:.0f}秒'.format(
            unit, entry['above_threshold_windows'],entry['accepted_windows'],
            entry['longest_windows'],entry['longest_span_seconds']))
        ax.set_ylim(-.02,1.02); ax.set_ylabel('模型泥石流分数')
        ax.set_xlabel('处理视频时间轴（分钟）'); ax.grid(alpha=.2)
        ax.legend(loc='upper right', fontsize=9, ncol=2)
    fig.suptitle('EXP-019：普通RGB基线的两条误报含持续高分区间', fontsize=15)
    fig.tight_layout(rect=(0,.035,1,.95))
    fig.text(.02,.01,'全部7条负视频均统计；只绘A的2条误报。缺失网格不连线；重叠窗不是独立样本。', fontsize=10)
    for suffix in ('png','pdf'):
        fig.savefig(args.out/('negative-score-trajectories.'+suffix), dpi=160)
    plt.close(fig)
    result = dict(scope='descriptive development analysis; no new model, threshold or performance claim',
                  threshold=.5, grid_stride_seconds=1.5, scored_videos=len(all_units),
                  scored_windows=len(seen), negative_videos=len(negative), summaries=summaries,
                  episode_definition='adjacent accepted target-grid windows with score >=0.5; gaps break runs',
                  span_definition='last window last source PTS minus first window first source PTS; overlapping input coverage, not independently measured alarm duration',
                  input_sha256={str(p.relative_to(root)):sha(p) for p in
                                (score_path,plan_path,sources_path)},
                  generator_sha256=sha(source_copy),
                  figure_sha256={p.name:sha(p) for p in args.out.glob('*.p*')})
    (args.out/'analysis.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    with (args.out/'summary.csv').open('x',encoding='utf-8-sig',newline='') as handle:
        keys=[k for k in summaries[0] if k!='episode_intervals']
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in summaries)
    print(json.dumps([{k:r[k] for k in ('unit_id','mode','above_threshold_windows',
                                      'longest_windows','longest_span_seconds')} for r in summaries]))


if __name__ == '__main__':
    main()
