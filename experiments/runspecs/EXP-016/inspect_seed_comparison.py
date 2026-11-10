"""Read-only four-case analysis from audited seed8 scores and existing JPEGs."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT/'data/results/exp016/paired-s8-20261031-v1'
OUT = ROOT/'data/process/EXP-016/seed8-error-analysis-v1'
FIG = ROOT/'docs/process/experiments/figures/EXP-016-seeds'
CASES = {'v1_120': 'neg', 'v1_62': 'neg', 'v5_04': 'neg', 'v1_33': 'pos'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    from PIL import Image, ImageDraw, ImageFont
    audit = json.loads((RUN/'audit-v1/audit.json').read_text(encoding='utf-8'))
    path = RUN/'paired-dense/window-scores.csv'
    if audit['status'] != 'paired_artifacts_audited' or sha(path) != audit['sources_sha256'][path.relative_to(ROOT).as_posix()]:
        raise ValueError('Audited score bytes required')
    if OUT.exists():
        raise FileExistsError(OUT)
    windows = {uid: [] for uid in CASES}
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['unit_id'] in windows:
                windows[row['unit_id']].append(row)
    summary = []
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    for ax, (uid, rows) in zip(axes, windows.items()):
        counts = {}
        for mode, label, color in [('rgb_zero', 'RGB', '#2f64a2'), ('rgb_diff', 'RGB＋帧差', '#cc7c32')]:
            scores = [float(r[mode]) for r in rows]
            peak = max(rows, key=lambda r: float(r[mode]))
            count = sum(v >= .5 for v in scores)
            counts[mode] = count
            summary.append(dict(unit_id=uid, video_label=CASES[uid], mode=mode, windows=len(rows),
                                above_threshold=count, fraction=count/len(rows),
                                score_min=min(scores), score_max=max(scores),
                                peak_start_seconds=float(peak['start']), peak_end_seconds=float(peak['end'])))
            ax.scatter([(float(r['start'])+float(r['end']))/120 for r in rows], scores,
                       s=4, color=color, label=label)
        ax.axhline(.5, linestyle='--', color='gray', linewidth=.8)
        ax.set(title=f"{uid}，标注{CASES[uid]}：过线窗 RGB {counts['rgb_zero']}／{len(rows)}；帧差 {counts['rgb_diff']}／{len(rows)}",
               xlabel='窗口中点时间（分）', ylabel='窗口分数（0–1）', ylim=(-.04, 1.04))
        ax.legend(loc='upper right', fontsize=8)
    fig.suptitle('seed8：误报替换、共同误报与被补回的漏报；固定0.5，不改标签')
    fig.tight_layout()
    OUT.mkdir(parents=True)
    outputs = []
    for ext in ('png', 'pdf', 'svg'):
        p = FIG/('seed8-error-timelines.'+ext)
        if p.exists():
            raise FileExistsError(p)
        fig.savefig(p, dpi=180, bbox_inches='tight')
        outputs.append(p)
    plt.close(fig)
    with (OUT/'window-patterns.csv').open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    # These are fixed supervised validation windows, not dense peak windows.
    # Reuse verified JPEGs: no decoder or model inference is invoked.
    plan = json.loads((ROOT/'data/metadata/EXP-016/training-plan-v2/plan.json').read_text(encoding='utf-8'))
    resource = ROOT/plan['resource']
    preds = {}
    for folder, epoch in [('rgb', 16), ('diff', 9)]:
        preds[folder] = json.loads((RUN/folder/f'validation-epoch-{epoch:02d}.json').read_text(encoding='utf-8'))
    bank = {}
    with (resource/'windows.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['unit_id'] in ('v1_33', 'v5_04'):
                bank[r['clip_id']] = r
    selected = []
    for uid in ('v1_33', 'v5_04'):
        prediction = max((r for r in preds['rgb'] if bank.get(r['clip_id'], {}).get('unit_id') == uid),
                         key=lambda r: r['logit'])
        clip = bank[prediction['clip_id']]
        b = next(r for r in preds['diff'] if r['clip_id'] == prediction['clip_id'])
        selected.append((clip, prediction, b))
    wanted = {(clip['unit_id'], i) for clip, _, _ in selected for i in clip['frame_indices']}
    index = next(ROOT/r['path'] for r in plan['sources'] if r['path'].endswith('verified-cache-files.csv'))
    frames = {}
    with index.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            key = row['unit_id'], int(row['frame_index'])
            if key in wanted:
                frames[key] = row
    if set(frames) != wanted:
        raise ValueError('Cached example frames missing')
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    examples = []
    for clip, a, b in selected:
        uid = clip['unit_id']
        pa, pb = [(1+math.tanh(r['logit']/2))/2 for r in (a, b)]
        sheet = Image.new('RGB', (1040, 1184), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 6), f'{uid} 固定验证窗 {clip["kind"]}，RGB {pa:.4f}／帧差 {pb:.4f}；非密评峰值', font=font, fill='black')
        hashes = []
        for j, (i, pts) in enumerate(zip(clip['frame_indices'], clip['source_pts_seconds'])):
            row = frames[uid, i]
            p = ROOT/row['cache_path']
            if sha(p) != row['jpeg_sha256']:
                raise ValueError('Cached example SHA changed')
            x, y = 4+(j % 4)*260, 40+(j//4)*286
            with Image.open(p) as im:
                if im.size != (256, 256):
                    raise ValueError('Unexpected cache size')
                sheet.paste(im.convert('RGB'), (x, y))
            draw.text((x, y+258), f'#{i} {pts:.3f}s', font=font, fill='black')
            hashes.append(dict(path=row['cache_path'], sha256=row['jpeg_sha256']))
        p = OUT/(uid+'-fixed-validation.png')
        sheet.save(p)
        outputs.append(p)
        examples.append(dict(clip=clip, rgb_probability=pa, diff_probability=pb,
                             selection='highest RGB score among frozen supervised validation clips for this video',
                             is_dense_peak=False, frames=hashes, sheet=p.relative_to(ROOT).as_posix()))
    (OUT/'generator-source.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'analysis.json').write_text(json.dumps(dict(status='read_only_four_case_analysis',
        source_scores_sha256=sha(path), generator_sha256=sha(Path(__file__)),
        threshold=.5, labels_unchanged=True, decoder_invoked=False, model_inference_invoked=False,
        window_patterns=summary, existing_jpeg_examples=examples,
        outputs_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in outputs}),
        ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(status='read_only_four_case_analysis', cases=len(windows), selected_frames=len(wanted))))


if __name__ == '__main__':
    main()
