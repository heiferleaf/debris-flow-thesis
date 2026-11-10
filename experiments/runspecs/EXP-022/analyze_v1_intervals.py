# -*- coding: utf-8 -*-
"""Describe saved v1 interval errors and reuse nearby audit JPEGs; no decoding or inference."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'data/results/exp022/v1-score-review-20261106-v1'
VISUAL = ROOT / 'data/process/EXP-022/v1-content-review-20261106-v1/targeted'
RUN = ROOT / 'data/results/exp022/linear-cv-s7-20261104-v1'
SEGMENTS = ROOT / 'data/metadata/EXP-016/metadata-v4/segments.csv'


def read(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(p, rows):
    with p.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def summarize(rows):
    p = [r for r in rows if r['kind'] == 'pos']
    n = [r for r in rows if r['kind'] == 'neg']
    tp = sum(r['score'] >= .5 for r in p)
    fp = sum(r['score'] >= .5 for r in n)
    return dict(P=len(p), N=len(n), TP=tp, FN=len(p)-tp, FP=fp, TN=len(n)-fp,
                P_recall=tp/len(p) if p else None, N_FPR=fp/len(n) if n else None,
                P_median=median(r['score'] for r in p) if p else None,
                N_median=median(r['score'] for r in n) if n else None)


def build():
    if OUT.exists() or VISUAL.exists():
        raise FileExistsError('Use a new version')
    segments = defaultdict(list)
    for s in read(SEGMENTS):
        segments[s['unit_id']].append(s)
    windows = defaultdict(list)
    inputs = [SEGMENTS, Path(__file__).resolve()]
    for fold in range(1, 6):
        path = RUN / ('f%02d-dense' % fold) / 'window-scores.csv'
        inputs.append(path)
        for row in read(path):
            if not row['unit_id'].startswith('v1_'):
                continue
            for k in ['start', 'end', 'score', 'target_start_seconds']:
                row[k] = float(row[k])
            matches = [s for s in segments[row['unit_id']]
                       if float(s['annotation_start_sec']) <= row['start']
                       and row['end'] < float(s['annotation_end_sec'])]
            assert len(matches) <= 1
            row['kind'] = matches[0]['kind'] if matches else 'boundary_or_unannotated'
            row['segment_id'] = matches[0]['segment_id'] if matches else ''
            row['fold'] = fold
            windows[row['unit_id']].append(row)
    assert len(windows) == 106
    rows = []
    for uid, ws in sorted(windows.items()):
        for s in segments[uid]:
            selected = [r for r in ws if r['segment_id'] == s['segment_id']]
            rows.append(dict(unit_id=uid, fold=ws[0]['fold'], group_id=s['group_id'],
                             segment_id=s['segment_id'], kind=s['kind'],
                             annotation_start=s['annotation_start_sec'], annotation_end=s['annotation_end_sec'],
                             **summarize(selected)))
    OUT.mkdir(parents=True)
    write(OUT/'interval-counts.csv', rows)
    video_rows = [dict(unit_id=u, fold=ws[0]['fold'], **summarize(ws)) for u, ws in sorted(windows.items())]
    write(OUT/'video-counts.csv', video_rows)
    grouped = defaultdict(list)
    for u, ws in windows.items():
        grouped[segments[u][0]['group_id']].extend(ws)
    write(OUT/'group-counts.csv', [dict(group_id=g, **summarize(ws)) for g, ws in sorted(grouped.items())])
    VISUAL.mkdir()
    audit = defaultdict(list)
    audit_root = ROOT/'data/process/video_frames/video1_segment'
    for r in read(audit_root/'frames_index.csv'):
        audit[r['video_key']].append((float(r['t_sec']), audit_root/'frames'/r['frame_file']))
    # Selected after the first, score-hidden visual pass. These are context sequences,
    # not the exact16 physical model frames and not estimates of annotation accuracy.
    requests = [('v1_60',250), ('v1_63',500), ('v1_63',343), ('v1_65',1260),
                ('v1_105',600), ('v1_106',900), ('v1_122',1020),
                ('v1_95',650), ('v1_95',1230), ('v1_95',1335), ('v1_91',100)]
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    seq = []
    for uid, target in requests:
        window = min(windows[uid], key=lambda r: abs((r['start']+r['end'])/2-target))
        center = (window['start']+window['end'])/2
        available = sorted(audit[uid])
        picked = []
        for offset in [-5,-3,-1,1,3,5]:
            f = min(available, key=lambda f: abs(f[0]-center-offset))
            if f not in picked:
                picked.append(f)
        sheet = Image.new('RGB',(1500,665),'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8,3),'{} context; saved3s [{:.2f},{:.2f}] p={:.4f} {}'.format(
            uid,window['start'],window['end'],window['score'],window['kind']),font=font,fill='#183442')
        for i,(t,path) in enumerate(picked):
            with Image.open(path) as im:
                thumb=im.convert('RGB');thumb.thumbnail((490,275),Image.Resampling.LANCZOS)
            x,y=(i%3)*500+4,(i//3)*310+35
            sheet.paste(thumb,(x,y));draw.text((x,y+278),'audit t={:.2f}s'.format(t),font=font,fill='#183442')
            seq.append(dict(unit_id=uid,target_center=target,window_start=window['start'],window_end=window['end'],
                            score=window['score'],kind=window['kind'],audit_time=t,
                            source=path.relative_to(ROOT).as_posix()))
        sheet.save(VISUAL/('{}-{}.jpg'.format(uid,target)),quality=95)
    write(VISUAL/'index.csv',seq)
    result=dict(videos=len(windows), supervised_segments=len(rows), all=summarize([r for ws in windows.values() for r in ws]),
                conditional_scope='Frozen pureP/N windows in existing cut+QC domain, fixed0.5; overlapping windows not independent.',
                image_scope='Existing2s audit context frames, not exact16 model inputs; no video decoding.',
                targeted_sequences=len(requests),sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    print(json.dumps(build(),ensure_ascii=False,indent=2))
