# -*- coding: utf-8 -*-
"""Build bounded contact sheets from existing audit JPEGs; never decode video or relabel."""
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'data/process/EXP-022/v1-content-review-20261106-v1'
SOURCE = ROOT / 'data/metadata/EXP-017/role-free-plan-v3/sources.csv'
AUDIT = ROOT / 'data/process/video_frames/video1_segment'


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build():
    if OUT.exists():
        raise FileExistsError(OUT)
    sources = {r['unit_id']: r for r in read(SOURCE) if r['unit_id'].startswith('v1_')}
    units = sorted(sources, key=lambda u: int(u.split('_')[1]))
    assert len(units) == 106
    batches = {u: 'ABC'[min(i // 36, 2)] for i, u in enumerate(units)}
    frames = defaultdict(list)
    for r in read(AUDIT / 'frames_index.csv'):
        if r['video_key'] in sources:
            frames[r['video_key']].append((float(r['t_sec']), AUDIT / 'frames' / r['frame_file']))
    for rows in frames.values():
        rows.sort()
    events = [dict(r, event_line=i) for i, r in enumerate(read(AUDIT / 'events.csv'), 2)
              if r['video_key'] in sources and r['confidence'] == 'real']
    events.sort(key=lambda r: (int(r['video_key'].split('_')[1]), float(r['start_sec'])))
    assert len(events) == 183
    OUT.mkdir(parents=True)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 19)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    panels = defaultdict(list)
    manifest = []
    interval_rows = []
    ids = Counter()
    for r in events:
        uid = r['video_key']
        a, b = float(r['start_sec']), float(r['end_sec'])
        ids[uid] += 1
        interval = '{}-I{:02d}'.format(uid, ids[uid])
        available = [(t, p) for t, p in frames[uid] if a <= t < b and p.is_file()]
        if not available:
            raise ValueError('No existing JPEG in {} [{},{})'.format(uid, a, b))
        m = min(1., (b-a)/10)
        targets = [a+m, a+.25*(b-a), a+.5*(b-a), a+.75*(b-a), b-m]
        selected = []
        for target in targets:
            remaining = [v for v in available if v not in selected]
            if not remaining:
                break
            selected.append(min(remaining, key=lambda v: abs(v[0]-target)))
        selected.sort()
        panel = Image.new('RGB', (1800, 260), 'white')
        draw = ImageDraw.Draw(panel)
        draw.text((8, 2), '{}  {} [{:g},{:g})s'.format(interval, r['kind'], a, b), font=font, fill='#102d3a')
        for col, (t, path) in enumerate(selected):
            with Image.open(path) as image:
                w, h = image.size
                thumb = image.convert('RGB')
                thumb.thumbnail((350, 195), Image.Resampling.LANCZOS)
            panel.paste(thumb, (col*360+5, 30))
            draw.text((col*360+5, 231), 't={:g}s  {}x{}'.format(t, w, h), font=small, fill='#324a58')
            manifest.append(dict(interval_id=interval, unit_id=uid, batch=batches[uid], kind=r['kind'],
                                 start=a, end=b, time=t, path=path.relative_to(ROOT).as_posix(), width=w, height=h))
        panels[batches[uid]].append((interval, panel))
        extras = r.get(None, [])
        note = ','.join([r.get('note', ''), *extras]).strip(',')
        interval_rows.append(dict(interval_id=interval, unit_id=uid, batch=batches[uid], kind=r['kind'],
                                  start=a, end=b, group_id=sources[uid]['group_id'], event_line=r['event_line'],
                                  sampled_frames=len(selected), note=note))
    sheets = []
    for batch, items in sorted(panels.items()):
        for offset in range(0, len(items), 4):
            chunk = items[offset:offset+4]
            name = '{}-{:02d}.jpg'.format(batch, offset//4+1)
            sheet = Image.new('RGB', (1800, 260*len(chunk)), 'white')
            for i, (_, panel) in enumerate(chunk):
                sheet.paste(panel, (0, i*260))
            sheet.save(OUT/name, quality=94)
            sheets.append(dict(batch=batch, sheet=name, interval_ids=';'.join(k for k, _ in chunk)))
    write(OUT/'sampled-frames.csv', manifest)
    write(OUT/'intervals.csv', interval_rows)
    write(OUT/'sheets.csv', sheets)
    (OUT/'events.snapshot.csv').write_bytes((AUDIT/'events.csv').read_bytes())
    result = dict(videos=len(units), intervals=len(events), frames=len(manifest),
                  interval_counts=dict(Counter(r['kind'] for r in interval_rows)),
                  batches={b: dict(videos=sum(batches[u] == b for u in units),
                                   intervals=sum(r['batch'] == b for r in interval_rows),
                                   sheets=sum(r['batch'] == b for r in sheets)) for b in 'ABC'},
                  decoding=False, label_changes=False, scores_shown=False,
                  sampling='Near endpoints and 25/50/75 percent, within each interval; short intervals may have fewer distinct JPEGs.',
                  limits='Static sparse coverage only; no claim of every-second label accuracy or physical motion.',
                  sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [SOURCE, AUDIT/'events.csv', AUDIT/'frames_index.csv', Path(__file__).resolve()]})
    (OUT/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False, indent=2))
