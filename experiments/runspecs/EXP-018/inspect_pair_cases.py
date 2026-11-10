"""Review existing misordered validation clips; no decoder or model inference."""
import csv
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    out = ROOT/'data/process/EXP-018/pair-case-review-v1'
    out.mkdir(parents=True, exist_ok=False)
    predictions_path = ROOT/'data/results/exp016/paired-s8-20261031-v1/rgb/validation-epoch-16.json'
    plan_path = ROOT/'data/metadata/EXP-016/training-plan-v2/plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    resource = ROOT/plan['resource']
    predictions = json.loads(predictions_path.read_text(encoding='utf-8'))
    selected = []
    for uid in ('v1_122', 'v1_63', 'v1_82'):
        current = [p for p in predictions if p['clip_id'].startswith(uid+':')]
        for label, pick in ((1, min), (0, max)):
            p = pick([p for p in current if p['label'] == label], key=lambda p:p['logit'])
            selected.append(dict(p, unit_id=uid))
    ids = {p['clip_id'] for p in selected}
    bank = {}
    with (resource/'windows.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['clip_id'] in ids:
                bank[r['clip_id']] = r
    assert len(bank) == 6
    segments = {(r['unit_id'], r['segment_id']):r for r in rows(resource/'segments.csv')}
    wanted = {(r['unit_id'], i) for r in bank.values() for i in r['frame_indices']}
    completion_path = ROOT/'data/process/EXP-016/metadata-v4/cache-completion.json'
    completion = json.loads(completion_path.read_text(encoding='utf-8'))
    frames = {}
    for r in rows(ROOT/completion['verified_cache_index']):
        key = (r['unit_id'], int(r['frame_index']))
        if key in wanted:
            path = ROOT/r['cache_path']
            assert sha(path) == r['jpeg_sha256']
            frames[key] = dict(path=r['cache_path'], sha256=r['jpeg_sha256'])
    assert set(frames) == wanted
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    records = []
    for uid in ('v1_122', 'v1_63', 'v1_82'):
        sheet = Image.new('RGB', (1048, 684), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), uid+' | existing seed8 RGB best16 | fixed clips only', fill='black', font=font)
        for side, p in enumerate([p for p in selected if p['unit_id'] == uid]):
            window = bank[p['clip_id']]
            segment = segments[(uid, window['segment_id'])]
            assert window['split'] == 'validation'
            assert window['kind'] == ('pos' if p['label'] else 'neg')
            first, last = window['source_pts_seconds'][0], window['source_pts_seconds'][-1]
            assert float(segment['sampling_start_sec']) <= first < last < float(segment['sampling_end_sec'])
            x0 = 8+side*520
            title = f"{'POS' if p['label'] else 'NEG'} logit={p['logit']:.5f} | {first:.2f}-{last:.2f}s"
            draw.text((x0, 35), title, fill='black', font=font)
            draw.text((x0, 57), f"annotation {segment['start_sec']}-{segment['end_sec']}s", fill='black', font=font)
            for j, (index, pts) in enumerate(zip(window['frame_indices'], window['source_pts_seconds'])):
                ref = frames[(uid, index)]
                with Image.open(ROOT/ref['path']) as image:
                    tile = image.convert('RGB').resize((126, 126))
                x, y = x0+(j%4)*128, 86+(j//4)*146
                sheet.paste(tile, (x, y))
                draw.text((x, y+128), f'{pts:.2f}s', fill='black', font=font)
            records.append(dict(prediction=p, window=window, segment=segment,
                                frames=[frames[(uid,i)] for i in window['frame_indices']]))
        sheet.save(out/(uid+'.png'))
    (out/'generator-source.py').write_bytes(SOURCE.read_bytes())
    result = dict(status='pending_visual_review', inference=False, video_decoding=False,
                  labels_changed=False, rows=records,
                  source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in
                                 (SOURCE, predictions_path, plan_path, resource/'windows.jsonl',
                                  resource/'segments.csv', completion_path)},
                  figures={uid+'.png':sha(out/(uid+'.png')) for uid in ('v1_122','v1_63','v1_82')})
    (out/'cases.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=out.relative_to(ROOT).as_posix(), clips=6, verified_frames=len(frames))))


if __name__ == '__main__':
    main()
