"""Extract only fixed error-analysis windows; never change labels or scores."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from experiments.common.video.dense_evaluation import checked_frames
    from experiments.common.video.pts_clip_sampling import jpeg95_bytes
    from experiments.common.video.supervised_plan import csv_rows, digest
    run = ROOT/'data/results/exp016/paired-s7-20261030-v2'
    metadata = ROOT/'data/metadata/EXP-016/metadata-v4'
    out = ROOT/'data/process/EXP-016/error-examples-v1'
    out.mkdir(parents=True, exist_ok=False)
    source = {r['unit_id']: r for r in csv_rows(metadata/'video-inputs.csv')}
    targets = {uid: [] for uid in ('v1_120', 'v1_62', 'v1_121')}
    with (run/'paired-dense/window-scores.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['unit_id'] in targets:
                targets[row['unit_id']].append(row)
    wanted = {}
    for uid, rows in targets.items():
        peak = max(rows, key=lambda r: float(r['rgb_diff']))
        cases = [('peak', peak)]
        if uid != 'v1_121':
            below = [r for r in rows if float(r['rgb_diff']) < .5]
            low = min(below, key=lambda r: abs(float(r['start'])-float(peak['start'])))
            cases.append(('nearest-below-threshold', low))
        wanted[uid] = cases
    frozen = {}
    with (metadata/'dense-eval-windows.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            uid = row['unit_id']
            if uid in wanted:
                for name, score in wanted[uid]:
                    if (abs(row['source_pts_seconds'][0]-float(score['start'])) < 1e-9
                            and abs(row['source_pts_seconds'][-1]-float(score['end'])) < 1e-9):
                        frozen[(uid, name)] = row
    if len(frozen) != 5:
        raise ValueError('Error-analysis grid lookup mismatch')
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    records = []
    for uid, cases in wanted.items():
        info = source[uid]
        pts = np.array([float(r['pts_time']) for r in csv_rows(ROOT/info['coord_path'])])
        indices = sorted({i for name, _ in cases for i in frozen[(uid, name)]['frame_indices']})
        directory = out/uid
        directory.mkdir()
        proof = {}
        frames = {}
        for i, bgr in checked_frames(ROOT/info['middleware_path'], pts,
                                    Path('D:/CodexCache/debris-flow-thesis/tools/ffmpeg-9.0.2/ffmpeg.exe'),
                                    proof, indices, directory/'select.txt'):
            path = directory/f'{i:07d}.jpg'
            path.write_bytes(jpeg95_bytes(bgr))
            frames[i] = path
        for name, score in cases:
            window = frozen[(uid, name)]
            canvas = Image.new('RGB', (1040, 1184), 'white')
            draw = ImageDraw.Draw(canvas)
            title = f"{uid} | {name} | 标注neg | 融合分数{float(score['rgb_diff']):.4f}"
            draw.text((8, 8), title, fill='black', font=font)
            for j, index in enumerate(window['frame_indices']):
                x, y = 4+(j % 4)*260, 40+(j // 4)*286
                with Image.open(frames[index]) as image:
                    canvas.paste(image.convert('RGB'), (x, y))
                draw.text((x, y+258), f'#{index}  {pts[index]:.3f}s', fill='black', font=font)
            image_path = directory/(name+'.png')
            canvas.save(image_path)
            records.append(dict(unit_id=uid, name=name, score=float(score['rgb_diff']), window=window,
                                sheet=image_path.relative_to(ROOT).as_posix(), sheet_sha256=digest(image_path),
                                source_path=info['middleware_path'], source_sha256=digest(ROOT/info['middleware_path']),
                                frames={str(i): dict(path=frames[i].relative_to(ROOT).as_posix(), sha256=digest(frames[i]))
                                        for i in window['frame_indices']}, decoder=proof))
        print(json.dumps(dict(unit=uid, source_frames=proof['decoded'], selected_frames=len(indices))), flush=True)
    (out/'generator-source.py').write_bytes(Path(__file__).read_bytes())
    (out/'examples.json').write_text(json.dumps(dict(
        status='read_only_error_examples', score_csv_sha256=digest(run/'paired-dense/window-scores.csv'),
        frozen_grid_sha256=digest(metadata/'dense-eval-windows.jsonl'), generator_sha256=digest(__file__),
        labels_unchanged=True, threshold_unchanged=True, examples=records), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
