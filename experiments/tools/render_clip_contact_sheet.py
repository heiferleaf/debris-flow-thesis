"""Render a real 16-frame manifest clip for visual QC, without changing frames."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--kind', choices=['pos', 'neg'], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    metadata = (root / args.metadata).resolve()
    with (metadata / 'windows.jsonl').open(encoding='utf-8') as stream:
        clip = next((row for row in map(json.loads, stream)
                     if row['unit_id'] == args.unit and row['kind'] == args.kind), None)
    if clip is None:
        raise ValueError('No matching manifest clip')
    with (metadata / 'frames-required.csv').open(encoding='utf-8-sig', newline='') as stream:
        frames = {int(row['frame_index']): root / row['cache_path']
                  for row in csv.DictReader(stream) if row['unit_id'] == args.unit}
    assert len(clip['frame_indices']) == 16
    relative_dataset = metadata.relative_to((root / 'data/metadata').resolve())
    output = root / 'data/process' / relative_dataset / 'visual-qc'
    output.mkdir(parents=True, exist_ok=True)
    target = output / ('%s-%s-first.png' % (args.unit, args.kind))
    sidecar = target.with_suffix('.json')
    if target.exists() or sidecar.exists():
        raise FileExistsError('QC output exists; refusing to overwrite')
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    canvas = Image.new('RGB', (1040, 1184), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), '%s | %s | %s' % (args.unit, args.kind, clip['split']), fill='black', font=font)
    evidence = []
    for i, (index, pts) in enumerate(zip(clip['frame_indices'], clip['source_pts_seconds'])):
        path = frames[index]
        with Image.open(path) as image:
            image.load()
            if image.size != (256, 256):
                raise ValueError('Unexpected cached frame size: %s' % path)
            x, y = 4 + (i % 4) * 260, 40 + (i // 4) * 286
            canvas.paste(image.convert('RGB'), (x, y))
            draw.text((x, y + 258), '#%d  %.3fs' % (index, pts), fill='black', font=font)
        evidence.append({'path': path.relative_to(root).as_posix(), 'sha256': digest(path)})
    canvas.save(target, format='PNG')
    record = {'clip': clip, 'frames': evidence, 'renderer_sha256': digest(Path(__file__)),
              'command': [sys.executable, 'experiments/tools/render_clip_contact_sheet.py'] + sys.argv[1:]}
    with sidecar.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print(target)


if __name__ == '__main__':
    main()
