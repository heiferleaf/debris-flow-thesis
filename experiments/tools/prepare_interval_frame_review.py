"""Render every existing audit frame in a requested interval; never decode video."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.tools import prepare_saved_transition_review as renderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--start', type=float, required=True)
    parser.add_argument('--end', type=float, required=True)
    parser.add_argument('--tag', required=True, help='New receipt name; existing output is never overwritten')
    args = parser.parse_args()
    match = re.fullmatch(r'v([145])_\d+', args.unit)
    if not match or not 0 <= args.start < args.end or not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        raise ValueError('Expected development-pool unit, increasing nonnegative interval, and safe tag')
    index = ROOT / ('data/process/video_frames/video%s_segment/frames_index.csv' % match.group(1))
    frames, missing = [], []
    with index.open(encoding='utf-8-sig', newline='') as stream:
        for record, row in enumerate(csv.DictReader(stream), 2):
            if row['video_key'] != args.unit or not args.start <= float(row['t_sec']) < args.end:
                continue
            path = index.parent / 'frames' / row['frame_file']
            t = float(row['t_sec'])
            frame = {'path': path.relative_to(ROOT).as_posix(), 't_sec': t, 't_sec_text': row['t_sec'],
                     'frame_file': row['frame_file'], 'index_csv_record': record,
                     'index_in_video': int(row['frame_file'].split('_frame_')[1].split('_')[0])}
            (frames if path.is_file() else missing).append(frame)
    # Preserve source index order and expose backsteps rather than silently repairing time.
    backsteps = []
    for i, frame in enumerate(frames):
        gap = None if not i else frame['t_sec'] - frames[i-1]['t_sec']
        frame.update(sequence_index=i+1, gap_from_previous_sec=gap)
        if gap is not None and gap <= 0:
            backsteps.append({'previous': frames[i-1]['index_csv_record'], 'current': frame['index_csv_record'], 'delta': gap})
    if not frames:
        raise ValueError('No existing audit frame in the requested interval')
    temp = ROOT / '.codex/docs/.temp/full-temporal-20261108-v1' / ('fine-' + args.tag)
    metadata = ROOT / 'data/metadata/annotation-review/full-temporal-20261108-v1/reviews' / (args.tag + '-preparation.json')
    if temp.exists() or metadata.exists():
        raise FileExistsError('Choose a new tag; previous evidence is preserved')
    renderer.TITLE = 'Existing cached interval review'
    panels = renderer.contact_sheets(args.unit, frames, temp)
    result = {'status': 'prepared_not_reviewed', 'unit_id': args.unit,
              'requested_interval_seconds': [args.start, args.end],
              'time_scope': 'existing audit index t_sec; not a new original-frame time mapping',
              'ordering': 'original CSV source-index order', 'PTS_backsteps': backsteps,
              'frame_count': len(frames), 'frames': frames, 'missing_indexed_files': missing,
              'panels': panels, 'video_decode_count': 0, 'label_changes': 0}
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'frames': len(frames), 'missing': len(missing), 'backsteps': len(backsteps),
                      'metadata': metadata.relative_to(ROOT).as_posix(), 'panels': [p['path'] for p in panels]}))


if __name__ == '__main__':
    main()
