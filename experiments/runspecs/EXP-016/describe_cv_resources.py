"""Describe CV data/compute feasibility from frozen metadata; no media decode.

Existing fold maps are reported, never rewritten or promoted to a new card.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.supervised_plan import csv_rows, digest, verify_plan, write_json
from experiments.common.video.pts_clip_sampling import dense_candidates
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    out.relative_to(ROOT/'data/metadata/EXP-016')
    if out.exists():
        raise FileExistsError(out)
    began = time.monotonic()
    plan_dir = ROOT/'data/metadata/EXP-016/training-plan-v2'
    plan = verify_plan(ROOT, plan_dir)
    resource = ROOT/plan['resource']
    videos = csv_rows(resource/'video-inputs.csv')
    manifest = {r['unit_id']: r for r in csv_rows(ROOT/'data/process/EXP-008-units-v2/unit-manifest.csv')}
    if len(videos) != 150 or {r['unit_id'] for r in videos} != set(manifest):
        raise ValueError('Expected unchanged 150-video development pool')
    kinds = defaultdict(lambda: defaultdict(int))
    with (resource/'windows.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            kinds[row['unit_id']][row['kind']] += 1
    scene_path = ROOT/'data/metadata/splits/EXP-008-units-scene-folds.json'
    session_path = ROOT/'data/metadata/splits/EXP-008-units-folds.json'
    scene_spec = json.loads(scene_path.read_text(encoding='utf-8'))
    session_spec = json.loads(session_path.read_text(encoding='utf-8'))
    scene_fold = {scene: int(fold['fold']) for fold in scene_spec['folds'] for scene in fold['val_scenes']}
    session_fold = {group: int(fold['fold']) for fold in session_spec['folds'] for group in fold['val_groups']}
    sources = {p.relative_to(ROOT).as_posix(): digest(p) for p in
               (SOURCE, plan_dir/'plan.json', resource/'video-inputs.csv', resource/'windows.jsonl',
                ROOT/'data/process/EXP-008-units-v2/unit-manifest.csv', scene_path, session_path,
                ROOT/'experiments/common/video/pts_clip_sampling.py')}
    rows = []
    for video in videos:
        uid, gid = video['unit_id'], video['group_id']
        parts = gid.split('|')
        if len(parts) != 3 or parts[0] != 'd':
            raise ValueError('Unexpected session group')
        scene = parts[1]
        if manifest[uid]['group_id'] != gid or int(manifest[uid]['fold']) != session_fold[gid]:
            raise ValueError('Existing session fold mapping conflict')
        coord = ROOT/video['coord_path']
        if digest(coord) != video['coord_sha256']:
            raise ValueError('Coordinate fingerprint changed')
        sources[video['coord_path']] = video['coord_sha256']
        with coord.open(encoding='utf-8-sig', newline='') as f:
            pts = np.fromiter((float(r['pts_time']) for r in csv.DictReader(f)), dtype=np.float64)
        grid, skipped = dense_candidates(pts, json.loads(video['clean_runs_json']))
        if video['dense_count'] and len(grid) != int(video['dense_count']):
            raise ValueError('Existing dense count differs from exact metadata grid')
        row = dict(unit_id=uid, label=video['video_label'], session_group=gid, scene_group=scene,
                   historical_session_fold=session_fold[gid], historical_scene_fold=scene_fold[scene],
                   pos_quota=min(4, kinds[uid].get('pos', 0)), neg_quota=min(4, kinds[uid].get('neg', 0)),
                   dense_windows=len(grid), skipped_targets_or_short_runs=len(skipped),
                   video_frames=int(video['n_frames']))
        if row['pos_quota']+row['neg_quota'] == 0:
            raise ValueError('Video has no usable supervised input')
        rows.append(row)
        if len(rows) % 25 == 0:
            print(json.dumps(dict(stage='metadata_grid_counts', videos=len(rows), total=150)), flush=True)

    def describe(subset, grouping):
        return dict(videos=len(subset), P=sum(r['label'] == 'pos' for r in subset),
                    N=sum(r['label'] == 'neg' for r in subset), groups=len({r[grouping] for r in subset}),
                    negative_groups=len({r[grouping] for r in subset if r['label'] == 'neg'}),
                    supervised_pos=sum(r['pos_quota'] for r in subset), supervised_neg=sum(r['neg_quota'] for r in subset),
                    dense_windows=sum(r['dense_windows'] for r in subset))
    protocols = {}
    for protocol, fold_key, group_key in [('session', 'historical_session_fold', 'session_group'),
                                         ('scene', 'historical_scene_fold', 'scene_group')]:
        groups = {r[group_key] for r in rows}
        if any(len({r[fold_key] for r in rows if r[group_key] == gid}) != 1 for gid in groups):
            raise ValueError('Group crosses historical folds')
        protocols[protocol] = [dict(fold=i, train=describe([r for r in rows if r[fold_key] != i], group_key),
                                   outer=describe([r for r in rows if r[fold_key] == i], group_key)) for i in range(1, 6)]
    out.mkdir(parents=True)
    with (out/'videos.csv').open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    result = dict(status='cv_resource_description_only', seconds=time.monotonic()-began,
                  total=describe(rows, 'session_group'), scene_total=describe(rows, 'scene_group'),
                  historical_fold_distributions=protocols, sources_sha256=sources,
                  limitations=['Existing video-balanced folds are described, not newly window-stratified or approved',
                               'Dense counts use PTS and known QC/cut metadata only; no pixel/decoder verification is added',
                               'No inner split, new training plan, model selection, or training authorization created'])
    write_json(out/'result.json', result)
    (out/'generator-source.py').write_bytes(SOURCE.read_bytes())
    print(json.dumps(dict(status=result['status'], total=result['total'], scene_total=result['scene_total'])))


if __name__ == '__main__':
    main()
