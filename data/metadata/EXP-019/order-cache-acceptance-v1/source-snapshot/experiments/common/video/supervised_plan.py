"""Freeze EXP-016 assignments from the already pixel-verified clip bank."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def rooted(root, relative):
    p = (root / relative).resolve()
    p.relative_to(root.resolve())
    return p


def bucket_schedule(segment_banks, epochs=20):
    """One video/class shares <=4 slots; cover original segments then rotate.

    Each original segment keeps its own bank; no frames are stitched together.
    Within a segment's allocation, use disjoint chronological strata.
    """
    banks = [tuple(x) for x in segment_banks if len(x)]
    if epochs < 1:
        raise ValueError('Positive epoch budget required')
    if not banks:
        return [[] for _ in range(epochs)]
    flat = [x for b in banks for x in b]
    if len(flat) != len(set(flat)):
        raise ValueError('Repeated clip in video/class bank')
    k = min(4, len(flat))
    allocation_schedule = []
    for epoch in range(epochs):
        allocations = Counter()
        cursor = (epoch * k) % len(banks)
        for _ in range(k):
            while allocations[cursor] == len(banks[cursor]):
                cursor = (cursor + 1) % len(banks)
            allocations[cursor] += 1
            cursor = (cursor + 1) % len(banks)
        allocation_schedule.append(allocations)
    pools, visited = {}, Counter()
    for segment, bank in enumerate(banks):
        visits = sum(a[segment] for a in allocation_schedule)
        count = min(visits, len(bank))
        if count == 0:
            continue
        representatives = (np.array([(len(bank) - 1) // 2]) if count == 1
                           else np.rint(np.linspace(0, len(bank) - 1, count)).astype(int))
        max_slots = max(a[segment] for a in allocation_schedule)
        strata = np.array_split(representatives, max_slots)
        # Interleave chronological strata; short budgets still cover the bank.
        pools[segment] = [bank[int(part[i])] for i in range(max(map(len, strata)))
                          for part in strata if i < len(part)]
    output = []
    for allocations in allocation_schedule:
        selected = []
        for segment, count in sorted(allocations.items()):
            pool = pools[segment]
            for _ in range(count):
                selected.append(pool[visited[segment] % len(pool)])
                visited[segment] += 1
        if len(selected) != k or len(set(selected)) != k:
            raise ValueError('Invalid shared quota')
        output.append(selected)
    first = {x for row in output[:(len(banks) + k - 1) // k] for x in row}
    if epochs >= (len(banks) + k - 1) // k and any(not first.intersection(bank) for bank in banks):
        raise ValueError('Initial original-segment coverage incomplete')
    return output


def freeze_plan(root, resource, destination):
    root, resource, destination = map(lambda p: Path(p).resolve(), (root, resource, destination))
    destination.relative_to(root / 'data/metadata/EXP-016')
    if destination.exists():
        raise FileExistsError(destination)
    completion = root / 'data/process/EXP-016/metadata-v4/cache-completion.json'
    complete = json.loads(completion.read_text(encoding='utf-8'))
    if complete['status'] != 'supervised_cache_ready' or complete['videos_verified'] != 150:
        raise ValueError('Verified 150-video cache required')
    if digest(resource / 'frozen-plan-closure.json') != complete['frozen_plan_closure_sha256']:
        raise ValueError('Resource closure fingerprint mismatch')
    closure = json.loads((resource / 'frozen-plan-closure.json').read_text(encoding='utf-8'))
    closure_files = {r['path']: r for r in closure['files']}
    used = [resource / name for name in ('windows.jsonl', 'segments.csv', 'video-inputs.csv',
                                        'dense-eval-windows.jsonl', 'dense-transfer-references.csv')]
    for p in used:
        expected = closure_files[p.relative_to(root).as_posix()]
        if p.stat().st_size != expected['size_bytes'] or digest(p) != expected['sha256']:
            raise ValueError(f'Resource changed: {p}')
    index = rooted(root, complete['verified_cache_index'])
    if digest(index) != complete['verified_cache_index_sha256']:
        raise ValueError('Verified pixel index changed')
    holdout = root / 'data/metadata/splits/EXP-016-development-holdout.json'
    split = json.loads(holdout.read_text(encoding='utf-8'))
    videos = csv_rows(resource / 'video-inputs.csv')
    by_video = {r['unit_id']: r for r in videos}
    if len(by_video) != 150:
        raise ValueError('Unexpected source video count')
    groups = {side: {r['group_id'] for r in videos if r['split'] == side}
              for side in ('train', 'validation')}
    if groups['train'] & groups['validation']:
        raise ValueError('Group leakage')
    for side, key in [('train', 'train_video_ids'), ('validation', 'validation_video_ids')]:
        if {r['unit_id'] for r in videos if r['split'] == side} != set(split[key]):
            raise ValueError('Frozen holdout mismatch')
    segments = {(r['unit_id'], r['segment_id']): r for r in csv_rows(resource / 'segments.csv')}
    bank = jsonl(resource / 'windows.jsonl')
    by_id, buckets, seen_tuples = {}, defaultdict(lambda: defaultdict(list)), set()
    for row in bank:
        uid, sid, kind, cid = row['unit_id'], row['segment_id'], row['kind'], row['clip_id']
        seg, video = segments[(uid, sid)], by_video[uid]
        ix, pts = row['frame_indices'], row['source_pts_seconds']
        key = (uid, tuple(ix))
        if cid in by_id or key in seen_tuples:
            raise ValueError('Duplicate clip ID or frame tuple')
        if (kind != seg['kind'] or row['split'] != video['split'] or row['group_id'] != video['group_id']
                or len(ix) != 16 or len(pts) != 16 or any(np.diff(ix) <= 0)
                or any(np.diff(pts) <= 0) or not np.isfinite(pts).all()
                or not 2.9 - 1e-9 <= pts[-1] - pts[0] <= 3.1 + 1e-9
                or pts[0] < float(seg['sampling_start_sec']) - 1e-9
                or pts[-1] >= float(seg['sampling_end_sec'])):
            raise ValueError(f'Invalid supervised bank row: {cid}')
        clean = json.loads(video['clean_runs_json'])
        if not any(ix[0] >= a and ix[-1] < b for a, b in clean):
            raise ValueError(f'Clip bridges quality exclusion: {cid}')
        seen_tuples.add(key)
        by_id[cid] = row
        buckets[(row['split'], uid, kind)][sid].append(cid)
    assignments, coverage = [], []
    for (side, uid, kind), segment_map in sorted(buckets.items()):
        segment_ids = sorted(segment_map, key=lambda sid: (float(segments[(uid, sid)]['start_sec']), sid))
        banks = [sorted(segment_map[sid], key=lambda cid: by_id[cid]['candidate_index']) for sid in segment_ids]
        schedule = bucket_schedule(banks, 20 if side == 'train' else 1)
        for epoch, ids in enumerate(schedule):
            for cid in ids:
                assignments.append(dict(split=side, epoch=epoch if side == 'train' else -1,
                                        clip_id=cid, unit_id=uid, kind=kind))
        coverage.append(dict(split=side, unit_id=uid, kind=kind, quota=len(schedule[0]),
                             original_segments=segment_ids,
                             first_coverage_epochs=(len(banks) + len(schedule[0]) - 1) // len(schedule[0])))
    expected = {'train': Counter(pos=356, neg=148), 'validation': Counter(pos=96, neg=40)}
    for epoch in range(20):
        counts = Counter(r['kind'] for r in assignments if r['split'] == 'train' and r['epoch'] == epoch)
        if counts != expected['train']:
            raise ValueError(f'Card training counts mismatch: {counts}')
    if Counter(r['kind'] for r in assignments if r['split'] == 'validation') != expected['validation']:
        raise ValueError('Card validation counts mismatch')
    if {r['unit_id'] for r in assignments} != set(by_video):
        raise ValueError('A video silently disappeared')
    destination.mkdir(parents=True, exist_ok=False)
    assign_path = destination / 'assignments.csv'
    with assign_path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(assignments[0]))
        writer.writeheader()
        writer.writerows(assignments)
    write_json(destination / 'coverage.json', coverage)
    sources = used + [resource / 'frozen-plan-closure.json', completion, index, holdout]
    manifest = dict(spec_id='EXP-016', version=1, resource=resource.relative_to(root).as_posix(),
                    epochs=20, seed=7, train_count=504, train_pos=356, train_neg=148,
                    pos_weight=37 / 89, validation_count=136, effective_batch=24,
                    microbatch_max=8, dense_windows=32567,
                    unique_clips=len({r['clip_id'] for r in assignments}),
                    sources=[dict(path=p.relative_to(root).as_posix(), sha256=digest(p),
                                  size_bytes=p.stat().st_size) for p in sources],
                    files={p.name: digest(p) for p in (assign_path, destination / 'coverage.json')},
                    generator_sha256=digest(__file__), independent_test_used=False)
    write_json(destination / 'plan.json', manifest)
    return manifest


def verify_plan(root, directory):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
    for name, sha in plan['files'].items():
        if digest(rooted(directory, name)) != sha:
            raise ValueError(f'Assignment changed: {name}')
    for ref in plan['sources']:
        p = rooted(root, ref['path'])
        if p.stat().st_size != ref['size_bytes'] or digest(p) != ref['sha256']:
            raise ValueError(f'Input changed: {ref["path"]}')
    return plan
