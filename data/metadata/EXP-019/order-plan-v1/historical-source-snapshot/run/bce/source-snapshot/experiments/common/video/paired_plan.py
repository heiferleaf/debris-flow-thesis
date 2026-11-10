"""EXP-018 frozen paired batches, reusing every EXP-016 endpoint exactly once."""
from __future__ import annotations
from collections import Counter, defaultdict
from itertools import permutations
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from experiments.common.video.supervised_plan import csv_rows, digest, jsonl, rooted, verify_plan, write_json

PAIR_QUOTAS = (4, 4, 4, 4, 12, 16, 16, 20)
BASE_PLAN = 'data/metadata/EXP-016/training-plan-v2'


def validate_window(row, segment, video):
    """Half-open original-class and quality intervals, with 16 increasing frames."""
    ix, pts = row['frame_indices'], np.asarray(row['source_pts_seconds'], dtype=float)
    if (row['kind'] not in ('pos', 'neg')
            or any(row[k] != segment[k] for k in ('unit_id', 'segment_id', 'kind', 'split', 'group_id'))
            or any(row[k] != video[k] for k in ('unit_id', 'split', 'group_id'))
            or len(ix) != 16 or len(pts) != 16
            or any(not isinstance(i, Integral) or isinstance(i, bool) or i < 0 for i in ix)
            or any(np.diff(ix) <= 0) or not np.isfinite(pts).all() or any(np.diff(pts) <= 0)
            or not 2.9 - 1e-9 <= pts[-1] - pts[0] <= 3.1 + 1e-9
            or pts[0] < float(segment['sampling_start_sec']) - 1e-9
            or pts[-1] >= float(segment['sampling_end_sec'])):
        raise ValueError('Invalid closed supervised window: %s' % row['clip_id'])
    if not any(ix[0] >= a and ix[-1] < b for a, b in json.loads(video['clean_runs_json'])):
        raise ValueError('Supervised window bridges a quality/cut exclusion')


def _base_inputs(root, directory):
    plan = verify_plan(root, directory)
    expected = dict(epochs=20, seed=7, train_count=504, train_pos=356, train_neg=148,
                    validation_count=136, effective_batch=24, microbatch_max=8)
    if any(plan.get(k) != v for k, v in expected.items()) or plan['pos_weight'] != 148 / 356:
        raise ValueError('Registered EXP-016 base contract changed')
    assignments = csv_rows(directory / 'assignments.csv')
    resource = rooted(root, plan['resource'])
    selected = {r['clip_id'] for r in assignments}
    raw_bank = jsonl(resource / 'windows.jsonl')
    bank = {r['clip_id']: r for r in raw_bank if r['clip_id'] in selected}
    if len(bank) != len(selected) or len({r['clip_id'] for r in raw_bank}) != len(raw_bank):
        raise ValueError('Missing/duplicate clip ID in base bank')
    segments = {(r['unit_id'], r['segment_id']): r for r in csv_rows(resource / 'segments.csv')}
    videos = {r['unit_id']: r for r in csv_rows(resource / 'video-inputs.csv')}
    groups = {side: {r['group_id'] for r in videos.values() if r['split'] == side}
              for side in ('train', 'validation')}
    if groups['train'] & groups['validation']:
        raise ValueError('Base group leakage')
    for row in bank.values():
        validate_window(row, segments[(row['unit_id'], row['segment_id'])], videos[row['unit_id']])
    schedule, validation = {i: [] for i in range(20)}, []
    for row in assignments:
        clip = bank[row['clip_id']]
        if any(row[k] != clip[k] for k in ('unit_id', 'kind', 'split')):
            raise ValueError('Base assignment label/source mismatch')
        if row['split'] == 'validation' and row['epoch'] == '-1':
            validation.append(row['clip_id'])
        elif row['split'] == 'train' and int(row['epoch']) in schedule:
            schedule[int(row['epoch'])].append(row['clip_id'])
        else:
            raise ValueError('Unexpected assignment split/epoch')
    if (len(validation) != len(set(validation)) or len(validation) != 136
            or Counter(bank[c]['kind'] for c in validation) != Counter(pos=96, neg=40)):
        raise ValueError('Fixed validation coverage changed')
    return plan, bank, schedule, validation


def _training_quota(ids, bank):
    if (len(ids) != 504 or len(set(ids)) != 504
            or Counter(bank[c]['kind'] for c in ids) != Counter(pos=356, neg=148)
            or any(bank[c]['split'] != 'train' for c in ids)):
        raise ValueError('Expected distinct original 356P/148N training endpoints')
    by_video = Counter((bank[c]['unit_id'], bank[c]['kind']) for c in ids)
    by_segment = Counter((bank[c]['unit_id'], bank[c]['segment_id']) for c in ids)
    if max(by_video.values()) > 4 or max(by_segment.values()) * 16 > 64:
        raise ValueError('Video/class shared quota or original segment 64-frame quota exceeded')
    groups = defaultdict(lambda: {'pos': [], 'neg': []})
    for cid in ids:
        r = bank[cid]
        groups[r['group_id']][r['kind']].append(cid)
    quotas = {g: min(len(v['pos']), len(v['neg'])) for g, v in groups.items()}
    quotas = {g: q for g, q in quotas.items() if q}
    if tuple(sorted(quotas.values())) != PAIR_QUOTAS:
        raise ValueError('Registered eight-group pairing quotas changed')
    return groups, quotas


def build_epoch(ids, bank, rng, epoch=0):
    """Only the supplied Generator consumes randomness; torch/global NumPy untouched."""
    groups, quotas = _training_quota(ids, bank)
    queues = {}
    for group in sorted(quotas):
        p = [groups[group]['pos'][i] for i in rng.permutation(len(groups[group]['pos']))]
        n = [groups[group]['neg'][i] for i in rng.permutation(len(groups[group]['neg']))]
        queues[group] = list(zip(p[:quotas[group]], n[:quotas[group]]))
    blocks, endpoints = [], set()
    for _ in range(20):
        selected, per_group = [], Counter()
        for _ in range(4):
            candidates = [g for g in queues if queues[g] and per_group[g] < 2]
            if not candidates:
                raise ValueError('Cannot pack four pairs with maximum two per group')
            largest = max(len(queues[g]) for g in candidates)
            tied = sorted(g for g in candidates if len(queues[g]) == largest)
            group = tied[int(rng.integers(len(tied)))]
            selected.append((group, queues[group].pop()))
            per_group[group] += 1
        selected = [selected[i] for i in rng.permutation(4)]
        group_ids = [g for g, _ in selected]
        valid = [perm for perm in permutations(range(4))
                 if all(group_ids[i] != group_ids[perm[i]] for i in range(4))]
        if not valid:
            raise ValueError('No strict cross-group negative permutation')
        perm = valid[int(rng.integers(len(valid)))]
        clips = [p for _, (p, _) in selected] + [n for _, (_, n) in selected]
        endpoints.update(clips)
        blocks.append(dict(clip_ids=clips, same_pairs=[[i, i + 4] for i in range(4)],
                           cross_pairs=[[i, perm[i] + 4] for i in range(4)]))
    if any(queues.values()):
        raise ValueError('Unpacked pairs remain')
    plain = sorted(set(ids) - endpoints)
    plain = [plain[i] for i in rng.permutation(len(plain))]
    plain_blocks = [dict(clip_ids=plain[i:i + 8], same_pairs=[], cross_pairs=[])
                    for i in range(0, len(plain), 8)]
    batches = []
    for i in range(21):
        chunks = ([blocks[i]] + plain_blocks[2*i:2*i+2] if i < 20 else plain_blocks[40:43])
        chunks = [chunks[j] for j in rng.permutation(3)]
        batch = dict(clip_ids=[], same_pairs=[], cross_pairs=[])
        for chunk in chunks:
            offset = len(batch['clip_ids'])
            batch['clip_ids'].extend(chunk['clip_ids'])
            for kind in ('same_pairs', 'cross_pairs'):
                batch[kind].extend([[p + offset, n + offset] for p, n in chunk[kind]])
        batches.append(batch)
    result = dict(epoch=epoch, batches=[batches[i] for i in rng.permutation(len(batches))])
    validate_epoch(result, ids, bank)
    return result


def validate_epoch(row, original_ids, bank):
    """Independent closure checker; validates both relationships, not generator internals."""
    _, expected_quota = _training_quota(original_ids, bank)
    batches = row['batches']
    all_ids = [c for b in batches for c in b['clip_ids']]
    if (len(batches) != 21 or any(len(b['clip_ids']) != 24 for b in batches)
            or len(all_ids) != 504 or len(set(all_ids)) != 504 or set(all_ids) != set(original_ids)):
        raise ValueError('Epoch endpoint coverage differs from original base plan')
    observed, endpoint_set, paired_batches = Counter(), set(), 0
    for batch in batches:
        ids, same, cross = batch['clip_ids'], batch['same_pairs'], batch['cross_pairs']
        if len(same) != len(cross) or len(same) not in (0, 4):
            raise ValueError('Exactly one four-pair microblock per paired effective batch required')
        if not same:
            continue
        paired_batches += 1
        endpoints_by_kind = []
        for kind, pairs in (('same', same), ('cross', cross)):
            seen, pos, neg, micros = set(), set(), set(), set()
            for pair in pairs:
                if (len(pair) != 2 or any(not isinstance(i, Integral) or isinstance(i, bool)
                                          or not 0 <= i < 24 for i in pair)):
                    raise ValueError('Invalid pair local index')
                p, n = pair
                if p // 8 != n // 8 or p in seen or n in seen:
                    raise ValueError('Pair must be closed in one microblock with unique endpoints')
                rp, rn = bank[ids[p]], bank[ids[n]]
                if rp['kind'] != 'pos' or rn['kind'] != 'neg':
                    raise ValueError('Pair endpoints must be ordered positive, negative')
                if (rp['group_id'] == rn['group_id']) != (kind == 'same'):
                    raise ValueError('Same/cross group pairing mismatch')
                seen.update((p, n)); pos.add(p); neg.add(n); micros.add(p // 8)
                if kind == 'same':
                    observed[rp['group_id']] += 1
            if len(micros) != 1 or len(seen) != 8:
                raise ValueError('All four pairs must occupy a single microblock')
            endpoints_by_kind.append((pos, neg))
        if endpoints_by_kind[0] != endpoints_by_kind[1]:
            raise ValueError('Cross permutation changed paired endpoints')
        pos, neg = endpoints_by_kind[0]
        if max(Counter(bank[ids[p]]['group_id'] for p in pos).values()) > 2:
            raise ValueError('A microblock contains more than two pairs from one group')
        current = {ids[i] for i in pos | neg}
        if current & endpoint_set:
            raise ValueError('An endpoint participates twice in an epoch')
        endpoint_set.update(current)
    if paired_batches != 20 or dict(observed) != expected_quota or len(endpoint_set) != 160:
        raise ValueError('Eighty-pair endpoint/group quota incomplete')


def freeze_pair_plan(root, base_plan, destination):
    root, base_plan, destination = (Path(p).resolve() for p in (root, base_plan, destination))
    destination.relative_to(root / 'data/metadata/EXP-018')
    if destination.exists():
        raise FileExistsError(destination)
    base, bank, schedule, validation = _base_inputs(root, base_plan)
    if base_plan.relative_to(root).as_posix() != BASE_PLAN:
        raise ValueError('EXP-018 only reuses registered training-plan-v2')
    rng = np.random.default_rng(7)
    epochs = [build_epoch(schedule[i], bank, rng, epoch=i) for i in range(20)]
    destination.mkdir(parents=True, exist_ok=False)
    epoch_file = destination / 'epochs.jsonl'
    with epoch_file.open('x', encoding='utf-8') as f:
        for row in epochs:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    plan = dict(spec_id='EXP-018', version=1, seed=7, epochs=20,
                base_plan_path=base_plan.relative_to(root).as_posix(),
                base_plan_sha256=digest(base_plan / 'plan.json'), resource=base['resource'],
                train_count=504, train_pos=356, train_neg=148, validation_count=136,
                validation_clip_ids=validation, pos_weight=148 / 356, effective_batch=24,
                microbatch=8, optimizer_updates_per_epoch=21, pairs_per_epoch=80,
                rank_lambda=.1, rank_scale=21 / 80, pair_group_quotas={
                    g: q for g, q in sorted(_training_quota(schedule[0], bank)[1].items())},
                files={'epochs.jsonl': digest(epoch_file)}, generator_sha256=digest(Path(__file__).resolve()),
                independent_test_used=False)
    write_json(destination / 'plan.json', plan)
    return plan


def load_pair_plan(root, plan_dir):
    root, plan_dir = Path(root).resolve(), Path(plan_dir).resolve()
    plan = json.loads((plan_dir / 'plan.json').read_text(encoding='utf-8'))
    required = dict(spec_id='EXP-018', version=1, seed=7, epochs=20, train_count=504,
                    train_pos=356, train_neg=148, validation_count=136, effective_batch=24,
                    microbatch=8, optimizer_updates_per_epoch=21, pairs_per_epoch=80,
                    rank_lambda=.1, rank_scale=21 / 80, base_plan_path=BASE_PLAN)
    if any(plan.get(k) != v for k, v in required.items()) or plan.get('pos_weight') != 148 / 356:
        raise ValueError('Unregistered EXP-018 paired plan contract')
    if set(plan['files']) != {'epochs.jsonl'}:
        raise ValueError('Unexpected paired-plan file list')
    for name, sha in plan['files'].items():
        if digest(rooted(plan_dir, name)) != sha:
            raise ValueError('Paired schedule fingerprint changed')
    base_dir = rooted(root, plan['base_plan_path'])
    if digest(base_dir / 'plan.json') != plan['base_plan_sha256']:
        raise ValueError('Base plan fingerprint changed')
    base, bank, schedule, validation = _base_inputs(root, base_dir)
    if plan['resource'] != base['resource'] or plan['validation_clip_ids'] != validation:
        raise ValueError('Paired plan changed base resource/validation order')
    epochs = jsonl(plan_dir / 'epochs.jsonl')
    if len(epochs) != 20 or [r['epoch'] for r in epochs] != list(range(20)):
        raise ValueError('Expected frozen sequential twenty-epoch schedule')
    for i, row in enumerate(epochs):
        validate_epoch(row, schedule[i], bank)
        if plan['pair_group_quotas'] != _training_quota(schedule[i], bank)[1]:
            raise ValueError('Registered group quotas changed across epochs')
    return plan, epochs
