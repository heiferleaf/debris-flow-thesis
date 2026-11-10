"""EXP-017 role-free metadata only; reuse frozen candidates and old sampling rules."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

from experiments.common.video.pts_clip_sampling import fixed_quota_rotation, tuple_digest
from experiments.common.video.supervised_plan import bucket_schedule, csv_rows, digest, jsonl, rooted, write_json

EPOCHS = 20
SPLIT = 'data/metadata/splits/EXP-017-session-proposal-v1.json'
SPLIT_SHA = '4fd318e78074417b2ea6dd2f7e0bc4cfd4c47e0eb8b68b8f89b38d7a48c8ab5a'
CLOSURE = 'data/metadata/EXP-016/metadata-v4/frozen-plan-closure.json'
CLOSURE_SHA = '735bb8e3a6b25842c673ad8720b81e19029b018e69cc23a7a1e300445e82a06f'
METADATA = 'data/metadata/EXP-016/metadata-v4'


def representative_indices(n):
    """Always construct twenty-epoch representatives, independent of outer E."""
    rotation, pools = fixed_quota_rotation(n, EPOCHS)
    return sorted({int(i) for pool in pools for i in pool}), rotation.tolist()


def video_schedules(banks):
    epochs, fixed = [[] for _ in range(EPOCHS)], []
    for kind in ('neg', 'pos'):
        parts = banks.get(kind, [])
        selected = bucket_schedule(parts, EPOCHS)
        for epoch, row in enumerate(selected): epochs[epoch].extend(row)
        fixed.extend(bucket_schedule(parts, 1)[0])
    return epochs, fixed


def prefix_epochs(epochs, selected_epochs):
    if (len(epochs) != EPOCHS or type(selected_epochs) is not int or not 1 <= selected_epochs <= EPOCHS):
        raise ValueError('Outer E must select a prefix of the complete frozen twenty epochs')
    return epochs[:selected_epochs]


def validate_epoch_quota(ids, bank):
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('An assignment must contain distinct clips')
    video = Counter((bank[c]['unit_id'], bank[c]['kind']) for c in ids)
    segment = Counter((bank[c]['unit_id'], bank[c]['segment_id']) for c in ids)
    if max(video.values()) > 4 or max(segment.values()) * 16 > 64:
        raise ValueError('Shared video/class quota or original-segment64-frame cap exceeded')


def training_weight(ids, bank):
    labels = [bank[c]['label'] for c in ids]
    if any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError('Training labels must be binary integers')
    p, n = sum(labels), len(labels) - sum(labels)
    if not p or not n: raise ValueError('Both training classes are required')
    return n / p


def validate_partitions(roles, sources):
    expected = {'inner_train', 'inner_val', 'outer_train', 'outer_eval'}
    if set(roles) != expected: raise ValueError('Missing CV role')
    sides = {k: set(v) for k, v in roles.items()}
    if any(len(sides[k]) != len(roles[k]) or not sides[k] <= set(sources) for k in roles):
        raise ValueError('Duplicate or unknown CV video identity')
    if (sides['inner_train'] & sides['inner_val'] or sides['outer_train'] & sides['outer_eval']
            or sides['outer_train'] != sides['inner_train'] | sides['inner_val']
            or sides['outer_train'] | sides['outer_eval'] != set(sources)):
        raise ValueError('CV role overlap, complement or refit union mismatch')
    group_sets = {k: {sources[u]['group_id'] for u in v} for k, v in sides.items()}
    if (group_sets['inner_train'] & group_sets['inner_val']
            or group_sets['outer_train'] & group_sets['outer_eval']):
        raise ValueError('A session group crossed CV sides')


def _write_csv(path, rows):
    import csv
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _write_jsonl(path, rows):
    with path.open('x', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def _counts(video_ids, sources, clips, bank):
    labels = Counter(sources[u]['video_label'] for u in video_ids)
    windows = Counter(bank[c]['kind'] for c in clips)
    return dict(videos=len(video_ids), P=labels['pos'], N=labels['neg'],
                pos_windows=windows['pos'], neg_windows=windows['neg'],
                groups=len({sources[u]['group_id'] for u in video_ids}),
                negative_groups=len({sources[u]['group_id'] for u in video_ids if sources[u]['video_label'] == 'neg'}))


def _folds(split, sources, bank, epochs, fixed):
    result, outer_coverage = [], Counter()
    names = {'inner_train': 'inner_train', 'inner_val': 'inner_validation',
             'outer_train': 'outer_refit_train', 'outer_eval': 'outer_evaluation'}
    if split['n_folds'] != 5 or [f['fold'] for f in split['folds']] != list(range(1, 6)):
        raise ValueError('Frozen five-fold identity changed')
    for fold in split['folds']:
        roles = {name: fold[old]['video_ids'] for name, old in names.items()}
        validate_partitions(roles, sources)
        row = {'fold': fold['fold']}
        for name, old in names.items():
            ids = list(roles[name]); wanted = set(ids)
            ordered_epochs = [[c for c in e['clip_ids'] if bank[c]['unit_id'] in wanted] for e in epochs]
            fixed_ids = [c for c in fixed['clip_ids'] if bank[c]['unit_id'] in wanted]
            counts = _counts(ids, sources, fixed_ids, bank)
            groups = sorted({sources[u]['group_id'] for u in ids})
            if counts != fold[old]['counts'] or groups != fold[old]['group_ids']:
                raise ValueError('Actual role-free quotas differ from frozen fold proposal')
            stage = dict(video_ids=ids, group_ids=groups, counts=counts)
            if name in ('inner_train', 'outer_train'):
                expected = Counter(bank[c]['kind'] for c in ordered_epochs[0])
                weight = training_weight(ordered_epochs[0], bank)
                for sequence in ordered_epochs:
                    validate_epoch_quota(sequence, bank)
                    if Counter(bank[c]['kind'] for c in sequence) != expected or training_weight(sequence, bank) != weight:
                        raise ValueError('Training N/P varies by epoch')
                # Same old per-fit default_rng(seed7) permutation sequence. Do
                # this only after the complete20-epoch assignment bank exists.
                rng = np.random.default_rng(7)
                shuffled = [[sequence[int(i)] for i in rng.permutation(len(sequence))] for sequence in ordered_epochs]
                stage.update(pos_weight=weight, epoch_clip_ids=shuffled,
                    weight_source='this_training_side_only; never inner/outer validation labels',
                    order='np.random.default_rng(7), one permutation per complete epoch; trainer must not reshuffle')
                if any(prefix_epochs(shuffled, e) != shuffled[:e] for e in (1, 2, 7, 20)):
                    raise ValueError('Frozen outer prefix check failed')
            elif name == 'inner_val':
                validate_epoch_quota(fixed_ids, bank)
                stage.update(clip_ids=fixed_ids, fixed=True, selection_metric='unweighted_BCE_earliest_minimum')
            else:
                stage.update(training_or_selection_use=False, evaluation='full_video_label_independent_grid_in_later_stage')
                outer_coverage.update(ids)
            row[name] = stage
        inner = row['inner_val']['counts']; train = row['inner_train']['counts']
        fraction = ((inner['pos_windows'] + inner['neg_windows']) /
                    (row['outer_train']['counts']['pos_windows'] + row['outer_train']['counts']['neg_windows']))
        if (not .15 <= fraction <= .30 or min(inner['P'], inner['N']) < 4
                or inner['negative_groups'] < 2 or train['negative_groups'] < 4):
            raise ValueError('Frozen inner coverage constraints failed')
        if not row['outer_eval']['counts']['P'] or not row['outer_eval']['counts']['N']:
            raise ValueError('Outer evaluation video class missing')
        row['audit'] = dict(session_isolation=True, exact_refit_union=True, inner_window_fraction=fraction,
            no_outer_labels_in_training_weight=True, outer_epoch_prefix_of_frozen20=True)
        result.append(row)
    if set(outer_coverage) != set(sources) or set(outer_coverage.values()) != {1}:
        raise ValueError('Each of the150 videos must be evaluated exactly once')
    return result


def freeze_cv_plan(root, destination, split_path=None, max_seconds=300):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    destination.relative_to(root / 'data/metadata/EXP-017')
    if destination.exists(): raise FileExistsError(destination)
    if not 0 < max_seconds <= 300: raise ValueError('Metadata CPU preparation cap is300seconds')
    split_path = Path(split_path).resolve() if split_path else root / SPLIT
    if split_path != root / SPLIT: raise ValueError('Only frozen proposal-v1 may be reused; no split search')
    destination.mkdir(parents=True, exist_ok=False)
    began, fingerprints = time.monotonic(), {}
    def check():
        if time.monotonic() - began >= max_seconds: raise TimeoutError('Metadata300-second cap reached')
    def bind(path, expected=None):
        check(); path = Path(path).resolve(); relative = path.relative_to(root).as_posix()
        value = digest(path)
        if expected is not None and value != expected: raise ValueError('Frozen input SHA changed: ' + relative)
        fingerprints[relative] = dict(path=relative, sha256=value, size_bytes=path.stat().st_size)
        return path
    try:
        closure = json.loads(bind(root / CLOSURE, CLOSURE_SHA).read_text(encoding='utf-8'))
        refs = {r['path']: r for r in closure['files']}
        def frozen(relative):
            record = refs[relative]; path = bind(root / relative, record['sha256'])
            if path.stat().st_size != record['size_bytes']: raise ValueError('Frozen metadata size changed')
            return path
        videos = csv_rows(frozen(METADATA + '/video-inputs.csv'))
        segment_rows = csv_rows(frozen(METADATA + '/segments.csv'))
        legacy_summary = json.loads(frozen(METADATA + '/summary.json').read_text(encoding='utf-8'))
        if len(videos) != 150 or len({v['unit_id'] for v in videos}) != 150:
            raise ValueError('All150 original videos, including prior long-interval cases, are required')
        if Counter(v['video_label'] for v in videos) != Counter(pos=113, neg=37):
            raise ValueError('Original video-label coverage changed')
        split = json.loads(bind(split_path, SPLIT_SHA).read_text(encoding='utf-8'))
        source_paths = [Path(__file__).resolve(), root / 'experiments/runspecs/EXP-017/prepare_plan.py',
                        root / 'experiments/common/video/pts_clip_sampling.py', root / 'experiments/common/video/supervised_plan.py',
                        root / 'experiments/common/video/supervised_training.py']
        for p in source_paths:
            bind(p, legacy_summary['common_sampling_sha256'] if p.name == 'pts_clip_sampling.py' else None)
        for relative in ('docs/process/experiments/EXP-017.md', 'docs/process/experiments/EXP-006-数据使用方案.md',
                         '.agents/skills/data-usage-protocol/SKILL.md'):
            bind(root / relative)
        if not legacy_summary['supervision_core_review'] or legacy_summary['supervision_core_review']['decision'] != 'approved_supervision_core':
            raise ValueError('Existing v1_09 conservative supervision core required')
        grouped = defaultdict(list)
        for seg in segment_rows: grouped[seg['unit_id']].append(seg)
        if set(grouped) != {v['unit_id'] for v in videos}: raise ValueError('Segment/source coverage differs')
        sources, banks, all_windows, segment_banks, frame_rows = {}, {}, {}, [], []
        by_video_epochs, by_video_fixed = {}, {}
        legacy_bank_counts = Counter()
        for video in sorted(videos, key=lambda r: r['unit_id']):
            check(); uid = video['unit_id']
            if video['video_label'] not in ('pos', 'neg') or video['split'] not in ('train', 'validation'):
                raise ValueError('Invalid original source identity/legacy role')
            coord = csv_rows(bind(root / video['coord_path'], video['coord_sha256']))
            qc = csv_rows(bind(root / video['qc_path'], video['qc_sha256']))
            n = int(video['n_frames'])
            if (len(coord) != n or len(qc) != n or [int(r['frame']) for r in coord] != list(range(n))
                    or [int(r['frame']) for r in qc] != list(range(n))):
                raise ValueError('Source coordinate/QC count or exact frame order mismatch')
            pts = np.array([float(r['pts_time']) for r in coord], dtype=np.float64)
            qt, std = np.array([float(r['pts_time']) for r in qc]), np.array([float(r['gray_std']) for r in qc])
            if not np.isfinite(pts).all() or not np.isfinite(qt).all() or not np.isfinite(std).all() or np.max(abs(qt - pts)) > .001001:
                raise ValueError('Source coordinate/QC finite or PTS alignment failure')
            source = dict(video); source['legacy_split'] = source.pop('split')
            source['media_hash_scope'] = 'frozen_v4_reference_not_rehashed_or_decoded_in_this_metadata_stage'
            sources[uid] = source
            relative = 'data/process/EXP-016/metadata-v4/candidates/%s.npz' % uid
            candidate_path = frozen(relative)
            segments = sorted(grouped[uid], key=lambda s: (float(s['start_sec']), s['segment_id']))
            if len({s['segment_id'] for s in segments}) != len(segments): raise ValueError('Duplicate original segment ID')
            video_banks = {'neg': [], 'pos': []}; frame_tuples = set()
            with np.load(candidate_path, allow_pickle=False) as archive:
                if set(archive.files) != {s['segment_id'] for s in segments}: raise ValueError('Candidate segment keys changed')
                for seg in segments:
                    sid, kind = seg['segment_id'], seg['kind']
                    candidates = archive[sid]
                    if (kind not in ('pos', 'neg') or seg['group_id'] != video['group_id']
                            or candidates.shape != (int(seg['n_candidates']), 16)
                            or not np.issubdtype(candidates.dtype, np.integer)
                            or tuple_digest(candidates) != seg['candidates_sha256']):
                        raise ValueError('Candidate count/kind/group/shape/digest mismatch')
                    chosen, rotation = representative_indices(len(candidates))
                    ids = []
                    a, b = float(seg['sampling_start_sec']), float(seg['sampling_end_sec'])
                    if not float(seg['annotation_start_sec']) <= a < b <= float(seg['annotation_end_sec']):
                        raise ValueError('Conservative sampling core extended original annotation')
                    clean = json.loads(video['clean_runs_json'])
                    for index in chosen:
                        ix = candidates[index].astype(np.int64)
                        if np.any(ix < 0) or np.any(ix >= n) or np.any(np.diff(ix) <= 0):
                            raise ValueError('Invalid representative physical indices')
                        t = pts[ix]
                        if (np.any(np.diff(t) <= 0) or not 2.9 - 1e-9 <= t[-1] - t[0] <= 3.1 + 1e-9
                                or t[0] < a - 1e-9 or t[-1] >= b or np.min(std[ix]) < 20.
                                or not any(ix[0] >= l and ix[-1] < r for l, r in clean)):
                            raise ValueError('Representative breaches PTS/class/core/quality/cut closure')
                        key = tuple(map(int, ix))
                        if key in frame_tuples: raise ValueError('Duplicate representative frame tuple across original segments')
                        frame_tuples.add(key)
                        cid = '%s:%s:%s' % (uid, sid, tuple_digest(ix)[:20])
                        if cid in all_windows: raise ValueError('Duplicate role-free clip identity')
                        ids.append(cid)
                        all_windows[cid] = dict(clip_id=cid, unit_id=uid, group_id=video['group_id'],
                            segment_id=sid, kind=kind, label=int(kind == 'pos'), origin=seg['origin'],
                            sampling_scope=seg['sampling_scope'], sampling_start_sec=a, sampling_end_sec=b,
                            candidate_index=index, frame_indices=ix.tolist(), source_pts_seconds=t.tolist())
                    video_banks[kind].append(ids)
                    segment_banks.append(dict(unit_id=uid, segment_id=sid, kind=kind,
                        n_raw_candidates=len(candidates), representative_candidate_indices=chosen,
                        clip_ids=ids, segment_rotation_candidate_indices20=rotation))
            epochs, fixed = video_schedules(video_banks)
            if not epochs[0] or not any(all_windows[c]['kind'] == video['video_label'] for c in epochs[0]):
                raise ValueError('A video lost all legal inputs of its own label')
            by_video_epochs[uid], by_video_fixed[uid] = epochs, fixed
            referenced = {c for e in epochs for c in e} | set(fixed)
            needed = sorted({i for c in referenced for i in all_windows[c]['frame_indices']})
            for i in needed:
                frame_rows.append(dict(unit_id=uid, frame_index=i, source_pts_seconds=float(pts[i]), frozen_qc_std=float(std[i])))
            legacy_bank_counts[video['split']] += len(frame_tuples)
        epochs = [dict(epoch=e, clip_ids=[c for uid in sorted(sources) for c in by_video_epochs[uid][e]]) for e in range(EPOCHS)]
        fixed = dict(clip_ids=[c for uid in sorted(sources) for c in by_video_fixed[uid]], by_video=by_video_fixed)
        referenced_train = {c for e in epochs for c in e['clip_ids']}; referenced_fixed = set(fixed['clip_ids'])
        for c, window in all_windows.items():
            window.update(used_in_20_epoch_schedule=c in referenced_train, used_in_fixed_validation=c in referenced_fixed)
        for assignment in [e['clip_ids'] for e in epochs] + [fixed['clip_ids']]:
            validate_epoch_quota(assignment, all_windows)
            if Counter(all_windows[c]['kind'] for c in assignment) != Counter(pos=452, neg=188):
                raise ValueError('Global shared video/class quotas differ from proposal452P/188N')
        folds = _folds(split, sources, all_windows, epochs, fixed)
        _write_csv(destination / 'sources.csv', [sources[u] for u in sorted(sources)])
        _write_jsonl(destination / 'windows.jsonl', [all_windows[c] for c in sorted(all_windows)])
        _write_jsonl(destination / 'epochs.jsonl', epochs)
        write_json(destination / 'fixed-validation.json', fixed)
        write_json(destination / 'folds.json', folds)
        write_json(destination / 'segment-banks.json', segment_banks)
        _write_csv(destination / 'frames-required.csv', frame_rows)
        audit = dict(status='metadata_checks_passed_pixels_unverified', videos=150, video_P=113, video_N=37,
            sessions=len({s['group_id'] for s in sources.values()}), raw_candidate_count=sum(s['n_raw_candidates'] for s in segment_banks),
            role_free_representative_clips=len(all_windows), referenced_training_clips=len(referenced_train),
            referenced_fixed_validation_clips=len(referenced_fixed), required_unique_unit_frames=len(frame_rows),
            legacy_role_video_coverage=dict(Counter(s['legacy_split'] for s in sources.values())),
            legacy_role_bank_clips=dict(legacy_bank_counts), all150_retained_without_legacy_role_or_long_interval_filter=True,
            shared_quota=dict(P=452, N=188, total=640, epochs=20),
            checks=['candidate SHA/count/shape/tuple digest', 'selected physical index/PTS/class/core/QC/clean-run closure',
                    'video-class<=4 and original-segment<=64frames every epoch and fixed validation',
                    'five-fold whole-session isolation and exact refit union', 'outer coverage exactlyonce for all150',
                    'per-training-side N/P independent from evaluation side', 'outer E prefix of frozen20bank and shuffle'],
            metadata_only=True, pixels_ready=False, training_ready=False, cache_manifest=None,
            actual_cache_missing_frames='not_inspected; separate subsequent cache-inventory/verification stage',
            native_timebase_caveat='v1_09 approved supervision core retained; full clock mapping remains unresolved')
        write_json(destination / 'audit.json', audit)
        for p in source_paths:
            relative = p.relative_to(root); target = destination / 'source-snapshot' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f: f.write(p.read_bytes())
        check()
        for ref in fingerprints.values():
            if (root / ref['path']).stat().st_size != ref['size_bytes'] or digest(root / ref['path']) != ref['sha256']:
                raise ValueError('Input/source/card/protocol changed while generating metadata')
        files = {p.relative_to(destination).as_posix(): digest(p) for p in sorted(destination.rglob('*')) if p.is_file()}
        plan = dict(spec_id='EXP-017', version=1, status='metadata_frozen_pixels_unverified',
            metadata_path=METADATA, epochs=20, seed=7, split_path=SPLIT, split_sha256=SPLIT_SHA,
            candidate_rule='fixed_quota_rotation(n,20) per original segment; sorted representative union; then bucket_schedule(segment_banks,20)',
            fixed_validation_rule='bucket_schedule(the_same_role_free_segment_banks,1)',
            training_order='per fold and training side, independent np.random.default_rng(7); one permutation each epoch as old EXP016; no trainer reshuffle',
            outer_rule='take firstE epoch_clip_ids from complete frozen20; reinitialize model externally; no bank regeneration',
            shuffle_source='experiments/common/video/supervised_training.py::train old np.default_rng(seed)/rng.permutation(len(current))',
            source_fingerprints=list(fingerprints.values()), files=files,
            media_fingerprint_scope='sources.csv binds existing middleware SHA from frozen v4; no media bytes read/decoded here',
            frame_requirements_scope='union of referenced20epoch+fixed-validation clips; unused representative-bank clips need no pixels yet',
            frame_lookup='(unit_id,frame_index); PTS/QC in frames-required.csv; source paths and SHA in sources.csv; cache inventory remains separate',
            pixels_ready=False, training_ready=False, cache_manifest=None, no_model_scores_read=True,
            seconds=time.monotonic() - began, cpu_metadata_seconds_cap=max_seconds)
        write_json(destination / 'plan.json', plan)
        return plan, audit
    except Exception as error:
        write_json(destination / 'failure.json', dict(status='metadata_preparation_failed_no_pixels_ready',
            error_type=type(error).__name__, error=str(error), seconds=time.monotonic() - began,
            source_fingerprints=list(fingerprints.values()), no_decode=True, no_training=True))
        raise


def load_cv_plan(root, directory, verify_sources=True):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
    if (plan['spec_id'] != 'EXP-017' or plan['epochs'] != 20 or plan['split_sha256'] != SPLIT_SHA
            or plan['pixels_ready'] or plan['cache_manifest'] is not None):
        raise ValueError('Unregistered or incorrectly pixel-promoted metadata plan')
    if (directory / 'failure.json').exists(): raise ValueError('Failed plan cannot be consumed')
    for name, fingerprint in plan['files'].items():
        if digest(rooted(directory, name)) != fingerprint: raise ValueError('Role-free plan file changed: ' + name)
    if verify_sources:
        for ref in plan['source_fingerprints']:
            path = rooted(root, ref['path'])
            if path.stat().st_size != ref['size_bytes'] or digest(path) != ref['sha256']:
                raise ValueError('Role-free plan source changed: ' + ref['path'])
    windows = jsonl(directory / 'windows.jsonl')
    bank = {r['clip_id']: r for r in windows}
    epochs = jsonl(directory / 'epochs.jsonl')
    fixed = json.loads((directory / 'fixed-validation.json').read_text(encoding='utf-8'))
    folds = json.loads((directory / 'folds.json').read_text(encoding='utf-8'))
    if len(bank) != len(windows) or [e['epoch'] for e in epochs] != list(range(20)):
        raise ValueError('Duplicate role-free clip or missing epoch')
    return plan, bank, epochs, fixed, folds
