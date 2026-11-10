"""EXP-019 imports the actual EXP-018 A input template, never CV representatives."""
from __future__ import annotations
from collections import Counter
import csv
import json
import re
from pathlib import Path
from pathlib import PurePosixPath
import time

import numpy as np

from experiments.common.video.paired_plan import load_pair_plan
from experiments.common.video.supervised_plan import csv_rows, digest, jsonl, rooted, write_json
from experiments.common.video.temporal_order import VERSION, PERMUTATION_SEED, window_order_record

BASE = 'data/metadata/EXP-016/training-plan-v2'
BASE_SHA = 'bb3af415133e1fe40a7a891bc80e2b22b0e3d44b97f2dec43ac0d6ca5750bb3f'
PAIRED = 'data/metadata/EXP-018/paired-plan-v1'
PAIRED_SHA = '7cda9b1c278e3af4b78ab7945bf656ff377ebbbdf07dd8e1d038b9d95b83b874'
EPOCHS_SHA = 'a1411779adbf41e6c9a6a4880a624df824ce94cf04f70760bde7bf4e54a5db12'
OLD_RUN = 'data/results/exp018/paired-s7-20261101-v1'
COLLECTOR = '.codex/logs/evidence/paired-s7-20261101-v1-collection.jsonl'
COLLECTOR_SHA = '51c79ce753bc2b8899f44167283570dc0f14bf3ed94cc9b7a7d146cfc483f41c'
DEPLOYMENT = 'data/metadata/EXP-018/deployment-v1'
DEPLOYMENT_INDEX_SHA = 'd37318d1a7dfef83c9bad62cee9a3f151b4562c9b3d2af3a94cd2aaf562c30c1'
INITIAL_SHA = 'c1847e1c0987ff003bff99805ab795af8eb59b07db210c7999e377b337ee9b81'


def write_jsonl(path, rows):
    with path.open('x', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def verify_snapshot_file(path, expected_sha, expected_size=None):
    """Validate archive bytes; never substitute a changed live governance file."""
    path = Path(path)
    if (expected_size is not None and path.stat().st_size != expected_size) or digest(path) != expected_sha:
        raise ValueError('Historical/current frozen source snapshot differs from recorded bytes')


def import_epochs(paired_rows, original_schedule, bank):
    if len(paired_rows) != 20 or [r['epoch'] for r in paired_rows] != list(range(20)):
        raise ValueError('Expected complete frozen twenty-epoch A template')
    result = []
    for epoch, row in enumerate(paired_rows):
        batches = row['batches']; ids = [c for b in batches for c in b['clip_ids']]
        if (len(batches) != 21 or any(len(b['clip_ids']) != 24 for b in batches)
                or len(ids) != 504 or len(set(ids)) != 504 or set(ids) != set(original_schedule[epoch])
                or Counter(bank[c]['kind'] for c in ids) != Counter(pos=356, neg=148)
                or any(bank[c]['split'] != 'train' for c in ids)):
            raise ValueError('A template batch/count/identity/source/class mismatch')
        quota = Counter((bank[c]['unit_id'], bank[c]['kind']) for c in ids)
        segments = Counter((bank[c]['unit_id'], bank[c]['segment_id']) for c in ids)
        if max(quota.values()) > 4 or max(segments.values()) * 16 > 64:
            raise ValueError('Shared quota/original-segment64-frame limit changed')
        result.append(dict(epoch=epoch, batches=batches, clip_ids=ids))
    return result


def validate_history_order(history, epoch_rows):
    if len(history) != 20: raise ValueError('Historical A training did not complete twenty epochs')
    for i, (record, epoch) in enumerate(zip(history, epoch_rows), 1):
        if (record['completed_epochs'] != i or record['samples'] != 504 or record['optimizer_steps'] != 21
                or record['total_updates'] != 21 * i or record['train_clip_ids'] != epoch['clip_ids']
                or record['objective_mode'] != 'bce' or record['rank_lambda'] != 0.):
            raise ValueError('Historical A did not use this exact batch sequence/budget')


def _historical_template(root, destination, bind):
    collector_path = bind(root / COLLECTOR, COLLECTOR_SHA, 'historical_origin')
    collection = jsonl(collector_path)
    records = {}
    for row in collection:
        if row.get('status') == 'download_sha_verified':
            if row['path'] in records and records[row['path']] != row: raise ValueError('Contradictory historical output receipt')
            records[row['path']] = row
    if (collection[-1].get('status') != 'collection_snapshot_complete'
            or collection[-1]['files'] != 186 or not collection[-1]['pipeline_result_downloaded']):
        raise ValueError('Historical output collection receipt incomplete')
    archived = []
    def historical(relative, expected=None):
        record = records[relative]
        if expected is not None and record['sha256'] != expected: raise ValueError('Output receipt/template fingerprint mismatch')
        source = bind(root / OLD_RUN / relative, record['sha256'], 'historical_origin')
        if source.stat().st_size != record['bytes']: raise ValueError('Historical output size mismatch')
        target = destination / 'historical-source-snapshot/run' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f: f.write(source.read_bytes())
        verify_snapshot_file(target, record['sha256'], record['bytes'])
        archived.append(dict(origin=source.relative_to(root).as_posix(),
            snapshot=target.relative_to(destination).as_posix(), sha256=record['sha256'], size_bytes=record['bytes']))
        return json.loads(source.read_text(encoding='utf-8')) if source.suffix == '.json' else source
    profile = historical('bce/profile.json')
    provenance = historical('bce/provenance.json')
    result = historical('bce/training-result.json')
    if (profile != result['profile'] or provenance['code'] != profile['code_sha256']
            or profile['objective_mode'] != 'bce' or profile['mode'] != 'rgb_zero' or profile['seed'] != 7
            or profile['initial_state_sha256'] != INITIAL_SHA or profile['plan_sha256'] != BASE_SHA
            or profile['pair_plan_sha256'] != PAIRED_SHA or profile['pair_schedule_sha256'] != EPOCHS_SHA
            or profile['requested_epochs'] != 20 or profile['requested_optimizer_updates'] != 420
            or profile['effective_batch'] != 24 or profile['microbatch'] != 8 or profile['rank_lambda'] != 0.
            or profile['smoke'] or result['completed_epochs'] != 20 or result['optimizer_steps'] != 420
            or result['status'] != 'supervised_complete_dense_evaluation_pending'):
        raise ValueError('Historical A profile/initialization/data/budget contract changed')
    historical('bce/paired-plan.json', PAIRED_SHA)
    historical('bce/paired-epochs.jsonl', EPOCHS_SHA)
    history_path = historical('bce/epochs.jsonl')
    for original, expected in profile['code_sha256'].items():
        historical('bce/source-snapshot/' + original, expected)
    index_path = bind(root / DEPLOYMENT / 'snapshot-index.json', DEPLOYMENT_INDEX_SHA, 'historical_origin')
    index = json.loads(index_path.read_text(encoding='utf-8'))
    manifest_path = bind(root / DEPLOYMENT / 'files.csv', index['manifest_sha256'], 'historical_origin')
    if len(index['sources']) != 22 or csv_rows(manifest_path) != [
            dict(path=r['path'], size_bytes=str(r['size_bytes']), sha256=r['sha256']) for r in index['sources']]:
        raise ValueError('Historical deployment source manifest disagrees with snapshot index')
    for ref in index['sources']:
        old = bind(root / DEPLOYMENT / 'source-snapshot' / ref['path'], ref['sha256'], 'historical_origin')
        if old.stat().st_size != ref['size_bytes']: raise ValueError('Historical deployment snapshot size changed')
    # Old governance is checked ONLY at immutable historical snapshots. Shared
    # data-import/pixel implementations must still match their historical code.
    for relative in ('experiments/common/video/paired_plan.py', 'experiments/common/video/supervised_plan.py',
                     'experiments/common/video/pts_clip_sampling.py'):
        bind(root / relative, profile['code_sha256'][relative], 'runtime_data')
    return profile, jsonl(history_path), archived


def _check_window(window, video, pts, std, supervised):
    indices = window['frame_indices']; times = np.asarray(window['source_pts_seconds'])
    if (len(indices) != 16 or any(type(i) is not int or i < 0 or i >= len(pts) for i in indices)
            or any(a >= b for a, b in zip(indices, indices[1:])) or len(times) != 16
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.allclose(pts[indices], times, rtol=0., atol=1e-8)
            or not 2.9 - 1e-9 <= times[-1] - times[0] <= 3.1 + 1e-9
            or np.any(std[indices] < 20.)
            or not any(indices[0] >= a and indices[-1] < b for a, b in json.loads(video['clean_runs_json']))):
        raise ValueError('Canonical window index/PTS/QC/clean-run closure failed')
    if supervised and (window['group_id'] != video['group_id'] or window['split'] != video['split']):
        raise ValueError('Supervised window role/group changed')


def freeze_order_plan(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    destination.relative_to(root / 'data/metadata/EXP-019')
    if destination.exists(): raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=False)
    started, references = time.monotonic(), {}
    def bind(path, expected=None, scope='runtime_data'):
        path = Path(path).resolve(); name = path.relative_to(root).as_posix()
        value = digest(path)
        if expected is not None and value != expected: raise ValueError('Input/source fingerprint mismatch: ' + name)
        existing = references.get((name, scope))
        ref = dict(path=name, sha256=value, size_bytes=path.stat().st_size, scope=scope)
        if existing and existing != ref: raise ValueError('Input changed within metadata generation')
        references[(name, scope)] = ref
        return path
    try:
        base_file = bind(root / BASE / 'plan.json', BASE_SHA)
        pair_file = bind(root / PAIRED / 'plan.json', PAIRED_SHA)
        bind(root / PAIRED / 'epochs.jsonl', EPOCHS_SHA)
        base = json.loads(base_file.read_text(encoding='utf-8'))
        paired = json.loads(pair_file.read_text(encoding='utf-8'))
        old_profile, old_history, archived = _historical_template(root, destination, bind)
        for name, expected in base['files'].items(): bind(root / BASE / name, expected)
        for ref in base['sources']:
            path = bind(root / ref['path'], ref['sha256'])
            if path.stat().st_size != ref['size_bytes']: raise ValueError('Frozen base input size changed')
        actual_pair, pair_rows = load_pair_plan(root, root / PAIRED)
        if actual_pair != paired: raise ValueError('Paired template changed while importing')
        resource = root / base['resource']
        sources = {v['unit_id']: v for v in csv_rows(resource / 'video-inputs.csv')}
        if len(sources) != 150 or Counter(v['split'] for v in sources.values()) != Counter(train=119, validation=31):
            raise ValueError('Original119/31 source role coverage changed')
        assignments = csv_rows(root / BASE / 'assignments.csv')
        chosen = {r['clip_id'] for r in assignments}
        bank = {w['clip_id']: w for w in jsonl(resource / 'windows.jsonl') if w['clip_id'] in chosen}
        if len(bank) != len(chosen): raise ValueError('Assigned clip absent from frozen original bank')
        original = {e: [r['clip_id'] for r in assignments if r['split'] == 'train' and int(r['epoch']) == e] for e in range(20)}
        epochs = import_epochs(pair_rows, original, bank)
        validate_history_order(old_history, epochs)
        validation = [r['clip_id'] for r in assignments if r['split'] == 'validation']
        if (validation != paired['validation_clip_ids'] or len(validation) != len(set(validation)) or len(validation) != 136
                or Counter(bank[c]['kind'] for c in validation) != Counter(pos=96, neg=40)):
            raise ValueError('Fixed136 validation identity/label/order changed')
        dense = jsonl(resource / 'dense-eval-windows.jsonl')
        if len(dense) != 32567 or {r['unit_id'] for r in dense} != {u for u,v in sources.items() if v['split'] == 'validation'}:
            raise ValueError('Original31-video dense32567 grid incomplete')
        data_by_unit = {u: [] for u in sources}
        for w in bank.values(): data_by_unit[w['unit_id']].append((w, True))
        for w in dense: data_by_unit[w['unit_id']].append((w, False))
        local_checks = []
        for uid, video in sorted(sources.items()):
            # Full local media identity check never decodes pixels. Runtime
            # inputs contain only actual dense media; training consumes JPEGs.
            scope = 'runtime_data' if video['split'] == 'validation' else 'local_media_identity'
            media = rooted(root, video['middleware_path'])
            media.relative_to(root / 'data/process/EXP-008-units-src256')
            bind(media, video['middleware_sha256'], scope)
            coordinates = csv_rows(bind(root / video['coord_path'], video['coord_sha256'], scope))
            qc = csv_rows(bind(root / video['qc_path'], video['qc_sha256'], scope))
            n = int(video['n_frames'])
            if (len(coordinates) != n or len(qc) != n
                    or [int(r['frame']) for r in coordinates] != list(range(n))
                    or [int(r['frame']) for r in qc] != list(range(n))):
                raise ValueError('Coordinate/QC actual source count/order mismatch')
            pts = np.array([float(r['pts_time']) for r in coordinates]); std = np.array([float(r['gray_std']) for r in qc])
            qt = np.array([float(r['pts_time']) for r in qc])
            if not np.isfinite(pts).all() or not np.isfinite(qt).all() or not np.isfinite(std).all() or np.max(abs(qt - pts)) > .001001:
                raise ValueError('Coordinate/QC finite/PTS mismatch')
            for window, supervised in data_by_unit[uid]:
                _check_window(window, video, pts, std, supervised)
                window['media_sha256'] = video['middleware_sha256']
                window['order_record'] = window_order_record(video['middleware_sha256'], window['frame_indices'])
            local_checks.append(dict(unit_id=uid, media_sha256=video['middleware_sha256'],
                coordinate_sha256=video['coord_sha256'], quality_sha256=video['qc_sha256'],
                metadata_frames_checked=n, scope=scope, media_pixels_decoded=False))
        completion = json.loads((root / 'data/process/EXP-016/metadata-v4/cache-completion.json').read_text(encoding='utf-8'))
        required = {(w['unit_id'], i) for w in bank.values() for i in w['frame_indices']}
        required_pts = {}
        for window in bank.values():
            for index, timestamp in zip(window['frame_indices'], window['source_pts_seconds']):
                key = window['unit_id'], index
                if key in required_pts and abs(required_pts[key] - timestamp) > 1e-8:
                    raise ValueError('Canonical bank gives contradictory PTS for one physical frame')
                required_pts[key] = timestamp
        cache_refs = []
        index_path = bind(root / completion['verified_cache_index'], completion['verified_cache_index_sha256'])
        with index_path.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                key = row['unit_id'], int(row['frame_index'])
                if key not in required: continue
                relative = PurePosixPath(row['cache_path'])
                if relative.is_absolute() or '..' in relative.parts or ':' in row['cache_path']:
                    raise ValueError('Old JPEG index path escapes project')
                timestamp = float(row['source_pts_seconds'])
                if not np.isfinite(timestamp) or abs(timestamp - required_pts[key]) > 1e-8:
                    raise ValueError('Verified-index PTS differs from canonical supervised physical frame')
                if any(re.fullmatch('[0-9a-f]{64}', row[k]) is None for k in ('jpeg_sha256', 'rgb112_sha256', 'source_bgr_sha256')):
                    raise ValueError('Invalid verified-index pixel/JPEG fingerprint')
                cache_refs.append(dict(unit_id=key[0], frame_index=key[1], source_pts_seconds=float(row['source_pts_seconds']),
                    cache_path=row['cache_path'], size_bytes=int(row['size_bytes']), jpeg_sha256=row['jpeg_sha256'],
                    rgb112_sha256=row['rgb112_sha256'], source_bgr_sha256=row['source_bgr_sha256']))
        if len(cache_refs) != len(required) or {(r['unit_id'], r['frame_index']) for r in cache_refs} != required:
            raise ValueError('Historical verified-index requirements missing or duplicated')
        current_names = ['experiments/common/video/temporal_order.py', 'experiments/common/video/order_plan.py',
            'experiments/runspecs/EXP-019/prepare.py', 'docs/process/experiments/EXP-019.md',
            'docs/process/experiments/EXP-006-数据使用方案.md', '.agents/skills/data-usage-protocol/SKILL.md']
        for name in current_names:
            path = bind(root / name, scope='current_source')
            target = destination / 'current-source-snapshot' / name; target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f: f.write(path.read_bytes())
            verify_snapshot_file(target, digest(path), path.stat().st_size)
        write_csv(destination / 'sources.csv', [sources[u] for u in sorted(sources)])
        write_jsonl(destination / 'windows.jsonl', [bank[c] for c in sorted(bank)])
        write_jsonl(destination / 'epochs.jsonl', epochs)
        write_json(destination / 'validation.json', dict(clip_ids=validation))
        write_jsonl(destination / 'dense-windows.jsonl', dense)
        write_csv(destination / 'supervised-cache-requirements.csv', sorted(cache_refs, key=lambda r: (r['unit_id'], r['frame_index'])))
        write_json(destination / 'golden-fixture.json', window_order_record('0123456789abcdef' * 4, list(range(0, 32, 2))))
        write_json(destination / 'template-a-provenance.json', dict(profile=old_profile, archived_sources=archived,
            old_data_source_sha256=BASE_SHA, old_batch_template_sha256=PAIRED_SHA, old_epoch_sha256=EPOCHS_SHA,
            historical_governance='strict historical snapshots; never substitute updated live cards/protocol',
            current_governance='separately frozen current EXP019 card and current protocol/skill'))
        write_json(destination / 'local-media-identity-checks.json', local_checks)
        physical = [w['order_record']['physical_window_sha256'] for w in list(bank.values()) + dense]
        audit = dict(status='metadata_and_media_identity_checked_pixels_unverified', supervised_clips=len(bank),
            train_count=504, train_P=356, train_N=148, validation_count=136, validation_P=96, validation_N=40,
            dense_windows=32567, train_videos=119, validation_videos=31, epochs=20, batches_per_epoch=21,
            required_unique_supervised_cache_frames=len(required), local_media_identities_checked=150,
            unique_physical_windows=len(set(physical)), shared_supervised_dense_physical_keys=len(physical) - len(set(physical)),
            cache_index_requirements_complete=True, actual_JPEG_pixel_acceptance_performed=False,
            pixels_ready=False, training_ready=False, no_video_decode=True, no_CV_bank_or_new_schedule=True)
        write_json(destination / 'audit.json', audit)
        for ref in references.values():
            path = root / ref['path']
            if path.stat().st_size != ref['size_bytes'] or digest(path) != ref['sha256']:
                raise ValueError('Input/source changed while freezing: ' + ref['path'])
        files = {p.relative_to(destination).as_posix(): digest(p) for p in sorted(destination.rglob('*')) if p.is_file()}
        plan = dict(spec_id='EXP-019', version=1, status='metadata_frozen_pixels_unverified', modes=['chronological', 'fixed_permutation'],
            permutation_version=VERSION, permutation_seed=PERMUTATION_SEED, seed=7, epochs=20,
            base_plan_path=BASE, base_plan_sha256=BASE_SHA, paired_plan_path=PAIRED, paired_plan_sha256=PAIRED_SHA,
            paired_epochs_sha256=EPOCHS_SHA, resource=base['resource'], expected_initial_state_sha256=INITIAL_SHA,
            pos_weight=148 / 356, effective_batch=24, microbatch=8, optimizer_updates=420,
            loader_contract=dict(workers=2, generator_seed=100007, one_loader=True,
                train_then_validation_each_epoch=True, extra_loader_probe=False, validation_reduction='old FP32 batch sums/count'),
            data_fingerprints=[r for r in references.values() if r['scope'] == 'runtime_data'],
            current_source_fingerprints=[r for r in references.values() if r['scope'] == 'current_source'],
            local_only_identity_fingerprints=[r for r in references.values() if r['scope'] == 'local_media_identity'],
            historical_origin_fingerprints=[r for r in references.values() if r['scope'] == 'historical_origin'],
            files=files, pixels_ready=False, cache_acceptance=None, training_ready=False,
            acceptance_rule='separate actual per-JPEG/jpegSHA/RGB112 pixel acceptance required before training; index hashes alone are not pixel acceptance',
            seconds=time.monotonic() - started)
        write_json(destination / 'plan.json', plan)
        return plan, audit
    except Exception as error:
        write_json(destination / 'failure.json', dict(status='order_plan_failed', error_type=type(error).__name__, error=str(error),
            seconds=time.monotonic() - started, pixels_ready=False, no_decode=True))
        raise


def load_order_plan(root, directory, verify_sources=True):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
    if (plan['spec_id'] != 'EXP-019' or plan['base_plan_sha256'] != BASE_SHA or plan['paired_plan_sha256'] != PAIRED_SHA
            or plan['paired_epochs_sha256'] != EPOCHS_SHA or plan['permutation_version'] != VERSION
            or plan['permutation_seed'] != 1901 or plan['pixels_ready'] or plan['cache_acceptance'] is not None):
        raise ValueError('Unregistered/promoted temporal-order metadata plan')
    if (directory / 'failure.json').exists(): raise ValueError('Failed plan cannot be consumed')
    for name, expected in plan['files'].items():
        if digest(rooted(directory, name)) != expected: raise ValueError('Frozen order-plan artifact changed: ' + name)
    if verify_sources:
        for ref in plan['data_fingerprints'] + plan['current_source_fingerprints']:
            path = rooted(root, ref['path'])
            if path.stat().st_size != ref['size_bytes'] or digest(path) != ref['sha256']:
                raise ValueError('Runtime input/current source changed: ' + ref['path'])
    windows = jsonl(directory / 'windows.jsonl'); bank = {r['clip_id']: r for r in windows}
    epochs = jsonl(directory / 'epochs.jsonl')
    validation = json.loads((directory / 'validation.json').read_text(encoding='utf-8'))['clip_ids']
    dense = jsonl(directory / 'dense-windows.jsonl')
    sources = {r['unit_id']: r for r in csv_rows(directory / 'sources.csv')}
    if len(bank) != len(windows) or len(bank) != 10180 or len(epochs) != 20 or len(validation) != 136 or len(dense) != 32567:
        raise ValueError('Order-plan complete coverage changed')
    for row in list(bank.values()) + dense:
        if row['order_record'] != window_order_record(row['media_sha256'], row['frame_indices']):
            raise ValueError('Temporal order record differs from shared pure function')
    return plan, bank, epochs, validation, dense, sources
