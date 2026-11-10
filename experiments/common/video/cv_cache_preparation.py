"""EXP-017 CPU input producer: old JPEG reuse, missing-only extraction, pixel proofs.

All sampling remains in the frozen role-free plan and existing dense_candidates.
New media decoding requires an explicit extract/dense command. Metadata readiness
is never promoted in place. Successful per-video receipts are immutable; resume
rehashes every sealed artifact and refuses an altered contract or partial result.
"""
from __future__ import annotations
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

import cv2
import numpy as np

from experiments.common.video.cv_supervised_plan import load_cv_plan
from experiments.common.video.pts_clip_sampling import dense_candidates, extract_checked_frames
from experiments.common.video.order_cache_acceptance import verify_required_rows
from experiments.common.video.supervised_plan import csv_rows, digest, jsonl, rooted, write_json

INDEX = 'data/process/EXP-016/metadata-v4/verified-cache-files.csv'
INDEX_SHA = '7e58186d125c67b52c54f6e926c3cedcb162c30ed82772da671f65160d7b018e'
OLD_REFERENCE = 'data/process/EXP-016/dense-pixel-verification-v3/result.json'
OLD_REFERENCE_SHA = '573d86e059545acac13b5d7ab198df0d307e4b2c00cd256db7324a0d54c03231'
PIXEL_VERSION = 'jpeg95-bgr-to-rgb-inter-linear-112-v1'
INDEX_FIELDS = ['cache_path','unit_id','frame_index','source_pts_seconds','physical_pts_seconds',
                'gray_std','frozen_qc_std','size_bytes','jpeg_sha256','source_bgr_sha256','rgb112_sha256']
NUMERIC_FILES = ('sources.csv','windows.jsonl','epochs.jsonl','fixed-validation.json','folds.json',
                 'segment-banks.json','frames-required.csv','audit.json')


def require(value, message):
    if not value: raise ValueError(message)


def key(row): return row['unit_id'], int(row['frame_index'])


def write_csv(path, fields, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def index_rows(rows):
    result = {}
    for row in rows:
        k = key(row); require(k not in result, 'Duplicate physical frame index')
        require(k[1] >= 0 and np.isfinite(float(row['source_pts_seconds'])), 'Invalid frame identity/PTS')
        result[k] = row
    return result


def compare_numeric_plans(old, new):
    result = {name: dict(old_sha256=digest(Path(old)/name), new_sha256=digest(Path(new)/name)) for name in NUMERIC_FILES}
    require(all(r['old_sha256'] == r['new_sha256'] for r in result.values()), 'Role-free numeric plan changed')
    return result


def split_requirements(required_rows, old_rows):
    required, old = index_rows(required_rows), index_rows(old_rows)
    reused = {}
    for k in required.keys() & old.keys():
        row = old[k]
        require(abs(float(row['source_pts_seconds'])-float(required[k]['source_pts_seconds'])) <= 1e-8,
                'Old JPEG index PTS differs from role-free plan')
        require(np.isfinite(float(row['gray_std'])) and float(row['gray_std']) >= 20
                and abs(float(row['frozen_qc_std'])-float(required[k]['frozen_qc_std'])) <= 1e-8,
                'Old indexed source QC differs from role-free requirement')
        for field in ('jpeg_sha256','rgb112_sha256','source_bgr_sha256'):
            require(re.fullmatch('[0-9a-f]{64}', row[field]) is not None, 'Invalid old cache fingerprint')
        reused[k] = row
    return required, reused, {k: required[k] for k in required.keys()-old.keys()}, old


def context(root, plan_dir):
    root, plan_dir = Path(root).resolve(), Path(plan_dir).resolve()
    plan, bank, epochs, fixed, folds = load_cv_plan(root, plan_dir)
    required_rows = csv_rows(plan_dir/'frames-required.csv')
    require(digest(root/INDEX) == INDEX_SHA, 'Original verified index changed')
    required, reused, missing, old = split_requirements(required_rows, csv_rows(root/INDEX))
    referenced = {cid for epoch in epochs for cid in epoch['clip_ids']} | set(fixed['clip_ids'])
    physical = {(bank[cid]['unit_id'], int(i)) for cid in referenced for i in bank[cid]['frame_indices']}
    require(set(required) == physical and len(required) == 191812 and len(reused) == 149964 and len(missing) == 41848,
            'Frozen191812 requirements /149964 reuse /41848 missing changed')
    sources = {r['unit_id']: r for r in csv_rows(plan_dir/'sources.csv')}
    require(len(sources) == 150 and len(folds) == 5, 'Original150-video/five-fold coverage changed')
    for row in reused.values(): require(rooted(root, row['cache_path']).is_file(), 'Indexed old JPEG is missing; do not silently reclassify')
    return dict(root=root, plan_dir=plan_dir, plan=plan, sources=sources, required=required,
                reused=reused, missing=missing, old=old, plan_sha256=digest(plan_dir/'plan.json'))


def source_paths(root):
    names = ('cv_cache_preparation.py','cv_supervised_plan.py','pts_clip_sampling.py',
             'supervised_plan.py','order_cache_acceptance.py','dense_evaluation.py','supervised_runtime.py')
    return [Path(root)/'experiments/common/video'/n for n in names]+[
        Path(root)/'experiments/runspecs/EXP-017'/n for n in ('prepare_inputs.py','verify_cache.py')]


def contract(c, kind, ffmpeg=None, **extra):
    result = dict(spec_id='EXP-017', kind=kind, format_version=1,
        plan_path=c['plan_dir'].relative_to(c['root']).as_posix(), plan_sha256=c['plan_sha256'],
        original_cache_index=INDEX, original_cache_index_sha256=INDEX_SHA,
        required_frame_count=len(c['required']), reused_frame_count=len(c['reused']), new_frame_count=len(c['missing']),
        source_sha256={p.relative_to(c['root']).as_posix(): digest(p) for p in source_paths(c['root'])},
        pixel_version=PIXEL_VERSION, opencv=cv2.__version__, numpy=np.__version__, no_training=True, **extra)
    if ffmpeg is not None:
        import subprocess
        ffmpeg = Path(ffmpeg).resolve()
        result.update(ffmpeg_path=str(ffmpeg), ffmpeg_sha256=digest(ffmpeg), ffmpeg_version=subprocess.run(
            [str(ffmpeg), '-version'], capture_output=True, text=True, check=True, timeout=15).stdout.splitlines()[0])
    return result


def guard(c, frozen):
    require(frozen['spec_id'] == 'EXP-017' and frozen['original_cache_index_sha256'] == INDEX_SHA
            and frozen['required_frame_count'] == len(c['required']) and frozen['reused_frame_count'] == len(c['reused'])
            and frozen['new_frame_count'] == len(c['missing']) and frozen['pixel_version'] == PIXEL_VERSION,
            'Producer contract scope changed')
    require(set(frozen['source_sha256']) == {p.relative_to(c['root']).as_posix() for p in source_paths(c['root'])},
            'Incomplete producer/source closure')
    require(digest(c['plan_dir']/'plan.json') == frozen['plan_sha256'] and digest(c['root']/INDEX) == INDEX_SHA, 'Plan/index changed')
    for name, value in frozen['source_sha256'].items(): require(digest(c['root']/name) == value, 'Producer/helper changed: '+name)
    for name, value in frozen.get('input_sha256', {}).items(): require(digest(rooted(c['root'], name)) == value, 'Producer dependency changed: '+name)
    for ref in c['plan']['source_fingerprints']:
        require(digest(c['root']/ref['path']) == ref['sha256'], 'Plan source/governance changed: '+ref['path'])
    if 'ffmpeg_path' in frozen: require(digest(frozen['ffmpeg_path']) == frozen['ffmpeg_sha256'], 'Decoder changed')


def open_output(c, out, frozen, resume):
    out = Path(out).resolve()
    require(any(parent in out.parents for parent in (c['root']/'data/process/EXP-017', c['root']/'data/metadata/EXP-017')),
            'New input output must remain inside EXP-017 process/metadata')
    if out.exists():
        require(resume, 'Existing output requires explicit --resume')
        require((out/'contract.json').is_file() and json.loads((out/'contract.json').read_text('utf-8')) == frozen,
                'Resume contract differs or initial contract is incomplete')
    else:
        out.mkdir(parents=True, exist_ok=False); write_json(out/'contract.json', frozen)
        for name, expected in frozen['source_sha256'].items():
            target = out/'source-snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f: f.write((c['root']/name).read_bytes())
            require(digest(target) == expected, 'Source snapshot readback differs')
    for name, expected in frozen['source_sha256'].items():
        require(digest(out/'source-snapshot'/name) == expected, 'Archived producer source changed')
    return out


def seal(root, receipt, contract_sha, uid, payload, paths):
    files = [dict(path=Path(p).resolve().relative_to(root).as_posix(), sha256=digest(p), size_bytes=Path(p).stat().st_size)
             for p in sorted(set(map(Path, paths)))]
    value = dict(status='complete', contract_sha256=contract_sha, unit_id=uid, payload=payload, files=files)
    write_json(receipt, value)
    return value


def read_seal(root, receipt, contract_sha, uid):
    if not Path(receipt).is_file(): return None
    value = json.loads(Path(receipt).read_text('utf-8'))
    require(value['status'] == 'complete' and value['contract_sha256'] == contract_sha and value['unit_id'] == uid,
            'Per-video receipt identity/contract differs')
    for row in value['files']:
        path = rooted(root, row['path'])
        require(path.is_file() and path.stat().st_size == row['size_bytes'] and digest(path) == row['sha256'],
                'Sealed per-video output changed: '+row['path'])
    return value


def attempt(out, stage):
    path = out/'attempts'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    write_json(path/'launch.json', dict(stage=stage, argv=sys.argv, started_utc=datetime.now(timezone.utc).isoformat()))
    return path


def record_cli_failure(root, out, stage, error):
    """Preserve pre-contract/initialization failures without replacing receipts."""
    root, out = Path(root).resolve(), Path(out).resolve()
    require(any(parent in out.parents for parent in (root/'data/process/EXP-017', root/'data/metadata/EXP-017')),
            'Failure evidence outside EXP-017 output is refused')
    out.mkdir(parents=True, exist_ok=True)
    run = attempt(out, stage)
    write_json(run/'failure.json', dict(status='failed_no_new_readiness', error_type=type(error).__name__,
        error=str(error), argv=sys.argv, source_sha256=digest(__file__)))


def coordinates(c, uid, verify_media=False):
    source = c['sources'][uid]; root = c['root']
    for field in ('coord','qc'):
        require(digest(root/source[field+'_path']) == source[field+'_sha256'], 'Coordinate/QC source changed')
    if verify_media: require(digest(root/source['middleware_path']) == source['middleware_sha256'], 'Actual middleware SHA differs')
    coord, qc = csv_rows(root/source['coord_path']), csv_rows(root/source['qc_path'])
    n = int(source['n_frames'])
    require(len(coord) == len(qc) == n and [int(r['frame']) for r in coord] == list(range(n))
            and [int(r['frame']) for r in qc] == list(range(n)), 'Coordinate/QC count/order differs')
    pts = np.array([float(r['pts_time']) for r in coord]); qp = np.array([float(r['pts_time']) for r in qc])
    std = np.array([float(r['gray_std']) for r in qc])
    require(np.isfinite(pts).all() and np.isfinite(qp).all() and np.isfinite(std).all()
            and np.max(abs(pts-qp)) <= .001001, 'Coordinate/QC finite/alignment differs')
    return pts, std


def extract_missing(root, plan_dir, out, ffmpeg, resume=False):
    c = context(root, plan_dir); frozen = contract(c, 'missing_supervised_JPEG95', ffmpeg)
    out = open_output(c, out, frozen, resume); contract_sha = digest(out/'contract.json')
    run = attempt(out, 'extract'); began = time.perf_counter()
    try:
        merged = dict(c['reused']); receipts = {}
        for uid in sorted(c['sources']):
            unit = out/'units'/uid; unit.mkdir(parents=True, exist_ok=True); receipt = unit/'complete.json'
            done = read_seal(c['root'], receipt, contract_sha, uid)
            wanted = sorted(i for u, i in c['missing'] if u == uid)
            if done is not None and wanted:
                require(digest(c['root']/c['sources'][uid]['middleware_path']) == c['sources'][uid]['middleware_sha256'], 'Completed video source identity changed')
            if done is None:
                payload = dict(selected_frames=0, new_index_rows=[]); artifacts = []
                if wanted:
                    pts, std = coordinates(c, uid, verify_media=True)
                    for i in wanted:
                        r = c['required'][uid, i]
                        require(abs(float(r['source_pts_seconds'])-pts[i]) <= 1e-8
                                and abs(float(r['frozen_qc_std'])-std[i]) <= 1e-8, 'Frozen missing requirement differs')
                    evidence = unit/('decode-'+uuid.uuid4().hex[:12]); frame_dir = out/'frames'/uid
                    summary = extract_checked_frames(c['root']/c['sources'][uid]['middleware_path'], pts, std,
                                                     wanted, frame_dir, evidence, ffmpeg, 20.)
                    require(summary['status'] == 'passed' and summary['selected_frames'] == len(wanted), 'Incomplete new-frame decode')
                    records = csv_rows(evidence/'frames-written.csv')
                    require([int(r['frame_index']) for r in records] == wanted, 'Extracted index order/coverage differs')
                    rows = []
                    for r in records:
                        i = int(r['frame_index']); path = frame_dir/('frame-%08d.jpg'%i)
                        rows.append(dict(cache_path=path.relative_to(c['root']).as_posix(), unit_id=uid, frame_index=i,
                            source_pts_seconds=r['source_pts_seconds'], physical_pts_seconds=r['physical_pts_seconds'],
                            gray_std=r['gray_std'], frozen_qc_std=r['frozen_qc_std'], size_bytes=r['jpeg_size_bytes'],
                            jpeg_sha256=r['jpeg_sha256'], source_bgr_sha256=r['source_bgr_sha256'], rgb112_sha256=r['rgb112_sha256']))
                    new_index = evidence/'new-index.csv'; write_csv(new_index, INDEX_FIELDS, rows)
                    require(digest(c['root']/c['sources'][uid]['middleware_path']) == c['sources'][uid]['middleware_sha256'], 'Middleware changed during extraction')
                    payload = dict(selected_frames=len(rows), new_index_path=new_index.relative_to(c['root']).as_posix(),
                                   new_index_sha256=digest(new_index), source_media_sha256=c['sources'][uid]['middleware_sha256'])
                    artifacts = [new_index]+[p for p in evidence.rglob('*') if p.is_file()]+[frame_dir/('frame-%08d.jpg'%i) for i in wanted]
                guard(c, frozen); done = seal(c['root'], receipt, contract_sha, uid, payload, artifacts)
            payload = done['payload']; require(payload['selected_frames'] == len(wanted), 'Resume video quota differs')
            if wanted:
                rows = index_rows(csv_rows(c['root']/payload['new_index_path']))
                require(set(rows) == {(uid, i) for i in wanted} and not (set(rows) & set(merged)), 'New index changed/overlaps old')
                merged.update(rows)
            receipts[uid] = digest(receipt)
            print(json.dumps(dict(stage='supervised_JPEG_increment', videos=len(receipts), total=150, unit_id=uid,
                                  selected_new=len(wanted), seconds=time.perf_counter()-began)), flush=True)
        require(set(merged) == set(c['required']), 'Merged index coverage not complete')
        path = out/'verified-cache-files.csv'
        if path.exists():
            require(index_rows(csv_rows(path)) == {k: {n: str(v) for n,v in row.items()} for k,row in merged.items()}, 'Existing final index differs')
        else: write_csv(path, INDEX_FIELDS, [{n: row[n] for n in INDEX_FIELDS} for k,row in sorted(merged.items())])
        guard(c, frozen)
        result = dict(status='cv_supervised_JPEGs_produced_pixels_acceptance_pending', plan_sha256=c['plan_sha256'],
            contract_sha256=contract_sha, required_frame_count=len(merged), reused_frame_count=len(c['reused']), new_frame_count=len(c['missing']),
            verified_cache_index=path.relative_to(c['root']).as_posix(), verified_cache_index_sha256=digest(path),
            completed_video_receipts=receipts, seconds=time.perf_counter()-began, no_training=True)
        if (out/'result.json').exists(): require(json.loads((out/'result.json').read_text('utf-8')) == dict(result, seconds=json.loads((out/'result.json').read_text('utf-8'))['seconds']), 'Existing producer result differs')
        else: write_json(out/'result.json', result)
        return result
    except Exception as error:
        write_json(run/'failure.json', dict(status='partial_not_cache_ready', error_type=type(error).__name__, error=str(error),
            seconds=time.perf_counter()-began, no_window_replacement=True)); raise


def checked_producer(c, directory):
    directory = Path(directory).resolve(); result = json.loads((directory/'result.json').read_text('utf-8'))
    frozen = json.loads((directory/'contract.json').read_text('utf-8')); guard(c, frozen)
    require(result['status'] == 'cv_supervised_JPEGs_produced_pixels_acceptance_pending' and result['plan_sha256'] == c['plan_sha256']
            and result['contract_sha256'] == digest(directory/'contract.json') and set(result['completed_video_receipts']) == set(c['sources']), 'Producer incomplete/bound to another plan')
    for uid, expected in result['completed_video_receipts'].items():
        receipt = directory/'units'/uid/'complete.json'; require(digest(receipt) == expected, 'Video completion receipt changed')
        read_seal(c['root'], receipt, result['contract_sha256'], uid)
    path = rooted(c['root'], result['verified_cache_index']); require(digest(path) == result['verified_cache_index_sha256'], 'Merged index changed')
    merged = index_rows(csv_rows(path)); require(set(merged) == set(c['required']), 'Merged index does not cover requirements')
    for k, row in merged.items():
        require(abs(float(row['source_pts_seconds'])-float(c['required'][k]['source_pts_seconds'])) <= 1e-8, 'Merged PTS differs')
        if k in c['reused']: require(row == c['reused'][k], 'An old indexed JPEG was replaced')
    return result, path, merged


def accept_cache(root, plan_dir, producer, out, workers=4):
    c = context(root, plan_dir); produced, path, merged = checked_producer(c, producer)
    producer_name = (Path(producer).resolve()/'result.json').relative_to(c['root']).as_posix()
    frozen = contract(c, 'actual_all_supervised_JPEG_acceptance', producer_result=producer_name,
        producer_result_sha256=digest(Path(producer)/'result.json'), verified_cache_index=produced['verified_cache_index'],
        verified_cache_index_sha256=digest(path), input_sha256={
            producer_name: digest(Path(producer)/'result.json'),
            path.relative_to(c['root']).as_posix(): digest(path)})
    out = open_output(c, out, frozen, False); began = time.perf_counter(); count = byte_count = 0; failure = None
    requirements = [dict(unit_id=u, frame_index=i, source_pts_seconds=float(r['source_pts_seconds']), cache_path=r['cache_path'],
                         size_bytes=int(r['size_bytes']), jpeg_sha256=r['jpeg_sha256'], rgb112_sha256=r['rgb112_sha256']) for (u,i),r in sorted(merged.items())]
    verified = out/'verified-required-frames.csv'
    try:
        cv2.setNumThreads(1)
        totals = verify_required_rows(c['root'], requirements, verified, out/'failed-required-frames.jsonl', workers,
            progress=lambda n,b: print(json.dumps(dict(stage='all191812_actual_JPEG_acceptance', verified=n, total=len(requirements),
                                                       bytes_read=b, seconds=time.perf_counter()-began)), flush=True))
        count, byte_count = totals['verified'], totals['bytes_read']; require(count == len(requirements) and totals['failed'] == 0, 'Incomplete all-frame acceptance')
        guard(c, frozen); require(digest(path) == frozen['verified_cache_index_sha256'] and digest(Path(producer)/'result.json') == frozen['producer_result_sha256'], 'Producer changed during acceptance')
    except Exception as error:
        failure = dict(error_type=type(error).__name__, error=str(error))
        if verified.is_file():
            partial = csv_rows(verified); count = len(partial); byte_count = sum(int(r['size_bytes']) for r in partial)
    proof = dict(status='cv_supervised_cache_ready' if failure is None else 'cv_supervised_cache_failed', spec_id='EXP-017',
        plan_path=frozen['plan_path'], plan_sha256=c['plan_sha256'], verified_cache_index=produced['verified_cache_index'],
        verified_cache_index_sha256=digest(path), producer_result=frozen['producer_result'], producer_result_sha256=frozen['producer_result_sha256'],
        required_frame_count=len(requirements), verified_required_frame_count=count,
        failed_required_frames=1 if failure else 0, unverified_required_frames=max(0,len(requirements)-count-(1 if failure else 0)),
        verified_required_frames_index=verified.relative_to(c['root']).as_posix(), verified_required_frames_index_sha256=digest(verified) if verified.is_file() else None,
        actual_JPEG_byte_SHA_checked=failure is None, actual_RGB112_pixel_SHA_checked=failure is None,
        actual_verified_JPEG_reads=count,
        pixel_version=PIXEL_VERSION, bytes_read=byte_count, seconds=time.perf_counter()-began,
        source_sha256=frozen['source_sha256'], metadata_flags_unchanged=True, no_media_decode=True, no_training=True, failure=failure)
    write_json(out/'acceptance.json', proof)
    if failure: raise RuntimeError('All-frame pixel acceptance failed: '+str(out/'acceptance.json'))
    return proof


def dense_grid(c):
    grouped, total = {}, 0
    for uid in sorted(c['sources']):
        pts, _ = coordinates(c, uid)
        grid, skipped = dense_candidates(pts, json.loads(c['sources'][uid]['clean_runs_json']))
        grouped[uid] = [dict(unit_id=uid, target_start_seconds=start, frame_indices=ix.tolist(),
                            source_pts_seconds=pts[ix].tolist()) for start,ix in grid]
        require(bool(grid), 'Video lost all label-independent dense windows')
        total += len(grid)
    require(total == 143470, 'Original150-video143470 dense grid changed')
    return grouped


def consume_dense(batches, info, windows):
    expected = {tuple(w['frame_indices']): w for w in windows}; seen, frames = set(), {}
    require(len(expected) == len(windows), 'Duplicate planned dense window')
    for u8, metadata in batches:
        require(u8.dtype == np.uint8 and u8.shape == (len(metadata),16,112,112,3), 'Invalid canonical dense RGB112 batch')
        for pixels, row in zip(u8, metadata):
            identity = tuple(row['frame_indices'])
            require(identity in expected and identity not in seen and row == expected[identity], 'Dense physical window/PTS changed or repeated')
            seen.add(identity)
            for index, pts, rgb in zip(row['frame_indices'], row['source_pts_seconds'], pixels):
                value = (float(pts), hashlib.sha256(rgb.tobytes()).hexdigest())
                if index in frames: require(frames[index] == value, 'One physical frame produced different RGB pixels/PTS')
                frames[index] = value
    require(seen == set(expected) and info.get('windows') == len(windows)
            and isinstance(info.get('rgb112_digest'), str) and re.fullmatch('[0-9a-f]{64}', info['rgb112_digest'])
            and info.get('minimum_selected_gray_std', -1) >= 20, 'Incomplete canonical dense pixel closure')
    return frames


def prepare_dense(root, plan_dir, producer, acceptance_path, out, ffmpeg, resume=False):
    from experiments.common.video.dense_evaluation import dense_batches, compatible_pixel_reference
    c = context(root, plan_dir); produced, index_path, merged = checked_producer(c, producer)
    acceptance_path = Path(acceptance_path).resolve(); accepted = json.loads(acceptance_path.read_text('utf-8'))
    require(accepted['status'] == 'cv_supervised_cache_ready' and accepted['plan_sha256'] == c['plan_sha256']
            and accepted['required_frame_count'] == accepted['verified_required_frame_count'] == len(c['required'])
            and accepted['failed_required_frames'] == accepted['unverified_required_frames'] == 0
            and accepted['actual_JPEG_byte_SHA_checked'] is True and accepted['actual_RGB112_pixel_SHA_checked'] is True
            and accepted['verified_cache_index_sha256'] == digest(index_path)
            and accepted['producer_result_sha256'] == digest(Path(producer)/'result.json'), 'Full supervised acceptance is required before dense preparation')
    require(digest(c['root']/accepted['verified_required_frames_index']) == accepted['verified_required_frames_index_sha256'], 'Actual acceptance frame ledger changed')
    require(digest(c['root']/OLD_REFERENCE) == OLD_REFERENCE_SHA, 'Original31 label-free pixel reference changed')
    reference = json.loads((c['root']/OLD_REFERENCE).read_text('utf-8'))
    base_path = c['root']/'data/metadata/EXP-016/training-plan-v2/plan.json'
    require(digest(base_path) == 'bb3af415133e1fe40a7a891bc80e2b22b0e3d44b97f2dec43ac0d6ca5750bb3f', 'Original pixel-reference-compatible base changed')
    base = json.loads(base_path.read_text('utf-8')); origin = c['root']/Path(OLD_REFERENCE).parent/'source-plan.json'
    compatible_pixel_reference(base, digest(base_path), reference, origin)
    require(set(reference['videos']) == {u for u,s in c['sources'].items() if s['legacy_split'] == 'validation'}, 'Original31 reference identities differ')
    grouped = dense_grid(c)
    old_grid = defaultdict(list)
    for row in jsonl(c['root']/c['plan']['metadata_path']/'dense-eval-windows.jsonl'): old_grid[row['unit_id']].append(row)
    for uid in reference['videos']:
        require(grouped[uid] == old_grid[uid], 'Original31 physical window grid differs; do not reuse its pixel digest')
    frozen = contract(c, 'canonical150_dense_pixel_reference', ffmpeg,
        acceptance_path=acceptance_path.relative_to(c['root']).as_posix(), acceptance_sha256=digest(acceptance_path),
        merged_index_path=index_path.relative_to(c['root']).as_posix(), merged_index_sha256=digest(index_path),
        original31_reference_path=OLD_REFERENCE, original31_reference_sha256=OLD_REFERENCE_SHA,
        original31_source_plan_sha256=digest(origin), base_plan_sha256=digest(base_path), windows=143470,
        input_sha256={acceptance_path.relative_to(c['root']).as_posix(): digest(acceptance_path),
            accepted['verified_required_frames_index']: accepted['verified_required_frames_index_sha256'],
            index_path.relative_to(c['root']).as_posix(): digest(index_path), OLD_REFERENCE: OLD_REFERENCE_SHA,
            origin.relative_to(c['root']).as_posix(): digest(origin), base_path.relative_to(c['root']).as_posix(): digest(base_path)})
    out = open_output(c, out, frozen, resume); run = attempt(out, 'dense'); began = time.perf_counter()
    contract_sha = digest(out/'contract.json'); grid_path = out/'dense-windows.jsonl'
    if not grid_path.exists():
        with grid_path.open('x', encoding='utf-8') as f:
            for uid in sorted(grouped):
                for row in grouped[uid]: f.write(json.dumps(row, allow_nan=False)+'\n')
    else: require(jsonl(grid_path) == [w for uid in sorted(grouped) for w in grouped[uid]], 'Existing canonical grid changed')
    known = defaultdict(dict)
    for (u,i), row in merged.items(): known[u][i] = row
    videos, receipts = {}, {}
    try:
        for uid in sorted(c['sources']):
            unit = out/'units'/uid; unit.mkdir(parents=True, exist_ok=True); receipt = unit/'complete.json'
            done = read_seal(c['root'], receipt, contract_sha, uid)
            if done is not None:
                require(digest(c['root']/c['sources'][uid]['middleware_path']) == c['sources'][uid]['middleware_sha256'], 'Completed dense video source changed')
            if done is None:
                pts, _ = coordinates(c, uid, verify_media=True); source = c['sources'][uid]; windows = grouped[uid]
                if uid in reference['videos']:
                    info = dict(reference['videos'][uid]); require(info['windows'] == len(windows), 'Reused31 dense coverage differs')
                    info.update(reference_origin='reused_original31_label_free_pixel_reference', reference_sha256=OLD_REFERENCE_SHA)
                    artifacts = []
                else:
                    evidence = unit/('decode-'+uuid.uuid4().hex[:12]); evidence.mkdir(parents=True, exist_ok=False)
                    info = {}; stream = dense_batches(c['root']/source['middleware_path'], pts, windows, ffmpeg, info,
                        known_pixels=known[uid], batch_size=8, filter_path=evidence/'select-filter.txt')
                    try: frames = consume_dense(stream, info, windows)
                    finally: stream.close()
                    require(info['decoded'] == len(pts) and info['max_pts_error'] <= .001001, 'Full physical frame/PTS closure failed')
                    hashes = evidence/'rgb112-frame-hashes.csv'
                    write_csv(hashes, ['unit_id','frame_index','source_pts_seconds','rgb112_sha256'],
                              [dict(unit_id=uid, frame_index=i, source_pts_seconds=t, rgb112_sha256=h) for i,(t,h) in sorted(frames.items())])
                    info.update(reference_origin='new_local119_full_physical_decode', unique_selected_frames=len(frames),
                        frame_hash_index=hashes.relative_to(c['root']).as_posix(), frame_hash_index_sha256=digest(hashes))
                    artifacts = [p for p in evidence.rglob('*') if p.is_file()]
                require(digest(c['root']/source['middleware_path']) == source['middleware_sha256'], 'Dense source changed during preparation')
                info.update(source_media_sha256=source['middleware_sha256'], coord_sha256=source['coord_sha256'], qc_sha256=source['qc_sha256'],
                            source_media_path=source['middleware_path'], source_media_size_bytes=(c['root']/source['middleware_path']).stat().st_size,
                            canonical_reference_matched=True, labels_used_for_sampling=False)
                video_path = unit/('video-'+uuid.uuid4().hex[:12]+'.json'); write_json(video_path, info); artifacts.append(video_path)
                guard(c, frozen); done = seal(c['root'], receipt, contract_sha, uid, info, artifacts)
            videos[uid] = done['payload']; receipts[uid] = digest(receipt)
            print(json.dumps(dict(stage='canonical_dense_reference', videos=len(videos), total=150, unit_id=uid,
                windows=videos[uid]['windows'], origin=videos[uid]['reference_origin'], seconds=time.perf_counter()-began)), flush=True)
        guard(c, frozen)
        require(digest(acceptance_path) == frozen['acceptance_sha256'] and digest(index_path) == frozen['merged_index_sha256'], 'Supervised evidence changed')
        result = dict(status='cv_dense_pixels_verified', spec_id='EXP-017', plan_sha256=c['plan_sha256'], videos=videos,
            grid_path=grid_path.relative_to(c['root']).as_posix(), grid_sha256=digest(grid_path), windows=143470, video_count=150,
            media_refs=[dict(unit_id=u, path=v['source_media_path'], sha256=v['source_media_sha256'], size_bytes=v['source_media_size_bytes']) for u,v in sorted(videos.items())],
            metadata_refs=[r for r in c['plan']['source_fingerprints'] if r['path'] in {s[f+'_path'] for s in c['sources'].values() for f in ('coord','qc')}],
            preprocessing=dict(pixel_version=PIXEL_VERSION, helper_path='experiments/common/video/pts_clip_sampling.py',
                helper_sha256=frozen['source_sha256']['experiments/common/video/pts_clip_sampling.py']),
            original31_reference_path=OLD_REFERENCE, original31_reference_sha256=OLD_REFERENCE_SHA,
            new119_full_decode_count=sum(v['reference_origin']=='new_local119_full_physical_decode' for v in videos.values()),
            completed_video_receipts=receipts, contract_sha256=contract_sha, seconds=time.perf_counter()-began,
            no_saved_dense_pixels=True, no_training=True, independent_test_used=False)
        require(len(videos) == 150 and result['new119_full_decode_count'] == 119 and sum(v['windows'] for v in videos.values()) == 143470, 'Full dense scope not closed')
        if (out/'result.json').exists():
            prior = json.loads((out/'result.json').read_text('utf-8')); require(prior == dict(result, seconds=prior['seconds']), 'Existing dense result differs')
        else: write_json(out/'result.json', result)
        return result
    except Exception as error:
        write_json(run/'failure.json', dict(status='partial_dense_reference_withheld', completed_videos=len(videos), error_type=type(error).__name__, error=str(error)))
        raise
