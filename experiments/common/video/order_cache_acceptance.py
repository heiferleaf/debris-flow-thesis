"""Actual existing-JPEG acceptance for EXP-019, independent of metadata readiness."""
from __future__ import annotations
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import time

import cv2
import numpy as np

from experiments.common.video.order_plan import load_order_plan
from experiments.common.video.pts_clip_sampling import rgb112_from_jpeg
from experiments.common.video.supervised_plan import csv_rows, digest, rooted, write_json

INDEX = 'data/process/EXP-016/metadata-v4/verified-cache-files.csv'
PIXEL_VERSION = 'jpeg95-bgr-to-rgb-inter-linear-112-v1'
CSV_FIELDS = ['unit_id', 'frame_index', 'source_pts_seconds', 'cache_path', 'size_bytes',
              'jpeg_sha256', 'actual_jpeg_sha256', 'rgb112_sha256', 'actual_rgb112_sha256',
              'rgb112_shape', 'rgb112_dtype', 'status']


def required_physical_frames(bank):
    required = {}
    for window in bank.values():
        indices, times = window['frame_indices'], window['source_pts_seconds']
        if (len(indices) != 16 or len(times) != 16 or any(type(i) is not int or i < 0 for i in indices)
                or any(a >= b for a, b in zip(indices, indices[1:]))
                or not np.isfinite(times).all() or any(a >= b for a, b in zip(times, times[1:]))):
            raise ValueError('Canonical supervised window has invalid index/PTS shape')
        for index, timestamp in zip(indices, times):
            key = window['unit_id'], index
            if key in required and abs(required[key] - timestamp) > 1e-8:
                raise ValueError('Conflicting PTS for one physical frame')
            required[key] = float(timestamp)
    if not required: raise ValueError('Empty supervised frame requirements')
    return required


def checked_requirements(root, plan_dir, plan, bank):
    """Cross-check plan requirements against the ORIGINAL frozen cache index."""
    root, plan_dir = Path(root).resolve(), Path(plan_dir).resolve()
    wanted = required_physical_frames(bank)
    fingerprints = {r['path']: r['sha256'] for r in plan['data_fingerprints']}
    expected = fingerprints.get(INDEX)
    if expected is None or digest(rooted(root, INDEX)) != expected:
        raise ValueError('Original verified-cache index not strictly bound to order plan')
    proposed = csv_rows(plan_dir / 'supervised-cache-requirements.csv')
    expected_file_sha = plan['files']['supervised-cache-requirements.csv']
    if digest(plan_dir / 'supervised-cache-requirements.csv') != expected_file_sha:
        raise ValueError('Order-plan cache requirements artifact changed')
    by_key = {}
    with rooted(root, INDEX).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            key = row['unit_id'], int(row['frame_index'])
            if key not in wanted: continue
            if key in by_key: raise ValueError('Duplicate required physical frame in original index')
            by_key[key] = row
    if set(by_key) != set(wanted) or len(proposed) != len(wanted):
        raise ValueError('Original index/proposed requirements do not cover canonical bank')
    result, seen = [], set()
    for row in proposed:
        key = row['unit_id'], int(row['frame_index'])
        if key not in wanted or key in seen: raise ValueError('Duplicate/unknown proposed physical requirement')
        seen.add(key); old = by_key[key]
        for name in ('cache_path', 'jpeg_sha256', 'rgb112_sha256', 'source_bgr_sha256'):
            if row[name] != old[name]: raise ValueError('Plan requirements differ from original cache identity')
        if int(row['size_bytes']) != int(old['size_bytes']) or int(row['size_bytes']) <= 0:
            raise ValueError('Expected JPEG byte count differs from original index')
        if (not np.isfinite(float(row['source_pts_seconds']))
                or abs(float(row['source_pts_seconds']) - wanted[key]) > 1e-8
                or abs(float(old['source_pts_seconds']) - wanted[key]) > 1e-8):
            raise ValueError('Cache-index PTS differs from canonical bank')
        relative = PurePosixPath(row['cache_path'])
        if relative.is_absolute() or '..' in relative.parts or ':' in row['cache_path']:
            raise ValueError('JPEG cache path must remain project-relative')
        rooted(root, row['cache_path'])
        if any(re.fullmatch('[0-9a-f]{64}', row[name]) is None for name in ('jpeg_sha256', 'rgb112_sha256')):
            raise ValueError('Invalid frozen JPEG/RGB112 fingerprint')
        result.append(dict(unit_id=key[0], frame_index=key[1], source_pts_seconds=wanted[key],
                           cache_path=row['cache_path'], size_bytes=int(row['size_bytes']),
                           jpeg_sha256=row['jpeg_sha256'], rgb112_sha256=row['rgb112_sha256']))
    if seen != set(wanted): raise ValueError('Required physical frame coverage incomplete')
    return sorted(result, key=lambda r: (r['unit_id'], r['frame_index'])), expected


def verify_jpeg(root, requirement):
    """One actual read; JPEG bytes and actual frozen-helper RGB112 pixels."""
    path = rooted(Path(root).resolve(), requirement['cache_path'])
    before = path.stat()
    if before.st_size != requirement['size_bytes']: raise ValueError('Actual JPEG byte count mismatch')
    encoded = path.read_bytes(); after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('JPEG changed while reading')
    actual_jpeg = hashlib.sha256(encoded).hexdigest()
    if len(encoded) != requirement['size_bytes'] or actual_jpeg != requirement['jpeg_sha256']:
        raise ValueError('Actual JPEG SHA256 mismatch')
    rgb = rgb112_from_jpeg(encoded)  # Also requires the decoded JPEG to be256x256x3.
    if rgb.shape != (112, 112, 3) or rgb.dtype != np.uint8 or not np.isfinite(rgb).all():
        raise ValueError('Actual decoded RGB112 has invalid shape/dtype/finite pixels')
    actual_rgb = hashlib.sha256(rgb.tobytes()).hexdigest()
    if actual_rgb != requirement['rgb112_sha256']: raise ValueError('Actual RGB112 SHA256 mismatch')
    return dict(requirement, actual_jpeg_sha256=actual_jpeg, actual_rgb112_sha256=actual_rgb,
                rgb112_shape='112x112x3', rgb112_dtype='uint8', status='verified')


def verify_required_rows(root, requirements, output_csv, failure_jsonl, workers=4, progress=None):
    """Bounded in-flight CPU reads; deterministic CSV order, immediate failure."""
    if type(workers) is not int or not 1 <= workers <= 8: raise ValueError('Cache workers must be1..8')
    output_csv, failure_jsonl = Path(output_csv), Path(failure_jsonl)
    iterator, pending, verified, failed, byte_count = iter(requirements), deque(), 0, 0, 0
    executor = ThreadPoolExecutor(max_workers=workers)
    def submit():
        try: row = next(iterator)
        except StopIteration: return False
        pending.append((row, executor.submit(verify_jpeg, root, row))); return True
    try:
        for _ in range(workers * 2):
            if not submit(): break
        with output_csv.open('x', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS); writer.writeheader()
            while pending:
                requirement, future = pending.popleft()
                try: result = future.result()
                except Exception as error:
                    failed += 1
                    with failure_jsonl.open('x', encoding='utf-8') as failure:
                        failure.write(json.dumps(dict(requirement=requirement, error_type=type(error).__name__,
                            error=str(error), verified_before_failure=verified), ensure_ascii=False, allow_nan=False) + '\n')
                    raise
                writer.writerow(result); verified += 1; byte_count += result['size_bytes']
                if progress and (verified % 5000 == 0 or verified == len(requirements)):
                    f.flush(); progress(verified, byte_count)
                submit()
        return dict(verified=verified, failed=failed, bytes_read=byte_count)
    finally:
        for _, future in pending: future.cancel()
        executor.shutdown(wait=True)


def accept_order_cache(root, plan_dir, out, workers=4):
    root, plan_dir, out = (Path(p).resolve() for p in (root, plan_dir, out))
    out.relative_to(root / 'data/metadata/EXP-019')
    if out.exists(): raise FileExistsError('Do not overwrite an earlier cache acceptance')
    out.mkdir(parents=True, exist_ok=False)
    began, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    required_count, index_sha, verified, byte_count, failure = 0, None, 0, 0, None
    plan_sha = digest(plan_dir / 'plan.json') if (plan_dir / 'plan.json').is_file() else None
    code = {}
    index_csv = out / 'verified-required-frames.csv'
    failure_rows = out / 'failed-required-frames.jsonl'
    try:
        plan, bank, _, _, _, _ = load_order_plan(root, plan_dir)
        if digest(plan_dir / 'plan.json') != plan_sha: raise ValueError('Plan changed during loading')
        requirements, index_sha = checked_requirements(root, plan_dir, plan, bank)
        required_count = len(requirements)
        helper = 'experiments/common/video/pts_clip_sampling.py'
        bound = {r['path']: r['sha256'] for r in plan['data_fingerprints'] + plan['current_source_fingerprints']}
        if bound.get(helper) != digest(root / helper): raise ValueError('Frozen JPEG/LINEAR112 helper changed')
        paths = [Path(__file__).resolve(), root / 'experiments/runspecs/EXP-019/verify_cache.py',
                 root / helper, root / 'experiments/common/video/order_plan.py',
                 root / 'experiments/common/video/temporal_order.py', root / 'experiments/common/video/supervised_plan.py']
        for path in paths:
            name = path.relative_to(root).as_posix(); code[name] = digest(path)
            target = out / 'source-snapshot' / name; target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f: f.write(path.read_bytes())
            if digest(target) != code[name]: raise ValueError('Acceptance source snapshot changed')
        write_json(out / 'launch.json', dict(argv=sys.argv, order_plan_sha256=plan_sha,
            verified_cache_index=INDEX, verified_cache_index_sha256=index_sha, source_sha256=code,
            started_utc=started_utc, workers=workers, pipeline=PIXEL_VERSION,
            media_source_identity='use frozen order-plan data/source fingerprints; no video decode or new frame extraction'))
        cv2.setNumThreads(1)
        result = verify_required_rows(root, requirements, index_csv, failure_rows, workers,
            progress=lambda n, size: print(json.dumps(dict(status='verifying_existing_JPEGs', verified=n,
                total=required_count, bytes_read=size, seconds=time.perf_counter()-began)), flush=True))
        verified, byte_count = result['verified'], result['bytes_read']
        if verified != required_count or result['failed'] != 0: raise ValueError('Incomplete actual pixel acceptance')
        if digest(plan_dir / 'plan.json') != plan_sha or digest(rooted(root, INDEX)) != index_sha:
            raise ValueError('Plan/original verified-cache index changed during acceptance')
        for name, expected in code.items():
            if digest(root / name) != expected: raise ValueError('Acceptance/helper source changed during verification')
        for ref in plan['current_source_fingerprints']:
            if digest(rooted(root, ref['path'])) != ref['sha256']: raise ValueError('Current plan-governance source changed')
        if digest(plan_dir / 'supervised-cache-requirements.csv') != plan['files']['supervised-cache-requirements.csv']:
            raise ValueError('Frozen cache requirements changed during verification')
    except Exception as error:
        failure = dict(error_type=type(error).__name__, error=str(error))
        if index_csv.is_file():
            completed = csv_rows(index_csv); verified = len(completed); byte_count = sum(int(r['size_bytes']) for r in completed)
    failed_count = 1 if failure_rows.is_file() else 0
    proof = dict(status='order_supervised_cache_ready' if failure is None else 'order_supervised_cache_failed',
        spec_id='EXP-019', format_version=1, order_plan_path=plan_dir.relative_to(root).as_posix(),
        order_plan_sha256=plan_sha, verified_cache_index=INDEX, verified_cache_index_sha256=index_sha,
        required_frame_count=required_count, verified_required_frame_count=verified, failed_required_frames=failed_count,
        unverified_required_frames=max(0, required_count-verified-failed_count),
        actual_verified_JPEG_reads=verified, actual_JPEG_byte_SHA_checked=True if failure is None else False,
        actual_RGB112_pixel_SHA_checked=True if failure is None else False, pixel_version=PIXEL_VERSION,
        verified_required_frames_index='verified-required-frames.csv' if index_csv.is_file() else None,
        verified_required_frames_index_sha256=digest(index_csv) if index_csv.is_file() else None,
        metadata_flags_unchanged_by_producer=True, original_cache_and_index_read_only=True, no_media_decode=True,
        no_new_frames=True, no_training=True, bytes_read=byte_count, source_sha256=code, failure=failure,
        started_utc=started_utc, completed_utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter()-began,
        environment=dict(python=sys.version, platform=platform.platform(), opencv=cv2.__version__, numpy=np.__version__))
    write_json(out / 'acceptance.json', proof)
    files = [dict(path=p.relative_to(out).as_posix(), sha256=digest(p), size_bytes=p.stat().st_size)
             for p in sorted(out.rglob('*')) if p.is_file()]
    write_json(out / 'output-manifest.json', dict(files=files))
    if failure is not None: raise RuntimeError('Actual cache acceptance failed; see ' + str(out / 'acceptance.json'))
    return proof
