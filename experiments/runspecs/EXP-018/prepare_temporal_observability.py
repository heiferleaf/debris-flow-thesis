"""Freeze first, review source, then explicitly replay fixed delivery-frame ranges."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video import observability_temporal as common

PRIOR = 'data/process/EXP-018/drift-blind-review-v1/mapping.json'
PRIOR_SHA = 'cff9b57c14af8796488963428a226635019c554c2692618d1fe967588cdac7fb'
BRIDGE = 'data/process/EXP-018/input-observability-v1/mapping.json'
BRIDGE_SHA = '30ba41a4f83600b839fb5470df9cc54b8ba59db2220d6e5f5fbb113a2d1d4231'
DEFAULT_OUT = ROOT / 'data/process/EXP-018/temporal-observability-v2'
SEED = 18025616


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))


def load_tables(root, case, check_quality=True, freeze_range=False):
    coordinates = csv_rows(common.rooted(root, case['coord_path']))
    quality = csv_rows(common.rooted(root, case['qc_path']))
    n = int(case['source_frame_count'])
    if (len(coordinates) != n or len(quality) != n
            or [int(r['frame']) for r in coordinates] != list(range(n))
            or [int(r['frame']) for r in quality] != list(range(n))):
        raise ValueError('Coordinate/QC source count/order mismatch')
    pts = [float(r['pts_time']) for r in coordinates]
    qt, std = [float(r['pts_time']) for r in quality], [float(r['gray_std']) for r in quality]
    import numpy as np
    if (not np.isfinite(pts).all() or not np.isfinite(qt).all() or not np.isfinite(std).all()
            or max(abs(a - b) for a, b in zip(pts, qt)) > .001001):
        raise ValueError('Coordinate/QC PTS/finite check failed')
    required = case['delivery_frame_indices']
    selected_pts = [pts[i] for i in required]
    range_issue = None
    try:
        positions, span = common.validate_range(required, selected_pts,
            case['anchor_frame_indices'], case['anchor_pts_seconds'], case['clean_runs'])
    except ValueError as error:
        if not freeze_range: raise
        # Freeze the original case and defect, never replace it or invent a
        # timeline. Execute will refuse this complete window before decoding.
        positions = [i - required[0] for i in case['anchor_frame_indices']]
        span = float(selected_pts[-1] - selected_pts[0])
        range_issue = str(error)
    if check_quality and any(std[i] < 20. for i in required):
        raise ValueError('At least one full-range frame fails frozen QC: whole window missing')
    return pts, std, positions, span, range_issue


def freeze(root, out, ffmpeg):
    if out.exists(): raise FileExistsError('Refuse existing temporal contract/output')
    inputs = {}
    def record(relative, expected=None):
        path = common.rooted(root, relative); before = path.stat()
        actual = common.digest_file(path)
        after = path.stat()
        if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                or (expected is not None and actual != expected)):
            raise ValueError('Frozen input hash/stability mismatch: ' + relative)
        inputs[relative] = dict(sha256=actual, size_bytes=after.st_size)
        return path
    prior = json.loads(record(PRIOR, PRIOR_SHA).read_text(encoding='utf-8'))
    bridge = json.loads(record(BRIDGE, BRIDGE_SHA).read_text(encoding='utf-8'))
    old = sorted(prior['cases'], key=lambda r: r['anonymous_id'])
    if ([r['anonymous_id'] for r in old] != ['C%02d' % i for i in range(1, 13)]
            or len({r['unit_id'] for r in old}) != 12 or len({r['clip_id'] for r in old}) != 12):
        raise ValueError('Only the original twelve unique windows may be frozen')
    bridge_by_id = {r['case_id']: r for r in bridge['cases']}
    helper = 'experiments/common/video/pts_clip_sampling.py'
    source_paths = [SOURCE.relative_to(root).as_posix(), Path(common.__file__).resolve().relative_to(root).as_posix(), helper]
    for name in source_paths:
        record(name, bridge['actual_preprocessing']['helper_sha256'] if name == helper else None)
    metadata_path = 'data/metadata/EXP-016/metadata-v4/video-inputs.csv'
    expected_metadata = next(r['sha256'] for r in prior['input_fingerprints'] if r['path'] == metadata_path)
    videos = {r['unit_id']: r for r in csv_rows(record(metadata_path, expected_metadata))}
    flips = [False] * 6 + [True] * 6; random.Random(SEED).shuffle(flips)
    cases = []
    for c, flip in zip(old, flips):
        v = videos[c['unit_id']]
        if v != c['source_video_metadata'] or v['split'] != 'validation':
            raise ValueError('Frozen case video metadata differs')
        b = bridge_by_id[c['anonymous_id']]
        if b['clip_id'] != c['clip_id'] or b['unit_id'] != c['unit_id']:
            raise ValueError('Spatial bridge identity differs from fixed window')
        for stem in ('middleware', 'coord', 'qc'):
            relative = v[stem + '_path']
            if stem == 'middleware':
                common.rooted(root, relative).relative_to(root / 'data/process/EXP-008-units-src256')
                if Path(relative).suffix.lower() != '.mp4': raise ValueError('Middleware video only')
            record(relative, v[stem + '_sha256'])
        w = c['window']; indices = w['frame_indices']; times = w['source_pts_seconds']
        required = list(range(indices[0], indices[-1] + 1))
        item = dict(case_id=c['anonymous_id'], clip_id=c['clip_id'], unit_id=c['unit_id'],
            middleware_path=v['middleware_path'], coord_path=v['coord_path'], qc_path=v['qc_path'],
            source_frame_count=int(v['n_frames']), clean_runs=json.loads(v['clean_runs_json']),
            anchor_frame_indices=indices, anchor_pts_seconds=times, delivery_frame_indices=required,
            condition_order=['all_delivered_frames', 'sampled_sixteen'] if flip else ['sampled_sixteen', 'all_delivered_frames'],
            anchors=[])
        coordinate_pts, quality, positions, span, issue = load_tables(root, item, check_quality=False, freeze_range=True)
        item.update(anchor_positions_in_delivery=positions, actual_span_seconds=span,
                    frozen_low_quality_delivery_indices=[i for i in required if quality[i] < 20.],
                    coordinate_range_issue=issue,
                    adjacent_nonincreasing_coordinate_pairs=[[a, b] for a, b in zip(required, required[1:])
                                                            if coordinate_pts[b] <= coordinate_pts[a]])
        bridge_frames = {f['source_frame_index']: f for f in b['frames']}
        if len(c['frames']) != 16 or len(bridge_frames) != 16:
            raise ValueError('Exactly sixteen frozen anchors required')
        for frame, index, timestamp in zip(c['frames'], indices, times):
            expected = bridge_frames[index]
            for key in ('cache_path', 'jpeg_sha256', 'rgb112_sha256'):
                if frame[key] != expected[key]: raise ValueError('Anchor cache/bridge mismatch')
            if int(frame['frame_index']) != index or abs(float(frame['source_pts_seconds']) - timestamp) > 1e-8:
                raise ValueError('Frozen anchor coordinate mismatch')
            path = record(frame['cache_path'], frame['jpeg_sha256'])
            if path.stat().st_size != int(frame['size_bytes']): raise ValueError('Anchor JPEG size changed')
            item['anchors'].append(dict(frame_index=index, cache_path=frame['cache_path'],
                jpeg_sha256=frame['jpeg_sha256'], source_bgr_sha256=frame['source_bgr_sha256'],
                rgb112_sha256=frame['rgb112_sha256'], rgb256_decoded_sha256=expected['rgb256_decoded_sha256']))
        cases.append(item)
    if sum(len(c['delivery_frame_indices']) for c in cases) != 525 or sum(len(c['anchors']) for c in cases) != 192:
        raise ValueError('Fixed case range/anchor count changed')
    if any(abs(c['actual_span_seconds'] - 2.96) > 1e-8 for c in cases if c['case_id'] in ('C08', 'C09')):
        raise ValueError('C08/C09 actual2.96-second endpoints must not be changed')
    ffmpeg = ffmpeg.resolve(strict=True)
    ffmpeg_sha = common.digest_file(ffmpeg)
    pre_registered_missing = {}
    for cid, count in (('C02', 30), ('C05', 1)):
        case = next(c for c in cases if c['case_id'] == cid)
        if len(case['adjacent_nonincreasing_coordinate_pairs']) != count or not case['coordinate_range_issue']:
            raise ValueError('Pre-registered duplicate-PTS finding changed; review required')
        pre_registered_missing[cid] = dict(reason='freeze_preflight_duplicate_coordinate_PTS',
            adjacent_nonincreasing_pairs=count, discovered_before_real_decode=True,
            original_sixteen_anchors_unchanged=True, whole_temporal_window_missing=True)
    contract = dict(format_version=2, review_id='temporal-observability-v2',
        status='frozen_pending_code_review_no_real_decode', created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Same verified256 middleware: original sixteen times versus every delivered frame in the identical inclusive coordinate-index range; not raw/native/all captured frames',
        no_training=True, no_model=True, no_content_judgment=True, no_raw_video=True, cost_cny=0,
        first_case_gate='C01 must fully pass all31 delivery frames and all16 anchor fingerprints before C02..C12',
        window_failure_policy='Whole window missing; retain private partial files and reason; never drop bad frames/stitch/replace a case',
        hard_limits=dict(seconds=common.MAX_SECONDS, output_bytes=common.MAX_BYTES, audit_reserve_bytes=common.AUDIT_RESERVE,
                         scope='entire execute stage including input hashes, extraction and publishing; budget includes existing contract/source snapshot bytes'),
        decoder_route='FFmpeg -copyts -threads1; showinfo BEFORE select; format=bgr24; fps_mode passthrough; full middleware physical-counter audit',
        pixel_route='frozen jpeg95_bytes(BGR256)->JPEG decode->RGB256 PNG; rgb112_from_jpeg is actual INTER_LINEAR112 anchor check; no enhancement',
        quality=dict(actual_gray_std_min=20., frozen_qc_gray_std_min=20., max_actual_frozen_difference=.055001,
                     max_physical_coord_pts_difference=.001001),
        playback='each condition holds own frame to next actual PTS; identical endpoints; no16/3fps, interpolation, rate change, audio or loop',
        randomization=dict(seed=SEED, six_first_each_condition=True),
        expected_windows=12, expected_anchors=192, expected_delivery_frames=525,
        pre_registered_missing_cases=pre_registered_missing,
        expected_replay_eligible_windows=10, expected_replay_eligible_delivery_frames=402,
        expected_replay_eligible_anchors=160,
        revision=dict(supersedes='data/process/EXP-018/temporal-observability-v1',
            old_contract_sha256='9f47c9230e3c106816d9858f61831b0ea016ddb6c9e073d398f753886ac8b1d7',
            same_cases_and_inputs=True, old_contract_and_source_snapshots_preserved=True,
            reasons=['nonzero CLI on unexpected failure', 'single-visible-condition actual displayed-frame coverage gate',
                     'publish only after final input closure and no unexpected failure']),
        inputs=inputs, ffmpeg=dict(path=str(ffmpeg), sha256=ffmpeg_sha, size_bytes=ffmpeg.stat().st_size),
        source_paths=source_paths, cases=cases)
    # Private contract contains identities; viewer is not created until execution.
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'private/frozen-contract.json', contract)
    for name in source_paths:
        target = out / 'private/source-snapshot' / name; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f: f.write(common.rooted(root, name).read_bytes())
        if common.digest_file(target) != inputs[name]['sha256']: raise ValueError('Source snapshot changed')
    (out / 'README.md').write_text(README, encoding='utf-8')
    print(json.dumps(dict(status=contract['status'], contract_sha256=common.digest_file(out / 'private/frozen-contract.json'),
        cases=12, anchors=192, delivery_frames=525, real_video_decoded=False)))
    return contract


def verify_inputs(root, contract, budget):
    for relative, ref in contract['inputs'].items():
        path = common.rooted(root, relative); budget.check()
        if path.stat().st_size != ref['size_bytes'] or common.digest_file(path, budget) != ref['sha256']:
            raise ValueError('Frozen source/input changed: ' + relative)
    decoder = Path(contract['ffmpeg']['path'])
    if decoder.stat().st_size != contract['ffmpeg']['size_bytes'] or common.digest_file(decoder, budget) != contract['ffmpeg']['sha256']:
        raise ValueError('Frozen FFmpeg changed')


def publish_case(out, case, proof, budget):
    for frame in proof['frames']:
        source = out / 'private/assets' / frame['path']; target = out / 'viewer' / frame['path']
        source.resolve().relative_to(out.resolve()); target.resolve().relative_to(out.resolve())
        budget.check(); target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists(): raise FileExistsError(target)
        source.rename(target)  # Same-output move; no media duplication or new bytes.
    public = dict(case_id=case['case_id'], duration_seconds=case['actual_span_seconds'], conditions=[])
    for alias, actual in zip(('K1', 'K2'), case['condition_order']):
        positions = case['anchor_positions_in_delivery'] if actual == 'sampled_sixteen' else list(range(len(proof['frames'])))
        rows = [proof['frames'][i] for i in positions]
        public['conditions'].append(dict(condition_id=alias,
            relative_pts_seconds=[r['relative_pts_seconds'] for r in rows],
            frames=[dict(path=r['path'], sha256=r['png_sha256']) for r in rows]))
    return public


def execute(root, out, expected_contract_sha):
    contract_path = out / 'private/frozen-contract.json'
    if out.resolve() == (root / 'data/process/EXP-018/temporal-observability-v1').resolve():
        raise FileExistsError('v1 contract/source archive is immutable; execute reviewed v2 only')
    if (out / 'private/execution-start.json').exists() or (out / 'viewer').exists():
        raise FileExistsError('One execution per frozen contract; no retry/overwrite')
    budget = common.Budget(out); watchdog = threading.Thread(target=budget.watchdog, daemon=True); watchdog.start()
    public_cases, statuses, replayed, verified = [], [], [], 0
    closure, stop_reason, unexpected_failure, contract = False, None, False, None
    try:
        budget.json(out / 'private/execution-start.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
            contract_sha256=expected_contract_sha, argv=sys.argv, no_torch='torch' not in sys.modules))
        if common.digest_file(contract_path, budget) != expected_contract_sha:
            raise ValueError('Reviewed contract SHA differs')
        contract = json.loads(contract_path.read_text(encoding='utf-8'))
        if (contract.get('format_version') != 2 or contract['status'] != 'frozen_pending_code_review_no_real_decode'
                or [c['case_id'] for c in contract['cases']] != ['C%02d' % i for i in range(1, 13)]
                or set(contract['pre_registered_missing_cases']) != {'C02', 'C05'}
                or contract['hard_limits']['seconds'] != common.MAX_SECONDS
                or contract['hard_limits']['output_bytes'] != common.MAX_BYTES):
            raise ValueError('Unexpected frozen contract')
        if 'torch' in sys.modules: raise RuntimeError('Unexpected torch import')
        verify_inputs(root, contract, budget)
        decoder = Path(contract['ffmpeg']['path'])
        version = subprocess.run([str(decoder), '-version'], capture_output=True, text=True, check=True,
            timeout=min(10., common.MAX_SECONDS - (time.monotonic() - budget.started))).stdout.splitlines()[0]
        budget.json(out / 'private/environment.json', dict(python=sys.version, opencv=common.cv2.__version__,
            numpy=common.np.__version__, ffmpeg_version=version, frozen_ffmpeg_sha256=contract['ffmpeg']['sha256']))
        for case in contract['cases']:
            budget.check()
            cid = case['case_id']
            if cid in contract['pre_registered_missing_cases']:
                statuses.append(dict(case_id=cid, status='pre_registered_timeline_missing',
                    **contract['pre_registered_missing_cases'][cid]))
                continue
            if cid != 'C01' and not (out / 'private/C01-gate-passed.json').is_file():
                raise RuntimeError('C01 gate must pass before any remaining eligible window')
            try:
                # Pre-existing JPEG anchors are read-only and checked against all
                # old JPEG/RGB256/LINEAR112 fingerprints before each replay.
                for anchor in case['anchors']:
                    encoded = common.rooted(root, anchor['cache_path']).read_bytes(); budget.check()
                    if common.digest_bytes(encoded) != anchor['jpeg_sha256']: raise ValueError('Old JPEG changed')
                    pixels = common.cv2.imdecode(common.np.frombuffer(encoded, common.np.uint8), common.cv2.IMREAD_COLOR)
                    rgb = common.cv2.cvtColor(pixels, common.cv2.COLOR_BGR2RGB)
                    if (common.digest_bytes(rgb.tobytes()) != anchor['rgb256_decoded_sha256']
                            or common.digest_bytes(common.rgb112_from_jpeg(encoded).tobytes()) != anchor['rgb112_sha256']):
                        raise ValueError('Old RGB256/LINEAR112 anchor differs')
                load_tables(root, case)
                proof = common.extract_case(root, case, decoder, version, budget, out / 'private')
                budget.json(out / ('private/%s/replay-proof.json' % case['case_id']), proof)
                if case['case_id'] == 'C01':
                    budget.json(out / 'private/C01-gate-passed.json', dict(status='C01_full_replay_passed',
                        delivery_frames=len(proof['frames']), verified_anchors=16,
                        proof_sha256=common.digest_file(out / 'private/C01/replay-proof.json', budget),
                        contract_sha256=expected_contract_sha))
                # Keep every verified frame private until GLOBAL final closure.
                replayed.append((case, proof)); verified += 16
                statuses.append(dict(case_id=case['case_id'], status='passed', delivery_frames=len(proof['frames']),
                                     verified_anchors=16, actual_span_seconds=case['actual_span_seconds']))
                print(json.dumps(statuses[-1]), flush=True)
            except Exception as error:
                reason = type(error).__name__ + ': ' + str(error)
                statuses.append(dict(case_id=case['case_id'], status='unexpected_failure', reason=reason))
                failure = out / ('private/%s/failure.json' % case['case_id'])
                if not failure.exists(): budget.json(failure, statuses[-1], audit=True)
                # No non-preregistered error may continue to a later case.
                raise
        expected = {c['case_id'] for c in contract['cases']} - set(contract['pre_registered_missing_cases'])
        if {c['case_id'] for c, _ in replayed} != expected:
            raise RuntimeError('Unexpected eligible-window coverage missing')
        verify_inputs(root, contract, budget)
        if 'torch' in sys.modules: raise RuntimeError('Unexpected torch import at final closure')
        closure = True
        for case, proof in replayed:
            public_cases.append(publish_case(out, case, proof, budget))
    except Exception as error:
        stop_reason = type(error).__name__ + ': ' + str(error)
        unexpected_failure = True
        public_cases = []
    finally:
        budget.stop.set(); watchdog.join(timeout=1)
        observed = {s['case_id'] for s in statuses}
        for cid in ['C%02d' % i for i in range(1, 13)]:
            if cid not in observed:
                statuses.append(dict(case_id=cid, status='not_extracted', reason=stop_reason or budget.expired or 'not_reached'))
        if 'torch' in sys.modules:
            stop_reason = stop_reason or 'Unexpected torch import; measurement withheld'
            unexpected_failure = True
        if unexpected_failure or not closure:
            public_cases = []
        success = closure and not unexpected_failure and len(public_cases) == 10
        public = dict(review_id='temporal-observability-v2', cases=public_cases,
            publication_status='published_after_final_closure' if success else 'withheld_due_unexpected_failure',
            prediction_and_selection_role_blinded=True, condition_names_anonymized=True,
            fully_condition_blinded=False, identity_blind=False, native_OSD_retained=True,
            display=dict(canvas_raster=[256, 256], canvas_css=[256, 256], image_smoothing=False, filters='none'),
            playback=dict(actual_relative_pts=True, hold_to_next_frame=True, stop_at_common_last_anchor=True,
                no_interpolation=True, no_rate_change=True, no_audio=True, no_loop=True,
                full_independent_playback_before_parallel=True), missing_case_ids=[s['case_id'] for s in statuses if s['status'] != 'passed'])
        budget.json(out / 'viewer/manifest.json', public, audit=True)
        budget.write(out / 'viewer/manifest.js', ('window.TEMPORAL_REVIEW=' + json.dumps(public, ensure_ascii=True, allow_nan=False) + ';\n').encode('utf-8'), audit=True)
        budget.write(out / 'viewer/index.html', common.HTML.encode('utf-8'), audit=True)
        summary = dict(status='measurement_complete_with_pre_registered_missing' if success else 'unexpected_failure_measurement_withheld',
            execution_success=success, unexpected_failure=unexpected_failure,
            no_training=True, no_model=True, no_content_judgment=True, no_raw_video=True, cost_cny=0,
            contract_sha256=expected_contract_sha, windows_replayed=len(replayed), windows_published=len(public_cases), verified_anchors=verified,
            expected_anchors=192, cases=statuses, inputs_unchanged_after_execution=closure,
            pre_registered_missing_case_ids=['C02', 'C05'], replay_eligible_expected_anchors=160,
            stop_reason=stop_reason or budget.expired, seconds=time.monotonic() - budget.started,
            byte_cap=common.MAX_BYTES, bytes_before_final_summary=budget.used,
            limit_policy='processing/decoder killed at deadline; reserved bounded audit finalization retains partial/missing reasons')
        budget.json(out / 'private/execution-summary.json', summary, audit=True)
        records = []
        for p in sorted(out.rglob('*')):
            if p.is_file(): records.append(dict(path=p.relative_to(out).as_posix(), sha256=common.digest_file(p), size_bytes=p.stat().st_size))
        budget.json(out / 'output-manifest.json', dict(files=records), audit=True)
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > common.MAX_BYTES:
            raise common.LimitReached('Output byte hard cap violated')
        print(json.dumps(dict(status=summary['status'], published=len(public_cases), verified_anchors=verified, reason=summary['stop_reason'])), flush=True)
    return summary


README = '''# Temporal observability measurement v2

v2 supersedes temporal-observability-v1 without modifying its frozen contract or
source snapshots. It retains exactly the same twelve cases and source inputs.
Review fixes: unexpected failures now stop subsequent cases and return nonzero;
the independent-playback gate requires actual visible-single-condition frame
coverage (a start-to-end callback jump cannot complete); verified frames remain
private until final input closure, with all public cases withheld on any error.
C02/C05 are pre-registered whole-timeline missing cases from static duplicate-PTS
findings, not new decoder failures. Eligible ten windows contain402 delivery
frames and at most160 actual replay anchors. The original192 cached anchors and
their sixteen-unique-time training inputs are not declared invalid.

Only viewer/ may be given to an independent observer. private/ contains source
identities, original cache links, condition mapping and replay proof. The fixed
twelve drift-blind-review-v1 cases are never replaced. No model, training,
visibility judgment, raw-video decode, enhancement or source/cache mutation.

This is the same verified256 middleware, comparing the original16 sampled
moments with every delivered frame in the identical inclusive coordinate-index
range. It is not original/native resolution or all captured/source-native frames.
C08/C09 retain actual2.96-second endpoints. Every anchor must reproduce old
sourceBGR/JPEG95/decodedRGB256/actualINTER_LINEAR112 fingerprints. Each added frame
must satisfy exact index count/order, physical/coordinate PTS, full clean-range
coverage and unchanged source/frozen-QC checks. One bad frame makes the entire
window missing, without deletion, stitching, replacement or tolerance relaxation.

Freeze performs hashes/static checks only. After source/contract review, execute
requires its SHA explicitly; C01 must pass the complete31-frame/16-anchor chain
before any remaining window is replayed. Existing execution is never overwritten
or retried. Failures and private partial assets remain, with missing reasons.

The entire execute stage has a3600-second processing deadline and1,000,000,000-byte
output ceiling, including the existing contract/source snapshot bytes. All writes
reserve bytes before opening, keeping1MB for bounded failure/audit finalization.
A watchdog kills an active decoder at the deadline. Failure finalization may
write bounded status/manifest metadata after stopping media processing; it cannot
resume frames or exceed the byte cap. Partial cases are never published.

Both conditions are256x256 RGB PNGs with disk pixel checks, shown in identical
256x256 canvases with smoothing disabled, no filters, no audio and no loop.
Each timeline independently holds its current frame until the next actual PTS,
then both stop at their common final anchor. Browser refresh may delay/miss a
presentation; source timing is not replaced by16/3fps. Observe each condition
independently from start to end before parallel comparison becomes available.
Changing position invalidates that independent-playback completion attempt.
Only frame indices drawn in the currently visible single condition during that
attempt count; hidden-condition traces and parallel display never add coverage.
Dropped presentation frames keep the gate closed; timing is never slowed or
frames interpolated to force completion.
Condition names/order are anonymized; frame density and retained OSD may reveal
condition/identity. This is prediction/selection-role blinding, not full blinding.

Commands (repository root; existing freeze/execute outputs are refused):
.venv/Scripts/python.exe -B experiments/runspecs/EXP-018/prepare_temporal_observability.py --stage freeze --ffmpeg D:/CodexCache/debris-flow-thesis/tools/ffmpeg-9.0.2/ffmpeg.exe
.venv/Scripts/python.exe -B experiments/runspecs/EXP-018/prepare_temporal_observability.py --stage execute --contract-sha256 REVIEWED_CONTRACT_SHA256

The execution command is reserved for after source review. Frozen-contract.json
and source-snapshot bind all source/coord/QC/cache/decoder/code SHA values. Only
the newly created output is writable. This tool does not change any old results.
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=('freeze', 'execute'), default='freeze')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--ffmpeg', type=Path)
    ap.add_argument('--contract-sha256')
    a = ap.parse_args(); out = a.out.resolve(); out.relative_to(ROOT / 'data/process/EXP-018')
    if a.stage == 'freeze':
        if 'torch' in sys.modules: raise RuntimeError('Unexpected torch import')
        if a.ffmpeg is None: ap.error('--ffmpeg is required to freeze its executable SHA')
        freeze(ROOT, out, a.ffmpeg)
    else:
        if a.contract_sha256 is None: ap.error('--contract-sha256 from reviewed frozen contract is required')
        summary = execute(ROOT, out, a.contract_sha256)
        if not summary['execution_success']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
