"""Streaming complete-video evaluation on a frozen label-independent grid.

Decode once for both models. Retain only frames still referenced by unfinished
windows; never materialize the full video or all dense RGB clips in memory.
"""
from __future__ import annotations
from collections import Counter, defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import threading
import time
import sys
import platform

import cv2
import numpy as np
import torch

from experiments.common.video.pts_clip_sampling import jpeg95_bytes, rgb112_from_jpeg, selection_expression
from experiments.common.video.r2plus1d_classifier import VideoClassifier
from experiments.common.video.supervised_plan import csv_rows, digest, jsonl, verify_plan, write_json
from experiments.common.video.supervised_runtime import ClipInput, score_clips


_RAW_DTS_WARNING = re.compile(
    r'Application provided invalid, non monotonically increasing dts to muxer in stream \d+: -?\d+ >= -?\d+')
_PHYSICAL_INFO = re.compile(r'\bn:\s*(\d+)\s+pts:\s*\S+\s+pts_time:\s*([-+0-9.eE]+)')


def observe_decoder_line(line, pts, state, errors, diagnostics):
    # FFmpeg stderr writers can interleave this known raw-output warning with
    # showinfo. Remove only its exact message; still check the frame and other
    # diagnostic text on the same line.
    warnings = _RAW_DTS_WARNING.findall(line)
    state['raw_muxer_dts_warnings'] += len(warnings)
    line = _RAW_DTS_WARNING.sub('', line)
    match = _PHYSICAL_INFO.search(line)
    if match:
        index, timestamp = int(match[1]), float(match[2])
        if index != state['decoded'] or index >= len(pts) or not np.isfinite(timestamp) or 's:256x256' not in line:
            errors.append('Physical frame count/order mismatch')
        elif abs(timestamp - pts[index]) > .001001:
            errors.append(f'PTS mismatch at {index}')
        else:
            state['max_pts_error'] = max(state['max_pts_error'], abs(timestamp - pts[index]))
        state['decoded'] += 1
    if 'error' in line.lower() or 'invalid' in line.lower():
        diagnostics.append(line.strip())


def video_metrics(rows):
    if any(r['score'] is None for r in rows):
        return dict(status='incomplete_coverage_metrics_withheld', total=len(rows),
                    unknown=sum(r['score'] is None for r in rows))
    y = np.array([r['label'] for r in rows], dtype=int)
    score = np.array([r['score'] for r in rows], dtype=float)
    if not np.isfinite(score).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid video prediction/label')
    p, n = score[y == 1], score[y == 0]
    tp, fp = int(np.sum(p >= .5)), int(np.sum(n >= .5))
    fn, tn = len(p) - tp, len(n) - fp
    return dict(status='complete', P=len(p), N=len(n), TP=tp, FP=fp, FN=fn, TN=tn,
                Recall=tp / len(p) if len(p) else None, FPR=fp / len(n) if len(n) else None,
                Precision=tp / (tp + fp) if tp + fp else 0.,
                F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                AUC=float(np.mean((p[:, None] > n) + .5 * (p[:, None] == n))) if len(p) and len(n) else None)


def checked_frames(source, pts, ffmpeg, evidence, selected=None, filter_path=None):
    """Physical sequential frame counter/PTS checked against frozen coordinates."""
    version = subprocess.run([str(ffmpeg), '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    major = int(re.search(r'ffmpeg version (?:n)?(\d+)', version).group(1))
    selected = list(range(len(pts))) if selected is None else sorted(set(selected))
    if not selected or selected[0] < 0 or selected[-1] >= len(pts):
        raise ValueError('Invalid physical frame selection')
    command = [str(ffmpeg), '-hide_banner', '-nostdin', '-copyts', '-threads', '1', '-i', str(source)]
    if filter_path is not None:
        script = f"showinfo,select='{selection_expression(selected)}',format=bgr24"
        Path(filter_path).write_text(script, encoding='utf-8')
        command += ['-/filter:v' if major >= 8 else '-filter_script:v', str(filter_path)]
    elif len(selected) != len(pts):
        raise ValueError('Selection requires an explicit reproducible filter file')
    else:
        command += ['-vf', 'showinfo,format=bgr24']
    command += ['-an']
    command += ['-fps_mode', 'passthrough'] if major >= 5 else ['-vsync', '0']
    command += ['-f', 'rawvideo', 'pipe:1']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1024**2)
    diagnostics, errors = deque(maxlen=10), []
    state = dict(decoded=0, max_pts_error=0., raw_muxer_dts_warnings=0)

    def drain():
        for raw in iter(process.stderr.readline, b''):
            line = raw.decode('utf-8', errors='replace')
            observe_decoder_line(line, pts, state, errors, diagnostics)
    worker = threading.Thread(target=drain, daemon=True)
    worker.start()
    count, frame_bytes = 0, 256 * 256 * 3
    try:
        while True:
            data = process.stdout.read(frame_bytes)
            if not data:
                break
            if len(data) != frame_bytes:
                raise ValueError('Incomplete raw frame')
            if count >= len(selected):
                raise ValueError('More physical frames than coordinate rows')
            yield selected[count], np.frombuffer(data, dtype=np.uint8).reshape(256, 256, 3)
            count += 1
        process.wait()
        worker.join()
        if process.returncode or errors or count != len(selected) or state['decoded'] != len(pts) or diagnostics:
            raise RuntimeError(dict(returncode=process.returncode, raw_frames=count,
                                    expected=len(selected), parsed=state['decoded'], errors=errors[:4],
                                    decoder_errors=list(diagnostics)))
        evidence.update(state, ffmpeg=version, command=command)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()
        worker.join(timeout=5)


def dense_batches(source, pts, windows, ffmpeg, evidence, known_pixels=None, batch_size=8, filter_path=None):
    ordered = sorted(windows, key=lambda r: (r['frame_indices'][-1], r['frame_indices'][0]))
    if not ordered:
        evidence.update(windows=0, rgb112_digest=hashlib.sha256().hexdigest(), minimum_selected_gray_std=None)
        return
    remaining = Counter(i for r in ordered for i in r['frame_indices'])
    cache, batch, meta, cursor = {}, [], [], 0
    pixel_digest = hashlib.sha256()
    minimum_std = float('inf')
    for index, bgr in checked_frames(source, pts, ffmpeg, evidence,
                                     selected=sorted(remaining) if filter_path else None, filter_path=filter_path):
        if index not in remaining:
            continue
        gray_std = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).std())
        if not np.isfinite(gray_std) or gray_std < 20:
            raise ValueError(f'Dense selected source frame {index} failed exact gray_std: {gray_std}')
        minimum_std = min(minimum_std, gray_std)
        rgb = rgb112_from_jpeg(jpeg95_bytes(bgr))
        if known_pixels and index in known_pixels:
            expected = known_pixels[index]
            if (hashlib.sha256(bgr.tobytes()).hexdigest() != expected['source_bgr_sha256']
                    or hashlib.sha256(rgb.tobytes()).hexdigest() != expected['rgb112_sha256']):
                raise ValueError(f'Dense/training pixel mismatch at source frame {index}')
        pixel_digest.update(str(index).encode('ascii') + b'\t' + rgb.tobytes())
        cache[index] = rgb
        while cursor < len(ordered) and ordered[cursor]['frame_indices'][-1] <= index:
            row = ordered[cursor]
            batch.append(np.stack([cache[i] for i in row['frame_indices']]))
            meta.append(row)
            for i in row['frame_indices']:
                remaining[i] -= 1
                if remaining[i] == 0:
                    del remaining[i], cache[i]
            cursor += 1
            if len(batch) == batch_size:
                yield np.stack(batch), meta
                batch, meta = [], []
    if batch:
        yield np.stack(batch), meta
    if cursor != len(ordered) or remaining:
        raise ValueError('Dense window coverage incomplete')
    evidence.update(windows=len(ordered), rgb112_digest=pixel_digest.hexdigest(),
                    minimum_selected_gray_std=minimum_std if ordered else None)


def compatible_pixel_reference(plan, plan_sha, reference, origin_plan=None):
    if reference.get('status') != 'dense_pixels_verified':
        raise ValueError('A completed label-free pixel verification is required')
    if reference['plan_sha256'] == plan_sha:
        return
    if origin_plan is None or digest(origin_plan) != reference['plan_sha256']:
        raise ValueError('Pixel reference original plan missing or changed')
    original = json.loads(Path(origin_plan).read_text(encoding='utf-8'))
    if original['resource'] != plan['resource'] or original['sources'] != plan['sources']:
        raise ValueError('Dense input resources differ; repeat pixel verification')


def load_selected_checkpoint(path, plan_sha, mode):
    path = Path(path)
    result = json.loads((path.parent / 'training-result.json').read_text(encoding='utf-8'))
    selected = result.get('selected_checkpoint', {})
    if (result['status'] != 'supervised_complete_dense_evaluation_pending'
            or path.name != selected.get('filename') or path.name != 'best.pt'):
        raise ValueError('Only the selected best of a completed real training run may be evaluated')
    before = digest(path)
    if before != selected['sha256']:
        raise ValueError('Selected checkpoint fingerprint mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if digest(path) != before:
        raise ValueError('Checkpoint changed while loading')
    profile = checkpoint['profile']
    if (profile['mode'] != mode or profile['plan_sha256'] != plan_sha or profile['smoke']
            or result['completed_epochs'] != profile['requested_epochs']
            or result['optimizer_steps'] != profile['requested_optimizer_updates']
            or checkpoint['completed_epochs'] != result['best_epoch']
            or checkpoint['completed_epochs'] != selected['completed_epochs']):
        raise ValueError('Checkpoint profile/selection/budget mismatch')
    return checkpoint, before


def checkpoint_input_modes(checkpoints, input_modes=None):
    """Separate result names from actual model inputs; old callers are unchanged."""
    checkpoints = checkpoints or {}
    modes = dict(input_modes) if input_modes is not None else {name:name for name in checkpoints}
    if set(modes) != set(checkpoints) or any(mode not in ('rgb_zero','rgb_diff') for mode in modes.values()):
        raise ValueError('Every checkpoint needs exactly one registered input mode')
    return modes


def evaluate(root, plan_dir, out, ffmpeg, checkpoints=None, device='cuda', pixel_reference=None,
             input_modes=None):
    root, plan_dir, out = map(Path, (root, plan_dir, out))
    code = [Path(__file__), root / 'experiments/common/video/supervised_runtime.py',
            root / 'experiments/common/video/pts_clip_sampling.py',
            root / 'experiments/common/video/r2plus1d_classifier.py',
            root / 'experiments/runspecs/EXP-016/evaluate.py',
            root / 'experiments/runspecs/EXP-016/run.py']
    execution_sha = {p.relative_to(root).as_posix(): digest(p) for p in code}
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    if out.exists():
        raise FileExistsError(out)
    plan = verify_plan(root, plan_dir)
    resource = root / plan['resource']
    for ref in csv_rows(resource / 'dense-transfer-references.csv'):
        p = root / ref['local_path']
        if p.stat().st_size != int(ref['size_bytes']) or digest(p) != ref['sha256']:
            raise ValueError('Dense input file changed')
    grouped = defaultdict(list)
    for row in jsonl(resource / 'dense-eval-windows.jsonl'):
        grouped[row['unit_id']].append(row)
    videos = [r for r in csv_rows(resource / 'video-inputs.csv') if r['split'] == 'validation']
    if len(videos) != 31 or sum(map(len, grouped.values())) != 32567:
        raise ValueError('Unexpected dense evaluation size')
    if any(r['video_label'] not in ('pos', 'neg') for r in videos):
        raise ValueError('Unknown video ground truth label')
    known = defaultdict(dict)
    index_path = next(root / r['path'] for r in plan['sources'] if r['path'].endswith('verified-cache-files.csv'))
    wanted_videos = {r['unit_id'] for r in videos}
    with index_path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['unit_id'] in wanted_videos:
                known[row['unit_id']][int(row['frame_index'])] = row
    reference = json.loads(Path(pixel_reference).read_text(encoding='utf-8')) if pixel_reference else None
    if reference:
        compatible_pixel_reference(plan, digest(plan_dir / 'plan.json'), reference,
                                   Path(pixel_reference).parent / 'source-plan.json')
    modes = checkpoint_input_modes(checkpoints, input_modes)
    models, loaded_hashes, profiles = {}, {}, []
    for mode, ckpt_path in (checkpoints or {}).items():
        checkpoint, loaded_hashes[mode] = load_selected_checkpoint(ckpt_path, digest(plan_dir / 'plan.json'), modes[mode])
        profile = checkpoint['profile']
        profiles.append(profile)
        net = VideoClassifier(pretrained=False).to(device).eval()
        net.load_state_dict(checkpoint['state_dict'])
        models[mode] = (net, ClipInput(modes[mode]).to(device))
    if len(models) > 1:
        for field in ('initial_state_sha256', 'seed', 'effective_batch', 'microbatch', 'checkpoint_backbone',
                      'precision', 'requested_epochs', 'requested_optimizer_updates'):
            if any(profile[field] != profiles[0][field] for profile in profiles[1:]):
                raise ValueError(f'Unpaired training configurations: {field}')
    out.mkdir(parents=True, exist_ok=False)
    evidence, scores = {}, {mode: [] for mode in models}
    started = time.perf_counter()
    with (out / 'window-scores.csv').open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['unit_id', 'start', 'end', *models])
        writer.writeheader()
        for video in videos:
            uid = video['unit_id']
            pts = np.array([float(r['pts_time']) for r in csv_rows(root / video['coord_path'])])
            info, maxima = {}, {mode: None for mode in models}
            for u8, metadata in dense_batches(root / video['middleware_path'], pts, grouped[uid],
                                              ffmpeg, info, known[uid], filter_path=out / f'{uid}-filter.txt'):
                results = {mode: torch.sigmoid(score_clips(net, torch.from_numpy(u8), transform)).numpy()
                           for mode, (net, transform) in models.items()}
                for i, row in enumerate(metadata):
                    output = dict(unit_id=uid, start=row['source_pts_seconds'][0], end=row['source_pts_seconds'][-1])
                    for mode, values in results.items():
                        score = float(values[i])
                        output[mode] = score
                        maxima[mode] = score if maxima[mode] is None else max(maxima[mode], score)
                    if models:
                        writer.writerow(output)
            if reference and info['rgb112_digest'] != reference['videos'][uid]['rgb112_digest']:
                raise ValueError(f'Dense preprocessing changed: {uid}')
            evidence[uid] = info
            # Ground-truth video labels are consulted ONLY after scoring.
            for mode in models:
                scores[mode].append(dict(unit_id=uid, label=int(video['video_label'] == 'pos'),
                                         score=maxima[mode], group_id=video['group_id'], windows=info['windows']))
            print(json.dumps(dict(video=uid, windows=info['windows'], seconds=time.perf_counter() - started)), flush=True)
    result = dict(status='dense_evaluation_complete' if models else 'dense_pixels_verified',
                  plan_sha256=digest(plan_dir / 'plan.json'), videos=evidence, seconds=time.perf_counter() - started,
                  metrics={mode: video_metrics(rows) for mode, rows in scores.items()},
                  checkpoint_sha256=loaded_hashes, checkpoint_input_modes=modes,
                  scores=scores, independent_test_used=False,
                  scope='development validation; frozen grid uses known quality/cut information')
    result['metrics_by_group'] = {mode: {group: video_metrics([r for r in rows if r['group_id'] == group])
                              for group in sorted({r['group_id'] for r in rows})}
                              for mode, rows in scores.items()}
    if any(digest(root / path) != sha for path, sha in execution_sha.items()):
        raise RuntimeError('Evaluation source changed during execution')
    result['code_sha256'] = execution_sha
    result['environment'] = dict(python=sys.version, torch=torch.__version__, opencv=cv2.__version__,
                                 platform=platform.platform(), device=str(device), precision='FP32')
    if pixel_reference:
        result['pixel_reference_sha256'] = digest(pixel_reference)
    for mode, path in (checkpoints or {}).items():
        if digest(path) != loaded_hashes[mode]:
            raise ValueError('Model checkpoint changed during evaluation')
    # Keep the exact originating plan beside pixel evidence for safe reuse when
    # only supervised assignments change and every resource fingerprint agrees.
    (out / 'source-plan.json').write_bytes((plan_dir / 'plan.json').read_bytes())
    write_json(out / 'result.json', result)
    return result
