"""One canonical dense decode per inner-role video, with A/B/C batch sharing."""
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from experiments.common.video.cv_supervised_training import _json, _sha
from experiments.common.video.dense_evaluation import dense_batches, video_metrics
from experiments.common.video.head_comparison_training import LinearVideoClassifier
from experiments.common.video.oriented_energy import OrientedEnergy, read_artifact
from experiments.common.video.oriented_energy_model import EnergyClassifier, RGB_INITIAL_SHA
from experiments.common.video.pts_clip_sampling import tuple_digest
from experiments.common.video.r2plus1d_classifier import configure_fp32, state_digest
from experiments.common.video.supervised_runtime import ClipInput


def selected_models(context, run, fold, device):
    plan, root = context['plan'], context['root']
    data = read_artifact(root / plan['kernel_path'], plan['kernel_sha256'])
    models, bindings = {}, {}
    for arm in ('A', 'B', 'C'):
        ref = plan['A_references'][str(fold)]
        directory = context['reference_root'] / ref['item'] if arm == 'A' else run / ('f%02d-%s-s7' % (fold, arm))
        result = json.loads((directory / 'training-result.json').read_text(encoding='utf-8'))
        selected = directory / 'best.pt'
        fingerprint = _sha(selected)
        if (result['status'] != 'fit_complete' or result['stage'] != 'inner_select' or result['fold'] != fold
                or result['completed_epochs'] != 20 or result['selected_checkpoint']['filename'] != 'best.pt'
                or result['selected_checkpoint']['sha256'] != fingerprint):
            raise ValueError('Only selected best of full inner fit is admissible')
        if arm == 'A':
            if result['selected_epochs'] != ref['selected_epochs']:
                raise ValueError('Original A selected E differs')
            for name, expected in ref['files'].items():
                p = context['reference_root'] / name
                if p.stat().st_size != expected['size_bytes'] or _sha(p) != expected['sha256']:
                    raise ValueError('Explicit mapped A reference bytes differ')
        checkpoint = torch.load(selected, map_location='cpu', weights_only=False)
        profile = checkpoint['profile']
        if (checkpoint['completed_epochs'] != result['selected_epochs'] or profile['seed'] != 7
                or profile['fold'] != fold or profile['stage'] != 'inner_select' or profile['mode'] != 'rgb_zero'
                or profile['execution_kind'] != 'registered_fit' or profile['requested_epochs'] != 20):
            raise ValueError('Selected inner profile mismatch')
        if arm == 'A':
            if profile['head'] != 'linear' or profile['initial_state_sha256'] != RGB_INITIAL_SHA:
                raise ValueError('A must be the original linear RGB head')
            net = LinearVideoClassifier(pretrained=False)
        else:
            if (profile['energy_mode'] != arm or profile['RGB_component_initial_state_sha256'] != RGB_INITIAL_SHA
                    or profile['kernel_sha256'] != plan['kernel_sha256'] or profile['head_parameters'] != 625
                    or profile['initial_state_sha256'] != result['initial_state_sha256']
                    or profile['input_fingerprints'] != context['fingerprints']):
                raise ValueError('Fusion profile/operator differs')
            net = EnergyClassifier(root / plan['kernel_path'], plan['kernel_sha256'], arm, pretrained=False)
        net.load_state_dict(checkpoint['state_dict'])
        if state_digest(net) != checkpoint['state_sha256']:
            raise ValueError('Selected tensor digest differs')
        if arm != 'A' and state_digest(net.energy) != state_digest(OrientedEnergy(data)):
            raise ValueError('Checkpoint changed frozen operator buffers')
        models[arm] = net.to(device).eval()
        bindings[arm] = dict(checkpoint_path=str(selected), checkpoint_sha256=fingerprint, selected_epochs=result['selected_epochs'])
        del checkpoint
    return models, bindings


def score_stream(batches, info, expected_digest, models, transform, callbacks):
    """Provisional window scores; max is accepted only after canonical EOF."""
    maxima = {arm: None for arm in ('A', 'B', 'C')}
    seen = set(); count = 0
    with torch.inference_mode():
        for pixels, metadata in batches:
            u8 = torch.from_numpy(pixels) if isinstance(pixels, np.ndarray) else pixels
            x = transform(u8.to(next(models['A'].parameters()).device))
            descriptions = models['B'].energy(x, ('B', 'C'))
            logits = {arm: (models[arm](x) if arm == 'A' else models[arm](x, descriptions=descriptions)).reshape(-1).float().cpu() for arm in models}
            scores = {arm: torch.sigmoid(z).numpy() for arm, z in logits.items()}
            if any(len(v) != len(metadata) or not np.isfinite(v).all() for v in scores.values()) or any(not torch.isfinite(z).all() for z in logits.values()):
                raise FloatingPointError('Nonfinite A/B/C dense score')
            for i, row in enumerate(metadata):
                key = tuple_digest(row['frame_indices'])
                if key in seen:
                    raise ValueError('Duplicate physical dense window')
                seen.add(key); count += 1
                for arm in models:
                    value = float(scores[arm][i])
                    maxima[arm] = value if maxima[arm] is None else max(maxima[arm], value)
                    callbacks[arm](dict(unit_id=row['unit_id'], target_start_seconds=row['target_start_seconds'],
                        start=row['source_pts_seconds'][0], end=row['source_pts_seconds'][-1], frame_tuple_sha256=key,
                        logit=float(logits[arm][i]), score=value))
    if not count or info.get('windows') != count or info.get('rgb112_digest') != expected_digest:
        raise ValueError('Canonical RGB112 EOF/grid closure failed; provisional scores withheld')
    return dict(maxima=maxima, windows=count, canonical_reference_matched=True)


def evaluate_role(context, run, fold, ffmpeg, device='cuda'):
    run = Path(run).resolve(); root, plan = context['root'], context['plan']
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('No requested CUDA device; no fallback')
    configure_fp32(7); cv2.setNumThreads(1); torch.set_num_threads(4)
    models, bindings = selected_models(context, run, fold, device)
    transform = ClipInput('rgb_zero').to(device)
    directories = {arm: run / ('dense-f%02d-%s' % (fold, arm)) for arm in models}
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=False)
    streams, writers = {}, {}
    scores = {arm: [] for arm in models}; evidence = {}; started = time.perf_counter()
    uids = plan['roles'][str(fold)]['video_ids']
    try:
        for arm, directory in directories.items():
            streams[arm] = (directory / 'window-scores.csv').open('x', encoding='utf-8-sig', newline='')
            writers[arm] = csv.DictWriter(streams[arm], fieldnames=['unit_id', 'target_start_seconds', 'start', 'end', 'frame_tuple_sha256', 'logit', 'score'])
            writers[arm].writeheader()
        for uid in uids:
            source = context['sources'][uid]
            with (root / source['coord_path']).open(encoding='utf-8-sig', newline='') as f:
                pts = np.array([float(r['pts_time']) for r in csv.DictReader(f)])
            windows = context['dense']['windows'][uid]; info = {}
            batches = dense_batches(root / source['middleware_path'], pts, windows, ffmpeg, info, context['known'][uid], batch_size=8,
                filter_path=directories['A'] / (uid + '-filter.txt'))
            closed = score_stream(batches, info, context['dense']['reference']['videos'][uid]['rgb112_digest'], models, transform,
                {arm: writer.writerow for arm, writer in writers.items()})
            evidence[uid] = dict(info, windows=closed['windows'], canonical_reference_matched=True)
            for arm in models:
                scores[arm].append(dict(unit_id=uid, group_id=source['group_id'], label=int(source['video_label'] == 'pos'),
                    score=closed['maxima'][arm], windows=closed['windows'], fold=fold))
            print(json.dumps(dict(status='dense_video_closed', fold=fold, unit_id=uid, windows=closed['windows'])), flush=True)
        if set(evidence) != set(uids) or sum(v['windows'] for v in evidence.values()) != sum(len(context['dense']['windows'][u]) for u in uids):
            raise ValueError('Incomplete registered inner dense role')
        for arm, binding in bindings.items():
            if _sha(binding['checkpoint_path']) != binding['checkpoint_sha256']:
                raise ValueError('Selected weight changed while evaluating')
        for name, expected in context['fingerprints'].items():
            if _sha(root / name) != expected:
                raise ValueError('Source/input changed while evaluating')
        results = {}
        for stream in streams.values():
            stream.flush(); stream.close()
        for arm, directory in directories.items():
            result = dict(status='inner_role_dense_complete', fold=fold, arm=arm, scores=scores[arm], metrics=video_metrics(scores[arm]),
                videos=evidence, selected_checkpoint=bindings[arm], window_scores_sha256=_sha(directory / 'window-scores.csv'),
                plan_sha256=_sha(context['plan_dir'] / 'plan.json'), kernel_sha256=plan['kernel_sha256'] if arm != 'A' else None,
                threshold=.5, aggregation='max', selection_used_dense=False, shared_B_C_description=True,
                canonical_digest_timing='single canonical stream; scores provisional until each video EOF',
                independent_test=False, input_source='actual dense decoded frames through unchanged jpeg95/RGB112 helper; no supervised-descriptor cache',
                seconds=time.perf_counter() - started)
            _json(directory / 'result.json', result); results[arm] = result
        return results
    except Exception as e:
        for directory in directories.values():
            _json(directory / 'failure.json', dict(status='dense_metrics_withheld', error_type=type(e).__name__, error=str(e), accepted_videos=list(evidence)))
        raise
    finally:
        for stream in streams.values():
            if not stream.closed:
                stream.close()
