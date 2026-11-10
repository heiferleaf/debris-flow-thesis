"""Bounded, read-only EXP-017 training-exposure replay; never trains/decodes video.

The prepare phase reads audited metadata only. The replay supervisor permits one
CPU worker, persists each accepted window, and kills it after at most 900 seconds.
Panel metrics describe this deterministic training panel, not full training risk
or generalization. No result from this module authorizes another training run.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

AUDIT_SHA = '4b94b4094336c99bf35e675bd07a72420498fb97bd56f8ee366a75a2b2971c45'
DEFAULT_AUDIT = 'data/results/exp017/audit-resume-cv-s7-20261102-v2-v2/audit.json'
CORE_FILES = tuple('experiments/common/video/' + n for n in (
    'supervised_runtime.py', 'r2plus1d_classifier.py', 'pts_clip_sampling.py'))
SELECTION = dict(version='exp017-training-panel-v2', seed=7, target_per_class=32,
                 max_per_video_class=2, physical_key='unit_id + ordered 16 frame indices',
                 order='SHA256(canonical JSON [version,seed,fold,label,kind,identity])',
                 priority='cover each candidate group; global unused-video group round robin before any second window',
                 scores_used=False, epochs='actual completed fixed_outer epochs only')
LAYER_WIDTHS = dict(backbone512=512, linear128=128, relu128=128,
                    linear64=64, relu64=64, linear32=32, relu32=32, logit=1)
NARROW_FOLDS = (1, 2, 5)
EPSILON = 1e-12
AUC_SCORE_KIND = 'FP32_logit'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      separators=(',', ':'), allow_nan=False).encode()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value, *, replace=False):
    path = Path(path)
    if replace:
        temp = path.with_name(path.name + '.tmp')
        with temp.open('w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    else:
        with path.open('x', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')


def rooted(root, relative):
    path = (Path(root) / relative).resolve()
    path.relative_to(Path(root).resolve())
    return path


def exposure_candidates(inputs, history, completed_epochs):
    """Use actual exposure logs; reject the unused tail of the twenty-row plan."""
    if (type(completed_epochs) is not int or not 1 <= completed_epochs <= 20
            or len(history) != completed_epochs or len(inputs['epoch_clip_ids']) != 20
            or inputs['validation_clip_ids'] is not None):
        raise ValueError('Require actual fixed-outer exposure length and no validation')
    bank = inputs['clip_bank']
    physical = {}
    train_counts = None
    for epoch, recorded in enumerate(history, 1):
        ids = inputs['epoch_clip_ids'][epoch - 1]
        counts = Counter(bank[c]['label'] for c in ids)
        if (not ids or len(ids) != len(set(ids)) or recorded['completed_epochs'] != epoch
                or recorded['train_clip_ids'] != ids or recorded['samples'] != len(ids)
                or recorded['train_P'] != counts[1] or recorded['train_N'] != counts[0]
                or not counts[0] or not counts[1] or recorded['finite'] is not True):
            raise ValueError('Actual exposure and frozen schedule/counts differ')
        if train_counts is None:
            train_counts = counts
        elif counts != train_counts:
            raise ValueError('Per-epoch training class counts changed')
        for cid in ids:
            row = bank[cid]
            indices = row['frame_indices']
            if (type(row['label']) is not int or row['label'] not in (0, 1)
                    or len(indices) != 16 or any(type(i) is not int or i < 0 for i in indices)
                    or any(a >= b for a, b in zip(indices, indices[1:]))
                    or len(row['frames']) != 16 or not row['group_id'] or not row['unit_id']):
                raise ValueError('Invalid physical training window')
            key = (row['unit_id'], tuple(indices))
            if key not in physical:
                physical[key] = dict(clip_id=cid, unit_id=row['unit_id'], group_id=row['group_id'],
                    label=row['label'], frame_indices=indices, source_pts_seconds=row['source_pts_seconds'],
                    frames=row['frames'], exposure_epochs=[], exposure_clip_ids=[])
            item = physical[key]
            if any(item[k] != row[k] for k in ('label', 'group_id', 'frames', 'source_pts_seconds')):
                raise ValueError('Conflicting aliases of one physical window')
            if epoch not in item['exposure_epochs']:
                item['exposure_epochs'].append(epoch)
            if cid not in item['exposure_clip_ids']:
                item['exposure_clip_ids'].append(cid)
    return list(physical.values()), dict(P=train_counts[1], N=train_counts[0])


def selection_hash(fold, label, kind, identity):
    return hashlib.sha256(canonical([SELECTION['version'], 7, fold, label, kind, identity])).hexdigest()


def choose_panel(candidates, fold, target=32):
    """No model scores are consulted; class/group/video constraints precede hash."""
    chosen, coverage = [], {}
    for label in (0, 1):
        groups = defaultdict(list)
        for source in candidates:
            if source['label'] == label:
                row = dict(source)
                row['selection_sha256'] = selection_hash(fold, label, 'window',
                    [row['unit_id'], row['frame_indices']])
                groups[row['group_id']].append(row)
        order = sorted(groups, key=lambda g: selection_hash(fold, label, 'group', g))
        counts, selected, used = Counter(), [], set()
        def take(group, unused_only=False):
            eligible = [r for r in groups[group] if (r['unit_id'], tuple(r['frame_indices'])) not in used
                        and counts[r['unit_id']] < 2
                        and (not unused_only or counts[r['unit_id']] == 0)]
            if not eligible:
                return False
            row = min(eligible, key=lambda r: (counts[r['unit_id']] > 0, r['selection_sha256']))
            selected.append(row)
            used.add((row['unit_id'], tuple(row['frame_indices'])))
            counts[row['unit_id']] += 1
            return True
        # The first pass gives every eligible group an opportunity before any refill.
        for group in order:
            if len(selected) < target:
                take(group)
        while len(selected) < target:
            before = len(selected)
            # Exhaust every globally unused video before admitting a second
            # window from a small group. Group order still controls each pass.
            for group in order:
                if len(selected) < target:
                    take(group, unused_only=True)
            if len(selected) > before:
                continue
            for group in order:
                if len(selected) < target:
                    take(group)
            if len(selected) == before:
                break
        coverage[str(label)] = dict(candidate_windows=sum(map(len, groups.values())),
            candidate_groups=sorted(groups), selected_groups=sorted({r['group_id'] for r in selected}),
            uncovered_groups=sorted(set(groups) - {r['group_id'] for r in selected}),
            selected_windows=len(selected), selected_videos=len(counts),
            per_video_counts=dict(sorted(counts.items())), shortage=target-len(selected))
        chosen.extend(selected)
    keys = [(r['unit_id'], tuple(r['frame_indices'])) for r in chosen]
    if len(keys) != len(set(keys)):
        raise ValueError('Panel repeats one physical window')
    return chosen, coverage


def load_audit(root, audit_path=DEFAULT_AUDIT):
    path = rooted(root, audit_path)
    if sha(path) != AUDIT_SHA:
        raise ValueError('Require the frozen passed five-fold audit SHA')
    audit = read_json(path)
    if (audit['status'] != 'audit_passed' or len(audit['fits']) != 5
            or [r['fold'] for r in audit['fits']] != [1, 2, 3, 4, 5]):
        raise ValueError('Require complete audited five-fold fits')
    return audit, path


def prepare(root, out, audit_path=DEFAULT_AUDIT):
    """Metadata-only panel preparation. Does not import torch or read JPEG pixels."""
    root, out = Path(root).resolve(), Path(out).resolve()
    out.relative_to(root)
    if out.exists():
        raise FileExistsError(out)
    audit, audit_file = load_audit(root, audit_path)
    bindings = {audit_file.relative_to(root).as_posix(): AUDIT_SHA}
    folds = []
    for fit in audit['fits']:
        directory = rooted(root, fit['artifact_directory'])
        files = {}
        for name in ('inputs.json', 'epochs.jsonl', 'profile.json', 'training-result.json'):
            path = directory / 'outer' / name
            relative = path.relative_to(root).as_posix()
            expected = audit['source_sha256'][relative]
            if sha(path) != expected:
                raise ValueError('Audited outer artifact changed: ' + relative)
            files[name] = path
            bindings[relative] = expected
        result, profile, inputs = (read_json(files[n]) for n in ('training-result.json', 'profile.json', 'inputs.json'))
        history = [json.loads(s) for s in files['epochs.jsonl'].read_text(encoding='utf-8').splitlines()]
        E = fit['selected_epochs']
        if (result['status'] != 'fit_complete' or result['stage'] != 'fixed_outer'
                or result['completed_epochs'] != E or result['selected_epochs'] != E
                or result['fold'] != fit['fold'] or profile['stage'] != 'fixed_outer'
                or profile['fold'] != fit['fold'] or profile['requested_epochs'] != E
                or profile['mode'] != 'rgb_zero' or profile['seed'] != 7
                or profile['execution_kind'] != 'registered_fit'
                or inputs['validation_clip_ids'] is not None):
            raise ValueError('Wrong selected outer fit')
        execution_file = directory.parent / 'execution-inputs.json'
        execution_relative = execution_file.relative_to(root).as_posix()
        if sha(execution_file) != audit['source_sha256'][execution_relative]:
            raise ValueError('Audited execution closure changed')
        bindings[execution_relative] = audit['source_sha256'][execution_relative]
        execution = read_json(execution_file)
        for relative in CORE_FILES:
            expected = execution['code_sha256'][relative]
            snapshot = directory.parent / 'source-snapshot' / relative
            if sha(snapshot) != expected or sha(rooted(root, relative)) != expected:
                raise ValueError('Shared runtime must equal frozen execution source: ' + relative)
            bindings[relative] = expected
            bindings[snapshot.relative_to(root).as_posix()] = expected
        candidates, counts = exposure_candidates(inputs, history, E)
        if (counts['P'] != result['train_P'] or counts['N'] != result['train_N']
                or not math.isclose(result['pos_weight'], counts['N']/counts['P'], abs_tol=1e-12)
                or profile['input_fingerprints'] != result['input_fingerprints']
                or profile['parsed_inputs_sha256'] != result['parsed_inputs_sha256']):
            raise ValueError('Original training class weight/input identity differs')
        selected, coverage = choose_panel(candidates, fit['fold'])
        # Resolve via the original accepted cache index, never a filename search.
        cache_name = 'data/process/EXP-017/cv-inputs-v1/verified-cache-files.csv'
        expected_cache = result['input_fingerprints'][cache_name]
        if sha(rooted(root, cache_name)) != expected_cache:
            raise ValueError('Frozen accepted cache index changed')
        bindings[cache_name] = expected_cache
        wanted = {(r['unit_id'], i) for r in selected for i in r['frame_indices']}
        index = {}
        with rooted(root, cache_name).open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                key = row['unit_id'], int(row['frame_index'])
                if key in wanted:
                    if key in index:
                        raise ValueError('Duplicate selected accepted FrameRef')
                    index[key] = row
        if set(index) != wanted:
            raise ValueError('Selected existing JPEG lacks accepted FrameRef')
        for row in selected:
            row['fold'] = fit['fold']
            row['physical_sha256'] = hashlib.sha256(canonical([row['unit_id'], row['frame_indices']])).hexdigest()
            for i, ref in zip(row['frame_indices'], row['frames']):
                accepted = index[(row['unit_id'], i)]
                local = rooted(root, accepted['cache_path'])
                # The stored cloud path and accepted relative path must share the
                # complete cache suffix; this is explicit relocation, not guessing.
                suffix = '/' + accepted['cache_path'].replace('\\', '/')
                if (not ref['path'].replace('\\', '/').endswith(suffix)
                        or any(ref[k] != accepted[k] for k in ('jpeg_sha256', 'rgb112_sha256'))
                        or not local.is_file() or local.stat().st_size != int(accepted['size_bytes'])):
                    raise ValueError('Selected FrameRef/accepted local JPEG differs')
                ref['original_path'] = ref['path']
                ref['path'] = accepted['cache_path']
        checkpoint = directory / 'outer/selected.pt'
        checkpoint_relative = checkpoint.relative_to(root).as_posix()
        expected_checkpoint = audit['source_sha256'][checkpoint_relative]
        if (result['selected_checkpoint']['filename'] != 'selected.pt'
                or result['selected_checkpoint']['sha256'] != expected_checkpoint
                or result['selected_checkpoint']['completed_epochs'] != E):
            raise ValueError('Audited selected checkpoint binding differs')
        folds.append(dict(fold=fit['fold'], completed_epochs=E, actual_exposure_windows=len(candidates),
            training_counts=counts, original_pos_weight=result['pos_weight'],
            checkpoint_path=checkpoint_relative, checkpoint_sha256=expected_checkpoint,
            final_state_sha256=result['final_state_sha256'], profile=profile,
            training_result_path=files['training-result.json'].relative_to(root).as_posix(),
            coverage=coverage, panel_quota_complete=all(c['shortage'] == 0 and not c['uncovered_groups']
                                                      for c in coverage.values()), windows=selected))
        del inputs, candidates
    manifest = dict(status='prepared_no_model_execution', scope='deterministic actual-training-exposure panel only',
        audit_path=audit_file.relative_to(root).as_posix(), audit_sha256=AUDIT_SHA,
        selection=SELECTION, folds=folds, source_sha256=bindings,
        implementation_sha256={n: sha(root/n) for n in ('experiments/common/video/training_replay_diagnostic.py',
                                                      'experiments/runspecs/EXP-017/diagnose_fit.py')},
        pixel_check='Existing accepted FrameRef hashes; JPEG and RGB112 rechecked by shared PixelCache at replay',
        pixel_relocation='Original audited cloud FrameRef -> hash-matched accepted cache index relative path',
        model_execution=False, training=False, video_decoding=False, paid_compute=False,
        max_seconds=900, device='cpu', precision='FP32', threads=4)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'manifest.json', manifest)
    return manifest


def bce(logit, label):
    return max(logit, 0)-label*logit+math.log1p(math.exp(-abs(logit)))


def probability(logit):
    z = math.exp(-abs(logit))
    return 1/(1+z) if logit >= 0 else z/(1+z)


def constant_reference(P, N, weight):
    if P <= 0 or N <= 0 or not math.isfinite(weight) or weight <= 0:
        return dict(q_star=None, optimum_weighted_bce=None, q_half_weighted_bce=None)
    q = weight*P/(weight*P+N)
    return dict(q_star=q,
        optimum_weighted_bce=-(weight*P*math.log(q)+N*math.log1p(-q))/(P+N),
        q_half_weighted_bce=(weight*P+N)*math.log(2)/(P+N))


def panel_metrics(rows, weight):
    import numpy as np
    if not rows:
        return dict(P=0, N=0, AUC=None, negative_Recall=None, positive_Recall=None,
                    AUC_score_kind=AUC_SCORE_KIND,
                    per_class={}, equal_class_bce=None, original_weight_panel_bce=None,
                    constant_reference=constant_reference(0, 0, weight))
    if any(r['label'] not in (0, 1) or not all(math.isfinite(r[k]) for k in ('logit', 'score')) for r in rows):
        raise FloatingPointError('Nonfinite/nonbinary panel predictions')
    positive = [r for r in rows if r['label'] == 1]
    negative = [r for r in rows if r['label'] == 0]
    P, N = len(positive), len(negative)
    TP = sum(r['score'] >= .5 for r in positive)
    FP = sum(r['score'] >= .5 for r in negative)
    per_class = {}
    for label, subset in ((0, negative), (1, positive)):
        if subset:
            per_class[str(label)] = dict(count=len(subset),
                bce=math.fsum(bce(r['logit'], label) for r in subset)/len(subset),
                probability_quantiles={str(q): float(np.quantile([r['score'] for r in subset], q))
                                       for q in (0, .1, .25, .5, .75, .9, 1)})
    # Compare logits rather than rounded/saturated probabilities: sigmoid is monotone.
    auc = (math.fsum((p['logit'] > n['logit']) + .5*(p['logit'] == n['logit'])
                     for p in positive for n in negative)/(P*N)) if P and N else None
    return dict(P=P, N=N, TP=TP, FP=FP, FN=P-TP, TN=N-FP, AUC=auc,
        AUC_score_kind=AUC_SCORE_KIND,
        AUC_rank_input='FP32 logit (monotone probability rank, avoids sigmoid saturation ties)',
        negative_Recall=(N-FP)/N if N else None, positive_Recall=TP/P if P else None,
        per_class=per_class,
        equal_class_bce=(per_class['0']['bce']+per_class['1']['bce'])/2 if P and N else None,
        original_weight_panel_bce=math.fsum(bce(r['logit'], r['label'])*(weight if r['label'] else 1)
                                           for r in rows)/(P+N),
        constant_reference=constant_reference(P, N, weight))


class LayerStatistics:
    """FP64 Welford sufficient statistics, without saving feature tensors."""
    def __init__(self):
        self.n = 0
        self.mean = self.m2 = self.sum_squares = self.zero_counts = None
        self.norms = []

    def add(self, value):
        import numpy as np
        x = np.asarray(value, dtype=np.float64).reshape(-1)
        if not x.size or not np.isfinite(x).all():
            raise FloatingPointError('Empty/nonfinite activation')
        if self.mean is None:
            self.mean = np.zeros_like(x)
            self.m2 = np.zeros_like(x)
            self.sum_squares = np.zeros_like(x)
            self.zero_counts = np.zeros(x.size, dtype=np.int64)
        if x.shape != self.mean.shape:
            raise ValueError('Layer width changed')
        self.n += 1
        delta = x-self.mean
        self.mean += delta/self.n
        self.m2 += delta*(x-self.mean)
        self.sum_squares += x*x
        self.zero_counts += x == 0
        self.norms.append(float(np.linalg.norm(x)))
        if not all(np.isfinite(a).all() for a in (self.mean, self.m2, self.sum_squares, self.norms)):
            raise FloatingPointError('Nonfinite FP64 sufficient statistic')

    def summary(self):
        import numpy as np
        if not self.n:
            return dict(samples=0)
        total = float(self.sum_squares.sum(dtype=np.float64))
        centered = max(0., float(self.m2.sum(dtype=np.float64)))
        return dict(samples=self.n, units=len(self.mean), accumulator_dtype='float64', epsilon=EPSILON,
            centered_sum_squares=centered, response_sum_squares=total,
            relative_sample_variation=centered/(total+EPSILON), all_zero=total == 0,
            sample_l2=dict(mean=float(np.mean(self.norms)), min=min(self.norms), max=max(self.norms)),
            zero_fraction=float(self.zero_counts.sum())/(self.n*len(self.mean)),
            always_zero_unit_fraction=float(np.mean(self.zero_counts == self.n)),
            interpretation='Descriptive sample variation/norm/zero statistics, not information loss rate')


def frozen_route(folds):
    """Apply the pre-replay rule only to five technically complete 32P/32N panels."""
    if len(folds) != 5 or {r['fold'] for r in folds} != {1, 2, 3, 4, 5}:
        return 'undetermined'
    for fold in folds:
        metrics = fold.get('metrics', {})
        values = [metrics.get('AUC'), metrics.get('negative_Recall')]
        if (fold.get('status') != 'complete' or fold.get('panel_quota_complete') is not True
                or metrics.get('P') != 32 or metrics.get('N') != 32
                or fold.get('scored_windows') != fold.get('selected_windows')
                or fold.get('selected_windows') != 64
                or any(v is None or not math.isfinite(v) for v in values)):
            return 'undetermined'
    if (all(f['metrics']['AUC'] >= .8 for f in folds)
            and sum(f['metrics']['negative_Recall'] <= .1 for f in folds if f['fold'] in NARROW_FOLDS) >= 2):
        return 'calibration_candidate'
    return 'linear_head_candidate'


def score_one(net, transform, u8):
    """Exactly one CPU/FP32/eval/no_grad forward and the required eight hooks."""
    import numpy as np
    import torch
    from torch import nn
    if any(m.training for m in net.modules()) or transform.training:
        raise ValueError('Every replay module must be in eval mode')
    for tensor in (*net.parameters(), *net.buffers(), *transform.buffers()):
        if (tensor.device.type != 'cpu' or tensor.is_floating_point() and tensor.dtype != torch.float32
                or tensor.is_floating_point() and not torch.isfinite(tensor).all()):
            raise ValueError('Require finite CPU FP32 model and preprocessing')
    modules = [('backbone512', net.backbone)]
    for name, index, cls in (('linear128', 0, nn.Linear), ('relu128', 1, nn.ReLU),
                             ('linear64', 3, nn.Linear), ('relu64', 4, nn.ReLU),
                             ('linear32', 6, nn.Linear), ('relu32', 7, nn.ReLU), ('logit', 9, nn.Linear)):
        if not isinstance(net.head[index], cls):
            raise ValueError('Frozen classifier head contract differs')
        modules.append((name, net.head[index]))
    captured, handles = {}, []
    def capture(name):
        def hook(_module, _inputs, output):
            if name in captured or output.shape != (1, LAYER_WIDTHS[name]):
                raise ValueError('Repeated hook or unexpected layer width')
            captured[name] = output.detach().cpu().numpy().astype(np.float64, copy=True).reshape(-1)
        return hook
    try:
        for name, module in modules:
            handles.append(module.register_forward_hook(capture(name)))
        with torch.no_grad():
            x = transform(u8)
            if x.device.type != 'cpu' or x.dtype != torch.float32 or not torch.isfinite(x).all():
                raise ValueError('Invalid shared ClipInput output')
            output = net(x)
        if (set(captured) != set(LAYER_WIDTHS) or output.numel() != 1
                or not torch.isfinite(output).all() or any(not np.isfinite(v).all() for v in captured.values())
                or output.requires_grad or any(p.grad is not None for p in net.parameters())):
            raise FloatingPointError('Incomplete/nonfinite/gradient-bearing replay')
        logit = float(output.reshape(-1)[0])
        if logit != float(captured['logit'][0]):
            raise ValueError('Final hook and classifier output differ')
        return logit, captured, hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
    finally:
        for handle in handles:
            handle.remove()


def verify_manifest(root, manifest):
    if (manifest['audit_sha256'] != AUDIT_SHA or manifest['selection'] != SELECTION
            or manifest['max_seconds'] != 900 or manifest['device'] != 'cpu'
            or manifest['precision'] != 'FP32' or manifest['threads'] != 4):
        raise ValueError('Prepared contract differs')
    load_audit(root, manifest['audit_path'])
    for name, expected in {**manifest['source_sha256'], **manifest['implementation_sha256']}.items():
        if sha(rooted(root, name)) != expected:
            raise ValueError('Frozen preparation source changed: ' + name)
    if [f['fold'] for f in manifest['folds']] != [1, 2, 3, 4, 5]:
        raise ValueError('Five original outer models required')
    for fold in manifest['folds']:
        rows = fold['windows']
        counts = Counter((r['unit_id'], r['label']) for r in rows)
        keys = {(r['unit_id'], tuple(r['frame_indices'])) for r in rows}
        if (len(rows) > 64 or len(keys) != len(rows) or any(n > 2 for n in counts.values())
                or any(sum(r['label'] == y for r in rows) > 32 for y in (0, 1))):
            raise ValueError('Prepared panel exceeds physical/class/video quota')


def worker(root, prepared, nonce):
    """Private worker. Only the supervisor may create launch.json and invoke it."""
    started = time.monotonic()
    prepared = Path(prepared)
    launch = read_json(prepared/'launch.json')
    if launch['nonce'] != nonce or sha(prepared/'manifest.json') != launch['manifest_sha256']:
        raise ValueError('Supervisor launch binding differs')
    # Exclusive claim prohibits manual or accidental replay of this worker.
    with (prepared/'worker-claimed').open('x') as f:
        f.write(nonce)
    # A second guard also bounds the real interpreter behind a Windows venv
    # launcher, whose Popen PID alone may otherwise leave a descendant alive.
    timer = threading.Timer(max(0., launch['deadline_monotonic']-time.monotonic()), lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    import numpy as np
    import torch
    import torchvision
    import cv2
    from experiments.common.video.supervised_runtime import ClipDataset, ClipInput, FrameRef
    from experiments.common.video.r2plus1d_classifier import VideoClassifier, configure_fp32, state_digest
    manifest = read_json(prepared/'manifest.json')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    configure_fp32(7)
    verify_manifest(root, manifest)
    progress = dict(status='partial', folds=[],
        environment=dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__,
                         numpy=np.__version__, opencv=cv2.__version__, device='cpu', precision='FP32', threads=4),
        seconds=0, optimizer_created=False, gradient_updates=0, video_decoding=False, paid_compute=False)
    write_json(prepared/'progress.json', progress)
    with (prepared/'window-scores.jsonl').open('x', encoding='utf-8') as stream:
        for fold in manifest['folds']:
            current = dict(fold=fold['fold'], status='partial', panel_quota_complete=fold['panel_quota_complete'],
                selected_windows=len(fold['windows']), scored_windows=0, completed_epochs=fold['completed_epochs'],
                checkpoint_sha256=fold['checkpoint_sha256'], original_pos_weight=fold['original_pos_weight'],
                coverage=fold['coverage'], layers={}, metrics=panel_metrics([], fold['original_pos_weight']))
            progress['folds'].append(current)
            write_json(prepared/'progress.json', progress, replace=True)
            checkpoint_path = rooted(root, fold['checkpoint_path'])
            if sha(checkpoint_path) != fold['checkpoint_sha256']:
                raise ValueError('Audited selected checkpoint bytes changed')
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
            if (checkpoint['profile'] != fold['profile'] or checkpoint['completed_epochs'] != fold['completed_epochs']
                    or checkpoint['state_sha256'] != fold['final_state_sha256']):
                raise ValueError('Checkpoint tensor/profile/epoch binding differs')
            net = VideoClassifier(pretrained=False).cpu().float().eval()
            net.load_state_dict(checkpoint['state_dict'])
            del checkpoint
            net.requires_grad_(False)
            before = state_digest(net)
            if before != fold['final_state_sha256']:
                raise ValueError('Checkpoint tensors differ from original fit')
            transform = ClipInput('rgb_zero').cpu().float().eval()
            dataset = ClipDataset([(r['physical_sha256'], tuple(FrameRef(str(rooted(root, f['path'])),
                f['jpeg_sha256'], f['rgb112_sha256']) for f in r['frames']), r['label']) for r in fold['windows']],
                cache_bytes=8*1024**2)
            layer_stats = {k: LayerStatistics() for k in LAYER_WIDTHS}
            scored = []
            for index, row in enumerate(fold['windows']):
                try:
                    pixels, label, _ = dataset[index]
                    logit, captured, input_sha = score_one(net, transform, torch.from_numpy(pixels[None]))
                    record = dict(fold=fold['fold'], physical_sha256=row['physical_sha256'], clip_id=row['clip_id'],
                        unit_id=row['unit_id'], group_id=row['group_id'], label=label, logit=logit,
                        score=probability(logit), bce=bce(logit, label),
                        rgb112_clip_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                        model_input_fp32_sha256=input_sha)
                    for name, values in captured.items():
                        layer_stats[name].add(values)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    scored.append(record)
                    current.update(scored_windows=len(scored), metrics=panel_metrics(scored, fold['original_pos_weight']),
                                   layers={k: v.summary() for k, v in layer_stats.items()})
                    progress['seconds'] = time.monotonic()-started
                    write_json(prepared/'progress.json', progress, replace=True)
                except Exception as error:
                    current['error'] = dict(window=index, physical_sha256=row['physical_sha256'],
                                           type=type(error).__name__, message=str(error))
                    write_json(prepared/'progress.json', progress, replace=True)
                    raise
            if state_digest(net) != before or sha(checkpoint_path) != fold['checkpoint_sha256']:
                raise ValueError('Read-only model/checkpoint invariant failed')
            current.update(status='complete', model_state_unchanged=True)
            write_json(prepared/'progress.json', progress, replace=True)
            del net, transform, dataset, layer_stats
    verify_manifest(root, manifest)
    progress.update(status='complete', seconds=time.monotonic()-started)
    write_json(prepared/'progress.json', progress, replace=True)


def accepted_prefix(prepared, manifest, progress):
    """Reconcile durable score rows after a kill between score/progress writes."""
    if 'folds' not in manifest:  # CPU supervisor timeout fixture has no panels.
        return progress.get('folds', []), None
    expected = {(f['fold'], r['physical_sha256']): r for f in manifest['folds'] for r in f['windows']}
    scores, seen, issue = defaultdict(list), set(), None
    score_path = Path(prepared)/'window-scores.jsonl'
    if score_path.exists():
        with score_path.open('rb') as stream:
            for number, raw in enumerate(stream, 1):
                try:
                    # Decode inside the recovery boundary: a hard kill may
                    # split a multibyte group name in the final journal row.
                    row = json.loads(raw.decode('utf-8'))
                    key = row['fold'], row['physical_sha256']
                    source = expected[key]
                    if (key in seen or any(row[k] != source[k] for k in ('label', 'unit_id', 'group_id'))
                            or not all(math.isfinite(row[k]) for k in ('logit', 'score', 'bce'))
                            or row['score'] != probability(row['logit']) or row['bce'] != bce(row['logit'], row['label'])):
                        raise ValueError('Invalid or repeated durable score')
                    seen.add(key)
                    scores[row['fold']].append(row)
                except (ValueError, KeyError, TypeError) as error:
                    issue = dict(type='durable_score_prefix_error', line=number, message=str(error))
                    break
    previous = {f['fold']: f for f in progress.get('folds', [])}
    result = []
    for fold in manifest['folds']:
        current = dict(previous.get(fold['fold'], dict(fold=fold['fold'], status='partial', layers={})))
        rows = scores[fold['fold']]
        current.update(panel_quota_complete=fold['panel_quota_complete'], selected_windows=len(fold['windows']),
                       scored_windows=len(rows), metrics=panel_metrics(rows, fold['original_pos_weight']),
                       original_pos_weight=fold['original_pos_weight'], coverage=fold['coverage'])
        # A kill can preserve one extra score before its layer summary snapshot.
        # Report both counts and keep the fold partial in that case.
        current['layer_summary_windows'] = current.get('layers', {}).get('backbone512', {}).get('samples', 0)
        if (len(rows) != len(fold['windows']) or current['layer_summary_windows'] != len(rows)
                or not current.get('model_state_unchanged') or issue):
            current['status'] = 'partial'
        result.append(current)
    return result, issue


def replay(root, prepared, entrypoint, *, seconds=900, expected_manifest_sha256=None):
    """One subprocess, hard total deadline, no automatic retry or larger panel."""
    if type(seconds) is not int or not 1 <= seconds <= 900:
        raise ValueError('Replay deadline must be <=900 seconds')
    root, prepared = Path(root).resolve(), Path(prepared).resolve()
    prepared.relative_to(root)
    manifest = read_json(prepared/'manifest.json')
    actual_manifest_sha = sha(prepared/'manifest.json')
    if expected_manifest_sha256 is not None and actual_manifest_sha != expected_manifest_sha256:
        raise ValueError('Prepared manifest differs from independently reviewed SHA')
    nonce = uuid.uuid4().hex
    launch = dict(nonce=nonce, manifest_sha256=actual_manifest_sha, seconds_limit=seconds,
        deadline_monotonic=time.monotonic()+seconds,
        argv=sys.argv, device='cpu', precision='FP32', threads=4, training=False, paid_compute=False,
        no_retry=True, model_forwards_max=320)
    # A previous launch (including failure/timeout) is a permanent refusal to rerun.
    write_json(prepared/'launch.json', launch)
    began = time.monotonic()
    timed_out, returncode, error = False, None, None
    with (prepared/'worker.log').open('x', encoding='utf-8') as log:
        try:
            process = subprocess.Popen([sys.executable, str(entrypoint), '_worker', '--prepared', str(prepared),
                                        '--nonce', nonce], cwd=root, stdout=log, stderr=subprocess.STDOUT)
            try:
                returncode = process.wait(timeout=max(.001, seconds-(time.monotonic()-began)))
            except subprocess.TimeoutExpired:
                timed_out = True
                if os.name == 'nt':
                    # Windows venv python.exe can be a launcher with a child.
                    # Terminate the whole owned process tree, without a shell.
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   stdout=log, stderr=subprocess.STDOUT, timeout=10, check=False)
                else:
                    process.kill()
                returncode = process.wait(timeout=10)
        except Exception as caught:
            error = dict(type=type(caught).__name__, message=str(caught))
    progress = read_json(prepared/'progress.json') if (prepared/'progress.json').exists() else dict(folds=[])
    folds, prefix_error = accepted_prefix(prepared, manifest, progress)
    error = error or prefix_error
    if returncode not in (0, None) and not timed_out and error is None:
        error = dict(type='worker_failure', message='See worker.log and fold/window error; no retry')
    complete = (returncode == 0 and not timed_out and not error and progress.get('status') == 'complete'
                and len(folds) == 5 and all(f['status'] == 'complete' for f in folds))
    result = dict(status='complete' if complete else 'partial', seconds=time.monotonic()-began,
        seconds_limit=seconds, worker_returncode=returncode, timed_out=timed_out, error=error,
        manifest_sha256=launch['manifest_sha256'], audit_sha256=AUDIT_SHA,
        scope='Selected training-exposure panel only; neither full training risk nor generalization',
        folds=folds, environment=progress.get('environment'),
        routing=frozen_route(folds) if complete else 'undetermined',
        routing_rule=dict(AUC_score_kind=AUC_SCORE_KIND, all_folds_AUC_minimum=.8,
                          narrow_folds_negative_Recall_maximum=.1, narrow_folds_required=2),
        frozen_narrow_folds=list(NARROW_FOLDS), threshold=.5, positive_rule='score >= 0.5',
        negative_Recall_definition='TN/(TN+FP)',
        model_execution=True, optimizer_created=False, gradient_updates=0, video_decoding=False,
        paid_compute=False, training_authorized=False, automatic_retry=False,
        partial_policy='Preserve accepted prefix; no low-AUC inference, panel extension or automatic rerun',
        layer_interpretation='Descriptive variation, norms and ReLU sparsity; not information loss rate')
    write_json(prepared/'result.json', result)
    return result
