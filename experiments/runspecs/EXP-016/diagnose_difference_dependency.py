"""One registered CPU-only D-zero diagnostic; never trains or decodes video."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[3]
sys.path.insert(0, str(ROOT))
ATOL = RTOL = 1e-4
SECONDS_LIMIT = 1800


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def zero_difference(x):
    import torch
    if (x.ndim != 5 or x.shape[1] != 6 or x.shape[2] != 16
            or x.dtype != torch.float32 or x.device.type != 'cpu' or not torch.isfinite(x).all()):
        raise ValueError('Expected finite CPU FP32 [N,6,16,H,W]')
    y = x.clone()
    y[:, 3:].zero_()
    if not torch.equal(y[:, :3], x[:, :3]) or y[:, 3:].count_nonzero():
        raise RuntimeError('Intervention changed RGB or retained D')
    return y


def probability(z):
    if z >= 0:
        return 1/(1+math.exp(-z))
    exp = math.exp(z)
    return exp/(1+exp)


def bce(z, label):
    return max(z, 0)-z*label+math.log1p(math.exp(-abs(z)))


def summarize(rows):
    import numpy as np
    if not rows:
        raise ValueError('Empty diagnostic summary')
    result = dict(clips=len(rows), P=sum(r['label'] for r in rows))
    result['N'] = len(rows)-result['P']
    for prefix in ('normal', 'zero'):
        tp = sum(r['label'] == 1 and r[prefix+'_logit'] >= 0 for r in rows)
        fp = sum(r['label'] == 0 and r[prefix+'_logit'] >= 0 for r in rows)
        result[prefix] = dict(mean_bce=float(np.mean([r[prefix+'_bce'] for r in rows])),
                              mean_probability=float(np.mean([r[prefix+'_probability'] for r in rows])),
                              TP=tp, FP=fp, FN=result['P']-tp, TN=result['N']-fp)
    for name in ('logit', 'probability', 'bce'):
        values = np.array([r['delta_'+name] for r in rows])
        result['delta_'+name] = dict(mean=float(values.mean()), mean_abs=float(np.abs(values).mean()),
                                    min=float(values.min()), max=float(values.max()),
                                    q10=float(np.quantile(values, .1)), median=float(np.median(values)),
                                    q90=float(np.quantile(values, .9)))
    return result


def main():
    began = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=ROOT/'data/results/exp016/paired-s8-20261031-v1')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    run.relative_to(ROOT/'data/results/exp016')
    out.relative_to(ROOT/'data/results/exp016')
    out.mkdir(parents=True, exist_ok=False)

    def timeout():
        time.sleep(max(0, SECONDS_LIMIT-(time.monotonic()-began)))
        try:
            write_json(out/'failure.json', dict(status='time_limit_reached', seconds_limit=SECONDS_LIMIT))
        finally:
            os._exit(124)
    threading.Thread(target=timeout, daemon=True).start()
    write_json(out/'launch.json', dict(argv=sys.argv, seconds_limit=SECONDS_LIMIT, device='cpu',
        logit_atol=ATOL, logit_rtol=RTOL, intervention='normal D minus zero D',
        training=False, video_decoding=False, external_test=False, authorization='user: ok，那你继续做吧'))
    try:
        import numpy as np
        import torch
        import torchvision
        import cv2
        from experiments.common.video.supervised_plan import digest
        from experiments.common.video.supervised_training import load_inputs
        from experiments.common.video.supervised_runtime import ClipDataset, ClipInput, EpochBatches
        from experiments.common.video.r2plus1d_classifier import VideoClassifier, configure_fp32, state_digest
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        configure_fp32(8)
        plan_dir = ROOT/'data/metadata/EXP-016/training-plan-v2'
        spec = ROOT/'docs/process/experiments/EXP-016-D通道依赖诊断.md'
        sources = [SOURCE, spec, plan_dir/'plan.json', plan_dir/'assignments.csv',
                   run/'audit-v1/audit.json', run/'diff/profile.json', run/'diff/training-result.json',
                   run/'diff/validation-epoch-09.json']
        sources += [ROOT/'experiments/common/video'/name for name in
                    ('supervised_plan.py', 'supervised_training.py', 'supervised_runtime.py',
                     'r2plus1d_classifier.py', 'pts_clip_sampling.py')]
        hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}
        (out/'generator-source.py').write_bytes(SOURCE.read_bytes())
        audit = json.loads((run/'audit-v1/audit.json').read_text(encoding='utf-8'))
        training = json.loads((run/'diff/training-result.json').read_text(encoding='utf-8'))
        path = run/'diff/best.pt'
        expected_sha = audit['sources_sha256'][path.relative_to(ROOT).as_posix()]
        if (audit['status'] != 'paired_artifacts_audited' or audit['seed'] != 8
                or training['selected_checkpoint']['sha256'] != expected_sha or digest(path) != expected_sha
                or training['completed_epochs'] != 20 or training['optimizer_steps'] != 420):
            raise ValueError('Audited selected seed8 checkpoint required')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        profile = checkpoint['profile']
        if (profile != training['profile'] or profile['seed'] != 8 or profile['mode'] != 'rgb_diff'
                or profile['smoke'] or profile['precision'] != 'FP32' or profile['microbatch'] != 8
                or profile['effective_batch'] != 24 or checkpoint['completed_epochs'] != 9
                or training['best_epoch'] != 9 or profile['plan_sha256'] != digest(plan_dir/'plan.json')):
            raise ValueError('Checkpoint configuration or selection differs')
        net = VideoClassifier(pretrained=False).eval()
        net.load_state_dict(checkpoint['state_dict'])
        del checkpoint
        if any(module.training for module in net.modules()):
            raise RuntimeError('All modules must be in eval mode')
        before = state_digest(net)
        plan, bank, _, validation = load_inputs(ROOT, plan_dir)
        selected = [bank[i] for i in validation]
        stored_path = run/'diff/validation-epoch-09.json'
        if digest(stored_path) != audit['sources_sha256'][stored_path.relative_to(ROOT).as_posix()]:
            raise ValueError('Reference predictions changed since audit')
        stored = json.loads(stored_path.read_text(encoding='utf-8'))
        if (len(selected) != 136 or sum(r[2] for r in selected) != 96
                or [(r[0], r[2]) for r in selected] != [(r['clip_id'], r['label']) for r in stored]):
            raise ValueError('Frozen reference identities, order or labels differ')
        clip_ids = {r[0] for r in selected}
        details = {}
        with (ROOT/plan['resource']/'windows.jsonl').open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                if row['clip_id'] in clip_ids:
                    details[row['clip_id']] = row
        ds = ClipDataset(selected, cache_bytes=8*1024**2)
        pixels = np.empty((136, 16, 112, 112, 3), dtype=np.uint8)
        identities = []
        for i, (cid, frames, label) in enumerate(selected):
            pixels[i] = ds[i][0]
            identities.append(dict(clip_id=cid, label=label, unit_id=details[cid]['unit_id'],
                                   frame_indices=details[cid]['frame_indices'],
                                   input_sha256=hashlib.sha256(pixels[i].tobytes()).hexdigest(),
                                   jpeg_sha256=[r.jpeg_sha256 for r in frames]))
        del ds, bank
        write_json(out/'inputs.json', identities)
        effective = EpochBatches(24)
        effective.set_order(range(136))
        batches = [list(batch[i:i+8]) for batch in effective for i in range(0, len(batch), 8)]
        transform = ClipInput('rgb_diff').eval()
        normal, zero, rgb_hashes = [], [], []
        with torch.inference_mode():
            for ids in batches:
                x = transform(torch.from_numpy(pixels[ids]))
                rgb_hashes.append(hashlib.sha256(x[:, :3].contiguous().numpy().tobytes()).hexdigest())
                logits = net(x)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite normal logit')
                normal.extend(logits.tolist())
                print(json.dumps(dict(stage='normal_reproduction', clips=len(normal), total=136)), flush=True)
            reference = np.array([r['logit'] for r in stored])
            error = np.abs(np.array(normal)-reference)
            failed = error > ATOL+RTOL*np.abs(reference)
            proof = dict(status='passed' if not failed.any() else 'failed', atol=ATOL, rtol=RTOL,
                         max_abs_logit_error=float(error.max()), mean_abs_logit_error=float(error.mean()),
                         failed_clip_ids=[stored[i]['clip_id'] for i in np.flatnonzero(failed)],
                         stored_sha256=digest(stored_path), normal_logits=normal)
            write_json(out/'reproduction.json', proof)
            if failed.any():
                raise ValueError('Normal input fails preregistered reproduction tolerance; intervention not run')
            for ids, rgb_sha in zip(batches, rgb_hashes):
                x = transform(torch.from_numpy(pixels[ids]))
                y = zero_difference(x)
                if hashlib.sha256(y[:, :3].contiguous().numpy().tobytes()).hexdigest() != rgb_sha:
                    raise RuntimeError('RGB differs between normal and zero passes')
                logits = net(y)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite zero-D logit')
                zero.extend(logits.tolist())
                print(json.dumps(dict(stage='zero_difference', clips=len(zero), total=136)), flush=True)
        if len(normal) != 136 or len(zero) != 136 or before != state_digest(net) or digest(path) != expected_sha:
            raise RuntimeError('Coverage or immutable model/checkpoint check failed')
        if any(digest(ROOT/name) != value for name, value in hashes.items()):
            raise RuntimeError('Diagnostic source or frozen input changed during execution')
        paired = []
        for identity, original, z0, saved in zip(identities, normal, zero, stored):
            row = {k: identity[k] for k in ('clip_id', 'unit_id', 'label')}
            row['stored_cloud_logit'] = saved['logit']
            for prefix, z in [('normal', original), ('zero', z0)]:
                row.update({prefix+'_logit': z, prefix+'_probability': probability(z),
                            prefix+'_bce': bce(z, row['label'])})
            for key in ('logit', 'probability', 'bce'):
                row['delta_'+key] = row['normal_'+key]-row['zero_'+key]
            paired.append(row)
        with (out/'paired-clips.csv').open('x', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
        result = dict(status='difference_dependency_diagnostic_complete', seconds=time.monotonic()-began,
            scope='136 fixed supervised validation clips; not complete-video or independent-test metrics',
            delta_direction='normal D minus zero D', normal_reproduction=proof,
            checkpoint_sha256=expected_sha, model_state_sha256=before, model_state_unchanged=True,
            sources_sha256=hashes, input_manifest_sha256=digest(out/'inputs.json'),
            paired_predictions_sha256=digest(out/'paired-clips.csv'),
            environment=dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__,
                             opencv=cv2.__version__, numpy=np.__version__, device='cpu', threads=2, microbatch=8),
            all_clips=summarize(paired), by_class={str(y): summarize([r for r in paired if r['label'] == y]) for y in (0, 1)},
            by_video_fixed_clips={uid: summarize([r for r in paired if r['unit_id'] == uid]) for uid in sorted({r['unit_id'] for r in paired})},
            optimizer_steps=0, video_decoding=False, external_test_used=False,
            registered_training_replication_condition_changed=False,
            limitations=['D removal changes input distribution; dependence is not correct motion understanding',
                         'Zero-D fused checkpoint is not the separately trained RGB baseline',
                         'No effect would not exclude training-time influence of D; seed7 failure not explained'])
        write_json(out/'result.json', result)
        print(json.dumps(dict(status=result['status'], seconds=result['seconds'], by_class=result['by_class'])), flush=True)
    except Exception as error:
        write_json(out/'failure.json', dict(status='diagnostic_failed', error_type=type(error).__name__,
                                           error=str(error), seconds=time.monotonic()-began))
        raise


if __name__ == '__main__':
    main()
