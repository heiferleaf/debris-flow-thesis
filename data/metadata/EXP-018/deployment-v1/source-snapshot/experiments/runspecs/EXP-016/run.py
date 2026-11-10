"""Local/cloud preflight or budget-bounded probe -> paired training -> dense eval.

Does not start/rent an instance. The caller must separately authorize its fees.
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import sys
import threading
import time
import math
import subprocess

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.dense_evaluation import evaluate, compatible_pixel_reference
from experiments.common.video.r2plus1d_classifier import configure_fp32
from experiments.common.video.supervised_plan import csv_rows, digest, rooted, write_json
from experiments.common.video.supervised_training import load_inputs, train


def hard_stop(seconds, out):
    time.sleep(seconds)
    try:
        (out / 'time-limit-reached.txt').write_text('Run stopped at registered wall-clock limit.\n', encoding='utf-8')
    finally:
        os._exit(124)  # Must also exit if the disk/log write fails.


def verify_uploaded_files(root, manifest, required):
    """Hash uploaded bytes without decoding media or constructing model inputs."""
    root, manifest = Path(root).resolve(), Path(manifest)
    before = digest(manifest)
    records = csv_rows(manifest)
    indexed = {r['path']: r for r in records}
    if not records or len(indexed) != len(records):
        raise ValueError('Empty or duplicate transfer manifest')
    for name, sha in required.items():
        if name not in indexed or indexed[name]['sha256'] != sha:
            raise ValueError(f'Upload manifest does not cover frozen input: {name}')
    parents, total = {}, 0
    started = time.monotonic()
    for index, row in enumerate(records, 1):
        rel = PurePosixPath(row['path'])
        if rel.is_absolute() or '..' in rel.parts or ':' in row['path'] or '\\' in row['path']:
            raise ValueError('Unsafe transfer manifest path')
        parent = str(rel.parent)
        if parent not in parents:
            parents[parent] = rooted(root, parent)
        target = parents[parent] / rel.name
        if (target.is_symlink() or not target.is_file()
                or target.stat().st_size != int(row['size_bytes']) or digest(target) != row['sha256']):
            raise ValueError(f'Uploaded file changed: {row["path"]}')
        total += int(row['size_bytes'])
        if index % 25000 == 0:
            print(json.dumps(dict(stage='static_file_hashes', verified=index, total=len(records))), flush=True)
    if digest(manifest) != before:
        raise ValueError('Transfer manifest changed during verification')
    return dict(status='cloud_static_inputs_verified', decoded_frames=0,
                files_sha_verified=len(records), bytes_sha_verified=total,
                manifest_sha256=before, seconds=time.monotonic() - started)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['check', 'full'], default='check')
    ap.add_argument('--seed', type=int, choices=(7, 8), default=7)
    ap.add_argument('--plan', type=Path, default=ROOT / 'data/metadata/EXP-016/training-plan-v2')
    ap.add_argument('--pixel-reference', type=Path, required=True)
    ap.add_argument('--ffmpeg', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--preflight-result', type=Path)
    ap.add_argument('--transfer-manifest', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--cache-mib', type=int, default=128)
    ap.add_argument('--microbatch', type=int, choices=range(1, 9), default=8)
    ap.add_argument('--checkpoint', action='store_true')
    ap.add_argument('--hourly-price', type=float, default=1.48)
    ap.add_argument('--seconds-limit', type=int, default=12 * 3600)
    ap.add_argument('--budget-yuan', type=float, default=20.)
    a = ap.parse_args()
    a.out.resolve().relative_to(ROOT / ('data/process' if a.stage == 'check' else 'data/results'))
    if a.out.exists():
        raise FileExistsError(a.out)
    if (not math.isfinite(a.hourly_price) or not math.isfinite(a.budget_yuan)
            or a.hourly_price <= 0 or a.budget_yuan <= 0 or a.seconds_limit <= 0
            or a.seconds_limit * a.hourly_price / 3600 > a.budget_yuan):
        raise ValueError('Time limit must fit the explicit fee budget')
    reference = json.loads(a.pixel_reference.read_text(encoding='utf-8'))
    if reference['status'] != 'dense_pixels_verified':
        raise ValueError('Matching completed CPU dense pixel verification is required')
    plan = json.loads((a.plan / 'plan.json').read_text(encoding='utf-8'))
    compatible_pixel_reference(plan, digest(a.plan / 'plan.json'), reference,
                               a.pixel_reference.parent / 'source-plan.json')
    configure_fp32(a.seed)
    import torch
    import cv2
    import torchvision
    ffmpeg_version = subprocess.run([str(a.ffmpeg), '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    runtime = dict(torch=torch.__version__, torchvision=torchvision.__version__, opencv=cv2.__version__,
                   python=list(sys.version_info[:3]), ffmpeg_sha256=digest(a.ffmpeg), ffmpeg_version=ffmpeg_version)
    code_paths = [ROOT / 'experiments/common/video' / n for n in
                  ('supervised_plan.py', 'supervised_runtime.py', 'supervised_training.py',
                   'r2plus1d_classifier.py', 'dense_evaluation.py', 'pts_clip_sampling.py')]
    code_paths += [Path(__file__).resolve(), ROOT / 'experiments/runspecs/EXP-016/train.py',
                   ROOT / 'experiments/runspecs/EXP-016/evaluate.py',
                   ROOT / 'experiments/runspecs/EXP-016/run_cloud.sh',
                   ROOT / 'experiments/runspecs/EXP-016/check_cloud.sh',
                   ROOT / 'docs/process/experiments/EXP-016.md',
                   ROOT / 'docs/process/experiments/EXP-006-数据使用方案.md']
    if a.seed == 8:
        code_paths.append(ROOT / 'docs/process/experiments/EXP-016-seed8.md')
    code = {p.relative_to(ROOT).as_posix(): digest(p) for p in code_paths}
    if a.stage == 'check':
        p, rows, schedule, val = load_inputs(ROOT, a.plan, check_files=True)
        required = {Path(frame.path).relative_to(ROOT).as_posix(): frame.jpeg_sha256
                    for _, frames, _ in rows for frame in frames}
        supervised_count = len(required)
        refs = csv_rows(ROOT / p['resource'] / 'dense-transfer-references.csv')
        if len(refs) != 93:
            raise ValueError('Expected 93 frozen dense source/coordinate/QC inputs')
        required.update({r['local_path']: r['sha256'] for r in refs})
        required.update(code)
        proof = verify_uploaded_files(ROOT, a.transfer_manifest, required)
        proof.update(supervised_images_sha_verified=supervised_count, dense_inputs_sha_verified=len(refs))
        a.out.mkdir(parents=True, exist_ok=False)
        write_json(a.out / 'static-inputs.json', proof)
        if any(digest(ROOT / path) != sha for path, sha in code.items()):
            raise RuntimeError('Source changed during static preflight')
        preflight = dict(status='cloud_static_preflight_passed', decoded_frames=0,
                         seed=a.seed,
                         plan_sha256=digest(a.plan / 'plan.json'),
                         pixel_reference_sha256=digest(a.pixel_reference), code_sha256=code, environment=runtime,
                         source_plan_sha256=digest(a.pixel_reference.parent / 'source-plan.json'),
                         transfer_manifest_sha256=digest(a.transfer_manifest),
                         static_result_sha256=digest(a.out / 'static-inputs.json'),
                         next_gate='Short GPU training runtime probe, paired training, then complete video evaluation with local pixel digest checks')
        write_json(a.out / 'preflight.json', preflight)
        print(json.dumps(preflight))
        return
    if a.preflight_result is None:
        raise ValueError('GPU run requires the completed no-GPU preflight report')
    preflight = json.loads(a.preflight_result.read_text(encoding='utf-8'))
    if (preflight.get('status') != 'cloud_static_preflight_passed'
            or preflight.get('decoded_frames') != 0
            or preflight.get('seed', 7) != a.seed
            or preflight['plan_sha256'] != digest(a.plan / 'plan.json')
            or preflight['pixel_reference_sha256'] != digest(a.pixel_reference)
            or preflight['source_plan_sha256'] != digest(a.pixel_reference.parent / 'source-plan.json')
            or preflight['transfer_manifest_sha256'] != digest(a.transfer_manifest)
            or preflight['code_sha256'] != code or preflight['environment'] != runtime
            or preflight['static_result_sha256'] != digest(a.preflight_result.parent / 'static-inputs.json')):
        raise ValueError('Completed CPU preflight no longer matches code/data/environment')
    if not torch.cuda.is_available():
        raise RuntimeError('A GPU run requires CUDA; no CPU fallback')
    a.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    threading.Thread(target=hard_stop, args=(a.seconds_limit, a.out), daemon=True).start()
    write_json(a.out / 'launch.json', dict(argv=sys.argv, hourly_price=a.hourly_price,
                                          seed=a.seed,
                                          budget_yuan=a.budget_yuan, seconds_limit=a.seconds_limit,
                                          pixel_reference_sha256=digest(a.pixel_reference)))
    kwargs = dict(root=ROOT, plan_dir=a.plan, seed=a.seed, device='cuda', workers=a.workers,
                  cache_mib=a.cache_mib, microbatch=a.microbatch, checkpoint=a.checkpoint)
    probe = train(out=a.out / 'probe-rgb', mode='rgb_zero', epochs=1, **kwargs)
    # Use only measured execution costs; do not branch on candidate scores.
    # Training reads verified JPEGs; video decoding follows supervised training.
    # Reserve one hour for paired evaluation, above the earlier single-model
    # 27-video/1002s observation. This is a reservation, not an extra pre-run.
    evaluation_reserve_seconds = 3600
    forecast_seconds = 1.5 * 2 * 20 * probe['seconds'] + evaluation_reserve_seconds
    elapsed = time.monotonic() - started
    forecast = dict(probe_seconds=elapsed, remaining_forecast_seconds=forecast_seconds,
                    paired_evaluation_reserve_seconds=evaluation_reserve_seconds,
                    total_forecast_yuan=(elapsed + forecast_seconds) / 3600 * a.hourly_price,
                    decision_uses='runtime and correctness only; not model metrics')
    write_json(a.out / 'budget-probe.json', forecast)
    if elapsed + forecast_seconds > a.seconds_limit or forecast['total_forecast_yuan'] > a.budget_yuan:
        raise RuntimeError('Measured forecast exceeds budget; paired training not started')
    train(out=a.out / 'rgb', mode='rgb_zero', epochs=20, **kwargs)
    train(out=a.out / 'diff', mode='rgb_diff', epochs=20, **kwargs)
    try:
        result = evaluate(ROOT, a.plan, a.out / 'paired-dense', a.ffmpeg,
                          {'rgb_zero': a.out / 'rgb/best.pt', 'rgb_diff': a.out / 'diff/best.pt'},
                          'cuda', a.pixel_reference)
    except Exception as error:
        write_json(a.out / 'evaluation-failure.json', dict(
            status='training_complete_video_evaluation_failed', error_type=type(error).__name__,
            error=str(error), training_artifacts_preserved=True, automatic_retraining=False))
        raise
    if (len(result['videos']) != 31 or sum(v['windows'] for v in result['videos'].values()) != 32567
            or any(m['status'] != 'complete' for m in result['metrics'].values())):
        raise RuntimeError('Incomplete video coverage; full metrics/promotion withheld')
    if any(digest(ROOT / p) != sha for p, sha in code.items()):
        raise RuntimeError('Pipeline source/card/protocol changed during execution')
    write_json(a.out / 'pipeline-result.json', dict(status='completed',
               seed=a.seed,
               seconds=time.monotonic() - started, metrics=result['metrics'],
               next_step='result audit; do not automatically start another seed or formal CV'))


if __name__ == '__main__':
    main()
