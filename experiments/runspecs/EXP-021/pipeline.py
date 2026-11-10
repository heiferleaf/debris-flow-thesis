"""Four independent fresh fits then two shared A/B/C dense role passes."""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.cv_supervised_training import _json, _sha
from experiments.common.video.oriented_energy_plan import load, read, new_sources
from experiments.common.video.oriented_energy_evaluation import evaluate_role
from experiments.common.video.oriented_energy_metrics import clip_metrics, gate


def inventory(directory, prefix=''):
    rows = []
    for p in sorted(directory.rglob('*')):
        if p.is_symlink():
            raise ValueError('No symlink output artifacts')
        if p.is_file():
            rows.append(dict(path=prefix + p.relative_to(directory).as_posix(), size_bytes=p.stat().st_size, sha256=_sha(p)))
    return rows


def close_stage(out, item, status):
    rows = inventory(out / item, item + '/')
    log = out / (item + '.log')
    if log.is_file():
        rows.append(dict(path=log.name, size_bytes=log.stat().st_size, sha256=_sha(log)))
    _json(out / ('closed-' + item + '.json'), dict(status=status, path=item, item=item, files=rows))


def hard_stop(deadline, out, stop):
    if stop.wait(max(0, deadline - time.time())):
        return
    try:
        try: _json(out / 'hard-time-limit.json', dict(status='actual_authorized_money_deadline_reached', metrics_withheld=True))
        except Exception: pass
        try: subprocess.run(['/usr/bin/timeout', '--kill-after=2s', '15s', '/bin/bash', '/usr/bin/shutdown'], timeout=18, check=False)
        except Exception: pass
    finally:
        os._exit(124)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'out', 'reference-root', 'ffmpeg', 'platform-timer-receipt'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--money-deadline-unix', type=float, required=True)
    a = p.parse_args(argv)
    out = a.out.resolve()
    if (os.name != 'posix' or not math.isfinite(a.money_deadline_unix) or a.money_deadline_unix <= time.time()
            or not a.platform_timer_receipt.is_file() or not a.platform_timer_receipt.stat().st_size):
        raise ValueError('Production route requires actual fee deadline and operator-verified platform timer')
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < 5 * 1024 ** 3:
        raise ValueError('Need5GiB on actual output mount for four fits, evidence and temporary checkpoints')
    out.mkdir(parents=True, exist_ok=False)
    stop = threading.Event(); terminal = 'run_failed_closed'
    watch = threading.Thread(target=hard_stop, args=(a.money_deadline_unix, out, stop), daemon=True); watch.start()
    try:
        context = load(ROOT, a.plan, a.reference_root)
        plan = context['plan']
        if a.reference_root.resolve() != Path(plan['reference_roots']['cloud']).resolve():
            raise ValueError('Production must use the explicit cloud A reference root')
        ffmpeg_sha = _sha(a.ffmpeg)
        original_ffmpeg = plan['ffmpeg_sha256']
        if ffmpeg_sha != original_ffmpeg:
            raise ValueError('Use original already checked cloud ffmpeg7 bytes')
        source_names = set(new_sources(ROOT)) | {name for name in plan['source_fingerprints']
            if name.startswith('experiments/') or name.endswith('SKILL.md') or name.startswith('docs/process/experiments/')}
        code = {name: plan['source_fingerprints'][name] for name in sorted(source_names)}
        for name, expected in code.items():
            target = out / 'source-snapshot' / name; target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream: stream.write((ROOT / name).read_bytes())
            if _sha(target) != expected:
                raise ValueError('Source changed while snapshotting')
        _json(out / 'execution-inputs.json', dict(plan_sha256=_sha(a.plan / 'plan.json'), input_fingerprints=context['fingerprints'],
            code_sha256=code, output_directory=str(out), reference_root=str(a.reference_root.resolve()),
            reference_mapping=plan['reference_roots'], money_deadline_unix=a.money_deadline_unix,
            platform_timer_receipt=str(a.platform_timer_receipt.resolve()), platform_timer_receipt_sha256=_sha(a.platform_timer_receipt),
            ffmpeg_sha256=ffmpeg_sha, local_archive_mapping='data/results/exp021/' + out.name))
        _json(out / 'launch.json', dict(argv=sys.argv if argv is None else argv, forecast_policy='advisory_only', no_automatic_repeat=True))
        initial_profiles, durations = [], []
        for item in plan['fit_order']:
            if time.time() >= a.money_deadline_unix:
                raise TimeoutError('Actual authorized fee deadline reached')
            started = time.perf_counter()
            command = [sys.executable, '-B', str(Path(__file__).with_name('train.py')), '--plan', str(a.plan),
                '--reference-root', str(a.reference_root), '--item', item, '--out', str(out / item), '--execution-inputs', str(out / 'execution-inputs.json')]
            print('starting fit ' + item, flush=True)
            with (out / (item + '.log')).open('xb') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=max(.001, a.money_deadline_unix - time.time()), check=False)
            if result.returncode != 0 or not (out / item / 'training-result.json').is_file():
                if (out / item).is_dir(): close_stage(out, item, 'fit_failed_closed')
                raise RuntimeError('Technical fit failure: ' + item)
            close_stage(out, item, 'fit_closed')
            initial_profiles.append(read(out / item / 'profile.json'))
            for key in ('initial_state_sha256', 'post_model_CPU_RNG_sha256', 'RGB_component_initial_state_sha256', 'backbone_state_sha256'):
                if len({r[key] for r in initial_profiles}) != 1:
                    raise ValueError('B/C/fold initialization or RNG mismatch: ' + key)
            durations.append(time.perf_counter() - started)
            _json(out / ('forecast-after-' + item + '.json'), dict(policy='advisory_only',
                remaining_fit_seconds=1.3 * max(durations) * (4 - len(durations)), dense_seconds='not_yet_measured; no stop decision',
                remaining_actual_fee_seconds=a.money_deadline_unix - time.time(), next_stage_not_cancelled=True))
            print('closed fit ' + item, flush=True)
        context = load(ROOT, a.plan, a.reference_root, with_dense=True)
        dense, fragment = {}, {}
        for fold in (1, 4):
            fragment[str(fold)] = {}
            for arm in ('A', 'B', 'C'):
                directory = a.reference_root / plan['A_references'][str(fold)]['item'] if arm == 'A' else out / ('f%02d-%s-s7' % (fold, arm))
                fragment[str(fold)][arm] = clip_metrics(directory, context['bank'], context['schedules'][str(fold)]['validation_clip_ids'])
            dense[str(fold)] = evaluate_role(context, out, fold, a.ffmpeg)
            for arm in ('A', 'B', 'C'):
                close_stage(out, 'dense-f%02d-%s' % (fold, arm), 'dense_closed')
        verdict = gate(dense, fragment, context['sources'], plan['roles'])
        _json(out / 'comparison.json', dict(status='four_fits_six_dense_complete_pending_independent_audit', fragment=fragment, gate=verdict,
            per_role_dense={f: {arm: r['metrics'] for arm, r in arms.items()} for f, arms in dense.items()},
            A_reference_root=str(a.reference_root.resolve()), no_outer=True, no_external=True, no_automatic_followup=True))
        if time.time() >= a.money_deadline_unix:
            raise TimeoutError('Actual authorized fee deadline reached before closure')
        _json(out / 'pipeline-result.json', dict(status='four_fits_six_dense_complete_pending_independent_audit',
            comparison_sha256=_sha(out / 'comparison.json'), promote_to_matched_formal_design_only=verdict['promote_to_matched_formal_design_only']))
        terminal = 'run_closed'
    except Exception as e:
        _json(out / 'pipeline-failure.json', dict(status='incomplete_metrics_withheld', error_type=type(e).__name__, error=str(e)))
        raise
    finally:
        try:
            _json(out / 'closed-run.json', dict(status=terminal, files=inventory(out), excludes_later_operator_shutdown_logs=True))
        finally:
            stop.set()


if __name__ == '__main__':
    main()
