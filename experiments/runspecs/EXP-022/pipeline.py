"""EXP022 five-fold Linear controller: two reused inners, eight fresh fits, five dense folds."""
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
from experiments.common.video.linear_cv_execution import load, read, require, reused_inner, comparison


def stage_order():
    return [item for f in range(1,6) for item in
            ([('f%02d-inner' % f)] if f in (2,3,5) else []) + ['f%02d-outer' % f, 'f%02d-dense' % f]]


def inventory(directory, prefix=''):
    rows = []
    for path in sorted(directory.rglob('*')):
        require(not path.is_symlink(), 'No symlink result artifacts')
        if path.is_file():
            rows.append(dict(path=prefix + path.relative_to(directory).as_posix(),
                             size_bytes=path.stat().st_size, sha256=_sha(path)))
    return rows


def close_stage(out, item, status):
    files = inventory(out / item, item + '/')
    log = out / (item + '.log')
    if log.is_file():
        files.append(dict(path=log.name, size_bytes=log.stat().st_size, sha256=_sha(log)))
    _json(out / ('closed-' + item + '.json'), dict(status=status, path=item, item=item, files=files))


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
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','out','reference-root','ffmpeg','platform-timer-receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--money-deadline-unix', type=float, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    require(os.name == 'posix' and math.isfinite(args.money_deadline_unix) and args.money_deadline_unix > time.time()
            and args.platform_timer_receipt.is_file() and args.platform_timer_receipt.stat().st_size > 0,
            'Production needs actual authorized fee deadline and operator-verified platform timer')
    require(not out.exists(), 'Refuse repeated run/output overwrite')
    out.parent.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(out.parent).free >= 8 * 1024 ** 3, 'Need8GiB on the actual output mount for16 standard checkpoints and temporary/evidence bytes')
    out.mkdir(parents=True, exist_ok=False)
    stop = threading.Event(); terminal = 'run_failed_closed'
    watch = threading.Thread(target=hard_stop, args=(args.money_deadline_unix, out, stop), daemon=True)
    watch.start()
    try:
        context = load(ROOT, args.plan, args.reference_root)
        plan = context['plan']
        require(args.reference_root.resolve() == Path(plan['reference_roots']['cloud']).resolve(), 'Production reference root must use explicit cloud mapping')
        require(_sha(args.ffmpeg) == plan['ffmpeg_sha256'], 'Use original checked ffmpeg7 bytes')
        reused = {str(f):reused_inner(context, f) for f in (1,4)}
        _json(out / 'inner-reuse.json', dict(status='two_closed_inner_stages_reused_no_retraining',
            original_run_id='head-s7s8-20261103-v2', reference_root=str(args.reference_root.resolve()),
            items={str(f):dict(item=plan['reused_inner'][str(f)]['item'], selected_epochs=reused[str(f)]['selected_epochs'],
                closure_sha256=plan['reused_inner'][str(f)]['closure_sha256'], source_profile_sha256=reused[str(f)]['profile_sha256']) for f in (1,4)},
            scope='References only; old results not copied or rewritten as new fits'))
        code = {name:fingerprint for name,fingerprint in plan['source_fingerprints'].items()
                if name.startswith('experiments/') or name.startswith('docs/') or name.endswith('SKILL.md')}
        for name, fingerprint in code.items():
            target = out / 'source-snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream: stream.write((ROOT / name).read_bytes())
            require(_sha(target) == fingerprint, 'Source changed during snapshot')
        _json(out / 'execution-inputs.json', dict(status='linear_cv_controller_active', controller_pid=os.getpid(),
            plan_sha256=_sha(args.plan / 'plan.json'), input_fingerprints=context['fingerprints'], code_sha256=code,
            output_directory=str(out), reference_root=str(args.reference_root.resolve()), reference_mapping=plan['reference_roots'],
            stage_order=stage_order(), money_deadline_unix=args.money_deadline_unix,
            platform_timer_receipt=str(args.platform_timer_receipt.resolve()), platform_timer_receipt_sha256=_sha(args.platform_timer_receipt),
            ffmpeg_sha256=_sha(args.ffmpeg), local_archive_mapping='data/results/exp022/' + out.name))
        _json(out / 'launch.json', dict(argv=sys.argv if argv is None else argv, forecast_policy='advisory_only', no_automatic_repeat=True))
        durations = []
        for item in stage_order():
            require(time.time() < args.money_deadline_unix, 'Actual fee deadline reached')
            command = [sys.executable, '-B', str(Path(__file__).with_name('stage.py')), '--plan', str(args.plan),
                '--reference-root', str(args.reference_root), '--item', item, '--out', str(out / item),
                '--execution-inputs', str(out / 'execution-inputs.json'), '--ffmpeg', str(args.ffmpeg)]
            print('starting ' + item, flush=True)
            started = time.perf_counter()
            with (out / (item + '.log')).open('xb') as stream:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                    timeout=max(.001, args.money_deadline_unix - time.time()), check=False)
            kind = 'dense' if item.endswith('dense') else 'fit'
            marker = 'result.json' if kind == 'dense' else 'training-result.json'
            if completed.returncode != 0 or not (out / item / marker).is_file():
                if (out / item).is_dir(): close_stage(out, item, kind + '_failed_closed')
                raise RuntimeError('Technical stage failure: ' + item)
            close_stage(out, item, kind + '_closed')
            durations.append(time.perf_counter() - started)
            _json(out / ('forecast-after-' + item + '.json'), dict(policy='advisory_only',
                observed_stage_seconds=durations[-1], remaining_actual_fee_seconds=args.money_deadline_unix - time.time(),
                remaining_stages=len(stage_order()) - len(durations), next_stage_not_cancelled=True))
            print('closed ' + item, flush=True)
        results = [read(out / ('f%02d-dense' % f) / 'result.json') for f in range(1,6)]
        require(sum(r['videos'][u]['windows'] for r in results for u in r['videos']) == 143470,
                'Incomplete150-video physical-window total')
        derived = comparison(results, read(args.plan / 'deep-baseline.json'), context['sources'])
        _json(out / 'comparison.json', derived)
        _json(out / 'pipeline-result.json', dict(status='five_fold_linear_cv_complete_pending_independent_audit',
            comparison_sha256=_sha(out / 'comparison.json'), source_inner_reuse_sha256=_sha(out / 'inner-reuse.json'),
            fold_artifact_origins=[dict(fold=f, inner=(str(args.reference_root / plan['reused_inner'][str(f)]['item']) if f in (1,4)
                else 'f%02d-inner' % f), outer='f%02d-outer' % f, dense='f%02d-dense' % f) for f in range(1,6)],
            scientific_gate_passed=derived['supports_overall_head_simplification_video_improvement']))
        require(time.time() < args.money_deadline_unix, 'Actual fee deadline reached before terminal closure')
        terminal = 'run_closed'
    except Exception as error:
        _json(out / 'pipeline-failure.json', dict(status='incomplete_metrics_withheld', error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        try:
            _json(out / 'closed-run.json', dict(status=terminal, files=inventory(out), excludes_later_operator_shutdown_logs=True))
        finally:
            stop.set()


if __name__ == '__main__':
    main()
