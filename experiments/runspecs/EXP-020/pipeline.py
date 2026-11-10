"""Six fresh subprocesses; forecasts are advisory and never stop the matrix."""
import argparse
import json
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
from plan import load, read, comparison, _sha, _json, NEW_CODE


def inventory(directory, prefix=''):
    result = []
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('No symlink output artifacts')
        if path.is_file():
            result.append(dict(path=prefix + path.relative_to(directory).as_posix(), size_bytes=path.stat().st_size, sha256=_sha(path)))
    return result


def hard_stop(deadline, out, stop):
    if stop.wait(max(0, deadline - time.time())):
        return
    try:
        try:
            _json(out / 'hard-time-limit.json', dict(status='actual_authorized_money_deadline_reached', partial_artifacts_preserved=True))
        except Exception:
            pass
        try:
            subprocess.run(['/bin/bash', '/usr/bin/shutdown'], timeout=15, check=False)
        except Exception:
            pass
    finally:
        os._exit(124)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--money-deadline-unix', type=float, required=True)
    p.add_argument('--platform-timer-receipt', type=Path, required=True)
    a = p.parse_args(argv)
    out = a.out.resolve()
    if os.name != 'posix' or not math.isfinite(a.money_deadline_unix) or time.time() >= a.money_deadline_unix:
        raise ValueError('Production POSIX and future actual money deadline required')
    if not a.platform_timer_receipt.is_file() or not a.platform_timer_receipt.stat().st_size:
        raise ValueError('Require actual operator-verified platform fee timer receipt')
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < 6 * 1024 ** 3:
        raise ValueError('Need6GiB on actual output mount for six fits and temporary checkpoints')
    out.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    watch = threading.Thread(target=hard_stop, args=(a.money_deadline_unix, out, stop), daemon=True)
    watch.start()
    closed = []
    status = 'run_failed_closed'
    try:
        plan, schedules, bank, _, fingerprints = load(ROOT, a.plan)
        code = dict(plan['code_sha256'])
        for name, sha in code.items():
            target = out / 'source-snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
            if _sha(target) != sha:
                raise ValueError('Source changed while snapshotting')
        _json(out / 'execution-inputs.json', dict(plan_sha256=_sha(a.plan / 'plan.json'),
            input_fingerprints=fingerprints, code_sha256=code, output_directory=str(out),
            money_deadline_unix=a.money_deadline_unix, platform_timer_receipt=str(a.platform_timer_receipt.resolve()),
            platform_timer_receipt_sha256=_sha(a.platform_timer_receipt), local_archive_mapping='data/results/exp020/' + out.name))
        _json(out / 'launch.json', dict(argv=sys.argv if argv is None else argv, no_automatic_repeat=True,
            forecast_policy='advisory_only; never stop for predicted duration or model scores'))
        elapsed = []
        for item in plan['fit_order']:
            if time.time() >= a.money_deadline_unix:
                raise TimeoutError('Actual authorized fee deadline reached')
            began = time.perf_counter()
            print(json.dumps(dict(status='starting_fit', item=item)), flush=True)
            command = [sys.executable, '-B', str(Path(__file__).with_name('train.py')), '--plan', str(a.plan),
                '--item', item, '--out', str(out / item), '--execution-inputs', str(out / 'execution-inputs.json')]
            with (out / (item + '.log')).open('xb') as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=max(.001, a.money_deadline_unix - time.time()), check=False)
            elapsed.append(time.perf_counter() - began)
            complete = process.returncode == 0 and (out / item / 'training-result.json').is_file()
            files = inventory(out / item, item + '/') if (out / item).is_dir() else []
            files += [dict(path=item + '.log', size_bytes=(out / (item + '.log')).stat().st_size, sha256=_sha(out / (item + '.log')))]
            _json(out / ('closed-' + item + '.json'), dict(status='fit_closed' if complete else 'fit_failed_closed', item=item, path=item, files=files))
            closed.extend(files)
            if not complete:
                raise RuntimeError('Technical fit failure: ' + item)
            remaining = len(plan['fit_order']) - len(elapsed)
            _json(out / ('forecast-after-' + item + '.json'), dict(policy='advisory_only',
                expected_remaining_seconds=1.3 * max(elapsed) * remaining + 300,
                actual_remaining_fee_seconds=a.money_deadline_unix - time.time(), next_fit_not_cancelled=True))
            print(json.dumps(dict(status='fit_closed', item=item, seconds=elapsed[-1])), flush=True)
        answer = comparison(plan, out, ROOT, bank=bank, schedules=schedules)
        if time.time() >= a.money_deadline_unix:
            raise TimeoutError('Actual authorized fee deadline reached before comparison closure')
        _json(out / 'comparison.json', answer)
        _json(out / 'pipeline-result.json', dict(status='six_new_fits_complete_pending_independent_audit',
            reused_items=[x for x in plan['items'] if x['action'] == 'reuse'], comparison_sha256=_sha(out / 'comparison.json')))
        status = 'run_closed'
    except Exception as e:
        _json(out / 'pipeline-failure.json', dict(status='incomplete_metrics_withheld', error_type=type(e).__name__, error=str(e)))
        raise
    finally:
        # Only completed subprocess artifacts are closed; an OS-abort/deadline
        # leaves no ready receipt for the actively written stage.
        try:
            rest = [r for r in inventory(out) if r['path'] not in {x['path'] for x in closed}]
            _json(out / 'closed-run.json', dict(status=status, files=closed + rest,
                excludes_later_operator_shutdown_logs=True))
        finally:
            stop.set()


if __name__ == '__main__':
    main()
