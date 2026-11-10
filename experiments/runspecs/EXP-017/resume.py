"""Prepare immutable parent lineage, or continue EXP-017 folds2--5 unchanged."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.cv_execution import load_execution
from experiments.common.video.cv_resume_execution import (PARENT_NAME, continuation_budget, load_parent,
    plain, prepare_parent, require, resume_sources, run_remaining, verify_inventory)
from experiments.common.video.supervised_plan import digest, write_json

SPEC = importlib.util.spec_from_file_location('exp017_original_pipeline', Path(__file__).with_name('pipeline.py'))
ORIGINAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORIGINAL)


def shutdown(out):
    """The shell also independently shuts down on Python abort/abnormal exit."""
    log = None
    try:
        try:
            log = (out / 'shutdown.log').open('xb')
        except Exception:
            pass
        r = subprocess.run(['/bin/bash', '/usr/bin/shutdown'], stdin=subprocess.DEVNULL,
            stdout=log if log else subprocess.DEVNULL, stderr=subprocess.STDOUT,
            start_new_session=True, timeout=15, check=False)
        return dict(status='shutdown_requested' if r.returncode == 0 else 'shutdown_failed', returncode=r.returncode)
    except Exception as e:
        return dict(status='shutdown_failed', error_type=type(e).__name__, error=str(e))
    finally:
        if log is not None:
            log.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'full'], default='prepare')
    for n in ('plan', 'acceptance', 'dense-reference', 'preflight', 'transfer-manifest', 'out'):
        parser.add_argument('--' + n, type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=ROOT / 'data/results/exp017' / PARENT_NAME)
    parser.add_argument('--parent-binding', type=Path)
    parser.add_argument('--parent-binding-sha256')
    parser.add_argument('--ffmpeg', type=Path)
    parser.add_argument('--gpu-start-unix', type=float)
    parser.add_argument('--platform-deadline-unix', type=float)
    parser.add_argument('--platform-shutdown-receipt', type=Path)
    parser.add_argument('--hourly-price', type=float, default=1.48)
    for n in ('prior-gpu-seconds', 'prior-cpu-seconds', 'prior-stage-spend-yuan', 'prior-total-spend-yuan'):
        parser.add_argument('--' + n, type=float)
    a = parser.parse_args(argv)
    for n in ('plan', 'acceptance', 'dense_reference', 'preflight', 'transfer_manifest', 'out', 'parent'):
        p = getattr(a, n).resolve()
        plain(ROOT, p.relative_to(ROOT).as_posix())
        setattr(a, n, p)
    if a.stage == 'prepare':
        a.out.relative_to(ROOT / 'data/metadata/EXP-017')
        context = load_execution(ROOT, a.plan, a.acceptance, a.dense_reference)
        answer = prepare_parent(context, a.parent, a.preflight, a.transfer_manifest, a.out)
        print(json.dumps(dict(**answer, parent_binding_sha256=digest(a.out / 'parent.json')), allow_nan=False))
        return
    require(os.name == 'posix' and os.geteuid() == 0 and Path('/usr/bin/shutdown').is_file(), 'Only the bounded AutoDL root production route may execute')
    a.out.relative_to(ROOT / 'data/results/exp017')
    match = re.fullmatch(r'cv-s7-20261102-v([0-9]+)', a.out.name)
    require(match is not None and int(match.group(1)) >= 2, 'Need a new explicit child run ID')
    require(not a.out.exists(), 'Child output already exists; no overwrite/retry')
    a.out.mkdir(parents=True, exist_ok=False)
    stop = threading.Event(); watch = None; started = time.perf_counter()
    try:
        require(all(getattr(a, n) is not None for n in ('parent_binding', 'parent_binding_sha256', 'ffmpeg',
            'gpu_start_unix', 'platform_deadline_unix', 'platform_shutdown_receipt', 'prior_gpu_seconds',
            'prior_cpu_seconds', 'prior_stage_spend_yuan', 'prior_total_spend_yuan')), 'Actual launch/ledger/lineage/timer evidence required')
        a.parent_binding = a.parent_binding.resolve()
        plain(ROOT, a.parent_binding.relative_to(ROOT).as_posix())
        receipt = a.platform_shutdown_receipt.resolve()
        plain(ROOT, receipt.relative_to(ROOT).as_posix())
        require(receipt.is_file() and receipt.stat().st_size > 0, 'Actual operator-verified platform timer receipt required')
        budget = continuation_budget(time.time(), a.gpu_start_unix, a.platform_deadline_unix, a.hourly_price,
            a.prior_gpu_seconds, a.prior_cpu_seconds, a.prior_stage_spend_yuan, a.prior_total_spend_yuan)
        watch = threading.Thread(target=ORIGINAL._hard_stop, args=(budget['deadline_unix'], a.out, stop), daemon=True)
        watch.start()
        write_json(a.out / 'launch.json', dict(argv=sys.argv if argv is None else argv, budget=budget,
            platform_timer_receipt=receipt.relative_to(ROOT).as_posix(), platform_timer_receipt_sha256=digest(receipt),
            timer_authenticity='Operator actually verifies platform timer; file presence is not UI confirmation', no_automatic_repeat=True))
        context = load_execution(ROOT, a.plan, a.acceptance, a.dense_reference)
        binding, frozen, first, parent_rows = load_parent(context, a.parent_binding, a.parent_binding_sha256, a.preflight, a.transfer_manifest)
        require(a.parent == plain(ROOT, binding['parent_path']), 'CLI parent differs from bound parent')
        require(digest(a.ffmpeg) == frozen['ffmpeg_sha256'], 'Original checked ffmpeg differs')
        source = resume_sources(ROOT)
        lineage = dict(parent_run_id=binding['parent_run_id'], parent_path=binding['parent_path'],
            parent_binding_path=a.parent_binding.relative_to(ROOT).as_posix(), parent_binding_sha256=a.parent_binding_sha256,
            parent_execution_inputs_sha256=binding['parent_execution_inputs_sha256'],
            parent_artifacts_csv_sha256=binding['files_sha256'], reused_folds=[1], newly_executed_folds=[2, 3, 4, 5],
            inherited_parent_weights=False, inherited_parent_optimizer=False)
        for name, sha in dict(frozen['code_sha256'], **source).items():
            target = a.out / 'source-snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(plain(ROOT, name).read_bytes())
            require(digest(target) == sha, 'Execution source changed while snapshotting')
        write_json(a.out / 'execution-inputs.json', dict(**frozen, resume_lineage=lineage, resume_code_sha256=source))
        summary, complete, origins = run_remaining(context, a.out, first, a.ffmpeg, budget, ORIGINAL.remaining_forecast)
        verify_inventory(a.parent, parent_rows)
        require(resume_sources(ROOT) == source, 'Continuation controller changed during execution')
        for name, sha in frozen['input_fingerprints'].items():
            require(digest(plain(ROOT, name)) == sha, 'Original execution inputs changed')
        require(time.time() < budget['deadline_unix'], 'Actual authorized fee/time boundary reached')
        write_json(a.out / 'pipeline-result.json', dict(status='complete_pending_independent_audit', summary=summary,
            resume_lineage=lineage, fold_artifact_origins=origins,
            per_fold=[dict(fold=i + 1, selected_epochs=x['inner']['selected_epochs'], metrics=x['dense']['metrics']) for i, x in enumerate(complete)],
            seconds=time.perf_counter() - started, budget=budget, no_automatic_repeat=True,
            final_invoice_and_actual_shutdown_require_operator_confirmation=True))
    except Exception as e:
        try:
            write_json(a.out / 'pipeline-failure.json', dict(status='incomplete_five_fold_metrics_withheld',
                error_type=type(e).__name__, error=str(e), partial_artifacts_preserved=True))
        finally:
            raise
    finally:
        # Keep the fee watchdog alive until a shutdown request has completed.
        receipt = shutdown(a.out)
        try:
            write_json(a.out / 'shutdown-receipt.json', receipt)
        except Exception:
            pass
        stop.set()
        if watch:
            watch.join(timeout=1)
        if receipt['status'] != 'shutdown_requested':
            raise RuntimeError('Shutdown request failed; independent platform timer remains required')


if __name__ == '__main__':
    main()
