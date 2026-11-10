"""Single bounded five-fold CV execution. Root owns rental and platform receipts."""
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

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.common.video.cv_execution import load_execution,train_stage,summarize_folds,deployment_records,validate_deployment_manifest
from experiments.common.video.cv_dense_evaluation import evaluate_fold
from experiments.common.video.supervised_plan import csv_rows,digest,rooted,write_json

GPU_LIMIT=14400.;CPU_LIMIT=7200.;STAGE_CEILING=7.;TOTAL_CEILING=20.


def code_paths():
    common=ROOT/'experiments/common/video'
    paths=[common/n for n in ('cv_execution.py','cv_dense_evaluation.py','cv_supervised_training.py','cv_supervised_plan.py',
        'cv_cache_preparation.py','supervised_runtime.py','supervised_plan.py','supervised_training.py','dense_evaluation.py',
        'r2plus1d_classifier.py','pts_clip_sampling.py')]
    paths += [ROOT/'experiments/runspecs/EXP-017'/n for n in ('train.py','pipeline.py','run.sh','prepare_transfer.py','audit.py',
        'prepare_plan.py','prepare_inputs.py','verify_cache.py','README.md')]
    return paths+[ROOT/'docs/process/experiments/EXP-017.md',ROOT/'docs/process/experiments/EXP-006-数据使用方案.md',
                  ROOT/'.agents/skills/data-usage-protocol/SKILL.md']


def budget_deadline(now,start,platform_end,price,prior_gpu,prior_cpu,prior_stage,prior_total):
    if not all(math.isfinite(x) for x in (now,start,platform_end,price,prior_gpu,prior_cpu,prior_stage,prior_total)):
        raise ValueError('Nonfinite budget input')
    if (not 0<price<=1.48 or not 0<=prior_gpu<GPU_LIMIT or not 0<=prior_cpu<=CPU_LIMIT
            or not 0<=prior_stage<STAGE_CEILING or not 2.87+prior_stage-1e-8<=prior_total<TOTAL_CEILING or start>now+5):
        raise ValueError('Invalid registered price/cumulative time/ledger')
    gpu_end=start+GPU_LIMIT-prior_gpu
    if not now<platform_end<=gpu_end+1:raise ValueError('Actual platform timer exceeds cumulative GPU bound')
    cpu_reserve=(CPU_LIMIT-prior_cpu)*.1/3600
    money=min(STAGE_CEILING-prior_stage,TOTAL_CEILING-prior_total)-cpu_reserve-.1
    end=min(gpu_end,platform_end,start+money/price*3600)-60
    if end<=now:raise ValueError('No time/money left for registered run')
    return dict(deadline_unix=end,actual_gpu_start_unix=start,prior_gpu_seconds=prior_gpu,prior_cpu_seconds=prior_cpu,
        prior_stage_yuan=prior_stage,prior_total_yuan=prior_total,hourly_price=price,gpu_limit_seconds=GPU_LIMIT,
        cpu_limit_seconds=CPU_LIMIT,remaining_cpu_cost_reserve=cpu_reserve,stage_ceiling=STAGE_CEILING,total_ceiling=TOTAL_CEILING,
        shutdown_reserve_seconds=60,platform_deadline_unix=platform_end)


def remaining_forecast(completed,remaining_folds,dense_windows):
    """Only elapsed time/update counts. Future outer fits conservatively use20."""
    if not completed:raise ValueError('Complete first registered fold required')
    if any(not math.isfinite(x[k]['seconds']) or x[k]['seconds']<=0 or x[k]['optimizer_steps']<=0 for x in completed for k in ('inner','outer')):
        raise ValueError('Invalid timing/update evidence')
    if any(not math.isfinite(x['dense']['seconds']) or x['dense']['seconds']<=0 or not sum(r['windows'] for r in x['dense']['scores']) for x in completed):
        raise ValueError('Invalid complete-video timing evidence')
    step=max(x[k]['seconds']/x[k]['optimizer_steps'] for x in completed for k in ('inner','outer'))
    rate=max(x['dense']['seconds']/sum(r['windows'] for r in x['dense']['scores']) for x in completed)
    updates=sum(20*(math.ceil(len(f['inner_train']['epoch_clip_ids'][0])/24)+math.ceil(len(f['outer_train']['epoch_clip_ids'][0])/24)) for f in remaining_folds)
    windows=sum(len(dense_windows[u]) for f in remaining_folds for u in f['outer_eval']['video_ids'])
    return dict(seconds=1.3*(step*updates+rate*windows)+300,worst_case_remaining_updates=updates,
        remaining_dense_windows=windows,seconds_per_update=step,dense_seconds_per_window=rate,
        basis='technical completeness and elapsed time only; future outer E bounded by20; no score/quality gate')


def _shutdown(out):
    log=None
    try:
        try:log=(out/'hard-stop-shutdown.log').open('xb')
        except Exception:pass
        r=subprocess.run(['/bin/bash','/usr/bin/shutdown'],stdin=subprocess.DEVNULL,
            stdout=log if log is not None else subprocess.DEVNULL,stderr=subprocess.STDOUT,
            start_new_session=True,timeout=15,check=False)
        return dict(returncode=r.returncode)
    finally:
        if log is not None:log.close()


def _hard_stop(deadline,out,stop):
    if stop.wait(max(0,deadline-time.time())):return
    try:
        try:write_json(out/'hard-time-limit.json',dict(status='failed_time_limit',metrics_withheld=True))
        except Exception:pass
        try:receipt=dict(status='shutdown_requested',**_shutdown(out))
        except Exception as e:receipt=dict(status='shutdown_failed',error=str(e))
        try:write_json(out/'hard-stop-shutdown.json',receipt)
        except Exception:pass
    finally:os._exit(124)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['check','full'],default='check')
    for name in ('plan','acceptance','dense-reference','out','ffmpeg','transfer-manifest'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',type=Path);p.add_argument('--gpu-start-unix',type=float);p.add_argument('--platform-deadline-unix',type=float)
    p.add_argument('--platform-shutdown-receipt',type=Path);p.add_argument('--hourly-price',type=float,default=1.48)
    p.add_argument('--prior-gpu-seconds',type=float,default=0.);p.add_argument('--prior-cpu-seconds',type=float,required=True)
    p.add_argument('--prior-stage-spend-yuan',type=float);p.add_argument('--prior-total-spend-yuan',type=float)
    a=p.parse_args(argv)
    for name in ('plan','acceptance','dense_reference','out','ffmpeg','transfer_manifest'):setattr(a,name,getattr(a,name).resolve())
    a.out.relative_to(ROOT/('data/process/EXP-017' if a.stage=='check' else 'data/results/exp017'))
    if a.out.exists():raise FileExistsError(a.out)
    if not 0<=a.prior_cpu_seconds<=CPU_LIMIT:raise ValueError('Cumulative no-GPU rental exceeds2h')
    stop=threading.Event();watch=None;began=time.perf_counter();budget=None
    a.out.mkdir(parents=True,exist_ok=False)
    try:
        if a.stage=='full':
            if any(x is None for x in (a.preflight,a.gpu_start_unix,a.platform_deadline_unix,a.platform_shutdown_receipt,a.prior_stage_spend_yuan,a.prior_total_spend_yuan)):
                raise ValueError('Need actual boot/ledger/platform timer evidence and successful static preflight')
            if not a.platform_shutdown_receipt.is_file() or not a.platform_shutdown_receipt.stat().st_size:raise ValueError('Missing actual platform timer receipt')
            budget=budget_deadline(time.time(),a.gpu_start_unix,a.platform_deadline_unix,a.hourly_price,a.prior_gpu_seconds,a.prior_cpu_seconds,a.prior_stage_spend_yuan,a.prior_total_spend_yuan)
            watch=threading.Thread(target=_hard_stop,args=(budget['deadline_unix'],a.out,stop),daemon=True);watch.start()
        write_json(a.out/'launch.json',dict(argv=sys.argv if argv is None else argv,budget=budget,no_automatic_repeat=True,
            timer_authenticity='Operator must actually set and verify platform timer; file nonempty alone is not platform confirmation'))
        context=load_execution(ROOT,a.plan,a.acceptance,a.dense_reference)
        code={x.relative_to(ROOT).as_posix():digest(x) for x in code_paths()}
        context['fingerprints'].update(code)
        manifest=csv_rows(a.transfer_manifest)
        validate_deployment_manifest(manifest,deployment_records(context,code_paths()))
        frozen=dict(code_sha256=code,plan_path=a.plan.relative_to(ROOT).as_posix(),acceptance_path=a.acceptance.relative_to(ROOT).as_posix(),
            dense_reference_path=a.dense_reference.relative_to(ROOT).as_posix(),plan_sha256=digest(a.plan/'plan.json'),acceptance_sha256=digest(a.acceptance),
            dense_reference_sha256=digest(a.dense_reference),transfer_manifest_sha256=digest(a.transfer_manifest),ffmpeg_sha256=digest(a.ffmpeg))
        if a.stage=='check':
            for r in manifest:
                path=rooted(ROOT,r['path'])
                if path.stat().st_size!=int(r['size_bytes']) or digest(path)!=r['sha256']:raise ValueError('Transferred file differs: '+r['path'])
            if shutil.disk_usage(ROOT).free<12*1024**3:raise ValueError('Need12GiB additional headroom for fit weights/temporary checkpoints/evidence')
            write_json(a.out/'preflight.json',dict(status='cv_static_preflight_passed',decoded_frames=0,files_verified=len(manifest),**frozen));return
        previous=json.loads(a.preflight.read_text(encoding='utf-8'))
        if previous.get('status')!='cv_static_preflight_passed' or any(previous.get(k)!=v for k,v in frozen.items()):raise ValueError('GPU execution differs from static preflight')
        # No second full-JPEG/media scan on paid GPU: dataset hashes and canonical
        # per-video RGB112 closure remain enforced while they are actually used.
        for name in code:
            target=a.out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write((ROOT/name).read_bytes())
            if digest(target)!=code[name]:raise ValueError('Execution source changed while snapshotting')
        write_json(a.out/'execution-inputs.json',dict(**frozen,input_fingerprints=context['fingerprints']))
        completed=[]
        for f in context['folds']:
            fold=f['fold'];directory=a.out/('fold-%02d'%fold);directory.mkdir()
            inner=train_stage(context,directory/'inner',fold,'inner_select')
            outer=train_stage(context,directory/'outer',fold,'fixed_outer',inner)
            dense=evaluate_fold(context,directory/'dense',fold,directory/'outer',inner['selected_epochs'],a.ffmpeg)
            completed.append(dict(inner=inner,outer=outer,dense=dense))
            future=context['folds'][fold:]
            if future:
                forecast=remaining_forecast(completed,future,context['dense']['windows']);remaining=budget['deadline_unix']-time.time()
                write_json(directory/'remaining-budget.json',dict(status='passed' if forecast['seconds']<=remaining else 'refused',remaining_seconds=remaining,**forecast))
                if forecast['seconds']>remaining:raise RuntimeError('Complete remaining folds do not fit; no shortening, fold selection, or automatic rerental')
        summary=summarize_folds([x['dense'] for x in completed],context['sources'])
        if time.time()>budget['deadline_unix']:raise TimeoutError('Budget exhausted before complete five-fold acceptance')
        write_json(a.out/'pipeline-result.json',dict(status='complete_pending_independent_audit',summary=summary,
            per_fold=[dict(fold=i+1,selected_epochs=x['inner']['selected_epochs'],metrics=x['dense']['metrics']) for i,x in enumerate(completed)],
            seconds=time.perf_counter()-began,budget=budget,no_automatic_repeat=True,final_invoice_and_actual_shutdown_require_operator_confirmation=True))
    except Exception as e:
        write_json(a.out/'pipeline-failure.json',dict(status='incomplete_five_fold_metrics_withheld',error_type=type(e).__name__,error=str(e),partial_artifacts_preserved=True))
        raise
    finally:
        stop.set()
        if watch:watch.join(timeout=1)


if __name__=='__main__':main()
