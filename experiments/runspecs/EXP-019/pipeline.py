"""One EXP-019 execution: separate timing probes, fresh A/B, shared dense decode.

No rental/start/platform API calls. Root/operator owns the actual GPU-start time,
prior ledger, platform shutdown timer and its receipt. run.sh supplies EXIT shutdown.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

SOURCE=Path(__file__).resolve()
ROOT=SOURCE.parents[3]
sys.path.insert(0,str(ROOT))
from experiments.common.video.order_training import MODES,_load_contract,probe,train
from experiments.common.video.order_dense_evaluation import evaluate
from experiments.common.video.supervised_plan import csv_rows,digest,rooted,write_json
from experiments.common.video.dense_evaluation import compatible_pixel_reference

GPU_LIMIT=7200.
STAGE_LIMIT=4.
TOTAL_LIMIT=20.
DENSE_OBSERVED_THREE_MODELS=1998.1201581954956
DENSE_RESERVE=2500.  # >=1.25x the measured THREE-model run, retained for TWO models.
FINALIZATION_RESERVE=300.
SHUTDOWN_RESERVE=60.
CHARGE_MARGIN=.10
NO_GPU_COST_RESERVE=.05  # Up to0.5h x0.10 CNY/h; operator still guards its time.


def code_paths():
    common=ROOT/'experiments/common/video'
    names=('temporal_order.py','order_plan.py','order_training.py','order_dense_evaluation.py',
           'order_cache_acceptance.py','paired_plan.py','paired_runtime.py','supervised_plan.py',
           'supervised_runtime.py','supervised_training.py','r2plus1d_classifier.py',
           'dense_evaluation.py','pts_clip_sampling.py')
    paths=[common/name for name in names]
    paths += [ROOT/'experiments/runspecs/EXP-019'/name for name in
              ('pipeline.py','run.sh','train.py','prepare.py','verify_cache.py')]
    paths += [ROOT/'docs/process/experiments'/name for name in
              ('EXP-019.md','EXP-006-数据使用方案.md')]
    paths.append(ROOT/'.agents/skills/data-usage-protocol/SKILL.md')
    return paths


def budget_deadline(*,now,gpu_start,platform_deadline,price,prior_gpu_seconds,
                    prior_stage_yuan,prior_total_yuan,seconds_limit):
    values=(now,gpu_start,platform_deadline,price,prior_gpu_seconds,prior_stage_yuan,prior_total_yuan,seconds_limit)
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite budget input')
    if (not 0<price<=1.48 or not 0<seconds_limit<=GPU_LIMIT
            or not 0<=prior_gpu_seconds<GPU_LIMIT or not 0<=prior_stage_yuan<STAGE_LIMIT
            or not prior_stage_yuan<=prior_total_yuan<TOTAL_LIMIT or gpu_start>now+5):
        raise ValueError('Unregistered price/time/ledger values')
    gpu_end=gpu_start+GPU_LIMIT-prior_gpu_seconds
    if platform_deadline<=now or platform_deadline>gpu_end+1:
        raise ValueError('Confirmed platform timer must be within remaining cumulative2h GPU limit')
    money=min(STAGE_LIMIT-prior_stage_yuan,TOTAL_LIMIT-prior_total_yuan)-CHARGE_MARGIN-NO_GPU_COST_RESERVE
    if money<=0:raise ValueError('No stage/total budget after shutdown and no-GPU reserves')
    money_end=gpu_start+money/price*3600
    deadline=min(gpu_end,platform_deadline,money_end,now+seconds_limit)-SHUTDOWN_RESERVE
    if deadline<=now:raise ValueError('No runtime remains inside registered limits')
    return dict(deadline_unix=deadline,remaining_seconds=deadline-now,gpu_session_started_unix=gpu_start,
        cumulative_gpu_limit_seconds=GPU_LIMIT,prior_gpu_seconds=prior_gpu_seconds,
        hourly_price=price,prior_stage_spend_yuan=prior_stage_yuan,prior_total_spend_yuan=prior_total_yuan,
        stage_ceiling_yuan=STAGE_LIMIT,total_ceiling_yuan=TOTAL_LIMIT,platform_deadline_unix=platform_deadline,
        shutdown_reserve_seconds=SHUTDOWN_RESERVE,charge_margin_yuan=CHARGE_MARGIN,
        remaining_no_gpu_cost_reserve_yuan=NO_GPU_COST_RESERVE)


def forecast(probes):
    if set(probes)!=set(MODES):raise ValueError('Need both independent one-epoch runtime probes')
    loops=[];setups=[]
    for row in probes.values():
        if (row['status']!='runtime_probe_complete' or row['completed_epochs']!=1 or row['optimizer_steps']!=21
                or not math.isfinite(row['seconds']) or row['seconds']<=0
                or not math.isfinite(row['wall_seconds']) or row['wall_seconds']<row['seconds']):
            raise ValueError('Incomplete/nonfinite runtime probe')
        loops.append(row['seconds']);setups.append(row['wall_seconds']-row['seconds'])
    training=1.25*40*max(loops)
    setup=2*max(setups)
    return dict(remaining_forecast_seconds=training+setup+DENSE_RESERVE+FINALIZATION_RESERVE,
        training_forecast_seconds=training,setup_forecast_seconds=setup,dense_reserve_seconds=DENSE_RESERVE,
        finalization_reserve_seconds=FINALIZATION_RESERVE,
        observed_exp018_three_model_dense_seconds=DENSE_OBSERVED_THREE_MODELS,
        linear_two_of_three_seconds=2*DENSE_OBSERVED_THREE_MODELS/3,
        dense_policy='retain >=1.25x measured3-model wall time for2 models; do not scale shared decoding down by2/3',
        selection_basis='elapsed time and technical completeness only; no scores/metrics/selected-epoch quality gate')


def _snapshot(out,code,inputs):
    for name,expected in {**code,**inputs}.items():
        source=rooted(ROOT,name)
        encoded=source.read_bytes()
        if digest(source)!=expected:raise ValueError('Snapshot source changed: '+name)
        target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(encoded)
        if digest(target)!=expected:raise ValueError('Snapshot readback differs')


def _request_shutdown(out):
    """Fixed AutoDL command, bounded and detached from inherited tee/worker pipes."""
    log=None;log_error=None
    try:
        try:log=(out/'hard-stop-shutdown.log').open('xb')
        except Exception as error:log_error=type(error).__name__+': '+str(error)
        completed=subprocess.run(['/bin/bash','/usr/bin/shutdown'],stdin=subprocess.DEVNULL,
            stdout=log if log is not None else subprocess.DEVNULL,stderr=subprocess.STDOUT,
            start_new_session=True,timeout=15,check=False)
        return dict(returncode=completed.returncode,log_file='hard-stop-shutdown.log' if log is not None else None,
                    log_open_error=log_error,timeout_seconds=15)
    finally:
        if log is not None:log.close()


def _hard_stop(deadline,out,stop):
    if stop.wait(max(0,deadline-time.time())):return
    try:
        try:
            write_json(out/'hard-time-limit.json',dict(status='time_limit_reached',deadline_unix=deadline,now_unix=time.time(),metrics_withheld=True))
        except Exception:pass  # A full/unwritable disk must not suppress the shutdown request.
        try:attempt=dict(status='shutdown_command_returned',**_request_shutdown(out))
        except Exception as error:
            attempt=dict(status='shutdown_command_failed',error_type=type(error).__name__,error=str(error))
        try:write_json(out/'hard-stop-shutdown.json',attempt)
        except Exception:pass
    finally:os._exit(124)  # Outer EXIT trap and the independent platform timer are additional guards.


def _parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['check','full'],default='check')
    for key in ('plan','out','cache-acceptance','pixel-reference','ffmpeg','transfer-manifest'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--preflight',type=Path)
    p.add_argument('--gpu-start-unix',type=float)
    p.add_argument('--platform-deadline-unix',type=float)
    p.add_argument('--platform-shutdown-receipt',type=Path)
    p.add_argument('--hourly-price',type=float,default=1.48)
    p.add_argument('--prior-gpu-seconds',type=float,default=0.)
    p.add_argument('--prior-stage-spend-yuan',type=float)
    p.add_argument('--prior-total-spend-yuan',type=float)
    p.add_argument('--seconds-limit',type=float,default=6600.)
    return p


def main(argv=None):
    a=_parser().parse_args(argv)
    for name in ('plan','out','cache_acceptance','pixel_reference','ffmpeg','transfer_manifest'):
        setattr(a,name,getattr(a,name).resolve())
    a.out.relative_to(ROOT/('data/process' if a.stage=='check' else 'data/results'))
    if a.out.exists():raise FileExistsError(a.out)
    budget=None;stop=threading.Event();watchdog=None
    if a.stage=='full':
        if any(x is None for x in (a.gpu_start_unix,a.platform_deadline_unix,a.prior_stage_spend_yuan,a.prior_total_spend_yuan,a.preflight,a.platform_shutdown_receipt)):
            raise ValueError('Full stage requires preflight, actual GPU-start/ledger and confirmed platform timer receipt')
        a.platform_shutdown_receipt=a.platform_shutdown_receipt.resolve()
        if not a.platform_shutdown_receipt.is_file() or not a.platform_shutdown_receipt.stat().st_size:
            raise ValueError('Missing independent platform shutdown receipt')
        budget=budget_deadline(now=time.time(),gpu_start=a.gpu_start_unix,platform_deadline=a.platform_deadline_unix,
            price=a.hourly_price,prior_gpu_seconds=a.prior_gpu_seconds,prior_stage_yuan=a.prior_stage_spend_yuan,
            prior_total_yuan=a.prior_total_spend_yuan,seconds_limit=a.seconds_limit)
        a.out.mkdir(parents=True,exist_ok=False)
        watchdog=threading.Thread(target=_hard_stop,args=(budget['deadline_unix'],a.out,stop),daemon=True);watchdog.start()
        write_json(a.out/'launch.json',dict(argv=sys.argv if argv is None else argv,budget=budget,
            platform_timer_receipt_sha256=digest(a.platform_shutdown_receipt),
            platform_timer_confirmation_scope='operator-supplied receipt; root/controller owns setting and verifying platform API',
            status='started_preflight_pending',no_automatic_repeat=True))
    began=time.monotonic()
    try:
        plan,base,rows,epochs,val,bank=_load_contract(ROOT,a.plan,a.cache_acceptance)
        reference=json.loads(a.pixel_reference.read_text(encoding='utf-8'))
        compatible_pixel_reference(base,plan['base_plan_sha256'],reference,a.pixel_reference.parent/'source-plan.json')
        if len(reference['videos'])!=31:raise ValueError('Expected the existing31-video label-free reference')
        code={p.relative_to(ROOT).as_posix():digest(p) for p in code_paths()}
        small={p.relative_to(ROOT).as_posix():digest(p) for p in a.plan.iterdir() if p.is_file()}
        small[a.cache_acceptance.relative_to(ROOT).as_posix()]=digest(a.cache_acceptance)
        frozen=dict(code_sha256=code,order_plan_sha256=digest(a.plan/'plan.json'),
            cache_acceptance_sha256=digest(a.cache_acceptance),pixel_reference_sha256=digest(a.pixel_reference),
            ffmpeg_sha256=digest(a.ffmpeg),transfer_manifest_sha256=digest(a.transfer_manifest))
        records=csv_rows(a.transfer_manifest)
        if not records or len({r['path'] for r in records})!=len(records):raise ValueError('Invalid transfer manifest')
        indexed={r['path']:r for r in records}
        for row in records:
            path=rooted(ROOT,row['path'])
            if path.is_symlink() or path.stat().st_size!=int(row['size_bytes']) or digest(path)!=row['sha256']:
                raise ValueError('Transferred artifact differs: '+row['path'])
        for name,expected in {**code,**small}.items():
            if name not in indexed or indexed[name]['sha256']!=expected:raise ValueError('Execution/plan not fully bound by transfer manifest: '+name)
        if a.stage=='check':
            a.out.mkdir(parents=True,exist_ok=False)
            write_json(a.out/'preflight.json',dict(status='exp019_static_preflight_passed',decoded_frames=0,
                actual_cache_acceptance_bound=True,training_windows_per_epoch=504,fixed_validation=136,
                new_artifacts_verified=len(records),**frozen))
            print(json.dumps(dict(status='exp019_static_preflight_passed',decoded_frames=0)),flush=True);return
        previous=json.loads(a.preflight.read_text(encoding='utf-8'))
        if previous.get('status')!='exp019_static_preflight_passed' or any(previous.get(k)!=v for k,v in frozen.items()):
            raise ValueError('Full stage differs from completed static preflight')
        import torch
        if not torch.cuda.is_available():raise RuntimeError('Full stage requires actual CUDA')
        _snapshot(a.out,code,small)
        write_json(a.out/'execution-inputs.json',dict(**frozen,preflight_sha256=digest(a.preflight),
            budget=budget,canonical_validation_ids=[rows[i][0] for i in val]))
        kwargs=dict(root=ROOT,plan_dir=a.plan,device='cuda',workers=2,cache_mib=128,cache_acceptance=a.cache_acceptance)
        probes={};measured_init=None
        for mode in MODES:
            started=time.monotonic()
            result=probe(out=a.out/('probe-'+mode),mode=mode,
                         expected_initial_state_sha256=measured_init,**kwargs)
            result=dict(result,wall_seconds=time.monotonic()-started)
            probes[mode]=result
            if measured_init is None:measured_init=result['initial_state_sha256']
        projected=forecast(probes);remaining=budget['deadline_unix']-time.time()
        gate=dict(status='passed' if projected['remaining_forecast_seconds']<=remaining else 'refused',
            gpu_runtime_remaining_seconds=remaining,probes={mode:{k:r[k] for k in ('status','seconds','wall_seconds','completed_epochs','optimizer_steps','initial_state_sha256')} for mode,r in probes.items()},
            **projected)
        write_json(a.out/'budget-probe.json',gate)
        if gate['status']!='passed':raise RuntimeError('Measured runtime cannot fit BOTH20-epoch branches plus complete shared dense evaluation')
        results={}
        for mode in MODES:
            if time.time()>=budget['deadline_unix']:raise TimeoutError('Runtime budget reached')
            if results:
                remaining=budget['deadline_unix']-time.time()
                needed=1.25*max(r['seconds'] for r in results.values())+DENSE_RESERVE+FINALIZATION_RESERVE
                write_json(a.out/'budget-before-second-branch.json',dict(
                    status='passed' if needed<=remaining else 'refused',
                    remaining_seconds=remaining,remaining_forecast_seconds=needed,
                    basis='first formal branch elapsed time only; no model scores'))
                if needed>remaining:raise RuntimeError('Remaining budget cannot complete the second full branch and dense evaluation')
            result=train(out=a.out/mode,mode=mode,expected_initial_state_sha256=measured_init,**kwargs)
            if result['completed_epochs']!=20 or result['optimizer_steps']!=420 or result['initial_state_sha256']!=measured_init:
                raise RuntimeError('Incomplete/unmatched formal order branch')
            results[mode]=result
        remaining=budget['deadline_unix']-time.time()
        write_json(a.out/'budget-before-dense.json',dict(status='passed' if remaining>=DENSE_RESERVE+FINALIZATION_RESERVE else 'refused',
            remaining_seconds=remaining,required_reserve_seconds=DENSE_RESERVE+FINALIZATION_RESERVE))
        if remaining<DENSE_RESERVE+FINALIZATION_RESERVE:
            raise RuntimeError('Complete dense evaluation no longer fits; preserve training and stop')
        dense=evaluate(ROOT,a.plan,a.out/'two-dense',a.ffmpeg,
            {mode:a.out/mode/'best.pt' for mode in MODES},pixel_reference=a.pixel_reference,
            cache_acceptance=a.cache_acceptance,device='cuda')
        if (len(dense['videos'])!=31 or sum(v['accepted_windows'] for v in dense['videos'].values())!=32567
                or set(dense['metrics'])!=set(MODES) or any(m['status']!='complete' for m in dense['metrics'].values())):
            raise RuntimeError('Incomplete evaluation; full comparison metrics withheld')
        if any(digest(ROOT/name)!=value for name,value in code.items()):raise RuntimeError('Execution sources changed')
        now=time.time();gpu_session_seconds=now-a.gpu_start_unix
        stage_projected=a.prior_stage_spend_yuan+gpu_session_seconds*a.hourly_price/3600
        total_projected=a.prior_total_spend_yuan+gpu_session_seconds*a.hourly_price/3600
        if now>budget['deadline_unix'] or stage_projected>STAGE_LIMIT or total_projected>TOTAL_LIMIT:
            raise RuntimeError('Final time/cost accounting outside registered limits')
        write_json(a.out/'pipeline-result.json',dict(status='completed_pending_independent_audit',
            seconds=time.monotonic()-began,stage_projected_spend_yuan=stage_projected,
            total_projected_spend_yuan=total_projected,projected_cost_is_not_final_invoice=True,
            metrics=dense['metrics'],order_plan_sha256=frozen['order_plan_sha256'],
            dense_result_sha256=digest(a.out/'two-dense/result.json'),budget=budget,
            next_step='independent audit/invoice/shutdown receipt; no automatic seed/restart/CV'))
    except Exception as error:
        if a.out.is_dir() and not (a.out/'pipeline-failure.json').exists():
            write_json(a.out/'pipeline-failure.json',dict(status='failed_complete_metrics_withheld',
                error_type=type(error).__name__,error=str(error),seconds=time.monotonic()-began,
                no_automatic_repeat=True,partial_artifacts_preserved=True))
        raise
    finally:
        stop.set()
        if watchdog:watchdog.join(timeout=1)


if __name__=='__main__':main()
