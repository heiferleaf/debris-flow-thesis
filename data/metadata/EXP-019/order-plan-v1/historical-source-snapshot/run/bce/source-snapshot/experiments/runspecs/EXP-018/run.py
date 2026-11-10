"""Authorized three-condition pilot; one shared decode for complete video scoring."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

SOURCE=Path(__file__).resolve()
ROOT=SOURCE.parents[3]
sys.path.insert(0,str(ROOT))
from experiments.common.video.supervised_plan import digest, rooted, csv_rows, write_json
from experiments.common.video.supervised_training import load_inputs
from experiments.common.video.paired_plan import load_pair_plan
from experiments.common.video.paired_training import train
from experiments.common.video.dense_evaluation import evaluate, compatible_pixel_reference


def code_paths():
    common=ROOT/'experiments/common/video'
    return ([common/name for name in ('paired_plan.py','paired_runtime.py','paired_training.py',
             'supervised_plan.py','supervised_runtime.py','supervised_training.py',
             'r2plus1d_classifier.py','dense_evaluation.py','pts_clip_sampling.py')]+
            [ROOT/'experiments/runspecs/EXP-018'/name for name in ('run.py','train.py','prepare.py','run_cloud.sh')]+
            [ROOT/'experiments/runspecs/EXP-016'/name for name in ('run.py','evaluate.py')]+
            [ROOT/'docs/process/experiments'/name for name in ('EXP-018.md','EXP-006-数据使用方案.md')]+
            [ROOT/'.agents/skills/data-usage-protocol/SKILL.md'])


def hard_stop(seconds,out):
    time.sleep(seconds)
    try:(out/'time-limit-reached.txt').write_text('Registered wall limit reached.\n',encoding='utf-8')
    finally:os._exit(124)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=('check','full'),default='check')
    ap.add_argument('--plan',type=Path,default=ROOT/'data/metadata/EXP-018/paired-plan-v1')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--pixel-reference',type=Path,required=True)
    ap.add_argument('--ffmpeg',type=Path,required=True)
    ap.add_argument('--transfer-manifest',type=Path,required=True)
    ap.add_argument('--preflight',type=Path)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--cache-mib',type=int,default=128)
    ap.add_argument('--hourly-price',type=float,default=1.48)
    ap.add_argument('--seconds-limit',type=int,default=6600)
    a=ap.parse_args()
    a.plan=a.plan.resolve()
    a.out=a.out.resolve()
    a.pixel_reference=a.pixel_reference.resolve()
    a.transfer_manifest=a.transfer_manifest.resolve()
    a.out.resolve().relative_to(ROOT/('data/process' if a.stage=='check' else 'data/results'))
    if a.out.exists():raise FileExistsError(a.out)
    if not (math.isfinite(a.hourly_price) and 0<a.hourly_price<=1.48 and 0<a.seconds_limit<=7200):
        raise ValueError('Refuse unregistered price/time')
    paired,epochs=load_pair_plan(ROOT,a.plan)
    base=ROOT/paired['base_plan_path']
    plan,rows,schedule,validation=load_inputs(ROOT,base,check_files=True)
    reference=json.loads(a.pixel_reference.read_text(encoding='utf-8'))
    compatible_pixel_reference(plan,digest(base/'plan.json'),reference,a.pixel_reference.parent/'source-plan.json')
    code={p.relative_to(ROOT).as_posix():digest(p) for p in code_paths()}
    if not a.ffmpeg.is_file():raise FileNotFoundError(a.ffmpeg)
    frozen=dict(code_sha256=code,ffmpeg_sha256=digest(a.ffmpeg),pair_plan_sha256=digest(a.plan/'plan.json'),
                base_plan_sha256=digest(base/'plan.json'),pixel_reference_sha256=digest(a.pixel_reference),
                transfer_manifest_sha256=digest(a.transfer_manifest))
    if a.stage=='check':
        # The original inputs are unchanged. Confirm paths here; JPEG and RGB
        # hashes are checked during actual training, dense pixels during scoring.
        # No expensive cloud video predecode or repeated full cache rehash.
        records=csv_rows(a.transfer_manifest)
        if not records or len({r['path'] for r in records})!=len(records):
            raise ValueError('Invalid code/plan transfer manifest')
        for r in records:
            p=rooted(ROOT,r['path'])
            if p.is_symlink() or p.stat().st_size!=int(r['size_bytes']) or digest(p)!=r['sha256']:
                raise ValueError('Uploaded code or plan changed')
        required=dict(code)
        required.update({(a.plan/filename).relative_to(ROOT).as_posix():digest(a.plan/filename)
                         for filename in ('plan.json','epochs.jsonl')})
        indexed={r['path']:r['sha256'] for r in records}
        if any(indexed.get(name)!=sha for name,sha in required.items()):
            raise ValueError('Upload does not bind complete execution code and plan')
        a.out.mkdir(parents=True)
        result=dict(status='exp018_static_preflight_passed',decoded_frames=0,
                    original_clip_rows=len(rows),validation_windows=len(validation),
                    new_artifacts_verified=len(records),input_policy='existing immutable cache; per-read SHA in training',
                    **frozen)
        write_json(a.out/'preflight.json',result)
        print(json.dumps(dict(status=result['status'],decoded_frames=0,new_files=len(records))),flush=True)
        return
    if a.preflight is None:raise ValueError('Completed no-GPU preflight required')
    previous=json.loads(a.preflight.read_text(encoding='utf-8'))
    if previous.get('status')!='exp018_static_preflight_passed' or any(previous[k]!=v for k,v in frozen.items()):
        raise ValueError('Preflight fingerprints differ')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU stage requires actual CUDA')
    a.out.mkdir(parents=True)
    started=time.monotonic()
    threading.Thread(target=hard_stop,args=(a.seconds_limit,a.out),daemon=True).start()
    write_json(a.out/'launch.json',dict(argv=sys.argv,seconds_limit=a.seconds_limit,
               hourly_price=a.hourly_price,pilot_ceiling_yuan=8,total_goal_ceiling_yuan=20,**frozen))
    kwargs=dict(root=ROOT,plan_dir=a.plan,seed=7,device='cuda',workers=a.workers,
                cache_mib=a.cache_mib,microbatch=8)
    probe=train(out=a.out/'probe-same',mode='same',epochs=1,**kwargs)
    forecast=time.monotonic()-started + 1.25*60*probe['seconds'] + 3000
    write_json(a.out/'budget-probe.json',dict(probe_seconds=probe['seconds'],
                total_forecast_seconds=forecast,dense_reserve_seconds=3000,
                basis='runtime/correctness only; never candidate scores',fits=forecast<=a.seconds_limit))
    if forecast>a.seconds_limit:raise RuntimeError('Measured budget insufficient for the full registered pilot')
    outputs={}
    for mode in ('bce','cross','same'):
        outputs[mode]=train(out=a.out/mode,mode=mode,epochs=20,**kwargs)
    try:
        dense=evaluate(ROOT,base,a.out/'three-dense',a.ffmpeg,
                       {mode:a.out/mode/'best.pt' for mode in outputs},'cuda',a.pixel_reference,
                       input_modes={mode:'rgb_zero' for mode in outputs})
    except Exception as error:
        write_json(a.out/'evaluation-failure.json',dict(status='training_complete_dense_failed',
                   error_type=type(error).__name__,error=str(error),no_automatic_retraining=True))
        raise
    if (len(dense['videos'])!=31 or sum(v['windows'] for v in dense['videos'].values())!=32567
            or set(dense['metrics'])!={'bce','cross','same'}
            or any(v['status']!='complete' for v in dense['metrics'].values())):
        raise RuntimeError('Incomplete evaluation; full pilot metrics withheld')
    if any(digest(ROOT/name)!=sha for name,sha in code.items()):
        raise RuntimeError('Execution sources changed during pilot')
    write_json(a.out/'pipeline-result.json',dict(status='completed',seconds=time.monotonic()-started,
        metrics=dense['metrics'],pair_plan_sha256=frozen['pair_plan_sha256'],
        next_step='independent audit; no automatic repeat seed or CV'))


if __name__=='__main__':main()
