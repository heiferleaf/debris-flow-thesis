"""Publish one finished local trial without replacing historical media."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import time

import cv2

ROOT=Path(__file__).resolve().parents[3]
JOBS=ROOT/'data/process/synthetic-video/local-workbench/jobs'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def publish(args):
    output=args.output.resolve()
    base=read(JOBS/args.base_job/'state.json')
    previous=read(JOBS/args.previous_job/'state.json') if args.previous_job else base
    cap=cv2.VideoCapture(str(output/'candidate.mp4'))
    frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    width,height=int(cap.get(3)),int(cap.get(4))
    if frames!=288 or fps!=24:raise ValueError('Expected 12 seconds at 24 fps')
    cap.set(cv2.CAP_PROP_POS_FRAMES,144);ok,frame=cap.read();cap.release()
    if not ok:raise ValueError('Cannot decode middle frame')
    cv2.imwrite(str(output/'review-6s.png'),frame)
    identity=args.job_id or secrets.token_hex(12)
    folder=JOBS/identity
    if folder.exists():raise ValueError('Published job already exists')
    dest=folder/'output';dest.mkdir(parents=True)
    record=read(output/'result.json')
    reference=args.reference or (JOBS/previous['id']/'output/first-frame.png')
    for source,name in [(output/'candidate.mp4','candidate.mp4'),(output/'result.json','result.json'),(reference,'first-frame.png')]:
        os.link(source,dest/name)
    state={k:copy.deepcopy(base[k]) for k in ('scene_id','upload_id','first_frame_url','geometry') if k in base}
    state.update(id=identity,name=args.name,batch='water-physics-comparison',status='completed',mode='full',
        quality='physics-flow-neural-appearance',duration=12,fps=24,progress=100,created_at=time.time(),
        message=args.description,preview_description=args.description,user_accepted=False,
        direction_accepted=args.direction_accepted,training_ready=False,active_output=str(output.relative_to(ROOT)),
        size=[width,height],method_record=record,previous_comparison_id=previous['id'],previous_label='播放上一个版本',
        video_sha256=hashlib.sha256((output/'candidate.mp4').read_bytes()).hexdigest())
    save(folder/'state.json',state)
    base['physics_comparison_id']=identity;base.pop('physics_before_after_id',None)
    base['quality_issue']=args.description
    save(JOBS/args.base_job/'state.json',base)
    print(json.dumps({'job_id':identity,'frames':frames,'fps':fps,'size':[width,height]}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--base-job',required=True)
    p.add_argument('--previous-job');p.add_argument('--job-id');p.add_argument('--reference',type=Path)
    p.add_argument('--name',required=True);p.add_argument('--description',required=True)
    p.add_argument('--direction-accepted',action='store_true')
    publish(p.parse_args())
