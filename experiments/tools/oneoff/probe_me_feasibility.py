"""One frozen eight-clip CPU input probe. No training, predictions or downloads."""
from pathlib import Path, PurePosixPath
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[3]
SELECTION_SHA='74f0ee37f8d5b3b5e7740b4f4deb6d5b5c17c0f56364283baee4287a96522ba7'
RUBRIC_SHA='302ede3e4e3dd3baf65d8c446ba2f1ff948b6d0223afd71908e02c94720b75f7'
WEIGHT=Path('C:/Users/20443/.cache/torch/hub/checkpoints/r2plus1d_18-91a641e6.pth')
WEIGHT_SHA='91a641e6c2ab531d1aca5f4321b4d802ec5c3babc15df855cdb6e39c6a1107c8'
EXPECTED_OUT=ROOT/'data/process/temporal-me-feasibility/20261102-v1'
HELPERS={'experiments/common/video/supervised_runtime.py':'3b2b4119a0d8c7846b36d355c356d7730a17aa6c4c0a2324ec7bbddbe4fd3d9a',
         'experiments/common/video/pts_clip_sampling.py':'931f67394c82949d81570cb1607d7fb1b3a811d549841d8d46050188b1232178'}


def require(value,message):
    if not value:raise ValueError(message)


def sha_bytes(data):return hashlib.sha256(data).hexdigest()


def safe(name):
    p=PurePosixPath(name)
    require(p.parts and not p.is_absolute() and '..' not in p.parts and p.as_posix()==name and not any(ord(c)<32 or ord(c) in (58,92) for c in name),'Unsafe project-relative input')
    path=ROOT.joinpath(*p.parts);path.resolve().relative_to(ROOT)
    for q in [path]+list(path.parents):
        if q==ROOT:break
        require(not q.is_symlink(),'Symlink input refused')
    return path


def checked(path,expected):
    data=Path(path).read_bytes();require(sha_bytes(data)==expected,'Input SHA mismatch: '+str(path));return data


def csv_bytes(data):return csv.DictReader(io.StringIO(data.decode('utf-8-sig')))


class Output:
    """All output writes are exclusive and capped, with16KiB failure reserve."""
    def __init__(self,path):
        self.path=path;path.mkdir(parents=True,exist_ok=False)
        self.used=0;self.files={};self.lock=threading.Lock();self.end=time.monotonic()+600;self.stop=threading.Event()
        threading.Thread(target=self.watchdog,daemon=True).start()

    def check(self):require(time.monotonic()<self.end,'CPU probe600second limit reached')

    def write(self,name,data):
        self.check()
        with self.lock:
            require(self.used+len(data)<=100000000-16384,'Probe100MB output limit reached')
            target=self.path/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(data)
            self.used+=len(data);self.files[name]=dict(size_bytes=len(data),sha256=sha_bytes(data))

    def json(self,name,value):self.write(name,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8'))

    def failure(self,error):
        payload=json.dumps(dict(status='failed_partial_diagnostic_no_claim',error_type=type(error).__name__,error=str(error)[:2000]),allow_nan=False).encode('utf-8')
        if self.lock.acquire(timeout=.05):
            try:
                target=self.path/'metadata/failure.json';target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists() and self.used+len(payload)<=100000000:
                    with target.open('xb') as f:f.write(payload)
            finally:self.lock.release()

    def watchdog(self):
        if not self.stop.wait(600):
            try:self.failure(TimeoutError('Hard CPU600second stop; partial outputs retained'))
            finally:os._exit(124)


def validate(selection,data):
    paths={r['path']:r['sha256'] for r in selection['inputs']}
    require(len(paths)==6,'Unexpected metadata input list')
    source=list(csv_bytes(data['data/metadata/EXP-016/metadata-v4/video-inputs.csv']))
    sources={r['unit_id']:r for r in source}
    windows={r['clip_id']:r for r in (json.loads(line) for line in data['data/metadata/EXP-016/metadata-v4/windows.jsonl'].decode('utf-8').splitlines())}
    segments={(r['unit_id'],r['segment_id']):r for r in csv_bytes(data['data/metadata/EXP-016/metadata-v4/segments.csv'])}
    assigned={r['clip_id']:r for r in csv_bytes(data['data/metadata/EXP-016/training-plan-v2/assignments.csv']) if r['split']=='train' and r['epoch']=='0'}
    train=set(json.loads(data['data/metadata/splits/EXP-016-development-holdout.json'])['train_video_ids'])
    require(len(train)==119 and len(selection['cases'])==4 and len({r['camera'] for r in selection['cases']})==4,'Frozen119/four-camera scope changed')
    wanted=set();clips=[]
    for number,case in enumerate(selection['cases'],1):
        uid=case['unit_id'];require(uid in train and sources[uid]['split']=='train' and case['pair_id']=='G%02d'%number,'Nontraining/changed case')
        require(len(case['clips'])==2 and {c['window']['kind'] for c in case['clips']}=={'pos','neg'},'Same-video P/N pair changed')
        for suffix,item in zip(('A','B'),case['clips']):
            w=item['window'];segment=item['segment'];cid=w['clip_id'];ix=w['frame_indices'];pts=w['source_pts_seconds']
            require(item['alias']==case['pair_id']+suffix and w==windows[cid] and segment==segments[(uid,w['segment_id'])],'Frozen alias/window/segment differs')
            require(w['unit_id']==uid and w['group_id']==case['group_id']==sources[uid]['group_id'] and w['origin']==segment['origin']=='explicit_human_event','Case source/label provenance differs')
            require(cid in assigned and all(assigned[cid][k]==w[k] for k in ('unit_id','kind')),'Window not in actual epoch0 training')
            require(len(ix)==len(pts)==16 and all(type(i) is int and 0<=i<int(sources[uid]['n_frames']) for i in ix) and all(b>a for a,b in zip(ix,ix[1:])),'Invalid16 real indices')
            require(all(isinstance(t,(int,float)) and __import__('math').isfinite(t) for t in pts) and all(b>a for a,b in zip(pts,pts[1:])) and 2.9-1e-8<=pts[-1]-pts[0]<=3.1+1e-8,'Invalid source PTS')
            require(float(segment['sampling_start_sec'])<=pts[0] and pts[-1]<float(segment['sampling_end_sec']) and float(segment['annotation_start_sec'])<=pts[0] and pts[-1]<float(segment['annotation_end_sec']),'Window crosses annotation/sampling boundary')
            runs=json.loads(sources[uid]['clean_runs_json']);require(any(a<=ix[0] and ix[-1]<b for a,b in runs),'Window crosses cut/QC run; never splice frames')
            for i in ix:require((uid,i) not in wanted,'Repeated selected physical frame');wanted.add((uid,i))
            clips.append(item)
    require(len(wanted)==128 and len(clips)==8 and len({c['window']['clip_id'] for c in clips})==8,'Exact eight-clip/128frame scope changed')
    return wanted,clips


def viewer(pairs):
    # Public payload contains only aliases, relative PTS, and alias PNG paths.
    encoded=json.dumps(pairs,ensure_ascii=True,allow_nan=False)
    return ('''<!doctype html><meta charset="utf-8"><title>16采样帧观察</title>
<style>body{font:16px sans-serif;max-width:1100px;margin:24px auto;background:#f5f5f5;color:#222}section{background:white;padding:14px;margin:18px 0}.pair{display:flex;gap:26px;flex-wrap:wrap}img{display:block;width:224px;height:224px;image-rendering:pixelated;background:#222}button{margin:8px 5px 8px 0}p{line-height:1.5}.time{font-variant-numeric:tabular-nums}</style>
<h1>模型16采样帧，非原生完整帧率录像</h1>
<p>112×112实际模型输入，224×224仅2倍显示，无图像增强。各序列独立播放，按原采样PTS相对间隔保持到下一帧，在第16帧停止；A/B不是连续片段，不跨两段计算运动。原画面文字和水印保留，别名不保证身份盲化。</p>
<p>观察可见运动、可追踪结构和主要变化来源。干沟与有流的静态外观差异本身不支持ME；不由此判断准确率或物理流速。</p><main id="clips"></main><script>
const pairs='''+encoded+''';const controllers=[];
function player(d,parent){let i=0,elapsed=0,base=0,playing=false,request=0;const box=document.createElement('article');box.innerHTML='<h3>'+d.alias+'</h3><img alt="'+d.alias+'"><div class="time"></div><button class="prev">上一帧</button><button class="play" disabled>播放</button><button class="next">下一帧</button><div class="note">载入16帧…</div>';parent.append(box);
const img=box.querySelector('img'),time=box.querySelector('.time'),play=box.querySelector('.play'),note=box.querySelector('.note');
function show(n){i=n;img.src=d.images[i];time.textContent='帧 '+(i+1)+'/16 · 相对PTS '+d.times[i].toFixed(3)+' 秒';}
function pause(){if(playing)elapsed=Math.min(d.times[15],(performance.now()-base)/1000);playing=false;cancelAnimationFrame(request);play.textContent='播放';}
function tick(now){if(!playing)return;elapsed=(now-base)/1000;let n=0;while(n<15&&d.times[n+1]<=elapsed)n++;if(n!==i)show(n);if(elapsed>=d.times[15]){show(15);pause();return;}request=requestAnimationFrame(tick);}
play.onclick=()=>{if(playing){pause();return;}if(elapsed>=d.times[15]){elapsed=0;show(0);}base=performance.now()-elapsed*1000;playing=true;play.textContent='暂停';request=requestAnimationFrame(tick);};
box.querySelector('.prev').onclick=()=>{pause();show(Math.max(0,i-1));elapsed=d.times[i];};box.querySelector('.next').onclick=()=>{pause();show(Math.min(15,i+1));elapsed=d.times[i];};
Promise.all(d.images.map(src=>new Promise((resolve,reject)=>{const p=new Image();p.onload=resolve;p.onerror=reject;p.src=src;}))).then(()=>{show(0);play.disabled=false;note.textContent='可单步检查全部16帧；不插帧、不改速。';}).catch(()=>{note.textContent='缺少PNG，拒绝播放。';});controllers.push(pause);}
for(const p of pairs){const section=document.createElement('section');section.innerHTML='<h2>'+p.pair+'</h2><div class="pair"></div>';document.querySelector('main').append(section);for(const d of p.clips)player(d,section.querySelector('.pair'));}
document.addEventListener('visibilitychange',()=>{if(document.hidden)controllers.forEach(pause=>pause());});
</script>''').encode('utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True);parser.add_argument('--selection-sha256',required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    require(args.selection_sha256==SELECTION_SHA,'Only the frozen eight-clip selection is allowed')
    selection_data=checked(args.selection,SELECTION_SHA);selection=json.loads(selection_data)
    rubric_data=checked(args.selection.with_name('observation-rubric.json'),RUBRIC_SHA);rubric=json.loads(rubric_data)
    require(rubric['selection_sha256']==SELECTION_SHA and args.out.resolve()==EXPECTED_OUT,'Rubric/output scope differs')
    require(selection['limits']['runtime_seconds']==600 and selection['limits']['output_bytes']==100000000,'Frozen resource limit differs')
    out=Output(args.out.resolve());began=time.monotonic()
    try:
        source=Path(__file__).resolve();code=source.read_bytes();data={r['path']:checked(safe(r['path']),r['sha256']) for r in selection['inputs']}
        for name,digest in HELPERS.items():checked(safe(name),digest)
        wanted,clips=validate(selection,data)
        completion=json.loads(data['data/process/EXP-016/metadata-v4/cache-completion.json'])
        require(completion['status']=='supervised_cache_ready','Existing full cache acceptance incomplete')
        # Exactly one index-file read and one CSV traversal; retain128 rows only.
        index_data=checked(safe(completion['verified_cache_index']),completion['verified_cache_index_sha256']);refs={}
        for row in csv_bytes(index_data):
            key=row['unit_id'],int(row['frame_index'])
            if key in wanted:require(key not in refs,'Repeated selected cache index identity');refs[key]=row
        require(set(refs)==wanted,'Selected128frames missing from actual verified cache index')
        del index_data
        checked(WEIGHT,WEIGHT_SHA);require(WEIGHT.stat().st_size==126162996,'Public K400 file size differs')
        sys.path.insert(0,str(ROOT))
        import cv2
        import numpy as np
        import torch
        from torch import nn
        from experiments.common.video.supervised_runtime import PixelCache,FrameRef,ClipInput
        cv2.setNumThreads(1);torch.set_num_threads(4)
        state=torch.load(str(WEIGHT),map_location='cpu',weights_only=True)
        require(state['stem.0.weight'].shape==(45,3,1,7,7),'Public spatial stem shape differs')
        stem=nn.Sequential(nn.Conv3d(6,45,(1,7,7),(1,2,2),(0,3,3),bias=False),nn.BatchNorm3d(45),nn.ReLU())
        with torch.no_grad():stem[0].weight.zero_();stem[0].weight[:,:3].copy_(state['stem.0.weight'])
        stem[1].load_state_dict({k:state['stem.1.'+k] for k in stem[1].state_dict()});del state
        stem.requires_grad_(False);stem.eval();transform=ClipInput('rgb_zero').eval();cache=PixelCache(16*1024**2)
        out.write('metadata/selection.json',selection_data);out.write('metadata/observation-rubric.json',rubric_data);out.write('metadata/generator-source.py',code)
        public=[];results=[];frames=[]
        for pair in selection['cases']:
            visible=dict(pair=pair['pair_id'],clips=[])
            for item in pair['clips']:
                out.check();alias=item['alias'];w=item['window'];images=[]
                for i in w['frame_indices']:
                    row=refs[(w['unit_id'],i)];require(abs(float(row['source_pts_seconds'])-w['source_pts_seconds'][len(images)])<=1e-8,'Cache PTS differs from selected real frame')
                    images.append(cache.get(FrameRef(str(safe(row['cache_path'])),row['jpeg_sha256'],row['rgb112_sha256'])))
                pixels=np.stack(images);require(pixels.shape==(16,112,112,3),'Actual RGB112 shape differs')
                with torch.inference_mode():
                    normalized=transform(torch.from_numpy(pixels[None]));feature=stem(normalized)[0]
                    require(tuple(feature.shape)==(45,16,56,56) and bool(torch.isfinite(feature).all()),'Invalid frozen spatial stem tensor')
                    feat=feature.numpy().copy()
                signed=feat[:,1:]-feat[:,:-1];delta=np.abs(signed);raw=np.abs(pixels[1:].astype(np.float32)-pixels[:-1].astype(np.float32))/255.
                signed_mean=signed.mean(axis=(2,3),dtype=np.float64).T;absolute_mean=delta.mean(axis=(2,3),dtype=np.float64).T
                cancellation=[[None if denominator==0 else float(abs(numerator)/denominator) for numerator,denominator in zip(a,b)] for a,b in zip(signed_mean,absolute_mean)]
                feature_edges=np.linspace(0,56,4,dtype=int).tolist();rgb_edges=np.linspace(0,112,4,dtype=int).tolist()
                def grid(array,edges,channel_first):
                    return [[float(array[:,:,edges[r]:edges[r+1],edges[c]:edges[c+1]].mean()) if channel_first else float(array[:,edges[r]:edges[r+1],edges[c]:edges[c+1],:].mean()) for c in range(3)] for r in range(3)]
                # Retain per-transition3x3 means, never select a high-change ROI.
                fgrid=[grid(delta[:,t:t+1],feature_edges,True) for t in range(15)]
                rgbgrid=[grid(raw[t:t+1],rgb_edges,False) for t in range(15)]
                metric=dict(alias=alias,window=w,annotation_edge_margin_seconds=item['annotation_edge_margin_seconds'],tensor_summary=dict(shape=list(feat.shape),dtype=str(feat.dtype),sha256=sha_bytes(feat.tobytes(order='C')),minimum=float(feat.min()),maximum=float(feat.max()),mean=float(feat.mean()),std=float(feat.std())),mean_abs_feature_per_channel=np.abs(feat).mean(axis=(1,2,3),dtype=np.float64).tolist(),signed_spatial_mean_delta_15x45=signed_mean.tolist(),absdiff_spatial_mean_15x45=absolute_mean.tolist(),abs_signed_mean_over_mean_abs_15x45=cancellation,cancellation_denominator_zero='None; ratio undefined when all spatial differences are zero',raw_RGB_absdiff_mean_15=raw.mean(axis=(1,2,3)).tolist(),raw_RGB_absdiff_mean_15x3=raw.mean(axis=(1,2)).tolist(),feature_absdiff_grid_15x3x3=fgrid,raw_RGB_absdiff_grid_15x3x3=rgbgrid,grid_edges=dict(feature=feature_edges,rgb112=rgb_edges),proxy_scope='identity-G difference only; signed-mean and mean-absolute are distinct. No learned G/full ME, displacement, threshold, AUC or performance inference; no cross-clip or loop-wrap difference')
                require(all(np.isfinite(v).all() for v in (feat,delta,raw)),'Nonfinite diagnostic')
                out.json('metadata/'+alias+'-metrics.json',metric);results.append(dict(alias=alias,clip_id=w['clip_id'],kind=w['kind'],metrics='metadata/'+alias+'-metrics.json'))
                names=[]
                for t,rgb in enumerate(pixels):
                    ok,png=cv2.imencode('.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));require(ok,'PNG encode failed')
                    decoded=cv2.cvtColor(cv2.imdecode(png,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB);require(np.array_equal(decoded,rgb),'PNG must preserve actual RGB112')
                    name=alias+'/%02d.png'%t;out.write('public/'+name,png.tobytes());names.append(name)
                    row=refs[(w['unit_id'],w['frame_indices'][t])];frames.append(dict(alias=alias,frame=t,frame_index=w['frame_indices'][t],cache_path=row['cache_path'],jpeg_sha256=row['jpeg_sha256'],rgb112_sha256=row['rgb112_sha256'],png='public/'+name,png_sha256=sha_bytes(png.tobytes())))
                times=[float(t-w['source_pts_seconds'][0]) for t in w['source_pts_seconds']];visible['clips'].append(dict(alias=alias,times=times,images=names))
                print(json.dumps(dict(stage='single_CPU_spatial_stem_probe_complete',alias=alias)),flush=True)
            public.append(visible)
        require(cache.misses==128 and len(frames)==128 and len(results)==8,'Unexpected reads/output scope')
        for name,expected in [(r['path'],r['sha256']) for r in selection['inputs']]+list(HELPERS.items()):require(sha_bytes(safe(name).read_bytes())==expected,'Input changed during probe')
        require(sha_bytes(source.read_bytes())==sha_bytes(code) and sha_bytes(WEIGHT.read_bytes())==WEIGHT_SHA,'Probe code/public weight changed')
        out.json('metadata/frames.json',frames)
        out.json('metadata/result.json',dict(status='eight_clip_CPU_diagnostic_complete_descriptive_only',selection_sha256=SELECTION_SHA,rubric_sha256=RUBRIC_SHA,pretrained_weight_path=str(WEIGHT),pretrained_weight_sha256=WEIGHT_SHA,pretrained_weight_bytes=126162996,code_sha256=sha_bytes(code),input_sha256={r['path']:r['sha256'] for r in selection['inputs']},cache_index_sha256=completion['verified_cache_index_sha256'],helper_sha256=HELPERS,environment=dict(python=sys.version,torch=torch.__version__,opencv=cv2.__version__,numpy=np.__version__,platform=platform.platform(),device='cpu',threads=4),stem='Only stem[0] spatial1x7x7 + stem[1] BN(eval) + stem[2] ReLU; temporal RF1, no stem[3]/layer1/head',parameters_require_grad=any(p.requires_grad for p in stem.parameters()),BN_training=stem[1].training,clips=results,cache_reads=cache.misses,seconds=time.monotonic()-began,viewer_identity_scope='aliases only in HTML/JSON; original pixel watermarks retained, not identity blindness',forbidden_claims=selection['forbidden_claims'],files_before_result=dict(out.files)))
        # Publish viewer only after all eight clips and source closure pass.
        out.write('public/index.html',viewer(public))
        out.json('metadata/output-manifest.json',dict(status='complete',bytes_before_manifest=out.used,files=dict(out.files),no_large_media_or_JPEG_copies=True,full_feature_tensors_saved=False))
    except BaseException as error:out.failure(error);raise
    finally:out.stop.set()


if __name__=='__main__':main()
