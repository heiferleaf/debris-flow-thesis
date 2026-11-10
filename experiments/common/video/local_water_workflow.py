"""Local upload, geometry and bounded subprocess jobs; no provider API calls."""
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time

from PIL import Image, ImageOps, UnidentifiedImageError


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else (default or {})


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(path.name+'.'+secrets.token_hex(4)+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def points(value, minimum=3, maximum=160):
    if not isinstance(value,list) or not minimum<=len(value)<=maximum:
        raise ValueError('水面与保护区至少3点，流向至少2点。')
    if any(not isinstance(p,list) or len(p)!=2 or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<=v<=1 for v in p) for p in value):
        raise ValueError('标记必须在图片内。')
    return value


class LocalWater:
    def __init__(self, root, cloud_enabled=True):
        self.root=Path(root).resolve()
        self.base=self.root/'data/process/synthetic-video/local-workbench'
        self.uploads=self.base/'uploads';self.jobs=self.base/'jobs'
        self.lock=threading.RLock();self.active=None;self.process=None
        self.cloud_enabled=cloud_enabled

    def location(self, kind, identity):
        if not isinstance(identity,str) or not re.fullmatch('[a-f0-9]{24}',identity):
            raise ValueError('记录编号无效。')
        return (self.uploads if kind=='upload' else self.jobs)/identity

    def environment(self):
        pretrained=self.root/'data/results/synthetic-video/pretrained/slr-sfs-fa8d610'
        weights=pretrained/'baseline2-motion2-fixedMotionFinetune.pth'
        source=pretrained/'source/SLR-SFS-fa8d61046dfa19d80c5605be4a64e71f499b58f7'
        return {'ready':weights.is_file() and (source/'models/networks/utilities.py').is_file(),
                'weights_ready':weights.is_file(),'source_ready':source.is_dir(),
                'message':'本机运行，不上传素材，不调用付费API。',
                'cloud_enabled':self.cloud_enabled}

    def snapshot(self):
        jobs=[]
        if self.jobs.exists():
            for p in sorted(self.jobs.glob('*/state.json'),key=lambda x:x.stat().st_mtime,reverse=True)[:40]:
                value=read(p)
                if value.get('status')=='running' and value.get('id')!=self.active:
                    value={**value,'status':'interrupted','message':'服务已重启；此任务未完成，可按原配置重新提交。'}
                jobs.append(value)
        return {'environment':self.environment(),'jobs':jobs,'active_id':self.active}

    def upload(self,data):
        encoded=data.get('image','')
        if not isinstance(encoded,str) or len(encoded)>28_000_000:
            raise ValueError('图片过大，请使用20 MB以内的PNG或JPEG。')
        try:
            raw=base64.b64decode(encoded.split(',',1)[-1],validate=True)
            im=Image.open(io.BytesIO(raw))
            if im.format not in ('PNG','JPEG','WEBP') or im.width*im.height>24_000_000 or min(im.size)<128:
                raise ValueError('请使用至少128像素、总像素不超过2400万的PNG、JPEG或WebP。')
            im=ImageOps.exif_transpose(im).convert('RGB')
        except (UnidentifiedImageError, OSError, base64.binascii.Error):
            raise ValueError('无法读取图片，请上传有效的PNG或JPEG。')
        original_size=list(im.size)
        im.thumbnail((2048,2048),Image.Resampling.LANCZOS)
        w,h=im.size
        if w%2 or h%2:im=im.resize((w-w%2,h-h%2),Image.Resampling.LANCZOS)
        identity=secrets.token_hex(12);folder=self.location('upload',identity);folder.mkdir(parents=True)
        path=folder/'wet-first-frame.png';im.save(path)
        value={'id':identity,'name':str(data.get('name','首帧'))[:100],'size':list(im.size),'original_size':original_size,
               'source_sha256':hashlib.sha256(raw).hexdigest(),'working_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
               'created_at':time.time(),'image_url':f'/api/local/uploads/{identity}/image.png'}
        write(folder/'source.json',value)
        return value

    def draft(self,identity,data=None):
        folder=self.location('upload',identity)
        if not (folder/'source.json').is_file():raise ValueError('请先上传首帧。')
        if data is not None:
            draft={'polygon':points(data.get('polygon'),0),'flow':points(data.get('flow'),0,80),
                   'exclusions':[points(p) for p in data.get('exclusions',[])]}
            if len(draft['exclusions'])>30:raise ValueError('保护区最多30个。')
            write(folder/'draft.json',draft)
            source=read(folder/'source.json')
            if source.get('role')=='dry_scene_base':
                path=self.base/'scene-preparation'/(source['scene_id']+'.json')
                state=read(path)
                if state.get('geometry')!=draft:
                    state.update(status='awaiting_user_geometry')
                    state.pop('confirmed_at',None)
                    write(path,state)
        return {'source':read(folder/'source.json'),'draft':read(folder/'draft.json',{'polygon':[],'flow':[],'exclusions':[]})}

    def scene_base(self,workflow,identity):
        scene=workflow.scene(identity)
        path=self.base/'scene-preparation'/(identity+'.json')
        with self.lock:
            existing=read(path)
            if existing:
                return {**self.draft(existing['upload_id']),'preparation':existing}
            image=workflow.asset(identity,'dry')
            value=self.upload({'image':base64.b64encode(image.read_bytes()).decode('ascii'),
                               'name':scene['camera']+' · 场景底图'})
            value.update(role='dry_scene_base',scene_id=identity)
            write(self.location('upload',value['id'])/'source.json',value)
            state={'scene_id':identity,'camera':scene['camera'],'upload_id':value['id'],
                   'status':'awaiting_user_geometry','old_water_candidates_used':False,'created_at':time.time()}
            write(path,state)
            return {'source':value,'draft':{'polygon':[],'flow':[],'exclusions':[]},'preparation':state}

    def confirm_scene(self,data):
        identity=data.get('upload_id');folder=self.location('upload',identity)
        source=read(folder/'source.json')
        if source.get('role')!='dry_scene_base':raise ValueError('此图片不是待标记的场景底图。')
        polygon=points(data.get('polygon'));flow=points(data.get('flow'),2,80)
        if data.get('confirmed') is not True:raise ValueError('请确认范围和流向。')
        saved=self.draft(identity,{**data,'polygon':polygon,'flow':flow})
        path=self.base/'scene-preparation'/(source['scene_id']+'.json')
        state=read(path)
        state.update(status='geometry_confirmed_waiting_first_frame',confirmed_at=time.time(),geometry=saved['draft'])
        write(path,state)
        return state

    def start(self,data):
        if not self.environment()['ready']:raise ValueError('本机模型尚未准备，请按使用说明完成安装。')
        identity=data.get('upload_id');folder=self.location('upload',identity)
        polygon=points(data.get('polygon'));flow=points(data.get('flow'),2,80)
        exclusions=[points(p) for p in data.get('exclusions',[])]
        if len(exclusions)>30:raise ValueError('保护区最多30个。')
        source=read(folder/'source.json')
        if not source:raise ValueError('首帧不存在，请重新上传。')
        if source.get('role')=='dry_scene_base':raise ValueError('这只是场景底图，需要先制作并确认有水首帧，不能直接生成视频。')
        # Reject degenerate polygons and a flow whose complete path misses water.
        import numpy as np
        import cv2
        mask=np.zeros((256,256),np.uint8)
        cv2.fillPoly(mask,[np.rint(np.array(polygon)*255).astype(np.int32)],255)
        for p in exclusions:cv2.fillPoly(mask,[np.rint(np.array(p)*255).astype(np.int32)],0)
        if (mask>0).sum()<100:raise ValueError('可流动水面太小，或被保护区完全覆盖。')
        if any(mask[round(p[1]*255),round(p[0]*255)]==0 for p in flow):
            raise ValueError('流向点需要位于水面内，并避开固定物体保护区。')
        if sum(math.dist(a,b) for a,b in zip(flow,flow[1:]))<.015:raise ValueError('流向路线太短。')
        quality=data.get('quality','detail');size={'balanced':512,'detail':768}.get(quality)
        if not size:raise ValueError('画质规格无效。')
        speed=data.get('speed',1)
        if isinstance(speed,bool) or not isinstance(speed,(int,float)) or not math.isfinite(speed) or not .25<=speed<=2:
            raise ValueError('速度范围为0.25至2倍。')
        mode=data.get('mode','full')
        if mode not in ('preview','full'):raise ValueError('处理方式无效。')
        with self.lock:
            if self.active:raise ValueError('已有本地任务运行，请完成或停止后再开始。')
            job_id=secrets.token_hex(12);job=self.location('job',job_id);inp=job/'input';inp.mkdir(parents=True)
            try:os.link(folder/'wet-first-frame.png',inp/'wet-first-frame.png')
            except OSError:
                import shutil
                shutil.copy2(folder/'wet-first-frame.png',inp/'wet-first-frame.png')
            geometry={'polygon':polygon,'flow':flow,'exclusions':exclusions}
            write(inp/'user-region.json',{'draft':geometry});write(folder/'draft.json',geometry)
            output=job/'output'
            venv=self.root/'.venv/Scripts/python.exe'
            python=str(venv) if venv.is_file() else sys.executable
            command=[python,'-u',str(self.root/'experiments/tools/controlled_water_preview.py'),
                     '--input-dir',str(inp),'--output',str(output),'--water-crop','--water-only',
                     '--size',str(size),'--duration','12','--fps','12','--speed',str(speed*2),'--threads','4']
            # Default to the previously accepted learned-motion route. Native
            # detail transport is not enabled after the user's rock-drift report.
            if mode=='preview':command.append('--probe')
            value={'id':job_id,'upload_id':identity,'name':source['name'],'status':'running',
                   'mode':mode,'quality':quality,'speed':speed,'duration':12,'fps':12,'progress':0,
                   'created_at':time.time(),'message':'正在载入模型','training_ready':False,
                   'texture_mode':'learned_renderer_without_native_detail','command':command,
                   'source_sha256':source['working_sha256'],'geometry':geometry}
            write(job/'state.json',value);self.active=job_id
            threading.Thread(target=self._run,args=(job_id,command),daemon=True).start()
            return value

    def _run(self,identity,command):
        folder=self.location('job',identity)
        try:
            flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
            with self.lock:
                if read(folder/'state.json').get('status')=='cancelled':return
                self.process=subprocess.Popen(command,cwd=str(self.root),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                              text=True,encoding='utf-8',errors='replace',creationflags=flags)
                process=self.process
            with (folder/'render.log').open('w',encoding='utf-8') as log:
                for line in process.stdout:
                    log.write(line);log.flush()
                    match=re.search(r'frame (\d+)/(\d+)',line)
                    if match:
                        with self.lock:
                            value=read(folder/'state.json')
                            if value['status']=='running':
                                value.update(progress=round(int(match[1])/int(match[2])*100),message='正在渲染水面')
                                write(folder/'state.json',value)
                code=process.wait()
            with self.lock:
                value=read(folder/'state.json')
                if value['status']!='cancelled':
                    if code or not (folder/'output/result.json').is_file():
                        value.update(status='failed',message='处理失败，可下载运行记录查看原因。')
                    else:
                        result=read(folder/'output/result.json')
                        value.update(status='completed',progress=100,message='检查帧已生成' if value['mode']=='preview' else '12秒视频已生成',
                                     size=result.get('render_dimensions'),elapsed_seconds=result.get('elapsed_seconds'))
                        value['preview_frames']=[p.name for p in sorted((folder/'output').glob('probe-*.png'))]
                    write(folder/'state.json',value)
        except Exception as exc:
            with self.lock:
                value=read(folder/'state.json');value.update(status='failed',message=str(exc)[:200]);write(folder/'state.json',value)
        finally:
            with self.lock:self.active=None;self.process=None

    def cancel(self,identity):
        with self.lock:
            if self.active!=identity:raise ValueError('该任务当前未运行。')
            folder=self.location('job',identity);value=read(folder/'state.json')
            value.update(status='cancelled',message='已停止，未完成文件不作为结果。');write(folder/'state.json',value)
            if self.process:self.process.terminate()
            return value

    def asset(self,identity,name):
        folder=self.location('job',identity)
        if name=='record.json':return folder/'state.json'
        if name=='render.log':return folder/'render.log'
        if name not in ('candidate.mp4','inspection12.jpg','first-frame.png','motion-controls.jpg') and not re.fullmatch(r'probe-\d{4}\.png',name):
            raise ValueError('素材名称无效。')
        value=read(folder/'state.json')
        if value.get('status')!='completed':raise ValueError('任务尚未完成。')
        return folder/'output'/name
