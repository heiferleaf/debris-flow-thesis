"""Bounded delivery-frame replay measurement; no models or content judgments."""
from __future__ import annotations
from bisect import bisect_right
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import threading
import time

import cv2
import numpy as np

from experiments.common.video.pts_clip_sampling import (
    jpeg95_bytes, rgb112_from_jpeg, selection_expression, validate_pixel_std,
)

MAX_SECONDS = 3600.
MAX_BYTES = 1_000_000_000
AUDIT_RESERVE = 1_000_000
PHYSICAL = re.compile(r'\bn:\s*(\d+)\s+pts:\s*\S+\s+pts_time:\s*([-+0-9.eE]+)')
RAW_WARNING = re.compile(r'Application provided invalid, non monotonically increasing dts to muxer in stream \d+: -?\d+ >= -?\d+')


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def digest_file(path, budget=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 ** 2), b''):
            if budget: budget.check()
            h.update(chunk)
    return h.hexdigest()


def rooted(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or ':' in str(relative):
        raise ValueError('Expected project-relative input path')
    result = (Path(root) / p).resolve()
    result.relative_to(Path(root).resolve())
    return result


class LimitReached(RuntimeError):
    pass


class Budget:
    """All new writes reserve exact bytes before opening; watchdog kills decoder."""
    def __init__(self, out, seconds=MAX_SECONDS, byte_limit=MAX_BYTES, clock=time.monotonic):
        if not 0 < seconds <= MAX_SECONDS or not AUDIT_RESERVE < byte_limit <= MAX_BYTES:
            raise ValueError('Unregistered output/time cap')
        self.out, self.clock = Path(out).resolve(), clock
        self.started, self.seconds, self.byte_limit = clock(), seconds, byte_limit
        self.used = sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.expired = None
        self.process = None

    def check(self):
        if self.expired or self.clock() - self.started >= self.seconds:
            self.expired = self.expired or 'one_hour_wall_limit'
            raise LimitReached(self.expired)

    def write(self, path, raw, audit=False):
        path = Path(path).resolve(); path.relative_to(self.out)
        with self.lock:
            if not audit: self.check()
            ceiling = self.byte_limit if audit else self.byte_limit - AUDIT_RESERVE
            if self.used + len(raw) > ceiling:
                self.expired = 'one_GB_output_limit'
                raise LimitReached(self.expired)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as f: f.write(raw)
            self.used += len(raw)

    def json(self, path, value, audit=False):
        self.write(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'), audit)

    def watchdog(self):
        while not self.stop.wait(.05):
            if self.clock() - self.started >= self.seconds:
                self.expired = 'one_hour_wall_limit'
            if self.expired and self.process is not None and self.process.poll() is None:
                self.process.kill()


def index_at(times, elapsed):
    if (len(times) < 2 or times[0] != 0 or any(not math.isfinite(t) for t in times)
            or any(a >= b for a, b in zip(times, times[1:])) or not math.isfinite(elapsed)):
        raise ValueError('Expected a finite strictly increasing relative PTS timeline')
    return max(0, min(len(times) - 1, bisect_right(times, max(0., elapsed)) - 1))


def validate_range(indices, pts, anchor_indices, anchor_pts, clean_runs):
    indices = list(indices); pts = np.asarray(pts, dtype=np.float64)
    if (len(anchor_indices) != 16 or len(set(anchor_indices)) != 16
            or any(type(i) is not int for i in anchor_indices) or len(anchor_pts) != 16
            or any(a >= b for a, b in zip(anchor_indices, anchor_indices[1:]))
            or indices != list(range(anchor_indices[0], anchor_indices[-1] + 1))
            or not indices or len(indices) != len(pts) or not np.isfinite(pts).all()
            or np.any(np.diff(pts) <= 0)
            or not any(indices[0] >= a and indices[-1] < b for a, b in clean_runs)):
        raise ValueError('Full delivery index/PTS/clean-range closure failed')
    positions = [i - indices[0] for i in anchor_indices]
    if not np.allclose(pts[positions], anchor_pts, rtol=0., atol=1e-8):
        raise ValueError('Original sixteen anchors differ from coordinate range')
    span = float(pts[-1] - pts[0])
    if not 2.9 <= span <= 3.1:
        raise ValueError('Unexpected actual endpoint duration')
    return positions, span


class PhysicalAudit:
    def __init__(self, pts):
        self.pts, self.decoded, self.maximum_error = np.asarray(pts), 0, 0.
        self.errors, self.selected_pts, self.raw_warnings = deque(maxlen=20), {}, 0

    def observe(self, raw, selected):
        self.raw_warnings += len(RAW_WARNING.findall(raw))
        line = RAW_WARNING.sub('', raw)
        match = PHYSICAL.search(line)
        if match:
            i, t = int(match[1]), float(match[2])
            if i != self.decoded or i >= len(self.pts) or 's:256x256' not in line or not math.isfinite(t):
                self.errors.append('Physical frame index/dimension/finite-PTS mismatch')
            else:
                error = abs(t - self.pts[i]); self.maximum_error = max(error, self.maximum_error)
                if error > .001001: self.errors.append('Physical/coordinate PTS mismatch')
                if i in selected: self.selected_pts[i] = t
            self.decoded += 1
        if 'error' in line.lower() or 'invalid' in line.lower() or 'corrupt' in line.lower():
            self.errors.append(line.strip()[:500])

    def finish(self, selected):
        if self.errors or self.decoded != len(self.pts) or set(self.selected_pts) != set(selected):
            raise ValueError('Physical replay incomplete: %s' % list(self.errors))


def processed_frame(bgr, qc_std, anchor=None):
    if bgr.shape != (256, 256, 3) or bgr.dtype != np.uint8:
        raise ValueError('Expected delivered uint8 BGR256 frame')
    actual_std = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).std())
    validate_pixel_std(actual_std, float(qc_std), 20.)
    jpeg = jpeg95_bytes(bgr)
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != (256, 256, 3):
        raise ValueError('JPEG95 decode failed')
    rgb256 = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    rgb112 = rgb112_from_jpeg(jpeg)
    linear = cv2.resize(rgb256, (112, 112), interpolation=cv2.INTER_LINEAR)
    if not np.array_equal(rgb112, linear):
        raise ValueError('Frozen helper no longer reproduces LINEAR112')
    proof = dict(jpeg_sha256=digest_bytes(jpeg), source_bgr_sha256=digest_bytes(bgr.tobytes()),
                 rgb256_decoded_sha256=digest_bytes(rgb256.tobytes()), rgb112_sha256=digest_bytes(rgb112.tobytes()),
                 gray_std=actual_std, frozen_qc_std=float(qc_std), anchor=anchor is not None)
    if anchor is not None:
        for key in ('jpeg_sha256', 'source_bgr_sha256', 'rgb256_decoded_sha256', 'rgb112_sha256'):
            if proof[key] != anchor[key]:
                raise ValueError('Frozen anchor replay mismatch: ' + key)
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(rgb256, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok: raise RuntimeError('PNG encode failed')
    restored = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if not np.array_equal(restored, rgb256): raise ValueError('PNG RGB256 roundtrip failed')
    return encoded.tobytes(), proof


def extract_case(root, case, ffmpeg, version, budget, private_dir):
    """Exact old sequential physical-counter route plus bounded new PNG outputs.

    We reuse frozen JPEG95/LINEAR112/QC helpers, but own the bounded streaming
    wrapper. showinfo precedes select, and all middleware frames are audited.
    No seek, fps conversion, source copy, frame interpolation or normalization.
    """
    pts, qc = [], []
    import csv
    with rooted(root, case['coord_path']).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f)); pts = [float(r['pts_time']) for r in rows]
    with rooted(root, case['qc_path']).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f)); qc = [float(r['gray_std']) for r in rows]
    required = case['delivery_frame_indices']; required_set = set(required)
    anchors = {r['frame_index']: r for r in case['anchors']}
    evidence = private_dir / case['case_id']
    selected_expression = selection_expression(required)
    script = evidence / 'select-filter.txt'
    budget.write(script, ("showinfo,select='%s',format=bgr24" % selected_expression).encode('utf-8'))
    major = int(re.search(r'ffmpeg version (?:n)?(\d+)', version).group(1))
    command = [str(ffmpeg), '-hide_banner', '-nostdin', '-copyts', '-threads', '1', '-i', str(rooted(root, case['middleware_path'])),
               '-/filter:v' if major >= 8 else '-filter_script:v', str(script), '-an']
    command += ['-fps_mode', 'passthrough'] if major >= 5 else ['-vsync', '0']
    command += ['-f', 'rawvideo', 'pipe:1']
    budget.json(evidence / 'command.json', command)
    audit, records, diagnostics = PhysicalAudit(pts), [], deque(maxlen=30)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1024 ** 2)
    budget.process = process
    def drain():
        try:
            for line in iter(process.stderr.readline, b''):
                text = line.decode('utf-8', errors='replace')
                audit.observe(text, required_set)
                if not PHYSICAL.search(text): diagnostics.append(text.strip()[:1000])
        except Exception as error: audit.errors.append('Diagnostic reader: ' + str(error))
    reader = threading.Thread(target=drain, daemon=True); reader.start()
    error = None
    try:
        for position, index in enumerate(required):
            budget.check(); raw = bytearray()
            while len(raw) < 256 * 256 * 3:
                chunk = process.stdout.read(256 * 256 * 3 - len(raw))
                if not chunk: raise ValueError('Selected raw pixel stream ended early')
                raw.extend(chunk); budget.check()
            bgr = np.frombuffer(raw, np.uint8).reshape(256, 256, 3)
            encoded, proof = processed_frame(bgr, qc[index], anchors.get(index))
            relative = 'images/%s/F%05d.png' % (case['case_id'], position)
            # Assets stay PRIVATE until the complete case (all frames+16 anchors)
            # passes. Failure never publishes a shortened/stitched observation.
            target = private_dir / 'assets' / relative
            budget.write(target, encoded)
            if digest_file(target, budget) != digest_bytes(encoded): raise ValueError('PNG disk SHA mismatch')
            restored = cv2.cvtColor(cv2.imread(str(target), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            if digest_bytes(restored.tobytes()) != proof['rgb256_decoded_sha256']:
                raise ValueError('PNG disk pixels mismatch')
            records.append(dict(frame_index=index, coordinate_pts_seconds=float(pts[index]),
                relative_pts_seconds=float(pts[index] - pts[required[0]]), path=relative,
                png_sha256=digest_bytes(encoded), size_bytes=len(encoded), **proof))
        if process.stdout.read(1): raise ValueError('Extra selected pixel bytes')
        process.wait(); reader.join(); budget.check()
        if process.returncode: raise RuntimeError('Decoder exited %d' % process.returncode)
        audit.finish(required)
        for row in records: row['physical_pts_seconds'] = audit.selected_pts[row['frame_index']]
        if len(records) != len(required) or sum(r['anchor'] for r in records) != 16:
            raise ValueError('Delivery or anchor count mismatch')
        return dict(status='passed', case_id=case['case_id'], frames=records,
                    verified_anchor_count=16, actual_span_seconds=case['actual_span_seconds'],
                    physical_frames_audited=audit.decoded, max_physical_coord_error=audit.maximum_error,
                    raw_muxer_dts_warnings=audit.raw_warnings, decoder_diagnostics=list(diagnostics))
    except Exception as caught:
        error = caught
        raise
    finally:
        if process.poll() is None: process.kill()
        process.wait(); reader.join(timeout=5); budget.process = None
        if reader.is_alive(): raise RuntimeError('Decoder diagnostic thread did not terminate')
        process.stdout.close(); process.stderr.close()
        if error is not None:
            budget.json(evidence / 'failure.json', dict(status='whole_window_missing', case_id=case['case_id'],
                error_type=type(error).__name__, error=str(error), completed_delivery_frames=len(records),
                completed_anchor_matches=sum(r['anchor'] for r in records),
                physical_frames_audited=audit.decoded, diagnostics=list(audit.errors),
                partial_assets_private=True, no_frame_deletion_or_stitching=True), audit=True)


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>时序输入观察</title>
<style>*{box-sizing:border-box}body{font:16px/1.5 system-ui;margin:0;background:#f2f4f6;color:#18232c}
main{max-width:940px;margin:24px auto;padding:0 20px}.card{background:white;padding:20px;border-radius:12px;border:1px solid #d9e0e6}
.bar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:14px 0}button,select,input{font:inherit}button,select{padding:7px 12px}
.panels{display:flex;gap:28px;flex-wrap:wrap}.panel{width:280px}canvas{width:256px;height:256px;background:#111;image-rendering:pixelated;filter:none;transform:none}
.note{font-size:14px;color:#536271}h1{font-size:26px}h2{font-size:18px}[hidden]{display:none!important}#error{color:#a00016;white-space:pre-wrap}input[type=range]{width:380px}</style>
<main><h1>时序输入观察</h1><p>每个条件先从首帧独立播放至共同末端；完成两次独立播放后才可并排。</p>
<p class="note">本工具比较同一已验256中间件的稀疏采样与范围内交付帧，不代表原片native或全部采集帧。
条件名称匿名，原生机位/日期叠字保留；帧数和运动密度可能透露条件，不能称完全条件盲化。</p>
<div class="card"><div class="bar"><label>窗口 <select id="case"></select></label><button id="k1">条件 K1</button><button id="k2">条件 K2</button><button id="compare" disabled>并排</button></div>
<p id="modeNote" class="note"></p><div class="panels"><section class="panel"><h2 id="title1"></h2><canvas id="canvas1" width="256" height="256"></canvas></section>
<section class="panel" id="panel2" hidden><h2 id="title2"></h2><canvas id="canvas2" width="256" height="256"></canvas></section></div>
<div class="bar"><button id="play" disabled>播放</button><button id="pause" disabled>暂停</button><button id="prev" disabled>上一帧</button><button id="next" disabled>下一帧</button>
<input id="time" type="range" min="0" max="3" step="0.001" value="0" disabled></div><p id="status"></p><pre id="error"></pre>
<p class="note">各条件按自身真实相对PTS保持到下一帧，共同末端停止；无声、无插帧、无改速、无增强、不自动循环。
两个256×256像素框均禁用平滑；浏览器刷新可能延迟显示，未呈现的刷新间帧不当作已经观察。</p></div></main>
<script src="manifest.js"></script><script>'use strict';
const data=window.TEMPORAL_REVIEW,$=id=>document.getElementById(id),ctx=[$('canvas1').getContext('2d'),$('canvas2').getContext('2d')];
let clip,images={},state={},raf=0,origin=0,generation=0;
function at(times,t){let i=0;while(i+1<times.length&&times[i+1]<=t)i++;return i;}
function stop(){if(raf)cancelAnimationFrame(raf);raf=0;state.playing=false;}
function conditionData(alias){return clip.conditions.find(c=>c.condition_id===alias);}
function draw(){const aliases=state.mode==='compare'?['K1','K2']:[state.condition];$('panel2').hidden=aliases.length===1;
aliases.forEach((alias,j)=>{const c=conditionData(alias),i=at(c.relative_pts_seconds,state.elapsed),im=images[alias][i];
ctx[j].imageSmoothingEnabled=false;ctx[j].clearRect(0,0,256,256);ctx[j].drawImage(im,0,0,256,256);$('title'+(j+1)).textContent=alias+' · '+(i+1)+'/'+c.frames.length;});
if(state.playing&&state.eligible&&state.mode==='single'){
const i=at(conditionData(state.condition).relative_pts_seconds,state.elapsed),seen=state.displayed[state.condition];
if(!seen.includes(i))seen.push(i);}
$('time').value=state.elapsed;$('status').textContent=clip.case_id+' · +'+state.elapsed.toFixed(3)+'秒 / '+clip.duration_seconds.toFixed(3)+'秒'+(state.elapsed>=clip.duration_seconds?' · 末端停止':'');
$('compare').disabled=state.completed.length<2;$('modeNote').textContent=state.mode==='compare'?'共同实际时间轴，各条件独立左保持。':'已完整独立播放：'+(state.completed.join('、')||'尚无');}
function tick(now){if(!state.playing)return;state.elapsed=Math.min((now-origin)/1000,clip.duration_seconds);draw();
state.trace.push({elapsed:state.elapsed,wall_ms:now,frames:clip.conditions.map(c=>at(c.relative_pts_seconds,state.elapsed))});
if(state.elapsed>=clip.duration_seconds){const c=conditionData(state.condition),seen=state.displayed[state.condition];
if(state.mode==='single'&&state.eligible&&c.frames.every((_,i)=>seen.includes(i))&&!state.completed.includes(state.condition))state.completed.push(state.condition);
stop();draw();return;}raf=requestAnimationFrame(tick);}
function play(){if(!state.ready||state.playing)return;if(state.elapsed>=clip.duration_seconds)state.elapsed=0;
if(state.elapsed===0){state.eligible=state.mode==='single';state.trace=[];if(state.eligible)state.displayed[state.condition]=[];}
origin=performance.now()-1000*state.elapsed;state.playing=true;draw();raf=requestAnimationFrame(tick);}
function pause(){if(state.playing)state.elapsed=Math.min((performance.now()-origin)/1000,clip.duration_seconds);stop();if(state.ready)draw();}
function seek(t){if(!state.ready)return;stop();state.eligible=false;state.elapsed=Math.max(0,Math.min(t,clip.duration_seconds));draw();}
function condition(alias){if(!state.ready)return;pause();state.mode='single';state.condition=alias;state.elapsed=0;state.eligible=false;draw();}
function compare(){if(!state.ready||state.completed.length<2)return;pause();state.mode='compare';state.elapsed=0;state.eligible=false;draw();}
function step(d){const c=state.mode==='compare'?clip.conditions.reduce((a,b)=>a.frames.length>b.frames.length?a:b):conditionData(state.condition);seek(c.relative_pts_seconds[Math.max(0,Math.min(c.frames.length-1,at(c.relative_pts_seconds,state.elapsed)+d))]);}
async function load(id){stop();const token=++generation;state={ready:false,playing:false};$('error').textContent='';$('compare').disabled=true;
['play','pause','prev','next','time','k1','k2'].forEach(k=>$(k).disabled=true);clip=data.cases.find(c=>c.case_id===id);if(!clip)throw Error('Unknown case');
const loaded={};await Promise.all(clip.conditions.map(async c=>{const t=c.relative_pts_seconds;
if(t.length!==c.frames.length||t.length<2||t[0]!==0||t[t.length-1]!==clip.duration_seconds||t.some((x,i)=>!Number.isFinite(x)||(i&&x<=t[i-1])))throw Error('Invalid timeline');
loaded[c.condition_id]=await Promise.all(c.frames.map(f=>new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>im.naturalWidth===256&&im.naturalHeight===256?resolve(im):reject(Error('Wrong pixel dimensions'));im.onerror=()=>reject(Error('Image load failed'));im.src=f.path;})));}));
if(token!==generation)return;images=loaded;state={ready:true,playing:false,elapsed:0,mode:'single',condition:'K1',completed:[],eligible:false,trace:[],displayed:{K1:[],K2:[]}};$('time').max=clip.duration_seconds;
['play','pause','prev','next','time','k1','k2'].forEach(k=>$(k).disabled=false);draw();}
function fail(e){stop();state.ready=false;$('error').textContent=String(e);}
data.cases.forEach(c=>{const o=document.createElement('option');o.value=c.case_id;o.textContent=c.case_id;$('case').append(o);});
$('case').onchange=()=>load($('case').value).catch(fail);$('play').onclick=play;$('pause').onclick=pause;$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);
$('time').oninput=()=>seek(Number($('time').value));$('k1').onclick=()=>condition('K1');$('k2').onclick=()=>condition('K2');$('compare').onclick=compare;
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
window.temporalPlayer={at,play,pause,seek,condition,compare,load,snapshot:()=>({case_id:clip.case_id,...state,canvas_raster:[256,256],canvas_css:[$('canvas1').getBoundingClientRect().width,$('canvas1').getBoundingClientRect().height],DPR:devicePixelRatio,image_smoothing:ctx.map(c=>c.imageSmoothingEnabled)})};
if(data.cases.length)load(data.cases[0].case_id).catch(fail);else $('status').textContent='无完整通过窗口；缺失原因见私有证据。';
</script></html>'''
