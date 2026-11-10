# -*- coding: utf-8 -*-
"""Prepare a score/selection-role-blinded RGB112 <-> decoded RGB256 bridge.

Reuses exactly 12 frozen clips/192 verified JPEGs. No torch, decoder, model,
training, label changes or visibility adjudication. Native conditions remain pending.
"""
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import sys

import cv2
import numpy as np

SOURCE = Path(__file__).resolve()
ROOT = SOURCE.parents[3]
OUT = ROOT / 'data/process/EXP-018/input-observability-v1'
PRIOR = ROOT / 'data/process/EXP-018/drift-blind-review-v1/mapping.json'
PRIOR_SHA = 'cff9b57c14af8796488963428a226635019c554c2692618d1fe967588cdac7fb'
SEED = 180112256


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def index_at(times, elapsed):
    """Left-held sampled frame; the last anchor stops playback, never loops."""
    if not len(times) == 16 or times[0] != 0 or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError('Expected sixteen increasing relative times beginning at zero')
    if not math.isfinite(elapsed):
        raise ValueError('Nonfinite playback time')
    return max(i for i, t in enumerate(times) if t <= max(0, elapsed))


def png_bytes(rgb):
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError('PNG encoding failed')
    restored = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if not np.array_equal(restored, rgb):
        raise ValueError('Lossless PNG memory roundtrip differs')
    return encoded.tobytes()


HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>输入观察工具</title>
<style>
*{box-sizing:border-box}body{font:16px/1.5 system-ui,sans-serif;background:#f2f4f6;color:#17212c;margin:0}
main{max-width:900px;margin:24px auto;padding:0 20px}h1{font-size:25px;margin:0 0 8px}
.card{background:white;border:1px solid #d8dfe6;border-radius:12px;padding:18px;margin:16px 0}
.bar{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin:12px 0}
button,select,input{font:inherit}button,select{padding:7px 12px;border:1px solid #a8b5c1;border-radius:6px;background:white}
button{cursor:pointer}button:disabled{cursor:default;opacity:.45}button.active{background:#183e61;color:white}
#panels{display:flex;gap:24px;flex-wrap:wrap}.panel{width:280px}h2{font-size:18px;margin:4px 0 12px}
.viewport{width:256px;height:256px;background:#111;overflow:hidden}
canvas{display:block;width:256px;height:256px;image-rendering:pixelated;filter:none;transform:none}
.note{color:#536271;font-size:14px}#status{min-height:25px}input[type=range]{width:min(440px,100%)}
[hidden]{display:none!important}#error{color:#a00016;white-space:pre-wrap}
</style>
<main><h1>输入观察工具</h1>
<p>先分别打开两个条件，再切换并排。编号条件的顺序已固定随机化。</p>
<p class="note">条件名称匿名；原生机位和日期叠字保留，不是身份或完全条件盲化。
这是相同的16个稀疏时刻，不能代表完整原片运动或native条件。</p>
<section class="card"><div class="bar"><label>窗口 <select id="case"></select></label>
<button id="condition1">条件 K1</button><button id="condition2">条件 K2</button>
<button id="compare" disabled>并排查看</button><button id="single">单条件</button></div>
<p id="modeNote" class="note"></p><div id="panels">
<section class="panel" id="panel1"><h2 id="title1"></h2><div class="viewport"><canvas id="canvas1" width="256" height="256"></canvas></div></section>
<section class="panel" id="panel2" hidden><h2 id="title2"></h2><div class="viewport"><canvas id="canvas2" width="256" height="256"></canvas></div></section>
</div><div class="bar"><button id="play" disabled>播放</button><button id="pause" disabled>暂停</button>
<button id="previous" disabled>上一帧</button><button id="next" disabled>下一帧</button>
<label>帧 <input id="frame" type="range" min="0" max="15" step="1" value="0" disabled></label></div>
<p id="status" aria-live="polite">正在载入</p><pre id="error"></pre>
<p class="note">按相邻真实PTS间隔播放，到末帧停止、不自动循环。可暂停和逐帧。
两条件均显示为256×256像素框，使用最近邻显示；不平滑、不锐化、不增强。
保存的PNG仍保留各自原有像素尺寸，显示放大不增加细节。</p></section>
</main><script src="manifest.js"></script><script>
'use strict';
const data=window.REVIEW_DATA,$=id=>document.getElementById(id);
let clip=null,images={},generation=0,raf=0,origin=0;
let state={ready:false,playing:false,frame:0,elapsed:0,mode:'single',condition:'K1',visited:[],trace:[]};
const contexts=[$('canvas1').getContext('2d'),$('canvas2').getContext('2d')];
contexts.forEach(c=>{c.imageSmoothingEnabled=false;});
function frameAt(times,t){let i=0;while(i+1<times.length&&times[i+1]<=t)i++;return i;}
function stop(){if(raf)cancelAnimationFrame(raf);raf=0;state.playing=false;}
function display(index){
  state.frame=index;$('frame').value=index;
  const aliases=state.mode==='compare'?['K1','K2']:[state.condition];
  $('panel2').hidden=state.mode!=='compare';
  aliases.forEach((alias,j)=>{
    const c=contexts[j],im=images[alias][index];c.imageSmoothingEnabled=false;
    c.clearRect(0,0,256,256);c.drawImage(im,0,0,im.naturalWidth,im.naturalHeight,0,0,256,256);
    $('title'+(j+1)).textContent=alias;
  });
  $('status').textContent=`${clip.case_id} · ${index+1}/16 · +${clip.relative_pts_seconds[index].toFixed(3)}秒`+
    (index===15&&!state.playing?' · 已到末帧，停止':'');
  $('condition1').classList.toggle('active',state.mode==='single'&&state.condition==='K1');
  $('condition2').classList.toggle('active',state.mode==='single'&&state.condition==='K2');
  $('compare').disabled=state.visited.length<2;
  $('modeNote').textContent=state.mode==='compare'?'并排同步：相同帧号、相同相对时间。':
    (state.visited.length<2?'单条件查看；打开另一条件后可并排。':'已分别打开两个条件，可以并排。');
}
function record(index,wall){state.trace.push({frame:index,relative_pts_seconds:clip.relative_pts_seconds[index],wall_ms:wall});}
function tick(now){
  if(!state.playing)return;
  state.elapsed=Math.min((now-origin)/1000,clip.relative_pts_seconds[15]);
  const i=frameAt(clip.relative_pts_seconds,state.elapsed);
  if(i!==state.frame){record(i,now);display(i);}
  if(state.elapsed>=clip.relative_pts_seconds[15]){stop();display(15);return;}
  raf=requestAnimationFrame(tick);
}
function play(){
  if(!state.ready||state.playing)return;
  if(state.frame===15){state.elapsed=0;display(0);}
  state.trace=[];origin=performance.now()-state.elapsed*1000;state.playing=true;
  record(state.frame,performance.now());raf=requestAnimationFrame(tick);
}
function pause(){
  if(state.playing)state.elapsed=Math.min((performance.now()-origin)/1000,clip.relative_pts_seconds[15]);
  stop();if(state.ready)display(frameAt(clip.relative_pts_seconds,state.elapsed));
}
function seek(index){if(!state.ready)return;stop();state.elapsed=clip.relative_pts_seconds[index];display(index);}
function condition(alias){
  if(!state.ready)return;pause();state.mode='single';state.condition=alias;
  if(!state.visited.includes(alias))state.visited.push(alias);display(state.frame);
}
function compare(){if(!state.ready||state.visited.length<2)return;pause();state.mode='compare';display(state.frame);}
async function load(id){
  stop();const token=++generation;state.ready=false;$('error').textContent='';
  ['play','pause','previous','next','frame','condition1','condition2','single'].forEach(k=>$(k).disabled=true);
  $('compare').disabled=true;$('status').textContent='正在载入';
  clip=data.cases.find(c=>c.case_id===id);if(!clip)throw Error('Unknown window');
  if(clip.relative_pts_seconds.length!==16||clip.relative_pts_seconds[0]!==0||
     clip.relative_pts_seconds.some((x,i,a)=>!Number.isFinite(x)||(i&&x<=a[i-1])))throw Error('Invalid PTS');
  const loaded={};
  await Promise.all(clip.conditions.map(async item=>{
    loaded[item.condition_id]=await Promise.all(item.frames.map(row=>new Promise((resolve,reject)=>{
      const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(Error('Image load failed'));image.src=row.path;
    })));
  }));
  if(token!==generation)return;
  images=loaded;state={ready:true,playing:false,frame:0,elapsed:0,mode:'single',condition:'K1',visited:['K1'],trace:[]};
  ['play','pause','previous','next','frame','condition1','condition2','single'].forEach(k=>$(k).disabled=false);display(0);
}
function fail(error){stop();state.ready=false;$('error').textContent=String(error);$('status').textContent='载入失败，请核对本地文件';}
data.cases.forEach(c=>{let option=document.createElement('option');option.value=c.case_id;option.textContent=c.case_id;$('case').append(option);});
$('case').onchange=()=>load($('case').value).catch(fail);$('play').onclick=play;$('pause').onclick=pause;
$('previous').onclick=()=>seek(Math.max(0,state.frame-1));$('next').onclick=()=>seek(Math.min(15,state.frame+1));
$('frame').oninput=()=>seek(Number($('frame').value));$('condition1').onclick=()=>condition('K1');
$('condition2').onclick=()=>condition('K2');$('compare').onclick=compare;$('single').onclick=()=>condition(state.condition);
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
window.reviewPlayer={play,pause,seek,condition,compare,load,frameAt,snapshot:()=>({case_id:clip.case_id,...state,
  device_pixel_ratio:window.devicePixelRatio,canvas_raster:[256,256],canvas_css:[$('canvas1').getBoundingClientRect().width,$('canvas1').getBoundingClientRect().height],
  image_smoothing:contexts.map(c=>c.imageSmoothingEnabled),condition_native_sizes:clip.conditions.map(c=>({condition_id:c.condition_id,width:images[c.condition_id]?.[0]?.naturalWidth,height:images[c.condition_id]?.[0]?.naturalHeight}))})};
load(data.cases[0].case_id).catch(fail);
</script></html>'''


README = '''# Input observability bridge v1

This local bridge covers only the already fixed12 windows / same16 real PTS per
window: actual unnormalized-equivalent RGB112 versus JPEG-decoded RGB256.
It does not adjudicate visibility or truth. Neither native-resolution condition
nor the full actual-frame/native-motion condition has been completed.

Give an independent reviewer only viewer/. Open viewer/index.html directly in a
browser; no network, server, model or torch is needed. First view K1 and K2
separately; then the parallel button becomes available. Play, pause and frame
steps share the same timeline. Original site/date OSD is kept. Scores, labels,
roles, source identities and the condition-code mapping are outside viewer/.
Condition names/order are anonymized; visible resolution may identify conditions,
so this is prediction/selection-role blinding, not full condition/identity blinding.

PNG files preserve exact112x112 or256x256 RGB pixels. The browser uses a256x256
canvas and fixed256x256 CSS square for both, with smoothing disabled and nearest
neighbour display. The112 display is therefore an explicit enlargement, not an
extra sharpening step or newly recovered detail. Device/browser zoom may change
physical screen geometry; inspect reviewPlayer.snapshot() for DPR and CSS size.
No CSS filters, sharpening, colour enhancement or synthesized frames are used.

Playback holds each selected frame until the next real relative PTS, uses the
first and final selected PTS as anchors, and stops at the final frame without
looping. Remaining display of the last still frame is not extra source duration
or a new event. Browser refresh may delay a presentation; source intervals are
never replaced by16/3fps. This is sparse-frame motion, not full native video.

Actual frozen helper is BGR JPEG decode -> RGB -> cv2.INTER_LINEAR112. The initial
AREA wording was contradicted by the code fingerprint and pixel hashes: all192
LINEAR frames match verified rgb112_sha256, while0 AREA frames match. No training
input/code changed. The RGB112 PNG is the uint8 view before mean/std normalization,
equivalent to undoing the fixed input normalization; no normalized tensor/model
is evaluated. RGB256 is the decoded JPEG cache, not original-resolution pixels.

Run from repository root:
.venv/Scripts/python.exe -B experiments/runspecs/EXP-018/prepare_observability.py

Existing output directories are refused. mapping.json is private and records all
original PTS, frame indices, source JPEG SHA, decoded-pixel SHA, actual112 SHA,
anonymous condition mapping, image SHA and input/source fingerprints. The public
manifest is a strict whitelist without predictions, labels, groups or roles.
Every PNG is read back and byte-for-byte RGB verified. No video/cloud/training
operation and no changes to raw, existing JPEG, frozen plan or labels occurred.
'''


def main():
    if OUT.exists():
        raise FileExistsError('Refuse existing observability outputs')
    inputs = {}
    def read(path, expected=None):
        raw = path.read_bytes()
        fingerprint = digest(raw)
        if expected is not None and fingerprint != expected:
            raise ValueError('Input SHA mismatch: ' + str(path))
        inputs[path.relative_to(ROOT).as_posix()] = dict(sha256=fingerprint, size_bytes=len(raw))
        return raw
    read(SOURCE)
    prior = json.loads(read(PRIOR, PRIOR_SHA).decode('utf-8'))
    cases = sorted(prior['cases'], key=lambda c: c['anonymous_id'])
    if ([c['anonymous_id'] for c in cases] != ['C%02d' % i for i in range(1, 13)]
            or len({c['clip_id'] for c in cases}) != 12 or len({c['unit_id'] for c in cases}) != 12):
        raise ValueError('Expected the frozen twelve unique windows/videos')
    prior_inputs = {r['path']: r['sha256'] for r in prior['input_fingerprints']}
    cache_path = ROOT / 'data/process/EXP-016/metadata-v4/verified-cache-files.csv'
    cache_rows = csv.DictReader(io.StringIO(read(cache_path, prior_inputs[cache_path.relative_to(ROOT).as_posix()]).decode('utf-8-sig')))
    wanted = {(c['unit_id'], int(i)) for c in cases for i in c['window']['frame_indices']}
    refs = {}
    for row in cache_rows:
        key = (row['unit_id'], int(row['frame_index']))
        if key in wanted:
            if key in refs:
                raise ValueError('Duplicate cache frame')
            refs[key] = row
    if len(wanted) != 192 or set(refs) != wanted:
        raise ValueError('Missing/distinct-frame mismatch in verified index')
    profile_path = ROOT / 'data/results/exp018/paired-s7-20261101-v1/bce/profile.json'
    profile = json.loads(read(profile_path).decode('utf-8'))
    if profile['objective_mode'] != 'bce' or profile['smoke']:
        raise ValueError('Not actual ordinary BCE profile')
    helper_path = ROOT / 'experiments/common/video/pts_clip_sampling.py'
    helper_sha = profile['code_sha256'][helper_path.relative_to(ROOT).as_posix()]
    read(helper_path, helper_sha)
    spec = importlib.util.spec_from_file_location('observability_frozen_pixels', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if 'torch' in sys.modules:
        raise RuntimeError('Unexpected torch import')
    flips = [False] * 6 + [True] * 6
    random.Random(SEED).shuffle(flips)
    prepared, area_matches, linear_matches = [], 0, 0
    for c, flip in zip(cases, flips):
        w = c['window']
        indices, pts = w['frame_indices'], w['source_pts_seconds']
        if (len(indices) != 16 or len(set(indices)) != 16 or len(pts) != 16
                or not all(type(i) is int for i in indices) or not all(math.isfinite(t) for t in pts)
                or any(a >= b for a, b in zip(indices, indices[1:]))
                or any(a >= b for a, b in zip(pts, pts[1:]))
                or not 2.9 <= pts[-1] - pts[0] <= 3.1):
            raise ValueError('Frozen clip frame/PTS contract failed')
        rel = [t - pts[0] for t in pts]
        for i, t in enumerate(rel):
            if index_at(rel, t) != i:
                raise ValueError('Playback anchor fixture failed')
            if i < 15 and index_at(rel, (t + rel[i + 1]) / 2) != i:
                raise ValueError('Playback interval fixture failed')
        names = ['rgb112_actual', 'rgb256_jpeg_decoded']
        if flip:
            names.reverse()
        frames = []
        for j, (i, t) in enumerate(zip(indices, pts)):
            ref = refs[(c['unit_id'], i)]
            old = c['frames'][j]
            for k in ('cache_path', 'jpeg_sha256', 'rgb112_sha256'):
                if ref[k] != old[k]:
                    raise ValueError('Drift mapping/cache index differs')
            if abs(float(ref['source_pts_seconds']) - t) > 1e-8:
                raise ValueError('Cache PTS differs from frozen window')
            raw = read(ROOT / ref['cache_path'], ref['jpeg_sha256'])
            if len(raw) != int(ref['size_bytes']):
                raise ValueError('JPEG size mismatch')
            bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape != (256, 256, 3):
                raise ValueError('Invalid decoded JPEG')
            rgb256 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb112 = helper.rgb112_from_jpeg(raw)
            linear = cv2.resize(rgb256, (112, 112), interpolation=cv2.INTER_LINEAR)
            area = cv2.resize(rgb256, (112, 112), interpolation=cv2.INTER_AREA)
            actual_sha = digest(rgb112.tobytes())
            if actual_sha != ref['rgb112_sha256'] or not np.array_equal(rgb112, linear):
                raise ValueError('Actual112 pixels do not match verified training contract')
            linear_matches += 1
            area_matches += digest(area.tobytes()) == ref['rgb112_sha256']
            frames.append(dict(position=j + 1, source_frame_index=i, source_pts_seconds=t,
                relative_pts_seconds=rel[j], cache_path=ref['cache_path'], jpeg_sha256=ref['jpeg_sha256'],
                rgb256_decoded_sha256=digest(rgb256.tobytes()), rgb112_sha256=actual_sha,
                area_candidate_sha256=digest(area.tobytes()),
                arrays={'rgb112_actual': rgb112, 'rgb256_jpeg_decoded': rgb256}))
        prepared.append(dict(case_id=c['anonymous_id'], clip_id=c['clip_id'], unit_id=c['unit_id'],
            group_id=c['group_id'], relative_pts_seconds=rel, condition_order=names, frames=frames))
    if linear_matches != 192 or area_matches != 0:
        raise ValueError('Unexpected interpolation audit; stop rather than changing inputs')
    OUT.mkdir(parents=True, exist_ok=False)
    viewer = OUT / 'viewer'
    viewer.mkdir()
    public_cases, private_cases, images = [], [], []
    for c in prepared:
        public = dict(case_id=c['case_id'], relative_pts_seconds=c['relative_pts_seconds'], conditions=[])
        private = {k: v for k, v in c.items() if k != 'frames'}
        private['frames'] = [{k: v for k, v in f.items() if k != 'arrays'} for f in c['frames']]
        for alias, actual in zip(('K1', 'K2'), c['condition_order']):
            directory = viewer / 'images' / c['case_id'] / alias
            directory.mkdir(parents=True)
            condition = dict(condition_id=alias, frames=[])
            for f in c['frames']:
                rgb = f['arrays'][actual]
                target = directory / ('F%02d.png' % f['position'])
                encoded = png_bytes(rgb)
                with target.open('xb') as handle:
                    handle.write(encoded)
                stored = target.read_bytes()
                restored = cv2.cvtColor(cv2.imdecode(np.frombuffer(stored, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                if digest(stored) != digest(encoded) or not np.array_equal(restored, rgb):
                    raise ValueError('PNG disk readback differs')
                item = dict(path=target.relative_to(viewer).as_posix(), sha256=digest(stored),
                            rgb_sha256=digest(restored.tobytes()), width=rgb.shape[1], height=rgb.shape[0], size_bytes=len(stored))
                condition['frames'].append({k: item[k] for k in ('path', 'sha256')})
                images.append(dict(case_id=c['case_id'], condition_id=alias, actual_condition=actual,
                                   position=f['position'], **item))
            public['conditions'].append(condition)
        public_cases.append(public)
        private_cases.append(private)
    public_manifest = dict(review_id='input-observability-v1', case_count=12, frames_per_case=16,
        condition_ids=['K1', 'K2'], condition_names_anonymized=True, fully_condition_blinded=False,
        prediction_and_selection_role_blinded=True, identity_blind=False, native_OSD_retained=True,
        display_geometry=dict(canvas_raster=[256, 256], css_square=[256, 256],
                              image_smoothing=False, resizing='nearest-neighbour display only; native PNG pixels retained', filters='none'),
        playback=dict(clock='performance.now / requestAnimationFrame, left-held real relative PTS',
                      stop_at_last_anchor=True, loop=False, generated_frames=False, native_full_motion=False),
        cases=public_cases)
    # Strict public field construction; never serialize the private prior mapping here.
    write_json(viewer / 'manifest.json', public_manifest)
    with (viewer / 'manifest.js').open('x', encoding='utf-8') as f:
        f.write('window.REVIEW_DATA=' + json.dumps(public_manifest, ensure_ascii=True, allow_nan=False) + ';\n')
    with (viewer / 'index.html').open('x', encoding='utf-8') as f:
        f.write(HTML)
    with (OUT / 'README.md').open('x', encoding='utf-8') as f:
        f.write(README)
    with (OUT / 'generator-source.py').open('xb') as f:
        f.write(SOURCE.read_bytes())
    mapping = dict(review_id='input-observability-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Two-condition bridge only; native same16 timestamps and native all-frame conditions pending',
        cost_cny=0, no_model=True, no_torch=True, no_video_decoding=True, no_training=True,
        no_source_or_label_changes=True, no_visibility_judgment=True,
        input_fingerprints=inputs, prior_mapping_sha256=PRIOR_SHA,
        generator_source_sha256=digest(SOURCE.read_bytes()), environment=dict(python=sys.version, cv2=cv2.__version__, numpy=np.__version__),
        actual_preprocessing=dict(helper=helper_path.relative_to(ROOT).as_posix(), helper_sha256=helper_sha,
            route='JPEG cv2 decode BGR256 -> RGB256 -> INTER_LINEAR112; pre-normalization uint8 RGB view',
            training_profile=profile_path.relative_to(ROOT).as_posix(),
            pixel_version='jpeg95-bgr-to-rgb-inter-linear-112-v1', verified_linear_matches=linear_matches,
            verified_area_matches=area_matches, total_frames=192,
            correction='Initial AREA assumption was disproven by frozen helper and all192 RGB112 hashes; actual training contract was not changed.'),
        randomization=dict(seed=SEED, method='Sort C01..C12; shuffle sixFalse/sixTrue using Python Random(seed); reverse two actual conditions for True; present K1 before K2',
                           first_actual_condition_counts=dict(rgb112_actual=6, rgb256_jpeg_decoded=6)),
        display_geometry=public_manifest['display_geometry'], native_OSD_retained=True,
        condition_names_anonymized=True, fully_condition_blinded=False, identity_blind=False,
        prediction_and_selection_role_blinded=True, cases=private_cases, images=images,
        verification=dict(png_count=len(images), png_disk_rgb_exact_matches=len(images),
                          frozen_frame_count=192, clips=12, unique_pts_per_clip=16,
                          playback_boundary_interval_fixtures='passed for all twelve native PTS arrays'))
    write_json(OUT / 'mapping.json', mapping)
    evidence = []
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            evidence.append(dict(path=path.relative_to(OUT).as_posix(), sha256=digest(path.read_bytes()), size_bytes=path.stat().st_size))
    write_json(OUT / 'output-manifest.json', dict(files=evidence))
    for relative, item in inputs.items():
        if digest((ROOT / relative).read_bytes()) != item['sha256']:
            raise ValueError('Input changed during preparation: ' + relative)
    print(json.dumps(dict(output=OUT.relative_to(ROOT).as_posix(), clips=12, verified_source_jpegs=192,
        png_files=len(images), linear_hash_matches=linear_matches, area_hash_matches=area_matches,
        mapping_sha256=digest((OUT / 'mapping.json').read_bytes()),
        output_manifest_sha256=digest((OUT / 'output-manifest.json').read_bytes())), ensure_ascii=True))


if __name__ == '__main__':
    main()
