"""Four fixed training-video context pairs; metadata-first, one extraction, no model."""
from pathlib import Path
import csv
import json
import random
import re
import subprocess
import sys
import threading
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.common.video import observability_temporal as common

OUT = ROOT / 'data/process/temporal-meeting/20261101-v1'
IDS = ('v1_01', 'v1_16', 'v1_91', 'v1_95')
GROUPS = ('d|鸡冠山上游|2023-08-02', 'd|鸡冠山下游|2024-07-28', 'd|塔吊上游|2025-07-29', 'd|塔吊下游|2025-07-29')
VIDEO_SHA = '3b6be81e4a4fcd1943ab3f4b535868f6a3342a53435948957902e5b557668e3c'
SEGMENT_SHA = '58ab042cc82d969fe59b69d85afcfc255d795db54d4a01012c8012260727a52e'


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))


def choose(pts, std, segments, clean, kind):
    """Fixed margin priority; longest continuous interval, midpoint-nearest12s."""
    candidates = []
    for before, after in ((10., 3.), (3., 3.), (0., 0.)):
        for seg in segments:
            if seg['kind'] != kind: continue
            left = max(float(seg['sampling_start_sec']), float(seg['annotation_start_sec']) + before)
            right = min(float(seg['sampling_end_sec']), float(seg['annotation_end_sec']) - after)
            for ca, cb in clean:
                start = None
                for i in range(ca, cb + 1):
                    valid = i < cb and left <= pts[i] < right and std[i] >= 20.
                    split = start is not None and i < cb and pts[i] <= pts[i - 1]
                    if start is not None and (not valid or split):
                        if pts[i - 1] - pts[start] >= 12.:
                            candidates.append((float(pts[i - 1] - pts[start]), start, i - 1, seg, before, after))
                        start = None
                    if valid and start is None: start = i
        if candidates: break
    if not candidates: return dict(status='missing', reason='no_continuous_12s_interval_under_frozen_margin_priority')
    span, a, b, seg, before, after = sorted(candidates, key=lambda r: (-r[0], r[1], r[3]['segment_id']))[0]
    mid = (pts[a] + pts[b]) / 2
    lo = a + int(np.argmin(abs(pts[a:b + 1] - (mid - 6.))))
    hi = a + int(np.argmin(abs(pts[a:b + 1] - (mid + 6.))))
    actual = pts[lo:hi + 1]
    if (not 11.8 <= actual[-1] - actual[0] <= 12.2 or np.any(np.diff(actual) <= 0)
            or np.any(std[lo:hi + 1] < 20.)):
        return dict(status='missing', reason='midpoint_selected_window_failed_duration_or_PTS_or_QC_no_reselection')
    return dict(status='selected', original_kind=kind, segment_id=seg['segment_id'],
        original_annotation=[float(seg['annotation_start_sec']), float(seg['annotation_end_sec'])],
        sampling_core=[float(seg['sampling_start_sec']), float(seg['sampling_end_sec'])],
        boundary_margins_sec=[before, after], longest_legal_interval_indices=[a, b],
        longest_legal_interval_span_sec=span, frame_indices=list(range(lo, hi + 1)),
        source_pts_seconds=actual.tolist(), span_first_to_last_sec=float(actual[-1] - actual[0]),
        midpoint_error_sec=float(abs((actual[0] + actual[-1]) / 2 - mid)))


def command_for(case, video, decoder):
    ix = case['frame_indices']; target = OUT / 'private/media' / case['public_filename']
    vf = "select='between(n,%d,%d)',showinfo,setpts=PTS-STARTPTS" % (ix[0], ix[-1])
    return [str(decoder), '-hide_banner', '-nostdin', '-n', '-copyts', '-threads', '1', '-i', str(ROOT / video['middleware_path']),
        '-map', '0:v:0', '-vf', vf, '-an', '-map_metadata', '-1', '-map_chapters', '-1',
        '-c:v', 'libx264', '-crf', '10', '-preset', 'fast', '-pix_fmt', 'yuv420p',
        '-fps_mode', 'passthrough', '-enc_time_base', 'demux', '-video_track_timescale', '1000000',
        '-frames:v', str(len(ix)), '-fs', '24000000', '-movflags', '+faststart', str(target)]


def encode(case, video, decoder, probe, budget):
    name = case['public_filename']; target = OUT / 'private/media' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    ix, expected = case['frame_indices'], np.array(case['source_pts_seconds'])
    command = command_for(case, video, decoder)
    if command != case['command'] or common.digest_bytes(json.dumps(command).encode('utf-8')) != case['command_sha256']:
        raise ValueError('frozen extraction command differs')
    budget.check()
    if budget.used + 27_000_000 > budget.byte_limit - common.AUDIT_RESERVE: raise common.LimitReached('insufficient_reserved_clip_and_log_bytes')
    budget.json(OUT / 'private' / (name + '.command.json'), command)
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    budget.process = process; log, actual, log_bytes = [], [], 0
    try:
        for raw in iter(process.stderr.readline, b''):
            budget.check(); log_bytes += len(raw)
            if log_bytes > 2_000_000: raise common.LimitReached('per_clip_decoder_log_cap')
            line = raw.decode('utf-8', errors='replace'); log.append(line)
            match = common.PHYSICAL.search(line)
            if match:
                i, t = int(match[1]), float(match[2])
                if i != len(actual) or i >= len(expected) or 's:256x256' not in line or abs(t - expected[i]) > .001001:
                    raise ValueError('selected source index/count/PTS/dimension mismatch')
                actual.append(t)
            if any(s in line.lower() for s in ('error while decoding', 'invalid nal', 'corrupt decoded')):
                raise ValueError('source decoder error')
        if process.wait() or len(actual) != len(expected): raise ValueError('incomplete source frame extraction')
    finally:
        if process.poll() is None: process.kill()
        process.wait(); process.stderr.close(); budget.process = None
        if target.exists(): budget.used += target.stat().st_size
        budget.write(OUT / 'private' / (name + '.decoder.log'), ''.join(log).encode('utf-8'), audit=True)
    if target.stat().st_size > 25_000_000: raise common.LimitReached('per_clip_mp4_byte_cap')
    probe_command = [str(probe), '-v', 'error', '-show_frames', '-show_streams', '-show_format',
        '-show_entries', 'frame=best_effort_timestamp_time,duration_time,pkt_duration_time:stream=codec_type,codec_name,width,height,duration,time_base,nb_frames:format=duration', '-of', 'json', str(target)]
    result = subprocess.run(probe_command, capture_output=True, text=True, check=True, timeout=min(30., max(.01, 600 - (time.monotonic() - budget.started))))
    data = json.loads(result.stdout); frames = data['frames']; streams = data['streams']
    output_pts = np.array([float(f['best_effort_timestamp_time']) for f in frames])
    if (len(streams) != 1 or streams[0]['codec_name'] != 'h264' or streams[0]['width'] != 256 or streams[0]['height'] != 256
            or len(frames) != len(expected) or not np.isfinite(output_pts).all() or np.any(np.diff(output_pts) <= 0)
            or not np.allclose(output_pts, expected - expected[0], rtol=0., atol=.001001)):
        raise ValueError('encoded MP4 frame count/timeline/geometry/codec check failed')
    budget.json(OUT / 'private' / (name + '.ffprobe.json'), dict(command=probe_command, result=data))
    return dict(public_filename=name, frames=len(frames), source_first_to_last_sec=float(expected[-1] - expected[0]),
        encoded_first_to_last_sec=float(output_pts[-1] - output_pts[0]), format_duration_sec=float(data['format']['duration']),
        final_frame_display_duration_sec=(float(frames[-1].get('duration_time', frames[-1].get('pkt_duration_time')))
            if frames[-1].get('duration_time', frames[-1].get('pkt_duration_time')) is not None else None),
        sha256=common.digest_file(target, budget), size_bytes=target.stat().st_size, pixels='lossy_CRF10_reencode_not_model_input')


def main():
    if OUT.exists(): raise FileExistsError('Refuse prior meeting material; no rerun or overwrite')
    OUT.mkdir(parents=True); budget = common.Budget(OUT, seconds=600, byte_limit=250_000_000)
    watch = threading.Thread(target=budget.watchdog, daemon=True); watch.start()
    inputs, groups, results, published, error = {}, [], [], [], None
    def bind(path, expected=None):
        path = Path(path).resolve(); value = common.digest_file(path, budget)
        if expected is not None and value != expected: raise ValueError('input SHA changed: ' + str(path))
        inputs[str(path)] = dict(sha256=value, size_bytes=path.stat().st_size); return path
    try:
        source = Path(__file__).resolve(); bind(source); bind(Path(common.__file__).resolve())
        meta = ROOT / 'data/metadata/EXP-016/metadata-v4'
        videos = {r['unit_id']: r for r in rows(bind(meta / 'video-inputs.csv', VIDEO_SHA))}
        segments = rows(bind(meta / 'segments.csv', SEGMENT_SHA))
        tools = Path('D:/CodexCache/debris-flow-thesis/tools/ffmpeg-9.0.2')
        decoder, probe = bind(tools / 'ffmpeg.exe'), bind(tools / 'ffprobe.exe')
        flips = [False, False, True, True]; random.Random(20261101).shuffle(flips)
        for index, (uid, expected_group, flip) in enumerate(zip(IDS, GROUPS, flips), 1):
            v = videos[uid]
            if v['split'] != 'train' or v['group_id'] != expected_group: raise ValueError('fixed training-side group identity mismatch')
            common.rooted(ROOT, v['middleware_path']).relative_to(ROOT / 'data/process/EXP-008-units-src256')
            for stem in ('middleware', 'coord', 'qc'): bind(ROOT / v[stem + '_path'], v[stem + '_sha256'])
            coord, qc = rows(ROOT / v['coord_path']), rows(ROOT / v['qc_path'])
            if len(coord) != int(v['n_frames']) or len(qc) != len(coord): raise ValueError('source coord/QC count mismatch')
            if [int(r['frame']) for r in coord] != list(range(len(coord))) or [int(r['frame']) for r in qc] != list(range(len(qc))): raise ValueError('source frame order mismatch')
            pts = np.array([float(r['pts_time']) for r in coord]); std = np.array([float(r['gray_std']) for r in qc])
            qt = np.array([float(r['pts_time']) for r in qc])
            if not np.isfinite([pts, qt, std]).all() or np.max(abs(qt - pts)) > .001001: raise ValueError('source finite/PTS mismatch')
            selected = [choose(pts, std, [s for s in segments if s['unit_id'] == uid], json.loads(v['clean_runs_json']), kind)
                        for kind in (('neg', 'pos') if flip else ('pos', 'neg'))]
            for alias, item in zip(('A', 'B'), selected):
                item['public_filename'] = 'G%02d-%s.mp4' % (index, alias)
                if item['status'] == 'selected':
                    item['command'] = command_for(item, v, decoder)
                    item['command_sha256'] = common.digest_bytes(json.dumps(item['command']).encode('utf-8'))
            groups.append(dict(group_id='G%02d' % index, unit_id=uid, source_group_id=v['group_id'], selections=selected))
        plan = dict(status='selection_frozen_before_any_media_extraction', fixed_training_ids=IDS, groups=groups, input_fingerprints=inputs,
            rule='margin tiers(10,3),(3,3),(0,0); first tier with >=12s run; longest same-original-segment/core/clean/QC/strict-PTS interval, ties earliest index then segment; nearest indices to midpoint+-6s; no image-based reselection',
            limits=dict(seconds=600, bytes=250_000_000, per_clip_mp4_fs=24_000_000, per_clip_log_bytes=2_000_000),
            encoding='H264 CRF10 fast yuv420p; selected physical n range; original PTS gaps; relative PTS reset only;256x256; no audio/ROI/resize/interpolation/enhancement; pixels differ from model input',
            seed=20261101, no_model_scores=True, no_model=True, no_visibility_judgment=True)
        budget.json(OUT / 'private/frozen-plan.json', plan)
        budget.write(OUT / 'private/generator-source.py', source.read_bytes())
        for group in groups:
            if any(c['status'] != 'selected' for c in group['selections']):
                results.append(dict(group_id=group['group_id'], status='whole_group_missing', reason='one_class_has_no_frozen_legal_context')); continue
            group_results = [encode(c, videos[group['unit_id']], decoder, probe, budget) for c in group['selections']]
            results.append(dict(group_id=group['group_id'], status='verified', clips=group_results))
        for name, ref in inputs.items():
            if common.digest_file(Path(name), budget) != ref['sha256']: raise ValueError('input/source changed during processing')
        for result in results:
            if result['status'] != 'verified': continue
            for clip in result['clips']:
                old = OUT / 'private/media' / clip['public_filename']; new = OUT / 'public' / clip['public_filename']
                new.parent.mkdir(exist_ok=True); budget.check(); old.rename(new); published.append(clip)
    except Exception as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
    finally:
        budget.stop.set(); watch.join(timeout=1)
        public_note = '组会讨论材料。每个G编号的A/B来自同一已验256中间件的不同连续时间段，按播放器正常1倍速播放。\n无声；约12秒；H264 CRF10重新编码，像素不是模型原输入；未裁ROI、放大、增强、插帧或变速。\n原有OSD保留，不能称完全身份盲化。本材料不证明两类可辨，不是量化验证或独立样本；来源和原标签仅在private/frozen-plan.json中。\n'
        budget.write(OUT / 'public/README.txt', public_note.encode('utf-8'), audit=True)
        note = '# 训练侧同背景连续上下文材料\n\n仅供下次组会讨论动态线索；不作内容判断、可辨性或模型结论。\n\npublic/仅含匿名G01–G04的A/B短片与播放说明。private/frozen-plan.json为身份、原标签、选窗算法、完整源帧号/PTS及SHA的私有映射；decoder.log、command.json与ffprobe.json记录命令和时轴验收。\n\n最终帧的显示时长与首末帧时间跨度分别记录，容器时长不当作额外事件。任何缺失保留原因，不更换固定ID或看画面后改选。\n'
        budget.write(OUT / 'README.md', note.encode('utf-8'), audit=True)
        budget.json(OUT / 'private/result.json', dict(status='complete' if error is None else 'failed_no_retry',
            error=error, groups=results, published=published, seconds=time.monotonic() - budget.started,
            source_SHA_closed=error is None, no_content_judgment=True, cost_cny=0), audit=True)
        files = [dict(path=p.relative_to(OUT).as_posix(), sha256=common.digest_file(p), size_bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file()]
        budget.json(OUT / 'output-manifest.json', dict(files=files), audit=True)
        total = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        print(json.dumps(dict(status='complete' if error is None else 'failed_no_retry', groups=len(published) // 2, clips=len(published), bytes=total, seconds=time.monotonic() - budget.started, error=error)))
    if error is not None: raise SystemExit(1)


if __name__ == '__main__':
    main()
