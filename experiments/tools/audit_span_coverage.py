"""Metadata-only 3/7.5-second coverage audit of the unchanged EXP-016 v4 pool.

No media/image is opened, hashed, decoded, or scored. Counts describe feasible
physical tuples inside original supervision segments, never training exposure.
Real execution is separate from the small CPU fixtures and refuses overwrite.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video.pts_clip_sampling import segment_candidates, tuple_digest

BASE = 'data/metadata/EXP-016/training-plan-v2/plan.json'
BASE_SHA = 'bb3af415133e1fe40a7a891bc80e2b22b0e3d44b97f2dec43ac0d6ca5750bb3f'
RESOURCE = 'data/metadata/EXP-016/metadata-v4'
SAMPLING_SHA = '931f67394c82949d81570cb1607d7fb1b3a811d549841d8d46050188b1232178'
OLD22 = 'data/process/EXP-008-units-v3-gap05/unit-failures.csv'
OLD22_SHA = '28c44af6213edfd7213f61e7559a64cb8da35566854f17c59d42c109c75ad2ff'
OLD128 = 'data/process/EXP-008-units-v3-gap05/unit-manifest.csv'
OLD128_SHA = 'e5cd061d2dd3fef917937d0bb9bdd176dc7c975cdea7cda65c2a7b6495d9f2fe'
POINT_ERROR = .100001
LONG_SPAN = 7.5
SHORT_SPAN = 3.
CENTER_OFFSET = (LONG_SPAN-SHORT_SPAN)/2
EPS = 1e-9
ALLOWED_METADATA = {'.csv', '.json', '.jsonl', '.npz', '.py'}


def metadata_path(root, name):
    """Reject media before resolving/opening paths; no absolute/path traversal."""
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name or p.suffix.lower() not in ALLOWED_METADATA:
        raise ValueError('Not an allowed relative metadata/code path: '+name)
    path = (root/Path(*p.parts)).resolve()
    path.relative_to(root)
    return path


def metadata_sha(path):
    if path.suffix.lower() not in ALLOWED_METADATA:
        raise ValueError('Refuse to open/hash media or images')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):h.update(block)
    return h.hexdigest()


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:return list(csv.DictReader(f))


def write_json(path, value):
    with path.open('x', encoding='utf-8') as f:json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def validate_runs(pts, clean_runs):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 1 or not np.isfinite(pts).all():raise ValueError('Nonfinite/non-vector PTS')
    previous = 0
    for a, b in clean_runs:
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(pts) or a < previous:
            raise ValueError('Invalid/overlapping clean runs')
        if np.any(np.diff(pts[a:b]) < 0):raise ValueError('PTS backstep inside clean run')
        previous = b
    return pts


def intersections(pts, clean_runs, start, end):
    if not np.isfinite([start, end]).all() or not start < end:raise ValueError('Invalid sampling bounds')
    for a, b in clean_runs:
        local = pts[a:b]
        left = a+int(np.searchsorted(local, start, side='left'))
        right = a+int(np.searchsorted(local, end, side='left'))
        if left < right:yield left, right


def nearest_many(local, starts, span):
    """Same nearest/tie-left rule as v4; targets must fully fit one clean run."""
    local = np.asarray(local, dtype=np.float64);starts = np.asarray(starts, dtype=np.float64)
    if (local.ndim != 1 or not len(local) or starts.ndim != 1
            or not np.isfinite(local).all() or not np.isfinite(starts).all() or np.any(np.diff(local) < 0)
            or not np.isfinite(span) or span <= 0):raise ValueError('Invalid nearest-neighbour metadata')
    targets = starts[:, None]+np.linspace(0, span, 16)
    ri = np.clip(np.searchsorted(local, targets), 0, len(local)-1)
    li = np.maximum(ri-1, 0)
    ix = np.where(abs(local[li]-targets) <= abs(local[ri]-targets), li, ri)
    actual = local[ix];actual_spans = actual[:, -1]-actual[:, 0]
    good = ((starts >= local[0]-EPS) & (starts+span <= local[-1]+EPS)
            & np.all(np.diff(ix, axis=1) > 0, axis=1)
            & np.all(np.diff(actual, axis=1) > 0, axis=1)
            & (np.max(abs(actual-targets), axis=1) <= POINT_ERROR)
            & (actual_spans >= span-.1-EPS) & (actual_spans <= span+.1+EPS))
    return ix.astype(np.int32), good


def alias_records(long_ix, short_ix, long_good, short_good, anchors):
    """Compute paired eligibility BEFORE any alias deduplication of long tuples."""
    joint = long_good & short_good
    return long_ix[long_good], anchors[long_good], np.concatenate((short_ix[joint], long_ix[joint]), axis=1), anchors[joint]


def _unique(parts, anchors, width):
    if not parts:return np.empty((0, width), dtype=np.int32), np.empty(0)
    unique, first = np.unique(np.concatenate(parts), axis=0, return_index=True)
    return unique.astype(np.int32), np.concatenate(anchors)[first]


def _representatives(n):
    return np.unique(np.rint(np.linspace(0, n-1, min(3, n))).astype(int)) if n else []


def window_witness(pts, ix, start, span):
    targets = start+np.linspace(0, span, 16);actual = pts[ix]
    return dict(frame_indices=ix.tolist(), source_pts_seconds=actual.tolist(), target_seconds=targets.tolist(),
                point_errors_seconds=(actual-targets).tolist(), max_point_error_seconds=float(max(abs(actual-targets))),
                actual_span_seconds=float(actual[-1]-actual[0]), actual_center_seconds=float((actual[0]+actual[-1])/2))


def audit_segment(pts, clean_runs, start, end, chunk_size=32768):
    """Full enumeration, retaining only bounded witnesses and deterministic hashes."""
    pts = validate_runs(pts, clean_runs)
    if type(chunk_size) is not int or chunk_size < 1:raise ValueError('Invalid chunk size')
    short = segment_candidates(pts, clean_runs, start, end)
    long_parts=[];long_anchors=[];pair_parts=[];pair_anchors=[];net_runs=[]
    anchor_count=raw_long=raw_pairs=0
    for left, right in intersections(pts, clean_runs, start, end):
        local = pts[left:right]
        net_runs.append(dict(start_frame=left, end_frame_exclusive=right, n_frames=right-left,
                             first_pts=float(local[0]), last_pts=float(local[-1]), actual_span_seconds=float(local[-1]-local[0])))
        anchors = local[(local+LONG_SPAN <= end+EPS) & (local+LONG_SPAN <= local[-1]+EPS)]
        anchor_count += len(anchors)
        for offset in range(0, len(anchors), chunk_size):
            s = anchors[offset:offset+chunk_size]
            li, lg = nearest_many(local, s, LONG_SPAN)
            si, sg = nearest_many(local, s+CENTER_OFFSET, SHORT_SPAN)
            l, la, p, pa = alias_records(li+left, si+left, lg, sg, s)
            raw_long += len(l);raw_pairs += len(p)
            if len(l):long_parts.append(l);long_anchors.append(la)
            if len(p):pair_parts.append(p);pair_anchors.append(pa)
    long, ls = _unique(long_parts, long_anchors, 16)
    pair, ps = _unique(pair_parts, pair_anchors, 32)
    center_difference = ((pts[pair[:, 16]]+pts[pair[:, 31]])-(pts[pair[:, 0]]+pts[pair[:, 15]]))/2 if len(pair) else np.empty(0)
    witnesses = dict(three_seconds=[window_witness(pts, short[i], float(pts[short[i, 0]]), 3.) for i in _representatives(len(short))],
        seven_point_five_seconds=[window_witness(pts, long[i], float(ls[i]), 7.5) for i in _representatives(len(long))],
        common_nominal_center_pairs=[dict(long=window_witness(pts, pair[i, 16:], float(ps[i]), 7.5),
            short=window_witness(pts, pair[i, :16], float(ps[i])+CENTER_OFFSET, 3.),
            nominal_center_seconds=float(ps[i]+3.75), actual_long_minus_short_center_seconds=float(center_difference[i])) for i in _representatives(len(pair))])
    counts = dict(n_candidates_3s=len(short), candidates_3s_sha256=tuple_digest(short),
        n_candidates_7p5s=len(long), candidates_7p5s_sha256=tuple_digest(long),
        n_common_center_pairs=len(pair), common_center_pairs_sha256=tuple_digest(pair),
        eligible_long_anchor_rows=anchor_count, legal_long_before_dedup=raw_long, legal_pairs_before_dedup=raw_pairs,
        longest_clean_pts_span_seconds=max((r['actual_span_seconds'] for r in net_runs), default=0.),
        n_sampling_clean_intersections=len(net_runs),
        max_abs_actual_center_difference_seconds=float(max(abs(center_difference))) if len(pair) else 0.)
    return counts, witnesses, short, net_runs


def class_row(video, kind, segments, old22):
    totals = {key:sum(s[key] for s in segments) for key in ('n_candidates_3s','n_candidates_7p5s','n_common_center_pairs')}
    if not segments:state='no_supervision_segment'
    elif not totals['n_candidates_3s'] and not totals['n_candidates_7p5s']:state='zero_3s_and_7p5s_candidates'
    elif not totals['n_candidates_3s']:state='zero_3s_real_pts_anchored_candidates'
    elif not totals['n_candidates_7p5s']:state='zero_7p5s_candidates'
    elif not totals['n_common_center_pairs']:state='separately_available_but_zero_common_center_pairs'
    else:state='common_center_pairs_available'
    return dict(unit_id=video['unit_id'], split=video['split'], group_id=video['group_id'], video_label=video['video_label'],
        kind=kind, old22=video['unit_id'] in old22, is_own_video_label=kind==video['video_label'],
        supervision_segment_present=bool(segments), human_annotation_present=any(s['origin']=='explicit_human_event' for s in segments),
        three_seconds_available=totals['n_candidates_3s']>0, seven_point_five_seconds_available=totals['n_candidates_7p5s']>0,
        common_center_pairs_available=totals['n_common_center_pairs']>0,
        n_supervision_segments=len(segments), coverage_state=state,
        longest_clean_pts_span_seconds=max((s['longest_clean_pts_span_seconds'] for s in segments), default=0.), **totals)


def _coverage(rows):
    own = [r for r in rows if r['is_own_video_label']]
    return dict(videos=len(own), own_label_3s=sum(r['n_candidates_3s']>0 for r in own),
        own_label_7p5s=sum(r['n_candidates_7p5s']>0 for r in own), own_label_common_center_pairs=sum(r['n_common_center_pairs']>0 for r in own),
        own_label_both_separately=sum(r['n_candidates_3s']>0 and r['n_candidates_7p5s']>0 for r in own),
        own_label_zero_7p5s_ids=[r['unit_id'] for r in own if not r['n_candidates_7p5s']],
        own_label_zero_3s_ids=[r['unit_id'] for r in own if not r['n_candidates_3s']],
        own_label_separately_available_zero_pair_ids=[r['unit_id'] for r in own if r['n_candidates_3s']>0 and r['n_candidates_7p5s']>0 and not r['n_common_center_pairs']],
        class_rows=len(rows), coverage_states=dict(Counter(r['coverage_state'] for r in rows)))


def run(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.relative_to(root/'data/metadata/span-coverage')
    if out.exists():raise FileExistsError(out)
    verified={}
    def verify(name, sha, size=None):
        p=metadata_path(root,name)
        if (size is not None and p.stat().st_size!=size) or metadata_sha(p)!=sha:raise ValueError('Metadata fingerprint mismatch: '+name)
        verified[name]=sha;return p
    started=time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    try:
        base=json.loads(verify(BASE, BASE_SHA).read_text(encoding='utf-8'))
        if base['resource']!=RESOURCE:raise ValueError('Registered base points to another resource')
        refs={r['path']:r for r in base['sources']}
        closure_name=RESOURCE+'/frozen-plan-closure.json';ref=refs[closure_name]
        closure=json.loads(verify(closure_name,ref['sha256'],ref['size_bytes']).read_text(encoding='utf-8'))
        if closure['metadata_path']!=RESOURCE or len({r['path'] for r in closure['files']})!=len(closure['files']):raise ValueError('Wrong/duplicate metadata closure')
        for item in closure['files']:verify(item['path'],item['sha256'],item['size_bytes'])
        summary=json.loads(verify(RESOURCE+'/summary.json',closure['summary_sha256']).read_text(encoding='utf-8'))
        if summary['common_sampling_sha256']!=SAMPLING_SHA:raise ValueError('Changed frozen 3s helper contract')
        verify('experiments/common/video/pts_clip_sampling.py',SAMPLING_SHA)
        generator_name=Path(__file__).resolve().relative_to(root).as_posix()
        generator=verify(generator_name,metadata_sha(Path(__file__)))
        with (out/'generator-source.py').open('xb') as f:f.write(generator.read_bytes())
        if metadata_sha(out/'generator-source.py')!=verified[generator_name]:raise ValueError('Generator source copy changed')
        videos=csv_rows(metadata_path(root,RESOURCE+'/video-inputs.csv'))
        segments=csv_rows(metadata_path(root,RESOURCE+'/segments.csv'))
        ids={v['unit_id'] for v in videos}
        if len(videos)!=150 or len(ids)!=150 or Counter(v['split'] for v in videos)!=Counter(train=119,validation=31):raise ValueError('Changed 150/119/31 pool')
        if {v['group_id'] for v in videos if v['split']=='train'} & {v['group_id'] for v in videos if v['split']=='validation'}:raise ValueError('Group overlap')
        old22_rows=csv_rows(verify(OLD22,OLD22_SHA));old128_rows=csv_rows(verify(OLD128,OLD128_SHA))
        old22={r['unit_id'] for r in old22_rows};old128={r['unit_id'] for r in old128_rows}
        if len(old22_rows)!=22 or len(old22)!=22 or len(old128_rows)!=128 or len(old128)!=128 or old22!=ids-old128 or old128|old22!=ids:raise ValueError('Old22/128 identity complement mismatch')
        by_unit=defaultdict(list)
        for s in segments:
            if s['unit_id'] not in ids or s['kind'] not in ('pos','neg'):raise ValueError('Unknown supervision source/kind')
            by_unit[s['unit_id']].append(s)
        all_segments=[];all_classes=[]
        with (out/'witnesses.jsonl').open('x',encoding='utf-8') as witnesses_file:
            for video in videos:
                uid=video['unit_id'];unit_segments=by_unit[uid]
                if video['video_label'] not in ('pos','neg') or not unit_segments or len({s['segment_id'] for s in unit_segments})!=len(unit_segments):raise ValueError('Invalid labels/duplicate or missing segments')
                coord_path=verify(video['coord_path'],video['coord_sha256']);qc_path=verify(video['qc_path'],video['qc_sha256'])
                coord=np.loadtxt(coord_path,delimiter=',',skiprows=1,usecols=(0,1),ndmin=2)
                with qc_path.open(encoding='utf-8-sig') as f:header=next(csv.reader(f))
                qc=np.loadtxt(qc_path,delimiter=',',skiprows=1,usecols=(header.index('frame'),header.index('pts_time'),header.index('gray_std')),ndmin=2)
                n=int(video['n_frames']);pts=coord[:,1]
                if (len(coord)!=n or len(qc)!=n or not np.array_equal(coord[:,0],np.arange(n)) or not np.array_equal(qc[:,0],np.arange(n))
                        or not np.isfinite(coord).all() or not np.isfinite(qc).all() or max(abs(qc[:,1]-pts))>.001001):raise ValueError(uid+': coord/QC count/order/PTS failure')
                clean=json.loads(video['clean_runs_json']);pts=validate_runs(pts,clean)
                forbidden=json.loads(video['forbidden_ranges_json'])
                from experiments.common.video.pts_clip_sampling import complement_ranges
                if clean!=[list(r) for r in complement_ranges(forbidden,n)]:raise ValueError(uid+': clean/forbidden complement mismatch')
                if any(np.any(qc[a:b,2]<20.) for a,b in clean):raise ValueError(uid+': low-QC frame inside clean run')
                candidate_name='data/process/EXP-016/metadata-v4/candidates/'+uid+'.npz'
                if candidate_name not in verified:raise ValueError('Candidate arrays absent from bound closure')
                unit_results=[]
                with np.load(metadata_path(root,candidate_name),allow_pickle=False) as frozen:
                    if set(frozen.files)!={s['segment_id'] for s in unit_segments}:raise ValueError('Candidate segment identity mismatch')
                    for s in unit_segments:
                        a,b=float(s['sampling_start_sec']),float(s['sampling_end_sec'])
                        if not (float(s['annotation_start_sec'])<=a<b<=float(s['annotation_end_sec'])) or s['split']!=video['split'] or s['group_id']!=video['group_id']:raise ValueError('Sampling extended annotation or source roles changed')
                        counts,witness,three,net_runs=audit_segment(pts,clean,a,b)
                        stored=frozen[s['segment_id']]
                        if (stored.dtype!=np.dtype('int32') or stored.shape!=three.shape or not np.array_equal(stored,three)
                                or len(three)!=int(s['n_candidates']) or tuple_digest(three)!=s['candidates_sha256']):raise ValueError(uid+': frozen full3s candidate mismatch '+s['segment_id'])
                        evidence=dict(unit_id=uid,segment_id=s['segment_id'],kind=s['kind'],origin=s['origin'],
                            middleware_path=video['middleware_path'],middleware_sha256_inherited=video['middleware_sha256'],
                            coord_sha256=video['coord_sha256'],qc_sha256=video['qc_sha256'],candidates_npz_sha256=verified[candidate_name],
                            sampling_start_sec=a,sampling_end_sec=b,sampling_clean_intersections=net_runs,**witness)
                        witnesses_file.write(json.dumps(evidence,ensure_ascii=False,allow_nan=False)+'\n')
                        row=dict(s,**counts);unit_results.append(row);all_segments.append(row)
                for kind in ('pos','neg'):all_classes.append(class_row(video,kind,[r for r in unit_results if r['kind']==kind],old22))
        if len(all_classes)!=300 or sum(r['n_candidates_3s'] for r in all_segments)!=summary['candidate_tuples']:raise ValueError('Incomplete full-candidate enumeration')
        write_csv(out/'segments.csv',all_segments);write_csv(out/'video-classes.csv',all_classes)
        groups=[]
        for group in sorted({r['group_id'] for r in all_classes}):
            for kind in ('pos','neg'):
                rows=[r for r in all_classes if r['group_id']==group and r['kind']==kind]
                groups.append(dict(group_id=group,split=rows[0]['split'],kind=kind,videos=len(rows),
                    with_supervision_segments=sum(r['supervision_segment_present'] for r in rows),
                    human_annotated_videos=sum(r['human_annotation_present'] for r in rows),
                    with_3s=sum(r['n_candidates_3s']>0 for r in rows),with_7p5s=sum(r['n_candidates_7p5s']>0 for r in rows),
                    with_common_center_pairs=sum(r['n_common_center_pairs']>0 for r in rows),
                    old22_videos=sum(r['old22'] for r in rows),
                    old22_with_3s=sum(r['old22'] and r['n_candidates_3s']>0 for r in rows),
                    old22_with_7p5s=sum(r['old22'] and r['n_candidates_7p5s']>0 for r in rows),
                    old22_with_common_center_pairs=sum(r['old22'] and r['n_common_center_pairs']>0 for r in rows)))
        write_csv(out/'groups.csv',groups)
        for name,sha in verified.items():
            if metadata_sha(metadata_path(root,name))!=sha:raise RuntimeError('Metadata/code changed during audit: '+name)
        result=dict(status='metadata_coverage_audit_passed',source_verification_scope='metadata_subset_only_no_media_or_anchor_images',
            source_media_identity='inherited unchanged from bound metadata-v4 video-inputs; media not opened or hashed',
            historical_metadata_fingerprints='bound historical declarations only; live skill/generation sources/anchor images not revalidated',
            source_base_sha256=BASE_SHA,source_closure_sha256=verified[closure_name],frozen_common_sampling_sha256=SAMPLING_SHA,
            verified_metadata_and_code_sha256=verified,original_v4_summary_metadata_fingerprints=summary['metadata_fingerprints'],
            exact_3s_full_candidates_matched=True,candidates_3s_total=sum(r['n_candidates_3s'] for r in all_segments),
            candidates_7p5s_total=sum(r['n_candidates_7p5s'] for r in all_segments),common_center_pairs_total=sum(r['n_common_center_pairs'] for r in all_segments),
            count_domain='Counts sum original supervision segments, including protocol9 implicit segments; unique physical tuples within each segment, no quotas or training exposure. Long unique tuples and (short,long) unique pairs have separate domains.',
            nearest_rule='16 target points; tie left; original indices and actual PTS strictly increase; point error<=0.100001; actual span nominal+/-0.1; full nominal targets inside sampling [a,b) intersect one clean run',
            anchor_rule='7.5s starts at each real PTS; centered short target starts s+2.25, not necessarily real PTS; both generated before long/pair dedup',
            tuple_hash_format='SHA256 little-endian signed int32 row-major; long/pairs lexicographic; 3s uses original v4 center ordering',
            overall=_coverage(all_classes),by_split={s:_coverage([r for r in all_classes if r['split']==s]) for s in ('train','validation')},
            by_video_label={s:_coverage([r for r in all_classes if r['video_label']==s]) for s in ('pos','neg')},
            old22=dict(ids=sorted(old22),identity_only_no_old_capacity_recalculation=True,coverage=_coverage([r for r in all_classes if r['old22']]),
                by_video_label={s:_coverage([r for r in all_classes if r['old22'] and r['video_label']==s]) for s in ('pos','neg')},
                by_split={s:_coverage([r for r in all_classes if r['old22'] and r['split']==s]) for s in ('train','validation')}),
            media_or_images_opened=0,decoded_frames=0,model_scores_read=False,pixels_ready=False,training_plan_created=False,
            argv=sys.argv,seconds=time.perf_counter()-started)
        write_json(out/'summary.json',result)
        artifacts={p.name:metadata_sha(p) for p in out.iterdir() if p.is_file()}
        write_json(out/'artifacts.json',dict(status='complete',files_sha256=artifacts))
        return result
    except Exception as error:
        write_json(out/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error),partial_artifacts_preserved=True))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--out',type=Path,default=ROOT/'data/metadata/span-coverage/metadata-v4-20261101-v1')
    a=p.parse_args();r=run(a.root,a.out)
    print(json.dumps(dict(status=r['status'],overall=r['overall'],seconds=r['seconds']),ensure_ascii=False))


if __name__=='__main__':main()
