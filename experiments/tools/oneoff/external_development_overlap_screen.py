"""Fixed 54x450 pHash screen, not proof of independent external testing."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import unittest

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
META=ROOT/'data/metadata/external-videos/development-overlap-screen-20261102-v1'
OUT=ROOT/'data/process/external-videos/development-overlap-screen-20261102-v1'
PLAN=ROOT/'data/metadata/EXP-017/role-free-plan-v3'
ACCEPT=ROOT/'data/metadata/EXP-017/cv-cache-acceptance-v1/acceptance.json'
SELECTION=ROOT/'data/metadata/external-videos/label-review-20261101-v2/selection.json'
AUDIT=ROOT/'data/metadata/external-videos/external-input-audit-20261102-v1'
DENSE=ROOT/'data/process/EXP-017/dense-pixel-reference-v1/result.json'
FFMPEG=Path('D:/CodexCache/debris-flow-thesis/tools/ffmpeg-9.0.2/ffmpeg.exe')


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def table(path,values):
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)


def phash_bytes(data):
    gray=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if gray is None:gray=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_GRAYSCALE)
    require(gray is not None and gray.size>0,'Corrupt/empty image; no hash substituted')
    small=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients=cv2.dct(small)[:8,:8].reshape(-1)[1:]
    require(np.isfinite(coefficients).all(),'Nonfinite DCT')
    packed=0
    for bit in coefficients>np.median(coefficients):packed=(packed<<1)|int(bit)
    return packed


def distance(a,b):return bin(a^b).count('1')  # Python3.8, no int.bit_count.


def first_middle_last(records):
    records=sorted(records,key=lambda r:int(r['frame_index']))
    indices=[int(r['frame_index']) for r in records]
    require(len(indices)>=3 and len(set(indices))==len(indices),'Insufficient/repeated accepted indices')
    return [records[i] for i in (0,len(records)//2,len(records)-1)]


def freeze():
    META.mkdir(parents=True,exist_ok=False);OUT.mkdir(parents=True,exist_ok=False)
    plan=read(PLAN/'plan.json');accept=read(ACCEPT);selection=read(SELECTION);dense=read(DENSE)
    index=ROOT/accept['verified_required_frames_index'];sources=rows(PLAN/'sources.csv')
    require(sha(PLAN/'plan.json')==accept['plan_sha256'] and sha(PLAN/'sources.csv')==plan['files']['sources.csv'],'Development plan binding differs')
    require(accept['status']=='cv_supervised_cache_ready' and accept['verified_required_frame_count']==191812 and accept['failed_required_frames']==0 and sha(index)==accept['verified_required_frames_index_sha256'],'Actual JPEG acceptance binding differs')
    require(len(sources)==len({r['unit_id'] for r in sources})==150 and dense['status']=='cv_dense_pixels_verified' and dense['video_count']==150,'Development150 identity incomplete')
    selected=selection['selected_media'];require(len(selected)==18 and sum(r['proposed_video_label']=='pos' for r in selected)==7,'Unchanged selection18 required')
    accepted=rows(index);by_unit={r['unit_id']:[] for r in sources};seen=set()
    for r in accepted:
        key=(r['unit_id'],int(r['frame_index']));require(key not in seen and r['unit_id'] in by_unit and r['status']=='verified' and r['jpeg_sha256']==r['actual_jpeg_sha256'],'Accepted-frame identity/coverage invalid')
        seen.add(key);by_unit[r['unit_id']].append(r)
    require(len(accepted)==191812,'Accepted frame count differs')
    development=[];mapping=[]
    for source in sources:
        uid=source['unit_id'];require(dense['videos'][uid]['source_media_sha256']==source['middleware_sha256'],'Existing full dense source identity differs')
        for r in first_middle_last(by_unit[uid]):development.append(dict(r))
        mapping.append(dict(population='development',unit_id=uid,publisher_or_source=source['source'],site_or_group=source['group_id'],event_or_video=source['video_id'],media_path=source['middleware_path'],declared_media_sha256=source['middleware_sha256']))
    evidence=read(AUDIT/'delivery-manifest.json');known={r['path']:r['sha256'] for r in evidence['files']};external=[]
    inherited={rel(p):sha(p) for p in [PLAN/'plan.json',PLAN/'sources.csv',ACCEPT,index,SELECTION,DENSE,AUDIT/'reconciliation.json',AUDIT/'delivery-manifest.json',ROOT/'data/metadata/external-videos/catalog.csv']}
    catalog={r['source_id']:r for r in rows(ROOT/'data/metadata/external-videos/catalog.csv')}
    for source in selected:
        uid=source['source_id'];proof=AUDIT/(uid+'.reconciled.json')
        if not proof.exists():proof=AUDIT/(uid+'.json')
        require(sha(proof)==known[rel(proof)],'External audited record SHA differs')
        r=read(proof);frame_path=ROOT/r['frame_csv'];require(sha(frame_path)==r['output_sha256'][rel(frame_path)],'External full-frame QC identity differs')
        require(r['technical_status'] in ('completed','reused') and r['source_sha256_before']==r['source_sha256_after']==source['media_sha256_inherited']==catalog[uid]['sha256'],'External source identity does not close')
        frame_rows=rows(frame_path);a=int(source['proposed_start_frame']);b=int(source['proposed_end_frame_exclusive']);require(0<=a<b<=len(frame_rows),'External proposed interval invalid')
        pts=np.array([float(v['pts_seconds']) for v in frame_rows]);require(np.isfinite(pts).all() and np.all(np.diff(pts)>0),'External PTS not strict/finite')
        targets=pts[a]+np.array([.25,.5,.75])*(pts[b-1]-pts[a]);indices=[int(a+np.argmin(abs(pts[a:b]-t))) for t in targets]
        require(len(set(indices))==3,'External fixed quantiles repeated a source frame')
        external.append(dict(source_id=uid,media_path=source['media_path'],source_sha256=source['media_sha256_inherited'],size_bytes=r['size_bytes'],interval=[a,b],indices=indices,target_pts_seconds=targets.tolist(),actual_pts_seconds=[float(pts[i]) for i in indices],native_dimensions=r['native_dimensions']))
        mapping.append(dict(population='external',unit_id=uid,publisher_or_source='USGS' if uid.startswith('USGS') else ('USU/Zenodo' if uid.startswith('USU') else 'NERC/EIDC'),site_or_group=source['source_site_group'],event_or_video=source['original_event_id'],media_path=source['media_path'],declared_media_sha256=source['media_sha256_inherited']))
        inherited[rel(proof)]=sha(proof);inherited[rel(frame_path)]=sha(frame_path)
    require(len(development)==450 and len(external)==18,'Fixed sample counts differ')
    contract=dict(status='frozen_before_JPEG_reads_or_external_preview_decode',created_unix=time.time(),deadline_unix=time.time()+720,
        command=sys.argv,code_path=rel(__file__),code_sha256=sha(__file__),ffmpeg_sha256=sha(FFMPEG),opencv_version=cv2.__version__,numpy_version=np.__version__,
        input_sha256=inherited,development_samples=development,external_samples=external,
        sampling_rule='development accepted frame_index first/floor(n/2)/last; external nearest PTS at25/50/75percent of unchanged proposed first-to-last PTS; first index on tie; no replacement',
        phash_version='opencv-reduced-gray4-fallback-gray-area32-float32-DCT8x8-excludeDC-medianGT-pack63-v1',
        phash_basis_path='experiments/tools/cluster_by_perceptual_hash.py',phash_basis_sha256=sha(ROOT/'experiments/tools/cluster_by_perceptual_hash.py'),
        bit_scope='63 non-DC bits stored in16-hex-character uint64-style notation; old repository implementation uses63 coefficients despite64-bit storage',
        hamming_alert_threshold=10,comparisons=54*450,media_SHA_scope='compare declared256 middleware SHA with published external media SHA; inherit actual prior QC, no full media rehash in this screen',
        limitations='3 sparse samples per video; crops, framing, overlays, edits, time gaps, unpublished common origins and other unsampled moments can evade pHash. No model scores, class/selection changes or independent-test proof.')
    table(META/'provenance-map.csv',mapping);write(META/'contract.json',contract)
    (META/'generator-source.py').write_bytes(Path(__file__).read_bytes())
    return contract


def execute(contract):
    began=time.monotonic();end=contract['deadline_unix'];actual=[];external=[]
    def alive():require(time.time()<end,'Bounded720s screen deadline exhausted')
    try:
        require(sha(__file__)==contract['code_sha256'],'Frozen generator changed')
        for name,digest in contract['input_sha256'].items():require(sha(ROOT/name)==digest,'Input SHA changed: '+name)
        for r in contract['development_samples']:
            alive();source=ROOT/r['cache_path'];data=source.read_bytes();digest=hashlib.sha256(data).hexdigest()
            require(len(data)==int(r['size_bytes']) and digest==r['actual_jpeg_sha256'],'Actual accepted JPEG byte identity differs')
            folder=OUT/'development'/r['unit_id'];folder.mkdir(parents=True,exist_ok=True);copy=folder/('frame-%08d.jpg'%int(r['frame_index']))
            with copy.open('xb') as f:f.write(data)
            actual.append(dict(population='development',unit_id=r['unit_id'],frame_index=int(r['frame_index']),source_pts_seconds=float(r['source_pts_seconds']),source_path=r['cache_path'],sample_path=rel(copy),byte_sha256=digest,phash='%016x'%phash_bytes(data)))
        table(META/'development-samples.csv',actual)
        for r in contract['external_samples']:
            alive();source=ROOT/r['media_path'];before=source.stat();require(before.st_size==r['size_bytes'],'External source size changed since full audited QC')
            folder=OUT/'external'/r['source_id'];folder.mkdir(parents=True,exist_ok=False)
            expression='+'.join('eq(n,%d)'%i for i in r['indices'])
            cmd=[str(FFMPEG),'-hide_banner','-nostdin','-n','-noautorotate','-copyts','-threads','2','-i',str(source),'-map','0:v:0','-filter_threads','1','-vf',"select='"+expression+"'",'-an','-fps_mode','passthrough','-c:v','png','-threads','1',str(folder/'native-%02d.png')]
            write(folder/'command.json',cmd)
            with (folder/'decode.log').open('x',encoding='utf-8') as log:
                p=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=log);write(folder/'process.json',dict(pid=p.pid,command=cmd))
                try:code=p.wait(timeout=max(.1,end-time.time()))
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10);raise TimeoutError('External extraction deadline; partial evidence retained')
            after=source.stat();require(code==0 and (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'External extraction failed or source changed')
            pngs=sorted(folder.glob('native-*.png'));require(len(pngs)==3,'Missing fixed external source frame')
            for path,index,pts in zip(pngs,r['indices'],r['actual_pts_seconds']):
                data=path.read_bytes();img=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                require(img is not None and [img.shape[1],img.shape[0]]==r['native_dimensions'],'External native dimensions differ')
                external.append(dict(population='external',unit_id=r['source_id'],frame_index=index,source_pts_seconds=pts,source_path=r['media_path'],inherited_source_sha256=r['source_sha256'],sample_path=rel(path),byte_sha256=hashlib.sha256(data).hexdigest(),phash='%016x'%phash_bytes(data)))
            print(json.dumps(dict(stage='external_three_native_frames_complete',source_id=r['source_id'])),flush=True)
        table(META/'external-samples.csv',external);require(len(external)==54,'Expected54 external samples')
        distances=[];nearest=[];alerts=[]
        for ext in external:
            candidates=[]
            for dev in actual:
                d=distance(int(ext['phash'],16),int(dev['phash'],16));row=dict(external_id=ext['unit_id'],external_frame_index=ext['frame_index'],development_id=dev['unit_id'],development_frame_index=dev['frame_index'],hamming=d)
                distances.append(row);candidates.append(row)
                if d<=10:alerts.append(row)
            nearest.append(min(candidates,key=lambda row:(row['hamming'],row['development_id'],row['development_frame_index'])))
        table(META/'all-distances.csv',distances);table(META/'nearest-per-external-frame.csv',nearest)
        if alerts:table(META/'alerts.csv',alerts);print(json.dumps(dict(stage='manual_review_alerts_found',pairs=len(alerts),no_extra_sampling=True)),flush=True)
        mapping=rows(META/'provenance-map.csv');same=[dict(development_id=d['unit_id'],external_id=e['unit_id'],declared_sha256=d['declared_media_sha256']) for d in mapping if d['population']=='development' for e in mapping if e['population']=='external' and d['declared_media_sha256']==e['declared_media_sha256']]
        by_video=[min([v for v in nearest if v['external_id']==r['source_id']],key=lambda v:(v['hamming'],v['development_id'],v['development_frame_index'])) for r in contract['external_samples']]
        table(META/'nearest-per-external-video.csv',by_video)
        write(META/'result.json',dict(status='finite_screen_complete_manual_alert_review_pending' if alerts else 'finite_screen_complete_no_threshold_alert_in_sample',development_videos=150,external_videos=18,development_images=450,external_images=54,comparisons=len(distances),hamming_threshold=10,alerts=alerts,declared_media_sha_matches=same,minimum_sample_hamming=min(v['hamming'] for v in nearest),seconds=time.monotonic()-began,source_media_full_rehash_performed=False,contract_sha256=sha(META/'contract.json'),formal_test_frozen=False,model_scores_read=False,labels_or_selection_changed=False,limitation=contract['limitations']))
    except Exception as e:
        write(META/'failure.json',dict(status='incomplete_no_overlap_claim',error_type=type(e).__name__,error=str(e),seconds=time.monotonic()-began));raise


class Fixtures(unittest.TestCase):
    def test_same_image_zero_alert_corrupt_refused_unique_indices(self):
        image=np.arange(96*96,dtype=np.uint8).reshape(96,96);ok,png=cv2.imencode('.png',image);require(ok,'fixture encoding')
        h=phash_bytes(png.tobytes());self.assertEqual(distance(h,h),0);self.assertLessEqual(distance(h,h),10)
        with self.assertRaises(ValueError):phash_bytes(b'corrupt-image')
        sample=first_middle_last([dict(frame_index=i) for i in (8,2,5,1)])
        self.assertEqual([int(r['frame_index']) for r in sample],[1,5,8])
        with self.assertRaises(ValueError):first_middle_last([dict(frame_index=i) for i in (1,1,2)])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:
        if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Fixtures)).wasSuccessful():raise SystemExit(1)
    else:execute(freeze())
