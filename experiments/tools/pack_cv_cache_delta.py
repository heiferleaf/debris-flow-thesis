"""Byte-preserving uncompressed tar of the accepted EXP-017 missing-frame delta.

Transport only: never an extra training-input manifest entry. Extract into the
repository root remotely, then run the normal frozen files.csv verification.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import sys
import tarfile
import tempfile
import time
import unittest

ROOT=Path(__file__).resolve().parents[2]
PLAN_SHA='70394126cd15d015c4250bbea8f1c9a0c928141d92f00a03d66279a576c2c9b8'
OLD_INDEX='data/process/EXP-016/metadata-v4/verified-cache-files.csv'
OLD_INDEX_SHA='7e58186d125c67b52c54f6e926c3cedcb162c30ed82772da671f65160d7b018e'


def require(value,message):
    if not value:raise ValueError(message)


def sha_stream(stream):
    h=hashlib.sha256()
    for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def sha(path):
    with Path(path).open('rb') as f:return sha_stream(f)


def safe_path(root,name):
    require(isinstance(name,str) and '\\' not in name,'Only POSIX repository-relative member paths are allowed')
    p=PurePosixPath(name);w=PureWindowsPath(name)
    require(bool(p.parts) and not p.is_absolute() and not w.drive and not w.is_absolute()
            and '..' not in p.parts and p.as_posix()==name,'Unsafe tar member path')
    path=Path(root).joinpath(*p.parts)
    require(not path.is_symlink(),'Symlink members are refused')
    path.resolve().relative_to(Path(root).resolve())
    return path


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);result=list(reader)
    require(reader.fieldnames and len(reader.fieldnames)==len(set(reader.fieldnames))
            and all(None not in r and all(v is not None for v in r.values()) for r in result),'Malformed index CSV')
    return result


def keyed(records):
    result={}
    for row in records:
        key=row['unit_id'],int(row['frame_index'])
        require(key[1]>=0 and key not in result,'Invalid/duplicate physical frame identity')
        result[key]=row
    return result


def delta_rows(merged,old,expected_count=41848):
    merged,old=keyed(merged),keyed(old)
    for k in set(merged)&set(old):
        require(all(merged[k][f]==old[k][f] for f in ('cache_path','jpeg_sha256','rgb112_sha256','size_bytes')),
                'An original indexed JPEG identity was replaced')
    selected=[dict(unit_id=k[0],frame_index=k[1],path=merged[k]['cache_path'],
                   size_bytes=int(merged[k]['size_bytes']),sha256=merged[k]['jpeg_sha256']) for k in sorted(set(merged)-set(old))]
    require(len(selected)==expected_count and len({r['path'] for r in selected})==len(selected),'Unexpected/duplicate delta members')
    require(all(r['size_bytes']>0 and re.fullmatch('[0-9a-f]{64}',r['sha256']) for r in selected),'Invalid delta size/SHA')
    return selected


def write_tar(root,members,path):
    with Path(path).open('xb') as output,tarfile.open(fileobj=output,mode='w',format=tarfile.USTAR_FORMAT) as archive:
        for i,row in enumerate(members,1):
            source=safe_path(root,row['path']);before=source.stat()
            require(source.is_file() and before.st_size==row['size_bytes'],'Source JPEG size changed')
            info=tarfile.TarInfo(row['path']);info.size=row['size_bytes'];info.mode=0o644;info.mtime=0
            with source.open('rb') as f:archive.addfile(info,f)
            after=source.stat()
            require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Source changed during packing')
            if i%5000==0:print(json.dumps(dict(stage='tar_write',members=i,total=len(members))),flush=True)


def verify_tar(root,members,path):
    count=0;total=0
    with tarfile.open(path,mode='r|') as archive:
        for member in archive:
            require(count<len(members),'Extra tar member')
            row=members[count];safe_path(root,member.name)
            require(member.isfile() and member.name==row['path'] and member.size==row['size_bytes'],'Unexpected tar member type/order/size')
            with archive.extractfile(member) as f:require(sha_stream(f)==row['sha256'],'Packed member byte SHA differs: '+member.name)
            count+=1;total+=member.size
            if count%5000==0:print(json.dumps(dict(stage='tar_readback_SHA',members=count,total=len(members))),flush=True)
    require(count==len(members),'Incomplete tar member coverage')
    return total


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def pack(root,acceptance_path,out):
    root,acceptance_path,out=Path(root).resolve(),Path(acceptance_path).resolve(),Path(out).resolve()
    out.relative_to(root/'data/process/EXP-017')
    if out.exists():raise FileExistsError('Refuse existing output: '+str(out))
    out.mkdir(parents=True,exist_ok=False);began=time.perf_counter();source=Path(__file__).resolve();code_sha=sha(source)
    shutil.copyfile(source,out/'generator-source.py')
    inputs={};partial=out/'cache-delta.tar.partial';final=out/'cache-delta.tar'
    def bind(path,expected=None):
        path=Path(path).resolve();name=path.relative_to(root).as_posix();value=sha(path)
        require(expected is None or value==expected,'Input SHA differs: '+name)
        inputs[name]=value;return path
    write_json(out/'launch.json',dict(argv=sys.argv,python_executable=sys.executable,python_version=sys.version,
        source_sha256=code_sha,compression='none; USTAR tar; member JPEG bytes unchanged'))
    try:
        proof=json.loads(bind(acceptance_path).read_text('utf-8'))
        require(proof['status']=='cv_supervised_cache_ready' and proof['spec_id']=='EXP-017' and proof['plan_sha256']==PLAN_SHA
                and proof['required_frame_count']==proof['verified_required_frame_count']==proof['actual_verified_JPEG_reads']==191812
                and proof['failed_required_frames']==proof['unverified_required_frames']==0
                and proof['actual_JPEG_byte_SHA_checked'] is True and proof['actual_RGB112_pixel_SHA_checked'] is True,
                'Complete independently accepted191812-frame cache required')
        bind(safe_path(root,proof['plan_path'])/'plan.json',PLAN_SHA)
        producer_path=bind(safe_path(root,proof['producer_result']),proof['producer_result_sha256'])
        produced=json.loads(producer_path.read_text('utf-8'))
        require(produced['status']=='cv_supervised_JPEGs_produced_pixels_acceptance_pending' and produced['plan_sha256']==PLAN_SHA
                and produced['new_frame_count']==41848 and produced['reused_frame_count']==149964 and produced['required_frame_count']==191812
                and produced['verified_cache_index']==proof['verified_cache_index']
                and produced['verified_cache_index_sha256']==proof['verified_cache_index_sha256'],'Producer/acceptance identity differs')
        new_index=bind(safe_path(root,proof['verified_cache_index']),proof['verified_cache_index_sha256'])
        old_index=bind(root/OLD_INDEX,OLD_INDEX_SHA)
        bind(safe_path(root,proof['verified_required_frames_index']),proof['verified_required_frames_index_sha256'])
        merged=rows(new_index);require(len(merged)==191812,'Unexpected merged index count')
        members=delta_rows(merged,rows(old_index))
        for row in members:
            path=safe_path(root,row['path'])
            expected=producer_path.parent/'frames'/row['unit_id']/('frame-%08d.jpg'%row['frame_index'])
            require(path==expected,'Delta JPEG is outside the new producer frame directory')
        with (out/'members.csv').open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['unit_id','frame_index','path','size_bytes','sha256']);writer.writeheader();writer.writerows(members)
        write_tar(root,members,partial);total=verify_tar(root,members,partial)
        for name,expected in inputs.items():require(sha(root/name)==expected,'Input changed during packing: '+name)
        require(sha(source)==code_sha and sha(out/'generator-source.py')==code_sha,'Packer source changed')
        archive_sha=sha(partial);partial.rename(final)
        result=dict(status='transport_cache_delta_ready',member_count=len(members),original_member_bytes=total,
            archive_path=final.relative_to(root).as_posix(),archive_bytes=final.stat().st_size,archive_sha256=archive_sha,
            members_csv_sha256=sha(out/'members.csv'),source_sha256=code_sha,input_sha256=inputs,seconds=time.perf_counter()-began,
            every_member_actual_byte_SHA_checked=True,compression='none',jpeg_reencoded=False,
            old_cache_files_copied=0,source_videos_packed=0,no_training=True,
            remote_requirement='Extract at repository root, then verify all frozen files.csv entries; tar is transport only, not an extra training input/manifest row.')
        write_json(out/'manifest.json',result);print(json.dumps(result),flush=True);return result
    except Exception as error:
        write_json(out/'failure.json',dict(status='transport_not_ready',error_type=type(error).__name__,error=str(error),
            input_sha256=inputs,source_sha256=code_sha,seconds=time.perf_counter()-began,partial_preserved=True))
        raise


class PackFixtures(unittest.TestCase):
    def test_paths(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for bad in ('/absolute.jpg','../escape.jpg','data/../escape.jpg','C:/escape.jpg','C:relative.jpg','data\\file.jpg','data/./x.jpg'):
                with self.assertRaises(ValueError):safe_path(root,bad)
            self.assertEqual(safe_path(root,'data/new.jpg'),root/'data/new.jpg')

    def test_delta_success_and_wrong_packed_sha(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'data').mkdir();p=root/'data/new.jpg';p.write_bytes(b'unchanged-JPEG-fixture-bytes')
            old=dict(unit_id='u',frame_index='0',cache_path='data/old.jpg',size_bytes='1',jpeg_sha256='a'*64,rgb112_sha256='b'*64)
            new=dict(old,frame_index='1',cache_path='data/new.jpg',size_bytes=str(p.stat().st_size),jpeg_sha256=sha(p))
            members=delta_rows([old,new],[old],1);tar=root/'delta.tar'
            write_tar(root,members,tar);self.assertEqual(verify_tar(root,members,tar),p.stat().st_size)
            with self.assertRaises(FileExistsError):write_tar(root,members,tar)
            broken=[dict(members[0],sha256='0'*64)]
            with self.assertRaises(ValueError):verify_tar(root,broken,tar)
            with self.assertRaises(ValueError):delta_rows([old,new,new],[old],1)
            self.assertFalse((root/'data/old.jpg').exists());self.assertEqual(p.read_bytes(),b'unchanged-JPEG-fixture-bytes')

    def test_failed_pack_never_marks_ready(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);proof=root/'bad-acceptance.json';proof.write_text('{"status":"failed"}',encoding='utf-8')
            out=root/'data/process/EXP-017/transfer-fixture'
            with self.assertRaises((ValueError,KeyError)):pack(root,proof,out)
            self.assertEqual(json.loads((out/'failure.json').read_text('utf-8'))['status'],'transport_not_ready')
            self.assertFalse((out/'manifest.json').exists());self.assertFalse((out/'cache-delta.tar').exists())
            with self.assertRaises(FileExistsError):pack(root,proof,out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance',type=Path,default=ROOT/'data/metadata/EXP-017/cv-cache-acceptance-v1/acceptance.json')
    parser.add_argument('--out',type=Path,default=ROOT/'data/process/EXP-017/transfer-cache-v1')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PackFixtures)).wasSuccessful() else 1
    pack(ROOT,args.acceptance,args.out);return 0


if __name__=='__main__':sys.exit(main())
