"""Stream a verified file manifest to a dedicated AutoDL workspace over SSH.

Known-host verification is mandatory. No local media archive is created; the
receiver checks every incoming size/SHA before atomically committing a file.
Existing files are skipped ONLY after remote SHA verification.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import sys
import tarfile
import threading
import time
import uuid

import paramiko

ROOT = Path(__file__).resolve().parents[2]

def local_source(root, name):
    relative = PurePosixPath(name)
    if '\\' in name or relative.is_absolute() or '..' in relative.parts or ':' in name:
        raise ValueError('Unsafe manifest path')
    path = (Path(root) / relative).resolve()
    path.relative_to(Path(root).resolve())
    return path


PREPARE = r'''
import json,sys
from pathlib import Path
base=Path(sys.argv[1])
if not base.is_absolute() or base.parent != Path('/root/autodl-tmp') or not base.name.startswith('debris-flow-thesis-exp016'):
    raise ValueError('Refuse workspace outside dedicated task directory')
if base.is_symlink():raise ValueError('Workspace itself may not be a symlink')
resolved=base.resolve();resolved.relative_to(Path('/root/autodl-tmp').resolve())
marker=base/'.codex/managed-upload.json'
marker.parent.resolve().relative_to(resolved)
if marker.is_symlink():raise ValueError('Managed marker may not be a symlink')
if base.exists() and any(base.iterdir()) and not marker.is_file():
    raise ValueError('Existing directory is not managed by this task')
marker.parent.mkdir(parents=True,exist_ok=True)
if marker.exists():
    assert json.loads(marker.read_text())['task']=='EXP-016'
else:marker.write_text(json.dumps({'task':'EXP-016','purpose':'verified training upload'}))
transfers=(base/'.codex/transfers').resolve();transfers.relative_to(resolved)
transfers.mkdir(exist_ok=True)
print(json.dumps({'workspace':str(base),'ready':True}))
'''

AUDIT = r'''
import csv,hashlib,json,sys
from pathlib import Path,PurePosixPath
base=Path(sys.argv[1]); manifest=Path(sys.argv[2]); missing_path=Path(sys.argv[3])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
missing=[]; skipped=0; parents={}
with manifest.open(encoding='utf-8-sig',newline='') as f:
    for row in csv.DictReader(f):
        rel=PurePosixPath(row['path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe manifest member')
        parent=str(rel.parent)
        if parent not in parents:
            parents[parent]=(base/rel.parent).resolve()
            parents[parent].relative_to(base.resolve())
        target=parents[parent]/rel.name
        if not target.is_symlink() and target.is_file() and target.stat().st_size==int(row['size_bytes']) and sha(target)==row['sha256']:skipped+=1
        else:missing.append(row['path'])
missing_path.write_text(json.dumps(missing))
print(json.dumps({'missing':len(missing),'skipped_sha_verified':skipped,'paths':missing}))
'''

RECEIVE = r'''
import csv,hashlib,json,os,sys,tarfile,time,tempfile
from pathlib import Path,PurePosixPath
base=Path(sys.argv[1]).resolve(); manifest=Path(sys.argv[2]); missing_path=Path(sys.argv[3]); limit=int(sys.argv[4])
with manifest.open(encoding='utf-8-sig',newline='') as f:records={r['path']:r for r in csv.DictReader(f)}
names=json.loads(missing_path.read_text()); names=names[:limit] if limit else names
expected=set(names); seen=set(); received=0; began=time.monotonic(); last=began
progress=missing_path.with_suffix('.progress.json'); parents={}
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|',bufsize=1024**2) as archive:
    for member in archive:
        name=member.name; rel=PurePosixPath(name)
        if name not in expected or name in seen or not member.isfile() or rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unexpected/duplicate/nonregular archive entry')
        record=records[name]
        if member.size!=int(record['size_bytes']):raise ValueError('Wrong declared file size')
        parent=str(rel.parent)
        if parent not in parents:
            directory=(base/rel.parent).resolve();directory.relative_to(base)
            directory.mkdir(parents=True,exist_ok=True);parents[parent]=directory
        target=parents[parent]/rel.name
        source=archive.extractfile(member); h=hashlib.sha256(); size=0
        with tempfile.NamedTemporaryFile(mode='wb',prefix='.exp016-upload-',dir=target.parent,delete=False) as out:
            temporary=Path(out.name)
            while True:
                chunk=source.read(1024**2)
                if not chunk:break
                h.update(chunk);out.write(chunk);size+=len(chunk)
        if size!=int(record['size_bytes']) or h.hexdigest()!=record['sha256']:
            raise ValueError('Incoming file checksum mismatch: '+name)
        os.replace(temporary,target)
        seen.add(name);received+=size
        now=time.monotonic()
        if now-last>=5:
            progress.write_text(json.dumps({'files':len(seen),'bytes':received,'seconds':now-began}))
            last=now
if seen!=expected:raise ValueError('Incomplete upload batch')
result={'status':'batch_verified','files':len(seen),'bytes':received,'seconds':time.monotonic()-began,
        'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
progress.write_text(json.dumps(result));print(json.dumps(result))
'''


def command(script, *args):
    return ' '.join(shlex.quote(str(x)) for x in ('/root/miniconda3/bin/python', '-c', script, *args))


def run_checked(ssh, script, *args):
    _, stdout, stderr = ssh.exec_command(command(script, *args), timeout=120)
    result = stdout.read().decode('utf-8')
    error = stderr.read().decode('utf-8', errors='replace')
    if stdout.channel.recv_exit_status():
        raise RuntimeError(error)
    return json.loads(result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--host', required=True)
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--key', type=Path, required=True)
    ap.add_argument('--remote-root', default='/root/autodl-tmp/debris-flow-thesis-exp016')
    ap.add_argument('--max-files', type=int, default=0)
    a = ap.parse_args()
    if a.max_files < 0:
        ap.error('--max-files must be nonnegative')
    with a.manifest.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    by_name = {r['path']: r for r in rows}
    if len(by_name) != len(rows):
        raise ValueError('Duplicate manifest path')
    for name in by_name:
        p = PurePosixPath(name)
        if '\\' in name or p.is_absolute() or '..' in p.parts or ':' in name:
            raise ValueError('Unsafe manifest path')
    token = uuid.uuid4().hex[:12]
    log_path = ROOT / f'.codex/logs/exp016-upload-{token}.jsonl'
    log = log_path.open('x', encoding='utf-8')

    def report(value):
        log.write(json.dumps(value) + '\n'); log.flush()
        print(json.dumps(value), flush=True)
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys(str(Path.home() / '.ssh/known_hosts'))
    # Default RejectPolicy: never add/accept an unknown or changed host key.
    ssh.connect(a.host, port=a.port, username='root', key_filename=str(a.key),
                allow_agent=False, look_for_keys=False, timeout=20)
    ssh.get_transport().set_keepalive(30)
    report(run_checked(ssh, PREPARE, a.remote_root))
    sftp = ssh.open_sftp()
    remote_manifest = f'{a.remote_root}/.codex/transfers/{token}-manifest.csv'
    remote_missing = f'{a.remote_root}/.codex/transfers/{token}-missing.json'
    sftp.put(str(a.manifest), remote_manifest)
    audit = run_checked(ssh, AUDIT, a.remote_root, remote_manifest, remote_missing)
    names = audit.pop('paths')
    report(audit)
    names = names[:a.max_files] if a.max_files else names
    if not names:
        report({'status': 'all_files_already_verified'}); ssh.close(); log.close(); return
    channel = ssh.get_transport().open_session()
    channel.exec_command(command(RECEIVE, a.remote_root, remote_manifest, remote_missing, a.max_files))
    writer = channel.makefile_stdin('wb')
    errors = []

    def stderr_reader():
        errors.append(channel.makefile_stderr('rb').read().decode('utf-8', errors='replace'))
    thread = threading.Thread(target=stderr_reader, daemon=True); thread.start()
    began = last = time.monotonic(); sent = 0
    report({'status': 'streaming', 'files': len(names), 'bytes': sum(int(by_name[n]['size_bytes']) for n in names),
            'log': str(log_path)})
    try:
        with tarfile.open(fileobj=writer, mode='w|', format=tarfile.PAX_FORMAT,
                          bufsize=1024**2, copybufsize=1024**2) as archive:
            for count, name in enumerate(names, 1):
                record = by_name[name]
                info = tarfile.TarInfo(name); info.size = int(record['size_bytes']); info.mode = 0o644
                with local_source(ROOT, name).open('rb') as source:
                    if os.fstat(source.fileno()).st_size != info.size:
                        raise ValueError('Local source size changed: ' + name)
                    archive.addfile(info, source)
                sent += info.size
                now = time.monotonic()
                if now - last >= 10:
                    report({'files_sent': count, 'files_total': len(names), 'bytes_sent': sent,
                            'seconds': now - began, 'MiB_per_second': sent / (now - began) / 1024**2})
                    last = now
        writer.flush(); channel.shutdown_write()
        result = channel.makefile('rb').read().decode('utf-8')
        status = channel.recv_exit_status(); thread.join()
        if status:
            raise RuntimeError('Remote upload failed: ' + ''.join(errors))
        report(json.loads(result))
    finally:
        writer.close(); sftp.close(); ssh.close(); log.close()


if __name__ == '__main__':
    main()
