"""Make a hash-addressed upload manifest, without copying images or videos."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.supervised_plan import csv_rows, digest, jsonl, verify_plan, write_json


def prepare(destination):
    destination = Path(destination).resolve()
    destination.relative_to(ROOT / 'data/metadata/EXP-016')
    if destination.exists():
        raise FileExistsError(destination)
    plan_dir = ROOT / 'data/metadata/EXP-016/training-plan-v2'
    plan = verify_plan(ROOT, plan_dir)
    resource = ROOT / plan['resource']
    ids = {r['clip_id'] for r in csv_rows(plan_dir / 'assignments.csv')}
    required = {(r['unit_id'], i) for r in jsonl(resource / 'windows.jsonl')
                if r['clip_id'] in ids for i in r['frame_indices']}
    files = {}

    def add(relative, size, sha, role):
        p = Path(relative)
        if p.is_absolute() or '..' in p.parts or ':' in str(p):
            raise ValueError('Unsafe project-relative transfer path')
        key = p.as_posix()
        record = dict(path=key, size_bytes=int(size), sha256=sha, role=role)
        if key in files and (files[key]['sha256'], files[key]['size_bytes']) != (sha, int(size)):
            raise ValueError('Conflicting duplicate transfer record')
        files[key] = record

    index_path = next(ROOT / r['path'] for r in plan['sources'] if r['path'].endswith('verified-cache-files.csv'))
    found = set()
    with index_path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            key = (row['unit_id'], int(row['frame_index']))
            if key in required:
                add(row['cache_path'], row['size_bytes'], row['jpeg_sha256'], 'supervised_jpeg')
                found.add(key)
    if found != required:
        raise ValueError('Upload selection missing verified frames')
    for r in csv_rows(resource / 'dense-transfer-references.csv'):
        add(r['local_path'], r['size_bytes'], r['sha256'], r['role'])
    small_files = {ROOT / r['path'] for r in plan['sources']}
    small_files.update(plan_dir.iterdir())
    small_files.update((ROOT / 'experiments/runspecs/EXP-016').glob('*.py'))
    small_files.update((ROOT / 'experiments/runspecs/EXP-016').glob('*.sh'))
    small_files.update(ROOT / 'experiments/common/video' / n for n in
                       ('supervised_plan.py', 'supervised_runtime.py', 'supervised_training.py',
                        'r2plus1d_classifier.py', 'dense_evaluation.py', 'pts_clip_sampling.py'))
    small_files.update(ROOT / n for n in ('AGENTS.md', 'experiments/runspecs/EXP-016/README.md',
                       'docs/process/experiments/EXP-016.md',
                       'docs/process/experiments/EXP-016-seed8.md',
                       'docs/process/experiments/EXP-006-数据使用方案.md',
                       'data/process/EXP-016/dense-pixel-verification-v3/result.json',
                       'data/process/EXP-016/dense-pixel-verification-v3/source-plan.json'))
    for p in small_files:
        if p.is_file():
            add(p.relative_to(ROOT), p.stat().st_size, digest(p), 'code_or_metadata')
    destination.mkdir(parents=True, exist_ok=False)
    manifest = destination / 'files.csv'
    with manifest.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'size_bytes', 'sha256', 'role'])
        writer.writeheader()
        writer.writerows(files[k] for k in sorted(files))
    counts, sizes = Counter(), Counter()
    for row in files.values():
        counts[row['role']] += 1
        sizes[row['role']] += row['size_bytes']
    result = dict(status='upload_manifest_prepared_not_uploaded', files=len(files),
                  bytes_total=sum(sizes.values()), counts=dict(counts), bytes_by_role=dict(sizes),
                  plan_sha256=digest(plan_dir / 'plan.json'), manifest_sha256=digest(manifest),
                  generator_sha256=digest(__file__), command=sys.argv,
                  source_of_media_hashes='completed verified cache index and frozen dense source references',
                  remote_rule='skip only when remote SHA256 matches; stream selected files without local archive',
                  pretrained_weights='reuse verified KINETICS400_V1 cache from existing image; inspect after no-GPU boot')
    write_json(destination / 'summary.json', result)
    return result


if __name__ == '__main__':
    destination = ROOT / 'data/metadata/EXP-016/transfer-v1'
    if len(sys.argv) > 1:
        destination = ROOT / sys.argv[1]
    print(json.dumps(prepare(destination), ensure_ascii=False))
