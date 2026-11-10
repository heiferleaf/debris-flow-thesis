"""Freeze EXP-019 executable sources and the small changed transfer closure.

Existing JPEG caches and dense media stay in their original locations. They are
not copied into this deployment archive. Runtime checks still verify actual
inputs; a deployment manifest does not replace pixel acceptance.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.supervised_plan import digest, rooted, write_json


def freeze_deployment(root, directory, plan_dir, acceptance_path):
    root, directory, plan_dir, acceptance_path = (
        Path(p).resolve() for p in (root, directory, plan_dir, acceptance_path))
    directory.relative_to(root / 'data/metadata/EXP-019')
    acceptance_root = root / 'data/metadata/EXP-019'
    if (acceptance_path.parent.parent != acceptance_root
            or not acceptance_path.parent.name.startswith('order-cache-acceptance-')):
        raise ValueError('Cache proof must be inside its dedicated EXP-019 acceptance directory')
    if directory.exists():
        raise FileExistsError(directory)
    from experiments.common.video.order_plan import load_order_plan
    from experiments.common.video.order_training import verify_cache_acceptance
    plan, bank, _, _, _, _ = load_order_plan(root, plan_dir)
    verify_cache_acceptance(root, plan_dir, plan, bank, acceptance_path)
    pipeline = runpy.run_path(str(root / 'experiments/runspecs/EXP-019/pipeline.py'))
    paths = {Path(p).resolve(): 'execution_source' for p in pipeline['code_paths']()}
    paths[Path(__file__).resolve()] = 'execution_source'
    # Plan-owned snapshots include the historical evidence; no live historical
    # governance file is substituted for those immutable archived bytes.
    for p in plan_dir.rglob('*'):
        if p.is_file():
            paths[p.resolve()] = 'order_plan_artifact'
    for p in acceptance_path.parent.rglob('*'):
        if p.is_file():
            paths[p.resolve()] = 'cache_acceptance_artifact'
    # Runtime data are already installed on the reused instance. Include their
    # small metadata here; the 31 dense media remain identity-checked in place.
    for ref in plan['data_fingerprints']:
        p = rooted(root, ref['path'])
        if p.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv', '.jpg', '.jpeg', '.png'):
            continue
        paths.setdefault(p.resolve(), 'existing_runtime_metadata')
    rows = []
    directory.mkdir(parents=True, exist_ok=False)
    try:
        for p, kind in sorted(paths.items(), key=lambda item: str(item[0])):
            relative = p.relative_to(root).as_posix()
            if p.is_symlink() or not p.is_file():
                raise ValueError('Deployment entry must be a regular project file: ' + relative)
            value = digest(p)
            row = dict(path=relative, size_bytes=p.stat().st_size, sha256=value, kind=kind)
            # Archive executable sources, not another copy of large metadata.
            if kind == 'execution_source':
                target = directory / 'source-snapshot' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as f:
                    f.write(p.read_bytes())
                if digest(target) != value:
                    raise ValueError('Executable changed while snapshotting: ' + relative)
            rows.append(row)
        with (directory / 'files.csv').open('x', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['path', 'size_bytes', 'sha256', 'kind'])
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            p = rooted(root, row['path'])
            if p.stat().st_size != row['size_bytes'] or digest(p) != row['sha256']:
                raise ValueError('Deployment closure changed while freezing: ' + row['path'])
        result = dict(status='execution_and_transfer_closure_frozen', spec_id='EXP-019',
            files=len(rows), total_bytes=sum(r['size_bytes'] for r in rows),
            manifest_sha256=digest(directory / 'files.csv'),
            order_plan_sha256=digest(plan_dir / 'plan.json'),
            cache_acceptance_sha256=digest(acceptance_path),
            sources=rows, pixels_copied=False, media_copied=False,
            authorization='2026-11-01 goal; total CNY20; EXP019 stage <= CNY4',
            note='Reuse existing media/JPEG paths; runtime verification remains mandatory.')
        write_json(directory / 'snapshot-index.json', result)
        return result
    except Exception as error:
        write_json(directory / 'failure.json', dict(status='deployment_freeze_failed',
            error_type=type(error).__name__, error=str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--cache-acceptance', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'data/metadata/EXP-019/deployment-v1')
    args = parser.parse_args()
    result = freeze_deployment(ROOT, args.out, args.plan, args.cache_acceptance)
    print(json.dumps({k: result[k] for k in ('status', 'files', 'total_bytes', 'manifest_sha256')}))


if __name__ == '__main__':
    main()
