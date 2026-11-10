"""Snapshot current executable sources; transfer only changed small artifacts."""
import csv
import json
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.common.video.supervised_plan import digest, write_json


def main():
    directory=ROOT/'data/metadata/EXP-018/deployment-v1'
    directory.mkdir(parents=True,exist_ok=False)
    controller=runpy.run_path(str(ROOT/'experiments/runspecs/EXP-018/run.py'))
    paths=list(controller['code_paths']())
    paths.extend([ROOT/'data/metadata/EXP-018/paired-plan-v1'/name for name in ('plan.json','epochs.jsonl')])
    paths.extend([Path(__file__).resolve(),ROOT/'experiments/runspecs/EXP-018/audit.py'])
    rows=[]
    for p in sorted(set(paths)):
        relative=p.relative_to(ROOT).as_posix()
        target=directory/'source-snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        encoded=p.read_bytes()
        target.write_bytes(encoded)
        if digest(target)!=digest(p):raise ValueError('Snapshot bytes differ')
        rows.append(dict(path=relative,size_bytes=len(encoded),sha256=digest(p)))
    with (directory/'files.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','size_bytes','sha256']);writer.writeheader();writer.writerows(rows)
    write_json(directory/'snapshot-index.json',dict(status='execution_sources_snapshotted',
        files=len(rows),total_bytes=sum(r['size_bytes'] for r in rows),manifest_sha256=digest(directory/'files.csv'),
        sources=rows,authorization='2026-11-01 goal; total new spend <= CNY20; EXP018 first stage <= CNY8'))
    print(json.dumps(dict(manifest=(directory/'files.csv').relative_to(ROOT).as_posix(),
                         files=len(rows),bytes=sum(r['size_bytes'] for r in rows))))


if __name__=='__main__':main()
