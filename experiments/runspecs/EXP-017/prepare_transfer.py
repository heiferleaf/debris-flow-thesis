"""Freeze deployment references; never copy source media/JPEGs or rent hardware."""
import argparse
import csv
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from experiments.common.video.cv_execution import load_execution,deployment_records
from experiments.common.video.supervised_plan import digest,write_json


def freeze(root,out,plan,acceptance,dense_reference):
    root,out=Path(root).resolve(),Path(out).resolve();out.relative_to(root/'data/metadata/EXP-017')
    if out.exists():raise FileExistsError(out)
    context=load_execution(root,plan,acceptance,dense_reference)
    pipeline=runpy.run_path(str(root/'experiments/runspecs/EXP-017/pipeline.py'))
    records=deployment_records(context,pipeline['code_paths']())
    out.mkdir(parents=True,exist_ok=False)
    with (out/'files.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','kind','sha256','size_bytes']);w.writeheader();w.writerows(records[k] for k in sorted(records))
    for row in records.values():
        if row['kind']=='execution_source':
            target=out/'source-snapshot'/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write((root/row['path']).read_bytes())
            if digest(target)!=row['sha256']:raise ValueError('Source changed during deployment freeze')
    result=dict(status='deployment_references_frozen',files=len(records),manifest_sha256=digest(out/'files.csv'),
        total_referenced_bytes=sum(r['size_bytes'] for r in records.values()),images_or_media_copied=False,
        transfer_rule='Remote same-SHA existing files may be skipped; transfer missing/different files before static preflight; no overwrite of historical runs.')
    write_json(out/'snapshot-index.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','acceptance','dense-reference','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(freeze(ROOT,a.out,a.plan,a.acceptance,a.dense_reference)))


if __name__=='__main__':main()
