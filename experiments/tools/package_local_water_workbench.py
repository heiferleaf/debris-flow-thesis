"""Export an allowlisted portable tool, without private data or credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[2]
UI=Path('experiments/tools/synthetic_minimax_workbench')
SOURCE=Path('data/results/synthetic-video/pretrained/slr-sfs-fa8d610/source/SLR-SFS-fa8d61046dfa19d80c5605be4a64e71f499b58f7')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'.codex/docs/local-water-workbench-share.zip')
    args=parser.parse_args()
    files=[Path('experiments/tools/controlled_water_preview.py'),Path('experiments/tools/synthetic_minimax_workbench.py'),
           Path('experiments/tools/prepare_local_water.py'),Path('experiments/common/video/local_water_workflow.py')]
    files += [UI/name for name in ['local.html','local.js','local.css','style.css','requirements-local.txt','README-local.md']]
    files += [p.relative_to(ROOT) for p in (ROOT/SOURCE/'models').rglob('*.py') if '__pycache__' not in p.parts]
    files += [SOURCE/'LICENSE.md',SOURCE/'README.md']
    manifest=[];args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED) as z:
        for relative in files:
            content=(ROOT/relative).read_bytes()
            z.writestr('water-workbench-local/'+relative.as_posix(),content)
            manifest.append({'path':relative.as_posix(),'sha256':hashlib.sha256(content).hexdigest()})
        z.writestr('water-workbench-local/start-local.cmd',(ROOT/UI/'start-local.cmd').read_bytes())
        z.writestr('water-workbench-local/README-local.md',(ROOT/UI/'README-local.md').read_bytes())
        z.writestr('water-workbench-local/MANIFEST.json',json.dumps({'files':manifest,'includes_personal_media':False,'includes_credentials':False,'includes_weights':False},indent=2))
    print(json.dumps({'file':str(args.output),'bytes':args.output.stat().st_size,'files':len(manifest)+3}))

if __name__=='__main__':main()
