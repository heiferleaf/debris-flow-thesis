"""Restore NPZ annotation metadata only; no model input pixels/features."""
import argparse,json,numpy as np
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('metadata');p.add_argument('--out',required=True);a=p.parse_args()
z=json.loads(Path(a.metadata).read_text(encoding='utf-8'));out=Path(a.out)
if out.exists(): raise SystemExit('Refusing overwrite')
arrays={k:np.array(v['values'],dtype=v['dtype']).reshape(v['shape']) for k,v in z['arrays'].items()}
out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**arrays)
