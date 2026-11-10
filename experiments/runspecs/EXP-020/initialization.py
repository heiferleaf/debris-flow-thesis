"""Four CPU initializations only: no bank, cache index, optimizer or forward."""
import argparse
import os
from pathlib import Path
import platform
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
import torchvision
from experiments.common.video.head_comparison_training import initialization_records, require_local_weights, _sha, _json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--observe-local-environment', action='store_true')
    a = p.parse_args(argv)
    out = a.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(4)
    began = time.perf_counter()
    names = ['experiments/common/video/head_comparison_training.py', 'experiments/common/video/r2plus1d_classifier.py',
             'experiments/common/video/cv_supervised_training.py', 'experiments/runspecs/EXP-020/initialization.py']
    sources = {n: _sha(ROOT / n) for n in names}
    answer = dict(status='initialization_failed', source_sha256=sources, argv=sys.argv if argv is None else argv,
        environment=dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__, platform=platform.platform()),
        device='cpu', no_forward=True, no_optimizer=True, no_training=True, no_download=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        weights = require_local_weights()
        answer['public_weights_sha256'] = _sha(weights)
        answer['initializations'] = initialization_records(enforce_original=not a.observe_local_environment)
        if any(_sha(ROOT / n) != sha for n, sha in sources.items()):
            raise ValueError('Initialization source changed during execution')
        answer['status'] = 'local_environment_observation_not_production_proof' if a.observe_local_environment else 'production_initialization_verified'
    except Exception as e:
        answer.update(error_type=type(e).__name__, error=str(e))
        raise
    finally:
        answer['seconds'] = time.perf_counter() - began
        _json(out, answer)
    print(answer['status'], _sha(out), answer['seconds'])


if __name__ == '__main__':
    main()
