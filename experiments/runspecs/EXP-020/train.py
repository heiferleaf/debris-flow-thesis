"""One fresh full20 EXP020 inner fit, launched in a dedicated subprocess."""
import argparse
import math
from pathlib import Path
import sys
import time
import os
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from plan import load, read, _sha
from experiments.common.video.head_comparison_training import fit


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--item', required=True)
    p.add_argument('--execution-inputs', type=Path, required=True)
    a = p.parse_args(argv)
    launch = read(a.execution_inputs)
    deadline = launch['money_deadline_unix']
    if (not math.isfinite(deadline) or time.time() >= deadline or launch['plan_sha256'] != _sha(a.plan / 'plan.json')
            or a.out.resolve() != Path(launch['output_directory']).resolve() / a.item):
        raise ValueError('Missing/mismatched live bounded controller contract')
    plan, schedules, bank, refs, fingerprints = load(ROOT, a.plan, with_frames=True)
    item = next(x for x in plan['items'] if x['item'] == a.item)
    if item['action'] != 'fit':
        raise ValueError('Historical seed7 baseline must never be refitted')
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    schedule = schedules[str(item['fold'])]
    result = fit(ROOT, a.out, head=item['head'], seed=item['seed'], fold=item['fold'], bank=bank,
        verified_frame_index=refs, input_fingerprints=fingerprints, **schedule,
        expected_initial_state_sha256=plan['initializations'][item['head'] + '-s' + str(item['seed'])]['initial_state_sha256'],
        max_epochs=20, device='cuda', workers=2, cache_mib=128)
    if result['completed_epochs'] != 20:
        raise RuntimeError('Full twenty epochs required')


if __name__ == '__main__':
    main()
