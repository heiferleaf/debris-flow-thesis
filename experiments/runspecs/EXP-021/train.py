"""Internal one-fit EXP021 entry; no outer/dense selection and no continuation."""
import argparse
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.oriented_energy_plan import load, frame_refs, read
from experiments.common.video.oriented_energy_training import fit
from experiments.common.video.cv_supervised_training import _sha


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'reference-root', 'out', 'execution-inputs'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--item', required=True)
    a = p.parse_args(argv)
    execution = read(a.execution_inputs)
    if (time.time() >= execution['money_deadline_unix'] or execution['plan_sha256'] != _sha(a.plan / 'plan.json')
            or a.out.resolve() != Path(execution['output_directory']).resolve() / a.item
            or str(a.reference_root.resolve()) != execution['reference_root']):
        raise ValueError('Internal stage differs from bounded controller contract')
    context = load(ROOT, a.plan, a.reference_root)
    plan = context['plan']
    if a.item not in plan['fit_order']:
        raise ValueError('Unknown new fit or attempted A retraining')
    fold = int(a.item[1:3]); mode = a.item.split('-')[1]
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    result = fit(ROOT, a.out, mode=mode, kernel_path=ROOT / plan['kernel_path'], kernel_sha256=plan['kernel_sha256'],
        fold=fold, bank=context['bank'], verified_frame_index=frame_refs(context, fold),
        input_fingerprints=context['fingerprints'], **context['schedules'][str(fold)],
        max_epochs=20, device='cuda', workers=2, cache_mib=128)
    if result['completed_epochs'] != 20 or result['optimizer_steps'] != (340 if fold == 1 else 360):
        raise ValueError('Registered full20 fit/update count required')


if __name__ == '__main__':
    main()
