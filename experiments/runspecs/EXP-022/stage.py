"""Internal, controller-bound fresh fit or fixed-outer dense stage."""
import argparse
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.linear_cv_execution import load, read, require, reused_inner, train_stage
from experiments.common.video.linear_cv_evaluation import evaluate_fold
from experiments.common.video.cv_supervised_training import _sha


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','reference-root','out','execution-inputs'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--item', required=True)
    parser.add_argument('--ffmpeg', type=Path)
    args = parser.parse_args(argv)
    contract = read(args.execution_inputs)
    require(contract['status'] == 'linear_cv_controller_active' and os.getppid() == contract['controller_pid']
            and time.time() < contract['money_deadline_unix'] and
            contract['plan_sha256'] == _sha(args.plan / 'plan.json') and
            args.out.resolve() == Path(contract['output_directory']).resolve() / args.item and
            str(args.reference_root.resolve()) == contract['reference_root'] and
            _sha(Path(contract['platform_timer_receipt'])) == contract['platform_timer_receipt_sha256'],
            'Internal stage requires its live bounded controller and frozen inputs')
    require(args.item in contract['stage_order'], 'Unregistered stage')
    fold = int(args.item[1:3]); kind = args.item.split('-')[1]
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    context = load(ROOT, args.plan, args.reference_root, with_dense=kind == 'dense')
    if kind == 'inner':
        train_stage(context, args.out, fold, 'inner_select')
    else:
        inner = (reused_inner(context, fold) if fold in (1,4) else
                 read(args.out.parent / ('f%02d-inner' % fold) / 'training-result.json'))
        if kind == 'outer':
            train_stage(context, args.out, fold, 'fixed_outer', inner)
        elif kind == 'dense':
            require(args.ffmpeg is not None and _sha(args.ffmpeg) == contract['ffmpeg_sha256'], 'Wrong dense decoder')
            evaluate_fold(context, args.out, fold, args.out.parent / ('f%02d-outer' % fold),
                          inner['selected_epochs'], args.ffmpeg)
        else:
            raise ValueError('Unknown stage')


if __name__ == '__main__':
    main()
