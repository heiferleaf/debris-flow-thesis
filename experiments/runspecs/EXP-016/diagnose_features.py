"""Read-only CPU activation diagnostics of immutable EXP-016 checkpoints.

Uses a fixed small set of existing train/validation clips. No optimizer,
parameter updates, model selection, threshold search or external test data.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    run.relative_to(ROOT/'data/results/exp016')
    out.relative_to(ROOT/'data/results/exp016')
    if out.exists():
        raise FileExistsError(out)
    import numpy as np
    import torch
    import cv2
    import torchvision
    from experiments.common.video.supervised_plan import digest
    from experiments.common.video.supervised_training import load_inputs
    from experiments.common.video.supervised_runtime import ClipDataset, ClipInput
    from experiments.common.video.r2plus1d_classifier import VideoClassifier, configure_fp32, state_digest
    configure_fp32(7)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    plan_dir = ROOT/'data/metadata/EXP-016/training-plan-v2'
    plan, rows, schedule, validation = load_inputs(ROOT, plan_dir)
    dataset = ClipDataset(rows, cache_bytes=8*1024**2)
    chosen, identities = [], []
    for split, pool in [('train', schedule[0]), ('validation', validation)]:
        for label in (0, 1):
            selection = [i for i in pool if rows[i][2] == label][:2]
            if len(selection) != 2:
                raise ValueError('Diagnostic sample coverage incomplete')
            for i in selection:
                chosen.append(i)
                identities.append(dict(clip_id=rows[i][0], label=label, split=split))
    pixels = torch.from_numpy(np.stack([dataset[i][0] for i in chosen]))
    inventory = json.loads((run/'checkpoint-transfer.json').read_text(encoding='utf-8'))
    expected = {r['path']: r['sha256'] for r in inventory}
    report = dict(scope='read-only eight-clip diagnostic; not video metrics or independent test',
                  code_sha256=digest(__file__), plan_sha256=digest(plan_dir/'plan.json'),
                  transfer_manifest_sha256=digest(run/'checkpoint-transfer.json'),
                  implementation_sha256={n: digest(ROOT/'experiments/common/video'/n) for n in
                      ('supervised_runtime.py', 'supervised_plan.py', 'supervised_training.py',
                       'r2plus1d_classifier.py', 'pts_clip_sampling.py')},
                  samples=identities, environment=dict(python=sys.version, torch=torch.__version__,
                      torchvision=torchvision.__version__, opencv=cv2.__version__, numpy=np.__version__, device='cpu'), models={})
    matrices = {}
    for name, mode in [('probe-rgb/best.pt', 'rgb_zero'), ('rgb/best.pt', 'rgb_zero'),
                       ('rgb/last.pt', 'rgb_zero'), ('diff/best.pt', 'rgb_diff')]:
        path = run/name
        if digest(path) != expected[name]:
            raise ValueError('Checkpoint SHA mismatch: '+name)
        # mmap avoids materializing unused optimizer tensors on memory-limited
        # local machines. Only the model state is copied into the network.
        checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        profile = checkpoint['profile']
        if profile['mode'] != mode or profile['plan_sha256'] != digest(plan_dir/'plan.json') or profile['smoke']:
            raise ValueError('Checkpoint profile mismatch')
        training_path = path.parent/'training-result.json'
        training = json.loads(training_path.read_text(encoding='utf-8'))
        selected_epoch = training['best_epoch'] if path.name == 'best.pt' else training['completed_epochs']
        if profile != training['profile'] or checkpoint['completed_epochs'] != selected_epoch:
            raise ValueError('Training selection/profile mismatch')
        if path.name == 'best.pt' and expected[name] != training['selected_checkpoint']['sha256']:
            raise ValueError('Selected checkpoint SHA differs from training result')
        net = VideoClassifier(pretrained=False).eval()
        net.load_state_dict(checkpoint['state_dict'])
        before = state_digest(net)
        checkpoint_epoch = checkpoint['completed_epochs']
        step_values = sorted({float(s['step']) for s in checkpoint['optimizer']['state'].values()})
        difference_weight_max = float(net.backbone.stem[0].weight[:, 3:].abs().max())
        del checkpoint
        transform = ClipInput(mode)
        activations = {k: [] for k in ('backbone512', 'relu128', 'relu64', 'relu32')}
        handles = []
        for key, module in [('backbone512', net.backbone), ('relu128', net.head[1]),
                            ('relu64', net.head[4]), ('relu32', net.head[7])]:
            def capture(module, inputs, output, key=key):
                activations[key].append(output.detach().numpy().copy())
            handles.append(module.register_forward_hook(capture))
        logits = []
        with torch.inference_mode():
            inputs = transform(pixels)
            for i in range(len(chosen)):
                logits.append(float(net(inputs[i:i+1])[0]))
        for handle in handles:
            handle.remove()
        if state_digest(net) != before or digest(path) != expected[name]:
            raise ValueError('Model/checkpoint changed during diagnostic')
        info = dict(checkpoint_sha256=expected[name], completed_epochs=checkpoint_epoch,
                    requested_training_epochs=profile['requested_epochs'],
                    training_result_sha256=digest(training_path), model_state_unchanged=True,
                    optimizer_steps=step_values, difference_stem_weight_abs_max=difference_weight_max,
                    input_channel_std=inputs.std(dim=(0, 2, 3, 4), unbiased=False).tolist(), layers={}, logits=logits)
        for key, arrays in activations.items():
            values = np.concatenate(arrays)
            if not np.isfinite(values).all() or not np.isfinite(logits).all():
                raise ValueError('Nonfinite diagnostic activation')
            info['layers'][key] = dict(mean_sample_l2=float(np.linalg.norm(values, axis=1).mean()),
                                      mean_between_sample_std=float(values.std(axis=0).mean()),
                                      max_between_sample_std=float(values.std(axis=0).max()),
                                      zero_fraction=float(np.mean(values == 0)))
            matrices[name.replace('/', '__')+'__'+key] = values
        report['models'][name] = info
        print(json.dumps(dict(checkpoint=name, layers=info['layers'], logits=logits)), flush=True)
        del inputs, transform, net, activations
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out/'activations.npz', **matrices)
    (out/'diagnostic.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
