"""Replay saved EXP-008/011/012 heads, without training or changing source runs."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    import numpy as np
    import torch
    from sklearn.metrics import f1_score, roc_auc_score
    from experiments.common.video.train_mil import MilHead
    from experiments.common.video.train_strong import (
        StrongHead, StrongNetCtx, GatedCtx, SelfAttnCtx,
    )

    torch.set_num_threads(2)
    runs = root / 'data/results/exp008'
    fp = runs / 'unitfeats-v2-150/unit-features.npz'
    lp = root / 'data/process/EXP-008-units-v2/clip-labels.npz'
    data = np.load(fp, allow_pickle=True)
    cl = np.load(lp, allow_pickle=True)
    ids = [str(x) for x in data['unit_ids']]
    assert ids == list(cl['unit_ids'])
    X = torch.as_tensor(data['feats'], dtype=torch.float32)
    y = data['labels'].astype(int)
    lookup = {u: i for i, u in enumerate(ids)}
    manifest = list(csv.DictReader((root / 'data/process/EXP-008-units-v2/unit-manifest.csv').open(encoding='utf-8-sig')))
    assert [r['unit_id'] for r in manifest] == ids
    assert all((r['label'] == 'pos') == bool(y[i]) for i, r in enumerate(manifest))
    report = {
        'scope': 'Saved frozen features -> saved head weights -> video scores -> fold metrics. No training, raw-video decoding or backbone-feature regeneration.',
        'feature_sha256': digest(fp), 'labels_sha256': digest(lp),
        'feature_shape': list(X.shape), 'n_negative': int((y == 0).sum()),
        'n_positive': int((y == 1).sum()),
        'source_hashes': {str(p.relative_to(root)).replace('\\', '/'): digest(p) for p in [
            root / 'experiments/common/video/train_mil.py',
            root / 'experiments/common/video/train_strong.py',
            Path(__file__).resolve(),
        ]},
        'runs': [],
    }
    for path in sorted(runs.iterdir()):
        tag = path.name
        if not tag.startswith(('mil-v2-150-', 'strong-v2-150-')):
            continue
        if not (path / 'predictions.csv').exists():
            continue
        m = json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
        if m.get('smoke') or len(m.get('folds', [])) != 5:
            continue
        rows = list(csv.DictReader((path / 'predictions.csv').open(encoding='utf-8-sig')))
        assert len(rows) == len(ids) == len({r['unit_id'] for r in rows})
        caliber = m.get('clip_caliber', m.get('caliber'))
        mask = torch.as_tensor(cl[caliber] >= 0) if caliber else torch.ones(X.shape[:2], dtype=torch.bool)
        if tag.endswith('-iid'):
            sp = 'EXP-008-units-iid-folds.json'
        elif tag.endswith('-scene'):
            sp = 'EXP-008-units-scene-folds.json'
        else:
            sp = 'EXP-008-units-folds.json'
        splits = json.loads((root / 'data/metadata/splits' / sp).read_text(encoding='utf-8'))
        fold_reports = []
        for f in m['folds']:
            fold = int(f['fold'])
            fr = [r for r in rows if int(r['fold']) == fold]
            idx = [lookup[r['unit_id']] for r in fr]
            fs = next(q for q in splits['folds'] if int(q['fold']) == fold)
            if isinstance(fs.get('val_units'), list):
                expected = set(fs['val_units'])
            elif 'val_groups' in fs:
                expected = {r['unit_id'] for r in manifest if r['group_id'] in fs['val_groups']}
            else:
                expected = {ids[i] for i, v in enumerate(data['folds']) if int(v) == fold}
            assert expected == {r['unit_id'] for r in fr}, (tag, fold, 'split mismatch')
            assert all(int(r['label']) == int(y[i]) for r, i in zip(fr, idx))
            weak = tag.startswith('mil-')
            agg = m.get('context_agg', 'none')
            if weak:
                net = MilHead()
            elif agg in ('gated', 'selfattn'):
                extra = 64 if agg == 'gated' else 128
                net = StrongNetCtx(GatedCtx() if agg == 'gated' else SelfAttnCtx(), StrongHead(512 + extra))
            else:
                net = StrongHead()
            wp = path / ('%s_f%d.pt' % ('mil' if weak else 'strong', fold))
            net.load_state_dict(torch.load(wp, map_location='cpu', weights_only=True))
            net.eval()
            with torch.inference_mode():
                if not weak and agg in ('gated', 'selfattn'):
                    s = torch.stack([net.forward_unit(X[i], mask[i]) for i in idx])
                else:
                    s = net(X[idx]).squeeze(-1)
                if not weak:
                    s = torch.sigmoid(s)
                scores = s.masked_fill(~mask[idx], -float('inf')).max(dim=1).values.numpy()
            saved = np.array([float(r['score']) for r in fr])
            error = float(abs(scores - saved).max())
            auc = float(roc_auc_score(y[idx], scores))
            f1 = float(f1_score(y[idx], scores >= 0.5))
            assert error < 2e-6, (tag, fold, 'score mismatch', error)
            assert abs(auc - f['auc']) <= 0.000051, (tag, fold, 'auc mismatch')
            assert abs(f1 - f['f1']) <= 0.000051, (tag, fold, 'f1 mismatch')
            fold_reports.append({'fold': fold, 'checkpoint': str(wp.relative_to(root)).replace('\\', '/'),
                                 'checkpoint_sha256': digest(wp), 'max_score_error': error,
                                 'auc': auc, 'f1_at_0_5': f1, 'split_members_match': True})
        report['runs'].append({'run_id': tag, 'status': 'passed', 'folds': fold_reports,
                               'fold_mean_auc': float(np.mean([r['auc'] for r in fold_reports])),
                               'recorded_mean_auc': m['mean_auc'],
                               'predictions_sha256': digest(path / 'predictions.csv')})
        print(tag, 'PASS', flush=True)
    report['verified_run_count'] = len(report['runs'])
    report['verified_checkpoint_count'] = sum(len(x['folds']) for x in report['runs'])
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=REPO)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit('Refusing to overwrite an existing verification record')
    report = verify(args.root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['verified_run_count', 'verified_checkpoint_count']}))


if __name__ == '__main__':
    main()
