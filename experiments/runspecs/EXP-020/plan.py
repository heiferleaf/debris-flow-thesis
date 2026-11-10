"""Small immutable EXP020 plan; inherits accepted EXP017 inputs, never re-decodes."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

from experiments.common.video.cv_execution import verified_frames
from experiments.common.video.cv_supervised_plan import load_cv_plan
from experiments.common.video.cv_supervised_training import _canonical
from experiments.common.video.head_comparison_training import _sha, _json, WEIGHTS_SHA

ROOT = Path(__file__).resolve().parents[3]
AUDIT = 'data/results/exp017/audit-resume-cv-s7-20261102-v2-v2/audit.json'
AUDIT_SHA = '4b94b4094336c99bf35e675bd07a72420498fb97bd56f8ee366a75a2b2971c45'
CARD = 'docs/process/experiments/EXP-020.md'
CARD_SHA = 'cd4a4cbf05d1ae5ac7fecb7585fd202abc35ab63ee018c3065d82345598fa348'
BASE = 'data/metadata/EXP-017/role-free-plan-v3'
ACCEPTANCE = 'data/metadata/EXP-017/cv-cache-acceptance-v1/acceptance.json'
EVOLUTION = '.codex/logs/evidence/2026-11-03-exp020-protocol-evolution.json'
NEW_CODE = ['experiments/common/video/head_comparison_training.py'] + [
    'experiments/runspecs/EXP-020/' + n for n in ('plan.py', 'prepare.py', 'initialization.py', 'train.py', 'pipeline.py', 'run.sh', 'README.md')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def inside(root, name):
    p = (root / name).resolve()
    p.relative_to(root)
    if Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Non-relative input identity')
    return p


def epochs(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]


def epoch_block(path):
    text = Path(path).read_text(encoding='utf-8')
    return text[text.index('            for epoch, order in enumerate(schedule[:epochs], 1):'):
                text.index('        if completed != epochs or total_updates != planned_steps:')]


def verify_baseline(root, artifact, f, bank):
    """No outer substitution, labels or schedule changes may masquerade as reuse."""
    result, profile, inputs = [read(artifact / n) for n in ('training-result.json', 'profile.json', 'inputs.json')]
    history = epochs(artifact / 'epochs.jsonl')
    expected = f['inner_train']['epoch_clip_ids']; val = f['inner_val']['clip_ids']
    if (result['status'] != 'fit_complete' or result['stage'] != 'inner_select' or profile['stage'] != 'inner_select'
            or profile['seed'] != 7 or result['fold'] != f['fold'] or result['completed_epochs'] != 20
            or len(history) != 20 or inputs['epoch_clip_ids'] != expected or inputs['validation_clip_ids'] != val):
        raise ValueError('Wrong baseline role, schedule or completeness')
    for cid, row in inputs['clip_bank'].items():
        w = bank[cid]
        if any(row[k] != w[k] for k in ('unit_id', 'group_id', 'label', 'frame_indices', 'source_pts_seconds')):
            raise ValueError('Baseline clip identity/label differs')
    losses = []
    for i, row in enumerate(history, 1):
        predictions = read(artifact / ('validation-epoch-%02d.json' % i))
        if (row['train_clip_ids'] != expected[i - 1] or [p['clip_id'] for p in predictions] != val
                or any(p['label'] != bank[p['clip_id']]['label'] or not math.isfinite(p['logit']) for p in predictions)
                or row['optimizer_steps'] != math.ceil(len(expected[i - 1]) / 24)):
            raise ValueError('Baseline exposure/logit identity differs')
        value = math.fsum(max(p['logit'], 0.) - p['label'] * p['logit'] + math.log1p(math.exp(-abs(p['logit']))) for p in predictions) / len(predictions)
        if not math.isfinite(row['validation_unweighted_bce']) or abs(value - row['validation_unweighted_bce']) > 2e-6:
            raise ValueError('Baseline raw validation BCE mismatch')
        losses.append(row['validation_unweighted_bce'])
    if result['selected_epochs'] != min(range(1, 21), key=lambda e: losses[e - 1]):
        raise ValueError('Baseline earliest raw BCE selection differs')
    return result


def check_initialization(root, path):
    proof = read(path)
    records = proof['initializations']
    if (proof['status'] != 'production_initialization_verified' or proof['public_weights_sha256'] != WEIGHTS_SHA
            or records['deep-s7']['initial_state_sha256'] != 'c1847e1c0987ff003bff99805ab795af8eb59b07db210c7999e377b337ee9b81'
            or set(records) != {'deep-s7', 'deep-s8', 'linear-s7', 'linear-s8'}
            or len({r['backbone_state_sha256'] for r in records.values()}) != 1
            or len({r['stem_state_sha256'] for r in records.values()}) != 1
            or any(r['head_parameters'] != (513 if k.startswith('linear') else 76033) for k, r in records.items())):
        raise ValueError('Production initialization proof incomplete/mismatched; local observations are insufficient')
    expected = {NEW_CODE[0], 'experiments/common/video/r2plus1d_classifier.py',
                'experiments/common/video/cv_supervised_training.py', 'experiments/runspecs/EXP-020/initialization.py'}
    if set(proof['source_sha256']) != expected:
        raise ValueError('Initialization source closure missing')
    for name, sha in proof['source_sha256'].items():
        if _sha(inside(root, name)) != sha:
            raise ValueError('Initialization source changed')
    return proof


def prepare(root, out, initialization_proof):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.relative_to(root / 'data/metadata/EXP-020')
    if out.exists():
        raise FileExistsError(out)
    if _sha(root / AUDIT) != AUDIT_SHA or _sha(root / CARD) != CARD_SHA:
        raise ValueError('Registered audit/card changed')
    audit = read(root / AUDIT)
    if audit['status'] != 'audit_passed':
        raise ValueError('Parent audit not passed')
    original = read(root / 'data/results/exp017/cv-s7-20261102-v1/execution-inputs.json')
    evolution = read(root / EVOLUTION)
    changes = {r['logical_path']: r for r in evolution['documents']}
    frozen = {AUDIT: AUDIT_SHA, CARD: CARD_SHA, EVOLUTION: _sha(root / EVOLUTION)}
    # Two governance documents evolved explicitly; all other accepted original
    # execution sources/input metadata remain byte-identical. No media/JPEG scan.
    for name, sha in original['input_fingerprints'].items():
        if name in changes:
            r = changes[name]
            if r['historical_sha256'] != sha or _sha(root / name) != r['current_sha256']:
                raise ValueError('Unregistered governance evolution')
            for snap in r['historical_snapshots']:
                if _sha(root / snap) != sha:
                    raise ValueError('Historical protocol snapshot changed')
            frozen[name] = r['current_sha256']
        else:
            if _sha(inside(root, name)) != sha:
                raise ValueError('Original accepted input/source changed: ' + name)
            frozen[name] = sha
    p, bank, _, _, folds = load_cv_plan(root, root / BASE, verify_sources=False)
    if _sha(root / BASE / 'plan.json') != original['plan_sha256']:
        raise ValueError('Base plan differs from audited execution')
    for name in NEW_CODE:
        frozen[name] = _sha(root / name)
    legacy_block = epoch_block(root / 'experiments/common/video/cv_supervised_training.py')
    if epoch_block(root / NEW_CODE[0]) != legacy_block:
        raise ValueError('New epoch execution differs from frozen EXP017')
    initialization_proof = Path(initialization_proof).resolve()
    initialization_proof.relative_to(root)
    initial_proof = check_initialization(root, initialization_proof)
    frozen[initialization_proof.relative_to(root).as_posix()] = _sha(initialization_proof)
    initial = initial_proof['initializations']
    items, schedules = [], {}
    for fold in (1, 4):
        f = folds[fold - 1]
        counts = Counter(bank[c]['label'] for c in f['inner_train']['epoch_clip_ids'][0])
        if (counts[1], counts[0], len(f['inner_val']['clip_ids'])) != ((288, 120, 100) if fold == 1 else (292, 120, 104)):
            raise ValueError('Registered fold counts changed')
        artifact = inside(root, audit['fits'][fold - 1]['artifact_directory']) / 'inner'
        baseline = verify_baseline(root, artifact, f, bank)
        reuse = {}
        for path in [artifact / n for n in ('profile.json', 'training-result.json', 'inputs.json', 'epochs.jsonl')] + sorted(artifact.glob('validation-epoch-*.json')):
            name = path.relative_to(root).as_posix()
            if audit['source_sha256'].get(name) != _sha(path):
                raise ValueError('Reuse artifact differs from passed audit')
            reuse[name] = frozen[name] = _sha(path)
        schedules[str(fold)] = dict(epoch_clip_ids=f['inner_train']['epoch_clip_ids'], validation_clip_ids=f['inner_val']['clip_ids'])
        checkpoints = {}
        for name in ('best.pt', 'last.pt'):
            logical = (artifact / name).relative_to(root).as_posix()
            checkpoints[logical] = audit['source_sha256'][logical]
        if baseline['selected_checkpoint']['sha256'] != checkpoints[(artifact / 'best.pt').relative_to(root).as_posix()]:
            raise ValueError('Audited baseline checkpoint link differs')
        items.append(dict(item='f%02d-deep-s7' % fold, fold=fold, head='deep', seed=7, action='reuse',
            artifact_directory=artifact.relative_to(root).as_posix(), files_sha256=reuse,
            audited_checkpoint_sha256=checkpoints, selected_epochs=baseline['selected_epochs']))
        for head, seed in (('linear', 7), ('deep', 8), ('linear', 8)):
            items.append(dict(item='f%02d-%s-s%d' % (fold, head, seed), fold=fold, head=head, seed=seed, action='fit'))
    out.mkdir(parents=True, exist_ok=False)
    _json(out / 'schedules.json', schedules)
    plan = dict(spec_id='EXP-020', status='prepared_not_training_authorization', version=1,
        audit_path=AUDIT, audit_sha256=AUDIT_SHA, base_plan_path=BASE, acceptance_path=ACCEPTANCE,
        initializations=initial, items=items, fit_order=[x['item'] for x in items if x['action'] == 'fit'],
        epochs=20, total_new_epochs=120, total_new_updates=2100, dataloader_generator_seed=100007,
        initialization_proof=initialization_proof.relative_to(root).as_posix(), initialization_environment=initial_proof['environment'],
        code_sha256={n: frozen[n] for n in set(original['code_sha256']) | set(NEW_CODE) | {CARD}},
        schedules_sha256=_sha(out / 'schedules.json'), source_fingerprints=frozen,
        epoch_execution_sha256=hashlib.sha256(legacy_block.encode()).hexdigest(),
        original_plan_live_governance_not_revalidated=True, original_acceptance_reused_no_pixel_rescan=True,
        independent_test=False, environment=dict(python=sys.version, platform=platform.platform()))
    _json(out / 'plan.json', plan)
    for name, sha in frozen.items():
        if _sha(root / name) != sha:
            raise ValueError('Preparation source changed during initialization: ' + name)
    return plan


def load(root, directory, with_frames=False):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    directory.relative_to(root)
    plan = read(directory / 'plan.json')
    if plan['spec_id'] != 'EXP-020' or plan['epochs'] != 20 or plan['audit_sha256'] != AUDIT_SHA:
        raise ValueError('Unregistered comparison plan')
    for name, sha in plan['source_fingerprints'].items():
        if _sha(inside(root, name)) != sha:
            raise ValueError('Frozen comparison source changed: ' + name)
    if _sha(directory / 'schedules.json') != plan['schedules_sha256']:
        raise ValueError('Frozen schedule changed')
    schedules = read(directory / 'schedules.json')
    proof = check_initialization(root, inside(root, plan['initialization_proof']))
    if plan['initializations'] != proof['initializations']:
        raise ValueError('Plan initialization values differ from frozen production proof')
    _, bank, _, _, folds = load_cv_plan(root, root / plan['base_plan_path'], verify_sources=False)
    for fold in (1, 4):
        f, schedule = folds[fold - 1], schedules[str(fold)]
        if schedule != dict(epoch_clip_ids=f['inner_train']['epoch_clip_ids'], validation_clip_ids=f['inner_val']['clip_ids']):
            raise ValueError('Frozen inner schedule identity differs')
    refs, fingerprints = None, dict(plan['source_fingerprints'])
    fingerprints[(directory / 'plan.json').relative_to(root).as_posix()] = _sha(directory / 'plan.json')
    fingerprints[(directory / 'schedules.json').relative_to(root).as_posix()] = plan['schedules_sha256']
    if with_frames:
        _, refs, _, accepted = verified_frames(root, root / plan['base_plan_path'], root / plan['acceptance_path'])
        fingerprints.update(accepted)
    return plan, schedules, bank, refs, fingerprints


def comparison(plan, run, root, *, bank, schedules):
    """All eight selected inner results. Raw logit AUC is never pooled."""
    import numpy as np
    from experiments.common.video.dense_evaluation import video_metrics
    rows = []
    for item in plan['items']:
        path = inside(root, item['artifact_directory']) if item['action'] == 'reuse' else run / item['item']
        result = read(path / 'training-result.json')
        profile = read(path / 'profile.json')
        history = epochs(path / 'epochs.jsonl')
        schedule = schedules[str(item['fold'])]
        orders, val = schedule['epoch_clip_ids'], schedule['validation_clip_ids']
        expected_initial = plan['initializations'][item['head'] + '-s' + str(item['seed'])]['initial_state_sha256']
        if (len(history) != 20 or result['status'] != 'fit_complete' or result['completed_epochs'] != 20
                or result['stage'] != 'inner_select' or result['fold'] != item['fold']
                or result['initial_state_sha256'] != expected_initial
                or profile['seed'] != item['seed'] or profile.get('head', 'deep') != item['head']
                or result['optimizer_steps'] != sum(math.ceil(len(x) / 24) for x in orders)):
            raise ValueError('Cannot compare incomplete fit')
        for e, record in enumerate(history, 1):
            predictions = read(path / ('validation-epoch-%02d.json' % e))
            if (record['train_clip_ids'] != orders[e - 1] or [r['clip_id'] for r in predictions] != val
                    or any(r['label'] != bank[r['clip_id']]['label'] or not math.isfinite(r['logit']) for r in predictions)):
                raise ValueError('Fit exposure/fixed validation labels differ')
            bce = math.fsum(max(r['logit'], 0.) - r['label'] * r['logit'] + math.log1p(math.exp(-abs(r['logit']))) for r in predictions) / len(predictions)
            if not math.isfinite(record['validation_unweighted_bce']) or abs(record['validation_unweighted_bce'] - bce) > 2e-6:
                raise ValueError('Raw BCE replay differs')
        values = [r['validation_unweighted_bce'] for r in history]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite selection history')
        selected = min(range(1, 21), key=lambda e: values[e - 1])
        if result['selected_epochs'] != selected:
            raise ValueError('Earliest raw BCE selection changed')
        if item['action'] == 'fit' and _sha(path / 'best.pt') != result['selected_checkpoint']['sha256']:
            raise ValueError('Selected weight bytes differ')
        predictions = read(path / ('validation-epoch-%02d.json' % selected))
        z = np.array([p['logit'] for p in predictions], dtype=np.float64)
        y = np.array([p['label'] for p in predictions], dtype=int)
        if not np.isfinite(z).all() or not np.isin(y, [0, 1]).all() or not (y == 1).any() or not (y == 0).any():
            raise ValueError('Invalid selected prediction')
        q = np.exp(-np.abs(z)); prob = np.where(z >= 0, 1 / (1 + q), q / (1 + q))
        metrics = video_metrics([dict(label=int(label), score=float(score)) for label, score in zip(y, prob)])
        import torch
        fp32_prob = torch.sigmoid(torch.tensor(z, dtype=torch.float32)).numpy()
        fp32_metrics = video_metrics([dict(label=int(label), score=float(score)) for label, score in zip(y, fp32_prob)])
        pz, nz = z[y == 1], z[y == 0]
        auc_logit = float(np.mean((pz[:, None] > nz) + .5 * (pz[:, None] == nz)))
        positive, negative = metrics['Recall'], 1 - metrics['FPR']
        rows.append(dict(**item, actual_artifact_directory=str(path), selected_epochs=selected,
            score_kind='logit_AUC_and_stable_FP64_sigmoid_for_fixed_0.5_recalls',
            AUC_logit=auc_logit, derived_FP64_probability_AUC=metrics['AUC'],
            derived_FP32_probability_AUC=fp32_metrics['AUC'],
            logit_minus_derived_FP32_probability_AUC=auc_logit - fp32_metrics['AUC'],
            historical_probability_AUC_unchanged=True, positive_Recall=positive, negative_Recall=negative,
            TP=metrics['TP'], FP=metrics['FP'], FN=metrics['FN'], TN=metrics['TN'],
            probability_min=float(prob.min()), probability_max=float(prob.max()), probability_sd=float(prob.std()),
            gate_pass=auc_logit >= .75 and positive >= .5 and negative >= .5))
    pairs = []
    for fold in (1, 4):
        for seed in (7, 8):
            a = next(r for r in rows if (r['fold'], r['seed'], r['head']) == (fold, seed, 'deep'))
            b = next(r for r in rows if (r['fold'], r['seed'], r['head']) == (fold, seed, 'linear'))
            pairs.append(dict(fold=fold, seed=seed, deep_item=a['item'], linear_item=b['item'],
                delta_AUC_logit=b['AUC_logit'] - a['AUC_logit'],
                delta_positive_Recall=b['positive_Recall'] - a['positive_Recall'],
                delta_negative_Recall=b['negative_Recall'] - a['negative_Recall']))
    return dict(status='all_eight_items_complete_pending_independent_audit', items=rows, pairs=pairs,
        passed_by_head={h: sum(r['gate_pass'] for r in rows if r['head'] == h) for h in ('deep', 'linear')},
        linear_all_four_pass=all(r['gate_pass'] for r in rows if r['head'] == 'linear'),
        scope='Known development inner splits only; no outer, dense, external or temporal-contribution claim')
