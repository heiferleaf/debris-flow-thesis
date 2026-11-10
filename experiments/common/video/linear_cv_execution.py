"""EXP022 immutable role/model adapter; inherits accepted inputs without re-decoding."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from experiments.common.video.cv_execution import summarize_folds
from experiments.common.video.cv_supervised_plan import load_cv_plan, validate_partitions
from experiments.common.video.cv_supervised_training import _sha, _json
from experiments.common.video.dense_evaluation import video_metrics
from experiments.common.video.linear_cv_training import INITIAL_SHA, BACKBONE_SHA, fit
from experiments.common.video.supervised_runtime import FrameRef, PIXEL_VERSION

BASE = 'data/metadata/EXP-017/role-free-plan-v3'
BASE_SHA = '70394126cd15d015c4250bbea8f1c9a0c928141d92f00a03d66279a576c2c9b8'
AUDIT = 'data/results/exp017/audit-resume-cv-s7-20261102-v2-v2/audit.json'
AUDIT_SHA = '4b94b4094336c99bf35e675bd07a72420498fb97bd56f8ee366a75a2b2971c45'
CARD = 'docs/process/experiments/EXP-022.md'
CARD_SHA = '27a0e2fbc5f8efb30e912c8ccd5c07f3beb7af7ac9b28def5f183da5169f2f3a'
ORIGINAL = 'data/results/exp017/cv-s7-20261102-v1/execution-inputs.json'
ACCEPTANCE = 'data/metadata/EXP-017/cv-cache-acceptance-v1/acceptance.json'
DENSE = 'data/process/EXP-017/dense-pixel-reference-v1/result.json'
INIT = 'data/metadata/EXP-020/cloud-initialization-v1.json'
REVIEW20 = '.codex/logs/evidence/2026-11-03-exp020-results-independent-review.json'
LOCAL20 = 'data/results/exp020/head-s7s8-20261103-v2'
CLOUD20 = '/root/exp020-results/head-s7s8-20261103-v2'
REUSE = {1: (10, '70072ca03e94ea0df0bf980a9513714d4a60f1f8d0858e575bf4dc60d494f094'),
         4: (6, '31d9924c4950ad5e2f966d56bbe0114c5a8af8f7aa5dabb3eae75fef7900e04b')}
COUNTS = [(288,120,72,28,360,148,23,10), (288,120,72,32,360,152,23,6),
          (292,120,72,28,364,148,22,7), (292,120,72,32,364,152,22,7), (288,120,72,32,360,152,23,7)]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rooted(root, name):
    path = (root / name).resolve()
    path.relative_to(root)
    require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Require rooted relative identity')
    return path


def code_paths(root):
    return sorted([p.relative_to(root).as_posix() for p in (root / 'experiments/common/video').glob('linear_cv_*.py')]
                  + [p.relative_to(root).as_posix() for p in (root / 'experiments/runspecs/EXP-022').iterdir() if p.is_file()])


def epoch_block(path):
    text = Path(path).read_text(encoding='utf-8')
    return text[text.index('            for epoch, order in enumerate(schedule[:epochs], 1):'):
                text.index('        if completed != epochs or total_updates != planned_steps:')]


def check_fit_result(result, stage, fold, epochs):
    require(result['status'] == 'fit_complete' and result['stage'] == stage and result['fold'] == fold
            and result['completed_epochs'] == epochs and result['initial_state_sha256'] == INITIAL_SHA
            and result['fixed_outer_validation_evaluated'] is False, 'Incomplete/wrong Linear stage')
    require(type(result['selected_epochs']) is int and 1 <= result['selected_epochs'] <= 20, 'Invalid inner E')
    if stage == 'fixed_outer':
        require(result['selected_epochs'] == epochs and result['selected_checkpoint']['filename'] == 'selected.pt',
                'Outer requires fresh fixed-E selected.pt, no outer validation')


def validate_reuse(profile, result, inputs, history, fold, bank, E):
    """Identity/recipe checks only; inherit the completed raw-logit selection audit."""
    check_fit_result(result, 'inner_select', fold['fold'], 20)
    require(result['selected_epochs'] == E and profile['head'] == 'linear' and profile['head_parameters'] == 513
            and profile['seed'] == 7 and profile['stage'] == 'inner_select' and profile['fold'] == fold['fold']
            and profile['initial_state_sha256'] == INITIAL_SHA and profile['backbone_state_sha256'] == BACKBONE_SHA
            and profile['model_class'] == 'experiments.common.video.head_comparison_training.LinearVideoClassifier'
            and profile['mode'] == 'rgb_zero' and profile['dataloader_generator_seed'] == 100007,
            'Wrong old Linear role, architecture or public initialization')
    expected, val = fold['inner_train']['epoch_clip_ids'], fold['inner_val']['clip_ids']
    require(inputs['epoch_clip_ids'] == expected and inputs['validation_clip_ids'] == val
            and len(history) == 20, 'Reused inner schedule/validation mismatch')
    for cid in set(c for order in expected for c in order) | set(val):
        require(all(inputs['clip_bank'][cid][k] == bank[cid][k] for k in
                    ('unit_id', 'group_id', 'label', 'frame_indices', 'source_pts_seconds')), 'Reused inner clip identity changed')
    for epoch, row in enumerate(history, 1):
        require(row['completed_epochs'] == epoch and row['train_clip_ids'] == expected[epoch - 1]
                and row['validation_clip_ids'] == val and row['optimizer_steps'] == math.ceil(len(expected[epoch - 1]) / 24)
                and math.isfinite(row['validation_unweighted_bce']), 'Reused inner exposure/selection log changed')
    require(E == min(range(1, 21), key=lambda e: history[e - 1]['validation_unweighted_bce']), 'Old recorded earliest minimum E differs')
    require(profile['optimizer'] == dict(name='Adam', lr_backbone=1e-4, lr_head=1e-3, betas=[.9,.999], eps=1e-8,
                                       weight_decay=0, initial_state_entries=0)
            and profile['effective_batch'] == 24 and profile['microbatch'] == 8
            and profile['environment']['precision'] == 'FP32' and profile['environment']['amp'] is False,
            'Reused inner optimization recipe differs')


def validate_roles(folds, bank, sources):
    held = []
    require(len(folds) == 5, 'Five fixed folds required')
    for i, (fold, expected) in enumerate(zip(folds, COUNTS), 1):
        require(fold['fold'] == i, 'Fold order changed')
        validate_partitions({k:fold[k]['video_ids'] for k in ('inner_train','inner_val','outer_train','outer_eval')}, sources)
        for role, pair in [('inner_train', expected[:2]), ('outer_train', expected[4:6])]:
            require(len(fold[role]['epoch_clip_ids']) == 20, 'Frozen full20 schedule required')
            for order in fold[role]['epoch_clip_ids']:
                counts = Counter(bank[c]['label'] for c in order)
                require((counts[1], counts[0]) == pair, 'Role-specific train N/P differs')
        val = Counter(bank[c]['label'] for c in fold['inner_val']['clip_ids'])
        labels = Counter(sources[u]['video_label'] for u in fold['outer_eval']['video_ids'])
        require((val[1], val[0]) == expected[2:4] and (labels['pos'], labels['neg']) == expected[6:], 'Validation/outer label counts differ')
        held.extend(fold['outer_eval']['video_ids'])
    require(len(held) == len(set(held)) == 150 and set(held) == set(sources), 'Each outer video must appear exactly once')


def prepare(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.relative_to(root / 'data/metadata/EXP-022')
    require(not out.exists(), 'Refuse plan overwrite')
    require(_sha(root / CARD) == CARD_SHA and _sha(root / AUDIT) == AUDIT_SHA, 'Approved card/deep audit changed')
    original, audit = read(root / ORIGINAL), read(root / AUDIT)
    require(audit['status'] == 'audit_passed', 'Old deep-head audit not passed')
    fingerprints = {CARD:CARD_SHA, AUDIT:AUDIT_SHA, ORIGINAL:_sha(root / ORIGINAL), INIT:_sha(root / INIT), REVIEW20:_sha(root / REVIEW20)}
    governance = []
    for name, fingerprint in original['input_fingerprints'].items():
        if name.startswith('docs/') or name.startswith('.agents/'):
            snapshot = root / 'data/results/exp017/cv-s7-20261102-v1/source-snapshot' / name
            require(_sha(snapshot) == fingerprint, 'Historical source snapshot no longer matches')
            governance.append(dict(path=name, historical_sha256=fingerprint, historical_snapshot=str(snapshot.relative_to(root)).replace('\\','/'),
                                   current_sha256=_sha(root / name), interpretation='Explicit historical inheritance; old run bytes unchanged'))
            fingerprints[name] = _sha(root / name)
        else:
            require(_sha(rooted(root, name)) == fingerprint, 'Inherited scientific input/source changed: ' + name)
            fingerprints[name] = fingerprint
    proof = read(root / INIT)
    require(proof['status'] == 'production_initialization_verified' and
            proof['initializations']['linear-s7']['initial_state_sha256'] == INITIAL_SHA and
            proof['initializations']['linear-s7']['backbone_state_sha256'] == BACKBONE_SHA, 'Original production Linear proof differs')
    for name, fingerprint in proof['source_sha256'].items():
        require(_sha(root / name) == fingerprint, 'Original Linear construction changed')
        fingerprints[name] = fingerprint
    require(read(root / REVIEW20)['status'] == 'independent_result_and_formal_report_review_passed', 'Old per-fit result review missing')
    p, bank, _, _, folds = load_cv_plan(root, root / BASE, verify_sources=False)
    require(_sha(root / BASE / 'plan.json') == BASE_SHA, 'Unregistered base plan')
    with (root / BASE / 'sources.csv').open(encoding='utf-8-sig', newline='') as stream:
        sources = {r['unit_id']: r for r in csv.DictReader(stream)}
    validate_roles(folds, bank, sources)
    require(epoch_block(root / 'experiments/common/video/linear_cv_training.py') ==
            epoch_block(root / 'experiments/common/video/cv_supervised_training.py'), 'Frozen epoch block changed')
    references = {}
    for fold, (E, closure_sha) in REUSE.items():
        item = 'f%02d-linear-s7' % fold
        closure_name = 'closed-' + item + '.json'
        closure_path = root / LOCAL20 / closure_name
        require(_sha(closure_path) == closure_sha, 'Reused inner closure differs')
        closure = read(closure_path)
        require(closure['status'] == 'fit_closed' and closure['item'] == item, 'Old per-fit closure not complete')
        files = {r['path']:r for r in closure['files']}
        directory = root / LOCAL20 / item
        for name in ('profile.json','training-result.json','inputs.json','epochs.jsonl'):
            ref = files[item + '/' + name]
            require(_sha(directory / name) == ref['sha256'], 'Old inner metadata changed')
        result, profile, inputs = [read(directory / name) for name in ('training-result.json','profile.json','inputs.json')]
        history = [json.loads(line) for line in (directory / 'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
        validate_reuse(profile, result, inputs, history, folds[fold - 1], bank, E)
        require(result['selected_checkpoint']['sha256'] == files[item + '/best.pt']['sha256'], 'Reused best provenance differs')
        references[str(fold)] = dict(item=item, selected_epochs=E, closure_path=closure_name,
            closure_sha256=closure_sha, files={k:v for k,v in files.items() if k.rsplit('/',1)[-1] in
            ('profile.json','training-result.json','inputs.json','epochs.jsonl','best.pt')},
            checkpoint_validation_scope='Inherited 210-file collector byte verification and per-fit audit; no new model-byte hash')
        del inputs, history
    deep = []
    for item in audit['fits']:
        name = item['artifact_directory'] + '/dense/result.json'
        require(_sha(root / name) == audit['source_sha256'][name], 'Audited deep dense result changed')
        raw = read(root / name)
        require(raw['status'] == 'outer_fold_complete' and raw['fold'] == item['fold'], 'Wrong deep fold source')
        require(video_metrics(raw['scores']) == raw['metrics'], 'Deep saved-probability metrics differ')
        deep.append(dict(fold=item['fold'], scores=raw['scores'], metrics=raw['metrics'], selected_epochs=item['selected_epochs'],
            source_run_id=item['artifact_directory'].split('/')[3], artifact_directory=item['artifact_directory'],
            result_path=name, result_sha256=audit['source_sha256'][name],
            window_scores_sha256=raw['window_scores_sha256'], source='EXP017 audited original predictions, never rescored'))
    summarize_folds(deep, sources)
    for name in code_paths(root):
        fingerprints[name] = _sha(root / name)
    out.mkdir(parents=True, exist_ok=False)
    _json(out / 'deep-baseline.json', dict(folds=deep, audit_path=AUDIT, audit_sha256=AUDIT_SHA))
    plan = dict(spec_id='EXP-022', version=1, status='prepared_not_training_authorization', seed=7,
        base_plan_path=BASE, base_plan_sha256=BASE_SHA, source_fingerprints=fingerprints, historical_governance=governance,
        inherited_acceptance=ACCEPTANCE, dense_reference=DENSE, initialization_proof=INIT,
        expected_initial_state_sha256=INITIAL_SHA, expected_backbone_state_sha256=BACKBONE_SHA,
        reference_roots=dict(local=LOCAL20, cloud=CLOUD20), reused_inner=references,
        deep_baseline_sha256=_sha(out / 'deep-baseline.json'), ffmpeg_sha256=original['ffmpeg_sha256'],
        epoch_execution_sha256=hashlib.sha256(epoch_block(root / 'experiments/common/video/linear_cv_training.py').encode()).hexdigest(),
        new_inner_folds=[2,3,5], outer_folds=[1,2,3,4,5], inner_new_epochs=60, inner_new_updates=1040,
        maximum_new_epochs=136, maximum_new_updates=2712, dense_videos=150, dense_physical_windows=143470,
        outer_E_rule='own inner earliest minimum unweighted BCE, then fresh public K400 prefix; no outer selection',
        pixel_acceptance='Existing actual JPEG/RGB proof; each consumed JPEG still checked; no repeated global pixel scan',
        independent_test=False, forecast_policy='advisory_only')
    _json(out / 'plan.json', plan)
    require(all(_sha(root / name) == fingerprint for name, fingerprint in fingerprints.items()), 'Source changed during metadata prepare')
    return plan


def load(root, plan_dir, reference_root, with_dense=False):
    root, plan_dir, reference_root = Path(root).resolve(), Path(plan_dir).resolve(), Path(reference_root).resolve()
    plan_dir.relative_to(root)
    plan = read(plan_dir / 'plan.json')
    require(plan['spec_id'] == 'EXP-022' and plan['seed'] == 7 and plan['base_plan_sha256'] == BASE_SHA,
            'Unregistered Linear CV plan')
    require(reference_root in {(root / plan['reference_roots']['local']).resolve(), Path(plan['reference_roots']['cloud']).resolve()}, 'Wrong explicit reference root')
    for name, fingerprint in plan['source_fingerprints'].items():
        require(_sha(rooted(root, name)) == fingerprint, 'Frozen input/source changed: ' + name)
    require(_sha(plan_dir / 'deep-baseline.json') == plan['deep_baseline_sha256'], 'Derived audited deep baseline changed')
    _, bank, _, _, folds = load_cv_plan(root, root / BASE, verify_sources=False)
    with (root / BASE / 'sources.csv').open(encoding='utf-8-sig', newline='') as stream:
        sources = {r['unit_id']:r for r in csv.DictReader(stream)}
    validate_roles(folds, bank, sources)
    fingerprints = dict(plan['source_fingerprints'])
    for name in ('plan.json','deep-baseline.json'):
        fingerprints[(plan_dir / name).relative_to(root).as_posix()] = _sha(plan_dir / name)
    acceptance = read(root / ACCEPTANCE)
    require(acceptance['status'] == 'cv_supervised_cache_ready' and acceptance['pixel_version'] == PIXEL_VERSION
            and acceptance['actual_JPEG_byte_SHA_checked'] is True and acceptance['actual_RGB112_pixel_SHA_checked'] is True
            and acceptance['required_frame_count'] == acceptance['verified_required_frame_count'] == 191812
            and acceptance['failed_required_frames'] == 0, 'Existing actual cache acceptance missing')
    context = dict(root=root, plan_dir=plan_dir, plan=plan, bank=bank, folds=folds, sources=sources,
                   fingerprints=fingerprints, reference_root=reference_root, proof=acceptance, dense=None)
    if with_dense:
        reference = read(root / DENSE)
        require(reference['status'] == 'cv_dense_pixels_verified' and reference['video_count'] == 150
                and reference['windows'] == 143470 and set(reference['videos']) == set(sources), 'Incomplete dense reference')
        grouped, index = {uid:[] for uid in sources}, {}
        with (root / reference['grid_path']).open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line); grouped[row['unit_id']].append(row)
        require(all(len(grouped[u]) == reference['videos'][u]['windows'] for u in sources), 'Dense physical grid counts differ')
        with (root / acceptance['verified_cache_index']).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                key = row['unit_id'], int(row['frame_index'])
                require(key not in index, 'Duplicate accepted physical frame')
                index[key] = row
        context['dense'] = dict(path=root / DENSE, reference=reference, windows=grouped)
        context['cache_index'] = index
    return context


def frame_refs(context, fold, stage):
    f, bank = context['folds'][fold - 1], context['bank']
    require(stage in ('inner_select','fixed_outer'), 'Wrong frame stage')
    role = f['inner_train' if stage == 'inner_select' else 'outer_train']
    clips = {c for order in role['epoch_clip_ids'] for c in order}
    if stage == 'inner_select': clips.update(f['inner_val']['clip_ids'])
    required = {(bank[c]['unit_id'], i) for c in clips for i in bank[c]['frame_indices']}
    refs = {}
    with (context['root'] / context['proof']['verified_required_frames_index']).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            key = row['unit_id'], int(row['frame_index'])
            if key not in required: continue
            require(key not in refs and row['status'] == 'verified' and row['rgb112_shape'] == '112x112x3'
                    and row['actual_jpeg_sha256'] == row['jpeg_sha256'] and row['actual_rgb112_sha256'] == row['rgb112_sha256'],
                    'Required accepted frame differs')
            refs[key] = FrameRef(str(rooted(context['root'], row['cache_path'])), row['jpeg_sha256'], row['rgb112_sha256'])
    require(set(refs) == required, 'Missing accepted role frames')
    return refs


def reused_inner(context, fold):
    require(str(fold) in context['plan']['reused_inner'], 'No registered old inner for this fold')
    ref = context['plan']['reused_inner'][str(fold)]
    directory = context['reference_root'] / ref['item']
    require(_sha(context['reference_root'] / ref['closure_path']) == ref['closure_sha256'], 'Old per-fit closure changed')
    for name, record in ref['files'].items():
        path = context['reference_root'] / name
        require(path.stat().st_size == record['size_bytes'], 'Old inner file size changed')
        if path.suffix != '.pt': require(_sha(path) == record['sha256'], 'Old inner metadata SHA changed')
    result = read(directory / 'training-result.json')
    check_fit_result(result, 'inner_select', fold, 20)
    require(result['selected_epochs'] == ref['selected_epochs'], 'Reused inner E changed')
    return result


def train_stage(context, out, fold, stage, inner_result=None):
    require(type(fold) is int and 1 <= fold <= 5 and stage in ('inner_select','fixed_outer'), 'Wrong fit request')
    f = context['folds'][fold - 1]
    kwargs = dict(stage=stage, fold=fold, bank=context['bank'], verified_frame_index=frame_refs(context, fold, stage),
        input_fingerprints=context['fingerprints'], expected_initial_state_sha256=INITIAL_SHA, device='cuda', workers=2, cache_mib=128)
    if stage == 'inner_select':
        require(fold in context['plan']['new_inner_folds'] and inner_result is None, 'Do not retrain registered old inner')
        kwargs.update(epoch_clip_ids=f['inner_train']['epoch_clip_ids'], validation_clip_ids=f['inner_val']['clip_ids'], max_epochs=20)
        epochs = 20
    else:
        require(inner_result is not None, 'Outer requires its own completed inner')
        check_fit_result(inner_result, 'inner_select', fold, 20)
        epochs = inner_result['selected_epochs']
        kwargs.update(epoch_clip_ids=f['outer_train']['epoch_clip_ids'], selected_epochs=epochs)
    result = fit(context['root'], out, **kwargs)
    check_fit_result(result, stage, fold, epochs)
    profile = read(Path(out) / 'profile.json')
    require(profile['head'] == 'linear' and profile['execution_kind'] == 'registered_fit', 'Unregistered/fixture fit')
    return result


def comparison(results, baseline, sources):
    linear, deep = summarize_folds(results, sources), summarize_folds(baseline['folds'], sources)
    require(all({r['unit_id'] for r in new['scores']} == {r['unit_id'] for r in old['scores']}
                for new, old in zip(results, baseline['folds'])), 'Matched heads must evaluate the same videos in each fold')
    old = {row['unit_id']:row for fold in baseline['folds'] for row in fold['scores']}
    new = {row['unit_id']:row for fold in results for row in fold['scores']}
    changes = [dict(unit_id=u, group_id=sources[u]['group_id'], label=new[u]['label'],
                    deep_score=old[u]['score'], linear_score=new[u]['score'],
                    deep_prediction=int(old[u]['score'] >= .5), linear_prediction=int(new[u]['score'] >= .5))
               for u in sorted(new) if (old[u]['score'] >= .5) != (new[u]['score'] >= .5)]
    groups = sorted(linear['negative_group_metrics'])
    require(groups == sorted(deep['negative_group_metrics']) and len(groups) == 9, 'Nine fixed negative groups required')
    group_rows = [dict(group_id=g, deep_FPR=deep['negative_group_metrics'][g]['FPR'],
                       linear_FPR=linear['negative_group_metrics'][g]['FPR']) for g in groups]
    macro_new = statistics.mean(r['linear_FPR'] for r in group_rows)
    macro_old = statistics.mean(r['deep_FPR'] for r in group_rows)
    improved = [r['group_id'] for r in group_rows if r['linear_FPR'] < r['deep_FPR']]
    passed = linear['FP'] < deep['FP'] and linear['FN'] <= deep['FN'] and macro_new < macro_old and len(improved) >= 2
    return dict(status='five_fold_comparison_complete_pending_independent_audit', linear=linear, deep=deep,
        same_video_prediction_changes=changes, negative_groups=group_rows,
        linear_macro_negative_group_FPR=macro_new, deep_macro_negative_group_FPR=macro_old,
        improved_negative_groups=improved, supports_overall_head_simplification_video_improvement=passed,
        metric_kind='Per-fold AUC of max saved FP32 window probability, ties retained; no pooled AUC',
        no_automatic_followup=True, independent_test=False)
