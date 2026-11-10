"""EXP-017 continuation: immutable fold-1 evidence, fresh folds 2--5 only."""
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
import time

from experiments.common.video.cv_execution import INITIAL_SHA, PLAN_SHA, check_fit_result, summarize_folds, train_stage
from experiments.common.video.cv_dense_evaluation import evaluate_fold
from experiments.common.video.cv_supervised_training import _canonical
from experiments.common.video.dense_evaluation import video_metrics
from experiments.common.video.pts_clip_sampling import tuple_digest
from experiments.common.video.supervised_plan import csv_rows, digest, jsonl, write_json

PARENT_NAME = 'cv-s7-20261102-v1'
REFUSAL = 'Complete remaining folds do not fit; no shortening, fold selection, or automatic rerental'


def require(value, message):
    if not value:
        raise ValueError(message)


def resume_sources(root):
    names = ['experiments/common/video/cv_resume_execution.py',
             'experiments/runspecs/EXP-017/resume.py', 'experiments/runspecs/EXP-017/resume.sh']
    return {n: digest(plain(root, n)) for n in names}


def plain(root, name):
    """Canonical repository-relative regular paths; reject symlink components."""
    p = PurePosixPath(name)
    require(isinstance(name, str) and str(p) == name and name and not p.is_absolute()
            and '\\' not in name and ':' not in name and '..' not in p.parts, 'Unsafe evidence path')
    path = Path(root)
    for part in p.parts:
        path = path / part
        require(not path.is_symlink(), 'Symlink evidence is not accepted')
    return path


def artifact_inventory(parent):
    rows = []
    for p in sorted(Path(parent).rglob('*')):
        require(not p.is_symlink(), 'Parent contains a symlink')
        if p.is_dir():
            continue
        require(stat.S_ISREG(p.lstat().st_mode), 'Parent has a nonregular artifact')
        rows.append(dict(path=p.relative_to(parent).as_posix(), size_bytes=p.stat().st_size, sha256=digest(p)))
    require(bool(rows), 'Empty parent')
    return rows


def verify_inventory(parent, rows):
    expected = {r['path']: r for r in rows}
    require(len(expected) == len(rows) and bool(rows), 'Duplicate/empty parent identity list')
    paths = list(Path(parent).rglob('*'))
    require(not any(p.is_symlink() for p in paths), 'Parent contains a symlink')
    actual = {p.relative_to(parent).as_posix() for p in paths if not p.is_dir()}
    require(actual == set(expected), 'Parent artifact set changed')
    for name, r in expected.items():
        p = plain(parent, name)
        require(stat.S_ISREG(p.lstat().st_mode) and p.stat().st_size == int(r['size_bytes'])
                and digest(p) == r['sha256'], 'Parent artifact SHA/size changed: ' + name)
    return {n: r['sha256'] for n, r in expected.items()}


def checked_parsed_inputs(context, path, fold, role, validation, fingerprints):
    """One parsed JSON at a time; semantic digest is distinct from its byte SHA."""
    saved = json.loads(Path(path).read_text(encoding='utf-8'))
    orders = fold[role]['epoch_clip_ids']
    require(saved['epoch_clip_ids'] == orders and saved['validation_clip_ids'] == validation,
            'Parent parsed schedule/validation differs')
    ids = set(c for order in orders for c in order) | set(validation or [])
    require(set(saved['clip_bank']) == ids, 'Parent parsed clip bank differs')
    for cid, row in saved['clip_bank'].items():
        w = context['bank'][cid]
        require(all(row[k] == w[k] for k in ('unit_id', 'group_id', 'label', 'frame_indices', 'source_pts_seconds')),
                'Parent parsed window identity differs')
        require(len(row['frames']) == 16, 'Parent parsed frame count differs')
        for i, actual in zip(w['frame_indices'], row['frames']):
            ref = context['refs'][(w['unit_id'], i)]
            relative = Path(ref.path).relative_to(context['root']).as_posix()
            require(actual == dict(path='/root/autodl-tmp/debris-flow-thesis-exp016/' + relative,
                jpeg_sha256=ref.jpeg_sha256, rgb112_sha256=ref.rgb112_sha256), 'Parent FrameRef/byte/pixel identity differs')
    return _canonical(dict(metadata=saved['clip_bank'], epoch_clip_ids=orders,
                           validation_clip_ids=validation, files=fingerprints))


def validate_first_fold(context, parent, frozen, hashes):
    """No weights unpickled or huge parsed-input JSON loaded; audit recorded evidence."""
    def read(name):
        require(name in hashes, 'Missing bound parent evidence: ' + name)
        return json.loads(plain(parent, name).read_text(encoding='utf-8'))
    failure = read('pipeline-failure.json')
    require(failure.get('status') == 'incomplete_five_fold_metrics_withheld'
            and failure.get('error_type') == 'RuntimeError' and failure.get('error') == REFUSAL,
            'Only the registered forecast-only parent stop may be reused')
    require(not any(n.startswith('fold-') and not n.startswith('fold-01/') for n in hashes), 'Parent contains another fold')
    require(not any('failure.json' in n and n != 'pipeline-failure.json' or 'hard-stop' in n or 'hard-time-limit' in n
                    or n.endswith('.part') for n in hashes), 'Parent contains incomplete/failed stage evidence')
    forecast = read('fold-01/remaining-budget.json')
    require(forecast['status'] == 'refused' and math.isfinite(forecast['seconds'])
            and math.isfinite(forecast['remaining_seconds']) and forecast['seconds'] > forecast['remaining_seconds'],
            'Parent refusal timing evidence differs')
    for name, sha in frozen['code_sha256'].items():
        require(hashes.get('source-snapshot/' + name) == sha, 'Parent source snapshot differs')
    f = context['folds'][0]
    inner = read('fold-01/inner/training-result.json')
    check_fit_result(inner, 'inner_select', 1, 20)
    E = inner['selected_epochs']
    outer = read('fold-01/outer/training-result.json')
    check_fit_result(outer, 'fixed_outer', 1, E)
    losses = []
    for stage, role, epochs, result in [('inner', 'inner_train', 20, inner), ('outer', 'outer_train', E, outer)]:
        prefix = 'fold-01/' + stage + '/'
        profile = read(prefix + 'profile.json')
        require(hashes[prefix + 'profile.json'] == result['profile_sha256'], 'Parent profile byte SHA differs')
        parsed_sha = checked_parsed_inputs(context, plain(parent, prefix + 'inputs.json'), f, role,
            f['inner_val']['clip_ids'] if stage == 'inner' else None, frozen['input_fingerprints'])
        require(parsed_sha == result['parsed_inputs_sha256'] == profile['parsed_inputs_sha256'],
                'Parent canonical parsed input SHA differs')
        require(profile['execution_kind'] == 'registered_fit' and profile['stage'] == result['stage']
                and profile['fold'] == 1 and profile['seed'] == 7 and profile['mode'] == 'rgb_zero'
                and profile['initial_state_sha256'] == profile['expected_initial_state_sha256'] == INITIAL_SHA
                and profile['effective_batch'] == 24 and profile['microbatch'] == 8
                and profile['checkpoint_backbone'] is False and profile['requested_epochs'] == epochs
                and profile['input_fingerprints'] == result['input_fingerprints'] == frozen['input_fingerprints'],
                'Parent fit recipe/input/initialization differs')
        history = jsonl(plain(parent, prefix + 'epochs.jsonl'))
        require(len(history) == epochs, 'Incomplete parent epoch history')
        updates = 0
        for i, row in enumerate(history):
            ids = f[role]['epoch_clip_ids'][i]
            P = sum(context['bank'][c]['label'] for c in ids); N = len(ids) - P
            steps = math.ceil(len(ids) / 24); updates += steps
            require(row['completed_epochs'] == i + 1 and row['train_clip_ids'] == ids and row['samples'] == len(ids)
                    and row['train_P'] == P and row['train_N'] == N and row['optimizer_steps'] == steps
                    and row['total_updates'] == updates and row['finite'] is True
                    and math.isfinite(row['train_weighted_bce']) and row['initial_state_sha256'] == INITIAL_SHA
                    and row['parsed_inputs_sha256'] == parsed_sha and row['profile_sha256'] == result['profile_sha256'],
                    'Parent schedule/finite/update evidence differs')
            if stage == 'inner':
                pred = read(prefix + 'validation-epoch-%02d.json' % (i + 1))
                require([r['clip_id'] for r in pred] == f['inner_val']['clip_ids']
                        and all(r['label'] == context['bank'][r['clip_id']]['label'] and math.isfinite(r['logit']) for r in pred),
                        'Parent validation identities differ')
                value = math.fsum(max(r['logit'], 0) - r['label'] * r['logit']
                                 + math.log1p(math.exp(-abs(r['logit']))) for r in pred) / len(pred)
                recorded = row['validation_unweighted_bce']
                require(math.isfinite(recorded) and recorded >= 0 and math.isfinite(value)
                        and abs(value - recorded) <= 1e-5, 'Parent validation BCE differs')
                losses.append(recorded)
            else:
                require(not any(k.startswith('validation') for k in row), 'Parent outer used validation')
        require(updates == result['optimizer_steps'] and math.isclose(profile['supervised_pos_weight'], N / P, abs_tol=1e-12),
                'Parent updates/N/P differ')
        for key in ('selected_checkpoint', 'last_checkpoint'):
            ref = result[key]
            require(ref['completed_epochs'] == (E if key == 'selected_checkpoint' else epochs)
                    and hashes.get(prefix + ref['filename']) == ref['sha256'], 'Parent checkpoint identity differs')
    require(min(range(20), key=lambda i: losses[i]) + 1 == E, 'Parent E is not earliest minimum')
    dense = read('fold-01/dense/result.json'); uids = set(f['outer_eval']['video_ids'])
    require(dense['status'] == 'outer_fold_complete' and dense['fold'] == 1 and dense['selected_epochs'] == E
            and dense['plan_sha256'] == PLAN_SHA and dense['checkpoint_sha256'] == outer['selected_checkpoint']['sha256']
            and dense['input_fingerprints'] == frozen['input_fingerprints'] and dense['threshold'] == .5
            and dense['aggregation'] == 'max' and dense['outer_selection_use'] is False
            and dense['independent_test_used'] is False and dense['pixel_reference_sha256'] == frozen['dense_reference_sha256']
            and hashes['fold-01/dense/window-scores.csv'] == dense['window_scores_sha256'], 'Parent dense contract differs')
    require(len(dense['scores']) == len(uids) and {r['unit_id'] for r in dense['scores']} == set(dense['videos']) == uids,
            'Parent outer video coverage differs')
    grid = {(u, tuple_digest(w['frame_indices'])): w for u in uids for w in context['dense']['windows'][u]}
    seen = set(); maxima = {}; counts = {}
    for row in csv_rows(plain(parent, 'fold-01/dense/window-scores.csv')):
        u = row['unit_id']; key = (u, row['frame_tuple_sha256']); value = float(row['score'])
        require(key in grid and key not in seen and math.isfinite(value) and 0 <= value <= 1, 'Parent physical dense coverage differs')
        w = grid[key]
        require(all(math.isfinite(float(row[k])) and abs(float(row[k]) - expected) <= 1e-8 for k, expected in
                    [('target_start_seconds', w['target_start_seconds']), ('start', w['source_pts_seconds'][0]), ('end', w['source_pts_seconds'][-1])]),
                'Parent dense timing differs')
        seen.add(key); maxima[u] = max(maxima.get(u, 0.), value); counts[u] = counts.get(u, 0) + 1
    require(seen == set(grid), 'Parent dense grid has missing physical windows')
    for row in dense['scores']:
        u = row['unit_id']; source = context['sources'][u]; evidence = dense['videos'][u]
        reference = context['dense']['reference']['videos'][u]
        require(row['label'] == int(source['video_label'] == 'pos') and row['group_id'] == source['group_id']
                and row['fold'] == 1 and row['score'] == maxima[u] and row['windows'] == counts[u]
                and evidence['windows'] == counts[u] and evidence['score'] == maxima[u]
                and evidence['decoded'] == reference['decoded'] and evidence['canonical_reference_matched'] is True
                and evidence['rgb112_digest'] == reference['rgb112_digest'], 'Parent video/EOF/pixel/max closure differs')
    require(video_metrics(dense['scores']) == dense['metrics'], 'Parent dense metrics differ')
    return dict(inner=inner, outer=outer, dense=dense)


def bind_context(context, frozen, root, preflight, manifest):
    require(digest(manifest) == frozen['transfer_manifest_sha256'], 'Original deployment manifest changed')
    proof = json.loads(Path(preflight).read_text(encoding='utf-8'))
    require(proof.get('status') == 'cv_static_preflight_passed' and proof.get('files_verified') == 192461
            and all(proof.get(k) == v for k, v in frozen.items() if k != 'input_fingerprints'), 'Original static preflight differs')
    for name, sha in frozen['code_sha256'].items():
        require(digest(plain(root, name)) == sha, 'Original execution source changed: ' + name)
    context['fingerprints'].update(frozen['code_sha256'])
    require(context['fingerprints'] == frozen['input_fingerprints'], 'Continuation inputs differ from parent')
    for name, sha in context['fingerprints'].items():
        require(digest(plain(root, name)) == sha, 'Bound input changed: ' + name)


def prepare_parent(context, parent, preflight, manifest, out):
    root = context['root']; parent = Path(parent).resolve(); out = Path(out).resolve()
    require(parent == root / 'data/results/exp017' / PARENT_NAME, 'Unregistered parent location')
    require(not out.exists(), 'Parent binding output already exists')
    frozen = json.loads((parent / 'execution-inputs.json').read_text(encoding='utf-8'))
    bind_context(context, frozen, root, preflight, manifest)
    rows = artifact_inventory(parent); hashes = {r['path']: r['sha256'] for r in rows}
    first = validate_first_fold(context, parent, frozen, hashes)
    require(len(rows) == 94 and sum(r['size_bytes'] for r in rows) == 1625434039
            and len(first['dense']['scores']) == 33 and sum(r['windows'] for r in first['dense']['scores']) == 22781,
            'Registered recovered parent population differs')
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'files.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'size_bytes', 'sha256']); writer.writeheader(); writer.writerows(rows)
    binding = dict(status='parent_fold1_reuse_verified', schema_version=1, parent_run_id=PARENT_NAME,
        parent_path=parent.relative_to(root).as_posix(), parent_execution_inputs_sha256=hashes['execution-inputs.json'],
        files_sha256=digest(out / 'files.csv'), file_count=len(rows), bytes=sum(r['size_bytes'] for r in rows),
        preflight_sha256=digest(preflight), transfer_manifest_sha256=digest(manifest), plan_sha256=PLAN_SHA,
        selected_epochs=first['inner']['selected_epochs'], inner_updates=first['inner']['optimizer_steps'],
        outer_updates=first['outer']['optimizer_steps'], dense_videos=33, dense_windows=22781,
        reuse_folds=[1], execute_folds=[2, 3, 4, 5], parent_checkpoints_loaded=False,
        resume_code_sha256=resume_sources(root),
        origin='Same preregistered five folds; immutable evidence reused, no parent weights/optimizer inherited')
    write_json(out / 'parent.json', binding)
    return binding


def load_parent(context, binding_path, binding_sha, preflight, manifest):
    require(digest(binding_path) == binding_sha, 'Parent binding SHA differs')
    b = json.loads(Path(binding_path).read_text(encoding='utf-8')); root = context['root']
    require(b.get('status') == 'parent_fold1_reuse_verified' and b.get('parent_run_id') == PARENT_NAME
            and b.get('plan_sha256') == PLAN_SHA and b.get('reuse_folds') == [1] and b.get('execute_folds') == [2, 3, 4, 5]
            and digest(preflight) == b['preflight_sha256'] and digest(manifest) == b['transfer_manifest_sha256'], 'Invalid parent reuse binding')
    require(resume_sources(root) == b['resume_code_sha256'], 'Continuation controller changed after binding')
    parent = plain(root, b['parent_path'])
    require(parent == root / 'data/results/exp017' / PARENT_NAME, 'Wrong parent path')
    csv_path = Path(binding_path).with_name('files.csv')
    require(digest(csv_path) == b['files_sha256'], 'Parent identity CSV differs')
    rows = csv_rows(csv_path); hashes = verify_inventory(parent, rows)
    require(len(rows) == b['file_count'] == 94 and sum(int(r['size_bytes']) for r in rows) == b['bytes'] == 1625434039
            and hashes['execution-inputs.json'] == b['parent_execution_inputs_sha256'], 'Recovered parent identity differs')
    frozen = json.loads((parent / 'execution-inputs.json').read_text(encoding='utf-8'))
    bind_context(context, frozen, root, preflight, manifest)
    first = validate_first_fold(context, parent, frozen, hashes)
    require((first['inner']['selected_epochs'], first['inner']['optimizer_steps'], first['outer']['optimizer_steps'],
             len(first['dense']['scores']), sum(r['windows'] for r in first['dense']['scores']))
            == (b['selected_epochs'], b['inner_updates'], b['outer_updates'], b['dense_videos'], b['dense_windows']),
            'Parent reuse counts/selection changed')
    return b, frozen, first, rows


def continuation_budget(now, start, platform_end, price, prior_gpu, prior_cpu, prior_stage, prior_total):
    values = (now, start, platform_end, price, prior_gpu, prior_cpu, prior_stage, prior_total)
    require(all(math.isfinite(x) for x in values), 'Nonfinite budget')
    require(0 < price <= 1.48 and prior_gpu >= 2154 and 6325 <= prior_cpu <= 7200
            and 1.05 <= prior_stage < 7 and 2.87 + prior_stage - 1e-8 <= prior_total < 20
            and start <= now + 5, 'Invalid cumulative budget/actual start')
    cpu_reserve = (7200 - prior_cpu) * .1 / 3600
    money = min(7 - prior_stage, 20 - prior_total) - cpu_reserve - .1
    fee_end = start + money / price * 3600
    require(now < platform_end <= fee_end and fee_end - platform_end < 60.001,
            'Platform timer must be the fee deadline rounded down by at most60seconds')
    end = platform_end - 60
    require(now < end, 'Actual authorized fee/time boundary reached')
    return dict(deadline_unix=end, fee_ceiling_unix=fee_end, platform_deadline_unix=platform_end,
        actual_gpu_start_unix=start, hourly_price=price, prior_gpu_seconds=prior_gpu, prior_cpu_seconds=prior_cpu,
        prior_stage_yuan=prior_stage, prior_total_yuan=prior_total, stage_ceiling=7., total_ceiling=20.,
        remaining_cpu_cost_reserve=cpu_reserve, shutdown_cost_reserve_yuan=.1, shutdown_reserve_seconds=60,
        gpu_plan_target_seconds=16200, gpu_target_policy='advisory_only_not_a_stop_gate',
        elapsed_time_scope='Conservative request-to-settled-row upper bounds, not exact billed seconds')


def run_remaining(context, out, first, ffmpeg, budget, forecast, *, train=train_stage, evaluate=evaluate_fold):
    """Timing notices never gate a fold. Only the actual authorized boundary stops."""
    completed = [first]; origins = [dict(fold=1, reused=True, run_id=PARENT_NAME,
        path='data/results/exp017/' + PARENT_NAME + '/fold-01')]
    require([f['fold'] for f in context['folds']] == [1, 2, 3, 4, 5], 'Changed fold order')
    for f in context['folds'][1:]:
        require(time.time() < budget['deadline_unix'], 'Actual authorized fee/time boundary reached')
        number = f['fold']; directory = Path(out) / ('fold-%02d' % number)
        directory.mkdir()
        notice = dict(status='advisory_only', forecast_can_stop_run=False,
                      remaining_seconds=budget['deadline_unix'] - time.time(), **forecast(completed, context['folds'][number-1:], context['dense']['windows']))
        notice['forecast_exceeds_remaining'] = notice['seconds'] > notice['remaining_seconds']
        write_json(directory / 'timing-advisory.json', notice)
        print(json.dumps(dict(event='timing_advisory', fold=number, **notice), allow_nan=False), flush=True)
        inner = train(context, directory / 'inner', number, 'inner_select')
        require(time.time() < budget['deadline_unix'], 'Actual authorized fee/time boundary reached')
        outer = train(context, directory / 'outer', number, 'fixed_outer', inner)
        require(time.time() < budget['deadline_unix'], 'Actual authorized fee/time boundary reached')
        dense = evaluate(context, directory / 'dense', number, directory / 'outer', inner['selected_epochs'], ffmpeg)
        completed.append(dict(inner=inner, outer=outer, dense=dense))
        origins.append(dict(fold=number, reused=False, run_id=Path(out).name,
                            path=directory.relative_to(context['root']).as_posix()))
    require(time.time() < budget['deadline_unix'], 'Actual authorized fee/time boundary reached')
    require(sum(r['windows'] for x in completed for r in x['dense']['scores']) == 143470, 'Incomplete143470-window coverage')
    return summarize_folds([x['dense'] for x in completed], context['sources']), completed, origins
