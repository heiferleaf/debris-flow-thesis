"""Read-only audit of immutable parent fold1 plus continuation folds2--5."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.common.video.cv_execution import INITIAL_SHA, PLAN_SHA, check_fit_result, load_execution, summarize_folds
from experiments.common.video.cv_resume_execution import REFUSAL, checked_parsed_inputs, plain, require, verify_inventory
from experiments.common.video.dense_evaluation import video_metrics
from experiments.common.video.pts_clip_sampling import tuple_digest
from experiments.common.video.supervised_plan import csv_rows, digest, jsonl, write_json

SPEC = importlib.util.spec_from_file_location('exp017_frozen_audit', Path(__file__).with_name('audit.py'))
ORIGINAL = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(ORIGINAL)
PARENT = 'data/results/exp017/cv-s7-20261102-v1'
CHILD = 'data/results/exp017/cv-s7-20261102-v2'
BINDING = 'data/metadata/EXP-017/resume-parent-v1/parent.json'
BINDING_SHA = '02f485d9f46b2884548e86a2b1d745300e51de64aebaf1114a4d1e251cdf55b0'
RESUME_CODE = {
    'experiments/common/video/cv_resume_execution.py': '1b48806e14a66ce976c1271a4b8767c29d18c7eb1e6c0cc457ab6e301d180895',
    'experiments/runspecs/EXP-017/resume.py': 'cee58d42e09ad501683e40a6afaebb00bf9d4ce9d5df072e438709a90f8e9538',
    'experiments/runspecs/EXP-017/resume.sh': 'b669a47b06d39da692c31a2d0fa0c58ba0b123a105e8cde85b18b369ce167d6d',
}
FAILURES = ('pipeline-failure.json', 'hard-time-limit.json', 'hard-stop-shutdown.json', 'hard-stop-shutdown.log')


class Evidence:
    def __init__(self, root): self.root = root; self.hashes = {}
    def file(self, path):
        path = plain(self.root, Path(path).relative_to(self.root).as_posix())
        name = path.relative_to(self.root).as_posix()
        if name not in self.hashes: self.hashes[name] = digest(path)
        return self.hashes[name]
    def read(self, path):
        self.file(path)
        return json.loads(Path(path).read_text(encoding='utf-8'))


def origins_and_lineage(root, parent, child, pipeline, frozen, binding):
    require(parent == root / PARENT and child == root / CHILD, 'Unregistered parent/child paths')
    expected = dict(parent_run_id=parent.name, parent_path=PARENT, parent_binding_path=BINDING,
        parent_binding_sha256=BINDING_SHA, parent_execution_inputs_sha256=binding['parent_execution_inputs_sha256'],
        parent_artifacts_csv_sha256=binding['files_sha256'], reused_folds=[1], newly_executed_folds=[2,3,4,5],
        inherited_parent_weights=False, inherited_parent_optimizer=False)
    require(pipeline['resume_lineage'] == frozen['resume_lineage'] == expected, 'False or altered parent reuse lineage')
    origins = [dict(fold=i, reused=i == 1, run_id=parent.name if i == 1 else child.name,
                   path=(PARENT if i == 1 else CHILD) + '/fold-%02d' % i) for i in range(1,6)]
    require(pipeline['fold_artifact_origins'] == origins, 'Wrong or repeated fold artifact origins')
    require(not (child / 'fold-01').exists(), 'Copied/rerun child fold1 is forbidden')
    return [root / r['path'] for r in origins]


def completeness(parent, child):
    require(not any((child / n).exists() for n in FAILURES), 'Child failure/hard-stop prevents complete acceptance')
    require(not list(child.rglob('failure.json')), 'Child stage failure prevents complete acceptance')
    needed = [child / 'pipeline-result.json', child / 'execution-inputs.json', child / 'shutdown-receipt.json']
    for i in range(1,6):
        base = (parent if i == 1 else child) / ('fold-%02d' % i)
        for stage, names in [('inner', ['training-result.json','profile.json','inputs.json','epochs.jsonl','best.pt','last.pt']),
                             ('outer', ['training-result.json','profile.json','inputs.json','epochs.jsonl','selected.pt','last.pt']),
                             ('dense', ['result.json','window-scores.csv'])]:
            needed.extend(base / stage / n for n in names)
        needed.extend(base / 'inner' / ('validation-epoch-%02d.json' % e) for e in range(1,21))
    return [p for p in needed if not p.is_file()]


def audit_fit(context, directory, fold, frozen, evidence):
    inner = evidence.read(directory / 'inner/training-result.json'); number = fold['fold']
    check_fit_result(inner, 'inner_select', number, 20); E = inner['selected_epochs']
    outer = evidence.read(directory / 'outer/training-result.json'); check_fit_result(outer, 'fixed_outer', number, E)
    losses = []
    for stage, role, epochs, result in [('inner','inner_train',20,inner), ('outer','outer_train',E,outer)]:
        d = directory / stage; profile = evidence.read(d / 'profile.json')
        require(evidence.file(d / 'profile.json') == result['profile_sha256'], 'Fit profile byte SHA differs')
        parsed = checked_parsed_inputs(context, d / 'inputs.json', fold, role,
            fold['inner_val']['clip_ids'] if stage == 'inner' else None, frozen['input_fingerprints'])
        evidence.file(d / 'inputs.json')
        rule = 'minimum_fixed_inner_unweighted_BCE_earliest_exact_tie' if stage == 'inner' else 'fixed_E_from_inner_no_validation_or_best_selection'
        require(profile['execution_kind'] == 'registered_fit' and profile['fold'] == number and profile['stage'] == result['stage']
                and profile['mode'] == 'rgb_zero' and profile['objective'] == 'ordinary_weighted_bce' and profile['seed'] == 7
                and profile['pretrained'] == 'KINETICS400_V1' and profile['model_class'] == 'experiments.common.video.r2plus1d_classifier.VideoClassifier'
                and profile['initial_state_sha256'] == profile['expected_initial_state_sha256'] == INITIAL_SHA
                and profile['expected_inner_initial_state_sha256'] == (None if stage == 'inner' else INITIAL_SHA)
                and profile['effective_batch'] == 24 and profile['microbatch'] == 8 and profile['checkpoint_backbone'] is False
                and profile['workers'] == 2 and profile['cache_mib'] == 128 and profile['frozen_schedule_epochs'] == 20
                and profile['requested_epochs'] == epochs and profile['selection_rule'] == result['selection_rule'] == rule
                and profile['input_fingerprints'] == result['input_fingerprints'] == frozen['input_fingerprints']
                and profile['parsed_inputs_sha256'] == result['parsed_inputs_sha256'] == parsed, 'Fit recipe/input/initial state differs')
        require(profile['optimizer'] == dict(name='Adam',lr_backbone=1e-4,lr_head=1e-3,betas=[.9,.999],eps=1e-8,weight_decay=0,initial_state_entries=0), 'Optimizer recipe differs')
        env = profile['environment']
        require(env['device'] == 'cuda' and env['precision'] == 'FP32' and env['amp'] is False
                and env['tf32_matmul'] is False and env['tf32_cudnn'] is False and env['cudnn_benchmark'] is False
                and env['cudnn_deterministic'] is True, 'FP32 recipe differs')
        require(profile['code_sha256'] == result['code_sha256'], 'Fit code fingerprint differs')
        for name, sha in result['code_sha256'].items():
            prefix = '/root/autodl-tmp/debris-flow-thesis-exp016/'
            require(name.startswith(prefix) and frozen['code_sha256'].get(name[len(prefix):]) == sha, 'Fit code outside frozen source')
        evidence.file(d / 'epochs.jsonl'); history = jsonl(d / 'epochs.jsonl')
        require(len(history) == epochs, 'Wrong fit length'); updates = 0; counts = set()
        for i, row in enumerate(history):
            ids = fold[role]['epoch_clip_ids'][i]; P = sum(context['bank'][c]['label'] for c in ids); N = len(ids) - P
            counts.add((P,N)); steps = math.ceil(len(ids) / 24); updates += steps
            q, r = divmod(len(ids), steps)
            sizes = [q+1]*r + [q]*(steps-r)  # Frozen EpochBatches uses balanced np.array_split.
            require(row['completed_epochs'] == i+1 and row['train_clip_ids'] == ids and row['samples'] == len(ids)
                    and row['train_P'] == P and row['train_N'] == N and row['optimizer_steps'] == steps
                    and row['total_updates'] == updates and row['actual_effective_batch_sizes'] == sizes
                    and row['finite'] is True and math.isfinite(row['train_weighted_bce'])
                    and all(math.isfinite(row[k]) and row[k] >= 0 for k in ('grad_norm_mean','grad_norm_max','seconds'))
                    and row['initial_state_sha256'] == INITIAL_SHA and row['learning_rates'] == [1e-4,1e-3]
                    and row['profile_sha256'] == result['profile_sha256'] and row['parsed_inputs_sha256'] == parsed
                    and row['code_sha256'] == result['code_sha256'], 'Epoch schedule/updates/finite/fingerprint differs')
            if stage == 'inner':
                pred = evidence.read(d / ('validation-epoch-%02d.json' % (i+1)))
                require([r['clip_id'] for r in pred] == fold['inner_val']['clip_ids']
                        and all(r['label'] == context['bank'][r['clip_id']]['label'] and math.isfinite(r['logit']) for r in pred)
                        and row['validation_clip_ids'] == fold['inner_val']['clip_ids'] and row['validation_samples'] == len(pred), 'Inner validation identity/label differs')
                ORIGINAL.checked_validation_bce(pred,row['validation_unweighted_bce']); losses.append(row['validation_unweighted_bce'])
                require(row['best_epoch'] == min(range(len(losses)),key=lambda j:losses[j])+1, 'Inner running earliest minimum differs')
            else: require(not any(k.startswith('validation') for k in row), 'Outer used validation')
        require(len(counts) == 1 and P > 0 and N > 0 and updates == result['optimizer_steps'] == profile['planned_optimizer_steps']
                and (profile['train_P'],profile['train_N']) == (result['train_P'],result['train_N']) == (P,N)
                and math.isclose(profile['supervised_pos_weight'],N/P,abs_tol=1e-12)
                and result['pos_weight'] == profile['supervised_pos_weight']
                and result['final_state_sha256'] == history[-1]['model_state_sha256'], 'Fit N/P/update/final state record differs')
        require(not list(d.glob('failure.json')) and not list(d.glob('*.part')) and not list(d.glob('*.tmp')), 'Unclosed fit artifact')
        if stage == 'outer': require(not list(d.glob('validation-*')) and not (d/'best.pt').exists(), 'Outer contains selection artifacts')
        for key, name, epoch in [('selected_checkpoint','best.pt' if stage == 'inner' else 'selected.pt',E),('last_checkpoint','last.pt',epochs)]:
            ref = result[key]
            require(ref['filename'] == name and ref['completed_epochs'] == epoch and evidence.file(d/name) == ref['sha256'], 'Checkpoint actual bytes/epoch differs')
    require(E == min(range(20),key=lambda i:losses[i])+1 and inner['best_validation_unweighted_bce'] == losses[E-1], 'Selected E differs')
    return inner, outer


def audit_dense(directory, fold, outer, frozen, context, evidence):
    number = fold['fold']; E = outer['selected_epochs']; d = evidence.read(directory/'dense/result.json')
    uids = set(fold['outer_eval']['video_ids'])
    require(d['status'] == 'outer_fold_complete' and d['fold'] == number and d['selected_epochs'] == E
            and d['plan_sha256'] == PLAN_SHA and d['checkpoint_sha256'] == outer['selected_checkpoint']['sha256']
            and d['pixel_reference_sha256'] == frozen['dense_reference_sha256'] and d['input_fingerprints'] == frozen['input_fingerprints']
            and d['threshold'] == .5 and d['aggregation'] == 'max' and d['outer_selection_use'] is False
            and d['independent_test_used'] is False, 'Dense model/selection/input contract differs')
    require(len(d['scores']) == len(uids) and {r['unit_id'] for r in d['scores']} == set(d['videos']) == uids, 'Missing/duplicate outer videos')
    grid = {}
    for uid in uids:
        for w in context['dense']['windows'][uid]:
            key = (uid,tuple_digest(w['frame_indices'])); require(key not in grid,'Duplicate frozen physical window'); grid[key] = w
    path = directory/'dense/window-scores.csv'; require(evidence.file(path) == d['window_scores_sha256'], 'Window CSV byte SHA differs')
    maxima = {}; counts = {}; seen = set()
    for row in csv_rows(path):
        uid = row['unit_id']; key = (uid,row['frame_tuple_sha256']); value = float(row['score'])
        require(key in grid and key not in seen and math.isfinite(value) and 0 <= value <= 1, 'Unknown/duplicate/nonfinite dense window')
        w = grid[key]
        require(all(math.isfinite(float(row[k])) and abs(float(row[k])-v) <= 1e-8 for k,v in
            [('target_start_seconds',w['target_start_seconds']),('start',w['source_pts_seconds'][0]),('end',w['source_pts_seconds'][-1])]), 'Dense physical timing differs')
        seen.add(key); counts[uid] = counts.get(uid,0)+1; maxima[uid] = max(maxima.get(uid,0.),value)
    require(seen == set(grid), 'Missing canonical dense windows')
    for row in d['scores']:
        uid = row['unit_id']; source = context['sources'][uid]; ref = context['dense']['reference']['videos'][uid]; info = d['videos'][uid]
        require(row['fold'] == number and row['label'] == int(source['video_label']=='pos') and row['group_id'] == source['group_id']
                and row['score'] == maxima[uid] and row['windows'] == counts[uid] == info['windows']
                and info['score'] == maxima[uid] and info['decoded'] == ref['decoded']
                and info['canonical_reference_matched'] is True and info['rgb112_digest'] == ref['rgb112_digest'], 'Video max/label/group/EOF/pixel closure differs')
        evidence.file(directory/'dense'/(uid+'-filter.txt'))
    groups = {g:video_metrics([r for r in d['scores'] if r['group_id']==g]) for g in sorted({r['group_id'] for r in d['scores']})}
    require(video_metrics(d['scores']) == d['metrics'] and groups == d['metrics_by_group'], 'Fold/group metrics drift')
    return d, seen


def audit(parent, child, plan_dir, out, preflight, manifest, *, root=ROOT):
    root = Path(root).resolve(); parent,child,plan_dir,out,preflight,manifest = [Path(p).resolve() for p in (parent,child,plan_dir,out,preflight,manifest)]
    require(parent == root/PARENT and child == root/CHILD, 'Wrong parent/child run')
    require(all(out != p and p not in out.parents for p in (parent,child)), 'Audit output must be outside immutable runs')
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=False); evidence = Evidence(root)
    try:
        missing = completeness(parent,child)
        if missing:
            answer = dict(status='incomplete',metrics_withheld=True,missing=[p.relative_to(root).as_posix() for p in missing],audit_source_sha256=digest(Path(__file__)))
            write_json(out/'audit.json',answer); return answer
        pipeline = evidence.read(child/'pipeline-result.json'); frozen = evidence.read(child/'execution-inputs.json')
        require(pipeline['status'] == 'complete_pending_independent_audit', 'Child pipeline is incomplete')
        shutdown = evidence.read(child/'shutdown-receipt.json')
        require(shutdown.get('status') == 'shutdown_requested' and shutdown.get('returncode') == 0, 'Child shutdown request failed')
        binding_path = plain(root,BINDING); require(evidence.file(binding_path) == BINDING_SHA,'Fixed parent binding changed')
        b = evidence.read(binding_path); files = binding_path.with_name('files.csv')
        require(b['status'] == 'parent_fold1_reuse_verified' and b['resume_code_sha256'] == RESUME_CODE
                and b['reuse_folds'] == [1] and b['execute_folds'] == [2,3,4,5] and b['plan_sha256'] == PLAN_SHA
                and b['file_count'] == 94 and b['bytes'] == 1625434039, 'Unregistered parent binding')
        require(evidence.file(files) == b['files_sha256'], 'Parent CSV SHA changed')
        parent_rows = csv_rows(files); parent_hashes = verify_inventory(parent,parent_rows)
        require(len(parent_rows) == 94 and sum(int(r['size_bytes']) for r in parent_rows) == b['bytes'], 'Parent artifact coverage changed')
        evidence.hashes.update({(parent.relative_to(root)/n).as_posix():sha for n,sha in parent_hashes.items()})
        require(evidence.file(parent/'execution-inputs.json') == b['parent_execution_inputs_sha256'], 'Parent execution identity changed')
        original = evidence.read(parent/'execution-inputs.json')
        require({k:v for k,v in frozen.items() if k not in ('resume_lineage','resume_code_sha256')} == original
                and frozen['resume_code_sha256'] == RESUME_CODE, 'Child changed original science/input contract')
        directories = origins_and_lineage(root,parent,child,pipeline,frozen,b)
        failure = evidence.read(parent/'pipeline-failure.json')
        require(failure.get('status') == 'incomplete_five_fold_metrics_withheld' and failure.get('error_type') == 'RuntimeError'
                and failure.get('error') == REFUSAL, 'Parent stop is not the registered timing-only refusal')
        require(not any((parent/n).exists() for n in FAILURES[1:]) and not list(parent.rglob('failure.json')), 'Parent scientific stage failed')
        require(evidence.file(preflight) == b['preflight_sha256'] and evidence.file(manifest) == b['transfer_manifest_sha256'], 'Original static evidence differs')
        proof = evidence.read(preflight)
        require(proof['status'] == 'cv_static_preflight_passed' and proof['files_verified'] == 192461
                and all(proof.get(k) == v for k,v in original.items() if k != 'input_fingerprints'), 'Static proof contract differs')
        for name,sha in original['code_sha256'].items():
            for run in (parent,child): require(evidence.file(run/'source-snapshot'/name) == sha,'Original source snapshot changed')
        for name,sha in RESUME_CODE.items():
            require(evidence.file(child/'source-snapshot'/name) == sha,'Continuation source snapshot changed')
            require(evidence.file(plain(root,name)) == sha,'Continuation audit dependency changed')
        for name,sha in original['input_fingerprints'].items(): require(evidence.file(plain(root,name)) == sha,'Frozen execution input changed')
        require(plan_dir == root/original['plan_path'] and evidence.file(plan_dir/'plan.json') == PLAN_SHA,'Plan differs')
        context = load_execution(root,plan_dir,root/original['acceptance_path'],root/original['dense_reference_path'])
        context['fingerprints'].update(original['code_sha256'])
        require(context['fingerprints'] == original['input_fingerprints'],'Parsed execution closure differs')
        results = []; fits = []; seen = set()
        for fold,directory in zip(context['folds'],directories):
            inner,outer = audit_fit(context,directory,fold,original,evidence)
            dense,keys = audit_dense(directory,fold,outer,original,context,evidence)
            require(not seen.intersection(keys),'Repeated physical window across folds'); seen.update(keys); results.append(dense)
            fits.append(dict(fold=fold['fold'],artifact_directory=directory.relative_to(root).as_posix(),selected_epochs=inner['selected_epochs'],
                             inner_updates=inner['optimizer_steps'],outer_updates=outer['optimizer_steps']))
        expected = {(u,tuple_digest(w['frame_indices'])) for u,rows in context['dense']['windows'].items() for w in rows}
        require(len(seen) == len(expected) == 143470 and seen == expected,'Incomplete full development physical grid')
        require((fits[0]['selected_epochs'],fits[0]['inner_updates'],fits[0]['outer_updates'],len(results[0]['scores']),sum(r['windows'] for r in results[0]['scores']))
                == (b['selected_epochs'],b['inner_updates'],b['outer_updates'],b['dense_videos'],b['dense_windows']), 'Reused parent counts differ')
        summary = summarize_folds(results,context['sources'])
        per_fold = [dict(fold=i+1,selected_epochs=fits[i]['selected_epochs'],metrics=results[i]['metrics']) for i in range(5)]
        require(summary == pipeline['summary'] and per_fold == pipeline['per_fold'],'Combined summary/selection/metrics drift')
        answer = dict(status='audit_passed',summary=summary,fits=fits,physical_windows=len(seen),resume_lineage=pipeline['resume_lineage'],
            source_sha256=evidence.hashes,audit_source_sha256=digest(Path(__file__)),original_audit_source_sha256=digest(Path(ORIGINAL.__file__)),
            plan_sha256=PLAN_SHA,validation_BCE_absolute_tolerance=1e-5,no_model_execution=True,no_media_decode=True,
            limitations='Session-group development CV, not independent external/unseen-camera testing. Weight files hashed, not tensor-replayed. Actual bills and platform-off state separately verified by operator.')
        write_json(out/'audit.json',answer); return answer
    except Exception as e:
        answer = dict(status='audit_failed',metrics_withheld=True,error_type=type(e).__name__,error=str(e),source_sha256=evidence.hashes,audit_source_sha256=digest(Path(__file__)))
        write_json(out/'failure.json',answer); return answer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,default=ROOT/PARENT); p.add_argument('--child',type=Path,default=ROOT/CHILD)
    p.add_argument('--plan',type=Path,default=ROOT/'data/metadata/EXP-017/role-free-plan-v3')
    p.add_argument('--preflight',type=Path,default=ROOT/'.codex/logs/evidence/exp017-gpu-resume-v2/v2-preflight.json')
    p.add_argument('--transfer-manifest',type=Path,default=ROOT/'data/metadata/EXP-017/deployment-v1/files.csv')
    p.add_argument('--out',type=Path,required=True); a = p.parse_args()
    answer = audit(a.parent,a.child,a.plan,a.out,a.preflight,a.transfer_manifest)
    print(json.dumps(answer,ensure_ascii=True)); raise SystemExit({'audit_passed':0,'incomplete':2,'audit_failed':1}[answer['status']])


if __name__ == '__main__': main()
