"""EXP021 registered clip floor and unique-video/session gates, no new tuning."""
import math

from experiments.common.video.dense_evaluation import video_metrics
from experiments.common.video.cv_supervised_training import _json, _sha
from experiments.common.video.oriented_energy_plan import read


def clip_metrics(directory, bank, validation):
    result = read(directory / 'training-result.json')
    with (directory / 'epochs.jsonl').open(encoding='utf-8') as f:
        import json
        epochs = [json.loads(line) for line in f]
    if len(epochs) != 20 or result['status'] != 'fit_complete' or result['completed_epochs'] != 20:
        raise ValueError('Incomplete inner fit')
    values = [r['validation_unweighted_bce'] for r in epochs]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite inner BCE')
    selected = min(range(1, 21), key=lambda e: values[e - 1])
    if selected != result['selected_epochs']:
        raise ValueError('Selected E must be earliest raw BCE minimum')
    rows = read(directory / ('validation-epoch-%02d.json' % selected))
    if ([r['clip_id'] for r in rows] != validation or any(r['label'] != bank[r['clip_id']]['label']
            or not math.isfinite(r['logit']) for r in rows)):
        raise ValueError('Selected raw validation identity/label differs')
    pos = [r['logit'] for r in rows if r['label'] == 1]; neg = [r['logit'] for r in rows if r['label'] == 0]
    if not pos or not neg:
        raise ValueError('Both classes required')
    auc = math.fsum((a > b) + .5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
    def probability(z):
        q = math.exp(-abs(z)); return 1 / (1 + q) if z >= 0 else q / (1 + q)
    tp = sum(probability(z) >= .5 for z in pos); tn = sum(probability(z) < .5 for z in neg)
    pr, nr = tp / len(pos), tn / len(neg)
    return dict(selected_epochs=selected, AUC_logit=auc, positive_Recall=pr, negative_Recall=nr,
        TP=tp, FP=len(neg) - tn, FN=len(pos) - tp, TN=tn, score_kind='logit_AUC',
        passed=auc >= .75 and pr >= .5 and nr >= .5)


def gate(dense, fragment, sources, roles):
    """Same video's roles averaged first, then videos per group, then groups."""
    indexed = {}
    for fold in (1, 4):
        expected = set(roles[str(fold)]['video_ids'])
        for arm in ('A', 'B', 'C'):
            result = dense[str(fold)][arm]
            rows = result['scores']
            if (result['status'] != 'inner_role_dense_complete' or len(rows) != len(expected)
                    or {r['unit_id'] for r in rows} != expected):
                raise ValueError('Incomplete or duplicate dense role')
            for r in rows:
                uid = r['unit_id']; key = fold, arm, uid
                if (key in indexed or r['label'] != int(sources[uid]['video_label'] == 'pos')
                        or r['group_id'] != sources[uid]['group_id'] or not math.isfinite(r['score']) or not 0 <= r['score'] <= 1):
                    raise ValueError('Frozen video label/group/score differs')
                indexed[key] = r
            actual = video_metrics(rows)
            if any(actual[k] != result['metrics'][k] for k in ('TP', 'FP', 'FN', 'TN')):
                raise ValueError('Dense confusion arithmetic differs')
    uids = sorted({u for f, a, u in indexed if sources[u]['video_label'] == 'neg'})
    groups = sorted({sources[u]['group_id'] for u in uids})
    video_role_means = {arm: {u: math.fsum(indexed[(f, arm, u)]['score'] >= .5 for f in (1, 4) if (f, arm, u) in indexed)
        / sum((f, arm, u) in indexed for f in (1, 4)) for u in uids} for arm in ('A', 'B', 'C')}
    group_fpr = {arm: {g: math.fsum(video_role_means[arm][u] for u in uids if sources[u]['group_id'] == g)
        / sum(sources[u]['group_id'] == g for u in uids) for g in groups} for arm in ('A', 'B', 'C')}
    macro = {arm: math.fsum(group_fpr[arm].values()) / len(groups) for arm in group_fpr}
    references = {}
    for reference in ('A', 'B'):
        repaired, worsened, mixed, detail = [], [], [], []
        for uid in uids:
            records = [dict(fold=f, reference_positive=indexed[(f, reference, uid)]['score'] >= .5,
                            C_positive=indexed[(f, 'C', uid)]['score'] >= .5) for f in (1, 4) if (f, reference, uid) in indexed]
            repair = any(r['reference_positive'] and not r['C_positive'] for r in records)
            worse = any(not r['reference_positive'] and r['C_positive'] for r in records)
            if repair and not worse: repaired.append(uid)
            if worse: worsened.append(uid)
            if repair and worse: mixed.append(uid)
            if repair or worse: detail.append(dict(unit_id=uid, group_id=sources[uid]['group_id'], roles=records))
        improved = [g for g in groups if group_fpr['C'][g] < group_fpr[reference][g]]
        regressed = [g for g in groups if group_fpr['C'][g] > group_fpr[reference][g]]
        per_role = {str(f): all(dense[str(f)]['C']['metrics'][k] <= dense[str(f)][reference]['metrics'][k] for k in ('FP', 'FN')) for f in (1, 4)}
        references[reference] = dict(R=repaired, W=worsened, role_inconsistent=mixed, error_changes=detail,
            unique_net_repair=len(repaired) - len(worsened), improved_negative_groups=improved, regressed_negative_groups=regressed,
            per_role_FP_FN_nonincrease=per_role, macro_FPR_reference=macro[reference], macro_FPR_C=macro['C'],
            passed=all(per_role.values()) and len(repaired) - len(worsened) >= 2 and len(improved) >= 2 and macro['C'] < macro[reference])
    floor = all(fragment[str(f)][arm]['passed'] for f in (1, 4) for arm in ('B', 'C'))
    return dict(status='registered_development_gates_computed_pending_independent_audit', fragment_floor_all_four=floor,
        references=references, negative_video_role_mean_FPR=video_role_means, unique_group_FPR=group_fpr,
        unique_group_macro_FPR=macro, promote_to_matched_formal_design_only=floor and all(r['passed'] for r in references.values()),
        scope='42 unique developmental videos across47 roles; no independent-repeat, outer, external or physical-flow claim')
