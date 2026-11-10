"""Reproduce EXP-017 development-CV tables/figures from the passed joint audit."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.common.video.dense_evaluation import video_metrics
from experiments.common.video.supervised_plan import digest, write_json

AUDIT_SHA = '4b94b4094336c99bf35e675bd07a72420498fb97bd56f8ee366a75a2b2971c45'


def require(value,message):
    if not value: raise ValueError(message)


def probability(logit):
    q=math.exp(-abs(logit))
    return 1/(1+q) if logit>=0 else q/(1+q)


def probability_stats(values):
    x=np.asarray(values,dtype=np.float64)
    require(len(x)>0 and np.isfinite(x).all() and ((0<=x)&(x<=1)).all(),'Invalid probability series')
    quantiles=np.quantile(x,[0,.25,.5,.75,1])
    return dict(n=int(len(x)),min=float(quantiles[0]),q25=float(quantiles[1]),median=float(quantiles[2]),q75=float(quantiles[3]),max=float(quantiles[4]),
                mean=float(np.mean(x)),population_sd=float(np.std(x,ddof=0)),below_0_5=int(np.count_nonzero(x<.5)))


def write_csv(path,rows):
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def figures(folds,videos,directory):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    blue='#2878B5';orange='#D95319';gray='#444444';x=np.arange(1,6)
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.6));fig.subplots_adjust(left=.075,right=.985,bottom=.20,top=.86,wspace=.30)
    a,b=axes
    a.bar(x,[r['AUC'] for r in folds],color=blue,width=.6)
    a.axhline(.5,color=gray,lw=1,ls='--');a.set_ylim(0,1.06);a.set_yticks(np.arange(0,1.01,.2))
    a.set_xticks(x,['F%d\nE=%d'%(r['fold'],r['selected_epochs']) for r in folds]);a.set_ylabel('Video AUC');a.set_title('(a) Ranking within each held-out fold',fontsize=10)
    for i,r in enumerate(folds):a.text(i+1,r['AUC']+.025,'%.3f'%r['AUC'],ha='center',fontsize=8)
    tp=[r['TP'] for r in folds];fp=[r['FP'] for r in folds]
    b.bar(x,tp,color=blue,width=.6,label='TP: positive video predicted positive')
    b.bar(x,fp,bottom=tp,color=orange,width=.6,label='FP: negative video predicted positive')
    for i,r in enumerate(folds):
        b.text(i+1,tp[i]/2,str(tp[i]),ha='center',va='center',color='white')
        b.text(i+1,tp[i]+fp[i]/2,str(fp[i]),ha='center',va='center',color='white')
    b.set_xticks(x,['F%d'%i for i in x]);b.set_ylim(0,38);b.set_ylabel('Held-out videos (count)');b.set_title('(b) Fixed threshold 0.5: all videos positive',fontsize=10)
    b.text(.03,.97,'FN = TN = 0 in every fold',transform=b.transAxes,va='top',fontsize=8)
    b.legend(loc='upper center',bbox_to_anchor=(.48,-.11),fontsize=7.4,frameon=False)
    fig.text(.075,.965,'EXP-017: session-group development CV',fontsize=11,weight='bold',va='top')
    fig.text(.075,.035,'AUC mean %.3f, sample SD %.3f (five folds).\nVideo score = maximum window probability.'%
             (statistics.mean(r['AUC'] for r in folds),statistics.stdev(r['AUC'] for r in folds)),fontsize=8)
    save_figure(fig,directory/'auc-and-fixed-threshold','EXP-017 fold AUC and fixed-threshold outcomes')
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.6));fig.subplots_adjust(left=.075,right=.985,bottom=.19,top=.86,wspace=.27)
    a,b=axes
    for fold in range(1,6):
        for label,color,marker,shift in [(1,blue,'o',-.10),(0,orange,'^',.10)]:
            rows=sorted((r for r in videos if r['fold']==fold and r['label']==label),key=lambda r:r['unit_id'])
            jitter=np.linspace(-.055,.055,len(rows)) if len(rows)>1 else np.zeros(len(rows))
            a.scatter(fold+shift+jitter,[r['score'] for r in rows],s=22,c=color,marker=marker,alpha=.8,edgecolors='white',linewidths=.25)
    handles=[Line2D([],[],color=blue,marker='o',ls='',label='Positive video'),Line2D([],[],color=orange,marker='^',ls='',label='Negative video')]
    a.legend(handles=handles,loc='lower left',frameon=False,fontsize=8)
    a.set_title('(a) Held-out video maxima',fontsize=10);a.set_ylabel('Predicted probability');a.set_xticks(x,['F%d\nn=%d'%(r['fold'],r['P']+r['N']) for r in folds])
    for r in folds:
        f=r['fold'];b.plot([f,f],[r['window_min'],r['window_max']],c=blue,lw=1.1)
        b.plot([f,f],[r['window_q25'],r['window_q75']],c=blue,lw=7,solid_capstyle='butt')
        b.scatter([f],[r['window_median']],c='white',edgecolors=blue,s=26,zorder=3)
        if r['window_below_0_5']:
            b.annotate('%d windows < 0.5'%r['window_below_0_5'],xy=(f,r['window_min']),xytext=(2.45,.15),fontsize=8,
                       arrowprops=dict(arrowstyle='->',color=gray,lw=.8))
    b.set_title('(b) Dense window probability quantiles',fontsize=10);b.set_xticks(x,['F%d\nn=%d'%(r['fold'],r['windows']) for r in folds])
    b.legend(handles=[Line2D([],[],c=blue,lw=1,label='Min–max'),Line2D([],[],c=blue,lw=6,label='25–75%'),Line2D([],[],c=blue,marker='o',mfc='white',ls='',label='Median')],
             loc='lower left',frameon=False,fontsize=8)
    for ax in axes:
        ax.set_ylim(0,1.04);ax.set_xlim(.5,5.5);ax.set_yticks(np.arange(0,1.01,.2));ax.axhline(.5,c=gray,lw=.9,ls='--',zorder=0)
    fig.text(.075,.965,'EXP-017: score distributions remain fold-specific',fontsize=11,weight='bold',va='top')
    fig.text(.075,.035,'Descriptive distributions; overlapping windows are correlated. No pooled cross-model AUC.',fontsize=8)
    save_figure(fig,directory/'fold-specific-score-distributions','EXP-017 fold-specific video and window probabilities')


def save_figure(fig,prefix,title):
    fig.savefig(str(prefix)+'.png',dpi=240,metadata={'Software':'Matplotlib','Title':title})
    fig.savefig(str(prefix)+'.pdf',metadata={'Creator':'Matplotlib','Title':title,'Author':''})
    plt.close(fig)


def generate(audit_path,out,figure_dir,argv):
    require(digest(audit_path)==AUDIT_SHA,'Unregistered audit report SHA')
    audit=json.loads(audit_path.read_text(encoding='utf-8'));require(audit['status']=='audit_passed','Audit did not pass')
    require(not out.exists() and not figure_dir.exists(),'New output directories required; refuse overwrite')
    inputs={audit_path.relative_to(ROOT).as_posix():AUDIT_SHA}
    def bound(path):
        path=Path(path).resolve();name=path.relative_to(ROOT).as_posix();expected=audit['source_sha256'][name]
        require(digest(path)==expected,'Audited input changed: '+name);inputs[name]=expected;return path
    def read(path):return json.loads(bound(path).read_text(encoding='utf-8'))
    def rows(path):
        with bound(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    bound(ROOT/'experiments/common/video/dense_evaluation.py')
    sources={r['unit_id']:r for r in rows(ROOT/'data/metadata/EXP-017/role-free-plan-v3/sources.csv')}
    fold_rows=[];video_rows=[];curves=[];inner_rows=[];all_videos=[];groups={};all_windows=[];window_keys=set()
    for fit in audit['fits']:
        fold=fit['fold'];directory=ROOT/fit['artifact_directory'];dense=read(directory/'dense/result.json')
        require(dense['fold']==fold and dense['selected_epochs']==fit['selected_epochs'],'Fit identity changed')
        scores=dense['scores'];metrics=video_metrics(scores);require(metrics==dense['metrics'],'Video metrics differ')
        values=[];counts={};maxima={}
        for row in rows(directory/'dense/window-scores.csv'):
            uid=row['unit_id'];value=float(row['score']);key=(uid,row['frame_tuple_sha256'])
            require(key not in window_keys and math.isfinite(value) and 0<=value<=1,'Duplicate/nonfinite window')
            window_keys.add(key);values.append(value);counts[uid]=counts.get(uid,0)+1;maxima[uid]=max(maxima.get(uid,0.),value)
        for row in scores:
            uid=row['unit_id'];source=sources[uid]
            require(row['score']==maxima[uid] and row['windows']==counts[uid] and row['label']==int(source['video_label']=='pos')
                    and row['group_id']==source['group_id'],'Video/window/source closure differs')
            video_rows.append(dict(fold=fold,artifact_directory=fit['artifact_directory'],unit_id=uid,group_id=row['group_id'],label=row['label'],score=row['score'],windows=row['windows'],prediction_at_0_5=int(row['score']>=.5)))
            groups.setdefault(row['group_id'],[]).append(row)
        all_videos.extend(scores);all_windows.extend(values)
        history=[]
        with bound(directory/'outer/epochs.jsonl').open(encoding='utf-8') as f:
            for line in f:history.append(json.loads(line))
        outer=read(directory/'outer/training-result.json');require(len(history)==fit['selected_epochs'],'Outer E differs')
        for row in history:
            P=row['train_P'];N=row['train_N'];w=outer['pos_weight'];reference=math.log(2)*(P*w+N)/(P+N)
            require(math.isfinite(row['train_weighted_bce']) and math.isclose(w,N/P,abs_tol=1e-12),'Training loss/weight invalid')
            curves.append(dict(fold=fold,completed_epochs=row['completed_epochs'],online_dropout_train_weighted_bce=row['train_weighted_bce'],
                constant_q_0_5_reference=reference,train_P=P,train_N=N,batch_min=min(row['actual_effective_batch_sizes']),batch_max=max(row['actual_effective_batch_sizes'])))
        pred=read(directory/'inner'/('validation-epoch-%02d.json'%fit['selected_epochs']))
        for label in [1,0]:
            selected=[r for r in pred if r['label']==label];require(bool(selected),'Missing selected validation class')
            losses=[max(r['logit'],0)-label*r['logit']+math.log1p(math.exp(-abs(r['logit']))) for r in selected]
            require(all(math.isfinite(x) for x in losses),'Invalid validation loss')
            inner_rows.append(dict(fold=fold,selected_epochs=fit['selected_epochs'],label=label,clips=len(selected),
                unweighted_bce=math.fsum(losses)/len(losses),mean_probability=statistics.mean(probability(r['logit']) for r in selected)))
        row=dict(fold=fold,artifact_directory=fit['artifact_directory'],selected_epochs=fit['selected_epochs'],P=metrics['P'],N=metrics['N'],
                 groups=len({r['group_id'] for r in scores}),negative_groups=len({r['group_id'] for r in scores if r['label']==0}),windows=len(values),
                 TP=metrics['TP'],FP=metrics['FP'],FN=metrics['FN'],TN=metrics['TN'],AUC=metrics['AUC'],Recall=metrics['Recall'],FPR=metrics['FPR'],F1=metrics['F1'])
        for kind,stats in [('video',probability_stats([r['score'] for r in scores])),('window',probability_stats(values))]:
            row.update({kind+'_'+k:v for k,v in stats.items() if k!='n'})
        row.update(outer_final_online_dropout_bce=history[-1]['train_weighted_bce'],outer_constant_q_0_5_bce=reference,
                   selected_inner_pos_bce=inner_rows[-2]['unweighted_bce'],selected_inner_neg_bce=inner_rows[-1]['unweighted_bce'])
        fold_rows.append(row)
    require([r['fold'] for r in fold_rows]==[1,2,3,4,5] and len(all_videos)==len({r['unit_id'] for r in all_videos})==150
            and {r['unit_id'] for r in all_videos}==set(sources) and len(all_windows)==audit['physical_windows']==143470,'Incomplete development coverage')
    for key in ('TP','FP','FN','TN','P','N'):require(sum(r[key] for r in fold_rows)==audit['summary'][key],'Audited summary count differs')
    require([r['AUC'] for r in fold_rows]==audit['summary']['per_fold_auc'],'Audited fold AUC differs')
    negatives=[]
    for g,score_rows in sorted(groups.items()):
        m=video_metrics(score_rows)
        if m['N']:
            require(m==audit['summary']['negative_group_metrics'][g],'Negative group metrics differ')
            negatives.append(dict(group_id=g,P=m['P'],N=m['N'],TP=m['TP'],FP=m['FP'],FN=m['FN'],TN=m['TN'],FPR=m['FPR']))
    out.mkdir(parents=True);figure_dir.mkdir(parents=True)
    try:
        write_csv(out/'folds.csv',fold_rows);write_csv(out/'video-scores.csv',video_rows);write_csv(out/'negative-groups.csv',negatives)
        write_csv(out/'outer-learning-curves.csv',curves);write_csv(out/'selected-inner-class-bce.csv',inner_rows)
        figures(fold_rows,video_rows,figure_dir)
        source=Path(__file__).resolve();(out/'report_exp017.py').write_bytes(source.read_bytes())
        answer=dict(status='derived_from_passed_joint_audit',audit_path=audit_path.relative_to(ROOT).as_posix(),audit_sha256=AUDIT_SHA,
            command=argv,report_source_sha256=digest(source),input_sha256=inputs,
            environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__,matplotlib=matplotlib.__version__),
            dataset=dict(videos=len(sources),P=sum(r['video_label']=='pos' for r in sources.values()),N=sum(r['video_label']=='neg' for r in sources.values()),
                session_groups=len(groups),camera_names=len({g.split('|')[1] for g in groups}),physical_windows=len(all_windows)),
            summary=audit['summary'],negative_group_macro_FPR=statistics.mean(r['FPR'] for r in negatives),negative_groups=len(negatives),folds=fold_rows,
            descriptive_cross_fold_probabilities=dict(video=probability_stats([r['score'] for r in all_videos]),window=probability_stats(all_windows)),
            definitions=dict(probability_sd='Population SD, ddof=0; descriptive only.',AUC_sd='Sample SD across five fold AUCs, ddof=1.',
                constant_q_0_5_weighted_BCE='ln(2)*(pos_weight*P+N)/(P+N), with pos_weight=N/P.',
                outer_training_loss='Observed online loss during training with Dropout, not a fixed eval-mode training-set fit.',
                selected_inner_class_BCE='Stable FP64 logit BCE, unweighted within each actual clip label.'),
            limitations=['Historically reused session-group development pool; no independent external or unseen-camera test.',
                'No pooled AUC across outer models. Combined probability moments are descriptive, not a single-model calibration measure.',
                'Overlapping dense windows are correlated;143470 is input coverage, not an independent sample size.',
                'This script reads audited records only; it does not run a model, decode video, tune a threshold, or establish a causal failure mechanism.'])
        for name,sha in inputs.items():require(digest(ROOT/name)==sha,'Input changed during reporting')
        answer['outputs_sha256']={p.relative_to(ROOT).as_posix():digest(p) for d in (out,figure_dir) for p in sorted(d.iterdir()) if p.is_file()}
        write_json(out/'report.json',answer)
        return answer
    except Exception as e:
        write_json(out/'failure.json',dict(status='report_failed',error_type=type(e).__name__,error=str(e)));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,default=ROOT/'data/results/exp017/audit-resume-cv-s7-20261102-v2-v2/audit.json')
    p.add_argument('--out',type=Path,default=ROOT/'data/results/exp017/report-cv-s7-20261102-v2-v1')
    p.add_argument('--figures',type=Path,default=ROOT/'docs/process/experiments/figures/EXP-017-v1')
    a=p.parse_args();answer=generate(a.audit.resolve(),a.out.resolve(),a.figures.resolve(),sys.argv)
    print(json.dumps(dict(status=answer['status'],dataset=answer['dataset'],macro_negative_group_FPR=answer['negative_group_macro_FPR'],folds=answer['folds']),ensure_ascii=True))


if __name__=='__main__':main()
