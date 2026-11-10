"""Produce thesis-ready figures only after the complete pilot audit passes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

SOURCE=Path(__file__).resolve()
ROOT=SOURCE.parents[3]
MODES=('bce','cross','same')
LABELS={'bce':'普通分类','cross':'跨场次排序','same':'同场次排序'}
COLORS={'bce':'#3366a3','cross':'#c6862b','same':'#37815b'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,
                         'figure.dpi':140,'savefig.dpi':220,'pdf.fonttype':42,'ps.fonttype':42})
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--figures',type=Path,default=ROOT/'docs/process/experiments/figures/EXP-018')
    a=ap.parse_args()
    audit_path=a.run/'audit-v1/audit.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['status']!='audit_passed':raise ValueError('Only audited complete results may be plotted')
    if a.out.exists() or a.figures.exists():raise FileExistsError('Do not overwrite an earlier report')
    summary=audit['summaries']
    histories={m:[json.loads(line) for line in (a.run/m/'epochs.jsonl').read_text(encoding='utf-8').splitlines()] for m in MODES}
    if any(len(h)!=20 for h in histories.values()):raise ValueError('Incomplete curves')
    dense=json.loads((a.run/'three-dense/result.json').read_text(encoding='utf-8'))
    a.out.mkdir(parents=True);a.figures.mkdir(parents=True)
    saved=[]
    def save(fig,name):
        for suffix in ('png','pdf','svg'):
            p=a.figures/(name+'.'+suffix)
            fig.savefig(p,bbox_inches='tight')
            saved.append(dict(path=p.resolve().relative_to(ROOT).as_posix(),sha256=sha(p)))
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.9))
    for i,m in enumerate(MODES):
        v=summary[m]['video_metrics']
        axes[0].bar([i*3,i*3+1],[v['FP'],v['FN']],color=[COLORS[m],COLORS[m]],alpha=.85)
    axes[0].set_xticks([0,1,3,4,6,7]);axes[0].set_xticklabels(['FP','FN']*3)
    axes[0].set_xlabel('普通分类       跨场次排序       同场次排序')
    axes[0].set_ylabel('视频数量（条）');axes[0].set_title('固定0.5阈值：误报与漏报')
    peak_error=max(summary[m]['video_metrics'][k] for m in MODES for k in ('FP','FN'))
    axes[0].set_ylim(0,max(1,peak_error)*1.25)
    for rect in axes[0].patches:
        axes[0].text(rect.get_x()+rect.get_width()/2,rect.get_height()+.08,str(int(rect.get_height())),ha='center',va='bottom')
    for axis,key,title,unit,scale in (
        (axes[1],'clip_bce_fp64','136个固定片段的分类损失','未加权BCE',1),
        (axes[2],'mixed_video_inversion_macro','三个混合视频的宏错序率','错序率（%）',100)):
        values=[summary[m][key]*scale for m in MODES]
        axis.bar(range(3),values,color=[COLORS[m] for m in MODES])
        axis.set_xticks(range(3));axis.set_xticklabels([LABELS[m] for m in MODES],rotation=15)
        axis.set_ylabel(unit);axis.set_xlabel('训练目标');axis.set_title(title)
        axis.set_ylim(0,max(max(values)*1.25,1e-6))
        for i,value in enumerate(values):axis.text(i,value,f'{value:.3f}',ha='center',va='bottom')
    fig.suptitle('EXP-018：同一开发划分的三组匹配小试',y=1.03)
    fig.text(.01,-.08,'来源：audit-v1/audit.json；seed7，各20轮。视频24正／7负，固定片段96正／40负。三个混合视频并非独立测试。',fontsize=9)
    fig.tight_layout();save(fig,'three-condition-results')
    fig,axes=plt.subplots(2,1,figsize=(10,6.6),sharex=True)
    for m in MODES:
        h=histories[m];x=list(range(1,21))
        axes[0].plot(x,[r['train_weighted_bce'] for r in h],label=LABELS[m],color=COLORS[m])
        axes[1].plot(x,[r['validation_unweighted_bce'] for r in h],label=LABELS[m],color=COLORS[m])
        best=summary[m]['best_epoch']
        axes[1].scatter([best],[h[best-1]['validation_unweighted_bce']],color=COLORS[m],s=35,zorder=4)
    axes[0].set_title('训练分类分量：每轮504窗，不包含附加排序项')
    axes[0].set_ylabel('加权BCE／片段');axes[0].legend()
    axes[1].set_title('固定验证：未加权BCE；圆点为预定规则选中轮次')
    axes[1].set_ylabel('未加权BCE／片段');axes[1].set_xlabel('完成训练轮次（轮）')
    axes[1].set_xticks(range(1,21));axes[1].legend()
    for axis in axes:axis.grid(axis='y',alpha=.2)
    fig.text(.01,-.02,'来源：各组epochs.jsonl。训练有类权重及Dropout，验证没有；上下两图不作数值高低比较。',fontsize=9)
    fig.tight_layout();save(fig,'three-condition-learning-curves')
    with (a.out/'metrics.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['objective','best_epoch','clip_bce_fp64','mixed_inversion_macro','TP','FP','FN','TN','AUC','Recall','FPR'])
        writer.writeheader()
        for m in MODES:
            row=dict(objective=m,best_epoch=summary[m]['best_epoch'],clip_bce_fp64=summary[m]['clip_bce_fp64'],mixed_inversion_macro=summary[m]['mixed_video_inversion_macro'])
            row.update({k:summary[m]['video_metrics'][k] for k in ('TP','FP','FN','TN','AUC','Recall','FPR')});writer.writerow(row)
    indexed={m:{r['unit_id']:r for r in dense['scores'][m]} for m in MODES}
    with (a.out/'video-comparison.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['unit_id','group_id','label',*MODES,*[m+'_wrong' for m in MODES]])
        writer.writeheader()
        for uid in sorted(indexed['bce']):
            base=indexed['bce'][uid];row={k:base[k] for k in ('unit_id','group_id','label')}
            for m in MODES:
                row[m]=indexed[m][uid]['score'];row[m+'_wrong']=int((row[m]>=.5)!=bool(base['label']))
            writer.writerow(row)
    shutil.copyfile(SOURCE,a.out/'generator-source.py')
    (a.out/'figure-index.json').write_text(json.dumps(dict(audit_sha256=sha(audit_path),figures=saved,
        registered_gate=audit['registered_gate'],advance_condition_met=audit['advance_condition_met']),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(figures=len(saved),output=str(a.out),advance=audit['advance_condition_met'])))


if __name__=='__main__':main()
