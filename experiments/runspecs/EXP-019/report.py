# -*- coding: utf-8 -*-
"""Generate the EXP-019 report/figures from accepted artifacts; no model/media IO."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

SOURCE=Path(__file__).resolve()
ROOT=SOURCE.parents[3]
MODES=('chronological','fixed_permutation')
LABELS={'chronological':'A 真实顺序','fixed_permutation':'B 固定置换'}
COLORS={'chronological':'#28669e','fixed_permutation':'#c07825'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'data/results/exp019/order-s7-20261101-v1')
    p.add_argument('--report',type=Path,default=ROOT/'docs/process/experiments/EXP-019-训练报告.md')
    p.add_argument('--figures',type=Path,default=ROOT/'docs/process/experiments/figures/EXP-019-v1')
    a=p.parse_args(argv)
    a.run,a.report,a.figures=(x.resolve() for x in (a.run,a.report,a.figures))
    a.run.relative_to(ROOT/'data/results/exp019')
    a.report.relative_to(ROOT/'docs/process/experiments');a.figures.relative_to(ROOT/'docs/process/experiments/figures')
    if a.report.exists() or a.figures.exists():raise FileExistsError('Refuse overwriting a previous report or figure version')
    inputs={};input_bytes={}
    def read(path,jsonl=False):
        path=path.resolve();name=path.relative_to(ROOT).as_posix()
        encoded=path.read_bytes();inputs[name]=hashlib.sha256(encoded).hexdigest();input_bytes[name]=encoded
        text=encoded.decode('utf-8-sig')
        return [json.loads(line) for line in text.splitlines() if line.strip()] if jsonl else json.loads(text)
    audit=read(a.run/'audit-v1/audit.json')
    if audit['status']!='audit_passed' or not audit['metrics_accepted']:raise ValueError('Complete accepted audit required')
    summaries=audit['summaries'];gate=audit['registered_gate']
    dense=read(a.run/'two-dense/result.json');pipeline=read(a.run/'pipeline-result.json')
    histories={m:read(a.run/m/'epochs.jsonl',True) for m in MODES}
    training={m:read(a.run/m/'training-result.json') for m in MODES}
    span_path=ROOT/'data/metadata/span-coverage/metadata-v4-20261101-v1/summary.json'
    span=read(span_path)
    billing=read(ROOT/'.codex/logs/evidence/exp019-final-billing.json')
    ledger=read(ROOT/'.codex/logs/evidence/research-budget-20261101.json')
    if (span['status']!='metadata_coverage_audit_passed' or billing['instance_state']!='off'
            or ledger['confirmed_spend']!=billing['goal_total']):raise ValueError('Incomplete coverage/billing/off-state evidence')
    if dense['status']!='dense_evaluation_complete' or len(dense['videos'])!=31:raise ValueError('Incomplete dense evaluation')
    for m in MODES:
        h=histories[m];t=training[m];s=summaries[m]
        if (len(h)!=20 or [r['completed_epochs'] for r in h]!=list(range(1,21)) or t['completed_epochs']!=20 or t['optimizer_steps']!=420
                or any(r['samples']!=504 or r['optimizer_steps']!=21 for r in h)
                or s['video_metrics']!=dense['metrics'][m] or t['best_epoch']!=s['best_epoch']):raise ValueError('Audited schedule/metrics changed')
        recorded=[r['validation_unweighted_bce'] for r in h]
        if not all(math.isfinite(v) for v in recorded) or min(range(20),key=lambda i:recorded[i])+1!=s['best_epoch']:raise ValueError('Invalid earliest minimum selection')
    indexed={m:{r['unit_id']:r for r in summaries[m]['videos']} for m in MODES}
    if set(indexed[MODES[0]])!=set(indexed[MODES[1]]):raise ValueError('Mismatched video identities')
    neg=sorted(uid for uid,r in indexed[MODES[0]].items() if r['label']==0)
    if len(neg)!=7 or len({indexed[MODES[0]][u]['group_id'] for u in neg})!=3:raise ValueError('Unexpected negative coverage')
    fp={m:{u for u in neg if indexed[m][u]['score']>=.5} for m in MODES}
    common=sorted(fp[MODES[0]]&fp[MODES[1]])
    a_only=sorted(fp[MODES[0]]-fp[MODES[1]]);b_only=sorted(fp[MODES[1]]-fp[MODES[0]])
    old=ROOT/'data/results/exp018/paired-s7-20261101-v1/bce'
    old_history=read(old/'epochs.jsonl',True);old_training=read(old/'training-result.json')
    comparable=('completed_epochs','train_loss','train_weighted_bce','train_bce_sum','validation_unweighted_bce',
        'best_epoch','samples','optimizer_steps','total_updates','train_clip_ids','grad_norm_mean','grad_norm_max',
        'validation_P','validation_N','validation_pos_bce','validation_neg_bce','rank_lambda')
    if len(old_history)!=20 or any(x[k]!=y[k] for x,y in zip(histories[MODES[0]],old_history) for k in comparable):raise ValueError('Old ordinary BCE trace not exactly reproduced')
    logit_count=0
    for epoch in range(1,21):
        new=read(a.run/MODES[0]/('validation-epoch-%02d.json'%epoch))
        previous=read(old/('validation-epoch-%02d.json'%epoch))
        if new!=previous or len(new)!=136:raise ValueError('Old validation identity/logits changed')
        logit_count+=len(new)
    for key in ('initial_state_sha256','final_state_sha256'):
        if training[MODES[0]][key]!=old_training[key]:raise ValueError('Old endpoint model state differs')
    if training[MODES[0]]['profile']['environment']!=old_training['profile']['environment']:raise ValueError('Old environment differs')
    reproduction=dict(comparable_epoch_fields=list(comparable),epochs_exactly_matched=20,validation_rows_exactly_matched=logit_count,
        endpoint_state_sha256_and_environment_equal=True,
        old_intermediate_state_or_rng_available=any(any(k in r for k in ('model_state_sha256','rng_before','rng_after')) for r in old_history),
        scope='No claim of old per-epoch model/RNG equality: old artifacts do not contain those fields.')
    bills=[r for r in ledger['billing_records'] if r['mode'].startswith('EXP019')]
    if (not math.isclose(sum(r['paid'] for r in bills),billing['stage_total'],abs_tol=1e-9)
            or sum(r['billed_seconds'] for r in bills if r['mode']=='EXP019_gpu')!=billing['gpu_seconds']
            or sum(r['billed_seconds'] for r in bills if 'cpu_only' in r['mode'])!=billing['cpu_seconds']):raise ValueError('Billing detail differs')
    for name,expected in audit['source_sha256'].items():
        if name in inputs and inputs[name]!=expected:raise ValueError('Consumed artifact differs from accepted audit: '+name)
    a.figures.mkdir(parents=True,exist_ok=False)
    ledger_name='.codex/logs/evidence/research-budget-20261101.json'
    ledger_snapshot=a.figures/'budget-ledger-snapshot.json'
    with ledger_snapshot.open('xb') as f:f.write(input_bytes[ledger_name])
    if sha(ledger_snapshot)!=inputs[ledger_name]:raise ValueError('Ledger snapshot differs from consumed bytes')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    font=FontProperties(fname=str(font_path))
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,
        'figure.dpi':130,'savefig.dpi':240,'pdf.fonttype':42,'ps.fonttype':42})
    figures=[]
    captions={
        'validation-bce':'同一开发划分固定136窗的20轮未加权BCE；圆点为按记录FP32 BCE预定规则选中的第2/3轮。左图完整范围，右图放大0—1区间，不用于重新选轮。',
        'negative-video-scores':'7条负视频、3个场次组的max聚合分数与预定0.5阈值。A/B视频AUC由全部24正+7负计算，不能从本图7条负视频单独计算；视频不是7个独立场次。'}
    def save(fig,name):
        for suffix in ('png','pdf'):
            target=a.figures/(name+'.'+suffix);fig.savefig(target,bbox_inches='tight')
            figures.append(dict(path=target.relative_to(ROOT).as_posix(),sha256=sha(target),caption=captions[name]))
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for m in MODES:
        y=[r['validation_unweighted_bce'] for r in histories[m]];selected=summaries[m]['best_epoch']
        for axis in axes:
            axis.plot(range(1,21),y,'o-',ms=3,lw=1.4,color=COLORS[m],label=LABELS[m])
            axis.scatter([selected],[y[selected-1]],s=80,facecolor=COLORS[m],edgecolor='white',linewidth=1,zorder=5)
            axis.annotate('%s 第%d轮'%(m==MODES[0] and 'A' or 'B',selected),(selected,y[selected-1]),
                xytext=(6,13 if m==MODES[0] else 28),textcoords='offset points',color=COLORS[m],fontsize=9)
    axes[0].set_title('完整数值范围');axes[1].set_title('局部放大：BCE 0—1');axes[1].set_ylim(0,1)
    for axis in axes:
        axis.set_xlabel('完成训练轮次');axis.set_ylabel('未加权验证 BCE');axis.set_xticks([1,2,3,5,10,15,20]);axis.grid(alpha=.2);axis.legend(loc='upper right')
    clipped={m:sum(r['validation_unweighted_bce']>1 for r in histories[m]) for m in MODES}
    fig.suptitle('EXP-019：真实顺序与固定置换的开发验证轨迹')
    fig.text(.02,.015,'seed7；每支20轮/420更新；固定136窗（96正/40负）。右图超出上界：A %d点、B %d点，完整数值见左图。'%(clipped[MODES[0]],clipped[MODES[1]]),fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.95));save(fig,'validation-bce')
    fig,axis=plt.subplots(figsize=(12.5,5.2))
    for j,m in enumerate(MODES):
        x=[i+(j-.5)*.34 for i in range(len(neg))];values=[indexed[m][u]['score'] for u in neg]
        metric=summaries[m]['video_metrics']
        bars=axis.bar(x,values,width=.32,color=COLORS[m],label='%s：FP=%d；视频AUC=%.4f'%(LABELS[m],metric['FP'],metric['AUC']))
        for bar,value in zip(bars,values):
            axis.text(bar.get_x()+bar.get_width()/2,value+.012,'%.3f'%value,ha='center',va='bottom',fontsize=8)
    axis.axhline(.5,color='#555555',linestyle='--',lw=1,label='固定报警阈值 0.5',zorder=0)
    axis.set_ylim(0,1.16);axis.set_xticks(range(len(neg)));axis.set_xticklabels(neg)
    axis.set_ylabel('视频分数（质量合格网格窗分数的最大值）');axis.set_title('7条负视频：报警计数与跨类别排序AUC是不同读数')
    axis.grid(axis='y',alpha=.2);axis.legend(loc='upper center',fontsize=9,ncol=3)
    positions={}
    for i,uid in enumerate(neg):positions.setdefault(indexed[MODES[0]][uid]['group_id'],[]).append(i)
    for group,ix in positions.items():
        _,site,date=group.split('|');axis.text(sum(ix)/len(ix),-.13,site+'\n'+date,ha='center',va='top',transform=axis.get_xaxis_transform(),fontsize=9)
        if ix[-1]<len(neg)-1:axis.axvline(ix[-1]+.5,color='#dddddd',lw=.8)
    fig.text(.02,.02,'来源：通过审计的31视频分数；24正/7负，负视频仅3组。AUC用全部31视频计算；此图展示负视频阈值越界，未扫描阈值。',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,1));save(fig,'negative-video-scores')
    A,B=(summaries[m] for m in MODES);vA,vB=A['video_metrics'],B['video_metrics']
    rows=[]
    for m in MODES:
        s=summaries[m];v=s['video_metrics']
        rows.append('|%s|%d|%.10f|%.10f|%d / %d / %d / %d|%.10f|%.4f|'
            %(LABELS[m],s['best_epoch'],s['clip_bce_fp64'],s['clip_metrics']['AUC'],v['TP'],v['FP'],v['FN'],v['TN'],v['AUC'],v['FPR']))
    def code_list(values):return '、'.join('`'+x+'`' for x in values) or '无'
    figure_relative=lambda n:(a.figures/(n+'.png')).relative_to(a.report.parent).as_posix()
    report=f'''# EXP-019：真实顺序与固定置换训练报告

使用了 data-usage-protocol skill。本轮按已授权的开发初筛范围执行：第6条沿用视频×类别共享每轮最多4窗的例外，同一原段仍最多64帧，未称严格等权；第10条采用预先冻结、场次隔离的单一开发划分，尚不构成正式组级交叉验证、未见机位验证或独立测试。`signed_by`未代填。

## 问题与对照

本轮检验：在当前3秒、K400预训练与微调配方下，保留真实时间排列，相比相同16张RGB图像的固定置换，是否呈现任务信号。A按真实顺序，B每个物理窗口固定一个排列；两支训练、固定验证与完整视频网格评价均使用各自相同规则。置换只改变模型输入T轴，不改物理帧集合、PTS、标签、空间像素或零D通道，也不消耗训练随机流。

两支复用EXP-018普通BCE的119训练/31开发验证和实际批次：每轮504窗（356P/148N），固定验证136窗（96P/40N）；各自从相同公开K400初始化重新训练20轮、420次更新。模型为R(2+1)D-18、`[RGB,0]`，FP32，BN统计固定；Adam骨干/头学习率1e-4/1e-3，有效批24、micro8，训练seed7，置换seed1901。分别按本支记录FP32未加权验证BCE的精确最小值选轮，同分取最早；表中BCE是审计由保存logit以FP64重算的一致性读数。

完整视频评价使用原31视频的{audit['checks']['dense_physical_windows']:,}个标签无关3秒/1.5秒网格窗，质量/cut排除、max聚合和固定0.5阈值。31视频含{vA['P']}正/{vA['N']}负，负视频来自{len(positions)}个场次组；重叠窗口和同组视频均不能当独立灾害样本。两支先各做独立1轮测速，正式训练重新初始化，不接续探针权重。

## 结果与误报差异

|输入条件|选中轮次|固定片段BCE（FP64）|片段AUC|视频 TP / FP / FN / TN|视频AUC|视频FPR|
|---|---:|---:|---:|---|---:|---:|
{chr(10).join(rows)}

两支均通过预登记最低非退化检查（片段AUC>0.5且概率极差≥0.05）。固定0.5阈值下，A与B均无视频漏报，A的FP由B的{vB['FP']}条变为{vA['FP']}条，满足预登记的FP/FN均不增加且至少一项减少条件，审计记为 `positive_development_signal={str(gate['positive_development_signal']).lower()}`。同时，B的视频AUC为{vB['AUC']:.10f}，高于A的{vA['AUC']:.10f}，必须与报警计数同等报告：AUC衡量跨类别评分排序，固定阈值报警则取决于分数相对0.5的位置，本次不是所有读数同向改善。

共同FP为{code_list(common)}；A独有FP为{code_list(a_only)}；B独有FP为{code_list(b_only)}。因此，相对B，A修复{len(b_only)}条误报，同时新增{len(a_only)}条误报，净少{len(fp[MODES[1]])-len(fp[MODES[0]])}条，错误集合并非完全包含。该差异不足以说明各类负场景都得到改善。

![20轮验证BCE与预定选轮点]({figure_relative('validation-bce')})

图1：{captions['validation-bce']} 全程数值有明显波动，完成20轮不等于末轮优于早期选中轮。

![7条负视频分数与固定阈值]({figure_relative('negative-video-scores')})

图2：{captions['negative-video-scores']} 可同时看到A独有的误报和B独有的误报，不能只报净差而省略新增错误。

## 可以支持的结论与限制

当前证据支持：在此3秒输入、K400预训练、seed7和复用开发划分下，真实排列相对固定置换满足预登记的有限正向任务信号。A的20轮可比输入/损失/梯度记录及全部{logit_count:,}条验证logit与旧EXP-018普通A精确相同，初始/最终模型SHA和环境一致；新A没有形成相对旧RGB普通基线的性能改善。旧EXP-018没有逐轮模型state/RNG记录，不能追称这些中间轨迹也已精确复现。

本轮不提出新模型或创新性能结论，也不证明模型理解了泥石流的物理运动。固定置换保留16帧的外观变化，同时改变局部相邻关系及与K400预训练的适配，无法把差异全部归因于灾害内在动态。单seed、7条负视频/3个负组、反复复用的开发验证限制了结论；不作显著性、独立泛化、未见机位或部署主张，不把{audit['checks']['dense_physical_windows']:,}个重叠窗当独立重复试验。预登记门槛通过只支持继续聚焦时序问题，不自动授权新增训练、扫描seed/阈值或方法晋级。

## 跨度覆盖与剩余证据

现行v4的纯元数据核查覆盖150条及186个原监督段。按视频自身标签，3秒合法输入覆盖{span['overall']['own_label_3s']}/{span['overall']['videos']}，7.5秒及同名义中心配对均覆盖{span['overall']['own_label_7p5s']}/{span['overall']['videos']}；旧22条中，长窗覆盖{span['old22']['coverage']['own_label_7p5s']}/22，零长窗10条均在训练侧。该结果是在修正后v4上重新核查，媒体身份继承冻结SHA，没有重读媒体/锚图或验证像素。可行性不是性能比较，不能据此认定3秒优于7.5秒，也不据此删除10条。3秒/7.5秒的匹配性能证据仍未完成。

后续若要形成方法贡献，仍需围绕导师提出的时序利用问题，先确定可证伪的机制与匹配对照，再按预登记完成训练侧选轮的组级评价和独立测试。外部材料仍为候选，正式独立测试尚未成立且未评分；本报告不把新下载文件或同源时间块视作新增独立事件。本轮到此结束，不自动追加训练。

## 运行、费用与复现

run_id：`{a.run.name}`。正式离线审计`audit-v1/audit.json`为`audit_passed`，耗时{audit['seconds']:.3f}秒；统一pipeline耗时{pipeline['seconds']:.3f}秒。账单核实GPU {billing['gpu_seconds']}秒、无卡 {billing['cpu_seconds']}秒，阶段实付{billing['stage_total']:.2f}元（{' + '.join(f"{r['paid']:.2f}" for r in bills)}），本轮20元总账已付{billing['goal_total']:.2f}元，本轮授权预算剩余{billing['remaining_authorized']:.2f}元，实例状态`{billing['instance_state']}`。费用使用最终账单，不以pipeline预估金额替代。

报告/图生成命令：`.venv/Scripts/python.exe -B experiments/runspecs/EXP-019/report.py`。入口拒绝覆盖既有报告或图目录，只读JSON/JSONL，不运行模型、训练或解码。图以PNG/PDF保存；`figures/EXP-019-v1/figure-source.json`保存逐项输入/源码SHA、图SHA、完整曲线与负视频分数、图注及复现核对范围。

证据入口：[正式审计](../../../data/results/exp019/{a.run.name}/audit-v1/audit.json)、[完整视频分数](../../../data/results/exp019/{a.run.name}/two-dense/result.json)、[跨度核查](../../../{span_path.relative_to(ROOT).as_posix()})、[最终费用](../../../.codex/logs/evidence/exp019-final-billing.json)、[总账](../../../.codex/logs/evidence/research-budget-20261101.json)。
'''
    a.report.parent.mkdir(parents=True,exist_ok=True)
    with a.report.open('x',encoding='utf-8') as f:f.write(report)
    for name,expected in inputs.items():
        if name!=ledger_name and sha(ROOT/name)!=expected:raise RuntimeError('Consumed input changed while generating report: '+name)
    with (a.figures/'generator-source.py').open('xb') as f:f.write(SOURCE.read_bytes())
    evidence=dict(status='audited_result_report_generated',run_id=a.run.name,command=sys.argv if argv is None else argv,
        input_sha256=inputs,input_snapshots={ledger_name:dict(path=ledger_snapshot.relative_to(ROOT).as_posix(),sha256=sha(ledger_snapshot),scope='exact consumed bytes; mutable live ledger is not required to retain this SHA')},
        generator_sha256=sha(SOURCE),matplotlib_version=matplotlib.__version__,font_path=str(font_path),
        font_sha256=sha(font_path),report_sha256=sha(a.report),figures=figures,reproduction=reproduction,
        registered_gate=gate,negative_video_scores=[dict(unit_id=u,group_id=indexed[MODES[0]][u]['group_id'],
            **{m:indexed[m][u]['score'] for m in MODES}) for u in neg],
        validation_recorded_fp32_bce={m:[r['validation_unweighted_bce'] for r in histories[m]] for m in MODES},
        no_model_execution=True,no_media_decode=True,conclusion_scope=audit['scope'])
    with (a.figures/'figure-source.json').open('x',encoding='utf-8') as f:json.dump(evidence,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=evidence['status'],report=str(a.report),figures=[r['path'] for r in figures],report_sha256=evidence['report_sha256']),ensure_ascii=True))


if __name__=='__main__':main()
