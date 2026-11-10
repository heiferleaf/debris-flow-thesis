# -*- coding: utf-8 -*-
"""判定线与误报诊断（只读，2026-10-27 立）。

回答三个问题，对应三个子命令：

  list-fp   0.5 判定线上的误报是哪几条视频，各自有多少片段过线。
            用来把误报分成「整段饱和」(几乎全部片段都过线) 与「个别片段冒尖」
            (只有一两个片段过线) 两类，两类的原因和修法不同。

  line      判定线本身能不能挪。测一条能上线的规则：先在某个机位录一段确认
            没有泥石流的画面，把这批已知负样本视频分数的第 q 百分位作为判定线，
            之后的视频超过就报警。折外执行（参考批次只取不属于被评估那一折的
            负样本），按场次组有放回重抽算区间。对照的基线是训练卡片写死的 0.5。

  saturation 分数分布。看视频级分数有多少落在 0 和 1 两端，这决定了挪判定线
            有没有空间：如果 90% 的分数都在两端，那么中间没有可以放线的地方。

  appearance 检验「分数跟着画面亮度或灰白面积走」这两个猜测。两条都被实测否掉，
            保留这个子命令是为了让「否掉」这个结论可以复算，不是留作结论。

  aggregate 把「整条视频的分数 = 可用片段里取最大」这一条聚合规则换成别的，看误报和
            漏报怎么变。只读已经落盘的片段分数，不训练、不改判定线。用途是回答
            「要不要另训一个图像复核网络来处理个别片段冒尖的误报」。

用法（仓库根目录）：

    python experiments/tools/diagnose_operating_point.py list-fp
    python experiments/tools/diagnose_operating_point.py line
    python experiments/tools/diagnose_operating_point.py saturation
    python experiments/tools/diagnose_operating_point.py appearance
    python experiments/tools/diagnose_operating_point.py aggregate

默认读 EXP-009 强标签微调那一次 run 的分数（150 条视频池）。
"""
import argparse
import csv
import pathlib
from collections import Counter

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / "data/results/exp008/EXP-009-strong-v2-150-strict-ft-noaug"
MANIFEST = REPO / "data/process/EXP-008-units-v2/unit-manifest.csv"
TH = 0.5
QS = [10, 30, 50, 70, 80, 90, 95, 98, 99, 99.5]
N_BOOT = 20000
SEED = 20261027


def load(run_dir):
    d = np.load(pathlib.Path(run_dir) / "clip-scores.npz", allow_pickle=True)
    ids = [str(x) for x in d["unit_ids"]]
    meta = {}
    with open(MANIFEST, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[r["unit_id"]] = r
    return dict(
        ids=ids,
        run_dir=pathlib.Path(run_dir),
        score=d["unit_score"].astype(np.float64),
        clip=d["clip_scores"].astype(np.float64),
        mask=d["mask"],
        label=d["labels"].astype(int),
        fold=d["folds"].astype(int),
        group=np.array([meta[i]["group_id"] for i in ids]),
        meta=meta,
    )


def metrics(tp, fp, fn, tn):
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, Precision=prec, Recall=rec, FPR=fpr, F1=f1)


def rule_reference(d, q, idx=None, th=TH):
    """折外参考批次定线：判定线取「不属于本折的负样本」分数的第 q 百分位。"""
    score, label, fold = d["score"], d["label"], d["fold"]
    if idx is not None:
        score, label, fold = score[idx], label[idx], fold[idx]
    tp = fp = fn = tn = 0
    thrs = []
    for f in sorted(set(fold.tolist())):
        ref = (label == 0) & (fold != f)
        if ref.sum() < 2:
            continue
        t = float(np.percentile(score[ref], q))
        thrs.append(t)
        sel = fold == f
        pred = score[sel] >= t
        y = label[sel]
        tp += int((pred & (y == 1)).sum())
        fp += int((pred & (y == 0)).sum())
        fn += int((~pred & (y == 1)).sum())
        tn += int((~pred & (y == 0)).sum())
    return tp, fp, fn, tn, thrs


def rule_fixed(d, th=TH, idx=None):
    score, label = d["score"], d["label"]
    if idx is not None:
        score, label = score[idx], label[idx]
    pred = score >= th
    return (int((pred & (label == 1)).sum()), int((pred & (label == 0)).sum()),
            int((~pred & (label == 1)).sum()), int((~pred & (label == 0)).sum()))


def cmd_list_fp(d, th=TH):
    fp = np.where((d["label"] == 0) & (d["score"] >= th))[0]
    tp, nfp, fn, tn = rule_fixed(d, th)
    print(f"{th} 判定线上：TP {tp} / FP {nfp} / FN {fn} / TN {tn}"
          f"（正样本 {int((d['label']==1).sum())} 条、负样本 {int((d['label']==0).sum())} 条）\n")
    print(f"误报 {len(fp)} 条：")
    print(f"{'视频':<9}{'折':<4}{'分数':<13}{'过线片段':<11}{'场次组'}")
    for i in fp:
        m = d["meta"][d["ids"][i]]
        over = int((d["clip"][i][d["mask"][i]] >= th).sum())
        use = int(d["mask"][i].sum())
        print(f"{d['ids'][i]:<9}{d['fold'][i]:<4}{d['score'][i]:<13.6f}{over:>2}/{use:<7}{m['group_id']}")
    print("\n按场次组归类：")
    cnt = Counter(d["meta"][d["ids"][i]]["group_id"] for i in fp)
    for g, n in cnt.most_common():
        tot = sum(1 for k, v in d["meta"].items()
                  if v["group_id"] == g and v["label"] == "neg")
        print(f"  {g:<32} 误报 {n} 条 / 该组负样本共 {tot} 条")
    print("\n两类分开看：")
    whole, spike = [], []
    for i in fp:
        r = (d["clip"][i][d["mask"][i]] >= th).mean()
        (whole if r >= 0.9 else spike).append(d["ids"][i])
    print(f"  整段饱和（≥90% 片段过线）{len(whole)} 条：{' '.join(whole)}")
    print(f"  个别片段冒尖（<90% 片段过线）{len(spike)} 条：{' '.join(spike)}")


def cmd_line(d):
    pos, neg = d["score"][d["label"] == 1], d["score"][d["label"] == 0]
    print(f"视频 {len(d['ids'])} 条，正 {len(pos)} 负 {len(neg)}，"
          f"场次组 {len(set(d['group'].tolist()))} 个\n")
    print("基线（训练卡片写死的 0.5）：")
    m = metrics(*rule_fixed(d))
    print("  TP={TP} FP={FP} FN={FN} TN={TN} Precision={Precision:.4f} "
          "Recall={Recall:.4f} FPR={FPR:.4f} F1={F1:.4f}\n".format(**m))

    print("参考负样本定线（折外）：")
    print("   q     五折判定线                                 TP  FP  FN  Precision  Recall  FPR     F1")
    for q in QS:
        tp, fp, fn, tn, thrs = rule_reference(d, q)
        m = metrics(tp, fp, fn, tn)
        tstr = " ".join(f"{t:.5f}" for t in thrs)
        print(f"  {q:>5}  {tstr}  {tp:>3} {fp:>3} {fn:>3}   "
              f"{m['Precision']:.4f}  {m['Recall']:.4f}  {m['FPR']:.4f}  {m['F1']:.4f}")

    print(f"\n按场次组有放回重抽 {N_BOOT} 次的 95% 区间：")
    rng = np.random.default_rng(SEED)
    groups = sorted(set(d["group"].tolist()))
    members = {g: np.where(d["group"] == g)[0] for g in groups}
    targets = [("固定 0.5", None)] + [(f"q={q}", q) for q in QS]
    acc = {n: {k: [] for k in ("TP", "FP", "FN", "TN")} for n, _ in targets}
    for _ in range(N_BOOT):
        draw = rng.choice(len(groups), size=len(groups), replace=True)
        idx = np.concatenate([members[groups[i]] for i in draw])
        for name, q in targets:
            out = rule_fixed(d, TH, idx) if q is None else rule_reference(d, q, idx)[:4]
            for k, v in zip(("TP", "FP", "FN", "TN"), out):
                acc[name][k].append(v)

    def ci(num, den):
        num, den = np.asarray(num, float), np.asarray(den, float)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = num / den
        r = r[np.isfinite(r)]
        return (np.percentile(r, 2.5), np.percentile(r, 97.5)) if r.size else (np.nan, np.nan)

    print(f"  {'规则':<10}{'FPR 点值':>9}{'FPR 95% 区间':>20}{'Recall 点值':>12}{'Recall 95% 区间':>20}")
    for name, q in targets:
        m = metrics(*rule_fixed(d)) if q is None else metrics(*rule_reference(d, q)[:4])
        a = acc[name]
        lo, hi = ci(a["FP"], np.add(a["FP"], a["TN"]))
        rlo, rhi = ci(a["TP"], np.add(a["TP"], a["FN"]))
        print(f"  {name:<10}{m['FPR']:>9.4f}  [{lo:.4f}, {hi:.4f}]"
              f"{m['Recall']:>11.4f}  [{rlo:.4f}, {rhi:.4f}]")


def cmd_saturation(d):
    s, l = d["score"], d["label"]
    print("150 条视频的分数分布")
    for lo, hi in [(0, 1e-4), (1e-4, 0.01), (0.01, 0.1), (0.1, 0.5),
                   (0.5, 0.9), (0.9, 0.99), (0.99, 0.9999), (0.9999, 1.0000001)]:
        sel = (s >= lo) & (s < hi)
        print(f"  [{lo:g}, {hi:g})  {int(sel.sum()):>3} 条   正 {int((sel & (l == 1)).sum()):>3}  "
              f"负 {int((sel & (l == 0)).sum()):>3}")
    mid = int(((s > 0.01) & (s < 0.99)).sum())
    print(f"\n落在 (0.01, 0.99) 中间的只有 {mid} 条（{mid/len(s):.0%}）")
    m = d["mask"]
    frac = float(((d["clip"][m] > 0.05) & (d["clip"][m] < 0.95)).mean())
    print(f"片段级：落在 (0.05, 0.95) 中间的占 {frac:.2%}")
    m99 = s >= 0.9999
    p, n = np.sort(s[m99 & (l == 1)]), np.sort(s[m99 & (l == 0)])
    if n.size:
        print(f"\n≥0.9999 这一档里：正样本 {p.size} 条（{p.min():.9f}–{p.max():.9f}），"
              f"负样本 {n.size} 条（{n.min():.9f}–{n.max():.9f}）")
        print(f"  两者 1−分数 的量级：正 {1-p.max():.1e}–{1-p.min():.1e}，"
              f"负 {1-n.max():.1e}–{1-n.min():.1e}；float32 在 1 附近的最小间隔约 1.19e-7")


def cmd_appearance(d):
    """检验两个外观假设，两个都应被否掉。"""
    from PIL import Image
    from scipy.stats import spearmanr

    frames = REPO / "data/process/EXP-008-units-v2/frames"
    rows = []
    for i, u in enumerate(d["ids"]):
        for c in range(len(d["clip"][i])):
            if not d["mask"][i][c]:
                continue
            p = frames / u / f"clip{c:02d}_f00.jpg"
            if not p.exists():
                continue
            a = np.asarray(Image.open(p).convert("RGB").resize((64, 36)), dtype=np.float32) / 255.0
            mx, mn = a.max(2), a.min(2)
            sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
            b = a.mean(2)
            rows.append((i, float(b.mean()), float(((sat < 0.15) & (b > 0.55)).mean()),
                         d["clip"][i][c]))
    vid = np.array([r[0] for r in rows])
    bright = np.array([r[1] for r in rows])
    grey = np.array([r[2] for r in rows])
    sco = np.array([r[3] for r in rows])
    print(f"片段 {len(sco)} 个，来自 {len(set(vid.tolist()))} 条视频\n")
    for name, x in (("平均亮度", bright), ("灰白面积比", grey)):
        rho, p = spearmanr(x, sco)
        print(f"  全部片段：分数 vs {name}  Spearman {rho:+.3f}（p={p:.1e}）")
    for name, x in (("平均亮度", bright), ("灰白面积比", grey)):
        rs = []
        for i in set(vid.tolist()):
            sel = vid == i
            if sel.sum() < 10 or sco[sel].max() - sco[sel].min() < 0.05:
                continue
            rs.append(spearmanr(x[sel], sco[sel])[0])
        print(f"  同一条视频内部：{len(rs)} 条可测，rho({name}) 中位 {np.median(rs):+.3f}")

    # 正负样本分开看：如果某个量能解释分数，它在两类之间的差别应该与分数差别同向同量级。
    lab = d["label"][vid]
    print("\n正负样本分开看（片段级中位数）：")
    for nm, v in (("正样本", 1), ("负样本", 0)):
        sel = lab == v
        print(f"  {nm} {int(sel.sum()):>4} 个片段：灰白面积比 {np.median(grey[sel]):.3f}，"
              f"平均亮度 {np.median(bright[sel]):.3f}，分数 {np.median(sco[sel]):.4f}")
    print("\n结论：两条都得不到支持，分数不能用画面亮度或灰白面积解释。")


def cmd_aggregate(d):
    """换掉「取最大」这条聚合规则，看误报与漏报怎么变。只读，不改判定线。"""
    usable = [d["clip"][i][d["mask"][i]] for i in range(len(d["ids"]))]
    label = d["label"]

    def agg(name):
        if name == "最大":
            return np.array([a.max() for a in usable])
        if name == "中位数":
            return np.array([np.median(a) for a in usable])
        if name == "均值":
            return np.array([a.mean() for a in usable])
        if name.endswith("大"):
            k = int(name[1:-1])
            return np.array([np.sort(a)[::-1][min(k, len(a)) - 1] for a in usable])
        if name.startswith("≥"):
            frac = float(name[1:].rstrip("%")) / 100
            # 要求「过线片段占可用片段的比例达到 frac」，返回 0/1，再由 stats 按判定线比。
            return np.array([float((a >= TH).mean() >= frac) for a in usable])
        raise ValueError(name)

    def stats(score, lab):
        pred = score >= TH
        return (int((pred & (lab == 1)).sum()), int((pred & (lab == 0)).sum()),
                int((~pred & (lab == 1)).sum()), int((~pred & (lab == 0)).sum()))

    base = np.array([a.max() for a in usable])
    print(f"核对：落盘的视频分数是否等于可用片段取最大 —— "
          f"{'是' if np.allclose(d['score'], base) else '否'}\n")

    rules = ["最大", "第2大", "第3大", "第5大", "中位数", "均值", "≥10%", "≥25%", "≥50%"]
    print("「第k大」= 把片段分数从高到低排，取第 k 个；「≥x%」= 过线片段占可用片段的比例达到 x%。")
    print(f"判定线一律 {TH}，负样本共 {int((label == 0).sum())} 条。\n")
    print(f"{'聚合方式':<8}{'TP':>5}{'FP':>4}{'FN':>5}{'TN':>5}"
          f"{'Precision':>11}{'Recall':>8}{'FPR':>8}{'F1':>8}")
    for name in rules:
        m = metrics(*stats(agg(name), label))
        print(f"{name:<8}{m['TP']:>5}{m['FP']:>4}{m['FN']:>5}{m['TN']:>5}"
              f"{m['Precision']:>11.4f}{m['Recall']:>8.4f}{m['FPR']:>8.4f}{m['F1']:>8.4f}")

    groups = sorted(set(d["group"].tolist()))
    members = {g: np.where(d["group"] == g)[0] for g in groups}
    rng = np.random.default_rng(SEED)
    draws = [rng.choice(len(groups), size=len(groups), replace=True) for _ in range(N_BOOT)]
    print(f"\n按场次组有放回重抽 {N_BOOT} 次的 95% 区间：")
    for name in ("最大", "第2大", "第5大", "中位数"):
        s = agg(name)
        fprs, recs = [], []
        for dr in draws:
            idx = np.concatenate([members[groups[i]] for i in dr])
            m = metrics(*stats(s[idx], label[idx]))
            fprs.append(m["FPR"])
            recs.append(m["Recall"])
        fprs, recs = np.asarray(fprs), np.asarray(recs)
        fprs, recs = fprs[np.isfinite(fprs)], recs[np.isfinite(recs)]
        m = metrics(*stats(s, label))
        print(f"  {name:<8}FPR {m['FPR']:.4f} [{np.percentile(fprs, 2.5):.4f}, "
              f"{np.percentile(fprs, 97.5):.4f}]   Recall {m['Recall']:.4f} "
              f"[{np.percentile(recs, 2.5):.4f}, {np.percentile(recs, 97.5):.4f}]")

    print("\n「取第 k 高分」逐个数扫一遍（看 k 有没有唯一解）：")
    print(f"  {'k':>3}{'TP':>5}{'FP':>4}{'FN':>5}{'TN':>5}")
    for k in range(1, 13):
        name = "最大" if k == 1 else f"第{k}大"
        m = metrics(*stats(agg(name), label))
        print(f"  {k:>3}{m['TP']:>5}{m['FP']:>4}{m['FN']:>5}{m['TN']:>5}")

    print("\n「第5大」还剩哪些误报：")
    s5 = agg("第5大")
    for i in np.where((label == 0) & (s5 >= TH))[0]:
        print(f"  {d['ids'][i]:<8}{s5[i]:>10.6f}  过线片段 "
              f"{int((usable[i] >= TH).sum()):>2}/{len(usable[i]):<3}{d['group'][i]}")

    print("\n「中位数」把哪些正样本判没了：")
    sm = agg("中位数")
    for i in np.where((label == 1) & (sm < TH))[0]:
        print(f"  {d['ids'][i]:<8}中位 {sm[i]:.6f}  最大 {usable[i].max():.6f}  "
              f"过线片段 {int((usable[i] >= TH).sum()):>2}/{len(usable[i])}")

    print("\n注意：「第k大」的 k 是在同一批分数上挑出来的，属于在评估集上选参数。"
          "\n拿它当方法必须先在实验卡片里把 k 的取值理由写死（例如按「泥石流至少占到"
          "\n整条录像的百分之几」定 k），再另跑一次；本子命令给的是诊断，不是成绩。")


def auc(pos, neg):
    """梯形 AUC：AUC = 随机取一条正样本和一条负样本，正样本分数更高的概率。"""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def cmd_folds(d):
    """逐折数混淆矩阵：误报是不是平均摊在五个折上。"""
    label, score, fold = d["label"], d["score"], d["fold"]
    folds = sorted(set(fold.tolist()))
    rows, tot = [], dict(TP=0, FP=0, FN=0, TN=0)
    for f in folds:
        s = fold == f
        m = metrics(int(((score[s] >= TH) & (label[s] == 1)).sum()),
                    int(((score[s] >= TH) & (label[s] == 0)).sum()),
                    int(((score[s] < TH) & (label[s] == 1)).sum()),
                    int(((score[s] < TH) & (label[s] == 0)).sum()))
        m["fold"] = f
        m["P"] = int((label[s] == 1).sum())
        m["N"] = int((label[s] == 0).sum())
        m["AUC"] = auc(score[s & (label == 1)], score[s & (label == 0)])
        rows.append(m)
        for k in tot:
            tot[k] += m[k]
    print(f"判定线 {TH}，视频分数 = 参与打分的片段里取最大；按训练时的折归属拆开：\n")
    print(f"  {'折':<4}{'P':>4}{'N':>4}{'TP':>5}{'FP':>4}{'FN':>4}{'TN':>4}"
          f"{'Recall':>9}{'FPR':>8}{'AUC':>9}")
    for m in rows:
        print(f"  {m['fold']:<4}{m['P']:>4}{m['N']:>4}{m['TP']:>5}{m['FP']:>4}"
              f"{m['FN']:>4}{m['TN']:>4}{m['Recall']:>9.3f}{m['FPR']:>8.3f}{m['AUC']:>9.4f}")
    m = metrics(tot["TP"], tot["FP"], tot["FN"], tot["TN"])
    print(f"  {'合计':<4}{sum(r['P'] for r in rows):>4}{sum(r['N'] for r in rows):>4}"
          f"{m['TP']:>5}{m['FP']:>4}{m['FN']:>4}{m['TN']:>4}"
          f"{m['Recall']:>9.4f}{m['FPR']:>8.4f}{auc(score[label == 1], score[label == 0]):>9.4f}")

    print("\n每折正负样本在分数轴上有没有排开（看最低的正样本与最低的负样本谁高）：")
    for f in folds:
        s = fold == f
        pos, neg = score[s & (label == 1)], score[s & (label == 0)]
        if not len(pos) or not len(neg):
            continue
        print(f"  折{f}: 正样本最低 {pos.min():.8f}   负样本最低 {neg.min():.8f}   "
              f"负样本最高 {neg.max():.8f}")
        if neg.min() > pos.min():
            gap = float(neg.min() - pos.min())
            ulp = 2.0 ** -24
            lg = float(np.log(neg.min() / (1 - neg.min())) - np.log(pos.min() / (1 - pos.min())))
            print("         ↑ 负样本的最低分高于正样本的最低分，正负在分数轴上没有排开")
            print(f"           两者差 {gap:.3e}，是 float32 在 1 附近最小间隔（{ulp:.2e}）的 "
                  f"{gap / ulp:.0f} 倍；换回对数几率差 {lg:.3f}")
            print("           → 这个差远大于浮点间隔，排序是模型真实给出的，不是舍入造成的。")

    print("\n每条视频「48 个片段里分数 ≥ 0.99 的比例」的中位数（饱和程度）：")
    clip, mask = d["clip"], d["mask"]
    for f in folds:
        for v, nm in ((1, "正"), (0, "负")):
            sel = [i for i in range(len(label)) if fold[i] == f and label[i] == v]
            if not sel:
                continue
            fr = [(clip[i][mask[i]] >= 0.99).mean() for i in sel]
            print(f"  折{f} {nm}样本 {len(sel):>2} 条：中位 {np.median(fr):>5.3f}")
    print("\n注意：AUC 是排序指标。分数全部挤在 1.0 附近时，只要有一条视频排错，"
          "\nAUC 就从 1.0000 掉下来，这个数字可能只由一两条视频决定，不能读成"
          "\n「这一折整体分得开」。要看清楚是哪几条在决定它，就对照上面那行"
          "\n「负样本最低分 vs 正样本最低分」，再用 list-fp 把分数接近的视频列出来。")


def cmd_models(d):
    """同一批画面被五个模型各打一遍分：看误报是画面的问题还是某一个模型的问题。"""
    run = d["run_dir"]
    files = sorted(run.glob("clip-scores-forcef*.npz"))
    if not files:
        print(f"{run} 里没有 clip-scores-forcef*.npz，无法做这一步。")
        return
    nets = {f.stem.split("forcef")[-1]: np.load(f, allow_pickle=True) for f in files}
    ks = sorted(nets)
    if [str(x) for x in nets[ks[0]]["unit_ids"]] != d["ids"]:
        print("各文件里的视频顺序不一致，不能逐条对照。")
        return
    S = {k: nets[k]["unit_score"].astype(np.float64) for k in ks}
    label, fold = d["label"], d["fold"]

    print("每个模型在它自己那折验证集上的成绩，以及拿去打全池负样本时报成「有」的条数：")
    print(f"  {'模型':<6}{'该折 P/N':>11}{'该折 AUC':>10}{'该折 FP/FN':>12}{'全池负样本里报成有':>20}")
    for k in ks:
        s, v = S[k], fold == int(k)
        p, n = s[v & (label == 1)], s[v & (label == 0)]
        fp = int(((n >= TH)).sum())
        fn = int(((p < TH)).sum())
        print(f"  {k:<6}{f'{len(p)}/{len(n)}':>11}{auc(p, n):>10.4f}"
              f"{f'{fp}/{fn}':>12}{int(((s >= TH) & (label == 0)).sum()):>20}")
    print("  （最后一列里每条视频都被其中几个模型训练过，只能横向比，不能当成绩。）")

    vote = sum((S[k] >= TH).astype(int) for k in ks)
    print("\n五个模型分歧最大的负样本（被至少一个模型报成有）：")
    print(f"  {'视频':<8}" + "".join(f"{'模型' + k:>10}" for k in ks))
    for i in np.where((label == 0) & (vote > 0))[0]:
        print(f"  {d['ids'][i]:<8}" + "".join(f"{S[k][i]:>10.4f}" for k in ks))
    only = np.where((label == 0) & (vote == 1))[0]
    n_multi = int(((label == 0) & (vote > 0)).sum())
    print(f"\n上面这一张表是「被至少一个模型报成有的负样本」，共 {n_multi} 条，不是判定线上的误报条数；"
          f"这 {n_multi} 条里有 {len(only)} 条只有 1 个模型报："
          + ("（" + "、".join(d["ids"][i] for i in only) + "）" if len(only) else ""))
    print("  这一类只要换一个模型就不会报，属于单个模型的不稳定，不是画面的问题。")

    fp_idx = np.where((label == 0) & (d["score"] >= TH))[0]
    print(f"\n判定线 {TH} 上的 {len(fp_idx)} 条误报，再按「有几个模型也报成有」拆一遍"
          f"（这一组数才是引用误报成绩时要写的）：")
    for n in sorted({int(vote[i]) for i in fp_idx}):
        vs = [d["ids"][i] for i in fp_idx if vote[i] == n]
        print(f"  被 {n} 个模型报成有：{len(vs)} 条（{'、'.join(vs)}）")
    no4 = sum(1 for i in fp_idx if S.get("4", np.zeros(len(label)))[i] >= TH)
    contra = sum(1 for i in fp_idx if vote[i] < len(ks))
    print(f"  这 {len(fp_idx)} 条误报里，模型 4 报成有的 {no4} 条；"
          f"五个模型里至少有一个给到 {TH} 以下的 {contra} 条"
          f"（这部分换个模型就不报，属于模型间训练不稳定，不是画面认不出）。")


def cmd_threshold(d):
    """折外选判定线：候选线 = 开发折上出现过的视频分数，不取整。

    规则一（漏报0条里误报最少）与规则二（F1 最大）都在开发折上选线、用到留出折。
    并列时取候选里最低的线。候选不取整——如果先对候选分数四舍五入再扫，
    选出的线会变成另一个数（如 0.9753），但留出折的混淆矩阵不变。
    """
    score, label, fold = d["score"], d["label"], d["fold"]
    for name in ("漏报0条里误报最少", "F1最大"):
        rows, tot = [], [0, 0, 0, 0]
        for f in sorted(set(fold.tolist())):
            dev = fold != f
            cand = np.unique(score[dev])
            best = None
            for t in cand:
                p = score[dev] >= t
                tp = int((p & (label[dev] == 1)).sum())
                fp = int((p & (label[dev] == 0)).sum())
                fn = int((~p & (label[dev] == 1)).sum())
                if name == "漏报0条里误报最少" and fn > 0:
                    continue
                key = (2 * tp / (tp + fp + fn)) if name == "F1最大" else -fp
                if best is None or key > best[0]:
                    best = (key, float(t))
            t = best[1]
            sel = fold == f
            p = score[sel] >= t
            tp = int((p & (label[sel] == 1)).sum())
            fp = int((p & (label[sel] == 0)).sum())
            fn = int((~p & (label[sel] == 1)).sum())
            tn = int((~p & (label[sel] == 0)).sum())
            rows.append((f, t, tp, fp, fn, tn))
            for i, v in enumerate((tp, fp, fn, tn)):
                tot[i] += v
        print(f"\n规则：{name}；候选线 = 开发折上出现过的视频分数（不取整），并列取最低的线")
        print(f"  {'折':<4}{'选出的判定线':>14}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}")
        for f, t, tp, fp, fn, tn in rows:
            print(f"  {f:<4}{t:>14.6f}{tp:>5}{fp:>5}{fn:>5}{tn:>5}")
        tp, fp, fn, tn = tot
        m = metrics(tp, fp, fn, tn)
        print(f"  {'合计':<4}{'':>14}{tp:>5}{fp:>5}{fn:>5}{tn:>5}"
              f"   Precision {m['Precision']:.4f}  Recall {m['Recall']:.4f}  "
              f"FPR {m['FPR']:.4f}  F1 {m['F1']:.4f}")
        th_of = {f: t for f, t, *_ in rows}
        lost = [(d["ids"][i], float(score[i]), int(fold[i]))
                for i in range(len(label))
                if label[i] == 1 and score[i] < th_of[int(fold[i])]]
        if lost:
            print("  被这条线漏掉的正样本：" + "、".join(
                f"{u}（{s:.4f}，第 {f} 折，该折线 {th_of[f]:.6f}）"
                for u, s, f in lost))
    print("\n注意：开发折上的分数来自「曾在留出折上训练过」的模型，所以这里的合计"
          "\n仍不是完全干净的成绩；完全干净的做法是嵌套交叉验证。")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list-fp", "line", "saturation", "appearance",
                                    "aggregate", "folds", "models", "threshold"])
    ap.add_argument("--run", default=str(DEFAULT_RUN), help="run 目录（含 clip-scores.npz）")
    ap.add_argument("--th", type=float, default=TH,
                    help=f"list-fp 用的判定线，默认 {TH}；抬高判定线看漏报代价时改这里")
    a = ap.parse_args()
    d = load(a.run)
    if a.cmd == "list-fp":
        cmd_list_fp(d, a.th)
        return
    {"list-fp": cmd_list_fp, "line": cmd_line,
     "saturation": cmd_saturation, "appearance": cmd_appearance,
     "aggregate": cmd_aggregate, "folds": cmd_folds, "models": cmd_models,
     "threshold": cmd_threshold}[a.cmd](d)


if __name__ == "__main__":
    main()
