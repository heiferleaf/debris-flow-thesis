# -*- coding: utf-8 -*-
"""从片段级分数看模型对「时间顺序」的依赖程度，并做两套设定之间的逐视频配对比较。

输入是 `experiments/tools/clip_scores_ft.py` 写出的三份分数文件（同一份权重、
同一批帧，只有 16 帧的排列不同）：
  clip-scores.npz            原样（16 帧按时间顺序）
  clip-scores-reverse.npz    倒序（16 帧倒过来，其余不变）
  clip-scores-freeze.npz     冻结（16 帧全换成中间那一帧，时间被抹平）

两种用法：

一、单套设定自检（报告用，看这套设定的模型到底用没用时间顺序）：
    python experiments/tools/motion_effect.py --run data/results/exp008/<run_id>

二、两套设定配对（EXP-014 主判据，看哪一套更依赖时间顺序）：
    python experiments/tools/motion_effect.py \
        --run-a data/results/exp008/EXP-014-strong-v2-128-strict-ft-noaug \
        --run-b data/results/exp008/EXP-014-strong-v3-128-strict-ft-noaug

  两套设定必须是同一批视频、同一套折：逐视频配对才有意义，脚本会逐位核对
  unit_ids / labels / folds 三列，不一致直接退出；同一套设定内部，倒序与冻结
  两份分数文件的这三列也必须与原样那份一致，否则同样退出。

判定规则（EXP-014 卡片第五节，跑之前写死）：
  每条视频的读数 = 它全部可用片段上的 |倒序分数 − 正序分数| 的均值
  （「可用片段」= 整块落在某段人工标注区间里的片段，掩码为 True 的那些；
   不分掩码、对全部 48 个片段取均值的结果只作对照，不作判据）；
  A 是原间隔、B 是新间隔，配对差 = B 的读数 − A 的读数；
  按视频有放回重抽 20000 次（每次在各折内部按该折规模抽），取 2.5% 与 97.5%
  分位数作 95% 区间。
  判「新间隔更依赖时间顺序」需同时满足三条：
    ① 配对差均值 > 0；② 区间不含 0；③ 第 1、2、3、5 折这四折的折内均值全为正
      （第 4 折只剩 2 条负样本，只报不作条件）。
  方向（倒序后分数整体是升还是降）单独报，不算通过条件。

本脚本只读分数文件，不训练、不改数据。随机种子固定，重复运行结果一致。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.paths import resolve_path  # noqa: E402
SEED = 20261025
N_BOOT = 20000
FOLD4 = 4          # 该折只有 2 条负样本，只报不作判据
NEED_FOLDS = [1, 2, 3, 5]


def auc(y, s):
    """秩和法的 AUC（并列取平均秩）。与 motion_signal_probe.py 的实现一致。"""
    y = np.asarray(y, dtype=np.int8)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[y == 1].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def load_scores(path):
    path = resolve_path(path)
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=True)
    return {
        "unit_ids": [str(u) for u in z["unit_ids"]],
        "scores": z["clip_scores"].astype(np.float64),   # [视频数, 48]
        "mask": z["mask"].astype(bool),                  # True=这个片段可用
        "labels": z["labels"].astype(int),
        "folds": z["folds"].astype(int),
    }


def unit_scores_from(scores, mask):
    """每条视频的分数 = 可用片段里的最大值（与训练、与正式成绩同一聚合方式）。"""
    out = np.full(scores.shape[0], np.nan)
    for i in range(scores.shape[0]):
        vals = scores[i][mask[i]]
        if vals.size:
            out[i] = vals.max()
    return out


def confusion(y, s, thr=0.5):
    pred = s >= thr
    y = y.astype(bool)
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    p, n = tp + fn, fp + tn
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / p if p else float("nan")
    fpr = fp / n if n else float("nan")
    spec = tn / n if n else float("nan")
    fnr = fn / p if p else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "P": p, "N": n,
            "Precision": prec, "Recall": rec, "FPR": fpr, "Specificity": spec,
            "FNR": fnr, "F1": f1, "pred_pos_rate": (tp + fp) / max(p + n, 1),
            "AUC": auc(y, s)}


def fmt_conf(c):
    return (f"TP {c['TP']} / FP {c['FP']} / FN {c['FN']} / TN {c['TN']}"
            f"（P {c['P']} / N {c['N']}）；Precision {c['Precision']:.3f}、"
            f"Recall {c['Recall']:.3f}、FPR {c['FPR']:.3f}、Specificity {c['Specificity']:.3f}、"
            f"FNR {c['FNR']:.3f}、F1 {c['F1']:.3f}、AUC {c['AUC']:.4f}、"
            f"pred_pos_rate {c['pred_pos_rate']:.3f}")


def fa(x):
    """AUC 之类的数，折内只有一类样本时是 nan，打印成 n/a。"""
    return "n/a" if x != x else f"{x:+.4f}"


def report_single(run_dir, clip_labels=None):
    run_dir = resolve_path(run_dir)
    if clip_labels is not None:
        clip_labels = resolve_path(clip_labels, must_exist=True)
    base = load_scores(run_dir / "clip-scores.npz")
    if base is None:
        raise SystemExit(f"{run_dir} 里没有 clip-scores.npz，先跑 clip_scores_ft.py（帧序 normal）")
    rev = load_scores(run_dir / "clip-scores-reverse.npz")
    frz = load_scores(run_dir / "clip-scores-freeze.npz")
    for name, other in (("reverse", rev), ("freeze", frz)):
        if other is None:
            continue
        if other["unit_ids"] != base["unit_ids"]:
            raise SystemExit(f"clip-scores-{name}.npz 与 clip-scores.npz 的视频顺序不一致")
        for key in ("labels", "folds"):
            if not np.array_equal(np.asarray(other[key]), np.asarray(base[key])):
                raise SystemExit(f"clip-scores-{name}.npz 与 clip-scores.npz 的 {key} 不一致"
                                 f"（同一套设定的三份分数必须是同一批视频、同一套折）")

    y, folds = base["labels"], base["folds"]
    thr = 0.5
    print(f"=== {run_dir.name}：{len(y)} 条视频（正 {int((y==1).sum())}/负 {int((y==0).sum())}）===")
    arms = [("原样", base)] + ([("倒序", rev)] if rev else []) + ([("冻结", frz)] if frz else [])
    u = {}
    for name, d in arms:
        us = unit_scores_from(d["scores"], d["mask"])
        u[name] = us
        aucs = [auc(y[folds == f], us[folds == f]) for f in sorted(set(folds))]
        c = confusion(y, us, thr)
        print(f"\n[{name}] 五折 AUC 均值 {np.mean(aucs):.4f}（±{np.std(aucs):.4f}）")
        print("  逐折 AUC：" + "  ".join(
            f"第{f}折 {a:.4f}" + ("（只报不作结论）" if f == FOLD4 else "")
            for f, a in zip(sorted(set(folds)), aucs)))
        print(f"  按 {thr} 报警：{fmt_conf(c)}"
              f"（这里的 AUC 是把五折分数混在一起算的参照值，"
              f"判定用的五折均值见上一行）")
        top = int(np.nanargmax(us)) if name == "原样" else -1
        if top >= 0:
            print(f"  分数最高的视频：{base['unit_ids'][top]}（{us[top]:.6f}）")

    # ---- 片段级变化 ----
    # 片段标签不必放在 run 目录里：EXP-009 的 run 目录就没有这份文件，标签的正本在
    # 数据目录（150 条池 = data/process/EXP-008-units-v2/clip-labels.npz）。用
    # --clip-labels 指过去即可，行序不一致会自动跳过并提示。
    cl_path = clip_labels if clip_labels is not None else (run_dir / "clip-labels.npz")
    clab = None
    if cl_path.is_file():
        z = np.load(cl_path, allow_pickle=True)
        if [str(x) for x in z["unit_ids"]] == base["unit_ids"]:
            clab = z["strict"].astype(int)          # 1 正 / 0 负 / -1 不可用
    mask = base["mask"]
    for name in [n for n, _ in arms if n != "原样"]:
        dlt = u[name] - u["原样"]
        ok = np.isfinite(dlt)
        absmax = float(np.abs(dlt[ok]).max()) if ok.any() else float("nan")
        same_call = np.array_equal(np.nan_to_num(u["原样"]) >= thr,
                                   np.nan_to_num(u[name]) >= thr)
        print(f"\n[{name}] 逐条视频分数差：最大 |差| {absmax:.4f}；"
              f"按 {thr} 的报警结果{'没有改变' if same_call else '改变了'}")
        # 先后顺序只在同一折内部比：不同折的视频由不同的折模型打分，跨折比先后没有意义。
        print("  同一折内部的先后顺序变化：", end="")
        tot_flip = tot_cross = 0
        per_fold = []
        for f in sorted(set(folds)):
            m = folds == f
            ua, ub, yy = u["原样"][m], u[name][m], y[m]
            da = np.sign(ua[:, None] - ua[None, :])
            db = np.sign(ub[:, None] - ub[None, :])
            iu = np.triu_indices(int(m.sum()), 1)
            flip = da[iu] != db[iu]
            cross = flip & (yy[iu[0]] != yy[iu[1]])
            tot_flip += int(flip.sum())
            tot_cross += int(cross.sum())
            if flip.any():
                per_fold.append(f"第{f}折 {int(flip.sum())} 对")
        print(f"共 {tot_flip} 对的先后变了（" + ("、".join(per_fold) or "无") + "）；"
              f"其中一条正一条负的 {tot_cross} 对——AUC 只由这种对决定")
        cs = base["scores"]
        d2 = None
        if name == "倒序" and rev:
            d2 = rev["scores"]
        elif name == "冻结" and frz:
            d2 = frz["scores"]
        if d2 is None:
            continue
        dd = d2 - cs
        avail = mask.copy()
        small = float((np.abs(dd[avail]) < 0.1).mean())
        print(f"  片段级：{100*small:.1f}% 的可用片段分数变化不到 0.1")
        if clab is not None:
            pos = avail & (clab == 1)
            neg = avail & (clab == 0)
            print(f"  按片段标签（strict 1/0）分：正片段 {int(pos.sum())} 个，"
                  f"平均变化 {dd[pos].mean():+.4f}；"
                  f"负片段 {int(neg.sum())} 个，平均变化 {dd[neg].mean():+.4f}；"
                  f"负片段里被推过 {thr} 的 {int(((cs[neg] < thr) & (d2[neg] >= thr)).sum())} 个，"
                  f"被拉回 {thr} 以下 {int(((cs[neg] >= thr) & (d2[neg] < thr)).sum())} 个")
            posvid = avail & np.repeat((y == 1)[:, None], cs.shape[1], axis=1)
            negvid = avail & np.repeat((y == 0)[:, None], cs.shape[1], axis=1)
            print(f"  按视频标签分（正视频的全部可用片段/负视频的全部可用片段）："
                  f"正片段 {int(posvid.sum())} 个，平均变化 {dd[posvid].mean():+.4f}；"
                  f"负片段 {int(negvid.sum())} 个，平均变化 {dd[negvid].mean():+.4f}；"
                  f"负片段里被推过 {thr} 的 {int(((cs[negvid] < thr) & (d2[negvid] >= thr)).sum())} 个，"
                  f"被拉回 {thr} 以下 {int(((cs[negvid] >= thr) & (d2[negvid] < thr)).sum())} 个")
    return 0


def reading(d, mask=None):
    """每条视频的读数：可用片段上 |倒序 − 正序| 的均值。

    mask 为 True 的片段才是「可用片段」（整块落在某段人工标注区间里）。不给 mask
    时对全部 48 个片段取均值，只在对照时使用；主判据必须带 mask，与卡片第五节
    写死的定义一致。
    """
    dd = np.abs(d)
    if mask is not None:
        dd = np.where(mask, dd, np.nan)
    return np.nanmean(dd, axis=1)


def report_paired(run_a, run_b):
    a0 = load_scores(run_a / "clip-scores.npz")
    b0 = load_scores(run_b / "clip-scores.npz")
    a1 = load_scores(run_a / "clip-scores-reverse.npz")
    b1 = load_scores(run_b / "clip-scores-reverse.npz")
    if a0 is None or b0 is None:
        raise SystemExit(f"缺 clip-scores.npz：{[n.name for n, v in ((run_a, a0), (run_b, b0)) if v is None]}")
    if a1 is None or b1 is None:
        raise SystemExit(f"缺 clip-scores-reverse.npz："
                         f"{[n.name for n, v in ((run_a, a1), (run_b, b1)) if v is None]}")
    for key in ("unit_ids", "labels", "folds"):
        if not np.array_equal(np.asarray(a0[key]), np.asarray(b0[key])) or \
                list(a0["unit_ids"]) != list(b0["unit_ids"]):
            raise SystemExit("两套设定的视频清单、标签或折归属不一致，不能逐视频配对")
    # 同一套设定内部：倒序那份必须与原样那份是同一批视频、同一套标签与折
    for name, ref, other in (("A", a0, a1), ("B", b0, b1)):
        if list(other["unit_ids"]) != list(ref["unit_ids"]):
            raise SystemExit(f"{name} 的 clip-scores-reverse.npz 与 clip-scores.npz 的视频顺序不一致")
        for key in ("labels", "folds"):
            if not np.array_equal(np.asarray(other[key]), np.asarray(ref[key])):
                raise SystemExit(f"{name} 的 clip-scores-reverse.npz 与 clip-scores.npz 的 {key} 不一致")
    y, folds = a0["labels"], a0["folds"]

    da = a1["scores"] - a0["scores"]
    db = b1["scores"] - b0["scores"]
    ra = reading(da, a0["mask"])
    rb = reading(db, b0["mask"])
    diff = rb - ra
    # 对照读数：不管掩码、对全部 48 个片段取均值。两套设定的掩码不同（原间隔可用
    # 5,930 个片段、新间隔 6,002 个），所以只作对照，不作判据。
    ra_all = reading(da)
    rb_all = reading(db)

    print(f"=== 配对比较：B（{run_b.name}）− A（{run_a.name}）===")
    print(f"{len(y)} 条视频（正 {int((y==1).sum())}/负 {int((y==0).sum())}），"
          f"读数 = 每条视频可用片段上 |倒序分数 − 正序分数| 的均值"
          f"（可用片段 = 整块落在人工标注区间里的片段，A 共 {int(a0['mask'].sum())} 个、"
          f"B 共 {int(b0['mask'].sum())} 个）")
    print(f"对照（不作判据）：不分掩码、对全部 48 个片段取均值，"
          f"A {np.nanmean(ra_all):.6f}、B {np.nanmean(rb_all):.6f}、"
          f"配对差 {np.nanmean(rb_all - ra_all):+.6f}")
    print(f"\nA 的读数均值 {np.nanmean(ra):.6f}；B 的读数均值 {np.nanmean(rb):.6f}")
    print(f"配对差均值 {np.nanmean(diff):+.6f}，中位 {np.nanmedian(diff):+.6f}")

    fold_means = {}
    for f in sorted(set(folds)):
        m = folds == f
        v = np.nanmean(diff[m])
        fold_means[int(f)] = float(v)
        n_neg = int((y[m] == 0).sum())
        print(f"  第{f}折（{int(m.sum())} 条，其中负样本 {n_neg} 条）：配对差均值 {v:+.6f}"
              + (f"（第 {FOLD4} 折按卡片规定只报不作结论）" if f == FOLD4 else ""))

    rng = np.random.default_rng(SEED)
    boots = np.empty(N_BOOT)
    idx_by_fold = [np.flatnonzero(folds == f) for f in sorted(set(folds))]
    for b in range(N_BOOT):
        pick = np.concatenate([rng.choice(ix, size=ix.size, replace=True) for ix in idx_by_fold])
        boots[b] = np.nanmean(diff[pick])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\n按视频有放回重抽 {N_BOOT} 次（各折内部按该折规模抽，种子 {SEED}）："
          f"95% 区间 [{lo:+.6f}, {hi:+.6f}]")

    pos_folds = sum(1 for f in NEED_FOLDS if fold_means.get(f, 0) > 0)
    c1 = np.nanmean(diff) > 0
    c2 = lo > 0
    c3 = pos_folds >= 4
    print(f"\n判定（三条全过才算「新间隔更依赖时间顺序」）：")
    print(f"  ① 配对差均值 > 0：{'过' if c1 else '不过'}（{np.nanmean(diff):+.6f}）")
    print(f"  ② 95% 区间不含 0：{'过' if c2 else '不过'}（[{lo:+.6f}, {hi:+.6f}]）")
    print(f"  ③ 第 1、2、3、5 折这四折的折内均值全为正：{'过' if c3 else '不过'}"
          f"（{pos_folds}/4 折为正）")
    print(f"  结论：{'三条全过，判「拉长间隔让模型更依赖时间顺序」' if (c1 and c2 and c3) else '未全部满足，判「本卡没有证据表明拉长间隔让模型更依赖时间顺序」'}")

    # 方向单独报：倒序后分数整体是升还是降，逐折 ΔAUC 的符号
    print("\n方向（只报，不作判据）：")
    for name, d, r in ((f"A（{run_a.name}）", a0, a1), (f"B（{run_b.name}）", b0, b1)):
        ua = unit_scores_from(d["scores"], d["mask"])
        ub = unit_scores_from(r["scores"], r["mask"])
        delta = np.nanmean(ub - ua)
        daucs = [auc(y[folds == f], ub[folds == f]) - auc(y[folds == f], ua[folds == f])
                 for f in sorted(set(folds))]
        print(f"  {name}：倒序后逐条分数平均变化 {delta:+.6f}；逐折 ΔAUC "
              + "  ".join(f"第{f}折 {fa(x)}" for f, x in zip(sorted(set(folds)), daucs)))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, default=None, help="单套设定自检的 run 目录")
    ap.add_argument("--run-a", type=Path, default=None, help="配对比较的 A（原间隔）")
    ap.add_argument("--run-b", type=Path, default=None, help="配对比较的 B（新间隔）")
    ap.add_argument("--clip-labels", dest="clip_labels", type=Path, default=None,
                    help="片段标签 npz（默认找 run 目录里的 clip-labels.npz）。"
                         "单套自检想看「正片段/负片段分别变了多少」时才需要，"
                         "散在数据目录时用这个参数指过去")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.run and (a.run_a or a.run_b):
        raise SystemExit("--run 与 --run-a/--run-b 只能选一种用法")
    if a.run_a and a.run_b:
        return report_paired(resolve_path(a.run_a), resolve_path(a.run_b))
    if a.run:
        return report_single(resolve_path(a.run),
                             resolve_path(a.clip_labels) if a.clip_labels else None)
    raise SystemExit("要么给 --run（单套自检），要么同时给 --run-a 与 --run-b（配对比较）")


if __name__ == "__main__":
    raise SystemExit(main())
