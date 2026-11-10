# -*- coding: utf-8 -*-
"""训练台：在浏览器里查看每条样本被判成什么、判错的那些错在哪，看图，并试算新视频。

对标 label_audit_server.py 的形态（http.server + 内嵌前端，不联网，只监听 127.0.0.1），
只读服务，不改任何数据。四种页签：

  【七套成绩对比】默认页签。把 EXP-008 跑过的 7 组设定并排成一张表，数字从各 run 的
    metrics.json 现读，并说明「为什么是 7 套而不是 1 套」。这一页与 --run 指定的那个
    run 无关，任何 run 下都能看。
  【样本与错误】列出全部样本（每条一行，带缩略图、真值、预测、分数），可只看判错的，
    点开某一条看它内部各部分的分数。
    时序线（train_temporal.py 的 run）：场次组 → 标注段 → 窗口 → 帧。
    EXP-008 线（train_mil.py 的 run）：场次组 → 整条视频 → 48 个等间隔采样位置 → 帧。
    强标签微调线（train_ft.py 的 run，EXP-009 / EXP-014）：同样到片段和帧，
    多一行「这块片段实际标注是什么」，因为训练时每个片段都有自己的答案。
  【图表】服务端 matplotlib 画 PNG，可直接下载进论文。图的诚实口径见「图表口径」节。
  【视频推理】输入一条视频，按 EXP-008 训练时的取材设定抽帧提特征打分（调
    experiments/tools/infer_video.py，不重写一份预处理）。

run 类型自动识别：目录下有 predictions.jsonl 走时序线；有 predictions.csv + mil_f*.pt
走 EXP-008 线；有 predictions.csv + ft_*.pt 走强标签微调线。三种契约各自有独立的
加载函数，前两种的逐折汇总共用 summarize_units。

用法（仓库根目录，venv）：
    .venv\\Scripts\\python.exe experiments\\tools\\training_workbench.py ^
        --run data\\results\\exp008\\mil-v2-150-strict [--port 8100]
    .venv\\Scripts\\python.exe experiments\\tools\\training_workbench.py ^
        --run data\\results\\temporal\\EXP-006-20260924-addrmap-scene-pred
    .venv\\Scripts\\python.exe experiments\\tools\\training_workbench.py ^
        --run data\\results\\exp008\\EXP-014-strong-v3-128-strict-ft-noaug

## 图表口径（2026-10-05 research-auditor 决策复核后定，不得擅自改）

- **不画「跨折均值 ROC」**：5 折的验证视频各不相同，跨折插值平均出的曲线不对应任何
  一个真实模型。
- **5 折分数混在一起画之前，先把分数换成同折百分位**：5 个折的模型各自把分数校准到
  自己那折的分布上，跨折原始分不可比（实测：151 条跨折混算 AUC 0.7743，逐折 AUC 均值
  0.9211，差 0.147 全来自尺度漂移）。混算的 ROC/直方图同时给未换算与换算后两条，
  并标注用的是哪一种。
- **逐折 ROC 画阶梯**（每折验证侧仅 29~32 条视频，是阶梯函数不是平滑曲线），标注
  每折正负视频数。
- **误差棒画折间总体标准差**（np.std, ddof=0，与 run-log S1 声明一致），标注
  「n=5，非置信区间」，并叠加 5 个逐折散点（否则 f4=0.7727 这样的离群折被均值掩盖）。
- **阈值滑条是 val 复用**：0.5 是卡片约定阈值，拖动所得最佳点是在同一验证集上挑的，
  含向上偏置。混淆矩阵旁必须显示 pred_pos_rate（比 F1 更早暴露「全部判成有泥石流」）。
- **单条视频的片段图 x 轴是「48 个等间隔采样片段序号」**，不是连续时间（linspace 采样 +
  换个位置重新抽帧，相邻序号在时间上不一定连续）；用针状图不连线；最高分片段不代表
  事件定位（弱标签，卡 D7 只复现到检测网络 #2，无定位能力）。
- **报告值口径**：EXP-008 的 AUC 是 5 折交叉验证下按条视频算的 AUC，验证折兼作选模集
  （选 best epoch）与报告集，含向上偏置（卡 M2 声明）。页面上任何数字都要与 run 名一起出现。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.paths import resolve_path  # noqa: E402
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "experiments/common/video"))
sys.path.insert(0, str(REPO / "experiments/tools"))

try:  # 包方式
    from experiments.common.temporal import adapters
except ImportError:
    from experiments.common.temporal import adapters

THUMB_W = 480  # 缩略图宽（与审计台一致，刷大图会卡页面的教训）

# 页头那句「每行成绩对应什么」按 run 类型换
KIND_NOTE = {"exp008": "EXP-008 实验：每行成绩对应一条视频",
             "ft": "强标签微调实验：每行成绩对应一条视频"}

# 取材正本根：2026-10-05 坏块事故重建后为 EXP-008-units-v2（旧根已作废）。
EXP008_UNITS_ROOT = REPO / "data/process/EXP-008-units-v2"
EXP008_FRAMES = EXP008_UNITS_ROOT / "frames"
EXP008_IID_FOLDS = REPO / "data/metadata/splits/EXP-008-units-iid-folds.json"
N_CLIP = 48
CLIP_LEN = 16


# ---------------------------------------------------------------- 指标计算
def _auc(y, s):
    """单类折上 AUC 无定义，返回 nan（与时序台口径一致）。"""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def fold_metrics(y, s, th=0.5):
    """折内指标。y = 真实标签（1 = 有泥石流），pred = 阈值 th 下的预测。

    混淆矩阵四个格子：TP 报对了 / FP 误报 / FN 漏报 / TN 正确排除。在它们之上算：
      - Recall = TP/(TP+FN)，实际有泥石流的视频里被报出来的比例。
      - Precision = TP/(TP+FP)，报出来的里面真的有的比例。
      - FPR = FP/(FP+TN)，实际没有泥石流的视频里被误报的比例。
      - FNR = FN/(FN+TP)，漏报比例，等于 1 − Recall。
      - Specificity = TN/(FP+TN)，等于 1 − FPR。
      - F1 = Precision 与 Recall 的调和平均。
      - Balanced accuracy = (Recall + Specificity)/2，类别不平衡时比 Accuracy 诚实。
      - MCC（Matthews correlation coefficient）：把四个格子一起考虑，不平衡时比 F1 稳。
      - pred_pos_rate：不是标准指标，是诊断量——模型把所有视频都判成有泥石流时
        它等于 1，立刻暴露。
    Accuracy 故意不算：150 条里 113 条是正的，把所有视频都判成有泥石流就有 75.3%，会骗人。
    """
    from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pred = (s >= th).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    rec = recall_score(y, pred, zero_division=0)
    return {
        "n": int(len(y)), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "auc": _auc(y, s),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(rec),
        "specificity": float(spec),
        "fpr": float(1 - spec),            # FPR
        "fnr": float(1 - rec),             # FNR
        "balanced_acc": float((rec + spec) / 2),
        "mcc": float(matthews_corrcoef(y, pred)) if len(set(y.tolist())) > 1 else float("nan"),
        "pred_pos_rate": float(pred.mean()) if len(pred) else float("nan"),
    }


def bootstrap_auc(units, n_boot=2000, seed=0, alpha=0.05):
    """按**场次组**重采样的 bootstrap 区间（不是按条视频重采样）。

    同一场次里的多条视频画面高度相似，不是独立样本；按条视频重采样会把样本量
    虚报成「池子里的视频总数」，区间假性变窄。按组重采样（组内整组一起进）才是
    诚实的口径。
    返回 (下界, 上界, 重采样次数)。
    """
    rng = np.random.default_rng(seed)
    groups = sorted({u["group_id"] for u in units})
    by = {g: [u for u in units if u["group_id"] == g] for g in groups}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(groups), size=len(groups), replace=True)
        us = [u for i in pick for u in by[groups[i]]]
        y = [u["label"] for u in us]
        s = [u["unit_score"] for u in us]
        a = _auc(y, s)
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return float("nan"), float("nan"), 0
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))), len(vals))


def bootstrap_fold_mean_auc(units, n_boot=2000, seed=0, alpha=0.05, min_folds=1):
    """按**场次组**重采样，给「5 折折间均值 AUC」算 95% 区间。

    为什么必须和 `bootstrap_auc` 分开：两者是两个不同的估计量。
      - 折间均值 = 5 折各自 AUC 的算术平均（mil-v2-150-strict：0.8699）
      - 跨折混算 = 把所有视频的分数放在一起算一次 AUC（同一 run：0.8146）
    混算值更低，是因为各折模型的分数尺度不同，混在一起算会被尺度差拉低。
    把混算区间的上下界贴在折间均值的点估计旁边，会出现「点估计落在自己区间之外」
    （2026-10-05 设计调研代理在旧池上发现：0.9524 不在 [0.739, 0.948] 内），答辩现场
    会被直接质疑。每个点估计必须配它自己那个估计量的区间。

    有放回地重抽样本时，抽的对象是场次组，与 `bootstrap_auc` 同理：同场次视频画面相似、
    不是独立样本。
    某次重采样里某折只剩单一类别时，该折 AUC 无定义，跳过、只对能算的折取均值。
    因此**每次重采样参与平均的折数并不都是 5**——调用方必须把折数分布一并披露，
    否则 `n_boot=2000` 会被读成「2000 次全是 5 折均值」。

    min_folds：参与平均所需的最少折数（默认 1，即只要有一折可算就算一次）。
    设成 3 是稳健性对照——实测（mil-strict-fix1，seed=0，2000 次）：把只用 1~2 折的
    43 次重采样剔掉后（余 1957 次）区间为 [0.8781, 0.9859]，与默认口径的
    [0.8735, 0.9861] 相差不到 0.006，说明拉宽区间的是那 2.2% 的 1~2 折重采样，
    结论对它不敏感。
    注意**不要**把门槛设到 5（要求每折都有两类）——那会把负样本富集的少数几个组反复
    抽进来，research-auditor 2026-10-05 实测接受样本里负样本数中位数从 38 升到 47
    （+27%），区间系统性偏窄且**有偏**。

    返回 dict：lo/hi/n_boot/fold_hist/median_folds。
      n_boot 是**有效重采样次数**（至少 min_folds 折可算的次数），不是「五折齐全的次数」。
    """
    rng = np.random.default_rng(seed)
    folds = sorted({int(u["fold"]) for u in units})
    groups = sorted({u["group_id"] for u in units})
    by = {g: [u for u in units if u["group_id"] == g] for g in groups}
    vals, hist = [], {}
    for _ in range(n_boot):
        pick = rng.choice(len(groups), size=len(groups), replace=True)
        us = [u for i in pick for u in by[groups[i]]]
        per = []
        for f in folds:
            fu = [u for u in us if int(u["fold"]) == f]
            a = _auc([u["label"] for u in fu], [u["unit_score"] for u in fu])
            if not np.isnan(a):
                per.append(a)
        if len(per) >= min_folds:
            vals.append(float(np.mean(per)))
            hist[len(per)] = hist.get(len(per), 0) + 1
    if not vals:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot": 0,
                "fold_hist": {}, "median_folds": float("nan")}
    return {
        "lo": float(np.percentile(vals, 100 * alpha / 2)),
        "hi": float(np.percentile(vals, 100 * (1 - alpha / 2))),
        "n_boot": len(vals),
        "fold_hist": dict(sorted(hist.items())),
        "median_folds": float(np.median([k for k, v in hist.items() for _ in range(v)])),
    }


def breakdown(units, key, th=0.5):
    """按某个字段（source / group_id / fold）分组算指标。

    整体 AUC 好看，可能藏着一个源很好、另一个源很差——合在一起看不出来。
    """
    out = {}
    for k in sorted({str(u[key]) for u in units}):
        us = [u for u in units if str(u[key]) == k]
        if len(us) < 2:
            continue
        out[k] = fold_metrics([u["label"] for u in us],
                              [u["unit_score"] for u in us], th)
    return out


def rank_normalize(scores, ref):
    """把 scores 按参考分布 ref 映射到 [0,1] 的分位（换成同折百分位）。

    跨折原始分数不可比（各折模型独立校准），跨折合并前必须先做这一步；这个变换保持
    折内排序不变，因此折内 AUC 逐位不变（research-auditor 2026-10-05 B4）。
    """
    ref = np.sort(np.asarray(ref, dtype=float))
    if len(ref) == 0:
        return np.full(len(scores), np.nan)
    return np.searchsorted(ref, np.asarray(scores, dtype=float), side="left") / len(ref)


# ---------------------------------------------------------------- 时序线加载
def load_temporal(run_dir: Path):
    """时序线（train_temporal.py 的 run）：窗口级，predictions.jsonl + metrics.json。

    两线与前端的契约要一致，但落盘结构差别很大，所以在加载层拍平：
      - metrics.json 的 per_fold 是「折 → 方法 → 层级(window/frame/seg) → 指标」三层嵌套，
        这里按方法抽 window 层，拍成 {方法: [逐折扁平行]}。
      - predictions.jsonl 的 kind 是 "pos"/"neg" 字符串，转成 0/1 的 label。
    """
    run_dir = resolve_path(run_dir, must_exist=True)
    preds = []
    for i, line in enumerate((run_dir / "predictions.jsonl")
                             .read_text(encoding="utf-8").splitlines()):
        if line.strip():
            p = json.loads(line)
            p["_i"] = i
            p["label"] = 1 if p.get("kind") == "pos" else 0
            p["fold"] = str(p.get("fold", "")).replace("fold", "")
            preds.append(p)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    methods = list(preds[0]["scores"]) if preds else []

    fold_metrics = {m: [] for m in methods}
    for pf in metrics.get("per_fold", []):
        meta = pf.get("_meta", {})
        n, n_pos = meta.get("n_val_win"), meta.get("pos_val")
        for m in methods:
            lv = (pf.get(m) or {}).get("window")
            if not lv:
                continue
            row = {"fold": str(pf.get("fold", "")).replace("fold", ""),
                   "n": n, "n_pos": n_pos,
                   "n_neg": (n - n_pos) if (n is not None and n_pos is not None) else None,
                   "auc": lv.get("auc"), "f1": lv.get("f1"),
                   "precision": lv.get("precision"), "recall": lv.get("recall"),
                   "pred_pos_rate": lv.get("pred_pos_rate"),
                   "best_epoch": (pf.get(m) or {}).get("best_epoch")}
            fold_metrics[m].append(row)

    return {
        "kind": "temporal", "run": run_dir.name,
        "methods": methods, "fold_metrics": fold_metrics,
        "bucket_dims": [{"key": "group_id", "label": "按组"},
                        {"key": "source", "label": "按源"},
                        {"key": "fold", "label": "按折"}],
        "items": preds,
        "note": ("时序线：一条样本是一个窗口（窗口级指标）。逐折指标取自 metrics.json 的 "
                 "window 层；方法可切换。"),
    }


# ---------------------------------------------------------------- 逐折与区间汇总
def summarize_units(units, metrics):
    """由每条视频的分数现算逐折指标、折内分数参照表与两个区间。

    EXP-008 线和强标签微调线共用这一段：两条线拿到「每条视频的分数」的方式不同
    （前者用折权重重新算，后者直接读训练时落盘的分数），但之后所有汇总完全一样。
    逐折 AUC 与 metrics.json 对不上就报错退出，不拿可疑数字起台。
    """
    per_fold, ref = [], {}
    for f in sorted({u["fold"] for u in units}, key=int):
        us = [u for u in units if u["fold"] == f]
        y = [u["label"] for u in us]
        s = [u["unit_score"] for u in us]
        m = fold_metrics(y, s)
        m["fold"] = f
        m["name"] = f"f{f}"
        pub = next((x for x in metrics.get("folds", []) if str(x.get("fold")) == f), None)
        if pub and abs(float(pub.get("auc", 0)) - m["auc"]) > 1e-3:
            sys.exit(f"[台] 折 {f} 现算 AUC {m['auc']:.4f} 与 metrics.json "
                     f"{pub['auc']} 不符，拒绝起台")
        if pub:
            m["best_epoch"] = pub.get("best_epoch")
        per_fold.append(m)
        ref[f] = sorted(float(x) for x in s)

    # 不确定度与分组拆分：单看一个均值读不出差别，得知道噪声有多大、藏在哪个源里
    # 两个区间分属两个不同的估计量，必须各自配套，不能混用（见两个函数的 docstring）
    plo, phi, pn = bootstrap_auc(units)                    # 跨折混算原始分
    fci = bootstrap_fold_mean_auc(units)                   # 5 折折间均值
    fci3 = bootstrap_fold_mean_auc(units, min_folds=3)     # 稳健性对照：剔掉 1~2 折的重采样
    from sklearn.metrics import average_precision_score
    y_all, raw_all, nrm_all, _ = _pooled_rank_norm(units)
    pooled = {
        "auc_raw": _auc(y_all, raw_all), "auc_ranknorm": _auc(y_all, nrm_all),
        "ap_raw": float(average_precision_score(y_all, raw_all)),
        "ap_ranknorm": float(average_precision_score(y_all, nrm_all)),
        "pos_rate": float(y_all.mean()),
    }
    ci = {"lo": fci["lo"], "hi": fci["hi"], "n_boot": fci["n_boot"],
          "by": "场次组", "estimator": "fold_mean",
          "fold_hist": fci["fold_hist"], "median_folds": fci["median_folds"],
          "robust_min3": {"lo": fci3["lo"], "hi": fci3["hi"],
                          "n_boot": fci3["n_boot"]},
          "note": ("5 折折间均值的区间；跨折混算口径的区间见 pooled_ci。"
                   "n_boot 是有效重采样次数（该次重采样至少 1 折可算），"
                   "不是五折齐全的次数——某折只剩单一类别时该折 AUC 无定义、跳过。"
                   "robust_min3 是剔掉只算得出 1~2 折的重采样后的稳健性对照。")}
    pooled_ci = {"lo": plo, "hi": phi, "n_boot": pn, "by": "场次组",
                 "estimator": "pooled_raw",
                 "note": "跨折混算原始分的区间，不可与折间均值的点估计并排显示"}
    return {"per_fold": per_fold, "fold_ref": ref, "pooled": pooled,
            "ci": ci, "pooled_ci": pooled_ci}


# ---------------------------------------------------------------- EXP-008 加载
def load_exp008(run_dir: Path, caliber: str = "auto", folds_json: str = ""):
    """读 train_mil.py 的 run：predictions.csv（按条视频）+ mil_f*.pt + 缓存特征。

    逐片段分数由 score_units.score_run 用折权重对缓存特征重算（秒级），并把整条视频的分数
    与 predictions.csv 对照——对不上说明折号或口径配错，直接报错而不是给可疑数字。
    """
    run_dir = resolve_path(run_dir, must_exist=True)
    from score_units import score_run, verify_against_predictions

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if caliber == "auto":
        caliber = metrics.get("clip_caliber") or "full"
    # 协议三（iid）的 run 必须用 iid 折正本；按 metrics 的 folds_source 判定
    fj = folds_json
    if not fj and "iid" in str(metrics.get("folds_source", "")):
        fj = str(EXP008_IID_FOLDS)
    # 特征缓存与片段标签路径优先取 metrics.json 记录的实际参数（2026-10-06 W2 整改后
    # 七套 run 均已回补）；未记录时才退回按目录名推的默认值。
    feats = (resolve_path(metrics["feats"], must_exist=True) if metrics.get("feats")
             else REPO / "data/results/exp008/unitfeats-v2-150/unit-features.npz")
    if caliber == "full":
        cl = None
    else:
        cl = (resolve_path(metrics["clip_labels"], must_exist=True) if metrics.get("clip_labels")
              else EXP008_UNITS_ROOT / "clip-labels.npz")
    units, meta = score_run(run_dir, feats, cl, caliber, fj or None)

    n_checked, worst = verify_against_predictions(run_dir, units)
    if n_checked and worst is not None and worst > 1e-5:
        sys.exit(f"[台] 逐片段重算与 {run_dir.name}/predictions.csv 最大绝对差 "
                 f"{worst:.2e}（应 <1e-5）——折号或掩码口径配错，拒绝用可疑数字起台")

    for i, u in enumerate(units):
        u["_i"] = i
        u["fold"] = str(u["fold"])
        u["kind"] = "pos" if u["label"] == 1 else "neg"
        u["scores"] = {"MIL": u["unit_score"]}
        u["n_win"] = len(u["clip_starts"]) or N_CLIP
        u["win_i"] = u["argmax_clip"] or 0
        u["frame_note"] = "16 帧是连续原生帧"
        fps = 0.0
        u["start_sec"] = (round(u["clip_starts"][u["argmax_clip"]] / 25.0, 1)
                          if u["argmax_clip"] is not None and u["clip_starts"] else None)

    summary = summarize_units(units, metrics)
    return {
        "kind": "exp008", "run": run_dir.name, "caliber": caliber,
        "frames_root": str(EXP008_FRAMES),
        "metrics": {k: metrics.get(k) for k in
                    ("mean_auc", "std_auc", "n_unit", "seed", "epochs", "folds_source")},
        "methods": ["MIL"],
        "fold_metrics": {"MIL": summary["per_fold"]},
        **summary,
        "breakdown": {"source": breakdown(units, "source"),
                      "group_id": breakdown(units, "group_id")},
        "bucket_dims": [{"key": "group_id", "label": "按组"},
                        {"key": "source", "label": "按源"},
                        {"key": "fold", "label": "按折"}],
        "items": units,
        "note": ("5 折交叉验证：5 个模型各在 4/5 的视频上训练，每条视频只被「没见过它」"
                 "的那个模型打分。不存在单独训练的「最终模型」。"),
    }


# ---------------------------------------------------------------- 强标签微调加载
# train_ft.py 的 run（EXP-009、EXP-014）：训练时每条视频 48 个片段的分数已经落盘在
# clip-scores.npz，不需要像 EXP-008 线那样拿折权重重算特征。帧目录不在 run 里，
# metrics.json 记的是 AutoDL 上的路径，这里按「有没有 v3 字样」映射回本机哪份帧。
FT_FRAMES_V3 = REPO / "data/process/EXP-008-units-v3-gap05/frames"
FT_FRAMES_V2 = REPO / "data/process/EXP-008-units-v2/frames"
FT_MANIFESTS = (REPO / "data/process/EXP-008-units-v3-gap05/unit-manifest.csv",
                REPO / "data/process/EXP-008-units-v2/unit-manifest.csv")
FT_METHOD = "片段级强标签"


def _localize(p) -> Path | None:
    """解析 metrics.json 记录的历史路径；空字段返回 None。"""
    return resolve_path(p) if p else None


def load_ft(run_dir: Path):
    """读 train_ft.py 的 run：clip-scores.npz（逐片段分数）+ predictions.csv + metrics.json。

    两道核对不通过就不起台：
      1. npz 里每条视频的分数与 predictions.csv 逐个对齐（最大绝对差应 < 1e-5）；
      2. 逐折 AUC 由分数现算后与 metrics.json 对照（由 summarize_units 做）。
    片段级答案（strict 标签）只用于在详情里对照，对不上就整列不显示，不影响前面两条。
    """
    run_dir = resolve_path(run_dir, must_exist=True)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    npz_path = run_dir / "clip-scores.npz"
    if not npz_path.is_file():
        sys.exit(f"[台] {run_dir.name} 没有 clip-scores.npz，强标签微调线缺分数文件，不起台")
    z = np.load(npz_path, allow_pickle=True)
    unit_ids = [str(x) for x in z["unit_ids"]]
    cs = z["clip_scores"].astype(float)
    mask = z["mask"].astype(bool)
    unit_score = z["unit_score"].astype(float)
    argmax_clip = z["argmax_clip"].astype(int)
    labels = z["labels"].astype(int)
    folds = z["folds"].astype(int)
    n_clip = cs.shape[1]

    preds = {r["unit_id"]: r for r in
             csv.DictReader((run_dir / "predictions.csv").open(encoding="utf-8"))}
    missing = [u for u in unit_ids if u not in preds]
    if missing:
        sys.exit(f"[台] clip-scores.npz 里有 {len(missing)} 条视频不在 predictions.csv，"
                 f"比如 {missing[0]}——两个文件不是同一次 run 的产物，不起台")
    # 容差与 clip_scores_ft.py 的回归门禁一致：本机 CPU 与训练时 GPU 的浮点差实测
    # 0.0023，取 0.01；线两侧的浮点误差不会改变任何一条视频的判定，所以 0.5 上不允许翻转。
    worst = max(abs(float(preds[u]["score"]) - unit_score[i]) for i, u in enumerate(unit_ids))
    flips = [u for i, u in enumerate(unit_ids)
             if (unit_score[i] >= 0.5) != (float(preds[u]["score"]) >= 0.5)]
    if flips:
        sys.exit(f"[台] clip-scores.npz 与 {run_dir.name}/predictions.csv 在 0.5 判定线上有 "
                 f"{len(flips)} 条视频翻转（如 {flips[0]}），不是同一次 run 的产物，不起台")
    if worst > 0.01:
        sys.exit(f"[台] clip-scores.npz 的整条视频分数与 {run_dir.name}/predictions.csv "
                 f"最大绝对差 {worst:.2e}（应 <1e-2，实机 CPU 与 GPU 的浮点差约 2e-3）——"
                 f"不是同一次 run 的产物，不起台")
    n_disagree = sum(int(preds[u]["label"]) != int(labels[i]) or
                     int(preds[u]["fold"]) != int(folds[i])
                     for i, u in enumerate(unit_ids))
    if n_disagree:
        sys.exit(f"[台] clip-scores.npz 与 predictions.csv 有 {n_disagree} 条视频的"
                 f"答案或折号不一致，不起台")

    # 片段级答案：训练时用的那份 strict 标签；顺序或掩码对不上就整列不显示
    cl_path = _localize(metrics.get("clip_labels"))
    clip_lab = None
    if cl_path and cl_path.is_file():
        zc = np.load(cl_path, allow_pickle=True)
        if [str(x) for x in zc["unit_ids"]] == unit_ids:
            strict = zc["strict"].astype(int)
            if bool(((strict == -1) == (~mask)).all()):
                clip_lab = strict
    if clip_lab is None:
        print(f"[台] 提示：{run_dir.name} 找不到能对上的片段级答案，"
              f"详情页不显示「这块片段实际是不是泥石流」这一行")

    # 场次组与数据源只在清单里；选包含全部视频的那一份
    man = None
    for path in FT_MANIFESTS:
        if not path.is_file():
            continue
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        by = {r["unit_id"]: r for r in rows}
        if all(u in by for u in unit_ids):
            man = by
            break
    if man is None:
        sys.exit(f"[台] 两份 unit-manifest.csv 都没有覆盖 {run_dir.name} 的全部视频，"
                 f"拿不到场次组，不起台")

    v3 = ("v3" in str(metrics.get("frames_root", ""))
          or "v3" in str(metrics.get("clip_labels", "")))
    frames = (_localize(metrics.get("frames_root"))
              or (FT_FRAMES_V3 if v3 else FT_FRAMES_V2))
    frame_note = ("16 帧每隔 0.5 秒取一帧，跨约 7.5 秒" if v3
                  else "16 帧是连续原生帧")
    units = []
    for i, uid in enumerate(unit_ids):
        r = man[uid]
        a = int(argmax_clip[i])
        starts = [int(x) for x in (r.get("clip_start_frames") or "").split(";") if x.strip()]
        fps = float(r.get("src_fps") or 25.0)
        units.append({
            "_i": i, "unit_id": uid, "video_id": r["video_id"],
            "source": r["source"], "group_id": r["group_id"],
            "label": int(labels[i]), "kind": "pos" if labels[i] == 1 else "neg",
            "fold": str(int(folds[i])), "unit_score": float(unit_score[i]),
            "argmax_clip": a, "win_i": a, "n_win": len(starts) or n_clip,
            "frame_note": frame_note,
            "scores": {FT_METHOD: float(unit_score[i])},
            "clip_scores": [None if not mask[i][k] else float(cs[i][k]) for k in range(n_clip)],
            "clip_labels": ([None if clip_lab[i][k] == -1 else int(clip_lab[i][k])
                             for k in range(n_clip)] if clip_lab is not None else None),
            "start_sec": round(starts[a] / fps, 1) if a < len(starts) else None,
        })

    summary = summarize_units(units, metrics)
    return {
        "kind": "ft", "run": run_dir.name, "caliber": metrics.get("caliber"),
        "frames_root": str(frames),
        "metrics": {k: metrics.get(k) for k in
                    ("mean_auc", "std_auc", "n_unit", "seed", "epochs", "augment")},
        "methods": [FT_METHOD],
        "fold_metrics": {FT_METHOD: summary["per_fold"]},
        **summary,
        "breakdown": {"source": breakdown(units, "source"),
                      "group_id": breakdown(units, "group_id")},
        "bucket_dims": [{"key": "group_id", "label": "按组"},
                        {"key": "source", "label": "按源"},
                        {"key": "fold", "label": "按折"}],
        "items": units,
        "note": ("5 折交叉验证：5 个模型各在 4/5 的视频上训练，每条视频只被「没见过它」"
                 "的那个模型打分。不存在单独训练的「最终模型」。"
                 "这套模型的训练用了逐片段答案：每个片段单独告诉模型"
                 "它的 16 帧是否完整落在人工标注的有泥石流区间内。"),
    }


def load_run(run_dir: Path, caliber="auto", folds_json=""):
    run_dir = resolve_path(run_dir, must_exist=True)
    if (run_dir / "predictions.jsonl").is_file():
        return load_temporal(run_dir)
    if (run_dir / "predictions.csv").is_file() and list(run_dir.glob("mil_f*.pt")):
        return load_exp008(run_dir, caliber, folds_json)
    if (run_dir / "predictions.csv").is_file() and list(run_dir.glob("ft_*.pt")):
        return load_ft(run_dir)
    sys.exit(f"{run_dir} 认不出 run 类型：既无 predictions.jsonl（时序线），"
             f"也无 predictions.csv + mil_f*.pt（EXP-008 线），"
             f"也无 predictions.csv + ft_*.pt（强标签微调线）")


# ---------------------------------------------------------------- 图表
CHART_FONT = "Microsoft YaHei"

# 图表色板。2026-10-05 设计调研后定：配色经 OKLab 色盲模拟（Machado 等 2009，
# severity 1.0）与 WCAG 对比度逐对校验，不是凭眼睛挑的。2026-10-08 起只保留这一套
# （叶海峰定：界面不要深色模式），原来的深色色板已删。
#   - 正类/负类用蓝/橙，不用红/绿：红绿对在色盲模拟下色差约 4（会塌成同色），
#     蓝橙对 24.7，是全色板里分离度最高的一对。
#   - 顺序色阶（混淆矩阵）单色相、明度单调，不用 jet/rainbow——彩虹色阶明度
#     不单调，人眼会把中间某段误读成峰值。
THEME = {
    "positive": "#2A78D6", "negative": "#EB6834", "neutral": "#71717A",
    "grid": "#E8E8EA", "axis": "#767680",
    "text": "#18181B", "text2": "#52525B",
    "cat": ["#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4"],
    "seq": ["#F2F6FD", "#86B6EF", "#3987E5", "#256ABF", "#0D366B"],
}


def _plt():
    """取 matplotlib，设好 rcParams。

    图表 PNG 用透明底：贴在卡片上不出现色块边框。
    文字与网格线用中性色 #767680（白底对比度 4.49:1，够 WCAG AA）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = THEME
    plt.rcParams.update({
        "font.sans-serif": [CHART_FONT, "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 130,
        "figure.facecolor": "none", "axes.facecolor": "none",
        "savefig.facecolor": "none", "savefig.transparent": True,
        "text.color": c["text"],
        "axes.labelcolor": c["text2"], "axes.labelsize": 9.5,
        "axes.edgecolor": c["axis"], "axes.linewidth": 1.0,
        "xtick.color": c["axis"], "ytick.color": c["axis"],
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.titlesize": 10.5,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "legend.handlelength": 1.6, "legend.handletextpad": 0.6,
        "legend.labelspacing": 0.4, "legend.borderaxespad": 0.8,
        "lines.linewidth": 1.6,
    })
    return plt


def _finish(ax, c, grid="y"):
    """统一收尾：只留左、下两条轴；只画水平网格线并压在数据下面。

    垂直网格线在时序图里会与数据线交叉成网，视觉噪声大；类别轴（柱状图）不需要网格线。
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(c["axis"])
    ax.spines["bottom"].set_color(c["axis"])
    ax.set_axisbelow(True)
    ax.grid(False)
    if grid:
        ax.grid(True, axis=grid, color=c["grid"], linewidth=1.0)


def _save(fig, out):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(out, format="png", transparent=True)
    plt.close(fig)


def _pooled_rank_norm(units):
    """跨折混算：先把分数换成同折百分位，再拼起来。返回 (y, s_raw, s_norm, folds)。"""
    by = {}
    for u in units:
        by.setdefault(u["fold"], []).append(u)
    y, raw, nrm, fo = [], [], [], []
    for f, us in by.items():
        ref = [x["unit_score"] for x in us]
        s = np.array([x["unit_score"] for x in us], dtype=float)
        for k, u in enumerate(us):
            y.append(u["label"]); raw.append(u["unit_score"])
            nrm.append(float(rank_normalize([s[k]], ref)[0])); fo.append(f)
    return np.array(y), np.array(raw), np.array(nrm), np.array(fo)


def chart_roc_perfold(units, out):
    plt = _plt()
    c = THEME
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    # 折配色绑定折的编号，不绑定排名：折 2 永远是第 2 个色，不管它这次成绩排第几。
    # 按成绩分配颜色的话，换个筛选条件颜色就整体重排，读者会看错。
    aucs = []
    for i, f in enumerate(sorted({u["fold"] for u in units}, key=int)):
        us = [u for u in units if u["fold"] == f]
        y = np.array([u["label"] for u in us])
        s = np.array([u["unit_score"] for u in us])
        if len(set(y.tolist())) < 2:
            continue
        aucs.append(_auc(y, s))
        fpr, tpr, _ = roc_curve(y, s)
        ax.step(fpr, tpr, where="post", lw=1.5, color=c["cat"][i % len(c["cat"])],
                label=f"折{f}　AUC {_auc(y, s):.3f}　(正{int((y==1).sum())}/负{int((y==0).sum())})")
    ax.plot([0, 1], [0, 1], color=c["axis"], ls=":", lw=1.0)
    ax.text(0.62, 0.55, "随机基线", fontsize=8, color=c["text2"], rotation=32)
    ax.set_xlabel("假正率 FPR"); ax.set_ylabel("真正率 TPR")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    # 标题里的区间从数据现算：写死数字换个 run 就对不上（fold4 曾低到 0.576）
    ax.set_title(f"逐折 ROC（各折 AUC {min(aucs):.3f} ~ {max(aucs):.3f}）\n"
                 "不画跨折均值曲线——各折验证的视频不同，均值曲线不对应任何真实模型",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    _finish(ax, c, grid=None)
    _save(fig, out)


def chart_roc_pooled(units, out):
    plt = _plt()
    c = THEME
    from sklearn.metrics import roc_curve
    y, raw, nrm, _ = _pooled_rank_norm(units)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for s, lab, col, ls in ((raw, f"未换算（AUC {_auc(y, raw):.3f}）", c["neutral"], "--"),
                            (nrm, f"换成同折百分位后（AUC {_auc(y, nrm):.3f}）", c["positive"], "-")):
        fpr, tpr, _ = roc_curve(y, s)
        ax.step(fpr, tpr, where="post", lw=1.8, ls=ls, color=col, label=lab)
    ax.plot([0, 1], [0, 1], color=c["axis"], ls=":", lw=1.0)
    ax.text(0.62, 0.55, "随机基线", fontsize=8, color=c["text2"], rotation=32)
    ax.set_xlabel("假正率 FPR"); ax.set_ylabel("真正率 TPR")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"5 折分数混算 ROC（{len(y)} 条视频）\n"
                 "未换算线偏低是各折模型打分松紧不同所致，不是模型变差", fontsize=10)
    ax.legend(loc="lower right", fontsize=8.5)
    _finish(ax, c, grid=None)
    _save(fig, out)


def chart_pr(units, out):
    plt = _plt()
    c = THEME
    from sklearn.metrics import precision_recall_curve, average_precision_score
    y, raw, nrm, _ = _pooled_rank_norm(units)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for s, lab, col, ls in ((raw, "未换算", c["neutral"], "--"),
                            (nrm, "换成同折百分位后", c["positive"], "-")):
        p, r, _ = precision_recall_curve(y, s)
        ax.step(r, p, where="post", lw=1.8, ls=ls, color=col,
                label=f"{lab}　AP {average_precision_score(y, s):.3f}")
    base = float(y.mean())
    ax.axhline(base, color=c["axis"], ls=":", lw=1.0)
    ax.text(0.02, base + 0.02, f"随机基线 {base:.3f}（正类占比）",
            fontsize=8, color=c["text2"])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.04)
    ax.set_title(f"PR 曲线（{int(y.sum())} 正 / {int((1 - y).sum())} 负，"
                 "类别不平衡时 AP 比 AUC 更诚实）", fontsize=10)
    ax.legend(loc="lower left", fontsize=8.5)
    _finish(ax, c)
    _save(fig, out)


def _text_on(rgba):
    """按单元格实际亮度选字色，保证数字在浅色阶上读得清。"""
    r, g, b = rgba[0], rgba[1], rgba[2]
    return "#18181B" if (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.55 else "#FFFFFF"


def chart_score_dist(units, out):
    plt = _plt()
    c = THEME
    y, raw, nrm, _ = _pooled_rank_norm(units)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, s, t in ((axes[0], raw, "未换算（各折模型打分松紧不同，重叠不能直接当可分性看）"),
                     (axes[1], nrm, "换成同折百分位后（跨折可比）")):
        ax.hist([s[y == 0], s[y == 1]], bins=20,
                label=[f"没有泥石流（{int((y == 0).sum())}）", f"有泥石流（{int((y == 1).sum())}）"],
                color=[c["negative"], c["positive"]], stacked=False)
        ax.set_title(t, fontsize=9.5); ax.set_xlabel("视频得分"); ax.set_ylabel("视频数")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8.5, loc="upper center")
        _finish(ax, c)
    fig.suptitle(f"视频得分分布（5 折分数混算，{len(y)} 条视频）", fontsize=11.5)
    _save(fig, out)


def chart_score_strip(units, out):
    """分数-标签 strip 图：所有点直接铺开，不受直方图 bin 宽度影响。"""
    plt = _plt()
    c = THEME
    y, raw, nrm, _ = _pooled_rank_norm(units)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    rng = np.random.default_rng(0)
    for ax, s, t in ((axes[0], raw, "未换算"), (axes[1], nrm, "换成同折百分位后")):
        for lab, col, yy in ((0, c["negative"], 0), (1, c["positive"], 1)):
            v = s[y == lab]
            ax.scatter(v, yy + rng.normal(0, 0.06, len(v)), s=14, alpha=0.7, color=col,
                       edgecolors="none")
        ax.axvline(0.5, color=c["axis"], ls="--", lw=1.0)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["没有泥石流", "有泥石流"])
        ax.set_xlim(0, 1); ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel("视频得分"); ax.set_title(t, fontsize=9.5)
        _finish(ax, c, grid="x")
    fig.suptitle("分数-标签散点（每点一条视频；虚线是约定阈值 0.5）", fontsize=11.5)
    _save(fig, out)


def chart_confusion(units, out, th=0.5):
    plt = _plt()
    c = THEME
    from matplotlib.colors import LinearSegmentedColormap
    y = np.array([u["label"] for u in units])
    s = np.array([u["unit_score"] for u in units])
    pred = (s >= th).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    m = np.array([[tn, fp], [fn, tp]])
    cmap = LinearSegmentedColormap.from_list("seq", c["seq"])
    ax.imshow(m, cmap=cmap, aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=16,
                    fontweight="600", color=_text_on(cmap(m[i, j] / max(m.max(), 1))))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["判成 N", "判成 P"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["实际 N", "实际 P"])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"混淆矩阵（阈值 {th:.2f}，5 折分数混算）\n"
                 f"pred_pos_rate {pred.mean():.3f}——比 F1 更早暴露「全部判成有泥石流」",
                 fontsize=10)
    _save(fig, out)


def chart_fold_auc(units, out):
    plt = _plt()
    c = THEME
    folds = sorted({u["fold"] for u in units}, key=int)
    aucs = []
    for f in folds:
        us = [u for u in units if u["fold"] == f]
        aucs.append(_auc([u["label"] for u in us], [u["unit_score"] for u in us]))
    aucs = np.array(aucs, dtype=float)
    mu, sd = float(np.nanmean(aucs)), float(np.nanstd(aucs))   # ddof=0，同 run-log S1
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    # 折间离散度画成横向色带。不用「每根柱子各带一根误差棒」——那是另一回事，
    # 会被读成「每折各自的置信区间」。色带压在柱子后面（zorder=0），
    # 否则灰底会盖住柱顶和数值标注，读数变糊。
    ax.axhspan(mu - sd, mu + sd, color=c["neutral"], alpha=0.13, zorder=0,
               label=f"折间标准差带 [{mu - sd:.3f}, {mu + sd:.3f}]")
    # 5 根柱子同色：5 折是同一方法的 5 次重复，不是 5 个可比较的对象
    ax.bar(range(len(folds)), aucs, color=c["positive"], width=0.55, zorder=2)
    ax.axhline(mu, color=c["negative"], ls="--", lw=1.4, zorder=3,
               label=f"折间均值 {mu:.4f}")
    # 数值写进柱子里（白字贴柱顶内侧），不写在柱外——柱外的位置正好被色带覆盖
    for i, a in enumerate(aucs):
        ax.text(i, a - 0.015, f"{a:.3f}", ha="center", va="top", fontsize=8.5,
                color="#FFFFFF", zorder=4)
    ax.set_xticks(range(len(folds))); ax.set_xticklabels([f"折{f}" for f in folds])
    ax.set_ylim(0.5, 1.03); ax.set_ylabel("按条视频算的 AUC（0.5 = 随机猜测）")
    ax.set_title(f"逐折 AUC：最低折 {float(np.nanmin(aucs)):.3f}、最高折 {float(np.nanmax(aucs)):.3f}\n"
                 "只看均值会掩盖离群折，所以逐折值一并画出", fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2, fontsize=8.5)
    _finish(ax, c)
    ax.text(0.0, -0.30, "色带为 5 折总体标准差（ddof=0），n=5，不是置信区间",
            transform=ax.transAxes, fontsize=8, color=c["text2"], va="top")
    _save(fig, out)


def chart_fold_table(units, out):
    plt = _plt()
    c = THEME
    folds = sorted({u["fold"] for u in units}, key=int)
    rows = []
    for f in folds:
        us = [u for u in units if u["fold"] == f]
        m = fold_metrics([u["label"] for u in us], [u["unit_score"] for u in us])
        rows.append([f"折{f}", str(m["n"]), str(m["n_pos"]), str(m["n_neg"]),
                     f"{m['auc']:.4f}", f"{m['f1']:.4f}", f"{m['pred_pos_rate']:.3f}"])
    fig, ax = plt.subplots(figsize=(7.2, 2.6)); ax.axis("off")
    t = ax.table(cellText=rows,
                 colLabels=["折", "视频", "P", "N", "AUC", "F1@0.5", "pred_pos_rate"],
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.5)
    for (r, _col), cell in t.get_celld().items():
        cell.set_edgecolor(c["grid"])
        cell.set_facecolor("none")
        cell.get_text().set_color(c["text"] if r else c["text2"])
    ax.set_title("折构成与逐折指标（验证折既选 best-epoch 又报成绩，含向上偏置）",
                 fontsize=10.5, pad=14)
    _save(fig, out)


def chart_unit(units, out, unit_id):
    plt = _plt()
    c = THEME
    u = next((x for x in units if x["unit_id"] == unit_id), None)
    if u is None:
        raise KeyError(unit_id)
    s = [np.nan if v is None else v for v in u["clip_scores"]]
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ok = [k for k, v in enumerate(s) if not np.isnan(v)]
    bad = [k for k, v in enumerate(s) if np.isnan(v)]
    ax.vlines(ok, 0, [s[k] for k in ok], color=c["positive"], lw=1.8)
    if bad:                       # 被掩码剔除的片段：贴着轴画一小段灰，不冒充数据
        ax.vlines(bad, 0, 0.03, color=c["neutral"], lw=1.8)
        ax.text(bad[0], 0.055, f"剔除 {len(bad)} 个片段（未参与打分）", fontsize=8, color=c["text2"])
    if u["argmax_clip"] is not None:
        ax.axvline(u["argmax_clip"], color=c["negative"], ls="--", lw=1.2,
                   label=f"最高分片段 #{u['argmax_clip']}")
    ax.axhline(0.5, color=c["axis"], ls=":", lw=1.0)
    ax.text(len(s) * 0.99, 0.52, "阈值 0.5", fontsize=8, color=c["text2"], ha="right")
    ax.set_xlabel("48 个等间隔采样片段序号（非连续时间）")
    ax.set_ylabel("片段分数"); ax.set_ylim(0, 1.02); ax.set_xlim(-0.5, len(s) - 0.5)
    ax.set_title(f"{unit_id}（真值 {'正' if u['label'] else '负'}，整条视频的分 "
                 f"{u['unit_score']:.4f}）——最高分片段不代表事件定位", fontsize=10)
    ax.legend(loc="upper right", fontsize=8.5)
    _finish(ax, c)
    _save(fig, out)


# 逐折/按源柱状图共用的四列指标。不放「Specificity + FPR」两列——两者互为补数
# （FPR = 1 − Specificity），并排画等于同一个数画两遍。
BAR_KEYS = [("precision", "Precision"), ("recall", "Recall"),
            ("f1", "F1"), ("fpr", "FPR")]


def _bar_panel(ax, c, labels, series, title, ylim=(0, 1.15), annot=None):
    """四指标分组柱状图。柱顶直接标数值——黄、青绿在白底上对比度低于 3:1，
    按无障碍要求必须配直接标注，不能只靠颜色读数。
    图例放到坐标区外面上方：柱子最高到 1.00，图例留在框内会压在柱顶的数值上。
    """
    x = np.arange(len(labels))
    w = 0.2
    for i, (k, lab) in enumerate(BAR_KEYS):
        v = [np.nan if s.get(k) is None else s[k] for s in series]
        b = ax.bar(x + (i - 1.5) * w, [0 if np.isnan(z) else z for z in v], w,
                   label=lab, color=c["cat"][i], zorder=2)
        for rect, z in zip(b, v):
            if np.isnan(z):
                ax.text(rect.get_x() + rect.get_width() / 2, 0.02, "无定义",
                        rotation=90, ha="center", va="bottom", fontsize=7,
                        color=c["text2"])
            else:
                ax.text(rect.get_x() + rect.get_width() / 2, z + 0.015, f"{z:.2f}",
                        ha="center", va="bottom", fontsize=7.5, color=c["text"])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(*ylim); ax.set_ylabel("比例")
    ax.set_title(title, fontsize=10, pad=30)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=4, fontsize=8.5)
    _finish(ax, c)
    if annot:
        ax.text(0.0, -0.22, annot, transform=ax.transAxes, fontsize=8,
                color=c["text2"], va="top", wrap=True)


def chart_fold_metrics(units, out):
    """逐折并排画四个阈值相关指标。

    只画一个 F1 会掩盖取舍：F1 高可能是把所有视频都判成有泥石流换来的。把 Precision 与 FPR
    摆在一起，才能看出这个工作点对负样本到底怎么样。
    """
    plt = _plt()
    c = THEME
    folds = sorted({u["fold"] for u in units}, key=int)
    ms = []
    for f in folds:
        us = [u for u in units if u["fold"] == f]
        ms.append(fold_metrics([u["label"] for u in us], [u["unit_score"] for u in us]))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    # 标题里的折号与条数从数据现算：写死「折 2 与折 5 的 FPR = 1.00」换个 run 就不成立
    hit = [(f, m) for f, m in zip(folds, ms) if m["fpr"] == 1.0 and m["n_neg"]]
    if hit:
        who = "、".join(f"折{f}" for f, _ in hit)
        line1 = (f"逐折指标（阈值 0.5）：{who} 的 FPR = 1.00——"
                 f"该折 {hit[0][1]['n_neg']} 条没有泥石流的视频全被判成有泥石流")
    else:
        w = max(zip(folds, ms), key=lambda t: (t[1]["fpr"] if np.isfinite(t[1]["fpr"]) else -1))
        line1 = f"逐折指标（阈值 0.5）：FPR 最高的是折{w[0]}（{w[1]['fpr']:.2f}）"
    _bar_panel(ax, c, [f"折{f}" for f in folds], ms,
               line1 + "\n只看 F1 会漏掉这种情况，因为把所有视频都判成有泥石流也能把 F1 撑高")
    _save(fig, out)


def chart_source(units, out):
    """按数据源分组画指标。整体好看可能藏着一个源很好、另一个很差。"""
    plt = _plt()
    c = THEME
    bd = breakdown(units, "source")
    if len(bd) < 2:
        raise KeyError("只有一个源，分组图无意义")
    names = list(bd)
    # 「哪个源一条正样本都没有」从数据现算，不写死源名
    empty = [n for n in names if not bd[n]["n_pos"]]
    line1 = "按数据源拆分（阈值 0.5）：" + (
        f"{'、'.join(empty)} 一条正样本都没有，AUC 无定义" if empty
        else "每个源正负样本都有")
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    _bar_panel(ax, c,
               [f"{n}\n（{bd[n]['n_pos']} 正 / {bd[n]['n_neg']} 负）" for n in names],
               [bd[n] for n in names],
               line1 + "\n整体分数被样本量最大的源主导，拆开才看得见各源真实水平")
    _save(fig, out)


CHARTS = {
    "roc_perfold": ("逐折 ROC（阶梯）", chart_roc_perfold),
    "roc_pooled": ("5 折分数混算 ROC（含百分位换算对照）", chart_roc_pooled),
    "pr": ("PR 曲线 + AP", chart_pr),
    "score_dist": ("视频得分分布", chart_score_dist),
    "score_strip": ("分数-标签散点", chart_score_strip),
    "confusion": ("混淆矩阵", chart_confusion),
    "fold_auc": ("逐折 AUC（含误差棒与散点）", chart_fold_auc),
    "fold_metrics": ("逐折指标（Precision / Recall / F1 / FPR）", chart_fold_metrics),
    "source": ("按数据源拆分", chart_source),
    "fold_table": ("折构成表", chart_fold_table),
}


def render_chart(name, units, th=0.5, unit_id=None):
    buf = io.BytesIO()
    if name == "unit":
        chart_unit(units, buf, unit_id)
    elif name == "confusion":
        chart_confusion(units, buf, th)
    elif name in CHARTS:
        CHARTS[name][1](units, buf)
    else:
        raise KeyError(name)
    return buf.getvalue()


# ---------------------------------------------------------------- 视频推理后台任务
class InferJob:
    """一次视频推理的后台任务。HTTP 线程只读它，推理线程只写它。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.state = "idle"      # idle | running | done | error
        self.frac = 0.0
        self.msg = ""
        self.result = None
        self.error = ""

    def progress(self, frac, msg):
        with self.lock:
            self.frac, self.msg = float(frac), str(msg)

    def snapshot(self):
        with self.lock:
            return {"state": self.state, "frac": self.frac, "msg": self.msg,
                    "result": self.result, "error": self.error}


JOB = InferJob()


# 台子认的 run 类型 -> 视频推理用哪条训练线（infer_video 的 --line）。
# exp008 线的 run 目录只有 mil_f*.pt（load_exp008 的识别条件就是这个），所以对 mil；
# ft 线是整网微调模型，对 ft；temporal 线不是「按条视频」的产物，不给推理入口。
INFER_LINE = {"exp008": "mil", "ft": "ft"}


def start_infer(video: str, run_dir: Path, out_dir: Path, line: str = "mil"):
    """在后台线程跑 infer_video.infer，进度写进 JOB。"""
    from infer_video import infer as _infer

    def worker():
        try:
            res = _infer(Path(video), Path(run_dir), out_dir, (),
                         None, progress=JOB.progress, kind=line)
            with JOB.lock:
                JOB.state, JOB.result, JOB.frac = "done", res, 1.0
        except BaseException as e:            # SystemExit 也要接住，否则线程静默死
            with JOB.lock:
                JOB.state, JOB.error = "error", str(e)

    with JOB.lock:
        JOB.reset()
        JOB.state = "running"
    threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------- 七套成绩对比
# 2026-10-08 新增。起因：叶海峰问「什么叫 7 套成绩，为什么会有 7 套」——展示台当时一次
# 只加载一个 run，回答不了这个问题。
# 这 7 个 run 是 150 条视频池上跑完全部有效设定后的成绩；runs/ 下其余目录是旧池的作废
# 成绩与冒烟测试，不列。表里每个数字都从各 run 的 metrics.json 现读或现算，不写死。
RUNSETS_DIR = REPO / "data/results/exp008"
# (run_id, 监督信号, 片段集, 划分协议代号, 用途)
RUNSET_SPECS = [
    ("mil-v2-150-strict", "弱标签 MIL", "strict", "一", "主对照一侧（论文核心数字）"),
    ("strong-v2-150-strict", "片段级强标签 BCE", "strict", "一",
     "主对照另一侧（论文核心数字）"),
    ("mil-v2-150-full", "弱标签 MIL", "full", "一", "片段集敏感性参考行"),
    ("strong-v2-150-loose", "片段级强标签 BCE", "loose", "一",
     "隐式负区口径敏感性参考行"),
    ("mil-v2-150-strict-iid", "弱标签 MIL", "strict", "三", "上限参照行"),
    ("strong-v2-150-strict-iid", "片段级强标签 BCE", "strict", "三", "上限参照行"),
    ("mil-v2-150-full-iid", "弱标签 MIL", "full", "三", "上限参照行"),
]
PROTOCOL_LABEL = {"一": "协议一（场次级）", "三": "协议三（同场次混分参照）"}
PROTOCOL_FOOTNOTE = "同场次混分参照，含场次内画面相似性，非泛化成绩"
RUNSETS_CORE = {"mil-v2-150-strict", "strong-v2-150-strict"}


def _finite(x):
    """能当数字用的就返回 float，否则 None（缺字段、null、NaN 一律算没有）。"""
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return None


def load_runsets():
    """读 7 个 run 的 metrics.json，现算每套的 5 折 AUC 均值与总体标准差。

    均值与标准差都从逐折 AUC 现算（ddof=0，与图表页误差棒同一口径），同时把
    metrics.json 里存的那两个数一并返回；两者对不上时前端会标出来，而不是悄悄挑一个用。
    逐折的正负视频数也一并求和，用来核对七套确实跑在同一批视频上。

    某个 run 目录不在（或没有 metrics.json）时不报错，把它列进 missing 交给页面显示——
    缺一套就少一行，比整页打不开好。
    """
    sets, missing, mismatch = [], [], []
    for run_id, signal, clips, proto, purpose in RUNSET_SPECS:
        path = RUNSETS_DIR / run_id / "metrics.json"
        if not path.is_file():
            missing.append(run_id)
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        folds = [{"fold": int(f["fold"]), "auc": _finite(f.get("auc")),
                  "n": f.get("n"), "n_pos": f.get("n_pos"), "n_neg": f.get("n_neg")}
                 for f in m.get("folds", [])]
        vals = [f["auc"] for f in folds if f["auc"] is not None]
        mean = float(np.mean(vals)) if vals else None
        std = float(np.std(vals, ddof=0)) if vals else None
        sm, ss = _finite(m.get("mean_auc")), _finite(m.get("std_auc"))
        if any(a is not None and b is not None and abs(a - b) > 5e-4
               for a, b in ((mean, sm), (std, ss))):
            mismatch.append(run_id)
        sets.append({
            "run_id": run_id, "signal": signal, "clips": clips,
            "protocol": PROTOCOL_LABEL[proto], "protocol_key": proto,
            "footnote": PROTOCOL_FOOTNOTE if proto == "三" else None,
            "purpose": purpose, "core": run_id in RUNSETS_CORE,
            "mean_auc": mean, "std_auc": std,
            "mean_auc_stored": sm, "std_auc_stored": ss,
            "folds": folds,
            "n_unit": m.get("n_unit"), "seed": m.get("seed"),
            "n_pos": sum(f["n_pos"] or 0 for f in folds),
            "n_neg": sum(f["n_neg"] or 0 for f in folds),
            "metrics_path": path.relative_to(REPO).as_posix(),
        })
    return {"sets": sets, "missing": missing, "mismatch": mismatch,
            "dir": RUNSETS_DIR.relative_to(REPO).as_posix(),
            "footnote": PROTOCOL_FOOTNOTE}


# ---------------------------------------------------------------- 前端
PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>训练台 __RUN__</title><style>
/* ============================================================ 设计令牌
   2026-10-05 重做，2026-10-08 起只保留浅色（叶海峰定：不要深色模式）。
   配色与字号经无障碍校验：正文 15px/24px，表格 14px/20px，最小可点区域 32px。 */
:root{
  color-scheme:light;
  --font-sans:system-ui,-apple-system,"Segoe UI","Microsoft YaHei UI",
    "Microsoft YaHei","PingFang SC","Hiragino Sans GB","Noto Sans CJK SC",sans-serif;
  --font-mono:"Cascadia Mono",Consolas,"SF Mono",Menlo,"DejaVu Sans Mono",monospace;
  --fs-display:30px;--lh-display:38px; --fs-h1:24px;--lh-h1:32px;
  --fs-h2:18px;--lh-h2:26px; --fs-h3:15px;--lh-h3:22px;
  --fs-body:15px;--lh-body:24px; --fs-table:14px;--lh-table:20px;
  --fs-caption:13px;--lh-caption:20px; --fs-label:12px;--lh-label:16px;
  --fw-regular:400;--fw-medium:500;--fw-semibold:600;
  --sp-1:4px;--sp-2:8px;--sp-3:12px;--sp-4:16px;--sp-5:20px;--sp-6:24px;
  --sp-8:32px;--sp-10:40px;--sp-12:48px;--sp-16:64px;
  --content-max:1440px;--gutter:24px;--card-pad:20px;--card-gap:16px;
  --radius-sm:3px;--radius-md:6px;--radius-lg:10px;--radius-full:999px;
  --bg-page:#F7F7F8;--bg-surface:#FFFFFF;--bg-raised:#FFFFFF;--bg-sunken:#F0F0F1;
  --bg-hover:rgba(24,24,27,.04);--bg-active:rgba(24,24,27,.07);--bg-selected:#EFF4FF;
  --text-primary:#18181B;--text-secondary:#52525B;--text-tertiary:#71717A;
  --text-disabled:#A1A1AA;--text-inverse:#FFFFFF;
  --border-subtle:#E4E4E7;--border-default:#D4D4D8;--border-strong:#8A8A93;
  --accent:#2563EB;--accent-hover:#1D4ED8;--accent-text:#1D4ED8;
  --accent-subtle:#EFF6FF;--accent-border:#BFDBFE;--accent-ring:rgba(37,99,235,.18);
  --success:#15803D;--success-subtle:#F0FDF4;--success-border:#BBF7D0;
  --warning:#B45309;--warning-subtle:#FFFBEB;--warning-border:#FDE68A;
  --danger:#B91C1C;--danger-subtle:#FEF2F2;--danger-border:#FECACA;
  --shadow-xs:0 1px 2px rgba(16,17,20,.05);
  --shadow-sm:0 1px 3px rgba(16,17,20,.07),0 1px 2px rgba(16,17,20,.05);
  --shadow-md:0 4px 12px rgba(16,17,20,.09),0 1px 3px rgba(16,17,20,.06);
  --shadow-lg:0 12px 32px rgba(16,17,20,.14),0 2px 8px rgba(16,17,20,.07);
  --table-head-h:40px;--table-row-h:36px;--table-cell-pad-x:12px;
  --table-zebra:rgba(24,24,27,.02);
  --viz-positive:#2A78D6;--viz-negative:#EB6834;--viz-baseline:#71717A;
  --viz-grid:#E8E8EA;--viz-axis:#767680;
}

/* ============================================================ 基础 */
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--bg-page);color:var(--text-primary);
  font:var(--fw-regular) var(--fs-body)/var(--lh-body) var(--font-sans);
  -webkit-font-smoothing:antialiased}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:var(--radius-sm)}
code{font-family:var(--font-mono);font-size:.92em;background:var(--bg-sunken);
  padding:1px 5px;border-radius:var(--radius-sm)}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}

/* ============================================================ 顶栏 */
.topbar{position:sticky;top:0;z-index:40;background:var(--bg-surface);
  border-bottom:1px solid var(--border-subtle);display:flex;align-items:center;
  gap:var(--sp-4);padding:0 var(--gutter);min-height:56px;flex-wrap:wrap}
.brand{display:flex;align-items:baseline;gap:var(--sp-3);min-width:0}
.brand h1{font-size:var(--fs-h3);line-height:var(--lh-h3);font-weight:var(--fw-semibold);
  margin:0;white-space:nowrap}
.brand .run{font-family:var(--font-mono);font-size:var(--fs-label);
  color:var(--text-tertiary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tabs{display:flex;gap:var(--sp-1);margin-left:auto;background:var(--bg-sunken);
  padding:3px;border-radius:var(--radius-md)}
.tab{appearance:none;border:0;background:transparent;color:var(--text-secondary);
  font:var(--fw-medium) var(--fs-table)/1 var(--font-sans);padding:0 var(--sp-4);
  height:30px;border-radius:var(--radius-sm);cursor:pointer;white-space:nowrap;
  display:inline-flex;align-items:center;gap:var(--sp-2)}
.tab:hover{color:var(--text-primary);background:var(--bg-hover)}
.tab[aria-selected="true"]{background:var(--bg-surface);color:var(--text-primary);
  box-shadow:var(--shadow-xs)}
/* 推理跑在服务端，切到别的页签它照样在跑。页签上加一个小圆点，
   用户切走后还知道有任务没结束。圆点只是视觉提示，文字靠 title 与
   旁边的 aria-label 传达，所以不用再给 ::before 补无障碍文本。 */
.tab.busy::before{content:"";width:7px;height:7px;border-radius:50%;
  background:var(--accent);flex:0 0 auto;animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
@media (prefers-reduced-motion:reduce){.tab.busy::before{animation:none}}
.iconbtn{appearance:none;border:1px solid var(--border-default);background:var(--bg-surface);
  color:var(--text-secondary);width:32px;height:32px;border-radius:var(--radius-md);
  cursor:pointer;display:inline-flex;align-items:center;justify-content:center;padding:0}
.iconbtn:hover{background:var(--bg-hover);color:var(--text-primary)}

/* ============================================================ 页签说明条 */
.kindnote{background:var(--bg-surface);border-bottom:1px solid var(--border-subtle);
  color:var(--text-secondary);font-size:var(--fs-caption);line-height:var(--lh-caption);
  padding:var(--sp-2) var(--gutter)}
.kindnote b{color:var(--text-primary);font-weight:var(--fw-semibold)}

/* ============================================================ 布局 */
main{padding:var(--sp-6) var(--gutter) var(--sp-16);max-width:var(--content-max);
  margin:0 auto;display:none}
main.on{display:block}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
  gap:var(--card-gap);margin-bottom:var(--sp-6)}
.card{background:var(--bg-surface);border:1px solid var(--border-subtle);
  border-radius:var(--radius-lg);padding:var(--card-pad);box-shadow:var(--shadow-xs)}
.card.warncard{border-color:var(--warning-border);background:var(--warning-subtle)}
.card h3{margin:0 0 var(--sp-1);font-size:var(--fs-label);line-height:var(--lh-label);
  font-weight:var(--fw-semibold);color:var(--text-tertiary);
  text-transform:none;letter-spacing:.02em}
.card .big{font-size:var(--fs-display);line-height:var(--lh-display);
  font-weight:var(--fw-semibold);font-variant-numeric:tabular-nums;
  letter-spacing:-.01em;margin:var(--sp-1) 0}
.card .big small{font-size:var(--fs-h3);font-weight:var(--fw-regular);
  color:var(--text-secondary);margin-left:var(--sp-2)}
.card .sub{font-size:var(--fs-caption);line-height:var(--lh-caption);
  color:var(--text-secondary)}
.card .sub + .sub{margin-top:var(--sp-2)}
.card .foot{margin-top:var(--sp-3);padding-top:var(--sp-3);
  border-top:1px solid var(--border-subtle);font-size:var(--fs-note);
  line-height:var(--lh-note);color:var(--text-tertiary)}
.cols{display:flex;gap:var(--card-gap);align-items:flex-start;flex-wrap:wrap}
.panel{background:var(--bg-surface);border:1px solid var(--border-subtle);
  border-radius:var(--radius-lg);padding:var(--card-pad);box-shadow:var(--shadow-xs)}
.panel>h2{margin:0 0 var(--sp-4);font-size:var(--fs-h3);line-height:var(--lh-h3);
  font-weight:var(--fw-semibold);display:flex;align-items:center;gap:var(--sp-2)}
.panel>h2 .hint{font-size:var(--fs-note);font-weight:var(--fw-regular);
  color:var(--text-tertiary)}

/* ============================================================ 表格 */
table{border-collapse:collapse;font-size:var(--fs-table);line-height:var(--lh-table);
  width:100%;font-variant-numeric:tabular-nums}
th,td{padding:0 var(--table-cell-pad-x);height:var(--table-row-h);
  border-bottom:1px solid var(--border-subtle);text-align:right;white-space:nowrap}
th{color:var(--text-secondary);font-weight:var(--fw-medium);
  font-size:var(--fs-label);height:var(--table-head-h);
  position:sticky;top:0;background:var(--bg-surface);z-index:1}
th.sortable{cursor:pointer;user-select:none}
th.sortable:hover{color:var(--text-primary);background:var(--bg-hover)}
/* 表头排序做成真 <button>：th 本身不可聚焦，键盘用户按不到。
   button 铺满整个表头格，用 text-align 对齐而不是 flex——flex 会把箭头
   变成一个独立的盒子，跟标签错开到两行去。 */
th.sortable{padding:0}
th.sortable .thbtn{appearance:none;border:0;background:none;font:inherit;color:inherit;
  width:100%;height:var(--table-head-h);padding:0 var(--table-cell-pad-x);
  cursor:pointer;text-align:right}
th.sortable .thbtn.l{text-align:left}
th.sortable .thbtn:focus-visible{outline:2px solid var(--accent);
  outline-offset:-2px;border-radius:var(--radius-sm)}
th .arrow{opacity:.35;margin-left:3px}
th[aria-sort] .arrow{opacity:1;color:var(--accent)}
td.l,th.l{text-align:left}
tbody tr:hover{background:var(--bg-hover)}
tbody tr.sel{background:var(--bg-selected)}
/* 主对照那两行是论文的核心数字，给一层浅底把它们从参考行里分出来。底色不是唯一线索：
   用途列里也写着「论文核心数字」。这两条必须写在 :hover 之后，否则同权重下会被盖掉。 */
tbody tr.row-core{background:var(--accent-subtle)}
tbody tr.row-core:hover{background:var(--bg-selected)}
.tw{overflow:auto;border:1px solid var(--border-subtle);
  border-radius:var(--radius-md)}
.tw:focus-visible,.list:focus-visible{outline:2px solid var(--accent);
  outline-offset:-2px}
.tw table{border:0}
.tw th,.tw td:last-child{border-right:0}
caption{caption-side:top;text-align:left;color:var(--text-tertiary);
  font-size:var(--fs-note);line-height:var(--lh-note);padding:0 0 var(--sp-2)}
/* 组名可能很长（d|鸡冠山上游|2025-07-19），窄栏里必须截断，否则整表横向滚动。
   完整名字放 title，鼠标停上去能看全。 */
td.trunc{max-width:180px;overflow:hidden;text-overflow:ellipsis}

/* ============================================================ 徽标与标签 */
.badge{display:inline-flex;align-items:center;gap:3px;padding:1px var(--sp-2);
  border-radius:var(--radius-sm);font-size:var(--fs-label);line-height:var(--lh-label);
  font-weight:var(--fw-medium);border:1px solid transparent;white-space:nowrap}
/* 判对/判错不只靠颜色：同时给 ✓ / ✗ 字形与文字，投影仪偏色或红绿色觉障碍下仍可读 */
.badge.pos{background:var(--accent-subtle);color:var(--accent-text);
  border-color:var(--accent-border)}
.badge.neg{background:var(--bg-sunken);color:var(--text-secondary);
  border-color:var(--border-default)}
.badge.ok{background:var(--success-subtle);color:var(--success);
  border-color:var(--success-border)}
.badge.bad{background:var(--danger-subtle);color:var(--danger);
  border-color:var(--danger-border)}
.badge.flag{background:var(--warning-subtle);color:var(--warning);
  border-color:var(--warning-border)}

/* ============================================================ 表单 */
label{font-size:var(--fs-caption);color:var(--text-secondary)}
input[type=text],input[type=search],select{
  background:var(--bg-surface);border:1px solid var(--border-strong);
  color:var(--text-primary);padding:0 var(--sp-3);height:32px;
  border-radius:var(--radius-md);font-family:inherit;font-size:var(--fs-table)}
input[type=text]:focus,input[type=search]:focus,select:focus{
  border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-ring);outline:none}
input[type=range]{accent-color:var(--accent);vertical-align:middle;width:150px}
input[type=checkbox]{accent-color:var(--accent);width:15px;height:15px;
  vertical-align:-2px}
button{appearance:none;background:var(--accent);color:#fff;border:1px solid transparent;
  padding:0 var(--sp-4);height:32px;border-radius:var(--radius-md);cursor:pointer;
  font-family:inherit;font-size:var(--fs-table);font-weight:var(--fw-medium)}
button:hover{background:var(--accent-hover)}
button:disabled{background:var(--bg-sunken);color:var(--text-disabled);
  cursor:not-allowed}
button.ghost{background:transparent;color:var(--text-secondary);
  border-color:var(--border-default)}
button.ghost:hover{background:var(--bg-hover);color:var(--text-primary)}
.toolbar{display:flex;gap:var(--sp-4);align-items:center;flex-wrap:wrap;
  margin-bottom:var(--sp-4)}
.field{display:inline-flex;align-items:center;gap:var(--sp-2)}

/* ============================================================ 列表（视频） */
.list{max-height:64vh;overflow:auto;border:1px solid var(--border-subtle);
  border-radius:var(--radius-md)}
.unit{display:flex;gap:var(--sp-3);align-items:center;padding:var(--sp-2) var(--sp-3);
  border-bottom:1px solid var(--border-subtle);cursor:pointer}
.unit:last-child{border-bottom:0}
.unit:hover{background:var(--bg-hover)}
.unit.sel{background:var(--bg-selected)}
.unit img{width:88px;height:50px;object-fit:cover;background:var(--bg-sunken);
  border-radius:var(--radius-sm);flex:none}
.unit .meta{flex:1;min-width:0}
.unit .uid{font-weight:var(--fw-medium);display:flex;align-items:center;
  gap:var(--sp-2);flex-wrap:wrap}
.unit .note{color:var(--text-tertiary);font-size:var(--fs-note);line-height:var(--lh-note);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.unit .sc{width:88px;flex:none;text-align:right;font-variant-numeric:tabular-nums;
  font-size:var(--fs-caption)}
.bar{height:6px;background:var(--bg-sunken);border-radius:var(--radius-full);
  overflow:hidden}
.bar>i{display:block;height:100%;border-radius:var(--radius-full)}
.bar>i.p{background:var(--viz-positive)}
.bar>i.n{background:var(--viz-negative)}
.bar>i.miss{background:var(--viz-baseline)}

/* ============================================================ 提示条 */
.warn,.err,.info{padding:var(--sp-3) var(--sp-4);border-radius:var(--radius-md);
  margin:var(--sp-3) 0;font-size:var(--fs-caption);line-height:var(--lh-caption);
  border:1px solid}
.warn{background:var(--warning-subtle);border-color:var(--warning-border);
  color:var(--text-primary)}
.err{background:var(--danger-subtle);border-color:var(--danger-border);
  color:var(--text-primary)}
.info{background:var(--accent-subtle);border-color:var(--accent-border);
  color:var(--text-primary)}
.note{color:var(--text-tertiary);font-size:var(--fs-note);line-height:var(--lh-note)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:var(--sp-2) var(--sp-4);
  font-size:var(--fs-caption);line-height:var(--lh-caption);align-items:baseline}
.kv b{color:var(--text-tertiary);font-weight:var(--fw-regular);white-space:nowrap}
.kv .v{font-variant-numeric:tabular-nums}
.empty{padding:var(--sp-10) var(--sp-4);text-align:center;color:var(--text-tertiary);
  font-size:var(--fs-caption)}
.empty .big{font-size:var(--fs-h2);color:var(--text-secondary);margin-bottom:var(--sp-2)}
.skeleton{background:linear-gradient(90deg,var(--bg-sunken) 25%,
  var(--bg-hover) 37%,var(--bg-sunken) 63%);background-size:400% 100%;
  animation:sk 1.4s ease infinite;border-radius:var(--radius-sm);height:14px}
@keyframes sk{0%{background-position:100% 50%}100%{background-position:0 50%}}
@media (prefers-reduced-motion:reduce){.skeleton{animation:none}}

/* ============================================================ 针状图与帧条 */
.needle{width:100%;height:150px;display:block}
.frames{display:flex;gap:var(--sp-1);overflow-x:auto;padding:var(--sp-2) 0}
.frames img{height:72px;border-radius:var(--radius-sm);cursor:zoom-in;
  background:var(--bg-sunken)}
.frames img:hover{outline:2px solid var(--accent);outline-offset:1px}

/* ============================================================ 放大查看器 */
dialog#viewer{border:0;padding:0;background:transparent;max-width:100vw;max-height:100vh;
  width:100vw;height:100vh;overflow:hidden}
dialog#viewer::backdrop{background:rgba(9,9,11,.86)}

/* 「这一页怎么读」弹窗。用普通主题色而不是查看器那套写死的深色——它落在常规页面上。 */
dialog#help{border:1px solid var(--border-default);border-radius:var(--radius-lg);
  background:var(--bg-surface);color:var(--text-primary);padding:0;
  max-width:min(560px,92vw);max-height:86vh;overflow:auto}
dialog#help::backdrop{background:rgba(9,9,11,.55)}
.hwrap{padding:var(--sp-5)}
.hbar{display:flex;align-items:center;gap:var(--sp-4);
  font-size:var(--fs-h2);font-weight:var(--fw-semibold);margin-bottom:var(--sp-3)}
.hbar .sp{margin-left:auto}
table.keys{width:100%;font-size:var(--fs-caption);line-height:var(--lh-caption)}
table.keys th{text-align:left;vertical-align:top;white-space:nowrap;width:1%;
  color:var(--text-primary);font-weight:var(--fw-medium);
  font-family:var(--font-mono);font-size:var(--fs-label);
  padding:var(--sp-1) var(--sp-4) var(--sp-1) 0;height:auto;position:static;
  background:none;border-bottom:0}
table.keys td{text-align:left;color:var(--text-secondary);height:auto;
  padding:var(--sp-1) 0;border-bottom:0;white-space:normal}
/* 分组标题行是唯一的跨两列 th。这里用属性选择器而不是 tbody + tbody：
   第一组也有标题行，兄弟选择器会把它漏掉。 */
table.keys th[colspan]{width:auto;color:var(--text-tertiary);
  padding-top:var(--sp-4);font-family:var(--font-sans);font-weight:var(--fw-semibold);
  font-size:var(--fs-label);letter-spacing:.02em}
table.keys tbody:first-child th[colspan]{padding-top:0}
.vwrap{display:flex;flex-direction:column;height:100%;padding:var(--sp-4);
  gap:var(--sp-3)}
.vbar{display:flex;align-items:center;gap:var(--sp-4);color:#E4E4E7;
  font-size:var(--fs-caption);flex:none}
.vbar .sp{margin-left:auto;display:flex;gap:var(--sp-2);align-items:center}
.vbar button{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);
  color:#fff}
.vbar button:hover{background:rgba(255,255,255,.22)}
.vstage{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;
  overflow:hidden;position:relative}
.vstage img{max-width:100%;max-height:100%;transform-origin:center center;
  user-select:none;-webkit-user-drag:none;cursor:zoom-in}
/* 帧图是 256×256 的小图，按原始尺寸摆在 1440 宽的屏幕上只有巴掌大，等于没放大。
   帧模式下让图铺满可用区域（object-fit:contain 保持比例、四周留边），这才是
   「放大查看」。图表模式相反——图本身比屏幕大，用滚轮缩放看细节。 */
.vstage.page img{width:100%;height:100%;object-fit:contain;cursor:default}
.vstage.grabbing img{cursor:grabbing}
.vstage.zoomed img{cursor:grab}
.vstrip{display:flex;gap:var(--sp-2);overflow-x:auto;flex:none;
  padding-bottom:var(--sp-1)}
.vstrip button{padding:0;border:2px solid transparent;background:transparent;
  height:auto;border-radius:var(--radius-sm);flex:none;overflow:hidden}
.vstrip button[aria-current="true"]{border-color:var(--accent)}
.vstrip img{display:block;height:52px;border-radius:var(--radius-sm)}
.vhint{color:rgba(228,228,231,.7);font-size:var(--fs-note);text-align:center;flex:none}

/* ============================================================ 图表页 */
.chartlist{display:flex;flex-direction:column;gap:var(--sp-1)}
.chartlist button{justify-content:flex-start;text-align:left;background:transparent;
  color:var(--text-secondary);border:0;font-weight:var(--fw-regular);width:100%}
.chartlist button:hover{background:var(--bg-hover);color:var(--text-primary)}
.chartlist button[aria-current="true"]{background:var(--accent-subtle);
  color:var(--accent-text);font-weight:var(--fw-medium)}
.chartlist .grp{font-size:var(--fs-label);color:var(--text-tertiary);
  padding:var(--sp-3) var(--sp-3) var(--sp-1)}
.chartbox{background:var(--bg-surface);border:1px solid var(--border-subtle);
  border-radius:var(--radius-md);padding:var(--sp-4);display:flex;
  align-items:center;justify-content:center;min-height:380px}
.chartbox img{max-width:100%;cursor:zoom-in}

/* ============================================================ 进度 */
.progress{height:6px;background:var(--bg-sunken);border-radius:var(--radius-full);
  overflow:hidden;margin:var(--sp-3) 0}
.progress>i{display:block;height:100%;background:var(--accent);width:0;
  transition:width .25s ease}

</style></head><body>

<header class="topbar">
  <div class="brand">
    <h1>训练台</h1>
    <span class="run">__RUN__</span>
  </div>
  <nav class="tabs" role="tablist" aria-label="页面切换">
    <button class="tab" role="tab" id="tab-sets" aria-controls="m-sets"
      aria-selected="true" data-t="sets">七套成绩对比</button>
    <button class="tab" role="tab" id="tab-drill" aria-controls="m-drill"
      aria-selected="false" data-t="drill">样本与错误</button>
    <button class="tab" role="tab" id="tab-chart" aria-controls="m-chart"
      aria-selected="false" data-t="chart">图表</button>
    <button class="tab" role="tab" id="tab-infer" aria-controls="m-infer"
      aria-selected="false" data-t="infer">视频推理</button>
  </nav>
  <button class="iconbtn" id="helpbtn" title="这一页怎么读"
    aria-label="这一页怎么读">?</button>
</header>
<div class="kindnote">__KINDNOTE__</div>

<main class="on" id="m-sets" role="tabpanel" aria-labelledby="tab-sets">
  <div class="panel" style="margin-bottom:var(--card-gap)">
    <h2>七套成绩 <span class="hint">数字从各 run 的 metrics.json 现读</span></h2>
    <p class="note">
      表里每一行是一组设定。成绩是 150 条视频上按 5 折交叉验证算出的 AUC：全部视频分成
      5 份，轮流拿 4 份训练、剩 1 份考试，得到 5 个模型，每条视频的得分来自没见过它的
      那个模型。「5 折 AUC 均值 ± 标准差」是把 5 折的 AUC 取平均，± 后面是 5 折之间的
      总体标准差，数字越大说明折与折之间越不稳。
    </p>
    <p class="note" style="margin-top:var(--sp-2)">
      表头里的<b>片段</b>指从一条视频里等间隔取的一个位置上的连续 16 帧，整条视频一共取
      48 个位置。这 48 个位置是隔着几十秒铺满整条视频的，不是把视频切成 48 段；落在剪辑
      区间（人工标为非事件的时段）里的位置不参与打分，所以一条视频实际参与打分的片段数
      可能少于 48，正样本里常见的是 47 个，最少的有 19 个。
      <b>场次组</b>指同一天、同一个机位拍的那一批视频。<b>监督信号</b>是喂给模型的标签：
      弱标签 MIL 只给整条视频一个标签、不指出是哪几秒，片段级强标签 BCE 逐个片段给标签。
      片段集有 strict / full / loose 三档、划分协议有协议一 / 协议三两种，逐条说明见下一节。
      其中 loose 多算作负样本的那部分时段叫<b>隐式负区</b>，指视频里没被标注成事件、
      也没被剔除的时段——这些时段按没有泥石流算。
    </p>
    <div class="tw" id="setstab" style="margin-top:var(--sp-3)"></div>
    <div class="note" id="setsnote" style="margin-top:var(--sp-3)"></div>
  </div>
  <div class="panel" style="margin-bottom:var(--card-gap)">
    <h2>为什么是 7 套而不是 1 套</h2>
    <p>这 7 套是 7 组不同的设定，不是同一件事重复跑了 7 次。用的视频是同一批 150 条，
      每组有三处不一样：</p>
    <ul>
      <li><b>喂给模型的标签不一样。</b>弱标签 MIL 只告诉模型「这条视频里有泥石流」，
        不指出是哪几秒；片段级强标签 BCE 逐个片段告诉模型这一段有还是没有。</li>
      <li><b>参与打分的片段不一样。</b>strict 只用完整落在一段标注区间里的片段（占全部
        片段的 96.9%）；loose 在 strict 之外，把完整落在 2 秒以上未标注空隙里的片段也算成
        没有泥石流（占 98.8%）；full 不筛，48 个位置全用。</li>
      <li><b>分折方式不一样。</b>协议一把场次组整组放进同一侧，考试用的场次组模型从没见过，
        回答「下一次泥石流能不能认出来」；协议三只按标签随机分 5 折，同一天同一机位的视频
        可能一半在训练、一半在考试，两边画面几乎一样，成绩必然偏高，只作上限参照，
        不是泛化成绩。</li>
    </ul>
    <p>同一组设定只跑一次：随机种子固定为 7，重跑的结果逐位一致。</p>
    <p>强标签这一路没有 full 这一档，是因为强标签要求 48 个片段每一个都有明确标签，
      而跨人工审计划出的标注区间边界的那些片段（占 3.1%）标不了，只能剔掉，所以强标签
      只有 strict 和 loose 两档。于是：弱标签 2 档片段集 × 2 种划分协议 = 4 套；强标签
      只有 strict 一档 × 2 种划分协议 = 2 套；两路合计 6 套。再加一套 strong-v2-150-loose
      做隐式负区判定口径的敏感性检查，一共 7 套。</p>
  </div>
  <div class="panel">
    <h2>论文里能写什么</h2>
    <ol>
      <li><b>能写（唯一硬结论）：</b>按一条视频算的 AUC 显著高于随机——这里的「显著」指
        实测值远在随机波动之外。依据：把标签打乱重算 2 万次，随机的 97.5% 分位是 0.608，
        实测 0.87~0.98 远在这个范围之外。</li>
      <li><b>能写，但要带限制：</b>误报是漏报的 3 倍，而且成整组出现（集中在与真事件同
        机位同日的硬负组）；这个偏向对「识别已出现、通知下游」的用法是保守方向，漏报少。</li>
      <li><b>能写，但必须一并写出最硬的限制条件：</b>37 条负样本视频集中在 10 个场次组，
        31 个场次组里 24 个只有单一类别，所以指标只能在「整体」与「按数据源」两级报告，
        拆不到场次级。</li>
    </ol>
    <p style="margin-top:var(--sp-4)"><b>不能写：</b></p>
    <ul>
      <li>任何「A 比 B 好 0.0X」的句子。负样本只有 37 条，一条负样本在排序里挪个位置，
        那一折 AUC 就能动 0.10~0.14，而表里各行之间的差只有 0.003~0.048。</li>
      <li>「强标签优于弱标签」——两者差的 95% 区间跨 0，没测出差别。</li>
    </ul>
    <p class="note" style="margin-top:var(--sp-4)">这一节的原文在
      <code>docs/process/experiments/run-logs/EXP-008-20261003-units-R1R2.md</code>
      的「成绩速览」一节，有出入以那里为准。</p>
  </div>
</main>

<main id="m-drill" role="tabpanel" aria-labelledby="tab-drill">
  <div class="cards" id="cards"></div>
  <div class="panel" style="margin-bottom:var(--card-gap)">
    <h2>逐折指标 <span class="hint">点表头可排序</span></h2>
    <div class="tw" id="foldtab"></div>
    <div class="note" id="foldnote" style="margin-top:var(--sp-3)"></div>
  </div>
  <div class="cols">
    <div class="panel" style="flex:1 1 540px;min-width:0">
      <h2>分组拆分 <span class="hint">整体分数好看，可能藏着一个源很差</span></h2>
      <div id="breakdown"></div>
    </div>
    <div class="panel" style="flex:1 1 370px;min-width:0">
      <h2>视频列表 <span class="hint" id="cnt"></span></h2>
      <div class="toolbar">
        <span class="field"><label for="th">阈值</label>
          <input type="range" id="th" min="0" max="1" step="0.01" value="0.5">
          <b class="v" id="thv" style="font-variant-numeric:tabular-nums">0.50</b></span>
        <span class="field"><input type="checkbox" id="onlyerr">
          <label for="onlyerr">只看判错的</label></span>
        <span class="field" id="methodwrap" hidden><label for="method">方法</label>
          <select id="method"></select></span>
        <span class="field"><label for="sort">排序</label>
          <select id="sort">
            <option value="score_desc">分数从高到低</option>
            <option value="score_asc">分数从低到高</option>
            <option value="group">按场次分组</option>
          </select></span>
        <span class="field"><label for="bucket" class="sr">分组统计</label>
          <select id="bucket"></select></span>
        <span class="field"><label for="q" class="sr">搜索</label>
          <input type="search" id="q" placeholder="搜索视频 / 场次组 / 来源" size="18"></span>
      </div>
      <div id="bucketbox"></div>
      <div class="list" id="list"></div>
    </div>
    <div class="panel" style="flex:1 1 420px;min-width:0" id="detailpanel">
      <h2>视频详情 <span class="hint">点左边任意一条</span></h2>
      <div id="detail"><div class="empty">还没有选中视频。<br>
        在左边列表里点一条，这里会显示它的 48 个片段的得分和具体画面。</div></div>
    </div>
  </div>
</main>

<main id="m-chart" role="tabpanel" aria-labelledby="tab-chart">
  <div class="cols">
    <div class="panel" style="flex:0 1 280px;min-width:220px">
      <h2>选图</h2>
      <div class="chartlist" id="chartlist"></div>
      <p class="note" style="margin-top:var(--sp-4)">
        图由服务端现画。点图可放大：滚轮缩放、按住拖动平移、双击复位、Esc 关闭。
      </p>
      <a id="dl" class="note" download>下载当前图</a>
    </div>
    <div class="panel" style="flex:1 1 620px;min-width:0">
      <h2 id="charttitle">图表</h2>
      <div class="chartbox"><img id="chartimg" alt=""></div>
      <div class="note" id="chartnote" style="margin-top:var(--sp-3)"></div>
    </div>
  </div>
</main>

<main id="m-infer" role="tabpanel" aria-labelledby="tab-infer">
  <div class="panel">
    <h2>输入一条新视频，看模型判它有没有泥石流</h2>
    <p class="note">请使用未剪辑的原视频，可直接选择文件或在下方填写路径。程序会先按训练时的流程转成
      256×256，再检查画面和时间；转换失败或找不到能用的画面时，不会给出分数。
      这次训练分 5 折，得到 5 个模型；这里让它们各给视频打一分。</p>
    <div class="toolbar" style="margin-top:var(--sp-4)">
      <input type="file" id="vfile" accept="video/*" aria-label="选择本地视频文件">
      <button id="upbtn">上传并推理</button>
    </div>
    <p class="note">上传会把文件复制一份到 <code>.codex/docs/.temp/infer-video/uploads/</code>（用完可删）。
      视频很大就用下面那栏直接填路径，服务读原文件、不复制。</p>
    <div class="toolbar" style="margin-top:var(--sp-2)">
      <input type="text" id="vpath" placeholder="视频绝对路径，例如 D:\data\video\xxx.mp4"
        style="flex:1 1 420px" aria-label="视频绝对路径">
      <button id="runbtn">开始推理</button>
      <span class="note" id="prog" aria-live="polite"></span>
    </div>
    <div class="progress" id="progbar" hidden role="progressbar"
      aria-label="推理进度" aria-valuemin="0" aria-valuemax="100"><i></i></div>
    <p class="note">程序会沿整条视频选最多 48 个位置，每处取一组 16 帧。
      原间隔模型取连续 16 帧；新间隔模型每隔约 0.5 秒取一帧。
      结果会显示实际取帧方式、低灰度帧数和参与打分的片段数。
      低灰度画面、时间异常和已标出的剪辑区间附近的片段会被避开。
      少于 48 个片段时照实际数量算，不能与 EXP-014 的 48 片段成绩直接比较。
      切换页面或刷新后，仍可接着查看进度。</p>
    <div id="inferout"></div>
  </div>
</main>

<dialog id="help" aria-labelledby="help-title">
  <div class="hwrap">
    <div class="hbar">
      <span id="help-title">这一页怎么读</span>
      <span class="sp"><button type="button" class="ghost" id="helpclose">关闭</button></span>
    </div>
    <table class="keys">
      <tbody>
        <tr><th colspan="2">页面上的东西</th></tr>
        <tr><th>视频</th><td>列表里的一行就是一条视频。</td></tr>
        <tr><th>片段</th><td>在整条视频的时间轴上等间隔选位置，每个位置取 16 帧作为一个片段。原间隔模型取连续帧，新间隔模型每隔约 0.5 秒取一帧。整条视频的得分取自参与打分的片段中得分最高的一个；落在禁区里的位置不参与打分。</td></tr>
        <tr><th>第 N 折的模型</th><td>全部视频分成 5 份，轮流拿 4 份训练、剩 1 份考试，得到 5 个模型。每条视频的得分来自没见过它的那个模型，没有单独训练的「最终模型」。</td></tr>
        <tr><th>场次组</th><td>同一天、同一个机位拍的那批视频算一组。分组按场次组切，因为同一组的画面几乎一样，一组进训练一组进考试会让成绩虚高。</td></tr>
        <tr><th>阈值</th><td>模型输出 0~1 的分数，滑条决定超过多少判成有泥石流，默认 0.5。</td></tr>
      </tbody>
      <tbody>
        <tr><th colspan="2">表里的指标</th></tr>
        <tr><th>AUC</th><td>随机抽一条有泥石流的和一条没有的，模型给前者打分更高的概率。0.5 是瞎猜，1.0 是全对。</td></tr>
        <tr><th>P / N</th><td>这一折里实际有泥石流的视频数 / 没有的视频数（Positive / Negative）。</td></tr>
        <tr><th>TP / FP<br>FN / TN</th><td>混淆矩阵的四个格子。TP 报对了，FP 误报（实际没有却报成有），FN 漏报（实际有却没报），TN 正确排除。</td></tr>
        <tr><th>Precision</th><td>= TP / (TP + FP)，模型报出「有泥石流」的里面真的有的占多少。</td></tr>
        <tr><th>Recall</th><td>= TP / (TP + FN)，实际有泥石流的里面被报出来的占多少。</td></tr>
        <tr><th>FPR</th><td>= FP / (FP + TN)，实际没有泥石流的里面被误报成有的占多少，也就是白跑的比例。</td></tr>
        <tr><th>Specificity</th><td>= TN / (FP + TN) = 1 − FPR，实际没有泥石流的里面判对的比例。表里不单列。</td></tr>
        <tr><th>FNR</th><td>= FN / (FN + TP) = 1 − Recall，漏报比例。表里不单列。</td></tr>
        <tr><th>F1</th><td>Precision 与 Recall 的调和平均。</td></tr>
        <tr><th>pred_pos_rate</th><td>被判成有泥石流的视频占比。不是标准指标，是诊断量——模型把所有视频都判成有泥石流时它等于 1，立刻暴露。</td></tr>
        <tr><th>换成同折百分位后</th><td>5 个模型的打分松紧不一样，把分数换成「在它那一折里排第几的百分比」之后，5 折的分数才能放在一起比。</td></tr>
        <tr><th>区间</th><td>把整批场次组有放回地重抽 2000 次、每次重算一遍成绩，取中间 95% 的范围。回答「换一批场次组，成绩会晃到哪」。</td></tr>
      </tbody>
      <tbody>
        <tr><th colspan="2">七套成绩对比页</th></tr>
        <tr><th>监督信号</th><td>喂给模型的标签形态。弱标签 MIL 只给整条视频一个标签、不指出是哪几秒；片段级强标签 BCE 逐个片段给标签。</td></tr>
        <tr><th>片段集</th><td>哪些片段参与训练和打分。strict 只用完整落在一段标注区间里的片段；loose 把完整落在 2 秒以上未标注空隙里的片段也算成没有泥石流；full 不筛，48 个位置全用。</td></tr>
        <tr><th>划分协议</th><td>考试分组方式。协议一按场次组整组切，考试用的场次组模型从没见过；协议三只按标签随机分 5 折，同一天同一机位的视频可能分到两侧，含场次内画面相似性，只作上限参照，不是泛化成绩。</td></tr>
        <tr><th>逐折 AUC</th><td>f1~f5 是 5 个折各自的 AUC。某一折特别低是组级划分的正常现象——那一折的场次组模型从没见过。</td></tr>
      </tbody>
    </table>
  </div>
</dialog>

<dialog id="viewer" aria-labelledby="viewer-title">
  <div class="vwrap">
    <div class="vbar">
      <!-- aria-live：点缩略图换帧时标题文字会变，屏幕阅读器靠这个属性把新帧号
           读出来，否则换帧之后用户听不到任何反馈。 -->
      <span id="viewer-title" aria-live="polite"></span>
      <span class="sp">
        <button type="button" id="vreset" class="ghost" hidden>适应窗口</button>
        <button type="button" id="vclose">关闭</button>
      </span>
    </div>
    <div class="vstage" id="vstage"><img id="vimg" alt=""></div>
    <div class="vstrip" id="vstrip" hidden></div>
    <div class="vhint" id="vhint"></div>
  </div>
</dialog>

<script>
const RUN = "__RUN__", KIND = "__KIND__";
let DATA = null, TH = 0.5, SEL = null, METHOD = null;
let TAB = 'sets', CURCHART = 'fold_auc', SORT = {key: null, dir: 1};

/* ---------------------------------------------------------- 页签与 URL 同步 */
function setTab(t, push) {
  if (!document.getElementById('m-' + t)) t = 'sets';
  TAB = t;
  document.querySelectorAll('.tab').forEach(x =>
    x.setAttribute('aria-selected', String(x.dataset.t === t)));
  document.querySelectorAll('main').forEach(m => m.classList.toggle('on', m.id === 'm-' + t));
  // 写 hash 而不是 replaceState：这样浏览器后退键能回到上一个页签。
  // 赋值前先比对，避免重复写同一个 hash 产生多余的历史记录。
  if (push !== false && location.hash.slice(1) !== t) location.hash = t;
  if (t === 'chart' && DATA && chartsAvailable()) pickChart(CURCHART);
  // 隐藏的页签尺寸归零，观察器会据此把可滚动容器的 tabindex 摘掉；切回来时
  // 容器里没有任何 DOM 变化，观察器不会再触发，tabindex 就补不回来了。这里补一次。
  markScrollers();
}
document.querySelectorAll('.tab').forEach(t => {
  t.onclick = () => setTab(t.dataset.t);
  t.onkeydown = e => {              // 页签之间用左右方向键切换（WAI-ARIA 页签模式）
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    const all = [...document.querySelectorAll('.tab')];
    const i = all.indexOf(t) + (e.key === 'ArrowRight' ? 1 : -1);
    const n = all[(i + all.length) % all.length];
    n.focus(); setTab(n.dataset.t);
  };
});
addEventListener('hashchange', () => setTab(location.hash.slice(1) || 'sets', false));

/* ---------------------------------------------------------- 通用小工具 */
const fmt = (v, n = 4) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : Number(v).toFixed(n);
const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct = v => (v === null || v === undefined || Number.isNaN(v)) ? '—'
  : (v * 100).toFixed(1) + '%';

/* 会横向溢出的表格容器默认不可聚焦，键盘用户滚不到右边被裁掉的列。
   内容超出时才加 tabindex="0"：没超出也加的话，Tab 会停在一个滚不动的框上。 */
function markScrollers(root) {
  (root || document).querySelectorAll('.tw,.list,dialog#help').forEach(el => {
    const over = el.scrollWidth > el.clientWidth + 1
              || el.scrollHeight > el.clientHeight + 1;
    if (over) el.setAttribute('tabindex', '0');
    else el.removeAttribute('tabindex');
  });
}
// 页面内容由 innerHTML 反复重写，逐处调用容易漏；用一个观察器统一在每次改动后补。
// 只观察 childList 不观察 attributes，否则本函数自己写的 tabindex 会把它再次触发。
new MutationObserver(() => markScrollers())
  .observe(document.body, {childList: true, subtree: true});
let mstimer;
addEventListener('resize', () => { clearTimeout(mstimer); mstimer = setTimeout(markScrollers, 150); });

function rankNorm(score, ref) {
  if (!ref || !ref.length) return NaN;
  let lo = 0, hi = ref.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (ref[m] < score) lo = m + 1; else hi = m; }
  return lo / ref.length;
}

/* ============================================================ 放大查看器
   底座用原生 <dialog>.showModal()：浏览器免费提供顶层显示、::backdrop 遮罩、
   Esc 关闭、Tab 焦点锁在弹层内、关闭后焦点回到触发元素。自己写 div 遮罩层
   这五件事都要手写且容易漏。图表要缩放平移，帧图要左右翻页，两套行为分开。 */
const viewer = document.getElementById('viewer');
const vimg = document.getElementById('vimg');
const vstage = document.getElementById('vstage');
const vstrip = document.getElementById('vstrip');
const vtitle = document.getElementById('viewer-title');
const vhint = document.getElementById('vhint');
const vreset = document.getElementById('vreset');
let vmode = null, vscale = 1, vtx = 0, vty = 0, vframes = null, vidx = 0, vlastFocus = null;

function vApply() {
  vimg.style.transform = `translate(${vtx}px,${vty}px) scale(${vscale})`;
  vstage.classList.toggle('zoomed', vscale > 1.001);
  vreset.hidden = !(vscale > 1.001);
}
function vFit() { vscale = 1; vtx = 0; vty = 0; vimg.style.transformOrigin = 'center center'; vApply(); }

function openImage(src, title) {
  vmode = 'zoom'; vframes = null;
  vstage.classList.remove('page');
  vimg.src = src; vimg.alt = title || '图表';
  vtitle.textContent = title || '图表';
  vstrip.hidden = true; vstrip.innerHTML = '';
  vhint.textContent = '滚轮缩放 · 按住拖动平移 · 双击在放大与适应窗口之间切换 · Esc 关闭';
  vFit(); show();
}
function openFrames(list, i, title) {
  vmode = 'page'; vframes = list; vidx = i;
  vstage.classList.add('page');   // 铺满可用区域，见 .vstage.page img 的说明
  vstrip.hidden = false;
  vstrip.innerHTML = list.map((f, k) =>
    `<button type="button" data-k="${k}" aria-current="${k === i}"
      aria-label="第 ${k + 1} 帧"><img src="${f}" alt=""></button>`).join('');
  vstrip.querySelectorAll('button').forEach(b => b.onclick = () => vShowFrame(+b.dataset.k));
  vtitle.textContent = title;
  vhint.textContent = '点下方缩略图换帧 · Esc 关闭';
  vFit(); show(); vShowFrame(i);
}
function vShowFrame(k) {
  vidx = (k + vframes.length) % vframes.length;
  vimg.src = vframes[vidx];
  vtitle.textContent = `${vframeTitle} · 第 ${vidx + 1} 帧 / 共 ${vframes.length} 帧`;
  vstrip.querySelectorAll('button').forEach((b, j) =>
    b.setAttribute('aria-current', String(j === vidx)));
  const cur = vstrip.querySelector('button[aria-current="true"]');
  if (cur) cur.scrollIntoView({block: 'nearest', inline: 'center'});
}
let vframeTitle = '';
function openFramesAt(list, i, title) { vframeTitle = title; openFrames(list, i, title); }

function show() {
  vlastFocus = document.activeElement;
  if (!viewer.open) viewer.showModal();
  document.getElementById('vclose').focus();
}
viewer.addEventListener('close', () => { vmode = null; vimg.src = ''; });
viewer.addEventListener('click', e => { if (e.target === viewer) viewer.close(); });
document.getElementById('vclose').onclick = () => viewer.close();
vreset.onclick = vFit;

/* 「这一页怎么读」弹窗。同样用 <dialog>，Esc 与点遮罩关闭都由浏览器提供。 */
const helpDlg = document.getElementById('help');
function toggleHelp(force) {
  const want = force === undefined ? !helpDlg.open : force;
  if (want && !helpDlg.open) { helpDlg.showModal(); markScrollers(); }
  else if (!want && helpDlg.open) helpDlg.close();
}
document.getElementById('helpbtn').onclick = () => toggleHelp();
document.getElementById('helpclose').onclick = () => toggleHelp(false);
helpDlg.addEventListener('click', e => { if (e.target === helpDlg) helpDlg.close(); });

// 滚轮缩放：以光标位置为锚点（地图类工具的共同行为，用户已形成预期）
vstage.addEventListener('wheel', e => {
  if (vmode !== 'zoom' || e.ctrlKey) return;   // 不抢浏览器的 Ctrl+滚轮页面缩放
  e.preventDefault();
  const r = vimg.getBoundingClientRect();
  const ox = (e.clientX - r.left) / r.width, oy = (e.clientY - r.top) / r.height;
  vscale = Math.min(8, Math.max(1, vscale * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
  vimg.style.transformOrigin = `${ox * 100}% ${oy * 100}%`;
  if (vscale === 1) { vtx = 0; vty = 0; }
  vApply();
}, {passive: false});

// 拖拽平移用 pointer 事件 + setPointerCapture：鼠标、触控笔、手指走同一套代码
let drag = null;
vstage.addEventListener('pointerdown', e => {
  if (vmode !== 'zoom' || vscale <= 1) return;
  drag = {x: e.clientX, y: e.clientY, tx: vtx, ty: vty};
  vstage.setPointerCapture(e.pointerId); vstage.classList.add('grabbing');
});
vstage.addEventListener('pointermove', e => {
  if (!drag) return;
  vtx = drag.tx + (e.clientX - drag.x); vty = drag.ty + (e.clientY - drag.y); vApply();
});
const endDrag = e => {
  if (!drag) return;
  drag = null; vstage.classList.remove('grabbing');
  try { vstage.releasePointerCapture(e.pointerId); } catch (err) { }
};
vstage.addEventListener('pointerup', endDrag);
vstage.addEventListener('pointercancel', endDrag);
vimg.addEventListener('dblclick', () => { vscale > 1 ? vFit() : (vscale = 2, vApply()); });

/* 键盘只保留浏览器自带的行为：Tab 移动焦点、回车/空格触发按钮、Esc 关弹层
   （原生 <dialog> 自带）。页面不再注册任何单键快捷键（2026-10-08 叶海峰定）。 */
</script>
<script id="app">
/* ============================================================ 数据装载
   页面先渲染骨架，数据到了再填。加载失败必须留下可见的错误和原因——
   空白页会让人以为是自己操作错了。 */
const $ = id => document.getElementById(id);
const isNum = v => v !== null && v !== undefined && !Number.isNaN(v);
const avg = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN;
const std = a => { const m = avg(a); return a.length ? Math.sqrt(avg(a.map(v => (v - m) ** 2))) : NaN; };

function fail(msg) {
  $('cards').innerHTML = `<div class="err" style="grid-column:1/-1">
    <b>数据没读出来。</b>${esc(msg)}<br>
    多半是服务端那个进程已经退出或报错了——回到终端看有没有 traceback，再刷新本页。</div>`;
  $('foldtab').innerHTML = ''; $('list').innerHTML = ''; $('detail').innerHTML = '';
}
$('cards').innerHTML = `<div class="card"><div class="skeleton" style="width:40%"></div>
  <div class="skeleton" style="width:70%;height:30px;margin:12px 0"></div>
  <div class="skeleton" style="width:90%"></div></div>`.repeat(3);

/* ---------------------------------------------------------- 七套成绩对比
   只读 data/results/exp008/<run_id>/metrics.json，数字全部现读或现算，
   页面里不写死。与 --run 指定的那个 run 无关，所以单独取，不挂在 /api/data 上——
   那边读失败也不该让这一页空着。 */
function buildSets() {
  const box = $('setstab'), note = $('setsnote');
  box.innerHTML = '<div class="skeleton" style="height:180px;margin:12px"></div>';
  fetch('/api/runsets').then(r => {
    if (!r.ok) throw new Error('服务端返回 HTTP ' + r.status);
    return r.json();
  }).then(d => {
    if (!d.sets.length) throw new Error('一个 run 的 metrics.json 都没读到');
    let h = '<table><thead><tr><th class="l">设定名</th><th class="l">监督信号</th>'
          + '<th>片段集</th><th class="l">划分协议</th>'
          + '<th>5 折 AUC 均值 ± 标准差</th><th class="l">逐折 AUC</th>'
          + '<th class="l">用途</th></tr></thead><tbody>';
    d.sets.forEach(s => {
      h += `<tr${s.core ? ' class="row-core"' : ''}>`
         + `<td class="l"><code>${esc(s.run_id)}</code></td>`
         + `<td class="l">${esc(s.signal)}</td><td>${esc(s.clips)}</td>`
         + `<td class="l">${esc(s.protocol)}`
         + (s.footnote ? ` <sup title="${esc(s.footnote)}">†</sup>` : '') + '</td>'
         + `<td><b>${fmt(s.mean_auc, 4)}</b> ± ${fmt(s.std_auc, 4)}</td>`
         + `<td class="l">${s.folds.map(f => `f${f.fold} ${fmt(f.auc, 3)}`).join(' · ')}</td>`
         + `<td class="l">${esc(s.purpose)}</td></tr>`;
    });
    box.innerHTML = h + '</tbody></table>';
    const a = d.sets[0];
    let n = `<b>†</b> ${esc(d.footnote)}。`;
    if (d.sets.every(s => s.n_unit === a.n_unit && s.n_pos === a.n_pos
                       && s.n_neg === a.n_neg && s.seed === a.seed)) {
      n += `七套都跑在同一批 ${a.n_unit} 条视频上（正 ${a.n_pos} / 负 ${a.n_neg}），`
         + `随机种子 ${a.seed}，各套逐折的正负视频数完全一致。`;
    } else {
      n += '注意：七套的视频数或正负构成不一致，横向比较前先查各套的 metrics.json。';
    }
    n += ' 逐折 AUC 下面是 5 个折各自的成绩，某一折特别低通常是那一折的场次组'
       + '模型从没见过，属组级划分的正常现象。';
    if (d.mismatch.length) {
      n += `<br><b>对不上：</b>${esc(d.mismatch.join('、'))} 的 metrics.json 里存的均值`
         + '或标准差与按逐折 AUC 现算的结果不一致（差超过 0.0005），表里显示的是现算值。';
    }
    if (d.missing.length) {
      n += `<br><b>没找到：</b>${esc(d.missing.join('、'))}——${esc(d.dir)} 下没有这几个`
         + ' run 的 metrics.json，表里缺这几行。';
    }
    note.innerHTML = n;
  }).catch(e => {
    box.innerHTML = `<div class="err" style="margin:0"><b>七套成绩没读出来。</b>`
      + esc(e && e.message ? e.message : String(e))
      + '<br>多半是服务端那个进程已经退出或报错了——回到终端看有没有 traceback，再刷新本页。</div>';
  });
}

buildSets();
fetch('/api/data').then(r => {
  if (!r.ok) throw new Error('服务端返回 HTTP ' + r.status);
  return r.json();
}).then(d => { DATA = d; METHOD = d.methods[0]; boot(); })
  .catch(e => fail(e && e.message ? e.message : String(e)));

function boot() {
  const ms = $('method');
  ms.innerHTML = DATA.methods.map(m => `<option>${esc(m)}</option>`).join('');
  $('methodwrap').hidden = DATA.methods.length < 2;
  ms.onchange = e => { METHOD = e.target.value; refreshAll(); };
  $('th').oninput = e => {
    TH = parseFloat(e.target.value);
    $('thv').textContent = TH.toFixed(2);
    refreshAll();
  };
  $('onlyerr').onchange = render;
  $('sort').onchange = render;
  $('q').oninput = render;
  buildBucketDim();
  refreshAll();
  buildCharts();
  if (KIND === 'temporal') buildInferNote();
  setTab(location.hash.slice(1) || 'sets', false);
}

function refreshAll() {
  buildCards(); buildFoldTable(); buildBreakdown(); render();
  if (SEL !== null) showDetail(SEL);
}

function curFoldMetrics() { return (DATA.fold_metrics || {})[METHOD] || []; }
function scoreOf(u) { return u.scores ? u.scores[METHOD] : u.unit_score; }
function uidOf(u) { return u.unit_id ?? (u.video_id + '/' + u.seg_id + '#' + u.win_i); }

/* ============================================================ 概览卡
   三张卡回答三个不同的问题：① 主成绩是多少、有多不确定；② 换一种算法口径
   是多少（两个口径不能混着引用）；③ 这个阈值实际会造成多少次白跑。
   数字随阈值滑条实时变，所以整体重算，不做增量。 */
function buildCards() {
  const pf = curFoldMetrics();
  const aucs = pf.map(f => f.auc).filter(isNum);
  const mean = avg(aucs), sd = std(aucs);
  const ci = DATA.ci || {};
  const pool = DATA.pooled || {};
  const nPos = DATA.items.filter(u => u.label === 1).length;
  const negs = DATA.items.filter(u => u.label === 0);
  const fps = negs.filter(u => scoreOf(u) >= TH);
  const byG = {};
  fps.forEach(u => { const k = u.group_id || '（无组）'; byG[k] = (byG[k] || 0) + 1; });
  const top = Object.entries(byG).sort((a, b) => b[1] - a[1])[0];
  const predPos = DATA.items.filter(u => scoreOf(u) >= TH).length;

  $('cards').innerHTML = `
  <div class="card">
    <h3>主成绩 · 按条视频算的 AUC</h3>
    <div class="big">${fmt(mean)}<small>± ${fmt(sd)}</small></div>
    <div class="sub">95% 区间 [${fmt(ci.lo, 3)}, ${fmt(ci.hi, 3)}]</div>
    <div class="foot">± 是 5 折成绩的总体标准差，不是置信区间。95% 区间按场次组
      有放回地重抽 ${ci.n_boot || 0} 次算出——同一天同一机位拍的视频画面相似，
      按条视频重抽会高估样本量。</div>
  </div>

  <div class="card">
    <h3>另一种算法 · 5 折的分数混在一起算</h3>
    <div class="big">${fmt(pool.auc_raw)}<small>换成同折百分位后 ${fmt(pool.auc_ranknorm, 3)}</small></div>
    <div class="sub">AP ${fmt(pool.ap_raw, 3)}（归一后 ${fmt(pool.ap_ranknorm, 3)}）
      · 正类占比 ${fmt(pool.pos_rate, 3)}</div>
    <div class="foot">每条视频由它自己那一折的模型打分，5 个模型打分的松紧不一样，
      所以这个数比上面那张卡低。两种算法不能混着引用。</div>
  </div>

  <div class="card ${fps.length ? 'warncard' : ''}">
    <h3>当前阈值 ${TH.toFixed(2)} 下的报警情况</h3>
    <div class="big">${fmt(predPos / DATA.items.length, 3)}<small>pred_pos_rate</small></div>
    <div class="sub">${negs.length} 条没有泥石流的视频里 <b>${fps.length}</b> 条被判成有泥石流
      ${top ? `，最集中的一组是 ${esc(top[0])}（${top[1]} 条）` : ''}</div>
    <div class="sub">${nPos} 条有泥石流的视频里漏报 ${nPos - DATA.items.filter(u => u.label === 1 && scoreOf(u) >= TH).length} 条</div>
    <div class="foot">阈值是在训练时用过的验证数据上挑的，成绩比真实部署乐观。</div>
  </div>`;
}

/* ============================================================ 逐折指标表
   Specificity（= 1 − FPR）与 FNR（= 1 − Recall）两列不列：它们跟已列的两列
   互为补数，并排画等于同一个数写两遍，还会把表挤到横向滚动。 */
const FOLD_COLS = [
  ['fold', '折', 'l', f => f.fold, v => '折' + v],
  ['n', '视频', '', f => f.n, v => v ?? '—'],
  ['n_pos', 'P', '', f => f.n_pos, v => v ?? '—'],
  ['n_neg', 'N', '', f => f.n_neg, v => v ?? '—'],
  ['auc', 'AUC', '', f => f.auc, v => fmt(v, 4)],
  ['precision', 'Precision', '', f => f.precision, v => fmt(v, 3)],
  ['recall', 'Recall', '', f => f.recall, v => fmt(v, 3)],
  ['fpr', 'FPR', '', f => f.fpr, v => fmt(v, 3)],
  ['f1', 'F1', '', f => f.f1, v => fmt(v, 3)],
  ['pred_pos_rate', 'pred_pos_rate', '', f => f.pred_pos_rate, v => fmt(v, 3)],
  ['best_epoch', '最佳轮次', '', f => f.best_epoch, v => v ?? '—'],
];
// 默认按第 0 列（折号）升序，也就是表格的自然顺序。
// 这里不能填 null：表格明明排好了序却不给任何一列写 aria-sort，
// 屏幕阅读器用户无从得知当前是按哪一列排的。
let SORTF = {i: 0, dir: 1};

function buildFoldTable() {
  const rows = curFoldMetrics().slice();
  if (SORTF.i !== null) {
    const get = FOLD_COLS[SORTF.i][3];
    rows.sort((a, b) => {
      const x = get(a), y = get(b);
      if (!isNum(x) && !isNum(y)) return 0;
      if (!isNum(x)) return 1;                 // 无定义的排最后，不管升降序
      if (!isNum(y)) return -1;
      return (x - y) * SORTF.dir;
    });
  }
  let h = '<table><thead><tr>';
  FOLD_COLS.forEach(([k, t, cls], i) => {
    const on = SORTF.i === i;
    h += `<th class="sortable ${cls}" data-i="${i}"
      ${on ? `aria-sort="${SORTF.dir > 0 ? 'ascending' : 'descending'}"` : ''}
      ><button type="button" class="thbtn ${cls}"
      title="点击按这一列排序">${t}<span class="arrow">${
        on ? (SORTF.dir > 0 ? '▲' : '▼') : '↕'}</span></button></th>`;
  });
  h += '</tr></thead><tbody>';
  for (const f of rows) {
    h += '<tr>';
    FOLD_COLS.forEach(([k, t, cls, , show]) => {
      h += `<td class="${cls}">${esc(show(f[k]))}</td>`;
    });
    h += '</tr>';
  }
  h += '</tbody></table>';
  $('foldtab').innerHTML = h;
  $('foldtab').querySelectorAll('th.sortable').forEach(th => th.onclick = () => {
    const i = +th.dataset.i;
    SORTF = {i, dir: SORTF.i === i ? -SORTF.dir : 1};
    buildFoldTable();
    // 重建表格换掉了按钮节点，焦点随之掉回 body。键盘用户按一次回车排完序，
    // 必须还能接着按，所以把焦点放回同一列的按钮上（鼠标点击不会显示焦点环）。
    const nb = $('foldtab').querySelector(`th.sortable[data-i="${i}"] button`);
    if (nb) nb.focus();
  });
  const note = 'FPR = 实际没有泥石流的视频里被判成有泥石流的比例，等于「白跑多少次」；'
    + 'Recall = 真出事报出来的比例，1 − Recall 就是 FNR，也就是漏报的比例。两列都随上面的阈值滑条变。'
    + 'Specificity（= 1 − FPR）与 FNR（= 1 − Recall）不单列，它们是同一个数的两面。';
  $('foldnote').textContent = note;
}

/* ============================================================ 分组拆分
   按 AUC 升序排：最差的那一组排最上面。整体分数好看，可能藏着一个源很差。 */
function buildBreakdown() {
  const bd = DATA.breakdown;
  if (!bd) { $('breakdown').innerHTML = ''; return; }
  let h = '';
  for (const [dim, label] of [['source', '按数据来源'], ['group_id', '按场次分组']]) {
    const rows = bd[dim];
    if (!rows || Object.keys(rows).length < 2) continue;
    const keys = Object.keys(rows).sort((a, b) => {
      const x = rows[a].auc, y = rows[b].auc;
      const nx = isNum(x), ny = isNum(y);
      // 单类组（全是正或全是负）的 AUC 无定义。比较函数必须自洽——两边都无定义时
      // 不能返回「a 在后」，否则 a<b 与 b<a 同时成立，排序结果随机。
      if (!nx && !ny) return String(a).localeCompare(String(b));
      if (!nx) return 1;
      if (!ny) return -1;
      return x - y;
    });
    h += `<div class="note" style="margin:var(--sp-4) 0 var(--sp-2)">${label}`
       + `（按 AUC 从低到高，最差的一组在最上面；一组里只有一种答案时 AUC 算不出来，排在最后）</div>`
       + '<div class="tw"><table><thead><tr><th class="l">组</th><th>视频</th>'
       + '<th>P/N</th><th>AUC</th><th>Recall</th>'
       + '<th>FPR</th></tr></thead><tbody>';
    for (const k of keys) {
      const m = rows[k];
      h += `<tr><td class="l trunc" title="${esc(k)}">${esc(k)}</td><td>${m.n}</td>`
         + `<td>${m.n_pos}/${m.n_neg}</td><td>${fmt(m.auc, 3)}</td>`
         + `<td>${fmt(m.recall, 3)}</td>`
         + `<td>${fmt(m.fpr, 3)}</td></tr>`;
    }
    h += '</tbody></table></div>';
  }
  $('breakdown').innerHTML = h || '<div class="note">这一跑没有分组统计。</div>';
}

/* ============================================================ 视频列表 */
function buildBucketDim() {
  const s = $('bucket');
  s.innerHTML = '<option value="">分组统计：不分组</option>' + DATA.bucket_dims
    .map(d => `<option value="${d.key}">分组统计：${esc(d.label)}</option>`).join('');
  s.onchange = render;
}

function shown() {
  const onlyErr = $('onlyerr').checked;
  const q = $('q').value.trim().toLowerCase();
  const sort = $('sort').value;
  const us = DATA.items.filter(u => {
    if (onlyErr) { if ((scoreOf(u) >= TH ? 1 : 0) === u.label) return false; }
    if (q) {
      const hay = [uidOf(u), u.group_id, u.source, u.video_id].join(' ').toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
  if (sort === 'score_desc') us.sort((a, b) => scoreOf(b) - scoreOf(a));
  else if (sort === 'score_asc') us.sort((a, b) => scoreOf(a) - scoreOf(b));
  else us.sort((a, b) => String(a.group_id).localeCompare(String(b.group_id))
    || scoreOf(b) - scoreOf(a));
  return us;
}

function render() {
  const items = shown();
  $('cnt').textContent = `显示 ${items.length} / ${DATA.items.length}`;
  const key = $('bucket').value;
  let box = '';
  if (key) {
    const g = {};
    for (const u of items) (g[u[key]] = g[u[key]] || []).push(u);
    let h = '<div class="tw"><table><thead><tr><th class="l">' + esc(key)
          + '</th><th>视频</th><th>P</th><th>N</th><th>AUC</th><th>pred_pos_rate</th>'
          + '</tr></thead><tbody>';
    for (const k of Object.keys(g).sort()) {
      const us = g[k], y = us.map(u => u.label), s = us.map(u => scoreOf(u));
      h += `<tr><td class="l">${esc(k)}</td><td>${us.length}</td>`
         + `<td>${y.filter(v => v === 1).length}</td>`
         + `<td>${y.filter(v => v === 0).length}</td>`
         + `<td>${fmt(aucOf(y, s), 3)}</td>`
         + `<td>${fmt(s.filter(v => v >= TH).length / s.length, 3)}</td></tr>`;
    }
    box = h + '</tbody></table></div>';
  }
  $('bucketbox').innerHTML = box;

  if (!items.length) {
    $('list').innerHTML = `<div class="empty"><div class="big">没有符合条件的视频</div>
      ${$('onlyerr').checked ? '把「只看判错的」取消试试。' : ''}
      ${$('q').value ? '把搜索框清空试试。' : ''}</div>`;
    return;
  }
  let h = '';
  for (const u of items) {
    const s = scoreOf(u), pred = s >= TH ? 1 : 0, ok = pred === u.label;
    h += `<div class="unit ${SEL === u._i ? 'sel' : ''}" data-i="${u._i}"
        tabindex="0" role="button" aria-label="${esc(uidOf(u))}">
      <img loading="lazy" decoding="async" src="${thumbUrl(u)}" alt="">
      <div class="meta">
        <div class="uid">${esc(uidOf(u))}
          <span class="badge ${u.label ? 'pos' : 'neg'}"
            title="${u.label ? 'P：这条视频里有泥石流' : 'N：这条视频里没有泥石流'}"
            >${u.label ? 'P' : 'N'}</span>
          <span class="badge ${ok ? 'ok' : 'bad'}">${ok ? '✓ 判对' : '✗ 判错'}</span></div>
        <div class="note">第${u.fold}折的模型 · ${esc(u.group_id)} · ${esc(u.source)}</div>
      </div>
      <div class="sc">
        <div class="bar"><i class="${u.label ? 'p' : 'n'}"
          style="width:${(s * 100).toFixed(1)}%"></i></div>
        <div class="note">${fmt(s, 4)}</div></div>
    </div>`;
  }
  const L = $('list');
  L.innerHTML = h;
  L.querySelectorAll('.unit').forEach(e => {
    const i = () => parseInt(e.dataset.i);
    e.onclick = () => selectUnit(i(), false);
    e.onkeydown = ev => {
      if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); selectUnit(i(), true); }
      else if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
        ev.preventDefault();
        const sib = ev.key === 'ArrowDown' ? e.nextElementSibling : e.previousElementSibling;
        if (sib) selectUnit(parseInt(sib.dataset.i), true);
      }
    };
  });
}

function selectUnit(i, keepFocus) {
  SEL = i; render(); showDetail(SEL);
  // render() 把整个列表重写了一遍，原来的节点被换掉，焦点随之掉回 body。
  // 键盘操作（回车选中、方向键上下移动）必须把焦点接回新列表的同一行，
  // 否则按一次方向键之后就再也按不动了。鼠标点击不需要接。
  if (keepFocus) {
    const nn = $('list').querySelector(`.unit[data-i="${i}"]`);
    if (nn) { nn.focus(); nn.scrollIntoView({block: 'nearest'}); }
  }
  // 窄屏上详情面板会折到下一行，点完列表看不到任何变化。
  // 只在它确实不在视口里时才滚动，否则宽屏下每次点击页面都会跳。
  const p = $('detailpanel').getBoundingClientRect();
  if (p.top > innerHeight - 80) $('detailpanel').scrollIntoView(
    {block: 'start', behavior: 'smooth'});
}

function aucOf(y, s) {   // 单类折无定义
  const pos = s.filter((_, i) => y[i] === 1), neg = s.filter((_, i) => y[i] === 0);
  if (!pos.length || !neg.length) return NaN;
  let c = 0;
  for (const a of pos) for (const b of neg) c += (a > b) ? 1 : (a === b ? 0.5 : 0);
  return c / (pos.length * neg.length);
}

function thumbUrl(u) {
  if (DATA.kind !== 'temporal') {
    const c = u.argmax_clip ?? 0;
    return `/thumb?unit=${encodeURIComponent(u.unit_id)}&c=${c}&f=0`;
  }
  const f = (u.frame_files || [])[0];
  return f ? `/thumb?src=${encodeURIComponent(u.source)}&p=${encodeURIComponent(f)}` : '';
}

/* ============================================================ 视频详情
   针状图是 48 根等宽柱子，不是折线——相邻序号在时间上不一定连续，
   连线会假装它们是等间隔的时间序列。 */
let CURCLIP = 0;

function showDetail(i) {
  const u = DATA.items[i];
  if (DATA.kind === 'temporal') { showDetailTemporal(u); return; }
  const s = u.clip_scores, n = s.length;
  CURCLIP = u.argmax_clip ?? 0;
  const nUsable = s.filter(v => v !== null).length;
  const labOf = k => (u.clip_labels && u.clip_labels[k] !== null)
    ? (u.clip_labels[k] === 1 ? '有泥石流' : '没有泥石流') : '没有片段级答案';

  $('detail').innerHTML = `
    <div class="kv" style="margin-bottom:var(--sp-3)">
      <b>视频</b><span>${esc(u.unit_id)}（${esc(u.video_id)}）</span>
      <b>实际答案</b><span>${u.label ? '有泥石流' : '没有泥石流'} · 来自第 ${u.fold} 折的模型 · 场次组 ${esc(u.group_id)}</span>
      <b>整条视频得分</b><span>${fmt(u.unit_score)}（${nUsable}/${n} 个可用片段取最大）</span>
      <b>最高分片段</b><span>#${u.argmax_clip}${u.clip_labels
        ? `（这块片段实际标注：${labOf(u.argmax_clip)}）` : ''}</span>
    </div>
    <div id="needlebox"></div>
    <div class="toolbar" style="margin-top:var(--sp-2)">
      <button class="ghost" id="cpprev">‹ 上一片段</button>
      <span class="note" id="cplab"></span>
      <button class="ghost" id="cpnext">下一片段 ›</button>
      <button class="ghost" id="cpzoom">放大这 16 帧</button>
    </div>
    <div class="frames" id="fr"></div>
    <div class="note" id="frtip"></div>
    <div class="note" style="margin-top:var(--sp-3)">
      x 轴 = ${n} 个等间隔抽出来的片段序号，<b>不是连续时间</b>。
      橙色那根是决定整条得分的片段；灰色短块是被剔除的片段
      （它的 16 帧没有完整落在同一条人工标注段内，不参与打分）。
      ${DATA.kind === 'ft'
        ? `这道 run 训练时每个片段都有自己的答案（这块片段的 16 帧是否完整落在
           人工标注的有泥石流区间内），把鼠标放到柱子上就能看到它和分数对不对得上。
           最高分片段同样不代表事件定位，它只是参与打分的片段里分数最高的那一个。`
        : `最高分片段不代表模型找到了泥石流在哪一秒——训练时只告诉模型
           「这条视频里有没有泥石流」，没有逐片段的答案。`}</div>`;

  drawNeedle(u);
  $('cpprev').onclick = () => stepClip(u, -1);
  $('cpnext').onclick = () => stepClip(u, 1);
  $('cpzoom').onclick = () => {
    const fr = frameList(u, CURCLIP);
    if (fr) openFramesAt(fr, 0, `${esc(uidOf(u))} · 片段 #${CURCLIP}`);
  };
  showFrames(u, CURCLIP);
}

function drawNeedle(u) {
  const s = u.clip_scores, n = s.length;
  let svg = `<svg class="needle" viewBox="0 0 ${n * 8} 100"
    preserveAspectRatio="none" role="img"
    aria-label="48 个片段的分数柱状图">`;
  svg += `<line x1="0" y1="50" x2="${n * 8}" y2="50"
    stroke="var(--viz-baseline)" stroke-dasharray="4 3" stroke-width="0.5"/>`;
  for (let k = 0; k < n; k++) {
    const v = s[k];
    if (v === null) {
      svg += `<rect x="${k * 8 + 2}" y="94" width="4" height="5"
        fill="var(--border-strong)"><title>片段 #${k} 被剔除，未参与打分</title></rect>`;
      continue;
    }
    const hh = Math.max(1, v * 94);
    const col = (k === CURCLIP) ? 'var(--viz-negative)' : 'var(--viz-positive)';
    const lab = (u.clip_labels && u.clip_labels[k] !== null)
      ? `，实际标注：${u.clip_labels[k] === 1 ? '有泥石流' : '没有泥石流'}` : '';
    svg += `<rect x="${k * 8 + 1}" y="${98 - hh}" width="6" height="${hh}"
      fill="${col}" opacity="${k === CURCLIP ? 1 : .78}">`
      + `<title>片段 #${k}：${v.toFixed(4)}${lab}</title></rect>`;
  }
  svg += '</svg>';
  $('needlebox').innerHTML = svg;
  const el = $('needlebox').querySelector('svg');
  el.style.cursor = 'pointer';
  el.onclick = ev => {
    const r = el.getBoundingClientRect();
    const k = Math.min(n - 1, Math.max(0, Math.floor((ev.clientX - r.left) / r.width * n)));
    setClip(u, k);
  };
}

function setClip(u, k) {
  CURCLIP = Math.max(0, Math.min(u.clip_scores.length - 1, k));
  drawNeedle(u); showFrames(u, CURCLIP);
}
function stepClip(u, d) { setClip(u, CURCLIP + d); }

function frameList(u, k) {
  if (k === null || u.clip_scores[k] === null) return null;
  return Array.from({length: 16},
    (_, f) => `/thumb?unit=${encodeURIComponent(u.unit_id)}&c=${k}&f=${f}`);
}

function showFrames(u, k) {
  CURCLIP = k;
  const sc = u.clip_scores[k];
  $('cplab').textContent = `片段 #${k} / ${u.clip_scores.length}　分数 ${fmt(sc)}`;
  const list = frameList(u, k);
  if (!list) {
    $('fr').innerHTML = '';
    $('frtip').textContent = `片段 #${k} 被剔除（它的 16 帧没有完整落在同一条人工标注段内），没有参与打分。`;
    $('cpzoom').disabled = true;
    return;
  }
  $('cpzoom').disabled = false;
  $('fr').innerHTML = list.map((src, f) =>
    `<img loading="lazy" decoding="async" src="${src}" alt="第 ${f + 1} 帧" tabindex="0"
      title="点开可放大、左右翻帧">`).join('');
  $('fr').querySelectorAll('img').forEach((im, f) => {
    im.onclick = () => openFramesAt(list, f, `${uidOf(u)} · 片段 #${k}`);
    im.onkeydown = ev => {
      if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault(); openFramesAt(list, f, `${uidOf(u)} · 片段 #${k}`);
      }
    };
  });
  $('frtip').textContent =
    `片段 #${k}：${u.frame_note || '16 帧是连续原生帧'}。片段分数 ${fmt(sc)}。点任意一帧可放大。`;
}

function showDetailTemporal(u) {
  const list = (u.frame_files || []).map(f =>
    `/thumb?src=${encodeURIComponent(u.source)}&p=${encodeURIComponent(f)}`);
  $('detail').innerHTML =
    `<div class="kv"><b>组</b><span>${esc(u.group_id)}</span>
     <b>源</b><span>${esc(u.source)}</span><b>折</b><span>${u.fold}</span>
     <b>段</b><span>${esc(u.seg_id)}（${u.start_sec}s ~ ${u.end_sec}s）</span>
     <b>真值</b><span>${esc(u.kind)}</span></div>
     <div class="note" style="margin-top:var(--sp-3)">窗口 ${u.win_i} / ${u.n_win}，
     ${esc(METHOD)} 分数 ${fmt(scoreOf(u))}</div>
     <div class="frames" id="fr">${list.map((src, f) =>
       `<img loading="lazy" decoding="async" src="${src}" alt="第 ${f + 1} 帧" tabindex="0">`).join('')}</div>
     <div class="note">点任意一帧可放大。</div>`;
  $('fr').querySelectorAll('img').forEach((im, f) => {
    im.onclick = () => openFramesAt(list, f, `${uidOf(u)} · 窗口 ${u.win_i}`);
    im.onkeydown = ev => {
      if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault(); openFramesAt(list, f, `${uidOf(u)} · 窗口 ${u.win_i}`);
      }
    };
  });
}

/* ============================================================ 图表
   每张图配一句它的口径或局限，正文引用时把那句一起带上。 */
const CHART_GROUPS = [
  ['整体成绩', [
    ['fold_auc', '逐折 AUC', '色带是 5 折成绩的总体标准差，只有 5 个数，不是置信区间。'],
    ['fold_metrics', '逐折指标', 'Precision / Recall / F1 / FPR 四列；Specificity = 1 − FPR，不重复画。'],
    ['confusion', '混淆矩阵', '随阈值滑条变。'],
  ]],
  ['分布', [
    ['score_dist', '视频得分分布', '有泥石流和没有泥石流两类视频的得分重叠多少，一眼能看出阈值放在哪会切掉什么。'],
    ['score_strip', '分数-标签散点', '每条视频一个点，比直方图更能看出离群的视频。'],
  ]],
  ['曲线', [
    ['roc_perfold', '逐折 ROC', '每折一条，阶梯状——每折视频少，只有有限个切点。'],
    ['roc_pooled', '5 折混算 ROC', '另一条线是换成同折百分位后的结果；它与主成绩是两种算法，不能混着引用。'],
    ['pr', 'PR 曲线 + AP', '想知道「报出来的有多少是真的」时，PR 比 ROC 更敏感。'],
  ]],
  ['拆分', [
    ['source', '按数据来源拆分', ''],
    ['fold_table', '折构成表', ''],
  ]],
];

function chartsAvailable() { return KIND !== 'temporal'; }

function buildCharts() {
  if (!chartsAvailable()) {   // 这批图的成绩都按条视频算，时序线的每行是一个窗口
    $('chartlist').innerHTML =
      '<div class="note">图表页签的图都按条视频算成绩；时序线的一行是一个窗口，画不了同一套图。'
      + '时序线的逐折指标见「样本与错误」页签的表格。</div>';
    $('chartimg').removeAttribute('src');
    $('charttitle').textContent = '图表';
    return;
  }
  $('chartlist').innerHTML = CHART_GROUPS.map(([g, items]) =>
    `<div class="grp">${esc(g)}</div>` + items.map(([k, t]) =>
      `<button type="button" data-k="${k}">${esc(t)}</button>`).join('')).join('');
  $('chartlist').querySelectorAll('button').forEach(b => {
    b.onclick = () => pickChart(b.dataset.k);
    b.onkeydown = ev => {
      if (ev.key !== 'ArrowDown' && ev.key !== 'ArrowUp') return;
      ev.preventDefault();
      const all = [...$('chartlist').querySelectorAll('button')];
      const n = all[(all.indexOf(b) + (ev.key === 'ArrowDown' ? 1 : -1) + all.length)
        % all.length];
      n.focus(); pickChart(n.dataset.k);
    };
  });
  pickChart(CURCHART);
}

function chartNote(k) {
  for (const [, items] of CHART_GROUPS)
    for (const [key, , note] of items) if (key === k) return note;
  return '';
}
function chartTitle(k) {
  for (const [, items] of CHART_GROUPS)
    for (const [key, t] of items) if (key === k) return t;
  return k;
}

function pickChart(k) {
  if (!k) return;
  CURCHART = k;
  const url = `/chart?name=${encodeURIComponent(k)}&th=${TH}`;
  const img = $('chartimg');
  img.alt = chartTitle(k);
  img.src = url;
  img.onclick = () => openImage(url, chartTitle(k));
  $('charttitle').textContent = chartTitle(k);
  const a = $('dl');
  a.href = url; a.download = RUN + '-' + k + '.png';
  $('chartnote').textContent = chartNote(k).replace('${pos}',
    fmt(DATA.items.filter(u => u.label === 1).length / DATA.items.length, 3));
  $('chartlist').querySelectorAll('button').forEach(b =>
    b.setAttribute('aria-current', String(b.dataset.k === k)));
}

/* ============================================================ 视频推理 */
let POLLING = false;

/* 按段的时序实验没有「一条视频一个分数」的模型，视频推理页签用不了。
   这里把页签内容换成说明，而不是让用户点下去才看到一个看不懂的报错。 */
function buildInferNote() {
  $('m-infer').innerHTML = `<div class="panel">
    <h2>这条 run 不能在网页里给新视频打分</h2>
    <p class="note">这条 run 的模型是按「段」训练的：一行成绩对应一段窗口，没有
      「一条视频一个分数」的模型，所以这里没有可以给新视频打分的东西。</p>
    <p class="note">要看这次训练里每条视频每个采样位置的分数、以及最高分片段是不是真的
      有泥石流，去「样本与错误」页签。</p>
  </div>`;
}

/* 两种入口共用这一段：先把界面切到「推理中」，再提交，再轮询进度。
   req 是一个已经构造好的 fetch 调用（返回 Promise）。 */
function submitInfer(req, hint) {
  $('inferout').innerHTML = '';
  setRunning(true);
  $('progbar').firstElementChild.style.width = '0%';
  $('progbar').setAttribute('aria-valuenow', '0');
  $('prog').textContent = hint;
  req()
    .then(r => r.json().then(j => {
      if (!r.ok || !j.ok) throw new Error(j.error || ('服务端返回 HTTP ' + r.status));
      return j;
    }))
    .then(() => { if (!POLLING) { POLLING = true; poll(); } })
    .catch(e => {
      POLLING = false;
      setRunning(false);
      $('prog').textContent = '';
      $('inferout').innerHTML = `<div class="err"><b>没能开始推理。</b>${esc(e.message || e)}<br>
        多半是服务端进程已经退出或路径不对——回到终端看有没有 traceback。</div>`;
    });
}

/* 上传：文件本体直接当请求体发（不套表单），文件名放头里。 */
$('upbtn').onclick = () => {
  const f = $('vfile').files[0];
  if (!f) { $('vfile').focus(); return; }
  const mb = (f.size / 1048576).toFixed(0);
  submitInfer(() => fetch('/api/upload', {
    method: 'POST',
    headers: {'X-Filename': encodeURIComponent(f.name)},
    body: f,
  }), `正在上传 ${f.name}（${mb} MB）…`);
};

$('runbtn').onclick = () => {
  const v = $('vpath').value.trim();
  if (!v) { $('vpath').focus(); return; }
  submitInfer(() => fetch('/api/infer', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({video: v})}), '正在提交…');
};

$('vpath').onkeydown = e => { if (e.key === 'Enter') $('runbtn').click(); };

/* 推理跑在服务端后台线程里，刷新页面或切走页签都不会停。所以按钮的禁用状态、
   页签上的小圆点、document.title 都由这一处统一管，进页面时也要按当前状态恢复。 */
function setRunning(on, doneMsg) {
  $('runbtn').disabled = on;
  $('upbtn').disabled = on;
  $('runbtn').textContent = on ? '推理中…' : '开始推理';
  $('upbtn').textContent = on ? '推理中…' : '上传并推理';
  $('progbar').hidden = !on;   // 进度条只在跑的时候占位
  document.getElementById('tab-infer').classList.toggle('busy', on);
  // 先把上次留下的 ● 或 ✓ 前缀摘掉再加新的，否则第二次跑会叠成「● ✓ 训练台」。
  const base = document.title.replace(/^[●✓] /, '');
  document.title = (on ? '● ' : doneMsg ? '✓ ' : '') + base;
}

function poll() {
  fetch('/api/infer/status').then(r => r.json()).then(s => {
    if (s.state === 'running') {
      setRunning(true);
      const f = isNum(s.frac) ? s.frac : 0;
      const pct = Math.round(f * 100);
      $('progbar').firstElementChild.style.width = pct + '%';
      $('progbar').setAttribute('aria-valuenow', String(pct));
      // 先写服务端报的真实步骤（如「提特征 12/48」）再写百分比：
      // 阶段之间的百分比是估算的，步骤数才是真的。
      $('prog').textContent = (s.msg ? s.msg + '　' : '') + pct + '%';
      $('progbar').setAttribute('aria-valuetext', (s.msg || '推理中') + '，' + pct + '%');
      setTimeout(poll, 700);
      return;
    }
    POLLING = false;
    $('progbar').hidden = true;
    $('prog').textContent = '';
    if (s.state === 'error') {
      setRunning(false);
      $('inferout').innerHTML = `<div class="err"><b>推理失败。</b>${esc(s.error)}</div>`;
      return;
    }
    if (s.state === 'idle') { setRunning(false); return; }   // 服务重启过，这次任务没了
    setRunning(false, true);
    // 结果要拿 DATA 里的模型参照分做百分位换算。数据还没到就只收尾，
    // 不渲染——等 /api/data 回来时这次任务已经结束了，不会再自动补上。
    if (!DATA) { $('prog').textContent = '推理已完成，但页面数据还没加载好，刷新一次即可看到结果。'; return; }
    showInfer(s.result);
  }).catch(e => {
    POLLING = false;
    $('progbar').hidden = true; $('prog').textContent = '';
    setRunning(false);
    $('inferout').innerHTML = `<div class="err"><b>取进度失败。</b>${esc(e.message || e)}</div>`;
  });
}

/* 进页面先问一次服务端有没有在跑的任务。推理跑在服务端，用户刷新页面或
   关掉标签页都不会让它停，回到本页必须自动接上进度，否则会以为任务丢了
   又去点一次「开始推理」。这一问不依赖 /api/data，所以放在这里单独发。 */
(function resumeIfRunning() {
  fetch('/api/infer/status').then(r => r.json()).then(s => {
    if (s.state === 'running') { POLLING = true; $('progbar').hidden = false; poll(); }
  }).catch(() => {});   // 问不到就当没有任务，不打扰用户
})();

function showInfer(r) {
  const folds = Object.keys(r.per_fold).sort();
  const scores = folds.map(f => r.per_fold[f].unit_score);
  const votes = scores.filter(v => v >= 0.5).length;
  // B2：主显示逐头分数与票数；单一数字先用同折百分位换算再平均（各折模型尺度不同）
  const norms = folds.map(f => rankNorm(r.per_fold[f].unit_score, DATA.fold_ref[f]));
  const agg = avg(norms);
  const cs = r.per_fold[folds[0]].clip_scores;
  const samplingText = r.sampling === 'v3'
    ? `每隔 ${r.frame_stride} 帧取一张（约 0.5 秒一张），每个片段共 16 帧`
    : '连续取 16 帧作为一个片段';
  const sourceRecord = r.transcode_record
    ? `<b>来源与处理记录</b><span><code style="overflow-wrap:anywhere">${esc(r.transcode_record)}</code>
      （记录原视频和转码后视频的路径，以及转码检查结果；模型分数在下方。）</span>`
    : '';
  let h = `<div class="warn">这条视频由 5 个模型分别打分：训练时把数据分成 5 份轮流当考试题，
    得到 5 个模型，每个只在其中 4 份数据上训练过。<b>没有单独训练一个「最终模型」</b>。
    分数没有校准成概率，只能横向比较。</div>
    <div class="kv" style="margin:var(--sp-4) 0">
      <b>视频</b><span>${esc(r.video)}</span>
      ${sourceRecord}
      <b>时长</b><span>${r.n_frames} 帧 / ${r.fps} fps / ${r.duration_sec} 秒</span>
      <b>取帧方式</b><span>${samplingText}</span>
      <b>全片低灰度帧</b><span>${r.n_lowstd_frames} 张；选片段前已避开这些帧及其附近</span>
      <b>参与打分的片段</b><span>${r.n_usable} / ${r.n_clips} 个${r.n_excluded
        ? `（另外 ${r.n_excluded} 个落在禁区里，已按训练同款规则剔除，不参与打分）`
        : ''}</span>
      <b>坏帧重抽次数</b><span>${r.retries}</span>
      <b>判成「有泥石流」的模型数</b><span><b style="color:var(--${votes * 2 >= folds.length ? 'danger' : 'accent-text'});
        font-size:var(--fs-h2)">${votes}/${folds.length}</b>（得分超过 0.5 算一票）</span>
      <b>综合分（百分位平均）</b><span>${fmt(agg)}（把每个模型的得分换算成「在它自己那折数据里
        排第几」的百分位再平均；原始分平均 ${fmt(avg(scores))}
        受各折打分松紧不同的影响，仅参考）</span>
    </div>`;
  if (r.n_clips < 48) {
    h += `<div class="warn">这条视频只取到了 ${r.n_clips} 个片段。EXP-014 每条视频固定取 48 个片段，
      所以这里的分数不能与 EXP-014 的成绩直接比较。</div>`;
  }
  if (r.n_excluded) {
    h += `<div class="warn">共 ${r.n_excluded} 个片段落在禁区里（人工标的剪辑区间、
      低灰度帧、时间戳倒步或时间空洞的邻域），已按训练同款规则剔除、不参与打分；
      整条视频的得分只在剩下 ${r.n_usable} 个片段上取最高。</div>`;
  }
  h += '<div class="tw"><table><thead><tr><th class="l">模型</th><th>整条视频得分</th>'
     + '<th>同折百分位</th><th>最高分片段</th></tr></thead><tbody>';
  folds.forEach((f, i) => {
    h += `<tr><td class="l">第 ${f} 折的模型</td><td>${fmt(scores[i])}</td>`
       + `<td>${fmt(norms[i], 3)}</td>`
       + `<td>#${r.per_fold[f].argmax_clip}</td></tr>`;
  });
  h += '</tbody></table></div>';
  h += `<div class="note" style="margin-top:var(--sp-4)">第 ${folds[0]} 折的模型给 ${cs.length}
    个片段的打分（x 轴 = 等间隔抽样的片段序号，不是连续时间）${r.n_excluded
      ? '；灰色矮柱是落在禁区里的片段，已剔除、不参与整条视频得分的取最高' : ''}：</div>
    <div id="ineedle"></div>
    <div class="toolbar" style="margin-top:var(--sp-2)">
      <button class="ghost" id="iprev">‹ 上一片段</button>
      <span class="note" id="iplab"></span>
      <button class="ghost" id="inext">下一片段 ›</button>
      <button class="ghost" id="izoom">放大这 16 帧</button>
    </div>
    <div class="frames" id="ifr"></div>
    <div class="note" id="iftip"></div>`;
  $('inferout').innerHTML = h;

  let k = r.per_fold[folds[0]].argmax_clip;
  // 落在禁区里的片段照常画柱子（灰度低矮柱），但不参与「整条视频得分」的取最高。
  const use = r.usable || cs.map(() => true);
  const draw = () => {
    let svg = `<svg class="needle" viewBox="0 0 ${cs.length * 8} 100"
      preserveAspectRatio="none" role="img" aria-label="片段分数柱状图">`;
    cs.forEach((v, j) => {
      const hh = Math.max(1, v * 94);
      svg += `<rect x="${j * 8 + 1}" y="${98 - hh}" width="6" height="${hh}"
        fill="var(--${!use[j] ? 'viz-baseline' : (j === k ? 'viz-negative' : 'viz-positive')})"
        opacity="${!use[j] ? .35 : (j === k ? 1 : .78)}"><title>片段 #${j}：${v.toFixed(4)}${use[j] ? '' : '（落在禁区里，已剔除，不参与打分）'}</title></rect>`;
    });
    svg += '</svg>';
    $('ineedle').innerHTML = svg;
    const el = $('ineedle').querySelector('svg');
    el.style.cursor = 'pointer';
    el.onclick = ev => {
      const rc = el.getBoundingClientRect();
      k = Math.min(cs.length - 1,
        Math.max(0, Math.floor((ev.clientX - rc.left) / rc.width * cs.length)));
      draw(); frames();
    };
  };
  const flist = () => Array.from({length: 16}, (_, f) => `/inferthumb?c=${k}&f=${f}`);
  const frames = () => {
    const list = flist();
    $('iplab').textContent = `片段 #${k} / ${cs.length}　分数 ${fmt(cs[k])}`;
    $('ifr').innerHTML = list.map((src, f) =>
      `<img loading="lazy" decoding="async" src="${src}" alt="第 ${f + 1} 帧" tabindex="0">`).join('');
    $('ifr').querySelectorAll('img').forEach((im, f) => {
      im.onclick = () => openFramesAt(list, f, `推理结果 · 片段 #${k}`);
      im.onkeydown = ev => {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault(); openFramesAt(list, f, `推理结果 · 片段 #${k}`);
        }
      };
    });
    $('iftip').textContent = `片段 #${k} 的 16 帧。点任意一帧可放大。`;
  };
  $('iprev').onclick = () => { k = Math.max(0, k - 1); draw(); frames(); };
  $('inext').onclick = () => { k = Math.min(cs.length - 1, k + 1); draw(); frames(); };
  $('izoom').onclick = () => openFramesAt(flist(), 0, `推理结果 · 片段 #${k}`);
  draw(); frames();
}
</script></body></html>"""


def _json_safe(obj):
    """把 NaN / ±Inf 换成 None，供 JSON 输出。

    有些指标在样本不足时算不出来（例如 video5 一条正样本都没有，AUC 无定义），
    值是 float('nan')。Python 的 json.dumps 默认会把它写成裸 NaN，而 JSON 标准
    不允许这个字面量，浏览器的 JSON.parse 会拒绝**整个**响应（实测报
    "Unexpected token 'N' ... is not valid JSON"，页面整块空白）。前端把 None
    显示成「—」，所以这里换成 None 即可；allow_nan=False 保证以后再有漏网的
    值直接抛错，而不是静默产出非法 JSON。
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class Handler(BaseHTTPRequestHandler):
    run_dir = None
    data = None
    infer_out = None

    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json; charset=utf-8", code=200, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(_json_safe(obj), ensure_ascii=False,
                              allow_nan=False).encode("utf-8"),
                   "application/json; charset=utf-8", code)

    def _receive_file(self, dest: Path):
        """把请求体按 Content-Length 分块写盘。

        视频可能几百 MB，不能整块读进内存（read() 无参数会把整个 body 拉进内存）。
        """
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0:
            raise IOError("没有请求体")
        dest.parent.mkdir(parents=True, exist_ok=True)
        left = n
        with open(dest, "wb") as fh:
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    raise IOError("连接中断")
                fh.write(chunk)
                left -= len(chunk)

    def _handle_upload(self):
        """浏览器里选一个视频文件，落盘后立刻开始推理。

        前端把文件本体当请求体直接发（application/octet-stream），文件名放
        X-Filename 头——这样服务端不必解析 multipart 表单，也就没有解析出错的空间。
        """
        if JOB.snapshot()["state"] == "running":
            return self._json({"ok": False, "error": "已有一个推理在跑"}, 409)
        if self.data["kind"] not in INFER_LINE:
            return self._json({"ok": False, "error": "这条 run 是按段的时序实验，"
                              "没有「一条视频一个分数」的模型，视频推理页签用不了"}, 400)
        raw = urllib.parse.unquote(self.headers.get("X-Filename", ""))
        name = Path(raw).name                     # 只取文件名，丢掉任何目录部分
        name = "".join(ch for ch in name if ch not in '\\/:*?"<>|').strip()
        if not name:
            name = "upload.mp4"
        dest = self.infer_out / "uploads" / name
        try:
            self._receive_file(dest)
        except (IOError, OSError) as e:
            return self._json({"ok": False, "error": f"上传中断：{e}"}, 400)
        if dest.stat().st_size == 0:
            return self._json({"ok": False, "error": "上传内容为空"}, 400)
        start_infer(str(dest), self.run_dir, self.infer_out,
                    INFER_LINE[self.data["kind"]])
        self._json({"ok": True, "video": str(dest), "bytes": dest.stat().st_size})

    def _send_thumb(self, path: Path):
        """缩略图：按路径缓存，中文路径用 imdecode 读。"""
        if not path.is_file():
            return self._send(b"", "text/plain", 404)
        cache = self.run_dir / ".wb-thumbs"
        key = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:16]
        cp = cache / f"{key}.jpg"
        if cp.is_file():
            return self._send(cp.read_bytes(), "image/jpeg",
                              extra={"Cache-Control": "max-age=86400"})
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return self._send(b"", "text/plain", 404)
        h, w = img.shape[:2]
        if w > THUMB_W:
            img = cv2.resize(img, (THUMB_W, max(1, int(h * THUMB_W / w))),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return self._send(b"", "text/plain", 500)
        cache.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(buf.tobytes())
        self._send(buf.tobytes(), "image/jpeg",
                   extra={"Cache-Control": "max-age=86400"})

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/":
            html = (PAGE.replace("__RUN__", self.data["run"])
                        .replace("__KIND__", self.data["kind"])
                        .replace("__KINDNOTE__", KIND_NOTE.get(
                            self.data["kind"], "时序实验：每行成绩对应一段窗口")))
            return self._send(html.encode("utf-8"), "text/html; charset=utf-8")
        if u.path == "/api/data":
            return self._json(self.data)
        if u.path == "/api/runsets":
            return self._json(load_runsets())
        if u.path == "/thumb":
            if "unit" in q:                       # 按条视频的线：视频 + 片段 + 帧
                uid = q["unit"][0]
                c, f = int(q.get("c", ["0"])[0]), int(q.get("f", ["0"])[0])
                root = Path(self.data.get("frames_root") or EXP008_FRAMES)
                return self._send_thumb(root / uid / f"clip{c:02d}_f{f:02d}.jpg")
            src, p = q["src"][0], q["p"][0]       # 时序线：源 + 相对帧路径
            root = adapters.frames_root_for_source(src)
            return self._send_thumb(root / p) if root else self._send(b"", "text/plain", 404)
        if u.path == "/inferthumb":
            c, f = int(q.get("c", ["0"])[0]), int(q.get("f", ["0"])[0])
            return self._send_thumb(self.infer_out / "frames" / f"clip{c:02d}_f{f:02d}.jpg")
        if u.path == "/chart":
            name = q.get("name", ["fold_auc"])[0]
            th = float(q.get("th", ["0.5"])[0])
            unit_id = q.get("unit", [None])[0]
            try:
                png = render_chart(name, self.data["items"], th, unit_id)
            except Exception as e:
                return self._send(f"渲染失败：{e}".encode("utf-8"), "text/plain", 500)
            return self._send(png, "image/png")
        if u.path == "/api/infer/status":
            return self._json(JOB.snapshot())
        self._send(b"", "text/plain", 404)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/upload":
            return self._handle_upload()
        if path != "/api/infer":
            return self._send(b"", "text/plain", 404)
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json({"ok": False, "error": "请求体不是 JSON"}, 400)
        if self.data["kind"] not in INFER_LINE:
            return self._json({"ok": False, "error": "这条 run 是按段的时序实验，"
                              "没有「一条视频一个分数」的模型，视频推理页签用不了"}, 400)
        video = (body.get("video") or "").strip()
        if not video:
            return self._json({"ok": False, "error": "缺少 video"}, 400)
        if not Path(video).is_file():
            return self._json({"ok": False, "error": f"视频不存在：{video}"}, 400)
        if JOB.snapshot()["state"] == "running":
            return self._json({"ok": False, "error": "已有一个推理在跑"}, 409)
        start_infer(video, self.run_dir, self.infer_out,
                    INFER_LINE[self.data["kind"]])
        self._json({"ok": True})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run 目录（含 predictions.csv 或 .jsonl）")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--caliber", choices=["auto", "strict", "loose", "full"],
                    default="auto", help="EXP-008 线的片段掩码规则；auto = 读 metrics.json")
    ap.add_argument("--folds-json", default="", help="协议三 run 传 iid 折正本")
    args = ap.parse_args()

    run_dir = resolve_path(args.run)
    if not run_dir.is_dir():
        sys.exit(f"run 目录不存在：{run_dir}")
    Handler.run_dir = run_dir
    Handler.data = load_run(run_dir, args.caliber, args.folds_json)
    Handler.infer_out = REPO / ".codex/docs/.temp/infer-video"

    d = Handler.data
    print(f"[台] {d['kind']} 线 · {d['run']} · {len(d['items'])} 条视频 · "
          f"方法 {d['methods']}")
    for m in d["methods"]:
        for f in d["fold_metrics"].get(m, []):
            print(f"     [{m}] 折{f['fold']}: n={f.get('n')} "
                  f"P{f.get('n_pos')}/N{f.get('n_neg')} "
                  f"AUC={f.get('auc')} pred_pos_rate={f.get('pred_pos_rate')}")
        if d["kind"] != "temporal":
            break        # 按条视频的线只有一个方法，不用重复打五遍
    print(f"[台] http://127.0.0.1:{args.port}/  （Ctrl+C 退出；只读，不改任何数据）")
    # Windows 上 SO_REUSEADDR 允许第二个进程抢绑同一端口，请求会随机落到旧进程——
    # 改了代码却看不到效果。关掉它，端口被占就直接报错退出。
    ThreadingHTTPServer.allow_reuse_address = False
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as e:
        sys.exit(f"端口 {args.port} 已被占用（{e}）——先关掉上一个台，或换 --port")
    srv.serve_forever()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    main()
