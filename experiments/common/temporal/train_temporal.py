# -*- coding: utf-8 -*-
"""EXP-006 训练:缓存特征上训三个轻量时序头(LSTM/TCN/单层注意力),与单帧基线、TSN 对照、融合对比。

评估协议(叶海峰 2026-09-17 定,取代单一 85/15 holdout 作正式口径):
- 默认 --protocol cv:组级分层 k 折交叉验证(k=5)。组 = group_id(场次视频组,整组同侧不拆,
  延续「同场次视频不 split」原则);分层 = 组内 pos/neg 段构成。全部数据都用上:
  每组 4 折当教材、恰有 1 折当考题;报告各折与均值±标准差。
- --protocol holdout:清单静态 85/15(train 列),快速迭代用,不作正式成绩。
- 不微调:ConvNeXt 骨干冻结(EXP-005 同权重),只训轻量头。

窗口(=EXP-006 卡):K=16 步长 8,只在段内滑不跨段,每段≤8 窗;类权重 BCE(pos_weight=负/正);
融合:score = a*单帧投票 + (1-a)*最优时序头,a 网格 {0,0.1,...,1} 在当折 val 上选。

指标(叶海峰 2026-09-17:不只 F1):F1 / Precision / Recall / AUC / 预测为正比例(pred_pos_rate,
接近 1.0 说明模型近似「全报正」,F1 会被正样本占比托高)。窗口级与段级都报。
输出:data/results/temporal/<时间戳>/{report.md, metrics.json}(不进 git 权重类产物口径)
用法(仓库根,venv): python experiments/common/temporal/train_temporal.py [--epochs 30] [--folds 5]
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:  # 包方式（python -m experiments.common.temporal.train_temporal）
    from experiments.common.temporal import adapters
    from experiments.common.temporal.adapters import SegmentManifestAdapter
except ImportError:  # 文件方式（python experiments/common/temporal/train_temporal.py）
    import adapters
    from adapters import SegmentManifestAdapter

SEED = 42
try:
    from sklearn.metrics import roc_auc_score

    def auc(y, s):
        return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")
except Exception:
    def auc(y, s):
        return float("nan")


def build_tsn_samples(rows, z, K=16):
    """TSN 对照:每段全跨度均匀抽 K 帧(不滑窗),帧级线性头打分后段内平均(TSN 共识)。
    与滑窗头的区别:采样覆盖整段,无时序编码,共识=平均。"""
    feats, probs, row_id = z["feats"], z["probs"], z["row_id"]
    samples = []
    for ri, r in enumerate(rows):
        idx = np.where(row_id == ri)[0]
        n = len(idx)
        if n == 0:
            continue
        # 始终取 K 个均匀索引;段短于 K 帧时 linspace 产生重复索引作确定性填充
        sel = idx[np.round(np.linspace(0, n - 1, K)).astype(int)]
        samples.append({"split": r["split"], "label": 1 if r["kind"] == "pos" else 0,
                        "seg": f'{r["source"]}/{r["video_id"]}/seg{r["seg_id"]}',
                        "feats": feats[sel]})
    return samples


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def build_windows(adapter):
    """窗口数据全部来自适配器（段→窗口→帧三步在适配器内完成，策略见 EXP006WindowPolicy）。
    返回 (rows, tr, va)：tr/va 为 Window 对象列表（seg_key / feats / label / mean_logit）。"""
    return adapter.rows, list(adapter.windows("train")), list(adapter.windows("val"))


def refit_single_head(adapter):
    """单帧头折内重训（协议变更，叶海峰 2026-09-18 指令，EXP-006 卡 2026-09-23 节）：
    每折只用该折 train 侧帧重训 768→1 逻辑回归（帧标签=段 kind，类权重同卡片口径
    pos_weight=负/正），覆盖缓存 probs 为新头概率——此后窗口 mean_logit、时序头第 769 维
    输入、单帧投票基线全部站在当折单帧模型肩上，单帧/TSN/时序头折内自洽。
    EXP-005 linear_head.pt 退役为历史参照（缓存的旧 probs 被覆盖，磁盘缓存不动）。"""
    from sklearn.linear_model import LogisticRegression
    X, y = [], []
    for s in adapter.single_frames("train"):
        X.append(s.feats)
        y.append(s.label)
    X, y = np.stack(X), np.array(y)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    clf = LogisticRegression(class_weight={0: 1.0, 1: n_neg / max(1, n_pos)},
                             max_iter=2000)
    clf.fit(X, y)
    adapter._cache["probs"] = clf.predict_proba(adapter._cache["feats"])[:, 1].astype(np.float32)
    return clf, {"n_train_frame": len(y), "pos_train_frame": n_pos}


class LSTMHead(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.proj = nn.Linear(768, d)
        self.rnn = nn.LSTM(d + 1, d, batch_first=True)
        self.out = nn.Linear(d, 1)

    def forward(self, x, ml):
        h, _ = self.rnn(torch.cat([self.proj(x), ml.unsqueeze(-1)], -1))
        return self.out(h[:, -1]).squeeze(-1)


class TCNHead(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.proj = nn.Linear(768, d)
        self.convs = nn.ModuleList([nn.Conv1d(d + 1 if i == 0 else d, d, 3, dilation=2 ** i)
                                    for i in range(3)])
        self.out = nn.Linear(d, 1)

    def forward(self, x, ml):
        h = torch.cat([self.proj(x), ml.unsqueeze(-1).expand(-1, x.size(1), -1)], -1).transpose(1, 2)
        for i, c in enumerate(self.convs):
            pad = 2 ** i * 2  # 因果:只补左侧
            h = nn.functional.pad(h, (pad, 0))
            h = torch.relu(c(h))
        return self.out(h[:, :, -1]).squeeze(-1)


class AttnHead(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.proj = nn.Linear(768, d - 1)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=256,
                                           batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, 1)
        self.out = nn.Linear(d, 1)

    def forward(self, x, ml):
        h = torch.cat([self.proj(x), ml.unsqueeze(-1)], -1)
        h = self.enc(h).mean(1)
        return self.out(h).squeeze(-1)


def _cls_metrics(y, s, n_seg=None):
    """F1/P/R/AUC/预测正占比(叶海峰 2026-09-17:指标不只 F1;
    pred_pos_rate≈1 说明近似全报正,F1 会被正样本占比托高)。"""
    y, s = np.asarray(y), np.asarray(s)
    pred = (s >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    out = {"f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4),
           "auc": round(auc(y, s), 4),
           "pred_pos_rate": round(float(pred.mean()), 4)}
    if n_seg is not None:
        out["n_seg"] = int(n_seg)
    return out


def _seg_windows(n_selected: int) -> int:
    """段 -> 窗口数(与 EXP006WindowPolicy 同步:K=16/步长 8/每段≤8 窗,短段补帧后至少 1 窗)。"""
    if n_selected < 16:
        return 1
    return min(8, (n_selected - 16) // 8 + 1)


def stratified_group_folds(rows, n_splits=5, seed=SEED, group_column="group_id"):
    """组级分层 k 折(手写;本机 sklearn 0.24 无 StratifiedGroupKFold)。

    组 = rows 的 group_column 列(协议一=group_id 场次组,协议二=scene_group 场景组),
    整组同侧绝不拆——同场次/同场景不 split 的既定原则。
    分层目标(2026-09-24 叶海峰指正补)= 组内 pos/neg **窗口数**:窗口才是训练的真实单元,
    旧版只按段数分层会折间窗口/帧规模失衡;段数/帧数随 folds JSON 落盘,不进代价。
    贪心:组按(正窗,负窗)降序依次分给「加入后各类窗口计数最接近 total/k」的折;
    同画像组间用 seed 洗牌保证确定性。返回 {组: fold_idx}。"""
    gcnt: dict[str, Counter] = {}
    for r in rows:
        w = _seg_windows(int(r["n_selected"]))
        gcnt.setdefault(r[group_column], Counter())[r["kind"]] += w
    kinds = ("pos", "neg")
    total = Counter()
    for c in gcnt.values():
        total.update(c)
    target = {k: max(1e-9, total[k] / n_splits) for k in kinds}
    rng = random.Random(seed)
    order = sorted(gcnt, key=lambda g: (-gcnt[g]["pos"], -gcnt[g]["neg"], rng.random()))
    stats = [Counter() for _ in range(n_splits)]
    assign = {}
    for g in order:
        best, best_cost = None, None
        for j in range(n_splits):
            cost = max((stats[j][k] + gcnt[g][k]) / target[k] for k in kinds)
            if best_cost is None or cost < best_cost:
                best, best_cost = j, cost
        assign[g] = best
        stats[best].update(gcnt[g])
    return assign


def _dump_window_predictions(adapter, va, score_map, fold_tag, pred_rows):
    """逐窗口预测落盘(训练台/错误分桶数据源,叶海峰 2026-09-24 要求):按 windows("val")
    的生成顺序重放(清单行序×窗口起点序,确定性),把各方法分数贴回组/视频/段/帧元数据。
    每行一窗:fold/组/段/窗口序/帧相对路径列表/各方法分数。TSN 对照为段样本级,不在此列。"""
    K, min_frames = adapter.policy.K, adapter.policy.min_frames
    wi = 0
    for ri, r in enumerate(adapter.rows):
        if r["split"] != "val":
            continue
        raw = adapter._segment_idxs(ri)
        n_raw = len(raw)
        # 与 _pad_to_min 同口径:短段 linspace 重复采样到 min_frames,位置序列即帧下标
        pos = (np.round(np.linspace(0, n_raw - 1, min_frames)).astype(int)
               if n_raw < min_frames else np.arange(n_raw))
        starts = adapter.policy.window_starts(len(pos))
        ff = r["frames"].split(";")
        n_win = len(starts)
        for j, s in enumerate(starts):
            pred_rows.append({
                "fold": fold_tag, "seg_key": adapter.seg_key(r),
                "source": r["source"], "video_id": r["video_id"],
                "group_id": r["group_id"], "scene_group": r.get("scene_group", ""),
                "seg_id": r["seg_id"], "kind": r["kind"],
                "start_sec": r["start_sec"], "end_sec": r["end_sec"],
                "n_selected": r["n_selected"],
                "win_i": j, "n_win": n_win,
                "frame_files": [ff[p] for p in pos[s:s + K]],
                "scores": {m: round(float(sc[wi]), 6) for m, sc in score_map.items()},
            })
            wi += 1
    assert wi == len(va), f"窗口重放 {wi} != val {len(va)},预测与元数据未对齐"


def run_fold(adapter, epochs, allow_no_pos=False, pred_rows=None, fold_tag=""):
    """在 adapter.rows 当前的 split 划分上训练全部方法,返回 {方法: {"window":…, "seg":…, …}}。
    调用前须已把 rows 的 split 字段设好(cv 协议按折改写,holdout 用清单静态列)。
    allow_no_pos:LOSO 留出全负/全正源(如 video5 整段 neg)时 val 可无正样本——此时
    按阈值 0.5 的 F1/AUC 退化(F1=0/AUC=nan),pred_pos_rate 即留出源上的误报率,仍有效;
    选最优 epoch 的 val F1 恒 0 会取到最后一个 epoch,报告里按退化折标注,不作正式比较。"""
    # 协议变更:先折内重训单帧头,再物化窗口(mean_logit/第 769 维都换成当折单帧头)
    _, refit_meta = refit_single_head(adapter)
    tr = list(adapter.windows("train"))
    va = list(adapter.windows("val"))
    ytr = np.array([w.label for w in tr])
    yva = np.array([w.label for w in va])
    if ytr.sum() == 0 or (yva.sum() == 0 and not allow_no_pos):
        sys.exit("某折正窗口为空,先检查组级分层")

    seg_kind = {adapter.seg_key(r): r["kind"] for r in adapter.rows}

    def frame_metrics():
        """帧级指标(单帧头,折内重训后):val 侧帧的正类 F1/P/R/AUC/预测正占比(H1/H2 对答案用)。"""
        probs = adapter._cache["probs"]
        yf, pf = [], []
        for ri, r in enumerate(adapter.rows):
            if r["split"] != "val":
                continue
            idx = np.where(adapter._cache["row_id"] == ri)[0]
            yf.extend(1 if r["kind"] == "pos" else 0 for _ in idx)
            pf.extend(probs[idx].tolist())
        return _cls_metrics(np.array(yf), np.array(pf)) if yf else {}

    def seg_metrics(scores_va):
        """段级:窗口分按段平均后过同一套指标(段 = 评估原子)。"""
        segs = {}
        for w, s in zip(va, scores_va):
            segs.setdefault(w.seg_key, []).append(float(s))
        ys, ss = [], []
        for g, arr in segs.items():
            ys.append(1 if seg_kind[g] == "pos" else 0)
            ss.append(float(np.mean(arr)))
        return _cls_metrics(np.array(ys), np.array(ss), n_seg=len(ys))

    def batched(wlist, bs=64):
        for i in range(0, len(wlist), bs):
            chunk = wlist[i:i + bs]
            x = torch.tensor(np.stack([w.feats for w in chunk]))
            ml = torch.tensor([w.mean_logit for w in chunk]).unsqueeze(1).expand(-1, x.size(1))
            yield x, ml, torch.tensor([w.label for w in chunk], dtype=torch.float32)

    results = {"_meta": {"n_train_win": len(tr), "n_val_win": len(va),
                         "pos_train": int(ytr.sum()), "pos_val": int(yva.sum()),
                         "degenerate_val": bool(yva.sum() == 0 or yva.sum() == len(yva)),
                         **refit_meta}}

    # a. 单帧基线(折内重训,协议变更 2026-09-18):768→1 逻辑回归只用当折 train 帧训练,
    #    窗口分=当折单帧头帧概率均值;EXP-005 linear_head.pt 不再参与打分(历史参照)
    results["a_单帧投票"] = {"window": _cls_metrics(yva, [w.mean_logit for w in va]),
                             "frame": frame_metrics(),
                             "note": "单帧头折内重训(768→1 逻辑回归,类权重 pos_weight=负/正),"
                                     "窗口分=当折每帧概率的均值"}

    # b. 三个时序头
    heads = {"b_LSTM": LSTMHead, "b_TCN": TCNHead, "b_Attn": AttnHead}
    best_states = {}
    head_scores = {}  # 各头当折 val 逐窗口分数(预测落盘用,与 va 顺序对齐)
    pos_w = torch.tensor(adapter.class_weight("train"))
    for name, cls in heads.items():
        t0 = time.time()
        model = cls()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        best = None
        for ep in range(epochs):
            model.train()
            for x, ml, y in batched(tr):
                opt.zero_grad()
                loss = lossf(model(x, ml), y)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                sv = []
                for x, ml, _ in batched(va):
                    sv.extend(torch.sigmoid(model(x, ml)).numpy().tolist())
            f1v = _cls_metrics(yva, sv)["f1"]
            if best is None or f1v >= best[0]:
                best = (f1v, sv, ep, {k: v.clone() for k, v in model.state_dict().items()})
        results[name] = {"window": _cls_metrics(yva, best[1]), "seg": seg_metrics(best[1]),
                         "best_epoch": best[2], "train_sec": round(time.time() - t0, 1)}
        best_states[name] = best[3]
        head_scores[name] = list(best[1])
        w = results[name]["window"]
        print(f"  {name}: val窗口 F1={w['f1']} P={w['precision']} R={w['recall']} AUC={w['auc']}"
              f" 预测正占比={w['pred_pos_rate']} (epoch {best[2]}, {results[name]['train_sec']}s)"
              f" | 段级 F1={results[name]['seg']['f1']} P={results[name]['seg']['precision']}"
              f" R={results[name]['seg']['recall']} AUC={results[name]['seg']['auc']}")

    # b'. TSN 对照:全段均匀抽 16 帧,帧级线性头 + 段内共识(平均),无时序编码
    t0 = time.time()
    tsz = adapter.cache_arrays  # feats/probs/row_id/labels,供按段均匀抽帧
    tsn_samples = build_tsn_samples(adapter.rows, tsz)
    ts_tr = [s for s in tsn_samples if s["split"] == "train"]
    ts_va = [s for s in tsn_samples if s["split"] == "val"]
    yts = np.array([s["label"] for s in ts_tr])
    yva_ts = np.array([s["label"] for s in ts_va])

    class TSN(nn.Module):
        def __init__(self):
            super().__init__()
            self.frame_head = nn.Linear(768, 1)

        def forward(self, x):  # x [B,K,768]
            return self.frame_head(x).mean(1).squeeze(-1)  # TSN 共识

    tsn = TSN()
    opt = torch.optim.Adam(tsn.parameters(), lr=1e-3)
    lossf2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((yts == 0).sum() / max(1, (yts == 1).sum())))
    tsn_best = None  # 与三个时序头一致:按 val F1 选最优 epoch
    for ep in range(epochs):
        tsn.train()
        for i in range(0, len(ts_tr), 32):
            chunk = ts_tr[i:i + 32]
            x = torch.tensor(np.stack([s["feats"] for s in chunk]))
            y = torch.tensor([s["label"] for s in chunk], dtype=torch.float32)
            opt.zero_grad()
            loss = lossf2(tsn(x), y)
            loss.backward()
            opt.step()
        tsn.eval()
        with torch.no_grad():
            sv_ep = []
            for i in range(0, len(ts_va), 32):
                chunk = ts_va[i:i + 32]
                x = torch.tensor(np.stack([s["feats"] for s in chunk]))
                sv_ep.extend(torch.sigmoid(tsn(x)).numpy().tolist())
        f1v = _cls_metrics(yva_ts, np.array(sv_ep))["f1"]
        if tsn_best is None or f1v >= tsn_best[0]:
            tsn_best = (f1v, sv_ep, ep)
    wts = _cls_metrics(yva_ts, np.array(tsn_best[1]))
    results["b_TSN对照"] = {"window": wts,
                            "note": "全段均匀抽16帧+帧级线性头共识,段样本级评估(非滑窗)",
                            "n_train_seg": len(ts_tr), "n_val_seg": len(ts_va),
                            "train_sec": round(time.time() - t0, 1)}
    print(f"  b_TSN对照: val F1={wts['f1']} P={wts['precision']} R={wts['recall']} AUC={wts['auc']}"
          f" 预测正占比={wts['pred_pos_rate']} ({len(ts_tr)}训练段/{len(ts_va)}验证段)")

    # c. 融合:α*单帧 + (1-α)*最优时序头,α 在当折 val 网格选(按窗口 F1)
    best_head = max(heads, key=lambda n: results[n]["window"]["f1"])
    model = heads[best_head]()
    model.load_state_dict(best_states[best_head])
    model.eval()
    with torch.no_grad():
        sv_best = []
        for x, ml, _ in batched(va):
            sv_best.extend(torch.sigmoid(model(x, ml)).numpy().tolist())
    sv_best = np.array(sv_best)
    base_arr = np.array([w.mean_logit for w in va])
    best = None
    fusion = {}
    for al in [i / 10 for i in range(11)]:
        m = _cls_metrics(yva, al * base_arr + (1 - al) * sv_best)
        fusion[f"alpha={al}"] = m
        if best is None or m["f1"] > best[1]["f1"]:
            best = (al, m)
    results["c_融合"] = {"best_alpha": best[0], "window": best[1], "with_head": best_head}

    if pred_rows is not None:
        score_map = {"a_单帧投票": [float(w.mean_logit) for w in va],
                     "b_LSTM": head_scores["b_LSTM"], "b_TCN": head_scores["b_TCN"],
                     "b_Attn": head_scores["b_Attn"],
                     "c_融合": list(best[0] * base_arr + (1 - best[0]) * sv_best)}
        _dump_window_predictions(adapter, va, score_map, fold_tag, pred_rows)
    return results, fusion


def _aggregate(per_fold):
    """各折指标 -> 均值±标准差(忽略 nan;AUC 单折退化时为 nan 不计入)。"""
    out = {}
    methods = [k for k in per_fold[0][1] if k != "_meta"]
    for m in methods:
        entry = {}
        for level in ("window", "seg", "frame"):
            dicts = [d[m][level] for _, d in per_fold if level in d[m]]
            if not dicts:
                continue
            agg = {}
            for k in dicts[0]:
                vals = [d[k] for d in dicts if k in d and d[k] == d[k]]
                if not vals:
                    agg[k] = {"mean": float("nan"), "std": float("nan")}
                else:
                    agg[k] = {"mean": round(float(np.mean(vals)), 4),
                              "std": round(float(np.std(vals)), 4)}
            entry[level] = agg
        extra = {k: [d[m].get(k) for _, d in per_fold if k in d[m]]
                 for k in ("best_alpha", "with_head") if any(k in d[m] for _, d in per_fold)}
        entry.update({k: v for k, v in extra.items() if v})
        out[m] = entry
    return out


def _dump_splits(assign, rows, gcol, tag, folds, manifest_str):
    """折划分落盘(叶海峰 2026-09-24 拍板:data/metadata/splits/ 是划分清单的家):
    folds JSON = 正本(含每折 val 组清单与段/窗/帧规模);另写 split_checks 兼容的每折
    train/val 清单到 .codex/docs/.temp/splits-check/ 并现场校验无组重叠。"""
    repo = Path(__file__).resolve().parents[3]
    # 留一组法只给含正样本的组分折,全负组(45 个)永远在 train 侧——assign.get(..., -1)
    # 落进末尾丢弃桶,不进任何折的考卷,但保留在 per/set(per) 里进 train 清单
    gsets = [set() for _ in range(folds + 1)]
    for r in rows:
        gsets[assign.get(r[gcol], -1)].add(r[gcol])
    per: dict[str, Counter] = {}
    for r in rows:
        c = per.setdefault(r[gcol], Counter())
        c["seg"] += 1
        c["win"] += _seg_windows(int(r["n_selected"]))
        c["frames"] += int(r["n_selected"])
    payload = {"name": f"EXP-006-{tag}-cv{folds}", "unit": "event_id", "task": "A",
               "meta": {"seed": SEED, "group_column": gcol,
                        "group_meaning": "场次组(机位×日期)" if gcol == "group_id" else "场景组(地址名)",
                        "manifest": manifest_str,
                        "note": "event_id 字段存划分组 ID(组=防泄题原子,整组同侧);train_temporal.py 生成"},
               "folds": [{"fold": i + 1, "val_groups": sorted(gsets[i]),
                          "val_segments": sum(per[g]["seg"] for g in gsets[i]),
                          "val_windows": sum(per[g]["win"] for g in gsets[i]),
                          "val_frames": sum(per[g]["frames"] for g in gsets[i])}
                         for i in range(folds)]}
    out = repo / "data" / "metadata" / "splits" / f"EXP-006-{tag}-folds.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    chk = repo / ".codex" / "docs" / ".temp" / "splits-check"
    chk.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for i in range(folds):
        m = {"name": f"EXP-006-{tag}-fold{i+1}", "unit": "event_id", "task": "A",
             "splits": {"train": [{"video_id": g, "label_origin": "real"}
                                  for g in sorted(set(per) - gsets[i])],
                        "val": [{"video_id": g, "label_origin": "real"} for g in sorted(gsets[i])],
                        "test": []}}
        p = chk / f"EXP-006-{tag}-fold{i+1}.json"
        p.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        r = subprocess.run([sys.executable, "-m", "experiments.splits.code.split_checks", str(p)],
                           cwd=str(repo), capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        ok = r.returncode == 0
        all_ok = all_ok and ok
        print(f"[splits] {tag} fold{i+1} 校验: {r.stdout.strip() or r.stderr.strip()}")
    print(f"[splits] 折划分正本 -> {out}"
          + ("(split_checks 全部通过)" if all_ok else "(有折未通过,见上方输出)"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--protocol", choices=["cv", "holdout"], default="cv",
                    help="cv=组级分层k折(默认,正式口径,每组恰考一次);"
                         "holdout=清单静态85/15(快速迭代用,不作正式成绩)")
    ap.add_argument("--group-column", choices=["group_id", "scene_group"], default="group_id",
                    help="双协议(叶海峰 2026-09-24 定,分别跑):group_id=协议一场次组"
                         "(机位×日期);scene_group=协议二场景组(地址名,同机位跨日期同侧)")
    ap.add_argument("--sufficiency-check", action="store_true", default=False,
                    help="整源留出附加检查,默认不跑(叶海峰 2026-09-24:训练集几乎无正样本时"
                         "训出的模型无效,该设置只有数据依赖论证的引用价值)")
    ap.add_argument("--logo", action="store_true", default=False,
                    help="留一组法(叶海峰 2026-09-24 批):仅协议二,每个含正样本的场景组"
                         "轮留当考卷、其余全练,每个场景得一条独立成绩——消除『大组独占一折』"
                         "的折间波动,直接回答『哪个场景好认哪个翻车』")
    ap.add_argument("--dump-predictions", action="store_true", default=False,
                    help="逐窗口预测落盘 predictions.jsonl(训练台/错误分桶数据源,"
                         "叶海峰 2026-09-24 要求:要能看到识别错误的 组/视频/段/窗口/帧)")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    pred_rows = [] if args.dump_predictions else None

    adapter = SegmentManifestAdapter()
    src_cnt = Counter(r["source"] for r in adapter.rows)
    grp_cnt = len({r[args.group_column] for r in adapter.rows})
    print(f"清单: {len(adapter.rows)} 段({dict(src_cnt)}) / {grp_cnt} 组({args.group_column})"
          f" | 协议: {args.protocol}"
          + (f" k={args.folds}" if args.protocol == "cv" else " (静态 85/15)"))

    per_fold = []
    loso = []
    if args.logo:
        if args.group_column != "scene_group":
            sys.exit("--logo 只用于协议二(--group-column scene_group):场景组整组同侧,"
                     "留一组=一个场景整场当考卷;场次组跨日期碎片多,留一组无意义")
        pos_groups = sorted({r["scene_group"] for r in adapter.rows if r["kind"] == "pos"})
        print(f"留一组法: {len(pos_groups)} 个含正样本场景组,每组轮留当考卷")
        assign = {g: i for i, g in enumerate(pos_groups)}
        _dump_splits(assign, adapter.rows, args.group_column, "scene-logo",
                     len(pos_groups), str(adapter.manifest_file))
        for gi, g in enumerate(pos_groups):
            for r in adapter.rows:
                r["split"] = "val" if r["scene_group"] == g else "train"
            va_rows = [r for r in adapter.rows if r["split"] == "val"]
            w_pos = sum(_seg_windows(int(r["n_selected"])) for r in va_rows if r["kind"] == "pos")
            w_neg = sum(_seg_windows(int(r["n_selected"])) for r in va_rows if r["kind"] != "pos")
            print(f"== 留一组 {gi + 1}/{len(pos_groups)}: {g}"
                  f"(val {len(va_rows)} 段/{w_pos}+{w_neg} 窗) ==")
            res, _ = run_fold(adapter, args.epochs, pred_rows=pred_rows, fold_tag=g)
            per_fold.append((g, res))
    elif args.protocol == "holdout":
        res, _ = run_fold(adapter, args.epochs, pred_rows=pred_rows, fold_tag="holdout")
        per_fold.append(("holdout", res))
    else:
        assign = stratified_group_folds(adapter.rows, args.folds, SEED, args.group_column)
        splits_json = _dump_splits(assign, adapter.rows, args.group_column,
                                   "session" if args.group_column == "group_id" else "scene",
                                   args.folds, str(adapter.manifest_file))
        for fi in range(args.folds):
            for r in adapter.rows:
                r["split"] = "val" if assign[r[args.group_column]] == fi else "train"
            va = [r for r in adapter.rows if r["split"] == "val"]
            n_g = len({r[args.group_column] for r in va})
            w_pos = sum(_seg_windows(int(r["n_selected"])) for r in va if r["kind"] == "pos")
            w_neg = sum(_seg_windows(int(r["n_selected"])) for r in va if r["kind"] == "neg")
            print(f"== 折 {fi + 1}/{args.folds}: val {n_g} 组/{len(va)} 段"
                  f"/{w_pos}+{w_neg} 窗(正+负)/{sum(int(r['n_selected']) for r in va)} 帧 ==")
            res, _ = run_fold(adapter, args.epochs, pred_rows=pred_rows,
                              fold_tag=f"fold{fi + 1}")
            per_fold.append((f"fold{fi + 1}", res))

        # 整源留出(2026-09-24 叶海峰质疑后默认停跑):抽走占主导源后训练集几乎无正样本,
        # 训出的模型本身无效,成绩只有「数据不可替代」的引用价值且属稻草人设置;
        # 数据依赖论证改用场景级成绩(没见过的机位 AUC)。需要引用时 --sufficiency-check 单独跑。
        if args.sufficiency_check:
            for src in sorted({r["source"] for r in adapter.rows}):
                for r in adapter.rows:
                    r["split"] = "val" if r["source"] == src else "train"
                n_va = sum(1 for r in adapter.rows if r["split"] == "val")
                print(f"== 数据充分性检查: 留出 {src}(val {n_va} 段)==")
                res, _ = run_fold(adapter, args.epochs, allow_no_pos=True,
                                  pred_rows=pred_rows, fold_tag=f"loso:{src}")
                loso.append({"left_out": src, **res})

    summary = _aggregate(per_fold)

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_id = os.environ.get("DF_RUN_ID") or ts
    out = adapters.RUN_ROOT / run_id  # 权重/报告不进 data/
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(adapter.manifest_file, out / adapter.manifest_file.name)  # 段清单副本随报告存档
    payload = {"config": {"protocol": args.protocol, "folds": args.folds if args.protocol == "cv" else 1,
                          "group_column": args.group_column,
                          "epochs": args.epochs, "seed": SEED,
                          "protocol_change": "单帧头折内重训(叶海峰 2026-09-18 指令);数据版本 2026-09-24"
                                             "(v4/v5 地址名映射+D12 拆两场景;v2 每段一组;deflow 每视频一组;"
                                             "折分层改窗口数),见 EXP-006 卡",
                          "manifest": str(adapter.manifest_file), "cache": str(adapter.cache_file)},
               "per_fold": [{"fold": tag, **res} for tag, res in per_fold],
               "summary": summary}
    if loso:
        payload["loso"] = loso
    (out / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    if pred_rows:
        (out / "predictions.jsonl").write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in pred_rows),
            encoding="utf-8")
        print(f"逐窗口预测 -> {out / 'predictions.jsonl'}({len(pred_rows)} 窗)")
    pol = adapter.policy
    lines = [f"# EXP-006 报告({args.protocol} 协议"
             + (f",k={args.folds}" if args.protocol == "cv" else "") + ";val 成绩,无独立 test)", "",
             f"- 窗口 K={pol.K} 步长 {pol.stride} 每段≤{pol.max_win_per_seg}窗 | 清单 {len(adapter.rows)} 段 / "
             f"{grp_cnt} {'场次组' if args.group_column == 'group_id' else '场景组(地址名)'}"
             f" | 不微调(骨干冻结,只训头)",
             "- 协议变更(叶海峰 2026-09-18 指令):单帧头折内重训(768→1 逻辑回归,只用当折 train 帧,"
             "类权重 pos_weight=负/正);EXP-005 linear_head.pt 退役为历史参照;数据版本 2026-09-24"
             "(v4/v5 地址名映射+D12 拆两场景;v2 每段一组;deflow 每视频一组;折分层改窗口数)", ""]
    if args.protocol == "cv" and not args.logo:  # logo 模式每折考卷恰 1 组,规模表无意义
        lines.append("## 各折考卷规模(组级划分的固有波动:组不可拆,大组进哪折哪折大)")
        lines.append("")
        lines.append("| 折 | 考卷组数 | 段 | 窗口(正+负) | 帧 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for fi in range(args.folds):
            va = [r for r in adapter.rows if assign[r[args.group_column]] == fi]
            wp = sum(_seg_windows(int(r["n_selected"])) for r in va if r["kind"] == "pos")
            wn = sum(_seg_windows(int(r["n_selected"])) for r in va if r["kind"] == "neg")
            lines.append(f"| fold{fi + 1} | {len({r[args.group_column] for r in va})} | {len(va)} "
                         f"| {wp}+{wn} | {sum(int(r['n_selected']) for r in va)} |")
        lines.append("")
    lines.append("## 各折均值±标准差(窗口级)")
    lines.append("")
    lines.append("| 方法 | F1 | Precision | Recall | AUC | 预测正占比 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for m, e in summary.items():
        w = e.get("window")
        if not w:
            continue
        fmt = lambda k: f"{w[k]['mean']:.3f}±{w[k]['std']:.3f}"
        lines.append(f"| {m} | {fmt('f1')} | {fmt('precision')} | {fmt('recall')} | "
                     f"{fmt('auc')} | {fmt('pred_pos_rate')} |")
    lines.append("")
    lines.append("## 各折均值±标准差(段级)")
    lines.append("")
    lines.append("| 方法 | F1 | Precision | Recall | AUC | 段数 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for m, e in summary.items():
        s = e.get("seg")
        if not s:
            continue
        fmt = lambda k: f"{s[k]['mean']:.3f}±{s[k]['std']:.3f}"
        lines.append(f"| {m} | {fmt('f1')} | {fmt('precision')} | {fmt('recall')} | "
                     f"{fmt('auc')} | {s['n_seg']['mean']:.0f} |")
    lines.append("")
    lines.append("## 各折均值±标准差(帧级,单帧头折内重训后;假设 H1/H2 对答案用)")
    lines.append("")
    lines.append("| 方法 | F1 | Precision | Recall | AUC | 预测正占比 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for m, e in summary.items():
        f = e.get("frame")
        if not f:
            continue
        fmt = lambda k: f"{f[k]['mean']:.3f}±{f[k]['std']:.3f}"
        lines.append(f"| {m} | {fmt('f1')} | {fmt('precision')} | {fmt('recall')} | "
                     f"{fmt('auc')} | {fmt('pred_pos_rate')} |")
    lines.append("")
    for tag, res in per_fold:
        lines.append(f"## {tag}(train {res['_meta']['n_train_win']} 窗/正{res['_meta']['pos_train']}"
                     f" · val {res['_meta']['n_val_win']} 窗/正{res['_meta']['pos_val']})")
        lines.append("```json")
        lines.append(json.dumps({k: v for k, v in res.items() if k != "_meta"},
                                ensure_ascii=False, indent=1))
        lines.append("```")
        lines.append("")

    if loso:
        lines.append("## 数据充分性检查(留一数据源:其余源训练,留出源整体当 val)")
        lines.append("")
        lines.append("定位(2026-09-24 叶海峰指正):这不是泛化成绩,不作模型能力结论——抽走样本量")
        lines.append("占主导的源后训练集骤减且几乎无正样本,成绩接近随机是设计使然;它回答的是")
        lines.append("「缺这个源练不练得成」,支撑「模型必须在部署沟的数据上训练」的部署主张。")
        lines.append("留出全负/全正源(如 video5 全负、deflow 全正)时按阈值 0.5 的 F1 与 AUC 退化,")
        lines.append("该行预测正占比 = 留出源上的误报率(全负)或漏报补充读数(全正),是唯一有效读数。")
        lines.append("")
        lines.append("| 留出源 | 方法 | 窗口 F1 | 窗口 AUC | 预测正占比 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for entry in loso:
            mark = "(退化折)" if entry["_meta"].get("degenerate_val") else ""
            for m, e in entry.items():
                if m in ("_meta", "left_out") or not isinstance(e, dict) or "window" not in e:
                    continue
                w = e["window"]
                lines.append(f"| {entry['left_out']}{mark} | {m} | {w['f1']:.3f} | "
                             f"{w['auc']:.3f} | {w['pred_pos_rate']:.3f} |")
        lines.append("")
        for entry in loso:
            lines.append(f"### LOSO 留出 {entry['left_out']}(train {entry['_meta']['n_train_win']} 窗"
                         f" · val {entry['_meta']['n_val_win']} 窗)")
            lines.append("```json")
            lines.append(json.dumps({k: v for k, v in entry.items() if k != "_meta"},
                                    ensure_ascii=False, indent=1))
            lines.append("```")
            lines.append("")

    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告 -> {out / 'report.md'}")
    print("\n窗口级均值±标准差:")
    for m, e in summary.items():
        w = e.get("window")
        if w:
            print(f"  {m}: F1={w['f1']['mean']:.3f}±{w['f1']['std']:.3f} "
                  f"P={w['precision']['mean']:.3f} R={w['recall']['mean']:.3f} "
                  f"AUC={w['auc']['mean']:.3f} 预测正占比={w['pred_pos_rate']['mean']:.3f}")


if __name__ == "__main__":
    main()
