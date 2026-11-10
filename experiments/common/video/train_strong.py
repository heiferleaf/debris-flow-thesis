# -*- coding: utf-8 -*-
"""EXP-008 阶段②：强标签对照头训练（同骨干同特征同折，唯一自变量 = 监督信号）。

对照设计（EXP-008 卡第三段 + research-auditor 2026-10-04 决策复核）：
  - 架构与 MIL 头同形：512→128→64→32→1 + 每隐藏层后 50% Dropout；
    末层不接 Sigmoid（BCEWithLogitsLoss 数值更稳；Sigmoid 为单调映射，函数族等价，
    MIL 头的 Sigmoid 是 Wu 原文形态）。
  - 损失：片段级 BCEWithLogitsLoss，pos_weight = 可用负片段数/可用正片段数
    （协议准则 7 精神；strict 口径 ≈ 0.40）。
  - 片段集：--caliber strict（主对照，两臂统一，复核 G2）/ loose（敏感性检查）。
    不可用片段（-1）不进损失、不进单位分数。
  - 评估口径与 train_mil.py 逐项一致：单位分数 = max(可用片段分数)，5 折 CV，
    折内 val 选 best epoch（M2 偏差声明沿用）。
  - 折：默认 npz folds（协议一）；--folds-json 切协议三。

  EXP-012 新增（2026-10-11 卡，签字后实现；不给新参数时与现役行为逐位一致）：
  - --seed（默认 7）、--fold（-1=全折，冒烟只跑一折）；
  - --context-agg gated|selfattn：在冻结特征与四级 MLP 头之间插入整视频上下文
    聚合结构（因子一）；片段级 BCE 监督、单位分数 = strict 掩码内最高分、
    θ=0.5 全部不变；表述只写「整视频上下文聚合」，不写时序建模；
  - --hard-neg-penalty λ：负侧超阈惩罚 L_pen = λ·mean(max(0, 负单位最高片段分−0.5))
    （因子二）；强标签线批级语义：负单位最高分 = 批内被采片段的最大值。

用法（仓库根，venv）：
  python experiments/common/video/train_strong.py \
      --feats data/results/exp008/unitfeats-v2-150/unit-features.npz \
      --caliber strict --tag strong-strict
环境变量：DF_RUNS_ROOT 覆盖 runs 根。
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.paths import resolve_path  # noqa: E402
RUNS_ROOT = resolve_path(os.environ.get("DF_RUNS_ROOT") or REPO / "data/results/exp008")
CLIP_LABELS = REPO / "data/process/EXP-008-units-v2/clip-labels.npz"
SEED = 7
BS = 256


class StrongHead(nn.Module):
    """与 MilHead 同形的四级 MLP（末层无 Sigmoid，配 BCEWithLogitsLoss）。"""

    def __init__(self, in_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class GatedCtx(nn.Module):
    """变体甲：门控注意力池化（gated attention，Ilse et al., ICML 2018；
    与顺序无关，EXP-012 卡预注册首选）。forward_unit 输入一条单位的
    特征 [48,512] 与可用掩码 [48] bool，返回每片段拼接用的附加特征
    [48,att_dim]（各片段同一份视频上下文 c = Σ a_k z_k）。
    """

    def __init__(self, in_dim=512, att_dim=64):
        super().__init__()
        self.V = nn.Linear(in_dim, att_dim)
        self.U = nn.Linear(in_dim, att_dim)
        self.w = nn.Linear(att_dim, 1, bias=False)

    def forward_unit(self, h, mask):
        hm = h * mask.unsqueeze(-1).to(h.dtype)          # 掩码位特征填 0（求和/平均类归约的正确方向）
        z = torch.tanh(self.V(hm)) * torch.sigmoid(self.U(hm))   # [48,att_dim]
        logits = self.w(z).squeeze(-1)                   # [48]
        # softmax 类归约：掩码位 logits 填 -inf，注意力权重恰为 0（无全掩码行，已核对）
        logits = logits.masked_fill(~mask, float("-inf"))
        a = torch.softmax(logits, dim=0)
        c = (a.unsqueeze(-1) * z).sum(dim=0)             # [att_dim]，掩码位贡献为 0
        return c.unsqueeze(0).expand(h.shape[0], -1)     # [48,att_dim]


class SelfAttnCtx(nn.Module):
    """变体乙：单层自注意力编码器（self-attention，Vaswani et al., 2017；
    先降维到 128 再上）。forward_unit 返回每片段各自的上下文增强表示
    [48,d_model]，与冻结特征 concat 后进头。
    """

    def __init__(self, in_dim=512, d_model=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.scale = float(d_model) ** 0.5

    def forward_unit(self, h, mask):
        hm = h * mask.unsqueeze(-1).to(h.dtype)          # 掩码位特征先填 0 再进 proj
        x = self.proj(hm)                                # [48,d_model]
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        logits = q @ k.transpose(0, 1) / self.scale      # [48,48]
        logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))   # 掩码键位 -inf
        a = torch.softmax(logits, dim=-1)
        return self.out(a @ v)                           # [48,d_model]


class StrongNetCtx(nn.Module):
    """带上下文聚合的打分网络（仅 --context-agg 非 none 时使用；
    forward_unit 输入一条单位的 [48,512] 特征与 [48] 掩码，返回 [48] logits）。
    """

    def __init__(self, agg, head):
        super().__init__()
        self.agg = agg
        self.head = head

    def forward_unit(self, Xi, mi):
        extra = self.agg.forward_unit(Xi, mi)            # [48,D_extra]
        return self.head(torch.cat([Xi, extra], dim=-1)).squeeze(-1)     # [48]


CTX_EXTRA_DIM = {"gated": 64, "selfattn": 128}
PENALTY_TAU = 0.5


def hard_neg_penalty(logits, unit_idx, unit_labels, lam, tau=PENALTY_TAU):
    """负侧超阈惩罚 L_pen = λ · mean(max(0, 负单位最高片段分 − τ))（EXP-012 因子二，
    语义同 EXP-011：对批内出现违规的负单位取违规均值，不对全批平均）。

    强标签线批级语义（如实记入 metrics.json 注记）：训练批是片段对采样，
    负单位的「最高片段分」= 批内该单位被采到的可用片段分数的最大值，
    不是全 48 片段的最大值。λ<=0 直接跳过；无违规返回 0 且不消耗随机数。
    """
    if lam <= 0:
        return torch.zeros((), dtype=logits.dtype)
    s = torch.sigmoid(logits)
    viols = []
    for u in torch.unique(unit_idx).tolist():
        if int(unit_labels[u]) != 0:
            continue
        v = torch.clamp(s[unit_idx == u].max() - tau, min=0.0)
        if float(v.item()) > 0.0:                        # 只对违规单位取均值，不对全批平均
            viols.append(v)
    if not viols:
        return torch.zeros((), dtype=logits.dtype)       # 无违规：恰为 0 且不携带梯度
    return lam * torch.stack(viols).mean()


def ctx_batch_logits(net, X, M, ui, ci):
    """聚合路径的批前向：训练批是片段对 (unit, clip)，对批内每个唯一单位
    用其全部 48 行特征过一次聚合器，片段表示 = concat(冻结特征, 上下文)。"""
    extras = {}
    for u in ui.tolist():
        u = int(u)
        if u not in extras:
            extras[u] = net.agg.forward_unit(X[u], M[u])
    reps = [torch.cat([X[int(ui[k]), int(ci[k])], extras[int(ui[k])][int(ci[k])]], dim=-1)
            for k in range(len(ui))]
    return net.head(torch.stack(reps)).squeeze(-1)


def report(name, y, p):
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
    binary = (p >= 0.5).astype(int)
    return {"set": name, "n": int(len(y)), "auc": round(auc, 4),
            "f1": round(f1_score(y, binary), 4) if len(set(y)) > 1 else float("nan"),
            "p": round(precision_score(y, binary, zero_division=0), 4),
            "r": round(recall_score(y, binary, zero_division=0), 4)}


def unit_scores(model, X, mask, idx, context_agg="none"):
    """单位分数 = max(可用片段 sigmoid 分数)。mask: [n,48] bool；idx: 单位下标。"""
    model.eval()
    out = []
    with torch.no_grad():
        for i in idx:
            if context_agg == "none":
                s = torch.sigmoid(model(X[i:i + 1]).squeeze(-1)).squeeze(0)   # [48]
            else:
                s = torch.sigmoid(model.forward_unit(X[i], torch.as_tensor(mask[i])))  # [48]
            out.append(float(s[mask[i]].max()))
    return np.array(out)


def run_fold(feats, unit_labels, clip_label, mask, tr, va, epochs, lr, fold_name,
             seed=SEED, context_agg="none", lam=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va = np.asarray(tr), np.asarray(va)
    # 模型构造：无聚合时与现役逐位同构（同一模块、同一初始化随机数消耗）
    agg = None
    if context_agg == "gated":
        agg = GatedCtx()
        head = StrongHead(512 + CTX_EXTRA_DIM["gated"])
    elif context_agg == "selfattn":
        agg = SelfAttnCtx()
        head = StrongHead(512 + CTX_EXTRA_DIM["selfattn"])
    else:
        head = StrongHead()
    model = head if agg is None else StrongNetCtx(agg, head)
    if agg is not None:
        ctx_params = sum(p.numel() for p in agg.parameters())
        print(f"  [{fold_name}] 上下文聚合 {context_agg}：聚合器参数量 {ctx_params}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(feats, dtype=torch.float32)
    M = torch.tensor(mask)                                    # [n,48] bool
    unit_labels = np.asarray(unit_labels)
    # 片段训练集：(unit_idx, clip_idx) 仅可用片段
    pairs = [(i, j) for i in tr for j in range(len(clip_label[i]))
             if clip_label[i][j] >= 0]
    y_clip = torch.tensor([clip_label[i][j] for i, j in pairs], dtype=torch.float32)
    idx_ui = torch.tensor([i for i, _ in pairs])
    idx_ci = torch.tensor([j for _, j in pairs])
    n_pos = int((y_clip == 1).sum())
    n_neg = int((y_clip == 0).sum())
    pos_weight = torch.tensor(n_neg / max(n_pos, 1))
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  [{fold_name}] 可用片段 {len(pairs)}（pos {n_pos}/neg {n_neg}，"
          f"pos_weight={pos_weight:.2f}）")
    best_auc, best_state, best_r, best_ep = -1.0, None, None, -1
    t0 = time.time()
    hist = []                                                 # 逐轮 BCE 与惩罚项分开记录（卡改动 3）
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(pairs))
        losses, bces, pens = [], [], []
        for b in range(0, len(perm), BS):
            sel = perm[b:b + BS]
            sel_ui = idx_ui[sel]
            if agg is None:
                logits = model(X[idx_ui[sel], idx_ci[sel]]).squeeze(-1)
            else:
                logits = ctx_batch_logits(model, X, M, sel_ui, idx_ci[sel])
            bce = crit(logits, y_clip[sel])
            if lam > 0:
                pen = hard_neg_penalty(logits, sel_ui, unit_labels, lam)
                loss = bce + pen
                pens.append(float(pen.item()))
            else:
                loss = bce                                    # λ=0 完全跳过惩罚路径
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{fold_name} ep{ep}: 损失非有限（{loss.item()}）——数值异常，"
                    "拒绝静默继续（2026-10-05 掩码 min 项失效事故后强制）")
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            bces.append(bce.item())
        ep_bce = float(np.mean(bces))
        ep_pen = float(np.mean(pens)) if pens else 0.0
        hist.append({"ep": ep, "bce": round(ep_bce, 6), "pen": round(ep_pen, 6)})
        p_va = unit_scores(model, X, M, va, context_agg)
        r_va = report("va" + fold_name, unit_labels[va], p_va)
        if not np.isnan(r_va["auc"]) and r_va["auc"] > best_auc:
            best_auc, best_state, best_r, best_ep = r_va["auc"], \
                {k: v.clone() for k, v in model.state_dict().items()}, r_va, ep
        pen_txt = f" pen={ep_pen:.4f}" if pens else ""
        print(f"  [{fold_name}] ep{ep} loss={np.mean(losses):.4f}{pen_txt} "
              f"vaAUC={r_va['auc']:.3f} vaF1={r_va['f1']:.3f}"
              f"{' *' if best_ep == ep else ''} ({time.time()-t0:.0f}s)")
    # 末轮模型对训练侧负单位算 masked max sigmoid 违规统计（λ 量级推导依据；
    # 纯评估 no_grad，不消耗随机数、不影响训练与预测）
    model.eval()
    viol = []
    with torch.no_grad():
        for i in tr:
            if int(unit_labels[i]) != 0:
                continue
            if agg is None:
                s = torch.sigmoid(model(X[i:i + 1]).squeeze(-1)).squeeze(0)
            else:
                s = torch.sigmoid(model.forward_unit(X[i], M[i]))
            viol.append(max(0.0, float(s[M[i]].max()) - PENALTY_TAU))
    viol = np.array(viol)
    if len(viol):
        neg_stats = {"n_neg_tr": int(len(viol)),
                     "viol_mean_violators": round(float(viol[viol > 0].mean()), 6)
                     if (viol > 0).any() else 0.0,
                     "viol_mean_all": round(float(viol.mean()), 6),
                     "viol_frac": round(float((viol > 0).mean()), 4)}
    else:
        neg_stats = {"n_neg_tr": 0, "viol_mean_violators": 0.0,
                     "viol_mean_all": 0.0, "viol_frac": 0.0}
    preds = {}
    if best_state is not None:
        model.load_state_dict(best_state)
        p_va = unit_scores(model, X, M, va, context_agg)
        preds = {i: float(s) for i, s in zip(va, p_va)}
    bce_last3 = round(float(np.mean([h["bce"] for h in hist[-3:]])), 6) if hist else float("nan")
    extra = {"hist": hist, "bce_last3": bce_last3, "neg_stats": neg_stats}
    if agg is not None:
        extra["ctx_params"] = ctx_params
    return best_state, best_r, best_ep, preds, extra


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--clip-labels", dest="clip_labels", default=str(CLIP_LABELS),
                    help="clip-labels.npz 路径；默认全量正本，冒烟时指向 tiny 夹具")
    ap.add_argument("--caliber", choices=["strict", "loose"], default="strict")
    ap.add_argument("--folds-json", default="",
                    help="协议三单位级折正本；不给则用 npz folds（协议一）")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：不划分折，全单位训练并自评（只验管线，2026-10-05 补）")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7,
                    help="随机种子（默认 7 与现役行为逐位一致；EXP-012 矩阵 7/8/9）")
    ap.add_argument("--context-agg", dest="context_agg",
                    choices=["none", "gated", "selfattn"], default="none",
                    help="上下文聚合结构（EXP-012 因子一；none=现役逐片段独立打分）")
    ap.add_argument("--hard-neg-penalty", dest="hard_neg_penalty", type=float, default=0.0,
                    help="负侧超阈惩罚 λ（EXP-012 因子二；0=现役，惩罚项完全跳过）")
    ap.add_argument("--fold", type=int, default=-1,
                    help="只跑第 fold 折（-1=全折；冒烟用）")
    ap.add_argument("--tag", default="strong-smoke")
    args = ap.parse_args()
    for name in ('feats', 'clip_labels', 'folds_json'):
        value = getattr(args, name)
        if value:
            setattr(args, name, str(resolve_path(value, must_exist=True)))

    d = np.load(args.feats, allow_pickle=True)
    feats, unit_labels = d["feats"], d["labels"].astype(int)
    unit_ids = d["unit_ids"]
    n = len(unit_labels)
    cl = np.load(args.clip_labels, allow_pickle=True)
    assert list(cl["unit_ids"]) == list(unit_ids), "clip-labels 与 npz 行序不一致"
    clip_label = cl[args.caliber].astype(int)                 # [n,48] -1/0/1
    mask = clip_label >= 0
    print(f"[STRONG] 口径 {args.caliber}：可用片段 {int(mask.sum())}/{mask.size} "
          f"（{100*mask.mean():.1f}%）")

    if args.smoke:
        # 冒烟：不划分折，全单位训练并自评，只验管线跑得通、损失有限（2026-10-05 补，
        # 依据 research-auditor 复核 W2：本脚本此前从未单独冒烟过）
        idx = list(range(n))
        state, best_r, best_ep, preds, extra = run_fold(
            feats, unit_labels, clip_label, mask, idx, idx,
            args.epochs, args.lr, "smoke", seed=args.seed,
            context_agg=args.context_agg, lam=args.hard_neg_penalty)
        folds_src = "smoke(全单位自评，不划分折)"
        bests, all_preds = {"smoke": state}, {"smoke": preds}
        runs = [{"fold": "smoke", "name": "smoke", "n": n,
                 "n_pos": int(unit_labels.sum()),
                 "n_neg": int(n - unit_labels.sum()),
                 "auc": best_r["auc"], "f1": best_r["f1"], "best_epoch": best_ep,
                 "bce_last3": extra["bce_last3"], "neg_stats": extra["neg_stats"]}]
        fold_extras = [extra]
    else:
        if args.folds_json:
            fj = json.load(open(args.folds_json, encoding="utf-8"))
            u2f = {u: k["fold"] for k in fj["folds"] for u in k["val_units"]}
            folds = np.array([u2f[u] for u in unit_ids])
            folds_src = Path(args.folds_json).name
        else:
            folds = d["folds"]
            folds_src = "npz(协议一 EXP-008-units-folds.json)"

        fold_nums = sorted(set(int(x) for x in folds))
        if args.fold >= 0:
            if args.fold not in fold_nums:
                raise SystemExit(f"--fold {args.fold} 不在折清单 {fold_nums} 中")
            fold_list = [args.fold]
        else:
            fold_list = fold_nums
        runs, bests, all_preds, fold_extras = [], {}, {}, []
        for f in fold_list:
            va = [i for i in range(n) if int(folds[i]) == f]
            tr = [i for i in range(n) if int(folds[i]) != f]
            state, best_r, best_ep, preds, extra = run_fold(
                feats, unit_labels, clip_label, mask, tr, va, args.epochs, args.lr,
                f"f{f}", seed=args.seed, context_agg=args.context_agg,
                lam=args.hard_neg_penalty)
            bests[f] = state
            all_preds[f] = preds
            fold_extras.append(extra)
            runs.append({"fold": f, "name": f"f{f}", "n": len(va),
                         "n_pos": int(unit_labels[va].sum()),
                         "n_neg": int(len(va) - unit_labels[va].sum()),
                         "auc": best_r["auc"], "f1": best_r["f1"], "best_epoch": best_ep,
                         "bce_last3": extra["bce_last3"], "neg_stats": extra["neg_stats"]})
    aucs = [r["auc"] for r in runs if not np.isnan(r["auc"])]
    out_dir = RUNS_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx_params = next((e["ctx_params"] for e in fold_extras if "ctx_params" in e), 0)
    summary = {"script": "train_strong.py", "caliber": args.caliber,
               "n_unit": n, "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
               "context_agg": args.context_agg,
               "hard_neg_penalty": args.hard_neg_penalty,
               "penalty_tau": PENALTY_TAU,
               "ctx_params": ctx_params,
               "neg_max_batch_note": "负视频最高片段分 = 训练批内该负单位被采到的"
                                     "可用片段分数的最大值（批级语义，非全 48 片段）",
               "folds_source": folds_src,
               "feats": str(Path(args.feats).resolve()),
               "clip_labels": str(Path(args.clip_labels).resolve()), "mean_auc": round(float(np.mean(aucs)), 4),
               "std_auc": round(float(np.std(aucs)), 4), "folds": runs}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    for f, state in bests.items():
        torch.save(state, out_dir / f"strong_f{f}.pt")
    with open(out_dir / "predictions.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "unit_id", "label", "score"])
        for fold, preds in all_preds.items():
            for i, s in preds.items():
                # 写全精度分数（不四舍五入）；metrics.json 的 mean_auc 同为全精度，为权威引用口径
                w.writerow([fold, unit_ids[i], int(unit_labels[i]), float(s)])
    print(f"[STRONG] 均值 AUC {summary['mean_auc']} ± {summary['std_auc']} "
          f"-> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
