# -*- coding: utf-8 -*-
"""EXP-008 R2 MIL 头训练：Wu 四级检测 MLP + 三项 MIL 损失（Eq.2/3/4/5 逐字）。

输入：extract_feats.py 产出的 unit-features.npz（[n_unit, 48, 512]）。
头：512→128→64→32→1，末段 Linear(32,1) 过 Sigmoid（分数 φ∈[0,1]），每隐藏层后 50% Dropout。
损失（逐字对照 Wu et al. 2025 原文 Eq.2/3/4/5，偏差 D6：叶海峰 2026-10-03 拍板忠实照跑）：
  L1 = max{0, 1 - (max_d - min_d - (max_n - min_n))}  正/负单位各 48 片段分数峰谷差，margin=1
  L2 = Σ_{k=1}^{47}(φ_k - φ_{k+1})²                   正单位内相邻片段平滑
  L3 = Σ_{k=1}^{48} φ_k                               正单位分数加总（稀疏）
  L  = L1 + λt·L2 + λs·L3，λt=λs=8e-5
训练组织：每 epoch 正单位全体洗牌后按 bs 无放回分批遍历（每正单位恰好进一个 batch），
      负单位洗牌后循环补齐到同批数；末批负单位取与正单位同长度（审计 C1 修复）。
评估：单位分数 = max(48 片段分数)，eval 模式（Dropout 关闭，审计 M1 配套）；
      单位级 AUC/F1/P/R，5 折 CV，折默认 = npz folds 字段（协议一正本
      data/metadata/splits/EXP-008-units-folds.json，2026-10-04 单位级分层重划）；
      --folds-json 切协议三（同场次混分参照，独立正本）。

用法（仓库根，venv）：
  python experiments/common/video/train_mil.py --feats runs/unitfeats-smoke/unit-features.npz \
      --smoke --epochs 8      # 冒烟：不划分折，全单位训练自评，只验管线
  python experiments/common/video/train_mil.py --feats runs/<tag>/unit-features.npz
环境变量：DF_RUNS_ROOT 覆盖 runs 根。
EXP-011（2026-10-11 签字）：新增 --seed（默认 7 = 旧硬编码，默认路径逐位同构）与
--hard-neg-penalty λ（默认 0 = 不启用，损失路径逐位同构）；λ>0 时在三项损失上叠加
L_pen = λ·mean(max(0, 负单位 masked max − 0.5))，实现见 hard_neg_penalty()，
观察量 negMax（训练侧负视频最高分批均值）为 W1 升档规则观察对象。
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
SEED = 7
LAM_T = 8e-5
LAM_S = 8e-5
MARGIN = 1.0
PEN_TAU = 0.5     # EXP-011 负样本视频超阈惩罚阈值 τ：判定阈值本身，先验冻结不扫


class MilHead(nn.Module):
    """Wu 四级检测 MLP（按我们骨干维度缩放）：512→128→64→32→1 + Sigmoid。
    每隐藏层后接 50% Dropout（Wu 原文 Fig.6，学习手册 §检测网络）。"""

    def __init__(self, in_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):               # x: [*, 512] -> [*, 1]
        return self.net(x)


def mil_loss(pos: torch.Tensor, neg: torch.Tensor, lam_t=LAM_T, lam_s=LAM_S):
    """pos/neg: [bs, 48] 各自单位的 48 片段分数（sigmoid 后 [0,1]）。
    返回 (总损失, L1, L2, L3) 标量。L2/L3 对每个正单位算再取 batch 平均。"""
    maxp, minp = pos.max(dim=1).values, pos.min(dim=1).values
    maxn, minn = neg.max(dim=1).values, neg.min(dim=1).values
    l1 = torch.max(torch.zeros_like(maxp),
                   1.0 - ((maxp - minp) - (maxn - minn))).mean()
    l2 = ((pos[:, 1:] - pos[:, :-1]) ** 2).sum(dim=1).mean()
    l3 = pos.sum(dim=1).mean()
    return l1 + lam_t * l2 + lam_s * l3, l1, l2, l3


def masked_max_min(scores: torch.Tensor, mask: torch.Tensor):
    """按掩码求每行的最大/最小分数，结果只落在可用片段上（mask=True 为可用）。

    填充方向必须与归约方向一致：求 max 把被排除位（~mask）填 -inf，
    求 min 把被排除位填 +inf。2026-10-05 修复：min 曾误写成
    `masked_fill(mask, +inf)`（填的是可用位），使 min 取到不可用片段；
    35 个「48 片段全可用」单位的 min 因此为 +inf、峰谷差为 -inf，L1 输出
    inf/NaN。该路径梯度仍是有限值（NaN 元素经 torch.max(0,·) 反向传播回传
    有限梯度），训练不报错、照常产出成绩，但目标函数已失效——详见 run-log
    「阶段②第一步」2026-10-05 更正节。回归测试：
    experiments/tests/test_mask_reduction.py。
    """
    mx = scores.masked_fill(~mask, float("-inf")).max(dim=1).values
    mn = scores.masked_fill(~mask, float("inf")).min(dim=1).values
    return mx, mn


def hard_neg_penalty(neg_scores: torch.Tensor, neg_mask: torch.Tensor,
                     tau=PEN_TAU, lam=1.0):
    """EXP-011 负样本视频超阈惩罚：L_pen = λ · mean( max(0, 最高片段分 − τ) )。

    neg_scores: [B,48] sigmoid 后片段分数；neg_mask: [B,48] bool（True = 可用片段）。
    最高片段分 = strict 掩码下的 masked max（~mask 填 -inf，与 masked_max_min 的
    max 分支同语义）。归一化只对违规行（最高分 > τ）取均值，不对全批平均；
    批内无违规时返回 0 标量张量（不进计算图、不消耗随机数）。
    返回 (penalty 标量张量, 违规行数 int)。损失输入只有视频级标签带来的
    「该行是否负单位」，不读场次组归属、不读片段级强标签。
    """
    mx = neg_scores.masked_fill(~neg_mask, float("-inf")).max(dim=1).values
    viol = torch.clamp(mx - tau, min=0)              # [B] 每行违规量，阈下为 0
    rows = viol > 0
    n_viol = int(rows.sum())
    if not rows.any():
        return torch.zeros(()), 0
    return lam * viol[rows].mean(), n_viol


def report(name, y, p):
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
    binary = (p >= 0.5).astype(int)
    f1 = f1_score(y, binary) if len(set(y)) > 1 else float("nan")
    return {"set": name, "n": int(len(y)), "auc": round(auc, 4),
            "f1": round(f1, 4), "p": round(precision_score(y, binary, zero_division=0), 4),
            "r": round(recall_score(y, binary, zero_division=0), 4)}


def run_fold(feats, labels, tr_idx, va_idx, epochs, lr, fold_name, clip_mask=None,
             seed=SEED, hard_neg_lam=0.0):
    """clip_mask: [n_unit,48] bool 或 None。非 None 时损失与单位分数只在可用片段上
    计算（G2：强弱标签两臂统一片段集）；max/min 用 ±inf 填充，L2 只计相邻双可用
    片段对，L3 只累加可用片段。clip_mask=None 时行为与旧版逐位一致。
    seed: 随机种子（EXP-011 起可传，默认 7 = 旧硬编码 SEED，默认路径逐位一致）。
    hard_neg_lam: 负样本视频超阈惩罚 λ（EXP-011）；0 = 不启用，损失路径与旧版逐位同构。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_idx = np.asarray(tr_idx)
    va_idx = np.asarray(va_idx)
    model = MilHead()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(feats, dtype=torch.float32)
    M = clip_mask if clip_mask is not None else None
    y = torch.tensor(labels, dtype=torch.float32)
    n_pos_tr = int(y[tr_idx].sum())
    n_neg_tr = len(tr_idx) - n_pos_tr
    assert n_pos_tr >= 1 and n_neg_tr >= 1, \
        f"{fold_name}: 训练折正 {n_pos_tr} 负 {n_neg_tr}，L1 需同时含两类单位"
    pos_tr = tr_idx[y[tr_idx] == 1]
    neg_tr = tr_idx[y[tr_idx] == 0]
    bs = min(8, max(n_pos_tr, n_neg_tr))
    best_auc, best_state, best_r, best_ep, hist = -1.0, None, None, -1, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        l_acc = {"l": 0.0, "l1": 0.0, "l2": 0.0, "l3": 0.0,
                 "l_pen": 0.0, "nv": 0.0, "neg_max": 0.0}
        # 无放回遍历（审计修复口径）：正单位每 epoch 恰好贡献一次梯度；
        # 负单位循环补齐到同总数，使用次数 = n_pos/n_neg（如实入 run-log）。
        pos_order = np.random.permutation(pos_tr)
        neg_order = np.random.permutation(neg_tr)
        n_batch = int(np.ceil(len(pos_order) / bs))
        neg_need = len(pos_order)
        if len(neg_order) < neg_need:
            neg_order = np.concatenate([np.tile(neg_order, neg_need // len(neg_order)),
                                        neg_order[:neg_need % len(neg_order)]])
        for b in range(n_batch):
            pi = pos_order[b * bs:(b + 1) * bs]
            ni = neg_order[b * bs:b * bs + len(pi)]
            p = model(X[pi]).squeeze(-1)          # [bs,48]
            n = model(X[ni]).squeeze(-1)
            if any(torch.isnan(t_).any() for t_ in (p, n)):
                print(f"  [!!] {fold_name} ep{ep} NaN 分数，中止")
                return None, hist, None
            if M is not None:
                # 掩码形态（G2 片段集统一）：max/min 只落在可用片段上，
                # L2 只计相邻双可用对，L3 只累加可用片段
                mp, mn = M[pi], M[ni]
                maxp, minp = masked_max_min(p, mp)
                maxn, minn = masked_max_min(n, mn)
                l1 = torch.max(torch.zeros_like(maxp),
                               1.0 - ((maxp - minp) - (maxn - minn))).mean()
                pv = mp[:, 1:] & mp[:, :-1]
                l2 = (((p[:, 1:] - p[:, :-1]) ** 2) * pv).sum(dim=1).mean()
                l3 = (p * mp).sum(dim=1).mean()
                loss = l1 + LAM_T * l2 + LAM_S * l3
            else:
                loss, l1, l2, l3 = mil_loss(p, n)
            if hard_neg_lam > 0:
                # EXP-011：负样本视频超阈惩罚叠加在三项损失之上（λ=0 完全跳过，
                # 保证默认路径与旧版逐位同构）。掩码用负单位行的可用片段。
                nm_b = M[ni] if M is not None else torch.ones_like(n, dtype=torch.bool)
                pen, n_viol = hard_neg_penalty(n, nm_b, tau=PEN_TAU, lam=hard_neg_lam)
                loss = loss + pen
            else:
                pen, n_viol = None, 0
            with torch.no_grad():
                # W1 升档观察量：训练批内负单位 masked max 的批均值（仅观测，
                # 不进计算图、不消耗随机数）
                if M is not None:
                    nmx = n.masked_fill(~M[ni], float("-inf")).max(dim=1).values
                else:
                    nmx = n.max(dim=1).values
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{fold_name} ep{ep}: 损失非有限（{loss.item()}）——数值或掩码"
                    "异常，拒绝静默继续（2026-10-05 掩码 min 项失效事故后强制）")
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in (("l", loss.item()), ("l1", l1.item()),
                         ("l2", l2.item()), ("l3", l3.item())):
                l_acc[k] += v / n_batch
            l_acc["l_pen"] += (float(pen) if pen is not None else 0.0) / n_batch
            l_acc["nv"] += n_viol / n_batch
            l_acc["neg_max"] += float(nmx.mean()) / n_batch
        model.eval()                              # 评估分数必须确定性（Dropout 关）
        if M is not None:
            p_tr = np.array([float(model(X[i]).squeeze(-1)[M[i]].max().detach())
                             for i in tr_idx])
            p_va = np.array([float(model(X[i]).squeeze(-1)[M[i]].max().detach())
                             for i in va_idx])
        else:
            p_tr = np.array([model(X[i]).squeeze(-1).detach().numpy().max() for i in tr_idx])
            p_va = np.array([model(X[i]).squeeze(-1).detach().numpy().max() for i in va_idx])
        r_tr = report("tr" + fold_name, labels[tr_idx], p_tr)
        r_va = report("va" + fold_name, labels[va_idx], p_va)
        hist.append({"epoch": ep, "loss": round(l_acc["l"], 5),
                     "l1": round(l_acc["l1"], 5), "l2": round(l_acc["l2"], 6),
                     "l3": round(l_acc["l3"], 5),
                     "l_pen": round(l_acc["l_pen"], 6),
                     "neg_max": round(l_acc["neg_max"], 5), **r_va})
        if not np.isnan(r_va["auc"]) and r_va["auc"] > best_auc:
            best_auc = r_va["auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_r = r_va
            best_ep = ep
        sign = " *" if best_r is r_va and not np.isnan(r_va["auc"]) else ""
        print(f"  [{fold_name}] ep{ep} L={l_acc['l']:.4f} "
              f"(L1={l_acc['l1']:.4f} L2={l_acc['l2']:.4e} L3={l_acc['l3']:.4f} "
              f"pen={l_acc['l_pen']:.4f} nViol={l_acc['nv']:.1f} "
              f"negMax={l_acc['neg_max']:.4f}) "
              f"trAUC={r_tr['auc']:.3f} vaAUC={r_va['auc']:.3f} vaF1={r_va['f1']:.3f}{sign} "
              f"({time.time()-t0:.0f}s)")
    # best epoch 下的逐单位预测（审计 W4/W5 配套：供逐折查看与复算）
    if best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        if M is not None:
            preds = {i: float(model(X[i]).squeeze(-1)[M[i]].max().detach())
                     for i in va_idx}
        else:
            preds = {i: float(model(X[i]).squeeze(-1).detach().numpy().max())
                     for i in va_idx}
    return best_state, hist, best_r, best_ep, preds if best_state is not None else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feats", required=True, help="unit-features.npz 路径")
    ap.add_argument("--folds-json", default="",
                    help="协议三模式：单位级折划分正本 JSON（val_units 列表）；"
                         "不给则用 npz folds 字段（协议一）")
    ap.add_argument("--smoke", action="store_true", help="冒烟：不划分折，全单位训练并自评")
    ap.add_argument("--clip-labels", dest="clip_labels", default="",
                    help="clip-labels.npz 路径；给定时启用片段掩码（G2 片段集统一）")
    ap.add_argument("--caliber", choices=["strict", "loose"], default="strict",
                    help="片段掩码口径（配合 --clip-labels）")
    ap.add_argument("--fold", type=int, default=-1, help="只跑第 fold 折（0..4）")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="随机种子（默认 7 = 旧硬编码值，默认路径与旧版逐位一致）")
    ap.add_argument("--hard-neg-penalty", dest="hard_neg_penalty", type=float, default=0.0,
                    help="EXP-011 负样本视频超阈惩罚 λ；0 = 不启用（默认，与旧版逐位同构）")
    ap.add_argument("--tag", default="mil-smoke", help="输出 run 目录标签")
    args = ap.parse_args()
    for name in ('feats', 'clip_labels', 'folds_json'):
        value = getattr(args, name)
        if value:
            setattr(args, name, str(resolve_path(value, must_exist=True)))

    d = np.load(args.feats, allow_pickle=True)
    feats, labels = d["feats"], d["labels"]
    unit_ids = d["unit_ids"]
    n = len(labels)
    if args.folds_json:
        # 协议三：单位级折映射来自独立正本 JSON（npz folds 字段=协议一，不读）
        fj = json.load(open(args.folds_json, encoding="utf-8"))
        u2f = {u: k["fold"] for k in fj["folds"] for u in k["val_units"]}
        missing = [u for u in unit_ids if u not in u2f]
        assert not missing, f"折 JSON 缺单位映射: {missing[:5]}"
        folds = np.array([u2f[u] for u in unit_ids])
        folds_src = Path(args.folds_json).name
    else:
        folds = d["folds"]
        folds_src = "npz(协议一 EXP-008-units-folds.json)"
    clip_mask = None
    if args.clip_labels:
        cl = np.load(args.clip_labels, allow_pickle=True)
        assert list(cl["unit_ids"]) == list(unit_ids), "clip-labels 与 npz 行序不一致"
        clip_mask = torch.tensor(cl[args.caliber].astype(int) >= 0)
        assert bool(clip_mask.any(dim=1).all()), \
            "存在无任何可用片段的单位：掩码下 max/min 归约无定义，需先处理该单位"
        folds_src += f" + clip掩码({args.caliber})"
        print(f"[MIL] 片段掩码 {args.caliber}：可用 {int(clip_mask.sum())}/"
              f"{clip_mask.numel()}")
    print(f"[MIL] 载入 {n} 单位 特征{feats.shape} 类别 {np.bincount(labels).tolist()} "
          f"折来源 {folds_src}")

    fold_nums = sorted(set(int(f) for f in folds))
    if args.smoke:
        tr_id = va_id = list(range(n))
        state, hist, best_r, best_ep, preds = run_fold(feats, labels, tr_id, va_id,
                                                       args.epochs, args.lr, "smoke",
                                                       clip_mask=clip_mask,
                                                       seed=args.seed,
                                                       hard_neg_lam=args.hard_neg_penalty)
        runs = [{"fold": "smoke", "name": "smoke",
                 "n": int(n), "auc": best_r["auc"], "f1": best_r["f1"],
                 "best_epoch": best_ep}] if hist and best_r else []
        bests, all_preds = {}, {"smoke": preds} if preds else {}
        bests = {}
    else:
        fold_list = [args.fold] if args.fold >= 0 else fold_nums
        runs, bests, all_preds = [], {}, {}
        for f in fold_list:
            va_id = [i for i in range(n) if int(folds[i]) == f]
            tr_id = [i for i in range(n) if int(folds[i]) != f]
            if len(set(labels[va_id])) < 2:
                print(f"  [MIL] 折 {f}: 验证单位单一类别，跳过（{len(va_id)} 单位）")
                continue
            n_pos_tr = int(np.asarray(labels)[tr_id].sum())
            n_neg_tr = len(tr_id) - n_pos_tr
            state, hist, best_r, best_ep, preds = run_fold(feats, labels, tr_id, va_id,
                                                           args.epochs, args.lr, f"f{f}",
                                                           clip_mask=clip_mask,
                                                           seed=args.seed,
                                                           hard_neg_lam=args.hard_neg_penalty)
            if state is None:
                print(f"[MIL] 折 {f} NaN 中止")
                continue
            bests[f] = state
            all_preds[f] = preds
            runs.append({"fold": f, "name": f"f{f}", "n": int(len(va_id)),
                         "n_pos": int(sum(1 for i in va_id if labels[i] == 1)),
                         "n_neg": int(sum(1 for i in va_id if labels[i] == 0)),
                         "neg_cycle_ratio": round(n_pos_tr / n_neg_tr, 2),
                         "auc": best_r["auc"], "f1": best_r["f1"],
                         "best_epoch": best_ep})
    if not runs:
        print("[MIL] 无有效折叠，退出")
        return

    out_dir = RUNS_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "script": "train_mil.py", "n_unit": n, "epochs": args.epochs, "lr": args.lr,
        "lam_t": LAM_T, "lam_s": LAM_S, "margin": MARGIN, "seed": args.seed,
        "hard_neg_penalty": args.hard_neg_penalty, "penalty_tau": PEN_TAU,
        "hard_neg_scope": "全部负视频；strict 掩码 masked max 超 tau 记违规，违规行均值×λ",
        "smoke": args.smoke, "folds_source": folds_src,
        "feats": str(Path(args.feats).resolve()),
        "clip_labels": str(Path(args.clip_labels).resolve()) if args.clip_labels else None,
        "clip_caliber": args.caliber if args.clip_labels else None, "folds": runs,
    }
    aucs = [r["auc"] for r in runs if not np.isnan(r["auc"])]
    if not args.smoke and aucs:
        summary["mean_auc"] = round(float(np.mean(aucs)), 4)
        summary["std_auc"] = round(float(np.std(aucs)), 4)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if not args.smoke:
        for f, state in bests.items():
            torch.save(state, out_dir / f"mil_f{f}.pt")
        with open(out_dir / "predictions.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fold", "unit_id", "label", "score"])
            for fold, preds in all_preds.items():
                for i, s in preds.items():
                    # 写全精度分数（不四舍五入）；metrics.json 的 mean_auc 同为全精度，为权威引用口径
                    w.writerow([fold, unit_ids[i], int(labels[i]), float(s)])
    print(f"[MIL] 完成 -> {out_dir / 'metrics.json'}")
    for r in runs:
        print(f"  {r['name']}: n={r['n']} vaAUC={r['auc']} vaF1={r['f1']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
