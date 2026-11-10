# -*- coding: utf-8 -*-
"""EXP-008 单位级 5 折划分（两模式，均一次落盘后冻结）。

模式一（默认）：单位级分层 + 组约束（协议一，research-auditor 2026-10-04 决策复核
通过，按草案 B 执行）。

模式二（--iid）：协议三「同场次混分参照」（叶海峰 2026-10-04 定）——仅按标签分层
随机均分，无组约束，独立正本；成绩只作上限参照，引用须带脚注（见 SKILL 硬约束 1）。

背景与依据（run-log EXP-008-20261003-units-R1R2「折划分决策复核与重划」节）：
  旧折沿用 EXP-006-session-folds.json，该折当年按窗口数分层、且含 EXP-008 池外的
  v2/v3 空组，对「整条视频单位」是错配（fold3 验证侧仅 3 个负单位，AUC 粒度过粗）。
  模式一按组级 pos/neg 的视频数重新分层划 5 折：组约束不破（同组全部视频同折）。

物理上限（复核修正 1）：负样本视频集中在 10 个组（组内负数 10,7,4,4,4,3,2,2,1,1），
  组不可拆，含 10 条负样本的大组整组入折 ⇒ 某折验证侧的负样本视频必然 ≥10 条；可达均衡为
  每折负 7~10、正 20~24（旧折 3~12）。

搜索方法（复核要求：普通贪心会留出空折）：大组优先基础序 + 固定 seed 随机重启
  贪心，目标 = 各折 (pos-目标)² + (neg-目标)² 之和，硬约束每折验证侧两类俱全；
  取 500 次重启中成本最低者。**一次生成、一次落盘后冻结**（复核修正 2），
  此后不得再动、不得换 seed 重挑。

生成命令（依据 = 2026-10-04 research-auditor 决策复核「按草案 B 执行」结论）：
  协议一：python experiments/splits/code/make_unit_folds.py --apply \
      --npz data/results/exp008/unitfeats-v2-150/unit-features.npz
  协议三：python experiments/splits/code/make_unit_folds.py --iid --apply
  --apply 重写 unit-manifest.csv 的 fold 列（仅协议一模式）；--npz 同步特征 npz
  的 folds 字段（特征数组与折无关，逐位不变）。manifest 与 npz 的 fold 均以
  data/metadata/splits/EXP-008-units-folds.json 为唯一正本；协议三正本独立
  （EXP-008-units-iid-folds.json），manifest fold 列不含协议三值。
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "data/process/EXP-008-units/unit-manifest.csv"
OUT_JSON = REPO / "data/metadata/splits/EXP-008-units-folds.json"
OUT_IID_JSON = REPO / "data/metadata/splits/EXP-008-units-iid-folds.json"
SEED = 42
N_FOLDS = 5
RESTARTS = 500


def load_groups():
    """manifest -> {group_id: [n_pos, n_neg]}（fold 列不参与，忽略旧值）。"""
    g = defaultdict(lambda: [0, 0])
    for r in csv.DictReader(open(MANIFEST, encoding="utf-8-sig", newline="")):
        g[r["group_id"]][0 if r["label"] == "pos" else 1] += 1
    return dict(g)


def cost_of(pos_f, neg_f, tp, tn):
    return sum((p - tp) ** 2 + (n - tn) ** 2 for p, n in zip(pos_f, neg_f))


def greedy(order, groups, tp, tn):
    """按给定顺序逐组放入增量成本最小的折；返回 (assign, cost, ok)。"""
    pos_f = [0] * N_FOLDS
    neg_f = [0] * N_FOLDS
    assign = {}
    for g in order:
        n_pos, n_neg = groups[g]
        best, bf = None, None
        for f in range(N_FOLDS):
            pos_f[f] += n_pos
            neg_f[f] += n_neg
            c = cost_of(pos_f, neg_f, tp, tn)
            pos_f[f] -= n_pos
            neg_f[f] -= n_neg
            if best is None or c < best:
                best, bf = c, f
        assign[g] = bf
        pos_f[bf] += n_pos
        neg_f[bf] += n_neg
    ok = all(p >= 1 and n >= 1 for p, n in zip(pos_f, neg_f))
    return assign, cost_of(pos_f, neg_f, tp, tn), ok


def main():
    ap = argparse.ArgumentParser(description="EXP-008 单位级分层 5 折划分")
    ap.add_argument("--iid", action="store_true",
                    help="协议三「同场次混分参照」：仅按标签分层随机均分（无组约束），"
                         "独立正本 EXP-008-units-iid-folds.json；不加此参数为协议一组约束模式")
    ap.add_argument("--apply", action="store_true",
                    help="落盘正本 JSON（协议一模式还会重写 manifest fold 列）；"
                         "默认只打印方案不落盘")
    ap.add_argument("--npz", default="",
                    help="协议一模式：同步该 unit-features.npz 的 folds 字段")
    args = ap.parse_args()

    out_json = OUT_IID_JSON if args.iid else OUT_JSON
    if out_json.exists():
        print(f"[folds] {out_json.name} 已存在——折一次落盘后冻结，不得重生成"
              f"（如确需重划须新开决策并过 research-auditor 复核）")
        sys.exit(1)

    if args.iid:
        # 协议三：仅按标签分层（组约束豁免，G4：不用窗口数）。确定序 + seed 洗牌 +
        # 向当前最少折发牌，正负计数自动均衡，每折两类俱全。
        rng = random.Random(SEED)
        by_label = defaultdict(list)
        for r in csv.DictReader(open(MANIFEST, encoding="utf-8-sig", newline="")):
            by_label[r["label"]].append(r["unit_id"])
        assign = {}
        fill = [0] * N_FOLDS
        for lab in ("pos", "neg"):
            units = sorted(by_label[lab])
            rng.shuffle(units)
            for uid in units:
                f = min(range(N_FOLDS), key=lambda i: (fill[i], i))
                assign[uid] = f
                fill[f] += 1
        neg_per_fold = [sum(1 for u in by_label["neg"] if assign[u] == f)
                        for f in range(N_FOLDS)]
        if any(v == 0 for v in neg_per_fold):
            print("[folds] 某折无负单位，中止（不落盘）")
            sys.exit(1)
        folds_out = []
        print("[folds-iid] 各折构成（验证侧，无组约束）：")
        for f in range(N_FOLDS):
            vu = sorted(u for u, ff in assign.items() if ff == f)
            p = sum(1 for u in vu if u in set(by_label["pos"]))
            n = len(vu) - p
            print(f"  fold{f + 1}: 正 {p} / 负 {n}")
            folds_out.append({"fold": f + 1, "val_units": vu,
                              "val_pos": p, "val_neg": n})
        if not args.apply:
            print("[folds-iid] 试算完成（未加 --apply，不落盘）")
            return
        out = {
            "name": "EXP-008-units-iid-cv5",
            "unit": "video_id(整条视频弱标签单位)",
            "task": "A",
            "meta": {
                "protocol": "协议三「同场次混分参照」（2026-10-04 叶海峰定）",
                "seed": SEED,
                "layering_key": "单位级标签分层（无组约束，同场次可拆两边）",
                "search": "确定序+seed洗牌+向最少折发牌（正负计数均衡）",
                "manifest": "data/process/EXP-008-units/unit-manifest.csv",
                "usage_limit": "成绩只作上限参照（i.i.d. split），凡引用必须带脚注"
                               "「同场次混分参照，含场次内画面相似性，非泛化成绩」，"
                               "不得用于任何泛化/部署主张（research-auditor 2026-10-04）",
                "generator": "experiments/splits/code/make_unit_folds.py --iid --apply",
                "generated_on": date.today().isoformat(),
                "frozen": "本折一次生成落盘后冻结，不得再动、不得换 seed 重挑",
                "note": "unit-manifest.csv 的 fold 列 = 协议一折；本文件是协议三唯一正本，"
                        "两协议正本分离，不得混用",
            },
            "folds": folds_out,
        }
        out_json.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"[folds-iid] 正本落盘 -> {out_json}")
        return

    groups = load_groups()
    tot_pos = sum(v[0] for v in groups.values())
    tot_neg = sum(v[1] for v in groups.values())
    tp, tn = tot_pos / N_FOLDS, tot_neg / N_FOLDS
    print(f"[folds] 组 {len(groups)}，单位 正{tot_pos}/负{tot_neg}，"
          f"目标每折 正{tp:.1f}/负{tn:.1f}")

    # 大组优先基础序 + 固定 seed 随机重启贪心（普通贪心会留出空折，复核实测）
    base = sorted(groups, key=lambda g: (-(groups[g][0] + groups[g][1]),
                                         -groups[g][1], g))
    rng = random.Random(SEED)
    best_assign, best_cost, best_ok = None, None, False
    for r in range(RESTARTS):
        order = base[:] if r == 0 else rng.sample(base, len(base))
        assign, cost, ok = greedy(order, groups, tp, tn)
        if ok and (best_cost is None or cost < best_cost):
            best_assign, best_cost, best_ok = assign, cost, r
    if not best_ok:
        print("[folds] 500 次重启均无法满足每折两类俱全，中止（不落盘）")
        sys.exit(1)

    folds_out = []
    print("[folds] 各折构成（验证侧）：")
    for f in range(N_FOLDS):
        vg = sorted(g for g, ff in best_assign.items() if ff == f)
        p = sum(groups[g][0] for g in vg)
        n = sum(groups[g][1] for g in vg)
        print(f"  fold{f + 1}: 正 {p} / 负 {n}（{len(vg)} 组）")
        folds_out.append({"fold": f + 1, "val_groups": vg,
                          "val_units": p + n, "val_pos": p, "val_neg": n})

    out = {
        "name": "EXP-008-units-cv5",
        "unit": "video_id(整条视频弱标签单位)",
        "task": "A",
        "meta": {
            "seed": SEED,
            "group_column": "group_id",
            "group_meaning": "场次组(机位×日期)",
            "layering_key": "组级 pos/neg 单位数（单位级分层，非窗口数）",
            "search": f"大组优先基础序+固定seed随机重启贪心 {RESTARTS} 次，"
                      f"目标=各折(pos-目标)²+(neg-目标)²，硬约束每折验证侧两类俱全",
            "manifest": "data/process/EXP-008-units/unit-manifest.csv",
            "physical_limit": "负样本视频集中在 10 个组（组内负数 10,7,4,4,4,3,2,2,1,1），"
                              "组不可拆，含 10 负组整组入折 ⇒ 某折验证侧负单位必然 ≥10；"
                              "可达均衡为每折负 7~10、正 20~24",
            "generator": "experiments/splits/code/make_unit_folds.py",
            "review": "research-auditor 2026-10-04 决策复核：按草案 B 执行"
                      "（修正1=分层目标 7~10 负/20~24 正；修正2=一次落盘后冻结）",
            "generated_on": date.today().isoformat(),
            "frozen": "本折一次生成落盘后冻结，不得再动、不得换 seed 重挑",
        },
        "folds": folds_out,
    }

    if not args.apply:
        print("[folds] 试算完成（未加 --apply，不落盘）")
        return
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"[folds] 正本落盘 -> {OUT_JSON}")

    # 单一正本化：manifest fold 列重写为新折
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig", newline="")))
    fields = list(rows[0].keys())
    g2f = {g: k["fold"] for k in folds_out for g in k["val_groups"]}
    for r in rows:
        r["fold"] = g2f[r["group_id"]]
    with open(MANIFEST, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[folds] manifest fold 列已同步（{len(rows)} 行）-> {MANIFEST}")

    # 特征 npz folds 字段同步（特征数组与折无关，逐位不变）
    if args.npz:
        d = dict(np.load(args.npz, allow_pickle=True))
        u2f = {r["unit_id"]: int(r["fold"]) for r in rows}
        new_folds = np.array([u2f[u] for u in d["unit_ids"]], dtype=d["folds"].dtype)
        changed = int((new_folds != d["folds"]).sum())
        d["folds"] = new_folds
        np.savez_compressed(args.npz, **d)
        print(f"[folds] npz folds 字段已同步（改动 {changed}/{len(new_folds)} 行，"
              f"feats 未动）-> {args.npz}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
