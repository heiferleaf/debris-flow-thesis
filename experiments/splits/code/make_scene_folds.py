#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""场景级（按拍摄地点整地点不拆）5 折划分设计 → EXP-008-units-scene-folds.json

设计规则（2026-10-10 叶海峰拍板「带限制跑」后，按执行计划第 1 步实施）：
- 划分原子 = 场景组 = 地址名（unit-manifest.csv group_id 的中段，d|地址名|日期 → 地址名）。
  同一场景组的所有视频强制落同一折（数据使用协议硬约束 1/2 的场景级形态）。
- 150 条视频共 11 个场景组，其中 5 个正负两类都有（混合场景），6 个只有单一类。
  混合场景各锚定一折（保证每折验证侧正负都有、AUC 可算）；6 个单类场景
  穷举分配（5^6=15625 组合，确定性，无随机数），目标：
  ①主目标：折间验证侧规模差最小（max-min）；②次目标：最差折的 P×N 最大
  （AUC 分辨率 = 1/(P×N)，P×N 越大单折 AUC 粒度越细）。
- 输出格式与 EXP-008-units-iid-folds.json 一致（val_units 为 unit_id 列表，
  train_mil.py / train_strong.py 的 --folds-json 直接可读），另附 val_scenes 供核对。

用法：python make_scene_folds.py            # 只打印设计结果，不写文件
      python make_scene_folds.py --apply    # 落盘 data/metadata/splits/
"""
import argparse
import csv
import json
import itertools
from collections import defaultdict
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "data/process/EXP-008-units-v2/unit-manifest.csv"
OUT = REPO / "data/metadata/splits/EXP-008-units-scene-folds.json"


def scene_of(group_id: str) -> str:
    parts = group_id.split("|")
    assert len(parts) == 3 and parts[0] == "d", f"意外的 group_id: {group_id}"
    return parts[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="落盘折正本")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    assert len(rows) == 150, f"清单应有 150 条，实为 {len(rows)}"

    # 场景组 → 单位列表 / 正负计数
    scene_units = defaultdict(list)
    scene_pn = defaultdict(lambda: [0, 0])
    scene_date = {}
    for r in rows:
        s = scene_of(r["group_id"])
        scene_units[s].append(r["unit_id"])
        scene_pn[s][0 if r["label"] == "pos" else 1] += 1
        scene_date[r["group_id"]] = s
    assert sum(len(v) for v in scene_units.values()) == 150

    mixed = sorted([s for s, (p, n) in scene_pn.items() if p > 0 and n > 0],
                   key=lambda s: -len(scene_units[s]))
    single = sorted([s for s, (p, n) in scene_pn.items() if p == 0 or n == 0],
                    key=lambda s: -len(scene_units[s]))
    assert len(mixed) == 5 and len(single) == 6, (mixed, single)
    print(f"混合场景（各锚定一折）: {[(s, len(scene_units[s])) for s in mixed]}")
    print(f"单类场景（穷举分配）  : {[(s, len(scene_units[s])) for s in single]}")

    # 穷举 6 个单类场景 → 5 折
    best = None
    for assign in itertools.product(range(5), repeat=len(single)):
        sizes, pns = [], []
        for f in range(5):
            extra = [single[i] for i in range(len(single)) if assign[i] == f]
            p = scene_pn[mixed[f]][0] + sum(scene_pn[s][0] for s in extra)
            n = scene_pn[mixed[f]][1] + sum(scene_pn[s][1] for s in extra)
            sizes.append(p + n)
            pns.append((p, n))
        spread = max(sizes) - min(sizes)
        worst_pn = min(p * n for p, n in pns)
        key = (spread, -worst_pn)
        if best is None or key < best[0]:
            best = (key, assign, sizes, pns)
    _, assign, sizes, pns = best
    print(f"最优分配: spread={sizes and max(sizes)-min(sizes)}, "
          f"各折规模={sizes}, 各折(正,负)={pns}, 最差折 P×N={min(p*n for p,n in pns)}")

    folds_out = []
    for f in range(5):
        scenes = [mixed[f]] + [single[i] for i in range(len(single)) if assign[i] == f]
        units = [u for s in scenes for u in sorted(scene_units[s])]
        p, n = pns[f]
        assert p >= 1 and n >= 1, f"折 {f+1} 单类: pos={p} neg={n}"
        folds_out.append({"fold": f + 1, "val_scenes": scenes,
                          "val_units": units, "val_pos": p, "val_neg": n})

    covered = [u for k in folds_out for u in k["val_units"]]
    assert sorted(covered) == sorted(r["unit_id"] for r in rows), "折划分未精确覆盖 150 条"
    assert sum(k["val_pos"] for k in folds_out) == 113
    assert sum(k["val_neg"] for k in folds_out) == 37

    # 定量注记：同日期跨场景的场次组对有多少被拆进不同折（审查 W2/S1 要求）
    g2fold = {}
    for k in folds_out:
        for s in k["val_scenes"]:
            g2fold[s] = k["fold"]
    date_groups = defaultdict(list)
    for gid, s in scene_date.items():
        date_groups[gid.split("|")[2]].append(s)
    total_pairs = same_fold_pairs = 0
    for d, ss in date_groups.items():
        for a, b in combinations(sorted(set(ss)), 2):
            total_pairs += 1
            if g2fold[a] == g2fold[b]:
                same_fold_pairs += 1
    split_pairs = total_pairs - same_fold_pairs
    sizes_all = [len(k["val_units"]) for k in folds_out]
    worst_pn = min(k["val_pos"] * k["val_neg"] for k in folds_out)
    worst_fold = [k["fold"] for k in folds_out
                  if k["val_pos"] * k["val_neg"] == worst_pn][0]
    print(f"同日期跨场景场次组对: {total_pairs} 对, 其中 {split_pairs} 对跨折 "
          f"(同折 {same_fold_pairs})")

    doc = {
        "name": "EXP-008-units-scene-cv5",
        "unit": "video_id(一条视频=一个弱标签单位)",
        "task": "A",
        "meta": {
            "protocol": "场景级折：划分原子=场景组(地址名)，整地点不拆；混合场景各锚一折，"
                        "单类场景穷举分配(主目标折规模差最小,次目标最差折P×N最大)，无随机数",
            "scene_count": 11,
            "mixed_anchors": mixed,
            "single_assigned": {s: assign[i] + 1 for i, s in enumerate(single)},
            "manifest": "data/process/EXP-008-units-v2/unit-manifest.csv",
            "usage_limit": "凡引用场景级成绩逐条带上，共 5 条："
                           f"①只作描述性对比、不做显著性主张——验证侧规模实测 "
                           f"{min(sizes_all)}~{max(sizes_all)} 条(正16~28/负4~13)，"
                           f"单折 AUC 分辨率=1/(P×N)，最粗 1/{worst_pn}(折{worst_fold})，"
                           "折间波动大；"
                           f"②这是「没见过的拍摄地点」留出，不是跨事件留出——同日期跨场景"
                           f"场次组对 {total_pairs} 对中 {split_pairs} 对被拆进不同折，"
                           "同一场事件的相邻机位画面可同时出现在两侧；"
                           "③设计为仓库自定(穷举全局最优)，非标准库做法，进论文方法节"
                           "标「本文自定」，并写明设计目标只用验证侧规模与 P×N 两个"
                           "不接触任何标签分数的量、穷举在训练前一次落盘冻结；"
                           "④折1/折2 弱标签 AUC=1.0000 不得单独引用为满分——负样本仅 "
                           "4/10 条，分辨率 1/112 与 1/210，满分与 0.99 在该规模下不可区分；"
                           "⑤F1 在部分折阈值 0.5 下退化(全部判成没有/全部判成有)，场景级成绩"
                           "引用口径只取 AUC，F1 不作引用。",
            "fold_format_note": "本文件 val_units 是 unit_id 列表，可直接喂 train_mil.py / "
                                "train_strong.py 的 --folds-json；EXP-008-units-folds.json "
                                "(协议一) 的 val_units 是计数整数、val_groups 才是分组，不能这样喂",
            "decision": "2026-10-02 叶海峰拍板协议二(场景级)本线不跑；2026-10-08 复核维持不跑"
                        "(朴素均分出单类验证折)；2026-10-10 叶海峰改拍板为「说明限制前提下跑」，"
                        "本文件为该决议的设计产物",
            "generator": "experiments/splits/code/make_scene_folds.py --apply",
        },
        "folds": folds_out,
    }
    print(json.dumps({k["fold"]: {"scenes": k["val_scenes"], "n": len(k["val_units"]),
                                  "pos": k["val_pos"], "neg": k["val_neg"]}
                      for k in folds_out}, ensure_ascii=False, indent=1))
    if args.apply:
        OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"已落盘: {OUT}")


if __name__ == "__main__":
    main()
