# -*- coding: utf-8 -*-
"""选轮次这一步到底带来了多少乐观：三种选法在同一批 run 上对比。

三种选法：
  A 现用做法   —— 在被考核的那一折自己的验证曲线上挑最高点（有泄漏）
  B 折外选轮   —— 在其余四折的验证曲线上挑平均最高的那一轮，再套到被考核折
  C 固定轮数   —— 所有折一律取同一轮

只读 metrics.json 的 hist 字段，不训练。
"""
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS = {
    "EXP-009-150-orig": "EXP-009-strong-v2-150-strict-ft-noaug",
    "EXP-014-128-orig": "EXP-014-strong-v2-128-strict-ft-noaug",
    "EXP-014-128-new": "EXP-014-strong-v3-128-strict-ft-noaug",
}


def load(run):
    p = REPO / "data/results/exp008" / run / "metrics.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    folds = m["folds"]
    hist = {f["fold"]: [x["va_auc"] for x in f["hist"]] for f in folds}
    return hist


def main():
    for name, run in RUNS.items():
        hist = load(run)
        folds = sorted(hist)
        n_ep = min(len(v) for v in hist.values())
        print("=" * 74)
        print(name)

        print(f"  各折共同可用轮数 {n_ep}；下文轮次均为从 0 开始的索引")

        # A 现用做法
        a = [max(hist[f][:n_ep]) for f in folds]
        a_ep = [hist[f].index(max(hist[f][:n_ep])) for f in folds]
        print(f"  A 被考核折自己挑  : 均值 {statistics.mean(a):.4f}  各折 {(', '.join(f'{v:.3f}' for v in a))}")
        print(f"                     选中轮次 {a_ep}")

        # B 折外选轮
        b = []
        b_ep = []
        for f in folds:
            others = [g for g in folds if g != f]
            prof = [statistics.mean(hist[g][e] for g in others) for e in range(n_ep)]
            e = prof.index(max(prof))
            b.append(hist[f][e])
            b_ep.append(e)
        print(f"  B 其余四折选轮    : 均值 {statistics.mean(b):.4f}  各折 {(', '.join(f'{v:.3f}' for v in b))}")
        print(f"                     选中轮次 {b_ep}   （每折不同，因为参照折不同）")

        # C 固定轮数
        print("  C 固定轮数        :")
        for e in range(n_ep):
            v = [hist[f][e] for f in folds]
            print(f"                     第 {e:>2} 轮 均值 {statistics.mean(v):.4f}  最低 {min(v):.3f}")

        # 最优固定轮
        best_fixed = max(
            range(n_ep),
            key=lambda e: statistics.mean(hist[f][e] for f in folds),
        )
        print(f"  -> 固定轮数里最好的一个是第 {best_fixed} 轮，"
              f"均值 {statistics.mean(hist[f][best_fixed] for f in folds):.4f}")
        print()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
