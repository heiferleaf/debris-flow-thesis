# -*- coding: utf-8 -*-
"""推理链往返自检：确认「展示台里的推理」= 「训练时的口径」（2026-10-05 立）。

背景：research-auditor 2026-10-05 决策复核把「上线前必做往返自检」列为展示台显示分数的
前提条件——原话「不通过则页面不得显示分数」。当时用一次性脚本跑过一次，但一次性脚本会
被销毁，检查就不可重复。本工具把它固化成常驻命令。

它验四件事，任何一条不过就非零退出：
  1. **槽位一致**：从建库使用的 256×256 中间视频重跑 `pick_slots`，48 个采样片段起点必须与
     `unit-manifest.csv` 的 `clip_start_frames` 逐个相同（含训练时因解码失败回退过的槽）。
  2. **禁区不误杀**：推理侧现场重算的禁区（人工标的剪辑区间、低灰度帧、时间戳倒步、
     时间空洞）不得把清单里任何一个片段判成不可用；可用片段数必须等于清单片段数。
  3. **分数一致**：重跑所得整条视频的分数与 `predictions.csv` 的分数在训练线对应的
     浮点容差内相同，且 0.5 判定线上的结果不翻转。FT 默认容差沿用 `clip_scores_ft.py`
     的 0.01，其他线为 1e-6；实测差值须逐条报告。
  4. **可解码**：整条链路不因坏帧中断（坏帧重采后仍须拿到 48 个完整片段）。

**只对训练集里的视频有意义**——`predictions.csv` 里只有训练过的视频。它证明的是
「台子用的预处理与打分和训练时是同一套」，不是「模型在没见过的视频上准不准」。

**旧清单槽位落入后来新增的倒步或空洞禁区时，往返自检无法通过**。原因：150 条视频池与
`unit-manifest.csv` 是 2026-10-05/06 建的，当时建库代码只有三类禁区（人工标的剪辑区间、
低灰度帧、v1_109 专属坏区块），「时间戳倒步」与「时间空洞」两类是 2026-10-25 才加进
`build_units_v2.py` 的。所以对含这两类禁区的视频，清单里的槽位是旧规则挑的、推理侧按新
规则挑，两边槽位可能不一样；现有输出没有分别记录新旧禁区，不能自动证明差异只由
新禁区造成。这类视频须先用离线一致性扫描
（`experiments/tools/oneoff/exp008-zone-audit/zone_consistency.py`）核对，扫描结论见该目录
的 `result.md`。

用法（仓库根，venv；建议给 2~3 个不同源的视频各跑一次）：
    python experiments/tools/check_infer_roundtrip.py --unit v1_07 --line mil
    python experiments/tools/check_infer_roundtrip.py --unit v1_07 --line ft --run data/results/exp008/EXP-009-strong-v2-150-strict-ft-noaug
    python experiments/tools/check_infer_roundtrip.py --unit v1_40 --line ft --run data/results/exp008/EXP-014-strong-v3-128-strict-ft-noaug
    python experiments/tools/check_infer_roundtrip.py --unit v4_20 v5_01 --run .../mil-v2-150-strict

耗时：每条视频要重抽 768 帧并跑一次骨干前向，本机约 3~6 分钟（含坏帧重采）。
"""
import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.paths import resolve_path  # noqa: E402
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/common/video"))
sys.path.insert(0, str(REPO / "experiments/tools"))

from infer_video import infer, sampling_from_run, blocks_for, slot_ok, stride_slot_ok  # noqa: E402

SRC256 = REPO / "data/process/EXP-008-units-src256"   # v2/v3 共用的 256 转码源
DEFAULT_RUN = REPO / "data/results/exp008/mil-v2-150-strict"


def source_video(uid: str):
    """只接受建库使用的 256×256 中间视频；原视频不能替代。"""
    p = SRC256 / f"{uid}.mp4"
    return p if p.is_file() else None


def load_manifest(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return {r["unit_id"]: r for r in csv.DictReader(fh)}


def load_predictions(run_dir: Path):
    p = resolve_path(run_dir) / "predictions.csv"
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8-sig", newline="") as fh:
        return {r["unit_id"]: float(r["score"]) for r in csv.DictReader(fh)}


def load_strict_counts(units_root: Path):
    """历史预测按严格标签掩码取最高分；新视频推理没有这些人工标签。"""
    with np.load(units_root / "clip-labels.npz", allow_pickle=True) as z:
        return {str(uid): int(np.count_nonzero(row >= 0))
                for uid, row in zip(z["unit_ids"], z["strict"])}


def check_one(uid, man, preds, strict_counts, run_dir, tol, keep, line, sampling):
    row = man.get(uid)
    if row is None:
        return "fail", f"{uid}: 不在 unit-manifest.csv 里"
    if uid not in preds:
        return "fail", f"{uid}: predictions.csv 里没有这条视频（不是本 run 训练过的？）"
    n_strict = strict_counts.get(uid)
    if n_strict != 48:
        return "na", (f"{uid}: 严格标签只有 {n_strict}/48 个可用片段，"
                      "新视频推理没有人工标签，不能拿全片段分数与历史预测直接对照")

    want_slots = [int(x) for x in row["clip_start_frames"].split(";") if x]
    want_score = preds[uid]
    fold = int(row["fold"])

    out = REPO / ".codex/docs/.temp" / f"roundtrip-{uid}"
    if out.exists() and not keep:
        shutil.rmtree(out, ignore_errors=True)
    video = source_video(uid)
    if video is None or not Path(video).is_file():
        return "fail", f"{uid}: 找不到建库使用的 256×256 中间视频"
    res = infer(video, run_dir, out, (), [fold], kind=line, prepared=True)

    got_slots = res["slots"]
    got_score = res["per_fold"][str(fold)]["unit_score"]

    ok_slot = got_slots == want_slots
    ok_score = (abs(got_score - want_score) <= tol
                and (got_score >= 0.5) == (want_score >= 0.5))
    # 2026-10-28 新增（research-auditor 决策复核前提五）：推理侧现场重算的禁区不得
    # 误杀任何一个清单认可的槽位。清单里这 48 个位置是建库时按同一套规则挑出来的，
    # 推理侧若把其中任何一个判成不可用，说明两边的禁区口径仍不一致，分数就不可比。
    ok_mask = res.get("n_usable", len(got_slots)) == len(got_slots)

    block = 16 if sampling == "v2" else blocks_for(res["frame_stride"])
    cleaned = [s for s in want_slots if _slot_valid(s, block, res["forbidden_ranges"], sampling)]
    lines = [f"{uid}（折{fold}，{res['n_frames']} 帧，重采 {res['retries']} 次，"
             f"低灰度重挑 {res.get('zone_rounds', 0)} 次）",
             f"  槽位 {'一致' if ok_slot else '不一致'}"
             f"（{len(got_slots)} vs {len(want_slots)}）",
             f"  禁区 不得误杀清单槽位 → {'通过' if ok_mask else '不通过'}"
             f"（可用片段 {res.get('n_usable', len(got_slots))}/{len(got_slots)}，"
             f"倒步 {res.get('n_jump_zones', 0)} 处、空洞 {res.get('n_gap_zones', 0)} 处）",
             f"  分数 重跑 {got_score:.6f} / predictions.csv {want_score:.6f}"
             f" / 绝对差 {abs(got_score - want_score):.2e}（容差 {tol:.0e}）"
             f" / 0.5 判定线翻转 {'是' if (got_score >= 0.5) != (want_score >= 0.5) else '否'}"
             f" → {'通过' if ok_score else '不通过'}"]
    if not ok_slot:
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(got_slots, want_slots)) if a != b]
        lines.append(f"  槽位差异 {len(diff)} 处：{diff[:6]}")
    if res["degenerate_clips"]:
        lines.append(f"  低灰度片段（已剔除，不参与打分）{res['degenerate_clips']}")
    ok_all = ok_slot and ok_score and ok_mask
    if ok_all:
        return "pass", "\n".join(lines)
    n_old_slots_newly_forbidden = len(want_slots) - len(cleaned)
    if sampling == "v2" and n_old_slots_newly_forbidden:
        lines.insert(0, f"{uid}：旧清单有 {n_old_slots_newly_forbidden} 个片段"
                        "按现行禁区规则不可用；仅凭合并禁区不能归因，需单独核对离线扫描。")
    return "fail", "\n".join(lines)


def _slot_valid(frame: int, block: int, ranges, sampling: str) -> bool:
    """按建库脚本的整个片段覆盖范围检查，不能只看起始帧。"""
    if frame % block:
        return False
    slot = frame // block
    return (slot_ok(slot, ranges) if sampling == "v2"
            else stride_slot_ok(slot, block, ranges))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", nargs="+", default=["v1_07"],
                    help="要自检的 unit_id（建议跨源各取一个）")
    ap.add_argument("--run", default=str(DEFAULT_RUN), help="要核对的 run 目录")
    ap.add_argument("--line", choices=("mil", "strong", "ft"), default="mil",
                    help="权重所属训练线；必须与 --run 对应")
    ap.add_argument("--tol", type=float, default=None,
                    help="分数容差；默认 FT 为 0.01（同 clip_scores_ft.py），其他线 1e-6")
    ap.add_argument("--keep", action="store_true", help="保留已抽的帧（调试用）")
    args = ap.parse_args()

    run_dir = resolve_path(args.run, must_exist=True)
    sampling = sampling_from_run(run_dir, args.line)
    units_root = Path(os.environ.get("DF_UNITS_ROOT") or
                      REPO / f"data/process/EXP-008-units-{'v3-gap05' if sampling == 'v3' else 'v2'}")
    man, preds = load_manifest(units_root / "unit-manifest.csv"), load_predictions(run_dir)
    strict_counts = load_strict_counts(units_root)
    tol = args.tol if args.tol is not None else (0.01 if args.line == "ft" else 1e-6)
    if tol < 0:
        ap.error("--tol 不得为负")
    print(f"[往返自检] run={run_dir.name} 训练线={args.line} 视频={args.unit} 容差={tol:.0e}")
    results = []
    for uid in args.unit:
        status, msg = check_one(uid, man, preds, strict_counts, run_dir, tol, args.keep,
                                args.line, sampling)
        print(msg)
        results.append((uid, status))
    bad = [u for u, s in results if s == "fail"]
    na = [u for u, s in results if s == "na"]
    passed = [u for u, s in results if s == "pass"]
    print(f"\n[往返自检] {len(passed)}/{len(results)} 通过"
          + (f"，不通过：{bad}——展示台不得显示分数" if bad else "")
          + (f"，不可直接对照：{na}" if na else "")
          + ("" if bad or na else "——推理链与训练口径一致"))
    return 1 if bad else (2 if na else 0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.exit(main())
