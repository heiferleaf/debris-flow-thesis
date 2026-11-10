# -*- coding: utf-8 -*-
"""比对两次解码产出的逐帧灰度表，回答「中间件重转之后，逐帧质检表还对不对得上」。

背景：2026-10-26 把 19 条软解中间件从原件重转了一遍，重算的灰度表（`EXP-008-units-src256/qc/`）
要能替代当年的表（`EXP-008-units-v2/qc/`）给抽帧用。判断依据只有两种可能：
帧数与时间戳逐帧相等、灰度值逐帧接近；任何一处错位都说明两张表指的不是同一套画面。
所以这里不做任何截断或容差换位——行数不等直接记为错误，绝不按较短的一边比。

为什么单独写成工具：这份证据要长期可复跑。此前那次比对用的是一次性脚本，还带手抄的
期望帧数，事后想复核就得重写一遍（research-auditor 2026-10-26 S1）。

用法（仓库根）：只给两个目录，默认比对两边都有的全部视频。

  .venv/Scripts/python.exe experiments/tools/verify_rebuild_qc.py \
      --old-qc data/process/EXP-008-units-v2/qc \
      --new-qc data/process/EXP-008-units-src256/qc \
      --out data/process/EXP-008-units-src256/qc-rebuild-verify.csv

  # 只比其中几条
  ... --units v1_19,v1_39

产出的一行一个视频：两边帧数、两边表本身的 md5、时间戳最大差、灰度标准差的最大差与
平均绝对差、差值超过 2 的帧数。判定看三件事：帧数相等、时间戳不漂、超差帧数为 0。
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COLS = ["unit_id", "n_old", "n_new", "md5_old", "md5_new", "max_pts_diff",
        "max_std_diff", "mean_abs_std_diff", "n_over_2", "verdict", "error"]


def _repo_path(p):
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


def read_table(path):
    """读一张逐帧灰度表，返回 (帧号列表, 时间戳列表, 灰度列表, md5)。"""
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    frames, pts, stds = [], [], []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            frames.append(int(r["frame"]))
            pts.append(float(r["pts_time"]))
            stds.append(float(r["gray_std"]))
    return frames, pts, stds, md5


def compare(uid, old_path, new_path):
    row = {"unit_id": uid}
    if not old_path.is_file() or not new_path.is_file():
        row["error"] = f"缺表（旧 {old_path.is_file()}、新 {new_path.is_file()}）"
        return row
    f_old, p_old, s_old, m_old = read_table(old_path)
    f_new, p_new, s_new, m_new = read_table(new_path)
    row.update(n_old=len(f_old), n_new=len(f_new), md5_old=m_old, md5_new=m_new)
    if len(f_old) != len(f_new):
        # 行数不等就是两张表覆盖的帧不一样，按较短的一边比等于把差异藏起来。
        row["error"] = f"行数不等（旧 {len(f_old)}、新 {len(f_new)}）"
        return row
    if f_old != f_new:
        bad = next(i for i in range(len(f_old)) if f_old[i] != f_new[i])
        row["error"] = f"帧号列不一致（第 {bad} 行：旧 {f_old[bad]}、新 {f_new[bad]}）"
        return row
    d_pts = max(abs(a - b) for a, b in zip(p_old, p_new))
    d_std = [abs(a - b) for a, b in zip(s_old, s_new)]
    over = sum(1 for d in d_std if d > 2)
    row.update(max_pts_diff=round(d_pts, 6),
               max_std_diff=round(max(d_std), 4),
               mean_abs_std_diff=round(sum(d_std) / len(d_std), 4),
               n_over_2=over)
    # 判定口径：帧数相等已在上面拦住；时间戳漂移超过一帧（0.05 秒）说明不是同一套取样；
    # 灰度差超过 2 说明画面内容不同，不是重转的舍入。
    ok = d_pts <= 0.05 and over == 0
    row["verdict"] = "一致" if ok else "不一致"
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old-qc", required=True, help="旧灰度表目录")
    ap.add_argument("--new-qc", required=True, help="新灰度表目录")
    ap.add_argument("--units", default="", help="逗号分隔的视频编号；不给则比两边都有的全部")
    ap.add_argument("--out", required=True, help="结果表路径")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    old_dir, new_dir = _repo_path(a.old_qc), _repo_path(a.new_qc)
    if a.units:
        ids = [s.strip() for s in a.units.split(",") if s.strip()]
    else:
        ids = sorted({p.stem for p in old_dir.glob("*.csv")} &
                     {p.stem for p in new_dir.glob("*.csv")})
    if not ids:
        raise SystemExit("两个目录里没有同名表可比")

    rows = []
    for uid in ids:
        row = compare(uid, old_dir / f"{uid}.csv", new_dir / f"{uid}.csv")
        rows.append(row)
        if "error" in row:
            print(f"  ! {uid} {row['error']}")
        else:
            print(f"  {uid} 帧 {row['n_old']}/{row['n_new']} 时间戳最大差 "
                  f"{row['max_pts_diff']:.4f}s 灰度最大差 {row['max_std_diff']:.2f} "
                  f"平均绝对差 {row['mean_abs_std_diff']:.3f} 超 2 的帧 "
                  f"{row['n_over_2']} → {row['verdict']}")

    out = _repo_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    bad = [r["unit_id"] for r in rows if r.get("verdict") != "一致"]
    print(f"\n一致 {len(rows) - len(bad)}/{len(rows)} 条 -> {out}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
