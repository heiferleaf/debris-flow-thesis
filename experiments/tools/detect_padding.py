"""扫描图片库,找出"为凑 16:9 而填充"的图:黑边填充 / 模糊边填充(竖屏视频转横屏)。

背景:dataset1/2 里混有竖屏视频截图,两侧(或上下)用纯黑或自身模糊拉伸补成横版。
填充区是合成像素,模型可能学到"黑边=泥石流"的捷径,训练前应裁掉。
本脚本只检测、只出清单与样例拼图,不改任何原图。

判定(对每条边独立):
  黑边:边带内灰度均值 < 30 且标准差 < 8(纯黑/近纯黑);
  模糊边:边带的拉普拉斯方差(高频能量) < 中心区的 1/8,且边带内存在
         明显的填充分界(梯度尖峰)。两条件同时满足才判模糊边,防止把
         天空等平坦真实区域误判成模糊填充。

用法(仓库根目录):
    .venv\\Scripts\\python.exe experiments\\tools\\detect_padding.py \\
        data/raw/dataset1 data/raw/dataset2 data/raw/dataset3 \\
        --out data/process/images_padded_free/填充边检测
产出:candidates.csv(路径,哪几条边,类型,边带占比)+ sample.jpg(前 24 张)。
(2026-09-16 data 结构重构:image 检测报告归 data/process/images_padded_free/,旧
 data/processed/label-review/填充边检测 已并入。)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def imread_unicode(path: Path) -> np.ndarray | None:
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def analyze(img: np.ndarray) -> dict | None:
    """返回各边判定结果;无任何填充返回 None。

    最终判据(2026-09-12 经 6 张已知正负例验证 6/6,含黑边/模糊边/单侧雾区误报):
      1. 从图边向内是连续低竖向梯度能量区(逐列中位数,抗水印/字幕;
         容忍零星噪声列,连续 run 列违规才算边界);
      2. 左右(或上下)填充必须近似对称(竖屏补边内容居中;|L-R|<=4%),
         且总填充带宽 >= 12% —— 单侧雾区/天空因此被排除;
      3. 分类:填充带亮度均值<30 且标准差<8 为 black,否则 blur。
    扫描深度上限 40%:9:16 竖屏补到 16:9 时单边最多 34.2%。
    """
    h, w = img.shape[:2]
    if h < 60 or w < 60:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gf = gray.astype(np.float32)
    res: dict = {}

    def pair_scan(energy: np.ndarray, n: int, body: slice, transposed: bool) -> bool:
        """energy: 逐列(或逐行,transposed=True)梯度能量中位数。判定一对边。"""
        k = max(3, n // 100)
        es = np.convolve(energy, np.ones(k) / k, mode="same")
        center = es[n // 5: 4 * n // 5].mean()
        if center < 1e-6:
            return False
        depth = int(0.4 * n)
        run = max(4, n // 150)

        def extent(low: np.ndarray) -> int:
            run_len = 0
            for i in range(depth):
                if low[i]:
                    run_len = 0
                else:
                    run_len += 1
                    if run_len >= run:
                        return i - run + 1
            return depth

        low_f = es[:depth] < center * 0.15
        low_r = es[::-1][:depth] < center * 0.15
        a, b = extent(low_f), extent(low_r)
        if a < 4 and b < 4:
            return False
        if abs(a - b) > 0.04 * n or a + b < 0.12 * n:
            return False

        def kind(seg_lo: int, seg_hi: int) -> str:
            # 非转置扫的是列方向(左右边):取整行的列带;转置扫的是行方向(上下边):取整列的行带
            seg = gray[:, seg_lo:seg_hi] if not transposed else gray[seg_lo:seg_hi, :]
            gm, gs = float(seg.mean()), float(seg.std())
            return "black" if (gm < 30 and gs < 8) else "blur"

        suffix = "_h" if transposed else ""
        res["left" + suffix if not transposed else "top"] = kind(0, a)
        res["right" + suffix if not transposed else "bottom"] = kind(n - b, n)
        res["left_frac" + suffix if not transposed else "top_frac"] = a / n
        res["right_frac" + suffix if not transposed else "bottom_frac"] = b / n
        return True

    dy = np.abs(cv2.Sobel(gf, cv2.CV_64F, 0, 1, ksize=3))
    dx = np.abs(cv2.Sobel(gf, cv2.CV_64F, 1, 0, ksize=3))
    found = pair_scan(np.median(dy, axis=0), w, slice(0, int(h * 0.85)), False)
    found = pair_scan(np.median(dx, axis=1), h, slice(0, int(w * 0.85)), True) or found
    if not res:
        return None
    res["band_frac"] = sum(
        res.get(s + "_frac", 0.0) + res.get(s + "_frac_h", 0.0)
        for s in ("left", "right", "top", "bottom") if res.get(s) or res.get(s + "_h")
    )
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="检测黑边/模糊边填充的图,只出清单不改图")
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default="data/process/images_padded_free/填充边检测",
                    help="输出根目录(新 data 结构 process/images_padded_free/;旧 label-review/填充边检测 已并入)")
    ap.add_argument("--max-side", type=int, default=640)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict] = []
    total = 0
    for root in args.roots:
        root = Path(root)
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file())
        print(f"[{root.name}] 扫描 {len(files)} 张 ...")
        for i, p in enumerate(files, 1):
            img = imread_unicode(p)
            total += 1
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = args.max_side / max(h, w)
            if scale < 1:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            res = analyze(img)
            if res:
                candidates.append({
                    "path": p.resolve().as_posix(),
                    "dataset": root.name,
                    "rel": p.resolve().relative_to(root.resolve()).as_posix(),
                    "sides": "+".join(res[s] for s in ("left", "right", "top", "bottom") if s in res) or "",
                    "edges": ",".join(s for s in ("left", "right", "top", "bottom") if s in res),
                    "band_frac": f"{res['band_frac']:.3f}",
                    "left_frac": f"{res.get('left_frac', 0):.3f}",
                    "right_frac": f"{res.get('right_frac', 0):.3f}",
                    "top_frac": f"{res.get('top_frac', 0):.3f}",
                    "bottom_frac": f"{res.get('bottom_frac', 0):.3f}",
                })
            if i % 2000 == 0:
                print(f"  {i}/{len(files)}  当前候选 {len(candidates)}")
    candidates.sort(key=lambda r: (r["dataset"], r["rel"]))
    with (out_dir / "candidates.csv").open("w", encoding="utf-8-sig", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["dataset", "rel", "path", "sides", "edges", "band_frac",
                                             "left_frac", "right_frac", "top_frac", "bottom_frac"])
        wcsv.writeheader()
        wcsv.writerows(candidates)

    # 样例拼图:黑边和模糊边各取前 12 张
    cells = []
    for row in candidates[:24]:
        img = imread_unicode(Path(row["path"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = 300 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        canvas = np.full((340, 340, 3), 255, np.uint8)
        canvas[10:10 + img.shape[0], 10:10 + img.shape[1]] = img
        cv2.putText(canvas, row["sides"], (8, 334), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
        cells.append(canvas)
    if cells:
        while len(cells) % 6:
            cells.append(np.full((340, 340, 3), 230, np.uint8))
        sheet = np.vstack([np.hstack(cells[i:i + 6]) for i in range(0, len(cells), 6)])
        ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            buf.tofile(str(out_dir / "sample.jpg"))
    black = sum(1 for c in candidates if "black" in c["sides"])
    blur = sum(1 for c in candidates if "blur" in c["sides"])
    print(f"完成:共扫 {total} 张,填充候选 {len(candidates)} 张(含黑边 {black},含模糊边 {blur})")
    print(f"清单:{out_dir / 'candidates.csv'}")


if __name__ == "__main__":
    main()
