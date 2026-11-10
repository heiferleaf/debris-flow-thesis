"""把 pHash 聚类结果按簇拼成总览图,供人工校验「同簇画面是否真的相似」。

输入 data/splits/phash-clusters.csv(每行一张图:cluster_id,source,official_group,relpath),
只为成员数 >= --min-size 的簇出图,按成员数从大到小排序。每个簇拼成
cols x rows 网格 sheet:标题写簇号与成员数,每格右下角写该图的短路径
(数据集名/拆分/文件名)。成员多于 --max-cells 时拆成 _partN 多张 sheet。

产出(写到 data/processed/label-review/clusters/):
  cluster_<id>.jpg 或 cluster_<id>_partNN.jpg  每张 sheet 一图
  index.txt  每行一个簇:出出的 sheet 与成员数;末尾汇总多图簇数与覆盖图片数

用法(仓库根目录):
    .venv\\Scripts\\python.exe experiments\\tools\\oneoff\\cluster_review_sheets.py --top 8
    .venv\\Scripts\\python.exe experiments\\tools\\oneoff\\cluster_review_sheets.py            # 全部多图簇

归档(2026-09-16 data 结构重构):簇校验总览为一次性审计件,当前簇分布已核验入档。
簇清单 data/splits/phash-clusters.csv → data/final/images/clusters/phash-clusters.csv;
产出目录 data/processed/label-review/clusters → process/ 下对应位置。若要重跑先按新路径
改 --csv/--out。
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# 图片实际存放位置:CSV 里的 source + relpath 相对 data/raw 拼接
RAW_ROOT = Path("data/raw")
CSV_PATH = Path("data/splits/phash-clusters.csv")
OUT_DIR = Path("data/processed/label-review/clusters")

# 缩略基准格:列数多时按比例缩小,保持 4:3
BASE_CELL_W, CELL_RATIO = 320, 0.75
# 标题条高度(基准格宽下;格缩小时同比例缩)
BASE_TITLE_H = 44


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pHash 簇校验总览图生成")
    parser.add_argument("--csv", default=str(CSV_PATH), help="聚类结果 CSV(默认 data/splits/phash-clusters.csv)")
    parser.add_argument("--out", default=str(OUT_DIR), help="产出目录(默认 data/processed/label-review/clusters)")
    parser.add_argument("--min-size", type=int, default=2, help="最小簇成员数,小于它的簇不出图(默认 2)")
    parser.add_argument("--max-cells", type=int, default=60, help="单张 sheet 最多多少格,超出拆 part(默认 60)")
    parser.add_argument("--top", type=int, default=0, help="只出成员数前 M 大的簇,0=全部(默认 0)")
    return parser.parse_args()


def imread_unicode(path: Path) -> np.ndarray | None:
    """cv2.imread 在 Windows 中文路径下读不出,改用 fromfile+imdecode;读不出返回 None。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def short_label(source: str, relpath: str) -> str:
    """把 CSV 里的相对路径缩成「数据集名/拆分/文件名」,方便画进格子里。"""
    parts = relpath.replace("\\", "/").split("/")
    if parts and parts[0].lower() == "images":  # 去掉无信息量的 images 前缀
        parts = parts[1:]
    if len(parts) > 2:  # 层级仍太深时只留首尾(scene1/Annotated/images/x.png -> scene1/x.png)
        parts = [parts[0], parts[-1]]
    return "/".join([source, *parts])


def make_sheet(
    cluster_id: str,
    members: list[tuple[str, str]],  # (source, relpath)
    part_no: int,
    part_total: int,
    max_cells: int,
    skipped: dict[str, int],
) -> np.ndarray:
    """把一个簇的一段成员拼成一张 sheet,返回 BGR 图。读不出的成员跳过并计数。"""
    n = len(members)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    # 自适应格宽:列越多格越小,但不小于 120px
    cell_w = max(120, min(BASE_CELL_W, int(BASE_CELL_W * 4 / cols)))
    cell_h = int(cell_w * CELL_RATIO)
    title_h = max(28, int(BASE_TITLE_H * cell_w / BASE_CELL_W))
    font_scale = max(0.32, 0.45 * cell_w / BASE_CELL_W)
    path_scale = max(0.30, 0.38 * cell_w / BASE_CELL_W)
    thickness = max(1, round(font_scale * 2))

    sheet = np.zeros((title_h + rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    title = f"cluster {cluster_id}  n={n}"
    if part_total > 1:
        title += f"  (part {part_no}/{part_total})"
    cv2.putText(sheet, title, (8, int(title_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 1.1, (255, 255, 255), thickness + 1, cv2.LINE_AA)
    cv2.line(sheet, (0, title_h), (sheet.shape[1], title_h), (90, 90, 90), 1)

    for i, (source, relpath) in enumerate(members):
        img = imread_unicode(RAW_ROOT / source / relpath)
        if img is None:
            skipped[source] += 1
            continue
        r, c = divmod(i, cols)
        x0, y0 = c * cell_w, title_h + r * cell_h
        ih, iw = img.shape[:2]
        scale = min(cell_w / iw, cell_h / ih)
        thumb = cv2.resize(img, (max(1, int(iw * scale)), max(1, int(ih * scale))),
                           interpolation=cv2.INTER_AREA)
        # 居中粘贴,不足补黑边
        ox, oy = x0 + (cell_w - thumb.shape[1]) // 2, y0 + (cell_h - thumb.shape[0]) // 2
        sheet[oy:oy + thumb.shape[0], ox:ox + thumb.shape[1]] = thumb
        label = short_label(source, relpath)
        # 字号超出格宽时按比例缩小,保证路径完整落在格内
        scale = path_scale
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        limit = cell_w - 8
        if tw > limit:
            scale = max(0.22, scale * limit / tw)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        lx, ly = x0 + cell_w - tw - 4, y0 + cell_h - 5
        # 半透明黑底衬字,浅色图上也可读
        cv2.rectangle(sheet, (lx - 2, ly - th - 3), (x0 + cell_w, y0 + cell_h), (0, 0, 0), -1)
        cv2.putText(sheet, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 1, cv2.LINE_AA)
    return sheet


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    """cv2.imwrite 在 Windows 中文路径下会静默失败,改用 imencode+tofile。"""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise SystemExit(f"编码失败:{path.name}")
    buf.tofile(str(path))


def main() -> None:
    args = parse_args()
    csv_path, out_dir = Path(args.csv), Path(args.out)
    if not csv_path.exists():
        raise SystemExit(f"聚类结果不存在:{csv_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 按簇归组;utf-8-sig 兼容带 BOM 的表头
    clusters: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            clusters[row["cluster_id"]].append((row["source"], row["relpath"]))

    # 只留多图簇,按成员数从大到小;--top 截前 M 个
    multi = sorted(
        ((cid, mem) for cid, mem in clusters.items() if len(mem) >= args.min_size),
        key=lambda kv: len(kv[1]), reverse=True,
    )
    if args.top > 0:
        multi = multi[: args.top]
    print(f"多图簇 {len(multi)} 个,开始出图 -> {out_dir}")

    skipped: dict[str, int] = defaultdict(int)
    index_lines: list[str] = []
    for cid, members in multi:
        parts = [members[i:i + args.max_cells] for i in range(0, len(members), args.max_cells)]
        width = max(2, len(str(len(parts))))  # part 序号补零对齐
        sheet_names = []
        for pno, chunk in enumerate(parts, 1):
            sheet = make_sheet(cid, chunk, pno, len(parts), args.max_cells, skipped)
            name = f"cluster_{cid}.jpg" if len(parts) == 1 else f"cluster_{cid}_part{pno:0{width}d}.jpg"
            imwrite_unicode(out_dir / name, sheet)
            sheet_names.append(name)
        shown = f"{', '.join(sheet_names)}" if len(sheet_names) == 1 \
            else f"{sheet_names[0]} .. {sheet_names[-1]} 共 {len(sheet_names)} 张"
        index_lines.append(f"簇 {cid}  成员 {len(members)} 张  ->  {shown}")

    covered = sum(len(m) for _, m in multi)
    index_lines.append("")
    index_lines.append(f"共 {len(multi)} 个多图簇(>={args.min_size} 张),覆盖 {covered} 张图。")
    total_skip = sum(skipped.values())
    if total_skip:
        by_src = ", ".join(f"{src}: {n}" for src, n in sorted(skipped.items()))
        index_lines.append(f"读不出/跳过 {total_skip} 张({by_src})。")
    (out_dir / "index.txt").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    print(f"完成:{len(multi)} 个簇,覆盖 {covered} 张图;"
          f"读不出跳过 {total_skip} 张;索引见 {(out_dir / 'index.txt')}")


if __name__ == "__main__":
    main()
