"""把带 YOLO 标注的图片画框拼成大总览图(contact sheet),一眼抽查几十张标签质量。

产出(写到 data/processed/label-review/<数据集名>/):
  sheet_NNN.jpg  每张放 --sample 张小图,8 列网格;画类别框,左上角写文件名与类别 id,
                 标签为空的小图标黄字 EMPTY;白底小图也可见
  index.txt      每张 sheet 包含哪些文件(含 split 与空标签标记),便于回查

用法(仓库根目录):
    .venv\\Scripts\\python.exe experiments\\tools\\oneoff\\label_review_sheets.py ^
        data/raw/dataset2 --split test --classes data/raw/dataset2/labels/test/classes.txt
    .venv\\Scripts\\python.exe experiments\\tools\\oneoff\\label_review_sheets.py ^
        data/raw/dataset1 --only-empty --sample 32

说明:
  --root 指向含 images/ 与 labels/ 的 YOLO 目录;split 取 images/ 下的子目录
  (train/val/test 等),若 images/ 下直接是图片(无 split 层)也能处理。
  标签文件缺失或没有有效行都算空标签,照常排进总览图(用于核验负样本)。
  读不出的图(如 0 字节坏图)跳过并计数,不中断。

归档(2026-09-16 data 结构重构):标签核验总览为一次性审计件,已完成入档。
root 指向 data/raw/(不动);输出 data/processed/label-review/ → process/ 下对应位置,
判决 CSV 已迁 data/final/images/labels/。若要重跑先按新路径改 --out-root。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CELL_W, CELL_H = 320, 240  # 每格小图尺寸
MARGIN = 2                 # 格间留白
TOP_BAND = 34              # 每格顶部文字条高度

# 类别 id -> BGR 颜色,固定映射,白底图上可见
PALETTE = [
    (0, 0, 255),      # 0 红
    (0, 200, 0),      # 1 绿
    (255, 0, 0),      # 2 蓝
    (0, 255, 255),    # 3 黄
    (255, 0, 255),    # 4 品红
    (0, 165, 255),    # 5 橙
    (255, 255, 0),    # 6 青
    (0, 128, 0),      # 7 深绿
    (128, 0, 128),    # 8 紫
    (180, 105, 255),  # 9 粉
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 标注总览拼图:画框抽查标签质量")
    parser.add_argument("root", help="数据集目录(含 images/ 与 labels/,如 data/raw/dataset2)")
    parser.add_argument("--sample", type=int, default=48, help="每张总览图放几张小图,默认 48")
    parser.add_argument("--cols", type=int, default=8, help="每张总览图列数,默认 8")
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"],
                        help="只看某个 split,默认 all")
    parser.add_argument("--only-empty", action="store_true", help="只看标签为空的图(核验负样本)")
    parser.add_argument("--only-suspect", default=None,
                        help="只看指定文件的文件名列表(每行一个文件名或含扩展名)")
    parser.add_argument("--classes", default=None,
                        help="classes.txt 路径(每行一个类别名,行号=id);不传则只写 id")
    parser.add_argument("--out-root", default="data/processed/label-review",
                        help="输出根目录,默认 data/processed/label-review")
    return parser.parse_args()


def imread_unicode(path: Path) -> np.ndarray | None:
    """cv2.imread 在 Windows 中文路径下读不出,改用 fromfile+imdecode。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray, params: list[int]) -> None:
    ok, buf = cv2.imencode(path.suffix.lower(), img, params)
    if not ok:
        raise SystemExit(f"编码失败:{path.name}")
    buf.tofile(str(path))


def load_classes(path: str | None) -> dict[int, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"警告:classes 文件不存在,只写类别 id:{p}")
        return {}
    names: dict[int, str] = {}
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        name = line.strip()
        if name:
            names[i] = name
    return names


def list_splits(root: Path) -> list[str]:
    """images/ 下的子目录是 split;若直接放图片则视为无 split 层,用空串表示。"""
    img_root = root / "images"
    if not img_root.is_dir():
        raise SystemExit(f"找不到 images 目录:{img_root}")
    subs = sorted(d.name for d in img_root.iterdir() if d.is_dir())
    return subs if subs else [""]


def collect_items(root: Path, splits: list[str], args: argparse.Namespace) -> list[dict]:
    """收集 (split, 图片路径, 标签路径, 是否空标签),按 split 与文件名排序保证可复现。"""
    suspect: set[str] | None = None
    if args.only_suspect:
        p = Path(args.only_suspect)
        suspect = {line.strip().lower() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}

    items: list[dict] = []
    for split in splits:
        if args.split != "all" and split != args.split:
            continue
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS or not img_path.is_file():
                continue
            if suspect is not None and img_path.name.lower() not in suspect:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            empty = not label_has_boxes(lbl_path)
            if args.only_empty and not empty:
                continue
            items.append({"split": split, "img": img_path, "lbl": lbl_path, "empty": empty})
    return items


def label_has_boxes(lbl_path: Path) -> bool:
    """标签文件缺失、空文件或只有空行都算空标签。"""
    if not lbl_path.is_file():
        return False
    try:
        return any(line.strip() for line in lbl_path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeDecodeError):
        return False


def parse_label(lbl_path: Path) -> list[tuple[int, float, float, float, float]]:
    """读 YOLO 行,返回 (cls, cx, cy, w, h);坏行跳过。"""
    boxes: list[tuple[int, float, float, float, float]] = []
    if not lbl_path.is_file():
        return boxes
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            boxes.append((int(float(parts[0])), *(float(v) for v in parts[1:5])))
        except ValueError:
            continue
    return boxes


def render_cell(item: dict, classes: dict[int, str]) -> tuple[np.ndarray, bool]:
    """单格:小图 + 画框 + 顶部文字条(文件名 / 类别);空标签画 EMPTY。
    返回 (格子, 图片是否读出),读不出的图占灰底格并写 UNREADABLE。"""
    cell = np.full((CELL_H + TOP_BAND, CELL_W, 3), 255, dtype=np.uint8)

    img = imread_unicode(item["img"])
    if img is None:
        cell[TOP_BAND:, :] = 160
        cv2.putText(cell, "UNREADABLE", (8, TOP_BAND + CELL_H // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        readable = False
    else:
        readable = True
        small = cv2.resize(img, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        for cls, cx, cy, bw, bh in parse_label(item["lbl"]):
            color = PALETTE[cls % len(PALETTE)]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            sx1, sy1 = max(0, round(x1 * CELL_W / w)), max(0, round(y1 * CELL_H / h))
            sx2, sy2 = min(CELL_W - 1, round(x2 * CELL_W / w)), min(CELL_H - 1, round(y2 * CELL_H / h))
            if sx2 <= sx1 or sy2 <= sy1:
                continue
            cv2.rectangle(small, (sx1, sy1), (sx2, sy2), color, 2)
        cell[TOP_BAND:, :] = small

    stem = item["img"].stem
    name_txt = stem if len(stem) <= 26 else stem[:23] + "..."
    cls_ids = sorted({b[0] for b in parse_label(item["lbl"])})
    if cls_ids:
        cls_txt = ",".join(f"{c}{classes.get(c, '')}" for c in cls_ids)
    else:
        cls_txt = ""
    cv2.putText(cell, name_txt, (6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    if item["empty"]:
        cv2.putText(cell, "EMPTY", (CELL_W - 66, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.putText(cell, "EMPTY", (6, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2, cv2.LINE_AA)
    elif cls_txt:
        cv2.putText(cell, cls_txt[:44], (6, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    return cell, readable


def stack_sheet(cells: list[np.ndarray], cols: int) -> np.ndarray:
    rows = (len(cells) + cols - 1) // cols
    ch, cw = cells[0].shape[:2]
    sheet = np.full((rows * ch + (rows - 1) * MARGIN, cols * cw + (cols - 1) * MARGIN, 3), 40, dtype=np.uint8)
    for i, c in enumerate(cells):
        r, k = divmod(i, cols)
        y, x = r * (ch + MARGIN), k * (cw + MARGIN)
        sheet[y : y + ch, x : x + cw] = c
    return sheet


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"数据集目录不存在:{root}")
    classes = load_classes(args.classes)
    splits = list_splits(root)
    items = collect_items(root, splits, args)
    if not items:
        raise SystemExit("没有符合条件的图片(检查 --split / --only-empty / --only-suspect)")

    out_dir = Path(args.out_root) / root.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = args.cols
    per_sheet = args.sample
    sheet_no, unreadable, empty_total = 0, 0, 0
    index_lines: list[str] = []
    for start in range(0, len(items), per_sheet):
        batch = items[start : start + per_sheet]
        cells = []
        for it in batch:
            cell, readable = render_cell(it, classes)
            cells.append(cell)
            if not readable:
                unreadable += 1
            if it["empty"]:
                empty_total += 1
        imwrite_unicode(out_dir / f"sheet_{sheet_no:03d}.jpg",
                        stack_sheet(cells, cols), [cv2.IMWRITE_JPEG_QUALITY, 85])
        index_lines.append(f"sheet_{sheet_no:03d}.jpg")
        for it in batch:
            mark = " EMPTY" if it["empty"] else ""
            index_lines.append(f"  {it['split']}/{it['img'].name}{mark}")
        sheet_no += 1

    (out_dir / "index.txt").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    print(f"共 {len(items)} 张图,其中空标签 {empty_total} 张,读不出(跳过占灰格){unreadable} 张")
    print(f"产出 {sheet_no} 张总览图 → {out_dir}")
    print(f"索引:{out_dir / 'index.txt'}")


if __name__ == "__main__":
    main()
