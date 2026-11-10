"""剖析"文件夹形式"的 YOLO 数据集（images/ + labels/）。

给出：文件名前缀分布、类别 id 分布、每图框数、类别名（classes.txt / data.yaml）、样例图片分辨率。

用法：
    .venv\\Scripts\\python.exe experiments\\tools\\oneoff\\profile_folder_dataset.py "C:/Users/20443/Downloads/Debris-flow dataset"
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="剖析文件夹形式的 YOLO 数据集")
    parser.add_argument("root", help="数据集根目录（含 images/ 与 labels/ 或直接放图片）")
    parser.add_argument("--sample", type=int, default=20, help="抽样读取分辨率的图片数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"目录不存在：{root}")

    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    labels = sorted(p for p in root.rglob("*.txt") if p.name not in {"classes.txt"})
    print(f"根目录：{root}")
    print(f"图片 {len(images)} 张；标签 {len(labels)} 个\n")

    # 1) 文件名前缀（第一个下划线之前的部分）
    prefixes = Counter()
    for p in images:
        stem = p.stem
        prefixes[re.split(r"[_\-\s]", stem)[0]] += 1
    print("== 文件名前缀（可能的类别线索） ==")
    for name, count in prefixes.most_common(12):
        print(f"  {name:28s} {count}")

    # 2) 类别名文件
    print("\n== 找到的类别/配置文件 ==")
    found = False
    for candidate in list(root.rglob("classes.txt")) + list(root.rglob("*.yaml")) + list(root.rglob("*.yml")) + list(root.rglob("README*")):
        found = True
        print(f"  --- {candidate.relative_to(root)} ---")
        text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
        print("  " + text.replace("\n", "\n  ")[:1200])
    if not found:
        print("  （无 classes.txt / yaml / README）")

    # 3) 标签里的类别 id 与每图框数
    class_ids: Counter[int] = Counter()
    boxes_per_file: Counter[int] = Counter()
    empty_labels = 0
    for label in labels:
        rows = [r for r in label.read_text(encoding="utf-8", errors="ignore").splitlines() if r.strip()]
        if not rows:
            empty_labels += 1
        boxes_per_file[len(rows)] += 1
        for row in rows:
            parts = row.split()
            if parts and parts[0].lstrip("-").isdigit():
                class_ids[int(parts[0])] += 1
    print("\n== 标签统计 ==")
    print(f"  标签文件 {len(labels)} 个；空标签 {empty_labels} 个")
    print(f"  类别 id 分布：{dict(class_ids)}")
    print(f"  每图框数分布（前 8）：{dict(boxes_per_file.most_common(8))}")

    # 4) 抽样分辨率
    print("\n== 样例图片分辨率（抽样） ==")
    step = max(len(images) // args.sample, 1)
    sizes: Counter[tuple[int, int]] = Counter()
    for p in images[::step][: args.sample]:
        try:
            with Image.open(p) as im:
                sizes[im.size] += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  读取失败 {p.name}: {exc}")
    for size, count in sizes.most_common():
        print(f"  {size[0]}×{size[1]}  ×{count}")


if __name__ == "__main__":
    main()
