"""用感知哈希(pHash)把多来源图像聚成"画面簇"。

同一场景不同时序的帧(同一场泥石流、同一机位、相邻时刻)哈希几乎相同,
聚成一簇。簇是划分的原子:整簇只能进一个 split,防止同一场景跨 train/test 泄漏。

对多个根目录一起聚类即做跨库去重(如 mudslide-photos × debrisflow-types)。

用法(仓库根目录):
    .venv\\Scripts\\python.exe experiments\\tools\\cluster_by_perceptual_hash.py ^
        data/raw/mudslide-photos data/raw/debrisflow-types data/raw/yolov8mudslide ^
        --threshold 10 --output data/final/images/clusters/phash-clusters.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

POPCOUNT = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pHash 画面簇聚类(多来源跨库去重)")
    parser.add_argument("roots", nargs="+", help="图像来源根目录,可多个")
    parser.add_argument("--threshold", type=int, default=10, help="64 位哈希的汉明距离阈值")
    parser.add_argument("--output", default="data/final/images/clusters/phash-clusters.csv",
                        help="簇清单 CSV 输出路径(新 data 结构 final/images/clusters/;旧 data/splits/ 已并入)")
    parser.add_argument("--hash-cache", default="data/final/images/clusters/phash-hashes.npz",
                        help="pHash 缓存,图片清单不变时复用(默认同 output 目录)")
    return parser.parse_args()


def phash(img_path: Path) -> np.uint64 | None:
    """32×32 灰度 → 二维 DCT → 左上 8×8 低频(去掉直流分量) → 中位数二值化 → 64 位哈希。

    缩放、亮度、轻压缩噪声不改变哈希;画面内容不同则哈希不同。
    返回 None 表示图片读不出(损坏/0 字节)。
    """
    gray = cv2.imread(str(img_path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if gray is None:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8].flatten()[1:]
    packed = 0
    for bit in dct > np.median(dct):
        packed = (packed << 1) | int(bit)
    return np.uint64(packed)


def collect_images(roots: list[Path]) -> list[tuple[str, Path, Path]]:
    """返回 (来源名, 相对路径, 绝对路径),按绝对路径排序保证可复现。"""
    items: list[tuple[str, Path, Path]] = []
    for root in roots:
        for p in root.rglob("*"):
            if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file():
                items.append((root.name, p.relative_to(root), p))
    items.sort(key=lambda t: str(t[2]).lower())
    return items


def official_group(rel: Path) -> str:
    """从相对路径推断自带划分/场景组:train|val|test 或 sceneN;都没有则空串。"""
    for part in rel.parts:
        if part in {"train", "val", "test"}:
            return part
        if part.startswith("scene") and part[5:].isdigit():
            return part
    return ""


def near_duplicate_pairs(hashes: np.ndarray, threshold: int, chunk: int = 512) -> list[tuple[int, int]]:
    """分块算两两汉明距离,返回 (i, j) 且 j > i、距离 ≤ 阈值的下标对(均为全量下标)。"""
    n = len(hashes)
    pairs: list[tuple[int, int]] = []
    for start in range(0, n, chunk):
        block = hashes[start : start + chunk]
        xor = block[:, None] ^ hashes[None, :]
        dist = np.zeros(xor.shape, dtype=np.uint8)
        for shift in (0, 16, 32, 48):
            dist += POPCOUNT[(xor >> np.uint64(shift)) & np.uint64(0xFFFF)]
        rows = start + np.arange(len(block))
        mask = (dist <= threshold) & (np.arange(n)[None, :] > rows[:, None])
        for di, dj in np.argwhere(mask):
            pairs.append((int(rows[di]), int(dj)))
        if (start // chunk) % 8 == 0:
            print(f"  距离矩阵 {min(start + chunk, n)}/{n}", flush=True)
    return pairs


def main() -> None:
    args = parse_args()
    roots = [Path(r) for r in args.roots]
    for root in roots:
        if not root.exists():
            raise SystemExit(f"目录不存在:{root}")

    print("== 收集图片 ==", flush=True)
    items = collect_images(roots)
    print(f"共 {len(items)} 张图片,开始计算 pHash", flush=True)

    # pHash 计算(带缓存:图片清单不变时直接复用,调阈值不用重算)
    cache_path = Path(args.hash_cache)
    digest = np.frombuffer(
        hashlib.sha256("\n".join(str(p) for *_, p in items).encode("utf-8")).digest(), dtype=np.uint8
    )
    hashes = np.zeros(len(items), dtype=np.uint64)
    readable = np.zeros(len(items), dtype=bool)
    unreadable: Counter[str] = Counter()
    loaded = None
    if cache_path.exists():
        loaded = np.load(cache_path)
        if not np.array_equal(loaded["digest"], digest):
            loaded = None
    if loaded is not None:
        hashes, readable = loaded["hashes"], loaded["readable"]
        unreadable = Counter(source for (source, _, _), ok in zip(items, readable) if not ok)
        print(f"哈希缓存命中:{cache_path}(读不出 {sum(unreadable.values())} 张)", flush=True)
    else:
        for idx, (source, _rel, path) in enumerate(items):
            h = phash(path)
            readable[idx] = h is not None
            if h is None:
                unreadable[source] += 1
            else:
                hashes[idx] = h
            if (idx + 1) % 2000 == 0:
                print(f"  pHash {idx + 1}/{len(items)}", flush=True)
        print(f"pHash 完成;读不出的图片:{dict(unreadable) or '无'}", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, hashes=hashes, digest=digest, readable=readable)

    print("== 聚类(汉明距离 ≤ 阈值合并,只聚读得出的图) ==", flush=True)
    readable_positions = np.flatnonzero(readable)
    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in near_duplicate_pairs(hashes[readable_positions], args.threshold):
        parent[find(int(readable_positions[i]))] = find(int(readable_positions[j]))

    members: dict[int, list[int]] = {}
    for pos in readable_positions:
        idx = int(pos)
        members.setdefault(find(idx), []).append(idx)
    # 按簇大小降序、再按代表成员路径排序,保证簇 id 稳定
    ordered = sorted(members.values(), key=lambda m: (-len(m), str(items[m[0]][2]).lower()))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "source", "official_group", "relpath"])
        for cid, member_ids in enumerate(ordered):
            for idx in member_ids:
                source, rel, _ = items[idx]
                writer.writerow([cid, source, official_group(rel), str(rel)])
    print(f"簇清单已写入 {out_path}", flush=True)

    # 汇总
    sizes = [len(m) for m in ordered]
    per_source_total: Counter[str] = Counter(source for source, _, _ in items)
    cross_group = [m for m in ordered if len({official_group(items[i][1]) for i in m}) > 1]
    cross_source = [m for m in ordered if len({items[i][0] for i in m}) > 1]
    summary = {
        "threshold": args.threshold,
        "total_images": len(items),
        "unreadable": dict(unreadable),
        "per_source_total": dict(per_source_total),
        "num_clusters": len(ordered),
        "singleton_clusters": sum(1 for s in sizes if s == 1),
        "largest_clusters": [
            {"cluster_id": cid, "size": len(m),
             "sources": dict(Counter(items[i][0] for i in m)),
             "groups": dict(Counter(official_group(items[i][1]) for i in m if official_group(items[i][1])))}
            for cid, m in enumerate(ordered[:10])
        ],
        "cross_group_clusters": len(cross_group),
        "cross_source_clusters": len(cross_source),
    }
    summary_path = out_path.with_name(out_path.stem + "-summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
