"""按 pHash 簇生成可浏览的目录树(硬链接,不复制数据)。

读取簇清单 CSV 和划分清单 JSON,在输出根目录下每簇建一个目录:
    {split}_c{簇id:04d}_n{张数}
把簇内每张图用 os.link 硬链接进去(簇内重名文件加序号后缀),
并写 index.csv(cluster_id, split, n_images, sources, relpath)。

用法(仓库根目录):
    .venv\\Scripts\\python.exe experiments/tools/oneoff/build_cluster_view.py

归档(2026-09-16 data 结构重构):簇浏览树为一次性审计件,划分定稿后已成历史。
簇清单 data/splits/phash-clusters.csv → data/final/images/clusters/phash-clusters.csv;
输出 data/processed/cluster-view → 一次性产物(.temp/)。若要重跑先按新路径改 --clusters/--out/--manifest。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按簇建立硬链接浏览树")
    parser.add_argument("--clusters", default="data/splits/phash-clusters.csv")
    parser.add_argument("--manifest", default="data/metadata/splits/images-v1.json")
    parser.add_argument("--raw-root", default="data/raw", help="图像来源根目录(内含各 source 子目录)")
    parser.add_argument("--out", default="data/processed/cluster-view")
    return parser.parse_args()


def unique_link_name(dir_path: Path, filename: str) -> Path | None:
    """在 dir_path 里找不冲突的链接名;重名加 __2、__3 序号后缀。返回 None 表示全部被占。"""
    stem, suffix = Path(filename).stem, Path(filename).suffix
    candidate = dir_path / filename
    n = 1
    while True:
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
        n += 1
        candidate = dir_path / f"{stem}__{n}{suffix}"
        if n > 9999:
            return None


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    out_root = Path(args.out)

    # 划分清单:cluster_id → split(与 build_split_manifest 同一口径)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    split_of_cluster: dict[int, str] = {}
    for split, entries in manifest["splits"].items():
        for entry in entries:
            for cid in entry["cluster_ids"]:
                if cid in split_of_cluster:
                    raise SystemExit(f"簇 {cid} 在清单中出现于多个条目,口径不一致")
                split_of_cluster[cid] = split

    # 簇清单
    clusters: dict[int, list[dict[str, str]]] = defaultdict(list)
    with Path(args.clusters).open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            clusters[int(row["cluster_id"])].append(row)

    missing = sorted(set(clusters) - set(split_of_cluster))
    if missing:
        print(f"警告:{len(missing)} 个簇不在划分清单里,跳过(如 {missing[:5]})", flush=True)

    if out_root.exists():
        raise SystemExit(f"输出目录已存在,请先手动清理:{out_root}")
    out_root.mkdir(parents=True)

    total_links = 0
    failed: list[tuple[int, str, str]] = []
    with (out_root / "index.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "split", "n_images", "sources", "relpath"])
        for cid in sorted(clusters):
            if cid not in split_of_cluster:
                continue
            split = split_of_cluster[cid]
            members = clusters[cid]
            sources = sorted({m["source"] for m in members})
            dir_name = f"{split}_c{cid:04d}_n{len(members)}"
            cluster_dir = out_root / dir_name
            cluster_dir.mkdir()
            for m in members:
                src = raw_root / m["source"] / m["relpath"]
                link = unique_link_name(cluster_dir, Path(m["relpath"]).name)
                if link is None:
                    failed.append((cid, str(src), "无可用链接名"))
                    continue
                try:
                    os.link(src, link)
                    total_links += 1
                except OSError as exc:
                    failed.append((cid, str(src), str(exc)))
            writer.writerow([cid, split, len(members), ";".join(sources), str(cluster_dir).replace("\\", "/")])
            if (cid + 1) % 200 == 0:
                print(f"  已处理 {cid + 1}/{len(clusters)} 簇", flush=True)

    n_dirs = sum(1 for p in out_root.iterdir() if p.is_dir())
    print(f"完成:{n_dirs} 个簇目录,{total_links} 个硬链接", flush=True)
    if failed:
        print(f"失败 {len(failed)} 个链接,前 10 条:", flush=True)
        for cid, src, err in failed[:10]:
            print(f"  簇 {cid}: {src} — {err}", flush=True)


if __name__ == "__main__":
    main()
