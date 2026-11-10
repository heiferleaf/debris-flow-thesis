"""从 pHash 簇清单生成图像部分的划分清单(整簇不跨侧)。

长期工具(2026-09-16 由 oneoff/ 提升):训练划分原子定义,新图入集/重新聚簇后
重跑生成划分清单。簇清单来源为 data/final/images/clusters/phash-clusters.csv。

规则:
  1. 含 sceneN 成员的簇跟随该 scene 的归属(跨库簇与 scene 同侧,防跨库泄漏);
     一个簇含多个 scene 视为聚类异常,报错。
  2. 其余独立簇按图像量做贪心均衡分配(train/val/test 按 --ratio 加权)。
  3. scene → split 的映射由 --scene-split 给出(默认按 EXP-002 建议 2/1/1)。

输出进 data/metadata/splits/(只含 ID,进 git);video 事件的划分待 FTP 视频标注后另出。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.paths import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 pHash 簇生成图像划分清单")
    parser.add_argument("--clusters", default="data/final/images/clusters/phash-clusters.csv",
                        help="pHash 簇 CSV(默认新 data 结构 final/images/clusters/;旧 data/splits/ 已并入)")
    parser.add_argument("--out", default="data/metadata/splits/images-v1.json")
    parser.add_argument("--name", default="images-v1")
    parser.add_argument(
        "--scene-split", default="scene1=train,scene2=train,scene3=val,scene4=test",
        help="scene 到 split 的映射,逗号分隔",
    )
    parser.add_argument("--ratio", default="train=0.7,val=0.15,test=0.15", help="剩余簇的目标比例")
    return parser.parse_args()


def parse_kv(text: str, sep: str) -> dict[str, str]:
    out = {}
    for item in text.split(","):
        key, _, value = item.partition(sep)
        if not key or not value:
            raise SystemExit(f"参数格式错误:{item}")
        out[key.strip()] = value.strip()
    return out


def main() -> None:
    args = parse_args()
    args.clusters = str(resolve_path(args.clusters, must_exist=True))
    args.out = str(resolve_path(args.out))
    scene_split = parse_kv(args.scene_split, "=")
    ratio = {k: float(v) for k, v in parse_kv(args.ratio, "=").items()}
    if set(ratio) != {"train", "val", "test"}:
        raise SystemExit("--ratio 必须同时给出 train/val/test")

    clusters: dict[int, list[dict[str, str]]] = defaultdict(list)
    with Path(args.clusters).open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            clusters[int(row["cluster_id"])].append(
                {"source": row["source"], "official_group": row["official_group"], "relpath": row["relpath"]}
            )

    # 逐簇归属
    scene_of_cluster: dict[int, str] = {}
    pending: list[int] = []
    for cid, members in clusters.items():
        scenes = {m["official_group"] for m in members if m["official_group"].startswith("scene")}
        if len(scenes) > 1:
            raise SystemExit(f"簇 {cid} 含多个 scene {sorted(scenes)},聚类链式合并越界,须先处理")
        if scenes:
            scene_of_cluster[cid] = scenes.pop()
        else:
            pending.append(cid)

    # 剩余簇贪心均衡:按图像数降序,放进当前「加权图像量」最小的 split
    weight = {"train": ratio["train"], "val": ratio["val"], "test": ratio["test"]}
    load = {"train": 0.0, "val": 0.0, "test": 0.0}
    counts = {k: 0 for k in load}
    assign: dict[int, str] = {}
    for cid in sorted(pending, key=lambda c: (-len(clusters[c]), -c)):
        target = min(load, key=lambda k: load[k] / weight[k])
        assign[cid] = target
        counts[target] += 1
        load[target] += len(clusters[cid])

    def split_of(cid: int) -> str:
        scene = scene_of_cluster.get(cid)
        return scene_split[scene] if scene else assign[cid]

    # 组装清单:同一 scene 的所有簇合并成一条(划分原子是 scene=事件,video_id 必须唯一)
    entries: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    scene_cluster_ids: dict[str, list[int]] = defaultdict(list)
    for cid in sorted(scene_of_cluster):
        scene_cluster_ids[scene_of_cluster[cid]].append(cid)
    for scene, cids in sorted(scene_cluster_ids.items()):
        members = [m for cid in cids for m in clusters[cid]]
        entries[scene_split[scene]].append({
            "video_id": f"scene-{scene[5:]}",
            "label_origin": "auto",
            "location": scene,
            "n_images": len(members),
            "sources": sorted({m["source"] for m in members}),
            "cluster_ids": cids,
        })
    for cid in sorted(assign):
        members = clusters[cid]
        entries[assign[cid]].append({
            "video_id": f"cluster-{cid}",
            "label_origin": "auto",
            "location": f"cluster-{cid}",
            "n_images": len(members),
            "sources": sorted({m["source"] for m in members}),
            "cluster_ids": [cid],
        })

    manifest = {
        "name": args.name,
        "unit": "video_id",
        "task": "A",
        "splits": entries,
        "meta": {
            "created": date.today().isoformat(),
            "cluster_csv": args.clusters,
            "threshold": 10,
            "scene_map": scene_split,
            "rule": "整簇不跨侧;含 sceneN 的簇跟随 scene;同一 location 不跨 split;负样本须经人工核验后才可训练",
            "cluster_count": len(clusters),
            "pending_counts": counts,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"清单已写入 {out}")
    print(f"簇总数 {len(clusters)};scene 绑定 {len(scene_of_cluster)},独立簇分配 {counts}")
    for name in ("train", "val", "test"):
        images = sum(int(e["n_images"]) for e in entries[name])  # type: ignore[arg-type]
        print(f"  {name}: {len(entries[name])} 条,图像 {images} 张")


if __name__ == "__main__":
    main()
