"""Create a zero-copy, human-label-organized view of the reserved external videos."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data/metadata/external-videos"
FROZEN = BASE / "final-evaluation-20261105-v1"
OUT = ROOT / "data/raw/external-videos-labeled"
META = BASE / "labeled-view-20261105-v1"


def build() -> dict:
    if OUT.exists() or META.exists():
        raise FileExistsError("View already exists; inspect it rather than overwriting")
    annotation_bytes = (BASE / "human-review-20261104-v1/annotations.json").read_bytes()
    if annotation_bytes != (FROZEN / "human-annotations.snapshot.json").read_bytes():
        raise ValueError("Human annotations changed since reservation; reconcile them first")
    annotations = json.loads(annotation_bytes)
    with (FROZEN / "media.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({r["video_id"] for r in rows}) == 29
    names = {"positive": "positive", "negative": "negative",
             "exclude": "excluded", "uncertain": "uncertain"}
    source_devices = {}
    for row in rows:
        source = (ROOT / row["source_path"]).resolve(strict=True)
        if source.stat().st_size != int(row["size_bytes"]):
            raise ValueError("Media size changed: " + row["video_id"])
        source_devices[row["video_id"]] = source.stat().st_dev
        if row["video_id"] in annotations["annotations"]:
            assert annotations["annotations"][row["video_id"]]["label"] == row["video_label"]
    # Only the two retained DeFlow files are on D:. Link their common directory.
    cross_volume = [r for r in rows if source_devices[r["video_id"]] != OUT.parent.stat().st_dev]
    assert {r["video_id"] for r in cross_volume} == {"v028", "v029"}
    historical_parent = (ROOT / cross_volume[0]["source_path"]).parent.resolve()
    assert all((ROOT / r["source_path"]).parent.resolve() == historical_parent for r in cross_volume)
    assert {p.name for p in historical_parent.iterdir() if p.is_file()} == {"cam1.mp4", "cam2.mp4"}

    OUT.mkdir()
    for name in names.values():
        (OUT / name).mkdir()
    junction = OUT / "positive/DeFlow_历史复测"
    quote = lambda p: "'" + str(p).replace("'", "''") + "'"
    command = ("New-Item -ItemType Junction -Path " + quote(junction)
               + " -Target " + quote(historical_parent) + " | Out-Null")
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], check=True,
                   capture_output=True, text=True)

    index = []
    for row in rows:
        source = ROOT / row["source_path"]
        if row in cross_volume:
            destination = junction / source.name
            method = "directory_junction"
        else:
            destination = OUT / names[row["video_label"]] / (row["video_id"] + "__" + source.name)
            os.link(source, destination)
            method = "same_volume_hardlink"
        assert os.path.samefile(source, destination), row["video_id"]
        label = annotations["annotations"].get(row["video_id"], {})
        index.append({**row, "organized_path": destination.relative_to(ROOT).as_posix(),
                      "link_method": method,
                      "quality_flags_json": json.dumps(label.get("quality_flags", []), ensure_ascii=False)})
    META.mkdir()
    manifest = META / "media.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    (META / "human-annotations.snapshot.json").write_bytes(annotation_bytes)
    os.link(manifest, OUT / "标签与备注.csv")
    summary = {"view": OUT.relative_to(ROOT).as_posix(), "annotation_revision": annotations["revision"],
               "label_counts": dict(Counter(r["video_label"] for r in rows)),
               "role_counts": dict(Counter(r["evaluation_role"] for r in rows)),
               "video_hardlinks": len(rows) - len(cross_volume), "video_directory_links": len(cross_volume),
               "new_video_bytes": 0, "source_media_changed": False, "frozen_membership_changed": False,
               "same_file_identity_checks": len(index),
               "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest()}
    (META / "checks.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (META / "README.md").write_text(
        "# 人工标签分类入口\n\n"
        "统一入口：`data/raw/external-videos-labeled/`，使用人工revision167。\n\n"
        "- `positive/`：9条独立主外测预留正视频；其中子目录`DeFlow_历史复测/`另含2条历史曝光正视频。\n"
        "- `negative/`：14条负视频。\n"
        "- `excluded/`：人工排除3条；`uncertain/`：待定1条（v027）。\n"
        "- `标签与备注.csv`：完整标签、原路径、查看路径、正区间、备注、旋转记录和评价身份。\n\n"
        "原片采用同卷硬链接，DeFlow采用目录联接，没有复制大视频。只通过入口观看，"
        "不要直接覆盖或编辑视频内容，否则会改变原片。分类整理没有裁剪、旋转或重标视频，"
        "白线、画幅和v027旋转等备注仍保留在清单中。\n\n"
        "独立主外测仍为23条（9正14负），DeFlow另报历史来源补充复测。"
        "本目录是人工标签查看入口，不授权递归扫描所有文件作为独立测试集。"
        "冻结评价身份以`../final-evaluation-20261105-v1/media.csv`为准。\n",
        encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
