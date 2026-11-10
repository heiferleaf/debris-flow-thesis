# -*- coding: utf-8 -*-
"""Sample existing audit JPEGs across every declared interval, without changing labels."""
from __future__ import annotations

from collections import defaultdict, Counter
import argparse
import base64
import csv
import hashlib
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]


def build(groups: list[int], out: Path, gallery: Path) -> dict:
    out, gallery = out.resolve(), gallery.resolve()
    if out.exists() or gallery.exists():
        raise FileExistsError("Choose a new version rather than overwrite a review")
    source_rows = []
    snapshots = {}
    for group in groups:
        folder = ROOT / f"data/process/video_frames/video{group}_segment"
        raw = (folder / "events.csv").read_bytes()
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        snapshots[str(group)] = {"path": (folder / "events.csv").relative_to(ROOT).as_posix(),
                                 "sha256": hashlib.sha256(raw).hexdigest(), "raw": raw}
        indexed = defaultdict(list)
        with (folder / "frames_index.csv").open(encoding="utf-8-sig", newline="") as stream:
            for frame in csv.DictReader(stream):
                indexed[frame["video_key"]].append({"time": float(frame["t_sec"]),
                    "path": folder / "frames" / frame["frame_file"]})
        for frames in indexed.values():
            frames.sort(key=lambda f: f["time"])
        interval_numbers = Counter()
        for line, row in enumerate(rows, 2):
            if row["confidence"] != "real":
                continue
            unit = row["video_key"]
            interval_numbers[unit] += 1
            a, b = float(row["start_sec"]), float(row["end_sec"])
            available = [f for f in indexed[unit] if a <= f["time"] < b and f["path"].is_file()]
            if not available:
                raise ValueError("No saved audit frames in interval: " + unit + " " + str(line))
            margin = min(1., (b - a) / 10)
            targets = [a + margin, a + .25 * (b - a), a + .5 * (b - a), a + .75 * (b - a), b - margin]
            picked = []
            for t in targets:
                candidates = [f for f in available if f not in picked]
                if not candidates:
                    break
                picked.append(min(candidates, key=lambda f: abs(f["time"] - t)))
            picked.sort(key=lambda f: f["time"])
            extras = row.get(None, [])
            note = ",".join([row.get("note", ""), *extras]).strip(",")
            source_rows.append({"interval_id": f"{unit}-I{interval_numbers[unit]:02d}",
                "unit_id": unit, "video_id": row["video_id"], "kind": row["kind"],
                "start": a, "end": b, "event_line": line, "group": group,
                "note": note, "frames": picked})
    out.mkdir(parents=True)
    gallery.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 21)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
    panels = defaultdict(list)
    manifest = []
    articles = []
    for item in sorted(source_rows, key=lambda r: (r["group"], r["unit_id"], r["start"])):
        panel = Image.new("RGB", (1800, 260), "white")
        draw = ImageDraw.Draw(panel)
        title = f"{item['interval_id']} | {item['kind']} | [{item['start']:g}, {item['end']:g})s"
        draw.text((8, 3), title, fill="#182c3b", font=font)
        cards = []
        for j, frame in enumerate(item["frames"]):
            with Image.open(frame["path"]) as original:
                size = original.size
                thumbnail = original.convert("RGB")
                thumbnail.thumbnail((350, 195), Image.Resampling.LANCZOS)
            panel.paste(thumbnail, (j * 360 + 5, 31))
            draw.text((j * 360 + 7, 231), f"t={frame['time']:.2f}s  {size[0]}x{size[1]}", fill="#425b6c", font=small)
            memory = io.BytesIO()
            thumbnail.save(memory, format="JPEG", quality=88)
            data = base64.b64encode(memory.getvalue()).decode("ascii")
            full_uri = frame["path"].resolve().as_uri()
            cards.append(f'<a href="{html.escape(full_uri)}" target="_blank" rel="noopener"><img src="data:image/jpeg;base64,{data}" alt="{html.escape(title)} {frame["time"]:.2f}秒"><span>{frame["time"]:.2f}秒 · 点击看原图</span></a>')
            manifest.append({**{k: v for k, v in item.items() if k != "frames"},
                "frame_time": frame["time"], "frame_path": frame["path"].relative_to(ROOT).as_posix(),
                "width": size[0], "height": size[1]})
        camera = item["video_id"].split("_")[0]
        panels[f'v{item["group"]}-{camera}'].append(panel)
        articles.append(f'<section data-kind="{item["kind"]}"><h2>{html.escape(title)}</h2><div class="frames">{"".join(cards)}</div></section>')
    for name, rows in panels.items():
        contact = Image.new("RGB", (1800, 260 * len(rows)), "white")
        for i, panel in enumerate(rows):
            contact.paste(panel, (0, i * 260))
        contact.save(out / (name + ".jpg"), quality=95)
    counts = dict(Counter(r["kind"] for r in source_rows))
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v4、v5逐区间抽帧复核</title><style>
body{margin:0;background:#f5f7f9;color:#182c3b;font-family:"Microsoft YaHei",sans-serif}header{padding:20px 3vw;background:white;border-bottom:1px solid #ccd7df}h1{font-size:25px;margin:0 0 12px}p{line-height:1.8;margin:5px 0}main{padding:15px 2vw}section{padding:15px;background:white;border:1px solid #dce3e8;margin:0 0 15px;border-radius:6px}h2{font-size:18px;margin:0 0 14px}.frames{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}.frames a{display:flex;flex-direction:column;align-items:center;justify-content:space-between;color:#236e88;font-size:13px;text-decoration:none}.frames img{max-width:100%;max-height:210px;object-fit:contain}.frames span{margin-top:8px}@media(max-width:850px){.frames{grid-template-columns:repeat(2,minmax(0,1fr))}}</style></head><body><header><h1>v4、v5逐区间抽帧复核</h1><p>覆盖全部标注区间，每段取近起点、四分之一、中间、四分之三和近终点的5帧；使用已有审计JPEG，不根据模型分数选样。</p><p>pos＝正区间，neg＝负区间，cut＝已排除区间。静态抽样用于发现明显不一致，不能保证区间内每一时刻都已核实。点击缩略图可打开原分辨率画面。</p></header><main>''' + "".join(articles) + "</main></body></html>"
    gallery.write_text(page, encoding="utf-8")
    for group, snapshot in snapshots.items():
        (out / f"video{group}-events.snapshot.csv").write_bytes(snapshot.pop("raw"))
    with (out / "sampled-frames.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(manifest)
    summary = {"intervals": len(source_rows), "videos": len({r["unit_id"] for r in source_rows}),
        "interval_counts": counts, "frames": len(manifest), "source_snapshots": snapshots,
        "gallery": gallery.relative_to(ROOT).as_posix(),
        "contact_sheets": [p.relative_to(ROOT).as_posix() for p in sorted(out.glob("*.jpg"))],
        "labels_changed": False, "model_scores_used": False}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.groups, args.out, args.gallery), ensure_ascii=False, indent=2))
