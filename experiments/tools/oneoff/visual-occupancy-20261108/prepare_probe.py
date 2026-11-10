"""Prepare twelve fixed-view diagnostic triplets; no class labels are created."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "data/metadata/annotation-review/visual-occupancy-20261108-v1"
CASES = [("v1_16", 300), ("v1_16", 1036.04), ("v1_16", 1040.23), ("v1_16", 1200),
         ("v1_01", 20), ("v1_01", 60), ("v1_01", 140), ("v1_01", 800),
         ("v4_08", 20), ("v4_08", 300), ("v4_08", 600), ("v4_08", 1800)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    triplets = ROOT / ".codex/docs/.temp/visual-occupancy-20261108"
    triplets.mkdir(parents=True, exist_ok=True)
    units = {}
    for key, _ in CASES:
        source = ROOT / "data/process/video_frames" / ("video" + key[1] + "_segment")
        if key not in units:
            with (source / "frames_index.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = [row for row in csv.DictReader(stream) if row.get("video_key") == key]
            rows.sort(key=lambda row: float(row["t_sec"]))
            units[key] = (source, rows)
    items = []
    for number, (key, target) in enumerate(CASES, 1):
        source, rows = units[key]
        index = min(range(len(rows)), key=lambda i: abs(float(rows[i]["t_sec"]) - target))
        chosen = rows[max(0, index-1):min(len(rows), index+2)]
        central = rows[index]
        paths = [source / "frames" / row["frame_file"] for row in chosen]
        canvas = Image.new("RGB", (640*len(chosen), 388), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (path, row) in enumerate(zip(paths, chosen)):
            im = Image.open(path).convert("RGB"); im.thumbnail((640, 360))
            canvas.paste(im, (i*640, 0)); draw.text((i*640+8, 365), f"t={row['t_sec']} s", fill="black")
        item_id = f"V{number:02d}"
        panel = triplets / (item_id + ".jpg"); canvas.save(panel, quality=91)
        frame = source / "frames" / central["frame_file"]
        with Image.open(frame) as im:
            size = list(im.size)
        items.append({"item_id":item_id, "camera_candidate": key, "unit_id":key,
                      "t_sec":float(central["t_sec"]), "central_frame":frame.relative_to(ROOT).as_posix(),
                      "central_frame_sha256":hashlib.sha256(frame.read_bytes()).hexdigest(),
                      "size":size, "triplet":panel.relative_to(ROOT).as_posix(),
                      "context":[{"path":p.relative_to(ROOT).as_posix(),"t_sec":float(row["t_sec"])}
                                 for p,row in zip(paths,chosen)]})
    document = {"purpose":"Bounded feasibility probe for visual target occupancy, not training labels or validation performance",
                "ROI_rule":"One fixed visible active-channel-bed polygon per camera, including currently dry channel bed, excluding banks/vegetation/structures and fixed occlusion. No target-following ROI",
                "mask_rule":"Trace only visible moving target material supported by the triplet; stationary deposits and ordinary water are not automatically target. If type/extent cannot be judged, mark unresolved rather than invent a mask",
                "coordinates":"Normalized [0,1] x/y in original central frame; same ROI for all four frames of a camera",
                "training_ready":False, "items":items}
    (OUT / "blind-manifest.json").write_text(json.dumps(document,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"items":len(items),"camera_candidates":len(units),"output":str(OUT)}))


if __name__ == "__main__":
    main()
