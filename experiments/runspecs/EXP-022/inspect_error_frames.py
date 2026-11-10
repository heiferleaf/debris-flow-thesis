"""Extract only representative saved-score windows for visual diagnosis, without model calls."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "data/results/exp022/error-diagnosis-20261105-v1"
OUT = ROOT / "data/process/EXP-022/error-diagnosis-20261105-v1"


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def build() -> dict:
    if OUT.exists():
        raise FileExistsError("Inspection output already exists")
    requests = csv_rows(ANALYSIS / "representative-windows.csv")
    targets = {(r["unit_id"], round(float(r["target_start_seconds"]), 6)): r for r in requests}
    dense = {}
    with (ROOT / "data/process/EXP-017/dense-pixel-reference-v1/dense-windows.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = row["unit_id"], round(float(row["target_start_seconds"]), 6)
            if key in targets:
                dense[key] = row
    assert targets.keys() <= dense.keys()
    sources = {r["unit_id"]: r for r in csv_rows(ROOT / "data/metadata/EXP-017/role-free-plan-v3/sources.csv")}
    OUT.mkdir(parents=True)
    (OUT / "frames").mkdir()
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 13)
    records = []
    sheets = []
    for number, request in enumerate(requests):
        unit = request["unit_id"]
        target = dense[unit, round(float(request["target_start_seconds"]), 6)]
        tile = Image.new("RGB", (810, 200), "white")
        draw = ImageDraw.Draw(tile)
        draw.text((7, 3), f"{unit} {request['outcome']} / {request['selection']} / p={float(request['score']):.4f}", fill="#182c3b", font=font)
        capture = None
        for col, ti in enumerate([0, 7, 15]):
            index = target["frame_indices"][ti]
            source = sources[unit]
            cached = ROOT / source["reused_cache_directory"] / f"frame-{index:08d}.jpg"
            if cached.is_file():
                frame = cv2.imread(str(cached), cv2.IMREAD_COLOR)
                method = "existing_verified_jpeg"
                origin = cached
                reported = None
            else:
                origin = ROOT / source["middleware_path"]
                if capture is None:
                    capture = cv2.VideoCapture(str(origin))
                    if not capture.isOpened():
                        raise ValueError("Cannot open middleware: " + unit)
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                    raise ValueError("Frame seek failed: " + unit)
                ok, frame = capture.read()
                reported = capture.get(cv2.CAP_PROP_POS_FRAMES)
                if not ok or abs(reported - (index + 1)) > .01:
                    raise ValueError("Frame ordinal mismatch: " + unit)
                method = "middleware_ordinal_seek_for_visual_inspection_only"
            if frame is None:
                raise ValueError("Empty representative frame: " + unit)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            asset = OUT / "frames" / f"{unit}-{index:08d}.png"
            if not asset.exists():
                Image.fromarray(rgb).save(asset)
            image = Image.fromarray(rgb)
            image.thumbnail((260, 142))
            tile.paste(image, (col * 270 + 5, 31))
            time = target["source_pts_seconds"][ti]
            draw.text((col * 270 + 7, 175), f"t={time:.3f}s  frame={index}", fill="#485965", font=small)
            records.append({"unit_id": unit, "selection": request["selection"],
                            "target_start_seconds": float(request["target_start_seconds"]),
                            "score": float(request["score"]), "frame_index": index,
                            "declared_source_pts_seconds": time, "read_method": method,
                            "origin": origin.relative_to(ROOT).as_posix(),
                            "opencv_reported_next_frame": reported,
                            "gray_std": float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).std()),
                            "asset": asset.relative_to(ROOT).as_posix()})
        if capture:
            capture.release()
        sheets.append((request, tile))
        if (number + 1) % 10 == 0:
            print(f"Representative windows {number + 1}/{len(requests)}", flush=True)
    groups = {"false-negatives": [s for s in sheets if s[0]["outcome"] == "FN"],
              "controls": [s for s in sheets if s[0]["selection"] == "existing_positive_control"]}
    positives = [s for s in sheets if s[0]["outcome"] == "FP"]
    for i in range(0, len(positives), 8):
        groups[f"false-positives-{i // 8 + 1}"] = positives[i:i + 8]
    for name, rows in groups.items():
        if not rows:
            continue
        sheet = Image.new("RGB", (810, 200 * len(rows)), "white")
        for i, (_, tile) in enumerate(rows):
            sheet.paste(tile, (0, i * 200))
        sheet.save(OUT / f"{name}.jpg", quality=94)
    result = {"representative_windows": len(requests), "frame_records": len(records),
              "unique_frame_assets": len(list((OUT / "frames").glob("*.png"))),
              "read_methods": {m: sum(r["read_method"] == m for r in records) for m in {r["read_method"] for r in records}},
              "scope": "Visual diagnosis at declared frozen frame ordinals. Static sheets do not establish physical motion, velocity, geological truth or whole-dataset label accuracy.",
              "records": records}
    (OUT / "inspection-manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {k: v for k, v in result.items() if k != "records"}


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
