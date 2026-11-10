"""Snapshot current annotations and make limited cached-frame evidence sheets.

Never edits annotations or training membership. All sample times are audit times.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "data/process/annotation-recheck/20261110-user-edits-v1/root"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def paired_visuals():
    dest = OUT.parent / "paired-visual"
    dest.mkdir(exist_ok=True)
    indexes = {}
    for number in (4, 5):
        folder = ROOT / "data/process/video_frames" / ("video{}_segment".format(number))
        for row in read_csv(folder / "frames_index.csv"):
            indexes.setdefault(row["video_key"], []).append((float(row["t_sec"]), folder / "frames" / row["frame_file"]))
    pairs = [
        ("v5_15", [701,703,705,707], "v4_07", [104,106,108,110], [0.35,0.35,0.73,0.84], [0.20,0.28,0.80,0.63], "D12同站点不同视向，未完成几何配准"),
        ("v5_16", [703,705,707,709], "v4_07", [104,106,108,110], [0.35,0.35,0.73,0.84], [0.20,0.28,0.80,0.63], "D12同站点不同视向，未完成几何配准"),
    ]
    for key, points in [("v5_17",[1022,1024,1026,1028]),("v5_18",[1034,1036,1038,1040]),("v5_19",[1038,1040,1042,1044]),("v5_20",[1052,1054,1056,1058])]:
        pairs.append((key, points, "v4_12", [170,172,174,176], [0.25,0.32,0.80,0.90], [0.25,0.32,0.80,0.90], "D13同视向河槽；不同时间过程，不是同步测量"))
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    result = []
    for key, points, reference, reference_points, roi, reference_roi, note in pairs:
        board = Image.new("RGB", (1440, 640), "white")
        draw = ImageDraw.Draw(board)
        draw.text((8,5), key + " 与当前正区间对照；" + note, font=font, fill="black")
        draw.text((8,33), "每行间隔约2秒的真实缓存图。只做裁剪与等比缩放，不补帧、不生成图。", font=font, fill="black")
        witnesses = []
        for j, (unit, times, crop) in enumerate([(key, points, roi),(reference, reference_points, reference_roi)]):
            for i, target in enumerate(times):
                sec, path = min(indexes[unit], key=lambda item: abs(item[0]-target))
                with Image.open(path) as im:
                    width, height = im.size
                    box = tuple(int(round(crop[k]*(width if k%2==0 else height))) for k in range(4))
                    im = im.convert("RGB").crop(box)
                    im.thumbnail((354,247))
                    x, y = 360*i+(360-im.width)//2, 92+j*272+(247-im.height)//2
                    board.paste(im,(x,y))
                draw.text((360*i+5,65+j*272), "{}  审计 {:.2f}秒".format(unit,sec), font=font, fill="black")
                witnesses.append(dict(video_key=unit,requested_audit_sec=target,actual_audit_sec=sec,path=str(path.relative_to(ROOT)),roi_normalized=crop,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        filename=key+"-same-site-P-comparison.jpg"
        board.save(dest/filename,quality=93)
        result.append(dict(video_key=key,reference=reference,note=note,evidence=filename,samples=witnesses))
    (dest/"manifest.json").write_text(json.dumps(dict(status="visual_comparison_not_label_certification",pairs=result,limits="Cached 2-second samples do not establish solid concentration, physical velocity or whole-video identity."),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(output=str(dest),images=[r["evidence"] for r in result]),ensure_ascii=False))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((ROOT / "data/metadata/annotation-revisions/current-review.json").read_text(encoding="utf-8"))
    working = ROOT / pointer["working_annotations_dir"]
    events, fingerprints = {}, []
    snapshots = OUT / "annotation-snapshot"
    snapshots.mkdir(exist_ok=True)
    for path in sorted(working.rglob("events.csv")):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        dest = snapshots / (path.parent.name + "-" + digest[:12] + ".csv")
        if not dest.exists():
            dest.write_bytes(raw)
        fingerprints.append(dict(path=str(path.relative_to(ROOT)), sha256=digest, bytes=len(raw)))
        for row in read_csv(path):
            key = row.get("video_key") or row.get("video_id")
            if row.get("confidence") == "real":
                events.setdefault(key, []).append(dict(start=float(row["start_sec"]), end=float(row["end_sec"]), kind=row["kind"]))
    for value in events.values():
        value.sort(key=lambda r: r["start"])
    scope = read_csv(ROOT / "data/metadata/annotation-review/recheck-20261108-v2/all-dongchuan-scope.csv")
    training = {r["unit_id"] for r in scope if r["in_current150_development_pool"].lower() == "true"}
    edited = ["v1_75", "v1_74", "v5_08", "v1_13", "v1_16", "v1_17", "v1_50", "v1_91", "v1_94", "v1_95", "v1_97", "v4_01", "v4_02", "v4_03", "v4_04"] + ["v5_{:02d}".format(i) for i in range(15, 21)]
    selected = {k: dict(in_historical150=k in training, events=events.get(k, [])) for k in edited}
    summary = dict(working_annotations_dir=pointer["working_annotations_dir"], csv_fingerprints=fingerprints, selected=selected,
                   scope_note="Snapshot only; not a new annotation acceptance or training-pool release.")
    (OUT / "current-label-snapshot.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index = read_csv(ROOT / "data/process/video_frames/video1_segment/frames_index.csv")
    frames = [(float(r["t_sec"]), r["frame_file"]) for r in index if r["video_key"] == "v1_17"]
    samples = []
    evs = events["v1_17"]
    for n, ev in enumerate(evs):
        if ev["kind"] != "neg":
            continue
        times = [ev["start"] + (ev["end"]-ev["start"]) * f for f in (0.12, 0.38, 0.62, 0.88)]
        after = next((e for e in evs[n+1:] if e["kind"] == "pos"), None)
        if after:
            times += [after["start"] + min(after["end"]-after["start"], 16) * f for f in (0.12, 0.38, 0.62, 0.88)]
        rows = []
        board = Image.new("RGB", (1200, 358*((len(times)+1)//2)), "white")
        draw = ImageDraw.Draw(board)
        for j, target in enumerate(times):
            sec, filename = min(frames, key=lambda r: abs(r[0]-target))
            path = ROOT / "data/process/video_frames/video1_segment/frames" / filename
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((600, 326))
                board.paste(im, ((j%2)*600, (j//2)*358+28))
            kind = next((e["kind"] for e in evs if e["start"] <= sec < e["end"]), "unmarked")
            draw.text(((j%2)*600+5, (j//2)*358+7), "v1_17 audit {:.2f}s | {} | request {:.2f}s".format(sec, kind, target), fill="black")
            rows.append(dict(audit_seconds=sec, requested_seconds=target, frame_file=filename, current_kind=kind))
        dest = OUT / "v1_17-neg-{:02d}-and-next-pos.jpg".format(n)
        board.save(dest, quality=91)
        samples.append(dict(negative_interval=ev, evidence_image=str(dest.relative_to(ROOT)), samples=rows))
    (OUT / "v1_17-evidence.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(snapshot=str(OUT / "current-label-snapshot.json"), selected=selected, evidence_images=[r["evidence_image"] for r in samples]), ensure_ascii=False))


if __name__ == "__main__":
    if "--paired-visuals" in sys.argv:
        paired_visuals()
    else:
        main()
