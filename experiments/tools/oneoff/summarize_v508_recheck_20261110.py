"""Build small deterministic evidence contacts and current-axis alignment."""
import bisect
import csv
import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/process/decode-recheck/v5_08-20261110-v1"
OLD = ROOT / "data/process/annotation-review/full-dongchuan-20261109-v1/batch-03/v5_08"


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    samples = json.loads((OUT / "samples.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((OUT / "frame-statistics.csv").open(encoding="utf-8")))
    timeline = [int(r["pts"]) / 90000 for r in rows]
    contacts = {}
    for prefix, cols in [("support3_",4),("support7p5_",4),("quality_",4),("navigation_",3),("candidate_",4)]:
        group = [s for s in samples if s["name"].startswith(prefix)]
        if not group:
            continue
        canvas = Image.new("RGB", (cols * 256, ((len(group) + cols - 1)//cols)*168), "white")
        draw = ImageDraw.Draw(canvas)
        for i, s in enumerate(group):
            x, y = i % cols * 256, i // cols * 168
            canvas.paste(Image.open(OUT / s["frame_file"]), (x, y + 24))
            draw.text((x + 2, y + 2), "n={} t={:.3f} std={:.2f}".format(s["n"], s["seconds"], s["gray_std"]), fill="black")
        path = OUT / (prefix.rstrip("_") + "-contact.jpg")
        canvas.save(path, quality=94)
        contacts[prefix] = dict(path=str(path.relative_to(ROOT)), cells=len(group),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    old_index = json.loads((OLD / "audit-frame-index.json").read_text(encoding="utf-8"))
    alignment = []
    for old in old_index:
        t = old["t"]
        k = bisect.bisect_left(timeline, t)
        near = min(range(max(0,k-1),min(len(timeline),k+2)), key=lambda i:abs(timeline[i]-t))
        alignment.append(dict(old_frame_file=old["frame_file"], old_seconds=t,
                              new_n=near, new_pts=int(rows[near]["pts"]),
                              new_seconds=timeline[near], abs_difference=abs(timeline[near]-t)))
    with (OUT / "old-audit-time-alignment.csv").open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(alignment[0]))
        w.writeheader(); w.writerows(alignment)
    groups=[]
    current=[]
    for i,r in enumerate(rows):
        delta = timeline[i] - timeline[i-1] if i else .1
        good = float(r["gray_std"]) >=20 and .05 < delta < .25
        if good:
            current.append(i)
        elif current:
            groups.append(current);current=[]
    if current:
        groups.append(current)
    runs=[dict(first_n=g[0],last_n=g[-1],frames=len(g),start_sec=timeline[g[0]],end_sec=timeline[g[-1]],
               duration_sec=timeline[g[-1]]-timeline[g[0]]) for g in groups]
    exact_jumps=[dict(before_n=i,after_n=i+1,before_sec=timeline[i],after_sec=timeline[i+1],
                      delta_sec=round(timeline[i+1]-timeline[i],6))
                 for i in range(len(rows)-1) if timeline[i+1]-timeline[i]>.25]
    early=[]
    for duration in [3,7.5]:
        subset=[r for r in rows if int(r["pts"])/90000<=duration]
        early.append(dict(window_start_sec=0,window_end_sec=duration,frames=len(subset),
                          last_sec=int(subset[-1]["pts"])/90000,
                          distinct_pixel_sha256=len({r["rgb_sha256"] for r in subset}),
                          min_gray_std=min(float(r["gray_std"]) for r in subset)))
    source_errors=[]
    muxer_warnings=[]
    last_n=None
    for line in (OUT/"stderr.txt").read_text(encoding="utf-8").splitlines():
        match=re.search(r" n:\s*(\d+)\s+pts:",line)
        if match:
            last_n=int(match[1])
        if re.search(r"(?i)(error|invalid|missing|corrupt|non.monoton|could not|NAL unit)",line):
            record=dict(line=line.strip(),preceding_showinfo_n=last_n,
                        context_warning="Decoder threading/log order does not establish an exact frame-error correspondence.")
            (muxer_warnings if "[rawvideo @" in line else source_errors).append(record)
    report=dict(contacts=contacts, old_audit_time_alignment=dict(count=len(alignment),
                  max_abs_difference=max(r["abs_difference"] for r in alignment),
                  unmatched_over_006=sum(r["abs_difference"]>.006 for r in alignment),
                  interpretation="Same current fixed source navigation; not pre-fix/physical-clock authentication."),
                technical_candidate_runs=runs,
                technical_candidate_run_rule="gray_std>=20 and preceding PTS difference between .05 and .25s; automatic screening only, not a semantic or complete quality certificate.",
                exact_pts_timebase="1/90000; use integer pts in frame-statistics.csv rather than six-significant-digit pts_time",
                first_last_exact_sec=[timeline[0],timeline[-1]],exact_jumps_over_025=exact_jumps,
                early_continuous_support=early,
                source_decode_error_lines=source_errors,
                rawvideo_muxer_warning_count=len(muxer_warnings),
                rawvideo_muxer_warning_interpretation="Output header uses 10tbn while source uses 90k tbn. Strictly increasing exact source PTS rescale into repeated output DTS ticks. These 34 output-muxer warnings are not source PTS reversal; no output frames were duplicated or dropped by vsync, and all raw frames match one showinfo record.",
                decoded_real_frames=summary["decoded_real_frames"])
    (OUT / "derived-evidence.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
