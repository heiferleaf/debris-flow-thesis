"""Bounded, background-referenced cached-frame motion diagnostics (never N/P/T).

Uses actual adjacent frames_index times. At ~2 s spacing, short pulses may be
missed and optical flow may alias; px/s means image displacement / audit-time
interval, not physical fluid velocity. No original-video decoder is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "data/metadata/annotation-review/user-feedback-20261108-motion/manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_image(path: Path, scale: float) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unreadable cached frame: " + str(path))
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def mask_polygon(shape: tuple, points: list, scale: float, exclude: list) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(np.asarray(points) * scale).astype(np.int32)], 255)
    for x, y, w, h in exclude:
        x, y, w, h = np.rint(np.array([x, y, w, h]) * scale).astype(int)
        mask[y:y+h, x:x+w] = 0
    # Boundary context is shared for flow but not counted in ROI summaries.
    return cv2.erode(mask, np.ones((5, 5), np.uint8)).astype(bool)


def measure_pair(a: np.ndarray, b: np.ndarray, channel: np.ndarray,
                 background: np.ndarray, dt: float, scale: float,
                 max_dt: float = 4.5) -> dict:
    """Return measured quantities and diagnostic-quality flags; no class field.

    Operational QC floors: <1 gray-level RMS local gradient is low texture;
    saturated means <3/>252; tiny std <0.5; phase response <.15; shifts >12
    original pixels. These invalidate this measurement, never label semantics.
    Activity is > contemporaneous background's 99th percentile, not a learned
    semantic threshold. The 1 gray-level floor only avoids quantization zeros.
    """
    if a.shape != b.shape or a.shape != channel.shape or a.shape != background.shape:
        raise ValueError("Frames and ROI masks must have identical shapes")
    if dt <= 0 or not 0 < scale <= 1:
        raise ValueError("Positive actual dt and scale in (0,1] required")
    flags = []
    if dt > max_dt:
        flags.append("interval_too_large")
    if min(channel.sum(), background.sum()) < 100:
        raise ValueError("ROIs require at least 100 processed pixels")
    for name, im in (("first", a), ("second", b)):
        if im.mean() < 3 or im.mean() > 252 or im.std() < .5:
            flags.append(name + "_bad_frame")
    texture = float(np.sqrt(np.mean(
        cv2.Sobel(a, cv2.CV_32F, 1, 0)[channel] ** 2 +
        cv2.Sobel(a, cv2.CV_32F, 0, 1)[channel] ** 2)))
    if texture < 1:
        flags.append("channel_low_texture")
    yy, xx = np.where(background)
    sl = np.s_[yy.min():yy.max()+1, xx.min():xx.max()+1]
    p, q = a[sl].astype(np.float32), b[sl].astype(np.float32)
    bg_texture = float(np.sqrt(np.mean(cv2.Sobel(p, cv2.CV_32F, 1, 0)**2 +
                                           cv2.Sobel(p, cv2.CV_32F, 0, 1)**2)))
    if bg_texture < 1:
        flags.append("background_low_texture")
    window = cv2.createHanningWindow((p.shape[1], p.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(p - p.mean(), q - q.mean(), window)
    dx, dy = (float(v) for v in shift)
    if response < .15:
        flags.append("registration_low_response")
    if np.hypot(dx, dy) / scale > 12:
        flags.append("registration_large_shift")
    aligned = cv2.warpAffine(b, np.float32([[1, 0, -dx], [0, 1, -dy]]),
                             (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT101)
    # Remove only a global exposure offset estimated on the fixed background.
    bias = float(np.median(aligned[background].astype(float) - a[background]))
    corrected = np.clip(aligned.astype(np.float32) - bias, 0, 255)
    diff = np.abs(corrected - a.astype(np.float32))
    flow = cv2.calcOpticalFlowFarneback(a, corrected.astype(np.uint8), None,
                                      .5, 3, 19, 3, 5, 1.2, 0)
    speed = np.linalg.norm(flow, axis=2) / scale / dt
    difference_floor = max(float(np.quantile(diff[background], .99)), 1.)
    flow_floor = max(float(np.quantile(speed[background], .99)), 1e-6)
    result = dict(
        diagnostic_evaluable=not flags, diagnostic_flags=";".join(flags),
        scale=scale, dt_sec=dt, phase_response=float(response),
        global_dx_original_px=dx/scale, global_dy_original_px=dy/scale,
        background_exposure_bias_gray=bias, channel_texture_rms=texture,
        background_texture_rms=bg_texture,
        channel_diff_mean_gray=float(diff[channel].mean()),
        background_diff_mean_gray=float(diff[background].mean()),
        background_diff_q99_gray=difference_floor,
        channel_diff_exceed_background_fraction=float((diff[channel] > difference_floor).mean()),
        background_diff_exceed_background_fraction=float((diff[background] > difference_floor).mean()),
        channel_flow_mean_px_s=float(speed[channel].mean()),
        channel_flow_q50_px_s=float(np.quantile(speed[channel], .5)),
        channel_flow_q95_px_s=float(np.quantile(speed[channel], .95)),
        background_flow_mean_px_s=float(speed[background].mean()),
        background_flow_q99_px_s=flow_floor,
        channel_flow_exceed_background_fraction=float((speed[channel] > flow_floor).mean()),
        background_flow_exceed_background_fraction=float((speed[background] > flow_floor).mean()),
        # Jointly above background in appearance and displacement; a QC signal.
        channel_joint_exceed_background_fraction=float(
            ((diff[channel] > difference_floor) & (speed[channel] > flow_floor)).mean()),
        background_joint_exceed_background_fraction=float(
            ((diff[background] > difference_floor) & (speed[background] > flow_floor)).mean()),
        channel_pixels=int(channel.sum()), background_pixels=int(background.sum()))
    return result


def interval_summary(rows: list[dict], window: dict) -> dict:
    # Fully contained pairs only: never attribute a cross-boundary displacement
    # to one side or interpolate from unavailable original frames.
    selected = [r for r in rows if r["unit_id"] == window["unit_id"]
                and r["t0_sec"] >= window["start_sec"] and r["t1_sec"] <= window["end_sec"]]
    usable = [r for r in selected if r["diagnostic_evaluable"]]
    output = {**window, "pairs": len(selected), "evaluable_pairs": len(usable),
              "measured_pair_time_sec": sum(r["dt_sec"] for r in usable)}
    for key in ("channel_diff_mean_gray", "background_diff_mean_gray",
                "channel_diff_exceed_background_fraction", "channel_flow_mean_px_s",
                "background_flow_mean_px_s", "channel_flow_exceed_background_fraction",
                "channel_joint_exceed_background_fraction", "background_joint_exceed_background_fraction"):
        vals = [r[key] for r in usable]
        output[key + "_median"] = float(np.median(vals)) if vals else None
        output[key + "_q90"] = float(np.quantile(vals, .9)) if vals else None
    # A descriptive alert, explicitly the fraction of pixels above *this pair's*
    # fixed-background noise, not proof of stationarity or a N/P/T threshold.
    no_excess = [r for r in usable if r["channel_diff_exceed_background_fraction"] <=
                 r["background_diff_exceed_background_fraction"] and
                 r["channel_flow_exceed_background_fraction"] <=
                 r["background_flow_exceed_background_fraction"]]
    output["pairs_without_channel_excess_over_background"] = len(no_excess)
    quiet = [r for r in usable if r["channel_joint_exceed_background_fraction"] <=
             r["background_joint_exceed_background_fraction"]]
    output["pairs_with_joint_excess_at_or_below_background"] = len(quiet)
    longest = current = 0.
    end = None
    for r in usable:
        if r["channel_joint_exceed_background_fraction"] <= r["background_joint_exceed_background_fraction"]:
            current = current+r["dt_sec"] if end is not None and abs(r["t0_sec"]-end) < .001 else r["dt_sec"]
            end = r["t1_sec"]
            longest = max(longest, current)
        else:
            current, end = 0., None
    output["longest_adjacent_joint_at_or_below_background_sec"] = longest
    typical_not_above = bool(usable) and (
        np.median([r["channel_joint_exceed_background_fraction"] for r in usable]) <=
        np.median([r["background_joint_exceed_background_fraction"] for r in usable]))
    output["review_alert"] = (
        "Old T contains a >=20 s sampled run at/below background joint activity, or its median joint activity is at/below background over >=20 s. Review quiet portions; isolated spikes remain in pair records; this does not assert continuous stationarity."
        if window["working_state"] == "T" and (longest >= 20 or
            (output["measured_pair_time_sec"] >= 20 and typical_not_above)) else "")
    return output


def boundary_summary(rows: list[dict], window: dict) -> dict:
    result = dict(window)
    intervals = []
    for name in ("left", "right"):
        lo, hi = window[name + "_sec"]
        selected = [r for r in rows if r["unit_id"] == window["unit_id"] and
                    r["diagnostic_evaluable"] and r["t0_sec"] >= lo and r["t1_sec"] <= hi]
        vals = [r["channel_joint_exceed_background_fraction"] for r in selected]
        stats = {"pairs": len(vals), "q10": float(np.quantile(vals, .1)) if vals else None,
                 "q50": float(np.median(vals)) if vals else None,
                 "q90": float(np.quantile(vals, .9)) if vals else None}
        result[name] = stats
        intervals.append(stats)
    left, right = intervals
    overlap = (left["pairs"] >= 3 and right["pairs"] >= 3 and
               max(left["q10"], right["q10"]) <= min(left["q90"], right["q90"]))
    result["central_80_percent_activity_ranges_overlap"] = bool(overlap)
    result["review_alert"] = ("Motion distributions overlap on opposite sides of old P/T boundary; motion alone does not justify their semantic difference. Not an equivalence test."
                              if overlap else "")
    return result


def plot_results(rows: list[dict], manifest: dict, out: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".codex/docs/.temp/motion-feedback/mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    for unit in ("v1_96", "v1_91"):
        rr = [r for r in rows if r["unit_id"] == unit and
              (unit != "v1_91" or r["t0_sec"] >= 1320)]
        fig, axes = plt.subplots(3, 1, figsize=(12, 7.3), sharex=True, constrained_layout=True)
        ts = [(r["t0_sec"]+r["t1_sec"])/2 for r in rr]
        axes[0].plot(ts, [r["channel_diff_mean_gray"] for r in rr], lw=.9, label="Channel")
        axes[0].plot(ts, [r["background_diff_mean_gray"] for r in rr], lw=.9, label="Fixed background")
        axes[0].set_ylabel("Mean abs. gray change\n(0-255, registered)")
        axes[0].legend(loc="upper right")
        axes[1].plot(ts, [r["channel_flow_mean_px_s"] for r in rr], lw=.9, label="Channel mean flow")
        axes[1].plot(ts, [r["background_flow_q99_px_s"] for r in rr], lw=.9, label="Background 99th percentile")
        axes[1].set_ylabel("Image displacement / dt\n(original-cache px/s)")
        axes[1].legend(loc="upper right")
        axes[2].plot(ts, [r["channel_diff_exceed_background_fraction"]*100 for r in rr], lw=.9, label="Gray change > background q99")
        axes[2].plot(ts, [r["channel_joint_exceed_background_fraction"]*100 for r in rr], lw=.9, label="Both gray change and flow > background q99")
        axes[2].set_ylabel("Channel pixels (%)")
        axes[2].set_xlabel("Actual cached-frame audit time (s); ~2 s samples may miss short pulses")
        axes[2].legend(loc="upper right")
        for interval in manifest["working_intervals"]:
            if interval["unit_id"] != unit:
                continue
            color = {"T": "#d6b173", "P": "#d87777", "N": "#94b6c7"}[interval["working_state"]]
            for ax in axes:
                ax.axvspan(interval["start_sec"], interval["end_sec"], color=color, alpha=.18)
            axes[0].text((interval["start_sec"]+interval["end_sec"])/2, .95,
                         interval["working_state"], transform=axes[0].get_xaxis_transform(),
                         ha="center", va="top", fontsize=8)
        for ax in axes:
            ax.grid(alpha=.2)
            for r in rr:
                if not r["diagnostic_evaluable"]:
                    ax.axvspan(r["t0_sec"], r["t1_sec"], color="black", alpha=.12)
        axes[0].set_title(f"{unit}: motion diagnostic, background-referenced; shading = old working labels, no relabeling")
        axes[0].set_xlim(min(ts), max(ts))
        fig.savefig(out / f"{unit}-motion-timeseries.png", dpi=150, metadata={"Software": "Matplotlib"})
        plt.close(fig)


def contact_sheets(manifest: dict, selected: dict, out: Path) -> None:
    from PIL import Image, ImageDraw
    for unit, times in manifest["contact_times"].items():
        spec = manifest["videos"][unit]
        candidates = selected[unit]
        canvas = Image.new("RGB", (4*384, ((len(times)+3)//4)*240), "white")
        draw = ImageDraw.Draw(canvas)
        for i, t in enumerate(times):
            row = min(candidates, key=lambda r: abs(float(r["t_sec"])-t))
            im = Image.open(ROOT / manifest["frame_root"] / row["frame_file"]).convert("RGB")
            d = ImageDraw.Draw(im)
            d.line([tuple(p) for p in spec["channel_polygon"] + [spec["channel_polygon"][0]]], fill="#ff983a", width=5)
            d.line([tuple(p) for p in spec["background_polygon"] + [spec["background_polygon"][0]]], fill="#43e9e1", width=5)
            for x, y, w, h in spec["exclude_rectangles"]:
                d.rectangle((x, y, x+w, y+h), outline="#ff4868", width=4)
            im.thumbnail((384, 216))
            x, y = (i%4)*384, (i//4)*240
            canvas.paste(im, (x, y+24))
            draw.text((x+5, y+5), f'{unit} audit t={row["t_sec"]}s', fill="black")
        canvas.save(out / f"{unit}-roi-contact.png")


def run(manifest_path: Path, out: Path) -> dict:
    document = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    out.mkdir(parents=True, exist_ok=True)
    index_path = ROOT / document["frames_index"]
    with index_path.open(encoding="utf-8-sig", newline="") as f:
        all_rows = list(csv.DictReader(f))
    results, inputs, selected = [], {}, {}
    scale = float(document["processing_scale"])
    for unit, spec in document["videos"].items():
        candidates = sorted((r for r in all_rows if r["video_key"] == unit), key=lambda r: float(r["t_sec"]))
        selected[unit] = candidates
        a, prev = None, None
        for n, row in enumerate(candidates):
            t = float(row["t_sec"])
            if not any(lo <= t <= hi for lo, hi in spec["scan_ranges_sec"]):
                a, prev = None, None
                continue
            path = ROOT / document["frame_root"] / row["frame_file"]
            b = read_image(path, scale)
            inputs[path.relative_to(ROOT).as_posix()] = sha256(path)
            if a is not None:
                channel = mask_polygon(b.shape, spec["channel_polygon"], scale, spec["exclude_rectangles"])
                background = mask_polygon(b.shape, spec["background_polygon"], scale, spec["exclude_rectangles"])
                measured = measure_pair(a, b, channel, background, t-float(prev["t_sec"]), scale,
                                        document["max_pair_interval_sec"])
                results.append(dict(unit_id=unit, t0_sec=float(prev["t_sec"]), t1_sec=t,
                                    frame0=prev["frame_file"], frame1=row["frame_file"], **measured))
            a, prev = b, row
        print(f'{unit}: {sum(r["unit_id"] == unit for r in results)} cached pairs', flush=True)
    summaries = [interval_summary(results, w) for w in document["summary_windows"]]
    boundaries = [boundary_summary(results, w) for w in document.get("boundary_windows", [])]
    metadata = dict(
        manifest_sha256=sha256(manifest_path), script_sha256=sha256(Path(__file__)),
        index_path=document["frames_index"], index_sha256=sha256(index_path),
        versions={"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
        method=document["method"], limitations=document["limitations"],
        input_frame_sha256=inputs, summaries=summaries, boundary_summaries=boundaries, pair_count=len(results),
        evaluable_pair_count=sum(r["diagnostic_evaluable"] for r in results),
        training_or_labels_changed=False)
    (out / "results.json").write_text(json.dumps({**metadata, "pairs": results}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    with (out / "pairs.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    (out / "interval-summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (out / "boundary-summary.json").write_text(json.dumps(boundaries, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    contact_sheets(document, selected, out)
    plot_results(results, document, out)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path, default=ROOT / "data/process/annotation-review/user-feedback-20261108-motion")
    args = parser.parse_args()
    result = run(args.manifest, args.out)
    print(json.dumps({k: result[k] for k in ("pair_count", "evaluable_pair_count", "training_or_labels_changed")}))


if __name__ == "__main__":
    main()
