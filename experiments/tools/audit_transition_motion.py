"""Reproduce a bounded motion counterexample; never produces class labels.

Reads only listed cached JPEG pairs, writes small evidence and static figures.
No decoder, learned model, classifier fitting, media edits, or network calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "data/metadata/annotation-review/npt-evidence-20261108-v1/motion-pairs.json"


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unreadable listed JPEG: " + str(path))
    return cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)


def measure(document: dict) -> list[dict]:
    output = []
    for unit, role, name1, name2, t1, t2 in document["pairs"]:
        dt = float(t2) - float(t1)
        if dt <= 0:
            raise ValueError("Pair time must strictly increase")
        source = "video" + unit.split("_")[0][1:] + "_segment"
        paths = [ROOT / document["frame_root"] / source / "frames" /
                 (unit + "_frame_" + suffix + ".jpg") for suffix in (name1, name2)]
        x, y = [read_gray(path) for path in paths]
        flow = cv2.calcOpticalFlowFarneback(x, y, None, .5, 3, 15, 3, 5, 1.2, 0)
        roi = np.s_[45:169, :]
        difference = np.abs(x.astype(np.float64) - y.astype(np.float64))[roi]
        speed = np.linalg.norm(flow, axis=2)[roi] / np.hypot(320, 180) / dt
        output.append({"unit_id": unit, "provisional_role": role, "t1": t1, "t2": t2,
                       "dt": dt, "images": [{"path": p.relative_to(ROOT).as_posix(),
                                               "sha256": fingerprint(p)} for p in paths],
                       "diff_s": float(difference.mean() / 255 / dt),
                       "flow_s": float(speed.mean()),
                       "flow95_s": float(np.quantile(speed, .95)),
                       "diff_gt10": float((difference > 10).mean())})
    return output


def figures(rows: list[dict], destination: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".codex/docs/.temp/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 11})
    destination.mkdir(parents=True, exist_ok=True)
    # Include all four95 examples, including the two late examples.
    selected = [row for row in rows if row["unit_id"] == "v1_95" or
                row["unit_id"] == "v1_96" and row["t1"] == 0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), constrained_layout=True)
    for ax, key, title in zip(axes, ("flow_s", "diff_s"),
                              ("归一化平均运动量／秒", "归一化平均灰度变化／秒")):
        for row in selected:
            is_n = row["unit_id"] == "v1_96"
            ax.scatter(row[key], 0 if is_n else 1, s=65, marker="s" if is_n else "o",
                       color="#28669a" if is_n else "#c67425")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["1-96：旧标 N（0秒取对）", "1-95：旧候选 T"])
        ax.set_ylim(-.55, 1.55); ax.set_xlabel(title); ax.grid(axis="x", alpha=.25)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle("历史标签下的运动量交叠：不是全局N/T分类器", fontsize=13)
    fig.savefig(destination / "motion-counterexample.png", dpi=180,
                metadata={"Software": "Matplotlib"}); plt.close(fig)
    r = np.linspace(0, 1, 501)
    fig, ax = plt.subplots(figsize=(7.5, 3.5), constrained_layout=True)
    for y, color, label in ((np.maximum(1 - 2*r, 0), "#28669a", "N：全负锚点"),
                            (1 - np.abs(2*r-1), "#c67425", "T：一半正、一半负锚点"),
                            (np.maximum(2*r-1, 0), "#b13b44", "P：全正锚点")):
        ax.plot(r, y, lw=2.3, color=color, label=label)
    ax.set(xlabel="r：已确认 P 占整段时间的比例", ylabel="软目标值", xlim=(0, 1), ylim=(0, 1.04))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.24), ncol=3, fontsize=9)
    ax.grid(alpha=.2)
    fig.savefig(destination / "composition-anchors.png", dpi=180,
                metadata={"Software": "Matplotlib"}); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    data = json.loads(args.pairs.read_text(encoding="utf-8-sig"))
    rows = measure(data)
    result = {"purpose": data["purpose"], "pair_manifest_sha256": fingerprint(args.pairs),
              "script_sha256": fingerprint(Path(__file__)),
              "versions": {"opencv": cv2.__version__, "numpy": np.__version__},
              "method": {"decode": "IMREAD_GRAYSCALE, then INTER_AREA320x180", "roi": [0, 45, 320, 169],
                         "farneback": [.5, 3, 15, 3, 5, 1.2, 0],
                         "flow_s": "mean flow magnitude / image diagonal / actual dt",
                         "diff_s": "mean absolute gray difference /255 /actual dt"},
              "limitations": ["Selected diagnostic pairs, no population performance claim",
                              "Not a semantic channel mask or flow-type measurement",
                              "May contain vehicle motion, illumination changes, or compression artifacts"],
              "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.figures:
        figures(rows, args.figures)
    n = next(row for row in rows if row["unit_id"] == "v1_96" and row["t1"] == 0)
    t = [row for row in rows if row["unit_id"] == "v1_95"]
    print(json.dumps({"pairs": len(rows), "N96_flow": n["flow_s"],
                      "T95_flow_min": min(row["flow_s"] for row in t),
                      "T95_flow_max": max(row["flow_s"] for row in t),
                      "training_or_classifier_run": False}))


if __name__ == "__main__":
    main()
