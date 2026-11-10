"""Plot the two calibrated measurable moments, not training labels or accuracy."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "data/metadata/annotation-review/visual-occupancy-20261108-v1"


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    manifest = json.loads((SOURCE / "calibrated-manifest.json").read_text(encoding="utf-8-sig"))
    outlines = [json.loads((SOURCE / ("measurement-" + c + ".json")).read_text(encoding="utf-8-sig")) for c in ("a", "b")]
    measurements = json.loads((SOURCE / "calibrated-measurements.json").read_text(encoding="utf-8-sig"))
    measured = {x["item_id"]: x for x in measurements["results"]}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2))
    for row, key in enumerate(("V03", "V04")):
        item = next(x for x in manifest["items"] if x["item_id"] == key)
        im = Image.open(ROOT / item["central_frame"]).convert("RGB")
        width, height = im.size
        for col, source in enumerate(outlines):
            ax = axes[row, col]
            ax.imshow(im)
            roi = source["rois"][item["unit_id"]]
            ax.add_patch(Polygon([(x*(width-1), y*(height-1)) for x, y in roi], fill=False, edgecolor="#ffd34d", linewidth=1.4))
            entry = next(x for x in source["items"] if x["item_id"] == key)
            for polygon in entry["flow_polygons"]:
                ax.add_patch(Polygon([(x*(width-1), y*(height-1)) for x, y in polygon], color="#00d8ef", alpha=.38, linewidth=.6))
            value = measured[key]["s_" + ("A" if col == 0 else "B")]
            ax.set_title("描边%s｜%s｜证据占比 %.2f%%" % ("A" if col == 0 else "B", "新前缘" if row == 0 else "明显主体", value*100), fontsize=13)
            ax.axis("off")
    fig.suptitle("同一固定视角的两个可测时刻：黄色为固定流道，青色为可辨认证据", fontsize=14, y=.98)
    fig.text(.5, .025, "仅为局部校准；目标类型依赖既有锚点。20%是候选工程阈值，不代表危险程度，也未验证跨机位有效。", ha="center", fontsize=10)
    fig.subplots_adjust(left=.02, right=.98, bottom=.08, top=.90, hspace=.20, wspace=.08)
    out = ROOT / "docs/data/figures/npt-20261108/visual-occupancy-calibration.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, metadata={"Title": "固定流道证据占比的局部校准"})
    plt.close(fig)
    print(str(out))


if __name__ == "__main__":
    main()
