"""Show archived EXP022 binary scores at their actual availability times."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = ROOT / "data/results/exp022/linear-cv-s7-20261104-v1/f03-dense/window-scores.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = [{**r, **{k: float(r[k]) for k in ("start", "end", "score", "target_start_seconds")}}
                for r in csv.DictReader(stream) if r["unit_id"] == "v1_16"]
    selected = [r for r in rows if 1028 <= r["end"] <= 1055]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax, windows) = plt.subplots(2, 1, figsize=(10, 5.7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1.4]})
    ax.plot([r["end"] for r in selected], [r["score"] for r in selected], "o-", color="#20738b", label="旧模型真实二类分数")
    ax.axhline(.5, color="#b25a38", linestyle="--", label="旧实验阈值0.5")
    ax.axvline(1036.04, color="#a5801d", linestyle=":", label="审计首个可见新前缘（约）")
    ax.set_ylim(0, 1); ax.set_ylabel("正类分数（未经概率校准）")
    ax.set_title("每个输入片段得到一个分数；收齐片段后才能显示", fontsize=14)
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=.18)
    examples = [min(rows, key=lambda r: abs(r["start"]-t)) for t in (1035.04,1036.52,1039.52)]
    for i, r in enumerate(examples):
        windows.plot([r["start"],r["end"]],[i,i],linewidth=7,color="#20738b",solid_capstyle="round")
        windows.plot(r["end"], i, "o", color="#b25a38")
        windows.text(r["end"]+.5, i, "结束%.2f秒 → %.3f"%(r["end"],r["score"]), va="center", fontsize=9)
    windows.set_yticks(range(3)); windows.set_yticklabels(["输入片段1","输入片段2","输入片段3"])
    windows.invert_yaxis(); windows.set_xlabel("视频时间／秒（横坐标为该次分数可获得的时刻）")
    windows.set_xlim(1028,1055); windows.grid(axis="x",alpha=.18)
    fig.text(.5,.015,"已有EXP-022、v1_16、外折3；不是新三分类结果。前缘审计秒数与中间件PTS仍有对齐误差，不据此报告精确告警延迟。",ha="center",fontsize=8)
    fig.tight_layout(rect=[0,.045,1,1])
    output=ROOT/"docs/data/figures/model-usage-20261108";output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/"actual-binary-window-scores.png",dpi=180,metadata={"Title":"已有二类模型的逐片段分数示例"});plt.close(fig)
    data=ROOT/"data/metadata/model-usage-example/20261108-v1";data.mkdir(parents=True,exist_ok=True)
    (data/"example.json").write_text(json.dumps({"unit_id":"v1_16","run_id":"linear-cv-s7-20261104-v1","outer_fold":3,
        "source":source.relative_to(ROOT).as_posix(),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "availability":"end is final input PTS, not start","all_unit_windows":len(rows),"plotted_rows":selected,
        "examples":examples,"is_new_three_class_result":False,"new_training_performed":False},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"unit_windows":len(rows),"plotted_windows":len(selected),"new_training":False}))


if __name__ == "__main__":
    main()
