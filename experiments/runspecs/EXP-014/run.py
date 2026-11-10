# -*- coding: utf-8 -*-
"""EXP-014 全套流水线：两套设定各训五折，再各算三套片段分数，最后跑配对判据。

为什么要有这个脚本：本卡在租来的卡上跑 20–48 小时，十几条命令行若靠手敲，
任何一处参数写错都会烧掉几小时租金、而且可能烧完才发现（例如把原间隔的帧目录
传给新间隔那套设定，训练照样跑得通，成绩却无从解释）。把命令固定成一张表，
跑之前 `--dry-run` 打印出来逐行核对，跑的时候一步失败立即停。

步骤（顺序固定，前一步不成功不进下一步；条数以 `--dry-run` 打印出来的为准）：

  0. 预算探针：原间隔这套设定**只跑第 1 折**，量一折要多久（产物写进单独目录，
     不进正式成绩）。卡片第四节的「跑完第一折重估总时长」靠这一步，不靠中途杀进程：
     `train_ft.py` 一次调用就把五折跑完，中途杀掉再用 `--fold 2` 之类续跑，新进程写盘时
     会把 `metrics.json` 与 `predictions.csv` 覆盖成只剩那一折，后面 `clip_scores_ft.py`
     的回归门禁就核不上了。
  1. 原间隔这套设定：`train_ft.py` 五折 20 轮（`--line strong --augment none`）
  2. 用折权重给全部 128 条打三套分数：正常、反序、时间冻结
  3. `motion_effect.py --run` 的单套自检：把时间冻结那一套与正常那一套放在一起读，
     打印「只看单帧时分数掉了多少」这个辅助读数（卡片 §5.3 要求）
  4. 新间隔这套设定的第 1 到 3 步
  5. `motion_effect.py --run-a <原间隔> --run-b <新间隔>` 出配对差与 95% 区间

每一步的完整命令行、起止时间、耗时与退出码写进 `<runs-root>/EXP-014-pipeline.log`。

用法（仓库根）：

  .venv/Scripts/python.exe experiments/runspecs/EXP-014/run.py --dry-run
  .venv/Scripts/python.exe experiments/runspecs/EXP-014/run.py --only probe-v2   # 只为估预算
  .venv/Scripts/python.exe experiments/runspecs/EXP-014/run.py --only train-v2
  .venv/Scripts/python.exe experiments/runspecs/EXP-014/run.py                 # 全跑

本脚本不训练、不算分，只按固定顺序调用已有脚本；真正的训练设定在
`train_ft.py` 与卡片 `docs/process/experiments/EXP-014.md` 里。
"""
import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PY = sys.executable
POOL = REPO / "data/process/EXP-008-units-pool128"
FOLDS = REPO / "data/metadata/splits/EXP-008-units-folds-128.json"

SETTINGS = {
    "v2": {  # 原间隔：连续 16 帧
        "name": "原间隔",
        "run": "EXP-014-strong-v2-128-strict-ft-noaug",
        "frames": REPO / "data/process/EXP-008-units-v2/frames",
        "labels": POOL / "clip-labels-v2-128.npz",
        # 两套设定的片段标签视频顺序完全相同、又都登记在 train_ft.py 的 CARD_MD5 里，
        # 所以「新间隔的帧配原间隔的标签」能过那边的指纹检查。这里按设定各记一个 md5，
        # 并且要求帧目录所在的中间件目录名与标签文件名后缀对应（EXP-008-units-v2 ↔ clip-labels-v2）。
        "frames_tag": "EXP-008-units-v2",
        "labels_md5": "07ad71057234c990fb9498a5c5e59a0c",
    },
    "v3": {  # 新间隔：每 0.5 秒取一帧
        "name": "新间隔",
        "run": "EXP-014-strong-v3-128-strict-ft-noaug",
        "frames": REPO / "data/process/EXP-008-units-v3-gap05/frames",
        "labels": POOL / "clip-labels-v3-128.npz",
        "frames_tag": "EXP-008-units-v3-gap05",
        "labels_md5": "fa2c53c1a0f93d62869b1aeebbc80f4b",
    },
}
META_MD5 = "4b3e8fdb7c01c9cffef59a09513a2446"      # meta-128.npz
RUNS = REPO / "data/results/exp008"
# 预算探针的单独目录，不进正式成绩。
# 名字必须带「-noaug」后缀：`train_ft.py` 的开跑前核对表从 run 名后缀反查增强档位，
# 查不到就拒绝开跑（这是防「名字写 noaug、实际跑 hflip」的那道检查，不放宽）。
# 探针同样按 `--augment none` 跑，所以后缀照实写。
PROBE_RUN = "EXP-014-budget-probe-v2-f1-noaug"

# 租来的卡每小时多少钱，与卡片第三节「设备」那一行一致（RTX 3090，¥1.48/时）。
# 只用来在控制台与日志里折算这一步花了多少钱，不影响任何训练或评估。
CARD_PRICE_PER_HOUR = 1.48


def train_cmd(key, fold=-1, tag=None):
    s = SETTINGS[key]
    return [PY, "-u", str(REPO / "experiments/common/video/train_ft.py"),
            "--line", "strong", "--augment", "none", "--caliber", "strict",
            "--epochs", "20", "--seed", "7", "--fold", str(fold),
            "--card", "EXP-014", "--tag", tag or s["run"],
            "--frames-root", str(s["frames"]),
            "--clip-labels", str(s["labels"]),
            "--meta", str(POOL / "meta-128.npz"),
            "--folds-json", str(FOLDS)]


def probe_cmd():
    """预算探针：原间隔这套设定只跑第 1 折，写进单独目录。

    为什么单独一个目录：正式那次必须是一次调用跑完五折——`train_ft.py` 每折跑完都
    重写 `metrics.json` 与 `predictions.csv`，换进程用 `--fold 2` 接着跑会把这些文件
    覆盖成只剩那一折，`clip_scores_ft.py` 的回归门禁（要求 predictions.csv 覆盖全部
    128 条）随即失败。所以预算复核用一次只跑一折的独立训练量速度，多花约一折租金
    （¥2 到 ¥5），换来正式产物与 EXP-009 走同一代码路径。
    """
    return train_cmd("v2", fold=1, tag=PROBE_RUN)


def run_dir_of(name, cmd):
    """这一步会写进的 run 目录；打分与判据步骤只读，返回 None（不去查它们的存在性）。"""
    if name == "probe-v2":
        return RUNS / PROBE_RUN
    if not name.startswith("train-"):
        return None
    for k in SETTINGS:
        if SETTINGS[k]["run"] in " ".join(cmd):
            return RUNS / SETTINGS[k]["run"]
    return None


def score_cmd(key, order):
    s = SETTINGS[key]
    # --features 只用来取 unit_ids / labels / folds 三列（不读特征本身），但它的默认值
    # 是 150 条那个缓存：不显式传就会因视频条数对不上而退出。传池子的 meta-128.npz，
    # 视频顺序与 clip-labels 逐条相等（2026-10-26 已核对）。
    return [PY, "-u", str(REPO / "experiments/tools/clip_scores_ft.py"),
            "--run", str(RUNS / s["run"]),
            "--units-root", str(s["frames"].parent),
            "--clip-labels", str(s["labels"]),
            "--features", str(POOL / "meta-128.npz"),
            "--frames", str(s["frames"]),
            "--frame-order", order]


def judge_cmd():
    # 配对模式只读两个 run 目录里的分数文件，--clip-labels 在这个模式下不起作用，
    # 所以不传（传了会让人误以为它参与了比较）。
    return [PY, "-u", str(REPO / "experiments/tools/motion_effect.py"),
            "--run-a", str(RUNS / SETTINGS["v2"]["run"]),
            "--run-b", str(RUNS / SETTINGS["v3"]["run"])]


def selfcheck_cmd(key):
    """单套自检：把正常、反序、时间冻结三套分数放在一条报告里读。

    配对判据只读正常与反序两套；冻结那一套（把 16 帧全换成中间一帧，等于抹掉时间）
    只剩这里有机会读。不跑这一步，卡片 §5.3 要求的时间冻结辅助读数就没人算。
    """
    s = SETTINGS[key]
    return [PY, "-u", str(REPO / "experiments/tools/motion_effect.py"),
            "--run", str(RUNS / s["run"]),
            "--clip-labels", str(s["labels"])]


def steps():
    out = [("probe-v2", "预算探针：原间隔这套设定只跑第 1 折，量一折耗时",
            probe_cmd())]
    for key in ("v2", "v3"):
        out.append((f"train-{key}", f"{SETTINGS[key]['name']}这套设定：五折 20 轮",
                    train_cmd(key)))
        for order in ("normal", "reverse", "freeze"):
            out.append((f"score-{key}-{order}",
                        f"{SETTINGS[key]['name']}这套设定：算 {order} 片段分数",
                        score_cmd(key, order)))
        out.append((f"selfcheck-{key}",
                    f"{SETTINGS[key]['name']}这套设定：三套帧序的自检报告",
                    selfcheck_cmd(key)))
    out.append(("judge", "两套设定配对判据", judge_cmd()))
    return out


def preflight(name):
    """跑之前先确认这一步要读的文件都在，缺了返回一句话说明，齐了返回 None。

    这样做是因为训练一步在租来的卡上要几小时：如果折权重没落盘就去打分，
    等到几小时后才发现缺文件，等于白花租金。
    """
    if name == "probe-v2" or name.startswith("train-"):
        key = "v2" if name == "probe-v2" else name.split("-", 1)[1]
        s = SETTINGS[key]
        if s["frames"].parent.name != s["frames_tag"]:
            return (f"{s['name']}这套设定的帧目录在 {s['frames'].parent.name}，"
                    f"应当是在 {s['frames_tag']}——帧与片段标签配错了")
        if not s["labels"].is_file():
            return f"缺片段标签 {s['labels']}"
        got = hashlib.md5(s["labels"].read_bytes()).hexdigest()
        if got != s["labels_md5"]:
            return (f"{s['labels'].name} 的 md5 是 {got}，"
                    f"与登记的 {s['labels_md5']} 不符（文件被换过或拷坏了）")
        got = hashlib.md5((POOL / "meta-128.npz").read_bytes()).hexdigest()
        if got != META_MD5:
            return f"meta-128.npz 的 md5 是 {got}，与登记的 {META_MD5} 不符"
    if name.startswith("score-"):
        key = name.split("-")[1]
        rd = RUNS / SETTINGS[key]["run"]
        missing = [f"ft_strong_f{f}.pt" for f in range(1, 6)
                   if not (rd / f"ft_strong_f{f}.pt").is_file()]
        if missing:
            return f"{rd} 里缺折权重 {missing}，先跑 train-{key}"
    if name.startswith("selfcheck-"):
        key = name.split("-")[1]
        rd = RUNS / SETTINGS[key]["run"]
        missing = [f.name for f in (rd / "clip-scores.npz", rd / "clip-scores-reverse.npz",
                                    rd / "clip-scores-freeze.npz") if not f.is_file()]
        if missing:
            return f"{rd} 里缺 {missing}，先跑 score-{key}-*"
    if name == "judge":
        missing = [str(RUNS / SETTINGS[k]["run"] / f"clip-scores{s}.npz")
                   for k in ("v2", "v3") for s in ("", "-reverse")
                   if not (RUNS / SETTINGS[k]["run"] / f"clip-scores{s}.npz").is_file()]
        if missing:
            return f"缺打分产物 {missing}，配对判据读不了"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--only", default="", help="只跑指定步骤（写步骤名，如 train-v2）")
    ap.add_argument("--upto", default="",
                    help="跑到指定步骤为止（含该步）。注意 --upto train-v2 会跑完原间隔"
                         "这套设定的全部 5 折——训练脚本一次调用就把五折跑完，不会在第 1 "
                         "折停下。预算复核用 probe-v2 这一步单独量一折，不靠中途杀进程")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    todo = steps()
    if args.only:
        todo = [t for t in todo if t[0] == args.only]
        if not todo:
            raise SystemExit(f"没有这一步：{args.only}；可选 "
                             f"{[t[0] for t in steps()]}")
    if args.upto:
        names = [t[0] for t in steps()]
        if args.upto not in names:
            raise SystemExit(f"没有这一步：{args.upto}；可选 {names}")
        todo = [t for t in todo if names.index(t[0]) <= names.index(args.upto)]
    if not todo:
        # --only 指定的步骤排在 --upto 之后时交集为空。不拦的话脚本会一步不跑、
        # 打印「全部步骤完成」并退出码 0，看的人以为跑过了。
        print(f"--only {args.only} 与 --upto {args.upto} 没有交集，本次一步都不会执行；"
              f"这两个参数只用其中一个", flush=True)
        raise SystemExit(2)

    log_path = RUNS / "EXP-014-pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for name, desc, cmd in todo:
        print(f"\n=== {name}：{desc}", flush=True)
        print(" ".join(cmd), flush=True)
        problem = preflight(name)
        run_dir = run_dir_of(name, cmd)
        clash = run_dir is not None and run_dir.exists()
        if args.dry_run:
            if problem:
                print(f"    [前置文件不全] {problem}", flush=True)
            if clash:
                print(f"    [输出目录已存在] {run_dir}；真跑时这一步会被拒绝", flush=True)
            continue
        if problem:
            raise SystemExit(f"{name} 的前置文件不全：{problem}")
        if clash:
            raise SystemExit(f"{run_dir} 已存在，拒绝覆盖已有成绩。"
                             f"要重跑请先改名或移走该目录")
        t0 = time.time()
        start = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n=== {name}：{desc}\n开始 {start}\n{' '.join(cmd)}\n")
            log.flush()
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
            used = time.time() - t0
            log.write(f"--- {name} 退出码 {rc}，用时 {used / 60:.1f} 分钟\n")
        # 单价跟着卡片第三节的设备走：原稿是 RTX 3080 Ti 的 ¥1.08，2026-10-27 换
        # RTX 3090 后卡片改成 ¥1.48。这里写成常量而不是散在格式化串里，换卡只改一处。
        print(f"    退出码 {rc}，用时 {used / 60:.1f} 分钟（约 "
              f"¥{used / 3600 * CARD_PRICE_PER_HOUR:.2f}）", flush=True)
        if rc != 0:
            raise SystemExit(f"{name} 失败（退出码 {rc}），后续步骤不执行；"
                             f"完整输出见 {log_path}")
    print(f"\n全部步骤完成；日志 {log_path}", flush=True)


if __name__ == "__main__":
    main()
