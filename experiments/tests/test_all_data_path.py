# -*- coding: utf-8 -*-
"""在 CPU 上用 10 条假视频跑通 train_ft.py 的两条路径。

要回答的问题：新加的「全部视频进训练、不留验证折」（`--all-data`）这条分支，
有没有改坏原来的五折路径，以及它自己的产物对不对。

假池子的做法：每条视频的 768 张帧图全部用硬链接指向同一张真实 jpg（不占第二份磁盘），
片段标签只放开 1 个片段，这样一轮只读 10 个片段，CPU 上也跑得动。折划分是 5 折 × 2 条，
每折一正一负。假池子建在系统临时目录，不写进仓库。

只在本次测试里替换、不改仓库文件的两项：
  - CARD_MD5：假池子自己的两个指纹（不绕过指纹核对，只是换成假池子的）；
  - CARD_BY_ID：轮数 1，免得在 CPU 上跑卡片登记的轮数。

两条路径用两个不同的卡号，因为「训练方式」这条核对按卡号判断该走哪种：卡号 EXP-013
要求 `--all-data`，其他卡号要求五折。五折路径若沿用 EXP-013，会被这条核对按设计拦下
（拦得对，但什么也没测到）。

用法（仓库根目录）：
  .venv/Scripts/python.exe experiments/tests/test_all_data_path.py             # 两条都跑
  .venv/Scripts/python.exe experiments/tests/test_all_data_path.py --only normal
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
POOL = Path(tempfile.gettempdir()) / "df-all-data-smoke" / "pool"
SRC_JPG = REPO / "data/process/EXP-008-units-v2/frames/v1_01/clip00_f00.jpg"
IDS = [f"smoke_{i:02d}" for i in range(10)]
LABELS = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]          # 每折（2 条）一正一负
FOLDS = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def build_pool():
    frames = POOL / "frames"
    if (frames / IDS[-1] / "clip47_f15.jpg").exists():
        print(f"假池子已存在，直接用：{POOL}")
        return
    if not SRC_JPG.is_file():
        sys.exit(f"缺参照帧图 {SRC_JPG}，先确认 data/process/EXP-008-units-v2/frames 在库")
    clip_label = np.full((len(IDS), 48), -1, dtype=int)
    for i, uid in enumerate(IDS):
        d = frames / uid
        d.mkdir(parents=True, exist_ok=True)
        # 每条视频用自己的那份拷贝做硬链接的源：NTFS 单个文件最多 1024 个硬链接，
        # 10 条视频 × 768 张全部指向同一个文件会超限（2026-10-27 实测 WinError 1142）。
        src = POOL / f"src_{uid}.jpg"
        if not src.exists():
            shutil.copyfile(SRC_JPG, src)
        for ci in range(48):
            for fi in range(16):
                p = d / f"clip{ci:02d}_f{fi:02d}.jpg"
                if not p.exists():
                    os.link(src, p)                # 硬链接：不复制内容
        clip_label[i, i] = LABELS[i]               # 只放开 1 个片段
    POOL.mkdir(parents=True, exist_ok=True)
    np.savez(POOL / "clip-labels.npz", unit_ids=np.array(IDS), strict=clip_label)
    np.savez(POOL / "unit-features.npz", unit_ids=np.array(IDS),
             labels=np.array(LABELS), folds=np.array(FOLDS))
    (POOL / "folds.json").write_text(json.dumps(
        {"folds": [{"fold": f, "val_units": sum(1 for x in FOLDS if x == f)}
                   for f in range(5)]}, ensure_ascii=False), encoding="utf-8")
    print(f"假池子已建好：{POOL}")


def run_branch(normal):
    """跑一条路径，返回 (产物目录, 该目录里的文件名字典)。"""
    os.environ["DF_RUNS_ROOT"] = str(POOL / ("runs-normal" if normal else "runs-all"))
    sys.path.insert(0, str(REPO / "experiments/common/video"))
    import train_ft as T

    clip = POOL / "clip-labels.npz"
    meta = POOL / "unit-features.npz"
    card = "SMOKE-NORMAL" if normal else "EXP-013"
    # 假池子登记自己的指纹：这是本次测试的输入，不是要绕过核对。
    T.CARD_MD5[card] = {"clip_labels": T.file_md5(clip),
                        "meta_npz": T.file_md5(meta)}
    T.CARD_BY_ID[card] = {"epochs": 1}

    argv = ["train_ft.py", "--line", "strong", "--augment", "none", "--card", card,
            "--epochs", "1", "--workers", "0",
            "--clip-labels", str(clip), "--meta", str(meta),
            "--frames-root", str(POOL / "frames"), "--folds-json", str(POOL / "folds.json")]
    if normal:
        argv += ["--tag", "smoke-normal-noaug"]
    else:
        argv += ["--tag", "smoke-all-data-noaug", "--all-data"]
    sys.argv = argv
    print("\n命令：" + " ".join(argv[1:]))
    T.main()
    out = Path(os.environ["DF_RUNS_ROOT"]) / argv[argv.index("--tag") + 1]
    files = {p.name: p.stat().st_size for p in sorted(out.iterdir())}
    print(f"\n产物目录：{out}")
    for k, v in files.items():
        print(f"  {k}  {v} 字节")
    return out, files


def check_normal(files):
    """五折路径：五个折权重 + 逐视频分数表，且不该出现全量训练那两个产物。"""
    bad = []
    for f in range(5):
        if f"ft_strong_f{f}.pt" not in files:
            bad.append(f"缺折权重 ft_strong_f{f}.pt")
    if "predictions.csv" not in files:
        bad.append("缺 predictions.csv")
    for name in ("final_strong.pt", "predictions-all.csv"):
        if name in files:
            bad.append(f"五折路径不应产出 {name}")
    if "metrics.json" not in files:
        bad.append("缺 metrics.json")
    return bad


def check_all_data(out, files):
    """全量路径：一个最终权重 + 训练集内分数表，且不该出现折权重。"""
    bad = []
    for name in ("final_strong.pt", "predictions-all.csv", "metrics.json"):
        if name not in files:
            bad.append(f"缺 {name}")
    for name in files:
        if name.startswith("ft_strong_f"):
            bad.append(f"全量路径不应产出折权重 {name}")
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    if m.get("all_data") is not True:
        bad.append(f"metrics.json 的 all_data 应为 true，实为 {m.get('all_data')!r}")
    auc = m.get("mean_auc")
    if auc is None or not (isinstance(auc, float) and auc != auc):
        bad.append(f"全量路径的 mean_auc 应为 NaN（不产生成绩），实为 {auc!r}")
    pred = (out / "predictions-all.csv").read_text(encoding="utf-8-sig").splitlines()
    if len(pred) - 1 != len(IDS):
        bad.append(f"predictions-all.csv 应有 {len(IDS)} 行，实有 {len(pred) - 1} 行")
    return bad


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    only = ""
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    build_pool()
    plan = [True, False] if not only else (
        [True] if only == "normal" else [False] if only == "all-data"
        else sys.exit("--only 只接受 normal 或 all-data"))
    fails = 0
    for normal in plan:
        name = "五折" if normal else "全量训练"
        out, files = run_branch(normal)
        bad = check_normal(files) if normal else check_all_data(out, files)
        if bad:
            fails += 1
            print(f"\n[{name}] 不通过 {len(bad)} 项：")
            for b in bad:
                print(f"  - {b}")
        else:
            print(f"\n[{name}] 通过")
    print(f"\n两条路径{'全部通过' if not fails else f'有 {fails} 条不通过'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
