# -*- coding: utf-8 -*-
"""按片段给**微调模型**打分：输出每条视频 48 个片段的分数、最高分片段号、整条分数。

与 `experiments/common/video/score_units.py` 的分工：

  `score_units.py`      用**已缓存的特征**给冻结骨干的两条线打分（--head mil / strong），秒级；
                        读不了微调模型的权重（它没有 HEADS 项，也没有读帧图的路径）。
  `clip_scores_ft.py`   用**帧图 + 骨干前向**给微调模型的 run 目录打分（权重 `ft_strong_f{折}.pt`），
                        本机 CPU 跑 150 条约 45 分钟。

为什么要这个工具（EXP-010 代码前置修复第 5 条）：三段级联要回答「第三段复核前后，
被撤掉的是哪些误报」，前提是知道每条视频 48 个片段各自的分数和最高分片段号。run 目录
的 `predictions.csv` 只存了「48 个片段取最大」这一个数，看不出片段级分布。

为什么带回归门禁：重算的「48 个片段取最大」必须与 run 目录里的 `predictions.csv` 逐条对上。
对不上就说明取帧、预处理、折归属或权重加载中有一处与训练时不同，输出不能用于任何结论。
门禁不通过即非零退出；这不只是提示，是硬性拒绝。

用法（仓库根）：

  .venv/Scripts/python.exe experiments/tools/clip_scores_ft.py \
      --run data/results/exp008/EXP-009-strong-v2-150-strict-ft-noaug

  # 只算几条
  .venv/Scripts/python.exe experiments/tools/clip_scores_ft.py --run <run 目录> v1_96 v5_19

产出（默认落在 run 目录，不覆盖任何已有文件）：

  clip-scores.npz    unit_ids / clip_scores[150,48] / mask[150,48] / labels / folds /
                     unit_score（可用片段的最大分）/ argmax_clip（取得最大分的片段号）

断点续跑（2026-10-25 加）：每算完 20 条落一次半成品 `<产物名>.partial.npz`，进程被
系统杀掉后，用同样的命令再启动就接着算，不从头再来。最终产物写好后半成品自动删除。

注意：片段分数是**过 Sigmoid 之后**的值（与 `predictions.csv` 同口径）。Sigmoid 单调，
所以「取得最大分的片段号」与 logit 口径相同。
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.paths import resolve_path  # noqa: E402

MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
CLIP_LEN = 16
N_CLIPS = 48
CHUNK = 8              # 分段前向，峰值显存/内存压得住（与训练同口径）
DECIDE_THR = 0.5       # 回归门禁的判定线


class Head(nn.Module):
    """与 train_ft.Head 同形（末层无 Sigmoid）。

    这里必须是一份逐层照抄的复制品，不能从 train_ft 导入——train_ft 在导入时会解析
    命令行并触碰 GPU 相关初始化。形态若有改动，两处必须同时改；回归门禁会抓住不一致。
    """

    def __init__(self, in_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def build(run_dir: Path, fold: int, pattern: str):
    """建网并载入第 fold 折的微调权重，返回 eval() 状态的网络。"""
    from torchvision.models.video import r2plus1d_18

    net = nn.Module()
    net.backbone = r2plus1d_18(weights=None)
    net.backbone.fc = nn.Identity()
    net.head = Head()
    run_dir = resolve_path(run_dir, must_exist=True)
    ckpt = run_dir / pattern.format(fold)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    net.load_state_dict(sd)
    net.eval()
    # 训练时 BatchNorm 全程 eval()；推理也必须如此，否则分数与 predictions.csv 对不上。
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
    return net


def preprocess(u8):
    """uint8 [N,16,H,W,3] -> float [N,3,16,H,W]，与 train_ft.preprocess 同口径。"""
    x = torch.from_numpy(np.ascontiguousarray(u8)).float().div_(255.0)
    m = torch.tensor(MEAN).view(1, 1, 1, 1, 3)
    sd = torch.tensor(STD).view(1, 1, 1, 1, 3)
    x = (x - m) / sd
    return x.permute(0, 4, 1, 2, 3)


def read_unit(frames_dir: Path, uid: str, order: str = "normal"):
    """读一条视频的 48 个片段 -> [48,16,112,112,3] uint8。

    与训练、与 `score_units.py` 同一取帧口径：cv2 读入、BGR->RGB、112×112 双线性。

    order 控制 16 帧的排列，用来做「模型到底看不看时间顺序」的对照，三种取值：
      normal  按原始时间顺序（训练与全部正式成绩用的就是这一种）；
      reverse 把 16 帧倒过来，其余一模一样。若模型真的用到运动方向，分数应变化；
      freeze  把 16 帧全部换成中间那一帧，等于把时间抹平，只剩单帧外观信息。
    三种排列只改帧的顺序或内容，不改取哪 48 个片段、不改预处理、不改权重。
    """
    d = resolve_path(frames_dir) / uid
    out = np.empty((N_CLIPS, CLIP_LEN, 112, 112, 3), dtype=np.uint8)
    for ci in range(N_CLIPS):
        for fi in range(CLIP_LEN):
            p = d / f"clip{ci:02d}_f{fi:02d}.jpg"
            buf = np.fromfile(str(p), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
            if img is None:
                raise FileNotFoundError(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out[ci, fi] = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    if order == "reverse":
        out = out[:, ::-1]
    elif order == "freeze":
        out = np.repeat(out[:, CLIP_LEN // 2:CLIP_LEN // 2 + 1], CLIP_LEN, axis=1)
    elif order != "normal":
        raise ValueError(f"未知的帧序：{order}")
    return out


def rank_max(scores_row, mask_row):
    """可用片段里的最大分与取得它的片段号；并列取片段号最小（与 D9 一致）。"""
    usable = np.flatnonzero(mask_row)
    if usable.size == 0:
        return float("nan"), -1
    vals = scores_row[usable]
    best = vals.max()
    # 并列时取片段号最小的那个：np.argmax 返回第一个最大值的位置，正是此意。
    return float(best), int(usable[int(np.argmax(vals))])


def main():
    # allow_abbrev=False：不加这个，写 `--units v1_01` 会被 argparse 当成 `--units-root`
    # 的缩写，把视频编号当成目录，报一句看不懂的「找不到 v1_01\clip-labels.npz」
    # （2026-10-25 实测踩到）。视频编号是位置参数，只能不带 -- 写。
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    ap.add_argument("--run", required=True, type=Path, help="run 目录（含 ft_*_f{折}.pt 与 predictions.csv）")
    ap.add_argument("--units-root", type=Path,
                    default=REPO / "data/process/EXP-008-units-v2",
                    help="取材正本根（含 clip-labels.npz 与 frames/）")
    ap.add_argument("--clip-labels", dest="clip_labels", type=Path, default=None,
                    help="片段标签 npz，默认 <units-root>/clip-labels.npz。换池子时"
                         "（如 EXP-014 的 clip-labels-v2-128.npz）必须显式传入："
                         "池目录里没有名为 clip-labels.npz 的文件，指回 150 条正本"
                         "又会因行数不同被拒（research-auditor 2026-10-25 B5）")
    ap.add_argument("--features", type=Path,
                    default=REPO / "data/results/exp008/unitfeats-v2-150/unit-features.npz",
                    help="特征缓存（只取其中的 unit_ids / labels / folds 三列，不读特征本身）")
    ap.add_argument("--frames", type=Path, default=None,
                    help="帧图根，默认 <units-root>/frames")
    ap.add_argument("--pattern", default="ft_strong_f{}.pt",
                    help="折权重文件名格式，默认 ft_strong_f{}.pt。全量训练只有一个"
                         "模型时传 final_strong.pt（文件名里没有折号占位，"
                         "format 后仍是它自己）")
    ap.add_argument("--pred-csv", dest="pred_csv", default="predictions.csv",
                    help="run 目录里逐视频分数表的文件名，默认 predictions.csv。"
                         "全量训练（EXP-013）写的叫 predictions-all.csv，用它才能把"
                         "回归门禁接上，而不是让门禁按「缺文件」直接判不通过")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出路径，默认写进 run 目录的 clip-scores.npz；已存在则拒绝覆盖")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="门禁允许的最大绝对差（本机 CPU 与 GPU 的浮点差实测 0.0023）")
    ap.add_argument("--force-fold", type=int, default=None,
                    help="用指定的折权重给全部视频打分（五折集成估计用）；此模式下跳过回归门禁，"
                         "因为分数本就不应与 predictions.csv 相同")
    ap.add_argument("--frame-order", dest="frame_order", default="normal",
                    choices=["normal", "reverse", "freeze"],
                    help="16 帧的排列：normal=原时间顺序；reverse=倒序；freeze=全换成中间帧。"
                         "非 normal 时跳过回归门禁（分数本就应当与 predictions.csv 不同）")
    ap.add_argument("units", nargs="*", help="只算这些视频（不给则算全部）")
    a = ap.parse_args()

    run_dir = resolve_path(a.run, must_exist=True)
    a.units_root = resolve_path(a.units_root)
    a.features = resolve_path(a.features, must_exist=True)
    if a.clip_labels is not None:
        a.clip_labels = resolve_path(a.clip_labels, must_exist=True)
    if a.frames is not None:
        a.frames = resolve_path(a.frames, must_exist=True)
    frames_dir = a.frames if a.frames is not None else (a.units_root / "frames")
    if a.out is not None:
        out_path = a.out
    elif a.force_fold is not None:
        out_path = run_dir / f"clip-scores-forcef{a.force_fold}.npz"
    elif a.frame_order != "normal":
        out_path = run_dir / f"clip-scores-{a.frame_order}.npz"
    else:
        out_path = run_dir / "clip-scores.npz"
    if out_path.exists():
        raise SystemExit(f"输出已存在，拒绝覆盖：{out_path}")

    z = np.load(a.features, allow_pickle=True)
    unit_ids = [str(u) for u in z["unit_ids"]]
    labels, folds = z["labels"], z["folds"]
    cl = np.load(a.clip_labels or (a.units_root / "clip-labels.npz"), allow_pickle=True)
    if [str(u) for u in cl["unit_ids"]] != unit_ids:
        raise SystemExit("片段标签与特征缓存的视频顺序不一致")
    mask = cl["strict"].astype(int) >= 0        # True = 这个片段可用

    preds = {}
    pred_csv = run_dir / a.pred_csv
    if pred_csv.is_file():
        with open(pred_csv, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                preds[r["unit_id"]] = float(r["score"])

    want = set(a.units)
    todo = [i for i, u in enumerate(unit_ids) if not want or u in want]
    missing = want - {unit_ids[i] for i in todo}
    if missing:
        raise SystemExit(f"清单里没有这些视频：{sorted(missing)}")
    todo_all = list(todo)          # 本次要覆盖的全部行；末了只核对这些行有没有分数

    # ---- 断点续跑：本机 150 条一遍约 70 分钟，中间被系统杀掉（实测：内存吃紧时
    # 进程收到 SIGSEGV，退出码 139）不该从头再来。每算完 20 条落一次半成品，
    # 下次用同样的命令启动就接着算。半成品与最终产物的差别只是文件名。
    partial_path = out_path.with_name(out_path.stem + ".partial.npz")
    scores = np.full((len(unit_ids), N_CLIPS), np.nan, dtype=np.float32)
    ffold = a.force_fold if a.force_fold is not None else -1
    done = set()
    if partial_path.is_file():
        # 必须用 with 关掉文件句柄：np.load 读 npz 时惰性打开文件，句柄不关，后面
        # unlink 半成品会在 Windows 上报「另一个程序正在使用此文件」（2026-10-25 实测）。
        with np.load(partial_path, allow_pickle=True) as pz:
            pz_ids = [str(u) for u in pz["unit_ids"]]
            pz_order = str(pz["frame_order"])
            pz_ffold = int(pz["force_fold"])
            pz_scores = pz["clip_scores"].astype(np.float32)
        same = (pz_ids == unit_ids and pz_order == a.frame_order and pz_ffold == ffold)
        if not same:
            raise SystemExit(f"半成品文件与本次参数不匹配（视频清单/帧序/折权重有一项不同），"
                             f"拒绝续跑。确认不要了再手动删除：{partial_path}")
        scores = pz_scores
        done = {unit_ids[i] for i in range(len(unit_ids)) if np.isfinite(scores[i]).any()}
        print(f"续跑：半成品里已有 {len(done)} 条分数，跳过它们")
    todo = [i for i in todo if unit_ids[i] not in done]

    def save_partial():
        """先写临时文件再改名，避免写到一半被杀留下坏文件。"""
        tmp = partial_path.with_name(partial_path.stem + ".tmp.npz")
        np.savez(tmp, unit_ids=np.array(unit_ids), clip_scores=scores, mask=mask,
                 labels=labels, folds=folds, frame_order=np.array(a.frame_order),
                 force_fold=np.array(ffold))
        os.replace(tmp, partial_path)

    print(f"要算 {len(todo)} 条视频；本机 CPU 线程 {torch.get_num_threads()}"
          f"（内存吃紧时可用环境变量 OMP_NUM_THREADS=4 降低线程数）")
    t0 = time.time()
    cur_fold, net = None, None
    for n, i in enumerate(todo, 1):
        f = a.force_fold if a.force_fold is not None else int(folds[i])
        if f != cur_fold:
            net = build(run_dir, f, a.pattern)
            cur_fold = f
        u8 = read_unit(frames_dir, unit_ids[i], a.frame_order)
        with torch.no_grad():
            outs = []
            for s in range(0, N_CLIPS, CHUNK):
                h = net.backbone(preprocess(u8[s:s + CHUNK]))
                outs.append(torch.sigmoid(net.head(h)).squeeze(-1))
            scores[i] = torch.cat(outs).numpy()
        best, _ = rank_max(scores[i], mask[i])
        used = time.time() - t0
        print(f"[{n}/{len(todo)}] {unit_ids[i]} 折{f} 可用{mask[i].sum()} "
              f"最大{best:.6f} 已用 {used:.0f}s 预计剩余 {used / n * (len(todo) - n):.0f}s",
              flush=True)
        if n % 20 == 0 and n < len(todo):
            save_partial()

    unit_score = np.full(len(unit_ids), np.nan, dtype=np.float32)
    argmax_clip = np.full(len(unit_ids), -1, dtype=np.int32)
    for i in range(len(unit_ids)):
        unit_score[i], argmax_clip[i] = rank_max(scores[i], mask[i])
    hole = [unit_ids[i] for i in todo_all if not np.isfinite(scores[i]).any()]
    if hole:
        raise SystemExit(f"这些视频没有分数，拒绝写最终产物（半成品保留可续跑）：{hole[:5]}")

    np.savez(out_path, unit_ids=np.array(unit_ids), clip_scores=scores, mask=mask,
             labels=labels, folds=folds, unit_score=unit_score, argmax_clip=argmax_clip,
             frame_order=np.array(a.frame_order))
    if partial_path.is_file():
        partial_path.unlink()
    print(f"写到 {out_path}（帧序 {a.frame_order}）")

    # ---- 回归门禁：重算的整条分数必须与 predictions.csv 对得上 ----
    if a.force_fold is not None:
        print(f"门禁跳过：--force-fold {a.force_fold} 是折外集成用途，"
              f"整数条分数本就与 predictions.csv 不同")
        return 0
    if a.frame_order != "normal":
        print(f"门禁跳过：帧序 {a.frame_order} 故意改了输入，分数本就应与 predictions.csv 不同。"
              f"这一遍的用途是与 normal 那一遍比分数变化，不是复现训练成绩。")
        return 0
    if not preds:
        print(f"门禁：run 目录没有 {a.pred_csv}，无法核对，按未通过处理")
        return 1
    common = [i for i in todo_all if unit_ids[i] in preds
              and np.isfinite(unit_score[i])]
    # 覆盖不全必须报错，不能只核对交集：predictions.csv 若只含部分折（例如训练是
    # 逐折分段跑、最后一次写盘只留下最后一折），交集变小、差值照样很小，门禁会
    # 静默通过，看起来「核对通过」而实际有几十条视频根本没核对。
    # 分母是「本次要算的行」而不是整个池子：命令行点名了 3 条视频时，另外 147 条
    # 本来就没有分数，拿全池当分母会把这条检查变成永远不通过（2026-10-27 审查 B2）。
    if len(common) != len(todo_all):
        cs = set(common)
        miss = [str(unit_ids[i]) for i in todo_all if i not in cs]
        print(f"门禁未通过：{a.pred_csv} 只覆盖 {len(common)}/{len(todo_all)} 条，"
              f"缺 {len(miss)} 条（前 5 条 {miss[:5]}）")
        return 1
    diffs = np.array([abs(float(unit_score[i]) - preds[unit_ids[i]]) for i in common])
    flips = [unit_ids[i] for i in common
             if (unit_score[i] >= DECIDE_THR) != (preds[unit_ids[i]] >= DECIDE_THR)]
    print(f"门禁：核对 {len(common)} 条，最大绝对差 {diffs.max():.6f}，"
          f"判定线 {DECIDE_THR} 上翻转 {len(flips)} 条")
    if flips:
        print(f"门禁未通过：翻转的视频 {flips[:10]}{' ...' if len(flips) > 10 else ''}")
        return 1
    if diffs.max() > a.tol:
        print(f"门禁未通过：最大绝对差 {diffs.max():.6f} 超过容差 {a.tol}")
        return 1
    print("门禁通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
