# -*- coding: utf-8 -*-
"""EXP-009：骨干解冻微调（唯一自变量 = 骨干是否参与训练，其余照抄现役脚本）。

对照 EXP-008 卡偏差 D5 预留的第一候选修正（docs/process/experiments/EXP-009.md，
叶海峰 2026-10-10 签字「明天跑微调」）：
  - 骨干：同一 torchvision r2plus1d_18（KINETICS400_V1），全部参数解冻；**每折独立
    重新构建（重新加载预训练权重）并新建优化器**（research-auditor 2026-10-14 B1，
    五折不得共用权重）；
  - 学习率分层：头 lr 1e-3（沿用），骨干 1e-4（预训练权重小学习率才不塌）；
  - 输入：每 epoch 直读帧图（data/process/EXP-008-units-v2/frames/<unit>/clip*.jpg，
    已过五组校验/锚点核对/人工抽查），随机水平翻转为唯一增强（**两线统一按片段
    翻转**：一个片段 16 帧同翻，片段间独立掷硬币——research-auditor 2026-10-14
    决策 2）。注记口径（W6）：冻结版管线无任何增强，微调成绩 = 解冻 + 翻转的
    合并效应，如实写入 metrics.json；
  - 损失与评估逐项照抄：MIL = masked L1+λt·L2+λs·L3（λt=λs=8e-5，margin 1，
    max/min 归约**直接复用 train_mil.masked_max_min**——research-auditor 2026-10-14
    B2/S1：掩码归约不写第二份实现，2026-10-05 事故即源于此）；强标签 =
    片段对 BCEWithLogitsLoss + pos_weight(负/正)，批组织/BS=256/可用片段对与
    train_strong.py 一致；
  - 评估：单位分数 = max(可用片段分数)（强标签线无条件 sigmoid，与 train_strong
    unit_scores 同口径，W1），折内 val 选 best epoch，每折 rng 以 seed=7 重置（W7）；
  - 显存适配（12 GB 租卡）：骨干前向按 CHUNK=8 片段分段 + 梯度检查点（反向
    重算换显存）。2026-10-15 远端实测（probe_mem/probe_mem2）：r2plus1d_18 的
    stem/layer1 输出为 64ch×56²×16T（空间 56²、时间 16 保持到 layer2/3 才降），
    95 片段整批激活图 ≈11 GiB；torch 2.5.1 checkpoint(use_reentrant=False) 在
    「输入不带梯度标记」时不丢弃激活（警告 None of the inputs have
    requires_grad=True，CHUNK=24/48、重入/非重入四种组合全部实测 OOM）——修法
    为分段输入 x.requires_grad_(True)（transformers/accelerate 同款标准做法）
    后检查点恢复省显存，重算时单段激活全活峰值 ≈2 GiB。96 片段整批激活图
    ≈11 GiB；**纯分段不用检查点不可行**：训练态所有段的激活图必须保留到反向，
    分段数不减总量（冒烟 11.25 GiB OOM 即此数）。数值路径：检查点语义 = 普通前向，
    本地 CPU 冒烟回归两次损失一致；FP32 不用 AMP——微调数值路径不引入新变量；
    检查点重算会二次更新 BN 统计量，故骨干 BN 固定用运行统计量（BN 层 eval，
    仿射参数 γ/β 照常训练；research-auditor 2026-10-14 决策 1 认可，冒烟须留
    running_mean 前后差值全零 + γ 梯度非零的执行证据，S4）。

用法（仓库根，venv 或云主机）：
  python experiments/common/video/train_ft.py --line strong --augment none --tag EXP-009-strong-v2-150-strict-ft-noaug
  python experiments/common/video/train_ft.py --line mil    --augment none --tag EXP-009-mil-v2-150-strict-ft-noaug
  python experiments/common/video/train_ft.py --line strong --augment none --smoke --epochs 1   # 冒烟自动选 1 正 1 负
  # 最终模型（EXP-013）：全部视频进训练、不留验证折，轮数事先定死
  python experiments/common/video/train_ft.py --line strong --augment none --card EXP-013 \
      --all-data --epochs 2 --tag EXP-013-final-150
环境变量：DF_RUNS_ROOT 覆盖 runs 根。
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from torch.utils.checkpoint import checkpoint as _ckpt   # 显式导入，不依赖惰性挂载

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from train_mil import masked_max_min                     # noqa: E402  唯一掩码归约来源
from card_checks import (CardCheck, count_trainable,     # noqa: E402  卡片核对框架
                         optimizer_lr_of, optimizer_covers_all, bn_training_states)


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments.paths import resolve_path  # noqa: E402
RUNS_ROOT = resolve_path(os.environ.get("DF_RUNS_ROOT") or REPO / "data/results/exp008")
CLIP_LABELS = REPO / "data/process/EXP-008-units-v2/clip-labels.npz"
FRAMES_ROOT = REPO / "data/process/EXP-008-units-v2/frames"
SEED = 7
LAM_T = LAM_S = 8e-5          # 与 train_mil.py 同源
BS_STRONG = 256               # train_strong.py 的片段对批大小
BS_MIL_UNITS = 8              # train_mil.py 的单位批大小（正单位无放回，负循环补齐）
CHUNK = 8                     # 骨干分段前向的片段数（带梯度检查点）。取小不取大：
                              # 反向重算时整段激活图全活，64ch@56²@16T 单张量
                              # 103 MB/8片段，单段全活 ≈2 GiB；CHUNK=24 重算全活
                              # ≈6.4 GiB 峰值过紧（2026-10-15 实测定 8）。
                              # 分段数学等价以「骨干 BN 固定 eval（运行统计量）」
                              # 为前提——BN 若改回 train 态，分段统计会静默偏离
                              # 整批口径（research-auditor 2026-10-15 S-2）。
DEGENERATE_AUC = 0.55         # 折内验证 AUC 不超过此值（≤ 0.55，与停跑规则的
                              # trailing_low_auc_streak 同一口径）即视为「与瞎猜
                              # 无异或更差」
DEGENERATE_STREAK = 5         # 末尾连续这么多轮的验证 AUC 都不高于上面那个值，
                              # 才把这一折判为退化。
                              #
                              # 为什么不能拿「折内最好的一轮」来判：2026-10-15 那次
                              # 弱标签线 f2 的验证 AUC 从第 0 轮的 0.738 一路掉到第 9
                              # 轮起恒为 0.500，最好的一轮恰恰是崩塌之前的 0.738，大于
                              # 0.55，于是「最好的一轮 ≤ 0.55」这条判定永远不触发。
                              # 判退化要看末尾，不看峰值。
                              #
                              # 为什么取 5：同一次 f1 折的验证 AUC 在 0.000 到 1.000
                              # 之间来回跳，最长的连续低分是 3 轮；真崩的 f2 是 11 轮。
                              # 取 5 落在两者中间，既不误伤来回跳的折，也不会漏掉真崩的。
DEGENERATE_STREAK_LABEL = "连续低分轮数"


def trailing_low_auc_streak(hist, thr=DEGENERATE_AUC):
    """返回 `hist` 末尾连续多少轮的验证 AUC 不高于 `thr`。

    `hist` 是逐轮记录表，每项含 `va_auc`。末尾断一次就重新计数，所以只看最后
    一段连续的。验证 AUC 为 nan（验证折单一类别）时视为断点。"""
    k = 0
    for h in reversed(hist):
        a = h.get("va_auc")
        if a is None or a != a or a > thr:
            break
        k += 1
    return k


def early_stop_due(hist, stop_low_streak, fold_name):
    """这一折是否该当场停跑；该停时打印原因并返回真。

    `stop_low_streak` 为 0 表示不停（冒烟折与单折调试走这条）。"""
    if stop_low_streak <= 0:
        return False
    k = trailing_low_auc_streak(hist)
    if k < stop_low_streak:
        return False
    print(f"  [停跑] {fold_name} 末尾连续 {k} 轮验证 AUC ≤ {DEGENERATE_AUC}"
          f"（已跑 {len(hist)} 轮），按卡片预登记规则停这一折；"
          f"已跑的轮次与最好一轮照常落盘", flush=True)
    return True
N_WORKERS = 16                # 帧解码进程数（租卡机 80 核）
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


class Head(nn.Module):
    """强标签头：与 StrongHead 同形（末层无 Sigmoid，配 BCEWithLogitsLoss）。"""

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


class MilHead(nn.Module):
    """Wu 四级检测 MLP（与 train_mil.MilHead 同构）：512→128→64→32→1 + Sigmoid。"""

    def __init__(self, in_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class FTNet(nn.Module):
    """骨干（解冻）+ 头。backbone 输入 [N,3,16,112,112] -> [N,512]。

    每折新建（构造即加载 KINETICS400_V1 预训练权重），折间不共享权重与优化器
    （research-auditor 2026-10-14 B1）。"""

    def __init__(self, line):
        super().__init__()
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        self.backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        self.backbone.fc = nn.Identity()      # 与 extract_feats.make_backbone 同：取 512 维
        self.head = MilHead() if line == "mil" else Head()

    def train(self, mode=True):
        """BN 层保持 eval（运行统计量不更新，规避检查点重算的二次更新）；
        其余层照常 train/eval。仿射参数 γ/β 仍 requires_grad，照常参与训练
        （research-auditor 2026-10-14 决策 1：与 FrozenBatchNorm 数学等价）。"""
        super().train(mode)
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()
        return self


# ---------------------------------------------------------------- 数据读取

def read_clip(unit_dir: Path, ci: int, clip_len: int = 16) -> np.ndarray:
    """读一片段 16 帧 jpg -> [16,112,112,3] uint8（与 extract_feats.load_clip 同口径：
    cv2 BGR->RGB，112×112 INTER_LINEAR；归一化移到 GPU 做）。"""
    frames = np.empty((clip_len, 112, 112, 3), dtype=np.uint8)
    for fi in range(clip_len):
        p = unit_dir / f"clip{ci:02d}_f{fi:02d}.jpg"
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"缺失帧 {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames[fi] = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    return frames


class UnitDataset(Dataset):
    """一条单位 = 48 片段帧图，[48,16,112,112,3] uint8（MIL 训练与全部评估用）。"""

    def __init__(self, unit_ids, frames_root):
        self.unit_ids = list(unit_ids)
        self.root = Path(frames_root)

    def __len__(self):
        return len(self.unit_ids)

    def __getitem__(self, k):
        d = self.root / str(self.unit_ids[k])
        return np.stack([read_clip(d, ci) for ci in range(48)])


def horizontal_flip(u8):
    """左右翻转：把宽度那一维倒过来。

    u8 的形状是 [..., 16 帧, 高, 宽, 3 通道]，所以宽度维是倒数第二维（下标 -2），
    不是最后一维（颜色通道，下标 -1）。写成 flip(-1) 会把红蓝两个通道对调——
    程序不报错、训练照跑、成绩照出，只是增强变成了颜色扰动（2026-10-15 实测
    事故：卡片写「随机水平翻转」，代码实际翻颜色通道，两线都受影响）。
    本函数与 assert_augment_axis() 配套：翻转只此一处实现，两条线共用，
    启动时自证翻的是哪一维。"""
    return u8.flip(-2)


def augment_batch(u8, flip):
    """按 flip 掩码对 u8 做左右翻转——**全脚本唯一的增强入口**。

    调用方不得在别处自己写 `.flip(...)`：2026-10-15 的事故就是两处调用点各写
    一行 `u8[flip] = u8[flip].flip(-1)`，错则两处同错，且审查时没人会去逐个
    调用点确认翻的是哪一维。收成一个入口后，「翻的是哪一维」只在这个函数里
    有答案，`assert_augment_axis()` 每次启动自证一次。

    u8 形状 [..., 16 帧, 高, 宽, 3 通道]（在 CPU 上），flip 的形状与 u8 的前几维
    一致（如 [批内视频数, 每视频片段数]）。就地改写并返回同一个张量。"""
    if flip.any():
        u8[flip] = horizontal_flip(u8[flip])
    return u8


def augment_label(augment):
    """增强设定对应的文字。启动打印与 metrics.json 共用这一份。

    两处共用是为了防止「参数说无增强、落盘的字段却写随机水平翻转」——那正是
    2026-10-15 这次事故的镜像：字段与实现不符，只不过方向反过来。"""
    return {"none": "无增强",
            "hflip": "随机水平翻转（片段级，16 帧同翻）"}[augment]


def apply_augment(u8, flip_shape, rng, augment):
    """按 augment 设定决定这一批翻不翻。

    augment="none" 时**连随机数都不消耗**：若照样掷硬币再丢弃，无增强与有增强
    两套的随机流就不一样，「增强」这一项之外会多出一个不受控的变量，两套成绩
    不可比（research-auditor 2026-10-15）。"""
    if augment == "none":
        return u8
    flip = torch.from_numpy(rng.random(flip_shape) < 0.5)
    return augment_batch(u8, flip)


def assert_augment_axis():
    """启动自检：证明 horizontal_flip 翻的是宽度维、没有动颜色维。

    这类「轴写错但不报错」的错误在本仓库出现过三次（掩码填充方向写反、
    v1_34 时间轴坐标偏离、本次翻转轴写错），共同点都是程序照跑、数照出，
    只有专门去问「翻的是哪一维」才查得出来。所以把这个问题写进代码，
    每次启动自动回答一次，耗时 0 秒且跑在计费之前。

    探针用六维张量、每维长度互不相同（2×3×16×4×5×3），与真实张量
    [视频, 片段, 16 帧, 高, 宽, 通道] 同形。六维且各维不等长，是为了让
    「翻错了维」必定改变结果：若某一维长度为 1 或两维等长，翻错维可能恰好
    得到相同张量，自检就放行了（research-auditor 2026-10-15 G2）。"""
    probe = torch.arange(2 * 3 * 16 * 4 * 5 * 3, dtype=torch.uint8).reshape(2, 3, 16, 4, 5, 3)
    out = horizontal_flip(probe)
    assert out.shape == probe.shape, "horizontal_flip 改了张量形状"
    assert torch.equal(out, probe.flip(-2)), "horizontal_flip 翻转的不是宽度维"
    assert not torch.equal(out, probe.flip(-1)), "horizontal_flip 翻的是颜色通道（轴写成了 -1）"
    return True


def preprocess(u8):
    """[N,16,H,W,3] uint8 -> [N,3,16,112,112] float（Kinetics 归一化，GPU 上做）。"""
    x = u8.permute(0, 1, 4, 2, 3).float().div_(255.0)      # [N,16,3,H,W]
    m = torch.tensor(MEAN, device=x.device).view(1, 1, 3, 1, 1)
    sd = torch.tensor(STD, device=x.device).view(1, 1, 3, 1, 1)
    x = (x - m) / sd
    return x.permute(0, 2, 1, 3, 4).contiguous()           # [N,3,16,H,W]


def backbone_chunks(backbone, u8, train_mode):
    """uint8 [N,16,112,112,3] 分段预处理 + 前向 -> [N,512]。
    归一化也按段做：整批 256 片段一次转 float 是 616 MB 本体、加中间拷贝峰值
    约 1.8 GB，与已驻留项叠加后在 12 GB 卡上 OOM（2026-10-15 远端冒烟实测），
    拆段后归一化峰值只剩 0.19 GB。
    训练态：checkpoint(use_reentrant=False) + 分段输入 requires_grad_(True)。
    **必须带梯度标记**：torch 2.5.1 对「输入无 requires_grad」的检查点不丢弃
    激活（2026-10-15 实测四种组合全 OOM，警告 None of the inputs have
    requires_grad=True），标记后检查点恢复省显存；x 是无参数运算的叶张量，
    标记不改变前向数值，其梯度算出即随作用域丢弃。评估态：调用方已 no_grad，
    直接逐段前向。"""
    outs = []
    for s in range(0, u8.shape[0], CHUNK):
        x = preprocess(u8[s:s + CHUNK])
        if train_mode:
            x.requires_grad_(True)
            outs.append(_ckpt(backbone, x, use_reentrant=False))
        else:
            outs.append(backbone(x))
    return torch.cat(outs, dim=0)


class fixed_sampler(torch.utils.data.Sampler):
    """按给定顺序取下标（顺序由主进程 rng 决定，DataLoader 只负责并行解码）。"""

    def __init__(self, order):
        self.order = np.asarray(order)

    def __iter__(self):
        return iter(self.order.tolist())

    def __len__(self):
        return len(self.order)


class IndexedClipDataset(Dataset):
    """一个可用片段 = 16 帧帧图 + 自身下标（对齐 y_clip[perm]）。"""

    def __init__(self, unit_ids, pairs, frames_root):
        self.unit_ids = list(unit_ids)
        self.pairs = list(pairs)                  # [(unit_idx, clip_idx), ...]
        self.root = Path(frames_root)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        i, j = self.pairs[k]
        return read_clip(self.root / str(self.unit_ids[i]), j), k


def bn_layers(net):
    return [m for m in net.backbone.modules() if isinstance(m, nn.BatchNorm3d)]


# ---------------------------------------------------------------- MIL 线

def mil_loss_masked(p, n, mp, mn):
    """与 train_mil.py masked 路径逐位一致：L1 峰谷差 margin 1（±inf 填充经
    train_mil.masked_max_min 单一来源，B2/S1），L2 只计相邻双可用对，L3 只累加
    可用片段。p/n: [B,48] sigmoid 分数；mp/mn: [B,48] bool（True=可用）。"""
    maxp, minp = masked_max_min(p, mp)
    maxn, minn = masked_max_min(n, mn)
    l1 = torch.max(torch.zeros_like(maxp),
                   1.0 - ((maxp - minp) - (maxn - minn))).mean()
    pv = mp[:, 1:] & mp[:, :-1]
    l2 = (((p[:, 1:] - p[:, :-1]) ** 2) * pv).sum(dim=1).mean()
    l3 = (p * mp).sum(dim=1).mean()
    return l1 + LAM_T * l2 + LAM_S * l3, l1, l2, l3


def unit_batches(pos_idx, neg_idx, rng):
    """train_mil.py 批组织：正单位洗牌无放回，负单位洗牌后循环补齐到同总数；
    返回 [(pi, ni), ...]，每批负单位数 = 同批正单位数。"""
    pos_order = rng.permutation(pos_idx)
    neg_order = rng.permutation(neg_idx)
    if len(neg_order) < len(pos_order):
        neg_order = np.concatenate([
            np.tile(neg_order, len(pos_order) // len(neg_order)),
            neg_order[:len(pos_order) % len(neg_order)]])
    batches = []
    for b in range(0, len(pos_order), BS_MIL_UNITS):
        pi = pos_order[b:b + BS_MIL_UNITS]
        batches.append((pi, neg_order[b:b + len(pi)]))
    return batches


def run_mil(net, opt, dev, unit_ids, unit_labels, clip_label, mask, tr, va,
            epochs, fold_name, rng, loader_kw, augment, stop_low_streak=0):
    """MIL 微调：单位批照抄 train_mil.py；骨干在环（checkpoint 分段前向）；
    翻转按片段（16 帧同翻，片段间独立掷硬币，决策 2），翻转决定在主进程 rng。
    clip_label 形参仅为与 run_strong 保持同一调用签名，MIL 损失只用单位标签
    与片段可用掩码，不读片段标签。"""
    tr, va = np.asarray(tr), np.asarray(va)
    y = np.asarray(unit_labels)
    pos_tr, neg_tr = tr[y[tr] == 1], tr[y[tr] == 0]
    assert len(pos_tr) >= 1 and len(neg_tr) >= 1, \
        f"{fold_name}: 训练折正 {len(pos_tr)} 条、负 {len(neg_tr)} 条，L1 需同时含两类视频"
    M = torch.tensor(mask, device=dev)
    ds = UnitDataset(unit_ids, FRAMES_ROOT)
    best_auc, best_state, best_r, best_ep = -1.0, None, None, -1
    t0 = time.time()
    hist = []
    smoke = fold_name == "smoke"
    for ep in range(epochs):
        net.train()
        batches = unit_batches(pos_tr, neg_tr, rng)
        flat = np.concatenate([np.concatenate(b) for b in batches])
        # 同 epoch 一次预取全部参训单位的帧（单位级 48×16 帧），主进程再切批
        ld = DataLoader(ds, batch_size=None, sampler=fixed_sampler(flat),
                        **loader_kw)
        cache = {}
        for k, u in zip(flat, ld):
            cache[int(k)] = u
        acc = {"l": 0.0, "l1": 0.0, "l2": 0.0, "l3": 0.0,
               "h_mean": 0.0, "h_std": 0.0, "h_norm": 0.0,
               "posMax": 0.0, "negMax": 0.0,
               "posRange": 0.0, "negRange": 0.0}
        for bi, (pi, ni) in enumerate(batches):
            idx = np.concatenate([pi, ni])
            u8 = torch.stack([cache[int(i)] for i in idx])   # [B,48,16,H,W,3]
            # 片段级翻转：[len(idx), 48] 掷硬币，命中的片段 16 帧同翻（决策 2）
            u8 = apply_augment(u8, (len(idx), 48), rng, augment)
            B, N = u8.shape[:2]
            if smoke and ep == 0 and bi == 0:
                snap = bn_evidence(net, dev, "before")
            # uint8 整批上 GPU（16×48 片段仅 462 MB），预处理在分段循环内逐段做
            h = backbone_chunks(net.backbone, u8.view(B * N, *u8.shape[2:]).to(dev), True)
            # 头输入尺度观测量（2026-10-15 立）：弱标签线这次跑到分数塌成常数，
            # 有两种可能——头输入尺度漂移把 Sigmoid 推饱和，或损失本身不约束分数
            # 的绝对水平。这两个量的走势能把两种原因分开（判据预登记在卡片里）。
            with torch.no_grad():
                acc["h_mean"] += float(h.mean())
                acc["h_std"] += float(h.std())
                acc["h_norm"] += float(h.norm(dim=1).mean())
            s = net.head(h).squeeze(-1).view(B, N)
            p, n = s[:len(pi)], s[len(pi):]
            mp, mn = M[pi], M[ni]
            loss, l1, l2, l3 = mil_loss_masked(p, n, mp, mn)
            with torch.no_grad():
                # 正负两类视频各自的片段分数峰值（批均值），照抄冻结线
                # train_mil.py 的算法；仅观测，不进计算图
                pmax = p.masked_fill(~mp, float("-inf")).max(dim=1).values
                pmin = p.masked_fill(~mp, float("inf")).min(dim=1).values
                nmax = n.masked_fill(~mn, float("-inf")).max(dim=1).values
                nmin = n.masked_fill(~mn, float("inf")).min(dim=1).values
                acc["posMax"] += float(pmax.mean())
                acc["negMax"] += float(nmax.mean())
                # 峰谷差本身（一类之内最高分减最低分）。L1 只看两类的峰谷差**之差**，
                # 从 l1 的数值反推不出两者各自多大，所以必须单独记：2026-10-15 那次
                # 弱标签线第 2 折「两类都塌成常数」与第 3 折「峰谷差拉开了但分数整体
                # 反序」两种形态，只看 l1 分不开（research-auditor 2026-10-15 S6/W7）。
                acc["posRange"] += float((pmax - pmin).mean())
                acc["negRange"] += float((nmax - nmin).mean())
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{fold_name} ep{ep}: 损失非有限（{loss.item()}）——拒绝静默继续")
            opt.zero_grad()
            loss.backward()
            if smoke and ep == 0 and bi == 0:
                g = float(bn_layers(net)[0].weight.grad.norm())
                print(f"  [S4] BN γ 梯度范数（应非零）: {g:.6f}", flush=True)
            opt.step()
            acc["l"] += float(loss.item()); acc["l1"] += float(l1.item())
            acc["l2"] += float(l2.item()); acc["l3"] += float(l3.item())
        for k in acc:
            acc[k] /= len(batches)
        if smoke and ep == 0:
            print(f"  [S4] BN running_mean 前后最大差值（应全零）: "
                  f"{bn_evidence(net, dev, 'after', snap)}", flush=True)
        r = eval_units(net, dev, ds, y, M, va, loader_kw, sigmoid=False)
        hist.append({"ep": ep, **{k: round(v, 6) for k, v in acc.items()},
                     "va_auc": r["auc"]})
        if not np.isnan(r["auc"]) and r["auc"] > best_auc:
            best_auc, best_r = r["auc"], r
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
            best_ep = ep
        print(f"  [{fold_name}] ep{ep} L={acc['l']:.4f} (L1={acc['l1']:.4f} "
              f"L2={acc['l2']:.4e} L3={acc['l3']:.4f}) vaAUC={r['auc']:.3f} "
              f"vaF1={r['f1']:.3f}{' *' if best_ep == ep else ''} "
              f"hNorm={acc['h_norm']:.3f} posMax={acc['posMax']:.3f} "
              f"negMax={acc['negMax']:.3f} posRange={acc['posRange']:.4f} "
              f"negRange={acc['negRange']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if early_stop_due(hist, stop_low_streak, fold_name):
            break
    return best_state, best_auc, best_r, best_ep, hist


# ---------------------------------------------------------------- 强标签线

def run_strong(net, opt, dev, unit_ids, unit_labels, clip_label, mask,
               tr, va, epochs, fold_name, rng, loader_kw, augment,
               stop_low_streak=0):
    """强标签微调：片段对批组织照抄 train_strong.py（全部可用 (unit,clip) 对
    洗牌，BS=256，pos_weight=负/正）。骨干在环：批内片段按 CHUNK 分段前向
    （checkpoint），一次 backward。翻转按片段（16 帧同翻），决定在主进程 rng。

    `va` 传空表示全部视频都拿来训练、不留言出折（EXP-013 最终模型）：没有验证集
    就没有逐轮评估、没有按验证 AUC 选出的「最好一轮」，每轮结束都把权重记一份，
    循环结束时留下的就是最后一轮的权重。此时返回的 best_auc 是 nan、best_r 是
    占位的 nan 字典——训练集内分数不能当成绩，调用方也不许拿它充数。"""
    tr, va = np.asarray(tr), np.asarray(va)
    holdout = len(va) > 0
    y_unit = np.asarray(unit_labels)
    M = torch.tensor(mask, device=dev)
    pairs = [(i, j) for i in tr for j in range(48) if clip_label[i][j] >= 0]
    y_clip = np.array([clip_label[i][j] for i, j in pairs], dtype=np.float32)
    n_pos, n_neg = int((y_clip == 1).sum()), int((y_clip == 0).sum())
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), device=dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  [{fold_name}] 可用片段 {len(pairs)}（pos {n_pos}/neg {n_neg}，"
          f"pos_weight={pos_weight:.2f}）", flush=True)
    ds = IndexedClipDataset(unit_ids, pairs, FRAMES_ROOT)
    best_auc, best_state, best_r, best_ep = -1.0, None, None, -1
    t0 = time.time()
    hist = []
    smoke = fold_name == "smoke"
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(pairs))
        ld = DataLoader(ds, batch_size=BS_STRONG, sampler=fixed_sampler(perm),
                        **loader_kw)
        losses = []
        for u8, sel in ld:                                 # sel = perm 内位置
            sel = sel.numpy()
            u8 = apply_augment(u8, (len(u8),), rng, augment)
            if smoke and ep == 0 and not losses:
                snap = bn_evidence(net, dev, "before")
            # uint8 整批上 GPU（256 片段仅 154 MB），预处理在分段循环内逐段做
            h = backbone_chunks(net.backbone, u8.to(dev), True)
            logits = net.head(h).squeeze(-1)               # [bs]
            loss = crit(logits, torch.from_numpy(y_clip[sel]).to(dev))
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{fold_name} ep{ep}: 损失非有限（{loss.item()}）——拒绝静默继续")
            opt.zero_grad()
            loss.backward()
            if smoke and ep == 0 and not losses:
                g = float(bn_layers(net)[0].weight.grad.norm())
                print(f"  [S4] BN γ 梯度范数（应非零）: {g:.6f}", flush=True)
            opt.step()
            losses.append(float(loss.item()))
        if smoke and ep == 0:
            print(f"  [S4] BN running_mean 前后最大差值（应全零）: "
                  f"{bn_evidence(net, dev, 'after', snap)}", flush=True)
        if holdout:
            r = eval_units(net, dev, UnitDataset(unit_ids, FRAMES_ROOT), y_unit,
                           M, va, loader_kw, sigmoid=True)  # 强标签线无条件 sigmoid（W1）
            hist.append({"ep": ep, "bce": round(float(np.mean(losses)), 6),
                         "va_auc": r["auc"]})
            if not np.isnan(r["auc"]) and r["auc"] > best_auc:
                best_auc, best_r = r["auc"], r
                best_state = {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()}
                best_ep = ep
            print(f"  [{fold_name}] ep{ep} bce={np.mean(losses):.4f} "
                  f"vaAUC={r['auc']:.3f} vaF1={r['f1']:.3f}"
                  f"{' *' if best_ep == ep else ''} ({time.time()-t0:.0f}s)", flush=True)
        else:
            # 没有验证集：不评估，每轮都存一份权重，最后一轮的那份就是返回值。
            hist.append({"ep": ep, "bce": round(float(np.mean(losses)), 6),
                         "va_auc": float("nan")})
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
            best_ep = ep
            print(f"  [{fold_name}] ep{ep} bce={np.mean(losses):.4f}"
                  f"（全部视频训练，无留出折，不评估）({time.time()-t0:.0f}s)",
                  flush=True)
        if early_stop_due(hist, stop_low_streak, fold_name):
            break
    if not holdout:
        _nan = float("nan")
        best_auc = _nan
        best_r = {"auc": _nan, "f1": _nan, "precision": _nan, "recall": _nan}
    return best_state, best_auc, best_r, best_ep, hist


# ---------------------------------------------------------------- 评估与主流程

def aggregate_unit_score(clip_scores, avail):
    """一条视频的分数 = 它所有**可用**片段分数的最大值。

    clip_scores: [48] 张量（该视频 48 个片段的分数）；avail: [48] 布尔，True = 可用。
    不可用的片段（落在剪辑区间里，即掩码剔除的那些）不参与取最大。归约只此一处，
    评估侧不得另写第二份（2026-10-05 掩码事故的成因之一就是同一份归约写了两遍、
    只核对了一遍）。"""
    return clip_scores[avail].max()


def eval_units(net, dev, ds, unit_labels, M, idx, loader_kw, sigmoid):
    """单位分数 = max(可用片段分数)（eval 模式，Dropout 关；无翻转）。
    mil：头自带 Sigmoid，分数直接用；strong：无条件 sigmoid（W1，与
    train_strong.unit_scores 同口径，不做数值范围启发式）。"""
    net.eval()
    ld = DataLoader(ds, batch_size=None, sampler=fixed_sampler(idx),
                    **{**loader_kw, "pin_memory": False})
    rows = {}
    with torch.no_grad():
        for k, u in zip(idx, ld):
            h = backbone_chunks(net.backbone, u.to(dev), False)  # uint8 入，段内预处理
            s = net.head(h).squeeze(-1)                # [48]
            if sigmoid:
                s = torch.sigmoid(s)
            rows[int(k)] = s
    scores = np.array([float(aggregate_unit_score(rows[int(i)], M[i])) for i in idx])
    y = np.asarray(unit_labels)[idx]
    auc = roc_auc_score(y, scores) if len(set(y.tolist())) > 1 else float("nan")
    binary = (scores >= 0.5).astype(int)
    return {"auc": round(float(auc), 4),
            "f1": round(float(f1_score(y, binary)), 4),
            # 键名写全 Precision / Recall：单字母 p/r 在同一个 JSON 里和「正类条数」
            # 的 n、以及各处的 p 变量撞脸，读的人得回去翻代码才知道指什么。
            "precision": round(float(precision_score(y, binary, zero_division=0)), 4),
            "recall": round(float(recall_score(y, binary, zero_division=0)), 4),
            "scores": scores}


def bn_evidence(net, dev, phase, snap=None):
    """S4：BN=运行统计量决策的执行证据。
    phase='before'：返回前 3 层 BN 的 running_mean 快照；
    phase='after'：返回与快照的最大差值列表（应全零——BN 层被固定在 eval，
    running stats 不得被训练或检查点重算改动）。"""
    ls = bn_layers(net)
    if phase == "before":
        return [m.running_mean.cpu().clone() for m in ls[:3]]
    return [round(float((m.running_mean.cpu() - s).abs().max()), 10)
            for m, s in zip(ls[:3], snap)]


def file_md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def preflight_frames(unit_ids, frames_root):
    """S1：启动即核对 meta 每条单位的帧图齐全（768 张 = 48 片段 × 16 帧），
    把「epoch 中途缺帧崩」提前到「启动即崩」（research-auditor 2026-10-15 S1；
    远端按小时计费，空烧一个 epoch 才崩也是钱）。"""
    bad = []
    for uid in unit_ids:
        d = Path(frames_root) / str(uid)
        n = sum(1 for _ in d.glob("clip*_f*.jpg")) if d.is_dir() else -1
        if n != 768:
            bad.append((str(uid), n))
    if bad:
        raise SystemExit(f"preflight：{len(bad)} 条视频的帧图不齐（应为 768 张）"
                         f"，前 5 条：{bad[:5]}")


def build_net_opt(line, dev, lr_backbone, lr_head):
    """建一套「骨干解冻 + 头」的网络与优化器（每折调用一次；折间不共享权重，
    也不共享优化器状态，research-auditor 2026-10-14 B1）。

    参数分组只此一处：训练主流程与开跑前的卡片核对用同一段代码建优化器，
    核对读到的学习率因此就是训练真正用的那个，不会出现「核对读的是另一份写法」。"""
    net = FTNet(line).to(dev)
    opt = torch.optim.Adam([
        {"params": net.backbone.parameters(), "lr": lr_backbone},
        {"params": net.head.parameters(), "lr": lr_head},
    ])
    return net, opt


# 每张卡片「开跑前数据核对」记录的训练输入指纹：训练输入必须与该卡登记的文件
# 逐字节一致，防止「远端拷了旧版划分也不报错」这种静默的不可比。
#
# 2026-10-25 改成按卡片号区分（research-auditor 决策复核 B3）：原来只写死 EXP-009
# 那两个 150 条池的文件，换池子的卡片（EXP-014 的 128 条池）一开跑就被当场拒绝。
# 新卡片必须在这里登记自己的指纹；未登记直接退出，**不提供跳过比对的开关**——
# 放松比对等于拆掉这道防线。
CARD_MD5 = {
    "EXP-009": {
        "clip_labels": "075b3eb1307c0a620534ec14c048df0b",
        "meta_npz": "b5e2a75c1df7f4e31a5cc331ffb48767",
    },
    # EXP-014 两套设定共用同一份池子元数据，片段标签两套设定各一份（原间隔 / 新间隔），
    # 两份都必须是这里登记过的文件。写成列表而不写成任意值：只放行登记过的指纹，
    # 比对强度与单值版本相同。
    "EXP-014": {
        "clip_labels": ["07ad71057234c990fb9498a5c5e59a0c",     # clip-labels-v2-128.npz
                        "fa2c53c1a0f93d62869b1aeebbc80f4b"],    # clip-labels-v3-128.npz
        "meta_npz": "4b3e8fdb7c01c9cffef59a09513a2446",          # meta-128.npz
    },
    # EXP-013 最终模型：与 EXP-009 用同一份 150 条池、同一份片段标签与同一份冻结版
    # 特征，所以两个指纹与 EXP-009 那一项逐字相同。仍然单独登记一项，不借用
    # "EXP-009" 这个名字：卡片号要能与实际跑的跑法对得上，否则事后看运行日志分不清
    # 这次跑的是哪张卡的设定。
    "EXP-013": {
        "clip_labels": "075b3eb1307c0a620534ec14c048df0b",
        "meta_npz": "b5e2a75c1df7f4e31a5cc331ffb48767",
    },
}

# 卡片（docs/process/experiments/EXP-009.md）「方法设定」节写死的值。
# 核对时拿这些**字面值**与实际对象比，不拿命令行参数比：参数既是建优化器的输入、
# 又是核对的被查对象，拿参数比就是循环论证，永远相等（2026-10-15 本核对表自测时
# 抓到过这个错——把骨干学习率故意写错，核对照样全绿）。
CARD = {
    "lr_backbone": 1e-4,
    "lr_head": 1e-3,
    "epochs": 20,
    "folds": 5,
    "seed": 7,
    "bs_strong": 256,
    "bs_mil_units": 8,
    "lam": 8e-5,
    "caliber": "strict",
    # 卡片「偏差三」登记的两套增强设定：主对照不做任何增强，附加对照做水平翻转。
    # 下面这张「run_id 后缀 -> 该用哪一档增强」的表是**第二份字面值**，与命令行
    # 参数各写各的。核对时拿它比参数，才能查出「run_id 写 noaug、实际跑 hflip」
    # 这类名实不符（2026-10-15 事故的形状）；若拿参数与参数比，永远相等。
    "augment_by_tag_suffix": {"noaug": "none", "hflip": "hflip"},
    "augment_main": "none",
    # 卡片登记的「训练线 × 增强档位」组合（卡片偏差三 + 2026-10-17 附注）。
    # 只查 run_id 后缀查不出「mil 线挂 strong 的 run 名」这类张冠李戴——后缀比对
    # 是拿 tag 的尾巴比增强档位，完全不碰 --line（2026-10-17 审查 W-C③）。
    "registered_line_augment": {
        ("strong", "none"): "主对照（2026-10-17 起唯一开跑的一套）",
        ("strong", "hflip"): "附加对照（2026-10-17 起暂不跑，待主对照成绩后定）",
        # ("mil", "none") 已随「弱标签线不再微调」决定（2026-10-17）取消：
        # 不在登记表里 = 传了就核对不通过、拒绝开跑。
    },
}

# 划分清单正本（协议一，场次级）。每折视频数拿它与训练输入逐折对照，
# 不用「合计等于池内条数」这种任何划分都成立的写法（research-auditor 2026-10-15 S1）。
FOLDS_JSON = REPO / "data/metadata/splits/EXP-008-units-folds.json"

# 按卡片号覆盖上面 CARD 里的字面值，只写确有差别的项，其余照 CARD。
# EXP-013 是最终模型：卡片第三节写死「固定 2 轮、不早停」——全部视频进训练就没有
# 验证集，也就没有早停的依据，轮数只能事先定死。其余设定照抄 EXP-009，只覆盖轮数。
CARD_BY_ID = {
    "EXP-009": {},
    "EXP-013": {"epochs": 2},
    "EXP-014": {},
}


def _mil_loss_reference(p, n, mp, mn):
    """MIL 掩码损失的参照式：照抄 `train_mil.py` 第 174–182 行的公式，但
    **不调用 `masked_max_min`**，改用逐行循环自己取最大值和最小值。

    为什么必须绕开 `masked_max_min`：这一条核对的目的就是「训练路径的归约被改
    坏了要能查出来」。若参照式也调同一个函数，把 2026-10-05 那次反向填充的坏版本
    塞进 `masked_max_min`，两边一起变、差值仍为 0，核对照样通过——这正是那起事故
    的直接防线，却拦不住它（research-auditor 2026-10-15 S2，已用坏版本实测）。
    绕开之后，归约写错必在两边之一露出来。

    循环写法慢，但只在开跑前的核对里跑一次，不在训练路径上。这与「同一份归约
    不得写两遍」不矛盾：那一条管的是训练路径，这里管的是判据。若 `train_mil.py`
    的公式真的变更，本函数须同步改，核对失败正是想要的信号。

    全程按双精度（float64）算：训练路径是 float32，逐元素相加的顺序与这里不同，
    同一公式在 float32 下也会差到 1e-7 量级（实测「部分片段可用」那一组差 6e-8）。
    把参照式升到双精度，差值就只剩训练路径自己的 float32 舍入，而不是两份实现的
    累加顺序之差；核对阈值因此可以取得很紧，公式真被改动时差值是 O(1)，照样拦得住。"""
    p, n = p.double(), n.double()
    rows = []
    for b in range(p.shape[0]):
        pv, nv = p[b][mp[b]], n[b][mn[b]]
        assert pv.numel() and nv.numel(), "每行至少一个可用片段，否则归约无定义"
        l1 = torch.clamp(1.0 - ((pv.max() - pv.min()) - (nv.max() - nv.min())),
                         min=0.0)
        adj = torch.zeros(())
        for j in range(p.shape[1] - 1):
            if bool(mp[b, j]) and bool(mp[b, j + 1]):
                adj = adj + (p[b, j + 1] - p[b, j]) ** 2
        rows.append(l1 + LAM_T * adj + LAM_S * pv.sum())
    return float(sum(rows) / len(rows))


def card_checks(args, unit_ids, unit_labels, clip_label, mask, folds):
    """开跑前的「实验卡片 vs 本脚本」逐条核对，跑在任何 GPU 计算之前。

    每一条查的都是实际对象——优化器真的建出来读它的学习率，批归一化层真的找
    出来看它处于什么状态——不是读代码文本。任何一条不通过即非零退出，不开 GPU。
    （2026-10-15 增强轴事故后立。叶海峰定：要改的是「确保所有环节都不出错」，
    不是补这一个洞。）

    探针网络建在 CPU 上，不占显存、不计 GPU 时间。建网会消耗 torch 的随机数流，
    所以建完把随机状态还原，使「跑没跑核对」不影响训练本身的随机流——否则核对
    自己就成了一个改变结果的自变量。"""
    cc = CardCheck(f"{args.card} 卡片 vs train_ft.py（--line {args.line}）")

    rng_state = torch.get_rng_state()
    net, opt = build_net_opt(args.line, torch.device("cpu"),
                             args.lr_backbone, args.lr_head)
    torch.set_rng_state(rng_state)

    # 骨干与头是否解冻（卡片「方法设定」：全部参数解冻）
    n_bb, tot_bb = count_trainable(net.backbone)
    cc.add("骨干是否解冻", "全部参数解冻",
           f"{n_bb}/{tot_bb} 个参数张量参与训练", n_bb == tot_bb and tot_bb > 0)
    n_hd, tot_hd = count_trainable(net.head)
    cc.add("头是否解冻", "全部参数解冻",
           f"{n_hd}/{tot_hd} 个参数张量参与训练", n_hd == tot_hd and tot_hd > 0)

    # 优化器里实际生效的学习率（读 param_groups，不在源码里搜数字）
    lr_bb = optimizer_lr_of(opt, net.backbone.parameters())
    lr_hd = optimizer_lr_of(opt, net.head.parameters())
    cc.add("骨干学习率", str(CARD["lr_backbone"]),
           f"命令行传入 {args.lr_backbone}，优化器实测 {lr_bb}",
           lr_bb == CARD["lr_backbone"] == args.lr_backbone)
    cc.add("头学习率", str(CARD["lr_head"]),
           f"命令行传入 {args.lr_head}，优化器实测 {lr_hd}",
           lr_hd == CARD["lr_head"] == args.lr_head)

    # 没有参数被漏在优化器外（漏掉的参数永远不更新，且不报任何错）
    covered, missing = optimizer_covers_all(opt, net)
    cc.add("参与训练的参数是否都进了优化器", "全部覆盖",
           "全部覆盖" if covered else f"漏了 {missing}", covered)

    # 批归一化状态（卡片：骨干批归一化固定用运行统计量）
    net.train()                       # 先摆成训练态，再看谁把它压回评估态
    states = bn_training_states(net.backbone)
    n_train_bn = sum(1 for _, t in states if t)
    cc.add("骨干批归一化处于什么状态", "评估态（用运行统计量，不更新）",
           f"{len(states)} 层中 {n_train_bn} 层仍在训练态", n_train_bn == 0)

    # 增强翻的是哪一维（卡片：随机水平翻转）
    try:
        assert_augment_axis()
        aug_ok, aug_why = True, "翻宽度维，颜色通道不动"
    except AssertionError as exc:
        aug_ok, aug_why = False, str(exc)
    cc.add("增强翻的是哪一维", "画面左右翻转", aug_why, aug_ok)

    # 增强开关的实际行为（卡片：无增强那一套要与冻结版输入分布逐位相同）
    probe = torch.arange(2 * 48 * 16 * 4 * 5 * 3, dtype=torch.uint8).reshape(
        2, 48, 16, 4, 5, 3)

    gen = np.random.RandomState(args.seed)
    out_none = apply_augment(probe.clone(), (2, 48), gen, "none")
    cc.add("增强设为 none 时是否真的不翻转", "输入逐位不变",
           "逐位不变" if torch.equal(out_none, probe) else "被改动了",
           torch.equal(out_none, probe))
    # 不消耗随机数的判据：同一个种子起步，走过 none 之后抽出的数与全新种子抽出的
    # 数相同，说明中间没抽过。若照样掷硬币再丢弃，两套成绩会多出一个不受控变量。
    got = gen.random(4)
    ref = np.random.RandomState(args.seed).random(4)
    cc.add("增强设为 none 时是否消耗随机数", "不消耗",
           "不消耗" if np.array_equal(got, ref) else "消耗了",
           bool(np.array_equal(got, ref)))

    gen2 = np.random.RandomState(args.seed)
    out_hflip = apply_augment(probe.clone(), (2, 48), gen2, "hflip")
    hit = np.random.RandomState(args.seed).random((2, 48)) < 0.5
    expected = probe.clone()
    hit_t = torch.from_numpy(hit)
    expected[hit_t] = horizontal_flip(probe[hit_t])
    cc.add("增强设为 hflip 时是否真的翻转", "按掷硬币结果翻转宽度维",
           f"命中 {int(hit.sum())}/{hit.size} 个片段",
           torch.equal(out_hflip, expected) and bool(hit.any()))
    # 增强设定：从 run_id 后缀反查卡片登记的档位，再与命令行参数比。
    # 后缀表是另一份字面值，所以这一条能查出「run_id 说 noaug、实际跑 hflip」这类
    # 名实不符。原先这一条写的是「两档文字不同」，那是自指检查——任何两档不同的
    # 文字都通过，与卡片无关（research-auditor 2026-10-15 W1/S3）。
    suffix = next((s for s in CARD["augment_by_tag_suffix"]
                   if args.tag.endswith(s)), None)
    if suffix is None:
        cc.add("增强设定与卡片登记一致",
               "run_id 后缀 noaug→无增强、hflip→水平翻转（卡片偏差三）",
               f"tag={args.tag} 不带卡片登记的后缀"
               + ("（冒烟跑，按设计不比对）" if args.smoke
                  else "，全量跑必须按卡片登记的名字命名"),
               bool(args.smoke))
    else:
        want = CARD["augment_by_tag_suffix"][suffix]
        cc.add("增强设定与卡片登记一致",
               f"run_id 后缀 {suffix} → 增强 {want}（卡片偏差三）",
               f"tag={args.tag} → 命令行增强 {args.augment}",
               args.augment == want)

    # 训练线 × 增强档位的组合必须整体在卡片登记过：--line 与 --tag 是两个独立的
    # 命令行参数，只查后缀拦不住「--line mil 挂 strong 主对照的 run 名」（2026-10-17
    # 审查 W-C③；mil 组合已随弱标签线停微调决定取消，传了就拒跑）。
    combo_reg = CARD["registered_line_augment"].get((args.line, args.augment))
    cc.add("训练线与增强档位的组合已登记",
           "strong×none=主对照；strong×hflip=附加对照；mil 组合已于 2026-10-17 取消",
           f"{args.line}×{args.augment}"
           + (f"（{combo_reg}）" if combo_reg else "（未在卡片登记，拒绝开跑）"),
           combo_reg is not None)

    # 评估口径：卡片只登记 strict。loose 的掩码更宽，任何「无全掩码视频」类断言
    # 在它下面只会更容易通过，于是用 loose 跑出来的成绩核对表不会提示它与卡片
    # 不符（research-auditor 2026-10-15 W2）。
    cc.add("评估口径与卡片一致", CARD["caliber"],
           f"命令行 {args.caliber}"
           + ("" if args.caliber == CARD["caliber"]
              else "（卡片只登记了 strict，loose 未登记）"),
           args.caliber == CARD["caliber"])

    # 轮数、折数、每折视频数（卡片：协议一 5 折、20 轮；EXP-013 固定 2 轮）
    # 冒烟模式按设计只跑 1 轮（冒烟本来就是「先跑通一次」的短跑，不是卡片的实验），
    # 所以这一条在冒烟下如实记为「按设计偏离」；全量下必须等于卡片值。两种情形都
    # 照常打印实际轮数，读者看得见差别，不会把冒烟当成一次真跑。
    card_ep = {**CARD, **CARD_BY_ID.get(args.card, {})}["epochs"]
    ep_ok = args.epochs == card_ep or (args.smoke and args.epochs == 1)
    cc.add("训练轮数", str(card_ep),
           f"{args.epochs}" + ("（冒烟模式，按设计只跑 1 轮）" if args.smoke else ""),
           ep_ok)
    # 训练方式：EXP-013 是全量重训（卡片第二节「没有留出」），其余卡片都是五折
    # 交叉验证。两种方式的训练集大小不同，混着跑出来的权重没有可比性，所以在
    # 开跑前就对上，而不是等跑完看产物名字才发现。
    want_all = args.card == "EXP-013"
    cc.add("训练方式",
           "全部视频进训练、不留验证折（--all-data）" if want_all
           else "五折交叉验证（--fold -1，每折留出 1/5）",
           "全部视频训练" if args.all_data else "按折留出训练",
           bool(args.all_data) == want_all)
    fold_nums = sorted(set(int(x) for x in folds))
    cc.add("折数", str(CARD["folds"]), str(len(fold_nums)),
           len(fold_nums) == CARD["folds"])
    # 每折视频数：与登记在 data/metadata/splits/ 的划分清单逐折对照。
    # 不能只断言「合计等于池内条数」——任何把 150 条分进 5 个折的划分都满足它，
    # 是恒真式，起不到对照作用（research-auditor 2026-10-15 S1，已用
    # folds=[0]*150 这个反例证实）。
    registered = {int(d["fold"]): int(d["val_units"])
                  for d in json.loads(Path(args.folds_json).read_text(encoding="utf-8"))["folds"]}
    got_per_fold = {f: int((folds == f).sum()) for f in fold_nums}
    # 只比每折条数、不比逐折名单：名单级的一致性由 meta npz 的 md5 指纹核对覆盖
    # （折划分来自 npz 的 folds 元字段，npz 变一个字节指纹就对不上、开跑即拒），
    # 所以这里查条数就是防「清单与输入错位」的冗余第二道，不承担名单核对
    # （2026-10-17 审查 W-C④，注明即可）。
    cc.add("每折视频数（对照划分清单）",
           "逐折等于 " + str([registered[f] for f in sorted(registered)]),
           f"实得 {[got_per_fold[f] for f in sorted(got_per_fold)]}"
           f"，合计 {sum(got_per_fold.values())}",
           got_per_fold == registered)

    # 停跑判据必须能在这条线真实崩过的那一折上触发，且不误伤来回跳但没崩的那一折。
    # 两条轨迹都是 2026-10-15 那次跑出来的真实逐轮验证 AUC，写死在这里当参照：
    # 判据改方向、改阈值、改成看峰值，这两条里必有一条翻车。
    # 两条轨迹都必须写满真实轮数（20 轮）。写短了会虚增「余量」：若只写到第 14 轮，
    # 末尾连续低分恰好 5 轮，刚好等于阈值、余量为 0，阈值一改成 6 这条检查立刻翻车，
    # 而它本该有 11 轮的余量（research-auditor 2026-10-15 W9）。
    known_bad = [0.7381, 0.1905, 0.2560, 0.4881, 0.6607, 0.6964, 0.6905, 0.6667,
                 0.6250, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                 0.5000, 0.5000, 0.5000, 0.5000]      # f2：真崩，末尾 11 轮贴瞎猜线
    known_ok = [0.8052, 0.7662, 0.0844, 0.5844, 0.1558, 0.0000, 0.1429, 0.5584,
                0.9870, 0.8506, 0.8864, 0.9740, 0.8831, 1.0000, 0.2662, 1.0000,
                1.0000, 0.8929, 0.1753, 0.4091]      # f1：来回跳但没崩，最长连续 3 轮
    bad_streak = trailing_low_auc_streak([{"va_auc": a} for a in known_bad])
    ok_streak = trailing_low_auc_streak([{"va_auc": a} for a in known_ok])
    cc.add("停跑判据能否在真崩过的那一折触发",
           f"末尾连续 ≥ {DEGENERATE_STREAK} 轮验证 AUC ≤ {DEGENERATE_AUC}",
           f"实得末尾连续 {bad_streak} 轮", bad_streak >= DEGENERATE_STREAK)
    cc.add("停跑判据会不会误伤没崩但来回跳的那一折",
           f"末尾连续 < {DEGENERATE_STREAK} 轮验证 AUC ≤ {DEGENERATE_AUC}",
           f"实得末尾连续 {ok_streak} 轮", ok_streak < DEGENERATE_STREAK)

    # strict 片段集（卡片：strict 掩码，无全掩码视频）
    n_all_masked = int((~mask.any(axis=1)).sum())
    cc.add("strict 片段集", "每条视频至少一个可用片段",
           f"可用 {int(mask.sum())}/{mask.size}，全掩码视频 {n_all_masked} 条",
           n_all_masked == 0)

    # 随机种子（卡片：seed=7）
    cc.add("随机种子", str(CARD["seed"]),
           f"命令行传入 {args.seed}，torch 初始种子 {torch.initial_seed()}",
           torch.initial_seed() == CARD["seed"] == args.seed)

    # 训练输入指纹（卡片「开跑前数据核对」第 1 条；EXP-009 与 EXP-014 各登记一份。
    # 期望值可以是单个 md5，也可以是「允许的若干个 md5」的列表——EXP-014 两套设定的
    # 片段标签各一份文件，但每一份都必须登记过）
    expect = CARD_MD5[args.card]
    for key, label, path in (("clip_labels", "片段标签文件", args.clip_labels),
                             ("meta_npz", "冻结版特征文件", args.meta)):
        want = expect[key]
        want = [want] if isinstance(want, str) else list(want)
        got = file_md5(path)
        cc.add(f"{label}指纹", " 或 ".join(w[:12] for w in want), got[:12], got in want)

    # 一条视频的分数怎么算（卡片：取可用片段的最大值）
    probe_scores = torch.tensor([0.1, 0.9, 0.3, 0.2])
    probe_avail = torch.tensor([True, False, True, True])
    got_score = float(aggregate_unit_score(probe_scores, probe_avail))
    # 容差 1e-6：0.30 存成 float32 是 0.30000001192，不能用严格相等
    cc.add("一条视频的分数怎么算", "取可用片段的最大值（不可用片段不参与）",
           f"探针实测 {got_score:.6f}（最大值 0.9 落在不可用片段上，应取 0.3）",
           abs(got_score - 0.3) < 1e-6)

    # 损失公式与常量（卡片：MIL = L1 + λt·L2 + λs·L3，λt=λs=8e-5，margin 1）
    if args.line == "mil":
        # 用局部随机数发生器，不碰全局随机流：调 torch.manual_seed 会改掉训练
        # 自己的种子，核对本身就成了改变结果的自变量（2026-10-15 本核对表
        # 第一次自测时抓到过这个错）。
        gen = torch.Generator().manual_seed(0)
        # 三种掩码形态各查一遍：2026-10-05 那起事故就出在掩码填充方向，而它在
        # 「全部片段可用」这一种形态下看不出来（没有位置被填充），必须把「部分可用」
        # 与「每条只剩一个可用」也喂进去（research-auditor 2026-10-15 S2）。
        all_on = torch.ones(3, 48, dtype=torch.bool)
        partial = torch.rand(3, 48, generator=gen) > 0.3
        partial[:, 0] = True
        single = torch.zeros(3, 48, dtype=torch.bool)
        single[:, 0] = True
        for case, (mp, mn) in (("全部片段可用", (all_on, all_on)),
                               ("部分片段可用", (partial, partial.clone())),
                               ("每条只剩一个可用片段", (single, single.clone()))):
            p = torch.rand(3, 48, generator=gen)
            n_ = torch.rand(3, 48, generator=gen)
            got_l = float(mil_loss_masked(p, n_, mp, mn)[0])
            ref_l = _mil_loss_reference(p, n_, mp, mn)
            # 阈值 1e-5：参照式在双精度下算，差值只剩训练路径的 float32 舍入
            # （实测 6e-8 量级）；公式真被改动时差值是 O(1)，隔了五个数量级。
            cc.add(f"MIL 损失公式（{case}）",
                   "L1 + 8e-5·L2 + 8e-5·L3（照抄 train_mil.py）",
                   f"实测 {got_l:.8f}，参照式 {ref_l:.8f}", abs(got_l - ref_l) < 1e-5)
        cc.add("MIL 每批正视频条数", str(CARD["bs_mil_units"]), str(BS_MIL_UNITS),
               BS_MIL_UNITS == CARD["bs_mil_units"])
    else:
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))
        got_b = float(crit(torch.zeros(2), torch.tensor([1.0, 0.0])))
        ref_b = 1.5 * float(np.log(2.0))      # 正样本项加权 2 倍后取平均
        cc.add("强标签损失形态", "BCEWithLogitsLoss + pos_weight（正样本项加权）",
               f"探针实测 {got_b:.6f}，参照式 {ref_b:.6f}",
               abs(got_b - ref_b) < 1e-6)
        cc.add("强标签片段批大小", str(CARD["bs_strong"]), str(BS_STRONG),
               BS_STRONG == CARD["bs_strong"])

    cc.add("损失权重常数", f"λt = λs = {CARD['lam']}",
           f"λt={LAM_T}, λs={LAM_S}",
           LAM_T == CARD["lam"] and LAM_S == CARD["lam"])

    # 归一化用的均值与标准差（卡片：与冻结版输入分布一致）
    cc.add("归一化均值与标准差", "Kinetics 值（与冻结版同一份）",
           f"均值 {MEAN.tolist()}，标准差 {STD.tolist()}",
           np.allclose(MEAN, [0.43216, 0.394666, 0.37645])
           and np.allclose(STD, [0.22803, 0.22145, 0.216989]))

    torch.set_rng_state(rng_state)      # 核对全程不得改变训练的随机流
    cc.report()


def main():
    global FRAMES_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--line", choices=["mil", "strong"], required=True)
    ap.add_argument("--clip-labels", dest="clip_labels", default=str(CLIP_LABELS))
    ap.add_argument("--meta", default=str(REPO / "data/results/exp008/unitfeats-v2-150/unit-features.npz"),
                    help="冻结版特征 npz（只读 unit_ids/labels/folds 元字段，不读 feats）")
    ap.add_argument("--frames-root", dest="frames_root", default=str(FRAMES_ROOT))
    ap.add_argument("--folds-json", dest="folds_json", default=str(FOLDS_JSON),
                    help="逐折条数登记清单，供开跑前核对表对照。默认是 150 条池的 "
                         "EXP-008-units-folds.json；换池子（如 EXP-014 的 128 条池）时"
                         "指向该池子的登记清单，历史跑法不给这个参数即不受影响")
    ap.add_argument("--caliber", choices=["strict", "loose"], default="strict")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr-head", dest="lr_head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", dest="lr_backbone", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--fold", type=int, default=-1, help="只跑第 fold 折（-1=全折）")
    ap.add_argument("--all-data", dest="all_data", action="store_true",
                    help="全部视频都拿来训练、不留验证折（EXP-013 最终模型）。"
                         "没有验证集就没有早停，轮数由 --epochs 事先定死；"
                         "权重存 final_<线>.pt，逐视频分数存 predictions-all.csv"
                         "（训练集内，只作一致性检查，不当成绩）")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：不划分折，选 1 条正、1 条负视频训练并自评（W4）")
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    # 必填，不给默认值：漏写时必须当场报错退出，不能静默跑成另一档增强。
    # 核对表「增强设定与卡片登记一致」只在 run_id 带卡片登记的后缀时比对，冒烟跑
    # 按设计不比对，所以冒烟路径曾经会静默放行漏写的 --augment（2026-10-15 决策
    # 复核 W3）。必填把这条通道一并关掉，且不改变任何一次正式跑的实际配置——
    # 卡片与运行日志里的每条启动命令都显式传了这个参数。
    ap.add_argument("--augment", choices=["none", "hflip"], required=True,
                    help="必填。none=不翻转（与冻结版输入分布逐位相同）；"
                         "hflip=片段级随机水平翻转，一个片段 16 帧同翻")
    ap.add_argument("--tag", default="ft-smoke")
    ap.add_argument("--card", default="EXP-009",
                    help="这份跑法按哪张实验卡核对（决定用哪一组训练输入指纹）。"
                         "默认 EXP-009；换池子的卡（如 EXP-014 的 128 条池）必须显式"
                         "传入，且该卡的指纹要先在 CARD_MD5 里登记")
    args = ap.parse_args()
    for name in ("frames_root", "clip_labels", "meta", "folds_json"):
        value = getattr(args, name)
        if value:
            setattr(args, name, str(resolve_path(value, must_exist=True)))

    if args.card not in CARD_MD5:
        raise SystemExit(
            f"卡片 {args.card} 的训练输入指纹尚未在 train_ft.py 顶部的 CARD_MD5 里登记"
            f"（已登记：{sorted(CARD_MD5)}）。等该卡的数据产物定稿后，把 clip-labels 与"
            f" meta 两个 md5 补进去再开跑；这里不提供跳过比对的开关。")

    # 三种跑法互斥，组合错了当场退出：--all-data 与 --fold 同时给，读者无法判断
    # 到底是想只跑一折还是全量训练；与 --smoke 同时给，冒烟那条路径根本不看折，
    # 静默忽略等于给了个不生效的开关。全量训练没有验证折，弱标签线（mil）的
    # MIL 损失本来就靠折内正负样本组织，本卡也不涉及，一并拒绝。
    if args.all_data and args.smoke:
        raise SystemExit("--all-data 与 --smoke 不能同时用：冒烟固定取 1 正 1 负两条视频")
    if args.all_data and args.fold >= 0:
        raise SystemExit("--all-data 与 --fold 不能同时用：全量训练不分折")
    if args.all_data and args.line != "strong":
        raise SystemExit("--all-data 只支持 --line strong（EXP-013 卡片用的是强标签线）")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    FRAMES_ROOT = Path(args.frames_root)

    cl = np.load(args.clip_labels, allow_pickle=True)
    # 标签与折划分取冻结版特征 npz 的元字段（labels/folds），保证与 EXP-008 冻结线
    # 同一划分、同一行序；特征 feats 字段不读取（微调用不到特征缓存）。
    meta = np.load(args.meta, allow_pickle=True)
    assert list(meta["unit_ids"]) == list(cl["unit_ids"]), \
        "clip-labels 与特征 npz 行序不一致"
    unit_ids = meta["unit_ids"]
    unit_labels = meta["labels"].astype(int)
    clip_label = cl[args.caliber].astype(int)              # [n,48] -1/0/1
    mask = clip_label >= 0
    assert torch.tensor(mask).any(dim=1).all(), \
        "存在 48 片段全掩码的视频——评估侧取最大将遇空张量（W5，沿用 train_mil 断言）"
    folds = meta["folds"].astype(int)
    n = len(unit_ids)
    preflight_frames(unit_ids, FRAMES_ROOT)               # S1：启动即核对帧图齐全
    # 卡片逐条核对（含增强轴自检）：全部跑在任何 GPU 计算之前，任一条不通过即退出
    card_checks(args, unit_ids, unit_labels, clip_label, mask, folds)
    print(f"[FT-{args.line}] 口径 {args.caliber}：{n} 条视频，可用片段 "
          f"{int(mask.sum())}/{mask.size}（{100*mask.mean():.1f}%）；"
          f"骨干解冻，lr 骨干 {args.lr_backbone} / 头 {args.lr_head}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kw = {"num_workers": args.workers, "pin_memory": dev.type == "cuda"}
    run = run_strong if args.line == "strong" else run_mil

    pred_rows = []                                         # (fold, unit_id, label, score)
    runs, bests = [], {}
    out_dir = RUNS_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = set()

    def write_outputs():
        """每折结束即落盘（research-auditor 2026-10-15 W-1）：12 小时级长跑若中
        途崩（瞬时 CUDA 错误、磁盘满、断连），已完成折的 metrics/权重/分数不丢。
        权重只写一次（saved 去重），metrics/predictions 每次全量重写。"""
        aucs = [r["auc"] for r in runs if r["auc"] == r["auc"]]
        summary = {"script": "train_ft.py", "line": args.line,
                   "caliber": args.caliber,
                   "n_unit": n, "epochs": args.epochs,
                   "lr_head": args.lr_head, "lr_backbone": args.lr_backbone,
                   "seed": args.seed,
                   "backbone": "r2plus1d_18 KINETICS400_V1 全解冻（每折独立初始化）",
                   "bn": "骨干 BN 用运行统计量（eval），仿射参数照常训练"
                         "（research-auditor 2026-10-14 决策 1）",
                   "augment": augment_label(args.augment),
                   "augment_axis": "宽度维（张量倒数第二维，flip(-2)）；"
                                   "颜色通道维不动",
                   "augment_note": (
                       "微调设定含翻转增强而冻结版无任何增强，微调成绩 = 解冻 + 翻转"
                       "的合并效应（research-auditor 2026-10-14 W6 注记口径）"
                       if args.augment == "hflip" else
                       "微调不设任何增强，输入分布与冻结版逐位相同，微调成绩 = "
                       "解冻的单独效应"),
                   "precision": f"FP32 + 梯度检查点（CHUNK={CHUNK}，分段输入 requires_grad）",
                   "folds_source": "冻结版特征 npz folds 元字段（协议一，与 EXP-008 冻结线同划分）",
                   "clip_labels": str(Path(args.clip_labels).resolve()),
                   "meta_npz": str(Path(args.meta).resolve()),       # S3：元文件对账
                   # 帧图根路径（research-auditor 2026-10-15 W5）：产物从远端拉回本地后，
                   # 光看 metrics.json 分不出它读的是哪一份帧图，本地与远端目录结构不同。
                   "frames_root": str(Path(args.frames_root).resolve()),
                   # 逐折条数核对用的登记清单（EXP-014 换池子后必须指向该池子的清单）
                   "folds_json": str(Path(args.folds_json).resolve()),
                   "md5": {"clip_labels": file_md5(args.clip_labels),
                           "meta_npz": file_md5(args.meta),
                           # 脚本自身的指纹（research-auditor 2026-10-27 W1）：
                           # 不记它，「跑的就是审过的那一份」事后无法证实。
                           "script": file_md5(__file__)},   # S3：与冻结版逐字节可比
                   "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
                   "smoke": bool(args.smoke),
                   # 全量训练（EXP-013）：全部视频进训练、不留验证折。写进产物里，
                   # 免得事后拿一份 metrics.json 分不清它是五折里的哪一折还是最终模型。
                   "all_data": bool(args.all_data),
                   "mean_auc": round(float(np.mean(aucs)), 4) if aucs else float("nan"),
                   "std_auc": round(float(np.std(aucs)), 4) if aucs else float("nan"),
                   "folds": runs}
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        for f, state in bests.items():
            if state is not None and f not in saved:
                # 全量训练只有一个模型，文件名照 EXP-013 卡片第七节写 final_<线>.pt；
                # 五折的五个模型仍按折号命名。
                name = (f"final_{args.line}.pt" if f == "final"
                        else f"ft_{args.line}_f{f}.pt")
                torch.save(state, out_dir / name)
                saved.add(f)
        pred_name = "predictions-all.csv" if args.all_data else "predictions.csv"
        with open(out_dir / pred_name, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["fold", "unit_id", "label", "score"])
            for row in pred_rows:
                w.writerow(row)
        if args.all_data:
            print(f"[FT-{args.line}] 已落盘（全部视频训练，不留出折，没有任何成绩数字）"
                  f" -> {out_dir / 'metrics.json'}", flush=True)
        else:
            print(f"[FT-{args.line}] 已落盘（fold {len(runs)} 个）均值 AUC "
                  f"{summary['mean_auc']} ± {summary['std_auc']} "
                  f"-> {out_dir / 'metrics.json'}", flush=True)

    if args.smoke:
        # 冒烟选 1 正 1 负（W4：--units 2 取前两条可能同类，触发断言崩溃）
        pos_i = [i for i in range(n) if unit_labels[i] == 1][0]
        neg_i = [i for i in range(n) if unit_labels[i] == 0][0]
        sel = [pos_i, neg_i]
        rng = np.random.RandomState(args.seed)             # 每折重置随机流（W7）
        net, opt = build_net_opt(args.line, dev, args.lr_backbone, args.lr_head)
        n_bb = sum(p.numel() for p in net.backbone.parameters())
        n_hd = sum(p.numel() for p in net.head.parameters())
        print(f"[FT-{args.line}] 骨干 {n_bb/1e6:.3f}M + 头 {n_hd/1e3:.1f}K 参数，"
              f"设备 {dev}（{torch.cuda.get_device_name(0) if dev.type=='cuda' else 'cpu'}），"
              f"BN=运行统计量（eval），增强={augment_label(args.augment)}", flush=True)
        state, best_auc, best_r, best_ep, hist = run(
            net, opt, dev, unit_ids, unit_labels, clip_label, mask,
            sel, sel, args.epochs, "smoke", rng, loader_kw, args.augment)
        if state is not None:
            bests["smoke"] = state
        runs.append({"fold": "smoke", "auc": best_auc, "best_epoch": best_ep,
                     "f1": best_r["f1"], "precision": best_r["precision"],
                     "recall": best_r["recall"],
                     # 实际跑了几轮（W6）：顶层的 epochs 写的是命令行传的值，早停的折
                     # 只跑了 5~13 轮；不看这个字段就得去数 hist 的长度才知道跑没跑满。
                     "epochs_run": len(hist),
                     # 冒烟折不带 n/n_pos/n_neg（验证折视频条数）：冒烟不是一折，
                     # 是 1 正 1 负两条视频的跑通测试，按设计不参与折级对账
                     # （2026-10-17 审查 W-C②，注明即可）。
                     "hist": hist,
                     # 冒烟折同样带 degenerate 字段：卡片写「每折带 degenerate」，
                     # 少一个字段的话读 JSON 的人分不清「这折没退化」和「这折没记」。
                     # 判定与正式折同一套（看末尾连续低分轮数），冒烟只跑 1 轮，
                     # 达不到连续 5 轮，所以必然是 False——这是设计如此，不是漏判。
                     "degenerate": bool(trailing_low_auc_streak(hist)
                                        >= DEGENERATE_STREAK),
                     DEGENERATE_STREAK_LABEL: trailing_low_auc_streak(hist)})
        if state is not None:
            net.load_state_dict(state)
            r = eval_units(net, dev, UnitDataset(unit_ids, FRAMES_ROOT),
                           unit_labels, torch.tensor(mask, device=dev), sel, loader_kw,
                           sigmoid=(args.line == "strong"))
            for i, s in zip(sel, r["scores"]):
                pred_rows.append(["smoke", str(unit_ids[i]),
                                  int(unit_labels[i]), float(s)])
        write_outputs()
    else:
        fold_nums = sorted(set(int(x) for x in folds))
        if args.all_data:
            fold_list = ["final"]              # 全量训练只有一个「折」，名字就叫 final
        else:
            fold_list = [args.fold] if args.fold >= 0 else fold_nums
            if args.fold >= 0 and args.fold not in fold_nums:
                raise SystemExit(f"--fold {args.fold} 不在折清单 {fold_nums} 中")
        n_bb = n_hd = 0
        deg_streak = 0                                    # 连续退化折计数（见下）
        for f in fold_list:
            if args.all_data:
                tr, va = list(range(n)), []
            else:
                va = [i for i in range(n) if int(folds[i]) == f]
                tr = [i for i in range(n) if int(folds[i]) != f]
            # 验证折单一类别的检查只管五折：全量训练没有验证折，这条不适用。
            if not args.all_data and len(set(unit_labels[va].tolist())) < 2:
                print(f"  [f{f}] 验证折单一类别（正 {int(unit_labels[va].sum())}/"
                      f"负 {len(va)-int(unit_labels[va].sum())}），跳过（W3，"
                      f"沿用 train_mil 惯例）", flush=True)
                runs.append({"fold": f, "name": f"f{f}", "n": len(va),
                             "n_pos": int(unit_labels[va].sum()),
                             "n_neg": int(len(va) - int(unit_labels[va].sum())),
                             "auc": float("nan"), "best_epoch": -1,
                             "epochs_run": 0,
                             # 卡片停跑规则写「每折带 degenerate 与连续低分轮数」，
                             # 被跳过的折也不例外：没有这两个字段，读 JSON 的人
                             # 分不清「这折没退化」和「这折没记」（2026-10-17
                             # 审查 W-C①）。跳过的折没跑任何轮，末尾连续低分
                             # 轮数按 0 记、degenerate 按假记——「没跑」不等于
                             # 「退化」，与冒烟折的记法同一套。
                             "degenerate": False,
                             DEGENERATE_STREAK_LABEL: 0,
                             "note": "验证折单一类别，跳过"})
                continue
            rng = np.random.RandomState(args.seed)         # 每折重置随机流（W7）
            # torch 也要每折重种：不重种的话，第 3 折的随机流取决于前两折各消耗了
            # 多少随机数，于是「单独跑 --fold 3」与「五折一起跑时的第 3 折」初始化
            # 不同、结果对不上，单折重跑就没法用来复现。冻结线（train_mil.py:133、
            # train_strong.py:196）本来就是每折重种，这里对齐（research-auditor W4）。
            torch.manual_seed(args.seed)
            net, opt = build_net_opt(args.line, dev, args.lr_backbone, args.lr_head)
            if not n_bb:
                n_bb = sum(p.numel() for p in net.backbone.parameters())
                n_hd = sum(p.numel() for p in net.head.parameters())
                print(f"[FT-{args.line}] 骨干 {n_bb/1e6:.3f}M + 头 {n_hd/1e3:.1f}K 参数，"
                      f"设备 {dev}（{torch.cuda.get_device_name(0) if dev.type=='cuda' else 'cpu'}），"
                      f"BN=运行统计量（eval），增强={augment_label(args.augment)}", flush=True)
            state, best_auc, best_r, best_ep, hist = run(
                net, opt, dev, unit_ids, unit_labels, clip_label, mask,
                tr, va, args.epochs, "final" if args.all_data else f"f{f}",
                rng, loader_kw, args.augment,
                stop_low_streak=0 if args.all_data else DEGENERATE_STREAK)
            bests[f] = state
            if args.all_data:
                # 全部视频进训练：没有验证折，也就没有折外成绩可报。记下实际跑了
                # 几轮、正负条数，仅供参考，auc/f1 等成绩字段一律 nan。
                runs.append({"fold": f, "name": "final", "n": n,
                             "n_pos": int(unit_labels.sum()),
                             "n_neg": int(n - unit_labels.sum()),
                             "auc": float("nan"), "best_epoch": best_ep,
                             "epochs_run": len(hist),
                             "degenerate": False,
                             DEGENERATE_STREAK_LABEL: 0,
                             "f1": float("nan"), "precision": float("nan"),
                             "recall": float("nan"),
                             "hist": hist,
                             "note": "全部视频训练，无留出折；本模型不产生任何成绩"})
            else:
                # 退化标记：看末尾连续多少轮贴在瞎猜线上，不看折内最好的一轮。
                # 卡片第 40 行的失败中止条款原先只写「AUC 崩塌即停」，没有可执行的判定，
                # 2026-10-15 弱标签线 f2 就已退化却跑到 f3 第 14 轮才停；后来补的判定又
                # 错用了「最好的一轮」，在已知失败上永远不触发（research-auditor B1）。
                low_streak = trailing_low_auc_streak(hist)
                degenerate = bool(low_streak >= DEGENERATE_STREAK)
                runs.append({"fold": f, "name": f"f{f}", "n": len(va),
                             "n_pos": int(unit_labels[va].sum()),
                             "n_neg": int(len(va) - int(unit_labels[va].sum())),
                             "auc": best_auc, "best_epoch": best_ep,
                             "epochs_run": len(hist),  # W6：跑满 20 轮还是早停在第几轮
                             "degenerate": degenerate,
                             DEGENERATE_STREAK_LABEL: low_streak,
                             "f1": best_r["f1"], "precision": best_r["precision"],
                             "recall": best_r["recall"],
                             "hist": hist})            # 逐 epoch 损失与 va_auc（N1）
            # 最终权重复算逐视频分数 -> predictions.csv（五折）/ predictions-all.csv
            # （全量训练，W2：配对重抽样输入）。全量训练给 150 条全部打分，只是
            # 训练集内的一致性检查材料，不是成绩。
            net.load_state_dict(state)
            score_idx = list(range(n)) if args.all_data else va
            r = eval_units(net, dev, UnitDataset(unit_ids, FRAMES_ROOT),
                           unit_labels, torch.tensor(mask, device=dev), score_idx,
                           loader_kw, sigmoid=(args.line == "strong"))
            for i, s in zip(score_idx, r["scores"]):
                pred_rows.append([f, str(unit_ids[i]), int(unit_labels[i]), float(s)])
            write_outputs()                                # W-1：每折落盘
            if args.all_data:
                continue                    # 全量训练只有一轮循环，没有折可判退化
            # 停跑规则（预登记在卡片「成功/失败」节，开跑前写死，不看结果再定）：
            # 连续两折被判退化即停并记录，不再跑后续折。已跑完的折照常落盘，已花的
            # 钱不白费；退化事实如实写进 metrics.json 与 run-log。
            deg_streak = deg_streak + 1 if degenerate else 0
            if deg_streak >= 2:
                print(f"  [停跑] {args.line} 线连续 {deg_streak} 折退化（{f} 及前一折，"
                      f"末尾 {DEGENERATE_STREAK} 轮验证 AUC 均 ≤ {DEGENERATE_AUC}），"
                      f"按卡片预登记规则停止，不再跑后续折；已完成折的数字照常落盘",
                      flush=True)
                break

    write_outputs()                                        # 终态再写一次（含全部折）


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
