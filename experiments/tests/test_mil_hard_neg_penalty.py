# -*- coding: utf-8 -*-
"""EXP-011 负样本视频超阈惩罚回归测试（实验卡「代码改动 3」五种情形）。

被测对象：experiments/common/video/train_mil.py 的 hard_neg_penalty()，以及
λ=0 时训练损失路径与旧版的逐位同构性。

机制回顾：L_pen = λ · mean( max(0, 负单位最高片段分 − 0.5) )，最高片段分取
strict 掩码下的 masked max（~mask 填 -inf）；均值只对违规行（最高分 > 0.5）
取，不对全批平均；批内无违规时该项恰为 0 且不消耗随机数。

运行（仓库根）：
  .venv/Scripts/python.exe -X utf8 -m pytest experiments/tests/test_mil_hard_neg_penalty.py -q
  .venv/Scripts/python.exe -X utf8 experiments/tests/test_mil_hard_neg_penalty.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments.common.video.train_mil as tm


def _loss_like_run_fold(p, n, mp, mn):
    """复刻 run_fold 掩码分支的三项损失（不含惩罚项），供同构断言用。"""
    maxp, minp = tm.masked_max_min(p, mp)
    maxn, minn = tm.masked_max_min(n, mn)
    l1 = torch.max(torch.zeros_like(maxp),
                   1.0 - ((maxp - minp) - (maxn - minn))).mean()
    pv = mp[:, 1:] & mp[:, :-1]
    l2 = (((p[:, 1:] - p[:, :-1]) ** 2) * pv).sum(dim=1).mean()
    l3 = (p * mp).sum(dim=1).mean()
    return l1 + tm.LAM_T * l2 + tm.LAM_S * l3


def test_case1_lam0_bitwise_identical():
    """情形①：λ=0（惩罚未启用）时损失路径与原路径逐位同构。

    run_fold 在 λ=0 时结构性跳过惩罚；本用例再锁一层——即便把 λ=0 的惩罚值
    加进损失，数值也逐位不变（防止将来有人把跳过改成「算了再加」后引入偏差）。
    """
    torch.manual_seed(7)
    p = torch.rand(4, 48)
    n = torch.rand(4, 48)
    mp = torch.ones(4, 48, dtype=torch.bool)
    mp[1, :10] = False
    mn = torch.ones(4, 48, dtype=torch.bool)
    mn[2, 40:] = False
    loss_old = _loss_like_run_fold(p, n, mp, mn)
    pen, n_viol = tm.hard_neg_penalty(n, mn, tau=tm.PEN_TAU, lam=0.0)
    loss_new = loss_old + pen
    assert loss_new.item() == loss_old.item(), \
        f"λ=0 惩罚改变损失值：旧 {loss_old.item()} 新 {loss_new.item()}"
    # λ=0 时即使有违规行，惩罚值也必须为 0
    assert pen.item() == 0.0, f"λ=0 惩罚值应为 0，实得 {pen.item()}"
    assert n_viol == int((n.max(dim=1).values > 0.5).sum())


def test_case2_all_below_threshold():
    """情形②：批内负视频全部阈下 → penalty==0 且对分数的梯度为 0。"""
    n = torch.full((3, 48), 0.3)
    n[0, 5] = 0.49
    mask = torch.ones(3, 48, dtype=torch.bool)
    s = n.detach().clone().requires_grad_(True)
    base = _loss_like_run_fold(s, s, mask, mask)
    pen, n_viol = tm.hard_neg_penalty(s, mask, tau=0.5, lam=1.0)
    assert pen.item() == 0.0, f"全阈下惩罚应为 0，实得 {pen.item()}"
    assert n_viol == 0, f"违规行数应为 0，实得 {n_viol}"
    # 无违规 → 返回值不进计算图：对分数的梯度贡献恰为 0
    assert pen.requires_grad is False, "阈下惩罚不应进计算图"
    (base + pen).backward()
    s2 = n.detach().clone().requires_grad_(True)
    _loss_like_run_fold(s2, s2, mask, mask).backward()
    assert torch.equal(s.grad, s2.grad), "阈下惩罚改变了梯度（应逐位为 0 差异）"


def test_case3_single_violator():
    """情形③：单个负视频超阈 → 激活值正确（max=0.7 → 惩罚 0.2），梯度只回传到该行 argmax。"""
    n = torch.full((2, 48), 0.2)
    n[0, 7] = 0.7                      # 行 0 违规：max=0.7
    mask = torch.ones(2, 48, dtype=torch.bool)
    s = n.detach().clone().requires_grad_(True)
    pen, n_viol = tm.hard_neg_penalty(s, mask, tau=0.5, lam=1.0)
    assert n_viol == 1
    assert abs(pen.item() - 0.2) < 1e-7, f"单违规行惩罚应为 0.2，实得 {pen.item()}"
    pen.backward()
    g = s.grad
    assert abs(g[0, 7].item() - 1.0) < 1e-7, \
        f"单违规行 argmax 梯度应为 1.0（=λ/违规行数），实得 {g[0, 7].item()}"
    assert g.abs().sum().item() == g[0, 7].item(), "梯度泄漏到非 argmax 位"
    assert g[1].abs().sum().item() == 0.0, "阈下行不应有梯度"


def test_case4_mixed_batch_mean_over_violators_only():
    """情形④：混合批（2 条违规 1 条阈下）→ 均值只对违规行（(0.2+0.1)/2=0.15）。"""
    n = torch.full((3, 48), 0.2)
    n[0, 3] = 0.7                      # 违规 0.2
    n[1, 9] = 0.4                      # 阈下
    n[2, 11] = 0.6                     # 违规 0.1
    mask = torch.ones(3, 48, dtype=torch.bool)
    s = n.detach().clone().requires_grad_(True)
    pen, n_viol = tm.hard_neg_penalty(s, mask, tau=0.5, lam=1.0)
    assert n_viol == 2
    assert abs(pen.item() - 0.15) < 1e-7, \
        f"违规均值应为 (0.2+0.1)/2=0.15（不对全批平均），实得 {pen.item()}"
    pen.backward()
    g = s.grad
    assert abs(g[0, 3].item() - 0.5) < 1e-7, \
        f"违规行 argmax 梯度应为 1/2，实得 {g[0, 3].item()}"
    assert abs(g[2, 11].item() - 0.5) < 1e-7, \
        f"违规行 argmax 梯度应为 1/2，实得 {g[2, 11].item()}"
    assert g[1].abs().sum().item() == 0.0, "阈下行不应有梯度"
    # λ 缩放：λ=10 时惩罚值与梯度同乘 10
    s10 = n.detach().clone().requires_grad_(True)
    pen10, nv10 = tm.hard_neg_penalty(s10, mask, tau=0.5, lam=10.0)
    assert nv10 == 2 and abs(pen10.item() - 1.5) < 1e-6
    pen10.backward()
    assert abs(s10.grad[0, 3].item() - 5.0) < 1e-6


def test_case5_masked_rows():
    """情形⑤：掩码行 → masked max 正确，被掩码位的高分不参与，无 inf/NaN。"""
    n = torch.tensor([
        [0.9, 0.4, 0.3] + [0.2] * 45,   # 行 0：掩码位 0.9 最高，可用 max=0.4 阈下
        [0.99, 0.8, 0.1] + [0.2] * 45,  # 行 1：掩码位 0.99 最高，可用 max=0.8 违规
        [0.2] * 48,                     # 行 2：全阈下
    ])
    mask = torch.tensor([
        [False] + [True] * 47,
        [False] + [True] * 47,
        [True] * 48,
    ])
    s = n.detach().clone().requires_grad_(True)
    pen, n_viol = tm.hard_neg_penalty(s, mask, tau=0.5, lam=1.0)
    assert torch.isfinite(pen), "惩罚值出现 inf/NaN：掩码填充方向与归约方向不一致"
    assert n_viol == 1, f"只有行 1 可用 max 超阈，违规行数应 1，实得 {n_viol}"
    assert abs(pen.item() - 0.3) < 1e-7, \
        f"违规量应按可用 max 0.8 计（0.8-0.5=0.3），实得 {pen.item()}"
    pen.backward()
    g = s.grad
    assert abs(g[1, 1].item() - 1.0) < 1e-7, "梯度应落在可用片段 argmax（位 1）"
    assert g[1, 0].item() == 0.0, "被掩码位不得有梯度（其高分不参与）"
    assert g[0].abs().sum().item() == 0.0, "可用 max 阈下的行不应有梯度"
    # 边界：某行全部片段被掩码 → masked max 为 -inf，违规量 clamp 到 0，
    # 不得污染其余行的惩罚值或出现 NaN（实际训练被「无可用片段」断言拦下，
    # 这里只锁函数本身不炸）
    mask2 = mask.clone()
    mask2[2] = False
    pen2, nv2 = tm.hard_neg_penalty(n.detach(), mask2, tau=0.5, lam=1.0)
    assert torch.isfinite(pen2) and nv2 == 1, \
        f"全掩码行应被 clamp 到 0 且不影响其余行，实得 pen={pen2.item()} nv={nv2}"


def main():
    test_case1_lam0_bitwise_identical()
    test_case2_all_below_threshold()
    test_case3_single_violator()
    test_case4_mixed_batch_mean_over_violators_only()
    test_case5_masked_rows()
    print("test_mil_hard_neg_penalty: OK (5 cases)")


if __name__ == "__main__":
    main()
