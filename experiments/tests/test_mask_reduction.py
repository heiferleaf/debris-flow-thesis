# -*- coding: utf-8 -*-
"""掩码归约回归测试（2026-10-05 立）。

背景：`train_mil.py` 的 `--clip-labels` 掩码分支曾把 min 项的填充方向写反
（写成 `masked_fill(mask, +inf)`，应为 `masked_fill(~mask, +inf)`），使最小值
取到被排除的片段；35 个「48 片段全可用」单位的 min 因此为 +inf、峰谷差为 -inf，
L1 输出 inf/NaN。该路径梯度仍是有限值，训练不报错、静默产出错误成绩
（详见 run-log「阶段②第一步」2026-10-05 更正节）。本测试锁定 masked_max_min
的语义，防止同类写法再次溜过。

运行（仓库根）：python experiments/tests/test_mask_reduction.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common.video.train_mil import masked_max_min


def test_end_to_end():
    """端到端锁定（2026-10-05 追加，依据 research-auditor 复核 W1）。

    上面的用例只测被抽出来的函数。若有人把 max/min 两行重新内联回 run_fold
    （正是当初出错的位置），函数测试仍全绿而 bug 原样复活。本用例直接跑训练
    循环，锁住「掩码分支真的经过正确的归约」这一环：

      正例——正常代码跑 run_fold，损失必须有限、不抛错、预测齐全；
      反证——把 masked_max_min 换成旧写法（min 填可用位），run_fold 必须抛
            RuntimeError（有限性守卫），证明这条链真的能拦住旧 bug。
    """
    import numpy as np
    import torch

    import experiments.common.video.train_mil as tm

    rng = np.random.default_rng(0)
    n_unit, n_clip = 4, 4
    feats = rng.normal(size=(n_unit, n_clip, 512)).astype("float32")
    labels = np.array([1, 1, 0, 0], dtype="float32")
    # 单位 0/2 全部片段可用（旧 bug 的直接触发条件）；单位 1 部分可用；单位 3 仅剩一片
    mask = np.array([
        [True, True, True, True],
        [False, True, True, False],
        [True, True, True, True],
        [False, False, True, False],
    ])
    idx = np.arange(n_unit)

    def _run():
        return tm.run_fold(feats, labels, idx, idx, 2, 1e-3, "e2e",
                           clip_mask=torch.tensor(mask))

    best_state, hist, best_r, best_ep, preds = _run()
    assert best_state is not None, "run_fold 返回 None：训练中出现 NaN 分数"
    assert hist, "未产生逐 epoch 记录"
    for h in hist:
        assert np.isfinite(h["loss"]), f"逐 epoch 损失非有限：{h}"
        assert np.isfinite(h["l1"]), f"L1 非有限：{h}"
    assert len(preds) == n_unit, f"best epoch 预测不齐：{len(preds)}/{n_unit}"
    assert all(np.isfinite(v) for v in preds.values()), f"预测含非有限值：{preds}"

    # 反证：旧写法（min 把「可用位」填 +inf）必须被有限性守卫拦下
    def buggy(scores, m):
        mx = scores.masked_fill(~m, float("-inf")).max(dim=1).values
        mn = scores.masked_fill(m, float("inf")).min(dim=1).values   # 旧 bug
        return mx, mn

    real = tm.masked_max_min
    tm.masked_max_min = buggy
    try:
        try:
            _run()
        except RuntimeError as e:
            assert "损失非有限" in str(e), f"抛错信息不符：{e}"
        else:
            raise AssertionError(
                "旧写法（min 填可用位）未被拦下——有限性守卫失效或已被绕过")
    finally:
        tm.masked_max_min = real


def main():
    # 行 0：四位全可用；行 1：可用位 1、3；行 2：仅可用位 2
    scores = torch.tensor([
        [0.10, 0.50, 0.30, 0.90],
        [0.20, 0.80, 0.40, 0.60],
        [0.70, 0.10, 0.35, 0.20],
    ])
    mask = torch.tensor([
        [True, True, True, True],
        [False, True, False, True],
        [False, False, True, False],
    ])
    mx, mn = masked_max_min(scores, mask)

    assert torch.equal(mx, torch.tensor([0.90, 0.80, 0.35])), f"max 错: {mx}"
    assert torch.equal(mn, torch.tensor([0.10, 0.60, 0.35])), f"min 错: {mn}"

    # 关键性质一：结果必须有限——全可用行不得出现 ±inf（旧 bug 的直接症状）
    assert torch.isfinite(mx).all() and torch.isfinite(mn).all(), \
        "归约结果含 inf：掩码填充方向与归约方向不一致"

    # 关键性质二：min 不得取到被排除位（旧 bug 会把行 1 的 min 算成 0.20）
    assert abs(mn[1].item() - 0.60) < 1e-6, "min 取到了被排除的片段（旧 bug 回归）"
    assert abs(mn[2].item() - 0.35) < 1e-6, "单可用行的 min 应等于该位本身（旧 bug 回归）"

    # 关键性质三：全可用行的 max/min 应等于无掩码的 max/min
    assert mx[0].item() == scores[0].max().item()
    assert mn[0].item() == scores[0].min().item()

    test_end_to_end()
    print("test_mask_reduction: OK")


if __name__ == "__main__":
    main()
