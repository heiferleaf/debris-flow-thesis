# -*- coding: utf-8 -*-
"""EXP-012 上下文聚合结构与负侧超阈惩罚单元测试（卡「代码改动 2」）。

覆盖（EXP-012 卡步骤 2）：
  ①λ=0 且无结构 → 与现役损失路径逐位同构（固定合成张量，预测分数全等）；
  ②惩罚项：负单位全阈下 → 0 且梯度为 0；单单位超阈激活正确；
    多单位混合 → 只对违规单位取均值；λ=0 完全跳过；
  ③gated 掩码三情形（全可用/部分可用/单可用）→ 输出有限、形状正确，
    且改被掩码位置的特征值 → 输出逐位不变（无泄漏）；
  ④selfattn 同样三情形 + 无泄漏；
  ⑤掩码填 0 的归约方向：聚合结果与「只 gather 可用位置手算」一致；
  ⑥两变体参数量 ≤300,000（并按卡上结构钉死精确值，防结构漂移），打印实际值。

运行（仓库根）：python experiments/tests/test_strong_ctx_penalty.py
（pytest 可用时也可 -m pytest 运行，同一套用例。）
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiments.common.video.train_strong as ts


def _make_synth(n_unit=4, n_clip=48, seed=0):
    """固定 seed 合成数据：特征 [n,48,512]、单位标签、片段标签 -1/0/1 与掩码。"""
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n_unit, n_clip, 512)).astype("float32")
    unit_labels = np.array([1, 1, 0, 0][:n_unit], dtype=int)
    clip_label = rng.choice([-1, 0, 1], size=(n_unit, n_clip), p=[0.2, 0.4, 0.4])
    clip_label[:, 0] = 1          # 保底：每单位至少一个可用片段（卡：无全掩码行）
    mask = clip_label >= 0
    assert (mask.sum(axis=1) >= 1).all()
    return feats, unit_labels, clip_label, mask


def _reference_run_fold(feats, unit_labels, clip_label, mask, tr, va, epochs, lr, seed=7):
    """现役（EXP-012 改动前）run_fold 的逐行复刻，作逐位同构的参照实现。

    与 train_mil/test_mask_reduction 同理：改动后必须有人守住「不给新参数时
    与现役逐位一致」这条防线；本函数独立于新代码路径，防止新旧共用代码
    掩盖回归。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va = np.asarray(tr), np.asarray(va)
    model = ts.StrongHead()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(feats, dtype=torch.float32)
    M = torch.tensor(mask)
    pairs = [(i, j) for i in tr for j in range(len(clip_label[i]))
             if clip_label[i][j] >= 0]
    y_clip = torch.tensor([clip_label[i][j] for i, j in pairs], dtype=torch.float32)
    idx_ui = torch.tensor([i for i, _ in pairs])
    idx_ci = torch.tensor([j for _, j in pairs])
    n_pos = int((y_clip == 1).sum())
    n_neg = int((y_clip == 0).sum())
    pos_weight = torch.tensor(n_neg / max(n_pos, 1))
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_auc, best_state, best_ep = -1.0, None, -1
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(pairs))
        for b in range(0, len(perm), ts.BS):
            sel = perm[b:b + ts.BS]
            logits = model(X[idx_ui[sel], idx_ci[sel]]).squeeze(-1)
            loss = crit(logits, y_clip[sel])
            assert torch.isfinite(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        p_va = []
        with torch.no_grad():
            for i in va:
                s = torch.sigmoid(model(X[i:i + 1]).squeeze(-1)).squeeze(0)
                p_va.append(float(s[M[i]].max()))
        p_va = np.array(p_va)
        auc = roc_auc_score(unit_labels[va], p_va) if len(set(unit_labels[va])) > 1 else float("nan")
        if not np.isnan(auc) and auc > best_auc:
            best_auc, best_state, best_ep = auc, \
                {k: v.clone() for k, v in model.state_dict().items()}, ep
    preds = {}
    if best_state is not None:
        model.load_state_dict(best_state)
        with torch.no_grad():
            for i in va:
                s = torch.sigmoid(model(X[i:i + 1]).squeeze(-1)).squeeze(0)
                preds[int(i)] = float(s[M[i]].max())
    return preds, best_ep


def test_identity_path_bitwise():
    """①λ=0 且无结构 → 新 run_fold 与现役参照逐位同构（分数全等、best_epoch 相同）。"""
    feats, unit_labels, clip_label, mask = _make_synth()
    tr, va = [0, 1], [2, 3]
    preds_ref, ep_ref = _reference_run_fold(
        feats, unit_labels, clip_label, mask, tr, va, epochs=3, lr=1e-3)
    _, _, ep_new, preds_new, extra = ts.run_fold(
        feats, unit_labels, clip_label, mask, tr, va, 3, 1e-3, "t1")
    assert ep_new == ep_ref, f"best_epoch 不一致：新 {ep_new} vs 参照 {ep_ref}"
    assert set(preds_new) == set(preds_ref), "预测键集合不一致"
    for k in preds_ref:
        assert preds_new[k] == preds_ref[k], \
            f"单位 {k} 分数不同构：新 {preds_new[k]!r} vs 参照 {preds_ref[k]!r}"
    # 同构路径也必须产出 λ 推导读数且有限
    assert np.isfinite(extra["bce_last3"])
    assert extra["neg_stats"]["n_neg_tr"] == int((unit_labels[tr] == 0).sum())


def test_penalty_all_below_threshold():
    """②a 负单位全部阈下 → 惩罚恰为 0，且不携带梯度（对损失的梯度贡献为 0）。"""
    logits = torch.tensor([-5.0, -3.0, -1.0], requires_grad=True)
    ui = torch.tensor([0, 0, 0])
    unit_labels = np.array([0])
    pen = ts.hard_neg_penalty(logits, ui, unit_labels, lam=2.0)
    assert pen.item() == 0.0, f"全阈下惩罚应为 0，实得 {pen.item()}"
    assert not pen.requires_grad, "全阈下惩罚不应挂接计算图（梯度贡献应恰为 0）"


def test_penalty_single_unit():
    """②b 单负单位超阈：惩罚 = λ·(σ(max)−0.5)，梯度只流向取到 max 的片段。"""
    logits = torch.tensor([3.0, 2.0], requires_grad=True)
    ui = torch.tensor([0, 0])
    unit_labels = np.array([0])
    expected = 2.0 * (float(torch.sigmoid(torch.tensor(3.0))) - 0.5)
    pen = ts.hard_neg_penalty(logits, ui, unit_labels, lam=2.0)
    assert abs(pen.item() - expected) < 1e-6, f"{pen.item()} != {expected}"
    pen.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert float(logits.grad[1]) == 0.0, "梯度不应流向未取到 max 的片段"
    assert float(logits.grad[0]) > 0.0


def test_penalty_mixed_units():
    """②c 多单位混合：只对违规负单位取均值；正单位高分不进惩罚；λ=0 跳过。"""
    # 单位 0=负（片段 3.0/-1.0，max=σ(3)，违规）；单位 1=负（2.5，违规）；
    # 单位 2=正（4.0，不得计入）
    logits = torch.tensor([3.0, -1.0, 2.5, 4.0])
    ui = torch.tensor([0, 0, 1, 2])
    unit_labels = np.array([0, 0, 1])
    sig = torch.sigmoid(logits)
    expected = 1.5 * ((float(sig[0]) - 0.5) + (float(sig[2]) - 0.5)) / 2
    pen = ts.hard_neg_penalty(logits, ui, unit_labels, lam=1.5)
    assert abs(pen.item() - expected) < 1e-6, f"{pen.item()} != {expected}"
    # λ=0：即使有违规也必须为 0（完全跳过路径）
    pen0 = ts.hard_neg_penalty(logits, ui, unit_labels, lam=0.0)
    assert pen0.item() == 0.0 and not pen0.requires_grad
    # 批内负单位全阈下（即使存在正单位高分）→ 0
    logits_lo = torch.tensor([-3.0, -2.0, -1.0, 4.0])
    pen_lo = ts.hard_neg_penalty(logits_lo, ui, unit_labels, lam=1.5)
    assert pen_lo.item() == 0.0


def _no_leak_case(agg, n_clip=48, seed=3):
    """③/④ 共用：全可用/部分可用/单可用三情形下，改被掩码位置的特征值，
    forward_unit 输出必须逐位不变（无泄漏）。"""
    rng = np.random.default_rng(seed)
    h = torch.tensor(rng.normal(size=(n_clip, 512)).astype("float32"))
    cases = {
        "all": torch.ones(n_clip, dtype=torch.bool),
        "partial": torch.tensor([i % 3 != 1 for i in range(n_clip)]),
        "single": torch.tensor([i == 5 for i in range(n_clip)]),
    }
    for name, mask in cases.items():
        out1 = agg.forward_unit(h, mask)
        assert torch.isfinite(out1).all(), f"{name}: 输出含非有限值"
        h2 = h.clone()
        h2[~mask] = torch.tensor(rng.normal(size=(int((~mask).sum()), 512)).astype("float32")) * 7.0
        out2 = agg.forward_unit(h2, mask)
        assert torch.equal(out1, out2), f"{name}: 改被掩码位置特征值改变了输出（泄漏）"
        yield name, mask, h, out1


def test_gated_masks():
    """③ gated 三情形：形状 [48,64]、有限、无泄漏；全可用时各片段附加特征相同。"""
    agg = ts.GatedCtx()
    for name, mask, h, out in _no_leak_case(agg):
        assert out.shape == (48, 64), f"{name}: 形状 {tuple(out.shape)} != (48,64)"
    # 全可用时每片段拿到同一份上下文（expand 语义）
    mask = torch.ones(48, dtype=torch.bool)
    out = agg.forward_unit(h, mask)
    assert torch.equal(out[0], out[17]), "全可用时各片段上下文应相同"


def test_selfattn_masks():
    """④ selfattn 三情形：形状 [48,128]、有限、无泄漏。"""
    agg = ts.SelfAttnCtx()
    last = None
    for name, mask, h, out in _no_leak_case(agg):
        assert out.shape == (48, 128), f"{name}: 形状 {tuple(out.shape)} != (48,128)"
        last = (name, mask, h, out)
    assert last is not None


def test_mask_reduction_direction():
    """⑤ 掩码填 0 的归约方向：聚合输出 = 只对可用位置手算的结果（逐位一致）。

    手算完全绕开「填 0 + 掩码 softmax」的写法，只 gather 可用位置单独算，
    若填 0 方向写反（例如掩码位漏排除）两者必然分开。
    """
    rng = np.random.default_rng(11)
    h = torch.tensor(rng.normal(size=(48, 512)).astype("float32"))
    mask = torch.tensor([i % 4 != 2 for i in range(48)])
    valid = mask.nonzero().squeeze(-1)

    # gated：c = softmax(w·z)_{valid} 对 z_{valid} 加权求和
    agg = ts.GatedCtx()
    out = agg.forward_unit(h, mask)
    z = torch.tanh(agg.V(h[valid])) * torch.sigmoid(agg.U(h[valid]))
    a = torch.softmax(agg.w(z).squeeze(-1), dim=0)
    c_manual = a.unsqueeze(-1) * z
    c_manual = c_manual.sum(dim=0)
    assert torch.allclose(out[0], c_manual, atol=1e-5), \
        "gated 上下文与「只用可用位置手算」不一致——掩码归约方向可能有误"

    # selfattn：valid 查询行的输出 = softmax(q·k_valid/√d) 对 v_valid 加权后过 out
    sat = ts.SelfAttnCtx()
    out_s = sat.forward_unit(h, mask)
    hm = h * mask.unsqueeze(-1).to(h.dtype)
    x = sat.proj(hm)
    q, k, v = sat.Wq(x), sat.Wk(x), sat.Wv(x)
    i = int(valid[3])                                  # 任取一个可用查询位
    a_s = torch.softmax((q[i] @ k[valid].T) / sat.scale, dim=0)
    o_manual = sat.out(a_s @ v[valid])
    assert torch.allclose(out_s[i], o_manual, atol=1e-5), \
        "selfattn 输出与「只用可用键位手算」不一致——掩码归约方向可能有误"


def test_param_counts():
    """⑥ 两变体参数量 ≤300,000；并按卡上写死的结构核对精确值（防结构漂移）。"""
    g = ts.GatedCtx()
    s = ts.SelfAttnCtx()
    pg = sum(p.numel() for p in g.parameters())
    ps = sum(p.numel() for p in s.parameters())
    print(f"    参数量：gated={pg}  selfattn={ps}（上限 300,000）")
    assert pg == 65728, f"gated 参数量 {pg} != 65728（V/U 各 512·64+64，w 64）"
    assert ps == 131712, f"selfattn 参数量 {ps} != 131712（proj 512·128+128 + QKV/输出各 128·128+128）"
    assert pg <= 300_000 and ps <= 300_000


def test_e2e_ctx_and_penalty_finite():
    """端到端冒烟：聚合两变体与惩罚路径各跑 2 epoch，损失、读数、预测全部有限。"""
    feats, unit_labels, clip_label, mask = _make_synth(seed=1)
    tr, va = [0, 1], [2, 3]
    for ctx, lam in [("gated", 0.0), ("selfattn", 0.0), ("gated", 0.7), ("none", 0.7)]:
        state, best_r, ep, preds, extra = ts.run_fold(
            feats, unit_labels, clip_label, mask, tr, va, 2, 1e-3, f"e2e-{ctx}-{lam}",
            context_agg=ctx, lam=lam)
        for h in extra["hist"]:
            assert np.isfinite(h["bce"]) and np.isfinite(h["pen"]), f"{ctx}/{lam}: 读数非有限 {h}"
            assert h["pen"] >= 0.0, f"{ctx}/{lam}: 惩罚项为负 {h}"
        for v in preds.values():
            assert np.isfinite(v), f"{ctx}/{lam}: 预测非有限"
        assert set(extra["neg_stats"]) == {"n_neg_tr", "viol_mean_violators",
                                           "viol_mean_all", "viol_frac"}


def main():
    tests = [test_identity_path_bitwise,
             test_penalty_all_below_threshold,
             test_penalty_single_unit,
             test_penalty_mixed_units,
             test_gated_masks,
             test_selfattn_masks,
             test_mask_reduction_direction,
             test_param_counts,
             test_e2e_ctx_and_penalty_finite]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print(f"全部 {len(tests)} 条通过")


if __name__ == "__main__":
    main()
