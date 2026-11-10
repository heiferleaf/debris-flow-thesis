# -*- coding: utf-8 -*-
"""EXP-009 微调增强轴回归测试（2026-10-15 立）。

背景：`train_ft.py` 的增强曾在两处写成 `u8[flip] = u8[flip].flip(-1)`。张量形状
是 `[视频, 片段, 16 帧, 高, 宽, 3 通道]`，最后一维是颜色通道，所以 `flip(-1)`
把红蓝对调，而不是把画面左右翻转；宽度维是倒数第二维，应写 `flip(-2)`。这个错
不报错、不改形状、训练照跑、成绩照出，强标签线 5 折与弱标签线 2 折跑完后才发现，
两条线成绩作废（详见 run-log《EXP-009-20261015-ft》第七节）。本测试锁定三件事：

  1. 翻转翻的必须是宽度维，不是颜色维、不是高度维；
  2. 增强只动被选中的那些片段，没选中的逐位不变；
  3. 一条视频的分数取「可用片段」的最大值，被掩码剔除的片段不参与。

运行（仓库根）：python experiments/tests/test_ft_augment.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common.video import train_ft as tf


def _probe():
    """与真实张量同形的探针：各维长度互不相同，翻错任何一维都必定改变结果。

    形状 [视频, 片段, 16 帧, 高, 宽, 通道] = [2, 3, 16, 4, 5, 3]。宽度 5、通道 3、
    高度 4 三者不等，所以「翻成通道」和「翻成高度」都会被断言抓住。"""
    return torch.arange(2 * 3 * 16 * 4 * 5 * 3, dtype=torch.uint8).reshape(
        2, 3, 16, 4, 5, 3)


def test_axis():
    """翻转翻的是宽度维（倒数第二维），形状不变。"""
    probe = _probe()
    out = tf.horizontal_flip(probe)
    assert out.shape == probe.shape, f"翻转改了形状：{out.shape} != {probe.shape}"
    assert torch.equal(out, probe.flip(-2)), "翻转的不是宽度维"
    assert not torch.equal(out, probe.flip(-1)), "翻转的是颜色通道（轴写成了 -1）"
    assert not torch.equal(out, probe.flip(-3)), "翻转的是高度维（轴写成了 -3）"
    assert torch.equal(tf.horizontal_flip(out), probe), "翻两次应还原"


def test_startup_selfcheck_catches_old_bug():
    """反证：把 horizontal_flip 换回旧写法，启动自检必须报错。"""
    real = tf.horizontal_flip
    tf.horizontal_flip = lambda u8: u8.flip(-1)          # 2026-10-15 的旧写法
    try:
        try:
            tf.assert_augment_axis()
        except AssertionError as exc:
            msg = str(exc)
            assert "宽度维" in msg or "颜色通道" in msg, f"报错信息不符：{msg}"
        else:
            raise AssertionError("旧写法（flip(-1) 翻颜色通道）未被启动自检拦下")
    finally:
        tf.horizontal_flip = real


def test_augment_batch_touches_only_selected():
    """augment_batch 只改被选中的片段，其余逐位不变。"""
    u8 = _probe().clone()
    before = u8.clone()
    # 选第 0 条视频的第 1 个片段、第 1 条视频的第 0 个片段
    flip = torch.zeros(2, 3, dtype=torch.bool)
    flip[0, 1] = True
    flip[1, 0] = True

    out = tf.augment_batch(u8, flip)
    assert out.shape == before.shape, "augment_batch 改了形状"
    assert torch.equal(out[0, 1], before[0, 1].flip(-2)), "选中的片段没被翻转"
    assert torch.equal(out[1, 0], before[1, 0].flip(-2)), "选中的片段没被翻转"
    # 没选中的片段必须逐位不变
    keep = ~flip
    assert torch.equal(out[keep], before[keep]), "没选中的片段被改动了"
    # 选中的片段翻的是宽度，不是颜色：与「颜色翻转」的结果必须不同。
    # （不能逐像素比对颜色维——水平翻转本来就会把像素连同通道值一起挪位置。）
    assert not torch.equal(out[0, 1], before[0, 1].flip(-1)), \
        "选中的片段翻的是颜色通道（轴写成了 -1）"

    # 一个都没选中时，结果应与输入逐位相同
    none = torch.zeros(2, 3, dtype=torch.bool)
    assert torch.equal(tf.augment_batch(_probe().clone(), none), _probe()), \
        "没有任何片段被选中时结果被改动了"


def test_unit_score_uses_available_clips_only():
    """一条视频的分数 = 可用片段的最大值，被剔除的片段不参与。"""
    scores = torch.tensor([0.10, 0.90, 0.30, 0.20])
    avail = torch.tensor([True, False, True, True])
    got = float(tf.aggregate_unit_score(scores, avail))
    # 容差取 1e-6：0.30 存成 float32 是 0.30000001192，不能用严格相等
    assert abs(got - 0.30) < 1e-6, \
        f"取到了被掩码剔除的片段：期望 0.30（0.90 落在不可用片段上），实得 {got}"

    # 只留一个可用片段时，结果应等于该片段本身
    only = torch.tensor([False, False, True, False])
    assert abs(float(tf.aggregate_unit_score(scores, only)) - 0.30) < 1e-6


def test_no_stray_flip_callsites():
    """结构性封口：源码里除了 horizontal_flip 与 assert_augment_axis，别处不得直接调 .flip()。

    启动自检只证明 horizontal_flip 这一处写对了，证明不了没有人在别处新写一行翻错
    维的代码——那正是 2026-10-15 事故的形状（同一行复制在两个调用点）。这里用语法树
    扫，不扫文本：文本里注释和文档字符串也会出现 `.flip(-1)` 字样，扫文本会误报。"""
    import ast

    allowed = {"horizontal_flip", "assert_augment_axis"}
    # 扫整个 exp008 目录，不只扫 train_ft.py：另外两个脚本（train_mil.py /
    # train_strong.py）当前没有翻转代码，但将来若有人往那里加一行翻错维的增强，
    # 只扫一个文件是看不见的（research-auditor 2026-10-15 S5）。
    root = Path(tf.__file__).resolve().parent
    stray = []
    for src in sorted(root.glob("*.py")):
        tree = ast.parse(src.read_text(encoding="utf-8"))
        funcs = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if not (isinstance(f, ast.Attribute) and f.attr == "flip"):
                continue
            # 找包住这个调用的最内层函数
            owner = None
            for fn in funcs:
                end = fn.end_lineno or fn.lineno
                if fn.lineno <= node.lineno <= end and (owner is None
                                                        or fn.lineno > owner.lineno):
                    owner = fn
            if owner is None or owner.name not in allowed:
                stray.append(f"{src.name}:{getattr(owner, 'name', '模块顶层')}"
                             f":{node.lineno}")
    assert not stray, f"这些地方绕过了 horizontal_flip 直接调 .flip()：{stray}"


def test_augment_switch():
    """增强开关：none 不翻转也不消耗随机数；hflip 按掷硬币结果翻转。"""
    import numpy as np

    probe = torch.arange(2 * 8 * 16 * 4 * 5 * 3, dtype=torch.uint8).reshape(
        2, 8, 16, 4, 5, 3)

    gen = np.random.RandomState(7)
    out = tf.apply_augment(probe.clone(), (2, 8), gen, "none")
    assert torch.equal(out, probe), "增强设为 none 时输入被改动了"
    assert np.array_equal(gen.random(4), np.random.RandomState(7).random(4)), \
        "增强设为 none 时消耗了随机数——两套成绩会多出一个不受控变量"

    gen = np.random.RandomState(7)
    out = tf.apply_augment(probe.clone(), (2, 8), gen, "hflip")
    hit = torch.from_numpy(np.random.RandomState(7).random((2, 8)) < 0.5)
    expected = probe.clone()
    expected[hit] = probe[hit].flip(-2)
    assert torch.equal(out, expected), "增强设为 hflip 时的翻转结果不符预期"

    assert tf.augment_label("none") != tf.augment_label("hflip"), \
        "两档增强的文字相同，落盘的 metrics.json 会分不出是哪一套"


def main():
    test_axis()
    test_startup_selfcheck_catches_old_bug()
    test_augment_batch_touches_only_selected()
    test_unit_score_uses_available_clips_only()
    test_no_stray_flip_callsites()
    test_augment_switch()
    print("test_ft_augment: OK")


if __name__ == "__main__":
    main()
