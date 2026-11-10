# -*- coding: utf-8 -*-
"""EXP-009 微调停跑判据的回归测试（2026-10-15 立）。

背景：卡片「成功/失败」节预登记的停跑规则先后写错过两次。

  1. 最初只写「AUC 崩塌即停」，没有可执行的判定。2026-10-15 弱标签线第 2 折就
     已经崩了，靠人现场判断，跑到第 3 折第 14 轮才停，多花约 ¥2。
  2. 补上的判定写成「折内**最好的一轮**验证 AUC ≤ 0.55」。可那一折的验证 AUC 是
     从第 0 轮的 0.738 一路掉到第 9 轮起恒为 0.500——最好的一轮恰恰是崩塌**之前**
     的 0.738，大于 0.55，于是这条判定在已知失败上永远不会触发（research-auditor
     B1）。

现在改成看**末尾**连续多少轮贴在瞎猜线上。本测试用那次跑出来的两条真实逐轮记录
把判据钉死：一条是真崩的折，必须触发；一条是来回跳但没崩的折，必须不触发。
判据改方向、改阈值、改成看峰值，这两条里必有一条翻车。

运行（仓库根）：python experiments/tests/test_ft_stop_rule.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common.video import train_ft as tf

# 2026-10-15 弱标签线 f2 的真实逐轮验证 AUC。第 9 轮起恒为 0.500，一直到最后。
KNOWN_COLLAPSED = [0.7381, 0.1905, 0.2560, 0.4881, 0.6607, 0.6964, 0.6905, 0.6667,
                   0.6250, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
                   0.5000, 0.5000, 0.5000, 0.5000]

# 同一次 f1 折的真实逐轮验证 AUC。在 0.000 到 1.000 之间来回跳，最长的连续低分
# 只有 3 轮（第 4~6 轮），但最终学出来了（最好一轮 1.000）。
KNOWN_UNSTABLE_BUT_OK = [0.8052, 0.7662, 0.0844, 0.5844, 0.1558, 0.0000, 0.1429,
                         0.5584, 0.9870, 0.8506, 0.8864, 0.9740, 0.8831, 1.0000,
                         0.2662, 1.0000, 1.0000, 0.8929, 0.1753, 0.4091]


def _hist(aucs):
    return [{"va_auc": a} for a in aucs]


def test_collapsed_fold_triggers():
    """真崩过的那一折必须被判为退化。"""
    streak = tf.trailing_low_auc_streak(_hist(KNOWN_COLLAPSED))
    assert streak >= tf.DEGENERATE_STREAK, (
        f"真崩过的折末尾连续低分只有 {streak} 轮，够不到 {tf.DEGENERATE_STREAK} 轮，"
        f"判据在已知失败上不触发")


def test_unstable_fold_does_not_trigger():
    """来回跳但没崩的那一折不能被误判为退化。"""
    streak = tf.trailing_low_auc_streak(_hist(KNOWN_UNSTABLE_BUT_OK))
    assert streak < tf.DEGENERATE_STREAK, (
        f"没崩的折末尾连续低分 {streak} 轮，够到了 {tf.DEGENERATE_STREAK} 轮，"
        f"判据会误伤正常的折")


def test_uses_tail_not_peak():
    """反证：判据必须看末尾，不能看折内最好的一轮。

    真崩的那一折最好的一轮是 0.7381，大于 0.55——这正是上一版判据失效的原因。
    这里把「看峰值」这个写法显式实现一遍，断言它在已知失败上不触发，以此证明
    两种写法确实不同、而我们现在用的不是这一种。"""
    peak = max(KNOWN_COLLAPSED)
    assert peak > tf.DEGENERATE_AUC, (
        "这条夹具的前提变了：真崩那一折的最好一轮本该高于阈值，"
        "否则证明不了「看峰值」会漏判")
    peak_based_fires = peak <= tf.DEGENERATE_AUC
    assert not peak_based_fires, "「看峰值」竟然触发了，夹具前提与记录不符"
    assert tf.trailing_low_auc_streak(_hist(KNOWN_COLLAPSED)) >= tf.DEGENERATE_STREAK, \
        "换成看末尾之后必须触发"


def test_streak_counts_only_trailing_run():
    """只数末尾那一段连续的低分，中间断过就重新计。"""
    aucs = [0.10, 0.10, 0.10, 0.90, 0.10, 0.10]
    assert tf.trailing_low_auc_streak(_hist(aucs)) == 2, "末尾连续段数错了"
    assert tf.trailing_low_auc_streak(_hist([0.10, 0.10, 0.10])) == 3
    assert tf.trailing_low_auc_streak(_hist([0.90])) == 0


def test_nan_breaks_the_run():
    """验证折单一类别时验证 AUC 是 nan，nan 必须当作断点，不能算成低分。"""
    assert tf.trailing_low_auc_streak(_hist([0.10, 0.10, float("nan"), 0.10])) == 1


def test_early_stop_switch():
    """冒烟折与单折调试传 0，必须永不早停。"""
    hist = _hist(KNOWN_COLLAPSED)
    assert not tf.early_stop_due(hist, 0, "smoke"), "传 0 时不该早停"
    assert tf.early_stop_due(hist, tf.DEGENERATE_STREAK, "f2"), "该早停时没早停"
    # 没崩的那一折不早停
    assert not tf.early_stop_due(_hist(KNOWN_UNSTABLE_BUT_OK),
                                 tf.DEGENERATE_STREAK, "f1")


def _check_stop_modes(expr):
    """核对正式调用两种模式；未知表达式直接拒绝。"""
    import ast

    def value(node, all_data):
        if isinstance(node, ast.Name) and node.id == "DEGENERATE_STREAK":
            return tf.DEGENERATE_STREAK
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if (isinstance(node, ast.IfExp)
                and isinstance(node.test, ast.Attribute)
                and isinstance(node.test.value, ast.Name)
                and node.test.value.id == "args"
                and node.test.attr == "all_data"):
            return value(node.body if all_data else node.orelse, all_data)
        raise AssertionError("无法确认 stop_low_streak 表达式的两种模式")

    assert value(expr, False) == tf.DEGENERATE_STREAK, "普通五折必须启用连续低分停跑"
    assert value(expr, True) == 0, "all_data 不得按验证 AUC 提前停跑"


def test_stop_mode_check_rejects_wrong_wiring():
    import ast

    _check_stop_modes(ast.parse("0 if args.all_data else DEGENERATE_STREAK", mode="eval").body)
    for source in ("0", "DEGENERATE_STREAK",
                   "DEGENERATE_STREAK if args.all_data else 0"):
        try:
            _check_stop_modes(ast.parse(source, mode="eval").body)
        except AssertionError:
            continue
        raise AssertionError(f"错误接法未被拒绝：{source}")


def test_stop_rule_is_wired_into_the_training_loop():
    """接线检查：判据写对了，还得真的接上，否则等于没有。

    上面五个用例查的都是纯函数，与训练循环无关。把全量分支那次 `run(...)` 调用的
    `stop_low_streak` 改成 0，或者删掉 `run_mil` / `run_strong` 里的
    `early_stop_due` 调用，上面全过、卡片核对也全过，退化折照跑满 20 轮——
    与 2026-10-15 增强轴事故同形的静默失败（research-auditor 2026-10-15 B2）。

    用语法树扫源码，不扫文本：注释里也会出现这两个名字。"""
    import ast

    tree = ast.parse(Path(tf.__file__).read_text(encoding="utf-8"))
    problems = []

    # 一、两个训练循环里都要有 early_stop_due(...)，且它的返回值被 if 用来跳出循环
    for fn_name in ("run_mil", "run_strong"):
        fn = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == fn_name), None)
        if fn is None:
            problems.append(f"找不到函数 {fn_name}")
            continue
        guarded = False
        for n in ast.walk(fn):
            if not isinstance(n, ast.If):
                continue
            if not any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                       and c.func.id == "early_stop_due"
                       for c in ast.walk(n.test)):
                continue
            # 光把 early_stop_due 放进 if 还不够：循环体里没有 break 的话，
            # 这一折照样会跑满全部轮次（把 break 写成 pass 就是这个形状）。
            if any(isinstance(b, ast.Break) for b in ast.walk(n)):
                guarded = True
        if not guarded:
            problems.append(f"{fn_name} 里 early_stop_due 的结果没有用来 break 跳出"
                            f"训练轮次循环")

    # 二、冒烟调用使用默认值；正式调用普通五折启用停跑，all_data 禁用。
    run_calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "run"]
    if len(run_calls) != 2:
        problems.append(f"run(...) 调用点应为 2 处（冒烟一处、全量一处），"
                        f"实为 {len(run_calls)} 处")
    else:
        passing = [kw.value for c in run_calls for kw in c.keywords
                   if kw.arg == "stop_low_streak"]
        if len(passing) != 1:
            problems.append(f"两处 run(...) 中应恰有一处传 stop_low_streak"
                            f"（全量那处），实为 {len(passing)} 处")
        else:
            try:
                _check_stop_modes(passing[0])
            except AssertionError as exc:
                problems.append(str(exc))

    assert not problems, "停跑判据没接上训练循环：" + "；".join(problems)


def main():
    test_collapsed_fold_triggers()
    test_unstable_fold_does_not_trigger()
    test_uses_tail_not_peak()
    test_streak_counts_only_trailing_run()
    test_nan_breaks_the_run()
    test_early_stop_switch()
    test_stop_rule_is_wired_into_the_training_loop()
    test_stop_mode_check_rejects_wrong_wiring()
    print("test_ft_stop_rule: OK")


if __name__ == "__main__":
    main()
