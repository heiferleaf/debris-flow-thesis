# -*- coding: utf-8 -*-
"""开跑前的「实验卡片 vs 训练代码」逐条核对（长期工具）。

## 为什么有这个东西

本仓库出现过三次同一类事故：代码与文档不符，程序不报错，成绩照出，事后才发现。

  1. `train_mil.py` 掩码填充方向写反——L1 项取到 `inf`，静默产出两个错误成绩；
  2. `v1_34` 中间件时间轴坐标表偏离审计基准——该视频的重建取材作废；
  3. EXP-009 增强把「画面左右翻转」写成了翻转颜色通道——两条线的成绩作废。

三次的共同点：错误藏在代码里，靠人读文档对不出来；只有专门去问「这一条到底
怎么实现的」才查得出来。而「专门去问」靠人是靠不住的——审查委托单里只写了
「审我改动的部分」，圈外那一行就没人过问（第 3 次正是这样漏掉的）。

所以把卡片里每一条设定写成一段**可执行**的检查，在任何 GPU 计算之前跑完。
检查的是实际对象——真的把优化器建出来读它的学习率，真的把批归一化层找出来
看它处于什么状态——不是读代码文本。任何一条不通过就非零退出，不许开 GPU。

## 用法

    from card_checks import CardCheck

    cc = CardCheck("EXP-009 / train_ft.py")
    cc.add("骨干解冻", "全部参数参与训练", f"{n_tr}/{n_all} 个参数张量", n_tr == n_all)
    ...
    cc.report()      # 有任何一条不通过即抛 SystemExit（非零退出码），通过则打印全表

## 写检查的三条规矩

1. **查实际对象，不查文本**：读 `opt.param_groups[i]["lr"]`，不要在源码里搜
   `lr=1e-3`。文本检查对重命名、默认值覆盖、变量拼接全都失明。
2. **通过也要记**：报告里同时列出通过与不通过的条目。只列失败项的话，读者
   分不清「查过且没问题」和「根本没写这条检查」。
3. **一条检查只回答一个问题**：条目名写成读者能直接判断「这说的是哪件事」的
   短语，不写代号。
"""


class CardCheck:
    """一张「卡片要求 vs 实测值」的对照表。"""

    def __init__(self, subject):
        self.subject = subject
        self.rows = []

    def add(self, item, card_says, actual, ok):
        """记一条核对结果。

        item      —— 这一条查的是哪件事（短语，读者一眼能懂）
        card_says —— 卡片/文档里对这件事的写法
        actual    —— 从实际对象上读到的值
        ok        —— 两者是否一致（真假值）
        """
        self.rows.append({"item": item, "card": card_says,
                          "actual": str(actual), "ok": bool(ok)})
        return bool(ok)

    @property
    def failed(self):
        return [r for r in self.rows if not r["ok"]]

    def report(self):
        """打印全表；有任何一条不通过就抛 SystemExit（非零退出码）。"""
        width = max(len(r["item"]) for r in self.rows)
        print(f"\n=== 卡片核对：{self.subject} ===", flush=True)
        for r in self.rows:
            mark = "通过  " if r["ok"] else "不通过"
            print(f"  [{mark}] {r['item']:<{width}}｜卡片写：{r['card']}"
                  f"｜实测：{r['actual']}", flush=True)
        if self.failed:
            detail = "；".join(
                f"{r['item']}（卡片写 {r['card']}，实测 {r['actual']}）"
                for r in self.failed)
            raise SystemExit(
                f"\n卡片核对不通过 {len(self.failed)}/{len(self.rows)} 条，"
                f"拒绝开跑：{detail}\n")
        print(f"  全部 {len(self.rows)} 条通过\n", flush=True)


def count_trainable(module):
    """返回 (参与训练的参数张量数, 全部参数张量数)。

    参与训练 = `requires_grad` 为真。数张量个数而不是参数元素个数，是为了让
    「有一层被漏掉」这种情况直接显示成「少 1」而不是被大数字淹没。"""
    all_p = list(module.parameters())
    return sum(1 for p in all_p if p.requires_grad), len(all_p)


def optimizer_lr_of(opt, params):
    """返回优化器里管着 `params` 这一组参数的实际学习率；没有则返回 None。

    用参数对象本身比对（`is`），不用顺序或名字——顺序和名字都会在重构后漂移。"""
    ids = {id(p) for p in params}
    for g in opt.param_groups:
        if ids and ids <= {id(p) for p in g["params"]}:
            return g["lr"]
    return None


def optimizer_covers_all(opt, module):
    """检查模块里每个参与训练的参数都在优化器里。

    不在优化器里的参数永远不会被更新，而程序不会报任何错——正是要防的那种静默。"""
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    missing = [n for n, p in module.named_parameters()
               if p.requires_grad and id(p) not in in_opt]
    return len(missing) == 0, missing


def bn_training_states(module):
    """返回模块里所有批归一化层的 (层名, 是否处于训练态)。

    批归一化处于训练态时用当前批的统计量、并更新运行统计量；处于评估态时用
    运行统计量、不更新。两种状态算出的数值不同，而程序都不报错。"""
    import torch.nn as nn
    out = []
    for name, m in module.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            out.append((name, bool(m.training)))
    return out
