# 泥石流识别：实验代码与结果归档

每个登记实验对应一个分支；仅收最终有效运行。没有训练结果的设计明确标记。原始数据不公开，部分历史材料缺失，不能据此宣称所有实验均已端到端复现。

共享代码保持原相对路径；输入为元信息说明及冻结小清单。已回放通过的小头权重随分支保存；其余权重尚未完成本轮验证，未上传。

## 实验入口

- [EXP-003：单帧迁移试点](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-003) — 历史试点；原始指标文件缺失
- [EXP-004：外部负图片及本地冒烟](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-004) — 准备与冒烟；无独立正式成绩
- [EXP-005：单帧正式基线](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-005) — 已运行；原始指标文件缺失
- [EXP-006：二维CNN与时序头](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-006) — 最终修订运行已完成
- [EXP-008：论文三项MIL损失重实现](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-008) — 最终150条重跑及头权重回放通过
- [EXP-009：强标签全参数微调](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-009) — 最终无增强运行已完成
- [EXP-010：图像复核级联](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-010) — 未训练草案
- [EXP-011：弱标签负包超阈惩罚](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-011) — 有效阴性实验；头权重回放通过
- [EXP-012：强标签整视频上下文](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-012) — 有效中性实验；头权重回放通过
- [EXP-013：历史全量训练设想](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-013) — 未执行草案
- [EXP-014：增大帧间隔](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-014) — 最终对照已完成
- [EXP-015：原间隔重采样草案](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-015) — 未执行草案
- [EXP-016：RGB与帧差](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-016) — 双种子开发实验完成
- [EXP-017：RGB场次分组五折参照](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-017) — 成功续跑后的五折完成
- [EXP-018：正负配对排序约束](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-018) — 三组匹配开发实验完成
- [EXP-019：真实顺序与固定置换](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-019) — 匹配开发实验完成
- [EXP-020：简化分类头小试](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-020) — 配对开发实验完成
- [EXP-021：局部时空能量](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-021) — 有效阴性开发实验完成
- [EXP-022：线性头完整开发五折](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-022) — 完整五折及结果复核完成
- [EXP-023：时间跨度受控对照](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-023) — 未执行草案
- [EXP-024：连续两窗确认](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-024) — 已有预测上的有限开发比较完成
- [EXP-025：完整开发连续确认方案](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-025) — 元数据覆盖检查；未完成全量评价
- [EXP-026：三状态与跨界软目标](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-026) — 历史设计；未训练
- [EXP-027：时间组成三锚点](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-027) — 未采用设计；未训练
- [EXP-028：逐片段参照草案](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-028) — 已暂停设计；未训练
- [EXP-029：局部三类头与跨片段上下文](https://github.com/heiferleaf/debris-flow-thesis/tree/codex/EXP-029) — 当前设计；未训练

## 复算与环境

EXP-008/011/012冻结头回放命令：`python -B experiments/tools/verify_frozen_head_archive.py --out <新的核对记录.json>`。需先在本地恢复原特征缓存和NPZ标签；它们的指纹在`data/metadata/experiment-archive/frozen-head-replay-20261110.json`。回放不重新训练、不生成原视频。

历史环境常见本地torch 2.4.1+cpu，云端运行环境以各原run保存文件为准；未提供一个未经验证的统一requirements冒充所有实验环境。

源代码迁移后的当前快照与历史执行源码分开；已有原始代码快照才作历史代码证据。结果没有独立测试时明确称开发结果。

总机器索引：`catalog.json`；共享源指纹：`SOURCE_MANIFEST.json`。
