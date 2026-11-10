# 通用工具

此目录只接收跨实验可复用、参数化的命令行工具。共享算法进`experiments/common/`，实验专用脚本进对应`runspecs/EXP-NNN/`，有复现价值的一次性工具进`oneoff/`；无用途临时件用后删除。

- 数据与媒体：下载、解码对照、取帧、裁剪、帧缓存、来源核查。
- 标注与审计：事件审计台、标注版本合并、区间图证、通用时间/质量核查。
- 研究准备：输入覆盖、取样盘点、通用统计与推理工作台。
- 合成工作流：可复用的工作台和参考/任务准备；具体试片参数与结果归data。

带明确EXP编号的分析/报告/收集实现已移至实验目录；旧命令查`.codex/logs/workspace-maintenance-20261110/tool-moves.json`，不保留重复实现或不工作的旧壳。

例：`runspecs/EXP-021/analyze_errors.py`、`runspecs/EXP-022/collect_results.py`、`runspecs/EXP-022/report.py`。完整规则见`.codex/docs/workspace-layout.md`。
