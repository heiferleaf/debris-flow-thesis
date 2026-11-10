# EXP-008 复现入口

公开分支当前采用已完成的跨场景三项MIL版本`mil-v2-150-strict-scene`。

- 完整报告：[跨场景MIL复现实验](../../../docs/process/experiments/EXP-008-跨场景MIL复现实验报告.md)。
- 输入证据：[输入与时间轴核查](../../../docs/process/experiments/EXP-008-输入与时间轴核查.md)。

- 实验卡：[EXP-008](../../../docs/process/experiments/EXP-008.md)
- 配置索引：navigation.json；实际超参数以签署卡片和原运行metrics为准。
- 共用实现：[train_mil.py](../../../experiments/common/video/train_mil.py)、[train_strong.py](../../../experiments/common/video/train_strong.py)
- 产物：data/results/exp008/<原run_id>/，保持历史产物原字节。

只读复算可以使用现有产物；重新训练仍须满足实验授权与数据协议。不要把实现复制成另一套训练循环。
