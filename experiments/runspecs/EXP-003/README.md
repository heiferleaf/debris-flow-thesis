# EXP-003 复现入口

历史实验；按卡片与报告解释结果。

- 实验卡：[EXP-003](../../../docs/process/experiments/EXP-003.md)
- 配置索引：navigation.json；实际超参数以签署卡片和原运行metrics为准。
- 共用实现：[run_pilot.py](../../../experiments/common/convnext/run_pilot.py)
- 产物：data/results/convnext/<原run_id>/，保持历史产物原字节。

只读复算可以使用现有产物；重新训练仍须满足实验授权与数据协议。不要把实现复制成另一套训练循环。
