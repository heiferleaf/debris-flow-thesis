# EXP-029 设计入口

本目录目前只有设计规格，没有训练脚本。实验卡见 `docs/process/experiments/EXP-029.md`，机器可读规格为 `design.json`。当前禁止执行训练、租卡或付费生成，所有授权字段均为false。

只读标签/网格盘点：

```text
python -X utf8 experiments/runspecs/EXP-029/prepare_design_inventory.py --out data/metadata/EXP-029/design-input-inventory-v1
```

该目录已生成，重算须给新版本目录。它只读取CSV/JSON/既有网格，不解码、不改标签、不运行模型。输出中的名义锚点数不是准备完成数。

候选角色成员生成命令（输出已存在时拒绝覆盖）：

```text
python -X utf8 experiments/runspecs/EXP-029/prepare_design_inventory.py --candidate-roles-out data/metadata/EXP-029/candidate-block-folds-v1.json
```

候选四折尚未冻结；实际物理场景关系和合法输入支持核定前不得当成可运行划分。后续实现优先调用 `experiments/common/video/` 共用模块，不复制旧六通道二分类训练循环。公共预训练冻结特征可复用，历史本池微调特征不可作无泄漏公共编码器。
