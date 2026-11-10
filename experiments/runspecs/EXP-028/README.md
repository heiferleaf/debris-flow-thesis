# EXP-028：逐片段存在二类的离线规格

**当前暂停执行。** 见`execution-hold.json`：标签方案尚未验收，且鸡冠山下游与1号塔下游已确认同一物理场景、现角色跨训练与外评。下文角色、输入计数和B0选择仅保留为旧离线规格，不能继续作为新训练依据；尚未生成替代划分。

本目录只有规格与元数据复现入口，没有训练入口，也不授权开机或付费。实验卡为`docs/process/experiments/EXP-028.md`，当前仅准备B0；T覆盖不足，V0不启动。

数据正本：

- 标签：`data/metadata/annotation-revisions/npt-20261108-evidence-working-v3/`
- 机位角色：`data/metadata/EXP-028/cross-camera-initial-role-v1.json`
- 物理网格及继承的技术检查：`data/metadata/EXP-028/full-coverage-plan-20261108-v1/`
- 当前语义覆盖与两套拟合侧权重：`data/metadata/EXP-028/full-coverage-label-overlay-20261108-v3/`

标签版本的复现使用`publish_annotation_working_release.py`及该版本`additional-decisions.json`，写入一个新的输出目录；不能覆盖v3或原events。语义重绑定只读原网格、标签和角色，不重解码、不重复QC。示例将输出路径替换为不存在的新目录：

```text
python experiments/runspecs/EXP-028/rebind_labels.py --parent data/metadata/EXP-028/full-coverage-plan-20261108-v1/windows.jsonl --release data/metadata/annotation-revisions/npt-20261108-evidence-working-v3/released-intervals.json --roles data/metadata/EXP-028/cross-camera-initial-role-v1.json --out data/metadata/EXP-028/reproduction-label-overlay
```

新风险的两个小夹具只检查改标不解除旧QC、未知不补N：

```text
python -m unittest experiments.tests.test_rebind_full_coverage_labels
```

当前64762个保守合法窗中417窗图片齐备，另外64345窗尚未增量准备。签署具体训练卡、完成这些输入与小样本冒烟、测速并获得新费用授权后，才增加训练入口。不能把本目录或`spec.json`当作训练已经就绪。
