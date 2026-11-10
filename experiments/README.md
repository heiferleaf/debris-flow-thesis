# 实验代码入口

当前研究状态见[唯一工作入口](../.codex/docs/当前研究状态.md)，实验卡及结果解释见[实验索引](../docs/process/experiments/README.md)。代码目录不维护另一份进度或训练授权。

## 代码归属

- `common/{convnext,temporal,video}/`：跨实验复用实现。
- `tools/`：参数化通用工具，如视频下载/取帧、审计台、数据检查与推理工作台。
- `splits/code/`：划分与泄漏检查；成员清单放`data/metadata/splits/`。
- `runspecs/EXP-NNN/`：各实验入口、配置、专用分析/报告/收集脚本；调用共享实现，不复制训练循环。
- `tools/oneoff/`：有复现价值的历史一次性工具。临时试验放`.codex/docs/.temp/`，用后清理。
- `tests/`：验证算法合同和关键行为的夹具测试，不用目录维护触发全量训练。

详细职责及生命周期见[工作区组织规范](../.codex/docs/workspace-layout.md)。不要恢复已迁走的`experiments/repro/`结构。

## 使用现有代码

先读对应`runspecs/EXP-NNN/README.md`及实验卡，再调用入口。配置中的历史路径须按迁移映射解释，不能改原指标或旧快照伪造复现。

当前数据使用以[跨场景规范](../docs/data/跨场景数据使用规范.md)和本实验卡为准；旧EXP-006双协议、旧max判决仅解释历史run，不是可任选的现行协议。租卡和训练需要对应授权。

通用审计台：

```text
python experiments/tools/event_annotator_server.py --dir data/process/video_frames --port 8768 --annotations-dir <当前标签目录>
```

当前标签目录通过`data/metadata/annotation-revisions/current-review.json`读取；不要仅凭上例启动新的并行标签入口。

划分检查实现位于[split_checks.py](splits/code/split_checks.py)；其他工具以实际`--help`和各卡片约定为准，不调用已删除的旧路径。

运行环境的本机路径示例在[paths.local.env.example](paths.local.env.example)；真实环境凭据不提交。训练产物位于`data/results/<线>/<run_id>/`。
