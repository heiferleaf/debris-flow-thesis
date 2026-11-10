# EXP-025 五折同分数判决比较

本入口不训练。所需一套 Linear 模型与全部五折 outer 预测由 EXP-022 原入口产生，默认模型 run 为 `data/results/exp022/linear-cv-s7-20261104-v1/`。A/B 使用同一组保存的 FP32 概率；不会回流训练或选 E，也不另加费用。

在仓库根目录执行新增五折归属夹具和元数据覆盖检查：

```powershell
python experiments/tests/test_confirmation_cv_mapping.py
python experiments/runspecs/EXP-025/compare.py prepare
```

`prepare` 只读取原 sources、五折清单和 143470 窗网格，检查 150 视频（113P/37N）、31 场次与 9 负场次，每视频与场次仅有一个 outer fold；不读取预测、视频、帧或权重。通过向已审单视频函数提供恒定占位概率，仅取得合法对计数，丢弃占位分数与判决。覆盖表、当前卡片／源码／输入指纹和源码副本保存在 `data/metadata/EXP-025/comparison-plan-v1/`，拒绝覆盖。任一缺覆盖仍保留全部视频并阻止完整比较。

B 原样调用 EXP-024 的 `describe`，在导入前要求该源文件 SHA256 为 `b2ea1dc33ab9d1865b23a55e3ecb05d921cedcefca600de8e626d2016e4f0442`。不重新解释确认规则，不重复旧边界测试。新入口源码在 `prepare` 冻结，真实读数时须保持相同；卡片与来源指纹也必须保持一致。

所有五折成功闭合并回收到本地后，执行一次：

```powershell
python experiments/runspecs/EXP-025/compare.py compare
```

默认写入 `data/results/exp025/confirmation-linear-cv-s7-20261104-v1/`，拒绝覆盖；`--out` 只能指定 EXP025 下新的直接子目录。先验证完整 run、五个阶段闭合清单及其文件指纹，再读取 CSV。支持原六列及额外含 logit 的七列；按元组 SHA、目标起点和实际首末帧时间绑定原网格。A 逐视频 max、数量、标签、场次和逐折原指标必须完全复现。只检查所需小元数据／预测文件，不哈希旧权重、不重复媒体验收。

输出逐视频 CSV、五折指标 CSV、9 负场次 CSV、全部正视频首次输出时刻与等待 CSV、源码快照和完整 JSON。无合法对或无触发时使用 `null`，不填 0；缺覆盖时完整指标与科学门槛受阻。AUC 只逐折计算，保留原概率并列，不计算跨模型 pooled AUC。负场次 FPR 先在场次内按视频计算，再对 9 场次宏平均。

所有正视频均公开 A/B 触发时刻、缺失情况和相对等待；分布保存全部有效差值、分母、中位数、均值、最大值及最大案例。该等待不等于相对真实事件起点的延迟。科学计数门槛与应用结论分别保存，应用适用性固定为未定。完整规格见 `docs/process/experiments/EXP-025.md`。
