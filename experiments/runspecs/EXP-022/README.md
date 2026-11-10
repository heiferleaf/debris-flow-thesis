# EXP-022 Linear场次分组开发五折

使用了 data-usage-protocol skill。沿用已批准的视频×类别共享配额与场次分组五折；无独立测试、未见机位或新时序机制主张。

仅新增三个20轮inner（fold2/3/5）和五个fresh fixed-E outer；fold1/4的EXP-020 seed7 inner按真实闭合来源复用，E10/6。所有outer只继承本折E，不加载inner参数或optimizer。原深头比较从已通过EXP-017联合audit所绑定的原始视频概率派生，保留父/子run_id，不重训或评分深头。

本地准备（无模型构造、forward或像素重扫）：

```text
.venv/Scripts/python.exe -B experiments/runspecs/EXP-022/prepare.py
.venv/Scripts/python.exe -B experiments/tests/test_linear_cv_execution.py
.venv/Scripts/python.exe -B experiments/runspecs/EXP-022/pipeline.py --help
```

计划目录`data/metadata/EXP-022/linear-plan-v1/`拒绝覆盖。只含小plan与deep-baseline，原20轮日程/窗口/缓存/像素证明原地引用。当前协议与历史源快照显式映射，不将旧live治理SHA假装仍有效。`epoch_execution_sha256`绑定原EXP-017逐轮主体，prepare要求新增主体与原源文本一致。

生产命令由租赁所有者填入实际费用截止与已经核实的平台定时回执，不能使用示例时间；阶段费用/时长预测只提醒，不拒绝下一阶段。

```text
/bin/bash experiments/runspecs/EXP-022/run.sh --plan data/metadata/EXP-022/linear-plan-v1 --out /root/exp022-results/linear-cv-s7-20261103-v1 --reference-root /root/exp020-results/head-s7s8-20261103-v2 --ffmpeg <原已验ffmpeg7绝对路径> --money-deadline-unix <实际金额截止Unix秒> --platform-timer-receipt <已核平台回执JSON>
```

输入root为原仓库；旧Linear引用root显式区分本地`data/results/exp020/head-s7s8-20261103-v2`与云端`/root/exp020-results/head-s7s8-20261103-v2`，不复制旧权重、不改旧profile绝对路径。pipeline启动在实际结果挂载检查8GiB余量。run.sh为LF/no-BOM，使用`/root/miniconda3/bin/python`；EXIT请求有界关机，失败不吞码，晚期shutdown回执不包含在closed-run中，实际off由租赁所有者核实。

阶段顺序逐fold：旧inner引用或新`f02/f03/f05-inner`，随后`f01..05-outer`及`f01..05-dense`。每fit独立进程fresh，core与dense生产模型均为既有`head_comparison_training.LinearVideoClassifier`；seed7完整初态必须f3b66df8…d288，公共骨干0093965b…0143。保留标准best/last（inner）或selected/last（outer）及optimizer，不改变checkpoint格式。原np.array_split均摊批次24、micro8、generator100007、固定BN、Adam和FP32保持。

`stage.py`为内部入口，要求活的直接controller父PID、相同plan/out/阶段/费用截止与平台回执。不得独立调用它绕过完整编排。

每阶段写`closed-<stage>.json`，状态`fit_closed/dense_closed`或失败对应状态；files列run-relative path/size_bytes/sha256，含根目录阶段log。顶层`closed-run.json`为`run_closed/run_failed_closed`，供只回收闭合文件的运维工具使用。原旧run成功/失败状态不变；`inner-reuse.json`仅记录旧来源，不复制为新拟合。`pipeline-result.json`明确五折产物及旧inner映射。

完整评价必须150视频113P37N、143470物理窗各一次。AUC逐折使用FP32窗口概率max，保留tie，不计算跨模型pooled AUC。全FP及9负场次宏FPR下降、FN不增加且至少2负场次改善才支持本卡头部简化的视频收益；即使不支持，也完成全部预定阶段并如实归档，不自动追加试验。
