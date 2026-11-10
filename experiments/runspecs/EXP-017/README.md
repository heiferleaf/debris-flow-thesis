# EXP-017 执行入口

只执行已冻结v3的普通RGB场次分组开发CV，不修改划分或配方。当前代码实现不代表像素准备完成或付费启动放行。独立缓存验收必须为`cv_supervised_cache_ready`，150视频的本地密评像素参考必须为`cv_dense_pixels_verified`。

`cv_execution.py`严格绑定v3计划SHA、实际191812监督输入验收及RGB112指纹。每折inner从公开K400/seed7训练20轮，固定内层片段BCE最小选E、同分最早；outer丢弃inner权重，从相同公开状态和新Adam训练原20轮计划前E轮。两次初始化在Adam创建前核C184，outer同时核inner实测初始SHA。outer不接收验证张量或选best。只在outer固定E完成后，对外折完整视频评价一次。

`cv_dense_evaluation.py`复用规范3秒/1.5秒网格及单流解码，在视频结束时核对本地预先取得的RGB112摘要。所有中途分数暂定，摘要失败或覆盖不全即非零退出。监督/密评均沿JPEG95→BGR解码→RGB→LINEAR112→原mean/std，零D；不在付费GPU增加另一次全视频预扫描。有限但不佳的成绩照常保存，不删折、换seed、扫阈值或自动续租。

## 静态准备

从仓库根执行。以下路径是命令模板，实际输入尚待producer与Sol验收后填入。

```text
python experiments/runspecs/EXP-017/prepare_transfer.py --plan data/metadata/EXP-017/role-free-plan-v3 --acceptance <acceptance.json> --dense-reference <dense-pixel-result.json> --out data/metadata/EXP-017/deployment-v1
python experiments/runspecs/EXP-017/pipeline.py --stage check --plan data/metadata/EXP-017/role-free-plan-v3 --acceptance <acceptance.json> --dense-reference <dense-pixel-result.json> --out data/process/EXP-017/cloud-preflight-v1 --ffmpeg <ffmpeg> --transfer-manifest <deployment/files.csv> --prior-cpu-seconds <actual_cumulative_no_GPU_seconds>
```

部署清单引用源媒体、必要JPEG和元数据，不复制大文件到源码快照。源媒体大小使用本地producer实际hash闭合的`media_refs`；不是从v3猜大小。远端同SHA文件可增量跳过，缺失项先传完；静态preflight核全部清单文件及至少12GiB剩余产物空间。无卡上传/回收累计不超过7200秒，root须实际设置剩余时长的平台定时并核实关机；`--prior-cpu-seconds`不是延长授权。

## 有界执行

完整GPU入口为`bash experiments/runspecs/EXP-017/run.sh`，附上述plan/acceptance/dense-reference/ffmpeg/transfer-manifest参数，`--out data/results/exp017/<unique_run_id>`，以及：

独立`train.py`生产CLI已禁用并非零退出；`train_stage`仅作为完整pipeline内部API复用。静态预检从已验context独立重建必需部署集合，缺少任何媒体或监督JPEG行、SHA或大小不符均拒绝，不能用截短清单缩小核验范围。

```text
--preflight <preflight.json>
--gpu-start-unix <actual_GPU_session_start>
--platform-deadline-unix <actually_set_platform_shutdown_time>
--platform-shutdown-receipt <actual_platform_receipt_file>
--prior-gpu-seconds <earlier_EXP017_GPU_seconds>
--prior-cpu-seconds <all_EXP017_no_GPU_seconds_so_far>
--prior-stage-spend-yuan <stage_spend_before_current_GPU_session>
--prior-total-spend-yuan <goal_spend_before_current_GPU_session>
--hourly-price 1.48
```

GPU累计≤14400秒、阶段≤7元、总目标≤20元（已完成阶段至少2.87元），为剩余无卡与关机留余量。真正的启动时刻包含预启动等待，不以脚本时间清零。平台定时回执真实性由root现场确认，非空文件不等于设置成功。硬截止自身固定请求`/bin/bash /usr/bin/shutdown`、隔离日志管道、15秒限时，任何证据写失败仍尝试关机，最终退出124；shell退出与平台定时另作保护。完成或失败后须核实际实例状态和账单。

首个预登记fold1完成后，按实际耗时/更新数及密评窗数预测余时；未知后续outer E按20轮上界，预留30%和300秒收尾，不读取成绩决定继续。不能装作额外探针接续权重，不能为了预算缩减评价或换折。完整产物保留10次拟合的输入/源码/环境、逐轮损失/内层logit、best或fixed-E selected与last权重、全部外折窗分数和视频分数。

```text
python experiments/runspecs/EXP-017/audit.py --run data/results/exp017/<run_id> --plan data/metadata/EXP-017/role-free-plan-v3 --out data/results/exp017/<run_id>/audit-v1
```

只有完整5折才汇总113P/37N各一次的TP/FP/FN/TN、Recall/FPR、逐负组结果。AUC仅逐折、折均值和样本标准差，跨折分数不属于同一个模型，不报混合AUC。该数据池被历史研究反复查看，不能称独立测试、未见机位泛化或模型创新；费用、关闭状态与科学引用还须独立审查。本入口不运行最终150条模型或外部候选评分。
