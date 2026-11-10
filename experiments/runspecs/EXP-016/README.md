# EXP-016：RGB与RGB＋帧差的配对开发试验

实验卡在 `docs/process/experiments/EXP-016.md`，seed7两组各20轮／420更新、31视频／32567窗完整评价与独立结果复算已完成。成功run为 `data/results/exp016/paired-s7-20261030-v2/`，训练约19.3分钟、完整评价26.6分钟、成功管线46.8分钟；实际总费用1.74元（含旧失败测速及所有无卡），已关机。见开发划分结果；该结果不是独立测试或正式CV。运行源码原字节在run内source-snapshot，后续状态更新不冒充原源码。

seed8于2026-10-31完整完成：run为`paired-s8-20261031-v1`，两组均20轮／420更新，31视频／32567窗完整评价及独立审计通过。RGB23／2／1／5、融合24／2／0／5，strict FP下降条件false；未自动追加训练。管线约47分钟，实付1.19元，已关机。比较入口`compare_seeds.py`，四案例入口`inspect_seed_comparison.py`，最新报告`EXP-016-双种子复核结果.md`。

本阶段获批新增费用上限4元，GPU≤2小时、单价≤1.48元/小时，无卡总计≤30分钟。run.py及审计入口均显式使用`--seed 8`，训练核心不变；静态预检的seed也必须一致。下面的预算命令保留供历史复现，不授权再次启动。

seed8实际启动参数：静态检查为`--seed 8 --budget-yuan 4 --seconds-limit 7200`，无卡平台限15分钟、外层`timeout 12m`；正式管线为`--seed 8 --hourly-price 1.48 --budget-yuan 3.8 --seconds-limit 6900`，GPU平台最迟11:20关机，从开机起不足2小时。另一次无卡取回的保护时间为10:27，实际10:14即关机，两次合计少于30分钟。独立审计加`--seed 8`；完整argv在run/launch.json，权重与源码分别保留，不能覆盖seed7。

当前取样为 `data/metadata/EXP-016/training-plan-v2/`：每轮356P/148N=504、固定验证96P/40N=136，按视频×类别共享最多4窗，位置覆盖整个已验代表库。旧v4的576/148及plan-v1保留作过程证据，图片不重提。

- 开发划分：`data/metadata/splits/EXP-016-development-holdout.json`；父数据/折指纹和ID在JSON中。
- 已验输入：`data/metadata/EXP-016/metadata-v4/`；监督资源库150条、158142张、5624649174字节，当前计划使用子集。实际文件路径逐行读取cache_path：149条复用v3缓存，不能自行拼成v4/frames路径。
- 数据：150条既有源池，首轮119训练/31开发验证，统一3秒16帧；新外部测试候选不进入。
- 两组：RGB输入[R,0]与融合[R,D]，同一六通道R(2+1)D初始化及训练预算。
- 优化：Adam骨干1e-4/头1e-3，BN统计固定、仿射训练，Dropout0.5，无增强；有效批次24、micro≤8，逐块反传且每批只更新一次，FP32禁用AMP/TF32。像素缓存及复用worker已接入，CUDA实测约29秒／轮；未做同功能优化前后对照，不报告提速倍数。
- 监督片段验证集按固定BCE选轮；选定权重仅做一次标签无关整视频密评。
- 共用实现位于 `experiments/common/video/`；输入在 `data/process/`，冻结清单在 `data/metadata/`，权重、指标及日志在 `data/results/`。

本地检查命令（仓库根目录运行，不租赁实例）：

```text
.venv/Scripts/python.exe experiments/runspecs/EXP-016/train.py --check-only
.venv/Scripts/python.exe experiments/runspecs/EXP-016/train.py --smoke --device cpu --workers 0 --cache-mib 16 --microbatch 1 --mode rgb_zero --out data/results/exp016/<unique-smoke-id>
```

真实CPU冒烟证据为 `data/results/exp016/smoke-local-rgb-s7-v2/` 与 `smoke-local-diff-s7-v2/`。只检查执行/梯度，不当模型成绩。

密评像素证据为 `data/process/EXP-016/dense-pixel-verification-v3/result.json`：31条、32567窗。旁存原v1 source-plan；v2所有sources一致，仅监督取样变化，故继承而非重跑。无卡预检仅核文件SHA、输入覆盖及环境，不解码；训练逐次核JPEG及其RGB112摘要，完整视频解码/摘要核对放到两组训练后的一次共享评价。

以下为seed7历史启动说明，不代表新的付费授权。上传代码、清单、选定图片及31条H264/coord/QC后，在原AutoDL无卡实例检查；同SHA文件可复用。`<FFMPEG>`必须是实际二进制路径，核对`/usr/bin/shutdown`可执行；历史上传上限为平台5小时定时关机。无卡检查不用高并行：

```bash
bash experiments/runspecs/EXP-016/check_cloud.sh \
  --transfer-manifest <本次远端完整上传清单CSV> \
  --pixel-reference data/process/EXP-016/dense-pixel-verification-v3/result.json \
  --ffmpeg <FFMPEG> --out data/process/EXP-016/<unique-cloud-check>
```

seed7历史无卡上传与检查合计最多5小时，成功或失败都关机。静态检查成功产生preflight.json与static-inputs.json，状态为cloud_static_preflight_passed，解码帧数0，不伪装完整云端像素通过。GPU阶段必须保持同份环境/代码/数据/FFmpeg，原20元范围的命令如下：

```bash
bash experiments/runspecs/EXP-016/run_cloud.sh \
  --transfer-manifest <与静态预检同份远端完整上传清单CSV> \
  --pixel-reference data/process/EXP-016/dense-pixel-verification-v3/result.json \
  --preflight-result data/process/EXP-016/<unique-cloud-check>/preflight.json \
  --ffmpeg <FFMPEG> --hourly-price 1.48 --budget-yuan 20 --seconds-limit 43200 \
  --out data/results/exp016/<unique-paired-run>
```

先做约30秒的1轮RGB训练预算探针，不遍历完整视频；只按正确性和耗时放行。通过后重新初始化两组各20轮，再共享一次解码评价各自best并完整核对31条/32567窗及像素摘要。预算中评价暂预留1小时（估计），总时间与费用硬限继续执行。不读取测试集、不自动加第二种子或五折。评价失败保留训练模型/日志，记评价失败并关机，不自动重训。GPU12小时硬限另配平台定时关机。20元含已发生及后续无卡费用，时间上限不是耗时预测。

公共层9项、取样4项、密评6项、预算保护2项CPU检查通过；另有真实网络冒烟和全部密评像素证据。每轮记录实际片段顺序、梯度、更新次数、学习率、P/N及分类别验证损失；保存best/last、profile、代码/卡/协议指纹。评分拒绝smoke/未完成/last或SHA不符模型。旧评分脚本固定CPU，不照搬旧耗时估GPU预算。第10条双协议CV尚未完成，本轮不宣称全部完成数据协议或得到独立测试成绩。

部署预处理无需正负区间：连续固定机位视频用3秒/1.5秒网格生成可变数量的16帧输入，不沿用训练4窗配额。训练与离线密评已接入六通道建模/像素处理；工作台的新模型接线另阶段完成，旧48/3通道入口不能只换权重。开发密评使用已知人工cut，不能称为完全无人工处理的部署能力。
