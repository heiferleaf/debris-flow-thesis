# 数据清单（唯一台账）

> 本文件只记当前有效的事实：有什么、在哪、多少、哪些不能用、怎么复算。
> 逐次决定与经过不在这里累积——实验经过在 `docs/process/experiments/` 的卡片、run log 与训练报告里，本文件只保留仍然约束现在取用的结论。
> 数量及核对时间以各节和对应正本为准；复算命令或输入路径随事实记录。

> **标签语义澄清（2026-11-02）：** 负责人确认，科考站提供的负样本可以包含细小流动；不能把neg统一解释为完全静止或物理上完全没有泥石流。沿用既有人工标签，正类起点不保证等于物理首次出现。历史指标保持原样，见[标签含义与边界核查](../docs/data/段级标签语义与边界核查.md)。

## 0. 现状速览

11月4日存储更新：按用户授权清理6份已处理来源下载件／重复ZIP，以及6份旧假数据冒烟权重，共6,717,729,417字节。当前媒体、待确认Calwood／Elk原件、正式训练输入与结果均保留。来源本地可用状态及证明见`metadata/external-videos/cleanup-20261104-v1/cleanup.json`和来源CSV新增storage字段；来源表的历史路径不再一概表示文件仍存在。

| 项 | 现状 |
| --- | --- |
| 现有视频开发数据 | 识别视频中是否已出现泥石流。原间隔150条在 `process/EXP-008-units-v2/`，长间隔128条在 `process/EXP-008-units-v3-gap05/`；EXP-016共同3秒16帧的150条监督缓存已验收，包含旧22条；入口metadata-v4，seed7／8两次配对训练及完整开发评价均完成，不称3秒最优 |
| 两套对比用的对齐子集 | `process/EXP-008-units-pool128/`：同128条、同折，但采样同时改变帧位置、跨度与合法片段集合，属于采样方案比较，不能当纯运动跨度的因果对照 |
| 新取得的外部候选 | catalog登记24个MP4：原22个（USGS正向事件来源8个、Sheaf／USU河流来源14个）加Las Lomas与Calwood未赋类候选各1个，另2个数字放大参考不计样本，共2,822,896,010字节。现有11项下载来源／24登记候选／13个USU成员分别见`metadata/external-videos/source-downloads.csv`、`catalog.csv`、`usu-members-20261030.csv`。2019 Firehose两来源ZIP均已完整归档，3段原生MOV另在`raw/web-research-20261102/review/`，不自动并入MP4登记数或按名称赋类。既有18媒体保持7P11N，Silverado两片分别保留；18＋两Flood MOV的20媒体技术记录已全闭合（89431帧），新增MOV亦完成2265帧技术检查及固定6帧有限预览。新增Las Lomas原下载在`raw/web-research-20261102/followup/`，约59.19秒；官方Public Domain，稀疏预览支持近似固定视角，事件日期未知，网页68秒与视频轨时长差异保留，已完成原生1774帧一次完整解码和PTS技术检查（正常EOF、0解码错误、无PTS回退），但不据此判定内容、标签或全片固定机位；技术产物在`process/external-videos/las-lomas-native-audit-v1/`。其登记证据在`metadata/external-videos/followup-20261102-v1/`，未改变既有18媒体选择。Calwood已去片头叠字并归档原下载，保留415秒未赋类处理片，证据在metadata/external-videos/calwood-promotion-20261103-v1/；输入像素、最终内容／字幕和有界独立性仍待核；正式测试未冻结、未读模型分数，不进训练、选轮或调阈值。 |
| 已核对的开发集读数 | EXP-009 原间隔强标签微调（150 条）：五折验证 AUC 0.9909±0.0182，TP 113 / FP 15 / FN 0 / TN 22，FPR 0.405；各折用本折验证 AUC 选最好轮次，不能当独立测试成绩。见 `docs/process/experiments/EXP-009-训练报告.md` |
| 图片侧 | EXP-005 单帧线已退役，只作历史参照 |
| 段标签来源 | 各汇总目录的 `events.csv`，即叶海峰在审计台上标的区间 |
| EXP-016配对训练归档 | EXP-016 seed7／8均完整完成。seed8 RGB：23TP／2FP／1FN／5TN；融合：24／2／0／5，严格FP下降未复现。两阶段实付1.74＋1.19＝2.93元，已关机。training-plan-v2每轮504／固定验证136，复用metadata-v4像素。未追加种子或正式CV。EXP-010/013旧草案未签，不沿用固定两轮计划。 |
| EXP-016诊断与EXP-017结果 | D置零诊断：136固定片段0翻转，CPU169.328秒／0元，结论限特定权重与片段。EXP-017：150条113P／37N，3秒16帧，场次组五折内层20轮选E后fresh重训，E为15／15／6／10／12；143470窗全部评价完成，父94项、子288项完整回收。第二次联合审计通过，首次批次均摊校验器误判与修正均留档。固定max／0.5为113TP／37FP／0FN／0TN，尚无有效负类拒绝能力；AUC逐折均值0.73404、样本SD0.28723，不是单模型或独立测试成绩。原输入、像素及父子来源见`results/exp017/audit-resume-cv-s7-20261102-v2-v2/audit.json`；原run不改。EXP-017运行已关机，训练侧320窗回放已完成（CPU132.312秒／0元），按预定规则进入EXP-020线性头小试；EXP-020后续结果见下一行；路线在`.codex/docs/research-plans/2026-11-03-完整五折后的研究决策.md`。 |
| 2026-11-01安排及外部核验 | EXP-018三组完成、1.59元，排序主门槛失败不晋级。EXP-019两支各20轮420更新、31视频32567窗、135产物回收及独立审计完成：正序24TP／2FP／0FN／5TN，打乱24／3／0／4，但打乱视频AUC更高；仅有限开发信号，不是新方法提升。该轮结束时实付1.28元、总账累计2.87元并已关机；此后EXP-017无卡阶段结算0.16元，截至2026-11-03 01:19累计已结算7.14元（EXP-017阶段4.27元），子run续跑GPU3.20元及最终CPU回收0.01元均已结算并关机；阶段7元／总20元限制不变。当前v4元数据核3秒覆盖150、7.5秒及同名义中心140，旧22中长窗可用12，像素与效果另验。外部登记23个MP4候选、正式测试0、未评分。见各实验报告、`docs/data/时间跨度覆盖核查.md`及外部边界说明。 |
| EXP-020分类头可靠性修订 | 六项新fit各20轮、共2100更新，复用两项旧inner；四个线性头logit AUC为0.9023～0.9601，预定片段门槛4/4通过，深头0/4。三对FP不变，F1 seed8减少FP同时增加5个FN；未评价新模型整视频或外部效果。210项4,827,059,819字节完整归档及派生结果复核通过，原聚合TypeError与失败回执保留，全部实例已关机。阶段实付1.37元、总账8.51／20元。详见[训练报告](../docs/process/experiments/EXP-020-训练报告.md)及`results/exp020/`中的原run、独立派生和复核记录。 |
| EXP-021局部时空能量开发小试 | 四次fresh拟合及六项inner整视频评价均完成并复核，合80轮／1400更新；三组同一批7条负视频误报，净修复0，负场次宏FPR均0.75，未晋级并停止该路线。230项产物已回收，阶段实付2.36元，总账10.87／20元，实例已关机。两角色的标签无关网格为27289／27818窗，23／24条验证列表共享5条，为42条不同视频，不是独立测试。run=energy-s7-20261103-v2；原始与派生产物在results/exp021/，实时状态见.codex/logs/2026-11-03-exp021-execution.json。 |

## 1. 目录结构

```text
data/
├── raw/                                    当前原视频及只读来源归档；替换须授权并保留映射
│   ├── dataset1/ dataset2/ dataset3/          Junction → D:\debris-flow-data\（GitHub 三库）
│   ├── dataset4-external-negatives/           实体目录（manifest.csv 记许可来源）
│   ├── video_dataset_1/ video_dataset_2/      Junction（东川站 FTP 下载）
│   ├── deflow/                                Junction（ETH Zurich DeFlow，2 段）
│   ├── webcam-negatives/                      Junction（自下载负样本）
│   ├── web-research-20261029/                 去片头后的HPCC当前原片候选
│   ├── web-research-20261030/                 其他USGS处理片、Sheaf及已解压USU候选；references单列放大参考
│   ├── web-research-20261101/positive/        Silverado同源两时间块，完整发布原片已移来源归档
│   ├── web-research-20261102/review/         两2019来源解出的3段原生MOV，未自动纳入selection18
│   ├── web-research-20261103/followup/       Calwood去片头与叠字后的415秒候选，类别未赋值
│   └── source-downloads-archive/             当前保留3份原下载MP4；已清理源件与重复ZIP见cleanup-20261104-v1
├── process/                                中间产物，可由 raw + 工具重建
│   ├── images_padded_free/                    图片补边图集
│   ├── video_frames/video1_segment/           东川 123 条：frames 74,020 张 + events 202 条区间 + index + mapping
│   ├── video_frames/video2_segment/           外部 12 个源视频：86 段目录 + extneg-scenes.csv（分段正本）
│   ├── video_frames/video3_segment/           DeFlow 2 条（v3_01/v3_02 各 120 帧）
│   ├── video_frames/video4_segment/           蒋家沟 22 条 26,099 帧（小型泥石流，正）
│   ├── video_frames/video5_segment/           蒋家沟 22 条 21,564 帧（爆发前常规水流，负）
│   ├── EXP-008-units-src256/                  256×256 软件解码中间件 150 个 mp4 + coord/ 逐帧坐标表（150 份）+ log/ + qc/ 逐帧灰度表（只有 2026-10-26 重转的 19 条，19 份；其余视频的灰度表在 EXP-008-units-v2/qc/），抽帧原料
│   ├── EXP-008-units-src256-hw/               显卡解码版的坐标表、逐帧灰度表、转码日志与清理记录（mp4 已于 2026-10-26 删除，腾出 855 MB；可用 experiments/tools/build_units_src256_hw.py 从原件重建）
│   ├── EXP-008-units-v2/                      原间隔训练帧 150 条 + 片段强标签 + qc/ 逐帧灰度表 + anchors/ 回源核对
│   ├── EXP-008-units-v3-gap05/                新间隔训练帧 128 条 + 片段强标签 + 逐帧回源核对结果 + 容量核对.csv（19 条在两套解码基座上各能放多少片段）
│   └── EXP-008-units-pool128/                 128 条对齐子集的元字段与两套片段强标签，不存帧
├── final/                                  最终训练数据（正本）
│   ├── images/labels/                         audit-dataset1..4.csv 图片判决
│   ├── images/clusters/                       pHash 簇清单 + 哈希缓存
│   └── video_windows/video2_segment/          外部场景帧集 86 段
└── 数据清单.md
```

| 层 | 职责 | 规则 |
| --- | --- | --- |
| `raw/` | 当前原视频与来源存档 | 来源归档内容只读；本批外部处理版经用户授权替换为当前原片，保留来源、处理、哈希和移动映射；大文件避免重复实体 |
| `process/` | 中间产物 | 可由 raw + 工具重建；产物附「生成命令 + 依据」 |
| `final/` | 训练正本 | 判决与划分的最终落位；改动需叶海峰确认 |

- 冻结划分清单在 `metadata/splits/` 并进Git；权重、特征缓存、指标和日志在 `results/<线>/<run_id>/`，运行产物不进Git。原run_id与历史文件保持原样。
- 训练加载经适配器，不直接感知目录：`experiments/common/temporal/adapters.py`，说明见 `docs/data/训练数据加载适配器.md`。

### EXP-016当前数据与结果入口（2026-10-31更新）

seed7两组已完成20轮训练及31视频／32567窗完整开发评价，权重、预测、运行环境、源码快照和独立复算在 `results/exp016/paired-s7-20261030-v2/`。RGB：TP24／FP7／FN0／TN0；RGB＋帧差：24／2／0／5。结果仍受单划分、同验证选轮及RGB退化限制；外部视频未进入该试验。误报联系图及源帧位于 `process/EXP-016/error-examples-v1/`，未修改任何标签或阈值。

seed8已完整完成，产物在 `results/exp016/paired-s8-20261031-v1/`：RGB为23／2／1／5、AUC0.964286；融合24／2／0／5、AUC0.976190。RGB恢复学习，融合误报严格下降未复现，不改原判据。四案例只读分析在 `process/EXP-016/seed8-error-analysis-v1/`，复用32张缓存帧，未重新解码或推理。两个种子共同评价同31条，不能合成62个独立样本。双种子报告为 `docs/process/experiments/EXP-016-双种子复核结果.md`。

后续D通道诊断产物在 `results/exp016/difference-dependency-s8-20261031-v2/`：同一seed8融合best9权重、136固定片段，正常输入先复现云端logit，再将D置零；0分类翻转，最大概率变化0.719484个百分点，费用0。结果仅限该权重与固定片段，不是另训RGB或整视频置零成绩。报告为 `docs/process/experiments/EXP-016-D通道诊断结果.md`。

`docs/process/experiments/EXP-017-训练报告.md`所述普通深头RGB场次组开发五折已完成，固定0.5下113TP／37FP／0FN／0TN；以下为已执行的数据准备与验收依据。名单在`metadata/splits/EXP-017-session-proposal-v1.json`，角色无关元数据为`metadata/EXP-017/role-free-plan-v3/`（plan SHA70394126cd15d015c4250bbea8f1c9a0c928141d92f00a03d66279a576c2c9b8）。31场次／9含负组、11机位／7含负机位，全池每轮配额452P＋188N＝640、密评网格143470窗。各外折内层选E后重新初始化拟合外层训练侧恰好E轮；不以外折选轮。新监督索引`process/EXP-017/cv-inputs-v1/verified-cache-files.csv`覆盖191812帧；真实验收在`metadata/EXP-017/cv-cache-acceptance-v1/acceptance.json`，读取6,827,772,585字节、约180.008秒，191812/191812通过。150条完整视频143470窗的像素参考亦已本地完成，其中119条新建、31条复用匹配旧参考，见`process/EXP-017/dense-pixel-reference-v1/result.json`（status=`cv_dense_pixels_verified`）；尚未运行正式CV，CPU部署准备另记，不能把输入验收或传输进展当作CV成绩或独立测试成立。

`metadata/EXP-016/metadata-v4/`保存150条源池的候选、监督段、原草案20轮轮换、固定片段验证、标签无关密评及传输引用。验收150/150通过、0失败，158142张去重监督JPEG，共5624649174字节（约5.62GB）。该版旧分配每轮576/验证148保留为资源库历史。当前共享配额已冻于 `metadata/EXP-016/training-plan-v2/`：训练356P/148N=504，验证96P/40N=136，共10180个不同16帧组合，使用已验图片子集；不改或重提图片。图片证据在 `process/EXP-016/metadata-v4/pixel-verification/20261030-183535-369061/run-summary.json`。

`process/EXP-016/dense-pixel-verification-v3/result.json`核31条/32567窗全部像素通过，最低实际源灰度标准差20.0393，本地约655.83秒；不含模型成绩。该run原plan=v1，旁存source-plan；v2所有sources一致只改监督取样，继承像素证据。云端无卡只核文件SHA和环境，不解码；训练读取已验JPEG，完整逐video摘要比对放在两组训练后的正式预测中，不能把继承称重跑。复算入口为 `experiments/runspecs/EXP-016/evaluate.py`，部署/训练命令见对应README。

149条的图片沿用v3已验缓存，v4_11用精确QC重算后在v4单独补抽1280张；每条真实路径以v4的 `frames-required.csv` 中cache_path为准，不按版本目录扫描或自行拼路径。旧v3中的被引用缓存仍是当前有效数据，不能当临时件整目录清理。候选、采样与复用约定见 `process/EXP-016/README.md`。

v1_09旧QC与当前coord版本不兼容，本次新QC在 `process/EXP-016/qc-refresh/v1_09.csv`，旧文件不改。本次监督采用已核正锚之间的保守[0,524)内核，原人工[0,526)不改；这不声明完整时钟已修复，也不用于密评删窗。

v4_11旧QC先将灰度标准差保留一位小数，本次实际像素检查在阈值20附近触发舍入差。新精确QC为 `process/EXP-016/qc-refresh/v4_11-fullprecision-20261030.csv`，未舍入值按同阈值和禁区规则重算；新required帧的最低源灰度标准差为20.03318。此差异不直接证明灰帧、解码丢帧或旧训练有错。

密评有32567个窗口，不包含在上述监督缓存齐全声明内；31条验证H264及coord/QC的93个传输引用见同一metadata目录 `dense-transfer-references.csv`。其中H264合计4018825923字节；优先核对远端相同哈希后增量上传，本地不另存一份视频。新外部候选不参与这一准备流程。

## 2. 数据集一览

| 数据集 | 来源 | 内容 | 用途 | 正 / 负 / 剔除 |
| --- | --- | --- | --- | --- |
| `dataset1` | [WZM-WH/Debris-flow-dataset](https://github.com/WZM-WH/Debris-flow-dataset) | 泥石流图片（YOLO 框已弃用） | 单帧训练 | 704 / 387 / 1,768 张 |
| `dataset2` | [chronosphere/Debris-flow-open-dataset](https://github.com/chronosphere/Debris-flow-open-dataset) | 多类型检测图（框已弃用；增强副本按簇去重） | 单帧训练 | 578 / 411 / 644 张（空标签 345 张计入负） |
| `dataset3` | [anzhi1211/Debris-Flow-Automatic-Annotation-Dataset](https://github.com/anzhi1211/Debris-Flow-Automatic-Annotation-Dataset) | 4 场事件连续帧 scene1–4，编号即时间顺序 | 单帧 + 时序两用 | 8,064 全正（叶海峰 2026-09-12 定）；各 scene 2,007 / 4,816 / 1,033 / 208 帧 |
| `dataset4` | 叶海峰联网自建（实体在仓库） | 外部单帧，难负为主 | 单帧难负 | 正 1 / 负 128 张 |
| `video_dataset_1` | 东川站 FTP（[NCDC 2025 元数据](https://www.ncdc.ac.cn/portal/metadata/d0e2aa4d-e274-458f-b790-3a0a5b4bf54a)） | 2025 年 3 场泥石流 × 7 机位，123 条 | 时序主力 | 正 46,469 / 负 2,157 / 剔 13,790 帧 |
| `video_dataset_2` | 东川站 FTP（[NCDC 2023–2024 元数据](https://www.ncdc.ac.cn/portal/metadata/33e48187-b80c-43a9-abc2-c8312358fb1e)） | 2023-08-02、2024-07-28 两次事件 × 鸡冠山上下游 | 时序 | 正 10,005 / 负 550 / 剔 685 帧 |
| `webcam-negatives` | 叶海峰自下载（B 站等，manifest.csv 逐条记 URL） | 12 个源视频，多场景拼接，切 86 段 | 时序难负 | 负 43 / 剔除 42 / 正 1 段。**这批素材不在现役 150 条内** |
| `deflow` | ETH Zurich [DeFlow](https://share.phys.ethz.ch/~gsg/DeFlow/) | 2 段正样本视频（各 240 秒、25 fps） | 时序正样本补充 | 2 条，整段正（组 `v3|deflow`） |
| `web-research-20261029/positive` | [USGS Hermits Peak-Calf Canyon](https://www.usgs.gov/media/videos/hermits-peak-calf-canyon-debris-flow)，Public Domain | 2022-08-03固定监测站事件录像；当前原片已去片头片尾，4433帧，约147.91秒。原下载202,320,899字节及原SHA保存在来源表和来源归档 | 外部正视频候选 | 已全解码核对帧数、PTS、尺寸；保留镜头水滴与模糊困难状态。事件区间和旧素材近重复仍待核，不进入训练/正式测试 |
| `web-research-20261030/negative` | [NERC EIDC River Sheaf](https://doi.org/10.5285/478db67e-bd58-48d9-a540-9e8e7e3ae68a)，OGL v3 | `27k_30fps_wide.mp4`，987,190,326字节；2704×1520、容器约29.97fps，5396帧；SHA256 `2d90a0ddca896d1b6ae6f43c6e94471719e0a2baf65a9bab418aae24574d5031` | 外部负视频候选 | 已下载，4时点抽查为定点水流；完整内容、事件及近重复未核。与正类源不同，不能以此两源即声称广泛泛化 |
| `web-research-20261030/positive` | USGS Chalk Cliffs 2016/2017、Elk 2025，Public Domain；逐条URL/SHA见统一catalog | 当前5个有效候选MP4：2016两个机位、2017原视角两个连续区间、Elk去片头／叠字后的3050帧处理片；另有2个放大参考在references，不进入有效媒体清单 | 外部正视频候选 | 2016/2017已去片头并分视图/连续区间；2017片内07-14和官方07-19日期冲突待核；Elk已完成一次新输出技术验收及首中末预览，原件已归档，真值仍未冻结；详见metadata/external-videos/elk-promotion-20261103-v1/。不同裁切、连续区间和放大不新增独立事件 |
| `web-research-20261030/negative/USU_fixed_river_cameras/` | [Zenodo 21047541](https://zenodo.org/records/21047541)，API许可CC BY 4.0 | 已解出13条MP4，BSF/First Dam/UWRL三站，视频共421,535,131字节；原ZIP在来源归档，原SHA见来源表 | 固定河流相机候选 | CRC、SHA、全解码帧数和递增PTS已核验；BSF三条有SEI截断警告并保留日志，其余10条未报该警告。全部已看中点预览，逐片标签/事件/全片质量未验收，不自动判为负或独立场次 |
| `video_dataset_3` | 叶海峰申请获得（数据方 2026-08 修复版） | 蒋家沟 2026-08-18 16:27–19:31 × D05/D12/D13/D15，22 条，小型泥石流到达塔吊断面 | 时序正样本（同沟同日配对） | events 28 条区间：pos 16 / neg 9 / cut 3 |
| `video_dataset_4` | 叶海峰申请获得（数据方 2026-09 修复版） | 蒋家沟 2026-08-18 11:33–14:01（上游 14:24 爆发前）× D02/D05/D12/D13/D15，22 条，常规水流 | 时序硬负样本 | events 整段负待审计 |
| 东川站平时录像 | 申请中（邮件已拟，收件人宋东日） | 同机位正常状态视频 | 负样本主力补充 | — |

按用户2026-10-30要求，HPCC与两条Chalk Cliffs生成的7个处理版已移动为raw下当前原片，其中2个数字放大版在references单列；未剪下载件3项及ZIP已移入 `raw/source-downloads-archive/`，未改来源字节。`process/external-videos/` 只保留QC图、坐标和日志，本批最终可用MP4数为0。`metadata/external-videos/segment-preparation-20261030.csv`记录帧区间、机位、原件/当前媒体哈希和复现命令；移动映射见同目录 `media-migration-20261030.json`。当前媒体与来源的只读校验、重建及解压命令见同目录README。文件夹名positive/negative只表示搜集来源方向，不替代逐片验收标签；全部尚未冻结为最终测试。

图片侧正负以审计判决为准（`final/images/labels/audit-datasetN.csv`，pass=正、neg=负、drop=剔、suspect=疑）：**正 9,346 / 负 798**。

## 3. 现役视频级训练数据

### 3.1 从源视频到训练帧

```text
源视频 → 256×256 中间件（150 条软件解码；显卡解码版只用于核对灰帧来源，已无训练用途）
       → 全片逐帧灰度质检（qc/<视频>.csv，每帧一行）
       → 划禁区：灰度标准差 < 20 的帧本身 + 前后各 16 帧 + PTS 倒步邻域 + 时间空洞前后 1 秒
       → 只在禁区外安排 48 个片段，每片段 16 帧
```

灰度标准差 = 一张图里像素亮度的离散程度。软件解码丢参考帧时，会一直吐整片均匀灰直到下一个关键帧，所以灰帧成段出现；它的灰度标准差接近 0，整段都会被划进禁区。**训练用的 jpg 里没有灰帧**：两套逐张复算，低于 20 的都是 0 张（见 3.2 表）。低灰度帧只存在于中间件 mp4 内，从未生成 jpg。

### 3.2 两套采样间隔

| 项 | 原间隔（v2） | 新间隔（v3-gap05） |
| --- | --- | --- |
| 一组 16 帧怎么取 | 连续 16 帧 | 源 20 帧/秒下每 10 帧取一张 |
| 一组跨多长 | 逐片段查PTS；现行禁区后有效6859个为0.60–2.32秒，中位0.75秒、75分位1.50秒（2026-10-29复算） | 目标7.5秒；20fps下占151帧窗口，实际以逐条PTS为准 |
| 入池视频 | 150 条（正 113 / 负 37） | 128 条（正 95 / 负 33） |
| 张数 | 115,200 | 98,304 |
| jpg 灰度标准差 < 20 | **0 张**（最低 20.04、中位 37.31） | **0 张**（最低 20.04、中位 36.68） |
| 片段强标签严格版可用 | 6,979 / 7,200 = 96.9%（正 5,022、负 1,957、不可用 221） | 6,002 / 6,144 = 97.7%（正 4,225、负 1,777、不可用 142） |
| 片段强标签宽松版可用 | 7,115 / 7,200 = 98.8% | 6,121 / 6,144 = 99.6% |

原间隔的 150 条构成：东川 106 + 蒋家沟视频 4 共 22 + 蒋家沟视频 5 共 22。新间隔的 128 条同为东川 84 + 22 + 22。

原间隔一栏的「张数 115,200」指清单内 150 条 × 768 张。`EXP-008-units-v2/frames/` 目录下实际有 151 个视频目录、115,968 张 jpg 加 151 个 `slots.txt`：多出的 768 张是 v1_34 的残留帧目录（这条已因中间件时间轴错误排除，不在任何清单里，占 24 MB，待叶海峰决定是否删除）。训练只读清单，不受影响。

**「这条视频在不在池里」与「它的 48 个片段是不是都参与打分」是两件事，不要混。** 每条入池视频都安排满 48 个片段位置（`EXP-008-units-v2/unit-manifest.csv` 的 `n_clips` 列 150 行全是 48）；某个位置落在人工标的剪辑区间里时，那个片段记为不可用、不参与打分。按严格标签数，原间隔 150 条里有 92 条实际参与打分的片段少于 48：正样本 113 条里 74 条（最少 19、中位 47），负样本 37 条里 18 条（最少 47、中位 48）。这 92 条都在池子里，EXP-009 的训练和评估都用了它们。新间隔排除的那 22 条是另一回事：换成 7.5 秒一组后它们凑不满 48 个合法位置，而在原间隔下它们参与打分的片段是 47 到 48 个，属于池子里最完整的一批。

```bash
# 92 条那四个数，以及这 22 条与原间隔池的关系（第一节与第五节）
.venv/Scripts/python.exe experiments/tools/oneoff/exp009-fp-analysis/short_video_bias.py
```

**新间隔排除的 22 条，凑不满 48 组具体卡在哪类禁区（2026-10-27 逐条复算）。** 禁区分两族：剪切剔除区＝人工标的剪辑区间前后各外扩 1 秒；画质族＝低灰度帧（前后各 16 帧）、PTS 倒步邻域、时间空洞（前后各 1 秒）。归因方法是：对每条视频，把某一族禁区单独留下、另一族当作不存在，看合法片段能不能到 48。结果：3 条由剪切剔除区单独卡死（v1_19、v1_55、v1_70——只留画质族时分别有 76、187、192 个合法位置）；18 条由画质族单独卡死（v1_110–v1_114、v1_117、v1_118、v1_44–v1_49、v1_73、v1_76–v1_79——只留剪切区时分别有 48 到 70 个合法位置）；1 条两族各卡掉一截、合起来才不够（v1_39：只留剪切区 23、只留画质族 23）。**没有任何一条是视频本身太短**：把 22 条的禁区全部拿掉，每条都能安排出至少 69 个片段位置。此前记的「0.5 秒容量不足」据此更正——缺的是过完禁区后的合法位置，不是原始位置。

```bash
# 22 条逐条归因（只读，不写数据文件）
.venv/Scripts/python.exe experiments/tools/oneoff/exp014-excluded-22/attribute_causes.py
```

复算命令：

```bash
# 抽帧（以新间隔为例）
.venv/Scripts/python.exe experiments/common/video/build_units_stride.py
.venv/Scripts/python.exe experiments/common/video/build_clip_labels_v3.py
# 逐张回源核对
.venv/Scripts/python.exe experiments/common/video/verify_units_v3.py --root EXP-008-units-v3-gap05
# 训练帧灰度复算
.venv/Scripts/python.exe experiments/tools/scan_frame_gray.py --frames-root <帧根> --out <csv>
# 某批视频在某个采样间隔下能放多少片段（不解码 mp4，只读坐标表与灰度表）
# 2026-10-26 重转的 19 条，灰度表在 src256/qc/（19 份）；其余视频用 v2/qc/（153 份）
.venv/Scripts/python.exe experiments/tools/check_clip_capacity.py --base 软解 \
    --src data/process/EXP-008-units-src256 --qc-dir data/process/EXP-008-units-src256/qc \
    --gap 0.5 --units v1_19,v1_39 --out data/process/EXP-008-units-v3-gap05/容量核对.csv
# 同一张表追加显卡基座 19 条，再追加只有软解数据的 3 条（--append），最终 41 行
.venv/Scripts/python.exe experiments/tools/check_clip_capacity.py --base 显卡 \
    --src data/process/EXP-008-units-src256-hw --qc-dir data/process/EXP-008-units-src256-hw/qc \
    --gap 0.5 --units v1_19,v1_39 --out data/process/EXP-008-units-v3-gap05/容量核对.csv --append
.venv/Scripts/python.exe experiments/tools/check_clip_capacity.py --base 软解 \
    --src data/process/EXP-008-units-src256 --qc-dir data/process/EXP-008-units-v2/qc \
    --gap 0.5 --units v1_44,v1_55,v1_70 --out data/process/EXP-008-units-v3-gap05/容量核对.csv --append
# 中间件逐帧灰度表（中间件重转过就必须重算，否则禁区与实际像素对不上）
.venv/Scripts/python.exe experiments/tools/scan_middleware_gray.py --units v1_19 --src <中间件目录> --out <中间件目录>/qc
# 重转前后的灰度表逐帧比对（按帧号硬对齐，行数不等即报错，不截断）
.venv/Scripts/python.exe experiments/tools/verify_rebuild_qc.py \
    --old-qc data/process/EXP-008-units-v2/qc --new-qc data/process/EXP-008-units-src256/qc \
    --out data/process/EXP-008-units-src256/qc-rebuild-verify.csv
```

### 3.3 128 条对齐子集（EXP-014 用）

`process/EXP-008-units-pool128/` 三份文件：`meta-128.npz`（视频编号、标签、折归属）、`clip-labels-v2-128.npz`（原间隔严格版可用 5,930/6,144 = 96.5%）、`clip-labels-v3-128.npz`（新间隔 6,002/6,144 = 97.7%）。折划分正本 `data/metadata/splits/EXP-008-units-folds-128.json`，逐折 29/30/24/15/30 条（正 22/23/14/13/23，负 7/7/10/2/7）。

两套的可用率差 1.2 个百分点，已过「不低于 80%」这条闸门。生成命令：

```bash
.venv/Scripts/python.exe experiments/common/video/make_pool_npz.py   # 写盘前与 v2 清单逐条核对折归属与标签
```

### 3.4 排除名单

| 视频 | 原因 | 记录在哪 |
| --- | --- | --- |
| v1_19、v1_39、v1_45、v1_46、v1_47、v1_48、v1_49、v1_73、v1_76、v1_77、v1_78、v1_79、v1_110、v1_111、v1_112、v1_113、v1_114、v1_117、v1_118（19 条） | 新间隔下合法片段 0–23 个（v1_118 23、v1_39 21、v1_19 11、v1_117 2、v1_78 1，其余 14 条 0 个）。一个片段要 151 个连续帧（v1_19、v1_39 是 76 个）整块不碰禁区。软件解码基座上低灰度帧占 25.5%–53.2%、禁区占 42.5%–92.2%，干净帧全拿来排也最多 40.2 个，不够 48；显卡解码基座同样 0/19 条达到 48——灰帧基本归零，但丢帧留下的时间空洞前后各 1 秒不许落片段，禁区占 25.9%–90.6%，干净区间碎成最长 99–1,853 帧的若干段，一段片段要占连续 136 帧。原间隔下这 19 条照常可用 | `EXP-008-units-v3-gap05/unit-failures.csv`；逐条容量与禁区拆解在 `data/process/EXP-008-units-v3-gap05/容量核对.csv` |
| v1_44（合法片段 28）、v1_55（43）、v1_70（24） | 新间隔下每组要连续 151 帧，这 3 条凑不满 48 组 | 同上 |
| v1_34 | 中间件时间轴错误（坐标表末端比源视频长 144 秒），已入池的 48 个片段里 42 个落进剔除区 | `EXP-008-units-v2/unit-failures.csv` |
| v1_74 | 剔除段占全片 72%，安全边距下合法片段 47 个，不足 48 | 同上 |
| 15 条东川视频 | 整条都是剔除段，本就不入池 | `EXP-008-units-v2/anchors/skipped.csv` |
| v5_08 | 源视频顺序解码只出 2 帧，判定解码器不兼容不可修复 | `mapping.csv` 标 `excluded=1` |

东川 123 条的去向：15 条整条剔除 → 108 条有中间件 → 减 v1_34、v1_74 → 106 条入原间隔池。

### 3.5 折划分正本

| 文件 | 用于 |
| --- | --- |
| `EXP-008-units-folds.json` | 原间隔 150 条这批，按组内正负视频数分层，seed 42 |
| `EXP-008-units-iid-folds.json` | 同场次混分参照，只作成绩上限，不是泛化成绩 |
| `EXP-008-units-folds-128.json` | 两套对比的公共 128 条 |
| `EXP-006-session-folds.json`、`EXP-006-scene-folds.json`、`EXP-006-scene-logo-folds.json` | EXP-006 时序线，保留作数据充分性检查 |

## 4. 历史两条线

| 线 | 做法与规模 | 状态 |
| --- | --- | --- |
| 单帧线 EXP-005 | 三库 12,556 张 → pHash 聚成 1,436 簇 → 整簇同侧切 70/15/15 → 每簇上限抽样 → 1,729 帧（train 1,271 / val 249 / test 209）。dataset1/2 作者自带的划分有 229 个簇跨两侧，已全部弃用 | 已退役，只作历史参照 |
| 时序线 EXP-006 | 按 events 区间切段（pos / neg / cut）→ 按「机位 × 场次」建组、组级分层 5 折 → 段内滑窗 16 帧、不跨段 → 5 折 CV。最后一次重建 224 段，特征缓存 12,027 帧 × 768 维 | 训练已停，段标签与「同场次不拆」的划分口径仍是所有实验的硬约束；成绩见 `docs/process/experiments/EXP-006-runlog-20260924.md` |

`splits/EXP-006-*` 三份折文件仍在，用于数据充分性检查（留出一个数据源，回答「缺这个源还练不练得成」，不作泛化成绩）。

## 5. 命名与标注口径

| 项 | 口径 |
| --- | --- |
| video_key | `v1_<两位序号>`（v1_01..v1_123）；v4_、v5_、v3_ 同理续号；帧文件名 = `<video_key>_<原帧名>`，可反解回原视频 |
| mapping.csv | 列 = `video_key, video_id, site, site_code, date, excluded`；`site` 为视频右下角地址水印名，`site_code` 存原 D 编号（仅 v4/v5 有） |
| 场次组 | 同一天、同一机位的一批视频算一组，划分时整组同侧；v2 外部素材每段一组，v3 DeFlow 每条一组 |
| 区间 → 帧标签 | 正区间内为正、剔除区间既不作正也不作负、其余为负 |
| 显式负 | 人工确认整段未达到正类判据时可整段标负，状态从「未标」变「已标」；允许细小流动，不因有运动自动改正类。原文“无泥石流”不能当作所有负类的物理定义（2026-11-02澄清，原events未改） |
| v1_35 | 门前沟中游 6:25；0–8 秒正、8–19 秒剔除、19–747 秒正，已归队参训 |

## 6. 已知数据缺陷

| 现象 | 影响 | 现在怎么处理 |
| --- | --- | --- |
| 软件解码中间件里有成片灰帧：152 条 3,491,095 帧中 117,153 帧（3.36%）灰度标准差 < 20，涉及 52 条 | 若直接抽帧会把灰图喂给模型 | 抽帧前逐帧质检、划禁区；两套 jpg 复算都是 0 张 |
| v1_45 前 5,000 多帧、v1_109 尾段是源自带的灰色画面 | 中间件里保留（不丢帧），抽帧时槽位避开 | 同禁区机制 |
| 96 条视频软件顺序解码比源少帧，合计 3,869 帧 | 时间轴可能偏移 | 一切秒↔帧换算只查 `coord/` 坐标表，禁用元数据帧率算 |
| v1_09 有 1 处 0.20 秒 PTS 倒步 | 该处时间不单调 | 条目保留、`pts_monotonic=0` 单独标记，倒步邻域禁入 |
| 19 条视频在新间隔下凑不满 48 个片段（两套解码基座都是 0/19） | 新间隔池只含 128 条 | 见 3.4。这 19 条的软件解码中间件已于 2026-10-26 从原件重转并逐帧核对一致，原间隔下照常可用，池子名单不变 |
| 段清单正本 `process/label-review/pilot-segment-manifest.csv` 已不在盘上，该目录为空 | EXP-006 那版段清单无法原地复现 | 副本在 `data/results/temporal/EXP-006-pilot/segment-manifest-20260917-2204.csv`；可用 `build_segment_manifest.py` 从 events.csv 重建 |
| `EXP-008-units-v2/qc/u.csv` 是残留下载件（780 行，不对应任何视频） | 若按目录总数统计会把帧数算多 780 | 统计时排除；文件未删，等确认后清理 |
| `extneg-scenes.csv` 有 3 段备注写「整段负样本」但类型不是负（1 段记正、2 段记剔除） | 影响外部素材的段数清点 | 未改动，待叶海峰核对；这批不在现役 150 条内 |
| 外部素材 86 段、DeFlow 2 条不在现役 150 条内 | 引用「全部数据」时要分清 | 现役池只含东川 106 + 蒋家沟 44 |

## 7. 判决样本的簇覆盖（2026-09-14 实测）

| | 样本数 | 覆盖簇数 | 最大簇 |
| --- | --- | --- | --- |
| 正 | 9,346 | 336 簇（dataset3 占约 10 簇，最大即 d3 巨簇） | 4,421 张 |
| 负 | 798 | 303 簇（平均 2.6 张/簇） | 81 张 |
| 正负混合簇 | — | **0 个** | — |

## 8. 待办

1. `qc/u.csv` 残留文件待确认后删除。
2. `extneg-scenes.csv` 里 3 段备注与类型不一致，待叶海峰核对。
3. 负样本扩充：现役负样本 33（新间隔）/ 37（原间隔）条，主要方向 = 泥石流发生前与退去后时段（同机位难负）+ 东川站平时录像（申请中）。
4. 图片侧正式划分清单：只在恢复单帧线时才需要，方案要先出、叶海峰确认后执行。
