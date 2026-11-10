# 泥石流视频识别：跨场景三项MIL实验

本实验研究：仅用整条视频的正负标签，能否在未参与训练的拍摄地点识别泥石流视频。方法采用视频级多实例学习（MIL）：将一条视频看作若干片段的集合，为片段评分，再聚合成视频判定。

**150条视频、跨场景五折、冻结R(2+1)D-18、四层分类网络、论文三项损失；五折平均AUC为0.9097。** 本报告对应已完成运行 `mil-v2-150-strict-scene`，提供五折权重、原始预测及复现步骤。结果属于开发实验，默认阈值表现仍较差。

## 1. 数据及监督

| 来源 | 视频数 | 正类 | 负类 |
| --- | ---: | ---: | ---: |
| 东川科考站FTP视频 | 106 | 97 | 9 |
| 百度网盘视频组v4 | 22 | 16 | 6 |
| 百度网盘视频组v5 | 22 | 0 | 22 |
| **合计** | **150** | **113** | **37** |

上述数量来自本实验的冻结标签，不用后来修订的N/P/T标签替换。正视频包含被标记为泥石流的时段，负视频没有正时段。训练的监督目标是视频标签；片段标注用于界定可使用的输入范围。

### 视频怎样成为模型输入

1. 原视频先制作256 × 256的解码中间件，记录帧号、时间戳和逐帧质量统计。
2. 排除已标记的cut、低灰度方差帧及其安全边距，在视频不同位置选择48个片段，每个片段包含16张相邻解码帧。
3. 用人工区间筛选片段：16帧须完整位于同一个正区间或负区间内。跨越正负边界或落在未覆盖范围的片段不参与训练和评分。这是本实验的 **strict规则**。
4. 最终保留 **6,979 / 7,200个片段**；每条视频实际使用19～48个。正视频中的合法负片段也保留，片段标签不直接作为分类损失的目标。
5. 每组16帧转换为RGB并缩放到112 × 112，经冻结骨干提取512维特征。所有特征只计算一次，分类网络训练时复用。

这里的16帧按解码帧号相邻，**并不保证跨视频具有相同秒数，也不是固定3秒输入实验**。已知时间连续性限制及核查证据见第8节。

## 2. 跨场景五折

按本实验冻结的拍摄地点分组，同一地点的全部日期和视频留在同一折。每次用四折训练、一折验证，得到五个模型；每条视频恰好作为一次留出样本。

| 折 | 训练正类 | 训练负类 | 验证正类 | 验证负类 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 85 | 33 | 28 | 4 |
| 2 | 92 | 27 | 21 | 10 |
| 3 | 90 | 31 | 23 | 6 |
| 4 | 88 | 33 | 25 | 4 |
| 5 | 97 | 24 | 16 | 13 |

划分共有11个地点组，完整成员冻结在[跨场景划分文件](data/metadata/splits/EXP-008-units-scene-folds.json)。它评价旧地点分组下的跨场景表现，不同时保证跨事件；同一天不同地点可能记录同一次事件。本次整理按用户要求保留原场景映射，不重新划分或训练。

## 3. 模型架构

```mermaid
flowchart LR
    A["视频中的16帧片段"] --> B["冻结R(2+1)D-18<br/>512维时空特征"]
    B --> C["共享分类网络<br/>512→128→64→32→1"]
    C --> D["合法片段分数取最大值<br/>得到视频分数"]
```

| 模块 | 结构与设置 | 是否训练 |
| --- | --- | --- |
| 视频骨干 | Kinetics400预训练R(2+1)D-18，移除原分类层 | 冻结 |
| 片段分类网络 | 512 → 128 → 64 → 32 → 1 | 更新参数 |
| 隐藏层 | ReLU和50% Dropout | — |
| 输出层 | Sigmoid，片段分数为0～1 | — |
| 视频评分 | 参与评分的片段分数取最大值 | 固定 |
| 默认分类阈值 | 分数≥0.5判为泥石流 | 固定 |

片段内的时空特征由骨干提取，片段之间没有新增的时序层。

## 4. 三项损失

每次配对一个正视频和一个负视频。两者的48个片段分数分别记为$p_k$和$n_k$，合法片段集合记为$V_p$、$V_n$。集合来自第1节的区间筛选；两者都至少包含一个片段。

包内峰谷差为：

$$
R_p=\max_{k\in V_p}p_k-\min_{k\in V_p}p_k,\qquad
R_n=\max_{k\in V_n}n_k-\min_{k\in V_n}n_k.
$$

**峰谷差约束**鼓励正视频的分数变化范围大于负视频：

$$
L_1=\max(0,\;1-R_p+R_n).
$$

**平滑约束**只计算正视频中相邻且同时合法的片段对：

$$
L_2=\sum_{\substack{k=1,\ldots,47\\k,k+1\in V_p}}(p_{k+1}-p_k)^2.
$$

**稀疏约束**限制正视频合法片段的分数总量：

$$
L_3=\sum_{k\in V_p}p_k.
$$

总损失对一个批次中的视频对取平均：

$$
\boxed{L=L_1+8\times10^{-5}L_2+8\times10^{-5}L_3.}
$$

后两项只作用于正视频；第一项比较包内峰谷差，不是正负视频最高分的直接差值。这里没有逐片段BCE分类损失。

## 5. 训练设置

| 设置 | 取值 |
| --- | --- |
| 优化器 | Adam，β=(0.9, 0.999)，ε=10⁻⁸，weight decay=0 |
| 学习率 | 0.001 |
| 每折预算 | 20轮 |
| 随机种子 | 7，每折重新设置 |
| 批次 | 每批最多8个正视频和8个负视频 |
| 样本组织 | 每轮遍历正视频，循环负视频补齐配对；末批两侧数量一致 |
| 模型选择 | 验证AUC最高的一轮；并列保留先出现者 |
| 推理 | 关闭Dropout，合法片段分数取最大值 |

## 6. 实验结果

### 各折AUC

| 折 | 验证视频数 | AUC | 选中轮次 |
| --- | ---: | ---: | ---: |
| 1 | 32 | 1.0000 | 4 |
| 2 | 31 | 1.0000 | 7 |
| 3 | 29 | 0.9348 | 14 |
| 4 | 29 | 0.7000 | 9 |
| 5 | 29 | 0.9135 | 18 |
| **逐折均值** | — | **0.9097** | — |

折间标准差为0.1104。轮次从1开始；原结果文件从0开始。AUC衡量正视频分数总体是否高于负视频，不依赖固定阈值。

### 默认阈值0.5的诊断结果

以下由五折原始预测合并统计，用于说明工作点问题，不把高AUC等同于可用的告警效果。

| 真实类别 | 预测泥石流 | 预测非泥石流 |
| --- | ---: | ---: |
| **泥石流，113条** | 55（TP） | 58（FN） |
| **非泥石流，37条** | 15（FP） | 22（TN） |

| 指标 | 数值 |
| --- | ---: |
| Precision | 0.7857 |
| Recall | 0.4867 |
| F1 | 0.6011 |
| Accuracy | 0.5133 |
| FPR | 0.4054 |

## 7. 结果意味着什么

在这批旧标签和地点分组下，方法具有一定的视频分数排序能力，但折间差异明显，默认阈值漏掉58条正视频。第4折AUC为0.7000，阈值0.5下Recall为0，说明排序能力和分类工作点必须分别考察，尚不能交付为可靠告警模型。

论文第一项损失约束的是分数变化范围；它并不直接要求正视频最高分高于负视频最高分。因此，较高排序AUC不能证明默认阈值已得到有效校准，也不能证明模型已经利用了物理运动。阈值问题、损失目标与最终判据的关系是值得继续研究的具体问题。

本实验在同一验证折上选轮和报告成绩，存在模型选择带来的乐观偏差；平均五折最高AUC不会消除这种偏差。它可作为**历史开发实验和方法重实现证据**，不能当作独立测试成绩，也不能直接证明新标签、固定时间跨度或新方法的有效性。

## 8. 输入质量与复算证据

| 证据 | 实际核查范围 |
| --- | --- |
| 输入完整性 | 150条各有48 × 16张图，115,200张全部可读取，尺寸均为256 × 256 |
| 灰帧筛查 | 全部所选图像及历史QC均未低于灰度标准差20的阈值；所选帧未碰到低方差禁区的16帧外扩范围 |
| 缓存身份 | 视频ID、标签、场次组和旧折字段与150条清单对齐，特征均为有限值 |
| 特征抽查 | 随机1组16帧重新计算512维特征，与缓存逐元素相同，最大误差0 |
| 模型复算 | 本运行五个分类网络从原特征得到的逐视频分数和逐折指标与保存记录一致 |

还发现一项实际限制：历史连续取16帧的规则未充分排除向前时间跳跃。6个相关片段中，1-32的5个已经被strict筛除；1-109的第46个片段仍参与本实验，其相邻帧时间跳跃4.05秒。在该视频的留出模型中，移除这一片段不改变视频最高分；这不等于已经消除了它对其他折训练的潜在影响。

此外，1-09现存坐标表比抽帧QC多1行，不能将现存表冒称抽帧时的字节级原件。原始cut表完整快照也未找回。上述证据支持结果数值的真实性和已有权重的复算，不支持“所有原片均可一键无差异重建”或“时间间隔已经严格受控”的说法。详见[输入与时间轴核查](docs/process/experiments/EXP-008-输入与时间轴核查.md)。

## 9. 与参考论文的关系

参考：[Wu等，2025，Deep learning-based debris flow hazard detection and recognition system: a case study](https://doi.org/10.1038/s41598-025-86471-4)。本实验保留视频级MIL及三项损失，做了以下调整：

| 部分 | 参考论文 | 本实验 |
| --- | --- | --- |
| 特征骨干 | 微调C3D，FC6输出4096维 | 冻结Kinetics400预训练R(2+1)D-18，输出512维 |
| 分类网络 | 4096 → 1024 → 512 → 32 → 1 | 512 → 128 → 64 → 32 → 1 |
| 片段组织 | 连续特征提取后重新组合 | 先选择48个片段，再提取特征和按区间筛选 |
| 完整系统 | 视频检测后另接图像复核 | 评价视频检测阶段 |

这是该方法在本数据上的重新实现，不是与原论文完全同设置的数值复现。

## 10. 复现步骤

命令在Windows PowerShell、仓库根目录执行。这里说明怎样复现，不表示本轮已经重新训练。

### ① 获取代码及环境

```powershell
git clone --single-branch --branch codex/EXP-008 https://github.com/heiferleaf/debris-flow-thesis.git
cd debris-flow-thesis
py -3.8 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r experiments/runspecs/EXP-008/requirements-reproduction.txt
```

### ② 准备对应版本的数据

复现者须已获得本实验的冻结特征。原视频、像素和特征不随公有代码上传。将文件放到：

```text
data/results/exp008/unitfeats-v2-150/unit-features.npz
```

| 数据项 | 内容 |
| --- | --- |
| feats | 150 × 48 × 512片段特征 |
| unit_ids | 150个视频ID，行序固定 |
| labels | 视频标签，1正0负 |
| groups | 原场次组，仅用于身份记录 |
| folds | 原场次折；本实验由下面的跨场景JSON覆盖 |

特征文件SHA256：

```text
2cbfe0e13e0603f73027eeb80073424ab729aaf55dafd75e34dff070b34c45a9
```

仓库提供冻结片段标注CSV和场景划分。下面把CSV转换为训练程序需要的掩码文件，不重新标注、不抽帧：

```powershell
.\.venv\Scripts\python.exe experiments/runspecs/EXP-008/prepare_clip_mask.py
```

预期显示保留6,979 / 7,200个片段。生成的`clip-labels-rebuilt.npz`只是标注元数据；0和1都表示该片段可用，-1表示排除。

<details>
<summary>已有固定片段图像时，如何生成特征？</summary>

准备`data/process/EXP-008-units-v2/unit-manifest.csv`和`frames/<video_id>/clip00_f00.jpg`至`clip47_f15.jpg`。另外安装与原制作环境对应的torchvision、OpenCV、Pillow及预训练权重。

```powershell
$env:DF_UNITS_ROOT = (Resolve-Path data/process/EXP-008-units-v2).Path
$env:DF_RUNS_ROOT = Join-Path (Get-Location) data/results/exp008
.\.venv\Scripts\python.exe experiments/common/video/extract_feats.py --tag unitfeats-regenerated
```

将下方训练命令的`--feats`改为生成文件。该入口要求固定片段图像；只有原视频时，当前分支没有经过完整验证的一键重建相同特征入口。

</details>

### ③ 运行跨场景五折

```powershell
$env:DF_RUNS_ROOT = Join-Path (Get-Location) data/results/exp008
.\.venv\Scripts\python.exe experiments/common/video/train_mil.py `
  --feats data/results/exp008/unitfeats-v2-150/unit-features.npz `
  --clip-labels data/process/EXP-008-units-v2/clip-labels-rebuilt.npz `
  --caliber strict `
  --folds-json data/metadata/splits/EXP-008-units-scene-folds.json `
  --epochs 20 --lr 0.001 --seed 7 `
  --tag mil-scene-reproduction
```

`--folds-json`不可省略，否则会使用特征文件内的场次折。新结果写入`data/results/exp008/mil-scene-reproduction/`，不覆盖参考运行。

### ④ 核对原始结果

参考运行位于`data/results/exp008/mil-v2-150-strict-scene/`，包含五个`mil_f*.pt`、`metrics.json`和`predictions.csv`。预期选中轮次为4、7、14、9、18；逐折AUC见第6节，混淆矩阵为TP55 / FP15 / FN58 / TN22。

以下命令按原文件六位小数的精度逐视频核对，不只比较平均AUC：

```powershell
@'
import csv
from pathlib import Path

def read(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return {(r["fold"], r["unit_id"]):
                (int(r["label"]), round(float(r["score"]), 6))
                for r in csv.DictReader(file)}

root = Path("data/results/exp008")
assert read(root / "mil-v2-150-strict-scene/predictions.csv") == read(
    root / "mil-scene-reproduction/predictions.csv")
print("All 150 video predictions match at the stated precision.")
'@ | .\.venv\Scripts\python.exe -
```

环境或数值计算差异可能造成微小变化；不匹配时保留新结果并检查输入和依赖，不能修改参考成绩来通过核对。

## 11. 证据与入口

| 文件 | 用途 |
| --- | --- |
| [训练和损失](experiments/common/video/train_mil.py) | 原三项MIL及掩码规则 |
| [特征提取](experiments/common/video/extract_feats.py) | 16帧到512维特征 |
| [取样清单](data/process/EXP-008-units-v2/unit-manifest.csv) | 视频身份及48个片段的位置 |
| [片段标注](data/process/EXP-008-units-v2/clip-labels.csv) | 冻结strict筛选元数据 |
| [场景划分](data/metadata/splits/EXP-008-units-scene-folds.json) | 五折留出成员 |
| [原指标](data/results/exp008/mil-v2-150-strict-scene/metrics.json) | 原运行记录 |
| [逐视频预测](data/results/exp008/mil-v2-150-strict-scene/predictions.csv) | 指标和混淆矩阵依据 |

权重推理复算、单组特征重算、原视频重建和完整重新训练是不同核查范围。本次完成前两项及全部固定输入图像的技术检查，没有重新训练或改写历史输入。
