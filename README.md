# 视频级泥石流识别：弱监督 MIL 基线

只标注一条视频“出现过泥石流”或“没有泥石流”，能否训练模型识别其中的泥石流画面？本实验使用预训练时空卷积网络提取短片段特征，再通过多示例学习（Multiple Instance Learning，MIL）进行视频级分类。

| 实验要点 | 内容 |
| --- | --- |
| 数据 | 150 条监控视频：113 条正样本、37 条负样本 |
| 输入 | 每条视频48个16帧片段，转换为48 × 512特征后训练 |
| 模型 | 冻结 R(2+1)D-18 + 四层全连接分类网络 |
| 监督 | 视频级正负标签，三项 MIL 损失 |
| 评价 | 按拍摄场次分组的五折交叉验证 |
| 主要结果 | 平均 AUC **0.9412**；Precision **0.8067**、Recall **0.8496**、F1 **0.8276** |

## 1. 研究问题

一条正视频可能只有部分时间出现泥石流，不能把它的所有片段直接视为正样本。MIL 将整条视频看作一个“包”，短片段是包内的“实例”：训练只知道正包中存在目标、负包中没有目标，让模型学习各片段的分数，再判断整条视频。

## 2. 数据与取样

### 数据来源

| 来源 | 视频数 | 正样本 | 负样本 |
| --- | ---: | ---: | ---: |
| 东川站 FTP 视频 | 106 | 97 | 9 |
| 百度网盘视频组 v4 | 22 | 16 | 6 |
| 百度网盘视频组 v5 | 22 | 0 | 22 |
| **合计** | **150** | **113** | **37** |

正样本表示视频中出现过目标泥石流，负样本表示按本实验标注未出现目标。标签属于整条视频，不要求正视频的每一帧都有泥石流。上表采用当时冻结的训练成员及标签。

### 一条视频如何成为训练样本

1. 排除人工剔除区间、损坏帧和时间异常区域。
2. 在整条视频的合法范围内均匀选取 **48 个片段**；每段包含原生解码帧序中连续的 **16 帧**，图像缩放为 **112 × 112**。
3. 使用冻结的预训练网络，将每个片段转换为 **512 维特征向量**。

因此，一条视频对应 **48 × 512** 的特征矩阵，整个训练数据形状为 **150 × 48 × 512**。16帧的物理时长由原视频时间轴决定。取样覆盖整条视频，正视频中的正常画面和泥石流画面都可以进入同一个样本。

## 3. 五折划分

以“**固定视角 × 拍摄日期**”为场次组，同组视频始终在同一侧。每次使用四折训练、一折验证，共训练五个模型；每条视频恰好作为一次留出验证样本。

| 折 | 训练正样本 | 训练负样本 | 验证正样本 | 验证负样本 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 91 | 30 | 22 | 7 |
| 2 | 89 | 30 | 24 | 7 |
| 3 | 91 | 27 | 22 | 10 |
| 4 | 91 | 31 | 22 | 6 |
| 5 | 90 | 30 | 23 | 7 |

不同日期的同一拍摄地点可以跨折，因此这里评价的是场次分组条件下的表现，不是严格的未见物理场景测试。

## 4. 模型架构

```mermaid
flowchart LR
    A["一条监控视频<br/>48 段 × 16 帧"] --> B["冻结 R(2+1)D-18<br/>48 × 512 特征"]
    B --> C["共享四层 MLP<br/>48 个片段分数"]
    C --> D["最大值聚合<br/>视频级正负判定"]
```

| 模块 | 结构或作用 | 训练状态 |
| --- | --- | --- |
| 特征提取网络 | Kinetics400 预训练 R(2+1)D-18，输出512维特征 | 冻结 |
| 分类网络 | 全连接层：512 → 128 → 64 → 32 → 1 | 更新参数 |
| 隐藏层 | ReLU 激活和50% Dropout | — |
| 输出层 | Sigmoid，片段分数范围为0～1 | — |
| 视频聚合 | 取48个分数的最大值；≥0.5判正 | 固定规则 |

MLP 指多层全连接网络。片段内的时空信息由 R(2+1)D-18 提取，所有片段共享同一个分类网络。

## 5. 三项损失

每次配对一个正视频和一个负视频，片段分数分别记为 $p_1,\ldots,p_{48}$ 和 $n_1,\ldots,n_{48}$。定义包内分数的峰谷差：

$$
R_p=\max_k p_k-\min_k p_k,\qquad R_n=\max_k n_k-\min_k n_k.
$$

**峰谷差约束**鼓励正视频的分数变化范围大于负视频：

$$
L_1=\max(0,\;1-R_p+R_n).
$$

**平滑约束**限制正视频相邻片段分数的突变：

$$
L_2=\sum_{k=1}^{47}(p_{k+1}-p_k)^2.
$$

**稀疏约束**限制正视频高分响应的总量：

$$
L_3=\sum_{k=1}^{48}p_k.
$$

总目标如下；一个批次内再对视频对取平均：

$$
\boxed{L=L_1+8\times10^{-5}L_2+8\times10^{-5}L_3.}
$$

后两项只作用于正视频。第一项比较的是**包内峰谷差**，不是正负视频最高分的直接差值。

## 6. 训练设置

| 设置 | 取值 |
| --- | --- |
| 更新参数 | 仅分类网络 |
| 优化器 | Adam |
| Adam参数 | β=(0.9, 0.999)，ε=10⁻⁸，weight decay=0 |
| 学习率 | 0.001 |
| 每折训练轮数 | 20 |
| 随机种子 | 7，每折重新设置 |
| 配对批次 | 每批最多8个正视频和8个负视频；末批使用剩余数量 |
| 每轮样本组织 | 正视频无放回遍历，负视频循环补齐配对 |
| 模型选择 | 保存验证AUC最高的一轮；并列时保留先出现者 |
| 视频分类阈值 | 0.5 |

## 7. 实验结果

### 五折验证 AUC

AUC 衡量正视频的分数是否总体高于负视频，不依赖某个固定分类阈值。

| 折 | 验证视频数 | AUC | 选中轮次 |
| --- | ---: | ---: | ---: |
| 1 | 29 | 1.0000 | 13 |
| 2 | 31 | 0.8333 | 20 |
| 3 | 32 | 0.9545 | 16 |
| 4 | 28 | 0.9242 | 15 |
| 5 | 30 | 0.9938 | 15 |
| **逐折均值** | — | **0.9412** | — |

表中轮次从1开始；原结果文件使用从0开始的编号。

### 阈值 0.5 下的分类结果

合并五折各自留出视频的判决，得到：

| 真实类别 | 预测：泥石流 | 预测：非泥石流 |
| --- | ---: | ---: |
| **泥石流，113条** | **96**（TP，正确检出） | **17**（FN，漏报） |
| **非泥石流，37条** | **23**（FP，误报） | **14**（TN，正确排除） |

| 指标 | 数值 | 含义 |
| --- | ---: | --- |
| Precision | **0.8067** | 报为泥石流的视频中，实际为正类的比例 |
| Recall | **0.8496** | 所有正视频中，被成功检出的比例 |
| F1 | **0.8276** | Precision 与 Recall 的调和平均 |
| Accuracy | 0.7333 | 全部视频中，判对的比例 |
| FPR | 0.6216 | 所有负视频中，被误报的比例 |

AUC采用逐折均值，混淆矩阵及派生指标采用五折判决合并统计。

## 8. 结果分析

模型在这批数据上获得了较高的排序AUC，但默认阈值下仍有 **23条误报和17条漏报**，尚未达到可部署水平。最高分聚合允许单个高分片段触发整条视频的正类判定，因此需要同时观察误报和漏报，不能只看AUC。

本实验在每折验证集上选择轮次，并在同一验证折报告成绩，结果可能偏乐观；它属于开发阶段结果，没有独立测试集成绩。

## 9. 与参考论文的关系

参考：[Wu 等，2025，*Deep learning-based debris flow hazard detection and recognition system: a case study*](https://doi.org/10.1038/s41598-025-86471-4)。本实验保留视频级监督、片段评分和三项损失，做了以下实现调整：

| 部分 | 参考论文 | 本实验 |
| --- | --- | --- |
| 视频特征网络 | 微调C3D，取FC6的4096维特征 | 冻结R(2+1)D-18，取512维特征 |
| 分类网络宽度 | 4096 → 1024 → 512 → 32 → 1 | 512 → 128 → 64 → 32 → 1 |
| 48个特征的构建 | 连续片段特征提取后重新组合 | 选定48个片段后提取特征 |
| 最终系统 | 视频检测后另接图像复核 | 本实验评价视频检测阶段 |

本实验是该方法在本数据上的重新实现，不直接与原论文的数据集成绩横比。

## 10. 复现步骤

以下命令在 **Windows PowerShell、仓库根目录**执行。数据须为本实验对应版本，不能替换为后来修改的标签或另一批同源视频。

### ① 获取代码与环境

```powershell
git clone --single-branch --branch codex/EXP-008 https://github.com/heiferleaf/debris-flow-thesis.git
cd debris-flow-thesis
py -3.8 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r experiments/runspecs/EXP-008/requirements-reproduction.txt
```

依赖文件固定了本地现有CPU训练环境中的关键库版本，Python版本为3.8.0。

### ② 放置对应数据

分类网络直接读取前文的 **150 × 48 × 512 视频特征数据**。将数据包中的冻结特征文件放到：

```text
data/results/exp008/unitfeats-v2-150/
└── unit-features.npz
```

| 文件内的数据项 | 内容 |
| --- | --- |
| `feats` | 150 × 48 × 512 片段特征 |
| `unit_ids` | 与特征行一一对应的150个视频ID |
| `labels` | 视频标签：1为正，0为负 |
| `groups` | 场次组 |
| `folds` | 每条视频所属验证折，取值1～5 |

原视频、片段帧和特征数据不随代码上传。这里假设复现者已获得上述对应数据包。下面的文件指纹（SHA256）用于确认拿到的是同一版冻结特征：

```text
2cbfe0e13e0603f73027eeb80073424ab729aaf55dafd75e34dff070b34c45a9
```

```powershell
Get-FileHash data/results/exp008/unitfeats-v2-150/unit-features.npz -Algorithm SHA256
```

<details>
<summary>数据包提供固定片段帧时，如何生成特征？</summary>

准备仓库中的训练成员清单，以及每条视频对应的48 × 16张片段帧：

```text
data/process/EXP-008-units-v2/
├── unit-manifest.csv
└── frames/
    ├── v1_01/
    │   ├── clip00_f00.jpg
    │   ├── ...
    │   └── clip47_f15.jpg
    └── ...
```

`extract_feats.py`按清单读取RGB图像、缩放为112 × 112，并通过Kinetics400预训练R(2+1)D-18提取512维特征。这一阶段另需与数据制作一致的torchvision、OpenCV、Pillow及预训练权重环境。

```powershell
$env:DF_UNITS_ROOT = (Resolve-Path data/process/EXP-008-units-v2).Path
$env:DF_RUNS_ROOT = Join-Path (Get-Location) data/results/exp008
.\.venv\Scripts\python.exe experiments/common/video/extract_feats.py --tag unitfeats-regenerated
```

输出为 `data/results/exp008/unitfeats-regenerated/unit-features.npz`；使用该文件时，相应修改下一步的 `--feats` 路径。

这一入口要求已经获得固定片段帧。只有原视频时，还需按冻结取样清单重建片段帧；当前分支没有经过验证的“原片一键重建相同特征”入口，不能与直接提供冻结特征的条件混同。

</details>

### ③ 训练五折分类网络

```powershell
$env:DF_RUNS_ROOT = Join-Path (Get-Location) data/results/exp008
.\.venv\Scripts\python.exe experiments/common/video/train_mil.py `
  --feats data/results/exp008/unitfeats-v2-150/unit-features.npz `
  --epochs 20 --lr 0.001 --seed 7 `
  --tag mil-full-reproduction
```

程序读取数据文件中的折编号，依次训练五折。新结果写入 `data/results/exp008/mil-full-reproduction/`，与原参考结果分开保存。

| 输出文件 | 内容 |
| --- | --- |
| `metrics.json` | 每折规模、AUC、F1、选中轮次及总体摘要 |
| `predictions.csv` | 留出视频的折编号、ID、标签和分数 |
| `mil_f1.pt`～`mil_f5.pt` | 五折各自选中的分类网络权重 |

### ④ 核对结果

对照第7节：各折选中轮次应为 **13、20、16、15、15**，逐折AUC相同，混淆矩阵为 **TP=96、FP=23、FN=17、TN=14**。

下面的命令按原文件保存的六位小数精度，逐视频比较新旧分数；只读文件，不修改结果：

```powershell
@'
import csv
from pathlib import Path

def predictions(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return {
            (row["fold"], row["unit_id"]):
            (int(row["label"]), round(float(row["score"]), 6))
            for row in csv.DictReader(file)
        }

root = Path("data/results/exp008")
reference = predictions(root / "mil-v2-150-full/predictions.csv")
reproduced = predictions(root / "mil-full-reproduction/predictions.csv")
assert reference == reproduced, "Predictions differ from the reference."
print("All 150 video predictions match at the saved precision.")
'@ | .\.venv\Scripts\python.exe -
```

仅AUC相同不足以证明逐视频结果相同。若核对不通过，先检查数据、依赖、种子与参数，保留新结果，不覆盖参考文件。

## 11. 代码与结果索引

| 文件 | 用途 |
| --- | --- |
| [特征提取代码](experiments/common/video/extract_feats.py) | 固定片段帧转换为特征 |
| [训练与损失代码](experiments/common/video/train_mil.py) | 分类网络、损失、五折训练和模型选择 |
| [成员与取样清单](data/process/EXP-008-units-v2/unit-manifest.csv) | 150条视频及其48个片段位置 |
| [场次划分](data/metadata/splits/EXP-008-units-folds.json) | 各折组成员与类别数量 |
| [原始指标](data/results/exp008/mil-v2-150-full/metrics.json) | 本报告的机器记录 |
| [逐视频预测](data/results/exp008/mil-v2-150-full/predictions.csv) | 混淆矩阵与逐视频比对依据 |

已保存的分类头经过原特征的推理核对；这里提供从初始化开始训练的复现步骤，不将说明书完备表述为本次已经完成五折重训验证。
