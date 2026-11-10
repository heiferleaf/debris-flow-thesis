本目录放一次性的处理工具：只为一轮数据准备或某个临时任务服务、已完成使命的脚本，通用可复用的工具在 `experiments/tools/` 根。

## 子目录

| 目录 | 内容 | 对应的结论在哪 |
| --- | --- | --- |
| `exp014-operating-point/` | EXP-014 判定规则与判定线的折外实验（`operating_point.py` 第一版、`operating_point2.py`/`operating_point3.py` 第二版、`预注册-判定规则与判定线.md` 预注册与两处修订）。2026-10-27 从 `.codex/docs/.temp/` 移入，因为 run-log 的数字要靠它复算 | `docs/process/experiments/run-logs/EXP-014-20261027-operating-point.md` |
| `exp009-fp-analysis/` | EXP-009 误报分析的附带测量：`grey_frames.py` 量那 6 条整段饱和视频喂进模型的首帧是不是发灰（结论：不是，发灰帧占比 0.0%）；`short_video_bias.py` 量「片段数少于 48 时取最高分会压低多少分」（结论：正样本几乎不动，负样本反而更安全）；`grey_std_15fp.py` 按训练同款判据（灰度标准差 < 20）复算 15 条误报的全部 11,520 张采样帧（结论：命中 0 张，最低 v4_01 的 23.83，按更严的 < 5 也为 0）。2026-10-27 从 `.codex/docs/.temp/` 移入，`grey_std_15fp.py` 2026-10-28 新增 | `docs/process/experiments/run-logs/EXP-009-20261027-false-positives.md` 第二节末；`EXP-009-训练报告.md` §4.4 |
| `exp014-excluded-22/` | EXP-014 在 0.5 秒间隔下排除的 22 条的排除原因逐条归因：`attribute_causes.py` 复用 `build_units_stride` 同一套禁区构造，对每条分别算「只留剪切剔除区」「只留画质族（低灰度/倒步/空洞）」「全拿掉」时的合法片段数。结论：3 条纯剪切区（v1_19、v1_55、v1_70）、18 条纯画质族、1 条两族共同（v1_39）、0 条容量不足 | `data/数据清单.md` 第三节「凑不满 48 组具体卡在哪类禁区」 |
| `exp008-zone-audit/` | `zone_consistency.py` 对 150 条池内视频做离线禁区一致性扫描：只解一遍逐帧时间戳、调推理侧同一套禁区函数，数出「冻结清单的 48 个槽位里有几个按今天的规则落在禁区里」。不跑模型。2026-10-28 新增，起因是 `infer_video.py` 改成按训练同款规则剔除禁区片段后，往返自检只能抽查几条 | 同目录 `result.md`；逐条数据在 `result.csv` |

移入时只改了两处：脚本里的仓库根目录层级（`parents[2]` 改成 `parents[4]`，目录深了一层）与示例命令路径。计算逻辑一字未动，移入后重跑，输出的 `R1 取最高分 线 0.50：FP 15 / FN 0`、`R2 全部片段均值 线 0.50：FP 6 / FN 3` 与对应 run-log 第五节的表一致。

`exp009-fp-analysis/` 移入后同样重跑过，`grey_frames.py` 输出「6 条整段饱和误报：发灰帧占比中位 0.0%」、`short_video_bias.py` 输出「正样本 113 条，不足 48 个片段的 74 条」，与 run-log 第二节末写的一致。`short_video_bias.py` 移入时另加了第四节（固定同一批有 48 个片段的视频、只改变抽样个数），因为前三节的每个 n 参与的视频不是同一批，分母在变。
