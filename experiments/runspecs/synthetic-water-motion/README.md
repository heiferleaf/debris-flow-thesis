# 普通水流外观试验

本目录是视频增强试验的专用实现，不是已经批准的训练方法。以下文件属于浅水流场驱动试片。通用工作台仍在 `experiments/tools/synthetic_minimax_workbench*`。

## 当前状态与证据

- 1号塔下游：`render-v4/` 被用户拒绝，原因是每1秒重播的感觉；`render-v5/` 因连续输运后水纹模糊，内部检查不通过，未发布。
- 本轮目录为 `data/process/synthetic-video/physical-water-20261110/camera-e3e5e0945cbc/`；`neural-v6-stride3/` 已生成并发布到工作台，采用已有预训练渲染器及计算流场，12秒、24fps、1280×720。**等待用户验收**，当前入口以 `current.json` 为准。
- `comparison-v6/` 对比一秒循环旧版与当前候选。水域去时间均值后，相隔1秒的相关系数由0.905降至0.158，仅支持短循环减弱，不证明物理真实。
- `comparison-v4/` 只是已拒绝旧版本的对照，不能视为已验收产物。
- 同级 `current.json` 记录入口、参数假设、代码指纹和未通过的中间版本。`render-v4/code/` 保存实际实现快照。
- 工作台 `http://127.0.0.1:8774/` → 1号塔下游 → 物理流场驱动试片；生成完成不等于验收，不自动进入训练。
- 当前为单视角试验，尚未将本方法推广到全部11个视角或通用上传入口。

## 实现与限制

`physics_core.py` 解二维深度平均浅水方程，采用静水重构、Rusanov通量、Manning底摩擦和水平动量扩散。基础检查覆盖静水、闭域质量、均匀摩擦解析解；不等于真实河道验证或完整数值收敛验证。

`simulate_scene.py` 将细化的水面范围映射到假设地平面，保存水深和动量。当前相机投影、14×60米参考范围、0.8米中心水深、0.002坡度、0.035糙率均为假设，没有现场测量。当前质量变化与进出水预算吻合，但不可将视频称为实测水动力复原。

`render_physical_scene.py` 投影计算流速，保护岸石背景，在水域内搬运首帧纹理。表面速度系数1.1、细波色调及纹理更新是外观近似。每1秒交替更新两套纹理，避免连续搬运把首帧拉成长条；可能出现周期感或交叉淡化，需播放验收。未解析真实三维反光、泡沫及破碎波。

`compare_scene.py` 制作288帧同步对照，并计算近岸和中部图像运动辅助量；光流或帧差不构成水动力真实性证明，也不替代用户验收。

新增 `export_renderer_flow.py` 将模拟的时间平均流速投影为全分辨率像素/帧，交给现有 SLR-SFS baseline 编码器/解码器。采用平均流场会丢弃这12秒的细小流速变化，不能宣称完整时变浅水渲染。原网络的前后向融合在整段12秒内进行，不每秒重置；末尾会趋近首帧，因此同样不是无限长度的水面模拟。

通用渲染器新增可选 `--flow-field`、`--water-mask`，默认入口保持原行为；可选 `--frame-stride 3` 每3帧计算网络输出，中间帧按流场补偿后融合，必须与逐帧解码对照。已保存 `neural-v6/candidate.mp4` 的66帧部分结果作核验，它不是完整视频，不应发布到验收入口。

当前候选完成插补比较：前66帧水域插补帧平均绝对像素差3.50/255，关键帧2.73/255；两次运行未编码首帧逐像素一致，编码比较含压缩误差，不可宣称插补完全等价。实际命令在 `neural-v6-stride3/run-command.json`，实现快照在其 `code/`；运行时帧数报告误将97个关键帧当总帧数，已经按视频实测的288帧/12秒更正，原报告保留为 `result-renderer-original.json`，同时已修正源码计数。

## 复现

在仓库根目录运行，输出须使用新的目录名，禁止覆盖已有候选：

```powershell
./.venv/Scripts/python.exe -X utf8 experiments/runspecs/synthetic-water-motion/simulate_scene.py --input data/process/synthetic-video/controlled-flow-20261110-boundary/camera-e3e5e0945cbc/input-refined-v2 --output data/process/synthetic-video/physical-water-20261110/camera-e3e5e0945cbc/simulation-reproduction
./.venv/Scripts/python.exe -X utf8 experiments/runspecs/synthetic-water-motion/render_physical_scene.py --simulation data/process/synthetic-video/physical-water-20261110/camera-e3e5e0945cbc/simulation-reproduction --output data/process/synthetic-video/physical-water-20261110/camera-e3e5e0945cbc/render-reproduction
```

基本输入与代码只适配这一假设场景，批量应用前须将相机和物理参数独立配置，并检查各场景岸线、固定物和入出流位置；不能直接套用同一个投影。

参考：Audusse等，2004，A Fast and Stable Well-Balanced Scheme with Hydrostatic Reconstruction for Shallow Water Flows，DOI:10.1137/S1064827503431090。方程成立不保证单图动画自然，也不证明合成增强对识别模型有收益。
