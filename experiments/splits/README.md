划分方法库。两层含义（2026-09-13 叶海峰定的结构原则）：

1. **划分方式本身是实验要选的方法**（机位级 holdout / 场次级留出 / k 折交叉验证 / 分层采样），每种复现线在**自己的子文件夹**里声明用哪种（写进该线的实验卡），通用说明沉淀在本 README。
2. 通用的划分清单 JSON 放这里（只含 ID），由 `python -m experiments.tools.split_checks experiments/splits/<name>.json` 校验。字段见 `docs/sop/templates/split-manifest.md`。

## 现有清单

- `EXP-006-{session,scene,scene-logo}-folds.json`：服务 EXP-006 线（窗口级单元，按窗口数分层），EXP-008 不再使用（2026-10-04 research-auditor 决策复核认定对整条视频单位属错配）。
- `EXP-008-units-folds.json`：服务 EXP-008 重开版（整条视频弱标签单位，按组级 pos/neg **单位数**分层，**协议一场次级**），由 `experiments/splits/code/make_unit_folds.py` 生成（seed 42，大组优先+随机重启贪心），2026-10-04 一次落盘后冻结；唯一正本，manifest fold 列与特征 npz folds 字段均已同步（manifest 的 fold 列只存协议一折值）。
- `EXP-008-units-iid-folds.json`：EXP-008 **协议三「同场次混分参照」**（2026-10-04 叶海峰定，无组约束，仅按标签分层随机 5 折），同脚本 `--iid --apply` 生成、一次落盘冻结；成绩只作上限参照，引用必须带脚注「同场次混分参照，非泛化成绩」。两份正本分离，不得混用。

图像划分清单的初版草稿 `images-v1.json` 已删除（草稿且过期，2026-09-11 由 `tools/oneoff/build_split_manifest.py` 从 pHash 簇生成）：待审计判决（通过/改负/剔除）应用后重新生成，再过 `split_checks`。
