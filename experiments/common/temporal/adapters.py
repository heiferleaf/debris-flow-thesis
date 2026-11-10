# -*- coding: utf-8 -*-
"""训练数据加载适配器（设计见 docs/data/训练数据加载适配器.md）。

职责边界：
- 训练脚本（train_temporal.py）只依赖 DataAdapter.single_frames(split) / windows(split)
  与 metrics_summary，不拼 data/ 绝对路径，不了解 data/ 物理布局。
- 窗口怎么切、每段几帧、短段加密、类权重 = 可插拔 WindowPolicy，默认 EXP006WindowPolicy，
  与当前试点（EXP-006）行为逐一对齐。
- 数据源差异（东川/外部）收敛到 VIDEO_SOURCES 的 videoN 映射与 resolve_frame()。

新 data/ 结构（2026-09-16 拍板，物理迁移未全部完成）：
    data/process/video_frames/video1_segment/   <- 东川初帧（旧 data/frames 的 122 视频目录合入）
    data/process/video_frames/video2_segment/   <- 网络场景帧（旧 data/frames-scenes）
    data/final/video_windows/video1_segment/、video2_segment/  <- 最终连续帧（短段加密后）
    data/final/images/clusters/、final/images/labels/          <- 单帧侧（pHash 簇 / audit 判决）
    权重/特征缓存   ->  data/results/<实验线>/<run_id>/

videoN 映射：video1 = 东川（data/raw/video_dataset_1 + video_dataset_2），
            video2 = 网络（data/raw/webcam-negatives），
            video4 = video_dataset_3（爆发后正样本）、video5 = video_dataset_4（爆发前负样本）。
路径全部可用环境变量覆盖；未迁移的旧路径（frames / frames-scenes / processed）自动 fallback，
保证现有存盘产物（manifest / cache.npz）在原位置原样可复现。
"""
import csv
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[3]  # 仓库根
sys.path.insert(0, str(REPO))
from experiments.paths import resolve_path  # noqa: E402
DATA_ROOT = resolve_path(os.environ.get("DF_DATA_ROOT") or (REPO / "data"))
RUN_ROOT = resolve_path(os.environ.get("DF_RUN_ROOT") or (REPO / "data/results/temporal"))

# ---------------------------------------------------------------- 路径配置

def _pick(new: Path, legacy: Path) -> Path:
    """读路径：新结构存在才用它，否则退回旧路径——目录迁移不应破坏现有可复现。
    新结构目录也不存在时返回新路径，让调用方在真实报错中看到目标位置。"""
    if new.exists():
        return new
    return legacy if legacy.exists() else new


def _write_path(new: Path, legacy: Path) -> Path:
    """写路径：目标父目录已就位才写新结构，否则写旧位置（未迁移时照旧写 processed）。"""
    if new.parent.is_dir():
        return new
    return legacy if legacy.parent.is_dir() else new


def video_frames_root(video_no: str) -> Path:
    """某个 videoN 的段帧根目录（manifest 里 frames 相对路径的挂载根）。"""
    env = os.environ.get(f"DF_VIDEO{video_no}_FRAMES")
    if env:
        return resolve_path(env)
    src = VIDEO_SOURCES[video_no]
    legacy = src.get("frames_legacy")
    if legacy is not None and legacy.exists():
        return legacy
    return src["frames_new"]


def frames_root_for_source(source: str) -> Path:
    """manifest source 字段（dongchuan / extneg）-> 帧根目录。"""
    return video_frames_root(_SOURCE_TO_VIDEO[source])


def resolve_frame(source: str, rel: str) -> Path:
    """source（东川/外部）+ manifest 内相对路径 -> 帧文件绝对路径。"""
    normalized = str(rel).replace("\\", "/")
    if (normalized.startswith(("/", "data/"))
            or (len(normalized) > 2 and normalized[1] == ":")):
        return resolve_path(rel)
    return frames_root_for_source(source) / rel


def manifest_path() -> Path:
    """段清单（段划分与帧列表入口）。"""
    env = os.environ.get("DF_MANIFEST")
    if env:
        return resolve_path(env)
    return _pick(
        DATA_ROOT / "process/label-review/pilot-segment-manifest.csv",
        REPO / "data/processed/label-review/pilot-segment-manifest.csv",
    )


def out_manifest_path() -> Path:
    """build_segment_manifest 的输出位置（写侧，避免写进不存在的目录）。"""
    env = os.environ.get("DF_MANIFEST") or os.environ.get("DF_MANIFEST_OUT")
    if env:
        return resolve_path(env)
    return _write_path(
        DATA_ROOT / "process/label-review/pilot-segment-manifest.csv",
        REPO / "data/processed/label-review/pilot-segment-manifest.csv",
    )


def scene_segments_csv() -> Path:
    """外部场景分段审计清单（build_segment_manifest 的输入之一）。"""
    env = os.environ.get("DF_SCENE_SEGMENTS_CSV")
    if env:
        return resolve_path(env)
    return _pick(
        DATA_ROOT / "process/video_frames/video2_segment/extneg-scene-frames.csv",
        REPO / "data/processed/label-review/extneg-scene-frames.csv",
    )


def cache_npz_path() -> Path:
    """特征缓存读路径：新 runs 根优先，未迁移时取旧 data/processed/weights/EXP-006-pilot。"""
    env = os.environ.get("DF_CACHE")
    if env:
        return resolve_path(env)
    return _pick(
        RUN_ROOT / "EXP-006-pilot/cache.npz",
        REPO / "data/processed/weights/EXP-006-pilot/cache.npz",
    )


def out_cache_path(cache_tag: str = "EXP-006-pilot") -> Path:
    """cache_features 的输出位置：data/results/temporal/。"""
    file_name = os.environ.get("DF_CACHE")
    if file_name:
        return resolve_path(file_name)
    return RUN_ROOT / cache_tag / "cache.npz"


def exp5_weights_dir() -> Path:
    """EXP-005 冻结权重（linear_head.pt）所在目录，供 cache_features 打每帧概率。
    T2 后正本在 convnext 线 runs/（EXP-005 是 ConvNeXt 单帧线产物）。"""
    env = os.environ.get("DF_EXP5_WEIGHTS")
    if env:
        return resolve_path(env)
    return _pick(REPO / "data/results/convnext/EXP-005-final",
                 REPO / "data/processed/weights/EXP-005-final")


# ---------------------------------------------------------------- videoN 映射

#: manifest source 字段 -> videoN
_SOURCE_TO_VIDEO = {"dongchuan": "1", "extneg": "2", "deflow": "3",
                    "video4": "4", "video5": "5"}

#: videoN -> 数据源描述。frames_* 为段帧根（新/旧）；raw 为原始视频实体位置（只读参考）。
VIDEO_SOURCES: dict = {
    "1": {  # 东川
        "name": "dongchuan",
        "raw_dirs": (DATA_ROOT / "raw/video_dataset_1", DATA_ROOT / "raw/video_dataset_2"),
        "frames_new": DATA_ROOT / "process/video_frames/video1_segment",
        "frames_legacy": REPO / "data/frames",
        "windows_new": DATA_ROOT / "final/video_windows/video1_segment",
    },
    "2": {  # 网络
        "name": "extneg",
        "raw_dirs": (DATA_ROOT / "raw/webcam-negatives",),
        # 段帧根:训练清单的段目录 <key>__segNN_kind 在 final 侧(T1 起为正本);
        # process/video_frames/video2_segment 只放审计初帧平铺(帧名带 key 前缀)
        "frames_new": DATA_ROOT / "final/video_windows/video2_segment",
        "frames_legacy": REPO / "data/frames-scenes",
        "windows_new": DATA_ROOT / "final/video_windows/video2_segment",
    },
    "3": {  # DeFlow(ETH Zurich,2026-09-17 入线;叶海峰确认整段 pos)
        "name": "deflow",
        "raw_dirs": (DATA_ROOT / "raw/deflow",),
        "frames_new": DATA_ROOT / "process/video_frames/video3_segment",
        "frames_legacy": None,
        "windows_new": DATA_ROOT / "process/video_frames/video3_segment",
    },
    "4": {  # video_dataset_3(2026-08-18 爆发后,小型泥石流到达塔吊断面;审计切段 pos16/neg9/cut3)
        "name": "video4",
        "raw_dirs": (DATA_ROOT / "raw/video_dataset_3",),
        "frames_new": DATA_ROOT / "process/video_frames/video4_segment",
        "frames_legacy": None,
        "windows_new": DATA_ROOT / "process/video_frames/video4_segment",
    },
    "5": {  # video_dataset_4(2026-08-18 爆发前常规水流,整段 neg;v5_08 解码不可修复剔除,mapping excluded=1)
        "name": "video5",
        "raw_dirs": (DATA_ROOT / "raw/video_dataset_4",),
        "frames_new": DATA_ROOT / "process/video_frames/video5_segment",
        "frames_legacy": None,
        "windows_new": DATA_ROOT / "process/video_frames/video5_segment",
    },
}


# ---------------------------------------------------------------- 采样策略

class WindowPolicy(ABC):
    """窗口采样策略：怎么切窗口、每段窗口上限、短段加密、类权重。可插拔。"""

    K: int = 16            # 窗口帧长
    stride: int = 8        # 窗口步长
    max_win_per_seg: int = 8   # 每段窗口上限
    min_frames: int = 16   # 短段加密目标：至少能出 1 个窗口（方案A）

    @abstractmethod
    def window_starts(self, n_frames: int) -> np.ndarray:
        """给定段内帧数，返回窗口起点集合（升序、唯一）。"""

    @abstractmethod
    def class_weight(self, n_pos: int, n_neg: int) -> float:
        """类别权重：BCE pos_weight = 负/正。"""


class EXP006WindowPolicy(WindowPolicy):
    """默认策略 = EXP-006 试点行为：linspace 均匀取 <=max_win 个起点，只在段内滑不跨段。"""

    def window_starts(self, n_frames: int) -> np.ndarray:
        if n_frames < self.K:
            return np.array([], dtype=int)
        max_start = n_frames - self.K
        n_w = min(self.max_win_per_seg, max_start // self.stride + 1)
        return np.unique(np.round(np.linspace(0, max_start, n_w)).astype(int))

    def class_weight(self, n_pos: int, n_neg: int) -> float:
        return n_neg / max(1, n_pos)


def _pad_to_min(idx: np.ndarray, min_frames: int) -> np.ndarray:
    """短段兜底（方案A 加载层实现）：帧不足 min_frames 时用 linspace 保序重复采样到 >=min_frames。
    与 build_tsn_samples 的确定性填充同法，保证同一段补帧结果唯一。"""
    if len(idx) >= min_frames:
        return idx
    sel = np.round(np.linspace(0, len(idx) - 1, min_frames)).astype(int)
    return idx[sel]


# ---------------------------------------------------------------- 统一接口

@dataclass
class Sample:
    """单帧样本（单帧训练模式）。"""
    seg_key: str
    feats: np.ndarray          # [D]
    label: int


@dataclass
class Window:
    """连续帧窗口（时序训练模式）：一段输出若干窗口。"""
    seg_key: str
    feats: np.ndarray          # [K, D]
    label: int
    mean_logit: float          # 窗口内帧级平均 logit（单帧基线用）


class DataAdapter(ABC):
    """训练脚本唯一依赖的加载接口。split = "train" / "val" / "test"。"""

    @abstractmethod
    def single_frames(self, split: str) -> Iterable[Sample]:
        """单帧模式：一段内每帧一个 Sample。"""

    @abstractmethod
    def windows(self, split: str) -> Iterable[Window]:
        """连续帧模式：段→窗口→帧三步在此完成。"""

    @property
    @abstractmethod
    def metrics_summary(self) -> dict:
        """各 split 的段数/窗口数/正负样本数（核对用）。"""


class SegmentManifestAdapter(DataAdapter):
    """段清单 + 特征缓存实现的适配器（当前视频侧唯一实现）。

    段→窗口→帧 的解析、帧根目录的 videoN 映射都在这层做；训练脚本只需 windows("train")。
    低层用途（cache_features / TSN 对照）可读 .rows 与 .cache_arrays 钩子。
    """

    def __init__(self, manifest=None, cache=None, policy: WindowPolicy = None):
        # 注:模块级 manifest_path()/cache_npz_path() 与属性名同前缀,此处用 globals() 显式取函数
        self.manifest_file = resolve_path(manifest or globals()["manifest_path"](), must_exist=True)
        self.cache_file = resolve_path(cache or globals()["cache_npz_path"](), must_exist=True)
        self.policy = policy or EXP006WindowPolicy()
        self.rows = list(csv.DictReader(open(self.manifest_file, encoding="utf-8-sig")))
        z = np.load(self.cache_file)
        self._cache = {"feats": z["feats"], "probs": z["probs"],
                       "row_id": z["row_id"], "labels": z["labels"]}

    @property
    def cache_arrays(self) -> dict:
        """特征缓存原始数组（feats/probs/row_id/labels），供 TSN 对照等按段手动取帧。"""
        return self._cache

    @staticmethod
    def seg_key(r: dict) -> str:
        """清单一行 -> 段唯一键（source/video_id/seg_id，与试点报告口径一致）。"""
        return f'{r["source"]}/{r["video_id"]}/seg{r["seg_id"]}'

    def _segment_idxs(self, ri: int) -> np.ndarray:
        """缓存中属于第 ri 个清单段的帧行索引（缓存按清单顺序追加，天然保序）。"""
        return np.where(self._cache["row_id"] == ri)[0]

    def single_frames(self, split: str) -> Iterable[Sample]:
        feats = self._cache["feats"]
        for ri, r in enumerate(self.rows):
            if r["split"] != split:
                continue
            idx = self._segment_idxs(ri)
            label = 1 if r["kind"] == "pos" else 0
            key = self.seg_key(r)
            for i in idx:
                yield Sample(key, feats[i], label)

    def windows(self, split: str) -> Iterable[Window]:
        feats, probs = self._cache["feats"], self._cache["probs"]
        for ri, r in enumerate(self.rows):
            if r["split"] != split:
                continue
            idx = _pad_to_min(self._segment_idxs(ri), self.policy.min_frames)
            n = len(idx)
            if n < self.policy.K:          # 加密后仍不足一个窗口（理论上不出现在兜底后）
                continue
            label = 1 if r["kind"] == "pos" else 0
            key = self.seg_key(r)
            for s in self.policy.window_starts(n):
                sel = idx[s:s + self.policy.K]
                yield Window(key, feats[sel], label, float(probs[sel].mean()))

    @property
    def metrics_summary(self) -> dict:
        out = {}
        for sp in ("train", "val", "test"):
            win = list(self.windows(sp))
            if not win:
                continue
            nseg = len({w.seg_key for w in win})
            out[sp] = {"segments": nseg, "windows": len(win),
                       "pos": sum(1 for w in win if w.label == 1),
                       "neg": sum(1 for w in win if w.label == 0)}
        return out

    def class_weight(self, split: str) -> float:
        """该 split 的类别权重（BCE pos_weight = 负/正），按窗口样本计。"""
        win = list(self.windows(split))
        n_pos = sum(1 for w in win if w.label == 1)
        n_neg = sum(1 for w in win if w.label == 0)
        return self.policy.class_weight(n_pos, n_neg)