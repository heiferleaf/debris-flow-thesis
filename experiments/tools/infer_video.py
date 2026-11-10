# -*- coding: utf-8 -*-
"""单条视频推理：按 EXP-008 训练时的取材与打分方式处理一个新视频（2026-10-05 立）。

叶海峰 2026-10-05 要求：「可以让我输入一个视频（保证没有需要剔除的），按照最终模型
处理测试集的方式处理输入视频，然后输入给模型，得到最后的预测结果」。

「同样的处理方式」不是重新实现一遍，而是**直接调用训练时用的那几个函数**，避免
两份实现漂移（2026-10-05 掩码 min 事故的教训就是同一件事在两条路径里各写一遍）：
  - 切片段与选 48 槽：`build_units.pick_slots` / `slot_ok`（等间隔 linspace + 最近合法槽）
  - 抽帧：与 `build_units.extract_unit` 同规则（整条视频顺序解码、路过各片段槽写
    16 帧，缩放 256×256；2026-10-05 起弃用逐槽 seek——HEVC 随机访问会静默解出坏块）
  - 提特征：`extract_feats.TF` / `make_backbone`（冻结 r2plus1d_18，112×112）
  - 打分：`--line` 选哪条训练线（与 `score_units.py` 的 `--head` 同口径）
      `--line mil`（默认）  读 `mil_f*.pt`，`MilHead` 输出直接就是分数
      `--line strong`       读 `strong_f*.pt`，`StrongHead` 输出是 logits，
                            要过一次 Sigmoid 才是分数（与 train_strong 同口径）
      `--line ft`           读 `ft_strong_f*.pt`（整网微调，EXP-009 / EXP-014）。
                            这条线的骨干参数也训练过，所以不能只算一次特征：每个片段
                            的 16 帧整段过整张网。建网与载权重复用
                            `clip_scores_ft.build`（那份代码带回归门禁，是仓库里唯一
                            验证过与训练同口径的加载路径）。

  片段内的取帧间隔由 run 的 metrics.json 确定：v2 是连续 16 帧，v3 是每隔
  round(0.5 × 实测帧率) 帧取一张。训练来源无法判定时拒绝出分。

禁区与打分（2026-10-28 起按训练同款规则，research-auditor 决策复核采纳「推理时现场
重算同一套禁区」）：训练时落在下列四类禁区里的片段不参与打分——
  1. 人工标的剪辑区间（events.csv 的秒区间，前后各外扩 `CUT_MARGIN_S`=1 秒）；
  2. 低灰度帧（灰度标准差 < `QC_STD_LOW`=20，前后各 16 帧并入禁区）；
  3. 时间戳倒步（PTS 回退处前后各 64 帧，20 fps 下约 ±3.2 秒）；
  4. 时间空洞（相邻帧间隔超过 max(0.2 秒, 3×帧间隔中位数) 处，前后各 1 秒）。
本工具现场重算同一套禁区：函数与阈值全部取自建库脚本（`build_units_v2` 的
`secs_to_ranges`/`pts_jump_zones`/`pts_gap_zones`/`zones_from_frames`，加
`build_units.QC_STD_LOW`），**不在本文件重写任何阈值**。选片段前先顺序解码整条视频，
在 256×256 全帧上逐帧查灰度并建立禁区。新输入默认先用训练同款命令转码为
中间视频（即使原视频也是 256×256），保留转码日志和来源记录。只有调用方明确
确认输入就是已准备好的训练中间视频，才跳过转码。整条视频的分数只在可用片段上取最高，并
报出「参与打分的片段数 / 被剔除的片段数」；全部片段都落在禁区里时报错、不出分
（EXP-014 建库要求凑足 48 个片段；本工具对较短的新视频会明确报告实际片段数）。

对新视频，取不到可靠的逐帧时间戳或全片灰度扫描不完整时拒绝出分。
但 2026-10-05/06 建的 150 条池与 `unit-manifest.csv` 早于「倒步」「空洞」两类禁区
（这两个函数 2026-10-25 才加进 `build_units_v2.py`），所以池内个别视频的清单槽位
按今天的规则会落在禁区里。那是历史清单与今天的规则不同，不是推理链走样；哪些视频
属于这种情况、各牵扯几个槽位，见 `experiments/tools/oneoff/exp008-zone-audit/result.md`。

2026-10-28 之前的版本不避低灰度帧与空洞，整条视频分数直接取全部片段的最大值，
上限偏乐观、与训练口径不同，那一版的分数不要与本版混用。

`--cuts` 仍留入口：训练时的剪辑区间来自 events.csv，新视频没有这份标注，若知道
区间就传进来；不传则只按其余三类禁区剔除。

坏帧重采保留（解码失败与内容无关，任何视频都可能遇到）。

用法（仓库根，venv）：
  python experiments/tools/infer_video.py \
      --video "D:/path/to/video.mp4" \
      --run data/results/exp008/mil-v2-150-strict
  # 强标签线的 run 要加 --line strong：
  python experiments/tools/infer_video.py \
      --video "D:/path/to/video.mp4" --line strong \
      --run data/results/exp008/strong-v2-150-strict

输出：stdout 打印逐折分数与平均分；`--out` 指定目录时把 768 帧落盘供展示台取图。
被 `training_workbench.py` 的「视频推理」页签作为库调用（progress 回调报进度）。
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.paths import resolve_path  # noqa: E402
sys.path.insert(0, str(REPO / "experiments/common/video"))
from build_units import (CLIP_LEN, N_CLIPS, QC_STD_LOW, VIDEO_KEY_OF_V1_109,  # noqa: E402
                         V1_109_BAD_FROM_S, extract_unit, merge_ranges,
                         pick_slots, slot_ok)
from build_units_v2 import (pts_gap_zones, pts_jump_zones,  # noqa: E402
                            secs_to_ranges, zones_from_frames)
from build_units_stride import (blocks_for, extract as extract_stride,  # noqa: E402
                                frame_list, pick_slots as pick_stride_slots,
                                slot_ok as stride_slot_ok)
from extract_feats import TF, make_backbone                          # noqa: E402
from train_mil import MilHead                                        # noqa: E402
from train_strong import StrongHead                                  # noqa: E402

MAX_PICK_RETRY = 12        # 解码失败重采上限（与 build_units.MAX_PICK_RETRY 一致）

KNOWN_CLIP_LABEL_MD5 = {
    "075b3eb1307c0a620534ec14c048df0b": "v2",
    "07ad71057234c990fb9498a5c5e59a0c": "v2",
    "fa2c53c1a0f93d62869b1aeebbc80f4b": "v3",
}

# 推理侧剔除的禁区族，与建库脚本 build_units_v2.py 的五条一一对应：人工标的剪辑区间、
# v1_109 专属坏区块、低灰度帧、时间戳倒步、时间空洞。
# 已知历史差异（2026-10-28 实测）：冻结的 150 条池子建于 2026-10-05/06，当时建库脚本
# 还没有空洞与倒步两类（pts_gap_zones / pts_jump_zones 是 2026-10-25 才加进来的）。
# 用「cuts + v1_109 + 低灰度」三族复算，v1_109 的 48 个片段与 unit-manifest.csv 逐个
# 相同；再加上空洞禁区会剔除其中 4 个（24144/24160/24304/24624 帧位）。所以对
# EXP-009 的旧视频池，有些原候选位置会被现行规则排除；重新选 48 个位置时又可能选中
# 历史清单之外的片段，因此旧分数不能直接与旧预测比较。EXP-014 v3 建库时已使用新禁区，
# 需按该池的清单单独核对，不能沿用 EXP-009 的历史差异结论。
ZONE_FAMILIES = ("cuts", "v1_109", "lowstd", "jump", "gap")

# 两条冻结骨干线的折权重文件名与判定头：`--line` 选哪一行。
# 与 `score_units.py` 的 HEADS 表同口径（那边按 `--head` 选），改一处要同步另一处。
# 第三条线 `ft`（整网微调，EXP-009 / EXP-014）不在这张表里：它不是「缓存特征 × 小头」，
# 而是骨干本身也训练过，必须把每个片段的 16 帧整段送进整张网，见 load_ft_nets。
HEADS = {
    "mil": (MilHead, "mil_f*.pt", False),
    "strong": (StrongHead, "strong_f*.pt", True),   # True = 输出是 logits，要过 Sigmoid
}
FT_PATTERN = "ft_strong_f{}.pt"     # 整网微调线的折权重文件名


def probe_video(video: Path):
    """返回 (总帧数, 帧率)。总帧数取容器声明值，为 0 时用逐帧数兜底。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    if n <= 0:                       # 容器没写帧数：顺序读到结束
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
    cap.release()
    return n, fps


def probe_pts(video: Path):
    """整条视频逐帧时间戳（秒），用 ffmpeg 的 showinfo 取；取不到时返回 None。

    cv2 的 CAP_PROP_POS_MSEC 在部分容器上对逐帧时间戳不可靠（整段给常数或用元数据
    帧率推算），训练侧建库时一律用 showinfo 的 pts_time（`build_units_src256_hw.
    probe_frames` 同法），这里沿用同一条路。成本 = 整条视频解码一遍。

    任一帧给不出 pts_time（N/A，记为 -1）时整条返回 None；调用方拒绝出分。
    """
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    cmd = [ffmpeg, "-hide_banner", "-v", "info", "-i", str(video),
           "-vf", "showinfo", "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              errors="replace", encoding="utf-8")
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    pat = re.compile(r"n:\s*(\d+) pts:\s*\S+ pts_time:\s*([0-9.]+|N/A)")
    pts = []
    frame_numbers = []
    for ln in proc.stderr.splitlines():
        m = pat.search(ln)
        if m:
            frame_numbers.append(int(m.group(1)))
            pts.append(-1.0 if m.group(2) == "N/A" else float(m.group(2)))
    if (not pts or any(t < 0 for t in pts)
            or frame_numbers != list(range(len(pts)))):
        return None
    return pts


def sampling_from_run(run_dir: Path, kind: str) -> str:
    """从训练产物确认片段内采样方式；证据不够时拒绝猜测。"""
    run_dir = resolve_path(run_dir)
    path = run_dir / "metrics.json"
    if not path.is_file():
        raise SystemExit(f"{run_dir} 缺 metrics.json，无法确定训练时怎样抽帧，拒绝出分")
    try:
        meta = json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, OSError) as exc:
        raise SystemExit(f"读不出 {path}，拒绝出分：{exc}") from exc
    script = {"mil": "train_mil.py", "strong": "train_strong.py",
              "ft": "train_ft.py"}.get(kind)
    if meta.get("script") != script:
        raise SystemExit(f"{run_dir} 的训练线与 --line {kind} 不符，拒绝出分")
    if kind == "ft":
        digest = meta.get("md5", {}).get("clip_labels")
        sampling = KNOWN_CLIP_LABEL_MD5.get(digest)
        label_name = resolve_path(meta["clip_labels"]).name if meta.get("clip_labels") else ""
        name_sampling = ("v3" if label_name == "clip-labels-v3-128.npz" else
                         "v2" if label_name in ("clip-labels-v2-128.npz", "clip-labels.npz")
                         else None)
        if sampling and sampling == name_sampling:
            return sampling
    elif ("v2" in run_dir.name and meta.get("n_unit") == 150
          and "EXP-008-units-folds" in str(meta.get("folds_source", ""))):
        # 早期冻结线没有输入文件指纹；run 名、150 条和折来源共同确定历史 v2 池。
        return "v2"
    raise SystemExit(f"{run_dir} 缺少足以确定 v2/v3 采样方式的训练记录，拒绝出分")


def first_frame_size(video: Path):
    """返回实际解出的首帧尺寸；读不到时交由训练转码函数尝试处理。"""
    cap = cv2.VideoCapture(str(video))
    try:
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        return (frame.shape[1], frame.shape[0]) if ok and frame is not None else None
    finally:
        cap.release()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_input_video(source: Path, progress=None, prepared: bool = False):
    """原始视频按训练命令转码；返回 (实际输入视频, 转码记录路径或 None)。

    每次调用单独建目录，避免两次推理覆盖中间件或日志。转码失败的记录也保留。
    """
    if prepared:
        if first_frame_size(source) != (256, 256):
            raise SystemExit("已准备好的输入必须是 256×256 训练中间视频，拒绝出分")
        return source, None
    from build_units_src256 import run_ffmpeg  # 直接使用训练侧的命令定义

    if progress:
        progress(0.005, "按训练流程转为 256×256 视频")
    source_digest = file_sha256(source)
    artifact_dir = (REPO / ".codex/docs/.temp/infer-video/transcodes"
                    / f"{source_digest[:12]}-{uuid.uuid4().hex[:8]}")
    artifact_dir.mkdir(parents=True, exist_ok=False)
    middleware = artifact_dir / "middleware.mp4"
    ffmpeg_log = artifact_dir / "ffmpeg.log"
    provenance = artifact_dir / "source.json"
    record = {
        "source": str(source.resolve()),
        "source_size_bytes": source.stat().st_size,
        "source_sha256": source_digest,
        "middleware": str(middleware.resolve()),
        "ffmpeg_log": str(ffmpeg_log.resolve()),
        "transcode_function": "experiments/tools/build_units_src256.py:run_ffmpeg",
        "transcode_arguments": ["infer", str(source.resolve()),
                                 str(middleware.resolve()), str(ffmpeg_log.resolve())],
        "transcode_parameters": "scale=256:256, -vsync 0, libx264, crf 10, g 30, -an",
        "reason": "与 EXP-008 建库使用同一条 256×256 转码流程",
        "status": "started",
    }
    provenance.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        rc, warns, seconds = run_ffmpeg("infer", source, middleware, ffmpeg_log)
        record.update(return_code=rc, warning_lines=len(warns),
                      duration_sec=round(seconds, 3))
        if rc != 0 or not middleware.is_file() or middleware.stat().st_size == 0:
            raise SystemExit(f"转码失败（退出码 {rc}），拒绝出分；日志：{ffmpeg_log}")
        if first_frame_size(middleware) != (256, 256):
            raise SystemExit(f"转码结果不是 256×256，拒绝出分；日志：{ffmpeg_log}")
        if file_sha256(source) != source_digest:
            raise SystemExit("原视频在转码过程中发生变化，拒绝出分")
        record.update(status="transcoded", middleware_sha256=file_sha256(middleware),
                      middleware_size_bytes=middleware.stat().st_size)
        return middleware, provenance
    except BaseException as exc:
        if not ffmpeg_log.exists():
            ffmpeg_log.write_text(str(exc), encoding="utf-8")
        record.update(status="failed", error=str(exc))
        raise
    finally:
        provenance.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                              encoding="utf-8")


def update_provenance(path, **fields):
    if path is None:
        return
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    record.update(fields)
    Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def scan_gray_video(video: Path, n_expected: int, progress=None):
    """与建库质检相同：256×256 全帧逐帧灰度 std；非训练尺寸或帧序错位即拒绝。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"打不开视频，无法逐帧检查：{video}")
    lowstd = []
    n_read = 0
    try:
        while n_read < n_expected:
            ok, fr = cap.read()
            if not ok or fr is None:
                raise SystemExit(f"逐帧检查在 {n_read}/{n_expected} 帧中断，拒绝出分")
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != n_read + 1:
                raise SystemExit(f"逐帧检查在第 {n_read} 帧失步，拒绝出分")
            if fr.shape[:2] != (256, 256):
                raise SystemExit(
                    f"输入视频第 {n_read} 帧为 {fr.shape[1]}×{fr.shape[0]}，"
                    "训练使用先转码成 256×256 的中间件；请先走训练转码流程再推理，拒绝出分")
            std = round(float(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).std()), 1)
            if std < QC_STD_LOW:
                lowstd.append(n_read)
            n_read += 1
            if progress and n_read % 1000 == 0:
                progress(0.02, f"逐帧检查 {n_read}/{n_expected}")
        if cap.grab():
            raise SystemExit(f"逐帧检查发现视频超过时间戳的 {n_expected} 帧，拒绝出分")
    finally:
        cap.release()
    return lowstd


def training_zones(video: Path, pts, n_frames: int, cuts=()):
    """按训练同款规则构造禁取帧区间 [(起, 止) 帧号，闭开)。返回 (区间表, 倒步处数, 空洞处数)。

    四个构造函数全部 import 自建库脚本、参数一律用默认值：在本文件重写阈值（例如把
    倒步外扩写成 ±1 秒、把空洞阈值写死 0.2 秒）会得到与训练不同的禁区，正是 2026-10
    两次口径事故的形态。
    """
    zones = list(secs_to_ranges(cuts, pts)) if cuts else []
    # v1_109 的专属坏区块（门前沟中游 19:57 之后整段坏），训练建库时按这条视频单独
    # 加了一条禁区。仅当文件名匹配 v1_109 时生效，新视频不会命中。
    if VIDEO_KEY_OF_V1_109 in Path(video).name:
        fb = next((j for j, t in enumerate(pts) if t >= V1_109_BAD_FROM_S), n_frames)
        if fb < n_frames:
            zones.append((fb, n_frames))
    jump = pts_jump_zones(pts)
    gaps = pts_gap_zones(pts)
    return merge_ranges(zones + jump + gaps), len(jump), len(gaps)


def slots_from_sidecar(frames_dir: Path, n_clips: int, block: int = CLIP_LEN):
    """复用已有帧目录时，从 `slots.txt` 旁证恢复每个片段对应的起始帧号（槽号）。

    `extract_unit` / 本工具调用新间隔抽帧后都会落这份旁证，内容是
    「片段起点的全局帧号」用分号连接。缺这份文件时返回 None：此时片段与视频帧位的
    对应关系不可知，拒绝出分。
    """
    p = frames_dir / "slots.txt"
    if not p.is_file():
        return None
    try:
        starts = [int(x) for x in p.read_text(encoding="utf-8").strip().split(";") if x]
    except ValueError:
        return None
    if len(starts) != n_clips or any(s % block for s in starts):
        return None
    return [s // block for s in starts]


def clip_usable(slots, n_clips, forb, lowstd, block: int = CLIP_LEN):
    """每个片段是否参与打分：不在任何禁区里，且 16 帧里没有低灰度帧。

    复用帧目录必须有有效 slots.txt；调用方已验证。
    """
    bad = {ci for ci, _fi, _sd in lowstd}
    out = []
    for ci in range(n_clips):
        ok = (True if slots is None else
              (slot_ok(slots[ci], forb) if block == CLIP_LEN
               else stride_slot_ok(slots[ci], block, forb)))
        out.append(bool(ok and ci not in bad))
    return out


def read_frame(path: Path) -> np.ndarray:
    """读一帧并转 RGB。用 imdecode 而非 cv2.imread——后者对含中文的路径会静默返回
    None（training_workbench 的缩略图服务已是这个写法）。"""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读不出帧：{path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract_features(frames_dir: Path, backbone, n_clips: int, progress=None):
    """逐片段前向 -> [n_clips, 512]。

    预处理逐位对齐 `extract_feats.load_clip`：112×112 INTER_LINEAR + Kinetics 归一化。
    与它的唯一差别是读帧用 imdecode（中文路径鲁棒），解码结果与 imread 相同。
    """
    out = np.zeros((n_clips, 512), dtype=np.float32)
    with torch.inference_mode():
        for ci in range(n_clips):
            frames = []
            for fi in range(CLIP_LEN):
                img = read_frame(frames_dir / f"clip{ci:02d}_f{fi:02d}.jpg")
                img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
                frames.append(TF(img))
            x = torch.stack(frames, dim=1).unsqueeze(0)     # [1,3,16,112,112]
            out[ci] = backbone(x).squeeze(0).cpu().numpy()
            if progress:
                progress(0.60 + 0.35 * (ci + 1) / n_clips,
                         f"提特征 {ci + 1}/{n_clips}")
    return out


def select_weight_files(files, folds, run_dir: Path, label: str):
    """默认要求 1..5 齐全；指定折时只返回所需文件。"""
    found = {}
    for path in files:
        try:
            fold = int(path.stem.rsplit("f", 1)[-1])
        except ValueError:
            continue
        found.setdefault(fold, []).append(path)
    required = list(range(1, 6)) if folds is None else list(folds)
    if not required or len(set(required)) != len(required) or any(
            not isinstance(f, int) or f not in range(1, 6) for f in required):
        raise SystemExit(f"请求的折号必须是 1..5 且不能重复：{required}")
    missing = [f for f in required if f not in found]
    if missing:
        raise SystemExit(f"{run_dir} 缺第 {missing} 折的 {label} 权重，拒绝出分")
    duplicates = [f for f in required if len(found[f]) != 1]
    if duplicates:
        raise SystemExit(f"{run_dir} 第 {duplicates} 折有重复的 {label} 权重，拒绝出分")
    extra = sorted(set(found) - set(required)) if folds is None else []
    if extra:
        raise SystemExit(f"{run_dir} 多出第 {extra} 折的 {label} 权重，无法确认五折模型，拒绝出分")
    return [(f, found[f][0]) for f in required]


def load_ft_nets(run_dir: Path, pattern: str = FT_PATTERN, folds=None):
    """整网微调模型 {折: 网络}，均置 eval。

    与上面两条线的区别：微调线一个权重文件装的是整张网（骨干 + 判定头），骨干参数也
    参与过训练，所以不能像 `extract_features` 那样先算一次 512 维特征再乘小头——每个
    片段的 16 帧都要整段过骨干（见 `score_clips_ft`）。

    建网与载权重直接复用 `clip_scores_ft.build`：那份代码带回归门禁（重算的整条视频
    分数必须与 run 目录的 predictions.csv 逐条对上），是仓库里唯一验证过与训练同口径的
    加载路径，这里不再抄第三份。
    """
    run_dir = resolve_path(run_dir)
    selected = select_weight_files(sorted(Path(run_dir).glob(pattern.format("*"))),
                                   folds, Path(run_dir), "ft")
    from clip_scores_ft import build          # 同目录，避免第三份网络定义
    nets = {}
    for fold, w in selected:
        nets[fold] = build(Path(run_dir), fold, pattern)
    return nets


def score_clips_ft(frames_dir: Path, nets, n_clips: int, progress=None):
    """整网微调线逐片段打分 -> {折: [片段分数]}，分数已过 Sigmoid。

    取帧与预处理与两条冻结骨干线共用同一段代码（`read_frame` + 112×112 双线性 +
    `extract_feats.TF` 的 Kinetics 归一化），所以三条线的差别只在模型，不在输入。
    """
    out = {f: np.zeros(n_clips, dtype=np.float32) for f in nets}
    with torch.inference_mode():
        for ci in range(n_clips):
            frames = []
            for fi in range(CLIP_LEN):
                img = read_frame(frames_dir / f"clip{ci:02d}_f{fi:02d}.jpg")
                img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
                frames.append(TF(img))
            x = torch.stack(frames, dim=1).unsqueeze(0)     # [1,3,16,112,112]
            for f, net in nets.items():
                logit = net.head(net.backbone(x)).squeeze(-1).reshape(-1)
                out[f][ci] = float(torch.sigmoid(logit)[0])
            if progress:
                progress(0.60 + 0.35 * (ci + 1) / n_clips,
                         f"逐片段过整网 {ci + 1}/{n_clips}")
    return out


def load_fold_heads(run_dir: Path, kind: str = "mil", folds=None):
    """run 目录下的全部折模型 {fold: 判定头}，均置 eval（关 Dropout，分数确定性）。

    kind 由 HEADS 表选：`mil` 读 `mil_f*.pt` + `MilHead`（输出即分数）；
    `strong` 读 `strong_f*.pt` + `StrongHead`（输出是 logits，见 infer 里的 Sigmoid）。
    """
    head_cls, pattern, _is_logits = HEADS[kind]
    run_dir = resolve_path(run_dir)
    selected = select_weight_files(sorted(Path(run_dir).glob(pattern)),
                                   folds, Path(run_dir), kind)
    heads = {}
    for fold, w in selected:
        h = head_cls()
        h.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
        h.eval()
        heads[fold] = h
    return heads


def fold_record(s, keep):
    """一个模型对这条视频的打分记录。

    片段分数全量保留（展示台画全部柱子），整条视频的分数只在可用片段上取最高——
    与训练侧的 `aggregate_unit_score` 同一条规则（落在禁区里的片段不参与打分）。

    keep 必须是**可用片段的下标数组**（如 `np.array([0,1,3,...])`），不是布尔掩码：
    `s[keep]` 取可用片段的分数，`keep[argmax]` 由此直接得到原片段号。传布尔掩码也能
    跑通、但取回的片段号会是 0/1，所以这里显式断言成整数数组。
    """
    keep = np.asarray(keep)
    assert keep.dtype.kind in "iu", "keep 要传可用片段的下标数组，不要传布尔掩码"
    s = np.asarray(s, dtype=float)
    s_use = s[keep]
    j = int(keep[int(s_use.argmax())])
    return {"clip_scores": [round(float(v), 6) for v in s],
            "unit_score": round(float(s_use.max()), 6),
            "argmax_clip": j}


def infer(video: Path, run_dir: Path, out_dir=None, cuts=(), folds=None,
          progress=None, kind: str = "mil", reuse_frames=None,
          prepared: bool = False):
    """完整推理一条视频。返回结果 dict（供 CLI 打印与展示台渲染）。

    folds：只用的折号列表（None = 全部）。往返自检要指定单折，因为一条视频
      在 predictions.csv 里只被它自己那一折的模型打过一次分。

    reuse_frames：直接用一个已经抽好帧的目录，跳过再次抽帧；时间戳与全片灰度
      仍重新检查。这项只给往返自检用，已登记中间件要同时传 prepared=True
      ——拿训练时落盘的帧目录再打一次分，结果须在该训练线的容差内一致、
      且 0.5 判定线不翻转，
      以证明「模型加载 + 预处理」这一环没走样。正常给新视频打分不要用它。

    progress(frac, msg) 可选，用于展示台轮询进度。
    """
    def pg(f, m):
        if progress:
            progress(f, m)

    source, run_dir = resolve_path(video), resolve_path(run_dir)
    sampling = sampling_from_run(run_dir, kind)
    if not source.is_file():
        raise SystemExit(f"视频不存在：{source}")
    video, provenance = prepare_input_video(source, progress=pg, prepared=prepared)
    pg(0.01, "读取逐帧时间戳")
    pts = probe_pts(video)
    if pts is None:
        update_provenance(provenance, status="rejected", error="逐帧时间戳不可靠")
        raise SystemExit("读不到可靠的逐帧时间戳，无法排除倒步与时间空洞，拒绝出分")
    try:
        n_frames_cv, fps_cv = probe_video(video)
    except Exception as exc:
        update_provenance(provenance, status="rejected", error=f"视频探测失败：{exc}")
        raise SystemExit(f"转码后视频无法探测帧数，拒绝出分：{exc}") from exc
    if abs(len(pts) - n_frames_cv) > 1:
        update_provenance(provenance, status="rejected",
                          error=f"时间戳 {len(pts)} 条与视频 {n_frames_cv} 帧不符")
        raise SystemExit(f"时间戳 {len(pts)} 条与视频声明的 {n_frames_cv} 帧不符，拒绝出分")
    n_frames = len(pts)
    if n_frames < CLIP_LEN:
        raise SystemExit(f"视频太短：不足 {CLIP_LEN} 帧，切不出片段")
    if sampling == "v3":
        measured_fps = ((n_frames - 1) / pts[-1] if pts[-1] > 0 else 10.0)
        stride = max(1, int(round(0.5 * measured_fps)))
        block = blocks_for(stride)
    else:
        stride, block = 1, CLIP_LEN
    fps = fps_cv if fps_cv > 0 else measured_fps if sampling == "v3" else 0.0
    if fps <= 0:
        raise SystemExit(f"读不到帧率：{video}")
    n_slots = n_frames // block
    if n_slots < 1:
        raise SystemExit(f"视频太短：按 {sampling} 的取帧方式切不出片段")
    pg(0.02, "逐帧检查低灰度画面")
    try:
        lowstd_frames = scan_gray_video(video, n_frames, progress=pg)
    except BaseException as exc:
        update_provenance(provenance, status="rejected", error=str(exc))
        raise
    update_provenance(provenance, status="validated", n_frames=n_frames,
                      n_lowstd_frames=len(lowstd_frames))
    lowstd, degen = [], []
    zone_rounds, pts_ok = 0, True
    forb, n_jump, n_gap = training_zones(source, pts, n_frames, cuts)
    forb = merge_ranges(forb + zones_from_frames(lowstd_frames, n_frames))
    if reuse_frames is not None:
        frames_dir = resolve_path(reuse_frames)
        if not frames_dir.is_dir():
            raise SystemExit(f"帧目录不存在：{frames_dir}")
        n_clips = len(list(frames_dir.glob("clip*_f00.jpg")))
        if n_clips < 1:
            raise SystemExit(f"{frames_dir} 里没有 clip*_f00.jpg，不是抽好帧的目录")
        retries, slots = 0, None
        pg(0.60, f"复用已抽好的 {n_clips} 个片段")
        slots = slots_from_sidecar(frames_dir, n_clips, block)
        if slots is None:
            raise SystemExit("复用帧目录缺有效 slots.txt，无法核对禁区，拒绝出分")
        usable = clip_usable(slots, n_clips, forb, lowstd, block)
        if sum(usable) < 1:
            raise SystemExit(
                f"这条视频在训练口径下不可训练：{n_clips} 个片段全部落在禁区里"
                f"（低灰度帧、时间戳倒步、时间空洞或人工标的剪辑区间），分数无意义。")
    else:
        max_retry = 6 if sampling == "v3" else MAX_PICK_RETRY
        work = Path(out_dir) if out_dir else REPO / ".codex/docs/.temp/infer-video"
        work.mkdir(parents=True, exist_ok=True)
        frames_dir = work / "frames"
        slots, retries, fail_at = None, 0, None
        while True:
            # 2026-10-27 补：pick_slots 要求 k 个片段互不相同，所以合法槽只有 m 个时
            # k>m 必然报错，哪怕这条视频并不短——只是禁区（人工标的剪辑区间、坏帧邻域）
            # 吃掉了大部分时长。这里先把 k 压到「实际可用的合法槽数」，才不会因为切片
            # 数量不够而拒绝出分（与 2026-10-14「不足 48 按实际槽数取片段」同一条口径）。
            legal = (lambda s: slot_ok(s, forb) if sampling == "v2"
                     else stride_slot_ok(s, block, forb))
            n_legal = sum(1 for s in range(n_slots) if legal(s))
            if n_legal < 1:
                raise SystemExit(
                    f"这条视频在训练口径下不可训练：{n_slots} 个候选位置全部落在禁区里"
                    f"（人工标的剪辑区间、低灰度帧、时间戳倒步或时间空洞的邻域），"
                    f"分数无意义。")
            n_clips = min(N_CLIPS, n_slots, n_legal)
            slots = (pick_slots(n_slots, forb, k=n_clips) if sampling == "v2"
                     else pick_stride_slots(n_slots, block, forb, k=n_clips))
            if len(slots) != n_clips:
                raise SystemExit(f"只能取到 {len(slots)}/{n_clips} 个片段，拒绝出分")
            if frames_dir.exists():
                shutil.rmtree(frames_dir, ignore_errors=True)
            # v2 调用原连续抽帧函数；v3 调用实验使用的按帧号列表抽帧函数。
            # 抽帧占整条推理的大头，这里把片段计数映射到 0.02→0.60 那一段进度上，
            # 否则进度条会在 1% 停很久。
            if sampling == "v2":
                n_written, fail, fail_frames = extract_unit(
                    video, slots, frames_dir, resume=False,
                    progress=lambda d, t: pg(0.02 + 0.58 * d / t, f"抽帧 {d}/{t}"))
            else:
                starts = ";".join(str(s * block) for s in slots)
                n_written, n_scanned, fail, fail_frames = extract_stride(
                    video, frame_list(slots, block, stride), frames_dir,
                    resume=False, expect_key=starts)
                if fail is None and n_scanned != n_frames:
                    raise SystemExit(f"抽帧时扫描 {n_scanned}/{n_frames} 帧，帧序不一致，拒绝出分")
                if fail is None:
                    (frames_dir / "slots.txt").write_text(starts, encoding="utf-8")
            if fail is None and n_written != n_clips * CLIP_LEN:
                raise SystemExit(f"抽帧只写出 {n_written}/{n_clips * CLIP_LEN} 张，拒绝出分")
            if fail is not None:
                if retries >= max_retry:
                    # 触顶拒绝出分：给一个「抽帧失败但仍算出来的」分数正是静默产出
                    # 错误成绩的翻版
                    raise SystemExit(
                        f"换位置重新抽帧 {max_retry} 次仍失败（末次坏帧 {fail_at}），"
                        f"拒绝出分——请检查视频是否可完整解码")
                zones = sorted(set(fail_frames)) or [fail]
                if sampling == "v3":
                    forb = merge_ranges(forb + zones_from_frames(
                        zones, n_frames, pad=16 * 2 ** retries))
                else:
                    grow = 0.5 * (2 ** retries)
                    for f in zones:
                        forb.append((max(0, int(f - grow * fps)),
                                     min(n_frames, int(f + CLIP_LEN + grow * fps))))
                retries += 1
                fail_at = fail
                pg(0.05, f"第 {retries} 次换位置重新抽帧（坏帧 {fail}）")
                continue
            break

        usable = clip_usable(slots, n_clips, forb, lowstd, block)
        # 全片低灰帧已在选片段前并入禁区，不会写进候选片段。
        degen = []
        if sum(usable) < 1:
            raise SystemExit(
                f"这条视频在训练口径下不可训练：{n_clips} 个片段全部落在禁区里"
                f"（人工标的剪辑区间、低灰度帧、时间戳倒步或时间空洞的邻域），"
                f"分数无意义。")
    n_usable = int(sum(usable))
    keep = np.array([i for i, ok in enumerate(usable) if ok], dtype=int)
    per_fold = {}
    if kind == "ft":
        pg(0.60, "整网打分")
        nets = load_ft_nets(run_dir, folds=folds)
        for fold, s in score_clips_ft(frames_dir, nets, n_clips, progress=pg).items():
            per_fold[str(fold)] = fold_record(s, keep)
    else:
        heads = load_fold_heads(run_dir, kind, folds=folds)
        pg(0.60, "提特征")
        feats = extract_features(frames_dir, make_backbone(), n_clips, progress=pg)
        pg(0.96, "打分")
        X = torch.tensor(feats[None], dtype=torch.float32)       # [1,n_clips,512]
        is_logits = HEADS[kind][2]
        with torch.inference_mode():
            for fold, h in heads.items():
                # 2026-10-15 审查更正：原来写 squeeze()，只有 1 个可用片段时会把整个
                # 张量压成标量，后面 s.max()/s.argmax() 直接崩。两条训练线共用此处。
                out = h(X).squeeze(-1).reshape(-1)               # [n_clips]
                if is_logits:
                    out = torch.sigmoid(out)    # 强标签线判定头输出 logits，过 Sigmoid 才是分数
                per_fold[str(fold)] = fold_record(out.numpy(), keep)
    pg(1.0, "完成")
    return {
        "video": str(source), "middleware": str(video),
        "transcode_record": str(provenance) if provenance else None,
        "prepared_input": prepared,
        "run": Path(run_dir).name, "line": kind,
        "n_frames": n_frames, "fps": round(fps, 3),
        "sampling": sampling, "frame_stride": stride,
        "n_lowstd_frames": len(lowstd_frames),
        "duration_sec": round(n_frames / fps, 1) if fps else None,
        "n_clips": n_clips,
        "n_usable": n_usable,
        "n_excluded": int(n_clips - n_usable),
        "usable": usable,
        "n_slots": n_slots,
        "slots": [int(s * block) for s in slots] if slots else None,
        "retries": retries, "zone_rounds": zone_rounds,
        "frames_dir": str(frames_dir),
        "degenerate_clips": degen,
        "lowstd_clips": sorted({ci for ci, _f, _s in lowstd}),
        "pts_ok": pts_ok, "n_jump_zones": n_jump, "n_gap_zones": n_gap,
        "forbidden_ranges": [[int(a), int(b)] for a, b in forb],
        "per_fold": per_fold,
        "elapsed_sec": None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True,
                    help="输入视频路径；默认按训练流程转码")
    ap.add_argument("--prepared", action="store_true",
                    help="明确确认输入是已准备好的 256×256 训练中间视频，跳过转码")
    ap.add_argument("--line", choices=["mil", "strong", "ft"], default="mil",
                    help="哪条训练线：mil=弱标签线（mil_f*.pt，输出即分数）；"
                         "strong=强标签线（strong_f*.pt，输出是 logits，过一次 "
                         "Sigmoid）；ft=整网微调线（ft_strong_f*.pt，骨干参数也训练过，"
                         "每个片段整段过整张网）。默认 mil，行为与加本参数前逐位相同")
    ap.add_argument("--run", required=True,
                    help="run 目录（--line mil 含 mil_f*.pt；--line strong 含 "
                         "strong_f*.pt；--line ft 含 ft_strong_f*.pt）")
    ap.add_argument("--out", default="", help="帧落盘目录（展示台取图用）；不给则用临时目录")
    ap.add_argument("--reuse-frames", default="",
                    help="直接用一个已抽好帧的目录，跳过再次抽帧（往返自检用："
                         "拿训练落盘的帧再打一次分，按该训练线容差核对分数与判定）")
    ap.add_argument("--cuts", default="",
                    help="可选：cut 区间「起,止;起,止」（秒），给了就照训练口径剔除")
    ap.add_argument("--fold", type=int, default=0,
                    help="只用第 N 折的模型打分（往返自检用；0 = 全部 5 个模型）")
    ap.add_argument("--json", action="store_true", help="额外打印一行 JSON 结果")
    args = ap.parse_args()

    cuts = []
    if args.cuts:
        for part in args.cuts.split(";"):
            if part.strip():
                a, b = part.split(",")
                cuts.append((float(a), float(b)))

    t0 = time.time()
    res = infer(Path(args.video), Path(args.run), Path(args.out) if args.out else None,
                cuts, [args.fold] if args.fold else None,
                progress=lambda f, m: print(f"  [{f * 100:5.1f}%] {m}"),
                kind=args.line,
                reuse_frames=Path(args.reuse_frames) if args.reuse_frames else None,
                prepared=args.prepared)
    res["elapsed_sec"] = round(time.time() - t0, 1)

    print(f"\n视频 {res['video']}")
    if res["transcode_record"]:
        print(f"  转码后视频 {res['middleware']}；来源与转码日志 {res['transcode_record']}")
    elif res["prepared_input"]:
        print("  输入按已准备好的训练中间视频使用，调用方明确跳过转码")
    if not args.reuse_frames:
        print(f"  {res['n_frames']} 帧 / {res['fps']} fps / {res['duration_sec']} 秒"
              f"（坏帧重抽 {res['retries']} 次）")
    else:
        print(f"  复用已抽好的帧目录：{res['frames_dir']}")
    print(f"  取帧方式 {res['sampling']}，片段内每隔 {res['frame_stride']} 帧取一张；"
          f"全片发现 {res['n_lowstd_frames']} 张低灰度帧并避开")
    print(f"  切出 {res['n_clips']} 个片段，其中 {res['n_usable']} 个参与打分"
          + (f"（另外 {res['n_excluded']} 个落在禁区里，按训练同款规则剔除）"
             if res["n_excluded"] else ""))
    if res["n_clips"] != N_CLIPS:
        print(f"  注意：片段数不是 {N_CLIPS} 个（视频较短或禁区较多），分数按这 "
              f"{res['n_usable']} 个可用片段取最高，与训练时 {N_CLIPS} 片段口径的"
              f"分数只可参考、不可直接对表")
    print(f"  禁区：人工标的剪辑区间 + 低灰度帧 + 时间戳倒步 {res['n_jump_zones']} 处"
          f" + 时间空洞 {res['n_gap_zones']} 处；逐帧时间戳已核对")
    if res.get("n_slots") and res["n_clips"] < res["n_slots"] \
            and res["n_clips"] != min(N_CLIPS, res["n_slots"]):
        print(f"  注意：整条视频有 {res['n_slots']} 个候选位置，其中 "
              f"{res['n_slots'] - res['n_clips']} 个落在禁区里（人工标的剪辑区间、"
              f"低灰度帧、时间戳倒步或时间空洞的邻域），只用剩下的 "
              f"{res['n_clips']} 个取片段。片段越少，整条视频的分数越不可靠。")
    print(f"模型 {res['run']}（{len(res['per_fold'])} 个模型）")
    scores = []
    for fold, v in res["per_fold"].items():
        scores.append(v["unit_score"])
        print(f"  第 {fold} 折的模型: 整条视频得分 {v['unit_score']:.4f}"
              f"（最高片段 #{v['argmax_clip']}）")
    if len(scores) > 1:
        votes = sum(1 for s in scores if s >= 0.5)
        print(f"  判成「有泥石流」的模型数 {votes}/{len(scores)}（得分超过 0.5 算一票）")
        print(f"  原始分平均 {np.mean(scores):.4f}（各折模型打分松紧不同，仅供参考，"
              f"展示台另给换成百分位后的综合分）")
    if res["degenerate_clips"]:
        print(f"  ⚠ 片段 {res['degenerate_clips']} 含低灰度帧（灰度标准差 < "
              f"{QC_STD_LOW}），已按训练同款规则剔除、不参与打分")
    print(f"\n说明：没有单独训练一个「最终模型」。训练时把全部视频分成 5 份轮流当考试题，"
          f"得到 5 个模型，上面就是这 5 个模型各自对这条视频的判断。"
          f"参考成绩 = 5 折按条视频算的 AUC（验证折同时用来选模型，成绩偏乐观）。"
          f"分数没有校准成概率，只用于横向比较。")
    if args.json:
        print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # 报错信息（SystemExit 的字符串）走 stderr；不重设的话 Windows 控制台按 GBK 写出，
    # 日志存成文件再按 UTF-8 看就是乱码
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
