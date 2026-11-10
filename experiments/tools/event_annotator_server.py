"""本地事件区间标注服务器:浏览器里逐帧浏览拆好的帧,标注泥石流出现的起止秒区间。

纯标准库 http.server + cv2(仅缩略图缩放用),只监听 127.0.0.1、不联网。
前端 HTML/JS/CSS 内嵌在本文件的字符串里;主视图帧图由服务端按请求原样回传字节
(与 label_audit_server 相同,规避 Windows 中文路径的编码问题)。

读入(来自 extract_frames.py 的产出,--dir 指向 data/process/video_frames/video1_segment/
  或其中某个视频目录;2026-09-16 data 结构重构后旧 data/frames/ 已并入该目录,
  迁移期 --dir 仍可手动指旧路径):
  frames/             frame_%06d_t{秒}s.jpg
  frames_index.csv    frame_file,t_sec,sheet(必需,时间轴按它展开)
  events.csv          video_id,start_sec,end_sec,confidence,kind,note(标注结果,即时重写)

区间类型(kind 列):
  pos  明确正样本 P——泥石流已明确可见的时段;
  transition  显现发展 T——有过程证据支持的泥石流显现、发展时段;
  unknown  历史类型未定 U——仅兼容已有记录，不开放新的人工选标入口;
  cut  剔除区间——雾大、全黑、遮挡等不适合进数据集的时段,既不作正也不作负;
  neg  明确负样本区间——画面确认无泥石流且要作负帧的时段(帧的标签与
       「区间外默认负」相同,但把显式声明落进 events.csv,使整段无事件的
       视频也能一键标为已处理,状态从「未标」变「已标」)。
  旧 5 列 CSV(无 kind 列)读入时按 pos 处理。四种类型两两之间同样禁止
  重叠(允许首尾相接);整段负样本视频用「整段标负」按钮写一条 0→末帧的 neg。

加载约定:
  events.csv 里 confidence=real 的行视为已确认区间(带 kind);
  其余行(旧预标注残留等)一律忽略不显示,下次保存即被清除——
  预标注候选机制已于 2026-09-17 整体移除(叶海峰口径:不用助手预标注,
  训练好模型后由模型预标注,且须单独存文件、人工确认后才并入 events.csv)。

多视频会话:
  GET /api/videos 列出 --dir 解析出的全部视频(汇总源按 video_key 展开,
  旧式目录按子目录),每项带 status 字段标注进度:annotated=已标(有确认区间)/
  unlabeled=未标,前端下拉框文字前缀 ✅ / ⬜ 一眼可辨;
  页面头部下拉切换(?v=<video_key 或目录名>),各视频的标注状态、缩略图缓存互相独立,
  会话按需惰性创建,缩略图预生成线程随会话启动(只写磁盘缓存,不常驻内存)。

设计取舍(两处均按任务要求选一种并说明):
  1. 重叠区间选「禁止」:新建的区间若与已有区间重叠则拒绝并提示
     (pos/cut/neg 两两之间同样禁止)。理由:标注约定允许一场事件多次龙头
     (多段正区间),合并会把相邻两段并成一段、丢失次数信息;禁止可保每段
     区间语义独立,cut/neg 段也不得吞掉相邻正区间的帧。
  2. 缩略图选「服务端磁盘缓存」:首次访问时生成约 120 张宽 160px 的小图,
     写入 <视频目录>/thumbs/(文件名与帧文件一一对应,重拆帧后旧缓存自动失效),
     之后每次刷新零重算。代价是数据目录里多一个 thumbs/ 子目录;好处是
     启动快、内存占用极小(单张小图几 KB,不常驻内存)。

页面交互:
  - 头部下拉切换视频;主视图大图 + 顶部大字时间 t = 123s (2:03) / 总长 1417s
  - 工具栏选区间类型(P/T/N/cut,默认 P),新建区间用所选类型;
    「整段标负」按钮一键写一条覆盖整个时间轴的 neg 区间;
    「上一视频/下一视频」按钮或 P/N 键按列表顺序快速切换视频(标注即时落盘,跳转不丢进度)
  - 底部缩略时间轴(均匀抽样 ≤120 张,横向滚动),正区间画绿色块、
    剔除区间画灰黑斜纹块、负样本区间画蓝色块,点击任意位置跳到该时刻
  - S 设起点 / E 设终点(先后顺序不限),只预览；点击「应用标注」才覆盖指定范围；
    C 取消待应用范围。旧区间两侧和各段备注保留，可手动逐次撤销。
  - ←/→ 步进 1 帧(即抽帧间隔 2 秒),Shift+←/→ 10 帧,
    PageUp/PageDown ±60 帧(约 2 分钟),Home/End 首末帧
  - 侧栏可直接选择类型、编辑起止小数秒与备注，点击「保存本段」生效；
    也可在当前帧真实时间分割已有区间，保留两段备注；重叠操作拒绝。
  - 时间判定统一为 [start,end)，保留原 CSV 边界及索引小数秒，不取整。
  - 当前帧类别和约 3 秒观察窗按现有人类区间显示覆盖；空隙未知，
    cut 独立显示。只浏览现有审计帧，不生成或声明模型16帧输入。
  - 每次改动立即整份重写 events.csv(video_id 用目录名,confidence 填 real),
    页面顶部显示「已保存 HH:MM:SS」,重启自动恢复

用法(仓库根目录;--dir 三种取值,前两种是 2026-09-17 新增的「汇总源」模式):
    可加 --annotations-dir <新版本目录>：按源完整复制旧CSV一次，随后只写
    版本CSV；sources.json 记录来源与初始SHA256，原帧/索引/缩略图复用。
    此模式不开放旧的隐式负场景导出，不自动修改任何已有区间。
    .venv\\Scripts\\python.exe experiments\\tools\\event_annotator_server.py ^
        --dir data\\process\\video_frames [--port 8100]
      (汇总根目录:自动发现其下全部汇总源/旧式目录,标注直写各汇总源
       的 canonical events.csv,共 125 个视频一页切换)
    .venv\\Scripts\\python.exe experiments\\tools\\event_annotator_server.py ^
        --dir data\\process\\video_frames\\video1_segment [--port 8100]
      (单个汇总源:展开其 mapping.csv 的全部 video_key,如 v1_01..v1_123)
    .venv\\Scripts\\python.exe experiments\\tools\\event_annotator_server.py ^
        --dir data\\process\\video_frames\\video1_segment\\<视频名> [--port 8100]
      (旧式单视频目录:frames/ + frames_index.csv + events.csv,行为同最早版本)

汇总源判定:目录含 mapping.csv 且 frames_index.csv 表头带 video_key 列
  (video1_segment/video3_segment 属此类;video2_segment 用独立文件名
   video2_frames_index.csv,按旧式目录处理)。key 模式下帧文件名带
  <video_key>_ 前缀平铺在汇总源 frames/ 里,events.csv/frames_index.csv
  按 video_key 列区分视频;保存时整文件读改写、只替换本 video_key 的行,
  其他视频行原样保留(带文件锁防并发覆盖)。

自测(不弄脏真实标注:全部跑在临时合成夹具上):
    .venv\\Scripts\\python.exe experiments\\tools\\event_annotator_server.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

EVENTS_FIELDS = ["video_id", "start_sec", "end_sec", "confidence", "kind", "note"]
KINDS = ("pos", "cut", "neg", "transition", "unknown")  # unknown 仅兼容历史记录，不作 N/cut 或训练真值
                                # neg=明确负样本(无泥石流且作负帧,如整段无事件视频的一键整段标负)
THUMB_COUNT = 120      # 时间轴缩略图抽样上限
THUMB_WIDTH = 160      # 缩略图宽度(px),单张几 KB
TEMPLATE_ROW = ["", "", "", "", "", "示例行:填实际起止秒后删除本行"]
REPO_ROOT = Path(__file__).resolve().parents[2]
RETIRED_SOURCES_JSON = REPO_ROOT / "data/metadata/source-retirement/retired-sources.json"
TRAINING_SCOPE_CSV = REPO_ROOT / "data/metadata/annotation-review/recheck-20261108-v2/all-dongchuan-scope.csv"


@lru_cache(maxsize=1)
def training_pool_units() -> frozenset[str]:
    """只读冻结清单的现役训练池成员；不以标签类别或复核列表推测纳入状态。"""
    if not TRAINING_SCOPE_CSV.is_file():
        return frozenset()
    with TRAINING_SCOPE_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        return frozenset(row["unit_id"] for row in csv.DictReader(stream)
                         if row["in_current150_development_pool"].strip().lower() == "true")


def active_video_specs(videos: dict) -> dict:
    """每次请求重读退役清单;历史映射和标注保留,退役来源不再开放审阅或保存。"""
    if not RETIRED_SOURCES_JSON.is_file():
        return videos
    with RETIRED_SOURCES_JSON.open("r", encoding="utf-8-sig") as f:
        retirement = json.load(f)
    retired_roots = [(REPO_ROOT / rel).resolve()
                     for rel in retirement["source_dirs"]]
    active = {}
    for name, spec in videos.items():
        source = spec["root"] if spec["kind"] == "key" else spec["dir"]
        source = source.resolve()
        if not any(source == root or root in source.parents for root in retired_roots):
            active[name] = spec
    return active


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地事件区间标注服务器(浏览器里标泥石流起止秒)")
    parser.add_argument("--dir", default="",
                        help="三种取值:汇总根目录(如 data/process/video_frames,自动发现全部"
                         "汇总源与旧式视频目录)/ 单个汇总源(如 video1_segment,按 mapping.csv "
                         "展开全部 video_key)/ 旧式单视频目录(含 frames_index.csv)")
    parser.add_argument("--port", type=int, default=8100, help="起始端口,被占用自动 +1(默认 8100)")
    parser.add_argument("--thumbs-dir", default="",
                        help="缩略图缓存目录(默认汇总源 <root>/thumbs/<key>/ 或 <视频目录>/thumbs/;"
                         "自测模式重定向到临时目录,一般不用传)")
    parser.add_argument("--selftest", action="store_true",
                        help="自测模式:在临时合成夹具上跑双模式(dir/key)全流程测试,"
                             "打印 PASS 后退出,不碰真实标注、不开浏览器")
    parser.add_argument("--review-file", default="", help="可选复核清单JSON，只添加提示和时间跳转，不改标注")
    parser.add_argument("--annotations-dir", default="",
                        help="可选独立标签版本目录：首次完整复制各源旧CSV，后续仅写版本CSV；复用原帧、索引和缩略图")
    parser.add_argument("--original-labels", action="store_true",
                        help="明确查看历史原标签；默认使用current-review.json登记的当前工作版")
    return parser.parse_args()


# ---------------------------------------------------------------- 数据读取

def time_value(value) -> int | float:
    """保留索引/CSV中的小数秒；整数仍按整数输出，拒绝非有限数。"""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("时间必须是有限数字")
    return int(number) if number.is_integer() else number


def kind_counts(events: list[dict]) -> dict[str, int]:
    counts = {kind: 0 for kind in KINDS}
    counts["unknown"] = 0
    for event in events:
        kind = event.get("kind") or "pos"  # 仅兼容旧格式缺失值；显式未知类型绝不当 P。
        counts[kind if kind in KINDS else "unknown"] += 1
    return counts

# 场景分段导出 CSV(相对仓库根,服务需在仓库根启动;自测会替换为临时路径)。
# 背景(叶海峰 2026-09-15 定):外部负样本视频不都是定点拍摄、有的多场景拼接,
# 整批当负样本;neg 区间在此语境下即"场景段",段=外部负样本的划分原子,
# 同一段落内的帧不得跨划分两侧。
# 2026-09-16 data 结构重构:分段标记随 video2 初帧同放 process/
SCENES_CSV = Path("data/process/video_frames/video2_segment/extneg-scenes.csv")
SCENES_FIELDS = ["video_id", "seg_id", "start_sec", "end_sec", "n_frames", "kind", "note", "exported"]


def save_scenes_csv(path: Path, video_id: str, segs: list[dict]) -> None:
    """把一个视频的场景分段写进 scenes CSV:同视频旧行替换、其他视频保留。

    文件是各视频 events.csv 区间的导出视图(含未标空隙的隐式负段),
    可随时重导出再生;段间边界即场景切分点,段=外部视频的划分原子。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if path.is_file():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("video_id") != video_id]
    today = time.strftime("%Y-%m-%d")
    for s in segs:
        rows.append({"video_id": video_id, "seg_id": s["seg_id"],
                     "start_sec": s["start"], "end_sec": s["end"],
                     "n_frames": s["n_frames"], "kind": s["kind"],
                     "note": s.get("note", ""), "exported": today})
    rows.sort(key=lambda r: (r["video_id"], int(r["seg_id"])))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCENES_FIELDS)
        w.writeheader()
        w.writerows(rows)


def load_frames_index(frames_dir: Path, index_csv: Path,
                      video_key: str | None = None) -> tuple[list[str], list[float]]:
    """读 frames_index.csv,返回 (帧文件名列表, 对应 t_sec 列表)。

    传 video_key 时只取该 key 的行(汇总索引:全源帧平铺在一个 frames/,
    文件名带 <video_key>_ 前缀,索引靠 video_key 列区分归属)。
    只接受纯文件名(拒绝带路径分隔符/.. 的行),防止索引被改后路径穿越。
    """
    if not index_csv.is_file():
        raise SystemExit(f"找不到 {index_csv},请先运行 extract_frames.py 拆帧")
    files: list[str] = []
    ts: list[float] = []
    with index_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if video_key is not None and (row.get("video_key") or "").strip() != video_key:
                continue
            name = (row.get("frame_file") or "").strip().replace("\\", "/")
            if not name or "/" in name or ".." in name:
                continue
            try:
                t = float(row.get("t_sec"))
            except (TypeError, ValueError):
                continue
            if not (frames_dir / name).is_file():
                continue
            files.append(name)
            ts.append(t)
    if not files:
        raise SystemExit(f"{index_csv} 里没有可用的帧记录(frames/ 下找不到对应文件)")
    return files, ts


def read_interval_csv(path: Path) -> list[dict]:
    """读区间类 CSV(events.csv),跳过空起止的模板行。

    kind 缺失(旧 5 列 CSV)时返回空串,由调用方按 pos 处理(向后兼容)。
    """
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                s = time_value(row.get("start_sec"))
                e = time_value(row.get("end_sec"))
            except (TypeError, ValueError):
                continue  # 模板示例行或损坏行
            if e <= s:
                continue
            rows.append({"start": s, "end": e, "confidence": (row.get("confidence") or "").strip(),
                         "kind": (row.get("kind") or "").strip().lower(),
                         "note": (row.get("note") or "").strip()})
    return rows


def load_annotations(events_csv: Path, video_id: str) -> list[dict]:
    """从 events.csv 得到已确认区间列表。

    confidence=real 的行是已确认(kind 缺失按 pos,兼容旧 5 列);
    其余行(auto/dismissed 等旧预标注残留)一律忽略不显示,下次保存即被清除。"""
    events: list[dict] = []
    for row in read_interval_csv(events_csv):
        if row["confidence"] == "real":
            events.append({"start": row["start"], "end": row["end"],
                           "kind": row["kind"] or "pos", "note": row["note"]})
    events.sort(key=lambda x: x["start"])
    return events


def save_events_csv(csv_path: Path, video_id: str, events: list[dict]) -> None:
    """整份重写 events.csv:全部行 confidence=real(带 kind);
    无区间时保留模板示例行。旧 dismissed/auto 行在重写时被清除。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(EVENTS_FIELDS)
        if not events:
            writer.writerow([video_id] + TEMPLATE_ROW)
        for ev in sorted(events, key=lambda x: x["start"]):
            writer.writerow([video_id, ev["start"], ev["end"], "real",
                             ev.get("kind") or "pos", ev.get("note", "")])


# 汇总层 events.csv 是多视频共用一份(7 列,多一个 video_key 列),
# 任意一个视频会话的保存都要整文件读改写,用锁防并发互相覆盖。
_CSV_LOCK = threading.Lock()
_CONSOLIDATED_FIELDS = ["video_id", "video_key", "start_sec", "end_sec",
                        "confidence", "kind", "note"]


def load_events_consolidated(root: Path, video_key: str,
                             events_csv: Path | None = None) -> list[dict]:
    """汇总层 events.csv 里取一个 video_key 的已确认区间(其余 key 的行不动)。"""
    return _read_consolidated_rows(root, events_csv).get(video_key, [])


def _read_consolidated_rows(root: Path,
                            events_csv: Path | None = None) -> dict[str, list[dict]]:
    """汇总层 events.csv -> {video_key: [区间]},只留 confidence=real 行。"""
    out: dict[str, list[dict]] = {}
    path = events_csv or root / "events.csv"
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("confidence") or "").strip() != "real":
                continue
            try:
                s = time_value(row.get("start_sec"))
                e = time_value(row.get("end_sec"))
            except (TypeError, ValueError):
                continue
            if e <= s:
                continue
            out.setdefault((row.get("video_key") or "").strip(), []).append(
                {"start": s, "end": e,
                 "kind": (row.get("kind") or "").strip().lower() or "pos",
                 "note": (row.get("note") or "").strip()})
    for v in out.values():
        v.sort(key=lambda x: x["start"])
    return out


def save_events_consolidated(root: Path, video_key: str, video_id: str,
                             events: list[dict],
                             events_csv: Path | None = None) -> None:
    """把一个 video_key 的区间整份写回汇总层 events.csv:只替换该 key 的行,
    其他 key 的行原样保留;全部写 confidence=real。"""
    path = events_csv or root / "events.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CSV_LOCK:
        kept: list[list] = []
        if path.is_file():
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.reader(f):
                    if not row or row[0].startswith("video_id"):
                        continue
                    if len(row) >= 2 and row[1].strip() == video_key:
                        continue
                    kept.append(row)
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(_CONSOLIDATED_FIELDS)
            w.writerows(kept)
            for ev in sorted(events, key=lambda x: x["start"]):
                w.writerow([video_id, video_key, ev["start"], ev["end"],
                            "real", ev.get("kind") or "pos", ev.get("note", "")])


def count_kinds_consolidated(root: Path,
                             events_csv: Path | None = None) -> dict[str, tuple[int, int, int]]:
    """汇总层 events.csv 每个视频 key 的 (正, 剔, 负) 区间计数;无文件返回 {}。"""
    rows = _read_consolidated_rows(root, events_csv)
    return {k: (sum(1 for e in v if e["kind"] == "pos"),
                sum(1 for e in v if e["kind"] == "cut"),
                sum(1 for e in v if e["kind"] == "neg"))
            for k, v in rows.items()}


def status_consolidated(root: Path, video_key: str) -> str:
    """汇总层口径的标注进度:该 key 有 real 行=已标,否则未标。"""
    return "annotated" if _read_consolidated_rows(root).get(video_key) else "unlabeled"


def normalize_events(raw: list) -> list[dict]:
    """保留小数秒，校验 [start,end) 区间及 N/T/P/U/cut，禁止所有类型间重叠。"""
    if not isinstance(raw, list):
        raise ValueError("区间列表格式错误")
    out: list[dict] = []
    for ev in raw:
        if not isinstance(ev, dict):
            raise ValueError("区间格式错误")
        try:
            s = time_value(ev.get("start"))
            e = time_value(ev.get("end"))
        except (TypeError, ValueError):
            raise ValueError("起止秒必须是数字")
        if s < 0 or e <= s:
            raise ValueError(f"区间 {s}–{e}s 必须满足 0 ≤ 起点 < 终点")
        kind = str(ev.get("kind") or "pos").strip().lower()
        if kind not in KINDS:
            raise ValueError(f"未知区间类型 {kind!r}(只支持 pos=P / transition=T / neg=N / unknown=U / cut=质量剔除)")
        out.append({"start": s, "end": e, "kind": kind, "note": str(ev.get("note", ""))})
    out.sort(key=lambda x: x["start"])
    for a, b in zip(out, out[1:]):
        if a["end"] > b["start"]:
            raise ValueError(f"区间 {a['start']}–{a['end']}s 与 {b['start']}–{b['end']}s 重叠"
                             "(本工具禁止重叠:请先删除或调整其中一段)")
    return out


def count_kinds(dir_path: Path) -> tuple[int, int, int]:
    """数 events.csv 里已确认的正/剔/负区间数(kind 缺失按 pos);没有文件返回 (0, 0, 0)。"""
    n_pos = n_cut = n_neg = 0
    for row in read_interval_csv(dir_path / "events.csv"):
        if row["confidence"] == "real":
            if row["kind"] == "cut":
                n_cut += 1
            elif row["kind"] == "neg":
                n_neg += 1
            elif not row["kind"] or row["kind"] == "pos":
                n_pos += 1
    return n_pos, n_cut, n_neg


def annotation_status(dir_path: Path) -> str:
    """据 events.csv 判定视频标注进度,供 /api/videos 与前端下拉框显示:
      annotated 已标——存在 confidence=real 的行(kind 为 pos/cut/neg;
        旧 5 列 CSV 无 kind 列的 real 行按 pos 兼容,同样算已标);
      unlabeled 未标——events.csv 不存在,或只有模板行等未处理行。
    模板行与空行由 read_interval_csv 统一跳过(BOM/编码同样在那里兼容)。
    """
    if not (dir_path / "events.csv").is_file():
        return "unlabeled"
    for row in read_interval_csv(dir_path / "events.csv"):
        if row["confidence"] == "real":
            return "annotated"
    return "unlabeled"


def sample_indices(n: int, count: int = THUMB_COUNT) -> list[int]:
    """从 n 帧里均匀抽 ≤count 个下标(首尾必含),供缩略时间轴使用。"""
    if n <= count:
        return list(range(n))
    return sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})


# ---------------------------------------------------------------- 缩略图缓存

class ThumbCache:
    """缩略图磁盘缓存:<thumbs_dir>/<帧文件名>,与帧一一对应,重拆帧后自然失效。

    ensure() 按需生成(线程安全);pregenerate() 在后台线程把抽样帧全部
    预生成,让用户几乎总是直接命中缓存。生成用 np.fromfile+cv2.imdecode,
    规避 cv2.imread 在 Windows 中文路径下静默失败的问题。
    """

    def __init__(self, frames_dir: Path, thumbs_dir: Path, files: list[str], sample_idx: list[int]):
        self.frames_dir = frames_dir
        self.thumbs_dir = thumbs_dir
        self.files = files
        self.sample_idx = sample_idx
        self.lock = threading.Lock()
        self.ready = 0
        self.total = len(sample_idx)
        self.stop = threading.Event()

    def thumb_path(self, i: int) -> Path:
        return self.thumbs_dir / self.files[i]

    def ensure(self, i: int) -> bool:
        """确保第 i 帧的缩略图存在(已缓存直接返回);失败返回 False。"""
        tp = self.thumb_path(i)
        if tp.is_file():
            return True
        try:
            buf = np.fromfile(str(self.frames_dir / self.files[i]), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("解码失败")
            h, w = img.shape[:2]
            thumb = cv2.resize(img, (THUMB_WIDTH, max(1, int(h * THUMB_WIDTH / w))),
                               interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise ValueError("编码失败")
            self.thumbs_dir.mkdir(parents=True, exist_ok=True)
            enc.tofile(str(tp))
        except (OSError, ValueError) as e:
            print(f"[缩略图] 第 {i} 帧生成失败:{e}")
            return False
        return True

    def pregenerate(self) -> None:
        done = 0
        for i in self.sample_idx:
            if self.stop.is_set():
                return
            if self.ensure(i):
                done += 1
                with self.lock:
                    self.ready = done


# ---------------------------------------------------------------- 前端页面

PAGE_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>事件区间标注</title>
<style>
  :root { color-scheme: light; --ink: #233044; --muted: #788394; --line: #e0e5eb; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 12px; font: 13px "Microsoft YaHei", system-ui, sans-serif;
         background: #eef1f4; color: var(--ink); }
  button, select, input, textarea { font: inherit; }
  button, select { cursor: pointer; }
  button { padding: 7px 10px; border: 1px solid var(--line); border-radius: 6px;
           background: #fff; color: var(--ink); white-space: nowrap; }
  button:hover { border-color: #9aa9ba; background: #f6f8fa; }
  button:focus-visible, summary:focus-visible { outline: 2px solid #4382c3; outline-offset: 2px; }
  select, input, textarea { min-width: 0; max-width: 100%; border: 1px solid #d5dce5;
                           border-radius: 5px; background: white; padding: 7px 8px; color: var(--ink); }
  header { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; min-width: 0; }
  header h1 { font-size: 15px; margin: 0 3px 0 0; white-space: nowrap; }
  .video-switch { display: flex; flex: 1; min-width: 0; gap: 5px; }
  #video-sel { width: 100%; min-width: 110px; text-overflow: ellipsis; }
  #list-sel { width: 108px; padding-left: 6px; padding-right: 4px; font-size: 12px; }
  #list-count { white-space: nowrap; font-size: 11px; color: var(--muted); }
  #version-short { max-width: 110px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
                   background: #e2e7ee; padding: 5px 7px; border-radius: 5px; font-size: 11px; }
  #saved-at { min-width: 105px; color: #327053; font-size: 11px; text-align: right; }
  main { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 12px; align-items: start; }
  #viewer { min-width: 0; background: #151c26; border-radius: 9px; overflow: hidden; }
  #frame-stage { width: 100%; aspect-ratio: 16 / 9; max-height: calc(100vh - 240px);
                 display: flex; justify-content: center; align-items: center; background: #0d121b; }
  #mainimg { width: 100%; height: 100%; min-height: 0; object-fit: contain; display: block; }
  #timebar { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;
             gap: 8px; padding: 10px 12px; color: #e6edf6; }
  .frame-controls { display: flex; align-items: center; gap: 8px; min-width: 0; }
  .frame-controls button { padding: 3px 9px; border-color: #3a4657; background: #243044; color: #fff; font-size: 17px; }
  #time-big { font: 13px ui-monospace, Consolas, monospace; white-space: nowrap; }
  #frame-kind { font-size: 11px; font-weight: normal; }
  #window-box { margin: 0 12px 10px; padding-top: 8px; border-top: 1px solid #303a49; color: #bdc9d9; font-size: 11px; }
  .window-heading { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
  #window-preview { white-space: normal; text-align: right; }
  #window-bar { display: flex; height: 5px; border-radius: 3px; overflow: hidden; background: #4e5360; }
  #window-bar .neg { background: #4c92cc; } #window-bar .transition { background: #e5af36; }
  #window-bar .pos { background: #57a47b; } #window-bar .cut { background: #8a929c; }
  #window-bar .unknown { background: #ac83bd; }
  #panel { min-width: 0; max-height: calc(100vh - 78px); display: flex; flex-direction: column;
           background: white; border: 1px solid var(--line); border-radius: 9px; overflow: hidden; }
  .panel-heading { display: flex; justify-content: space-between; align-items: center; padding: 12px; flex-shrink: 0; }
  .panel-heading h2 { font-size: 14px; margin: 0; } #event-count { font-size: 11px; font-weight: normal; color: var(--muted); }
  .panel-heading button { font-size: 12px; padding: 5px 9px; color: #276297; }
  #events-list { position: relative; flex: 1; padding: 0 10px 4px; overflow-y: auto; min-height: 70px; }
  .ev-row { border: 1px solid var(--line); border-radius: 7px; margin-bottom: 7px; overflow: hidden; }
  .ev-row.selected { border-color: #7e9dbb; box-shadow: 0 0 0 1px #d7e5f2; }
  .event-select { display: flex; align-items: center; gap: 7px; width: 100%; border: 0; border-radius: 0;
                  padding: 10px 9px; text-align: left; background: #fafbfd; }
  .event-select:hover { background: #f1f5f9; }
  .ev-time { font: 12px ui-monospace, Consolas, monospace; white-space: nowrap; }
  .draft-flag { margin-left: auto; font-size: 10px; color: #ad720a; }
  .edit-hint { margin-left: auto; color: #276297; font-size: 11px; white-space: nowrap; }
  .ev-summary { margin: 0; padding: 0 9px 9px; font-size: 11px; color: var(--muted); overflow-wrap: anywhere; }
  .event-editor { padding: 4px 9px 10px; }
  .event-editor label { display: block; font-size: 11px; color: #728092; margin-top: 8px; }
  .event-editor select, .event-editor textarea, .event-editor input { width: 100%; margin-top: 4px; font-size: 12px; }
  .event-editor textarea { resize: vertical; min-height: 55px; }
  .bound-fields { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .editor-actions { display: flex; gap: 7px; margin-top: 10px; }
  .editor-actions .primary { flex: 1; background: #2d659a; border-color: #2d659a; color: white; font-weight: 600; }
  .editor-actions .primary:hover { background: #23547f; }
  .editor-secondary { display: flex; justify-content: space-between; align-items: center; margin-top: 6px; }
  .editor-secondary button { padding: 4px 0; border: 0; background: transparent; font-size: 11px; color: #647991; }
  .editor-secondary .delete { color: #9a7777; }
  .badge { font-size: 11px; line-height: 1.2; padding: 4px 6px; border-radius: 4px; white-space: nowrap; }
  .badge.pos { background: #e2f1e7; color: #1c6b3c; }
  .badge.cut { background: #e4e4e4; color: #555; }
  .badge.neg { background: #e2ecf7; color: #1c5a8f; }
  .badge.transition { background: #fff0bf; color: #8b5700; }
  .badge.unknown { background: #f1e7f7; color: #754389; }
  .empty-tip { padding: 12px 4px; font-size: 12px; line-height: 1.7; color: var(--muted); }
  #toast { position: fixed; bottom: 18px; left: 50%; transform: translateX(-50%); max-width: min(440px, 92vw);
           background: #873e42; color: white; padding: 10px 14px; border-radius: 7px; font-size: 12px; z-index: 9; }
  #strip-zone { border-top: 1px solid #303a49; padding: 9px 10px 7px; }
  .timeline-heading { display: flex; justify-content: space-between; margin-bottom: 6px; color: #96a5ba; font-size: 10px; }
  #scroller { overflow-x: auto; overflow-y: hidden; padding: 0 0 3px; }
  #strip-inner { position: relative; }
  #overlay { position: absolute; top: 2px; left: 0; height: 14px; width: 100%; cursor: crosshair; }
  #overlay span { position: absolute; top: 0; height: 100%; }
  #overlay .ev { background: #2f7d4f; opacity: .85; border-radius: 2px; }
  #overlay .cut { background: repeating-linear-gradient(45deg, #444 0 4px, #999 4px 8px);
                  border-radius: 2px; }
  #overlay .neg { background: #2f6b9f; opacity: .85; border-radius: 2px; }
  #overlay .transition { background: #d59a12; opacity: .9; border-radius: 2px; }
  #overlay .unknown { background: #9866aa; border-radius: 2px; }
  #overlay .mark-s { background: #d22; width: 3px; }
  #overlay .mark-e { background: #26c; width: 3px; }
  #overlay .pending-range { background: #f0bd48; opacity: .4; border: 1px dashed #ffe397; }
  #overlay .caret { background: #fa0; width: 2px; top: -2px; height: calc(100% + 4px); }
  #thumbs { display: flex; padding-top: 19px; }
  .cell { flex: 0 0 120px; width: 120px; margin-right: 6px; cursor: pointer; }
  .cell img { width: 120px; display: block; border: 2px solid transparent; border-radius: 3px;
              background: #222; min-height: 40px; }
  .cell.cur img { border-color: #fa0; }
  .cell.in-ev img { border-color: #2f7d4f; }
  .cell.in-cut img { border-color: #666; }
  .cell.in-neg img { border-color: #2f6b9f; }
  .cell.in-transition img { border-color: #d59a12; }
  .cell.in-unknown img { border-color: #9866aa; }
  .cell.cur.in-ev img, .cell.cur.in-cut img, .cell.cur.in-neg img,
  .cell.cur.in-transition img, .cell.cur.in-unknown img { border-color: #fa0; }
  .cell .lab { font-size: 10px; color: #aebccc; text-align: center; font-variant-numeric: tabular-nums; }
  details { font-size: 12px; line-height: 1.7; } summary { cursor: pointer; color: #64758a; }
  .panel-details { border-top: 1px solid var(--line); padding: 9px 12px; flex-shrink: 0; }
  .panel-details[open] { overflow: auto; flex-shrink: 0; max-height: 44vh; }
  .panel-details summary { font-size: 12px; }
  .toolbar { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  #new-interval { margin: 0 12px 10px; padding-top: 8px; border-top: 1px solid #303a49; color: #bdc9d9; }
  #new-interval .toolbar { margin-top: 0; align-items: center; }
  #new-interval select { width: 110px; font-size: 12px; }
  #new-interval button { padding: 7px 8px; font-size: 12px; }
  .range-fields { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: 7px; }
  .range-fields label { display: flex; align-items: center; gap: 5px; font-size: 11px; }
  .range-fields input { width: 92px; padding: 6px; font: 12px ui-monospace, Consolas, monospace; }
  #new-interval .range-apply { background: #c89734; border-color: #c89734; color: #17212f; font-weight: 600; }
  button:disabled { opacity: .45; cursor: default; }
  #pending-info { display: block; margin-top: 6px; color: #e8be70; font-size: 11px; }
  #review-box p { font-size: 12px; margin: 8px 0; color: #637185; }
  #review-title { font-weight: normal; font-size: 11px; }
  #review-jumps button { margin: 3px 4px 3px 0; padding: 5px 7px; font-size: 11px; }
  #info-box { margin-top: 10px; padding: 8px 3px; color: #66758a; }
  #info-box .info-content { padding: 5px 8px; max-width: 900px; }
  #info-box p { margin: 9px 0; } #annotation-version, #video-name { overflow-wrap: anywhere; font-size: 11px; }
  #hints { line-height: 2; } .more-options { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }
  kbd { background: #eee; border: 1px solid #ccc; border-radius: 3px; padding: 0 4px; font-size: 11px; }
  [hidden] { display: none !important; }
  @media (min-width: 1200px) { main { grid-template-columns: minmax(0, 1fr) 320px; } }
  @media (max-width: 719px) {
    body { padding: 8px; } header { flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
    header h1 { font-size: 13px; } .video-switch { flex-basis: calc(100% - 80px); }
    #version-short { max-width: 180px; } #saved-at { margin-left: auto; }
    main { grid-template-columns: minmax(0, 1fr); gap: 10px; }
    #frame-stage { max-height: none; } #panel { max-height: none; }
    #events-list { max-height: 55vh; } #timebar { padding: 9px 10px; }
    #info-box { margin-top: 4px; }
  }
</style>
</head>
<body>
<header>
  <h1>区间审计</h1>
  <div class="video-switch"><button id="btn-prev-video" title="上一视频 P">上一</button>
    <select id="video-sel" aria-label="选择视频"></select>
    <button id="btn-next-video" title="下一视频 N">下一</button></div>
  <select id="list-sel" aria-label="视频列表范围"><option value="all">全部视频</option>
    <option value="training">现役训练池150条</option>
    <option value="review">边界复核</option><option value="all_n">全N视频</option></select><span id="list-count"></span>
  <span id="version-short" title="标签版本">独立版本</span>
  <span id="saved-at">已载入</span>
</header>
<main>
  <section id="viewer" aria-label="视频审阅画面">
    <div id="frame-stage"><img id="mainimg" alt="当前审计帧，完整画面"></div>
    <div id="timebar"><div class="frame-controls">
      <button id="btn-prev-frame" title="上一帧 ←" aria-label="上一帧">‹</button>
      <span id="time-big">加载中…</span>
      <button id="btn-next-frame" title="下一帧 →" aria-label="下一帧">›</button>
    </div><strong id="frame-kind" class="badge unknown">当前帧 · 未知</strong></div>
    <section id="new-interval" aria-label="范围改标">
      <div class="toolbar"><span>范围改标</span>
        <select id="kind-sel" aria-label="范围标注类型"><option value="pos">正类 P</option>
          <option value="transition" selected>显现发展 T</option><option value="neg">负类 N</option>
          <option value="cut">剔除 cut</option></select>
        <button id="btn-start">设起点 S</button><button id="btn-end">设终点 E</button><button id="btn-clear">清除 C</button></div>
      <div class="range-fields"><label>起<input id="range-start" type="number" min="0" step="any" aria-label="范围起点秒" placeholder="秒"></label>
        <label>止<input id="range-end" type="number" min="0" step="any" aria-label="范围终点秒" placeholder="秒"></label>
        <button id="btn-apply-range" class="range-apply" disabled>应用标注</button><button id="btn-undo" disabled>撤销</button></div>
      <span id="pending-info"></span>
    </section>
    <div id="window-box"><div class="window-heading"><span>居中3秒观察窗</span><span id="window-preview"></span></div>
      <div id="window-bar" aria-label="人类区间时间覆盖比例"></div></div>
    <div id="strip-zone"><div class="timeline-heading"><span>时间轴 · 点击跳转</span><span id="thumb-status"></span></div>
      <div id="scroller"><div id="strip-inner"><div id="overlay"></div><div id="thumbs"></div></div></div></div>
  </section>
  <aside id="panel" aria-label="区间编辑">
    <div class="panel-heading"><h2>区间标签 <span id="event-count"></span></h2><button id="btn-new">范围改标</button></div>
    <div id="events-list"></div>
    <details id="review-box" class="panel-details" hidden><summary>复核提示 <span id="review-title"></span></summary>
      <p id="review-question"></p><div id="review-jumps"></div>
      <p>按画面独立判断；疑问可写入区间备注。提示不会自动改标签。审计帧约每2秒一张，不必强行给出亚秒级起点。</p>
    </details>
  </aside>
</main>
<details id="info-box"><summary>类别说明 · 快捷键 · 来源与更多操作</summary><div class="info-content">
  <p><span class="badge neg">N</span> 明确无泥石流，可包括普通水流；<span class="badge transition">T</span> 有过程证据支持的泥石流显现、发展时段；
    <span class="badge pos">P</span> 泥石流目标明确可见；
    <span class="badge cut">cut</span> 质量剔除。未标空隙仍为未知；已保存不等于训练就绪。</p>
  <p>区间均为 [起点, 终点)，保留真实小数秒。范围改标无需选父区间：选类型，用 S/E 或输入秒数确定范围，再点击“应用标注”。应用会覆盖范围内旧类型、保留两侧与原分段备注。S/E 只做预览，“清除”取消未应用范围。右侧仍可整段修改并点击“保存本段”。撤销按编辑顺序逐次恢复，切视频后清空。</p>
  <p>居中3秒观察窗采用当前帧时间 ±1.5秒，片头片尾截短，仅汇总现有人类区间的时间覆盖。现有审计帧约每2秒一张；此处未生成模型16帧输入，也不表示训练窗口或预测规则。</p>
  <div id="hints">
  <kbd>←</kbd><kbd>→</kbd> 步进 1 帧(2 秒) · <kbd>Shift</kbd>+<kbd>←</kbd><kbd>→</kbd> 10 帧 ·
  <kbd>PgUp</kbd><kbd>PgDn</kbd> ±2 分钟 · <kbd>Home</kbd>/<kbd>End</kbd> 首末帧 ·
  <kbd>S</kbd> 设起点 · <kbd>E</kbd> 设终点 · <kbd>C</kbd> 清除标记 ·
  <kbd>N</kbd> 下一视频 · <kbd>P</kbd> 上一视频 ·
  点击缩略图或上方色条跳转；输入框与类型选择时不触发快捷键。
  </div><p id="video-name"></p><p id="annotation-version"></p>
  <div class="more-options"><label><input type="checkbox" id="from-start"> 从头开始</label>
    <label><input type="checkbox" id="hide-cut"> 隐藏 cut 区间</label>
    <button id="btn-neg-all">整段标负</button><button id="btn-scenes">导出场景分段</button></div>
</div></details>
<div id="toast" hidden></div>
<script>
"use strict";
var state = { ts: [], duration: 0, idx: 0, events: [],
              pendingStart: null, pendingEnd: null,
              thumbIdx: [], stripW: 0, thumbsBuilt: false };
var toastTimer = null;
var saveChain = Promise.resolve(true), savePending = 0;
var editDrafts = new Map(), selectedEvent = null, saveFailed = false, allowNavigation = false;
var undoStack = [], pendingTouched = false;

// 当前视频(?v= 参数,空 = 服务端默认视频);api() 给每个请求补上 v 定位会话
var V = decodeURIComponent((location.search.match(/[?&]v=([^&]*)/) || ["", ""])[1] || "");
var reviewItems = [], videoItems = [], currentVideo = V;
var listMode = parseListMode(location.search), reviewOnly = listMode === "review";
function api(p) { return p + (p.indexOf("?") >= 0 ? "&" : "?") + "v=" + encodeURIComponent(V); }

function $(id) { return document.getElementById(id); }

function fmt(s) {
  var m = Math.floor(s / 60);
  return m + ":" + (s % 60).toFixed(2).padStart(5, "0");
}
function sec(s) { return Number(s).toFixed(2); }

function toast(msg) {
  var el = $("toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(function () { el.hidden = true; }, 4000);
}

// 区间类型中文名(正=泥石流可见 / 剔=雾大全黑等不入数据集 / 负=确认无泥石流)
var KIND_CN = { pos: "P", transition: "T", cut: "cut", neg: "N", unknown: "U" };
var KIND_LONG = { pos: "明确正样本 P", transition: "显现发展 T", cut: "质量剔除 cut", neg: "明确负样本 N", unknown: "历史类型未定 U" };

// ---- pure interval helpers (used by the synthetic regression tests) --------
function parseListMode(search) {
  var query = new URLSearchParams(search), mode = query.get("list");
  if (mode === "all" || mode === "training" || mode === "review" || mode === "all_n") { return mode; }
  return query.get("review") === "1" ? "review" : "all";
}
function videosForMode(videos, reviews, mode) {
  if (mode === "training") {
    return videos.filter(function (v) { return v.in_training_pool === true; });
  }
  if (mode === "all_n") {
    return videos.filter(function (v) {
      return v.n_neg > 0 && v.n_pos === 0 && v.n_transition === 0 && (v.n_unknown || 0) === 0;
    });
  }
  if (mode === "review") {
    return reviews.map(function (r) { return videos.find(function (v) { return v.name === r.unit_id; }); }).filter(Boolean);
  }
  return videos;
}
function videoQuery(name, mode, time) {
  var query = "?v=" + encodeURIComponent(name);
  if (mode === "training") { query += "&list=training"; }
  else if (mode === "all_n") { query += "&list=all_n"; }
  else if (mode === "review") { query += "&list=review&review=1"; }
  if (time !== undefined && time !== null) { query += "&t=" + time; }
  return query;
}
function classAt(events, t) {
  for (var k = 0; k < events.length; k++) {
    var ev = events[k];
    if (ev.start <= t && t < ev.end) { return KIND_CN[ev.kind] ? ev.kind : "unknown"; }
  }
  return "unknown";
}
function windowCoverage(events, start, end) {
  var coverage = { neg: 0, transition: 0, pos: 0, cut: 0, unknown: 0 };
  var boundaries = [start, end];
  events.forEach(function (ev) {
    if (start < ev.start && ev.start < end) { boundaries.push(ev.start); }
    if (start < ev.end && ev.end < end) { boundaries.push(ev.end); }
  });
  boundaries = Array.from(new Set(boundaries)).sort(function (a, b) { return a - b; });
  for (var k = 0; k + 1 < boundaries.length; k++) {
    var a = boundaries[k], b = boundaries[k + 1];
    coverage[classAt(events, (a + b) / 2)] += b - a;
  }
  return { coverage: coverage, crosses: boundaries.length > 2 };
}
function validateEvents(events) {
  var ordered = events.slice().sort(function (a, b) { return a.start - b.start; });
  ordered.forEach(function (ev, i) {
    if (!Number.isFinite(ev.start) || !Number.isFinite(ev.end) || ev.start < 0 || ev.start >= ev.end) {
      throw new Error("起止秒必须满足 0 ≤ 起点 < 终点");
    }
    if (!KIND_CN[ev.kind]) { throw new Error("请选择 N / T / P / cut 类型"); }
    if (i && ordered[i - 1].end > ev.start) { throw new Error("修改后与其他区间重叠，已拒绝"); }
  });
  return ordered;
}
function editedEvents(events, index, replacement) {
  return validateEvents(events.map(function (ev, k) { return k === index ? replacement : ev; }));
}
function splitEvents(events, index, t) {
  var ev = events[index];
  if (!Number.isFinite(t) || !(ev.start < t && t < ev.end)) {
    throw new Error("请先把当前帧移到本段内部；分割点必须严格位于起止秒之间");
  }
  var result = events.slice();
  result.splice(index, 1, { start: ev.start, end: t, kind: ev.kind, note: ev.note },
                        { start: t, end: ev.end, kind: ev.kind, note: ev.note });
  return validateEvents(result);
}
function rangeBounds(start, end, duration) {
  if (start === null || end === null || (typeof start === "string" && !start.trim()) ||
      (typeof end === "string" && !end.trim())) { throw new Error("请设置起点和终点"); }
  var s = Number(start), e = Number(end);
  if (s > e) { var swap = s; s = e; e = swap; }
  if (!Number.isFinite(s) || !Number.isFinite(e) || s < 0 || s >= e) {
    throw new Error("范围须满足 0 ≤ 起点 < 终点");
  }
  if (Number.isFinite(duration) && e > duration) { throw new Error("范围不能超过视频时间轴末端"); }
  return { start: s, end: e };
}
function relabelRange(events, start, end, kind, duration) {
  var bounds = rangeBounds(start, end, duration), s = bounds.start, e = bounds.end;
  if (!KIND_CN[kind]) { throw new Error("请选择 N / T / P / cut 类型"); }
  var ordered = events.slice().sort(function (a, b) { return a.start - b.start; });
  ordered.forEach(function (ev, i) {
    if (!Number.isFinite(ev.start) || !Number.isFinite(ev.end) || ev.start < 0 || ev.start >= ev.end ||
        (i && ordered[i - 1].end > ev.start)) { throw new Error("原区间存在非法边界或重叠，请先修正"); }
  });
  var result = [], points = [s, e];
  events.forEach(function (ev) {
    if (ev.end <= s || e <= ev.start) { result.push(ev); return; }
    if (ev.start < s) { result.push(Object.assign({}, ev, { end: s })); }
    if (e < ev.end) { result.push(Object.assign({}, ev, { start: e })); }
    points.push(Math.max(s, ev.start), Math.min(e, ev.end));
  });
  points = Array.from(new Set(points)).sort(function (a, b) { return a - b; });
  for (var k = 0; k + 1 < points.length; k++) {
    var a = points[k], b = points[k + 1];
    var original = events.find(function (ev) { return ev.start <= a && b <= ev.end; });
    result.push(Object.assign({}, original || { note: "" }, { start: a, end: b, kind: kind }));
  }
  // 保留旧边界上的分段及各自备注；相邻同类也不盲目合并。
  return validateEvents(result);
}
function cloneEvents(events) { return events.map(function (event) { return Object.assign({}, event); }); }
function draftChanged(event, draft) {
  return draft && (draft.start !== String(event.start) || draft.end !== String(event.end) ||
                   draft.kind !== event.kind || draft.note !== (event.note || ""));
}
function hasUnsavedDrafts() {
  var changed = false;
  editDrafts.forEach(function (draft, event) { if (draftChanged(event, draft)) { changed = true; } });
  return changed;
}
function hasPendingRange() { return pendingTouched || state.pendingStart !== null || state.pendingEnd !== null; }
// ---- end pure interval helpers --------------------------------------------

// ---- 时间 ↔ 帧序号 -------------------------------------------------------

function frameAt(t) {
  var ts = state.ts, lo = 0, hi = ts.length - 1;
  while (lo < hi) { var mid = (lo + hi) >> 1; if (ts[mid] < t) { lo = mid + 1; } else { hi = mid; } }
  return lo;
}

function seekToT(t) { setIdx(frameAt(t)); }

function setIdx(i) {
  if (!state.ts.length) { return; }
  i = Math.max(0, Math.min(state.ts.length - 1, i));
  state.idx = i;
  $("mainimg").src = api("/img?i=" + i);
  // 预载相邻帧,步进翻页更跟手
  [i + 1, i - 1].forEach(function (j) {
    if (j >= 0 && j < state.ts.length) { var im = new Image(); im.src = api("/img?i=" + j); }
  });
  updateTime(); updateThumbClasses(); scrollStrip(); renderOverlay();
}

function updateTime() {
  var t = state.ts[state.idx] || 0;
  $("time-big").textContent = sec(t) + " / " + sec(state.duration) + " s";
  $("time-big").title = "当前 " + fmt(t) + " / 全长 " + fmt(state.duration);
  var kind = classAt(state.events, t);
  var explicitU = state.events.some(function (ev) { return ev.kind === "unknown" && ev.start <= t && t < ev.end; });
  $("frame-kind").textContent = "当前帧 · " + (kind === "unknown" ? (explicitU ? "U · 类型未定" : "未标注 / 未知") : KIND_CN[kind]);
  $("frame-kind").title = kind === "unknown" && !explicitU ? "未标注空隙或未知类型；不会自动补 N" : KIND_LONG[kind];
  $("frame-kind").className = "badge " + kind;
  var start = Math.max(state.ts[0] || 0, t - 1.5), end = Math.min(state.duration, t + 1.5);
  var preview = windowCoverage(state.events, start, end), parts = [], bar = $("window-bar");
  bar.innerHTML = "";
  ["neg", "transition", "pos", "cut", "unknown"].forEach(function (k) {
    var amount = preview.coverage[k];
    if (amount > 0) {
      var percent = 100 * amount / (end - start), segment = document.createElement("span");
      parts.push((k === "unknown" ? "U/未标" : KIND_CN[k]) + " " + percent.toFixed(0) + "%");
      segment.className = k; segment.style.width = percent + "%";
      segment.title = (KIND_LONG[k] || "未知") + " " + sec(amount) + "s";
      bar.appendChild(segment);
    }
  });
  $("window-preview").textContent = parts.join(" · ") + (preview.crosses ? " · 跨界" : "");
  $("window-box").title = "居中观察窗 [" + sec(start) + ", " + sec(end) + ") 秒，仅汇总人类区间的时间覆盖";
}

// ---- 区间操作 -------------------------------------------------------------

function findOverlap(s, e) {
  for (var k = 0; k < state.events.length; k++) {
    var ev = state.events[k];
    if (s < ev.end && ev.start < e) { return ev; }
  }
  return null;
}

// ---- edit transactions ----------------------------------------------------
function commitEvents(next, selection, action) {
  var normalized = validateEvents(next);
  undoStack.push({ events: cloneEvents(state.events), action: action });
  if (undoStack.length > 50) { undoStack.shift(); }
  state.events = normalized;
  editDrafts.forEach(function (draft, event) { if (!state.events.includes(event)) { editDrafts.delete(event); } });
  selectedEvent = state.events.includes(selection) ? selection : null;
  renderEvents(); renderPending();
  return save();
}
function undoLastEdit() {
  if (!undoStack.length) { return; }
  if (hasUnsavedDrafts()) { toast("请先保存区间草稿，再撤销已应用的编辑"); return; }
  var previous = undoStack.pop();
  state.events = cloneEvents(previous.events); editDrafts.clear();
  var t = (state.ts || [])[state.idx] || 0;
  selectedEvent = state.events.find(function (ev) { return ev.start <= t && t < ev.end; }) || null;
  renderEvents(); renderPending();
  return save();
}
// ---- end edit transactions ------------------------------------------------

function applyPendingRange() {
  try {
    var bounds = rangeBounds(state.pendingStart, state.pendingEnd, state.duration);
    var blocked = state.events.some(function (ev) {
      return ev.start < bounds.end && bounds.start < ev.end && draftChanged(ev, editDrafts.get(ev));
    });
    if (blocked) { throw new Error("覆盖范围内有未保存的区间草稿，请先保存本段修改"); }
    var next = relabelRange(state.events, bounds.start, bounds.end, $("kind-sel").value, state.duration);
    var selected = next.find(function (ev) { return ev.start === bounds.start; });
    state.pendingStart = state.pendingEnd = null; pendingTouched = false;
    commitEvents(next, selected, "范围改标");
  } catch (error) { toast(error.message); }
}

function setStart() {
  if (!state.ts.length) { return; }
  var t = state.ts[state.idx];
  state.pendingStart = t; pendingTouched = true;
  renderPending();
}

function setEnd() {
  if (!state.ts.length) { return; }
  var t = state.ts[state.idx];
  state.pendingEnd = t; pendingTouched = true;
  renderPending();
}

function clearPending() {
  state.pendingStart = state.pendingEnd = null;
  pendingTouched = false;
  renderPending();
}

function delEvent(k) {
  var event = state.events[k];
  if (draftChanged(event, editDrafts.get(event)) && !confirm("删除此段也会丢弃它的未保存修改，继续删除？")) { return; }
  var next = state.events.filter(function (_, index) { return index !== k; });
  commitEvents(next, null, "删除区间");
}

// 整段标负:写一条 0→末帧的 neg 区间,把整段无泥石流的视频显式标为已处理;
// 已有任何确认区间时拒绝(重叠禁止),想改请先删掉已有区间
function markAllNegative() {
  if (!state.ts.length) { return; }
  var s = state.ts[0], e = state.duration;
  if (s === e) { toast("视频太短(不足一个帧间隔),不成区间"); return; }
  var ov = findOverlap(s, e);
  if (ov) {
    toast("整段标负被拒绝:已有区间 " + ov.start + "–" + ov.end +
          "s 与之重叠(请先删除已有区间,或逐段标注)");
    return;
  }
  var created = { start: s, end: e, kind: "neg", note: "整段负样本" };
  commitEvents(state.events.concat([created]), created, "整段标负");
}

// 导出场景分段:把当前视频的 neg 区间作为场景段导出到 extneg-scenes.csv
// (段=外部负样本的划分原子)。neg 区间必须覆盖全片,有缺口会被服务端拒绝并列出缺口
function exportScenes() {
  fetch(api("/api/scenes"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}"
  }).then(function (r) {
    return r.json().then(function (j) { return { status: r.status, j: j }; });
  }).then(function (res) {
    if (res.status !== 200) { toast("导出被拒绝:" + (res.j.error || "未知错误")); return; }
    var kc = { pos: 0, transition: 0, neg: 0, cut: 0, unknown: 0 };
    res.j.segs.forEach(function (s) { kc[KIND_CN[s.kind] ? s.kind : "unknown"] += 1; });
    toast("已导出 " + res.j.n_seg + " 段(P" + kc.pos + "/T" + kc.transition + "/N" + kc.neg + "/U" + kc.unknown + "/cut" + kc.cut +
          ") -> extneg-scenes.csv");
  }).catch(function () { toast("导出失败(服务未响应)"); });
}

// ---- 保存 -----------------------------------------------------------------

function save() {
  // 每次捕获独立快照并顺序提交，防止后发请求先到后被旧快照覆盖。
  var snapshot = JSON.stringify({ events: state.events });
  savePending += 1;
  $("saved-at").textContent = "保存中…";
  saveChain = saveChain.then(function () {
    return fetch(api("/api/events"), {
      method: "POST", headers: { "Content-Type": "application/json" }, body: snapshot
    }).then(function (r) {
      return r.json().then(function (j) { return { status: r.status, j: j }; });
    }).then(function (res) {
      if (res.status !== 200) { throw new Error(res.j.error || "未知错误"); }
      if (savePending === 1) { $("saved-at").textContent = "已保存 " + res.j.saved_at; saveFailed = false; }
      return true;
    }).catch(function (error) {
      toast("保存失败：" + error.message + "。请重新保存或刷新以恢复已落盘内容。");
      if (savePending === 1) { $("saved-at").textContent = "保存失败，待重试"; saveFailed = true; }
      return false;
    }).finally(function () { savePending -= 1; });
  });
  return saveChain;
}

// ---- 渲染 -----------------------------------------------------------------

function loadState() {
  fetch(api("/api/state")).then(function (r) { return r.json(); }).then(function (d) {
    state.ts = d.ts;
    state.duration = d.duration;
    state.events = d.events;
    state.thumbIdx = d.thumb_idx;
    state.stripW = d.thumb_idx.length * 126;
    $("strip-inner").style.width = state.stripW + "px";
    $("video-name").textContent = d.video_id + "(共 " + d.n_frames + " 帧)";
    $("annotation-version").textContent = (d.annotations_dir ? "独立标签版本：" : "当前标签：") + d.csv_path;
    var versionName = d.annotations_dir ? d.annotations_dir.split(/[\\/]/).filter(Boolean).pop() : "原标签";
    $("version-short").textContent = versionName;
    $("version-short").title = d.annotations_dir || d.csv_path;
    $("btn-scenes").hidden = !d.scene_export_enabled;
    // 初始位置:默认跳到第一段未被任何 pos/cut/neg 区间覆盖处;勾选「从头开始」则从 0 开始
    var start = 0;
    if (!$("from-start").checked && state.events.length) {
      start = state.ts.length - 1;
      for (var i = 0; i < state.ts.length; i++) {
        var t = state.ts[i], covered = false;
        for (var k = 0; k < state.events.length; k++) {
          var ev = state.events[k];
          if (t >= ev.start && t < ev.end) { covered = true; break; }
        }
        if (!covered) { start = i; break; }
      }
    }
    var requestedTime = new URLSearchParams(location.search).get("t");
    if (requestedTime !== null && Number.isFinite(Number(requestedTime))) { start = frameAt(Number(requestedTime)); }
    selectedEvent = state.events.find(function (ev) { return ev.start <= state.ts[start] && state.ts[start] < ev.end; }) || null;
    renderEvents(); renderPending();
    setIdx(start);
    pollThumbs();
  });
}

function pollThumbs() {
  fetch(api("/api/state")).then(function (r) { return r.json(); }).then(function (d) {
    $("thumb-status").textContent = "缩略图 " + d.thumbs_ready + "/" + d.thumbs_total;
    if (d.thumbs_ready >= d.thumbs_total && d.thumbs_total > 0) {
      if (!state.thumbsBuilt) { buildStrip(); }
    } else {
      setTimeout(pollThumbs, 1500);
    }
  }).catch(function () { setTimeout(pollThumbs, 3000); });
}

// 头部下拉:列出 frames 根目录下所有视频(前缀标记标注状态 + 已确认正/剔计数),切换即整页跳转
var STATUS_MARK = { annotated: "✅ 已标 ", unlabeled: "⬜ 未标 " };
function loadVideoList() {
  Promise.all([fetch(api("/api/videos")).then(function (r) { return r.json(); }),
               fetch("/api/review").then(function (r) { return r.json(); })]).then(function (results) {
    var d = results[0]; reviewItems = results[1].items || []; videoItems = d.videos; currentVideo = d.current;
    $("list-sel").value = listMode;
    renderVideoOptions(); renderReview();
    $("video-sel").addEventListener("change", function () { navigateVideo(this.value); });
    $("list-sel").addEventListener("change", function () { setListMode(this.value); });
    var visible = videosForMode(videoItems, reviewItems, listMode);
    if (visible.length && !visible.some(function (v) { return v.name === currentVideo; })) {
      navigateVideo(visible[0].name);
    }
  });
}

function renderVideoOptions() {
    var sel = $("video-sel");
    sel.innerHTML = "";
    var videos = videosForMode(videoItems, reviewItems, listMode);
    $("list-sel").value = listMode;
    $("list-count").textContent = videos.length + " 条";
    $("list-count").title = listMode === "training" ? "按现役150条训练池清单列出，保留全P、全N及混合视频；成员身份不表示训练已获准" :
      (listMode === "all_n" ? "有明确 N，无 P/T/未知类型；cut 单独保留，不计作 N" : "当前列表视频数");
    videos.forEach(function (v) {
      var opt = document.createElement("option");
      opt.value = v.name;
      opt.textContent = v.name + " · " + (v.status === "annotated" ? "已标" : "未标") +
                        " · P" + v.n_pos + " T" + (v.n_transition || 0) + " N" + (v.n_neg || 0) + " cut" + v.n_cut +
                        (v.n_unknown ? " U/未知" + v.n_unknown : "");
      sel.appendChild(opt);
    });
    sel.value = currentVideo;
}

function navigateVideo(name, targetMode) {
  var mode = targetMode || listMode;
  var entry = reviewItems.find(function (r) { return r.unit_id === name; });
  var time = mode === "review" && entry && entry.regions.length ? entry.regions[0].start : undefined;
  var query = videoQuery(name, mode, time);
  return saveChain.then(function (ok) {
    if (!ok) { $("video-sel").value = currentVideo; $("list-sel").value = listMode; toast("最近一次保存失败，请先重新保存或刷新后再切换视频"); return false; }
    if ((hasUnsavedDrafts() || hasPendingRange()) && !confirm("有未保存的区间草稿或待应用范围。放弃并切换视频？")) {
      $("video-sel").value = currentVideo;
      $("list-sel").value = listMode;
      return false;
    }
    allowNavigation = true;
    location.href = "/" + query;
    return true;
  });
}

function setListMode(mode) {
  return saveChain.then(function (ok) {
    if (!ok) { $("list-sel").value = listMode; toast("最近一次保存失败，请先重新保存后再切换列表"); return false; }
    // 当前视频的标签可能刚被编辑；只刷新其列表计数，不改任何标签或纳入状态。
    var current = videoItems.find(function (v) { return v.name === currentVideo; });
    if (current && state.ts.length) {
      ["pos", "neg", "transition", "cut"].forEach(function (kind) {
        current["n_" + kind] = state.events.filter(function (e) { return e.kind === kind; }).length;
      });
      current.n_unknown = state.events.filter(function (e) { return e.kind === "unknown" || !KIND_CN[e.kind]; }).length;
    }
    var visible = videosForMode(videoItems, reviewItems, mode);
    if (!visible.length) { $("list-sel").value = listMode; toast("这个列表暂无视频"); return false; }
    if (!visible.some(function (v) { return v.name === currentVideo; })) { return navigateVideo(visible[0].name, mode); }
    // 不换视频时就地切列表，完整保留区间草稿、待应用范围和撤销栈。
    listMode = mode; reviewOnly = mode === "review"; renderVideoOptions();
    history.replaceState(null, "", "/" + videoQuery(currentVideo, mode, state.ts[state.idx]));
    return true;
  });
}

function renderReview() {
  $("review-box").hidden = !reviewItems.length;
  var entry = reviewItems.find(function (r) { return r.unit_id === currentVideo; });
  $("review-title").textContent = entry ? " · " + entry.category + " · " + entry.unit_id : " · 当前视频不在复核清单中";
  $("review-question").textContent = entry ? entry.question : "可在顶部选择“边界复核”返回本次审阅列表。";
  var holder = $("review-jumps"); holder.innerHTML = "";
  if (!entry) { return; }
  entry.regions.forEach(function (region) {
    var button = document.createElement("button");
    button.textContent = region.label + "（" + region.start + "–" + region.end + "秒）";
    button.addEventListener("click", function () { seekToT(region.start); });
    holder.appendChild(button);
    });
}

// 快速切换视频:P/N 或按钮,按头部下拉的列表顺序前后翻。
// 标注每次改动已即时落盘,直接整页跳转不会丢进度;到头/到尾只提示不环绕(避免绕晕方向感)
function switchVideo(delta) {
  var sel = $("video-sel");
  if (sel.selectedIndex < 0) { toast("视频列表尚未加载完,稍候再按"); return; }
  if (sel.options.length < 2) { toast("列表里只有当前一个视频"); return; }
  var i = sel.selectedIndex + delta;
  if (i < 0) { toast("已经是第一个视频"); return; }
  if (i >= sel.options.length) { toast("已经是最后一个视频"); return; }
  navigateVideo(sel.options[i].value);
}

function buildStrip() {
  state.thumbsBuilt = true;
  var holder = $("thumbs");
  holder.innerHTML = "";
  for (var k = 0; k < state.thumbIdx.length; k++) {
    var i = state.thumbIdx[k];
    var cell = document.createElement("div");
    cell.className = "cell";
    cell.setAttribute("data-frame", i);
    var img = document.createElement("img");
    img.src = api("/thumb?i=" + i);
    img.loading = "lazy";   // 横向滚动时按需加载,控制内存
    cell.appendChild(img);
    var lab = document.createElement("div");
    lab.className = "lab";
    lab.textContent = sec(state.ts[i]) + "s";
    cell.appendChild(lab);
    holder.appendChild(cell);
  }
  updateThumbClasses(); renderOverlay();
}

function updateThumbClasses() {
  var cells = document.querySelectorAll("#thumbs .cell");
  for (var c = 0; c < cells.length; c++) {
    var i = +cells[c].getAttribute("data-frame");
    var kind = classAt(state.events, state.ts[i]);
    var cls = "in-" + (kind === "pos" ? "ev" : kind);
    cells[c].classList.toggle("in-ev", cls === "in-ev");
    cells[c].classList.toggle("in-cut", cls === "in-cut");
    cells[c].classList.toggle("in-neg", cls === "in-neg");
    cells[c].classList.toggle("in-transition", cls === "in-transition");
    cells[c].classList.toggle("in-unknown", cls === "in-unknown");
    cells[c].classList.toggle("cur", i === state.idx);
  }
}

function scrollStrip() {
  var cell = document.querySelector('#thumbs .cell[data-frame="' + state.idx + '"]');
  if (!cell) { return; }
  var sc = $("scroller");
  var left = cell.offsetLeft, right = left + cell.offsetWidth;
  if (left < sc.scrollLeft + 20) { sc.scrollLeft = left - 20; }
  else if (right > sc.scrollLeft + sc.clientWidth - 20) { sc.scrollLeft = right - sc.clientWidth + 20; }
}

function renderOverlay() {
  var ov = $("overlay");
  if (!ov || !state.duration || !state.stripW) { return; }
  ov.innerHTML = "";
  function px(t) { return Math.round(t / state.duration * state.stripW); }
  function bar(cls, s, e, title) {
    var sp = document.createElement("span");
    sp.className = cls;
    sp.style.left = px(s) + "px";
    sp.style.width = Math.max(2, px(e) - px(s)) + "px";
    if (title) { sp.title = title; }
    ov.appendChild(sp);
  }
  for (var k = 0; k < state.events.length; k++) {
    var ev = state.events[k];
    var hideCut = $("hide-cut") && $("hide-cut").checked;
    if (hideCut && ev.kind === "cut") { continue; }   // 屏蔽剔除段:色条不画
    bar(ev.kind === "pos" ? "ev" : (KIND_CN[ev.kind] ? ev.kind : "unknown"), ev.start, ev.end,
        (KIND_LONG[ev.kind] || "未知类型 " + ev.kind) + " [" + sec(ev.start) + ", " + sec(ev.end) + ")s" +
        (ev.note ? " " + ev.note : ""));
  }
  if (state.pendingStart !== null && state.pendingEnd !== null) {
    try { var pending = rangeBounds(state.pendingStart, state.pendingEnd, state.duration);
      bar("pending-range", pending.start, pending.end, "待应用范围：尚未改写标签");
    } catch (_) { /* 非法范围只显示标记，不绘制覆盖预览。 */ }
  }
  if (state.pendingStart !== null) { bar("mark-s", state.pendingStart, state.pendingStart, "待定起点"); }
  if (state.pendingEnd !== null) { bar("mark-e", state.pendingEnd, state.pendingEnd, "待定终点"); }
  bar("caret", state.ts[state.idx] || 0, state.ts[state.idx] || 0, "当前帧");
}

function renderPending() {
  var s = state.pendingStart, e = state.pendingEnd, txt = "选类型 → S / E 设范围 → 应用标注；选点不会改写标签";
  var valid = false;
  if (s !== null && e === null) { txt = "待应用：起点 " + sec(s) + "s，移到终点按 E"; }
  else if (s === null && e !== null) { txt = "待应用：终点 " + sec(e) + "s，移到起点按 S"; }
  else if (s !== null && e !== null) {
    try {
      var bounds = rangeBounds(s, e, state.duration);
      var affected = state.events.filter(function (ev) { return ev.start < bounds.end && bounds.start < ev.end; });
      var kinds = Array.from(new Set(affected.map(function (ev) { return KIND_CN[ev.kind] || "未知(" + ev.kind + ")"; })));
      txt = "待应用 [" + sec(bounds.start) + ", " + sec(bounds.end) + ") → " + KIND_CN[$("kind-sel").value] +
        " · 影响 " + affected.length + " 段" + (kinds.length ? "（" + kinds.join(" / ") + "）" : "（原未标）") +
        "；保留分段备注" + (s > e ? "，已按时间先后排序" : "");
      valid = true;
    } catch (error) { txt = "待应用范围无效：" + error.message; }
  }
  [[$("range-start"), s], [$("range-end"), e]].forEach(function (pair) {
    if (document.activeElement !== pair[0]) { pair[0].value = pair[1] === null ? "" : pair[1]; }
  });
  $("btn-apply-range").disabled = !valid;
  $("btn-undo").disabled = !undoStack.length;
  $("btn-undo").title = undoStack.length ? "撤销最近一次：" + undoStack[undoStack.length - 1].action : "本视频尚无可撤销编辑";
  $("pending-info").textContent = txt;
  renderOverlay();
}

function renderEvents() {
  var evList = $("events-list");
  evList.innerHTML = "";
  $("event-count").textContent = state.events.length + " 段";
  if (!state.events.length) {
    evList.innerHTML = '<div class="empty-tip">暂无区间。用画面下方“范围改标”，设 S / E 后点击“应用标注”。</div>';
  }
  state.events.forEach(function (ev, k) {
    if ($("hide-cut") && $("hide-cut").checked && ev.kind === "cut") { return; }   // 屏蔽剔除段
    var row = document.createElement("article");
    row.className = "ev-row" + (selectedEvent === ev ? " selected" : "");
    var header = document.createElement("button");
    header.className = "event-select"; header.setAttribute("aria-expanded", String(selectedEvent === ev));
    header.title = "选择此段编辑；已有待应用范围时保持当前画面位置";
    var badge = document.createElement("span");
    badge.className = "badge " + (KIND_CN[ev.kind] === undefined ? "unknown" : ev.kind);
    badge.textContent = KIND_CN[ev.kind] || "未知";
    header.appendChild(badge);
    var t = document.createElement("span");
    t.className = "ev-time";
    t.textContent = sec(ev.start) + " – " + sec(ev.end) + " s";
    header.appendChild(t);
    var flag = document.createElement("span"); flag.className = "draft-flag";
    flag.textContent = draftChanged(ev, editDrafts.get(ev)) ? "未保存" : "";
    header.appendChild(flag);
    var editHint = document.createElement("span"); editHint.className = "edit-hint";
    editHint.textContent = selectedEvent === ev ? "收起" : "修改标签"; header.appendChild(editHint);
    header.addEventListener("click", function () {
      selectedEvent = selectedEvent === ev ? null : ev;
      renderEvents();
      if (selectedEvent && !hasPendingRange()) { seekToT(ev.start); }
    });
    row.appendChild(header); evList.appendChild(row);
    if (selectedEvent !== ev) {
      if (ev.note) { var summary = document.createElement("p"); summary.className = "ev-summary";
        summary.textContent = ev.note.length > 42 ? ev.note.slice(0, 42) + "…" : ev.note; row.appendChild(summary); }
      return;
    }
    var draft = editDrafts.get(ev);
    if (!draft) { draft = { start: String(ev.start), end: String(ev.end), kind: ev.kind, note: ev.note || "" };
      editDrafts.set(ev, draft); }
    var editor = document.createElement("div"); editor.className = "event-editor";
    row.appendChild(editor);
    function field(labelText, control, parent) {
      var label = document.createElement("label"); label.textContent = labelText;
      label.appendChild(control); (parent || editor).appendChild(label);
    }
    function updateDraft() {
      draft.start = start.value; draft.end = end.value; draft.kind = selector.value; draft.note = note.value;
      flag.textContent = draftChanged(ev, draft) ? "未保存" : "";
    }
    var selector = document.createElement("select");
    selector.setAttribute("aria-label", "区间类型");
    ["neg", "transition", "pos", "cut"].forEach(function (kind) {
      var option = document.createElement("option");
      option.value = kind; option.textContent = KIND_LONG[kind]; selector.appendChild(option);
    });
    if (ev.kind === "unknown" || !KIND_CN[ev.kind]) {
      var unknown = document.createElement("option"); unknown.value = ev.kind;
      unknown.textContent = ev.kind === "unknown" ? "历史类型未定 U（仅保留）" : "未知类型：" + ev.kind;
      unknown.disabled = true; selector.appendChild(unknown);
    }
    selector.value = draft.kind; field("修改本段标签", selector);
    var start = document.createElement("input"), end = document.createElement("input");
    var bounds = document.createElement("div"); bounds.className = "bound-fields"; editor.appendChild(bounds);
    [[start, "起点 / 秒", draft.start], [end, "终点 / 秒（不含）", draft.end]].forEach(function (entry) {
      var input = entry[0]; input.type = "number"; input.step = "any"; input.min = "0";
      input.setAttribute("aria-label", entry[1]); input.value = entry[2]; field(entry[1], input, bounds);
    });
    var note = document.createElement("textarea"); note.rows = 2;
    note.placeholder = "记录判断或疑问（可选）"; note.value = draft.note; field("备注", note);
    [selector, start, end, note].forEach(function (input) {
      input.addEventListener("input", updateDraft); input.addEventListener("change", updateDraft);
    });
    var actions = document.createElement("div"); actions.className = "editor-actions"; editor.appendChild(actions);
    var apply = document.createElement("button");
    apply.textContent = "保存本段"; apply.className = "primary";
    apply.addEventListener("click", function () {
      try {
        if (start.value.trim() === "" || end.value.trim() === "") { throw new Error("起止秒不能为空"); }
        var replacement = { start: Number(start.value), end: Number(end.value), kind: selector.value, note: note.value };
        var next = editedEvents(state.events, k, replacement);
        commitEvents(next, replacement, "修改区间");
      } catch (error) { toast(error.message); }
    });
    actions.appendChild(apply);
    var secondary = document.createElement("div"); secondary.className = "editor-secondary"; editor.appendChild(secondary);
    var split = document.createElement("button"); split.textContent = "在当前帧分割";
    split.title = "在当前帧真实时间分割已保存区间，保留原类型与备注；再分别编辑两段";
    split.addEventListener("click", function () {
      if (draftChanged(ev, draft)) { toast("请先保存本段修改，再执行分割"); return; }
      try { var splitAt = state.ts[state.idx];
        var next = splitEvents(state.events, k, splitAt);
        var selected = next.find(function (part) { return part.start === splitAt; });
        commitEvents(next, selected, "分割区间"); }
      catch (error) { toast(error.message); }
    });
    secondary.appendChild(split);
    var del = document.createElement("button");
    del.textContent = "删除此段"; del.className = "delete";
    del.addEventListener("click", function () { delEvent(k); });
    secondary.appendChild(del);
  });

  var active = evList.querySelector(".selected");
  if (active) {
    if (active.offsetTop < evList.scrollTop) { evList.scrollTop = active.offsetTop; }
    else if (active.offsetTop + active.offsetHeight > evList.scrollTop + evList.clientHeight) {
      evList.scrollTop = Math.max(0, active.offsetTop + active.offsetHeight - evList.clientHeight);
    }
  }
  renderOverlay(); updateThumbClasses(); updateTime();
}

// ---- 事件绑定 -------------------------------------------------------------

$("strip-inner").addEventListener("click", function (e) {
  if (e.target.tagName === "IMG") {   // 点缩略图:精确跳到那一帧
    var cell = e.target.parentElement;
    setIdx(+cell.getAttribute("data-frame"));
    return;
  }
  var rect = this.getBoundingClientRect();
  var frac = (e.clientX - rect.left) / rect.width;
  seekToT(frac * state.duration);
});
$("btn-start").addEventListener("click", setStart);
$("btn-end").addEventListener("click", setEnd);
$("btn-clear").addEventListener("click", clearPending);
$("btn-apply-range").addEventListener("click", applyPendingRange);
$("btn-undo").addEventListener("click", undoLastEdit);
[["range-start", "pendingStart"], ["range-end", "pendingEnd"]].forEach(function (pair) {
  $(pair[0]).addEventListener("input", function () {
    state[pair[1]] = this.value === "" ? null : Number(this.value);
    pendingTouched = true; renderPending();
  });
});
$("kind-sel").addEventListener("change", function () {
  try { localStorage.setItem("event-annotator-range-kind", this.value); } catch (_) { /* 本地存储不可用仍可标注。 */ }
  renderPending();
});
$("btn-neg-all").addEventListener("click", markAllNegative);
$("btn-scenes").addEventListener("click", exportScenes);
$("hide-cut").addEventListener("change", function () { renderOverlay(); renderEvents(); });
$("btn-prev-video").addEventListener("click", function () { switchVideo(-1); });
$("btn-next-video").addEventListener("click", function () { switchVideo(1); });
$("btn-prev-frame").addEventListener("click", function () { setIdx(state.idx - 1); });
$("btn-next-frame").addEventListener("click", function () { setIdx(state.idx + 1); });
$("btn-new").addEventListener("click", function () { $("new-interval").scrollIntoView({block: "nearest"}); $("kind-sel").focus(); });
window.addEventListener("beforeunload", function (event) {
  if (!allowNavigation && (hasUnsavedDrafts() || hasPendingRange() || savePending || saveFailed)) { event.preventDefault(); event.returnValue = ""; }
});

document.addEventListener("keydown", function (e) {
  if (e.target && (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT" || e.target.tagName === "SELECT")) { return; }
  if (e.ctrlKey || e.metaKey || e.altKey) { return; }
  var k = e.key, step = e.shiftKey ? 10 : 1;
  if (k === "ArrowLeft") { e.preventDefault(); setIdx(state.idx - step); }
  else if (k === "ArrowRight") { e.preventDefault(); setIdx(state.idx + step); }
  else if (k === "PageUp") { e.preventDefault(); setIdx(state.idx - 60); }
  else if (k === "PageDown") { e.preventDefault(); setIdx(state.idx + 60); }
  else if (k === "Home") { e.preventDefault(); setIdx(0); }
  else if (k === "End") { e.preventDefault(); setIdx(state.ts.length - 1); }
  else if (k === "s" || k === "S") { setStart(); }
  else if (k === "e" || k === "E") { setEnd(); }
  else if (k === "c" || k === "C") { clearPending(); }
  else if (k === "n" || k === "N") { switchVideo(1); }
  else if (k === "p" || k === "P") { switchVideo(-1); }
});

try { var preferredKind = localStorage.getItem("event-annotator-range-kind");
  if (["neg", "transition", "pos", "cut"].indexOf(preferredKind) >= 0) { $("kind-sel").value = preferredKind; }
} catch (_) { /* 缺省为 T。 */ }
loadVideoList();
loadState();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------- HTTP 服务

class Session:
    """一个视频的标注会话:帧索引、时间轴、标注状态、缩略图缓存各持一份,
    按 ?v= 惰性创建;缩略图预生成线程随会话启动(写磁盘缓存,不常驻内存)。

    两种来源(kind=dir 旧式单视频目录 / kind=key 汇总源里的一个 video_key),
    由 resolve_videos 给出的 spec 区分;key 模式下 events.csv 读写都落到
    源根目录的汇总文件(只动本 key 的行)。"""

    def __init__(self, spec: dict, name: str, thumbs_dir: Path):
        self.name = name
        if spec["kind"] == "key":
            self.dir = spec["root"]
            self.video_key = spec["video_key"]
            self.video_id = spec["video_id"]
            self.csv_path = spec.get("events_csv") or spec["root"] / "events.csv"
            frames_dir = spec["root"] / "frames"
            if not frames_dir.is_dir():
                raise SystemExit(f"找不到帧目录:{frames_dir}")
            self.frames_dir = frames_dir
            self.files, self.ts = load_frames_index(
                frames_dir, spec.get("index_csv") or spec["root"] / "frames_index.csv",
                video_key=spec["video_key"])
            self.events = load_events_consolidated(spec["root"], spec["video_key"],
                                                   events_csv=self.csv_path)
            self.consolidated = True
        else:
            video_dir = spec["dir"]
            self.dir = video_dir
            self.video_key = None
            self.video_id = video_dir.name
            self.csv_path = spec.get("events_csv") or video_dir / "events.csv"
            frames_dir = video_dir / "frames"
            if not frames_dir.is_dir():
                raise SystemExit(f"找不到帧目录:{frames_dir}")
            self.frames_dir = frames_dir
            self.files, self.ts = load_frames_index(
                frames_dir, spec.get("index_csv") or video_dir / "frames_index.csv")
            self.events = load_annotations(self.csv_path, self.video_id)
            self.consolidated = False
        self.duration = self.ts[-1] + (statistics.median([b - a for a, b in zip(self.ts, self.ts[1:])])
                                       if len(self.ts) > 1 else 2.0)
        self.thumb_idx = sample_indices(len(self.files))
        self.sample_set = set(self.thumb_idx)
        self.thumb_cache = ThumbCache(frames_dir, thumbs_dir, self.files, self.thumb_idx)
        self.lock = threading.Lock()
        threading.Thread(target=self.thumb_cache.pregenerate, daemon=True).start()

    def save_events(self, events: list[dict]) -> None:
        """按会话类型落盘:汇总源只替换本 video_key 的行,旧式目录整份重写。"""
        if self.consolidated:
            save_events_consolidated(self.dir, self.video_key, self.video_id,
                                     events, events_csv=self.csv_path)
        else:
            save_events_csv(self.csv_path, self.video_id, events)

    def frame_file(self, i: int) -> Path | None:
        """帧序号 → frames/ 内的文件;越界返回 None。"""
        if not isinstance(i, int) or i < 0 or i >= len(self.files):
            return None
        return self.frames_dir / self.files[i]


class AnnotatorHandler(BaseHTTPRequestHandler):
    """路由:GET / 页面;GET /api/videos 视频清单;GET /api/state?v= 全量状态;
    GET /img?v=&i= 原帧;GET /thumb?v=&i= 缩略图;POST /api/events?v= 整份提交区间;
    POST /api/scenes?v= 导出 neg 区间为场景分段。
    会话按 v 惰性创建,缺省用初始视频。"""

    # 由 make_server 注入(类属性,单进程内全局共享)
    videos: dict = {}            # 视频名 -> spec(resolve_videos 产出;key=名字)
    default_video: str = ""
    thumbs_base: Path | None = None   # 非 None 时缩略图统一放 <thumbs_base>/<视频名>/,否则 <视频目录>/thumbs/
    sessions: dict = {}
    sessions_lock = threading.Lock()
    review_file: Path | None = None

    def log_message(self, fmt, *args):  # 静默逐请求日志,只保留启动信息
        pass

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @classmethod
    def session_for(cls, name: str) -> "Session | None":
        """按名字取(或惰性建)视频会话;名字不在清单里返回 None(防路径穿越)。"""
        active = active_video_specs(cls.videos)
        name = (name or "").strip()
        if not name:
            name = cls.default_video if cls.default_video in active else next(iter(sorted(active)), "")
        spec = active.get(name)
        if spec is None:
            return None
        with cls.sessions_lock:
            sess = cls.sessions.get(name)
            if sess is None:
                if cls.thumbs_base:
                    thumbs_dir = cls.thumbs_base / name
                elif spec["kind"] == "key":
                    thumbs_dir = spec["root"] / "thumbs" / spec["video_key"]
                else:
                    thumbs_dir = spec["dir"] / "thumbs"
                try:
                    sess = Session(spec, name, thumbs_dir)
                except (OSError, SystemExit, ValueError) as e:
                    print(f"[会话] 打开视频 {name} 失败:{e}")
                    return None
                cls.sessions[name] = sess
            return sess

    def _resolve(self, parsed) -> tuple["Session | None", dict]:
        """从查询串解析会话(缺省用初始视频)与查询参数。"""
        qs = urllib.parse.parse_qs(parsed.query)
        return self.session_for((qs.get("v") or [""])[0]), qs

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            name = (urllib.parse.parse_qs(parsed.query).get("v") or [""])[0]
            if name and name not in active_video_specs(self.videos):
                self._json({"error": "未知视频或来源已退役"}, 404)
                return
            body = PAGE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/review":
            items = []
            if self.review_file:
                with self.review_file.open(encoding="utf-8-sig") as stream:
                    items = json.load(stream)["items"]
                active = active_video_specs(self.videos)
                items = [item for item in items if item["unit_id"] in active]
            self._json({"items": items})
        elif parsed.path == "/api/videos":
            active = active_video_specs(self.videos)
            videos = []
            # 汇总源视频:状态/计数按 video_key 从汇总 events.csv 统计(整文件只读一遍)
            cons_events = {spec["root"]: spec.get("events_csv")
                           for spec in active.values() if spec["kind"] == "key"}
            cons_cache: dict[Path, tuple] = {}
            for r, r_events in cons_events.items():
                source_events = _read_consolidated_rows(r, r_events)
                keys_in_root = [n for n, s in active.items()
                                if s["kind"] == "key" and s["root"] == r]
                for n in sorted(keys_in_root):
                    events = source_events.get(active[n]["video_key"], [])
                    counts = kind_counts(events)
                    videos.append({"name": n, **{"n_" + k: v for k, v in counts.items()},
                                   "status": "annotated" if events else "unlabeled"})
            # 旧式单视频目录:状态/计数按目录内 events.csv
            for n, spec in active.items():
                if spec["kind"] != "dir":
                    continue
                events = load_annotations(spec.get("events_csv") or spec["dir"] / "events.csv", n)
                counts = kind_counts(events)
                videos.append({"name": n, **{"n_" + k: v for k, v in counts.items()},
                               "status": "annotated" if events else "unlabeled"})
            videos.sort(key=lambda v: v["name"])
            training_units = training_pool_units()
            for video in videos:
                video["in_training_pool"] = video["name"] in training_units
            qs = urllib.parse.parse_qs(parsed.query)
            current = (qs.get("v") or [""])[0] or self.default_video
            if current not in active:
                current = next(iter(sorted(active)), "")
            self._json({"videos": videos, "current": current,
                        "training_count": sum(video["in_training_pool"] for video in videos)})
        elif parsed.path == "/api/state":
            sess, _ = self._resolve(parsed)
            if sess is None:
                self._json({"error": "未知视频(v 无效或目录缺 frames_index.csv)"}, 404)
                return
            with sess.lock:
                events = [dict(e) for e in sess.events]
            self._json({
                "video_id": sess.video_id, "dir": str(sess.dir),
                "csv_path": str(sess.csv_path),
                "annotations_dir": str(self.videos[sess.name].get("annotations_dir", "")),
                "scene_export_enabled": not bool(self.videos[sess.name].get("annotations_dir")),
                "n_frames": len(sess.files), "ts": sess.ts, "duration": sess.duration,
                "events": events,
                "thumb_idx": sess.thumb_idx,
                "thumbs_ready": sess.thumb_cache.ready,
                "thumbs_total": sess.thumb_cache.total,
            })
        elif parsed.path in ("/img", "/thumb"):
            sess, qs = self._resolve(parsed)
            if sess is None:
                self._json({"error": "未知视频(v 无效或目录缺 frames_index.csv)"}, 404)
                return
            try:
                i = int((qs.get("i") or [""])[0])
            except ValueError:
                i = -1
            if parsed.path == "/img":
                full = sess.frame_file(i)
                if full is None or not full.is_file():
                    self._json({"error": "帧序号越界或文件缺失", "i": i}, 404)
                    return
                try:
                    data = full.read_bytes()
                except OSError:
                    self._json({"error": "帧读取失败", "i": i}, 500)
                    return
            else:
                if i not in sess.sample_set:
                    self._json({"error": "非抽样帧,无缩略图", "i": i}, 404)
                    return
                if not sess.thumb_cache.ensure(i):
                    self._json({"error": "缩略图生成失败", "i": i}, 500)
                    return
                try:
                    data = sess.thumb_cache.thumb_path(i).read_bytes()
                except OSError:
                    self._json({"error": "缩略图读取失败", "i": i}, 500)
                    return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._json({"error": "unknown path"}, 404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path not in ("/api/events", "/api/scenes"):
            self._json({"error": "unknown path"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            self._json({"error": "bad json"}, 400)
            return
        if not isinstance(payload, dict):
            self._json({"error": "请求必须是对象"}, 400)
            return
        sess, _ = self._resolve(parsed)
        if sess is None:
            self._json({"error": "未知视频(v 无效或目录缺 frames_index.csv)"}, 404)
            return
        if parsed.path == "/api/events":
            try:
                events = normalize_events(payload.get("events", []))
            except ValueError as e:
                self._json({"error": str(e)}, 400)
                return
            with sess.lock:
                sess.save_events(events)
                sess.events[:] = events
            self._json({"ok": True, "saved_at": time.strftime("%H:%M:%S"), "n_events": len(events)})
            return
        # /api/scenes:把当前视频的已确认区间导出为场景分段(段=外部视频的划分原子)。
        # 口径(叶海峰 2026-09-15 定,外部视频不只负样本):各类型区间(pos/neg/cut)
        # 都算场景段;未标空隙=隐式负段(与"区间外默认负"同口径);
        # 帧归属左闭右开,每帧恰属一段,不许漏段。
        if parsed.path == "/api/scenes":
            if self.videos[sess.name].get("annotations_dir"):
                self._json({"error": "独立标签版本不使用旧版隐式负场景导出"}, 400)
                return
            with sess.lock:
                marks = sorted(({"start": e["start"], "end": e["end"],
                                 "kind": e.get("kind") or "pos",
                                 "note": e.get("note", "")}
                                for e in sess.events), key=lambda x: x["start"])
            if not marks:
                self._json({"error": "没有已确认区间——泥石流场景标 pos,无泥石流可留空"
                                    "(默认负)或标 neg/cut,然后重新点导出"}, 400)
                return
            ts0, ts1 = int(round(sess.ts[0])), int(round(sess.ts[-1]))
            step = int(round((sess.ts[-1] - sess.ts[0]) / max(1, len(sess.ts) - 1))) \
                if len(sess.ts) > 1 else 2
            spans: list[tuple[int, int, str, str]] = []
            cur = ts0
            for m in marks:
                if m["start"] - cur > step:
                    spans.append((cur, m["start"], "neg", "未标默认负"))
                spans.append((m["start"], m["end"], m["kind"], m["note"]))
                cur = max(cur, m["end"])
            if ts1 - cur > step:
                spans.append((cur, ts1, "neg", "未标默认负"))
            # 帧归属按左闭右开:段 si 计 [start_si, start_{si+1}) 内的帧,
            # 边界帧归前一段(如 529 与 531 之间的 t=530 归前段)
            bounds = [s for s, _, _, _ in spans] + [ts1 + 1]
            segs = []
            for si, (s, e, k, note) in enumerate(spans):
                n_fr = sum(1 for t in sess.ts if s <= t < bounds[si + 1])
                segs.append({"seg_id": si + 1, "start": s, "end": e, "kind": k,
                             "note": note, "n_frames": n_fr})
            save_scenes_csv(self.scenes_csv, sess.video_id, segs)
            self._json({"ok": True, "n_seg": len(segs), "segs": segs,
                        "csv": str(self.scenes_csv)})
            return


def make_server(videos: dict, default_video: str, thumbs_base: Path | None,
                port: int) -> ThreadingHTTPServer:
    AnnotatorHandler.sessions = {}
    AnnotatorHandler.videos = videos
    AnnotatorHandler.default_video = default_video
    AnnotatorHandler.thumbs_base = thumbs_base
    AnnotatorHandler.scenes_csv = SCENES_CSV
    return ThreadingHTTPServer(("127.0.0.1", port), AnnotatorHandler)


def _is_consolidated_source(d: Path) -> bool:
    """汇总源判定:有 mapping.csv 且(前缀变体)frames_index.csv 的表头带 video_key 列
    (全源帧平铺在一个 frames/、events/索引按 video_key 区分的多视频目录)。"""
    if not (d / "mapping.csv").is_file():
        return False
    idx = _find_legacy_index(d)
    if idx is None:
        return False
    with idx.open("r", encoding="utf-8-sig", newline="") as f:
        header = csv.reader(f).__next__()
    return "video_key" in [h.strip().lstrip("﻿") for h in header]


def _find_legacy_index(d: Path) -> Path | None:
    """旧式单视频目录的 frames_index.csv:精确名优先,兼容 video2_ 前缀变体
    (video2_segment 的索引叫 video2_frames_index.csv,避免与汇总源文件名冲突)。"""
    exact = d / "frames_index.csv"
    if exact.is_file():
        return exact
    cands = sorted(x for x in d.glob("*frames_index.csv") if x.is_file())
    return cands[0] if cands else None


def _find_legacy_events(d: Path) -> Path | None:
    """旧式目录的 events.csv,前缀变体同理(如 video2_events.csv)。
    其他 *scenes*/*frames* CSV 不匹配 *events.csv,不会被误认。"""
    exact = d / "events.csv"
    if exact.is_file():
        return exact
    cands = sorted(x for x in d.glob("*events.csv") if x.is_file())
    return cands[0] if cands else None


def resolve_videos(dir_arg: str) -> tuple[dict, str]:
    """--dir 兼容三种值,产出 视频名 -> spec 注册表与默认视频名:
    1. 汇总源目录(如 data/process/video_frames/video1_segment)→ 其 mapping.csv
       的每个 video_key 注册为一个视频(events/索引读写都落汇总文件);
    2. 多源父目录(如 data/process/video_frames)→ 汇总源子目录按规则 1 展开,
       其余含(前缀变体)frames_index.csv 的子目录按旧式单视频注册;
    3. 旧式单视频目录(含 frames_index.csv 无 mapping.csv)→ 注册自身。
    """
    p = Path(dir_arg)
    if not p.is_dir():
        raise SystemExit(f"目录不存在:{p}")

    def register_consolidated(root: Path) -> None:
        idx_csv = _find_legacy_index(root)
        ev_csv = _find_legacy_events(root)
        with (root / "mapping.csv").open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = (row.get("video_key") or "").strip()
                if not key:
                    continue
                if (row.get("excluded") or "").strip() == "1":
                    continue  # 已剔除视频(如帧已删的 v5_08),不注册避免打开报错
                videos[key] = {"kind": "key", "root": root, "video_key": key,
                               "video_id": (row.get("video_id") or key).strip(),
                               "index_csv": idx_csv, "events_csv": ev_csv}

    def register_legacy(d: Path) -> None:
        idx = _find_legacy_index(d)
        if idx:
            videos[d.name] = {"kind": "dir", "dir": d, "index_csv": idx,
                              "events_csv": _find_legacy_events(d)}

    videos: dict = {}
    if _is_consolidated_source(p):           # 规则 1:本身就是汇总源
        register_consolidated(p)
    else:
        for d in sorted(p.iterdir()):        # 规则 2:父目录扫描
            if not d.is_dir():
                continue
            if _is_consolidated_source(d):
                register_consolidated(d)
            else:
                register_legacy(d)
        if not videos:
            register_legacy(p)               # 规则 3:单视频
    videos = active_video_specs(videos)
    if not videos:
        raise SystemExit(f"{p} 下没有可标注的视频(汇总源需含 mapping.csv,单视频需含 frames_index.csv)")
    return videos, sorted(videos)[0]


def apply_annotations_dir(videos: dict, annotations_dir: Path) -> dict:
    """逐源初始化完整 CSV 副本并重定向标签；帧、索引和缩略图仍在原处。

    sources.json 保留首次复制时的来源与哈希。已有版本绝不重新覆盖。
    仓库内源使用相对路径生成稳定标识，搬迁整个仓库不改变映射。
    """
    annotations_dir = annotations_dir.resolve()
    manifest_path = annotations_dir / "sources.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8-sig"))
                if manifest_path.is_file() else {"schema_version": 1, "sources": {}})
    sources = manifest["sources"]
    result = {}
    changed = False
    for name, original in videos.items():
        spec = dict(original)
        source_root = (spec["root"] if spec["kind"] == "key" else spec["dir"]).resolve()
        source_csv = (spec.get("events_csv") or source_root / "events.csv").resolve()
        try:
            source_id = source_root.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            source_id = source_root.as_posix()
        source_key = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:12]
        entry = sources.get(source_key)
        if entry is None:
            relative_csv = Path(f"{source_root.name}--{source_key}") / "events.csv"
            entry = {"source_id": source_id, "origin_csv": str(source_csv),
                     "origin_sha256": hashlib.sha256(source_csv.read_bytes()).hexdigest()
                     if source_csv.is_file() else None,
                     "annotations_csv": relative_csv.as_posix()}
            changed = True
        elif entry["source_id"] != source_id:
            raise ValueError("标签版本来源标识冲突")
        target_csv = (annotations_dir / entry["annotations_csv"]).resolve()
        if annotations_dir not in target_csv.parents or target_csv == source_csv:
            raise ValueError("标签版本CSV必须位于独立版本目录内")
        if not target_csv.is_file():
            if source_key in sources:
                raise ValueError(f"版本CSV缺失，拒绝用后来原标签重新初始化：{target_csv}")
            target_csv.parent.mkdir(parents=True, exist_ok=True)
            if source_csv.is_file():
                shutil.copy2(source_csv, target_csv)
            else:
                fields = _CONSOLIDATED_FIELDS if spec["kind"] == "key" else EVENTS_FIELDS
                with target_csv.open("w", encoding="utf-8-sig", newline="") as stream:
                    csv.writer(stream).writerow(fields)
        sources[source_key] = entry
        spec["origin_events_csv"] = source_csv
        spec["events_csv"] = target_csv
        spec["annotations_dir"] = annotations_dir
        result[name] = spec
    if changed:
        annotations_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    return result


def current_review_annotations(input_dir: str) -> Path | None:
    """Use the registered working copy only for the project's audit media tree."""
    media_root = (REPO_ROOT / "data/process/video_frames").resolve()
    selected = Path(input_dir).resolve()
    if selected != media_root and media_root not in selected.parents:
        return None
    pointer = REPO_ROOT / "data/metadata/annotation-revisions/current-review.json"
    if not pointer.is_file():
        return None
    current = json.loads(pointer.read_text(encoding="utf-8-sig"))
    version = (REPO_ROOT / current["working_annotations_dir"]).resolve()
    revisions = (REPO_ROOT / "data/metadata/annotation-revisions").resolve()
    if revisions not in version.parents or not (version / "sources.json").is_file():
        raise ValueError("当前审计工作版登记无效，拒绝悄悄退回旧标签")
    return version


def run(args: argparse.Namespace) -> None:
    videos, default_video = resolve_videos(args.dir)
    if args.original_labels and args.annotations_dir:
        raise ValueError("--original-labels与--annotations-dir不可同时使用")
    annotations_dir = (Path(args.annotations_dir) if args.annotations_dir else
                       None if args.original_labels else current_review_annotations(args.dir))
    if annotations_dir:
        videos = apply_annotations_dir(videos, annotations_dir)
    thumbs_base = Path(args.thumbs_dir) if args.thumbs_dir else None
    AnnotatorHandler.review_file = Path(args.review_file).resolve() if args.review_file else None
    if AnnotatorHandler.review_file:
        with AnnotatorHandler.review_file.open(encoding="utf-8-sig") as stream:
            review = json.load(stream)
        if review.get("items"):
            default_video = review["items"][0]["unit_id"]
            if default_video not in videos:
                raise SystemExit("复核清单首项不是现役视频: " + default_video)

    port = args.port
    server = None
    for _ in range(10):
        try:
            server = make_server(videos, default_video, thumbs_base, port)
            break
        except OSError:
            print(f"[提示] 端口 {port} 被占用,换下一个")
            port += 1
    if server is None:
        raise SystemExit(f"端口 {args.port}~{port} 都被占用")

    # 默认视频的会话先建好(缩略图预生成线程随之启动),启动信息直接给出恢复情况
    sess = AnnotatorHandler.session_for(default_video)
    if sess is None:
        raise SystemExit(f"默认视频打不开:{default_video}")
    if sess.events:
        print(f"[恢复] {default_video} 已有确认区间 {len(sess.events)} 段:" +
              ", ".join(f"{e['start']}-{e['end']}s" for e in sess.events))

    n_videos = len(videos)
    print("=" * 60)
    print("事件区间标注工作台已启动(仅监听 127.0.0.1,不联网)")
    print(f"  地址:http://127.0.0.1:{port}/")
    print(f"  标注根目录:{Path(args.dir).resolve()}")
    print(f"  默认视频:{default_video}(页面头部下拉可切换,共 {n_videos} 个视频)")
    print(f"  帧数:{len(sess.files)}  时长≈{sess.duration / 60:.1f} 分钟")
    print(f"  标注落盘:{sess.csv_path.resolve()}")
    print("  快捷键:S 设起点 / E 设终点 / ←→ 步进 / PgUp PgDn 跳转")
    print("  按 Ctrl+C 停止")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。标注保存在:", sess.csv_path.resolve())


# ---------------------------------------------------------------- 自测

def _selftest_write_jpeg(path: Path, shade: int) -> None:
    """写一张 8x8 微型 JPEG(主视图/缩略图都要求能真解码,不能用空文件)。"""
    img = np.full((8, 8, 3), (shade % 255, 60, 120), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok, f"JPEG 编码失败:{path}"
    buf.tofile(path)


def _selftest_build_fixture(tmp: Path) -> tuple[Path, Path]:
    """合成双模式夹具,返回 (frames_root, 汇总源目录):

    <tmp>/frames_root/                  # --dir 取它,走 resolve_videos 规则 2(父目录扫描)
      cons_src/                         # 汇总源:key1/key2 各 3 帧(t=0,2,4)平铺一个 frames/
        frames_index.csv(带 video_key 列) + mapping.csv + events.csv(key2 预填 1 行 pos)
      plain_video/                      # 旧式单视频目录:5 帧 t=0..8,无 events.csv
    """
    root = tmp / "frames_root"
    cons = root / "cons_src"
    plain = root / "plain_video"
    (cons / "frames").mkdir(parents=True)
    (plain / "frames").mkdir(parents=True)

    idx_rows = []
    for n, t in enumerate((0, 2, 4)):
        for key in ("key1", "key2"):
            name = f"{key}_{n:06d}_t{t}s.jpg"
            _selftest_write_jpeg(cons / "frames" / name, n * 40)
            idx_rows.append((key, name, t))
    with (cons / "frames_index.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_key", "frame_file", "t_sec", "sheet"])
        w.writerows([[k, n, t, 0] for k, n, t in idx_rows])
    with (cons / "mapping.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_key", "video_id", "site", "date", "excluded"])
        w.writerow(["key1", "vid-a", "selftest", "", "0"])
        w.writerow(["key2", "vid-b", "selftest", "", "0"])
    with (cons / "events.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(_CONSOLIDATED_FIELDS)
        w.writerow(["vid-b", "key2", 0, 4, "real", "pos", "预填整段正样本(key2 隔离参照)"])

    plain_idx = []
    for n, t in enumerate((0, 2, 4, 6, 8)):
        name = f"frame_{n:06d}_t{t}s.jpg"
        _selftest_write_jpeg(plain / "frames" / name, n * 30)
        plain_idx.append((name, t))
    with (plain / "frames_index.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_file", "t_sec", "sheet"])
        w.writerows([[n, t, 0] for n, t in plain_idx])
    return root, cons


def selftest(args: argparse.Namespace) -> None:
    """--selftest:全部跑在临时合成夹具上,不碰真实标注。
    测:annotation_status 纯函数 → resolve_videos 三规则 → 起 HTTP 服务 →
    GET /、/api/videos(汇总源按 video_key 计数,status 状态字段)、/api/state、
    /img、/thumb、非法输入 400/404 → POST 区间(dir 模式落本目录 events.csv;
    key 模式只替换本 video_key 的行、其他 key 行原样保留)→ /api/scenes 导出
    → 打印 PASS。"""
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="annotator-selftest-"))
    server = None
    try:
        # --- 纯函数级:annotation_status(旧式目录口径)。
        # 覆盖:无文件 / 仅模板行 / real cut 行 / 旧 5 列无 kind 列(向后兼容)。
        st_dir = tmp / "status-only"
        st_dir.mkdir()
        csv_path_st = st_dir / "events.csv"

        def expect_status(exp: str, label: str) -> None:
            nonlocal ok
            got = annotation_status(st_dir)
            if got != exp:
                ok = False
                print(f"[自测失败] annotation_status({label}):得到 {got!r},期望 {exp!r}")

        expect_status("unlabeled", "无 events.csv")
        csv_path_st.write_bytes(
            "﻿video_id,start_sec,end_sec,confidence,kind,note\n"
            f"{st_dir.name},,,,,示例行:填实际起止秒后删除本行\n"
            "\n".encode("utf-8"))
        expect_status("unlabeled", "仅模板行")
        with csv_path_st.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([st_dir.name, 10, 20, "real", "cut", "自测剔除段"])
        expect_status("annotated", "有 real cut 行")
        csv_path_st.write_bytes(
            "video_id,start_sec,end_sec,confidence,note\n"
            f"{st_dir.name},10,20,real,旧格式一行\n".encode("utf-8"))
        expect_status("annotated", "旧 5 列 real 行(无 kind 列)")

        # --- 夹具 + resolve_videos(规则 2:父目录扫描)
        root, cons = _selftest_build_fixture(tmp)
        videos, default_video = resolve_videos(str(root))
        if set(videos) != {"key1", "key2", "plain_video"} \
                or default_video != "key1" \
                or videos["key1"]["kind"] != "key" or videos["key1"]["video_id"] != "vid-a" \
                or videos["plain_video"]["kind"] != "dir":
            ok = False
            print(f"[自测失败] resolve_videos 注册表不符:{videos} / 默认 {default_video!r}")

        server = make_server(videos, default_video, tmp / "thumbs", 0)
        scenes_tmp = tmp / "thumbs" / "extneg-scenes.csv"   # 自测导出走临时路径
        AnnotatorHandler.scenes_csv = scenes_tmp
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{port}"

        def get(path: str, expect: int = 200) -> bytes:
            nonlocal ok
            try:
                with urllib.request.urlopen(base + path, timeout=30) as resp:
                    if resp.status != expect:
                        ok = False
                        print(f"[自测失败] GET {path} 状态 {resp.status}(期望 {expect})")
                    return resp.read()
            except urllib.error.HTTPError as e:
                if e.code != expect:
                    ok = False
                    print(f"[自测失败] GET {path} 状态 {e.code}(期望 {expect})")
                return e.read()

        def post(path: str, obj: dict, expect: int = 200) -> dict:
            nonlocal ok
            req = urllib.request.Request(
                base + path, data=json.dumps(obj).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    if resp.status != expect:
                        ok = False
                        print(f"[自测失败] POST {path} 状态 {resp.status}(期望 {expect})")
                    return body
            except urllib.error.HTTPError as e:
                body = json.loads(e.read().decode("utf-8"))
                if e.code != expect:
                    ok = False
                    print(f"[自测失败] POST {path} 状态 {e.code}(期望 {expect})")
                return body

        def vid(name: str) -> dict | None:
            vids = json.loads(get("/api/videos").decode("utf-8"))
            return next((v for v in vids["videos"] if v["name"] == name), None)

        page = get("/")
        if b"mainimg" not in page or b"video-sel" not in page \
                or b"btn-next-video" not in page or b"btn-prev-video" not in page \
                or b"btn-scenes" not in page:
            ok = False
            print("[自测失败] 首页缺少主视图、视频下拉、上/下一视频或导出场景分段按钮元素")

        # /api/videos:三种模式共 3 个视频;key2 由预填 pos 行判已标
        vids = json.loads(get("/api/videos").decode("utf-8"))
        if {v["name"] for v in vids["videos"]} != {"key1", "key2", "plain_video"} \
                or vids.get("current") != default_video:
            ok = False
            print(f"[自测失败] /api/videos 清单或 current 不符:{vids}")
        k2 = vid("key2")
        if k2 is None or k2["status"] != "annotated" or k2["n_pos"] != 1:
            ok = False
            print(f"[自测失败] key2 应为已标(预填 1 行 pos):{k2}")
        k1 = vid("key1")
        if k1 is None or k1["status"] != "unlabeled":
            ok = False
            print(f"[自测失败] key1 应为未标:{k1}")

        # /api/state:key 模式的 video_id 来自 mapping.csv,区间只取本 key 的 real 行
        state = json.loads(get("/api/state?v=key1").decode("utf-8"))
        if state["video_id"] != "vid-a" or state["n_frames"] != 3 \
                or state["events"] != [] \
                or not state["ts"] or state["duration"] <= 0 \
                or not state["thumb_idx"] or len(state["thumb_idx"]) > THUMB_COUNT:
            ok = False
            print(f"[自测失败] /api/state(v=key1)不符:{state}")
        st2 = json.loads(get("/api/state?v=key2").decode("utf-8"))
        if st2["video_id"] != "vid-b" or len(st2["events"]) != 1 \
                or st2["events"][0]["start"] != 0 or st2["events"][0]["end"] != 4 \
                or st2["events"][0]["kind"] != "pos":
            ok = False
            print(f"[自测失败] /api/state(v=key2)应读到预填 pos 行:{st2}")

        # /img、/thumb、越界/穿越/缺省
        if not get("/img?i=0&v=key1"):
            ok = False
            print("[自测失败] /img 返回为空")
        si = state["thumb_idx"][0]
        if not get(f"/thumb?i={si}&v=key1"):
            ok = False
            print("[自测失败] /thumb 返回为空")
        sess = AnnotatorHandler.sessions.get("key1")
        if sess is None or not sess.thumb_cache.thumb_path(si).is_file():
            ok = False
            print("[自测失败] 缩略图未落盘缓存")
        if b"error" not in get("/img?i=999999&v=key1", expect=404):
            ok = False
            print("[自测失败] 越界帧序号未被拒绝")
        if b"error" not in get("/api/state?v=..%2F..%2Fetc", expect=404):
            ok = False
            print("[自测失败] 非法 v(路径穿越)未被拒绝")
        if json.loads(get("/api/state").decode("utf-8"))["video_id"] != "vid-a":
            ok = False
            print("[自测失败] 缺省 v 应落到默认视频 key1(vid-a)")

        # --- dir 模式 POST(plain_video):坏行/非法 kind/同类型与跨类型重叠都拒绝
        pq = "v=plain_video"
        post("/api/events?" + pq, {"events": [{"start": 20, "end": 10, "note": "坏行"}]}, expect=400)
        post("/api/events?" + pq,
             {"events": [{"start": 10, "end": 20, "kind": "bad", "note": ""}]}, expect=400)
        post("/api/events?" + pq, {"events": [{"start": 0, "end": 4, "note": ""},
                                              {"start": 2, "end": 6, "note": ""}]}, expect=400)
        post("/api/events?" + pq, {"events": [{"start": 0, "end": 4, "kind": "pos"},
                                              {"start": 2, "end": 6, "kind": "cut"}]}, expect=400)
        # 清空 → 旧式目录应只剩模板行,状态未标
        post("/api/events?" + pq, {"events": []}, expect=200)
        rows = list(csv.DictReader((root / "plain_video" / "events.csv")
                                   .open("r", encoding="utf-8-sig", newline="")))
        if len(rows) != 1 or rows[0]["start_sec"] != "":
            ok = False
            print(f"[自测失败] dir 清空后应只剩模板行,实际:{rows}")
        if (vid("plain_video") or {}).get("status") != "unlabeled":
            ok = False
            print("[自测失败] dir 清空后 status 应为 unlabeled")

        # dir 模式合法提交(cut/pos/neg 各一段;未写 kind 按 pos),校验落盘
        post("/api/events?" + pq, {"events": [{"start": 0, "end": 2, "kind": "cut", "note": "自测剔除段"},
                                              {"start": 4, "end": 6, "note": "自测一段"},
                                              {"start": 8, "end": 10, "kind": "neg", "note": "整段负样本"}]},
             expect=200)
        rows = list(csv.DictReader((root / "plain_video" / "events.csv")
                                   .open("r", encoding="utf-8-sig", newline="")))
        if len(rows) != 3 or rows[0]["video_id"] != "plain_video" \
                or rows[0]["start_sec"] != "0" or rows[0]["end_sec"] != "2" \
                or rows[0]["confidence"] != "real" or rows[0]["kind"] != "cut" \
                or rows[1]["start_sec"] != "4" or rows[1]["kind"] != "pos" \
                or rows[2]["kind"] != "neg" or rows[2]["note"] != "整段负样本":
            ok = False
            print(f"[自测失败] dir 模式 events.csv 内容不符:{rows}")
        pv = vid("plain_video")
        if pv is None or pv["n_pos"] != 1 or pv["n_cut"] != 1 or pv["n_neg"] != 1 \
                or pv["status"] != "annotated":
            ok = False
            print(f"[自测失败] /api/videos(dir)正/剔/负计数或 status 不符:{pv}")

        # --- /api/scenes(dir 模式):空区间 400;有区间时空隙成隐式负段
        st_plain = json.loads(get("/api/state?" + pq).decode("utf-8"))
        post("/api/events?" + pq, {"events": []}, expect=200)
        j = post("/api/scenes?" + pq, {}, expect=400)
        if "没有已确认区间" not in j.get("error", ""):
            ok = False
            print(f"[自测失败] /api/scenes 空区间未拒:{j}")
        # cut[0,2] + pos[6,8]:中间空隙 2→6(>间隔 2)成隐式负段 → 段=[cut, 隐负, pos]
        post("/api/events?" + pq, {"events": [{"start": 0, "end": 2, "kind": "cut", "note": "自测剔除段"},
                                              {"start": 6, "end": 8, "note": "自测一段"}]},
             expect=200)
        j = post("/api/scenes?" + pq, {}, expect=200)
        segs = j.get("segs", [])
        if j.get("n_seg") != 3 \
                or [s["kind"] for s in segs] != ["cut", "neg", "pos"] \
                or segs[1]["note"] != "未标默认负" \
                or sum(s["n_frames"] for s in segs) != st_plain["n_frames"]:
            ok = False
            print(f"[自测失败] /api/scenes 隐式负段结果不符:{j}")
        # 两段 neg 覆盖全片 → 2 段且帧数守恒
        post("/api/events?" + pq, {"events": [
            {"start": 0, "end": 4, "kind": "neg", "note": "自测场景甲"},
            {"start": 4, "end": 8, "kind": "neg", "note": "自测场景乙"}]},
             expect=200)
        j = post("/api/scenes?" + pq, {}, expect=200)
        segs = j.get("segs", [])
        if j.get("n_seg") != 2 or len(segs) != 2 \
                or segs[0]["n_frames"] + segs[1]["n_frames"] != st_plain["n_frames"]:
            ok = False
            print(f"[自测失败] /api/scenes 分段结果不符:{j}")
        if not scenes_tmp.is_file():
            ok = False
            print(f"[自测失败] /api/scenes 未落盘 {scenes_tmp}")
        else:
            srows = list(csv.DictReader(scenes_tmp.open(encoding="utf-8-sig", newline="")))
            if len(srows) != 2 or srows[0]["video_id"] != "plain_video" \
                    or srows[1]["kind"] != "neg" or srows[1]["seg_id"] != "2":
                ok = False
                print(f"[自测失败] extneg-scenes.csv 内容不符:{srows}")

        # --- key 模式 POST(key1):只替换本 key 的行,其他 key 行原样保留
        post("/api/events?v=key1", {"events": [{"start": 0, "end": 2, "kind": "cut", "note": "自测剔除段"},
                                               {"start": 4, "end": 6, "note": "自测一段"}]},
             expect=200)

        def cons_rows() -> list[list]:
            with (cons / "events.csv").open("r", encoding="utf-8-sig", newline="") as f:
                return [r for r in csv.reader(f) if r and not r[0].startswith("video_id")]

        rows = cons_rows()
        k1_rows = [r for r in rows if len(r) >= 2 and r[1] == "key1"]
        k2_rows = [r for r in rows if len(r) >= 2 and r[1] == "key2"]
        if len(k1_rows) != 2 or k1_rows[0][0] != "vid-a" or k1_rows[0][5] != "cut" \
                or k1_rows[1][3] != "6" or k1_rows[1][5] != "pos":
            ok = False
            print(f"[自测失败] key1 落盘行不符:{rows}")
        if k2_rows != [["vid-b", "key2", "0", "4", "real", "pos", "预填整段正样本(key2 隔离参照)"]]:
            ok = False
            print(f"[自测失败] key2 的预填行应原样保留,实际:{k2_rows}")
        k1v = vid("key1")
        if k1v is None or k1v["status"] != "annotated" or k1v["n_cut"] != 1 or k1v["n_pos"] != 1:
            ok = False
            print(f"[自测失败] key1 保存后 /api/videos 计数不符:{k1v}")
        # 清空 key1:行删除、key2 行仍在(key2 状态不受影响)
        post("/api/events?v=key1", {"events": []}, expect=200)
        rows = cons_rows()
        if any(len(r) >= 2 and r[1] == "key1" for r in rows):
            ok = False
            print(f"[自测失败] key1 清空后汇总文件仍有其行:{rows}")
        k2_rows = [r for r in rows if len(r) >= 2 and r[1] == "key2"]
        if len(k2_rows) != 1 or (vid("key2") or {}).get("status") != "annotated":
            ok = False
            print(f"[自测失败] key2 行应不受 key1 清空影响:{rows}")
        # key2 保存时 video_id 用 mapping 名,同样不碰其他 key
        post("/api/events?v=key2", {"events": [{"start": 0, "end": 2, "note": "自测一段"}]},
             expect=200)
        rows = cons_rows()
        k2_rows = [r for r in rows if len(r) >= 2 and r[1] == "key2"]
        if len(k2_rows) != 1 or k2_rows[0][0] != "vid-b" or k2_rows[0][3] != "2" \
                or k2_rows[0][5] != "pos" \
                or any(len(r) >= 2 and r[1] == "key1" for r in rows):
            ok = False
            print(f"[自测失败] key2 保存结果不符:{rows}")

        print(f"[自测] 夹具帧:cons 3+3 张 / plain 5 张,"
              f"缩略图抽样 {len(state['thumb_idx'])} 张")
    finally:
        if server is not None:
            server.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest(args)
    else:
        if not args.dir:
            raise SystemExit("缺少 --dir 参数(汇总根目录 / 单个汇总源 / 旧式单视频目录,"
                             "见 --help),或用 --selftest 自测")
        run(args)


if __name__ == "__main__":
    main()
