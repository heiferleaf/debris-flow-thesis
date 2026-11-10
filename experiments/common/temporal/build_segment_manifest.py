# -*- coding: utf-8 -*-
"""EXP-006 试点段清单:东川 122 视频 events.csv 区间映射成段 + 外部场景段,统一 train/val 划分。

段口径(与数据清单 B 节一致):
- pos 区间=pos 段, neg 区间=neg 段, cut 区间=cut 段(不训), 未标空隙=隐式 neg 段;
  标注区间 [s,e] 闭区间收编右端点压线帧(2026-09-23 叶海峰拍板修 1 帧尾巴:旧左闭右开判据把
  t==e 的末帧挤进未标空隙,生成 1 帧尾巴隐式 neg 段);
  零头不入清单(2026-09-24 叶海峰判决,替代同日早先的并入方案):未标区域只含 <=1 帧
  (审计台显示粒度以下的零头,典型为末帧落在最后一条标注末端之后的舍入缝隙——审计端点是
  整数秒、帧时间戳带小数)不生成隐式负段直接剔除;人工标注过的段一字不动;
  >=2 帧的未标区域仍按隐式 neg 生成;适用所有数据源,后续新数据集一律如此;
  相邻同类型段合并(仅限同一连续视频内的审计区间:v1/v3/v4/v5);v2 多场景拼接素材不合并(见下);
  每段帧内均匀子采样 <=MAX_PER_SEG 帧(段间等权);
- 划分单元 = group_id(场次视频组),按"是否含正段"分层 85/15,seed=42;
  group_id 东川/v4/v5 = d|{机位}|{日期}(同机位同日期的全部视频文件整组同侧,场次=一次泥石流/同场次);
  外部 v2: 每段一组(段=独立场景;2026-09-24 叶海峰指正:拼接视频的场景边界才是物理独立
  单元,不得按源视频绑组;组 v2|{video_id}|seg{seg_id:02d});
  v3: v3|{video_id} 每视频一组(两个视频=两个场景,2026-09-24 叶海峰指正)。
- 输入(T1 新布局): video1_segment/{events.csv,frames_index.csv,mapping.csv} + video2_segment/extneg-scene-frames.csv
  + video3_segment/(DeFlow,2026-09-17 入线,组 v3|deflow,叶海峰确认整段 pos)
  + video4_segment/(video_dataset_3,2026-09-23 入线,审计切段 pos16/neg9/cut3)
  + video5_segment/(video_dataset_4,2026-09-23 入线,整段 neg;v5_08 excluded=1 剔除,叶海峰拍板);
  v4/v5 组 = d|{地址名}|{日期}(mapping site 列=右下角地址水印,2026-09-24 叶海峰确认逐视频映射
  并按地点名与 v1 严格合并;site_code 列保留 D 编号;同机位同日期跨 v4/v5 同组,同场次不拆;
  D12 两批预置位不同=塔吊下方(v4)/塔吊下游(v5)两个场景)。
  scene_group 列 = 协议二场景组 s|{地址名}(同地址名跨日期整组同侧;extneg/deflow 每段/每视频
  即场景,与 group_id 同值;塔吊上游/塔吊下游与 v1 同名严格合并)。
  东川 mapping.csv 的 excluded=1 视频移 .temp 待定,不入清单;v5 mapping excluded=1(v5_08)跳过。
- 静态 split 只是快速迭代用缺省;正式口径 = 组级分层 k 折(见 train_temporal --protocol cv,
  叶海峰 2026-09-17 定:用全部数据,每组恰考一次)。
- d3 不入试点(EXP-006 卡)。
路径经 adapters 配置(新 data/ 结构 + 旧路径 fallback),不在此裸拼。
输出:adapters.out_manifest_path()
用法(仓库根,venv): python experiments/common/temporal/build_segment_manifest.py
依据: EXP-006 卡(叶海峰 2026-09-15 /goal 签字);审计 events.csv 与 extneg-scenes.csv 的人工判决只增不删;
工具预标注候选不得存在(叶海峰 2026-09-17 口径:不用助手预标注,今后由训练好的模型预标注),
events.csv 出现空 kind 行即报错拒绝构建(2026-09-17 已删 v1_19 的 5 行 dismissed 残留)。
"""
import csv
import random
import re
import sys
from pathlib import Path

import numpy as np

try:  # 包方式（python -m experiments.common.temporal.build_segment_manifest）
    from experiments.common.temporal import adapters
    from experiments.common.temporal.adapters import (
        VIDEO_SOURCES, frames_root_for_source, out_manifest_path, scene_segments_csv)
except ImportError:  # 文件方式（python experiments/common/temporal/build_segment_manifest.py）
    import adapters
    from adapters import VIDEO_SOURCES, frames_root_for_source, out_manifest_path, scene_segments_csv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCENE_CSV = scene_segments_csv()
OUT = out_manifest_path()
MAX_PER_SEG = 64
TRAIN_FRAC = 0.85
SEED = 42

# ---------------------------------------------------------------- 东川场次分组

# 东川 2023/2024 目录名日期段: "occurred on August 2, 2023（Jihguanshan..."
_MONTH_IDX = {m: i for i, m in enumerate((
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"), 1)}
_DATE_RE = re.compile(r"occurred on ([A-Z][a-z]+ \d{1,2}, \d{4})")
# 2025 raw 里日期文件夹名的可识别子串 -> ISO 日期（叶海峰口径 7-19/7-24/7-29 各一场次）
_DATE_ALIAS_2025 = {
    "2025年7月19日": "2025-07-19",
    "2025年7月24日": "2025-07-24",
    "2025年7月29日": "2025-07-29",
}
# 2025 目录名前缀的中文机位（"塔吊上游  Upstream  of the tower crane 5：44" 取 "塔吊上游"）
_SITE_CN_RE = re.compile(r"^([一-鿿]+)")


def _iso_date(txt):
    try:
        mm, dd, yyyy = txt.replace(",", " ").split()
        return f"{int(yyyy):04d}-{_MONTH_IDX[mm]:02d}-{int(dd):02d}"
    except (KeyError, ValueError):
        return txt


def build_dongchuan_date_map() -> dict:
    """枚举 data/raw/video_dataset_1/ 下视频文件,建 stem -> ISO日期 映射(2025 三场日期)。
    frames 目录名与 mp4 stem 精确一致(实测 105/105 无差异);raw 目录不可用时降级为空映射。"""
    date_map = {}
    raw = next((d for d in VIDEO_SOURCES["1"]["raw_dirs"] if d.exists()), None)
    if raw is None:
        print("[warn] 东川原始视频目录不可用,2025 分组退化为单视频组")
        return date_map
    for f in raw.rglob("*"):
        if f.suffix.lower() not in (".mp4", ".mov", ".avi", ".mkv", ".m4v"):
            continue
        date_dir = next((p.name for p in f.parents if "7月" in p.name), None)
        if date_dir is None:
            continue
        iso = next((v for k, v in _DATE_ALIAS_2025.items() if k in date_dir), None)
        if iso:
            date_map[f.stem] = iso
    return date_map


def site_of_dongchuan(video_name: str) -> str:
    """东川视频目录名 -> 机位中文 key（2023/2024 与 2025 同机位归一为同一 key）。"""
    name = video_name.strip()
    m = _SITE_CN_RE.match(name)
    if m:
        return m.group(1)  # 2025: "塔吊上游  Upstream..." -> "塔吊上游"
    if "downstream" in name or "下游" in name:
        return "鸡冠山下游"  # 2023/2024: "Jihguanshan downstream perspective"
    if "upstream" in name or "上游" in name:
        return "鸡冠山上游"
    return "未知"


def group_of_dongchuan(video_name: str, date_map: dict) -> str:
    """东川视频目录名 -> 划分单元 group_id = (日期, 机位)。

    2025(video_dataset_1): 日期来自 stem→日期映射(7-19/7-24/7-29),同机位×同日期多视频同组;
    2023/2024(video_dataset_2): 日期从目录名 "occurred on August 2, 2023..." 解析;
    同机位不同日期 = 不同场次可跨 side(主协议);组内所有段强制同 side——组=防泄题原子。
    """
    name = video_name.strip()
    date = date_map.get(video_name)  # 2025: stem 与 frames 目录名精确一致
    if not date:
        m = _DATE_RE.search(name)    # 2023/2024
        if m:
            date = _iso_date(m.group(1))
    if not date:
        print(f"[warn] 无法判定日期,按单视频组: {video_name}")
        return f"v1|{video_name}"
    return f"d|{site_of_dongchuan(name)}|{date}"


def read_intervals(path):
    if not path.is_file():
        return []
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["start_sec"]), float(r["end_sec"]), r["kind"].strip()))
    return sorted(rows)


def to_spans(intervals, t_end):
    """区间+未标空隙 -> [(start,end,kind,pure_implicit)],相邻同 kind 合并。
    pure_implicit=True 表示整段来自未标空隙(无任何审计区间参与)——是「零头不入清单」
    (2026-09-24 叶海峰判决:≤1 帧的未标区域不生成隐式负段)的过滤对象;
    与审计段相邻同类合并后按已标注处理(人工标注过的段一字不动)。
    重叠截断:后到的区间与已覆盖范围重叠时只保留未覆盖部分(按时间序先到先得)——
    防止重复/冲突区间把 cut 帧漏进相邻 kind 的段(2026-09-17 叶海峰口径:人工判决优先)。"""
    spans, cur = [], 0.0
    for s, e, k in intervals:
        s = max(s, cur)
        if e <= cur:  # 完全被前区间覆盖
            continue
        if s > cur:
            spans.append((cur, s, "neg", True))
        spans.append((s, e, k, False))
        cur = e
    if cur < t_end:
        spans.append((cur, t_end, "neg", True))
    merged = []
    for sp in spans:  # 合并相邻同类(任一部分被标注过,合并段即按已标注处理)
        if merged and merged[-1][2] == sp[2] and abs(merged[-1][1] - sp[0]) < 1e-6:
            merged[-1] = (merged[-1][0], sp[1], sp[2], merged[-1][3] and sp[3])
        else:
            merged.append(sp)
    return merged


def subsample(files, cap=MAX_PER_SEG):
    if len(files) <= cap:
        return files
    idx = np.unique(np.round(np.linspace(0, len(files) - 1, cap)).astype(int))
    return [files[i] for i in idx]


def add_summary_source(root, source, rows, groups, group_of, scene_of=None, skip_excluded=False):
    """汇总层三件套(events.csv + frames_index.csv + mapping.csv,帧平铺 frames/)的数据源入清单。
    video3(DeFlow)/video4/video5 同构;group_of(video_id, mapping行) 给协议一场次组;
    scene_of(video_id, mapping行) 给协议二场景组(缺省=场次组);skip_excluded=True 时跳过
    mapping excluded=1 的视频(v5_08 解码不可修复,叶海峰拍板剔除)。
    区间经 to_spans(空隙=隐式 neg,相邻同 kind 合并——同一连续视频的审计口径,与东川一致;
    v2 多场景拼接素材不合并,走下方独立块)。split 硬编码 train(静态划分仅快速迭代用;
    正式口径 = 组级分层 k 折,train_temporal 按折改写 split)。"""
    ev, idx, mp = {}, {}, {}
    for r in csv.DictReader(open(root / "events.csv", encoding="utf-8-sig", newline="")):
        if not r["kind"].strip():
            raise SystemExit(f"[预标注残留] {root.name}/events.csv 有空 kind 行,先清理再建清单")
        ev.setdefault(r["video_key"], []).append(
            (float(r["start_sec"]), float(r["end_sec"]), r["kind"].strip()))
    for r in csv.DictReader(open(root / "frames_index.csv", encoding="utf-8-sig", newline="")):
        idx.setdefault(r["video_key"], []).append((r["frame_file"], float(r["t_sec"])))
    for r in csv.DictReader(open(root / "mapping.csv", encoding="utf-8-sig", newline="")):
        mp[r["video_key"]] = r
    n = 0
    for key in sorted(ev):
        m = mp.get(key) or {}
        if skip_excluded and int(m.get("excluded") or 0):
            continue
        iv = sorted(ev[key])
        idx_rows = idx.get(key, [])
        if not idx_rows:
            continue
        t_end = idx_rows[-1][1] + 2.0
        has_pos = any(k == "pos" for _, _, k in iv)
        vid = m.get("video_id") or key
        gid = group_of(vid, m)
        scon = (scene_of(vid, m) if scene_of else None) or gid
        claimed: set = set()  # 本视频已归属段的帧(压线帧只归一段,不被相邻段重复收编)
        for si, (s, e, kind, imp) in enumerate(to_spans(iv, t_end), 1):
            if kind == "cut":
                continue
            # 帧归属(2026-09-23 叶海峰拍板修 1 帧尾巴):标注区间 [s,e] 闭区间收编右端点压线帧
            # (旧判 t<e-1e-6 会把 t==e 的末帧挤进后面未标空隙,生成 1 帧尾巴隐式 neg 段);
            # 已归前段的帧不重复归属;cut 段不收帧,其边界帧由相邻标注段按原规则取得
            fs = [fn for fn, t in idx_rows
                  if s - 1e-6 <= t <= e + 1e-6 and fn not in claimed]
            if imp and len(fs) <= 1:
                continue  # 零头不入清单(2026-09-24 叶海峰判决):≤1 帧的未标区域不生成隐式负段
            if not fs:
                continue
            claimed.update(fs)
            sel = subsample(fs)
            rel = [f"frames/{fn}" for fn in sel]
            rows.append({"source": source, "video_id": vid, "group_id": gid,
                         "scene_group": scon, "seg_id": si, "kind": kind, "start_sec": round(s, 2),
                         "end_sec": round(e, 2), "n_total": len(fs), "n_selected": len(sel),
                         "split": "train", "frames": ";".join(rel)})
            n += 1
        groups.append((gid, has_pos, source))
    return n


def main():
    random.seed(SEED)
    rows, groups = [], []  # groups: (group_id, has_pos, source)
    # ---- 东川（T1 新布局）：汇总 events.csv + frames_index.csv + mapping.csv（段帧根 = video1_segment）
    from adapters import video_frames_root
    v1root = video_frames_root("1")
    mapping = {r["video_id"]: r
               for r in csv.DictReader(open(v1root / "mapping.csv", encoding="utf-8-sig", newline=""))}
    ev_by_vid, idx_by_key = {}, {}
    for r in csv.DictReader(open(v1root / "events.csv", encoding="utf-8-sig", newline="")):
        if not r["kind"].strip():
            raise SystemExit(
                f"[预标注残留] events.csv 有空 kind 行({r['video_id']} {r['start_sec']}-{r['end_sec']})——"
                "按叶海峰 2026-09-17 口径,工具预标注候选不得存在,请先清理 events.csv 再建清单")
        ev_by_vid.setdefault(r["video_id"], []).append(
            (float(r["start_sec"]), float(r["end_sec"]), r["kind"].strip()))
    for r in csv.DictReader(open(v1root / "frames_index.csv", encoding="utf-8-sig", newline="")):
        idx_by_key.setdefault(r["video_key"], []).append((r["frame_file"], float(r["t_sec"])))
    n_skipped = 0
    for video_id in sorted(ev_by_vid):
        m = mapping.get(video_id)
        if m is None or int(m.get("excluded") or 0):
            n_skipped += 1
            continue
        iv = sorted(ev_by_vid[video_id])
        idx_rows = idx_by_key.get(m["video_key"], [])
        if not idx_rows:
            continue
        t_end = idx_rows[-1][1] + 2.0
        gid = f'd|{m["site"]}|{m["date"]}' if m.get("date") else f"v1|{video_id}"
        scon = f's|{m["site"]}' if m.get("site") else gid  # 协议二场景组=地址名,严格匹配
        has_pos = any(k == "pos" for _, _, k in iv)
        claimed: set = set()  # 本视频已归属段的帧(压线帧只归一段,不被相邻段重复收编)
        for si, (s, e, kind, imp) in enumerate(to_spans(iv, t_end), 1):
            if kind == "cut":
                continue
            # 帧归属(2026-09-23 叶海峰拍板修 1 帧尾巴):标注区间 [s,e] 闭区间收编右端点压线帧
            # (旧判 t<e-1e-6 会把 t==e 的末帧挤进后面未标空隙,生成 1 帧尾巴隐式 neg 段);
            # 已归前段的帧不重复归属;cut 段不收帧,其边界帧由相邻标注段按原规则取得
            fs = [fn for fn, t in idx_rows
                  if s - 1e-6 <= t <= e + 1e-6 and fn not in claimed]
            if imp and len(fs) <= 1:
                continue  # 零头不入清单(2026-09-24 叶海峰判决):≤1 帧的未标区域不生成隐式负段
            if not fs:
                continue
            claimed.update(fs)
            sel = subsample(fs)
            rel = [f"frames/{fn}" for fn in sel]
            rows.append({"source": "dongchuan", "video_id": video_id, "group_id": gid,
                         "scene_group": scon, "seg_id": si, "kind": kind, "start_sec": round(s, 2),
                         "end_sec": round(e, 2), "n_total": len(fs), "n_selected": len(sel),
                         "split": "", "frames": ";".join(rel)})
        groups.append((gid, has_pos, "dongchuan"))
    print(f"[东川] 跳过待定视频 {n_skipped} 个,参训视频 {sum(1 for g in groups)} 个")

    # ---- 汇总层数据源:video3(DeFlow)/video4/video5 三件套同构,统一走 add_summary_source
    v3root = video_frames_root("3")
    if v3root.is_dir():
        n_v3 = add_summary_source(
            v3root, "deflow", rows, groups,
            group_of=lambda vid, m: f"v3|{vid}")
        print(f"[DeFlow] video3_segment 入清单 {n_v3} 段(每视频一组,两个视频=两个场景)")

    # video4 = video_dataset_3(2026-08-18 爆发后,小型泥石流到达塔吊断面;审计切段 pos16/neg9/cut3)
    # video5 = video_dataset_4(2026-08-18 爆发前常规水流,整段 neg;v5_08 excluded=1 剔除)
    # 组 = d|{机位}|{日期}(mapping site/date 列;v4/v5 同机位同日期同组,同场次不拆)
    for vno, skip_ex in (("4", False), ("5", True)):
        root = video_frames_root(vno)
        if not root.is_dir():
            print(f"[warn] video{vno}_segment 目录不存在,跳过")
            continue
        n_vx = add_summary_source(
            root, f"video{vno}", rows, groups,
            group_of=lambda vid, m, _v=vno: (
                f'd|{m["site"]}|{m["date"]}' if m.get("site") and m.get("date") else f"v{_v}|{vid}"),
            scene_of=lambda vid, m: f's|{m["site"]}' if m.get("site") else None,
            skip_excluded=skip_ex)
        print(f"[video{vno}] video{vno}_segment 入清单 {n_vx} 段"
              + ("(v5_08 excluded 跳过)" if vno == "5" else ""))

    # 外部场景段(v2,2026-09-23 重切为 86 段口径):段帧根 = final/video_windows/video2_segment。
    # 口径(叶海峰定):v2 是多场景拼接素材,段 = 划分原子,**相邻同标签段不合并**(CSV 每行即一段,
    # 与东川 to_spans 的合并逻辑无关);组 = 每个源视频(video_id=video_key)一组,同源视频窗口必须同侧。
    # 段目录布局:2026-09-23 重切后为嵌套 <video_id>/segNN_kind/,dir 列写的是扁平名 <video_id>__segNN_kind
    # ——两者都探测,以磁盘上实际存在 frames_index.csv 的为准。
    if SCENE_CSV.is_file():
        v2root = frames_root_for_source("extneg")
        with open(SCENE_CSV, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                kind = r["kind"].strip()
                if kind == "cut":
                    continue
                vid = r["video_id"]
                sid = int(r["seg_id"])
                cands = []
                d = (r.get("dir") or "").replace("\\", "/")
                if "video2_segment/" in d:
                    cands.append(d.split("video2_segment/", 1)[1])
                cands += [f"{vid}/seg{sid:02d}_{kind}", f"{vid}__seg{sid:02d}_{kind}"]
                seg_rel = next((c for c in cands
                                if (v2root / c / "frames_index.csv").is_file()), None)
                if seg_rel is None:
                    print(f"[缺索引] {vid} seg{sid:02d}_{kind}(尝试过: {cands})")
                    continue
                fs = [rr["frame_file"] for rr in csv.DictReader(
                    open(v2root / seg_rel / "frames_index.csv", encoding="utf-8-sig"))]
                if not fs:
                    continue
                sel = subsample(fs)
                rel = [f"{seg_rel}/{fn}" for fn in sel]
                rows.append({"source": "extneg", "video_id": vid,
                             "group_id": f"v2|{vid}|seg{sid:02d}",
                             "scene_group": f"v2|{vid}|seg{sid:02d}",
                             "seg_id": sid, "kind": kind,
                             "start_sec": float(r["start_sec"]), "end_sec": float(r["end_sec"]),
                             "n_total": len(fs), "n_selected": len(sel),
                             "split": "train", "frames": ";".join(rel)})
                groups.append((f"v2|{vid}|seg{sid:02d}", kind == "pos", "extneg"))
    # 组级 85/15 分层(外部已在 train,不参与抽样): 分层单位 = group_id
    dch = [g for g in groups if g[2] == "dongchuan"]
    pos_g = sorted(g for g, hp, _ in dch if hp)
    neg_g = sorted(g for g, hp, _ in dch if not hp)
    rng = random.Random(SEED)
    val = set(rng.sample(pos_g, max(1, round(len(pos_g) * (1 - TRAIN_FRAC)))))
    val |= set(rng.sample(neg_g, max(1, round(len(neg_g) * (1 - TRAIN_FRAC)))))
    split_of = {g: ("val" if g in val else "train") for g, _, _ in dch}
    for r in rows:
        if r["source"] == "dongchuan":
            r["split"] = split_of[r["group_id"]]

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "video_id", "group_id", "scene_group",
                                          "seg_id", "kind", "start_sec", "end_sec", "n_total",
                                          "n_selected", "split", "frames"])
        w.writeheader()
        w.writerows(rows)

    n_tr_p = sum(1 for r in rows if r["split"] == "train" and r["kind"] == "pos")
    n_tr_n = sum(1 for r in rows if r["split"] == "train" and r["kind"] == "neg")
    n_va_p = sum(1 for r in rows if r["split"] == "val" and r["kind"] == "pos")
    n_va_n = sum(1 for r in rows if r["split"] == "val" and r["kind"] == "neg")
    f_tr = sum(r["n_selected"] for r in rows if r["split"] == "train")
    f_va = sum(r["n_selected"] for r in rows if r["split"] == "val")
    print(f"段总数 {len(rows)} | 组 {len(groups)} | train {n_tr_p}正/{n_tr_n}负 段({f_tr}帧) "
          f"| val {n_va_p}正/{n_va_n}负 段({f_va}帧)")
    print(f"val 组: {sorted(val)}")
    print(f"清单 -> {OUT}")


if __name__ == "__main__":
    main()