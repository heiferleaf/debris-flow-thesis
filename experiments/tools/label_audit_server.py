"""本地标签审计服务器:浏览器里逐张看图、一键判决(正负由人的判决定)。

实现(http.server + json + csv + cv2 仅缩略图缩放用),不联网,只监听 127.0.0.1。
前端 HTML/JS/CSS 直接内嵌在本文件的字符串里,
图片由服务端按请求原样回传,避免 Windows 中文路径下的编码问题;
同簇缩略图走 /thumb 服务端磁盘缓存(<root>/.audit-thumbs/,宽 480px JPEG),
原图响应带 Cache-Control(原图存档永不改动的仓库约定保证缓存安全)——
三者合作用来解决全尺寸原图刷缩略图导致的页面卡顿(2026-09-14 叶海峰报)。

页面交互:
  - 只显示原图,不画 YOLO 框、不显示 label 文本——源数据里的检测标签
    与本课题的「识别已出现」分类任务无关(2026-09-12 叶海峰定),页面不展示
  - 四个判决:通过(1) / 剔除(2) / 存疑(3) / 改负(4),判决后自动跳下一簇
    (传 --clusters 时;未启用簇功能则与旧版一致跳下一张);
    ← / → 或 A / D 前后浏览;已判决的图可改判(改判停在原地便于微调);
    可选备注输入框(默认收起)
  - 传 --clusters 时(pHash 簇 CSV):主图下方显示「同簇图片」横条,同簇成员
    缩略图点击可跳转,已审的带对勾;勾选「整簇同判」后,一次判决会应用到
    同簇所有未审成员(已审的不覆盖),浮窗提示应用/跳过数量
  - 每次判决立即 POST 到服务端写入 CSV,刷新不丢进度;
    重新启动自动跳到第一张未判决的图(勾选「从头重审」则从第 0 张开始)

用法(仓库根目录):
    .venv\\Scripts\\python.exe experiments\\tools\\label_audit_server.py ^
        --root data\\raw\\dataset1 [--port 8000] [--mode all|empty|suspect] ^
        [--suspect-file <txt,每行一个相对路径>] [--classes <classes.txt路径>] ^
        [--clusters data\\final\\images\\clusters\\phash-clusters.csv] [--out <csv路径>]

CSV 列:file,label_info,verdict,note,timestamp(按 file 去重,重复判决覆盖)。
默认输出 data/final/images/labels/audit-<root目录名>.csv(2026-09-16 data 结构重构:
判决 CSV=图像最终标签,归 final/images/labels/;旧 data/processed/label-review/ 已并入)。
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png"}
VERDICTS = {"pass": "通过", "drop": "剔除", "suspect": "存疑", "neg": "改负"}
CSV_FIELDS = ["file", "label_info", "verdict", "note", "timestamp"]
MAX_CLUSTER_THUMBS = 60  # 单簇缩略图路径条数上限(超出部分不进 JSON,防响应过大撑崩浏览器)
THUMB_WIDTH = 480        # 同簇缩略图宽度(px),JPEG 质量 80——只为扫一眼相似度,看细节点开原图
THUMB_CACHE_DIR = ".audit-thumbs"   # 缩略图磁盘缓存目录(建在 root 下,gitignore 由 data 规则覆盖)


def make_thumb(root: Path, full: Path, rel: str) -> bytes:
    """同簇缩略图字节:读磁盘缓存,未命中则缩到宽 THUMB_WIDTH 后落缓存再返回。
    缓存键 = md5(相对路径:文件大小),文件重排/替换自动失效;
    np.fromfile+imdecode 兼容 Windows 中文路径;任何失败回退原图字节——
    缩略图只是加速,不能挡住审计。原图存档永不改动的约定保证缓存内容安全。"""
    try:
        cache_dir = root / THUMB_CACHE_DIR
        size = full.stat().st_size
        cache_path = cache_dir / (hashlib.md5(f"{rel}:{size}".encode("utf-8")).hexdigest() + ".jpg")
        if cache_path.is_file() and cache_path.stat().st_size > 0:
            return cache_path.read_bytes()
        img = cv2.imdecode(np.fromfile(str(full), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return full.read_bytes()
        h, w = img.shape[:2]
        if w > THUMB_WIDTH:
            img = cv2.resize(img, (THUMB_WIDTH, max(1, round(h * THUMB_WIDTH / w))),
                             interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return full.read_bytes()
        cache_dir.mkdir(parents=True, exist_ok=True)
        data = enc.tobytes()
        tmp = cache_path.with_suffix(f".tmp{threading.get_ident()}")
        tmp.write_bytes(data)
        tmp.replace(cache_path)
        return data
    except OSError:
        try:
            return full.read_bytes()
        except OSError:
            return b""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地标签审计服务器(纯标准库,浏览器逐张审 YOLO 标签)")
    parser.add_argument("--root", required=True, help="数据集根目录,如 data/raw/dataset1")
    parser.add_argument("--port", type=int, default=8000, help="起始端口,被占用自动 +1(默认 8000)")
    parser.add_argument("--mode", choices=["all", "empty", "suspect", "rest"], default="all",
                        help="all=全部图;empty=只审空标签图;suspect=只审 --suspect-file 清单里的图;"
                             "rest=剩余所有(排除空标签与清单内的图,与另三类正好互补)")
    parser.add_argument("--suspect-file", default="", help="存疑清单 txt,每行一个相对 root 的图片路径")
    parser.add_argument("--classes", default="", help="classes.txt 路径(每行一个类别名);不传则 UI 显示类别 id")
    parser.add_argument("--clusters", default="",
                        help="pHash 簇 CSV 路径(如 data/final/images/clusters/phash-clusters.csv);"
                             "不传则不显示同簇图片、无整簇同判,行为与旧版完全一致")
    parser.add_argument("--out", default="", help="判决 CSV 输出路径(默认 data/final/images/labels/audit-<root目录名>.csv)")
    parser.add_argument("--selftest", action="store_true",
                        help="自测模式:启动→自请求→写测试 CSV→删→打印 PASS 后退出,不开浏览器")
    return parser.parse_args()


# ---------------------------------------------------------------- 数据扫描

def load_classes(path: str) -> list[str]:
    """读 classes.txt,每行一个类别名;读不到返回空列表(UI 回退显示 id)。"""
    p = Path(path) if path else None
    if not p or not p.is_file():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]


def load_suspect(path: str, root: Path) -> set[str]:
    """读存疑清单,每行一个相对路径,统一成 posix 相对路径字符串。"""
    p = Path(path) if path else None
    if not p or not p.is_file():
        return set()
    out = set()
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        rel = ln.strip().replace("\\", "/")
        if not rel or rel.startswith("#"):
            continue
        try:
            out.add((root / rel).resolve().relative_to(root.resolve()).as_posix())
        except ValueError:
            print(f"[跳过] 清单路径不在 root 内:{rel}")
    return out


def load_clusters(clusters_path: str, root: Path) -> tuple[dict[str, int], dict[int, list[str]], dict[int, int]]:
    """读 pHash 簇 CSV,归一化成相对当前 root 的映射。

    CSV 列:cluster_id,source,official_group,relpath。source 标明图片属于哪个
    数据集(如 dataset1),relpath 是相对该数据集 root 的路径(Windows 反斜杠,
    如 images\\train\\xxx.jpg 或 scene2\\Annotated\\images\\xxx.png);个别行
    relpath 可能自带 datasetN/ 前缀,两种写法都兼容。
    只保留属于当前 root 的成员,跨数据集的成员计入 cid2other(UI 提示
    「另有 N 张在别的数据集」)。
    返回 (file→cluster_id, cluster_id→本 root 成员列表, cluster_id→其他数据集成员数)。
    """
    p = Path(clusters_path) if clusters_path else None
    if not p or not p.is_file():
        return {}, {}, {}
    file2cid: dict[str, int] = {}
    cid2files: dict[int, list[str]] = {}
    cid2other: dict[int, int] = {}
    root_name = root.name.lower()
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw_rel = (row.get("relpath") or row.get("file") or row.get("path") or "").strip()
            raw_cid = (row.get("cluster_id") or "").strip()
            source = (row.get("source") or "").strip()
            if not raw_rel or not raw_cid:
                continue
            try:
                cid = int(raw_cid)
            except ValueError:
                continue
            rel = raw_rel.replace("\\", "/")
            # relpath 自带 datasetN/ 前缀时以前缀为准,否则按 source 列归属
            m = re.match(r"^(dataset\d+)/+(?P<rest>.+)$", rel, re.IGNORECASE)
            if m:
                owner, rel = m.group(1).lower(), m.group("rest")
            elif source:
                owner = source.lower()
            else:
                owner = root_name  # 无 source 列且无前缀:默认视为当前 root
            if owner != root_name:
                cid2other[cid] = cid2other.get(cid, 0) + 1
                continue
            if rel in file2cid:
                continue  # 同一文件重复行,只取第一条
            file2cid[rel] = cid
            cid2files.setdefault(cid, []).append(rel)
    return file2cid, cid2files, cid2other


def _label_path_for(root: Path, img_rel: str) -> Path:
    """图片相对路径 -> 标签路径:相对路径里最后一段 images 换成 labels,后缀 .txt。

    兼容两种布局:dataset1/2 的 images/train/x.jpg -> labels/train/x.txt;
    dataset3 的 scene3/Annotated/images/x.png -> scene3/Annotated/labels/x.png.txt 的 .txt。
    """
    parts = list(Path(img_rel).parts)
    idx = max(i for i, p in enumerate(parts) if p.lower() == "images")
    return root.joinpath(*parts[:idx], "labels", *parts[idx + 1:]).with_suffix(".txt")


def scan_images(root: Path) -> list[str]:
    """扫描 root 下的图片,返回相对 root 的 posix 路径列表(按路径排序)。

    布局兼容:
      - <root>/images 存在(dataset1/2):有 split 子目录就遍历,没有就直接扫;
      - 否则扫描 root 下所有名为 images 的目录(dataset3:sceneN/Annotated/images)。
    """
    images_dir = root / "images"
    if images_dir.is_dir():
        subdirs = [d for d in images_dir.iterdir() if d.is_dir()]
        bases = subdirs if subdirs else [images_dir]
    else:
        bases = sorted(d for d in root.rglob("images") if d.is_dir())
        if not bases:
            raise SystemExit(f"找不到图片目录:{images_dir}(也没有嵌套的 images 目录)")
    files = []
    for base in bases:
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                files.append(p.resolve().relative_to(root.resolve()).as_posix())
    return sorted(files)


def read_label(root: Path, img_rel: str) -> tuple[list[list[float]], str, bool, bool]:
    """读一张图对应的 label txt。

    返回 (框列表[[cls,cx,cy,w,h],...], 原始文本, label文件缺失?, 是否空标签)。
    找不到 label 文件视为空标签,并单独注明「label 文件缺失」。
    """
    label_path = _label_path_for(root, img_rel)
    if not label_path.is_file():
        return [], "", True, True
    text = label_path.read_text(encoding="utf-8", errors="replace")
    boxes = []
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            boxes.append([int(float(parts[0]))] + [float(x) for x in parts[1:5]])
        except ValueError:
            continue
    empty = not boxes  # 0 字节或只有无效行都算空标签
    return boxes, text, False, empty


def build_items(args: argparse.Namespace, root: Path,
                clusters: tuple[dict, dict, dict] | None = None) -> list[dict]:
    """按 --mode 过滤出待审计的图片,每张生成一条 item dict。

    传了 clusters(load_clusters 的返回值)时,每条 item 额外附带 cluster_id /
    cluster_size(本 root 内成员数)/ cluster_other(其他数据集成员数)/
    cluster_files(本 root 内成员路径列表,最多 MAX_CLUSTER_THUMBS 条);无簇信息则不带这些键。
    """
    all_files = scan_images(root)
    if args.mode == "suspect":
        wanted = load_suspect(args.suspect_file, root)
        if not wanted:
            raise SystemExit(f"--mode suspect 但清单为空或不存在:{args.suspect_file}")
        all_files = [f for f in all_files if f in wanted]
    elif args.mode == "rest":
        wanted = load_suspect(args.suspect_file, root)  # 清单可为空:只排除空标签
        print(f"[rest] 排除存疑清单 {len(wanted)} 张与空标签,审剩余全部")
    file2cid, cid2files, cid2other = clusters if clusters else ({}, {}, {})
    items = []
    for rel in all_files:
        boxes, text, missing, empty = read_label(root, rel)
        if args.mode == "empty" and not empty:
            continue
        if args.mode == "rest" and (empty or rel in wanted):
            continue
        it = {
            "file": rel,
            "label_info": "label 文件缺失" if missing else (f"{len(boxes)}框" if boxes else "空标签"),
        }
        cid = file2cid.get(rel)
        if cid is not None:
            members = cid2files.get(cid, [])
            it["cluster_id"] = cid
            it["cluster_size"] = len(members)
            it["cluster_other"] = cid2other.get(cid, 0)
            # 超大簇只带前 MAX_CLUSTER_THUMBS 条路径:8k 帧数据集的单簇可达数千成员,
            # 全量下发会让 /api/list 涨到 GB 级,浏览器渲染进程直接崩溃(STATUS_BREAKPOINT)
            it["cluster_files"] = members[:MAX_CLUSTER_THUMBS]
        items.append(it)
    return items


def load_verdicts(csv_path: Path) -> dict[str, dict]:
    """从已有 CSV 恢复判决进度,按 file 去重(后一行覆盖前一行)。"""
    verdicts: dict[str, dict] = {}
    if not csv_path.is_file():
        return verdicts
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            file = (row.get("file") or "").strip()
            if file:
                verdicts[file] = {k: row.get(k, "") for k in CSV_FIELDS}
    return verdicts


def save_verdicts(csv_path: Path, verdicts: dict[str, dict]) -> None:
    """整份重写 CSV(按 file 排序去重),防止重复行。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for file in sorted(verdicts):
            writer.writerow({k: verdicts[file].get(k, "") for k in CSV_FIELDS})


# ---------------------------------------------------------------- 前端页面

PAGE_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>标签审计工作台</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 12px 16px; font-family: "Microsoft YaHei", system-ui, sans-serif;
         background: #f4f4f2; color: #222; }
  header { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; margin-bottom: 10px; }
  header h1 { font-size: 17px; margin: 0; }
  .csv-path { font-size: 12px; color: #666; word-break: break-all; }
  #progress { flex: 1 1 200px; min-width: 160px; height: 10px; background: #ddd; border-radius: 5px; overflow: hidden; }
  #progress-bar { height: 100%; width: 0; background: #2f7d4f; transition: width .2s; }
  #counter { font-size: 13px; white-space: nowrap; }
  .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 10px; font-size: 13px; }
  .toolbar label { cursor: pointer; }
  main { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
  #viewer { flex: 1 1 560px; min-width: 300px; }
  #canvas-wrap { position: relative; background: #111; }
  canvas { max-width: 100%; height: auto; display: block; }
  #panel { flex: 0 1 340px; min-width: 280px; }
  #filename { font-size: 14px; font-weight: 600; word-break: break-all; margin-bottom: 6px; }
  .badge { display: inline-block; font-size: 12px; padding: 1px 8px; border-radius: 9px; margin-right: 6px; }
  .badge.empty { background: #ffe9c2; color: #8a5300; }
  .badge.missing { background: #ffd6d6; color: #a11; }
  .badge.ok { background: #d8f0e0; color: #1c6b3c; }
  .verdicts { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .verdicts button { font-size: 15px; padding: 10px 0; border: 1px solid #bbb; border-radius: 8px;
                     background: #fff; cursor: pointer; }
  .verdicts button small { display: block; font-size: 11px; color: #888; }
  .verdicts button:hover { border-color: #666; }
  .verdicts button.cur-pass { background: #2f7d4f; color: #fff; border-color: #2f7d4f; }
  .verdicts button.cur-drop { background: #b23; color: #fff; border-color: #b23; }
  .verdicts button.cur-suspect { background: #d88a1f; color: #fff; border-color: #d88a1f; }
  .verdicts button.cur-neg { background: #4468b0; color: #fff; border-color: #4468b0; }
  #note-toggle { margin-top: 10px; font-size: 13px; color: #555; cursor: pointer; background: none;
                 border: none; padding: 0; text-decoration: underline; }
  #note { width: 100%; margin-top: 6px; padding: 6px; font-size: 13px; border: 1px solid #ccc;
          border-radius: 6px; resize: vertical; }
  #note[hidden] { display: none; }
  .nav { margin-top: 12px; display: flex; gap: 8px; }
  .nav button { flex: 1; padding: 8px 0; font-size: 14px; }
  #img-error { color: #b23; font-size: 13px; margin-top: 8px; }
  kbd { background: #eee; border: 1px solid #ccc; border-radius: 3px; padding: 0 4px; font-size: 11px; }
  .hint { font-size: 12px; color: #777; margin-top: 10px; line-height: 1.7; }
  #batch-wrap { display: block; margin-bottom: 8px; font-size: 13px; }
  #batch-wrap[hidden] { display: none; }
  #cluster-bar { margin-top: 10px; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 8px; }
  .cluster-head { font-size: 12px; color: #555; margin-bottom: 6px; }
  .cluster-thumbs { display: flex; flex-wrap: wrap; gap: 6px; }
  .thumb { position: relative; width: 86px; height: 64px; border-radius: 4px; overflow: hidden;
           cursor: pointer; background: #222; border: 2px solid transparent; flex: 0 0 auto; }
  .thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .thumb.cur { border-color: #2f7d4f; }
  .thumb .check { position: absolute; top: 2px; right: 2px; background: #2f7d4f; color: #fff;
                  font-size: 11px; line-height: 1; padding: 2px 3px; border-radius: 3px; }
  #toast { position: fixed; left: 50%; bottom: 28px; transform: translateX(-50%);
           background: #333; color: #fff; padding: 8px 16px; border-radius: 6px; font-size: 13px; z-index: 10; }
  #toast[hidden] { display: none; }
</style>
</head>
<body>
<header>
  <h1>标签审计工作台</h1>
  <div id="counter">加载中…</div>
  <div id="progress"><div id="progress-bar"></div></div>
</header>
<div class="toolbar">
  <span id="mode-info" class="csv-path"></span>
  <span class="csv-path" id="csv-path"></span>
  <label><input type="checkbox" id="from-start"> 从头重审</label>
</div>
<main>
  <div id="viewer">
    <div id="canvas-wrap"><canvas id="canvas" width="640" height="360"></canvas></div>
    <div id="img-error" hidden></div>
    <div id="cluster-bar" hidden></div>
  </div>
  <div id="panel">
    <div id="filename"></div>
    <div id="badges"></div>
    <label id="batch-wrap" hidden><input type="checkbox" id="batch-mode"> 整簇同判(同簇未审的一起判)</label>
    <div class="verdicts">
      <button data-v="pass">通过<small><kbd>1</kbd></small></button>
      <button data-v="drop">剔除<small><kbd>2</kbd></small></button>
      <button data-v="suspect">存疑<small><kbd>3</kbd></small></button>
      <button data-v="neg">改负<small><kbd>4</kbd></small></button>
    </div>
    <button id="note-toggle">+ 备注</button>
    <textarea id="note" rows="2" hidden placeholder="备注(可选)…"></textarea>
    <div class="nav">
      <button id="prev">← 上一张</button>
      <button id="next">下一张 →</button>
    </div>
    <div class="hint">
      <kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd>/<kbd>4</kbd> 判决后自动跳下一簇
      (勾选整簇同判则一次判完一簇);<kbd>←</kbd><kbd>→</kbd> 或 <kbd>A</kbd><kbd>D</kbd> 前后浏览;
      点同簇缩略图可微调单张,改判后停在原地;判决即时保存。
    </div>
  </div>
</main>
<div id="toast" hidden></div>
<script>
"use strict";
var state = { items: [], idx: 0, imgOk: false,
              verdicts: {}, hasClusters: false };
var noteShown = false;
var toastTimer = null;

function $(id) { return document.getElementById(id); }

function curItem() { return state.items[state.idx] || null; }

function load() {
  fetch("/api/list").then(function (r) { return r.json(); }).then(function (data) {
    state.items = data.items;
    state.verdicts = data.verdicts || {};
    state.hasClusters = !!data.has_clusters;
    $("batch-wrap").hidden = !state.hasClusters;
    $("mode-info").textContent = "模式:" + data.mode + "  待审 " + data.items.length + " 张(root: " + data.root + ")";
    $("csv-path").textContent = "CSV: " + data.csv_path;
    var start = 0;
    if (!$("from-start").checked) {
      // 自动跳到第一张未判决的图;全审完则停在最后一张
      start = data.items.length - 1;
      for (var i = 0; i < data.items.length; i++) {
        if (!data.items[i].verdict) { start = i; break; }
      }
    }
    state.idx = start;
    render();
  });
}

function showItem() {
  var it = curItem();
  if (!it) { return; }
  $("filename").textContent = "[" + (state.idx + 1) + "/" + state.items.length + "] " + it.file;
  var badges = $("badges");
  badges.innerHTML = "";
  var note = $("note");
  if (note.value !== (it.note || "")) { note.value = it.note || ""; }
  var v = it.verdict || "";
  var btns = document.querySelectorAll(".verdicts button");
  for (var i = 0; i < btns.length; i++) {
    var b = btns[i];
    b.className = (b.getAttribute("data-v") === v) ? "cur-" + v : "";
  }
  loadImage(it.file);
  renderClusterBar(it);
}

function findIndexByFile(file) {
  for (var i = 0; i < state.items.length; i++) {
    if (state.items[i].file === file) { return i; }
  }
  return -1;
}

function renderClusterBar(it) {
  var bar = $("cluster-bar");
  bar.innerHTML = "";
  // 无簇信息或簇内只有本图时收起横条
  if (!it || typeof it.cluster_id !== "number" || it.cluster_size < 2) { bar.hidden = true; return; }
  bar.hidden = false;
  var head = document.createElement("div");
  head.className = "cluster-head";
  var t = "同簇图片(" + it.cluster_size + ")";
  if (it.cluster_other > 0) { t += " · 另有 " + it.cluster_other + " 张在其他数据集"; }
  head.textContent = t;
  bar.appendChild(head);
  var row = document.createElement("div");
  row.className = "cluster-thumbs";
  for (var i = 0; i < it.cluster_files.length; i++) {
    (function (f) {
      var cell = document.createElement("div");
      cell.className = "thumb" + (f === it.file ? " cur" : "");
      var im = document.createElement("img");
      im.loading = "lazy";
      im.src = "/thumb?file=" + encodeURIComponent(f);   // 服务端 480px 缓存小图,原图只留给主视图
      im.alt = f;
      im.title = f;
      cell.appendChild(im);
      if (state.verdicts[f]) {          // 已审的成员缩略图加对勾
        var ck = document.createElement("span");
        ck.className = "check";
        ck.textContent = "✓";
        cell.appendChild(ck);
      }
      cell.addEventListener("click", function () { jumpToFile(f); });
      row.appendChild(cell);
    })(it.cluster_files[i]);
  }
  bar.appendChild(row);
}

function jumpToFile(file) {
  var i = findIndexByFile(file);
  if (i >= 0) { state.idx = i; render(); }
  else { toast("该图不在当前审计列表(可能被模式过滤或图片缺失)"); }
}

function toast(msg) {
  var el = $("toast");
  el.textContent = msg;
  el.hidden = false;
  if (toastTimer) { clearTimeout(toastTimer); }
  toastTimer = setTimeout(function () { el.hidden = true; }, 2600);
}

function loadImage(rel) {
  var cv = $("canvas"), err = $("img-error");
  err.hidden = true;
  state.imgOk = false;
  var img = new Image();
  img.onload = function () {
    cv.width = img.naturalWidth; cv.height = img.naturalHeight;
    var ctx = cv.getContext("2d");
    ctx.drawImage(img, 0, 0);
    state.imgOk = true;
  };
  img.onerror = function () {
    var ctx = cv.getContext("2d");
    ctx.fillStyle = "#222"; ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = "#f66"; ctx.font = "16px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("图片读取失败(已计入跳过)", cv.width / 2, cv.height / 2);
    err.textContent = "图片读取失败:" + rel;
    err.hidden = false;
  };
  img.src = "/img?file=" + encodeURIComponent(rel);
}

function saveVerdict(verdict) {
  var it = curItem();
  if (!it) { return; }
  var wasReviewed = !!it.verdict || !!state.verdicts[it.file];  // 改判还是新判
  it.verdict = verdict;
  it.note = $("note").value.trim();
  var body = { file: it.file, verdict: verdict, note: it.note };
  if ($("batch-mode").checked) { body.cluster_apply = true; }   // 整簇同判
  fetch("/api/verdict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  }).then(function (r) { return r.json(); }).then(function (resp) {
    if (resp.applied > 1) {
      toast("整簇同判:已应用 " + resp.applied + " 张" +
            (resp.skipped ? "(跳过已审 " + resp.skipped + " 张)" : ""));
      // 同步本簇成员的本地判决状态(后端只应用了未审的,已审保持原判)
      state.verdicts[it.file] = verdict;
      var files = it.cluster_files || [];
      for (var i = 0; i < files.length; i++) {
        var f = files[i];
        var j = findIndexByFile(f);
        var had = (j >= 0 && state.items[j].verdict) || state.verdicts[f];
        if (!had) {
          state.verdicts[f] = verdict;
          if (j >= 0) { state.items[j].verdict = verdict; }
        }
      }
    } else {
      state.verdicts[it.file] = verdict;
    }
    updateProgress();
    showItem();          // 刷新当前图的按钮高亮(改判场景)
    // 主流程:新判(或整簇补判了新成员)后自动跳下一簇;
    // 单纯改判停在原地,便于继续微调;无簇模式保持旧的「跳下一张」
    if (!state.hasClusters) { go(1); }
    else if (resp.applied > 1 || !wasReviewed) { nextCluster(); }
    else { toast("已改判,停在当前图;点「下一簇」继续"); }
  });
}

function go(delta) {
  if (!state.items.length) { return; }
  var next = state.idx + delta;
  if (next < 0) { next = 0; }
  if (next >= state.items.length) { next = state.items.length - 1; }
  if (next !== state.idx) { state.idx = next; render(); }
}

function clusterKey(it) {
  // 簇唯一键:有簇号的按簇号;没进簇的每张自成一组
  return (typeof it.cluster_id === "number") ? "c" + it.cluster_id : "f:" + it.file;
}

function nextClusterIndex(items, fromIdx, curKey) {
  // 从 fromIdx 起环形找第一个未判决且与 curKey 不同簇的 item;没有返回 -1。
  // 同簇未审的也跳过(绕一圈回来再单独审)
  var n = items.length;
  for (var s = 1; s <= n; s++) {
    var i = (fromIdx + s) % n;
    var it = items[i];
    if (it.verdict) { continue; }
    if (clusterKey(it) === curKey) { continue; }
    return i;
  }
  return -1;
}

function nextUnreviewedIndex(items, fromIdx) {
  // 环形找第一个未判决 item(不限簇);没有返回 -1
  var n = items.length;
  for (var s = 1; s <= n; s++) {
    var i = (fromIdx + s) % n;
    if (!items[i].verdict) { return i; }
  }
  return -1;
}

function countPendingClusters() {
  // 剩余未审簇数(未审满的簇各计 1;无簇的单张也算一组)
  var seen = {}, n = 0;
  for (var i = 0; i < state.items.length; i++) {
    var it = state.items[i];
    if (it.verdict) { continue; }
    var k = clusterKey(it);
    if (!seen[k]) { seen[k] = true; n++; }
  }
  return n;
}

function nextCluster() {
  // 主导航:跳到下一个未审簇;无簇模式退化为旧的「下一张」
  if (!state.items.length) { return; }
  if (!state.hasClusters) { go(1); return; }
  var it = curItem();
  var t = nextClusterIndex(state.items, state.idx, it ? clusterKey(it) : "");
  if (t < 0) { t = nextUnreviewedIndex(state.items, state.idx); }  // 只剩当前簇未审时回退
  if (t < 0) { toast("全部审完!"); updateProgress(); return; }
  state.idx = t;
  render();
}

function updateProgress() {
  var done = 0;
  for (var i = 0; i < state.items.length; i++) { if (state.items[i].verdict) { done++; } }
  var txt = "已审 " + done + " / " + state.items.length;
  if (state.hasClusters) {
    var pending = countPendingClusters();
    txt += " · 未审簇 " + pending;
    $("next").textContent = "下一簇(" + pending + ") →";
  } else {
    $("next").textContent = "下一张 →";
  }
  $("counter").textContent = txt;
  $("progress-bar").style.width = (state.items.length ? done * 100 / state.items.length : 0) + "%";
}

function render() { updateProgress(); showItem(); }

document.addEventListener("keydown", function (e) {
  if (e.target && (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT")) { return; }
  var k = e.key;
  if (k === "1") { saveVerdict("pass"); }
  else if (k === "2") { saveVerdict("drop"); }
  else if (k === "3") { saveVerdict("suspect"); }
  else if (k === "4") { saveVerdict("neg"); }
  else if (k === "ArrowLeft" || k === "a" || k === "A") { go(-1); }
  else if (k === "ArrowRight" || k === "d" || k === "D") { nextCluster(); }
});

var vbtns = document.querySelectorAll(".verdicts button");
for (var i = 0; i < vbtns.length; i++) {
  vbtns[i].addEventListener("click", function () { saveVerdict(this.getAttribute("data-v")); });
}
$("prev").addEventListener("click", function () { go(-1); });
$("next").addEventListener("click", function () { nextCluster(); });
$("note-toggle").addEventListener("click", function () {
  noteShown = !noteShown;
  $("note").hidden = !noteShown;
  this.textContent = noteShown ? "− 备注" : "+ 备注";
});
$("note").addEventListener("change", function () {
  var it = curItem();
  if (it) { it.note = this.value.trim(); }
});
$("from-start").addEventListener("change", function () {
  if (this.checked) { state.idx = 0; render(); }
});

load();
</script>
</body>
</html>
"""



# ---------------------------------------------------------------- HTTP 服务

class AuditHandler(BaseHTTPRequestHandler):
    """路由:GET / 页面;GET /api/list 列表;GET /img?file= 原图;
    POST /api/verdict 判决(可带 cluster_apply=true 整簇同判)。"""

    # 由 make_handler 注入
    items: list = []
    classes: list = []
    root: Path = None
    csv_path: Path = None
    mode: str = "all"
    verdicts: dict = {}
    file2cid: dict = {}     # file→cluster_id(仅当前 root;未启用簇功能时为空)
    cid2files: dict = {}    # cluster_id→本 root 成员列表
    cid2other: dict = {}    # cluster_id→其他数据集成员数
    lock = threading.Lock()

    def log_message(self, fmt, *args):  # 静默逐请求日志,只保留启动信息
        pass

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _safe_path(self, rel: str) -> tuple[Path | None, int | None]:
        """把相对路径解析成 root 内的绝对路径,防路径穿越。

        返回 (绝对路径, None);越界返回 (None, 403),解析异常返回 (None, 400)。
        """
        try:
            full = (self.root / rel.replace("\\", "/")).resolve()
            root_res = self.root.resolve()
            if str(full) != str(root_res) and not str(full).startswith(str(root_res) + "\\") \
                    and not str(full).startswith(str(root_res) + "/"):
                return None, 403
            return full, None
        except (ValueError, OSError):
            return None, 400

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            body = PAGE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/list":
            qs = urllib.parse.parse_qs(parsed.query)
            with AuditHandler.lock:
                merged = []
                for it in self.items:
                    v = self.verdicts.get(it["file"], {})
                    merged.append({**it, "verdict": v.get("verdict", ""), "note": v.get("note", "")})
                # 全量判决映射:同簇缩略图标记「已审」用(成员可能不在当前列表里)
                verdict_map = {f: v.get("verdict", "") for f, v in self.verdicts.items()}
            self._json({
                "root": str(self.root), "mode": self.mode,
                "csv_path": str(self.csv_path), "classes": self.classes,
                "has_clusters": bool(self.file2cid),
                "verdicts": verdict_map,
                "items": merged, "total": len(merged),
            })
        elif parsed.path == "/img":
            qs = urllib.parse.parse_qs(parsed.query)
            rel = (qs.get("file") or [""])[0]
            full, err = self._safe_path(urllib.parse.unquote(rel))
            if full is None:
                self._json({"error": "路径越界" if err == 403 else "路径非法", "file": rel},
                           err or 400)
                return
            if not full.is_file():
                self._json({"error": "图片不存在或路径越界", "file": rel}, 404)
                return
            try:
                data = full.read_bytes()
            except OSError:
                self._json({"error": "图片读取失败", "file": rel}, 500)
                return
            ctype = "image/png" if full.suffix.lower() == ".png" else "image/jpeg"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            # 原图存档永不改动 → 长缓存安全,翻回看过的图不再重新下载/解码
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(data)
        elif parsed.path == "/thumb":
            qs = urllib.parse.parse_qs(parsed.query)
            rel = (qs.get("file") or [""])[0]
            full, err = self._safe_path(urllib.parse.unquote(rel))
            if full is None:
                self._json({"error": "路径越界" if err == 403 else "路径非法", "file": rel},
                           err or 400)
                return
            if not full.is_file():
                self._json({"error": "图片不存在或路径越界", "file": rel}, 404)
                return
            data = make_thumb(self.root, full, rel)
            if not data:
                self._json({"error": "图片读取失败", "file": rel}, 500)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(data)
        else:
            self._json({"error": "unknown path"}, 404)

    def do_POST(self) -> None:
        if urllib.parse.urlparse(self.path).path != "/api/verdict":
            self._json({"error": "unknown path"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            self._json({"error": "bad json"}, 400)
            return
        file = str(payload.get("file", ""))
        verdict = str(payload.get("verdict", ""))
        if verdict not in VERDICTS:
            self._json({"error": "verdict 必须是 " + "/".join(VERDICTS)}, 400)
            return
        cluster_apply = bool(payload.get("cluster_apply"))  # 整簇同判开关
        with AuditHandler.lock:
            # 整簇同判:当前图必判(含改判场景);同簇其他成员只补未审的,
            # 已审的不覆盖;跨数据集成员不在本 root 的映射里,自然跳过
            targets = [file]
            skipped = 0
            if cluster_apply:
                cid = self.file2cid.get(file)
                for member in (self.cid2files.get(cid, []) if cid is not None else []):
                    if member == file:
                        continue
                    if member in self.verdicts:
                        skipped += 1
                    else:
                        targets.append(member)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            for t in targets:
                self.verdicts[t] = {
                    "file": t, "verdict": verdict,
                    "note": str(payload.get("note", "")),
                    "timestamp": ts,
                }
            snapshot = dict(self.verdicts)   # 锁内只取快照,写盘放到锁外——磁盘慢时不再阻塞列表请求
        save_verdicts(self.csv_path, snapshot)
        self._json({"ok": True, "file": file, "verdict": verdict,
                    "applied": len(targets), "skipped": skipped})


def make_server(args: argparse.Namespace, root: Path, csv_path: Path, items: list,
                classes: list, verdicts: dict, port: int,
                clusters: tuple[dict, dict, dict] | None = None) -> ThreadingHTTPServer:
    AuditHandler.items = items
    AuditHandler.classes = classes
    AuditHandler.root = root
    AuditHandler.csv_path = csv_path
    AuditHandler.mode = args.mode
    AuditHandler.verdicts = verdicts
    AuditHandler.file2cid, AuditHandler.cid2files, AuditHandler.cid2other = \
        clusters if clusters else ({}, {}, {})
    return ThreadingHTTPServer(("127.0.0.1", port), AuditHandler)


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"root 不存在:{root}")
    csv_path = Path(args.out) if args.out else \
        Path("data/final/images/labels") / f"audit-{root.name}.csv"

    clusters = load_clusters(args.clusters, root) if args.clusters else None
    if args.clusters and not clusters[0]:
        print(f"[提示] --clusters 已指定,但没有匹配到本 root 的路径:{args.clusters}")
    items = build_items(args, root, clusters)
    classes = load_classes(args.classes)
    verdicts = load_verdicts(csv_path)
    if verdicts:
        print(f"[恢复] 已有判决 {len(verdicts)} 条,自动跳过(勾选「从头重审」可重看)")

    port = args.port
    server = None
    for _ in range(10):
        try:
            server = make_server(args, root, csv_path, items, classes, verdicts, port,
                                 clusters=clusters)
            break
        except OSError:
            print(f"[提示] 端口 {port} 被占用,换下一个")
            port += 1
    if server is None:
        raise SystemExit(f"端口 {args.port}~{port} 都被占用")

    already = sum(1 for it in items if it["file"] in verdicts)
    print("=" * 60)
    print(f"标签审计工作台已启动(仅监听 127.0.0.1,不联网)")
    print(f"  地址:http://127.0.0.1:{port}/")
    print(f"  root:{root.resolve()}  模式:{args.mode}")
    print(f"  待审:{len(items)} 张(已审 {already} 张,自动跳过)")
    print(f"  CSV:{csv_path.resolve()}")
    if clusters and clusters[0]:
        n_multi = sum(1 for fs in clusters[1].values() if len(fs) >= 2)
        print(f"  同簇:{len(clusters[0])} 张图有 pHash 簇号,含 {n_multi} 个多图簇"
              f"(整簇同判已启用)")
    if classes:
        print(f"  类别:{', '.join(classes)}")
    else:
        print("  类别:未提供 classes.txt,UI 显示类别 id")
    print("  按 Ctrl+C 停止")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。判决保存在:", csv_path.resolve())


# ---------------------------------------------------------------- 自测

def selftest(args: argparse.Namespace) -> None:
    """--selftest 入口:整簇功能用例(临时 mini root)→ 「下一簇」导航 JS 断言
    (node,可选)→ 真实数据冒烟(若指定 --clusters)→ 基础链路自测,
    全过打印 PASS 并退出码 0。"""
    ok = True
    ok = _selftest_cluster_mini() and ok
    ok = _selftest_nav_js() and ok
    if args.clusters:
        ok = _selftest_cluster_smoke(args) and ok
    ok = _selftest_basic(args) and ok
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def _selftest_cluster_mini() -> bool:
    """整簇功能用例:临时 mini root + 假 clusters CSV(3 张同簇 + 1 张单图簇,
    同簇另有 1 张跨库成员),验证簇信息展示、整簇同判只应用未审、路径越界防护。"""
    ok = True
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
    bs = chr(92)  # CSV 里用 Windows 反斜杠路径,与真实 phash-clusters.csv 一致
    with tempfile.TemporaryDirectory(prefix="audit-cluster-test-") as td:
        tmp = Path(td)
        root = tmp / "miniroot"
        (root / "images").mkdir(parents=True)
        same = ["images/c1_1.png", "images/c1_2.png", "images/c1_3.png"]
        lone = "images/x.png"
        for rel in same + [lone]:
            (root / rel).write_bytes(png)
        lines = ["cluster_id,source,official_group,relpath"]
        lines += ["0,miniroot,train," + rel.replace("/", bs) for rel in same]
        lines.append("0,other,train,images" + bs + "elsewhere.png")  # 跨库成员
        lines.append("1,miniroot,train," + lone.replace("/", bs))
        clusters_csv = tmp / "clusters.csv"
        clusters_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
        test_csv = tmp / "selftest-verdicts.csv"  # 临时 CSV,不碰真实判决文件

        clusters = load_clusters(str(clusters_csv), root)
        ns = argparse.Namespace(mode="all", suspect_file="")
        items = build_items(ns, root, clusters)
        by_file = {it["file"]: it for it in items}
        c1 = by_file.get("images/c1_1.png")
        if not c1 or c1.get("cluster_id") != 0 or c1.get("cluster_size") != 3 \
                or c1.get("cluster_other") != 1 or len(c1.get("cluster_files", [])) != 3:
            ok = False
            print(f"[自测失败] 同簇 item 簇信息不符:{c1}")
        x = by_file.get(lone)
        if not x or x.get("cluster_id") != 1 or x.get("cluster_size") != 1:
            ok = False
            print(f"[自测失败] 单图簇信息不符:{x}")

        server = make_server(ns, root, test_csv, items, [], {}, 0, clusters)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{port}"

        def get_status(path: str):
            """返回 (status, body);HTTP 错误码原样返回不抛异常。"""
            try:
                with urllib.request.urlopen(base + path, timeout=10) as resp:
                    return resp.status, resp.read()
            except urllib.error.HTTPError as e:
                return e.code, e.read()

        def post(body: dict) -> dict:
            req = urllib.request.Request(
                base + "/api/verdict", data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def csv_rows() -> dict:
            with test_csv.open("r", encoding="utf-8-sig", newline="") as f:
                return {r["file"]: r["verdict"] for r in csv.DictReader(f)}

        listing = json.loads(get_status("/api/list")[1].decode("utf-8"))
        if not listing.get("has_clusters"):
            ok = False
            print("[自测失败] /api/list 未标记 has_clusters")
        li = {it["file"]: it for it in listing["items"]}
        if li.get("images/c1_2.png", {}).get("cluster_size") != 3:
            ok = False
            print(f"[自测失败] /api/list item 缺簇信息:{li.get('images/c1_2.png')}")

        # 1) 单条判决(旧行为):只应用 1 张
        resp = post({"file": "images/c1_1.png", "verdict": "suspect"})
        if resp.get("applied") != 1:
            ok = False
            print(f"[自测失败] 单条判决 applied={resp.get('applied')}")
        # 2) 整簇同判:当前图(含改判)+ 2 张未审成员一并应用
        resp = post({"file": "images/c1_1.png", "verdict": "pass", "cluster_apply": True})
        if resp.get("applied") != 3 or resp.get("skipped") != 0:
            ok = False
            print(f"[自测失败] 整簇同判 applied/skipped={resp}")
        rows = csv_rows()
        if rows != {"images/c1_1.png": "pass", "images/c1_2.png": "pass",
                    "images/c1_3.png": "pass"}:
            ok = False
            print(f"[自测失败] 整簇同判 CSV:{rows}")
        # 3) 已审不覆盖:对已审的 c1_2 整簇判 drop → 只改 c1_2,其余 2 张跳过
        resp = post({"file": "images/c1_2.png", "verdict": "drop", "cluster_apply": True})
        if resp.get("applied") != 1 or resp.get("skipped") != 2:
            ok = False
            print(f"[自测失败] 已审跳过 applied/skipped={resp}")
        rows = csv_rows()
        if rows.get("images/c1_2.png") != "drop" or rows.get("images/c1_1.png") != "pass" \
                or rows.get("images/c1_3.png") != "pass":
            ok = False
            print(f"[自测失败] 已审不覆盖 CSV:{rows}")
        # 4) 单图簇不带开关:行为与旧版一致
        resp = post({"file": lone, "verdict": "neg"})
        if resp.get("applied") != 1:
            ok = False
            print(f"[自测失败] 单图判决 applied={resp.get('applied')}")
        # 5) 路径越界 → 403;root 内不存在的文件 → 404
        st, _ = get_status("/img?file=" + urllib.parse.quote("../secret.txt"))
        if st != 403:
            ok = False
            print(f"[自测失败] 越界路径应 403,实际 {st}")
        st, _ = get_status("/img?file=" + urllib.parse.quote("images/nope.png"))
        if st != 404:
            ok = False
            print(f"[自测失败] 不存在文件应 404,实际 {st}")

        server.shutdown()
    return ok


def _selftest_cluster_smoke(args: argparse.Namespace) -> bool:
    """真实 root + 真实 clusters CSV 冒烟:验证路径归一化映射与簇成员确实落盘,
    不启动服务器、不写任何判决 CSV。"""
    ok = True
    root = Path(args.root)
    if not root.is_dir():
        print(f"[冒烟跳过] root 不存在:{root}")
        return True
    file2cid, cid2files, cid2other = load_clusters(args.clusters, root)
    if not file2cid:
        print(f"[冒烟失败] 簇 CSV 未匹配到本 root 任何路径:{args.clusters}")
        return False
    scanned = set(scan_images(root))
    in_scan = sum(1 for f in file2cid if f in scanned)
    print(f"[冒烟] 本 root({root.name})匹配 {len(file2cid)} 张簇图,"
          f"其中 {in_scan} 张在磁盘扫描结果里")
    if in_scan < len(file2cid):
        print(f"[冒烟提示] {len(file2cid) - in_scan} 张簇图不在扫描结果"
              f"(可能已被移动或删除,不影响审计)")
    multi = {cid: fs for cid, fs in cid2files.items() if len(fs) >= 2}
    if not multi:
        print("[冒烟失败] 没有多图簇,簇映射可疑")
        return False
    cids = sorted(multi)
    cid = cids[len(cids) // 2]  # 取中间一个簇抽查,避免总抽同一个
    members = sorted(multi[cid])
    missing = [f for f in members if f not in scanned or not (root / f).is_file()]
    if missing:
        ok = False
        print(f"[冒烟失败] 簇 {cid} 成员在磁盘上找不到:{missing}")
    else:
        print(f"[冒烟] 抽查簇 {cid}({len(members)} 张)成员全部存在,如:{members[0]}")
    if cid2other:
        print(f"[冒烟] 有 {len(cid2other)} 个簇存在跨数据集成员,"
              f"UI 将提示「另有 N 张在别的数据集」")
    return ok


NAV_JS_STUBS = r"""
// ---- DOM / fetch / Image 桩(在页面脚本之前注入,使其可在 node 里直接求值) ----
var __toasts = [];
function __el() {
  return { hidden: false, textContent: "", value: "", checked: false, className: "",
           innerHTML: "", src: "", alt: "", title: "", loading: "", style: {},
           appendChild: function () {}, addEventListener: function () {},
           getContext: function () { return null; }, getAttribute: function () { return ""; } };
}
var document = {
  getElementById: function () { return __el(); },
  createElement: function () { return __el(); },
  createTextNode: function () { return {}; },
  querySelectorAll: function () { return []; },
  addEventListener: function () {}
};
function Image() { return {}; }
var fetch = function () { return new Promise(function () {}); };  // 永不 resolve,load() 不会渲染
"""

NAV_JS_ASSERTS = r"""
// ---- 「下一簇」导航行为断言(页面脚本求值后运行) ----
toast = function (msg) { __toasts.push(String(msg)); };           // 拦截浮窗便于断言

var __fails = [];
function __assertEq(actual, expected, msg) {
  if (actual !== expected) { __fails.push(msg + ": 期望 " + expected + " 实际 " + actual); }
}
function __mkItems(spec) {
  return spec.map(function (s) {
    var it = { file: s.file, verdict: s.verdict || "", note: "" };
    if (typeof s.cid === "number") {
      it.cluster_id = s.cid; it.cluster_size = s.size || 1;
      it.cluster_other = 0; it.cluster_files = [];
    }
    return it;
  });
}

// T1: 同簇多张,下一簇跳到第一个异簇未审
var t1 = __mkItems([{cid: 0, file: "a1"}, {cid: 0, file: "a2"}, {cid: 0, file: "a3"},
                    {cid: 1, file: "b1"}, {cid: 1, file: "b2"}, {cid: 2, file: "c"}]);
__assertEq(nextClusterIndex(t1, 0, "c0"), 3, "T1 同簇跳过");
__assertEq(nextClusterIndex(t1, 3, "c1"), 5, "T2 逐簇推进");
__assertEq(nextClusterIndex(t1, 5, "c2"), 0, "T2b 尾部环绕回卷");

// T3: 只判了簇内一张,同簇其余未审也跳过
t1[0].verdict = "pass";
__assertEq(nextClusterIndex(t1, 0, "c0"), 3, "T3 同簇未审也跳过");
t1[0].verdict = "";

// T4: 无簇 item 自成一组
var t4 = __mkItems([{cid: 0, file: "a1"}, {file: "solo"}, {cid: 1, file: "b1"}]);
__assertEq(clusterKey(t4[1]), "f:solo", "T4a 无簇键");
__assertEq(nextClusterIndex(t4, 1, clusterKey(t4[1])), 2, "T4b 无簇下一簇");

// T5: 全部审完
var t5 = __mkItems([{cid: 0, file: "a", verdict: "pass"}, {cid: 1, file: "b", verdict: "drop"}]);
__assertEq(nextClusterIndex(t5, 0, "c0"), -1, "T5a 全审完无下一簇");
__assertEq(nextUnreviewedIndex(t5, 0), -1, "T5b 全审完无未审");

// T6: 只剩当前簇未审 → 回退到第一个未审(同簇)
var t6 = __mkItems([{cid: 0, file: "a1"}, {cid: 0, file: "a2"}, {cid: 1, file: "b", verdict: "drop"}]);
__assertEq(nextClusterIndex(t6, 0, "c0"), -1, "T6a 无异簇未审");
__assertEq(nextUnreviewedIndex(t6, 0), 1, "T6b 回到同簇未审");

// T7: 未审簇计数(c1 剩 1 + c2 剩 1 + 无簇 1 = 3)
state.items = __mkItems([{cid: 0, file: "a1", verdict: "pass"}, {cid: 0, file: "a2", verdict: "pass"},
                         {cid: 0, file: "a3", verdict: "pass"}, {cid: 1, file: "b1", verdict: "pass"},
                         {cid: 1, file: "b2"}, {cid: 2, file: "c"}, {file: "solo"}]);
state.hasClusters = true;
__assertEq(countPendingClusters(), 3, "T7 未审簇计数");

// T8: 无簇模式 nextCluster 等价旧的「下一张」
state.items = __mkItems([{file: "x"}, {file: "y"}]);
state.hasClusters = false; state.idx = 0;
nextCluster();
__assertEq(state.idx, 1, "T8 无簇模式=下一张");

// T9: 簇模式 nextCluster 跳过同簇未审,落到下一簇
state.items = __mkItems([{cid: 0, file: "a1"}, {cid: 0, file: "a2"}, {cid: 1, file: "b1"}]);
state.hasClusters = true; state.idx = 0;
nextCluster();
__assertEq(state.idx, 2, "T9 跳到下一簇");

// T10: 全审完 nextCluster 停在原地并提示完成
state.items = __mkItems([{cid: 0, file: "a", verdict: "pass"}, {cid: 1, file: "b", verdict: "drop"}]);
state.idx = 0;
nextCluster();
__assertEq(state.idx, 0, "T10a 全审完不跳");
__assertEq(__toasts.length > 0 && __toasts[__toasts.length - 1].indexOf("全部审完") >= 0,
           true, "T10b 提示完成");

if (__fails.length) { console.log("NAV_FAILS: " + __fails.join(" | ")); }
else { console.log("NAV_OK"); }
"""


def _selftest_nav_js() -> bool:
    """「下一簇」主导航的 JS 行为自测:抽取 PAGE_HTML 的页面脚本,在 node 里
    连同桩(DOM/fetch/Image)与断言一起运行。node 不在 PATH 时跳过并注明
    (服务器本身纯标准库,不依赖 node,这里只做行为验证)。"""
    node = shutil.which("node")
    if not node:
        print("[自测跳过] 未找到 node,「下一簇」JS 行为断言未运行")
        return True
    m = re.search(r"<script>(.*)</script>", PAGE_HTML, re.DOTALL)
    if not m:
        print("[自测失败] PAGE_HTML 中找不到 <script> 块")
        return False
    with tempfile.TemporaryDirectory(prefix="audit-nav-test-") as td:
        path = Path(td) / "nav_test.js"
        path.write_text(NAV_JS_STUBS + "\n" + m.group(1) + "\n" + NAV_JS_ASSERTS,
                        encoding="utf-8")
        try:
            proc = subprocess.run([node, str(path)], capture_output=True,
                                  text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"[自测失败] node 运行出错:{e}")
            return False
        out = (proc.stdout or "").strip()
        if proc.returncode != 0 or "NAV_OK" not in out:
            print(f"[自测失败] 下一簇导航断言未通过:"
                  f"{out or (proc.stderr or '').strip()}")
            return False
        print("[自测] 下一簇导航 JS 断言通过(含同簇跳过/回卷/无簇模式回退/全审完提示)")
        return True


def _selftest_basic(args: argparse.Namespace) -> bool:
    """基础链路自测:启动→自请求 /、/api/list、/img→POST 一条判决→删测试 CSV。

    仅在自测里顺带检查图片可读性(整文件读一次),读不出的跳过并计数;
    正常使用不做该检查,避免启动变慢。
    """
    root = Path(args.root)
    # selftest 是测试产物,写完即删:挪 .temp/,不污染 final/images/labels/(2026-09-16 目录重构)
    csv_path = Path(args.out) if args.out else \
        Path(".codex/docs/.temp") / f"selftest-audit-{root.name}.csv"
    if csv_path.exists():
        csv_path.unlink()

    items = build_items(args, root)
    classes = load_classes(args.classes)

    unreadable = 0
    checked: list = []
    for it in items:
        full = root / it["file"]
        try:
            if full.stat().st_size == 0:
                unreadable += 1
            else:
                full.read_bytes()
                checked.append(it)
        except OSError:
            unreadable += 1
    if unreadable:
        print(f"[自测] {unreadable} 张图片读不出,已跳过")

    if not checked:
        raise SystemExit("[自测失败] 没有可读图片")

    # 用一个随机高位端口,避免与用户已开的服务冲突
    server = make_server(args, root, csv_path, checked, classes, {}, 0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    ok = True

    def get(path: str) -> bytes:
        nonlocal ok
        with urllib.request.urlopen(base + path, timeout=10) as resp:
            if resp.status != 200:
                ok = False
                print(f"[自测失败] GET {path} 状态 {resp.status}")
            return resp.read()

    page = get("/")
    if b"canvas" not in page:
        ok = False
        print("[自测失败] 首页缺少 canvas")
    listing = json.loads(get("/api/list").decode("utf-8"))
    if listing["total"] != len(checked):
        ok = False
        print(f"[自测失败] 列表数 {listing['total']} != 待审数 {len(checked)}")
    img = get("/img?file=" + urllib.parse.quote(checked[0]["file"]))
    if not img:
        ok = False
        print("[自测失败] /img 返回为空")

    # POST 一条判决,确认 CSV 落盘且内容正确
    target = checked[0]["file"]
    req = urllib.request.Request(
        base + "/api/verdict",
        data=json.dumps({"file": target, "verdict": "suspect", "note": "自测"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        json.loads(resp.read().decode("utf-8"))
    if not csv_path.is_file():
        ok = False
        print("[自测失败] CSV 未落盘")
    else:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 1 or rows[0]["file"] != target or rows[0]["verdict"] != "suspect" \
                or rows[0]["note"] != "自测" or not rows[0]["timestamp"]:
            ok = False
            print(f"[自测失败] CSV 内容不符:{rows}")

    server.shutdown()
    if csv_path.exists():
        csv_path.unlink()
    print(f"[自测] 待审 {len(checked)} 张,读不出跳过 {unreadable} 张")
    return ok


def main() -> None:
    args = parse_args()
    if args.selftest:
        selftest(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
