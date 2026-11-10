# -*- coding: utf-8 -*-
"""下载外部负样本候选图片（山区沟道/河道/溪流，画面中无泥石流）。

来源:
  1. Wikimedia Commons 全文搜索 (File 命名空间), 取 1600px 缩略图, 许可白名单过滤
  2. Openverse API (聚合 Flickr CC0/CC BY/CC BY-SA / StockSnap CC0 等)

输出:
  data/raw/dataset4-external-negatives/extneg_*.jpg|png|webp
  data/raw/dataset4-external-negatives/manifest.csv
    filename,source_url,page_url,license,download_time,width,height

过滤规则: 单边 >= 640px, 文件 >= 30KB 且 <= 15MB, PIL 可打开; 内容 md5 去重。
用法: python download_external_negatives.py

归档(2026-09-16 data 结构重构核对):下载产物写 data/raw/dataset4-external-negatives/,属 raw
只读层、目录结构不变,无需改动。补下载未了结时(还有待补充的负例)可重跑,不牵涉 path 迁移。
"""
import collections
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
from datetime import datetime

import requests
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(REPO, "data", "raw", "dataset4-external-negatives", "images")
MANIFEST = os.path.join(OUT, os.pardir, "manifest.csv")
UA = {"User-Agent": "debris-flow-thesis-research/1.0 (local script, academic use)"}
TARGET_TOTAL = 280
MIN_W = 640
MIN_BYTES = 30 * 1024
MAX_BYTES = 15 * 1024 * 1024
LICENSE_OK = re.compile(
    r"^(cc0|public domain|pdm|cc by(-sa)?( [0-9.]+)?( [a-z]{2})?|"
    r"attribution( [-a-z]+)?|by|by-sa)",
    re.I,
)

stats = collections.defaultdict(lambda: [0, 0])  # tag -> [ok, fail]
fail_log = []  # (query, source, reason)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_manifest_rows():
    if not os.path.exists(MANIFEST):
        return []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_manifest(row):
    new = not os.path.exists(MANIFEST)
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["filename", "source_url", "page_url", "license",
                        "download_time", "width", "height"])
        w.writerow([row["filename"], row["source_url"], row["page_url"],
                    row["license"], row["download_time"], row["width"], row["height"]])


def content_license_ok(lic):
    if not lic:
        return False
    return bool(LICENSE_OK.match(lic.strip()))


def ext_from_pil(img):
    fmt = (img.format or "JPEG").upper()
    return {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(fmt)


def fetch_bytes(url, attempts=3):
    """GET 二进制; 429/401/5xx 指数退避重试。"""
    for i in range(attempts):
        try:
            r = requests.get(url, headers=UA, timeout=40, stream=True)
        except Exception as e:
            if i == attempts - 1:
                return None, "ERR %s" % e
            time.sleep(5 * (i + 1))
            continue
        if r.status_code == 200:
            return r, None
        if r.status_code in (401, 403, 429, 500, 502, 503) and i < attempts - 1:
            time.sleep(8 * (i + 1))
            continue
        return None, "HTTP %d" % r.status_code
    return None, "retries exhausted"


def try_download(url, page_url, license_str, src_tag, seq, seen_md5, rows):
    """下载并验证单张图; 成功写 manifest 返回 True。"""
    try:
        r, err = fetch_bytes(url)
        if r is None:
            fail_log.append((src_tag, page_url, err))
            stats[src_tag][1] += 1
            return False
        data = r.raw.read(MAX_BYTES + 1, decode_content=True)
        if len(data) < MIN_BYTES:
            fail_log.append((src_tag, page_url, "too small %dB" % len(data)))
            stats[src_tag][1] += 1
            return False
        if len(data) > MAX_BYTES:
            fail_log.append((src_tag, page_url, "too large"))
            stats[src_tag][1] += 1
            return False
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception as e:
            fail_log.append((src_tag, page_url, "PIL %s" % e))
            stats[src_tag][1] += 1
            return False
        w, h = img.size
        if w < MIN_W and h < MIN_W:
            fail_log.append((src_tag, page_url, "lowres %dx%d" % (w, h)))
            stats[src_tag][1] += 1
            return False
        ext = ext_from_pil(img)
        if not ext:
            fail_log.append((src_tag, page_url, "fmt %s" % img.format))
            stats[src_tag][1] += 1
            return False
        md5 = hashlib.md5(data).hexdigest()
        if md5 in seen_md5:
            stats[src_tag][1] += 1
            return False
        seen_md5.add(md5)
        fname = "extneg_%s_%04d%s" % (src_tag, seq, ext)
        with open(os.path.join(OUT, fname), "wb") as f:
            f.write(data)
        append_manifest({
            "filename": fname, "source_url": url, "page_url": page_url,
            "license": license_str, "download_time": now(),
            "width": w, "height": h,
        })
        stats[src_tag][0] += 1
        return True
    except Exception as e:
        fail_log.append((src_tag, page_url, "ERR %s" % e))
        stats[src_tag][1] += 1
        return False


def commons_search(query, limit, seen_urls, seen_md5, rows, counter, cap):
    """Commons File 命名空间搜索; 返回本次新增下载数。"""
    got = 0
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrnamespace": 6, "gsrsearch": query, "gsrlimit": limit,
        "prop": "imageinfo", "iiprop": "url|size|extmetadata", "iiurlwidth": 1600,
    }
    try:
        d = requests.get("https://commons.wikimedia.org/w/api.php", params=params,
                         headers=UA, timeout=40).json()
    except Exception as e:
        fail_log.append((query, "commons-api", "ERR %s" % e))
        return 0
    pages = list(d.get("query", {}).get("pages", {}).values())
    for p in pages:
        if counter["total"] >= TARGET_TOTAL or got >= cap:
            break
        ii = (p.get("imageinfo") or [{}])[0]
        title = p.get("title", "")
        if not re.search(r"\.(jpe?g|png|webp|tiff?)$", title, re.I):
            continue  # svg/gif/视频等跳过
        url = ii.get("thumburl") or ii.get("url")
        if not url or url in seen_urls:
            continue
        lic = (ii.get("extmetadata", {}).get("LicenseShortName", {}) or {}).get("value", "")
        if not content_license_ok(lic):
            fail_log.append((query, title, "license %r" % lic))
            continue
        tw = ii.get("thumbwidth") or ii.get("width") or 0
        if tw and tw < MIN_W:
            continue
        if try_download(url, "https://commons.wikimedia.org/wiki/" +
                        title.replace(" ", "_"), lic, "commons", counter["commons"], seen_md5, rows):
            seen_urls.add(url)
            counter["commons"] += 1
            got += 1
            counter["total"] += 1
        time.sleep(1.2)
    return got


def openverse_query(query, limit_pages, seen_urls, seen_md5, rows, counter, cap,
                    extra=None):
    got = 0
    for page in range(1, limit_pages + 1):
        if counter["total"] >= TARGET_TOTAL or got >= cap:
            break
        params = {
            "q": query, "page_size": 50, "page": page,
            "license": "cc0,pdm,by,by-sa",
        }
        if extra:
            params.update(extra)
        try:
            r = requests.get("https://api.openverse.org/v1/images/", params=params,
                             headers=UA, timeout=40)
            if r.status_code != 200:
                fail_log.append((query, "openverse-api", "HTTP %d" % r.status_code))
                break
            results = r.json().get("results", [])
        except Exception as e:
            fail_log.append((query, "openverse-api", "ERR %s" % e))
            break
        if not results:
            break
        for it in results:
            if counter["total"] >= TARGET_TOTAL or got >= cap:
                break
            url = it.get("url") or ""
            if not url or url in seen_urls:
                continue
            lic = "%s %s" % (it.get("license", ""), it.get("license_version", ""))
            prov = it.get("source", "openverse")
            w = it.get("width") or 0
            if w and w < MIN_W:
                continue
            tag = counter["ov_tag"] if prov in ("stocksnap", "rawpixel") else "openverse"
            if try_download(url, it.get("foreign_landing_url") or url, lic,
                            tag, counter[tag], seen_md5, rows):
                seen_urls.add(url)
                counter[tag] += 1
                got += 1
                counter["total"] += 1
        time.sleep(1.0)
    return got


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load_manifest_rows()
    seen_urls = set(r["source_url"] for r in rows)
    seen_md5 = set()
    counter = {"total": len(rows), "commons": 0, "openverse": 0,
               "stocksnap": 0, "ov_tag": "stocksnap"}
    start_total = counter["total"]

    commons_queries = [
        ("mountain torrent ravine stream", 40),
        ("mountain stream boulders", 40),
        ("gravel bar river", 40),
        ("creek bed rocks", 40),
        ("mountain river gorge", 40),
        ("alluvial fan", 30),
        ("ephemeral stream wadi", 25),
        ("mountain creek clear water", 30),
        ("rocky riverbed", 30),
        ("debris flow deposit", 15),
    ]
    for q, lim in commons_queries:
        if counter["total"] >= TARGET_TOTAL:
            break
        n = commons_search(q, lim, seen_urls, seen_md5, rows, counter, cap=130)
        print("[commons] %r +%d (total %d)" % (q, n, counter["total"]))
        time.sleep(1.0)

    ov_queries = [
        "mountain stream rocks", "mountain river", "creek boulders",
        "gravel river bed", "mountain gorge river", "alluvial fan",
        "alpine stream", "rocky creek water", "mountain creek",
        "riverbed stones",
    ]
    for q in ov_queries:
        if counter["total"] >= TARGET_TOTAL:
            break
        n = openverse_query(q, 2, seen_urls, seen_md5, rows, counter, cap=110)
        print("[openverse] %r +%d (total %d)" % (q, n, counter["total"]))

    # StockSnap CC0 高清补充
    for q in ["mountain river", "stream rocks", "creek forest"]:
        if counter["total"] >= TARGET_TOTAL:
            break
        n = openverse_query(q, 1, seen_urls, seen_md5, rows, counter, cap=40,
                            extra={"source": "stocksnap"})
        print("[stocksnap] %r +%d (total %d)" % (q, n, counter["total"]))

    summary = {
        "downloaded_this_run": counter["total"] - start_total,
        "per_tag": {k: v for k, v in counter.items() if isinstance(v, int)},
        "stats_ok_fail": {k: v for k, v in stats.items()},
        "fails": fail_log,
    }
    with open(os.path.join(OUT, os.pardir, "_download_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "fails"},
                     ensure_ascii=False))
    print("fail count:", len(fail_log))


if __name__ == "__main__":
    main()
