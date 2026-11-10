# -*- coding: utf-8 -*-
"""修复 dataset4 manifest 中 10 行陈旧来源记录(第二轮补跑覆盖写入所致)。

约束(team-lead 2026-09-14):
  - 只改 manifest.csv 文本; images/ 内 127 张图一律不动。
  - 修复前提: images 侧 md5 复核(images/_run2_verify.json 由后台任务生成于目录根)
    中每条 URL 的重下载字节与恰好一个现存文件全等, 且该文件属于待修复清单。
  - 旧名映射用 manifest.csv.bak-重排前 交叉核对。

用法: python repair_dataset4_manifest.py [--dry-run]
"""
import csv
import json
import os
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                    "data", "raw", "dataset4-external-negatives"))
VERIFY = os.path.join(BASE, "_run2_verify.json")
MANIFEST = os.path.join(BASE, "manifest.csv")
BACKUP = os.path.join(BASE, "manifest.csv.bak-重排前")

# 第二轮(2026-09-14 15:49-15:52)来源记录, 键=run2 期文件名(1 基), 值=(URL, 来源页, 许可, 时间, 宽, 高)
RUN2 = {
    "extneg_commons_0001.jpg": ("https://upload.wikimedia.org/wikipedia/commons/6/66/Aspens%2C_Boulder_Mountain_%283678755867%29.jpg",
                                "https://commons.wikimedia.org/wiki/File:Aspens,_Boulder_Mountain_(3678755867).jpg",
                                "CC BY-SA 2.0", "2026-09-14 15:49:55", 1600, 1057),
    "extneg_commons_0002.jpg": ("https://upload.wikimedia.org/wikipedia/commons/6/60/Boulder_stream_north_of_Foel_Deg-arbedol_-_geograph.org.uk_-_1125836.jpg",
                                "https://commons.wikimedia.org/wiki/File:Boulder_stream_north_of_Foel_Deg-arbedol_-_geograph.org.uk_-_1125836.jpg",
                                "CC BY-SA 2.0", "2026-09-14 15:50:12", 480, 640),
    "extneg_commons_0003.jpg": ("https://upload.wikimedia.org/wikipedia/commons/e/e6/Boulders-5592%2C_Tralee_Bay%2C_Co._Kerry%2C_Ireland.jpg",
                                "https://commons.wikimedia.org/wiki/File:Boulders-5592,_Tralee_Bay,_Co._Kerry,_Ireland.jpg",
                                "CC BY-SA 4.0", "2026-09-14 15:50:43", 1920, 1080),
    "extneg_commons_0004.jpg": ("https://upload.wikimedia.org/wikipedia/commons/0/09/Boulders-9622%2C_Tralee_Bay%2C_Co._Kerry%2C_Ireland.jpg",
                                "https://commons.wikimedia.org/wiki/File:Boulders-9622,_Tralee_Bay,_Co._Kerry,_Ireland.jpg",
                                "CC BY-SA 4.0", "2026-09-14 15:50:46", 1920, 1080),
    "extneg_commons_0005.jpg": ("https://upload.wikimedia.org/wikipedia/commons/8/8d/Little_River_%26_boulders_of_Thunderhead_Sandstone_%28Neoproterozoic%3B_Great_Smoky_Mountains%2C_Tennessee%2C_USA%29_2_%2837095710366%29.jpg",
                                "https://commons.wikimedia.org/wiki/File:Little_River_&_boulders_of_Thunderhead_Sandstone_(Neoproterozoic;_Great_Smoky_Mountains,_Tennessee,_USA)_2_(37095710366).jpg",
                                "CC BY 2.0", "2026-09-14 15:51:42", 1600, 1200),
    "extneg_commons_0006.jpg": ("https://upload.wikimedia.org/wikipedia/commons/f/f8/Little_River_%26_boulders_of_Thunderhead_Sandstone_%28Neoproterozoic%3B_Great_Smoky_Mountains%2C_Tennessee%2C_USA%29_3_%2836472149353%29.jpg",
                                "https://commons.wikimedia.org/wiki/File:Little_River_&_boulders_of_Thunderhead_Sandstone_(Neoproterozoic;_Great_Smoky_Mountains,_Tennessee,_USA)_3_(36472149353).jpg",
                                "CC BY 2.0", "2026-09-14 15:51:46", 1600, 1200),
    "extneg_commons_0007.jpg": ("https://upload.wikimedia.org/wikipedia/commons/f/f3/Little_River_%26_boulders_of_Thunderhead_Sandstone_%28Neoproterozoic%3B_Great_Smoky_Mountains%2C_Tennessee%2C_USA%29_4_%2836472145863%29.jpg",
                                "https://commons.wikimedia.org/wiki/File:Little_River_&_boulders_of_Thunderhead_Sandstone_(Neoproterozoic;_Great_Smoky_Mountains,_Tennessee,_USA)_4_(37095710366).jpg",
                                "CC BY 2.0", "2026-09-14 15:51:58", 1600, 1200),
    "extneg_commons_0008.jpg": ("https://upload.wikimedia.org/wikipedia/commons/d/df/Little_River_%26_boulders_of_Thunderhead_Sandstone_%28Neoproterozoic%3B_Great_Smoky_Mountains%2C_Tennessee%2C_USA%29_5_%2837095696396%29.jpg",
                                "https://commons.wikimedia.org/wiki/File:Little_River_&_boulders_of_Thunderhead_Sandstone_(Neoproterozoic;_Great_Smoky_Mountains,_Tennessee,_USA)_5_(37095696396).jpg",
                                "CC BY 2.0", "2026-09-14 15:52:02", 1600, 1200),
    "extneg_commons_0011.jpg": ("https://upload.wikimedia.org/wikipedia/commons/3/3f/Old_Rag_boulders%2C_fog_%2812349832754%29.jpg",
                                "https://commons.wikimedia.org/wiki/File:Old_Rag_boulders,_fog_(12349832754).jpg",
                                "Public domain", "2026-09-14 15:52:17", 1712, 2288),
    "extneg_commons_0012.jpg": ("https://upload.wikimedia.org/wikipedia/commons/1/17/Snow_on_Boulder_Mountain_in_late_spring_%283685113158%29.jpg",
                                "https://commons.wikimedia.org/wiki/File:Snow_on_Boulder_Mountain_in_late_spring_(3685113158).jpg",
                                "CC BY-SA 2.0", "2026-09-14 15:52:29", 1600, 1200),
}


def norm_url(u):
    return u.split("?")[0]


def main():
    dry = "--dry-run" in sys.argv
    with open(VERIFY, encoding="utf-8") as f:
        verify = json.load(f)

    # 1) 每条 URL 状态 ok, 且恰好比对到一个现存文件
    url2file = {}
    for old, rec in RUN2.items():
        v = verify.get(old, {})
        if v.get("status") != "ok":
            raise SystemExit("验证未完成/失败: %s -> %r" % (old, v))
        if norm_url(v["url"]) != norm_url(rec[0]):
            raise SystemExit("URL 不一致: %s" % old)
        hits = v.get("match_files", [])
        if len(hits) != 1:
            raise SystemExit("URL 比对到 %d 个现存文件(预期 1): %s -> %r" % (len(hits), old, hits))
        url2file[old] = (hits[0], rec, v.get("size"))

    # 2) 比对到的现存文件必须正是当前 manifest 中尺寸/时间对不上的行
    with open(MANIFEST, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    by_name = {r[0]: r for r in data}
    for old, (cur, rec, size) in url2file.items():
        r = by_name[cur]
        if (int(r[5]), int(r[6])) == (rec[4], rec[5]) and r[4] == rec[3]:
            pass  # 已是正确记录
        else:
            print("[fix] %s <- %s (现名) 行记(%s,%s,%s) 改为 (%d,%d,%s)" %
                  (old, cur, r[5], r[6], r[4], rec[4], rec[5], rec[3]))

    # 3) 备份交叉核对: 备份里同 URL 的旧名应与 run2 期名一致或仅差 0 基/1 基偏移
    with open(BACKUP, encoding="utf-8-sig", newline="") as f:
        bdata = list(csv.reader(f))[1:]
    bak_by_url = {norm_url(r[1]): r[0] for r in bdata}
    mapping = []
    for old, (cur, rec, _s) in sorted(url2file.items()):
        bak_name = bak_by_url.get(norm_url(rec[0]), "?")
        mapping.append((old, bak_name, cur))
        print("[map] run2期名=%s  备份名=%s  现名=%s" % (old, bak_name, cur))

    if dry:
        print("dry-run, 未写入")
        return
    for old, (cur, rec, _s) in url2file.items():
        r = by_name[cur]
        r[1], r[2], r[3], r[4], r[5], r[6] = rec[0], rec[1], rec[2], rec[3], str(rec[4]), str(rec[5])
    with open(MANIFEST, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)
    print("manifest updated")


if __name__ == "__main__":
    main()
