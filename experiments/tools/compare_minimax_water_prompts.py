"""Three user-authorized prompt previews; no automatic generation retries."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video.minimax_preview_workflow import ACTIVE, MinimaxPreview, read_json
from experiments.common.video.synthetic_wan import atomic_json, utc_now

BATCH_ID = "water-prompts-20261109-v1"
VARIANTS = [
    ("A", "开头已有完整水流", "视频从第0秒起，河道内就已经有贯通上游到下游的连续水面。四秒中始终维持这个已存在的流水状态，不展示水从干河床出现、渗出、汇集或逐渐铺开的过程。"),
    ("B", "上游补给与流出路径", "这是一段由上游持续来水的河流。来水从画面右侧沟道的画外入口进入，顺着橙色路线流向左下方，继续从画面左下方流出。水量由上游连续补给，河床内部任何位置都不是出水源。水不要从地下冒出，也不要在干河床上凭空出现。"),
    ("C", "清楚可辨的水面运动", "把普通浅河水拍成能直接看见流动的视频：河道内水面的小波纹与少量自然白色水花持续向左下方输送，经过固定石块时局部变形并继续下行。第一帧起水就已经在流，水面纹理有明显的定向移动，河床石块本身不移动。"),
]


def create_batch(workflow, *, execute=False):
    manifest = workflow.metadata / "prompt-batches" / (BATCH_ID + ".json")
    if manifest.exists():
        raise ValueError("这组已创建；只可查询原任务，不能再次付费提交。")
    if any(job["status"] in ACTIVE for job in workflow.history()):
        raise ValueError("已有任务进行中或回执待核对。")
    draft = workflow.draft("camera-c9c07aee7f28")
    base = read_json(workflow.jobs / "minimax-official-f6349ffaed0d0fd9/state.json").get("prompt")
    if not base:
        raise ValueError("找不到上条试片的实际提示词。")
    prepared = []
    for code, name, addition in VARIANTS:
        current = workflow.prepare({**draft, "prompt": base + "\n本版本重点：" + addition, "reference_ok": True, "duration": 4})
        prepared.append({"variant_code": code, "variant_name": name, "id": "minimax-official-" + secrets.token_hex(8),
                         "intent": current["intent"], "prepared_fingerprint": current["fingerprint"]})
    if len({p["intent"]["reference_sha256"] for p in prepared}) != 1 or len({p["intent"]["locator_sha256"] for p in prepared}) != 1:
        raise ValueError("三组参考图必须完全相同。")
    total = round(sum(p["intent"]["estimated_cny"] for p in prepared), 2)
    if len(prepared) != 3 or total != 1.92:
        raise ValueError("本轮仅授权3条，共计约1.92元。")
    if not execute:
        return {"batch_id": BATCH_ID, "total_estimated_cny": total, "variants": prepared, "execute": False}
    atomic_json(manifest, {"batch_id": BATCH_ID, "created_at": utc_now(), "total_estimated_cny": total,
                          "authorization": "User requested several prompt dimensions with parallel generation in this conversation.",
                          "variants": prepared, "no_automatic_generation_retry": True, "training_ready": False})
    for row in prepared:
        atomic_json(workflow.jobs / row["id"] / "state.json", {**row["intent"], "id": row["id"],
                    "batch_id": BATCH_ID, "variant_code": row["variant_code"], "variant_name": row["variant_name"],
                    "prepared_fingerprint": row["prepared_fingerprint"], "created_at": utc_now(),
                    "status": "uploading", "message": "本轮提示词对照待提交；每版本仅一次。", "training_ready": False})
    return {"batch_id": BATCH_ID, "total_estimated_cny": total, "variants": prepared, "execute": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    workflow = MinimaxPreview(ROOT)
    batch = create_batch(workflow, execute=args.execute)
    print("Batch %s; three previews; estimated CNY %.2f" % (batch["batch_id"], batch["total_estimated_cny"]), flush=True)
    for row in batch["variants"]:
        print("%s %s %s" % (row["variant_code"], row["variant_name"], row["id"]), flush=True)
    if args.execute:
        def run(row):
            # Each concurrent request has its own HTTP session and key loader.
            worker = MinimaxPreview(ROOT)
            worker.perform(row["id"])
            state = read_json(worker.run_path(row["id"]))
            print("%s finished: %s; %s" % (row["variant_code"], state["status"], state.get("message", "")), flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(run, batch["variants"]))


if __name__ == "__main__":
    main()
