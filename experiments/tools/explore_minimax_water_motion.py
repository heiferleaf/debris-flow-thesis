"""One generic-flow control, then a conditional two-second motion-reference trial.

The current user authorized feasible low-cost attempts. The announced scope is
CNY0.64 for the control and CNY0.96 for transfer, at most one request per stage.
No automatic creation retries, no model fallback, no training release.
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video.minimax_preview_workflow import ACTIVE, MinimaxPreview, digest, read_json
from experiments.common.video import minimax_official as official
from experiments.common.video.synthetic_wan import atomic_json, utc_now

BATCH = "water-motion-reference-20261109-v1"
GENERIC_PROMPT = ("固定摄像机拍摄一条已经在流动的清浅溪流近景，水面占画面三分之二。"
    "从第一帧开始就是已有的连续流水，溪水从画面上方向下方持续流过。"
    "清楚可见水面纹理和少量自然白色水花向下游移动，经过固定石块时分流、变形并继续下行。"
    "河岸和石块保持原位，真实自然摄影，一镜到底。")
TRANSFER_PROMPT = ("图一是目标监控场景。视频一只提供真实感水流的运动方式，不采用视频一的河岸、石块、地形或镜头。"
    "保留图一的河岸、山体、道路、建筑和机位，在图一从画面右侧延伸到左下方的河道内呈现普通河水。"
    "第一帧起河道内就已经有连续水面，上游来水从画面右侧河道入口持续进入，顺河道流向左下方并流出画面。"
    "水面运动参考视频一：水纹连续输送、局部变形并绕过固定石块，水量保持稳定，不从干河床内部冒出。"
    "固定监控镜头，真实自然画面，不出现参考标记。")


def video_data(path):
    data = path.read_bytes()
    if len(data) > 10*1024*1024:
        raise ValueError("运动参考超出本轮大小范围。")
    return "data:video/mp4;base64," + base64.b64encode(data).decode("ascii")


def plan(workflow, stage):
    folder = workflow.metadata / "motion-reference-trial"
    record = folder / (stage + ".json")
    if record.exists():
        raise ValueError("这一步已有作业记录，只可查询原任务，不重复付费。")
    if any(row["status"] in ACTIVE for row in workflow.history()):
        raise ValueError("已有作业未结束或回执待核对。")
    content = [{"type":"text", "text":GENERIC_PROMPT if stage == "generic" else TRANSFER_PROMPT}]
    value = {"scene_id":"camera-c9c07aee7f28", "provider":"minimax_official", "model":official.MODEL,
             "resolution":"480P", "duration":4, "batch_id":BATCH, "training_ready":False,
             "polygon":[[0,0],[1,0],[1,1],[0,1]], "prompt":content[0]["text"]}
    if stage == "generic":
        value.update(input_mode="text_only", variant_code="D", variant_name="通用流水近景，无场景图", estimated_cny=.64,
                     target_scene_reference_used=False)
    else:
        source = read_json(folder / "generic.json")
        approval = read_json(folder / "generic-inspection.json")
        if not source.get("id") or approval.get("eligible_for_motion_reference") is not True:
            raise ValueError("通用流水对照尚未通过运动参考初检，停止第二步。")
        source_video = workflow.run_path(source["id"]).parent / "candidate.mp4"
        if not source_video.is_file():
            raise ValueError("运动源视频尚不存在。")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        folder.mkdir(parents=True, exist_ok=True)
        clip = folder / "generic-water-motion-2s.mp4"
        if not clip.exists():
            subprocess.run([ffmpeg,"-nostdin","-v","error","-ss","0.5","-i",str(source_video),"-t","2","-r","24","-an","-c:v","libx264","-pix_fmt","yuv420p","-crf","20",str(clip)],check=True)
        import cv2
        capture = cv2.VideoCapture(str(clip))
        fps,count = capture.get(cv2.CAP_PROP_FPS),capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.release()
        if fps != 24 or count != 48:
            raise ValueError("本轮输入运动视频须为2秒、24fps，不能超出0.96元估算范围。")
        reference = workflow.asset("camera-c9c07aee7f28","dry")
        content += [{"type":"image_url","role":"reference_image","image_url":{"url":"data:image/png;base64,"+base64.b64encode(reference.read_bytes()).decode("ascii")}},
                    {"type":"video_url","role":"reference_video","video_url":{"url":video_data(clip)}}]
        value.update(input_mode="image+motion_video",variant_code="E",variant_name="干床场景＋独立水流运动参考",estimated_cny=.96,
                     polygon=workflow.draft("camera-c9c07aee7f28")["polygon"],reference_sha256=digest(reference),
                     reference_video_sha256=digest(clip),reference_video_seconds=2,reference_parent_job=source["id"])
    body = {"model":official.MODEL,"content":content,"resolution":"480P","duration":4,"ratio":"16:9"}
    assert value["estimated_cny"] in (.64,.96) and official.MODEL == "MiniMax-H3-Lite"
    return record,value,body


def execute(workflow, record, value, body):
    identity = "minimax-official-" + secrets.token_hex(8)
    value.update(id=identity,created_at=utc_now(),status="submission_unknown",message="正在提交本轮明确授权的一次试片。")
    atomic_json(record,value)
    atomic_json(workflow.jobs/identity/"state.json",value)
    try:
        secret = workflow.secret_loader()
        response = workflow.http.post(official.BASE_URL+"/v2/video_generation",json=body,
                                      headers={"Authorization":"Bearer "+secret},timeout=90,allow_redirects=False)
        if not response.ok:
            workflow.update(identity,status="request_rejected" if 400 <= response.status_code < 500 else "submission_unknown",
                            message="请求未完成："+workflow.safe_error(response)+"；没有自动重试。")
            return identity
        task = response.json().get("task_id")
        if not isinstance(task,str) or not task.isdigit():
            raise ValueError("回执没有有效任务号。")
        workflow.update(identity,status="submitted",task_id=task,message="平台已接收，等待本轮视频。")
        print("Submitted %s %s" % (value["variant_code"],identity),flush=True)
        workflow.poll(identity)
    except Exception as exc:
        workflow.update(identity,status="submission_unknown",message="请求回执需核对（%s），不自动重试。" % type(exc).__name__)
    return identity


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("generic","transfer"),required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    workflow=MinimaxPreview(ROOT)
    record,value,body=plan(workflow,args.stage)
    print("Stage %s; %s;480P;4s;CNY%.2f" % (args.stage,value["model"],value["estimated_cny"]),flush=True)
    if args.execute:
        identity=execute(workflow,record,value,body)
        print("Finished %s: %s" % (identity,read_json(workflow.run_path(identity)).get("status")),flush=True)


if __name__ == "__main__":
    main()
