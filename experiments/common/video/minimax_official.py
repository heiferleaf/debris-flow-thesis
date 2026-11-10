"""MiniMax China V2 adapter; no credentials or network are used on import."""
import base64
import os
from pathlib import Path

from experiments.common.video.local_credentials import load_runway_secret, store_runway_secret, validated_secret

MODEL = "MiniMax-H3-Lite"
RESOLUTION = "480P"
DURATION = 4
RATE_CNY = .16
MAX_COST_CNY = .64
BASE_URL = "https://api.minimax.cn"
PROFILES = {
    "lite": {"model": MODEL, "resolution": RESOLUTION, "duration": DURATION, "rate_cny": RATE_CNY, "estimated_cny": MAX_COST_CNY},
    "standard-water-v2": {"model": "MiniMax-H3", "resolution": "768P", "duration": 4, "rate_cny": .50, "estimated_cny": 2.0},
}


def profile(name="lite"):
    if name not in PROFILES:
        raise ValueError("试片规格不存在。")
    return dict(PROFILES[name])


def key_path(root):
    return Path(root) / ".codex/secrets/minimax.key.dpapi"


def configured(root):
    return key_path(root).is_file() or bool(os.environ.get("MINIMAX_API_KEY"))


def load_secret(root):
    if key_path(root).is_file():
        return load_runway_secret(key_path(root))
    value = os.environ.get("MINIMAX_API_KEY")
    if value:
        return validated_secret(value)
    raise ValueError("请先在本机工作台配置 MiniMax 官方按量付费密钥。")


def store_secret(root, value, *, replace=False):
    if not isinstance(value, str) or not 16 <= len(value.strip()) <= 8192:
        raise ValueError("官方密钥格式不完整。")
    store_runway_secret(value, key_path(root), replace=replace)


def request_body(intent, reference, locator=None):
    expected = profile(intent.get("profile", "lite"))
    if any(intent.get(key) != expected[key] for key in ("model", "resolution", "duration", "estimated_cny")):
        raise ValueError("模型、画质、时长及费用必须与本次确认的规格一致。")
    first_frame = intent.get("input_mode") == "first_frame"
    content = [{"type": "text", "text": intent["prompt"]}]
    for path in ((reference,) if first_frame else (reference, locator)):
        data = Path(path).read_bytes()
        if len(data) > 20*1024*1024:
            raise ValueError("参考图片过大。")
        url = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": url}, "role": "first_frame" if first_frame else "reference_image"})
    result = {"model": expected["model"], "content": content, "resolution": expected["resolution"], "duration": expected["duration"]}
    if not first_frame:
        result["ratio"] = "16:9"
    return result


def motion_check(path, polygon):
    """Low-motion warning only; never declares realistic water or a valid N label."""
    import cv2
    import numpy as np
    capture = cv2.VideoCapture(str(path))
    try:
        fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or fps <= 0 or count < 2:
            return {"flag": "decode_failed", "message": "视频未能正常解码，质量未通过。"}
        frames = []
        for position in np.linspace(0, max(0,count-1), 9):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = capture.read()
            if ok:
                h,w = frame.shape[:2]
                frame = cv2.resize(frame, (640, max(1,round(h*640/w))))
                frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        if len(frames) < 3:
            return {"flag":"decode_failed","message":"有效帧不足，质量未通过。"}
        h,w = frames[0].shape
        mask=np.zeros((h,w),np.uint8)
        cv2.fillPoly(mask,[np.array([[min(w-1,int(x*w)),min(h-1,int(y*h))] for x,y in polygon],np.int32)],255)
        river=cv2.erode(mask,np.ones((3,3),np.uint8))>0
        background=~(cv2.dilate(mask,np.ones((15,15),np.uint8))>0)
        background[:int(h*.15),:]=False
        water_values,background_values=[],[]
        for previous,current in zip(frames,frames[1:]):
            flow=cv2.calcOpticalFlowFarneback(previous,current,None,.5,3,15,3,5,1.2,0)
            magnitude=np.linalg.norm(flow,axis=2)
            if river.any():water_values.append(float(np.percentile(magnitude[river],90)))
            if background.any():background_values.append(float(np.percentile(magnitude[background],90)))
        water=float(np.median(water_values)) if water_values else 0.
        fixed=float(np.median(background_values)) if background_values else 0.
        low=water < max(.2, fixed*1.25)
        return {"flag":"low_motion" if low else "needs_visual_review", "river_flow_p90_px":round(water,4),
                "background_flow_p90_px":round(fixed,4),"analysis_width":640,"frames_sampled":len(frames),
                "message":"低运动警告：未检测到显著高于固定背景的河道运动，请先播放核实。" if low else "检测到画面变化，仍须播放检查水流形态；这不证明真实水流。",
                "threshold_kind":"preview_warning_only_not_physical_criterion", "duration_seconds":round(count/fps,3)}
    finally:
        capture.release()
