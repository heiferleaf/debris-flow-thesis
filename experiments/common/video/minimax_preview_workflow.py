"""Local drafts, explicit single-job consent and separate media acceptance."""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import re
import secrets
import threading
import time
from urllib.parse import urlsplit

import requests
from PIL import Image, ImageDraw

from experiments.common.video.synthetic_wan import atomic_json, utc_now
from experiments.common.video import minimax_official as official

MODEL = official.MODEL
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,159}$")
ACTIVE = {"uploading", "submission_unknown", "submitted", "running", "poll_timeout", "download_failed"}
PROMPT = ("图一提供真实场景，图二仅用蓝色边界和橙色箭头说明河道位置及期望流向；最终视频不要出现这些标记。"
          "保持图一的机位、山体、道路、建筑和河岸布局，在图二标出的河道内生成已经持续流动的普通河水，"
          "从箭头起点向终点流动。水面局部波纹随水流持续移动、变化，在露出水面的固定石块附近分流与汇合。"
          "水流受河岸约束，不越过河岸，不形成泥石流阵头、滚石或新增物料堆积。"
          "固定监控镜头，一镜到底，水确实在流动；不是整张图片平移，也不是只有亮度闪烁。")
WET_PROMPT = ("延续首帧中已经存在的普通河水。水由画面左下方的上游持续进入，沿原河道向画面中部、右上方流动，"
              "经过中央弯道后从右侧河道流出。清楚可辨的水纹与少量自然白色水花沿这个方向随水流移动并变化，"
              "经过固定石块时分流并继续下行。保持首帧中已有的连续水域和水位，整段持续来水。"
              "固定监控镜头，河岸、山体、道路、建筑、石块和独立水池保持原位。不要从河床内部冒水，"
              "不要生成从无水到有水的过程，不要移动摄影机，不增加泥石流阵头、滚石或参考标记。")


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else ({} if default is None else default)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def points(value, minimum, maximum):
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError("河道至少需要 3 点，流向至少需要上游、下游 2 点。")
    result = []
    for p in value:
        if not isinstance(p, list) or len(p) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in p):
            raise ValueError("标记点必须位于图片范围内。")
        result.append([round(x, 6) for x in p])
    return result


def locator_bytes(image, polygon, flow):
    with Image.open(image) as original:
        picture = original.convert("RGB")
    w, h = picture.size
    draw = ImageDraw.Draw(picture)
    xy = [(int(x*w), int(y*h)) for x, y in polygon]
    draw.line(xy + xy[:1], fill="#00bce8", width=max(3, w//250))
    trajectory = [(int(x*w), int(y*h)) for x, y in flow]
    a, b = trajectory[-2:]
    draw.line(trajectory, fill="#ff791f", width=max(4, w//220))
    angle, size = math.atan2(b[1]-a[1], b[0]-a[0]), max(18, w//45)
    draw.polygon([b, (b[0]-size*math.cos(angle-.45), b[1]-size*math.sin(angle-.45)),
                  (b[0]-size*math.cos(angle+.45), b[1]-size*math.sin(angle+.45))], fill="#ff791f")
    stream = io.BytesIO()
    picture.save(stream, format="PNG")
    return stream.getvalue()


class MinimaxPreview:
    def __init__(self, root, *, http=None, secret_loader=None):
        self.root = Path(root).resolve()
        self.metadata = self.root / "data/metadata/synthetic-video/minimax-workbench-v2"
        self.jobs = self.root / "data/process/synthetic-video/jobs"
        self.inventory = self.root / "data/metadata/synthetic-video/dry-scenes-20261108-v2/source-inventory.json"
        self.lock = threading.RLock()
        self.challenges = {}
        self.http = http or requests.Session()
        self.http.trust_env = False
        self.secret_loader = secret_loader or self.load_secret
        self.polling = set()

    def load_secret(self):
        return official.load_secret(self.root)

    def configure_key(self, data):
        official.store_secret(self.root, data.get("key"), replace=data.get("replace") is True)
        return {"configured": True, "message": "官方密钥已在本机加密保存，未调用生成接口。"}

    def scenes(self):
        return read_json(self.inventory).get("items", [])

    def scene(self, identity):
        for row in self.scenes():
            if row["scene_id"] == identity:
                return row
        raise ValueError("未找到这个场景。")

    def asset(self, identity, kind):
        row = self.scene(identity)
        if kind in ("wet", "wet_crop", "wet_user"):
            filename = "wet-motion-crop.png" if kind == "wet_crop" else "wet-first-frame.png"
            version = "user-water-region-20261109-v2" if kind == "wet_user" else "wet-first-frame-20261109-v1"
            path = self.root / "data/process/synthetic-video" / version / identity / filename
            if not path.is_file():
                raise ValueError("这个场景尚未准备有水首帧。")
            return path
        field = {"dry": "output_path", "source": "source_path"}.get(kind)
        if not field:
            raise ValueError("素材类型无效。")
        path = (self.root / row[field]).resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            raise ValueError("场景路径不在项目内。")
        if not path.is_file():
            raise ValueError("场景素材不存在。")
        return path

    def draft(self, identity):
        self.scene(identity)
        return read_json(self.metadata / "drafts" / (identity + ".json"),
                         {"scene_id": identity, "polygon": [], "flow": [], "prompt": PROMPT,
                          "duration": 4, "reference_ok": False})

    def validate_draft(self, data, *, ready=False):
        identity = data.get("scene_id")
        self.scene(identity)
        duration = data.get("duration", 4)
        if type(duration) is not int or duration != official.DURATION:
            raise ValueError("本轮只生成 4 秒低成本试片。")
        prompt = data.get("prompt", "")
        mode = data.get("input_mode", "image_url+image_url")
        profile = data.get("profile", "lite")
        official.profile(profile)
        if mode not in ("first_frame", "image_url+image_url"):
            raise ValueError("输入方式无效。")
        if profile == "standard-water-v2" and mode != "first_frame":
            raise ValueError("本次标准版试片只使用新标记区域的有水首帧。")
        if mode == "first_frame":
            self.asset(identity, "wet_user" if profile == "standard-water-v2" else "wet")
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 7000:
            raise ValueError("请填写不超过 7000 字符的提示词。")
        polygon = points(data.get("polygon", []), 3 if ready else 0, 80)
        flow = points(data.get("flow", []), 2 if ready else 0, 16)
        if ready:
            if data.get("reference_ok") is not True:
                raise ValueError("请先核对场景图、河道范围和期望流向。")
            area = abs(sum(polygon[i][0]*polygon[(i+1)%len(polygon)][1]-polygon[(i+1)%len(polygon)][0]*polygon[i][1] for i in range(len(polygon)))) / 2
            if area < .0005 or math.dist(flow[0], flow[1]) < .02:
                raise ValueError("河道范围或流向过小，请重新标记。")
        return {"scene_id": identity, "polygon": polygon, "flow": flow, "prompt": prompt.strip(),
                "duration": duration, "input_mode": mode, "profile": profile, "reference_ok": data.get("reference_ok") is True}

    def save_draft(self, data):
        value = self.validate_draft(data)
        value.update(updated_at=utc_now())
        atomic_json(self.metadata / "drafts" / (value["scene_id"] + ".json"), value)
        return value

    def run_path(self, identity):
        if not isinstance(identity, str) or not NAME.fullmatch(identity):
            raise ValueError("作业编号无效。")
        path = self.jobs / identity / "state.json"
        if not identity.startswith(("minimax-official-", "minimax-water-v2-", "minimax-h3-")) or not path.is_file():
            raise ValueError("作业不存在。")
        return path

    def history(self):
        rows = []
        for path in sorted(self.jobs.glob("minimax-*/state.json"), reverse=True):
            value = read_json(path)
            if not path.parent.name.startswith(("minimax-official-", "minimax-water-v2-", "minimax-h3-")):
                continue
            row = {k: value.get(k) for k in ("scene_id", "status", "provider_status", "message", "model", "resolution", "provider", "motion_check", "duration", "estimated_cny", "created_at", "task_id", "prompt", "reference_sha256", "locator_sha256", "input_mode", "reference_kind", "batch_id", "variant_code", "variant_name", "presentation_video", "profile")}
            identity = path.parent.name
            row.update(id=identity, scene_id=value.get("scene_id") or "camera-c9c07aee7f28",
                       has_video=(path.parent / ("candidate-composited.mp4" if value.get("reference_kind") == "wet_crop" else "candidate.mp4")).is_file(),
                       has_inspection16=(path.parent / "inspection16.jpg").is_file(),
                       can_resume=bool(value.get("provider") == "minimax_official" and value.get("task_id") and identity not in self.polling and value.get("status") in ACTIVE),
                       review=read_json(self.metadata / "reviews" / (identity + ".json")))
            rows.append(row)
        rows.sort(key=lambda row: row.get("created_at") or "", reverse=True)
        return rows

    def video_path(self, identity):
        state_path = self.run_path(identity)
        state = read_json(state_path)
        composed = state_path.parent / "candidate-composited.mp4"
        if state.get("presentation_video") == composed.name and composed.is_file():
            return composed
        return state_path.parent / "candidate.mp4"

    def snapshot(self):
        return {"scenes": [{"scene_id": x["scene_id"], "camera": x["camera"], "index": x["index"], "wet_available": (self.root / "data/process/synthetic-video/wet-first-frame-20261109-v1" / x["scene_id"] / "wet-first-frame.png").is_file(), "wet_user_available": (self.root / "data/process/synthetic-video/user-water-region-20261109-v2" / x["scene_id"] / "wet-first-frame.png").is_file()} for x in self.scenes()],
                "jobs": self.history(), "default_prompt": PROMPT, "wet_prompt": WET_PROMPT, "credential_configured": official.configured(self.root),
                "profile": {"provider":"minimax_official","model":MODEL,"resolution":official.RESOLUTION,"duration":official.DURATION,"rate_cny":official.RATE_CNY,"estimated_cny":official.MAX_COST_CNY},
                "profiles": official.PROFILES,
                "training_ready": False}

    @staticmethod
    def fingerprint(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()

    def prepare(self, data):
        value = self.validate_draft(data, ready=True)
        with self.lock:
            if any(x["status"] in ACTIVE for x in self.history()):
                raise ValueError("已有未结束或回执待核对的任务，请先查询该任务。")
            reference_kind = ("wet_user" if value["profile"] == "standard-water-v2" else "wet") if value["input_mode"] == "first_frame" else "dry"
            image = self.asset(value["scene_id"], reference_kind)
            reference_sha = digest(image)
            locator = locator_bytes(image, value["polygon"], value["flow"])
            locator_sha = hashlib.sha256(locator).hexdigest()
            value.update(reference_sha256=reference_sha, locator_sha256=locator_sha,
                         provider="minimax_official", reference_kind=reference_kind, **official.profile(value["profile"]))
            fingerprint = self.fingerprint(value)
            prepared = self.metadata / "prepared" / fingerprint
            prepared.mkdir(parents=True, exist_ok=True)
            (prepared / "locator.png").write_bytes(locator)
            atomic_json(prepared / "intent.json", value)
            token = secrets.token_urlsafe(32)
            self.challenges.clear()
            self.challenges[token] = (time.monotonic() + 600, value, fingerprint)
            return {"token": token, "fingerprint": fingerprint, "intent": value,
                    "locator_url": "/api/prepared/" + fingerprint + "/locator.png"}

    def submit(self, data):
        with self.lock:
            entry = self.challenges.pop(data.get("token", ""), None)
            if data.get("approved") is not True or not entry or time.monotonic() > entry[0]:
                raise ValueError("请重新核对本条图片、提示词和费用。")
            _, value, fingerprint = entry
            if any(x["status"] in ACTIVE for x in self.history()):
                raise ValueError("已有任务待完成或核对，不能重复提交。")
            if digest(self.asset(value["scene_id"], value.get("reference_kind", "dry"))) != value["reference_sha256"]:
                raise ValueError("场景图已经变化，请重新核对。")
            locator = self.metadata / "prepared" / fingerprint / "locator.png"
            if digest(locator) != value["locator_sha256"]:
                raise ValueError("河道定位图已经变化，请重新核对。")
            if not official.configured(self.root) and self.secret_loader == self.load_secret:
                raise ValueError("请先配置 MiniMax 官方密钥；百炼密钥不用于本次生成。")
            if value.get("profile") == "standard-water-v2":
                phase = [x for x in self.history() if x.get("batch_id") == "user-water-region-20261109-v2"]
                if phase:
                    raise ValueError("本次标准版已分配一条作业，不能重复付费。")
                value = {**value, "batch_id":"user-water-region-20261109-v2", "variant_code":"H", "variant_name":"用户河道标记＋标准版H3"}
            elif value["input_mode"] == "first_frame":
                phase = [x for x in self.history() if x.get("batch_id") == "wet-first-frame-20261109-v1"]
                if len(phase) >= 2:
                    raise ValueError("本阶段两条低价试片额度已使用，需先讨论结果和后续费用。")
                value = {**value, "batch_id":"wet-first-frame-20261109-v1", "variant_code":"F" if not phase else "G",
                         "variant_name":"有水首帧＋核定流向" if not phase else "有水首帧第二次修订"}
            identity = "minimax-official-" + secrets.token_hex(8)
            path = self.jobs / identity / "state.json"
            atomic_json(path, {**value, "id": identity, "prepared_fingerprint": fingerprint,
                              "created_at": utc_now(), "status": "uploading", "message": "正在上传本次确认的素材。",
                              "training_ready": False})
            threading.Thread(target=self.perform, args=(identity,), daemon=True).start()
            return {"id": identity}

    def update(self, identity, **changes):
        with self.lock:
            path = self.run_path(identity)
            value = read_json(path)
            value.update(changes, updated_at=utc_now())
            atomic_json(path, value)

    @staticmethod
    def safe_error(response):
        code = ""
        try:
            value = response.json()
            value = value.get("error", value)
            candidate = value.get("code", value.get("type", ""))
            if isinstance(candidate, str) and NAME.fullmatch(candidate):
                code = " / " + candidate
        except (ValueError, AttributeError):
            pass
        return "HTTP %d%s" % (response.status_code, code)

    def perform(self, identity):
        posted = False
        try:
            value = read_json(self.run_path(identity))
            secret = self.secret_loader()
            body = official.request_body(value, self.asset(value["scene_id"], value.get("reference_kind", "dry")),
                                        self.metadata / "prepared" / value["prepared_fingerprint"] / "locator.png")
            self.update(identity, status="submission_unknown", message="正在向 MiniMax 官方提交一次 %.2f 元试片。" % value["estimated_cny"])
            posted = True
            response = self.http.post(official.BASE_URL + "/v2/video_generation", json=body,
                headers={"Authorization": "Bearer " + secret}, timeout=90, allow_redirects=False)
            if not response.ok:
                rejected = response.status_code in (400, 401, 402, 403, 404, 413, 422, 429)
                self.update(identity, status="request_rejected" if rejected else "submission_unknown", message="提交未完成：" + self.safe_error(response) + "。没有自动重新生成。")
                return
            task = response.json().get("task_id")
            if not isinstance(task, str) or not NAME.fullmatch(task):
                raise ValueError("回执没有有效任务号，请核对平台记录。")
            self.update(identity, status="submitted", task_id=task, message="平台已接收，等待生成。")
            self.poll(identity)
        except Exception as exc:
            detail = str(exc) if type(exc) is ValueError else type(exc).__name__
            self.update(identity, status="submission_unknown" if posted else "failed_before_submit",
                        message="操作未完成：%s。没有自动重新生成。" % detail)

    def poll(self, identity):
        with self.lock:
            if identity in self.polling:
                return
            self.polling.add(identity)
        try:
            value = read_json(self.run_path(identity))
            if value.get("provider") != "minimax_official":
                raise ValueError("这是百炼历史任务，不向官方接口发送其任务号或密钥。")
            if not value.get("task_id") or not NAME.fullmatch(value["task_id"]):
                raise ValueError("没有有效任务号，需在 MiniMax 官方控制台核对回执。")
            secret = self.secret_loader()
            for _ in range(480):
                response = self.http.get(official.BASE_URL + "/v2/query/video_generation/" + value["task_id"], headers={"Authorization": "Bearer " + secret}, timeout=30, allow_redirects=False)
                if not response.ok:
                    self.update(identity, status="poll_timeout", message="查询未完成：" + self.safe_error(response) + "。可查询已有任务，不要重新生成。")
                    return
                payload = response.json()
                output = payload.get("task", {})
                remote = output.get("status")
                if remote == "succeeded":
                    self.update(identity, provider_status=remote, status="download_failed", message="生成成功，正在保存视频。")
                    self.download(identity, secret, output)
                    return
                if remote in {"failed", "cancelled", "canceled", "expired"}:
                    code = output.get("error", {}).get("code", "")
                    code = code if isinstance(code, str) and NAME.fullmatch(code) else ""
                    self.update(identity, provider_status=remote, status="failed", message="平台任务%s：%s；没有自动重试。" % (remote, code))
                    return
                self.update(identity, status="running", provider_status=remote, message="平台%s，正在等待视频。" % remote)
                time.sleep(15)
            self.update(identity, status="poll_timeout", message="轮询超时；可查询已有任务，不要重新生成。")
        except Exception as exc:
            self.update(identity, status="poll_timeout", message="查询或保存中断（%s）；请查询已有任务。" % type(exc).__name__)
        finally:
            with self.lock:
                self.polling.discard(identity)

    def download(self, identity, secret, output):
        result = self.run_path(identity).parent / "candidate.mp4"
        if result.is_file():
            self.update(identity, status="succeeded", message="视频已保存，等待播放验收。")
            return
        result_url = output.get("content", {}).get("url", "")
        url = urlsplit(result_url)
        if url.scheme != "https" or not url.hostname:
            raise ValueError("结果地址无效。")
        response = self.http.get(result_url, timeout=120, stream=True, allow_redirects=False)
        response.raise_for_status()
        temporary = result.with_suffix(".mp4.part")
        size = 0
        with temporary.open("wb") as f:
            for chunk in response.iter_content(1024*1024):
                size += len(chunk)
                if size > 512*1024*1024:
                    raise ValueError("结果文件过大。")
                f.write(chunk)
        if not size:
            raise ValueError("结果为空。")
        temporary.replace(result)
        self.update(identity, status="succeeded", result_sha256=digest(result), usage=output.get("usage", {}),
                    actual_model=output.get("model"), message="视频已保存，等待播放验收。", training_ready=False)
        try:
            current = read_json(self.run_path(identity))
            report = official.motion_check(result, current["polygon"])
            self.update(identity, motion_check=report, message=report["message"])
        except Exception as exc:
            self.update(identity, motion_check={"flag":"check_failed","message":"运动检查未完成（%s），请播放核实。" % type(exc).__name__})

    def review(self, data):
        identity = data.get("job_id")
        path = self.run_path(identity)
        if not (path.parent / "candidate.mp4").is_file():
            raise ValueError("没有可验收的视频。")
        outcome = data.get("outcome")
        checks = data.get("checks", {})
        if outcome not in {"accepted", "rejected"} or not isinstance(checks, dict):
            raise ValueError("验收结果无效。")
        checks = {k: checks.get(k) is True for k in ("motion", "shape", "background")}
        if outcome == "accepted" and not all(checks.values()):
            raise ValueError("请播放视频，逐项确认三项条件；接口成功不能替代质量验收。")
        note = data.get("note", "")
        if not isinstance(note, str) or len(note) > 2000:
            raise ValueError("备注不超过 2000 字。")
        review = {"job_id": identity, "outcome": outcome, "checks": checks, "note": note,
                  "reviewer": "user", "updated_at": utc_now(), "training_ready": False}
        atomic_json(self.metadata / "reviews" / (identity + ".json"), review)
        return review
