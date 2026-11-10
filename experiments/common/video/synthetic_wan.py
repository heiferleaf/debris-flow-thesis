"""Narrow Wan 2.6 media-preview boundary; Python 3.8 stdlib only.

Import, plan and status never decrypt credentials or make HTTP requests. Actual
execution requires execute=True and a separate human authorization binding the
manifest, state path, exact request fingerprint and CNY cap. A lost POST receipt
reserves the full cost and is never retried automatically. This module never
creates an approval or assigns a ground-truth label to generated media.

Official contract checked 2026-11-08:
https://help.aliyun.com/zh/model-studio/legacy-image-to-video-api-reference/
Beijing's existing dashscope.aliyuncs.com endpoint remains supported. This
profile explicitly selects 720P/10s/audio=false, no audio input or prompt
rewriting. shot_type is omitted because it only applies with prompt_extend=true.
The CNY 0.15/s silent price was checked in the signed-in Beijing console.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib import error, parse, request


ROOT = Path(__file__).resolve().parents[3]
WAN_KEY_PATH = ROOT / ".codex" / "secrets" / "dashscope.key.dpapi"
API_BASE = "https://dashscope.aliyuncs.com"
MODEL = "wan2.6-i2v-flash"
COST_CENTS = 150
PROFILE = {"model": MODEL, "resolution": "720P", "duration": 10, "audio": False,
           "prompt_extend": False, "watermark": False, "seed": 20261108}
ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")
SHA = re.compile(r"^[a-f0-9]{64}$")
REMOTE_STATUSES = {"PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"}


class WorkflowError(ValueError):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code, self.status = code, status


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def credential_configured():
    # Presence only: no environment value or ciphertext is read for UI status.
    return "DASHSCOPE_API_KEY" in os.environ or WAN_KEY_PATH.is_file()


def store_wan_secret(value, path=WAN_KEY_PATH, replace=False):
    # Reuse the generic current-user DPAPI implementation with a NEW path only.
    from experiments.common.video.local_credentials import store_runway_secret
    return store_runway_secret(value, Path(path), replace=replace)


def load_wan_secret(path=WAN_KEY_PATH):
    from experiments.common.video.local_credentials import load_runway_secret, validated_secret
    value = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not value:
        try:
            value = load_runway_secret(Path(path))
        except FileNotFoundError:
            raise WorkflowError("credential_missing", "请先运行本地密钥配置命令。", 403) from None
    return validated_secret(value)


def safe_https(value):
    if not isinstance(value, str):
        raise ValueError("Expected HTTPS video URL")
    url = parse.urlsplit(value)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.fragment:
        raise ValueError("Expected HTTPS video URL without credentials")
    return value


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward API credentials to another endpoint.


class UrllibTransport:
    """One attempt per operation. Downloads have no Authorization header."""
    def __init__(self, timeout=60):
        self.timeout = timeout
        self.opener = request.build_opener(NoRedirect())

    def request(self, method, url, headers, body=None):
        req = request.Request(url, data=body, headers=headers, method=method)
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                return response.status, response.read(2 * 1024 * 1024)
        except error.HTTPError as response:
            return response.code, b""  # Do not echo provider errors containing input/key material.

    def download(self, url, stream):
        req = request.Request(safe_https(url), headers={"Accept": "video/*"}, method="GET")
        with self.opener.open(req, timeout=self.timeout) as response:
            if response.status != 200:
                raise ValueError("Video download did not return HTTP 200")
            count = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > 512 * 1024 * 1024:
                    raise ValueError("Video exceeds the local 512 MiB receipt limit")
                stream.write(chunk)


class WanWorkflow:
    def __init__(self, manifest_path, state_path, *, root=ROOT, output_dir=None,
                 execute=False, authorization_path=None, transport=None, secret_loader=None):
        self.root = Path(root).resolve()
        self.manifest_path, self.state_path = Path(manifest_path).resolve(), Path(state_path).resolve()
        if self.manifest_path == self.state_path:
            raise ValueError("Manifest and state must use separate paths")
        self.manifest_bytes = self.manifest_path.read_bytes()
        self.manifest_sha256 = digest(self.manifest_bytes)
        self.manifest = json.loads(self.manifest_bytes.decode("utf-8-sig"))
        jobs = self.manifest.get("jobs", [])
        if (self.manifest.get("purpose") != "media_preview" or self.manifest.get("training_ready") is not False
                or len(jobs) != 1 or not ID.fullmatch(jobs[0].get("id", ""))):
            raise ValueError("This workbench accepts exactly one explicit media-preview job")
        self.jobs = {job["id"]: job for job in jobs}
        self.output_dir = Path(output_dir or self.root / "data/process/synthetic-video/jobs").resolve()
        self.execute = execute is True
        self.authorization_path = Path(authorization_path).resolve() if authorization_path else None
        self.transport = transport or UrllibTransport()
        self.secret_loader = secret_loader or load_wan_secret

    def job(self, job_id):
        if job_id not in self.jobs:
            raise WorkflowError("job_not_found", "未注册的生成任务。", 404)
        return self.jobs[job_id]

    def state(self):
        state = read_json(self.state_path) if self.state_path.is_file() else {
            "schema_version": 1, "provider": "dashscope", "manifest_sha256": self.manifest_sha256,
            "jobs": {}, "activity": []}
        if (state.get("provider") != "dashscope" or state.get("manifest_sha256") != self.manifest_sha256
                or not isinstance(state.get("jobs"), dict)):
            raise WorkflowError("state_mismatch", "持久状态与当前清单不匹配；请先人工核对。", 409)
        for job_id, row in state["jobs"].items():
            if job_id not in self.jobs or row.get("id") != job_id:
                raise WorkflowError("state_mismatch", "持久状态中的任务身份不匹配。", 409)
            if row.get("reserved_cost_cents", 0) not in (0, COST_CENTS):
                raise WorkflowError("state_mismatch", "持久预算记录无效。", 409)
        return state

    @contextmanager
    def locked_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.state_path.with_name(self.state_path.name + ".lock")
        try:
            descriptor = os.open(str(lock), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            raise WorkflowError("state_locked", "任务状态正被使用；遗留锁须人工核对后处理。", 409) from None
        os.close(descriptor)
        try:
            yield self.state()
        finally:
            lock.unlink()

    def row(self, state, job_id):
        job = self.job(job_id)
        return state["jobs"].setdefault(job_id, {"id": job_id, "status": "draft",
            "prompt_zh": job["prompt_zh"], "prompt_en": job["prompt_en"], "reserved_cost_cents": 0,
            "accepted_label": None, "training_ready": False, "preview_only": True})

    def event(self, state, job_id, action, status, message):
        state.setdefault("activity", []).append({"at": utc_now(), "job_id": job_id,
            "action": action, "status": status, "message": message})

    def persist(self, state, job_id=None):
        atomic_json(self.state_path, state)
        if job_id and state["jobs"][job_id].get("reserved_cost_cents"):
            row = state["jobs"][job_id]
            atomic_json(self.output_dir / job_id / "manifest.json", {
                "schema_version": 1, "provider": "dashscope", "model": MODEL,
                "source_manifest_sha256": self.manifest_sha256, "job": row,
                "source_reference": self.job(job_id)["selected_reference"],
                "activity": [item for item in state["activity"] if item["job_id"] == job_id],
                "accepted_label": None, "training_ready": False})

    def draft(self, job_id, state=None):
        state = state if state is not None else self.state()
        job = self.job(job_id)
        return {**job, **state["jobs"].get(job_id, {})}

    def validate(self, job_id, state=None):
        job = self.draft(job_id, state)
        if (job.get("preview_only") is not True or job.get("training_ready") is not False
                or job.get("accepted_label") is not None or job.get("kind") != "negative_candidate"):
            raise WorkflowError("preview_contract", "此入口仅准备未验收的负类媒体候选。")
        if job.get("settings") != PROFILE:
            raise WorkflowError("profile_mismatch", "仅支持已核对的720P、10秒、无声固定参数。")
        for name in ("prompt_zh", "prompt_en"):
            value = job.get(name)
            if not isinstance(value, str) or not value.strip() or len(value) > 1500 or any(ord(c) < 32 and c not in "\n\t" for c in value):
                raise WorkflowError("invalid_prompt", "中英文提示词均须非空且不超过1500字符。")
        ref = job.get("selected_reference", {})
        path = (self.root / ref.get("path", "")).resolve()
        try:
            path.relative_to(self.root / "data")
        except ValueError:
            raise WorkflowError("reference_invalid", "参考图必须来自项目data目录中的注册路径。") from None
        if (ref.get("accepted") is not True or ref.get("lineage_complete") is not True
                or ref.get("source_camera") != job.get("source_camera")
                or not SHA.fullmatch(ref.get("sha256", "")) or not path.is_file()
                or digest(path.read_bytes()) != ref["sha256"]):
            raise WorkflowError("reference_invalid", "参考图验收、谱系或文件指纹未通过。")
        use = self.manifest.get("reference_use", {})
        if (use.get("external_test") is not False or use.get("checked_current_frozen_test_list") is not True
                or not isinstance(use.get("frozen_test_cameras"), list)
                or job["source_camera"] in use["frozen_test_cameras"]):
            raise WorkflowError("reference_scope", "参考图的独立测试范围核对未通过。")
        ancestry = ref.get("parent_lineage")
        if not isinstance(ancestry, list) or not ancestry:
            raise WorkflowError("reference_scope", "参考图缺少来源谱系。")
        for parent in ancestry:
            if (parent.get("source_camera") != job["source_camera"] or parent.get("split") in {"test", "external_test"}
                    or not parent.get("source_unit") or not parent.get("source_group")
                    or not SHA.fullmatch(parent.get("source_media_sha256", ""))):
                raise WorkflowError("reference_scope", "参考图谱系存在未认可或测试来源。")
        data = path.read_bytes()
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"} or len(data) > 20 * 1024 * 1024:
            raise WorkflowError("reference_invalid", "参考图格式或大小超出本入口支持范围。")
        return job, path

    def request_identity(self, job_id, state=None):
        job, _ = self.validate(job_id, state)
        identity = {"job_id": job_id, "model": MODEL, "prompt": job["prompt_en"],
            "prompt_zh_review_text": job["prompt_zh"], "parameters": {k: v for k, v in PROFILE.items() if k != "model"},
            "reference_sha256": job["selected_reference"]["sha256"]}
        return identity, digest(json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    def authorization(self, job_id, state=None):
        if not self.authorization_path or not self.authorization_path.is_file():
            raise WorkflowError("authorization_missing", "尚无独立人工授权文件；本轮仅准备工作台。", 403)
        try:
            auth = read_json(self.authorization_path)
            _, fingerprint = self.request_identity(job_id, state)
            cost = Decimal(str(auth.get("max_cost_cny"))) * 100
            state_path = Path(auth.get("state_path", ""))
            if not state_path.is_absolute():
                state_path = self.authorization_path.parent / state_path
            if (auth.get("approved") is not True or auth.get("provider") != "dashscope"
                    or auth.get("model") != MODEL or auth.get("manifest_sha256") != self.manifest_sha256
                    or state_path.resolve() != self.state_path or auth.get("allowed_job_ids") != [job_id]
                    or auth.get("request_sha256_by_job", {}).get(job_id) != fingerprint
                    or not all(isinstance(auth.get(key), str) and auth[key].strip() for key in
                               ("authorization_id", "approved_by", "approved_at"))
                    or not cost.is_finite() or cost != cost.to_integral_value() or cost < COST_CENTS
                    or cost > COST_CENTS or type(auth.get("max_jobs")) is not int or auth.get("max_jobs") != 1
                    or auth.get("automatic_recharge") is not False or auth.get("automatic_retry") is not False):
                raise ValueError("Authorization binding mismatch")
            datetime.fromisoformat(auth["approved_at"].replace("Z", "+00:00"))
            return auth
        except (ValueError, TypeError, KeyError, OSError, InvalidOperation):
            raise WorkflowError("authorization_invalid", "授权须绑定当前提示词、参考图、清单、状态路径及1.50元单条上限。", 403) from None

    def plan(self, job_id, state=None):
        state = state if state is not None else self.state()
        job = self.draft(job_id, state)
        blockers, fingerprint, identity = [], None, None
        try:
            identity, fingerprint = self.request_identity(job_id, state)
        except WorkflowError as exc:
            blockers.append(str(exc))
        auth_status, auth_message = "missing", "尚无独立人工授权。"
        try:
            self.authorization(job_id, state)
            auth_status, auth_message = "approved", "独立人工授权已匹配当前请求。"
        except WorkflowError as exc:
            auth_status = "missing" if exc.code == "authorization_missing" else "invalid"
            auth_message = str(exc)
            blockers.append(auth_message)
        if not self.execute:
            blockers.append("服务以准备模式启动，未启用真实执行。")
        if not credential_configured():
            blockers.append("尚未配置本地百炼API Key。")
        if job.get("status", "draft") != "draft":
            blockers.append("该任务已有提交记录；不允许重复提交。")
        return {"job_id": job_id, "network_requests": 0, "credentials_read": False,
            "request_preview": {"model": MODEL, "input": {"prompt": job.get("prompt_en"),
                "img_url": "[local accepted reference; base64 only on authorized submit]"},
                "parameters": {k: v for k, v in PROFILE.items() if k != "model"}},
            "estimated_cny": 1.5, "budget_authorized": auth_status == "approved",
            "authorization_status": auth_status, "authorization_message": auth_message,
            "submission_allowed": not blockers, "blockers": blockers,
            "request_sha256": fingerprint, "manifest_sha256": self.manifest_sha256,
            "reference_sha256": (identity or {}).get("reference_sha256"), "training_ready": False}

    def edit(self, job_id, changes):
        if not isinstance(changes, dict) or not changes or set(changes) - {"prompt_zh", "prompt_en"}:
            raise WorkflowError("draft_fields_only", "只能修改草稿中的中英文提示词。")
        with self.locked_state() as state:
            row = self.row(state, job_id)
            if row["status"] != "draft":
                raise WorkflowError("draft_locked", "提交过的任务提示词不可改写。", 409)
            row.update(changes)
            self.validate(job_id, state)
            row.pop("last_plan", None)
            self.event(state, job_id, "edit", "saved", "提示词草稿已保存；更改后须重新匹配授权。")
            self.persist(state)

    def save_plan(self, job_id):
        with self.locked_state() as state:
            plan = self.plan(job_id, state)
            row = self.row(state, job_id)
            row["last_plan"] = {"at": utc_now(), "request_sha256": plan["request_sha256"]}
            self.event(state, job_id, "plan", "complete", "离线请求预览已完成；未读取密钥，未发送参考图。")
            self.persist(state)
            return plan

    def execution_gate(self, job_id, state):
        if not self.execute:
            raise WorkflowError("execution_not_authorized", "当前为准备模式，真实生成未获授权。", 403)
        return self.authorization(job_id, state)

    def submit(self, job_id):
        with self.locked_state() as state:
            auth = self.execution_gate(job_id, state)
            row = self.row(state, job_id)
            if row["status"] != "draft":
                return dict(row)  # Strictly idempotent, including unknown/failed submissions.
            job, path = self.validate(job_id, state)
            _, fingerprint = self.request_identity(job_id, state)
            if sum(item.get("reserved_cost_cents", 0) for item in state["jobs"].values()) + COST_CENTS > COST_CENTS:
                raise WorkflowError("budget_exceeded", "累计预算已被占用；不自动充值或重试。", 403)
            reference_bytes = path.read_bytes()
            if digest(reference_bytes) != job["selected_reference"]["sha256"]:
                raise WorkflowError("reference_changed", "参考图已变化，请重新检查并确认。", 409)
            secret = self.secret_loader()  # Only after every local authorization/reference/budget gate.
            if not isinstance(secret, str) or not secret or any(ord(c) < 32 for c in secret):
                raise WorkflowError("credential_invalid", "本地凭据格式无效。", 403)
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            body = {"model": MODEL, "input": {"prompt": job["prompt_en"],
                "img_url": "data:" + mime + ";base64," + base64.b64encode(reference_bytes).decode("ascii")},
                "parameters": {k: v for k, v in PROFILE.items() if k != "model"}}
            payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            row.update(status="submission_unknown", reserved_cost_cents=COST_CENTS,
                authorization_id=auth["authorization_id"], request_sha256=fingerprint,
                payload_sha256=digest(payload), settings=dict(PROFILE), attempt_started_at=utc_now(),
                source_camera=job["source_camera"], generation_preview_camera_use=True)
            self.event(state, job_id, "submit", "active", "开始一次提交；已占用1.50元上限。")
            self.persist(state, job_id)  # Durable BEFORE the potentially chargeable POST.
            try:
                status, raw = self.transport.request("POST", API_BASE + "/api/v1/services/aigc/video-generation/video-synthesis",
                    {"Authorization": "Bearer " + secret, "X-DashScope-Async": "enable", "Content-Type": "application/json"}, payload)
                if not 200 <= status < 300:
                    raise ValueError("HTTP_%d" % status)
                result = json.loads(raw.decode("utf-8"))
                output = result.get("output", {})
                task_id, provider_status = output.get("task_id", ""), output.get("task_status")
                if not isinstance(task_id, str) or not ID.fullmatch(task_id) or secret in task_id or provider_status not in REMOTE_STATUSES:
                    raise ValueError("Invalid task receipt")
                row.update(status="submitted", remote_task_id=task_id, provider_status=provider_status,
                    submitted_at=utc_now(), provider_request_id=self.safe_identifier(result.get("request_id"), secret))
                row.pop("error", None)
                if provider_status == "RUNNING":
                    row["status"] = "running"
                elif provider_status == "SUCCEEDED" and output.get("video_url"):
                    url = safe_https(output["video_url"])
                    if secret in url:
                        raise ValueError("Unsafe output URL")
                    row.update(status="succeeded", output_video_url=url)
                elif provider_status in {"FAILED", "CANCELED", "UNKNOWN"}:
                    row["status"] = {"FAILED": "failed", "CANCELED": "canceled", "UNKNOWN": "provider_unknown"}[provider_status]
                    row["error"] = {"code": provider_status, "message": "服务端报告任务%s；不自动重试生成。" % provider_status}
                self.event(state, job_id, "submit", "complete", "已收到真实任务编号；等待查询。")
            except Exception as exc:
                # A parsed task ID is still useful if another field is malformed.
                # Keep that ID for reconciliation but never attempt another POST.
                row["status"] = "submission_unknown"
                row["error"] = {"code": "submission_receipt_unknown", "message": "提交回执不明（%s）；预算保留，须人工对账，不自动重试。" % type(exc).__name__}
                self.event(state, job_id, "submit", "failed", row["error"]["message"])
            self.persist(state, job_id)
            return dict(row)

    @staticmethod
    def safe_identifier(value, secret):
        return value if isinstance(value, str) and ID.fullmatch(value) and secret not in value else None

    def poll(self, job_id):
        with self.locked_state() as state:
            self.execution_gate(job_id, state)
            row = self.row(state, job_id)
            task_id = row.get("remote_task_id")
            if not task_id or not ID.fullmatch(task_id):
                raise WorkflowError("task_id_required", "尚无真实任务编号；提交不明时须人工对账。", 409)
            if row.get("provider_status") in {"FAILED", "CANCELED"} or (row.get("provider_status") == "SUCCEEDED" and row.get("output_video_url")):
                return dict(row)
            # UI refreshes do not translate into duplicate remote queries.
            last_attempt = row.get("last_poll_attempt_at")
            if last_attempt:
                try:
                    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(last_attempt)).total_seconds()
                    if elapsed < 15:
                        return dict(row)
                except ValueError:
                    pass
            secret = self.secret_loader()
            row["last_poll_attempt_at"] = utc_now()
            self.persist(state, job_id)
            try:
                status, raw = self.transport.request("GET", API_BASE + "/api/v1/tasks/" + parse.quote(task_id, safe=""),
                    {"Authorization": "Bearer " + secret})
                if status != 200:
                    raise ValueError("Task query HTTP failure")
                result = json.loads(raw.decode("utf-8"))
                output = result.get("output", {})
                provider_status = output.get("task_status")
                if output.get("task_id") != task_id or provider_status not in REMOTE_STATUSES:
                    raise ValueError("Unexpected task response")
                updates = {"provider_status": provider_status, "last_polled_at": utc_now(),
                    "status": {"PENDING": "submitted", "RUNNING": "running", "SUCCEEDED": "succeeded",
                               "FAILED": "failed", "CANCELED": "canceled", "UNKNOWN": "provider_unknown"}[provider_status]}
                if provider_status == "SUCCEEDED":
                    url = safe_https(output.get("video_url"))
                    if secret in url:
                        raise ValueError("Unsafe output URL")
                    updates["output_video_url"] = url
                row.update(updates)
                row.pop("error", None)
                if provider_status in {"FAILED", "CANCELED", "UNKNOWN"}:
                    # Codes only; arbitrary upstream message text can echo input credentials.
                    row["error"] = {"code": self.safe_identifier(output.get("code"), secret) or provider_status,
                        "message": "服务端报告任务%s；不自动重试生成。" % provider_status}
                self.event(state, job_id, "poll", provider_status.lower(), "真实服务端状态：" + provider_status)
            except Exception as exc:
                row["error"] = {"code": "poll_failed", "message": "查询未完成（%s）；保留已有任务，不重建任务。" % type(exc).__name__}
                self.event(state, job_id, "poll", "failed", row["error"]["message"])
            self.persist(state, job_id)
            return dict(row)

    def result_path(self, job_id, row=None):
        row = row or self.state()["jobs"].get(job_id, {})
        receipt = row.get("download", {})
        target = self.output_dir / job_id / "candidate.mp4"
        if (not receipt or receipt.get("path") != str(target) or not target.is_file()
                or digest(target.read_bytes()) != receipt.get("sha256")):
            return None
        return target

    def download(self, job_id):
        with self.locked_state() as state:
            self.execution_gate(job_id, state)
            row = self.row(state, job_id)
            if row.get("provider_status") != "SUCCEEDED" or not row.get("output_video_url"):
                raise WorkflowError("result_not_ready", "只有服务端SUCCEEDED且返回视频地址后才能收件。", 409)
            if self.result_path(job_id, row):
                return dict(row)
            target = self.output_dir / job_id / "candidate.mp4"
            if target.exists():
                raise WorkflowError("result_conflict", "结果文件已有未匹配收据；拒绝覆盖。", 409)
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=".candidate.", suffix=".part", dir=str(target.parent))
            temporary = Path(name)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    self.transport.download(safe_https(row["output_video_url"]), stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                if not temporary.stat().st_size:
                    raise ValueError("Empty result")
                receipt = {"path": str(target), "sha256": digest(temporary.read_bytes()),
                    "bytes": temporary.stat().st_size, "at": utc_now()}
                os.replace(str(temporary), str(target))
                row["download"] = receipt
                row.pop("error", None)
                self.event(state, job_id, "download", "complete", "生成候选已收件；尚未通过内容验收。")
            except Exception as exc:
                row["error"] = {"code": "download_failed", "message": "视频收件失败（%s）；可查询原任务后重试收件。" % type(exc).__name__}
                self.event(state, job_id, "download", "failed", row["error"]["message"])
            finally:
                if temporary.exists():
                    temporary.unlink()
            self.persist(state, job_id)
            return dict(row)

    def qc(self, job_id, ffprobe=None, ffmpeg=None):
        with self.locked_state() as state:
            row = self.row(state, job_id)
            path = self.result_path(job_id, row)
            if not path:
                raise WorkflowError("local_result_required", "需先收件并核对结果指纹。", 409)
            report = technical_qc(path, ffprobe=ffprobe, ffmpeg=ffmpeg)
            row["qc"] = report
            atomic_json(path.parent / "qc.json", report)
            self.event(state, job_id, "qc", report["technical_status"], "技术检查已记录；画面运动、地貌和标签仍待复核。")
            self.persist(state, job_id)
            return report


def technical_qc(path, ffprobe=None, ffmpeg=None):
    report = {"at": utc_now(), "technical_status": "pending", "duration_sec": None,
        "width": None, "height": None, "decode_status": "not_checked", "audio_stream_count": None,
        "physical_quality_status": "pending_human_or_assistant_review", "accepted_label": None,
        "training_ready": False, "checks": [], "limitations": [
            "尺寸、时长和解码通过不代表固定镜头、水流方向、地貌稳定或N语义通过。"]}
    probe, decoder = ffprobe or shutil.which("ffprobe"), ffmpeg or shutil.which("ffmpeg")
    if not probe or not decoder:
        report["checks"].append({"name": "tools", "status": "pending", "detail": "需本地ffprobe及ffmpeg后才能完成技术检查。"})
        return report
    try:
        run = subprocess.run([str(probe), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=True)
        info = json.loads(run.stdout.decode("utf-8"))
        videos = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if len(videos) != 1:
            raise ValueError("Expected one video stream")
        video = videos[0]
        duration = float(video.get("duration", info.get("format", {}).get("duration")))
        if not math.isfinite(duration):
            raise ValueError("Nonfinite duration")
        width, height = int(video["width"]), int(video["height"])
        audio_count = sum(s.get("codec_type") == "audio" for s in info.get("streams", []))
        report.update(duration_sec=duration, width=width, height=height, audio_stream_count=audio_count)
        for name, passed, detail in (("duration", abs(duration - 10) <= 0.5, "预期10秒，容差0.5秒。"),
                ("dimensions", 1200 <= width <= 1400 and 650 <= height <= 800, "本参考为16:9，预期约1280×720。"),
                ("silent", audio_count == 0, "预期无音轨。")):
            report["checks"].append({"name": name, "status": "passed" if passed else "failed", "detail": detail})
        result = subprocess.run([str(decoder), "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        decoded = result.returncode == 0 and not result.stderr.strip()
        report["decode_status"] = "passed" if decoded else "failed"
        report["checks"].append({"name": "decode", "status": report["decode_status"], "detail": "完整视频解码检查。"})
        report["technical_status"] = "passed" if all(c["status"] == "passed" for c in report["checks"]) else "failed"
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
        report["technical_status"] = "failed"
        report["checks"].append({"name": "probe", "status": "failed", "detail": "本地媒体探测或解码失败。"})
    return report
