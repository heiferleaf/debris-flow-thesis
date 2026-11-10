"""Local dry-scene gallery and one explicitly confirmed paid video preview.

python experiments/tools/synthetic_video_workbench_server.py
The gallery is http://127.0.0.1:8772/; single-job creation is /generate.
Opening either page does not decrypt credentials or contact the provider.
The new water-bearing first frame has its own manifest, state and consent scope.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import secrets
import sys
import threading
import time
from urllib import parse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video.synthetic_wan import (
    PROFILE, WanWorkflow, WorkflowError, atomic_json, credential_configured, digest, utc_now,
)

DEFAULT_DIRECTORY = ROOT / "data/metadata/synthetic-video/preview-wan26-flood-20261108-v1"
DEFAULT_MANIFEST = DEFAULT_DIRECTORY / "manifest.json"
DEFAULT_STATE = ROOT / "data/process/synthetic-video/jobs/wan26-flood-preview-workbench-state.json"
CONTROLLED_MANIFEST = ROOT / "data/metadata/synthetic-video/preview-wan26-water-controlled-20261108-v1/manifest.json"
CONTROLLED_STATE = ROOT / "data/process/synthetic-video/jobs/wan26-controlled-water-workbench-state.json"
ORDINARY_MANIFEST = ROOT / "data/metadata/synthetic-video/preview-wan26-ordinary-water-20261108-v4/manifest.json"
ORDINARY_STATE = ROOT / "data/process/synthetic-video/jobs/wan26-ordinary-water-workbench-state-v4.json"
ORDINARY_INVENTORY = ROOT / "data/metadata/synthetic-video/ordinary-water-20261108-v4/source-inventory.json"
ACCEPTED_DRY_INVENTORY = ROOT / "data/metadata/synthetic-video/dry-scenes-20261108-v2/source-inventory.json"
ASSET_DIRECTORY = ROOT / "experiments/tools/synthetic_video_workbench"
JOB_ROUTE = re.compile(r"^/api/jobs/([A-Za-z0-9_-]{1,160})(?:/(plan|submit|poll|download|qc|prepare-confirmation|confirm-submit))?$")
DRY_INVENTORY = ROOT / "data/metadata/synthetic-video/dry-scenes-20261108-v1/source-inventory.json"
DRY_DIRECTORY = ROOT / "data/process/synthetic-video/dry-scenes-20261108-v1"
ACCEPTANCE_RELATIVE = Path("data/metadata/synthetic-video/dry-scenes-20261108-v1/user-acceptance.json")


def acceptance_overrides(root):
    path = Path(root) / ACCEPTANCE_RELATIVE
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    return {item["path"]: item for item in document.get("items", [])}


class DrySceneGallery:
    """Filesystem-backed snapshots; no inference, credential or network clients."""
    readonly_gallery = True

    def __init__(self, inventory=DRY_INVENTORY, *, root=ROOT, assets=ASSET_DIRECTORY, spatial=None):
        self.root, self.inventory, self.assets = Path(root), Path(inventory), Path(assets)
        self.spatial = Path(spatial) if spatial else self.inventory.parent / "spatial-controls.json"

    @staticmethod
    def read(path):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            return {}

    def registered(self, value):
        return self.root / value if value else None

    def controls(self):
        document = self.read(self.spatial)
        rows = document.get("items", document.get("scenes", []))
        if isinstance(rows, dict):
            return rows
        return {row["scene_id"]: row for row in rows}

    def snapshot(self):
        controls, scenes = self.controls(), []
        acceptance = acceptance_overrides(self.root)
        acceptance.update({item['path']: item for item in
                           self.read(self.inventory.parent / 'user-acceptance.json').get('items', [])})
        for item in self.read(self.inventory).get("items", []):
            scene_id = item["scene_id"]
            source, generated = self.registered(item["source_path"]), self.registered(item["output_path"])
            metadata = self.read(self.registered(item["metadata_path"]))
            record = self.read(generated.parent / "static-10s.json")
            video = generated.parent / "static-10s.mp4"
            first_frame = generated.parent / "ordinary-water-first-frame.png"
            available = generated.is_file()
            current_hash = digest(generated.read_bytes()) if available else None
            video_current = (video.is_file() and current_hash and record.get("source_sha256") == current_hash)
            needs_revision = (metadata.get("fluid_suspected_in_channel") is True
                              or metadata.get("needs_revision") is True
                              or any(word in metadata.get("status", "").lower() for word in ("reject", "revision", "failed")))
            image_status = "needs_revision" if needs_revision else "complete" if available else "pending"
            user_acceptance = acceptance.get(item["output_path"], {})
            if user_acceptance.get("status") == "rejected_by_user":
                image_status = "rejected_by_user"
            scenes.append({**item, "metadata": metadata, "spatial": controls.get(scene_id, {}),
                "user_acceptance": user_acceptance,
                "first_frame_acceptance": acceptance.get(first_frame.relative_to(self.root).as_posix(), {}),
                "image_status": image_status, "image_available": available,
                "source_url": "/media/source/" + scene_id if source.is_file() else None,
                "image_url": "/media/dry/" + scene_id + "?v=" + current_hash[:12] if available else None,
                "video_url": "/media/static/" + scene_id + "?v=" + record.get("video_sha256", "")[:12] if video_current else None,
                "video_status": "complete" if video_current else "pending", "video_record": record if video_current else {},
                "first_frame_url": "/media/water-first-frame/" + scene_id if first_frame.is_file() else None,
                "training_ready": False})
        return {"schema_version": 2, "mode": "local_dry_scene_gallery", "updated_at": utc_now(),
            "batch": self.inventory.parent.name,
            "paid_submission_enabled": False, "paid_status_zh": "动态试片：待空间区域和水流首帧准备",
            "video_generation_fee_cny": 0, "physical_motion": False, "unique_source_images_per_video": 1,
            "summary": {"expected": len(scenes), "generated": sum(s["image_available"] for s in scenes),
                "needs_revision": sum(s["image_status"] in {"needs_revision", "rejected_by_user"} for s in scenes),
                "rejected_by_user": sum(s["image_status"] == "rejected_by_user" for s in scenes),
                "accepted_by_user": sum(s["user_acceptance"].get("status") == "accepted_by_user" for s in scenes),
                "pending": sum(s["image_status"] == "pending" for s in scenes),
                "video_complete": sum(s["video_status"] == "complete" for s in scenes)},
            "scenes": scenes, "training_ready": False, "poll_interval_seconds": 8,
            "legacy_previews": [{"id": identity, "title": title, "url": "/media/demo/" + identity}
                for identity, title in (("previous-water", "上一条普通水流试片"), ("photometric", "零费用成像变化代码演示"))
                if self.media("demo", identity)]}

    def media(self, kind, identity):
        for item in self.read(self.inventory).get("items", []):
            if item["scene_id"] == identity:
                generated = self.registered(item["output_path"])
                candidates = {"source": self.registered(item["source_path"]), "dry": generated,
                              "static": generated.parent / "static-10s.mp4",
                              "water-first-frame": generated.parent / "ordinary-water-first-frame.png"}
                path = candidates.get(kind)
                return path if path and path.is_file() else None
        if kind == "demo":
            registered = self.read(DEFAULT_MANIFEST).get("local_previews", {}).get(identity)
            if registered:
                path = self.registered(registered["path"])
                if path.is_file() and digest(path.read_bytes()) == registered["sha256"]:
                    return path
        return None


class OrdinaryWaterGallery(DrySceneGallery):
    """Hot-read independent water candidates; dry acceptance never transfers."""

    def snapshot(self):
        document, scenes = self.read(self.inventory), []
        acceptance = {row["path"]: row for row in
                      self.read(self.inventory.parent / "user-acceptance.json").get("items", [])}
        for item in document.get("items", []):
            generated = self.registered(item["output_path"])
            metadata = self.read(self.registered(item["metadata_path"]))
            available = generated.is_file()
            actual_hash = digest(generated.read_bytes()) if available else None
            accepted = acceptance.get(item["output_path"], {})
            status = ("rejected_by_user" if accepted.get("status") == "rejected_by_user" else
                      "accepted_by_user" if accepted.get("status") == "accepted_by_user" else
                      "complete_pending_user_review" if available else "pending")
            image_url = "/media/ordinary-water/" + item["scene_id"] + "?v=" + actual_hash[:12] if available else None
            scenes.append({**item, "metadata": metadata, "image_status": status,
                "image_available": available, "current_sha256": actual_hash,
                "image_url": image_url, "first_frame_url": image_url,
                "user_acceptance": accepted,
                "source_url": "/media/ordinary-water-source/" + item["scene_id"]
                    if self.registered(item["source_path"]).is_file() else None,
                "dry_url": "/media/ordinary-water-dry/" + item["scene_id"]
                    if self.registered(item.get("dry_path")).is_file() else None,
                "training_ready": False})
        return {"schema_version": 1, "batch": self.inventory.parent.name,
            "mode": "ordinary_water_reference_gallery", "updated_at": utc_now(),
            "acceptance_scope": "Reference content for media previews only; no N ground truth or training release.",
            "summary": {"expected": len(scenes), "generated": sum(row["image_available"] for row in scenes),
                "pending": sum(not row["image_available"] for row in scenes),
                "accepted_by_user": sum(row["image_status"] == "accepted_by_user" for row in scenes),
                "rejected_by_user": sum(row["image_status"] == "rejected_by_user" for row in scenes)},
            "scenes": scenes, "training_ready": False}

    def media(self, kind, identity):
        field = {"ordinary-water": "output_path", "ordinary-water-source": "source_path",
                 "ordinary-water-dry": "dry_path"}.get(kind)
        if not field:
            return None
        for item in self.read(self.inventory).get("items", []):
            if item["scene_id"] == identity:
                path = self.registered(item.get(field))
                return path if path and path.is_file() else None
        return None


class Workbench:
    def __init__(self, workflow, *, assets=ASSET_DIRECTORY, ffprobe=None, ffmpeg=None):
        self.workflow, self.assets = workflow, Path(assets).resolve()
        self.ffprobe, self.ffmpeg = ffprobe, ffmpeg
        self.mutex = threading.RLock()
        self.confirmations = {}

    def reference_acceptance(self, job_id):
        reference = self.workflow.job(job_id)["selected_reference"]["path"]
        overrides = acceptance_overrides(self.workflow.root)
        local_record = self.workflow.manifest_path.parent / "reference-acceptance.json"
        if local_record.is_file():
            document = json.loads(local_record.read_text(encoding="utf-8-sig"))
            overrides.update({item["path"]: item for item in document.get("items", [])})
        return overrides.get(reference, {})

    def reference_rejection_message(self, job_id):
        return self.reference_acceptance(job_id).get("reason_zh") or "当前参考图验收不通过，暂停本条付费生成。"

    def require_reference_not_rejected(self, job_id):
        status = self.reference_acceptance(job_id).get("status")
        if status == "paused_for_task_match_review":
            raise WorkflowError("reference_task_match_review_pending", self.reference_rejection_message(job_id), 409)
        if status == "rejected_by_user":
            raise WorkflowError("reference_rejected_by_user", self.reference_rejection_message(job_id), 409)

    def consent_path(self, job_id, fingerprint):
        binding = digest(json.dumps([self.workflow.manifest_sha256, str(self.workflow.state_path)],
                                    separators=(",", ":")).encode("utf-8"))[:16]
        return self.workflow.state_path.parent / "workbench-consent" / (job_id + "--" + binding + "--" + fingerprint + ".json")

    def workflow_for(self, job_id, state=None):
        """Reuse only an existing request-bound user receipt, never infer consent."""
        workflow = self.workflow
        if workflow.execute:
            return workflow
        try:
            _, fingerprint = workflow.request_identity(job_id, state)
        except WorkflowError:
            return workflow
        receipt = self.consent_path(job_id, fingerprint)
        if not receipt.is_file():
            return workflow
        candidate = WanWorkflow(workflow.manifest_path, workflow.state_path, root=workflow.root,
            output_dir=workflow.output_dir, execute=True, authorization_path=receipt,
            transport=workflow.transport, secret_loader=workflow.secret_loader)
        try:
            candidate.authorization(job_id, state)
        except WorkflowError:
            return workflow
        return candidate

    def confirmation_blockers(self, job_id, state):
        blockers = []
        if self.reference_acceptance(job_id).get("status") in {"rejected_by_user", "paused_for_task_match_review"}:
            blockers.append(self.reference_rejection_message(job_id))
        try:
            self.workflow.request_identity(job_id, state)
        except WorkflowError as exc:
            blockers.append(str(exc))
        row = self.workflow.draft(job_id, state)
        if row.get("status", "draft") != "draft" or row.get("reserved_cost_cents", 0):
            blockers.append("本条已有提交或费用预留记录，不能重复生成。")
        if not credential_configured():
            blockers.append("请先配置百炼密钥。")
        return blockers

    def prepare_confirmation(self, job_id):
        """Build a short-lived local review; no key loading or provider calls."""
        with self.mutex:
            self.require_reference_not_rejected(job_id)
            state = self.workflow.state()
            blockers = self.confirmation_blockers(job_id, state)
            if blockers:
                raise WorkflowError("confirmation_unavailable", "；".join(blockers), 409)
            identity, fingerprint = self.workflow.request_identity(job_id, state)
            job = self.workflow.draft(job_id, state)
            token = secrets.token_urlsafe(32)
            expires_at = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
            self.confirmations[job_id] = {"token": token, "request_sha256": fingerprint,
                "deadline": time.monotonic() + 600, "expires_at": expires_at}
            confirmation = {"token": token, "expires_at": expires_at, "job_id": job_id,
                "request_sha256": fingerprint, "reference_sha256": identity["reference_sha256"],
                "reference_url": "/media/reference/" + self.workflow.job(job_id)["scenario_id"],
                "prompt_zh": job["prompt_zh"], "prompt_en": job["prompt_en"],
                "model": PROFILE["model"], "duration": 10, "resolution": "720P", "audio": False,
                "estimated_cny": 1.5, "max_cost_cny": 1.5}
            return {"confirmation": confirmation, "workbench": self.snapshot()}

    def confirm_submission(self, job_id, payload):
        """The explicit final UI action binds one upload and at most CNY1.50."""
        with self.mutex:
            self.require_reference_not_rejected(job_id)
            expected = {"confirmation_token", "request_sha256", "accepted_upload", "max_cost_cny"}
            if (set(payload) != expected or payload.get("accepted_upload") is not True
                    or type(payload.get("max_cost_cny")) not in (int, float)
                    or payload["max_cost_cny"] != 1.5):
                raise WorkflowError("explicit_confirmation_required", "请明确同意本条参考图上传及1.50元费用上限。", 403)
            challenge = self.confirmations.get(job_id)
            token = payload.get("confirmation_token")
            if (not challenge or not isinstance(token, str)
                    or not secrets.compare_digest(token, challenge["token"])
                    or time.monotonic() > challenge["deadline"]):
                raise WorkflowError("confirmation_expired", "确认已失效，请重新打开生成确认窗口。", 409)
            state = self.workflow.state()
            blockers = self.confirmation_blockers(job_id, state)
            if blockers:
                raise WorkflowError("confirmation_unavailable", "；".join(blockers), 409)
            _, fingerprint = self.workflow.request_identity(job_id, state)
            if fingerprint != challenge["request_sha256"] or payload.get("request_sha256") != fingerprint:
                raise WorkflowError("request_changed", "提示词或参考图已变化，请重新确认。", 409)
            receipt = {"approved": True, "provider": "dashscope", "model": PROFILE["model"],
                "authorization_id": "workbench-user-" + secrets.token_hex(12),
                "approved_by": "explicit_local_workbench_user_action", "approved_at": utc_now(),
                "approval_method": "final_UI_upload_and_cost_confirmation",
                "manifest_sha256": self.workflow.manifest_sha256,
                "state_path": str(self.workflow.state_path), "allowed_job_ids": [job_id], "max_jobs": 1,
                "request_sha256_by_job": {job_id: fingerprint}, "max_cost_cny": "1.50",
                "automatic_recharge": False, "automatic_retry": False,
                "reference_upload_accepted": True}
            receipt_path = self.consent_path(job_id, fingerprint)
            if not receipt_path.exists():
                atomic_json(receipt_path, receipt)
            del self.confirmations[job_id]
            workflow = self.workflow_for(job_id, state)
            workflow.submit(job_id)
            return {"workbench": self.snapshot()}

    def snapshot(self):
        workflow = self.workflow
        with self.mutex:
            state = workflow.state()
            scenarios = []
            for source in workflow.manifest["scenarios"]:
                path = (workflow.root / source["reference_path"]).resolve()
                available = path.is_file() and digest(path.read_bytes()) == source["reference_sha256"]
                scenarios.append({**{k: v for k, v in source.items() if k not in {"reference_path", "reference_sha256"}},
                    "reference_url": "/media/reference/" + source["scenario_id"] if available else None,
                    "reference_available": available})
            jobs = []
            for job_id, source in workflow.jobs.items():
                row = workflow.draft(job_id, state)
                plan = self.workflow_for(job_id, state).plan(job_id, state)
                confirmation_blockers = self.confirmation_blockers(job_id, state)
                local_video = workflow.result_path(job_id, row)
                public = {"id": job_id, "scenario_id": source["scenario_id"], "title": source["title"],
                    "reference_acceptance": self.reference_acceptance(job_id),
                    "prompt_zh": row["prompt_zh"], "prompt_en": row["prompt_en"], "settings": dict(PROFILE),
                    "status": row.get("status", "draft"), "provider_status": row.get("provider_status"),
                    "remote_task_id": row.get("remote_task_id"), "local_video_url": "/media/result/" + job_id if local_video else None,
                    "progress": None, "error": row.get("error"), "cost_reserved_cny": row.get("reserved_cost_cents", 0) / 100,
                    "estimated_cny": 1.5, "authorization_status": plan["authorization_status"],
                    "submission_allowed": plan["submission_allowed"], "submission_blockers": plan["blockers"],
                    "user_confirmation_available": not confirmation_blockers,
                    "user_confirmation_blockers": confirmation_blockers,
                    "download_allowed": row.get("provider_status") == "SUCCEEDED" and bool(row.get("output_video_url")) and not local_video,
                    "poll_allowed": bool(row.get("remote_task_id")) and (row.get("provider_status") not in {"SUCCEEDED", "FAILED", "CANCELED"}
                        or row.get("provider_status") == "SUCCEEDED" and not row.get("output_video_url")),
                    "qc_allowed": bool(local_video), "qc": row.get("qc"), "last_polled_at": row.get("last_polled_at"),
                    "accepted_label": None, "training_ready": False, "preview_only": True,
                    "last_plan": row.get("last_plan")}
                public["stages"] = self.stages(public, plan)
                if public["reference_acceptance"].get("status") in {"rejected_by_user", "paused_for_task_match_review"}:
                    public["submission_allowed"] = False
                    public["submission_blockers"] = confirmation_blockers
                    public["stages"][0].update(status="blocked", detail=self.reference_rejection_message(job_id))
                jobs.append(public)
            first_id = next(iter(workflow.jobs))
            first_workflow = self.workflow_for(first_id, state)
            first_plan = first_workflow.plan(first_id, state)
            return {"schema_version": 1, "provider": {"id": "dashscope", "model": PROFILE["model"],
                "resolution": "720P", "duration": 10, "audio": False, "estimated_cny": 1.5,
                "region": "cn-beijing", "price_cny_per_second": 0.15, "free_quota": 0,
                "contract_source": "https://help.aliyun.com/zh/model-studio/legacy-image-to-video-api-reference/"},
                "credential_configured": credential_configured(), "scenarios": scenarios, "jobs": jobs,
                "configuration": {"execution_enabled": first_workflow.execute, "user_confirmation_enabled": True,
                    "authorization_status": first_plan["authorization_status"], "authorization_message": first_plan["authorization_message"],
                    "key_setup_command": "python experiments/tools/configure_wan_key.py",
                    "authorization_template": "data/metadata/synthetic-video/preview-wan26-20261108-v1/authorization-template.json",
                    "poll_interval_seconds": 15},
                "activity": state.get("activity", [])[-100:], "training_ready": False,
                "mode": "authorized_execution" if first_workflow.execute else "awaiting_user_confirmation"}

    @staticmethod
    def stages(job, plan):
        status, remote = job["status"], job["provider_status"]
        submitted = bool(job["remote_task_id"])
        technical = (job.get("qc") or {}).get("technical_status")
        stages = [
            ("reference", "参考图", "ready" if plan["reference_sha256"] else "blocked", "本条媒体预览参考图；训练标签另行验收；" +
                ("已用于本次提交。" if submitted else "提交回执待核实。" if status == "submission_unknown" else "未发送到外部。")),
            ("prompt", "提示词 / 参数", "complete" if job.get("last_plan") else "ready", "中英文可编辑；实际发送英文，720P / 10秒 / 无声。"),
            ("authorization", "独立授权", "complete" if plan["budget_authorized"] else "blocked", plan["authorization_message"]),
            ("submit", "提交", "complete" if submitted else "failed" if status == "submission_unknown" else "pending",
                "真实任务：" + job["remote_task_id"] if submitted else "回执不明，须人工对账；不重提。" if status == "submission_unknown" else "尚未提交，预计1.50元。"),
            ("generation", "生成", "complete" if remote == "SUCCEEDED" else "failed" if remote in {"FAILED", "CANCELED", "UNKNOWN"} else "active" if submitted else "pending",
                "真实服务端状态：" + remote if remote else "尚未开始；服务未提供百分比进度。"),
            ("qc", "质量检查", "failed" if technical == "failed" else "ready" if technical == "passed" else "active" if job["local_video_url"] else "pending",
                "技术检查：" + technical + "；物理质量及标签待复核。" if technical else "收件后检查时长、尺寸和解码；内容需另行复核。"),
            ("result", "结果", "ready" if job["local_video_url"] else "pending", "候选已收件，未验收、不用于训练。" if job["local_video_url"] else "尚无生成结果。"),
        ]
        return [{"id": key, "label": label, "status": stage_status, "detail": detail} for key, label, stage_status, detail in stages]

    def action(self, job_id, action, payload):
        if action == "submit":
            self.require_reference_not_rejected(job_id)
        if action == "prepare-confirmation":
            if payload:
                raise WorkflowError("body_fields_forbidden", "打开确认窗口无需参数。")
            return self.prepare_confirmation(job_id)
        if action == "confirm-submit":
            return self.confirm_submission(job_id, payload)
        if payload:
            raise WorkflowError("body_fields_forbidden", "该操作不接受额外参数。")
        with self.mutex:
            workflow = self.workflow_for(job_id)
            if action == "plan":
                return {"plan": workflow.save_plan(job_id), "workbench": self.snapshot()}
            if action == "qc":
                self.workflow.qc(job_id, ffprobe=self.ffprobe, ffmpeg=self.ffmpeg)
            elif action in {"submit", "poll", "download"}:
                getattr(workflow, action)(job_id)
            else:
                raise WorkflowError("action_not_found", "未注册的操作。", 404)
            return {"workbench": self.snapshot()}

    def media(self, kind, identity):
        if kind == "reference":
            for scenario in self.workflow.manifest["scenarios"]:
                if scenario["scenario_id"] == identity:
                    path = (self.workflow.root / scenario["reference_path"]).resolve()
                    if path.is_file() and digest(path.read_bytes()) == scenario["reference_sha256"]:
                        return path
        elif kind == "result" and identity in self.workflow.jobs:
            return self.workflow.result_path(identity)
        elif kind == "demo":
            registered = self.workflow.manifest.get("local_previews", {}).get(identity)
            if registered:
                path = (self.workflow.root / registered["path"]).resolve()
                allowed = (self.workflow.root / "data/process/synthetic-video/jobs",
                           self.workflow.root / ".codex/docs/.temp/photometric-demo-20261108-v1")
                if (any(base.resolve() in path.parents for base in allowed)
                        and path.is_file() and digest(path.read_bytes()) == registered["sha256"]):
                    return path
        return None


def handler_for(workbench, generation=None, water=None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "LocalVideoWorkbench/1"

        def log_message(self, format, *args):
            return  # Never log untrusted request paths/bodies or credentials.

        def json(self, value, status=200):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.common_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def common_headers(self):
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; media-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.send_header("Referrer-Policy", "no-referrer")

        def local_request(self, mutation=False):
            host = self.headers.get("Host", "")
            allowed = {"127.0.0.1:%d" % self.server.server_port, "localhost:%d" % self.server.server_port}
            if host not in allowed:
                raise WorkflowError("invalid_host", "仅接受本机工作台地址。", 403)
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + host:
                raise WorkflowError("invalid_origin", "仅接受工作台同源请求。", 403)
            if mutation and self.headers.get_content_type() != "application/json":
                raise WorkflowError("json_required", "修改与操作请求必须使用application/json。", 415)

        def payload(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > 16384:
                    raise ValueError()
                value = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                if not isinstance(value, dict):
                    raise ValueError()
                return value
            except (ValueError, UnicodeDecodeError):
                raise WorkflowError("invalid_json", "请求须为不超过16KiB的JSON对象。") from None

        def guarded(self, callback):
            try:
                self.local_request(self.command in {"POST", "PATCH"})
                callback()
            except WorkflowError as exc:
                self.json({"error": {"code": exc.code, "message": str(exc)}}, exc.status)
            except Exception:
                self.json({"error": {"code": "local_operation_failed", "message": "本地操作未完成；请核对状态、日志及注册文件。"}}, 500)

        def do_GET(self):
            self.guarded(self.get_route)

        def get_route(self):
            path = parse.urlsplit(self.path).path
            if path == "/api/workbench":
                snapshot = workbench.snapshot()
                if water:
                    snapshot["ordinary_water"] = water.snapshot()
                    water_by_id = {row["scene_id"]: row for row in snapshot["ordinary_water"]["scenes"]}
                    for scene in snapshot.get("scenes", []):
                        candidate = water_by_id.get(scene["scene_id"], {})
                        scene["first_frame_url"] = candidate.get("first_frame_url")
                        scene["first_frame_acceptance"] = candidate.get("user_acceptance", {})
                if generation:
                    current = generation.snapshot()["jobs"][0]
                    snapshot["generation"] = {"url": "/generate", "title": current["title"],
                        "status": current["status"], "confirmation_available": current["user_confirmation_available"]}
                self.json(snapshot)
                return
            if path == "/api/generation" and generation:
                self.json(generation.snapshot())
                return
            assets = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css",
                      "/generate": "generate.html", "/generate.html": "generate.html", "/generate.js": "generate.js"}
            target = workbench.assets / assets[path] if path in assets else None
            match = re.fullmatch(r"/media/(reference|result|demo|source|dry|static|water-first-frame|ordinary-water|ordinary-water-source|ordinary-water-dry)/([A-Za-z0-9_-]{1,160})", path)
            if match:
                kind, identity = match.groups()
                target = (water.media(kind, identity) if water and kind.startswith("ordinary-water") else
                          generation.media(kind, identity) if generation and kind in {"reference", "result"} else
                          workbench.media(kind, identity))
            if not target or not target.is_file():
                raise WorkflowError("not_found", "未注册或尚不存在的本地资源。", 404)
            self.send_file(target)

        def send_file(self, path):
            size = path.stat().st_size
            start, end = 0, size - 1
            requested = self.headers.get("Range")
            if requested:
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
                if not match:
                    self.send_response(416)
                    self.send_header("Content-Range", "bytes */%d" % size)
                    self.end_headers()
                    return
                start, value = match.groups()
                start, end = int(start), min(int(value) if value else size - 1, size - 1)
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", "bytes */%d" % size)
                    self.end_headers()
                    return
            self.send_response(206 if requested else 200)
            self.common_headers()
            self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            if requested:
                self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
            self.end_headers()
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def do_PATCH(self):
            self.guarded(self.patch_route)

        def patch_route(self):
            active = generation or workbench
            if getattr(active, "readonly_gallery", False):
                raise WorkflowError("paid_generation_locked", "动态试片：待空间区域和水流首帧准备。当前仅提供本地只读画廊。", 403)
            match = JOB_ROUTE.fullmatch(parse.urlsplit(self.path).path)
            if not match or match.group(2):
                raise WorkflowError("not_found", "未注册的草稿接口。", 404)
            with active.mutex:
                active.workflow.edit(match.group(1), self.payload())
                active.confirmations.pop(match.group(1), None)
                self.json(active.snapshot())

        def do_POST(self):
            self.guarded(self.post_route)

        def post_route(self):
            active = generation or workbench
            if getattr(active, "readonly_gallery", False):
                raise WorkflowError("paid_generation_locked", "动态试片：待空间区域和水流首帧准备。所有付费提交接口已停用。", 403)
            match = JOB_ROUTE.fullmatch(parse.urlsplit(self.path).path)
            if not match or not match.group(2):
                raise WorkflowError("not_found", "未注册的任务操作接口。", 404)
            if (match.group(2) in {"prepare-confirmation", "confirm-submit"}
                    and self.headers.get("Origin") != "http://" + self.headers.get("Host", "")):
                raise WorkflowError("same_origin_confirmation_required", "请从本地工作台打开并确认生成。", 403)
            self.json(active.action(match.group(1), match.group(2), self.payload()))

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=ACCEPTED_DRY_INVENTORY)
    parser.add_argument("--water-inventory", type=Path, default=ORDINARY_INVENTORY)
    parser.add_argument("--spatial", type=Path)
    parser.add_argument("--generation-manifest", type=Path, default=ORDINARY_MANIFEST)
    parser.add_argument("--generation-state", type=Path, default=ORDINARY_STATE)
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--ffprobe")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--self-test", action="store_true", help="Offline snapshot/plan check; starts no server")
    args = parser.parse_args(argv)
    try:
        workbench = DrySceneGallery(args.inventory, spatial=args.spatial)
        water = OrdinaryWaterGallery(args.water_inventory)
        workflow = WanWorkflow(args.generation_manifest, args.generation_state, execute=False)
        generation = Workbench(workflow, ffprobe=args.ffprobe, ffmpeg=args.ffmpeg)
        snapshot = workbench.snapshot()  # Validate files/state before opening the sole port.
        generation_snapshot = generation.snapshot()
        if args.self_test:
            print(json.dumps({"network_requests": 0, "credentials_read": False,
                "summary": snapshot["summary"], "ordinary_water": water.snapshot()["summary"],
                "generation": {"job_id": generation_snapshot["jobs"][0]["id"],
                    "status": generation_snapshot["jobs"][0]["status"],
                    "confirmation_available": generation_snapshot["jobs"][0]["user_confirmation_available"],
                    "submission_allowed_without_final_confirmation": generation_snapshot["jobs"][0]["submission_allowed"]}}, ensure_ascii=False, indent=2))
            return 0
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(workbench, generation, water))
        print("视频生成工作台：http://127.0.0.1:%d/（普通水流参考图 + 单条试片）" % args.port, flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    except (OSError, ValueError) as exc:
        print("工作台未启动：" + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
