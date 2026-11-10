"""Runway Dev execution boundary; stdlib only, no network or key reads on import.

The existing 88-task preparation manifest is accepted by plan(), not upgraded.
Execution requires an independently reviewed manifest with camera_partition =
{status: frozen, split_id: ..., roles: {registered_camera: train|val|test}},
and jobs/tasks containing submit_ready=true, split=train, source_camera,
provider_request={model, ratio, duration: 10, promptText, seed (when supported)},
and selected_reference={path, sha256,
accepted: true, source_camera, split: train, lineage_complete: true,
processing_history_accepted: true, processing_history_id,
parent_lineage: [{source_camera, split, source_unit, source_group,
source_media_sha256}, ...]}. Every ancestor must belong to that train camera.
submission_controls must confirm matched N/P processing history and registered
training ratio/source quotas; see validate_job(). No training-ready or accepted
label is inferred from submission or download.

Formal augmentation_scope=negative_only permits only no_target/ordinary_water/
flood intents, never a P task. It replaces the matched N/P-history requirement
with selected_reference.processing_history_reviewed=true and a nonempty
processing_history_id shared by all accepted references from the same source
(parent source_unit or identical parent media SHA). Frozen train-camera/source
lineage, reference acceptance, registered training ratio and source quotas, and
cost authorization still apply. No scope retains the legacy matched-N/P rule.

Single-clip exception: purpose=media_preview needs exactly one job with
kind=negative_candidate, preview_only=true, training_ready=false, no accepted
label, and no training split claim. It does not need a formal partition, P
control, or training quotas. reference_use must explicitly give external_test
=false, checked_current_frozen_test_list=true, and frozen_test_cameras=[...]
(an empty reviewed snapshot is allowed when no current test split is frozen).
Any supplied partition test role or reference/ancestor test designation still
blocks preview. Accepted image/hash and complete same-camera source ancestry
remain mandatory. A submitted preview records generation_preview camera use:
that camera is not an independent holdout for choices made using this preview.

Authorization JSON: approved=true, provider=runway_dev, authorization_id,
approved_by, approved_at, manifest_sha256, state_path, allowed_job_ids,
max_cost_usd. It records an existing human authorization; this module never signs
or creates one. Use one persistent state file for the authorized budget. A lost
POST receipt reserves its full base cost and requires manual reconciliation,
not a new state file or automatic retry. A stale .lock also needs human review.

Supported narrow profiles (not every option offered by these models):
gen4_turbo: 1280:720, uint32 seed, USD 0.50/10s (low-cost baseline).
gen4.5: 1280:720, uint32 seed, MP4, USD 1.20/10s.
seedance2_5: 1280:720, uint32 seed, audio=false, draft=false, MP4,
USD 3.00/10s. The verified SDK lists this ratio, not an adaptive ratio token.
kling3.0_pro: 1920:1080, audio=false, no seed field, USD 1.20/10s.
One accepted image is supplied as the first frame; no reference video/audio,
last-frame, multishot, upscaling, premium format, or automatic-duration options.
Profile-fixed options may be omitted, but conflicting explicit values fail.
Price assumptions are generation base prices, excluding tax/editing/retries.
Provider schema support does not establish debris-flow physical correctness.

Official contract checked 2026-11-08:
https://github.com/runwayml/sdk-python/blob/main/src/runwayml/resources/image_to_video.py
https://github.com/runwayml/sdk-python/blob/main/src/runwayml/types/image_to_video_create_params.py
https://github.com/runwayml/sdk-python/blob/main/src/runwayml/resources/tasks.py
https://docs.dev.runwayml.com/assets/inputs/
https://docs.dev.runwayml.com/guides/pricing/
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib import error, parse, request


API_BASE = "https://api.dev.runwayml.com"
API_VERSION = "2024-11-06"
DURATION = 10
MODEL_REGISTRY = {
    "gen4_turbo": {"credits_per_second": 5, "ratio": "1280:720", "seed": True,
                   "fixed_fields": {}, "first_frame_array": False, "prompt_limit_utf16": 1000},
    "gen4.5": {"credits_per_second": 12, "ratio": "1280:720", "seed": True,
               "fixed_fields": {"outputFormat": "mp4"}, "first_frame_array": True, "prompt_limit_utf16": 1000},
    "seedance2_5": {"credits_per_second": 30, "ratio": "1280:720", "seed": True,
                    "fixed_fields": {"audio": False, "draft": False, "outputFormat": "mp4"},
                    "first_frame_array": True, "prompt_limit_utf16": 15000},
    "kling3.0_pro": {"credits_per_second": 12, "ratio": "1920:1080", "seed": False,
                     "fixed_fields": {"audio": False}, "first_frame_array": True,
                     "prompt_limit_utf16": 1000},  # Conservative local cap; SDK gives no text-length limit.
}
MAX_DATA_URI_BYTES = 5_000_000  # Includes base64 expansion and data URI prefix.
JPEG_PREFIX = "data:image/jpeg;base64,"
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,200}$")
SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
PROVIDER_STATUSES = {"PENDING", "THROTTLED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"}


def generation_cost_cents(model: str, duration: int = DURATION) -> int:
    if model not in MODEL_REGISTRY or type(duration) is not int or duration != DURATION:
        raise ValueError("Select an explicitly supported model and duration=10")
    # USD 0.01/credit: the number of credits equals the number of USD cents.
    return MODEL_REGISTRY[model]["credits_per_second"] * duration


def usd_string(amount_cents: int) -> str:
    return "%d.%02d" % divmod(amount_cents, 100)


def model_catalog() -> dict:
    return {model: {**profile, "duration": DURATION, "base_cost_usd": usd_string(generation_cost_cents(model)),
                    "usd_per_credit": "0.01", "image_mode": "one accepted first frame",
                    "quality_verified_for_debris_flow": False}
            for model, profile in MODEL_REGISTRY.items()}


def validated_provider_spec(spec: dict) -> dict:
    if not isinstance(spec, dict) or spec.get("model") not in MODEL_REGISTRY:
        raise ValueError("provider_request.model must explicitly select a supported model; no priced default")
    model = spec["model"]
    profile = MODEL_REGISTRY[model]
    generation_cost_cents(model, spec.get("duration"))
    if spec.get("ratio") != profile["ratio"]:
        raise ValueError("Model %s requires ratio=%s in this supported profile" % (model, profile["ratio"]))
    allowed = {"model", "ratio", "duration", "promptText"} | set(profile["fixed_fields"])
    if profile["seed"]:
        allowed.add("seed")
    if set(spec) - allowed:
        raise ValueError("Unsupported fields for %s: %s" % (model, ", ".join(sorted(set(spec) - allowed))))
    for key, expected in profile["fixed_fields"].items():
        if key in spec and (type(spec[key]) is not type(expected) or spec[key] != expected):
            raise ValueError("Model %s requires %s=%s" % (model, key, json.dumps(expected)))
    prompt = spec.get("promptText")
    if (not isinstance(prompt, str) or not prompt.strip()
            or len(prompt.encode("utf-16-le")) // 2 > profile["prompt_limit_utf16"]):
        raise ValueError("promptText exceeds the supported nonempty UTF-16 prompt range")
    body = {key: spec[key] for key in ("model", "ratio", "duration", "promptText")}
    if profile["seed"]:
        if type(spec.get("seed")) is not int or not 0 <= spec["seed"] <= 4294967295:
            raise ValueError("Explicit uint32 seed required for this model; variation_key is not a provider seed")
        body["seed"] = spec["seed"]
    return {**body, **profile["fixed_fields"]}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def cents(value) -> int:
    try:
        number = Decimal(str(value)) * 100
    except InvalidOperation:
        raise ValueError("Cost cap must be a nonnegative USD amount")
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise ValueError("Cost cap must be finite, nonnegative, and in whole cents")
    return int(number)


def local_path(base: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def https_url(value: str) -> str:
    parsed = parse.urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or parsed.fragment):
        raise ValueError("Output must be a HTTPS URL without embedded credentials")
    return value


def load_api_secret() -> str:
    value = os.environ.get("RUNWAYML_API_SECRET", "").strip()
    if not value:
        # This loader is called only after the executed submit/poll local gates.
        from experiments.common.video.local_credentials import load_runway_secret
        try:
            value = load_runway_secret()
        except FileNotFoundError:
            raise ValueError("Run configure_runway_key.py or set RUNWAYML_API_SECRET before an authorized API request") from None
    if "\r" in value or "\n" in value:
        raise ValueError("Invalid API secret")
    return value


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward a Bearer token through a redirect.


class UrllibTransport:
    """No retries. Downloads use a separate, unauthenticated HTTPS request."""

    def __init__(self, timeout: float = 60):
        self.timeout = timeout
        self.opener = request.build_opener(NoRedirect())

    def request(self, method: str, url: str, headers: dict, body: bytes | None = None):
        req = request.Request(url, data=body, headers=headers, method=method)
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                return response.status, response.read()
        except error.HTTPError as response:
            # Do not return server text that might echo request data or credentials.
            return response.code, b""

    def download(self, url: str, stream) -> None:
        req = request.Request(https_url(url), headers={"Accept": "video/*"}, method="GET")
        with self.opener.open(req, timeout=self.timeout) as response:
            if response.status != 200:
                raise ValueError("Download did not return HTTP 200")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def locked_state(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    try:
        descriptor = os.open(str(lock), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        raise ValueError("State is locked; do not run concurrent writers or remove an unreviewed stale lock")
    os.close(descriptor)
    try:
        state = read_json(path) if path.exists() else {"schema_version": 1, "provider": "runway_dev", "jobs": {}}
        if state.get("provider") != "runway_dev" or not isinstance(state.get("jobs"), dict):
            raise ValueError("Invalid Runway state file")
        for job_id, row in state["jobs"].items():
            if (not isinstance(row, dict) or not ID_PATTERN.fullmatch(job_id) or row.get("job_id") != job_id
                    or type(row.get("reserved_cost_cents")) is not int):
                raise ValueError("Invalid persisted job identity or budget reservation")
            metadata = row.get("request_metadata", {})
            if row["reserved_cost_cents"] != generation_cost_cents(metadata.get("model"), metadata.get("duration")):
                raise ValueError("Persisted reservation does not match the submitted model and duration")
        yield state
    finally:
        lock.unlink()


class RunwayWorkflow:
    def __init__(self, manifest_path: Path, state_path: Path, transport=None):
        self.manifest_path, self.state_path = Path(manifest_path).resolve(), Path(state_path).resolve()
        if self.manifest_path == self.state_path:
            raise ValueError("Manifest and mutable task state must use different paths")
        self.manifest_bytes = self.manifest_path.read_bytes()
        self.manifest_sha256 = digest(self.manifest_bytes)
        self.manifest = json.loads(self.manifest_bytes.decode("utf-8-sig"))
        jobs = self.manifest.get("jobs", self.manifest.get("tasks", []))
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("Manifest requires a nonempty jobs or tasks list")
        self.jobs = {}
        for job in jobs:
            job_id = job.get("job_id", "")
            if not isinstance(job_id, str) or not ID_PATTERN.fullmatch(job_id) or job_id in self.jobs:
                raise ValueError("Missing, unsafe, or duplicate job_id")
            self.jobs[job_id] = job
        self.transport = transport if transport is not None else UrllibTransport()

    def select(self, job_ids=None) -> list:
        selected = list(self.jobs) if job_ids is None else list(dict.fromkeys(job_ids))
        if not selected or any(job_id not in self.jobs for job_id in selected):
            raise ValueError("Select at least one existing job_id")
        return selected

    def _validate_training_reference(self, job: dict) -> dict:
        camera = job.get("source_camera")
        partition = self.manifest.get("camera_partition", {})
        roles = partition.get("roles", {})
        if (partition.get("status") != "frozen" or not partition.get("split_id")
                or not isinstance(roles, dict) or not camera or roles.get(camera) != "train"):
            raise ValueError("A frozen registered train-camera partition is required")
        if job.get("split") != "train" or job.get("submit_ready") is not True or job.get("synthetic") is not True:
            raise ValueError("Job must be explicitly reviewed submit_ready=true, synthetic=true, split=train")
        if str(job.get("status", "")).startswith("blocked"):
            raise ValueError("Blocked preparation candidates cannot be submitted")
        controls = job.get("submission_controls", {})
        reference = job.get("selected_reference") or {}
        history = reference.get("processing_history_id")
        negative_only = self.manifest.get("augmentation_scope") == "negative_only"
        if negative_only:
            self._validate_negative_only_histories()
            if (controls.get("training_ratio_preregistered") is not True
                    or controls.get("per_source_quota_preregistered") is not True
                    or controls.get("holdout_may_select_prompts_models_or_samples") is not False):
                raise ValueError("Negative-only augmentation still requires preregistered training ratio and source quotas, without holdout-driven selection")
        elif (controls.get("reference_processing_match_status") != "accepted" or not history
              or controls.get("shared_reference_processing_history_id") != history
              or controls.get("training_ratio_preregistered") is not True
              or controls.get("per_source_quota_preregistered") is not True
              or controls.get("holdout_may_select_prompts_models_or_samples") is not False):
            raise ValueError("Matched N/P reference history and preregistered sampling controls are required")
        if (reference.get("accepted") is not True or reference.get("source_camera") != camera
                or reference.get("split") != "train" or reference.get("lineage_complete") is not True
                or (not negative_only and reference.get("processing_history_accepted") is not True)):
            raise ValueError("An accepted train reference with complete, accepted processing lineage is required")
        parents = reference.get("parent_lineage")
        if not isinstance(parents, list) or not parents:
            raise ValueError("Reference parent lineage is missing")
        for parent in parents:
            if (parent.get("source_camera") != camera or parent.get("split") != "train"
                    or roles.get(parent.get("source_camera")) != "train"
                    or not parent.get("source_unit") or not parent.get("source_group")
                    or not SHA_PATTERN.fullmatch(str(parent.get("source_media_sha256", "")))):
                raise ValueError("Every reference ancestor must have an identified train-camera source")
        return reference

    def _validate_negative_only_histories(self) -> None:
        source_histories = {}
        for candidate in self.jobs.values():
            if (candidate.get("condition") not in {"no_target", "ordinary_water", "flood"}
                    or candidate.get("desired_training_label", "N") != "N"
                    or candidate.get("accepted_label") is not None):
                raise ValueError("negative_only permits only no_target/ordinary_water/flood N intents, with accepted_label=null; P tasks are excluded")
            reference = candidate.get("selected_reference") or {}
            if reference.get("accepted") is not True:
                continue  # Unaccepted references still fail the ordinary selected-job gate.
            history = reference.get("processing_history_id")
            if (reference.get("processing_history_reviewed") is not True
                    or not isinstance(history, str) or not history.strip()):
                raise ValueError("Every accepted negative reference requires processing_history_reviewed=true and a processing_history_id")
            parents = reference.get("parent_lineage")
            if not isinstance(parents, list) or not parents:
                raise ValueError("Accepted negative references require identified source parents")
            for parent in parents:
                unit, media_sha = parent.get("source_unit"), str(parent.get("source_media_sha256", ""))
                camera = candidate.get("source_camera")
                if not camera or not unit or not SHA_PATTERN.fullmatch(media_sha):
                    raise ValueError("Negative reference source identity/hash is incomplete")
                # Both keys prevent filename aliases from hiding inconsistent processing.
                for source in (("unit", camera, unit), ("sha256", media_sha.lower())):
                    previous = source_histories.setdefault(source, history)
                    if previous != history:
                        raise ValueError("Accepted negative references from the same source must share processing_history_id")

    def _validate_preview_reference(self, job: dict) -> dict:
        if (len(self.jobs) != 1 or job.get("kind") != "negative_candidate"
                or job.get("preview_only") is not True or job.get("training_ready") is not False
                or self.manifest.get("preview_only", True) is not True
                or self.manifest.get("training_ready", False) is not False
                or job.get("accepted_label") is not None
                or job.get("split") not in (None, "preview_only", "unassigned")):
            raise ValueError("media_preview accepts only one negative_candidate marked preview_only=true, training_ready=false, without a training split or accepted label")
        camera = job.get("source_camera")
        use = self.manifest.get("reference_use") or {}
        frozen = use.get("frozen_test_cameras")
        if (not isinstance(camera, str) or not camera.strip() or use.get("external_test") is not False
                or use.get("checked_current_frozen_test_list") is not True
                or not isinstance(frozen, list) or any(not isinstance(item, str) or not item.strip() for item in frozen)):
            raise ValueError("Preview reference_use must confirm non-external provenance and a reviewed current frozen-test-camera list")
        roles = (self.manifest.get("camera_partition") or {}).get("roles", {})
        test_roles = {"test", "external_test", "independent_test", "frozen_test", "holdout_test"}
        if camera in frozen or str(roles.get(camera, "")).lower() in test_roles:
            raise ValueError("A frozen or external test camera cannot be used for generation preview")
        reference = job.get("selected_reference") or {}
        if (reference.get("accepted") is not True or reference.get("source_camera") != camera
                or reference.get("lineage_complete") is not True):
            raise ValueError("Preview needs an accepted reference with complete same-camera source lineage")
        parents = reference.get("parent_lineage")
        if not isinstance(parents, list) or not parents:
            raise ValueError("Reference parent lineage is missing")
        for entry in [reference] + parents:
            if (entry.get("external_test") is True
                    or any(str(entry.get(field, "")).lower() in test_roles for field in ("split", "source_role", "source_scope"))):
                raise ValueError("Test reference or test ancestor cannot be used for generation preview")
        for parent in parents:
            if (parent.get("source_camera") != camera or not parent.get("source_unit") or not parent.get("source_group")
                    or not SHA_PATTERN.fullmatch(str(parent.get("source_media_sha256", "")))):
                raise ValueError("Every preview ancestor must identify its same-camera source and media SHA256")
        return reference

    def preview_camera_use(self) -> dict | None:
        if self.manifest.get("purpose") != "media_preview":
            return None
        job = next(iter(self.jobs.values()))
        return {"activity": "generation_preview", "source_camera": job.get("source_camera"),
                "independent_holdout_for_choices_using_this_preview": False,
                "scope": "prompt/model/sample selection informed by this camera preview"}

    def validate_job(self, job: dict) -> tuple[dict, dict]:
        reference = (self._validate_preview_reference(job) if self.manifest.get("purpose") == "media_preview"
                     else self._validate_training_reference(job))
        body = validated_provider_spec(job.get("provider_request", {}))
        expected = str(reference.get("sha256", ""))
        if not SHA_PATTERN.fullmatch(expected) or not reference.get("path"):
            raise ValueError("Reference path and SHA256 required")
        path = local_path(self.manifest_path.parent, reference["path"])
        size = path.stat().st_size
        if len(JPEG_PREFIX) + 4 * ((size + 2) // 3) > MAX_DATA_URI_BYTES:
            raise ValueError("Reference exceeds the 5MB encoded data URI limit; edit separately and re-accept")
        image = path.read_bytes()
        if digest(image) != expected.lower() or not image.startswith(b"\xff\xd8\xff"):
            raise ValueError("Reference hash or JPEG encoding does not match its acceptance record")
        metadata = {**body, "reference_path": str(path), "reference_sha256": digest(image),
                    "reference_bytes": len(image), "promptImage_encoded_bytes": len(JPEG_PREFIX) + 4 * ((len(image) + 2) // 3),
                    "base_generation_cost_cents": generation_cost_cents(body["model"]), "pricing_checked_date": "2026-11-08"}
        uri = JPEG_PREFIX + base64.b64encode(image).decode("ascii")
        body["promptImage"] = ([{"position": "first", "uri": uri}]
                               if MODEL_REGISTRY[body["model"]]["first_frame_array"] else uri)
        return body, metadata

    def plan(self, job_ids=None) -> dict:
        selected = self.select(job_ids)
        rows = []
        for job_id in selected:
            raw_spec = self.jobs[job_id].get("provider_request")
            spec = raw_spec if isinstance(raw_spec, dict) else {}
            blockers = []
            try:
                metadata = validated_provider_spec(spec)
                cost = generation_cost_cents(metadata["model"])
            except (ValueError, TypeError) as exc:
                metadata = {key: spec.get(key) for key in ("model", "ratio", "duration", "seed")}
                cost = None
                blockers.append(str(exc))
            try:
                _, metadata = self.validate_job(self.jobs[job_id])
            except (ValueError, OSError, TypeError, AttributeError) as exc:
                metadata = {**metadata, "reference_path": (self.jobs[job_id].get("selected_reference") or {}).get("path")}
                if str(exc) not in blockers:
                    blockers.append(str(exc))
            rows.append({"job_id": job_id, "request_metadata": metadata, "blockers": blockers,
                         "base_generation_cost_usd": usd_string(cost) if cost is not None else None,
                         "submit_ready_in_source": self.jobs[job_id].get("submit_ready", False)})
        unpriced = sum(row["base_generation_cost_usd"] is None for row in rows)
        subtotal = sum(cents(row["base_generation_cost_usd"]) for row in rows if row["base_generation_cost_usd"] is not None)
        preview = self.manifest.get("purpose") == "media_preview"
        requirements = (["one negative_candidate, preview_only=true, training_ready=false",
                         "accept reference/hash and complete same-camera source lineage",
                         "confirm non-external reference and check the current frozen test camera list",
                         "independent authorization JSON plus execute and cost cap"] if preview else
                        ["freeze camera_partition and use train cameras only", "accept selected_reference and complete processing parent lineage",
                         "match N/P reference editing history and preregister sampling", "explicit 10s model-specific provider_request and revised promptText; seed only where supported",
                         "independent authorization JSON plus execute and cost cap"])
        if not preview and self.manifest.get("augmentation_scope") == "negative_only":
            requirements[2] = "review accepted N reference processing history shared per source; preregister training ratio and source quotas; no P pair required"
        return {"action": "plan", "network_requests": 0, "credentials_read": False,
                "purpose": self.manifest.get("purpose", "training"), "preview_only": preview,
                "augmentation_scope": None if preview else self.manifest.get("augmentation_scope", "legacy_matched_np"),
                "camera_use_if_submitted": self.preview_camera_use(),
                "manifest_sha256": self.manifest_sha256, "job_count": len(rows), "jobs": rows,
                "estimated_base_cost_usd": usd_string(subtotal) if not unpriced else None,
                "priced_subtotal_usd": usd_string(subtotal), "unpriced_job_count": unpriced,
                "pricing": {"model_profiles": model_catalog(), "usd_per_credit": "0.01", "source": "https://docs.dev.runwayml.com/guides/pricing/",
                            "excludes": ["reference editing", "taxes", "retries"], "budget_authorized": False},
                "duration_note": "10s is a generated candidate duration, not a promise of 3s/7.5s effective QC-passed support",
                "adaptation_required": requirements,
                "training_ready": False}

    def authorize(self, path: Path, selected: list, max_cost_usd) -> tuple[dict, int]:
        authorization = read_json(Path(path))
        if (authorization.get("approved") is not True or authorization.get("provider") != "runway_dev"
                or not all(authorization.get(key) for key in ("authorization_id", "approved_by", "approved_at"))
                or authorization.get("manifest_sha256", "").lower() != self.manifest_sha256
                or local_path(Path(path).resolve().parent, authorization.get("state_path", "")) != self.state_path
                or not set(selected).issubset(set(authorization.get("allowed_job_ids", [])))):
            raise ValueError("Authorization must bind reviewed jobs, manifest SHA, state path, approver, and provider")
        return authorization, min(cents(authorization.get("max_cost_usd")), cents(max_cost_usd))

    def submit(self, job_ids, *, execute=False, authorization_path=None, max_cost_usd=None) -> dict:
        if not execute:
            return self.plan(job_ids)
        selected = self.select(job_ids)
        if not authorization_path or max_cost_usd is None:
            raise ValueError("submit requires --execute, --authorization, and --max-cost-usd")
        authorization, cap = self.authorize(Path(authorization_path), selected, max_cost_usd)
        with locked_state(self.state_path) as state:
            pending = [job_id for job_id in selected if job_id not in state["jobs"]]
            prepared = {job_id: self.validate_job(self.jobs[job_id]) for job_id in pending}
            reserved = sum(row["reserved_cost_cents"] for row in state["jobs"].values())
            if reserved + sum(generation_cost_cents(prepared[job_id][0]["model"]) for job_id in pending) > cap:
                raise ValueError("Budget cap would be exceeded; unknown/failed submissions still reserve their base cost")
            if pending:
                secret = load_api_secret()  # Only after all local acceptance/authorization/budget gates.
                headers = {"Authorization": "Bearer " + secret, "X-Runway-Version": API_VERSION, "Content-Type": "application/json"}
            for job_id in pending:
                body, metadata = prepared[job_id]
                payload = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                row = {"job_id": job_id, "status": "submission_unknown", "attempt_started_utc": utc_now(),
                       "reserved_cost_cents": generation_cost_cents(body["model"]), "authorization_id": authorization["authorization_id"],
                       "manifest_sha256": self.manifest_sha256, "request_sha256": digest(payload), "request_metadata": metadata,
                       "quality_status": "not_generated", "accepted_label": None, "training_ready": False}
                if self.manifest.get("augmentation_scope") == "negative_only" and self.manifest.get("purpose") != "media_preview":
                    row.update(augmentation_scope="negative_only", condition=self.jobs[job_id]["condition"])
                camera_use = self.preview_camera_use()
                if camera_use is not None:
                    row.update(purpose="media_preview", kind="negative_candidate", preview_only=True,
                               source_camera=self.jobs[job_id]["source_camera"], generation_preview=camera_use)
                    state.setdefault("camera_uses", []).append({**camera_use, "job_id": job_id,
                                                                "attempt_started_utc": row["attempt_started_utc"]})
                state["jobs"][job_id] = row
                atomic_json(self.state_path, state)  # A crash after this point must never trigger another POST.
                try:
                    status, raw = self.transport.request("POST", API_BASE + "/v1/image_to_video", headers, payload)
                    if not 200 <= status < 300:
                        row["submission_error"] = "HTTP_%d_no_automatic_retry" % status
                    else:
                        task_id = json.loads(raw.decode("utf-8")).get("id", "")
                        if not isinstance(task_id, str) or not ID_PATTERN.fullmatch(task_id) or secret in task_id:
                            raise ValueError("Missing valid task ID")
                        row.update(status="submitted", task_id=task_id, submitted_utc=utc_now())
                except Exception as exc:
                    row["submission_error"] = type(exc).__name__ + "_receipt_unknown_no_automatic_retry"
                atomic_json(self.state_path, state)
                if row["status"] == "submission_unknown":
                    break  # Stop the batch until the uncertain charge is reconciled.
            return {"action": "submit", "jobs": {job_id: state["jobs"].get(job_id, {"status": "not_attempted"}) for job_id in selected},
                    "reserved_cost_cents": sum(row["reserved_cost_cents"] for row in state["jobs"].values())}

    def poll(self, job_ids, *, execute=False) -> dict:
        if not execute:
            return {"action": "poll_plan", "network_requests": 0, "credentials_read": False, "job_ids": self.select(job_ids)}
        selected = self.select(job_ids)
        with locked_state(self.state_path) as state:
            rows = [state["jobs"].get(job_id, {}) for job_id in selected]
            if any(not row.get("task_id") for row in rows):
                raise ValueError("poll requires a known provider task ID; submission_unknown needs manual reconciliation")
            secret = load_api_secret()
            for row in rows:
                try:
                    status, raw = self.transport.request("GET", API_BASE + "/v1/tasks/" + parse.quote(row["task_id"], safe=""),
                                                        {"Authorization": "Bearer " + secret, "X-Runway-Version": API_VERSION})
                    if status != 200:
                        raise ValueError("Task query did not return HTTP 200")
                    result = json.loads(raw.decode("utf-8"))
                    provider_status = result.get("status")
                    if provider_status not in PROVIDER_STATUSES or result.get("id", row["task_id"]) != row["task_id"]:
                        raise ValueError("Unexpected task response")
                    updates = {"provider_status": provider_status, "last_polled_utc": utc_now()}
                    if provider_status == "SUCCEEDED":
                        outputs = result.get("output")
                        if not isinstance(outputs, list) or not outputs or any(not isinstance(url, str) or secret in url for url in outputs):
                            raise ValueError("Successful response lacks safe output URLs")
                        updates.update(output_urls=[https_url(url) for url in outputs], status="succeeded", quality_status="generated_pending_qc")
                    elif provider_status in {"FAILED", "CANCELLED"}:
                        updates.update(status=provider_status.lower(), failure=str(result.get("failure", "")).replace(secret, "[REDACTED]"),
                                       failure_code=str(result.get("failureCode", "")).replace(secret, "[REDACTED]"))
                    row.update(updates)
                    row.pop("poll_error", None)
                except Exception as exc:
                    row["poll_error"] = type(exc).__name__ + "_query_failed_no_automatic_retry"
                atomic_json(self.state_path, state)
            return {"action": "poll", "jobs": {job_id: state["jobs"][job_id] for job_id in selected}}

    def download(self, job_ids, output_dir: Path, *, execute=False) -> dict:
        selected = self.select(job_ids)
        if not execute:
            return {"action": "download_plan", "network_requests": 0, "credentials_read": False,
                    "job_ids": selected, "output_dir": str(Path(output_dir).resolve())}
        destination = Path(output_dir).resolve()
        with locked_state(self.state_path) as state:
            rows = [state["jobs"].get(job_id, {}) for job_id in selected]
            if any(row.get("status") != "succeeded" or not row.get("output_urls") for row in rows):
                raise ValueError("download accepts only known SUCCEEDED tasks with HTTPS output URLs")
            destination.mkdir(parents=True, exist_ok=True)
            for row in rows:
                downloads = row.setdefault("downloads", {})
                for index, url in enumerate(row["output_urls"]):
                    https_url(url)
                    target = destination / (row["job_id"] + "-%02d.mp4" % index)
                    previous = downloads.get(str(index), {})
                    if target.exists():
                        if previous.get("path") == str(target) and digest(target.read_bytes()) == previous.get("sha256"):
                            continue
                        raise ValueError("Download target exists without a matching receipt; refusing overwrite")
                    descriptor, name = tempfile.mkstemp(prefix="." + target.name + ".", suffix=".tmp", dir=str(destination))
                    temporary = Path(name)
                    try:
                        with os.fdopen(descriptor, "wb") as stream:
                            self.transport.download(url, stream)  # No API key or Authorization header.
                            stream.flush()
                            os.fsync(stream.fileno())
                        size = temporary.stat().st_size
                        if not size:
                            raise ValueError("Empty download")
                        receipt = {"path": str(target), "sha256": digest(temporary.read_bytes()), "bytes": size, "downloaded_utc": utc_now()}
                        os.replace(str(temporary), str(target))
                        downloads[str(index)] = receipt
                        row.pop("download_error", None)
                    except Exception as exc:
                        row["download_error"] = type(exc).__name__ + "_download_failed"
                    finally:
                        if temporary.exists():
                            temporary.unlink()
                    atomic_json(self.state_path, state)
            return {"action": "download", "jobs": {job_id: state["jobs"][job_id] for job_id in selected}, "training_ready": False}
