#!/usr/bin/env python3
"""Loopback-only, byte-preserving human video review server (Python 3.8 stdlib)."""
import argparse
import copy
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit


HERE = Path(__file__).resolve().parent
LABELS = ("unreviewed", "positive", "negative", "uncertain", "exclude")
QUALITY_FLAGS = ("intro_outro", "scene_cut", "split_screen", "blur",
                 "lens_droplets", "orientation", "time_scale", "other")
FIELDS = {"label", "notes", "quality_flags", "positive_intervals", "valid_start",
          "valid_end", "rotation", "reviewed_at"}
MAX_BODY = 256 * 1024
CHUNK_SIZE = 256 * 1024


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def atomic_json(path, value):
    """Replace one complete JSON snapshot; never truncate an existing snapshot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, str(path))
    finally:
        if os.path.exists(name):
            os.unlink(name)


def default_annotation():
    return {"label": "unreviewed", "notes": "", "quality_flags": [],
            "positive_intervals": [], "valid_start": None, "valid_end": None,
            "rotation": 0, "reviewed_at": None}


def finite_seconds(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(field + " must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(field + " must be a finite nonnegative number")
    return value


def validate_annotation(value, duration=None):
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("annotation fields are invalid")
    if value["label"] not in LABELS:
        raise ValueError("unknown label")
    if not isinstance(value["notes"], str) or len(value["notes"]) > 50000:
        raise ValueError("notes must be text of at most 50000 characters")
    flags = value["quality_flags"]
    if not isinstance(flags, list) or len(flags) > len(QUALITY_FLAGS):
        raise ValueError("quality_flags must be a list")
    if any(not isinstance(flag, str) or flag not in QUALITY_FLAGS for flag in flags):
        raise ValueError("unknown quality flag")
    if len(flags) != len(set(flags)):
        raise ValueError("quality_flags contains duplicates")
    if isinstance(value["rotation"], bool) or value["rotation"] not in (0, 90, 180, 270):
        raise ValueError("rotation must be 0, 90, 180 or 270")
    start, end = value["valid_start"], value["valid_end"]
    for field in ("valid_start", "valid_end"):
        if value[field] is not None:
            finite_seconds(value[field], field)
            if duration is not None and value[field] > duration + 0.05:
                raise ValueError(field + " exceeds recorded video duration")
    if start is not None and end is not None and start >= end:
        raise ValueError("valid_start must precede valid_end")
    intervals = value["positive_intervals"]
    if not isinstance(intervals, list) or len(intervals) > 1000:
        raise ValueError("positive_intervals must be a list of at most 1000 intervals")
    previous_end = -1
    for interval in intervals:
        if not isinstance(interval, dict) or set(interval) != {"start", "end"}:
            raise ValueError("each positive interval needs start and end")
        left = finite_seconds(interval["start"], "interval start")
        right = finite_seconds(interval["end"], "interval end")
        if left >= right or left < previous_end:
            raise ValueError("positive intervals must be ordered, nonoverlapping and start < end")
        if duration is not None and right > duration + 0.05:
            raise ValueError("positive interval exceeds recorded video duration")
        if start is not None and left < start:
            raise ValueError("positive interval starts before valid_start")
        if end is not None and right > end:
            raise ValueError("positive interval ends after valid_end")
        previous_end = right
    reviewed = value["reviewed_at"]
    if reviewed is not None:
        if not isinstance(reviewed, str) or len(reviewed) > 64:
            raise ValueError("reviewed_at must be an ISO timestamp or null")
        try:
            parsed = datetime.fromisoformat(reviewed.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("reviewed_at must be an ISO timestamp or null")
        if parsed.tzinfo is None:
            raise ValueError("reviewed_at must include a timezone")


class AnnotationState:
    """Serialized writes, durable append journal and atomically replaced snapshot."""
    def __init__(self, data_dir, videos):
        self.data_dir = Path(data_dir)
        self.snapshot = self.data_dir / "annotations.json"
        self.history = self.data_dir / "history.jsonl"
        self.videos = {video["id"]: video for video in videos}
        self.lock = threading.RLock()
        self.data = read_json(self.snapshot)
        if self.data.get("schema") != "external-video-human-annotations/v1":
            raise ValueError("unsupported annotation schema")
        if not isinstance(self.data.get("revision"), int) or self.data["revision"] < 0:
            raise ValueError("invalid annotation revision")
        if set(self.data.get("annotations", {})) != set(self.videos):
            raise ValueError("annotation IDs differ from manifest; refusing to discard edits")
        for video_id, annotation in self.data["annotations"].items():
            validate_annotation(annotation, self.videos[video_id].get("duration_seconds"))
        if not self.history.exists():
            if self.data["revision"] != 0:
                raise ValueError("history missing for nonzero revision")
            self.history.touch(exist_ok=False)
        recovered = False
        last_history_revision = 0
        with self.history.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.endswith("\n"):
                    raise ValueError("history has an incomplete final line; preserve it before repair")
                event = json.loads(line)
                revision = event.get("revision")
                if revision != last_history_revision + 1:
                    raise ValueError("history revision is not contiguous at line %s" % line_number)
                last_history_revision = revision
                video_id = event.get("id")
                if video_id not in self.videos:
                    raise ValueError("unknown ID in annotation history")
                validate_annotation(event.get("after"), self.videos[video_id].get("duration_seconds"))
                if revision > self.data["revision"]:
                    self.data["annotations"][video_id] = event["after"]
                    self.data["revision"] = revision
                    recovered = True
        if last_history_revision < self.data["revision"]:
            raise ValueError("snapshot is ahead of durable history")
        if recovered:
            atomic_json(self.snapshot, self.data)

    def get(self):
        with self.lock:
            return copy.deepcopy(self.data)

    def update(self, video_id, patch):
        if video_id not in self.videos:
            raise KeyError(video_id)
        if not isinstance(patch, dict) or not patch or set(patch) - FIELDS:
            raise ValueError("body must contain only annotation fields")
        with self.lock:
            before = self.data["annotations"][video_id]
            after = copy.deepcopy(before)
            after.update(patch)
            validate_annotation(after, self.videos[video_id].get("duration_seconds"))
            after["reviewed_at"] = utc_now()
            revision = self.data["revision"] + 1
            event = {"schema": "external-video-human-history/v1", "revision": revision,
                     "id": video_id, "at": after["reviewed_at"],
                     "before": before, "after": after}
            encoded = (json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            with self.history.open("ab", buffering=0) as stream:
                pending = memoryview(encoded)
                while pending:
                    written = stream.write(pending)
                    if not written:
                        raise OSError("incomplete history append")
                    pending = pending[written:]
                os.fsync(stream.fileno())
            updated = copy.deepcopy(self.data)
            updated["revision"] = revision
            updated["annotations"][video_id] = after
            # If replacement fails, the durable journal still contains the edit.
            self.data = updated
            atomic_json(self.snapshot, updated)
            return {"ok": True, "revision": revision, "annotation": copy.deepcopy(after)}


def parse_range(header, size):
    """One RFC-style bytes range; invalid/multipart/unsatisfiable returns ValueError."""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if match is None or size <= 0:
        raise ValueError("invalid byte range")
    first, last = match.groups()
    if not first:
        if not last or int(last) <= 0:
            raise ValueError("invalid suffix byte range")
        return max(0, size - int(last)), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        raise ValueError("unsatisfiable byte range")
    return start, end


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address, manifest_path, data_dir, ui_file, repo_root):
        if address[0] != "127.0.0.1":
            raise ValueError("server must bind 127.0.0.1")
        self.repo_root = Path(repo_root).resolve()
        self.manifest = read_json(manifest_path)
        self.ui_file = Path(ui_file).resolve()
        self.videos = {video["id"]: video for video in self.manifest["videos"]}
        if len(self.videos) != len(self.manifest["videos"]):
            raise ValueError("duplicate IDs in manifest")
        self.media = {}
        for video_id, video in self.videos.items():
            if not re.fullmatch(r"v\d{3,}", video_id):
                raise ValueError("invalid manifest ID")
            alias = (self.repo_root / video["alias_path"]).resolve()
            if self.repo_root not in alias.parents or not alias.is_file():
                raise ValueError("missing or unsafe media alias")
            if alias.stat().st_size != video["size_bytes"]:
                raise ValueError("media size changed: " + video_id)
            self.media[video_id] = alias
        self.state = AnnotationState(data_dir, list(self.videos.values()))
        super().__init__(address, ReviewHandler)

    @property
    def base_url(self):
        return "http://127.0.0.1:%s" % self.server_address[1]


class ReviewHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "LocalHumanReview/1"

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, message, *args):
        # URLs contain only local route IDs, never absolute source paths.
        print("%s %s" % (utc_now(), message % args), flush=True)

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_bytes(self, status, payload, content_type="application/json; charset=utf-8", headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def send_json(self, status, payload, headers=None):
        self.send_bytes(status, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"), headers=headers)

    def error(self, status, message):
        self.send_json(status, {"ok": False, "error": message})

    def local_request(self, mutation=False):
        allowed = {"127.0.0.1:%s" % self.server.server_address[1],
                   "localhost:%s" % self.server.server_address[1]}
        if self.headers.get("Host") not in allowed:
            self.close_connection = True
            self.error(403, "local Host required")
            return False
        if mutation:
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + host for host in allowed}:
                self.close_connection = True
                self.error(403, "same-origin requests required")
                return False
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                self.close_connection = True
                self.error(403, "cross-site requests are refused")
                return False
        return True

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self.local_request():
            return
        try:
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            if path in ("/", "/index.html", "/ui/index.html"):
                if not self.server.ui_file.is_file():
                    self.error(503, "review UI is not ready")
                    return
                self.send_bytes(200, self.server.ui_file.read_bytes(), "text/html; charset=utf-8", {
                    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; media-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"})
            elif path == "/api/manifest":
                keys = ("id", "display_name", "filename", "source_id", "relative_path",
                        "size_bytes", "duration_seconds", "media_url")
                videos = [{key: video.get(key) for key in keys} for video in self.server.videos.values()]
                self.send_json(200, {"schema": self.server.manifest["schema"],
                                     "session_id": self.server.manifest["session_id"],
                                     "videos": videos, "counts": self.server.manifest["counts"]})
            elif path == "/api/annotations":
                self.send_json(200, self.server.state.get())
            elif path == "/api/export.json":
                self.send_json(200, {**self.server.state.get(), "schema": "external-video-human-export/v1",
                                     "session_id": self.server.manifest["session_id"],
                                     "exported_at": utc_now(), "videos": self.server.manifest["videos"]},
                               {"Content-Disposition": 'attachment; filename="human-review-20261104.json"'})
            elif path == "/api/export.csv":
                self.export_csv()
            elif path.startswith("/media/"):
                video_id = path[len("/media/"):]
                if video_id not in self.server.videos:
                    self.error(404, "unknown video ID")
                    return
                query = parse_qs(parsed.query)
                self.send_media(video_id, query.get("mime") == ["mp4"])
            else:
                self.error(404, "route not found")
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        except Exception as exc:
            self.log_message("GET failed (%s)", type(exc).__name__)
            self.close_connection = True
            self.error(500, "local read failed; inspect server log")

    def send_media(self, video_id, mp4_compat=False):
        path = self.server.media[video_id]
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size != self.server.videos[video_id]["size_bytes"]:
                self.error(409, "media size changed; stop review and inspect source")
                return
            start, end, status = 0, size - 1, 200
            if self.headers.get("Range"):
                try:
                    start, end = parse_range(self.headers["Range"], size)
                except ValueError:
                    self.send_bytes(416, b"", headers={"Content-Range": "bytes */%s" % size,
                                                       "Accept-Ranges": "bytes"})
                    return
                status = 206
            self.send_response(status)
            content_type = "video/quicktime" if path.suffix.lower() == ".mov" else "video/mp4"
            if mp4_compat:
                content_type = "video/mp4"
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(max(0, end - start + 1)))
            if status == 206:
                self.send_header("Content-Range", "bytes %s-%s/%s" % (start, end, size))
            self.end_headers()
            if self.command == "HEAD":
                return
            stream.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = stream.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def export_csv(self):
        output = io.StringIO(newline="")
        columns = ("id", "display_name", "filename", "source_id", "relative_path",
                   "size_bytes", "duration_seconds", "label", "notes", "quality_flags",
                   "positive_intervals", "valid_start", "valid_end", "rotation", "reviewed_at")
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        annotations = self.server.state.get()["annotations"]
        for video_id, video in self.server.videos.items():
            row = {key: video.get(key) for key in columns[:7]}
            row.update(annotations[video_id])
            row["quality_flags"] = json.dumps(row["quality_flags"], ensure_ascii=False)
            row["positive_intervals"] = json.dumps(row["positive_intervals"], ensure_ascii=False)
            # Keep spreadsheet software from executing free-text notes as formulas.
            for field in ("notes", "display_name", "filename", "source_id", "relative_path"):
                if isinstance(row[field], str) and row[field].lstrip().startswith(("=", "+", "-", "@")):
                    row[field] = "'" + row[field]
            writer.writerow(row)
        self.send_bytes(200, output.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8",
                        {"Content-Disposition": 'attachment; filename="human-review-20261104.csv"'})

    def do_POST(self):
        if not self.local_request(mutation=True):
            return
        path = unquote(urlsplit(self.path).path)
        prefix = "/api/annotations/"
        if not path.startswith(prefix) or path[len(prefix):] not in self.server.videos:
            self.close_connection = True
            self.error(404, "unknown annotation ID")
            return
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            self.error(400, "transfer encoding is unsupported")
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self.close_connection = True
            self.error(415, "application/json required")
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            self.close_connection = True
            self.error(413 if length > MAX_BODY else 411, "valid bounded Content-Length required")
            return
        try:
            encoded = self.rfile.read(length)
            if len(encoded) != length:
                raise ValueError("incomplete request body")
            patch = json.loads(encoded.decode("utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError("non-finite JSON number")))
            result = self.server.state.update(path[len(prefix):], patch)
            self.send_json(200, result)
        except (ValueError, UnicodeDecodeError) as exc:
            self.error(400, str(exc))
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception as exc:
            self.log_message("save failed (%s)", type(exc).__name__)
            self.close_connection = True
            self.error(500, "save failed; edit may be in durable history; inspect server log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--repo-root", type=Path, default=HERE.parents[3])
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--ui", type=Path, default=HERE / "ui" / "index.html")
    parser.add_argument("--server-info", type=Path, default=HERE / "server-info.json")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port must be 0..65535")
    data_dir = args.data_dir or args.repo_root / "data/metadata/external-videos/human-review-20261104-v1"
    # OS-held lock prevents a second server/process from racing the same history.
    lock_path = Path(data_dir) / ".server.lock"
    lock_stream = lock_path.open("a+b")
    if os.name == "nt":
        import msvcrt
        if lock_path.stat().st_size == 0:
            lock_stream.write(b"0")
            lock_stream.flush()
        lock_stream.seek(0)
        msvcrt.locking(lock_stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    server = ReviewServer(("127.0.0.1", args.port), args.manifest, data_dir, args.ui, args.repo_root)
    info = {"schema": "external-video-human-review-server/v1", "session_id": server.manifest["session_id"],
            "host": "127.0.0.1", "port": server.server_address[1], "url": server.base_url + "/",
            "pid": os.getpid(), "started_at": utc_now(), "status": "running"}
    atomic_json(args.server_info, info)
    print("Human review server: " + info["url"], flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        info.update(status="stopped", stopped_at=utc_now())
        atomic_json(args.server_info, info)
        lock_stream.close()


if __name__ == "__main__":
    main()
