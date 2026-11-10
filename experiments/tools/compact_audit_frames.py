"""Compact explicitly selected audit JPEGs in place, preserving names/indexes.

Requires an absolute frame root, a protection list, and an append-only JSONL
ledger. JPEG quality is 85, maximum long edge 1280; images are never enlarged.
Only a decoded, verified, smaller temporary JPEG may atomically replace a file.
Original videos and training directories are not discovered or opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import uuid

from PIL import Image
import PIL


REPO_ROOT = Path(__file__).resolve().parents[2]
LONG_EDGE = 1280
JPEG_QUALITY = 85
TEMP_MARKER = ".compact-audit-"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def load_protected_files(path: Path, frames_root: Path) -> tuple[set[str], str]:
    """Read UTF-8 lines, a JSON list, or {protected_files:[string/{path:...}]}.

    Paths are absolute or frame-root-relative; repository-relative data/... is
    also accepted for existing evidence manifests. Nothing is inferred by name.
    """
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    if path.suffix.lower() == ".json":
        document = json.loads(text)
        entries = document["protected_files"] if isinstance(document, dict) else document
        if not isinstance(entries, list):
            raise ValueError("protected_files must be a list")
    else:
        entries = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    protected = set()
    for entry in entries:
        value = entry.get("path") if isinstance(entry, dict) else entry
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Each protected entry must be a nonempty path")
        candidate = Path(value.replace("\\", "/"))
        if not candidate.is_absolute():
            candidate = (REPO_ROOT if candidate.parts[0] == "data" else frames_root) / candidate
        protected.add(path_key(candidate))
    return protected, sha256(raw)


class Ledger:
    """Flush/fsync every row; prepared records bridge replacement/log crashes."""

    def __init__(self, path: Path, context: dict):
        self.path = path
        self.lock = threading.Lock()
        self.latest = {}
        records = []
        discarded_tail = None
        truncate_at = None
        if path.is_file():
            raw = path.read_bytes()
            lines, offset = raw.splitlines(keepends=True), 0
            for number, line in enumerate(lines):
                try:
                    record = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    if number != len(lines) - 1:
                        raise ValueError("Ledger has a corrupt interior row")
                    discarded_tail = raw[offset:]
                    truncate_at = offset
                    break
                records.append(record)
                offset += len(line)
            if records and records[0] != context:
                raise ValueError("Ledger root, protection list, or encoding context differs")
            if raw and not records:
                raise ValueError("Ledger has no intact context record; no file was changed")
            if truncate_at is not None:
                with path.open("r+b") as stream:
                    stream.truncate(truncate_at)  # Only a matched ledger's incomplete tail.
            if discarded_tail is None and raw and not raw.endswith(b"\n"):
                with path.open("ab") as stream:
                    stream.write(b"\n")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", encoding="utf-8", newline="\n")
        for record in records:
            if record.get("event") == "file":
                self.latest[record["path"]] = record
        if not records:
            self.append(context)
        if discarded_tail is not None:
            self.append({"event": "recovered_incomplete_ledger_tail", "discarded_bytes": len(discarded_tail),
                         "discarded_sha256": sha256(discarded_tail)})

    def append(self, record: dict) -> None:
        with self.lock:
            self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            self.stream.flush()
            os.fsync(self.stream.fileno())
            if record.get("event") == "file":
                self.latest[record["path"]] = record

    def close(self) -> None:
        self.stream.close()


def _remove_own_temp(path: Path | None, source: Path, root: Path) -> None:
    if path is None:
        return
    resolved = path.resolve()
    prefix = "." + source.name + TEMP_MARKER
    if (within(resolved, root) and resolved.parent == source.parent.resolve()
            and resolved.name.startswith(prefix) and resolved.suffix == ".tmp"):
        try:
            resolved.unlink()
        except FileNotFoundError:
            pass


def _compact_one(path: Path, root: Path, protected: set[str], ledger: Ledger, run_id: str) -> dict:
    relative = path.relative_to(root).as_posix()
    record = {"event": "file", "run_id": run_id, "path": relative,
              "original_sha256": None, "original_bytes": None, "original_dimensions": None,
              "new_sha256": None, "new_bytes": None, "new_dimensions": None,
              "saved_logical_bytes": 0}
    temporary = None
    try:
        resolved = path.resolve(strict=True)
        if not within(resolved, root) or path.is_symlink():
            record.update(status="skipped", reason="outside_root_or_symlink")
        else:
            original = path.read_bytes()
            record.update(original_sha256=sha256(original), original_bytes=len(original))
            previous = ledger.latest.get(relative)
            unchanged_skip = bool(previous and previous.get("reason") in {"not_smaller", "resume_unchanged_not_smaller"}
                                  and previous.get("original_sha256") == record["original_sha256"])
            matched_new = bool(previous and previous.get("new_sha256") == record["original_sha256"] and not unchanged_skip)
            if previous and previous.get("temp_path"):
                _remove_own_temp(Path(previous["temp_path"]), path, root)
            # Protection takes precedence even when resuming a matching file.
            if path_key(path) in protected:
                with Image.open(io.BytesIO(original)) as image:
                    record["original_dimensions"] = list(image.size)
                record.update(status="skipped", reason="protected")
            elif previous and (matched_new or unchanged_skip):
                record.update(status="resume_matched", reason="already_compacted" if matched_new else "resume_unchanged_not_smaller",
                              original_dimensions=previous.get("new_dimensions") if matched_new else previous.get("original_dimensions"),
                              new_sha256=record["original_sha256"], new_bytes=len(original),
                              new_dimensions=previous.get("new_dimensions") if matched_new else previous.get("original_dimensions"),
                              previous_operation_id=previous.get("operation_id") or previous.get("previous_operation_id"),
                              recovered_prepared=previous.get("status") == "prepared")
            else:
                with Image.open(io.BytesIO(original)) as image:
                    record["original_dimensions"] = list(image.size)
                    if image.format != "JPEG":
                        record.update(status="skipped", reason="not_jpeg")
                    else:
                        image.load()
                        output = image.convert("RGB")
                        output.thumbnail((LONG_EDGE, LONG_EDGE), Image.Resampling.LANCZOS)
                        expected_size = output.size
                        descriptor, name = tempfile.mkstemp(prefix="." + path.name + TEMP_MARKER,
                                                            suffix=".tmp", dir=str(path.parent))
                        os.close(descriptor)
                        temporary = Path(name)
                        options = {key: image.info[key] for key in ("exif", "icc_profile") if image.info.get(key)}
                        output.save(temporary, format="JPEG", quality=JPEG_QUALITY, subsampling=2, optimize=True, **options)
                        output.close()
                if temporary is not None:
                    with Image.open(temporary) as checked:
                        checked.verify()
                    with Image.open(temporary) as checked:
                        checked.load()
                        if (checked.format != "JPEG" or checked.size != expected_size or max(checked.size) > LONG_EDGE
                                or any(new > old for new, old in zip(checked.size, record["original_dimensions"]))):
                            raise ValueError("Temporary JPEG failed size/readability validation")
                    compressed = temporary.read_bytes()
                    record.update(new_sha256=sha256(compressed), new_bytes=len(compressed), new_dimensions=list(expected_size))
                    if len(compressed) >= len(original):
                        record.update(status="skipped", reason="not_smaller")
                    else:
                        if sha256(path.read_bytes()) != record["original_sha256"]:
                            raise ValueError("Source changed while compacting; replacement refused")
                        with temporary.open("r+b") as stream:
                            stream.flush()
                            os.fsync(stream.fileno())
                        record.update(status="prepared", reason=None, operation_id=uuid.uuid4().hex,
                                      temp_path=str(temporary), saved_logical_bytes=len(original) - len(compressed))
                        ledger.append(dict(record))
                        os.replace(str(temporary), str(path))
                        temporary = None
                        record["status"] = "replaced"
        ledger.append(record)
        return record
    except Exception as error:
        record.update(status="error", reason=type(error).__name__ + ": " + str(error), saved_logical_bytes=0)
        ledger.append(record)
        return record
    finally:
        _remove_own_temp(temporary, path, root)


def compact_frames(frames_root: Path, protected_files: Path, ledger_path: Path, *,
                   limit: int | None = None, workers: int = 4) -> dict:
    root_argument = Path(frames_root)
    if not root_argument.is_absolute():
        raise ValueError("--frames-root must be an absolute path (including its Windows drive)")
    root = root_argument.resolve(strict=True)
    if not root.is_dir() or root == Path(root.anchor):
        raise ValueError("An explicit frame directory, not a filesystem root, is required")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("--limit must be positive when supplied")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("--workers must be between 1 and 8")
    protected, protection_sha = load_protected_files(Path(protected_files), root)
    ledger_path = Path(ledger_path).resolve()
    if ledger_path.suffix.lower() != ".jsonl" or within(ledger_path, root) or ledger_path == Path(protected_files).resolve():
        raise ValueError("--ledger must be a separate .jsonl outside the frame root and protection list")
    context = {"event": "context", "format_version": 1, "frames_root": str(root),
               "protected_files_sha256": protection_sha, "long_edge": LONG_EDGE, "jpeg_quality": JPEG_QUALITY,
               "subsampling": 2, "optimize": True, "pillow_version": PIL.__version__}
    ledger = Ledger(ledger_path, context)
    run_id = uuid.uuid4().hex
    summary = {"event": "summary", "run_id": run_id, "frames_root": str(root),
               "started_utc": datetime.now(timezone.utc).isoformat(), "workers": workers, "limit": limit,
               "processed": 0, "status_counts": {}, "reason_counts": {}, "replaced": 0,
               "saved_logical_bytes": 0, "original_bytes_replaced": 0, "new_bytes_replaced": 0}

    def candidates():
        count = 0
        for parent, directories, files in os.walk(root, followlinks=False):
            directories.sort()
            for name in sorted(files):
                if Path(name).suffix.lower() not in {".jpg", ".jpeg", ".jpe"}:
                    continue
                yield Path(parent) / name
                count += 1
                if limit is not None and count >= limit:
                    return

    statuses, reasons = Counter(), Counter()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for record in executor.map(lambda path: _compact_one(path, root, protected, ledger, run_id), candidates()):
                summary["processed"] += 1
                statuses[record["status"]] += 1
                if record.get("reason"):
                    reasons[record["reason"]] += 1
                if record["status"] == "replaced":
                    summary["replaced"] += 1
                    summary["saved_logical_bytes"] += record["saved_logical_bytes"]
                    summary["original_bytes_replaced"] += record["original_bytes"]
                    summary["new_bytes_replaced"] += record["new_bytes"]
        summary.update(status_counts=dict(statuses), reason_counts=dict(reasons),
                       completed_utc=datetime.now(timezone.utc).isoformat())
        ledger.append(summary)
        return summary
    finally:
        ledger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--protected-files", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Maximum number of JPEG candidates considered in this run")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    summary = compact_frames(args.frames_root, args.protected_files, args.ledger, limit=args.limit, workers=args.workers)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(1 if summary["status_counts"].get("error") else 0)


if __name__ == "__main__":
    main()
