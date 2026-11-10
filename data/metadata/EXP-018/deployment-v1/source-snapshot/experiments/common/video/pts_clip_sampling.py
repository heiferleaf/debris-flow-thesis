"""Deterministic PTS sampling and explicitly invoked checked frame extraction."""
from __future__ import annotations

import hashlib
import numpy as np

CLIP_LEN = 16
SPAN_SECONDS = 3.0
MAX_POINT_ERROR = 0.100001


def complement_ranges(forbidden, n):
    result, left = [], 0
    for a, b in sorted(forbidden):
        a, b = max(0, int(a)), min(n, int(b))
        if a > left:
            result.append((left, a))
        left = max(left, b)
    if left < n:
        result.append((left, n))
    return result


def nearest_window(pts, a, b, target_start):
    """Nearest points within one original-order monotone clean run [a,b)."""
    local = np.asarray(pts[a:b], dtype=np.float64)
    if not len(local) or not np.isfinite(local).all() or not np.isfinite(target_start) or np.any(np.diff(local) < 0):
        raise ValueError("Nearest-neighbour input must be one monotone run")
    targets = float(target_start) + np.linspace(0, SPAN_SECONDS, CLIP_LEN)
    right = np.clip(np.searchsorted(local, targets), 0, len(local) - 1)
    left = np.maximum(right - 1, 0)
    chosen = np.where(abs(local[left] - targets) <= abs(local[right] - targets), left, right)
    actual = local[chosen]
    if (np.any(np.diff(chosen) <= 0) or np.any(np.diff(actual) <= 0)
            or np.max(abs(actual - targets)) > MAX_POINT_ERROR
            or not 2.9 - 1e-9 <= actual[-1] - actual[0] <= 3.1 + 1e-9):
        return None
    return chosen.astype(np.int32) + a


def segment_candidates(pts, clean_runs, start, end):
    """Enumerate real-PTS anchored unique tuples in one unchanged [start,end).

    Quotas are applied by the caller after pooling all clean runs in this
    original segment. Nominal 3 s targets never extend the original label.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if end - start < SPAN_SECONDS - 1e-9:
        return np.empty((0, CLIP_LEN), dtype=np.int32)
    parts = []
    for a, b in clean_runs:
        local = pts[a:b]
        if np.any(np.diff(local) < 0):
            raise ValueError("Clean run contains a PTS backstep")
        left = a + int(np.searchsorted(local, start, side="left"))
        right = a + int(np.searchsorted(local, end, side="left"))
        local = pts[left:right]
        if len(local) < CLIP_LEN:
            continue
        anchors = local[(local + SPAN_SECONDS <= end + 1e-9)
                        & (local + SPAN_SECONDS <= local[-1] + 1e-9)]
        if not len(anchors):
            continue
        target = anchors[:, None] + np.linspace(0, SPAN_SECONDS, CLIP_LEN)
        ri = np.clip(np.searchsorted(local, target), 0, len(local) - 1)
        li = np.maximum(ri - 1, 0)
        ix = np.where(abs(local[li] - target) <= abs(local[ri] - target), li, ri)
        actual = local[ix]
        spans = actual[:, -1] - actual[:, 0]
        good = (np.all(np.diff(ix, axis=1) > 0, axis=1)
                & np.all(np.diff(actual, axis=1) > 0, axis=1)
                & (np.max(abs(actual - target), axis=1) <= MAX_POINT_ERROR)
                & (spans >= 2.9 - 1e-9) & (spans <= 3.1 + 1e-9))
        if good.any():
            parts.append((ix[good] + left).astype(np.int32))
    if not parts:
        return np.empty((0, CLIP_LEN), dtype=np.int32)
    unique = np.unique(np.concatenate(parts), axis=0)
    centers = (pts[unique[:, 0]] + pts[unique[:, -1]]) / 2
    return unique[np.lexsort((unique[:, 0], centers))]


def uniform_indices(n, k):
    if not 0 <= k <= n:
        raise ValueError((n, k))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    indices = np.rint(np.linspace(0, n - 1, k)).astype(np.int64)
    if len(np.unique(indices)) != k:
        raise ValueError("Uniform representative selection repeated an index")
    return indices


def fixed_quota_rotation(n, epochs=20):
    """k=min(4,n) nonempty chronological strata, <=epochs representatives each."""
    if n < 0 or epochs < 1:
        raise ValueError("Candidate count must be nonnegative and epochs positive")
    k = min(4, n)
    if not k:
        return np.empty((epochs, 0), dtype=np.int64), []
    strata = np.array_split(np.arange(n, dtype=np.int64), k)
    pools = [part[uniform_indices(len(part), min(epochs, len(part)))] for part in strata]
    schedule = np.array([[pool[epoch % len(pool)] for pool in pools] for epoch in range(epochs)])
    if any(len(set(row)) != k for row in schedule):
        raise ValueError("Repeated candidate within one segment/epoch")
    return schedule, [pool.tolist() for pool in pools]


def dense_candidates(pts, clean_runs):
    """QC-only 3 s/1.5 s grid with final end-aligned target and tuple dedup."""
    out, skipped, seen = [], [], set()
    for a, b in clean_runs:
        if pts[b - 1] - pts[a] < SPAN_SECONDS - 1e-9:
            skipped.append((a, b, "clean_run_shorter_than_3s"))
            continue
        final = float(pts[b - 1] - SPAN_SECONDS)
        count = int(np.floor((final - pts[a]) / 1.5 + 1e-9)) + 1
        starts = [float(pts[a] + i * 1.5) for i in range(count)]
        if abs(starts[-1] - final) > 1e-8:
            starts.append(final)
        for target in starts:
            ix = nearest_window(pts, a, b, target)
            if ix is None:
                skipped.append((a, b, f"invalid_grid_target:{target:.9f}"))
                continue
            key = tuple(map(int, ix))
            if key not in seen:
                seen.add(key)
                out.append((target, ix))
    return out, skipped


def tuple_digest(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i4").tobytes()).hexdigest()


def jpeg95_bytes(bgr):
    import cv2
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("JPEG95 encoding failed")
    return encoded.tobytes()


def rgb112_from_jpeg(encoded):
    """Shared cache/dense contract: JPEG decode, BGR->RGB, linear resize112."""
    import cv2
    bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape != (256, 256, 3):
        raise ValueError("Invalid cached JPEG dimensions")
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (112, 112), interpolation=cv2.INTER_LINEAR)


def selection_runs(indices):
    """Disjoint arithmetic runs representing exactly the requested index set."""
    selected = sorted(set(map(int, indices)))
    runs, i = [], 0
    while i < len(selected):
        if i + 2 < len(selected):
            step = selected[i + 1] - selected[i]
            j = i + 2
            while j < len(selected) and selected[j] - selected[j - 1] == step:
                j += 1
            if j - i >= 3:
                runs.append((selected[i], selected[j - 1], step))
                i = j
                continue
        runs.append((selected[i], selected[i], 1))
        i += 1
    return runs


def selection_expression(indices):
    terms = []
    for a, b, step in selection_runs(indices):
        if a == b:
            terms.append(f"eq(n,{a})")
        elif step == 1:
            terms.append(f"between(n,{a},{b})")
        else:
            terms.append(f"between(n,{a},{b})*eq(mod(n-{a},{step}),0)")
    def balanced(parts):
        if len(parts) == 1:
            return parts[0]
        middle = len(parts) // 2
        return f"({balanced(parts[:middle])}+{balanced(parts[middle:])})"
    return balanced(terms) if terms else "0"


def validate_pixel_std(actual, frozen, threshold):
    if (not np.isfinite(actual) or not np.isfinite(frozen) or actual < threshold
            or frozen < threshold or abs(actual - frozen) > 0.055001):
        raise ValueError(f"Source pixel/QC mismatch: actual={actual}, frozen={frozen}, threshold={threshold}")


def extract_checked_frames(source, pts, qc_stds, required, output_dir, evidence_dir, ffmpeg, qc_std_low):
    """One sequential H264 pass; select by FFmpeg's physical frame counter.

    showinfo BEFORE select verifies every source frame number/physical PTS
    against the frozen coord. Only selected full BGR frames cross the pipe.
    Existing JPEGs are reused only if regenerated bytes have identical SHA256.
    """
    import csv
    import json
    import re
    import subprocess
    import threading
    import time
    from pathlib import Path
    import cv2

    source, output_dir, evidence_dir = map(Path, (source, output_dir, evidence_dir))
    required = sorted(set(map(int, required)))
    if not required or required[0] < 0 or required[-1] >= len(pts):
        raise ValueError("Invalid/empty selected frame set")
    if len(qc_stds) != len(pts):
        raise ValueError("QC/coordinate count mismatch")
    evidence_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    version = subprocess.run([str(ffmpeg), "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    major = int(re.search(r"ffmpeg version (?:n)?(\d+)", version).group(1))
    script = evidence_dir / "select-filter.txt"
    expression = selection_expression(required)
    script.write_text(f"showinfo,select='{expression}',format=bgr24", encoding="utf-8")
    filter_option = "-/filter:v" if major >= 8 else "-filter_script:v"
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-copyts", "-threads", "1", "-i", str(source),
               filter_option, str(script), "-an"]
    command += ["-fps_mode", "passthrough"] if major >= 5 else ["-vsync", "0"]
    command += ["-f", "rawvideo", "pipe:1"]
    (evidence_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    physical = np.full(len(pts), np.nan, dtype=np.float64)
    errors, state = [], {"decoded": 0, "max_pts_error": 0.0}
    pattern = re.compile(r"\bn:\s*(\d+)\s+pts:\s*\S+\s+pts_time:\s*([-+0-9.eE]+)")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1024 * 1024)

    def read_diagnostics():
        try:
            with (evidence_dir / "decoder.log").open("x", encoding="utf-8") as log:
                for raw_line in iter(process.stderr.readline, b""):
                    line = raw_line.decode("utf-8", errors="replace")
                    match = pattern.search(line)
                    if match:
                        index, timestamp = int(match[1]), float(match[2])
                        if index != state["decoded"] or index >= len(pts) or "s:256x256" not in line:
                            errors.append(f"Source frame sequence/dimensions mismatch at {index}")
                        elif not np.isfinite(timestamp):
                            errors.append(f"Nonfinite physical PTS at {index}")
                        else:
                            physical[index] = timestamp
                            state["max_pts_error"] = max(state["max_pts_error"], abs(timestamp - pts[index]))
                        state["decoded"] += 1
                    elif "showinfo" not in line:
                        log.write(line)
                        if any(token in line.lower() for token in ("error while decoding", "invalid nal", "corrupt decoded")):
                            errors.append(line.strip())
        except Exception as error:
            errors.append(f"Diagnostic reader failed: {error}")

    reader = threading.Thread(target=read_diagnostics, daemon=True)
    reader.start()
    start = time.perf_counter()
    records = []
    try:
        for index in required:
            buffer = bytearray()
            while len(buffer) < 256 * 256 * 3:
                chunk = process.stdout.read(256 * 256 * 3 - len(buffer))
                if not chunk:
                    raise RuntimeError(f"Selected pixel stream ended before source frame {index}")
                buffer.extend(chunk)
            bgr = np.frombuffer(buffer, dtype=np.uint8).reshape(256, 256, 3)
            gray_std = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).std())
            validate_pixel_std(gray_std, qc_stds[index], qc_std_low)
            encoded = jpeg95_bytes(bgr)
            fingerprint = hashlib.sha256(encoded).hexdigest()
            path = output_dir / f"frame-{index:08d}.jpg"
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest() != fingerprint:
                    raise FileExistsError(f"Existing JPEG has different bytes, refused: {path}")
                action = "reused_same_sha256"
            else:
                with path.open("xb") as f:
                    f.write(encoded)
                action = "created"
            stored = path.read_bytes()
            if hashlib.sha256(stored).hexdigest() != fingerprint:
                raise RuntimeError(f"JPEG readback fingerprint mismatch: {path}")
            rgb = rgb112_from_jpeg(stored)
            records.append(dict(frame_index=index, source_pts_seconds=float(pts[index]),
                                physical_pts_seconds="", gray_std=gray_std, frozen_qc_std=float(qc_stds[index]),
                                jpeg_size_bytes=len(encoded), jpeg_sha256=fingerprint,
                                source_bgr_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(),
                                rgb112_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), action=action))
        if process.stdout.read(1):
            raise RuntimeError("Extra selected pixel bytes beyond expected frame count")
        process.wait()
        reader.join()
        if process.returncode or errors or state["decoded"] != len(pts) or not np.isfinite(physical).all() or state["max_pts_error"] > 0.001001:
            raise RuntimeError(f"Frame/PTS decode verification failed: rc={process.returncode}; state={state}; errors={errors[:5]}")
        for record in records:
            record["physical_pts_seconds"] = float(physical[record["frame_index"]])
        np.save(evidence_dir / "physical-pts.npy", physical, allow_pickle=False)
        with (evidence_dir / "frames-written.csv").open("x", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        summary = dict(status="passed", source_frames=state["decoded"], selected_frames=len(records),
                       max_physical_coord_pts_error=state["max_pts_error"], seconds=time.perf_counter() - start,
                       jpeg_bytes=sum(record["jpeg_size_bytes"] for record in records), ffmpeg_version=version,
                       cv2_version=cv2.__version__, preprocessing="JPEG95->decode->RGB->INTER_LINEAR112")
        (evidence_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    except Exception as error:
        if process.poll() is None:
            process.kill()
        process.wait()
        reader.join()
        (evidence_dir / "failure.json").write_text(json.dumps(dict(error=str(error), state=state,
                                                                   diagnostics=errors[:20], completed_selected_frames=len(records)), indent=2), encoding="utf-8")
        raise


def refresh_exact_qc(source, pts, output_csv, evidence_dir, ffmpeg):
    """Explicit repair: same FFmpeg BGR24 route, unrounded full-frame gray std.

    No RGB images are stored. The source and all previous QC/coord stay intact.
    """
    import csv
    import json
    from pathlib import Path
    import re
    import subprocess
    import time
    import cv2

    output_csv, evidence_dir = Path(output_csv), Path(evidence_dir)
    if output_csv.exists():
        raise FileExistsError(output_csv)
    evidence_dir.mkdir(parents=True, exist_ok=False)
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-copyts", "-threads", "1", "-i", str(source),
               "-vf", "showinfo,format=bgr24", "-an", "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"]
    (evidence_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    start = time.perf_counter()
    stds = np.empty(len(pts), dtype=np.float64)
    with (evidence_dir / "decoder.log").open("xb") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, bufsize=1024 * 1024)
        try:
            for index in range(len(pts)):
                buffer = bytearray()
                while len(buffer) < 256 * 256 * 3:
                    chunk = process.stdout.read(256 * 256 * 3 - len(buffer))
                    if not chunk:
                        raise RuntimeError(f"QC repair pixel stream ended at frame {index}")
                    buffer.extend(chunk)
                bgr = np.frombuffer(buffer, dtype=np.uint8).reshape(256, 256, 3)
                stds[index] = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).std())
            if process.stdout.read(1):
                raise RuntimeError("QC repair decoded extra frames")
            if process.wait():
                raise RuntimeError("QC repair decoder failed")
        except Exception:
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
    physical = []
    pattern = re.compile(r"\bn:\s*(\d+)\s+pts:\s*\S+\s+pts_time:\s*([-+0-9.eE]+)")
    for line in (evidence_dir / "decoder.log").open(encoding="utf-8", errors="replace"):
        match = pattern.search(line)
        if match:
            if int(match[1]) != len(physical) or "s:256x256" not in line:
                raise ValueError("QC repair physical frame/dimension mismatch")
            physical.append(float(match[2]))
    physical = np.asarray(physical, dtype=np.float64)
    if (len(physical) != len(pts) or not np.isfinite(physical).all() or not np.isfinite(stds).all()
            or np.max(abs(physical - pts)) > 0.001001):
        raise ValueError("QC repair frame count/physical PTS/finite check failed")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("x", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "pts_time", "gray_std", "coord_pts_time"])
        for index in range(len(pts)):
            writer.writerow([index, format(physical[index], ".9f"), format(stds[index], ".17g"), format(pts[index], ".9f")])
    return dict(frames=len(pts), seconds=time.perf_counter() - start,
                max_physical_coord_pts_error=float(np.max(abs(physical - pts))), gray_std_storage="17 significant digits",
                cv2_version=cv2.__version__, ffmpeg_version=subprocess.run([str(ffmpeg), "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0])
