"""Produce a small, zero-cost still-image photometric demonstration locally."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common.video import photometric_static as implementation  # noqa: E402
from experiments.common.video.photometric_static import (  # noqa: E402
    CONDITIONS, PhotometricSpec, photometric_metadata, simulate_photometric_frames,
)


DEFAULT_SOURCE = ROOT / "data/process/video_frames/video1_segment/frames/v1_16_frame_000143_t300s.jpg"
DEFAULT_MANIFEST = ROOT / "data/metadata/synthetic-video/photometric-demo-20261108-v1/manifest.json"
DEFAULT_PREVIEW_DIR = ROOT / ".codex/docs/.temp/photometric-demo-20261108-v1"
LABELS = {
    "baseline": "BASELINE | exact still repetition",
    "exposure_only": "EXPOSURE ONLY | amplitude 0.02 EV",
    "white_balance_only": "WHITE BALANCE ONLY | R/B +/-0.005",
    "sensor_noise_only": "NOISE ONLY | std 0.3/255 | AR(1) rho 0.95",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidate = Path("C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(candidate), size) if candidate.exists() else ImageFont.load_default()


def tile(frame: np.ndarray, label: str, width: int = 480) -> Image.Image:
    height = round(frame.shape[0] * width / frame.shape[1])
    panel = Image.new("RGB", (width, height + 30), "#14202b")
    panel.paste(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS), (0, 30))
    ImageDraw.Draw(panel).text((10, 7), label, fill="white", font=font(15))
    return panel


def preview_frame(frames: dict[str, np.ndarray], index: int, spec: PhotometricSpec) -> Image.Image:
    panels = [tile(frames[condition][index], LABELS[condition]) for condition in CONDITIONS]
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * 2, height * 2 + 70), "#0c141c")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), "PHOTOMETRIC DEMO | one ordinary-flow N still | NO PHYSICAL MOTION", fill="white", font=font(19))
    draw.text((12, 38), f"Frame {index + 1:02d}/{spec.frame_count} | {spec.preview_display_seconds:g} s display only | illustrative, uncalibrated | NOT TRAINING DATA", fill="#c1c9d2", font=font(16))
    for position, panel in enumerate(panels):
        canvas.paste(panel, ((position % 2) * width, 70 + (position // 2) * height))
    return canvas


def contact_sheet(frames: dict[str, np.ndarray]) -> Image.Image:
    panels = [(tile(frames[condition][0], LABELS[condition] + " | FIRST"),
               tile(frames[condition][-1], LABELS[condition] + " | LAST"))
              for condition in CONDITIONS]
    width, height = panels[0][0].size
    canvas = Image.new("RGB", (width * 2, height * len(CONDITIONS) + 55), "#0c141c")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "FIRST / LAST | one ordinary-flow N still | no physical motion", fill="white", font=font(20))
    draw.text((12, 33), "Illustrative photometric changes; not calibrated and not training-ready.", fill="#c1c9d2", font=font(15))
    for row, pair in enumerate(panels):
        for column, panel in enumerate(pair):
            canvas.paste(panel, (column * width, 55 + row * height))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--ffmpeg", type=Path, default=ROOT / ".venv/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win64-v4.2.2.exe")
    args = parser.parse_args()
    spec = PhotometricSpec(seed=args.seed)
    with Image.open(args.source) as image:
        source = np.asarray(image.convert("RGB"))
    frames = {condition: simulate_photometric_frames(source, condition, spec) for condition in CONDITIONS}
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    video_path = args.preview_dir / "photometric_comparison.mp4"
    contact_path = args.preview_dir / "first_last_contact_sheet.png"
    initial = preview_frame(frames, 0, spec)
    # Browser-compatible H.264 is only a viewing aid. Resize and video encoding
    # can suppress tiny changes; exact full-resolution array hashes and
    # pre-encoding delta statistics are retained in the manifest.
    command = [str(args.ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{initial.width}x{initial.height}", "-framerate", "16/3", "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "10",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for index in range(spec.frame_count):
            process.stdin.write(preview_frame(frames, index, spec).tobytes())
        process.stdin.close()
        assert process.stderr is not None
        error = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise RuntimeError(error)
    except BaseException:
        process.kill()
        process.wait()
        raise
    contact_sheet(frames).save(contact_path, optimize=True)
    condition_records = []
    for condition, output in frames.items():
        delta = np.abs(output.astype(np.int16) - source.astype(np.int16)[None])
        temporal_delta = np.abs(output[1:].astype(np.int16) - output[:-1].astype(np.int16))
        record = photometric_metadata(condition, spec)
        record.update({
            "output_shape": list(output.shape), "dtype": str(output.dtype),
            "in_memory_frame_array_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
            "mean_absolute_code_value_delta_from_source": float(delta.mean()),
            "mean_adjacent_frame_absolute_code_value_delta": float(temporal_delta.mean()),
            "changed_code_value_fraction_from_source": float((delta != 0).mean()),
            "max_absolute_code_value_delta_from_source": int(delta.max()),
        })
        condition_records.append(record)
    manifest = {
        "schema_version": 1, "purpose": "local_photometric_still_preview_only",
        "source": {"path": relative(args.source), "sha256": sha256(args.source),
                   "shape": list(source.shape), "semantic_anchor": "recognized ordinary-flow N",
                   "unique_source_images": 1},
        "physical_motion": "none", "unique_source_images": 1,
        "capture_observation_seconds": 0, "preview_display_seconds": spec.preview_display_seconds,
        "training_ready": False, "paid_api_calls": 0,
        "parameters": asdict(spec), "sensor_distribution_calibrated": False,
        "transform_description": {
            "exposure_only": "sRGB decode; linear RGB * 2**(amplitude_EV*sin(fixed_seed_phase+2*pi*t/period)); sRGB encode",
            "white_balance_only": "sRGB decode; linear RGB * [1+offset,1,1-offset]; offset=amplitude*sin(fixed_seed_phase+2*pi*t/period); sRGB encode",
            "sensor_noise_only": "encoded sRGB + stationary per-pixel/channel Gaussian AR(1); initial std=0.3/255; innovation std=std*sqrt(1-rho**2)",
            "quantization": "clip [0,255] and nearest-integer uint8 rounding",
            "geometry": "identity; no warp, crop, camera shake, water deformation or texture advection",
        },
        "conditions": condition_records,
        "code": [{"path": relative(path), "sha256": sha256(path)}
                 for path in (Path(implementation.__file__), Path(__file__))],
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "pillow": Image.__version__, "ffmpeg": relative(args.ffmpeg)},
        "preview_encoding": {"codec": "libx264", "pixel_format": "yuv420p", "crf": 10,
                             "display_frame_rate": "16/3", "panel_width": 480,
                             "purpose": "viewing aid; resize and encoding may suppress tiny changes"},
        "command": [".venv/Scripts/python.exe", "experiments/tools/preview_photometric_static.py",
                    "--source", relative(args.source), "--manifest", relative(args.manifest),
                    "--preview-dir", relative(args.preview_dir), "--seed", str(spec.seed),
                    "--ffmpeg", relative(args.ffmpeg)],
        "outputs": [{"path": relative(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                    for path in (video_path, contact_path)],
        "limitations": [
            "Parameters are illustrative defaults, not calibrated to real sensor distributions.",
            "No claim of improved realism or training effectiveness; nothing is connected to a trainer.",
            "16 displayed frames from one still are not 16 independent source frames or 3 seconds of captured observation.",
            "This is not a real 10-second video; physical ordinary-flow motion is absent.",
            "Small uint8 changes may be difficult to see at preview scale; this is not evidence that physical motion exists.",
        ],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": relative(args.manifest), "preview": relative(video_path),
                      "contact_sheet": relative(contact_path), "source_shape": list(source.shape),
                      "frame_count_per_condition": spec.frame_count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
