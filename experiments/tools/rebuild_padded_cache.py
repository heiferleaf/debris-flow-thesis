"""Verify or rebuild retired derived crop images from their recorded raw sources.

Default: verify the raw input and reconstructed bytes in memory; write no images.
--rebuild creates missing images only. Existing files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = "data/metadata/cache-retirement/images-padded-free-20261107-v1/manifest.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reconstruct(record: dict, repo_root: Path) -> bytes:
    source = repo_root / record["source_path"]
    source_bytes = source.read_bytes()
    if sha256(source_bytes) != record["source_sha256"]:
        raise ValueError(f"raw SHA256 changed: {record['source_path']}")
    image = cv2.imdecode(np.frombuffer(source_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode raw image: {source}")
    left, top, right, bottom = record["crop_box_xyxy"]
    cropped = image[top:bottom, left:right]
    if [cropped.shape[1], cropped.shape[0]] != record["cropped_size_wh"]:
        raise ValueError(f"crop size differs: {record['candidate_path']}")
    ok, encoded = cv2.imencode(record["encoding"]["extension"], cropped,
                               record["encoding"]["opencv_params"])
    if not ok:
        raise ValueError(f"cannot encode crop: {record['candidate_path']}")
    result = encoded.tobytes()
    if sha256(result) != record["expected_sha256"]:
        raise ValueError(f"reconstructed SHA256 differs: {record['candidate_path']}")
    return result


def run(manifest_path: Path, repo_root: Path, *, rebuild: bool = False,
        output_root: Path | None = None, only: list[str] | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = {"verified": 0, "rebuilt": 0, "existing": 0, "missing": 0,
               "ineligible_skipped": 0, "errors": []}
    selected = set(only or [])
    for record in manifest["images"]:
        if selected and record["relative_path"] not in selected:
            continue
        if not record["eligible_for_deletion"]:
            summary["ineligible_skipped"] += 1
            continue
        try:
            result = reconstruct(record, repo_root)
            target = (output_root / record["relative_path"] if output_root is not None
                      else repo_root / record["candidate_path"])
            if target.exists():
                if sha256(target.read_bytes()) != record["expected_sha256"]:
                    raise ValueError(f"existing image differs; never overwritten: {target}")
                summary["existing"] += 1
            elif rebuild:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(result)
                summary["rebuilt"] += 1
            else:
                summary["missing"] += 1
            summary["verified"] += 1
        except (OSError, ValueError, cv2.error) as error:
            summary["errors"].append({"candidate_path": record["candidate_path"], "error": str(error)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--rebuild", action="store_true", help="create missing image files; never overwrite")
    parser.add_argument("--output-root", type=Path, help="optional independent destination for the cache-relative tree")
    parser.add_argument("--only", action="append", help="verify/rebuild only this cache-relative path; repeatable")
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else args.repo_root / args.manifest
    summary = run(manifest_path, args.repo_root, rebuild=args.rebuild,
                  output_root=args.output_root, only=args.only)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(1 if summary["errors"] else 0)


if __name__ == "__main__":
    main()
