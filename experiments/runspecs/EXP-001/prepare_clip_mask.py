"""Rebuild the frozen, metadata-only clip mask from its published CSV."""
import sys
from pathlib import Path as _ArchivePath
sys.path.insert(0, str(_ArchivePath(__file__).resolve().parents[3]))
from experiments.common.video.artifact_paths import recorded_path
import argparse
import csv
from pathlib import Path

import numpy as np


def load_clip_arrays(folder):
    with (folder / "unit-manifest.csv").open(encoding="utf-8-sig", newline="") as file:
        ids = [row["unit_id"] for row in csv.DictReader(file)]
    index = {uid: i for i, uid in enumerate(ids)}
    arrays = {name: np.full((len(ids), 48), -2, dtype=np.int8) for name in ("strict", "loose")}
    with (folder / "clip-labels.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            i, j = index[row["unit_id"]], int(row["clip_idx"])
            if not 0 <= j < 48 or arrays["strict"][i, j] != -2:
                raise ValueError("Duplicate or out-of-range clip")
            for name, array in arrays.items():
                value = int(row[name])
                if value not in (-1, 0, 1):
                    raise ValueError("Unexpected clip label")
                array[i, j] = value
    if any(np.any(array == -2) for array in arrays.values()):
        raise ValueError("Incomplete clip metadata")
    return dict(unit_ids=np.asarray(ids), **arrays)


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=recorded_path(root / "data/process/EXP-008-units-v2/clip-labels-rebuilt.npz", repo=root))
    args = parser.parse_args()
    arrays = load_clip_arrays(recorded_path(root / "data/process/EXP-008-units-v2", repo=root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as file:
        np.savez_compressed(file, **arrays)
    print("Strict retained clips: %d/7200" % int((arrays["strict"] >= 0).sum()))
    print(args.output)


if __name__ == "__main__":
    main()
