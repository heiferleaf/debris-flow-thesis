"""Read existing checkpoints/logs to describe previous mechanisms; no network forward or fitting."""
from __future__ import annotations

import csv
import gc
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/results/exp022/error-diagnosis-20261105-v2/method-mechanisms-v2.json"


def json_lines(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def build() -> dict:
    if OUT.exists():
        raise FileExistsError("Mechanism audit already exists")
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    result = {"scope": "Only existing logs and read-only mmap checkpoint tensors; no model forward, epoch reselection, training or external testing.",
              "difference_stem": [], "ranking": {}, "energy_direct_bounds": [],
              "linear_inner_learning": [], "fixed_score_separability": []}
    for run in ["paired-s7-20261030-v2", "paired-s8-20261031-v1"]:
        path = ROOT / "data/results/exp016" / run / "diff/best.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        weight = checkpoint["state_dict"]["backbone.stem.0.weight"]
        rgb, difference = float(weight[:, :3].norm()), float(weight[:, 3:].norm())
        result["difference_stem"].append({"checkpoint": path.relative_to(ROOT).as_posix(),
            "selected_epoch": checkpoint["completed_epochs"], "RGB_L2": rgb, "difference_L2": difference,
            "difference_to_RGB_L2": difference / rgb, "interpretation": "Norm ratio is not information contribution."})
        del checkpoint, weight
        gc.collect()
    base = ROOT / "data/results/exp018/paired-s7-20261101-v1"
    for arm in ["bce", "cross", "same"]:
        history = json_lines(base / arm / "epochs.jsonl")
        epoch = history[-1]["best_epoch"]
        selected = history[epoch - 1]
        trace = []
        for e in range(1, 21):
            rows = json.loads((base / arm / f"validation-epoch-{e:02d}.json").read_text(encoding="utf-8"))
            wrong = 0.
            for unit in ["v1_122", "v1_63", "v1_82"]:
                ps = [r["logit"] for r in rows if r["clip_id"].split(":")[0] == unit and r["label"] == 1]
                ns = [r["logit"] for r in rows if r["clip_id"].split(":")[0] == unit and r["label"] == 0]
                assert len(ps) == len(ns) == 4
                wrong += sum((p < n) + .5 * (p == n) for p in ps for n in ns)
            trace.append({"epoch": e, "wrong_pairs": wrong, "pairs": 48, "inversion_rate": wrong / 48})
        schedule = json_lines(base / arm / "paired-epochs.jsonl")
        key = "cross_pairs" if arm == "cross" else "same_pairs"
        pairs = [(batch["clip_ids"][p], batch["clip_ids"][n]) for row in schedule
                 for batch in row["batches"] for p, n in batch[key]]
        result["ranking"][arm] = {"source": (base / arm).relative_to(ROOT).as_posix(), "selected_epoch": epoch,
            "training_pair_mean_first": history[0]["train_pair_mean"],
            "training_pair_mean_selected": selected["train_pair_mean"],
            "training_weighted_BCE_selected": selected["train_weighted_bce"],
            "lambda_times_pair_mean_selected": selected["rank_lambda"] * selected["train_pair_mean"],
            "selected_validation": trace[epoch - 1],
            "best_inversion_descriptive_only_not_reselection": min(trace, key=lambda r: r["inversion_rate"]),
            "all_epochs_validation_inversion": trace,
            "pair_exposures": len(pairs), "unique_pairs": len(set(pairs)),
            "unique_endpoints": len({v for pair in pairs for v in pair}),
            "same_video_pairs": sum(p.split(":")[0] == n.split(":")[0] for p, n in pairs)}
    for directory in sorted((ROOT / "data/results/exp021/energy-s7-20261103-v2").glob("f*-*-s7")):
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True, mmap=True)
        weight = checkpoint["state_dict"]["head.energy_weight"].reshape(4, 28)
        lo, hi = float(weight.amin(1).sum()), float(weight.amax(1).sum())
        bound = max(abs(lo), abs(hi))
        result["energy_direct_bounds"].append({"checkpoint": (directory / "best.pt").relative_to(ROOT).as_posix(),
            "selected_epoch": checkpoint["completed_epochs"], "logit_lower": lo, "logit_upper": hi,
            "absolute_logit_bound": bound, "probability_shift_bound_fixed_RGB": bound / 4,
            "condition": "Four simplex-normalized28-entry descriptors; no energy bias; fix this model's own RGB term."})
        del checkpoint, weight
        gc.collect()
    for fold in [2, 3, 5]:
        path = ROOT / f"data/results/exp022/linear-cv-s7-20261104-v1/f{fold:02d}-inner/epochs.jsonl"
        history = json_lines(path)
        epoch = history[-1]["best_epoch"]
        selected = history[epoch - 1]
        later_epoch = 18 if fold in {2, 3} else 20
        late = next(r for r in history if r["completed_epochs"] == later_epoch)
        result["linear_inner_learning"].append({"fold": fold, "source": path.relative_to(ROOT).as_posix(),
            "selected_epoch": epoch,
            "selected_training_BCE": selected.get("train_weighted_bce", selected.get("train_loss")),
            "selected_validation_BCE": selected["validation_unweighted_bce"],
            "later_example_epoch": late["completed_epochs"],
            "late_training_BCE": late.get("train_weighted_bce", late.get("train_loss")),
            "late_validation_BCE": late["validation_unweighted_bce"]})
    path = ROOT / "data/results/exp022/report-linear-cv-s7-20261104-v1-v1/video-pairs.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        pairs = list(csv.DictReader(stream))
    for fold in range(1, 6):
        positives = [float(r["linear_score"]) for r in pairs if int(r["fold"]) == fold and int(r["label"]) == 1]
        negatives = [float(r["linear_score"]) for r in pairs if int(r["fold"]) == fold and int(r["label"]) == 0]
        result["fixed_score_separability"].append({"fold": fold,
            "positive_min": min(positives), "negative_min": min(negatives), "negative_max": max(negatives),
            "negative_files_unavoidably_included_to_retain_all_positive_files": sum(n >= min(positives) for n in negatives),
            "strict_inversions": sum(p < n for p in positives for n in negatives),
            "ties": sum(p == n for p in positives for n in negatives),
            "note": "Optimistic per-fold oracle separability bound on saved probabilities; not a fitted threshold or revised metric."})
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = build()
    print(json.dumps({"saved": OUT.relative_to(ROOT).as_posix(),
                      "energy_bounds": result["energy_direct_bounds"],
                      "linear_inner_learning": result["linear_inner_learning"]}, ensure_ascii=False, indent=2))
