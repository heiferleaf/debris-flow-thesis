"""Measure independent diagnostic polygons; never infer debris-flow type."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def polygon_mask(polygons: list, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) < 3:
            raise ValueError("Polygon needs at least3 points")
        points = np.asarray(polygon, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
            raise ValueError("Coordinates must be finite normalized x/y")
        pixels = np.rint(points * np.array([width-1,height-1])).astype(np.int32)
        cv2.fillPoly(mask, [pixels], 1)
    return mask.astype(bool)


def ratio(material: np.ndarray, roi: np.ndarray) -> float | None:
    n = int(roi.sum())
    return float(np.logical_and(material,roi).sum()/n) if n else None


def iou(x: np.ndarray, y: np.ndarray) -> float | None:
    union = int(np.logical_or(x,y).sum())
    return float(np.logical_and(x,y).sum()/union) if union else None


def state_if_target(lower: float | None, upper: float | None, threshold: float) -> str | None:
    # A partial conservative outline is only a lower bound. Its small size
    # cannot certify that the complete visible target is small.
    if lower is None or upper is None:
        return None
    return "P" if lower >= threshold else "T" if upper < threshold else None


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--outline-a",type=Path,required=True)
    parser.add_argument("--outline-b",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    files=[args.manifest,args.outline_a,args.outline_b]
    manifest,a,b=[json.loads(p.read_text(encoding="utf-8-sig")) for p in files]
    lookup=[{r["item_id"]:r for r in d["items"]} for d in (a,b)]
    results=[]
    for item in manifest["items"]:
        uid=item["unit_id"];w,h=item["size"]
        entries=[d[item["item_id"]] for d in lookup]
        rois=[polygon_mask([d["rois"][uid]],w,h) for d in (a,b)]
        material=[polygon_mask(r.get("flow_polygons",[]),w,h) for r in entries]
        bounds=[]
        for m,r,entry in zip(material,rois,entries):
            if entry.get("mask_reliability")=="unresolved":
                bounds.append((None,None));continue
            lower=ratio(m,r)
            if entry.get("mask_complete") is True:
                upper=lower
            elif entry.get("possible_flow_polygons") and entry.get("upper_bound_validated") is True:
                possible=polygon_mask(entry["possible_flow_polygons"],w,h)
                upper=ratio(np.logical_or(possible,m),r)
            else:
                upper=1.0
            bounds.append((lower,upper))
        scores=[lo if lo==hi else None for lo,hi in bounds]
        sensitivity=[]
        for d,m,entry in zip((a,b),material,entries):
            base=np.asarray(d["rois"][uid],dtype=float);center=base.mean(axis=0)
            variants=[]
            for factor in (.975,1,1.025):
                points=np.clip(center+(base-center)*factor,0,1)
                region=polygon_mask([points.tolist()],w,h)
                lower=ratio(m,region) if entry.get("mask_reliability")!="unresolved" else None
                if lower is None:
                    upper=None
                elif entry.get("mask_complete") is True:
                    upper=lower
                elif entry.get("possible_flow_polygons") and entry.get("upper_bound_validated") is True:
                    upper=ratio(np.logical_or(m,polygon_mask(entry["possible_flow_polygons"],w,h)),region)
                else:
                    upper=1.0
                variants.append({"factor":factor,"roi_pixels":int(region.sum()),"s_lower":lower,"s_upper":upper,
                                 "conditional_states":{str(t):state_if_target(lower,upper,t) for t in (.1,.2,.3)}})
            sensitivity.append(variants)
        results.append({"item_id":item["item_id"],"unit_id":uid,"t_sec":item["t_sec"],
                        "roi_iou":iou(*rois),"material_iou":iou(*material),"s_A":scores[0],"s_B":scores[1],
                        "s_bounds_A":bounds[0],"s_bounds_B":bounds[1],
                        "complete":[e.get("mask_complete",False) for e in entries],
                        "absolute_s_difference":abs(scores[0]-scores[1]) if None not in scores else None,
                        "reliability":[e.get("mask_reliability") for e in entries],
                        "conditional_states_if_E1":{str(t):[state_if_target(lo,hi,t) for lo,hi in bounds] for t in (.1,.2,.3)},
                        "ROI_sensitivity":sensitivity,"presence_gate_verified_here":False})
    summary={"items":len(results),"both_outline_available":sum(r["s_A"] is not None and r["s_B"] is not None for r in results),
             "conditional_TP_disagreements":{str(t):sum(v[0] is not None and v[1] is not None and v[0]!=v[1]
                 for v in (r["conditional_states_if_E1"][str(t)] for r in results)) for t in (.1,.2,.3)}}
    output={"purpose":"Reproducibility and sensitivity of proposed2D visual occupancy, not model performance or target-type certification",
            "inputs":[{"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
            "summary":summary,"results":results,"training_ready":False,
            "limitations":["Material outlines are AI visual interpretations of sparse cached triplets, not semantic ground truth",
                           "N/T/P requires independent presence evidence; conditional states are not sample labels",
                           "Same-family AI agreement is not independent expert accuracy",
                           "Three views do not validate a universal rule on all150 videos"]}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary))


if __name__=="__main__":
    main()
