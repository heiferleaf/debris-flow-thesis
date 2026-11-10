"""Twelve fixed-center, real-PTS input-only 3/7.5s comparisons. No training."""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.tools.audit_span_coverage import nearest_many
from experiments.common.video.verify_units_v3 import sequential_frames
from experiments.common.video.pts_clip_sampling import jpeg95_bytes, rgb112_from_jpeg

OUT = ROOT / "data/process/annotation-recheck/20261110-user-edits-v1/span12"
SOURCES = ROOT / "data/metadata/EXP-017/role-free-plan-v3/sources.csv"
ANCHORS = {"v1_13":[14,30],"v1_16":[1033,1038],"v1_17":[68,96,312],
           "v1_91":[1518],"v1_94":[1283],"v1_95":[1265],"v1_97":[363,400]}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",encoding="utf-8")


def choose(coord, center, span):
    pts=coord[:,1]
    start=center-span/2
    positions=np.flatnonzero((pts>=start-.15)&(pts<=start+span+.15))
    local=pts[positions]
    if len(local)<16 or np.any(np.diff(local)<0):
        return dict(span_sec=span,supported=False,reason="insufficient_or_backstep_local_coordinate")
    # Keep the first original-order frame of each equal PTS; no synthetic repair.
    unique=np.r_[True,np.diff(local)>0]
    positions=positions[unique]
    local=pts[positions]
    indices,good=nearest_many(local,np.asarray([start]),span)
    chosen=positions[indices[0]]
    actual=pts[chosen]
    between=pts[chosen[0]:chosen[-1]+1]
    positive=np.diff(local);positive=positive[positive>0]
    gap_threshold=max(.2,3*float(np.median(positive)))
    discontinuities=np.flatnonzero((np.diff(between)<0)|(np.diff(between)>gap_threshold+1e-9))
    targets=start+np.linspace(0,span,16)
    return dict(span_sec=span,frame_count=16,nominal_input_start_sec=start,nominal_input_end_sec=start+span,
                supported=bool(good[0] and len(discontinuities)==0),
                frame_indices=coord[chosen,0].astype(int).tolist(),coordinate_row_positions=chosen.tolist(),
                pts_seconds=actual.tolist(),targets_seconds=targets.tolist(),
                point_errors_seconds=(actual-targets).tolist(),max_point_error_sec=float(max(abs(actual-targets))),
                actual_span_sec=float(actual[-1]-actual[0]),
                actual_center_sec=float((actual[0]+actual[-1])/2),
                local_duplicate_pts_removed=int((~unique).sum()),
                full_between_selected_max_pts_gap_sec=float(np.diff(between).max()),
                technical_gap_threshold_sec=gap_threshold,technical_discontinuity_count=len(discontinuities),
                reason="nearest_16_and_local_continuity_pass" if good[0] and not len(discontinuities) else "nearest_or_continuity_failure")


def contact(anchor, rendered, size):
    width=8*size;cell_h=size+31
    image=Image.new("RGB",(width,62+4*cell_h),"white")
    draw=ImageDraw.Draw(image)
    draw.text((4,4),f"{anchor['unit_id']} center={anchor['center_sec']}s | fixed output [{anchor['output_start_sec']},{anchor['output_end_sec']})",fill="black")
    draw.text((4,20),f"Actual RGB{size} input | rows 1-2: D=3s,F=16 | rows 3-4: D=7.5s,F=16",fill="black")
    draw.text((4,36),"D=input coverage seconds; F=input frame count. Longer input does not change output truth.",fill="black")
    for wi,window in enumerate(anchor["windows"]):
        for j,n in enumerate(window["frame_indices"]):
            x=j%8*size;y=62+(wi*2+j//8)*cell_h
            record=window["frames"][j]
            draw.text((x+2,y+1),f"f{n} t{record['pts_sec']:.3f}",fill="black")
            draw.text((x+2,y+14),f"D={window['span_sec']:g}s F=16",fill="black")
            image.paste(Image.fromarray(rendered[n]["rgb"+str(size)]),(x,y+31))
    return image


def main():
    started=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=False)
    sources={r["unit_id"]:r for r in csv.DictReader(SOURCES.open(encoding="utf-8-sig"))}
    index_path=ROOT/"data/process/video_frames/video1_segment/frames_index.csv"
    audit=list(csv.DictReader(index_path.open(encoding="utf-8-sig")))
    anchors=[];video_proofs=[];reference_panels=[]
    for uid,centers in ANCHORS.items():
        if time.monotonic()-started>580:
            raise RuntimeError("Bounded task exceeded 580 seconds; stop rather than expand")
        source=sources[uid]
        mw=ROOT/source["middleware_path"]
        coord_path=ROOT/source["coord_path"]
        qc_path=ROOT/source["qc_path"]
        coord=np.loadtxt(coord_path,delimiter=",",skiprows=1,ndmin=2)
        qc={int(r["frame"]):r for r in csv.DictReader(qc_path.open(encoding="utf-8-sig"))}
        requests=set();unit_anchors=[]
        for center in centers:
            windows=[choose(coord,center,span) for span in (3.,7.5)]
            anchor=dict(unit_id=uid,center_sec=center,output_start_sec=center-1.5,output_end_sec=center+1.5,
                        output_interval_closure="[start,end)",output_truth_changed=False,
                        time_axis="nominal middleware coord working seconds; no new physical/audit offset authentication",
                        windows=windows)
            near=int(np.argmin(abs(coord[:,1]-center)))
            anchor["reference_center_frame_index"]=int(coord[near,0])
            anchor["reference_center_pts_sec"]=float(coord[near,1])
            reference=min((r for r in audit if r["video_key"]==uid),key=lambda r:abs(float(r["t_sec"])-center))
            anchor["existing_audit_reference"]=dict(path="data/process/video_frames/video1_segment/frames/"+reference["frame_file"],
                audit_nominal_sec=float(reference["t_sec"]),audit_minus_requested_center_sec=float(reference["t_sec"])-center,
                content_alignment_review="pending_visual_review",offset_calibration="none; content comparison is not a calibrated offset")
            requests.add(anchor["reference_center_frame_index"])
            for window in windows:
                if "frame_indices" in window:
                    requests.update(window["frame_indices"])
            unit_anchors.append(anchor)
        before=mw.stat()
        got=sequential_frames(mw,sorted(requests))
        after=mw.stat()
        if set(got)!=requests:
            raise RuntimeError(uid+": requested frame missing in sequential decode")
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        rendered={}
        for n,bgr in got.items():
            if bgr.shape!=(256,256,3):
                raise RuntimeError(uid+": non256 middleware")
            encoded=jpeg95_bytes(bgr)
            decoded=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
            gray_std=float(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).std())
            frozen=float(qc[n]["gray_std"])
            rgb112=rgb112_from_jpeg(encoded)
            rendered[n]=dict(rgb112=rgb112,rgb256=cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB),
                proof=dict(frame_index=n,pts_sec=float(coord[n,1]),gray_std=gray_std,frozen_qc_std=frozen,
                    source_qc_abs_std_difference=abs(gray_std-frozen),
                    source_qc_std_pass=bool(gray_std>=20 and frozen>=20 and abs(gray_std-frozen)<=.055001),
                    source_bgr_sha256=sha(bgr.tobytes()),jpeg95_sha256=sha(encoded),
                    rgb112_sha256=sha(rgb112.tobytes()),rgb256_decoded_sha256=sha(cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB).tobytes()),
                    visual_quality_review="pending_contact_review"))
        for anchor in unit_anchors:
            for window in anchor["windows"]:
                window["frames"]=[rendered[n]["proof"].copy() for n in window["frame_indices"]]
                window["actual_qc_std_pass"]=all(r["source_qc_std_pass"] for r in window["frames"])
                window["supported"]=window["supported"] and window["actual_qc_std_pass"]
                window["distinct_rgb112_count"]=len({r["rgb112_sha256"] for r in window["frames"]})
            name=f"{uid}_center{anchor['center_sec']:04d}"
            for size in [112,256]:
                filename=name+f"_input{size}.png"
                contact(anchor,rendered,size).save(OUT/filename,optimize=True)
                anchor["contact"+str(size)]=filename
            ref=anchor["existing_audit_reference"]
            old=Image.open(ROOT/ref["path"]).convert("RGB").resize((256,144))
            new=Image.fromarray(rendered[anchor["reference_center_frame_index"]]["rgb256"]).resize((256,144))
            panel=Image.new("RGB",(512,184),"white");draw=ImageDraw.Draw(panel)
            draw.text((2,2),f"{uid} center{anchor['center_sec']} mw t{anchor['reference_center_pts_sec']:.3f}",fill="black")
            draw.text((258,2),f"audit t{ref['audit_nominal_sec']:.3f} delta{ref['audit_minus_requested_center_sec']:+.3f}s",fill="black")
            draw.text((2,18),"Middleware vs audit: content check only; no offset calibration",fill="black")
            panel.paste(new,(0,40));panel.paste(old,(256,40));reference_panels.append(panel)
            write_json(OUT/(name+".json"),anchor)
            anchors.append(anchor)
        video_proofs.append(dict(unit_id=uid,source_row=source,middleware_bytes=before.st_size,
            middleware_mtime_ns=before.st_mtime_ns,middleware_stat_unchanged=True,
            middleware_sha256=source["middleware_sha256"],middleware_hash_status="inherited; not rehashed in this bounded decode",
            coord_sha256_actual=sha(coord_path.read_bytes()),qc_sha256_actual=sha(qc_path.read_bytes()),
            sequential_decode_passes=1,decode_through_frame_inclusive=max(requests),requested_distinct_frames=len(requests),
            all_requests_returned=True,old_forbidden_ranges_not_used_for_sampling=True))
        print(json.dumps(dict(unit_id=uid,anchors=len(unit_anchors),frames=len(requests),max_index=max(requests),elapsed_sec=round(time.monotonic()-started,2))),flush=True)
    reference_sheet=Image.new("RGB",(1536,4*184),"white")
    for i,panel in enumerate(reference_panels):
        reference_sheet.paste(panel,(i%3*512,i//3*184))
    reference_sheet.save(OUT/"existing-audit-comparison.png",optimize=True)
    manifest=dict(status="generated_pending_all_contacts_visual_review",created_by_script=Path(__file__).resolve().relative_to(ROOT).as_posix(),
        script_sha256=sha(Path(__file__).read_bytes()),sources_csv_sha256=sha(SOURCES.read_bytes()),
        anchor_count=len(anchors),windows=len(anchors)*2,frame_references=len(anchors)*32,
        sampling="actual coord PTS nearest_many/tie-left; no seconds*fps, duplicate PTS removed, 16 distinct and strictly increasing PTS, <=.100001s point error",
        pixel_contract="actual 256 middleware BGR -> cv2 JPEG95 encode/decode -> RGB -> INTER_LINEAR112; PNG contacts preserve those actual pixels",
        fixed_output="[center-1.5,center+1.5); longer input adds context only; no output truth relabeling",
        no_training=True,no_features=True,no_raw_4k_decode=True,no_seek=True,no_annotation_writes=True,
        old_forbidden_ranges_not_used=True,
        anchors=anchors,videos=video_proofs,elapsed_sec=round(time.monotonic()-started,3),
        audit_reference_comparison="existing-audit-comparison.png",limitations=[
        "Visual QC of selected inputs does not certify every intervening pixel or all of each video.",
        "Coordinate continuity and point error are checked; source/audit working time is not newly authenticated to a physical clock.",
        "No performance preference between 3s and 7.5s is inferred."])
    write_json(OUT/"manifest.json",manifest)
    print(json.dumps(dict(anchors=len(anchors),windows=24,frame_references=384,supported=sum(w["supported"] for a in anchors for w in a["windows"]),elapsed_sec=manifest["elapsed_sec"])),flush=True)


if __name__=="__main__":
    main()
