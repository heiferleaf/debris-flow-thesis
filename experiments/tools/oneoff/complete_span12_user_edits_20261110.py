"""Complete the saved span12 receipt after contact visual review; no decode."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"data/process/annotation-recheck/20261110-user-edits-v1/span12"
SOURCE=ROOT/"data/metadata/EXP-017/role-free-plan-v3/sources.csv"
BUILD=ROOT/"experiments/tools/oneoff/build_span12_user_edits_20261110.py"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def main():
    sources={r["unit_id"]:r for r in csv.DictReader(SOURCE.open(encoding="utf-8-sig"))}
    anchors=[];proofs=[];pixels_checked=0
    paths=sorted(OUT.glob("v1_*_center*.json"))
    assert len(paths)==12
    for path in paths:
        anchor=json.loads(path.read_text(encoding="utf-8"))
        anchor["visual_review"]=dict(all_32_selected_frames_viewed_at_112=True,all_32_selected_frames_viewed_at_256=True,
            gray_or_large_block_damage_observed=False,quality_conclusion="No grey residual or large decoding blocks seen in these selected inputs.",
            resolution_note="112 preserves broad channel/water distribution while fine texture and small wave/front details are reduced; no training/performance preference inferred.")
        anchor["existing_audit_reference"]["content_alignment_review"]="viewed: same camera geometry and comparable visible water/channel state; not a calibrated temporal offset"
        for wi,window in enumerate(anchor["windows"]):
            assert len(window["frames"])==len(window["frame_indices"])==16
            assert len(set(window["frame_indices"]))==16
            assert np.all(np.diff(window["pts_seconds"])>0)
            assert window["max_point_error_sec"]<=.100001
            assert window["technical_discontinuity_count"]==0
            assert window["supported"] and window["actual_qc_std_pass"]
            for frame in window["frames"]:
                frame["visual_quality_review"]="viewed at both actual RGB112 and RGB256 in contacts; no grey residual or large block damage seen"
            for size,key in [(112,"rgb112_sha256"),(256,"rgb256_decoded_sha256")]:
                im=np.asarray(Image.open(OUT/anchor["contact"+str(size)]).convert("RGB"))
                for j,frame in enumerate(window["frames"]):
                    x=j%8*size;y=62+(wi*2+j//8)*(size+31)+31
                    cell=im[y:y+size,x:x+size]
                    assert cell.shape==(size,size,3) and sha(cell.tobytes())==frame[key]
                    pixels_checked+=1
        write(path,anchor);anchors.append(anchor)
    for uid in sorted({a["unit_id"] for a in anchors}):
        a=[x for x in anchors if x["unit_id"]==uid]
        requests={n for x in a for w in x["windows"] for n in w["frame_indices"]}
        requests.update(x["reference_center_frame_index"] for x in a)
        source=sources[uid];mw=ROOT/source["middleware_path"];stat=mw.stat()
        proofs.append(dict(unit_id=uid,source_row=source,middleware_bytes=stat.st_size,middleware_mtime_ns=stat.st_mtime_ns,
            middleware_sha256_inherited=source["middleware_sha256"],middleware_hash_recomputed=False,
            coord_sha256_actual=sha((ROOT/source["coord_path"]).read_bytes()),
            qc_sha256_actual=sha((ROOT/source["qc_path"]).read_bytes()),
            sequential_decode_passes=1,decode_through_frame_inclusive=max(requests),requested_distinct_frames=len(requests),
            all_requests_returned=True,middleware_stat_unchanged_assertion_passed_during_decode=True,
            old_forbidden_ranges_not_used_for_sampling=True))
    windows=[w for a in anchors for w in a["windows"]]
    frames=[(a["unit_id"],f) for a in anchors for w in a["windows"] for f in w["frames"]]
    manifest=dict(status="complete_selected_input_technical_and_visual_review",anchor_count=12,windows=24,frame_references=384,
        distinct_selected_video_frame_pairs=len({(uid,f["frame_index"]) for uid,f in frames}),
        supported_windows=sum(w["supported"] for w in windows),
        max_point_error_sec=max(w["max_point_error_sec"] for w in windows),
        min_actual_source_gray_std=min(f["gray_std"] for _,f in frames),
        max_source_qc_abs_std_difference=max(f["source_qc_abs_std_difference"] for _,f in frames),
        max_between_selected_pts_gap_sec=max(w["full_between_selected_max_pts_gap_sec"] for w in windows),
        max_abs_existing_audit_sample_minus_requested_center_sec=max(abs(a["existing_audit_reference"]["audit_minus_requested_center_sec"]) for a in anchors),
        all_384_reference_frames_actually_viewed_at_112_and_256=True,selected_grey_or_block_damage_count=0,
        contact_embedded_pixel_sha_checks=pixels_checked,
        sampling="actual coord PTS nearest_many/tie-left; no seconds*fps; remove equal PTS before selecting; sixteen different frame indices and strictly increasing PTS; point error <=.100001s",
        pixel_contract="actual 256 middleware BGR -> JPEG95 encode/decode -> RGB -> INTER_LINEAR112; lossless PNG contacts match all recorded pixel hashes",
        fixed_output="[center-1.5,center+1.5); longer input adds evidence only; no output truth change",
        no_training=True,no_features=True,no_raw_4k_decode=True,no_seek=True,no_annotation_writes=True,
        old_forbidden_ranges_not_used=True,decode_and_contact_elapsed_sec=34.86,
        footer_recovery="All seven decode passes and 24 contacts completed. Initial footer used a relative __file__ and failed after all per-anchor JSONs were saved. Fixed only footer path resolution; rebuilt this receipt from saved per-anchor evidence without decoding media again.",
        sources_csv_sha256=sha(SOURCE.read_bytes()),script_sha256_after_footer_path_fix=sha(BUILD.read_bytes()),
        completion_script_sha256=sha(Path(__file__).read_bytes()),
        audit_reference_comparison="existing-audit-comparison.png",
        limitations=["Selected-frame visual QC does not certify all intervening pixels or whole videos.",
          "All times are nominal middleware coordinate working seconds. Nearest old audit reference images show comparable content; their sample-minus-center differences are not measured timebase offsets.",
          "Historical first/last anchors and coord QC were previously checked; no new physical-clock or local-offset authentication is claimed.",
          "No model, feature extraction, training, performance ranking or annotation update was performed."],
        anchors=anchors,videos=proofs)
    write(OUT/"manifest.json",manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ["anchors","videos"]},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
