"""Apply the user's conditional authorization to v5_15..18 only.

Uses the exact pre-review snapshot as a concurrency guard. It preserves every
parsed field outside four kind cells and records before/after hashes.
"""
import csv
import hashlib
import io
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "data/process/annotation-recheck/20261110-user-edits-v1/root"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sync_live():
    result_path = OUT / "v5-negative-correction.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    path = ROOT / result["path"]
    if sha(path.read_bytes()) != result["after_sha256"]:
        raise RuntimeError("CSV changed after correction; no live update performed")
    before_rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig"), newline="")))
    receipts = []
    for key in result["changed_keys"]:
        url = "http://127.0.0.1:8768/api/"
        with urllib.request.urlopen(url + "state?v=" + key) as response:
            state = json.load(response)
        events = state["events"]
        if len(events) != 1 or events[0]["kind"] != "pos":
            raise RuntimeError("Live state changed; refusing overwrite for " + key)
        row = next(r for r in before_rows[1:] if r[1] == key)
        if events[0]["start"] != float(row[2]) or events[0]["end"] != float(row[3]) or events[0].get("note", "") != row[6]:
            raise RuntimeError("Live interval differs from reviewed interval for " + key)
        events[0]["kind"] = "neg"
        request = urllib.request.Request(url + "events?v=" + key,
            data=json.dumps(dict(events=events), ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request) as response:
            receipt = json.load(response)
        with urllib.request.urlopen(url + "state?v=" + key) as response:
            verified = json.load(response)["events"]
        assert verified == events
        receipts.append(dict(video_key=key, receipt=receipt, verified_events=verified))
    after_rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8-sig"), newline="")))
    header = before_rows[0]
    assert header == after_rows[0]
    old_other = [r for r in before_rows[1:] if r[1] not in result["changed_keys"]]
    new_other = [r for r in after_rows[1:] if r[1] not in result["changed_keys"]]
    assert old_other == new_other
    assert len(before_rows) == len(after_rows)
    result["live_server_sync"] = dict(receipts=receipts, after_sync_sha256=sha(path.read_bytes()),
        note="Live sessions cached pre-edit events. Updated through the server's own save API and verified readback; non-target CSV rows preserved. Target numeric formatting follows native server serialization.")
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result["live_server_sync"], ensure_ascii=False))


def withdraw_source_based_edit():
    result_path = OUT / "v5-negative-correction.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    path = ROOT / result["path"]
    if sha(path.read_bytes()) != result["live_server_sync"]["after_sync_sha256"]:
        raise RuntimeError("Working labels changed; refuse automatic rollback")
    receipts = []
    for prior in result["live_server_sync"]["receipts"]:
        key = prior["video_key"]
        base = "http://127.0.0.1:8768/api/"
        with urllib.request.urlopen(base + "state?v=" + key) as response:
            state = json.load(response)
        if state["events"] != prior["verified_events"]:
            raise RuntimeError("Live annotation changed; refuse overwrite for " + key)
        events = state["events"]
        events[0]["kind"] = "pos"
        request = urllib.request.Request(base + "events?v=" + key,
            data=json.dumps(dict(events=events), ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request) as response:
            receipt = json.load(response)
        with urllib.request.urlopen(base + "state?v=" + key) as response:
            after = json.load(response)["events"]
        assert after == events
        receipts.append(dict(video_key=key, receipt=receipt, verified_events=after))
    result["status"] = "withdrawn_and_user_P_restored"
    result["withdrawal"] = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        reason="User clarified acceptance requires same-scene visual comparison, not provider description. Existing visual comparison did not justify the N change. Restore prior user P, not an independent P certification.",
        receipts=receipts, final_sha256=sha(path.read_bytes()))
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(status=result["status"], keys=result["changed_keys"],
                         final_sha256=result["withdrawal"]["final_sha256"])))


def main():
    snapshot = json.loads((OUT / "current-label-snapshot.json").read_text(encoding="utf-8"))
    info = next(r for r in snapshot["csv_fingerprints"] if "video5_segment--" in r["path"])
    path = ROOT / info["path"]
    before = path.read_bytes()
    if sha(before) != info["sha256"]:
        raise RuntimeError("Annotations changed after snapshot; no write performed")
    text = before.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    header = rows[0]
    key_col, kind_col = header.index("video_key"), header.index("kind")
    original = [r[:] for r in rows]
    targets = {"v5_{:02d}".format(i) for i in range(15, 19)}
    changed = []
    for row in rows[1:]:
        if len(row) > kind_col and row[key_col] in targets:
            if row[kind_col] != "pos":
                raise RuntimeError("Unexpected non-P target; no write performed")
            row[kind_col] = "neg"
            changed.append(row[key_col])
    if len(changed) != 4 or set(changed) != targets:
        raise RuntimeError("Expected exactly four whole-video rows; no write performed")
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    after = output.getvalue().encode("utf-8-sig" if before.startswith(b"\xef\xbb\xbf") else "utf-8")
    reread = list(csv.reader(io.StringIO(after.decode("utf-8-sig"), newline="")))
    for a, b in zip(original, reread):
        if len(a) > key_col and a[key_col] in targets:
            check = b[:]
            check[kind_col] = "pos"
            assert a == check
        else:
            assert a == b
    assert len(original) == len(reread)
    if sha(path.read_bytes()) != sha(before):
        raise RuntimeError("Annotations concurrently edited; no write performed")
    temporary = path.with_name("events.v5-negative-correction.tmp")
    with temporary.open("xb") as f:
        f.write(after)
    os.replace(str(temporary), str(path))
    assert path.read_bytes() == after
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), path=str(path.relative_to(ROOT)),
                  before_sha256=sha(before), after_sha256=sha(after), changed_keys=changed,
                  cells_changed="kind only: pos -> neg; all other parsed CSV fields preserved",
                  authorization="User asked to review v5_15..20 and restore N if appropriate.",
                  basis="Provider explicitly identifies the v5 batch as ordinary water flow; actual sampled imagery did not contradict it. Source-supported labels, not independent visual geophysical certification.",
                  before_snapshot="annotation-snapshot/" + path.parent.name + "-" + sha(before)[:12] + ".csv",
                  training_authorized=False, historical_run_modified=False)
    (OUT / "v5-negative-correction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    if "--withdraw-source-edit" in sys.argv:
        withdraw_source_based_edit()
    elif "--sync-live" in sys.argv:
        sync_live()
    else:
        main()
