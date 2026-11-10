"""Synthetic-only regressions for four-state annotation and isolated versions.

Run: .venv\Scripts\python.exe -m unittest discover -s experiments/tests -p test_event_annotator_transition.py
"""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "event_annotator_server.py"
SPEC = importlib.util.spec_from_file_location("event_annotator_transition_test_target", MODULE_PATH)
annotator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(annotator)


class TransitionTests(unittest.TestCase):
    def test_version_source_mapping_survives_repo_move_and_missing_copy_fails(self):
        with tempfile.TemporaryDirectory(prefix="annotator-transition-mapping-") as directory:
            workspace = Path(directory) / "workspace"
            videos = {}
            for name in ("first", "second"):
                source = workspace / name / "same-source-name"
                source.mkdir(parents=True)
                (source / "events.csv").write_text("video_id,start_sec,end_sec,confidence,kind,note\n"
                                                 f"{name},0,2,real,neg,origin\n", encoding="utf-8")
                videos[name] = {"kind": "dir", "dir": source, "events_csv": source / "events.csv"}
            original_repo_root = annotator.REPO_ROOT
            try:
                annotator.REPO_ROOT = workspace
                version = workspace / "version"
                redirected = annotator.apply_annotations_dir(videos, version)
                self.assertNotEqual(redirected["first"]["events_csv"], redirected["second"]["events_csv"])
                manifest = (version / "sources.json").read_bytes()
                relative_targets = {name: spec["events_csv"].relative_to(version)
                                    for name, spec in redirected.items()}
                moved = Path(directory) / "moved"
                workspace.rename(moved)
                annotator.REPO_ROOT = moved
                moved_videos = {name: {"kind": "dir", "dir": moved / name / "same-source-name",
                    "events_csv": moved / name / "same-source-name" / "events.csv"} for name in videos}
                remapped = annotator.apply_annotations_dir(moved_videos, moved / "version")
                for name, spec in remapped.items():
                    self.assertEqual(spec["events_csv"].relative_to(moved / "version"), relative_targets[name])
                self.assertEqual((moved / "version" / "sources.json").read_bytes(), manifest)
                remapped["first"]["events_csv"].unlink()
                with self.assertRaisesRegex(ValueError, "版本CSV缺失"):
                    annotator.apply_annotations_dir(moved_videos, moved / "version")
            finally:
                annotator.REPO_ROOT = original_repo_root

    def test_precise_times_legacy_kind_and_unknown(self):
        event = annotator.normalize_events([
            {"start": 1.125, "end": 2.375, "kind": "transition", "note": "weak"}
        ])[0]
        self.assertEqual((event["start"], event["end"], event["kind"]), (1.125, 2.375, "transition"))
        unknown = annotator.normalize_events([{"start": 1.125, "end": 2.375, "kind": "unknown", "note": "类型未定"}])
        self.assertEqual(annotator.kind_counts(unknown), {"pos": 0, "cut": 0, "neg": 0, "transition": 0, "unknown": 1})
        with self.assertRaises(ValueError):
            annotator.normalize_events([{"start": 0, "end": 1, "kind": "mystery"}])
        for invalid in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                annotator.normalize_events([{"start": 0, "end": invalid, "kind": "neg"}])
        with tempfile.TemporaryDirectory(prefix="annotator-transition-unit-") as directory:
            path = Path(directory) / "events.csv"
            path.write_text("video_id,start_sec,end_sec,confidence,kind,note\n"
                            "v,0.125,1.375,real,transition,t\n"
                            "v,2.125,3.375,real,mystery,u\n"
                            "v,4.125,5.375,real,unknown,explicit U\n", encoding="utf-8")
            events = annotator.load_annotations(path, "v")
            self.assertEqual(events[0]["start"], 0.125)
            self.assertEqual(events[1]["kind"], "mystery")
            self.assertEqual(events[2]["kind"], "unknown")
            self.assertEqual(annotator.kind_counts(events)["pos"], 0)
            self.assertEqual(annotator.kind_counts(events)["unknown"], 2)
            path.write_text("video_id,start_sec,end_sec,confidence,note\nv,0,2,real,old\n", encoding="utf-8")
            self.assertEqual(annotator.load_annotations(path, "v")[0]["kind"], "pos")

    def test_overlay_http_roundtrip_isolation_and_statistics(self):
        with tempfile.TemporaryDirectory(prefix="annotator-transition-http-") as directory:
            tmp = Path(directory)
            root, source = annotator._selftest_build_fixture(tmp)
            # A preexisting legacy CSV and fractional index times must survive initialization.
            legacy_csv = root / "plain_video" / "events.csv"
            legacy_csv.write_text("video_id,start_sec,end_sec,confidence,kind,note\n"
                                  "plain_video,0.125,1.375,real,neg,old\n", encoding="utf-8")
            index = source / "frames_index.csv"
            with index.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                row["t_sec"] = str(float(row["t_sec"]) + 0.125)
            with index.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            originals = {path: path.read_bytes() for path in (source / "events.csv", legacy_csv)}
            videos, default = annotator.resolve_videos(str(root))
            version = tmp / "labels-version"
            redirected = annotator.apply_annotations_dir(videos, version)
            for name, spec in redirected.items():
                origin = spec["origin_events_csv"]
                self.assertEqual(spec["events_csv"].read_bytes(), originals[origin])
                self.assertEqual(spec["index_csv"], videos[name]["index_csv"])
            self.assertEqual(redirected["key1"]["events_csv"], redirected["key2"]["events_csv"])
            self.assertNotEqual(redirected["key1"]["events_csv"], redirected["plain_video"]["events_csv"])
            self.assertFalse(list(version.rglob("*.jpg")))
            manifest_bytes = (version / "sources.json").read_bytes()
            manifest = json.loads(manifest_bytes)
            self.assertEqual(len(manifest["sources"]), 2)
            for entry in manifest["sources"].values():
                self.assertEqual(entry["origin_sha256"], hashlib.sha256(originals[Path(entry["origin_csv"]) ]).hexdigest())

            server = annotator.make_server(redirected, default, tmp / "thumbs", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"

            def request(path, body=None, expected=200):
                req = urllib.request.Request(base + path,
                    data=None if body is None else json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"})
                try:
                    response = urllib.request.urlopen(req, timeout=5)
                except urllib.error.HTTPError as error:
                    response = error
                with response:
                    self.assertEqual(response.status, expected)
                    return json.loads(response.read().decode("utf-8"))

            try:
                state = request("/api/state?v=key1")
                self.assertEqual(state["ts"], [0.125, 2.125, 4.125])
                self.assertFalse(state["scene_export_enabled"])
                self.assertEqual(Path(state["csv_path"]), redirected["key1"]["events_csv"])
                events = [
                    {"start": 0.125, "end": 1.375, "kind": "neg", "note": "原始长备注" * 70},
                    {"start": 1.375, "end": 2.125, "kind": "transition", "note": "weak"},
                    {"start": 2.125, "end": 4.125, "kind": "pos", "note": "clear"},
                ]
                request("/api/events?v=key1", {"events": events})
                self.assertEqual(request("/api/state?v=key1")["events"], events)
                listing = {v["name"]: v for v in request("/api/videos")["videos"]}
                self.assertEqual(listing["key1"]["n_transition"], 1)
                self.assertEqual(listing["key2"]["n_pos"], 1)
                # Explicit type changes preserve the endpoints and note.
                events[1]["kind"] = "cut"
                request("/api/events?v=key1", {"events": events})
                events[1]["kind"] = "unknown"
                request("/api/events?v=key1", {"events": events})
                self.assertEqual(request("/api/state?v=key1")["events"][1]["kind"], "unknown")
                self.assertEqual(annotator.load_events_consolidated(source, "key1", redirected["key1"]["events_csv"])[1]["kind"], "unknown")
                listing_u = {v["name"]: v for v in request("/api/videos")["videos"]}
                self.assertEqual(listing_u["key1"]["n_unknown"], 1)
                self.assertEqual(listing_u["key1"]["n_neg"], 1)
                events[1]["kind"] = "transition"
                request("/api/events?v=key1", {"events": events})
                overlapping = [dict(event) for event in events]
                overlapping[1]["start"] = 1.2
                request("/api/events?v=key1", {"events": overlapping}, expected=400)
                self.assertEqual(request("/api/state?v=key1")["events"], events)
                request("/api/scenes?v=key1", {}, expected=400)
                version_csv = redirected["key1"]["events_csv"]
                with version_csv.open(encoding="utf-8-sig", newline="") as stream:
                    persisted = list(csv.DictReader(stream))
                self.assertEqual([row for row in persisted if row["video_key"] == "key2"][0]["note"],
                                 "预填整段正样本(key2 隔离参照)")
                request("/api/events?v=plain_video", {"events": [
                    {"start": 0.125, "end": 1.375, "kind": "transition", "note": "legacy overlay"}]})
                listing = {v["name"]: v for v in request("/api/videos")["videos"]}
                self.assertEqual(listing["plain_video"]["n_transition"], 1)
                self.assertEqual(listing["plain_video"]["n_neg"], 0)
                for path, original in originals.items():
                    self.assertEqual(path.read_bytes(), original)
                saved_copy = version_csv.read_bytes()
                annotator.apply_annotations_dir(videos, version)
                self.assertEqual(version_csv.read_bytes(), saved_copy)
                self.assertEqual((version / "sources.json").read_bytes(), manifest_bytes)
                # A later canonical change cannot silently reset provenance or version contents.
                (source / "events.csv").write_text("changed synthetic origin", encoding="utf-8")
                annotator.apply_annotations_dir(videos, version)
                self.assertEqual(version_csv.read_bytes(), saved_copy)
                self.assertEqual((version / "sources.json").read_bytes(), manifest_bytes)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                for session in annotator.AnnotatorHandler.sessions.values():
                    session.thumb_cache.stop.set()

    @unittest.skipUnless(shutil.which("node"), "Node is needed to execute the real UI interval helpers")
    def test_video_list_modes_filter_navigation_and_draft_preservation(self):
        script = annotator.PAGE_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
        kinds = re.search(r"var KIND_CN = .*?;", script).group()
        helpers = script.split("// ---- pure interval helpers", 1)[1].split("// ---- end pure interval helpers", 1)[0].split("\n", 1)[1]
        rendering = "function renderVideoOptions() {" + script.split("function renderVideoOptions() {", 1)[1].split("function navigateVideo(", 1)[0]
        navigation = "function navigateVideo(" + script.split("function navigateVideo(", 1)[1].split("function renderReview()", 1)[0]
        switching = "function switchVideo(delta) {" + script.split("function switchVideo(delta) {", 1)[1].split("function buildStrip()", 1)[0]
        checks = r"""
const assert = require('node:assert/strict');
function video(name, counts) { return Object.assign({name, n_pos:0, n_neg:0, n_transition:0, n_cut:0, n_unknown:0, status:'annotated'}, counts); }
var videoItems = [video('n',{n_neg:1}), video('n-cut',{n_neg:2,n_cut:1}), video('cut-only',{n_cut:1}),
                  video('p',{n_neg:1,n_pos:1}), video('t',{n_neg:1,n_transition:1}),
                  video('unknown',{n_neg:1,n_unknown:1}), video('empty',{})];
var reviewItems = [{unit_id:'p',regions:[{start:4}]},{unit_id:'n',regions:[]},{unit_id:'missing',regions:[]}];
assert.deepEqual(videosForMode(videoItems,reviewItems,'all_n').map(v=>v.name),['n','n-cut']);
assert.equal(videosForMode(videoItems,reviewItems,'all').length,7);
assert.deepEqual(videosForMode(videoItems,reviewItems,'review').map(v=>v.name),['p','n']);
assert.equal(parseListMode('?review=1'),'review');
assert.equal(parseListMode('?review=1&list=all'),'all');
assert.equal(parseListMode('?list=all_n'),'all_n');
var state = {events:[{start:0,end:10,kind:'neg',note:'kept'}],ts:[0,2],idx:1,pendingStart:1,pendingEnd:2};
var currentVideo='n', listMode='all',reviewOnly=false,editDrafts=new Map(),pendingTouched=true,allowNavigation=false;
var saveChain=Promise.resolve(true), location={href:null}, history={changes:[],replaceState(a,b,url){this.changes.push(url);}};
var elements = {};
function $(id) {
  if (!elements[id]) { elements[id]={value:'',textContent:'',title:'',options:[],appendChild(option){this.options.push(option);}};
    Object.defineProperty(elements[id],'innerHTML',{set(){this.options=[];}});
    Object.defineProperty(elements[id],'selectedIndex',{get(){return this.options.findIndex(o=>o.value===this.value);}});
  } return elements[id];
}
var document={createElement:()=>({})},toast=()=>{},confirm=()=>false;
async function checkModes() {
 const draftEvent=state.events[0];
 editDrafts.set(draftEvent,{start:'0',end:'10',kind:'transition',note:'kept'});
 assert.equal(await setListMode('all_n'),true);
 assert.equal(listMode,'all_n');
 assert.equal($('list-count').textContent,'2 条');
 assert.deepEqual($('video-sel').options.map(o=>o.value),['n','n-cut']);
 assert.equal(state.pendingStart,1);
 assert.equal(editDrafts.get(draftEvent).kind,'transition'); // Same-video mode changes keep all drafts.
 assert.equal(history.changes.at(-1),'/?v=n&list=all_n&t=2');
 assert.equal(await setListMode('review'),true);
 assert.deepEqual($('video-sel').options.map(o=>o.value),['p','n']);
 assert.match(history.changes.at(-1),/list=review&review=1/);
 currentVideo='p';state.events=[{start:0,end:10,kind:'pos',note:''}];
 let release;saveChain=new Promise(resolve=>release=resolve);
 const cancelled=setListMode('all_n');
 assert.equal(location.href,null);
 release(true);
 assert.equal(await cancelled,false); // Out-of-list navigation waits for saving, then honors cancellation.
 assert.equal(listMode,'review');assert.equal($('list-sel').value,'review');
 assert.equal(state.pendingStart,1);assert.equal(editDrafts.get(draftEvent).kind,'transition');
 editDrafts.clear();state.pendingStart=state.pendingEnd=null;pendingTouched=false;
 let finish;saveChain=new Promise(resolve=>finish=resolve);
 const changed=setListMode('all_n');
 await Promise.resolve();assert.equal(location.href,null);
 finish(true);assert.equal(await changed,true);
 assert.equal(location.href,'/?v=n&list=all_n');
 location.href=null;currentVideo='n';listMode='all_n';renderVideoOptions();
 switchVideo(1);await Promise.resolve();
 assert.equal(location.href,'/?v=n-cut&list=all_n'); // Next follows only the filtered dropdown.
 location.href=null;saveChain=Promise.resolve(false);
 assert.equal(await setListMode('all'),false);assert.equal(location.href,null);
}
checkModes().catch(error=>{console.error(error);process.exitCode=1;});
"""
        result = subprocess.run([shutil.which("node"), "-"], input=kinds + helpers + rendering + navigation + switching + checks,
                                text=True, encoding="utf-8", capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node is needed to execute the real UI interval helpers")
    def test_range_relabel_preview_notes_boundaries_and_transaction_undo(self):
        script = annotator.PAGE_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
        kinds = re.search(r"var KIND_CN = .*?;", script).group()
        helpers = script.split("// ---- pure interval helpers", 1)[1].split("// ---- end pure interval helpers", 1)[0].split("\n", 1)[1]
        operations = script.split("// ---- edit transactions", 1)[1].split("// 整段标负", 1)[0].split("\n", 1)[1]
        checks = r"""
const assert = require('node:assert/strict');
const longNote = '原始长备注'.repeat(70);
const original = [{start: 0, end: 100, kind: 'neg', note: longNote}];
var state = {events: cloneEvents(original), ts: [0, 40, 60, 100], idx: 1, duration: 200,
             pendingStart: null, pendingEnd: null};
var editDrafts = new Map(), selectedEvent = null, undoStack = [], pendingTouched = false;
var snapshots = [], messages = [];
var renderEvents = () => {}, renderPending = () => {}, toast = message => messages.push(message);
var $ = () => ({value: 'transition'});
var save = () => { snapshots.push(cloneEvents(state.events)); return Promise.resolve(true); };
setStart(); state.idx = 2; setEnd();
assert.equal(hasPendingRange(), true);
assert.deepEqual(state.events, original);
assert.equal(snapshots.length, 0); // S/E only preview, never save.
applyPendingRange();
const painted = [
  {start: 0, end: 40, kind: 'neg', note: longNote},
  {start: 40, end: 60, kind: 'transition', note: longNote},
  {start: 60, end: 100, kind: 'neg', note: longNote}
];
assert.deepEqual(state.events, painted);
assert.equal(hasPendingRange(), false);
assert.equal(snapshots.length, 1);
assert.ok(snapshots[0].every(event => event.note === longNote));
const changed = {...state.events[0], kind: 'cut'};
commitEvents(editedEvents(state.events, 0, changed), changed, '修改区间');
undoLastEdit();
assert.deepEqual(state.events, painted); // Undo the latest edit, not the earlier range.
undoLastEdit();
assert.deepEqual(state.events, original);
assert.deepEqual(snapshots.at(-1), original);
assert.equal(undoStack.length, 0);
const mixed = [{start: 0, end: 100, kind: 'neg', note: 'N来源'},
               {start: 100, end: 200, kind: 'pos', note: 'P来源'}];
assert.deepEqual(relabelRange(mixed, 80, 120, 'transition', 200), [
 {start: 0, end: 80, kind: 'neg', note: 'N来源'},
 {start: 80, end: 100, kind: 'transition', note: 'N来源'},
 {start: 100, end: 120, kind: 'transition', note: 'P来源'},
 {start: 120, end: 200, kind: 'pos', note: 'P来源'}]);
assert.deepEqual(relabelRange(mixed, 120, 80, 'transition', 200), relabelRange(mixed, 80, 120, 'transition', 200));
assert.deepEqual(relabelRange(original, 40.125, 60.375, 'transition', 200).map(e => [e.start,e.end]),
                 [[0,40.125], [40.125,60.375], [60.375,100]]);
const holes = [{start: 0, end: 1, kind: 'neg', note: 'A'}, {start: 2, end: 3, kind: 'cut', note: 'B'}];
const filled = relabelRange(holes, 0.5, 2.5, 'transition', 4);
assert.equal(classAt(filled, 1.5), 'transition');
assert.equal(classAt(filled, 3.5), 'unknown');
assert.equal(classAt(filled, 2.5), 'cut');
assert.equal(filled.find(e => e.start === 2).note, 'B'); // cut overwrite is explicit and note survives.
assert.equal(filled.find(e => e.start === 1).note, '');
for (const [s,e] of [[1,1], [-1,2], [0,NaN], [0,Infinity], [0,201], [null,1], ['',1]]) {
  assert.throws(() => relabelRange(original, s, e, 'transition', 200));
}
assert.throws(() => relabelRange(original, 1, 2, 'alien', 200));
const unknown = [{start: 0, end: 10, kind: 'alien', note: 'unknown origin'}];
assert.throws(() => relabelRange(unknown, 2, 4, 'transition', 10)); // Remaining unknown is refused, never P.
assert.equal(relabelRange(unknown, 0, 10, 'transition', 10)[0].note, 'unknown origin');
state.pendingStart = 1; state.pendingEnd = 2; pendingTouched = true; clearPending();
assert.equal(hasPendingRange(), false);
assert.deepEqual(state.events, original);
"""
        result = subprocess.run([shutil.which("node"), "-"], input=kinds + helpers + operations + checks,
                                text=True, encoding="utf-8", capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node is needed to execute the real UI interval helpers")
    def test_ui_half_open_gaps_edit_split_and_javascript_syntax(self):
        script = annotator.PAGE_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
        kinds = re.search(r"var KIND_CN = .*?;", script).group()
        helpers = script.split("// ---- pure interval helpers", 1)[1].split("// ---- end pure interval helpers", 1)[0]
        helpers = helpers.split("\n", 1)[1]
        checks = r"""
const assert = require('node:assert/strict');
const events = [
  {start: 0.125, end: 1.375, kind: 'neg', note: 'preserve'},
  {start: 1.375, end: 2.125, kind: 'transition', note: 'weak'},
  {start: 2.125, end: 4.125, kind: 'pos', note: 'clear'}
];
assert.equal(classAt(events, 1.3749), 'neg');
assert.equal(classAt(events, 1.375), 'transition');
assert.equal(classAt(events, 2.125), 'pos');
assert.equal(classAt(events, 4.125), 'unknown');
assert.equal(classAt([{start: 0, end: 1, kind: 'alien'}], 0.5), 'unknown');
const preview = windowCoverage(events, 0.625, 3.625);
assert.deepEqual(preview.coverage, {neg: 0.75, transition: 0.75, pos: 1.5, cut: 0, unknown: 0});
assert.equal(preview.crosses, true);
const gaps = [{start: 0, end: 1, kind: 'neg'}, {start: 2, end: 3, kind: 'cut'}];
assert.equal(classAt(gaps, 1.5), 'unknown');
assert.equal(classAt([{start:0,end:1,kind:'unknown'}],0.5),'unknown');
assert.equal(editedEvents(events,1,{...events[1],kind:'unknown'})[1].kind,'unknown');
assert.deepEqual(relabelRange([{start:0,end:3,kind:'neg',note:'keep'}],1,2,'unknown',3).map(e=>e.kind),['neg','unknown','neg']);
assert.deepEqual(windowCoverage(gaps, 0, 3).coverage, {neg: 1, transition: 0, pos: 0, cut: 1, unknown: 1});
const parts = splitEvents(events, 2, 3.125);
assert.equal(parts.length, 4);
assert.equal(parts[2].end, parts[3].start);
assert.equal(parts[2].note, 'clear');
assert.equal(parts[3].note, 'clear');
const changed = editedEvents(parts, 2, {...parts[2], kind: 'transition'});
assert.equal(changed[2].kind, 'transition');
assert.equal(changed[2].start, 2.125);
assert.equal(changed[2].end, 3.125);
assert.throws(() => editedEvents(events, 1, {...events[1], start: 1.2}), /重叠/);
assert.throws(() => splitEvents(events, 1, 1.375));
assert.throws(() => splitEvents(events, 1, 2.125));
"""
        save_function = "function save() {" + script.split("function save() {", 1)[1].split("// ---- 渲染", 1)[0]
        navigate_function = "function navigateVideo(" + script.split("function navigateVideo(", 1)[1].split("function renderReview()", 1)[0]
        queue_checks = r"""
var state = {events: [{start: 0, end: 1, kind: 'pos'}], pendingStart: null, pendingEnd: null};
var saveChain = Promise.resolve(true), savePending = 0;
var editDrafts = new Map(), saveFailed = false, allowNavigation = false, pendingTouched = false;
var reviewItems = [], reviewOnly = false, listMode = 'all', location = {href: null}, currentVideo = 'key1';
var api = p => p, $ = () => ({}), toast = () => {};
const calls = [], completions = [];
var fetch = (path, options) => {
  calls.push(JSON.parse(options.body));
  return new Promise(resolve => completions.push(resolve));
};
async function checkSaveQueue() {
  const first = save();
  state.events[0].kind = 'transition';
  const second = save();
  navigateVideo('key2');
  await Promise.resolve();
  assert.equal(calls.length, 1);
  assert.equal(calls[0].events[0].kind, 'pos');
  assert.equal(location.href, null);
  completions[0]({status: 200, json: () => Promise.resolve({saved_at: 'one'})});
  await first;
  await Promise.resolve();
  assert.equal(calls.length, 2);
  assert.equal(calls[1].events[0].kind, 'transition');
  assert.equal(location.href, null);
  completions[1]({status: 200, json: () => Promise.resolve({saved_at: 'two'})});
  assert.equal(await second, true);
  await Promise.resolve();
  assert.equal(savePending, 0);
  assert.equal(location.href, '/?v=key2');
  location.href = null;
  const failed = save();
  navigateVideo('key1');
  await Promise.resolve();
  completions[2]({status: 400, json: () => Promise.resolve({error: 'rejected'})});
  assert.equal(await failed, false);
  await Promise.resolve();
  assert.equal(location.href, null);
  saveChain = Promise.resolve(true);
  const draftEvent = {start: 0, end: 1, kind: 'neg', note: ''};
  editDrafts.set(draftEvent, {start: '0', end: '1', kind: 'transition', note: ''});
  assert.equal(hasUnsavedDrafts(), true);
  globalThis.confirm = () => false;
  navigateVideo('key2');
  await Promise.resolve();
  assert.equal(location.href, null);
  assert.equal(editDrafts.get(draftEvent).kind, 'transition');
  editDrafts.clear(); state.pendingStart = 0.125; state.pendingEnd = 2; pendingTouched = true;
  navigateVideo('key2');
  await Promise.resolve();
  assert.equal(location.href, null); // Unapplied S/E ranges also warn before leaving.
  assert.equal(state.pendingStart, 0.125);
}
checkSaveQueue().catch(error => { console.error(error); process.exitCode = 1; });
"""
        execution = ("new Function(" + json.dumps(script) + ");\n" + kinds + "\n" + helpers + checks
                     + save_function + navigate_function + queue_checks)
        result = subprocess.run([shutil.which("node"), "-"], input=execution, text=True,
                                encoding="utf-8", capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
