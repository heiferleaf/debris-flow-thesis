"""Finite CPU/JS contracts for delivery-frame replay; no real-video decoding."""
from pathlib import Path
from copy import deepcopy
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.common.video import observability_temporal as temporal


class TemporalTests(unittest.TestCase):
    def test_range_is_inclusive_exact_and_keeps_2_96_duration(self):
        ix = list(range(100, 138)); pts = np.arange(38) * .08 + 250.
        positions = np.rint(np.linspace(0, 37, 16)).astype(int)
        anchors = [ix[i] for i in positions]
        sampled = [float(pts[i]) for i in positions]
        chosen, span = temporal.validate_range(ix, pts, anchors, sampled, [[90, 150]])
        self.assertEqual(chosen, positions.tolist()); self.assertAlmostEqual(span, 2.96)
        relative = [float(t - pts[0]) for t in pts]
        for i, t in enumerate(relative):
            self.assertEqual(temporal.index_at(relative, t), i)
            if i + 1 < len(relative): self.assertEqual(temporal.index_at(relative, (t + relative[i+1]) / 2), i)
        self.assertEqual(temporal.index_at(relative, 100), 37)
        self.assertEqual(temporal.index_at(relative, -1), 0)
        for mutated in (ix[:-1], ix[:5] + ix[6:], ix[::-1]):
            with self.assertRaises(ValueError): temporal.validate_range(mutated, pts, anchors, sampled, [[90, 150]])
        with self.assertRaises(ValueError): temporal.validate_range(ix, pts, anchors, sampled, [[90, 120], [120, 150]])
        with self.assertRaises(ValueError): temporal.index_at([0., .1, .1], .1)

    def test_legacy_jpeg_rgb256_linear112_anchor_and_quality_gates(self):
        bgr = np.random.default_rng(2).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        std = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).std())
        png, proof = temporal.processed_frame(bgr, std)
        encoded = temporal.jpeg95_bytes(bgr)
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        self.assertEqual(proof['jpeg_sha256'], temporal.digest_bytes(encoded))
        self.assertEqual(proof['rgb256_decoded_sha256'], temporal.digest_bytes(rgb.tobytes()))
        self.assertEqual(proof['rgb112_sha256'], temporal.digest_bytes(cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR).tobytes()))
        second, _ = temporal.processed_frame(bgr, std, proof)
        self.assertEqual(png, second)
        for key in ('jpeg_sha256', 'source_bgr_sha256', 'rgb256_decoded_sha256', 'rgb112_sha256'):
            wrong = dict(proof); wrong[key] = '0' * 64
            with self.assertRaises(ValueError): temporal.processed_frame(bgr, std, wrong)
        with self.assertRaises(ValueError): temporal.processed_frame(bgr, std + .1)
        with self.assertRaises(ValueError): temporal.processed_frame(np.zeros_like(bgr), 0.)

    def test_physical_counter_pts_shape_and_complete_set(self):
        audit = temporal.PhysicalAudit([0., .08, .16])
        for i, t in enumerate([0., .08, .16]):
            audit.observe('[showinfo] n: %d pts: 0 pts_time: %s s:256x256' % (i, t), {1, 2})
        audit.finish({1, 2})
        self.assertEqual(audit.selected_pts, {1: .08, 2: .16})
        for bad in ('n: 1 pts: 0 pts_time: 0 s:256x256', 'n: 0 pts: 0 pts_time: .01 s:256x256',
                    'n: 0 pts: 0 pts_time: 0 s:255x256', 'error while decoding'):
            audit = temporal.PhysicalAudit([0.]); audit.observe(bad, {0})
            with self.assertRaises(ValueError): audit.finish({0})
        audit = temporal.PhysicalAudit([0., .08]); audit.observe('n: 0 pts: 0 pts_time: 0 s:256x256', {0})
        with self.assertRaises(ValueError): audit.finish({0})

    def test_exact_byte_reservation_and_deadline_kill_blocked_process(self):
        temp_root = ROOT / '.codex/docs/.temp/temporal-fixtures'; temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            out = Path(folder); clock = [0.]
            budget = temporal.Budget(out, seconds=1., byte_limit=1_000_100, clock=lambda: clock[0])
            budget.write(out / 'one.bin', b'x' * 100)
            with self.assertRaises(temporal.LimitReached): budget.write(out / 'over.bin', b'x')
            self.assertFalse((out / 'over.bin').exists())
            budget.write(out / 'failure.json', b'{}', audit=True)
            self.assertEqual(budget.used, 102)
            with self.assertRaises(ValueError): budget.write(out.parent / 'escape.bin', b'')
            deadline = temporal.Budget(out, seconds=.15, byte_limit=2_000_000)
            # Use one actual executable, like ffmpeg.exe. Windows venv python
            # is a launcher; killing only that launcher can leave its Python
            # child holding stdout and does not test a direct decoder process.
            child = subprocess.Popen(['node', '-e', 'setTimeout(()=>{},5000)'], stdout=subprocess.PIPE)
            deadline.process = child
            watch = threading.Thread(target=deadline.watchdog, daemon=True); watch.start()
            started = time.monotonic()
            try:
                self.assertEqual(child.communicate(timeout=2)[0], b''); child.wait(timeout=1)
                self.assertLess(time.monotonic() - started, 2.)
                with self.assertRaises(temporal.LimitReached): deadline.check()
            finally:
                deadline.stop.set(); watch.join(timeout=1)
                if child.poll() is None: child.kill(); child.wait()
                child.stdout.close()

    def test_actual_js_timelines_independent_completion_and_parallel_gate(self):
        player = temporal.HTML.split('<script>')[1].split('</script>')[0]
        fixture = {'cases': [{'case_id': 'C01', 'duration_seconds': 2.96, 'conditions': [
            {'condition_id': 'K1', 'relative_pts_seconds': [0., 1., 2.96], 'frames': [{'path': 'a'}] * 3},
            {'condition_id': 'K2', 'relative_pts_seconds': [0., .4, 1., 2., 2.96], 'frames': [{'path': 'b'}] * 5}]}]}
        environment = r'''
const assert=require('assert');let wall=0,callback=null;const elements=new Map();
global.window={TEMPORAL_REVIEW:DATA};global.devicePixelRatio=1;global.performance={now:()=>wall};
global.document={hidden:false,createElement:()=>({}),addEventListener:()=>{},getElementById:id=>{
if(!elements.has(id))elements.set(id,{disabled:false,hidden:false,value:0,textContent:'',append:()=>{},
getBoundingClientRect:()=>({width:256,height:256}),getContext:()=>({clearRect:()=>{},drawImage:()=>{},imageSmoothingEnabled:false})});return elements.get(id);}};
global.requestAnimationFrame=f=>{callback=f;return 1;};global.cancelAnimationFrame=()=>{callback=null;};
global.Image=class{set src(x){this.naturalWidth=256;this.naturalHeight=256;queueMicrotask(()=>this.onload());}};
'''.replace('DATA', json.dumps(fixture))
        checks = r'''
setImmediate(()=>{const p=window.temporalPlayer;
assert(p.snapshot().ready);p.compare();assert.equal(p.snapshot().mode,'single');
assert.equal(p.at([0,1,2.96],.5),0);assert.equal(p.at([0,.4,1,2,2.96],.5),1);
p.play();wall=2960;callback(wall);assert.deepEqual(p.snapshot().completed,[]);assert(!p.snapshot().playing);
assert.deepEqual(p.snapshot().displayed.K1,[0,2]);assert.deepEqual(p.snapshot().displayed.K2,[]);
p.compare();assert.equal(p.snapshot().mode,'single');
let base=wall;p.play();for(const t of [1,2.96]){wall=base+t*1000;callback(wall);}assert.deepEqual(p.snapshot().completed,['K1']);
p.condition('K2');p.seek(1);p.play();wall+=1960;callback(wall);assert.deepEqual(p.snapshot().completed,['K1']);
p.condition('K2');base=wall;p.play();for(const t of [.4,1,2,2.96]){wall=base+t*1000;callback(wall);}assert.deepEqual(p.snapshot().completed,['K1','K2']);
const coverage=JSON.stringify(p.snapshot().displayed);
p.compare();assert.equal(p.snapshot().mode,'compare');p.play();wall+=2960;callback(wall);
assert.equal(p.snapshot().elapsed,2.96);assert(!p.snapshot().playing);assert.equal(p.snapshot().trace.at(-1).frames[0],2);assert.equal(p.snapshot().trace.at(-1).frames[1],4);
assert.equal(JSON.stringify(p.snapshot().displayed),coverage);
assert.deepEqual(p.snapshot().canvas_raster,[256,256]);assert.deepEqual(p.snapshot().image_smoothing,[false,false]);
console.log('actual_JS_playback_fixture_passed');});
'''
        result = subprocess.run(['node', '-e', environment + player + checks], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('actual_JS_playback_fixture_passed', result.stdout)

    def test_unexpected_failures_stop_and_withhold_but_registered_missing_succeeds(self):
        path = ROOT / 'experiments/runspecs/EXP-018/prepare_temporal_observability.py'
        spec = importlib.util.spec_from_file_location('temporal_entry_fixture', path)
        entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
        contract = dict(format_version=2, status='frozen_pending_code_review_no_real_decode',
            hard_limits=dict(seconds=temporal.MAX_SECONDS, output_bytes=temporal.MAX_BYTES),
            ffmpeg={'path': 'fixture-does-not-run-ffmpeg', 'sha256': '0' * 64},
            pre_registered_missing_cases={c: {'reason': 'fixture_preflight_duplicate_pts'} for c in ('C02', 'C05')},
            cases=[dict(case_id='C%02d' % i, anchors=[], actual_span_seconds=3.) for i in range(1, 13)])
        temp_root = ROOT / '.codex/docs/.temp/temporal-execute-fixtures'; temp_root.mkdir(parents=True, exist_ok=True)
        for failure in ('contract_sha', 'initial_fingerprint', 'C01', 'C03', 'budget', 'final_fingerprint', 'final_torch', None):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(dir=temp_root) as folder:
                out = Path(folder); private = out / 'private'; private.mkdir()
                contract_path = private / 'frozen-contract.json'
                contract_path.write_text(json.dumps(contract), encoding='utf-8')
                calls, verifications = [], []
                def verify(*_):
                    verifications.append(1)
                    if failure == 'initial_fingerprint' or (failure == 'final_fingerprint' and len(verifications) == 2):
                        raise ValueError('fixture fingerprint refusal')
                    if failure == 'final_torch' and len(verifications) == 2:
                        sys.modules['torch'] = object()
                def extract(_root, case, *_):
                    calls.append(case['case_id'])
                    if failure == case['case_id']: raise ValueError('fixture unexpected case failure')
                    if failure == 'budget': raise temporal.LimitReached('fixture budget refusal')
                    return dict(status='passed', case_id=case['case_id'], frames=[{'fixture': 1}] * 31)
                with patch.object(entry, 'verify_inputs', side_effect=verify), \
                     patch.object(entry, 'load_tables', return_value=None), \
                     patch.object(temporal, 'extract_case', side_effect=extract), \
                     patch.object(entry, 'publish_case', side_effect=lambda _o, c, _p, _b: {'case_id': c['case_id']}) as publish, \
                     patch.object(entry.subprocess, 'run', return_value=SimpleNamespace(stdout='ffmpeg version9.0.2\n')), \
                     patch.dict(sys.modules, {}, clear=False):
                    expected_sha = '0' * 64 if failure == 'contract_sha' else temporal.digest_file(contract_path)
                    result = entry.execute(ROOT, out, expected_sha)
                public = json.loads((out / 'viewer/manifest.json').read_text(encoding='utf-8'))
                saved = json.loads((private / 'execution-summary.json').read_text(encoding='utf-8'))
                self.assertEqual(result, saved)
                if failure is None:
                    self.assertTrue(result['execution_success']); self.assertEqual(len(public['cases']), 10)
                    self.assertEqual(result['verified_anchors'], 160)
                    self.assertEqual(calls[0], 'C01'); self.assertNotIn('C02', calls); self.assertNotIn('C05', calls)
                    self.assertEqual(publish.call_count, 10)
                else:
                    self.assertFalse(result['execution_success']); self.assertTrue(result['unexpected_failure'])
                    self.assertEqual(public['cases'], []); self.assertFalse(publish.called)
                    if failure in ('C01', 'budget'): self.assertEqual(calls, ['C01'])
                    if failure == 'C03': self.assertEqual(calls, ['C01', 'C03'])
                    if failure in ('contract_sha', 'initial_fingerprint'): self.assertEqual(calls, [])

    def test_cli_propagates_measurement_failure_exit_code(self):
        path = ROOT / 'experiments/runspecs/EXP-018/prepare_temporal_observability.py'
        spec = importlib.util.spec_from_file_location('temporal_cli_fixture', path)
        entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
        argv = [str(path), '--stage', 'execute', '--contract-sha256', '0' * 64]
        with patch.object(sys, 'argv', argv), patch.object(entry, 'execute', return_value={'execution_success': False}):
            with self.assertRaises(SystemExit) as caught: entry.main()
            self.assertEqual(caught.exception.code, 1)


if __name__ == '__main__':
    unittest.main()
