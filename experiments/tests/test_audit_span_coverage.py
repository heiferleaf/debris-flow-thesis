"""Finite synthetic metadata fixtures; no real pool scan or media access."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('span_audit_fixture',ROOT/'experiments/tools/audit_span_coverage.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class SpanCoverageTests(unittest.TestCase):
    def test_three_seconds_is_the_existing_full_enumerator(self):
        pts=np.arange(0,12.0001,.05);clean=[[0,81],[100,len(pts)]]
        counts,witnesses,three,_=audit.audit_segment(pts,clean,0.,12.01,chunk_size=7)
        expected=audit.segment_candidates(pts,clean,0.,12.01)
        self.assertTrue(np.array_equal(three,expected))
        self.assertEqual(counts['candidates_3s_sha256'],audit.tuple_digest(expected))
        self.assertEqual(counts['n_candidates_3s'],len(expected))
        self.assertGreater(len(witnesses['three_seconds']),0)
        self.assertEqual(counts['n_candidates_7p5s'],0)  # Neither clean run is7.5s.

    def test_sampling_half_open_boundary_and_short_runs(self):
        pts=np.arange(0,8.0001,.05);clean=[[0,len(pts)]]
        short,_,_,_=audit.audit_segment(pts,clean,0.,3.)
        exact,_,_,_=audit.audit_segment(pts,clean,0.,3.05)
        before,_,_,_=audit.audit_segment(pts,clean,0.,7.5)
        after,witnesses,_,_=audit.audit_segment(pts,clean,0.,7.55)
        self.assertEqual(short['n_candidates_3s'],0)
        self.assertGreater(exact['n_candidates_3s'],0)
        self.assertEqual(before['n_candidates_7p5s'],0)
        self.assertGreater(after['n_candidates_7p5s'],0)
        for w in witnesses['common_nominal_center_pairs']:
            self.assertLess(w['long']['source_pts_seconds'][-1],7.55)
            self.assertEqual(w['short']['target_seconds'][0],w['long']['target_seconds'][0]+2.25)
            self.assertAlmostEqual((w['short']['target_seconds'][0]+w['short']['target_seconds'][-1])/2,w['nominal_center_seconds'])

    def test_nearest_left_tie_error_and_strict_actual_indices_pts(self):
        local=np.concatenate(([0.,.125],.0625+np.linspace(0,3.,16)[1:]))
        ix,good=audit.nearest_many(local,np.array([.0625]),3.)
        self.assertTrue(good[0]);self.assertEqual(ix[0,0],0)
        _,bad=audit.nearest_many(np.arange(0,8.0001,.4),np.array([0.]),7.5)
        self.assertFalse(bad[0])
        repeated=np.repeat(np.arange(0,3.0001,.2),2)
        ix,good=audit.nearest_many(repeated,np.array([0.]),3.)
        self.assertTrue(good[0]);self.assertTrue(np.all(np.diff(ix[0])>0))
        self.assertTrue(np.all(np.diff(repeated[ix[0]])>0))
        for local,starts in ((np.array([0.,float('nan')]),np.array([0.])),(np.array([0.,1.,.5]),np.array([0.])),(np.array([0.,1.]),np.array([float('inf')]))):
            with self.assertRaises(ValueError):audit.nearest_many(local,starts,3.)

    def test_backstep_outside_separate_runs_is_not_globally_sorted(self):
        pts=np.concatenate((np.arange(0,4.0001,.05),np.arange(0,4.0001,.05)))
        n=len(pts)//2
        counts,_,_,_=audit.audit_segment(pts,[[0,n],[n,len(pts)]],0.,4.1)
        self.assertEqual(counts['n_candidates_7p5s'],0)
        self.assertGreater(counts['n_candidates_3s'],0)
        with self.assertRaises(ValueError):audit.audit_segment(pts,[[0,len(pts)]],0.,4.1)

    def test_different_label_segments_are_never_joined(self):
        pts=np.arange(0,8.0001,.05);clean=[[0,len(pts)]]
        p,_,_,_=audit.audit_segment(pts,clean,0.,4.)
        n,_,_,_=audit.audit_segment(pts,clean,4.,8.05)
        self.assertGreater(p['n_candidates_3s'],0);self.assertGreater(n['n_candidates_3s'],0)
        self.assertEqual(p['n_candidates_7p5s'],0);self.assertEqual(n['n_candidates_7p5s'],0)
        merged,_,_,_=audit.audit_segment(pts,clean,0.,8.05)
        self.assertGreater(merged['n_candidates_7p5s'],0)  # Demonstrates why labels cannot be merged.

    def test_aliases_are_paired_before_long_dedup(self):
        long=np.tile(np.arange(16,dtype=np.int32),(2,1))
        short=np.array([np.arange(16)+100,np.arange(16)+200],dtype=np.int32)
        anchors=np.array([0.,.01]);lg=np.array([True,True])
        l,la,p,pa=audit.alias_records(long,short,lg,np.array([False,True]),anchors)
        unique,_=audit._unique([l],[la],16);pairs,_=audit._unique([p],[pa],32)
        self.assertEqual(len(unique),1);self.assertEqual(len(pairs),1)
        self.assertTrue(np.array_equal(pairs[0,:16],short[1]))
        l,la,p,pa=audit.alias_records(long,short,lg,np.array([True,True]),anchors)
        self.assertEqual(len(audit._unique([l],[la],16)[0]),1)
        self.assertEqual(len(audit._unique([p],[pa],32)[0]),2)

    def test_zero_long_is_valid_and_all_video_class_states_are_explicit(self):
        v=dict(unit_id='tiny',split='train',group_id='g',video_label='pos')
        empty=audit.class_row(v,'neg',[],set())
        self.assertEqual(empty['coverage_state'],'no_supervision_segment')
        segment=dict(origin='explicit_human_event',longest_clean_pts_span_seconds=3.,n_candidates_3s=1,n_candidates_7p5s=0,n_common_center_pairs=0)
        row=audit.class_row(v,'pos',[segment],{'tiny'})
        self.assertEqual(row['coverage_state'],'zero_7p5s_candidates')
        segment.update(n_candidates_7p5s=1)
        self.assertEqual(audit.class_row(v,'pos',[segment],set())['coverage_state'],'separately_available_but_zero_common_center_pairs')

    def test_metadata_allowlist_overwrite_and_fingerprint_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            for name in ('data/raw/source.mp4','data/process/anchor.jpg','../outside.csv','C:/absolute.csv'):
                with self.assertRaises(ValueError):audit.metadata_path(root,name)
            out=root/'data/metadata/span-coverage/existing';out.mkdir(parents=True)
            with self.assertRaises(FileExistsError):audit.run(root,out)
            base=root/audit.BASE;base.parent.mkdir(parents=True,exist_ok=True);base.write_text('{}',encoding='utf-8')
            out=root/'data/metadata/span-coverage/fingerprint-failure'
            with self.assertRaises(ValueError):audit.run(root,out)
            failure=json.loads((out/'failure.json').read_text(encoding='utf-8'))
            self.assertEqual(failure['status'],'failed')
            self.assertEqual(base.read_text(encoding='utf-8'),'{}')


if __name__=='__main__':unittest.main()
