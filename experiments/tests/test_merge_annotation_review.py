import json
from pathlib import Path
import tempfile
import unittest

from experiments.tools.merge_annotation_review import merge, replace_unit, sha


class ReviewMergeTests(unittest.TestCase):
    def test_cut_cannot_be_replaced_by_a_semantic_class(self):
        original = [{'start_sec': '0', 'end_sec': '2', 'kind': 'cut'},
                    {'start_sec': '2', 'end_sec': '4', 'kind': 'pos'}]
        with self.assertRaisesRegex(ValueError, 'cut coverage'):
            replace_unit(original, [{'start_sec': 0, 'end_sec': 4, 'state': 'T'}])

    def test_unlabelled_gap_is_not_filled(self):
        original = [{'start_sec': '0', 'end_sec': '2', 'kind': 'neg'},
                    {'start_sec': '3', 'end_sec': '4', 'kind': 'pos'}]
        with self.assertRaisesRegex(ValueError, 'gap filling'):
            replace_unit(original, [{'start_sec': 0, 'end_sec': 4, 'state': 'N'}])

    def test_live_edit_and_historical_output_are_never_overwritten(self):
        temp_root = Path(__file__).resolve().parents[2] / '.codex/docs/.temp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(temp_root), prefix='merge-test-') as directory:
            root = Path(directory)
            baseline, current, out = root / 'base', root / 'current', root / 'out'
            baseline.mkdir()
            current.mkdir()
            raw = b'video_id,video_key,start_sec,end_sec,confidence,kind,note\r\nv,u,0,4,real,pos,keep,legacy-extra\r\n'
            for folder in (baseline, current):
                (folder / 'events.csv').write_bytes(raw)
                (folder / 'sources.json').write_text(json.dumps({'sources': {'key': {
                    'annotations_csv': 'events.csv', 'initial_annotations_sha256': sha(raw)}}}), encoding='utf-8')
            proposals = root / 'proposals.json'
            proposals.write_text(json.dumps({'items': [{'unit_id': 'u', 'recommended_intervals': [
                {'start_sec': 0, 'end_sec': 4, 'state': 'T'}]}]}), encoding='utf-8')
            (current / 'events.csv').write_bytes(raw + b'\r\n')
            with self.assertRaisesRegex(ValueError, 'new edits'):
                merge(baseline, current, proposals, out, execute=True)
            self.assertFalse(out.exists())
            (current / 'events.csv').write_bytes(raw)
            self.assertTrue(merge(baseline, current, proposals, out)['dry_run'])
            self.assertFalse(out.exists())
            merge(baseline, current, proposals, out, execute=True)
            self.assertIn(b'transition', (out / 'events.csv').read_bytes())
            self.assertIn(b'keep,legacy-extra', (out / 'events.csv').read_bytes())
            self.assertEqual((baseline / 'events.csv').read_bytes(), raw)
            with self.assertRaises(FileExistsError):
                merge(baseline, current, proposals, out, execute=True)


if __name__ == '__main__':
    unittest.main()
