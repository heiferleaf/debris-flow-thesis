"""Small mathematical and metadata-only fixture for the time-composition contract."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common.video.window_composition import (  # noqa: E402
    Interval, compose_window, composition_class, dominant_class, normalize_intervals, reconstruct_fraction, soft_target,
)
from experiments.tools.audit_window_composition import audit  # noqa: E402


class WindowCompositionTest(unittest.TestCase):
    def test_anchor_targets_and_general_anchor_reconstruction(self):
        expected = {0: (1, 0, 0), .0625: (.875, .125, 0), .5: (0, 1, 0), 1: (0, 0, 1)}
        for fraction, target in expected.items():
            self.assertEqual(soft_target(fraction), target)
        for anchor in (.4, .5, .6):
            for fraction in (0, .0625, .25, .4, .5, .6, .75, 1):
                target = soft_target(fraction, anchor)
                self.assertTrue(all(value >= 0 for value in target))
                self.assertAlmostEqual(sum(target), 1)
                self.assertAlmostEqual(reconstruct_fraction(target, anchor), fraction)
        self.assertEqual(dominant_class(soft_target(.25)), "T")
        self.assertEqual(dominant_class(soft_target(.75)), "T")
        self.assertEqual(dominant_class(soft_target(.8, .6)), "T")
        self.assertEqual(dominant_class(soft_target(0)), "N")
        self.assertEqual(dominant_class(soft_target(1)), "P")
        with self.assertRaises(ValueError):
            soft_target(.5, 1)

    def test_fp32_predictions_and_composition_display_are_separate(self):
        # Standard-library conversion reproduces actual IEEE754 FP32 values.
        fp32 = tuple(struct.unpack("f", struct.pack("f", value))[0] for value in (.1, .2, .7))
        self.assertEqual(sum(fp32), .9999999925494194)
        self.assertAlmostEqual(reconstruct_fraction(fp32), .8, places=6)
        self.assertEqual(composition_class(fp32), "P")
        self.assertEqual(dominant_class((.4, .2, .4)), "N")
        self.assertEqual(reconstruct_fraction((.4, .2, .4)), .5)
        self.assertEqual(composition_class((.4, .2, .4)), "T")
        for fraction in (.25, .75):
            self.assertEqual(composition_class(soft_target(fraction)), "T")
        for anchor in (.4, .6):
            for fraction in (0, anchor / 2, anchor, (1 + anchor) / 2, 1):
                self.assertEqual(composition_class(soft_target(fraction, anchor), anchor),
                                 dominant_class(soft_target(fraction, anchor)))
        # The normalization tolerance can produce a tiny decoded overshoot;
        # it is endpoint roundoff rather than permission to extrapolate g(r).
        self.assertEqual(reconstruct_fraction((0, 5e-7, 1)), 1)
        self.assertEqual(composition_class((0, 5e-7, 1)), "P")
        for bad in ((float("nan"), .2, .8), (float("inf"), 0, 0),
                    (-1e-8, .2, .8), (.1, .2, .71), (1.000001, 0, 0), (.2, .3)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                reconstruct_fraction(bad)

    def test_real_time_is_not_sixteen_frame_vote(self):
        equal_time = compose_window(0, 16, [Interval(0, 15, "neg"), Interval(15, 16, "pos")])
        self.assertEqual(equal_time["target"], (.875, .125, 0))
        # The same 15 N + 1 P sampled frame labels can have a different duration.
        # Their counts alone do not supply this explicitly annotated .1s P span.
        uneven_time = compose_window(0, 10, [Interval(0, 9.9, "N"), Interval(9.9, 10, "P")])
        self.assertAlmostEqual(uneven_time["r"], .01)
        self.assertAlmostEqual(uneven_time["target"][1], .02)
        self.assertNotEqual(uneven_time["target"], equal_time["target"])

    def test_unknown_gaps_legacy_T_and_window_outside_annotation(self):
        for intervals in ([Interval(0, 15, "N"), Interval(15, 16, "transition")],
                          [Interval(0, 15, "N"), Interval(15, 16, "unknown")],
                          [Interval(0, 15, "N")]):
            composition = compose_window(0, 16, intervals)
            self.assertEqual(composition["status"], "U")
            self.assertIsNone(composition["target"])
            self.assertIsNone(composition["r"])
            self.assertEqual((composition["r_min"], composition["r_max"]), (0, 1 / 16))
        empty = compose_window(0, 2, [Interval(4, 5, "P")])
        self.assertEqual((empty["r_min"], empty["r_max"]), (0, 1))
        bounds = compose_window(0, 3, [Interval(0, 1, "P"), Interval(2, 3, "N")])
        self.assertEqual((bounds["r_min"], bounds["r_max"]), (1 / 3, 2 / 3))

    def test_cut_technical_veto_overlap_and_half_open_endpoint(self):
        cut = compose_window(0, 2, [Interval(0, 1, "N"), Interval(1, 2, "cut")])
        self.assertEqual(cut["status"], "technical_invalid")
        self.assertIn("untyped_cut", cut["reasons"])
        self.assertIsNone(cut["r_min"])
        self.assertIsNone(cut["r_max"])
        self.assertIsNone(cut["target"])
        veto = compose_window(0, 2, [Interval(0, 2, "P")], forbidden_ranges=[(.5, .6)])
        self.assertEqual(veto["status"], "technical_invalid")
        self.assertIsNone(veto["r"])
        edge = compose_window(0, 2, [Interval(0, 2, "N")], forbidden_ranges=[(2, 3)])
        self.assertEqual(edge["status"], "N")
        for intervals in ([Interval(0, 2, "N"), Interval(1, 3, "P")],
                          [Interval(0, 2, "N"), Interval(1, 3, "N")]):
            with self.assertRaises(ValueError):
                compose_window(0, 3, intervals)
        for start, end in ((0, 0), (float("nan"), 2), (0, float("inf"))):
            with self.assertRaises(ValueError):
                compose_window(start, end, [])

    def test_adjacent_same_label_csv_lines_do_not_change_target(self):
        contiguous = [Interval(0, .3, "N"), Interval(.3, 1, "P")]
        split = [Interval(0, .1, "neg"), Interval(.1, .3, "N"), Interval(.3, .9, "P"), Interval(.9, 1, "pos")]
        self.assertEqual(normalize_intervals(split), tuple(contiguous))
        self.assertEqual(compose_window(0, 1, contiguous), compose_window(0, 1, split))
        self.assertEqual(compose_window(0, 1, contiguous)["status"], "MIX")

    def test_metadata_fixture_dedup_gates_and_no_input_mutation(self):
        with tempfile.TemporaryDirectory(prefix="window-composition-") as directory:
            root = Path(directory)
            source = root / "data/process/video_frames/fixture"
            source.mkdir(parents=True)
            (source / "mapping.csv").write_text("video_key,video_id,excluded\na,source-a,0\n", encoding="utf-8")
            revision = root / "saved-v3"
            revision.mkdir()
            events = revision / "events.csv"
            with events.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["video_id", "video_key", "start_sec", "end_sec", "confidence", "kind", "note"])
                for start, end, kind in ((0, 5, "neg"), (5, 10, "pos"), (10, 14, "transition"), (14, 20, "cut")):
                    writer.writerow(["source-a", "a", start, end, "real", kind, ""])
            manifest = revision / "sources.json"
            manifest.write_text(json.dumps({"sources": {"fixture": {"source_id": "data/process/video_frames/fixture", "annotations_csv": "events.csv"}}, "training_ready": False}), encoding="utf-8")
            coord, qc = root / "coord.csv", root / "qc.csv"
            with coord.open("w", encoding="utf-8", newline="") as cstream, qc.open("w", encoding="utf-8", newline="") as qstream:
                cw, qw = csv.writer(cstream), csv.writer(qstream)
                cw.writerow(["frame", "pts_time"])
                qw.writerow(["frame", "pts_time", "gray_std"])
                for index in range(201):
                    # Equal PTS at an unsampled intermediate source frame is
                    # allowed; the selected even-frame PTS remain increasing.
                    timestamp = 1.0 if index == 11 else index / 10
                    cw.writerow([index, timestamp])
                    qw.writerow([index, timestamp, 30])
            pool = root / "sources.csv"
            with pool.open("w", encoding="utf-8", newline="") as stream:
                fields = ["unit_id", "source", "video_id", "group_id", "n_frames", "coord_path", "coord_sha256", "qc_path", "qc_sha256", "forbidden_ranges_json"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(unit_id="a", source="dongchuan", video_id="source-a", group_id="d|camera A|day", n_frames=201,
                                     coord_path="coord.csv", coord_sha256=hashlib.sha256(coord.read_bytes()).hexdigest(),
                                     qc_path="qc.csv", qc_sha256=hashlib.sha256(qc.read_bytes()).hexdigest(), forbidden_ranges_json="[[3,4]]"))
            windows = root / "windows.jsonl"
            rows = []
            for number, start in enumerate((0, 1, 3, 6, 10, 14, 1)):
                indices = list(range(start * 10, start * 10 + 31, 2))
                rows.append({"unit_id": "a", "clip_id": str(number), "frame_indices": indices,
                             "source_pts_seconds": [index / 10 for index in indices],
                             "sampling_start_sec": -999, "sampling_end_sec": 999})
            windows.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            paths = [pool, manifest, events, coord, qc, windows, source / "mapping.csv"]
            before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
            summary = audit(pool, manifest, windows, root / "out", repo_root=root, command=["fixture"])
            self.assertEqual(summary["totals"]["count"], 6)
            self.assertEqual(summary["totals"]["counts"], {"N": 1, "P": 1, "MIX": 1, "U": 1, "technical_invalid": 2})
            self.assertEqual(summary["window_input_schema"]["duplicate_physical_support_rows"], 1)
            self.assertEqual(summary["legacy_T_to_U"]["seconds"], 4)
            self.assertEqual(summary["totals"]["training_ready_count"], 0)
            self.assertEqual(summary["anchor_sensitivity"]["candidate_count"], 3)
            self.assertEqual(summary["by_unit"][0]["saved_N_P_contacts"], [5])
            witnesses = [json.loads(line) for line in (root / "out/window-witnesses.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(witnesses), 4)
            self.assertTrue(all(row["supervision_target"] is None for row in witnesses))
            self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
            # Wrong schema must fail before producing output, rather than guessing
            # a window from the deliberately misleading annotation bounds.
            windows.write_text(json.dumps({"unit_id": "a", "sampling_start_sec": 0, "sampling_end_sec": 3}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                audit(pool, manifest, windows, root / "bad-out", repo_root=root)
            self.assertFalse((root / "bad-out").exists())


if __name__ == "__main__":
    unittest.main()
