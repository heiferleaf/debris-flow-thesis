"""Small real-JPEG fixtures for protected pixels, compaction, and recovery."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from experiments.tools.compact_audit_frames import compact_frames


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def noisy_jpeg(path, size, quality=100):
    image = Image.effect_noise(size, 80).convert("RGB")
    image.save(path, "JPEG", quality=quality, subsampling=0)
    image.close()


class CompactAuditFramesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="compact-audit-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.frames = self.base / "frames"
        self.frames.mkdir()
        self.protection = self.base / "protected.txt"
        self.protection.write_text("", encoding="utf-8")
        self.ledger = self.base / "ledger.jsonl"

    def test_protected_and_index_unchanged_smaller_verified_jpeg_same_path(self):
        protected = self.frames / "a-protected.jpg"
        candidate = self.frames / "b-candidate.jpg"
        not_jpeg = self.frames / "c-misnamed.jpg"
        noisy_jpeg(protected, (1600, 900))
        noisy_jpeg(candidate, (2000, 1000))
        Image.new("RGB", (10, 8), "red").save(not_jpeg, format="PNG")
        index = self.base / "frames_index.csv"
        index.write_text("frame_file,t_sec\na-protected.jpg,0.125\nb-candidate.jpg,2.125\n", encoding="utf-8")
        self.protection = self.base / "protected.json"
        self.protection.write_text(json.dumps({"protected_files": [{"path": protected.name}]}), encoding="utf-8")
        protected_sha, index_sha, misnamed_sha = digest(protected), digest(index), digest(not_jpeg)
        original_sha, original_size = digest(candidate), candidate.stat().st_size
        summary = compact_frames(self.frames, self.protection, self.ledger, workers=2)
        self.assertEqual(summary["processed"], 3)
        self.assertEqual(summary["replaced"], 1)
        self.assertEqual(summary["reason_counts"]["protected"], 1)
        self.assertEqual(summary["reason_counts"]["not_jpeg"], 1)
        self.assertEqual((digest(protected), digest(index), digest(not_jpeg)), (protected_sha, index_sha, misnamed_sha))
        with Image.open(candidate) as checked:
            checked.load()
            self.assertEqual(checked.format, "JPEG")
            self.assertEqual(checked.size, (1280, 640))
        self.assertLess(candidate.stat().st_size, original_size)
        self.assertEqual(summary["saved_logical_bytes"], original_size - candidate.stat().st_size)
        entries = [json.loads(line) for line in self.ledger.read_text(encoding="utf-8").splitlines()]
        replaced = next(entry for entry in entries if entry.get("status") == "replaced")
        self.assertEqual(replaced["path"], candidate.name)
        self.assertEqual(replaced["original_sha256"], original_sha)
        self.assertEqual(replaced["new_sha256"], digest(candidate))
        self.assertEqual(replaced["original_dimensions"], [2000, 1000])
        self.assertEqual(replaced["new_dimensions"], [1280, 640])
        self.assertEqual(sorted(path.name for path in self.frames.iterdir()),
                         ["a-protected.jpg", "b-candidate.jpg", "c-misnamed.jpg"])

    def test_no_upscale_limit_and_not_smaller_never_replaced(self):
        small = self.frames / "a-small.jpg"
        tiny = self.frames / "b-tiny.jpg"
        noisy_jpeg(small, (320, 180))
        Image.new("RGB", (8, 8), "black").save(tiny, "JPEG", quality=85, subsampling=2, optimize=True)
        tiny_sha = digest(tiny)
        first = compact_frames(self.frames, self.protection, self.ledger, workers=1, limit=1)
        self.assertEqual(first["processed"], 1)
        with Image.open(small) as image:
            self.assertEqual(image.size, (320, 180))
        second = compact_frames(self.frames, self.protection, self.ledger, workers=1)
        self.assertEqual(second["replaced"], 0)
        self.assertEqual(second["reason_counts"]["not_smaller"], 1)
        self.assertEqual(digest(tiny), tiny_sha)
        self.assertEqual(second["saved_logical_bytes"], 0)
        with self.assertRaisesRegex(ValueError, "absolute path"):
            compact_frames(Path("relative-frames"), self.protection, self.ledger)

    def test_resume_after_replace_before_final_row_and_partial_tail(self):
        candidate = self.frames / "frame_000001_t2s.jpg"
        noisy_jpeg(candidate, (1800, 1200))
        first = compact_frames(self.frames, self.protection, self.ledger, workers=1)
        self.assertEqual(first["replaced"], 1)
        compacted_sha, compacted_mtime = digest(candidate), candidate.stat().st_mtime_ns
        lines = self.ledger.read_text(encoding="utf-8").splitlines()
        prepared = next(line for line in lines if json.loads(line).get("status") == "prepared")
        # Simulate a committed image replacement whose final journal row was lost.
        self.ledger.write_bytes((lines[0] + "\n" + prepared + '\n{"event":').encode("utf-8"))
        recovered = compact_frames(self.frames, self.protection, self.ledger, workers=1)
        self.assertEqual(recovered["replaced"], 0)
        self.assertEqual(recovered["status_counts"]["resume_matched"], 1)
        self.assertEqual(recovered["saved_logical_bytes"], 0)
        self.assertEqual((digest(candidate), candidate.stat().st_mtime_ns), (compacted_sha, compacted_mtime))
        rows = [json.loads(line) for line in self.ledger.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(row.get("event") == "recovered_incomplete_ledger_tail" for row in rows))
        self.assertTrue(any(row.get("recovered_prepared") for row in rows))
        again = compact_frames(self.frames, self.protection, self.ledger, workers=1)
        self.assertEqual(again["status_counts"]["resume_matched"], 1)
        self.assertEqual((digest(candidate), candidate.stat().st_mtime_ns), (compacted_sha, compacted_mtime))
        self.assertEqual(list(self.frames.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
