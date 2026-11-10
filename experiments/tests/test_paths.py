"""Directory relocation contract; all test writes stay in disposable fixtures."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.paths import HISTORICAL_WINDOWS_REPO, resolve_path


class PathRelocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def resolve(self, value, **kw):
        return resolve_path(value, repo=self.root, **kw)

    def test_old_runs_win_cloud_and_relative_match(self):
        rel = "experiments/repro/exp008/runs/run-a/metrics.json"
        expected = self.root / "data/results/exp008/run-a/metrics.json"
        for recorded in (rel, "/root/repo/" + rel,
                         (HISTORICAL_WINDOWS_REPO + "/" + rel).replace("/", "\\")):
            with self.subTest(recorded=recorded):
                self.assertEqual(self.resolve(recorded), expected)
        self.assertEqual(self.resolve("/root/repo/experiments/repro/temporal/runs/r/cache.npz"),
                         self.root / "data/results/temporal/r/cache.npz")
        self.assertEqual(self.resolve("experiments/repro/convnext/runs/r/head.pt"),
                         self.root / "data/results/convnext/r/head.pt")

    def test_longest_alias_and_component_boundaries(self):
        self.assertEqual(self.resolve("experiments/repro/exp008/train_ft.py"),
                         self.root / "experiments/common/video/train_ft.py")
        self.assertEqual(self.resolve("experiments/repro/exp008/make_unit_folds.py"),
                         self.root / "experiments/splits/code/make_unit_folds.py")
        self.assertEqual(self.resolve("experiments/repro/exp008_extra/runs/a"),
                         self.root / "experiments/repro/exp008_extra/runs/a")
        self.assertEqual(self.resolve("experiments/repro/exp008/runs-extra/a"),
                         self.root / "experiments/common/video/runs-extra/a")
        self.assertEqual(self.resolve("/root/frames-extra/u"), Path("/root/frames-extra/u"))

    def test_explicit_frame_aliases(self):
        self.assertEqual(self.resolve("/root/frames/u/clip00_f00.jpg"),
                         self.root / "data/process/EXP-008-units-v2/frames/u/clip00_f00.jpg")
        self.assertEqual(self.resolve("/root/autodl-tmp/frames-v3/u/clip00_f00.jpg"),
                         self.root / "data/process/EXP-008-units-v3-gap05/frames/u/clip00_f00.jpg")

    def test_splits_and_temp_paths(self):
        names = (
            "EXP-006-scene-folds.json", "EXP-006-scene-logo-folds.json",
            "EXP-006-session-folds.json", "EXP-008-units-folds-128.json",
            "EXP-008-units-folds.json", "EXP-008-units-iid-folds.json",
            "EXP-008-units-scene-folds.json",
        )
        for name in names:
            with self.subTest(name=name):
                for prefix in ("", "/root/repo/", HISTORICAL_WINDOWS_REPO + "/"):
                    self.assertEqual(self.resolve(prefix + "experiments/splits/" + name),
                                     self.root / "data/metadata/splits" / name)
        self.assertEqual(self.resolve("experiments/splits/unknown.json"),
                         self.root / "experiments/splits/unknown.json")
        self.assertEqual(self.resolve("experiments/splits/EXP-008-units-folds.json/child"),
                         self.root / "experiments/splits/EXP-008-units-folds.json/child")
        self.assertEqual(self.resolve(".claude/.temp/check.txt"),
                         self.root / ".codex/docs/.temp/check.txt")

    def test_current_source_and_results_paths_are_idempotent(self):
        current = (
            "experiments/splits/code/make_unit_folds.py",
            "experiments/splits/code/fixture.json",
            "experiments/splits/README.md",
            "data/metadata/splits/EXP-008-units-folds.json",
            "data/results/exp008/fixture/metrics.json",
            "data/results/temporal/fixture/cache.npz",
            "data/results/convnext/fixture/head.pt",
        )
        for relative in current:
            with self.subTest(relative=relative):
                expected = self.root / relative
                expected.parent.mkdir(parents=True, exist_ok=True)
                expected.write_bytes(b"current fixture")
                resolved = self.resolve(relative, must_exist=True)
                self.assertEqual(resolved, expected)
                self.assertEqual(self.resolve(resolved, must_exist=True), expected)

    def test_external_absolute_path_kept(self):
        outside = Path(self.temp.name) / "external-cache.npz"
        outside.write_bytes(b"fixture")
        other_repo = self.root / "repo"
        self.assertEqual(resolve_path(outside, repo=other_repo, must_exist=True), outside)
        self.assertEqual(self.resolve(r"Z:\external\cache.npz"), Path(r"Z:\external\cache.npz"))

    def test_missing_never_searches_by_basename(self):
        (self.root / "metrics.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            self.resolve("/root/repo/experiments/repro/exp008/runs/missing/metrics.json",
                         must_exist=True)
        with self.assertRaises(ValueError):
            self.resolve("")

    def test_resolving_history_does_not_rewrite_artifacts(self):
        source = "experiments/repro/exp008/runs/old/metrics.json"
        artifact = self.root / "data/results/exp008/old/metrics.json"
        artifact.parent.mkdir(parents=True)
        payload = {"seed": 7, "feats": "/root/repo/experiments/repro/exp008/runs/f/feats.npz",
                   "md5": {"clip_labels": "unchanged"}}
        artifact.write_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
        before = hashlib.sha256(artifact.read_bytes()).digest()
        for recorded in (source, "/root/repo/" + source):
            resolved = self.resolve(recorded, must_exist=True)
            self.assertEqual(json.loads(resolved.read_bytes()), payload)
        self.assertEqual(hashlib.sha256(artifact.read_bytes()).digest(), before)


if __name__ == "__main__":
    unittest.main()
