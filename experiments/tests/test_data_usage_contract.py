"""Small counterexamples for camera leakage and repeated-content contribution."""
import unittest

from experiments.common.video.data_usage_contract import (
    canonical_camera_identity, physical_window_key,
    validate_camera_partitions, validate_window_quotas,
)


class DataUsageContractTest(unittest.TestCase):
    def test_legacy_quota_is_explicit_and_not_a_global_four_window_limit(self):
        rows = [{"source_camera": "camera A", "source_media_sha256": "ab" * 32,
                 "frame_indices": list(range(i * 20, i * 20 + 16)), "state": "N",
                 "family_id": None, "family_status": "unresolved"} for i in range(6)]
        with self.assertRaises(TypeError):
            validate_window_quotas(rows, camera_cap=10, family_cap=10)
        self.assertEqual(validate_window_quotas(rows, camera_cap=10, family_cap=10,
                                              video_cap=6)["window_count"], 6)

    def test_mixed_windows_consume_existing_state_quotas_once(self):
        def window(index, **changes):
            return {"source_camera": "camera A", "source_media_sha256": "ab" * 32,
                    "frame_indices": list(range(index * 20, index * 20 + 16)),
                    "state": "MIX", "time_shares": {"N": .8, "T": 0, "P": .2},
                    "family_id": None, "family_status": "unresolved", **changes}
        mixed = [window(i, state="different-mixed-name-" + str(i)) for i in range(4)]
        audit = validate_window_quotas(mixed, camera_cap=16, family_cap=4, video_cap=4)
        self.assertEqual(audit["window_count"], 4)
        self.assertEqual(audit["mixed_window_count"], 4)
        self.assertEqual(audit["source_frame_exposures"], 64)
        self.assertEqual({x["state"]: x["count"] for x in audit["video_state_counts"]}, {"N": 4, "P": 4})
        with self.assertRaisesRegex(ValueError, "Original-video/state cap"):
            validate_window_quotas(mixed + [window(4, state="P", time_shares={"N": 0, "T": 0, "P": 1})],
                                   camera_cap=16, family_cap=4, video_cap=4)
        with self.assertRaisesRegex(ValueError, "MIX/U is not a new budget"):
            validate_window_quotas([window(0, time_shares=None)], camera_cap=16, family_cap=4, video_cap=4)
        with self.assertRaisesRegex(ValueError, "unknown time"):
            validate_window_quotas([window(0, time_shares={"N": .5, "T": 0, "P": 0})], camera_cap=16, family_cap=4, video_cap=4)

    def test_media_sha_case_alias_crosses_different_camera_roles(self):
        sources = {
            "a": {"source_camera": "camera A", "source_media_sha256": "ab" * 32},
            "b": {"source_camera": "camera B", "source_media_sha256": "AB" * 32},
        }
        original = {uid: dict(record) for uid, record in sources.items()}
        with self.assertRaisesRegex(ValueError, "Source media crossed"):
            validate_camera_partitions({"train": ["a"], "validation": ["b"]}, sources)
        self.assertEqual(sources, original)  # Canonical checks do not rewrite source metadata.
        sources["a"]["source_media_sha256"] = "cd" * 32
        derived = {"child": {"source_camera": "camera A", "parent_ids": ["a"],
                             "source_media_sha256": "ab" * 32}}
        with self.assertRaisesRegex(ValueError, "Source media crossed"):
            validate_camera_partitions({"train": ["b"], "validation": ["child"]}, sources, derived)
        for invalid in ("a" * 63, "g" * 64, "", None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "full source-media SHA256"):
                validate_camera_partitions({"train": ["bad"]},
                                           {"bad": {"source_camera": "camera A", "source_media_sha256": invalid}})

    def test_registered_view_dates_sources_and_parent_lineage(self):
        sources = {
            "a": {"source_camera": "塔吊下方", "site_code": "D12", "date": "2026-01-01", "source": "video4"},
            "b": {"source_camera": "塔吊下方", "site_code": "D12", "date": "2026-02-01", "source": "dongchuan"},
            "c": {"source_camera": "塔吊下游", "site_code": "D12", "date": "2026-01-01", "source": "video5"},
        }
        self.assertEqual(canonical_camera_identity(sources["a"]), canonical_camera_identity(sources["b"]))
        self.assertNotEqual(canonical_camera_identity(sources["a"]), canonical_camera_identity(sources["c"]))
        valid = validate_camera_partitions({"train": ["a", "b"], "validation": ["c"]}, sources)
        self.assertEqual(valid["role_camera_keys"]["train"], ["s|塔吊下方"])
        with self.assertRaisesRegex(ValueError, "Camera crossed"):
            validate_camera_partitions({"train": ["a"], "validation": ["b"]}, sources)
        with self.assertRaisesRegex(ValueError, "Camera crossed"):
            validate_camera_partitions({"inner_train": ["a"], "inner_val": ["b"],
                                        "outer_train": ["a", "b"], "outer_eval": ["c"]}, sources)
        derived = {"synthetic-a": {"source_camera": "塔吊下方", "synthetic": True, "parent_ids": ["a"]}}
        with self.assertRaisesRegex(ValueError, "Camera crossed"):
            validate_camera_partitions({"train": ["a"], "validation": ["synthetic-a"]}, sources, derived)
        with self.assertRaisesRegex(ValueError, "requires explicit parent_ids"):
            validate_camera_partitions({"train": ["bad"]}, sources,
                                       {"bad": {"source_camera": "塔吊下方", "synthetic": True}})
        with self.assertRaisesRegex(ValueError, "differs from parent"):
            validate_camera_partitions({"train": ["bad"]}, sources,
                                       {"bad": {"source_camera": "塔吊下游", "parent_ids": ["a"]}})
        with self.assertRaises(ValueError):
            canonical_camera_identity({"site_code": "D12"})

    def test_physical_identity_conflicts_and_fragmentation_do_not_add_quota(self):
        media = "ab" * 32
        indices = list(range(16))
        self.assertEqual(physical_window_key(media, indices), physical_window_key(media.upper(), indices))

        def window(index, **changes):
            return {"source_camera": "camera A", "source_media_sha256": media,
                    "frame_indices": list(range(index * 10, index * 10 + 16)), "state": "N",
                    "family_id": "same-static-content", **changes}

        with self.assertRaisesRegex(ValueError, "conflicting states"):
            validate_window_quotas([window(0, unit_id="first"), window(0, unit_id="alias", state="P")], camera_cap=16, family_cap=4, video_cap=4)
        with self.assertRaisesRegex(ValueError, "repeated within"):
            validate_window_quotas([window(0), window(0, unit_id="alias", csv_row=99)], camera_cap=16, family_cap=4, video_cap=4)
        original = [window(index, unit_id="one-original", csv_row=1) for index in range(4)]
        fragmented = [window(index, unit_id="different-alias-" + str(index), csv_row=index + 1,
                             cut_side="left" if index < 2 else "right") for index in range(4)]
        self.assertEqual(validate_window_quotas(original, camera_cap=16, family_cap=4, video_cap=4),
                         validate_window_quotas(fragmented, camera_cap=16, family_cap=4, video_cap=4))
        with self.assertRaisesRegex(ValueError, "Original-video/state cap"):
            validate_window_quotas(fragmented + [window(4, family_id="different-row")], camera_cap=16, family_cap=4, video_cap=4)
        # Five different files with the same content family still share four.
        with self.assertRaisesRegex(ValueError, "Content-family/state"):
            validate_window_quotas([window(i, source_media_sha256=f"{i+1:064x}") for i in range(5)], camera_cap=16, family_cap=4, video_cap=4)
        # The rule is not a blanket four-per-session cut: distinct motion strata can rotate.
        moving = [window(i, source_media_sha256="cd" * 32, family_id="flowing-water", motion_stratum="ordinary_water") for i in range(4)]
        audit = validate_window_quotas(original + moving, camera_cap=16, family_cap=4, video_cap=4)
        self.assertEqual(audit["window_count"], 8)
        self.assertEqual(len(audit["family_state_counts"]), 2)
        self.assertTrue(audit["basic_sampling_ready"])
        self.assertTrue(audit["family_review_complete"])
        self.assertGreater(audit["reused_source_frame_exposures"], 0)
        with self.assertRaisesRegex(ValueError, "Camera/state total"):
            validate_window_quotas([window(i, source_media_sha256=f"{i+1:064x}", family_id=str(i)) for i in range(17)], camera_cap=16, family_cap=4, video_cap=4)
        unresolved = [window(0, family_id=None, family_status="unresolved"),
                      window(1, family_id=None, family_status="unresolved")]
        audit = validate_window_quotas(unresolved, camera_cap=16, family_cap=4, video_cap=4)
        self.assertTrue(audit["basic_sampling_ready"])
        self.assertTrue(audit["quota_ready"])
        self.assertFalse(audit["family_review_complete"])
        self.assertFalse(audit["near_duplicate_removal_verified"])
        self.assertEqual(audit["unresolved_family_window_count"], 2)
        self.assertEqual(audit["family_state_counts"], [])
        with self.assertRaisesRegex(ValueError, "Original-video/state cap"):
            validate_window_quotas([window(i, family_id=None, family_status="unresolved") for i in range(5)],
                                   camera_cap=16, family_cap=4, video_cap=4)
        with self.assertRaisesRegex(ValueError, "Camera/state total"):
            validate_window_quotas([window(i, source_media_sha256=f"{i+1:064x}", family_id=None,
                                          family_status="unresolved") for i in range(17)],
                                   camera_cap=16, family_cap=4, video_cap=4)
        with self.assertRaisesRegex(ValueError, "explicitly unresolved"):
            validate_window_quotas([window(0, family_id=None)], camera_cap=16, family_cap=4, video_cap=4)


if __name__ == "__main__":
    unittest.main()
