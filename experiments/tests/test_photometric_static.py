"""Three small checks for the preview-only photometric still implementation."""
from __future__ import annotations

import unittest

import numpy as np

from experiments.common.video.photometric_static import (
    CONDITIONS, PhotometricSpec, photometric_metadata, simulate_photometric_frames,
)


class PhotometricStaticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = np.random.default_rng(10).integers(16, 240, (12, 20, 3), dtype=np.uint8)

    def test_deterministic_without_mutating_source(self) -> None:
        original = self.source.copy()
        for condition in CONDITIONS:
            with self.subTest(condition=condition):
                first = simulate_photometric_frames(self.source, condition)
                second = simulate_photometric_frames(self.source, condition)
                np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(self.source, original)

    def test_baseline_is_exact_repetition(self) -> None:
        output = simulate_photometric_frames(self.source, "baseline")
        self.assertEqual(output.shape, (16, 12, 20, 3))
        for frame in output:
            np.testing.assert_array_equal(frame, self.source)
        self.assertEqual(photometric_metadata("baseline")["temporal_information"], "none")

    def test_variants_preserve_dimensions_and_single_source_provenance(self) -> None:
        spec = PhotometricSpec(seed=30, exposure_amplitude_ev=0.04,
                               white_balance_amplitude=0.01, noise_std_normalized=0.6 / 255)
        for condition in CONDITIONS:
            with self.subTest(condition=condition):
                output = simulate_photometric_frames(self.source, condition, spec)
                self.assertEqual(output.shape, (16, *self.source.shape))
                self.assertEqual(output.dtype, np.uint8)
                record = photometric_metadata(condition, spec)
                self.assertEqual(record["unique_source_images"], 1)
                self.assertEqual(record["capture_observation_seconds"], 0)
                self.assertEqual(record["physical_motion"], "none")
                self.assertFalse(record["training_ready"])
                if condition != "baseline":
                    self.assertEqual(record["temporal_information"], "simulated_photometric_only")


if __name__ == "__main__":
    unittest.main()
