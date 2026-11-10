"""Deterministic photometric changes of ONE still image, for local preview only.

This does not create physical motion, independent observations, or a training
sampler. The parameter defaults are illustrative, not sensor calibration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


Condition = Literal["baseline", "exposure_only", "white_balance_only", "sensor_noise_only"]
CONDITIONS: tuple[Condition, ...] = (
    "baseline", "exposure_only", "white_balance_only", "sensor_noise_only",
)


@dataclass(frozen=True)
class PhotometricSpec:
    frame_count: int = 16
    preview_display_seconds: float = 3.0
    seed: int = 20261108
    exposure_amplitude_ev: float = 0.02
    exposure_period_seconds: float = 12.0
    white_balance_amplitude: float = 0.005
    white_balance_period_seconds: float = 12.0
    noise_std_normalized: float = 0.3 / 255.0
    noise_rho: float = 0.95

    def validate(self) -> None:
        if not isinstance(self.frame_count, int) or self.frame_count < 1:
            raise ValueError("frame_count must be a positive integer")
        values = asdict(self)
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("all parameters must be finite")
        if self.preview_display_seconds <= 0:
            raise ValueError("preview_display_seconds must be positive")
        if self.exposure_period_seconds <= 0 or self.white_balance_period_seconds <= 0:
            raise ValueError("photometric periods must be positive")
        if self.exposure_amplitude_ev < 0 or not 0 <= self.white_balance_amplitude < 1:
            raise ValueError("invalid exposure or white-balance amplitude")
        if self.noise_std_normalized < 0 or not 0 <= self.noise_rho < 1:
            raise ValueError("noise std must be nonnegative and rho in [0, 1)")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def _validate(condition: Condition, spec: PhotometricSpec) -> None:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    spec.validate()


def _wave(spec: PhotometricSpec, period_seconds: float) -> np.ndarray:
    """One smooth sinusoid with a fixed-seed phase; no per-frame redraws."""
    phase = np.random.default_rng(spec.seed).uniform(0, 2 * np.pi)
    display_times = np.arange(spec.frame_count) * spec.preview_display_seconds / spec.frame_count
    return np.sin(phase + 2 * np.pi * display_times / period_seconds)


def _decode_srgb(encoded: np.ndarray) -> np.ndarray:
    return np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)


def _encode_uint8(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0, 1)
    encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return np.rint(encoded * 255).clip(0, 255).astype(np.uint8)


def simulate_photometric_frames(
    source_rgb: np.ndarray,
    condition: Condition,
    spec: PhotometricSpec = PhotometricSpec(),
) -> np.ndarray:
    """Return uint8 [T,H,W,3] without modifying the input or touching the RNG state.

    Exposure and red/blue white-balance gains operate on linearized sRGB.
    Sensor-noise-only adds a stationary per-pixel/channel AR(1) field to encoded
    sRGB. Its initial field is random; subsequent fields use rho*previous plus
    sqrt(1-rho**2)*innovation, rather than independently replacing the field.
    No coordinate, object, water-surface, camera, or texture motion is synthesized.
    """
    _validate(condition, spec)
    if not isinstance(source_rgb, np.ndarray) or source_rgb.dtype != np.uint8:
        raise ValueError("source_rgb must be a uint8 numpy array")
    if source_rgb.ndim != 3 or source_rgb.shape[2] != 3 or min(source_rgb.shape[:2]) < 1:
        raise ValueError("source_rgb must have nonempty shape [H,W,3]")
    frames = np.repeat(source_rgb[None], spec.frame_count, axis=0)
    if condition == "baseline":
        return frames
    encoded = source_rgb.astype(np.float32) / 255.0
    if condition == "sensor_noise_only":
        rng = np.random.default_rng(spec.seed)
        noise = rng.standard_normal(source_rgb.shape, dtype=np.float32) * spec.noise_std_normalized
        innovation_scale = spec.noise_std_normalized * np.sqrt(1 - spec.noise_rho ** 2)
        for index in range(spec.frame_count):
            if index:
                noise = (spec.noise_rho * noise + innovation_scale *
                         rng.standard_normal(source_rgb.shape, dtype=np.float32))
            frames[index] = np.rint((encoded + noise) * 255).clip(0, 255).astype(np.uint8)
        return frames
    linear = _decode_srgb(encoded)
    if condition == "exposure_only":
        gains = np.exp2(spec.exposure_amplitude_ev * _wave(spec, spec.exposure_period_seconds))
        for index, gain in enumerate(gains):
            frames[index] = _encode_uint8(linear * gain)
    elif condition == "white_balance_only":
        offsets = spec.white_balance_amplitude * _wave(spec, spec.white_balance_period_seconds)
        for index, offset in enumerate(offsets):
            frames[index] = _encode_uint8(linear * np.array([1 + offset, 1, 1 - offset], dtype=np.float32))
    return frames


def photometric_metadata(condition: Condition, spec: PhotometricSpec = PhotometricSpec()) -> dict:
    """Describe display-time synthesis without counting it as captured duration."""
    _validate(condition, spec)
    return {
        "condition": condition,
        "temporal_information": "none" if condition == "baseline" else "simulated_photometric_only",
        "physical_motion": "none",
        "unique_source_images": 1,
        "capture_observation_seconds": 0,
        "preview_display_seconds": spec.preview_display_seconds,
        "frame_count": spec.frame_count,
        "training_ready": False,
        "sensor_distribution_calibrated": False,
        "parameters": asdict(spec),
        "display_frame_start_seconds": (
            np.arange(spec.frame_count) * spec.preview_display_seconds / spec.frame_count
        ).tolist(),
        "interpretation": "One ordinary-flow N still; frame count is not independent source observations.",
    }
