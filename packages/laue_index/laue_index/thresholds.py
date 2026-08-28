"""Image thresholding strategies.

REFACTOR_PLAN §4a / §6.3.  The legacy ``lsu.apply_threshold`` dispatched on a
``method`` string; this promotes each branch to an independently testable
strategy class while keeping a byte-for-byte-equivalent ``apply_threshold``
dispatch (which ``lsu`` now re-exports).

Strategies return ``(thresholded_image, threshold_value)``; the input is assumed
background-subtracted (>= 0).
"""
from __future__ import annotations

from typing import Protocol, Tuple

import numpy as np

try:
    from skimage import filters as _skfilters
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

__all__ = [
    "ThresholdStrategy", "NoiseFloorThreshold", "PercentileThreshold",
    "OtsuThreshold", "FixedThreshold", "THRESHOLD_STRATEGIES", "apply_threshold",
]


def _apply(image: np.ndarray, thresh: float) -> Tuple[np.ndarray, float]:
    out = image.copy()
    out[image <= thresh] = 0
    return out, float(thresh)


def _noise_floor(image: np.ndarray, k: float = 4.0) -> float:
    """Robust per-frame noise floor: median(nz) + k*1.4826*MAD(nz), >= 1.0.

    Replaces the old max(60*(1+std//60),1) ~240 fixed floor that gutted faint
    frames (the 2026-06-21 fix)."""
    nz = image[image > 0]
    if nz.size:
        med = float(np.median(nz))
        mad = 1.4826 * float(np.median(np.abs(nz - med)))
        return max(med + k * mad, 1.0)
    return 1.0


class ThresholdStrategy(Protocol):
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, float]: ...


class NoiseFloorThreshold:
    """Adaptive: threshold a few MADs above the per-frame noise floor."""
    def __init__(self, k: float = 4.0):
        self.k = k

    def __call__(self, image):
        return _apply(image, _noise_floor(image, self.k))


class PercentileThreshold:
    def __init__(self, pct: float = 90.0):
        self.pct = pct

    def __call__(self, image):
        return _apply(image, float(np.percentile(image.ravel(), self.pct)))


class OtsuThreshold:
    def __call__(self, image):
        if not HAS_SKIMAGE:
            raise RuntimeError("OtsuThreshold requires scikit-image")
        return _apply(image, float(_skfilters.threshold_otsu(image)))


class FixedThreshold:
    def __init__(self, value: float = 0.0):
        self.value = value

    def __call__(self, image):
        return _apply(image, float(self.value))


THRESHOLD_STRATEGIES = {
    "adaptive": NoiseFloorThreshold,
    "percentile": PercentileThreshold,
    "otsu": OtsuThreshold,
    "fixed": FixedThreshold,
}


def apply_threshold(
    image: np.ndarray,
    method: str = "adaptive",
    fixed_value: float = 0.0,
    percentile: float = 90.0,
) -> Tuple[np.ndarray, float]:
    """String-dispatch threshold, byte-for-byte equivalent to the legacy
    ``lsu.apply_threshold`` — including the otsu->adaptive fallback when
    scikit-image is unavailable.  Kept so existing callers (preprocess_image,
    RunImage) need no change."""
    if method == "percentile":
        thresh = float(np.percentile(image.ravel(), percentile))
    elif method == "otsu" and HAS_SKIMAGE:
        thresh = float(_skfilters.threshold_otsu(image))
    elif method == "fixed":
        thresh = float(fixed_value)
    else:  # adaptive (and otsu fallback when skimage absent)
        thresh = _noise_floor(image, k=4.0)
    return _apply(image, thresh)
