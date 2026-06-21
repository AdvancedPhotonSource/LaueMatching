"""Characterization: lock the current behaviour of ``apply_threshold``.

Pins (threshold value, surviving-pixel count) for each method on a set of
deterministic, seeded synthetic frames.  This is the behaviour anchor for the
planned ``ThresholdStrategy`` extraction (REFACTOR_PLAN §4a / §6.3): the new
strategy classes must reproduce these numbers exactly.
"""
import numpy as np
import pytest

import laue_stream_utils as lsu
from _golden import check_golden


def _synthetic_frame(spot_amp, n_spots=40, noise=8.0, size=256, seed=0,
                     n_hot=0, hot_val=65535.0, hot_seed=99):
    """A reproducible faint/bright frame, optionally with saturated hot pixels."""
    rng = np.random.RandomState(seed)
    img = np.abs(rng.normal(0, noise, (size, size)))
    for _ in range(n_spots):
        y, x = rng.randint(8, size - 8, 2)
        img[y - 1:y + 2, x - 1:x + 2] += spot_amp
    if n_hot:
        hrng = np.random.RandomState(hot_seed)
        for _ in range(n_hot):
            y, x = hrng.randint(0, size, 2)
            img[y, x] = hot_val
    return img


# A small battery of frames spanning the regimes the threshold must handle.
_FRAMES = {
    "bright":            dict(spot_amp=400.0, seed=1),
    "faint":             dict(spot_amp=80.0, seed=2),
    "faint_with_hot":    dict(spot_amp=150.0, seed=3, n_hot=33),
    "empty":             dict(spot_amp=0.0, n_spots=0, noise=0.0, seed=4),
}


def _measure(method, **kw):
    out = {}
    for fname, fkw in _FRAMES.items():
        img = _synthetic_frame(**fkw)
        thr_img, thr = lsu.apply_threshold(img, method=method, **kw)
        out[fname] = {"thresh": float(thr), "nonzero": int((thr_img > 0).sum())}
    return out


def test_char_threshold_adaptive():
    check_golden("threshold_adaptive", _measure("adaptive"))


def test_char_threshold_percentile():
    check_golden("threshold_percentile", _measure("percentile", percentile=99.5))


def test_char_threshold_fixed():
    check_golden("threshold_fixed", _measure("fixed", fixed_value=240.0))


@pytest.mark.skipif(not lsu.HAS_SKIMAGE, reason="skimage (otsu) unavailable")
def test_char_threshold_otsu():
    check_golden("threshold_otsu", _measure("otsu"))


if __name__ == "__main__":
    for fn in [test_char_threshold_adaptive, test_char_threshold_percentile,
               test_char_threshold_fixed]:
        fn(); print(f"PASS  {fn.__name__}")
    if lsu.HAS_SKIMAGE:
        test_char_threshold_otsu(); print("PASS  test_char_threshold_otsu")
