"""Regression / characterization tests for the 2026-06-21 LaueMatching
robustness fixes:

  1. filter_orientations_robust  -- twin/CSL-aware orientation filter that
     keeps real Sigma3 twins (and the matrix) instead of letting the
     winner-take-all unique-spot dedup delete one of a coincident pair.
  2. apply_threshold (adaptive)  -- robust per-frame noise-floor threshold
     so faint frames are not gutted by the old ~fixed-240 std formula.

Standalone: `python tests/test_robustness_fixes.py` or `pytest`.  Only needs
numpy + scripts/laue_stream_utils.py (no external fixtures).  When the library
is refactored, keep these as behaviour anchors and re-point the import.
"""
import math
import os
import sys

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "scripts"))
import laue_stream_utils as lsu  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _R(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)


# A well-determined cubic orientation (from a real Ni indexing).
_MAT = np.array([0.5425461, 0.7753705, 0.3231785,
                 0.8400185, -0.4991547, -0.2126346,
                 -0.0035545, 0.3868400, -0.9221400]).reshape(3, 3)


def _solution_row(grain, quality, n_matches, om, ncols=34):
    """RunImage solution layout: grain=col0, quality=col4, NMatches=col5,
    OrientMatrix=cols22:31."""
    r = np.zeros(ncols)
    r[0] = grain
    r[4] = quality
    r[5] = n_matches
    r[22:31] = np.asarray(om).reshape(-1)
    return r


# --------------------------------------------------------------------------
# 1. twin/CSL-aware orientation filter
# --------------------------------------------------------------------------
def test_is_csl_related_sigma3():
    twin = _MAT @ _R([1, 1, 1], 60)          # exact Sigma3
    assert lsu.is_csl_related(_MAT, twin, sigmas=(3,))
    not_csl = _MAT @ _R([2, 1, 0], 25)       # generic 25 deg
    assert not lsu.is_csl_related(_MAT, not_csl, sigmas=(3,))


def test_robust_filter_rescues_sigma3_twin_pair():
    """The core bug: a real matrix that shares Sigma3 reflections with a
    higher-goodness twin must NOT be deleted, while spurious / near-duplicate
    solutions still are."""
    twin = _MAT @ _R([1, 1, 1], 60)          # Sigma3 twin of matrix
    spurious = _MAT @ _R([1, 1, 0], 38.94)   # Sigma9-ish, NOT Sigma3 -> drop
    near_dup = twin @ _R([1, 0, 0], 1.5)     # ~1.5 deg from twin -> dedup
    independent = _MAT @ _R([2, 1, 0], 25)   # distinct real grain -> keep

    orient = np.array([
        _solution_row(1, 2789.0, 10, twin),         # highest quality
        _solution_row(2, 1998.0, 11, _MAT),         # MATRIX (most matches!)
        _solution_row(3, 1594.0, 5, spurious),
        _solution_row(4, 2500.0, 9, near_dup),
        _solution_row(5, 1200.0, 6, independent),
    ])
    # winner-take-all spot claim leaves the matrix & near-dup with ~0 unique
    usi = {
        1: {"count": 10, "unique_label_count": 8},
        2: {"count": 11, "unique_label_count": 0},   # starved by the twin
        3: {"count": 5, "unique_label_count": 1},
        4: {"count": 9, "unique_label_count": 0},
        5: {"count": 6, "unique_label_count": 5},
    }
    kw = dict(min_unique=2, grain_col=0, quality_col=4,
              om_start_col=22, nmatches_col=5,
              max_angle_deg=5.0, min_total_spots=5, csl_sigmas=(3,))

    legacy = {int(r[0]) for r in lsu.filter_orientations(orient, usi, min_unique=2)}
    robust = {int(r[0]) for r in lsu.filter_orientations_robust(orient, usi, **kw)}

    assert 2 not in legacy, "legacy SHOULD drop the matrix (documents the bug)"
    assert {1, 2, 5}.issubset(robust), "robust must keep twin+matrix+independent"
    assert 2 in robust, "robust must rescue the Sigma3 matrix"
    assert 3 not in robust, "spurious (non-CSL, low unique) must be dropped"
    assert 4 not in robust, "near-duplicate of the twin must be deduped"


# --------------------------------------------------------------------------
# 2. noise-floor adaptive threshold
# --------------------------------------------------------------------------
def _synthetic_frame(spot_amp, n_spots=40, noise=8.0, size=256, seed=0):
    rng = np.random.RandomState(seed)
    img = np.abs(rng.normal(0, noise, (size, size)))   # background noise
    for _ in range(n_spots):
        y, x = rng.randint(8, size - 8, 2)
        img[y - 1:y + 2, x - 1:x + 2] += spot_amp       # a few-pixel spot
    return img


def test_adaptive_threshold_recovers_faint_with_hot_pixels():
    """Reproduce the real bug: a faint frame with a few saturated/hot pixels.
    The OLD formula max(60*(1+std//60),1) is inflated by the hot pixels' std to
    ~hundreds, zeroing the faint real spots.  The new noise-floor threshold
    sits just above the noise so the faint spots survive."""
    faint = _synthetic_frame(spot_amp=150.0, n_spots=40, noise=8.0, seed=3)
    # a few saturated/hot pixels (like the 33 in real Ni end-frames)
    rng = np.random.RandomState(99)
    for _ in range(33):
        y, x = rng.randint(0, faint.shape[0], 2)
        faint[y, x] = 65535.0

    old_thresh = max(60.0 * (1.0 + np.std(faint) // 60.0), 1.0)  # legacy formula
    out_new, new_thresh = lsu.apply_threshold(faint, method="adaptive")

    assert new_thresh < 150.0 < old_thresh, (new_thresh, old_thresh)
    # faint spots survive the new threshold ...
    assert (out_new > 0).sum() >= 40 * 9 * 0.5, "faint spots must survive"
    # ... but the legacy formula would have gutted them (only hot pixels left)
    assert (faint > old_thresh).sum() < 100, "legacy formula guts the faint spots"


def test_adaptive_threshold_above_noise_floor():
    """Threshold must sit above the noise (so pure-noise frames stay empty-ish)
    but below real spots."""
    faint = _synthetic_frame(spot_amp=80.0, seed=2)
    out, thr = lsu.apply_threshold(faint, method="adaptive")
    assert thr > 0.0
    # threshold should be well below the spot amplitude (so spots survive)
    assert thr < 80.0


# --------------------------------------------------------------------------
def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
