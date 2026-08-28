"""Characterization: lock the CSL / disorientation geometry helpers.

Pins ``_disorientation_deg_axis`` and ``is_csl_related`` on fixed cubic
orientation pairs.  Behaviour anchor for REFACTOR_PLAN §3 (the pure
``geometry.py`` module that ``filtering.py`` will depend on).  These are pure
functions, so the numbers must be bit-stable across the move.
"""
import math

import numpy as np

import laue_stream_utils as lsu
from _golden import check_golden


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

_PAIRS = {
    "sigma3_111_60":   _MAT @ _R([1, 1, 1], 60),
    "sigma9_110_3894": _MAT @ _R([1, 1, 0], 38.94),
    "generic_210_25":  _MAT @ _R([2, 1, 0], 25),
    "small_100_1p5":   _MAT @ _R([1, 0, 0], 1.5),
}


def test_char_disorientation():
    snap = {}
    for name, B in _PAIRS.items():
        ang, axfam = lsu._disorientation_deg_axis(_MAT, B)
        snap[name] = {"angle_deg": float(ang), "axis_family": axfam.tolist()}
    check_golden("disorientation", snap)


def test_char_is_csl_related():
    snap = {}
    for name, B in _PAIRS.items():
        snap[name] = {
            "sigma3": bool(lsu.is_csl_related(_MAT, B, sigmas=(3,))),
            "sigma3_9_11": bool(lsu.is_csl_related(_MAT, B, sigmas=(3, 9, 11))),
        }
    check_golden("is_csl_related", snap)


if __name__ == "__main__":
    for fn in [test_char_disorientation, test_char_is_csl_related]:
        fn(); print(f"PASS  {fn.__name__}")
