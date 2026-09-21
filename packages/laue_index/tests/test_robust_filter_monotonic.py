"""The robust orientation filter must be monotonic in an orientation's OWN evidence.

Before 0.7.2 the evidence floor used the winner-take-all ``count`` but fell back
to ``NMatches`` whenever that count was 0. On real ``calculate_unique_spots``
output that made a Sigma3 twin whose spots were ALL claimed by its parent pass
(count 0 -> NMatches), while the same twin with 1-4 pixels of its own failed
(count 1-4 < MinNrSpots): more evidence, worse outcome. These tests sweep the
number of spots the twin shares with its parent and require the kept set to be
upward-closed in the twin's own evidence.

Everything goes through the real ``calculate_unique_spots`` -- the bug lived in
how its output was consumed, so a hand-written ``unique_spot_info`` would not
exercise it.
"""
import math

import numpy as np
import pytest

from laue_index.filtering import calculate_unique_spots, filter_orientations_robust
from laue_index.geometry import HEX_OPS, disorientation_deg_axis
from laue_index.postprocess import PostProcessor


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

_KW = dict(min_unique=2, grain_col=0, quality_col=4, om_start_col=22,
           nmatches_col=5, max_angle_deg=2.0, min_total_spots=5,
           csl_sigmas=(3,))


def _sol(grain, quality, n_matches, om):
    """RunImage layout: grain col0, quality col4, NMatches col5, OM cols 22:31."""
    r = np.zeros(34)
    r[0], r[4], r[5] = grain, quality, n_matches
    r[22:31] = np.asarray(om).reshape(-1)
    return r


def _spot(grain, x, y):
    """RunImage spots layout: grain col0, x col5, y col6."""
    r = np.zeros(11)
    r[0], r[5], r[6] = grain, x, y
    return r


def _frame(n_shared, second_om, n_second=10, n_parent=12):
    """Parent (grain 1, better quality) and a second orientation (grain 2) that
    shares ``n_shared`` of its ``n_second`` matched spots with the parent. Every
    spot sits in its own segmentation label, so shared spots are exactly the
    ones the parent claims first."""
    labels = np.zeros((200, 200), dtype=np.int32)
    spots = []
    parent_px = [(10 + 10 * i, 20) for i in range(n_parent)]
    for i, (x, y) in enumerate(parent_px):
        labels[y, x] = 1 + i
        spots.append(_spot(1, x, y))
    for j in range(n_second):
        if j < n_shared:
            x, y = parent_px[j]
        else:
            x, y = 10 + 10 * j, 100
            labels[y, x] = 100 + j
        spots.append(_spot(2, x, y))
    sols = np.array([_sol(1, 1000.0, n_parent, _MAT),
                     _sol(2, 500.0, n_second, second_om)])
    return sols, np.array(spots), labels


def _kept_by_own(second_om, n_second=10):
    out = {}
    for n_shared in range(n_second + 1):
        sols, spots, labels = _frame(n_shared, second_om, n_second)
        usi = calculate_unique_spots(sols, spots, labels)
        own = usi[2]["count"]
        assert own == n_second - n_shared   # the sweep really moves own evidence
        kept = {int(r[0]) for r in filter_orientations_robust(sols, usi, **_KW)}
        assert 1 in kept
        out[own] = 2 in kept
    return out


def _assert_upward_closed(kept_by_own):
    owns = sorted(kept_by_own)
    first = next((o for o in owns if kept_by_own[o]), None)
    if first is None:
        return
    bad = [o for o in owns if o >= first and not kept_by_own[o]]
    assert not bad, (
        f"not monotonic in own evidence: kept at own={first} but dropped at "
        f"own={bad} ({kept_by_own})")


def test_sigma3_twin_is_monotonic_in_own_evidence():
    twin = _MAT @ _R([1, 1, 1], 60)           # exact Sigma3
    kept = _kept_by_own(twin)
    _assert_upward_closed(kept)
    # the floor is MinNrSpots own pixels; the CSL exemption covers exclusivity
    assert all(kept[o] for o in range(5, 11))
    assert not any(kept[o] for o in range(0, 5))


def test_twin_with_every_spot_claimed_is_not_rescued_by_nmatches():
    """The old fallback: own count 0 read NMatches (10) and passed the floor.
    Zero pixels of its own cannot be told from a phantom Sigma3 variant."""
    twin = _MAT @ _R([1, 1, 1], 60)
    assert _kept_by_own(twin)[0] is False


def test_unrelated_grain_is_monotonic_too():
    other = _MAT @ _R([2, 1, 0], 25)          # generic, not CSL
    kept = _kept_by_own(other)
    _assert_upward_closed(kept)
    assert all(kept[o] for o in range(5, 11))


def test_missing_winner_take_all_info_falls_back_to_nmatches():
    """The NMatches fallback survives for the case it is for: no winner-take-all
    information supplied for a grain at all."""
    sols = np.array([_sol(1, 1000.0, 12, _MAT)])
    kept = filter_orientations_robust(sols, {}, **_KW)
    assert len(kept) == 1
    sols = np.array([_sol(1, 1000.0, 3, _MAT)])
    assert len(filter_orientations_robust(sols, {}, **_KW)) == 0


# ---------------------------------------------------------------------------
# near-duplicate removal for hexagonal groups
# ---------------------------------------------------------------------------

def _two_grain_frame(om_a, om_b):
    """Two orientations, each with 8 spots in 8 labels of its own."""
    labels = np.zeros((200, 200), dtype=np.int32)
    spots = []
    for g, y in ((1, 20), (2, 120)):
        for i in range(8):
            x = 10 + 10 * i
            labels[y, x] = 10 * g + i
            spots.append(_spot(g, x, y))
    sols = np.array([_sol(1, 1000.0, 8, om_a), _sol(2, 900.0, 8, om_b)])
    return sols, np.array(spots), labels


def test_hex_ops_see_a_60_degree_c_rotation_as_identity():
    """Bound 1e-3 deg: _MAT is printed to 7 decimals, so it is orthonormal only
    to ~1e-7, which shows up as ~1e-5 deg in the arccos."""
    a = _MAT
    b = _MAT @ _R([0, 0, 1], 60) @ _R([1, 0, 0], 1.0)
    ang, _ = disorientation_deg_axis(a, b, HEX_OPS)
    assert ang == pytest.approx(1.0, abs=1e-3)


def test_hexagonal_near_duplicates_are_removed():
    """The docstring promised near-duplicate removal; for non-cubic groups the
    filter used to skip it entirely. A hexagonal pair 1 deg apart once the
    6-fold is accounted for must collapse to one; with no symmetry tabulated
    (triclinic) the same pair is 60 deg apart and both stay."""
    b = _MAT @ _R([0, 0, 1], 60) @ _R([1, 0, 0], 1.0)
    sols, spots, labels = _two_grain_frame(_MAT, b)
    hexa = PostProcessor(robust=True, max_angle_deg=2.0, space_group=194)(
        sols, spots, labels)
    assert hexa.kept_grain_nrs == {1}
    tric = PostProcessor(robust=True, max_angle_deg=2.0, space_group=2)(
        sols, spots, labels)
    assert tric.kept_grain_nrs == {1, 2}


def test_hexagonal_distinct_grains_are_kept():
    b = _MAT @ _R([1, 0, 0], 30)
    sols, spots, labels = _two_grain_frame(_MAT, b)
    res = PostProcessor(robust=True, max_angle_deg=2.0, space_group=194)(
        sols, spots, labels)
    assert res.kept_grain_nrs == {1, 2}
