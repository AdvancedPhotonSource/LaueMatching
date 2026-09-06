"""Tests for the XMAS -> LaueMatching geometry conversion.

These pin the things that would silently produce a confident wrong answer:
the pixel-origin mapping, the P inversion, the radians convention, and the two
gauge/near-gauge degeneracies measured during the TPS 21A campaign.
"""
from __future__ import annotations

import numpy as np
import pytest

from laue_index.xmas import (
    MOUNTS, TILT_CONVENTIONS, TPS21A_A5, TPS21A_NB1_3,
    convert, enumerate_candidates, poni_pixel,
)
from laue_index.calibrate import (
    DetectorSpec, project, reciprocal_matrix, rodrigues_to_matrix,
)


# --- the pixel origin, from the data's own dual-convention statement ---------
@pytest.mark.parametrize("cal,albula_first,albula_second", [
    (TPS21A_A5, 1203.76, 1228.98),        # A5_fine_Condition.txt
    (TPS21A_NB1_3, 1211.62, 1228.73),     # Nb1_3_Condition.txt
])
def test_poni_matches_the_albula_centre_quoted_in_the_condition_file(
        cal, albula_first, albula_second):
    """Each Condition.txt states the centre in BOTH conventions. The mapping
    row = n_first - xcent, col = ycent - 1 must reproduce the Albula pair
    exactly -- it does, to both decimals, on both scans."""
    px, py = poni_pixel(cal)
    assert px == pytest.approx(albula_first, abs=5e-3)
    assert py == pytest.approx(albula_second, abs=5e-3)


def test_P_inverts_the_projection_back_to_the_poni():
    """P is defined by requiring the normal-incidence ray to land on the XMAS
    centre channel. Push a ray straight down the detector normal and check it
    comes back to that pixel."""
    for cal in (TPS21A_A5, TPS21A_NB1_3):
        g = convert(cal, "34IDE", "rpy_zxy_pos")
        px_poni, py_poni = poni_pixel(cal)
        px = (0.0 - g.P[0]) / g.px_x + 0.5 * (g.nr_px_x - 1)
        py = (0.0 - g.P[1]) / g.px_y + 0.5 * (g.nr_px_y - 1)
        assert px == pytest.approx(px_poni, abs=1e-9)
        assert py == pytest.approx(py_poni, abs=1e-9)


def test_axis_order_columns_are_nr_px_x():
    """LaueMatchingCPU.c indexes image[ipy*nrPxX + ipx], so NrPxX counts
    COLUMNS. The TPS Pilatus TIFF is (2527 rows, 2463 cols)."""
    g = convert(TPS21A_A5)
    assert (g.nr_px_x, g.nr_px_y) == (2463, 2527)


def test_R_is_radians_not_degrees():
    """|R| must be ~2.1 rad (a 120 deg mount), not ~120. GenerateHKLs --help
    says degrees; the code does cos(norm(R)). The docs are wrong."""
    g = convert(TPS21A_A5, "34IDE", "rpy_zxy_pos")
    assert np.linalg.norm(g.R) == pytest.approx(2.0999, abs=0.02)


def test_nominal_mount_sends_detector_normal_to_lab_Y():
    """The 34-ID-E family: det z -> lab +Y, det x -> lab +Z, det y -> lab +X."""
    R = MOUNTS["34IDE"]
    assert np.allclose(R[:, 0], [0, 0, 1])
    assert np.allclose(R[:, 1], [1, 0, 0])
    assert np.allclose(R[:, 2], [0, 1, 0])
    assert np.linalg.det(R) == pytest.approx(1.0)


# --- the two degeneracies ---------------------------------------------------
def _Rz(deg):
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@pytest.mark.parametrize("a,b", [
    ("34IDE", "flip"), ("34IDE_rot90", "flip_rot90"),
    ("34IDE_rot180", "flip_rot180"), ("34IDE_rot270", "flip_rot270"),
])
def test_flip_mounts_are_the_180deg_beam_gauge_not_new_geometries(a, b):
    """The four 'flip' mounts are the four others turned 180 deg about the
    BEAM, which is an exact gauge. Enumerating both would duplicate every
    candidate -- so enumerate_candidates must not."""
    assert np.allclose(_Rz(180) @ MOUNTS[a], MOUNTS[b], atol=1e-12)


def test_enumerate_candidates_is_32_and_omits_the_gauge_duplicates():
    cands = list(enumerate_candidates(TPS21A_A5))
    assert len(cands) == 4 * 2 * 4 == 32
    assert not any(c.mount.startswith("flip") for c in cands)
    assert {c.row_parity for c in cands} == {+1, -1}


def test_row_parity_negates_P1_and_nothing_else():
    """Parity -1 is 'the readout's row direction opposes detector y'. It shows
    up as P1 -> -P1 plus a flipped IMAGE; R is untouched (R must stay proper,
    which is exactly why the parity cannot live in the parameter file)."""
    p = convert(TPS21A_A5, "34IDE", "rpy_zxy_pos", +1)
    m = convert(TPS21A_A5, "34IDE", "rpy_zxy_pos", -1)
    assert m.P[0] == pytest.approx(p.P[0])
    assert m.P[1] == pytest.approx(-p.P[1])
    assert m.P[2] == pytest.approx(p.P[2])
    # bounded, not ==: R comes out of the same code path for both parities so
    # it is bit-identical here, but an exact float compare in a test is a
    # platform assertion the moment anything upstream of it changes.
    assert m.R == pytest.approx(p.R, abs=1e-15)
    assert m.row_parity == -1


def test_the_180deg_beam_gauge_leaves_every_predicted_pixel_alone():
    """Measured during the campaign at 2.0e-12 px / 1.4e-14 keV: (34IDE, U) and
    (flip, Rz(180) U) are the same pattern."""
    B = reciprocal_matrix([0.543102] * 3 + [90.0] * 3)
    rng = np.random.default_rng(0)
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    U = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                  [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                  [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    # hmax=10, not 6: at 6 only five reflections survive the panel and energy
    # window, which is too few for "the two poses predict the same pattern" to
    # mean anything.
    rr = range(-10, 11)
    hkl = np.array([[h, k, l] for h in rr for k in rr for l in rr
                    if (h, k, l) != (0, 0, 0) and h % 2 == k % 2 == l % 2],
                   dtype=float)

    def pat(mount, Uu):
        g = convert(TPS21A_A5, mount, "rpy_zxy_pos")
        spec = DetectorSpec(n_pix=(g.nr_px_x, g.nr_px_y), px_size=(g.px_x, g.px_y))
        px, py, en = project(Uu, B, hkl, g.P, rodrigues_to_matrix(g.R), spec)
        ok = ((px >= 0) & (px < g.nr_px_x) & (py >= 0) & (py < g.nr_px_y)
              & (en > 8.74) & (en < 26.0))
        return px[ok], py[ok], en[ok], ok

    p1 = pat("34IDE", U)
    p2 = pat("flip", _Rz(180) @ U)
    # Compare the surviving reflection COUNT and the positions, not the boolean
    # window mask: the mask is a float comparison against the panel edge and the
    # energy cutoff, so a reflection sitting within an ulp of either could flip
    # on one platform's libm and not another's. The physical claim is that the
    # two poses predict the same pattern, and that is what is asserted.
    assert p1[3].sum() == p2[3].sum() > 20
    assert np.abs(p1[0] - p2[0]).max() < 1e-9
    assert np.abs(p1[1] - p2[1]).max() < 1e-9
    assert np.abs(p1[2] - p2[2]).max() < 1e-12


def test_tps_pdf_tth_annotation_selects_the_34IDE_mount():
    """TPS's Detector Geometry.pdf: tth_high LEFT, tth_low RIGHT in the Albula
    display. Only the 34IDE mount puts the 2theta variation across COLUMNS with
    the low end on the right; rot90/rot270 put it across ROWS, rot180 reverses
    it. This is external metrology and settles the mount without indexing."""
    def edges(mount):
        g = convert(TPS21A_A5, mount, "rpy_zxy_pos")
        R = rodrigues_to_matrix(g.R)
        def tth(col, row):
            xd = (col - 0.5*(g.nr_px_x-1))*g.px_x + g.P[0]
            yd = (row - 0.5*(g.nr_px_y-1))*g.px_y + g.P[1]
            v = R @ np.array([xd, yd, g.P[2]]); v /= np.linalg.norm(v)
            return np.degrees(np.arccos(np.clip(v[2], -1, 1)))
        cy, cx = (g.nr_px_y-1)/2, (g.nr_px_x-1)/2
        return tth(0, cy), tth(g.nr_px_x-1, cy), tth(cx, 0), tth(cx, g.nr_px_y-1)

    L, R_, T, Bm = edges("34IDE")
    assert L - R_ > 40.0                      # 2theta spans ~44 deg across columns
    assert abs(T - Bm) < 1.0                  # and is flat across rows
    for other in ("34IDE_rot90", "34IDE_rot180", "34IDE_rot270"):
        l, r, t, b = edges(other)
        assert not (l - r > 40.0 and abs(t - b) < 1.0), f"{other} also matches"
