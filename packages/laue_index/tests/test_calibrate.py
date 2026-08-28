"""Tests for laue_index.calibrate.

The interesting cases here are the ones that actually went wrong on the Eiger
1M campaign at 34-ID-E: the mirrored solution that fits perfectly, the
row/column reading of a supplied reciprocal matrix that norms cannot
distinguish, and the beam-axis gauge that makes the orientation-free problem
rank-deficient.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from laue_index.calibrate import (
    Anchor,
    DetectorSpec,
    calibrate,
    matrix_to_rodrigues,
    orientation_candidates,
    project,
    reciprocal_matrix,
    rodrigues_to_matrix,
)

# A real 34-ID-E Eiger2 CdTe 1M geometry (Sheyfer_20260801).
SPEC = DetectorSpec(n_pix=(1028, 1062), px_size=(75e-6, 75e-6))
P_TRUE = (0.005895450006, 0.005050106521, 0.111976310851)
R_TRUE = (-0.010749055619, 2.208983593525, 2.226632777789)
SI = (0.543102, 0.543102, 0.543102, 90.0, 90.0, 90.0)


def diamond_hkl(m: int = 12) -> np.ndarray:
    out = []
    for h in range(-m, m + 1):
        for k in range(-m, m + 1):
            for l in range(-m, m + 1):
                if h == k == l == 0:
                    continue
                par = (h % 2, k % 2, l % 2)
                if par == (1, 1, 1):
                    out.append((h, k, l))
                elif par == (0, 0, 0) and (h + k + l) % 4 == 0:
                    out.append((h, k, l))
    return np.array(out, dtype=float)


def synth(seed: int = 3, n_min: int = 12):
    """A synthetic pattern on the real geometry: returns (U, hkl_on, obs)."""
    B = reciprocal_matrix(SI)
    H = diamond_hkl()
    rng = np.random.default_rng(seed)
    for _ in range(400):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        px, py, E = project(Q, B, H, P_TRUE, rodrigues_to_matrix(R_TRUE), SPEC)
        on = ((px >= 0) & (px < SPEC.n_pix[0]) & (py >= 0) & (py < SPEC.n_pix[1])
              & (E >= 5.0) & (E <= 30.0))
        if on.sum() >= n_min:
            return Q, H[on], np.stack([px[on], py[on]], axis=1), E[on]
    raise RuntimeError("no orientation put enough reflections on the panel")


def test_reciprocal_matrix_cubic():
    B = reciprocal_matrix(SI)
    expect = 2 * math.pi / SI[0]
    assert np.allclose(np.linalg.norm(B, axis=0), expect, rtol=1e-10)
    assert np.allclose(B, np.diag([expect] * 3), atol=1e-9)


def test_rodrigues_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = rng.normal(size=3)
        v *= rng.uniform(0.01, 3.0) / np.linalg.norm(v)
        M = rodrigues_to_matrix(v)
        assert np.allclose(M @ M.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(M) == pytest.approx(1.0, abs=1e-12)
        assert np.allclose(rodrigues_to_matrix(matrix_to_rodrigues(M)), M, atol=1e-10)


def test_recovers_the_truth_from_a_crude_start():
    U, hkl, obs, _ = synth()
    anchors = [Anchor(tuple(int(x) for x in h), tuple(o))
               for h, o in zip(hkl, obs)]
    res = calibrate(
        anchors,
        recip=U * (2 * math.pi / SI[0]),
        lattice=SI,
        spec=SPEC,
        frame_provenance="synthetic truth",
        initial_guess=(0.004, 0.004, 0.104,
                       R_TRUE[0] + 0.05, R_TRUE[1] - 0.04, R_TRUE[2] + 0.03),
    )
    assert res.rms_px < 1e-6
    assert np.allclose(res.p_array, P_TRUE, atol=1e-9)
    assert np.allclose(rodrigues_to_matrix(res.r_array),
                       rodrigues_to_matrix(R_TRUE), atol=1e-8)


def test_three_spots_are_enough_and_two_are_not():
    U, hkl, obs, _ = synth()
    mk = lambda n: [Anchor(tuple(int(x) for x in h), tuple(o))
                    for h, o in zip(hkl[:n], obs[:n])]
    kw = dict(recip=U * (2 * math.pi / SI[0]), lattice=SI, spec=SPEC,
              frame_provenance="synthetic truth",
              initial_guess=(0.004, 0.004, 0.106,
                             R_TRUE[0] + 0.02, R_TRUE[1], R_TRUE[2] - 0.02))
    res = calibrate(mk(3), **kw)
    assert np.allclose(res.p_array, P_TRUE, atol=1e-8)

    with pytest.raises(ValueError, match="at least 3"):
        calibrate(mk(2), **kw)


def test_frame_provenance_is_mandatory():
    """The module must refuse to invent the rotation about the beam."""
    U, hkl, obs, _ = synth()
    anchors = [Anchor(tuple(int(x) for x in h), tuple(o))
               for h, o in zip(hkl, obs)]
    for bad in ("", "   "):
        with pytest.raises(ValueError, match="frame_provenance is required"):
            calibrate(anchors, recip=U, lattice=SI, spec=SPEC,
                      frame_provenance=bad)


def test_norms_cannot_distinguish_row_from_column_but_the_fit_can():
    """The trap that cost the real campaign a day."""
    U, hkl, obs, _ = synth()
    recip = U * (2 * math.pi / SI[0])

    # Both readings have identical norms -- so norms can never break the tie.
    row_norms = np.linalg.norm(recip.T, axis=0)
    col_norms = np.linalg.norm(recip, axis=0)
    assert np.allclose(row_norms, col_norms, rtol=1e-12)

    anchors = [Anchor(tuple(int(x) for x in h), tuple(o))
               for h, o in zip(hkl, obs)]
    res = calibrate(anchors, recip=recip, lattice=SI, spec=SPEC,
                    frame_provenance="synthetic truth",
                    initial_guess=(0.004, 0.004, 0.106,
                                   R_TRUE[0], R_TRUE[1], R_TRUE[2]))
    # The projection decides, and it picks the reading the data was built with.
    assert res.convention == "columns"
    assert res.convention_scores["columns"]["rms_px"] < 1e-6
    assert res.convention_scores["rows"]["rms_px"] > 1.0


def test_orientation_candidates_are_transposes():
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    cands = orientation_candidates(Q * 11.5)
    assert np.allclose(cands["columns"], cands["rows"].T, atol=1e-12)


def test_the_beam_axis_gauge_is_exact():
    """Rotating detector and crystal together about the beam changes nothing."""
    U, hkl, _, _ = synth()
    B = reciprocal_matrix(SI)
    R_mat = rodrigues_to_matrix(R_TRUE)
    px0, py0, e0 = project(U, B, hkl, P_TRUE, R_mat, SPEC)
    for phi in (5.0, 68.523, 137.0, -90.0):
        t = math.radians(phi)
        Rz = np.array([[math.cos(t), -math.sin(t), 0],
                       [math.sin(t), math.cos(t), 0],
                       [0, 0, 1.0]])
        px, py, e = project(Rz @ U, B, hkl, P_TRUE, Rz @ R_mat, SPEC)
        assert np.max(np.abs(px - px0)) < 1e-8
        assert np.max(np.abs(py - py0)) < 1e-8
        assert np.max(np.abs(e - e0)) < 1e-12


def test_conditioning_reports_the_gauge_as_flat_and_the_fit_as_full_rank():
    U, hkl, obs, _ = synth()
    anchors = [Anchor(tuple(int(x) for x in h), tuple(o))
               for h, o in zip(hkl, obs)]
    res = calibrate(anchors, recip=U * (2 * math.pi / SI[0]), lattice=SI,
                    spec=SPEC, frame_provenance="synthetic truth",
                    initial_guess=(0.004, 0.004, 0.106, *R_TRUE))
    assert not res.conditioning.degenerate
    assert res.conditioning.condition_number < 1e5
    # with the orientation free there IS a flat direction
    assert res.conditioning.gauge_flat_ratio < 1e-6


def test_energies_do_not_move_the_pose():
    """Energy is a function of (orientation, hkl) alone."""
    U, hkl, _, _ = synth()
    B = reciprocal_matrix(SI)
    _, _, e0 = project(U, B, hkl, P_TRUE, rodrigues_to_matrix(R_TRUE), SPEC)
    moved_P = (P_TRUE[0] + 0.005, P_TRUE[1], P_TRUE[2] + 0.030)
    moved_R = (R_TRUE[0] + 0.3, R_TRUE[1] + 0.2, R_TRUE[2] + 0.1)
    px, _, e1 = project(U, B, hkl, moved_P, rodrigues_to_matrix(moved_R), SPEC)
    assert np.max(np.abs(e1 - e0)) < 1e-12       # energies: unchanged
    assert np.max(np.abs(px)) > 0                # but the pixels did move


def test_mirror_solution_is_rejected():
    """A negative standoff fits the angles but is unphysical."""
    U, hkl, obs, _ = synth()
    anchors = [Anchor(tuple(int(x) for x in h), tuple(o))
               for h, o in zip(hkl, obs)]
    res = calibrate(anchors, recip=U * (2 * math.pi / SI[0]), lattice=SI,
                    spec=SPEC, frame_provenance="synthetic truth",
                    initial_guess=(0.004, 0.004, -0.106, *R_TRUE))
    assert res.p_array[2] > 0


def test_held_out_validation_clears_a_measured_null():
    U, hkl, obs, _ = synth(n_min=20)
    n_fit = 6
    anchors = [Anchor(tuple(int(x) for x in h), tuple(o))
               for h, o in zip(hkl[:n_fit], obs[:n_fit])]
    res = calibrate(anchors, recip=U * (2 * math.pi / SI[0]), lattice=SI,
                    spec=SPEC, frame_provenance="synthetic truth",
                    initial_guess=(0.004, 0.004, 0.106, *R_TRUE),
                    held_out_hkl=hkl[n_fit:], observed_spots=obs[n_fit:],
                    tolerance_px=3.0, null_trials=300)
    v = res.validation
    assert v is not None
    assert v.n_used_in_fit == n_fit
    assert v.n_held_out_matched == v.n_held_out
    assert v.clears_null
    assert abs(v.residual_dx[0]) < 1e-6


def test_params_text_carries_the_provenance():
    U, hkl, obs, _ = synth()
    anchors = [Anchor(tuple(int(x) for x in h), tuple(o))
               for h, o in zip(hkl, obs)]
    prov = "recip1 from LaueGo CalibrationListOrange0, wire-calibrated 2026-07-07"
    res = calibrate(anchors, recip=U * (2 * math.pi / SI[0]), lattice=SI,
                    spec=SPEC, frame_provenance=prov,
                    initial_guess=(0.004, 0.004, 0.106, *R_TRUE))
    text = res.params_text(SI, 227, "F", spec=SPEC)
    assert prov in text
    assert "INHERITED" in text
    assert "P_Array" in text and "R_Array" in text
    assert "NrPxX 1028" in text
