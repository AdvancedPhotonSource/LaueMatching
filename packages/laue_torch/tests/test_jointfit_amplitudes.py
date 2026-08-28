"""Design matrix + non-negative amplitude solve.

The amplitudes are what let the joint fit avoid predicting absolute intensities
(no I0(E) available).  That freedom is also the main risk: with a free amplitude
per reflection the model is flexible, so the solver has to be exactly right and
the non-negativity has to actually bind.  ``scipy.optimize.nnls`` is used as an
independent oracle.
"""
from __future__ import annotations

import math

import pytest
import torch

from laue_torch.jointfit import (
    ROI,
    build_basis,
    gram_and_rhs,
    pixel_covariance,
    rois_from_labels,
    solve_amplitudes,
    solve_nnls,
    spots_in_roi,
    spread_covariance,
    suggested_window,
)
from laue_torch.jointfit.amplitudes import residual

scipy_nnls = pytest.importorskip("scipy.optimize").nnls


def _iso_cov(n: int, sigma: float) -> torch.Tensor:
    return (sigma**2) * torch.eye(2, dtype=torch.float64).expand(n, 2, 2).contiguous()


# ── ROI plumbing ───────────────────────────────────────────────────────────


def test_roi_crop_and_contains():
    img = torch.arange(100, dtype=torch.float64).reshape(10, 10)
    roi = ROI(x0=2, y0=3, nx=4, ny=5)
    assert roi.shape == (4, 5) and roi.n_pixels == 20
    torch.testing.assert_close(roi.crop(img), img[2:6, 3:8])
    px = torch.tensor([2.0, 1.9, 5.9, 6.0], dtype=torch.float64)
    py = torch.tensor([3.0, 3.0, 7.9, 7.0], dtype=torch.float64)
    assert roi.contains(px, py).tolist() == [True, False, True, False]


def test_rois_from_labels_pads_and_clips():
    labels = torch.zeros(40, 40, dtype=torch.long)
    labels[10:14, 20:23] = 7
    labels[0:2, 0:2] = 9                       # touches the corner
    rois = rois_from_labels(labels, [7, 9], pad=5)
    assert rois[0] == ROI(x0=5, y0=15, nx=14, ny=13)
    # Clipped at the image edge rather than going negative.
    assert rois[1].x0 == 0 and rois[1].y0 == 0


def test_rois_from_labels_rejects_missing_label():
    labels = torch.zeros(8, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="label 3 not present"):
        rois_from_labels(labels, [3])


def test_spots_in_roi_margin_admits_spillover():
    """A spot just outside the ROI still paints into it and must be kept."""
    roi = ROI(x0=10, y0=10, nx=20, ny=20)
    px = torch.tensor([15.0, 5.0, 40.0], dtype=torch.float64)
    py = torch.tensor([15.0, 15.0, 15.0], dtype=torch.float64)
    assert spots_in_roi(px, py, roi).tolist() == [0]
    assert spots_in_roi(px, py, roi, margin=6.0).tolist() == [0, 1]


def test_suggested_window_scales_with_footprint():
    small = suggested_window(_iso_cov(1, 1.06))
    big = suggested_window(_iso_cov(1, 19.0))
    assert small % 2 == 1 and big % 2 == 1
    assert big > small
    assert big >= 2 * int(4 * 19.0) + 1


# ── basis construction ─────────────────────────────────────────────────────


def test_basis_shape_and_superposition():
    """sum_k a_k B_k must equal a direct render of the same spots."""
    roi = ROI(x0=0, y0=0, nx=60, ny=60)
    px = torch.tensor([20.0, 35.5], dtype=torch.float64)
    py = torch.tensor([25.0, 40.25], dtype=torch.float64)
    cov = _iso_cov(2, 3.0)
    basis = build_basis(px, py, cov, roi, window=25)
    assert basis.shape == (2, roi.n_pixels)

    amps = torch.tensor([2.0, 0.5], dtype=torch.float64)
    from laue_torch.rasterize import anisotropic_gaussian_splat
    direct = anisotropic_gaussian_splat(px, py, amps, cov, roi.shape, window=25)
    torch.testing.assert_close((amps @ basis).reshape(roi.shape), direct,
                               rtol=1e-12, atol=1e-14)


def test_basis_respects_roi_offset():
    """ROI-local rendering must place the spot at the right absolute pixel."""
    px = torch.tensor([70.0], dtype=torch.float64)
    py = torch.tensor([80.0], dtype=torch.float64)
    cov = _iso_cov(1, 2.0)
    roi = ROI(x0=60, y0=70, nx=20, ny=20)
    img = build_basis(px, py, cov, roi, window=15).reshape(roi.shape)
    peak = torch.nonzero(img == img.max())[0]
    assert (int(peak[0]) + roi.x0, int(peak[1]) + roi.y0) == (70, 80)


def test_basis_empty_input():
    roi = ROI(x0=0, y0=0, nx=8, ny=8)
    empty = torch.zeros(0, dtype=torch.float64)
    basis = build_basis(empty, empty, torch.zeros(0, 2, 2, dtype=torch.float64),
                        roi, window=5)
    assert basis.shape == (0, 64)


def test_basis_is_differentiable_in_position_and_covariance():
    px = torch.tensor([30.0], dtype=torch.float64, requires_grad=True)
    py = torch.tensor([30.0], dtype=torch.float64, requires_grad=True)
    s = torch.tensor(9.0, dtype=torch.float64, requires_grad=True)
    cov = (s * torch.eye(2, dtype=torch.float64))[None]
    roi = ROI(x0=0, y0=0, nx=60, ny=60)
    build_basis(px, py, cov, roi, window=21).pow(2).sum().backward()
    for name, t in (("px", px), ("py", py), ("sigma", s)):
        assert t.grad is not None and float(t.grad.abs().sum()) > 0, name


# ── NNLS solver ────────────────────────────────────────────────────────────


def _random_problem(k=6, p=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    B = torch.rand(k, p, generator=g, dtype=torch.float64)
    a_true = torch.rand(k, generator=g, dtype=torch.float64) + 0.1
    return B, a_true, a_true @ B


def test_nnls_recovers_exact_nonnegative_solution():
    B, a_true, y = _random_problem()
    sol = solve_amplitudes(B, y, max_iter=20000, tol=1e-14)
    assert sol.converged, f"did not converge: max_step={sol.max_step}"
    torch.testing.assert_close(sol.amplitudes, a_true, rtol=1e-6, atol=1e-8)


def test_nnls_matches_scipy_oracle():
    """Independent implementation, including where the constraint binds."""
    g = torch.Generator().manual_seed(3)
    B = torch.rand(5, 80, generator=g, dtype=torch.float64)
    # A target that a non-negative combination cannot reach exactly, so the
    # active set is non-trivial and the two solvers must agree on WHICH
    # amplitudes get clamped to zero.
    y = torch.rand(80, generator=g, dtype=torch.float64) - 0.35
    sol = solve_amplitudes(B, y, max_iter=50000, tol=1e-15)
    ref, _ = scipy_nnls(B.T.numpy(), y.numpy())
    torch.testing.assert_close(sol.amplitudes, torch.as_tensor(ref, dtype=torch.float64),
                               rtol=1e-4, atol=1e-6)


def test_nnls_enforces_non_negativity():
    """Non-negativity must bind: no negative 'explaining away'."""
    g = torch.Generator().manual_seed(5)
    B = torch.rand(4, 60, generator=g, dtype=torch.float64)
    y = B[0] - 2.0 * B[1]                     # only reachable with a negative
    sol = solve_amplitudes(B, y, max_iter=20000, tol=1e-14)
    assert bool((sol.amplitudes >= 0).all())
    assert float(sol.amplitudes[1]) == pytest.approx(0.0, abs=1e-9)


def test_nnls_zero_target_gives_zero_amplitudes():
    B, _, _ = _random_problem()
    sol = solve_amplitudes(B, torch.zeros(B.shape[1], dtype=torch.float64))
    torch.testing.assert_close(sol.amplitudes, torch.zeros(6, dtype=torch.float64),
                               rtol=0, atol=1e-12)


def test_nnls_handles_duplicate_basis_with_ridge():
    """Two reflections rendered on top of each other: G is singular.

    The split between them is genuinely undetermined; the solver must stay
    finite and reproduce the total rather than diverge.
    """
    g = torch.Generator().manual_seed(7)
    row = torch.rand(50, generator=g, dtype=torch.float64)
    B = torch.stack([row, row.clone()])
    y = 3.0 * row
    sol = solve_amplitudes(B, y, ridge=1e-9, max_iter=20000, tol=1e-14)
    assert torch.isfinite(sol.amplitudes).all()
    assert float(sol.amplitudes.sum()) == pytest.approx(3.0, rel=1e-4)


def test_nnls_reports_non_convergence_rather_than_hiding_it():
    B, _, y = _random_problem(k=8, p=300, seed=11)
    sol = solve_amplitudes(B, y, max_iter=2, tol=1e-16)
    assert sol.n_iter == 2 and not sol.converged
    assert math.isfinite(sol.max_step)


def test_nnls_warm_start_agrees_with_cold_start():
    B, a_true, y = _random_problem(seed=13)
    cold = solve_amplitudes(B, y, max_iter=20000, tol=1e-14)
    gram, rhs = gram_and_rhs(B, y)
    warm = solve_nnls(gram, rhs, max_iter=20000, tol=1e-14,
                      init=a_true * 0.5)
    torch.testing.assert_close(warm.amplitudes, cold.amplitudes,
                               rtol=1e-6, atol=1e-8)


def test_gram_and_rhs_rejects_pixel_mismatch():
    B = torch.rand(3, 20, dtype=torch.float64)
    with pytest.raises(ValueError, match="basis has 20 pixels but target has 19"):
        gram_and_rhs(B, torch.rand(19, dtype=torch.float64))


def test_solve_amplitudes_accepts_2d_target():
    roi = ROI(x0=0, y0=0, nx=30, ny=30)
    px = torch.tensor([15.0], dtype=torch.float64)
    py = torch.tensor([15.0], dtype=torch.float64)
    basis = build_basis(px, py, _iso_cov(1, 2.0), roi, window=15)
    image = (2.5 * basis).reshape(roi.shape)          # 2-D, as it comes off disk
    sol = solve_amplitudes(basis, image, max_iter=20000, tol=1e-14)
    assert float(sol.amplitudes[0]) == pytest.approx(2.5, rel=1e-6)


# ── the piece the outer loop depends on ────────────────────────────────────


def test_residual_gradient_flows_to_geometry_with_detached_amplitudes():
    """Envelope theorem in practice: a* detached, gradient still reaches px.

    If this ever returns zero the outer loop is silently not optimizing.
    """
    roi = ROI(x0=0, y0=0, nx=60, ny=60)
    truth_x = torch.tensor([30.0], dtype=torch.float64)
    truth_y = torch.tensor([30.0], dtype=torch.float64)
    cov = _iso_cov(1, 3.0)
    target = (4.0 * build_basis(truth_x, truth_y, cov, roi, window=25)).detach()

    px = torch.tensor([27.0], dtype=torch.float64, requires_grad=True)
    py = torch.tensor([30.0], dtype=torch.float64, requires_grad=True)
    basis = build_basis(px, py, cov, roi, window=25)
    sol = solve_amplitudes(basis, target.reshape(-1))
    assert not sol.amplitudes.requires_grad          # detached by default
    residual(basis, target.reshape(-1), sol.amplitudes).backward()
    assert px.grad is not None and float(px.grad.abs()) > 0
    # Misplaced along +x by 3 px: the residual must decrease by moving right,
    # i.e. the gradient w.r.t. px is negative.
    assert float(px.grad) < 0, f"gradient points the wrong way: {px.grad}"


def test_residual_is_zero_at_the_truth():
    roi = ROI(x0=0, y0=0, nx=50, ny=50)
    px = torch.tensor([25.0, 33.0], dtype=torch.float64)
    py = torch.tensor([25.0, 18.0], dtype=torch.float64)
    cov = _iso_cov(2, 2.5)
    basis = build_basis(px, py, cov, roi, window=21)
    target = torch.tensor([3.0, 1.5], dtype=torch.float64) @ basis
    sol = solve_amplitudes(basis, target, max_iter=20000, tol=1e-14)
    assert float(residual(basis, target, sol.amplitudes)) < 1e-12


def test_streaked_footprint_amplitudes_recover():
    """End-to-end with a real streak footprint, not a toy isotropic blob."""
    jac = torch.zeros(2, 2, 3, dtype=torch.float64)
    jac[:, 0, 0] = 5000.0
    jac[:, 1, 1] = 5000.0
    cov_w = spread_covariance(
        0.22 * math.pi / 180, 0.043 * math.pi / 180,
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64),
    )
    cov = pixel_covariance(jac, cov_w, psf_sigma=1.06)
    roi = ROI(x0=0, y0=0, nx=160, ny=160)
    px = torch.tensor([60.0, 100.0], dtype=torch.float64)
    py = torch.tensor([80.0, 80.0], dtype=torch.float64)
    window = suggested_window(cov)
    basis = build_basis(px, py, cov, roi, window=window)
    a_true = torch.tensor([5.0, 2.0], dtype=torch.float64)
    sol = solve_amplitudes(basis, a_true @ basis, max_iter=50000, tol=1e-14)
    torch.testing.assert_close(sol.amplitudes, a_true, rtol=1e-4, atol=1e-6)
