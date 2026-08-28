"""Footprint algebra: Jacobian d(pixel)/d(omega) and pixel-space covariance.

The joint fit renders a streaked reflection as an anisotropic Gaussian whose
covariance is ``J Sigma_omega J^T + sigma_psf^2 I``.  If either the Jacobian or
the covariance assembly is wrong the streaks point the wrong way, which is
exactly the failure the wrong-axis null was built to catch -- so both are
checked against independent oracles here.
"""
from __future__ import annotations

import math

import pytest
import torch

from laue_torch.geometry import rodrigues_to_matrix
from laue_torch.jointfit import (
    finite_difference_jacobian,
    pixel_covariance,
    pixel_jacobian,
    spread_covariance,
    tangent_rotation,
)


# ── a projection stand-in with a known closed form ─────────────────────────
#
# Real projection goes through LaueForwardModel; for testing the footprint
# algebra we want something whose Jacobian we can also write down by hand.
# Orthographic projection of rotated reciprocal vectors onto (x, y), scaled:
#     p(OM) = SCALE * (OM @ g)[:2]
# so d p / d omega = SCALE * d/domega (R(omega) OM g)[:2] = -SCALE * [OM g]_x
# (the cross-product matrix), restricted to the first two rows.

SCALE = 1000.0


def _project(om: torch.Tensor, gvecs: torch.Tensor) -> torch.Tensor:
    return SCALE * (gvecs @ om.transpose(-1, -2))[:, :2]


def _analytic_jacobian(om: torch.Tensor, gvecs: torch.Tensor) -> torch.Tensor:
    """d(pixel)/d(omega) for ``_project``, from d(R(w) v)/dw|_0 = -[v]_x."""
    v = gvecs @ om.transpose(-1, -2)                     # (H, 3), rotated vectors
    zero = torch.zeros_like(v[:, 0])
    cross = torch.stack(
        [
            torch.stack([zero, v[:, 2], -v[:, 1]], dim=-1),
            torch.stack([-v[:, 2], zero, v[:, 0]], dim=-1),
            torch.stack([v[:, 1], -v[:, 0], zero], dim=-1),
        ],
        dim=-2,
    )                                                    # (H, 3, 3) = -[v]_x
    return SCALE * cross[:, :2, :]


@pytest.fixture
def setup():
    torch.manual_seed(0)
    om = rodrigues_to_matrix(torch.tensor([0.3, -0.2, 0.15], dtype=torch.float64))
    gvecs = torch.randn(7, 3, dtype=torch.float64)
    return om, gvecs


# ── tangent rotation: the aa=0 gradient trap ───────────────────────────────


def test_tangent_rotation_agrees_with_rodrigues_away_from_zero():
    """Same rotation as the shared helper wherever the helper is well-behaved."""
    for w in ([0.3, -0.2, 0.15], [1.0, 0.0, 0.0], [0.01, 0.02, -0.03]):
        omega = torch.tensor(w, dtype=torch.float64)
        torch.testing.assert_close(
            tangent_rotation(omega), rodrigues_to_matrix(omega),
            rtol=1e-12, atol=1e-14,
        )


def test_tangent_rotation_is_identity_at_zero():
    omega = torch.zeros(3, dtype=torch.float64)
    torch.testing.assert_close(
        tangent_rotation(omega), torch.eye(3, dtype=torch.float64),
        rtol=1e-14, atol=1e-15,
    )


def test_tangent_rotation_has_nonzero_gradient_at_zero():
    """REGRESSION: every Jacobian here is evaluated at exactly omega = 0.

    If the rotation is not differentiable at the origin, d/domega is identically
    zero, every footprint comes out perfectly round, and the anisotropy silently
    vanishes. The derivative of R at the origin is the skew generator, so
    dR[1,0]/domega_z must be +1, not 0.
    """
    omega = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    g = torch.autograd.grad(tangent_rotation(omega)[1, 0], omega)[0]
    assert abs(float(g[2]) - 1.0) < 1e-10, f"expected d/dwz = 1, got {g}"

    # geometry.rodrigues_to_matrix used to return a ZERO gradient here, via a
    # torch.where(near_zero, eye, R) that autograd cannot see through. It was
    # fixed to the smooth sin(t)/t formulation; this asserts the fix stays put,
    # because anyone composing a delta onto a seed orientation
    # (rodrigues_to_matrix(dr) @ U0, starting at dr = 0) gets a dead optimiser
    # otherwise -- the fit returns its input and reports success.
    omega2 = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    g2 = torch.autograd.grad(rodrigues_to_matrix(omega2)[1, 0], omega2)[0]
    assert abs(float(g2[2]) - 1.0) < 1e-10, (
        f"rodrigues_to_matrix lost its gradient at the origin again: {g2}"
    )

    # The two agree in value and derivative at the origin.
    assert torch.allclose(tangent_rotation(torch.zeros(3, dtype=torch.float64)),
                          rodrigues_to_matrix(torch.zeros(3, dtype=torch.float64)))


# ── Jacobian ───────────────────────────────────────────────────────────────


def test_jacobian_matches_analytic(setup):
    om, g = setup
    jac = pixel_jacobian(lambda o: _project(o, g), om)
    assert jac.shape == (7, 2, 3)
    torch.testing.assert_close(jac, _analytic_jacobian(om, g), rtol=1e-10, atol=1e-8)


def test_jacobian_matches_finite_difference(setup):
    """Autograd vs central differences -- independent numerical oracle."""
    om, g = setup
    jac = pixel_jacobian(lambda o: _project(o, g), om)
    fd = finite_difference_jacobian(lambda o: _project(o, g), om, eps=1e-6)
    torch.testing.assert_close(jac, fd, rtol=1e-6, atol=1e-5)


def test_jacobian_rejects_bad_orientation_shape(setup):
    _om, g = setup
    with pytest.raises(ValueError, match=r"om must be \(3, 3\)"):
        pixel_jacobian(lambda o: _project(o, g), torch.eye(4, dtype=torch.float64))


# ── spread covariance ──────────────────────────────────────────────────────


def test_spread_covariance_eigenstructure():
    """sigma_par^2 along the axis, sigma_perp^2 in the perpendicular plane."""
    axis = torch.tensor([1.0, 2.0, -0.5], dtype=torch.float64)
    s_par, s_perp = 0.22 * math.pi / 180, 0.043 * math.pi / 180
    cov = spread_covariance(s_par, s_perp, axis)

    n = axis / axis.norm()
    # The axis is an eigenvector with eigenvalue sigma_par^2.
    torch.testing.assert_close(cov @ n, (s_par**2) * n, rtol=1e-12, atol=1e-18)
    # Any perpendicular vector has eigenvalue sigma_perp^2.
    perp = torch.linalg.cross(n, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    perp = perp / perp.norm()
    torch.testing.assert_close(cov @ perp, (s_perp**2) * perp, rtol=1e-12, atol=1e-18)

    evals = torch.linalg.eigvalsh(cov)
    torch.testing.assert_close(
        evals, torch.tensor([s_perp**2, s_perp**2, s_par**2], dtype=torch.float64),
        rtol=1e-10, atol=1e-18,
    )


def test_spread_covariance_isotropic_is_scaled_identity():
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    cov = spread_covariance(0.01, 0.01, axis)
    torch.testing.assert_close(cov, 1e-4 * torch.eye(3, dtype=torch.float64),
                               rtol=1e-12, atol=1e-18)


def test_spread_covariance_normalizes_axis():
    a = torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(
        spread_covariance(0.02, 0.005, a),
        spread_covariance(0.02, 0.005, a / a.norm()),
        rtol=1e-12, atol=1e-18,
    )


def test_spread_covariance_batched():
    axes = torch.randn(5, 3, dtype=torch.float64)
    cov = spread_covariance(0.02, 0.004, axes)
    assert cov.shape == (5, 3, 3)
    for i in range(5):
        torch.testing.assert_close(
            cov[i], spread_covariance(0.02, 0.004, axes[i]), rtol=1e-12, atol=1e-18
        )


# ── pixel covariance ───────────────────────────────────────────────────────


def test_pixel_covariance_diagonal_case():
    """Hand-computable: diagonal J and diagonal Sigma_omega."""
    jac = torch.zeros(1, 2, 3, dtype=torch.float64)
    jac[0, 0, 0] = 100.0        # px per rad
    jac[0, 1, 1] = 50.0
    cov_w = torch.diag(torch.tensor([4e-4, 1e-4, 9e-4], dtype=torch.float64))
    out = pixel_covariance(jac, cov_w, psf_sigma=2.0)
    expect = torch.tensor(
        [[100.0**2 * 4e-4 + 4.0, 0.0], [0.0, 50.0**2 * 1e-4 + 4.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(out[0], expect, rtol=1e-12, atol=1e-12)


def test_pixel_covariance_zero_spread_is_psf_floor():
    jac = torch.randn(4, 2, 3, dtype=torch.float64)
    cov_w = torch.zeros(3, 3, dtype=torch.float64)
    out = pixel_covariance(jac, cov_w, psf_sigma=1.06)
    expect = (1.06**2) * torch.eye(2, dtype=torch.float64).expand(4, 2, 2)
    torch.testing.assert_close(out, expect, rtol=1e-12, atol=1e-12)


def test_pixel_covariance_symmetric_and_positive_definite():
    torch.manual_seed(1)
    jac = torch.randn(16, 2, 3, dtype=torch.float64) * 5000.0
    cov_w = spread_covariance(
        0.22 * math.pi / 180, 0.043 * math.pi / 180,
        torch.randn(16, 3, dtype=torch.float64),
    )
    out = pixel_covariance(jac, cov_w, psf_sigma=1.06)
    torch.testing.assert_close(out, out.transpose(-1, -2), rtol=0, atol=0)
    assert bool((torch.linalg.eigvalsh(out) > 0).all())


def test_pixel_covariance_streak_is_elongated():
    """The measured Ti-64 spread must produce a long, narrow footprint.

    sigma_par = 0.22 deg at J ~ 5000 px/rad is a ~19 px sigma along the streak,
    while sigma_perp = 0.043 deg is ~3.7 px across -- the 5:1 anisotropy that
    beat the wrong-axis null.
    """
    jac = torch.zeros(1, 2, 3, dtype=torch.float64)
    jac[0, 0, 0] = 5000.0
    jac[0, 1, 1] = 5000.0
    cov_w = spread_covariance(
        0.22 * math.pi / 180, 0.043 * math.pi / 180,
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
    )
    out = pixel_covariance(jac, cov_w, psf_sigma=1.06)[0]
    sig = torch.linalg.eigvalsh(out).sqrt()
    assert sig[1] / sig[0] > 4.0, f"expected elongated footprint, got {sig}"
    assert 15.0 < float(sig[1]) < 25.0, f"streak sigma out of range: {sig[1]}"


def test_pixel_covariance_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"jac must be"):
        pixel_covariance(
            torch.randn(4, 3, 3, dtype=torch.float64),
            torch.eye(3, dtype=torch.float64),
            psf_sigma=1.0,
        )


# ── strain jacobian and multi-mechanism covariance ────────────────────────


def test_strain_jacobian_shape_and_finite_difference(setup):
    """d(pixel)/d(deviatoric strain), checked against central differences."""
    from laue_torch.geometry import deviatoric5_to_symmetric
    from laue_torch.jointfit import strain_jacobian

    om, g = setup

    def proj(o, eps5):
        eps = deviatoric5_to_symmetric(eps5)
        eye = torch.eye(3, dtype=o.dtype)
        return _project((eye - eps) @ o, g)

    jac = strain_jacobian(proj, om)
    assert jac.shape == (7, 2, 5)

    fd = torch.zeros_like(jac)
    h = 1e-6
    for i in range(5):
        d = torch.zeros(5, dtype=torch.float64)
        d[i] = h
        fd[:, :, i] = (proj(om, d) - proj(om, -d)) / (2 * h)
    torch.testing.assert_close(jac, fd, rtol=1e-6, atol=1e-5)


def test_strain_jacobian_rejects_bad_orientation(setup):
    from laue_torch.jointfit import strain_jacobian
    _om, g = setup
    with pytest.raises(ValueError, match=r"om must be \(3, 3\)"):
        strain_jacobian(lambda o, e: _project(o, g),
                        torch.eye(2, dtype=torch.float64))


def test_combined_covariance_matches_single_term():
    """One term must reproduce pixel_covariance exactly."""
    from laue_torch.jointfit import combined_pixel_covariance

    torch.manual_seed(4)
    jac = torch.randn(6, 2, 3, dtype=torch.float64) * 100
    cov = spread_covariance(0.01, 0.002, torch.randn(6, 3, dtype=torch.float64))
    one = combined_pixel_covariance([(jac, cov)], psf_sigma=1.06)
    ref = pixel_covariance(jac, cov, psf_sigma=1.06)
    torch.testing.assert_close(one, ref, rtol=1e-12, atol=1e-14)


def test_combined_covariance_adds_mechanisms():
    """Independent mechanisms add; the PSF floor is counted once, not twice."""
    from laue_torch.jointfit import combined_pixel_covariance

    torch.manual_seed(5)
    jw = torch.randn(4, 2, 3, dtype=torch.float64) * 100
    je = torch.randn(4, 2, 5, dtype=torch.float64) * 50
    cw = spread_covariance(0.01, 0.002, torch.randn(4, 3, dtype=torch.float64))
    ce = (1e-6) * torch.eye(5, dtype=torch.float64)

    both = combined_pixel_covariance([(jw, cw), (je, ce)], psf_sigma=1.06)
    only_w = combined_pixel_covariance([(jw, cw)], psf_sigma=1.06)
    only_e = combined_pixel_covariance([(je, ce)], psf_sigma=1.06)
    psf = (1.06**2) * torch.eye(2, dtype=torch.float64)
    torch.testing.assert_close(both, only_w + only_e - psf, rtol=1e-12, atol=1e-12)
    assert bool((torch.linalg.eigvalsh(both) > 0).all())


def test_combined_covariance_validates_inputs():
    from laue_torch.jointfit import combined_pixel_covariance
    with pytest.raises(ValueError, match="at least one"):
        combined_pixel_covariance([], psf_sigma=1.0)
    with pytest.raises(ValueError, match="does not match"):
        combined_pixel_covariance(
            [(torch.randn(3, 2, 3, dtype=torch.float64),
              torch.eye(5, dtype=torch.float64))], psf_sigma=1.0)
    with pytest.raises(ValueError, match=r"each jac must be"):
        combined_pixel_covariance(
            [(torch.randn(3, 3, 3, dtype=torch.float64),
              torch.eye(3, dtype=torch.float64))], psf_sigma=1.0)


def test_pixel_covariance_is_differentiable():
    """Gradients must reach the spread parameters -- the outer loop needs them."""
    s_par = torch.tensor(0.004, dtype=torch.float64, requires_grad=True)
    jac = torch.randn(3, 2, 3, dtype=torch.float64) * 1000.0
    cov_w = spread_covariance(s_par, 0.0005,
                              torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))
    pixel_covariance(jac, cov_w, psf_sigma=1.06).sum().backward()
    assert s_par.grad is not None and torch.isfinite(s_par.grad)
    assert abs(float(s_par.grad)) > 0
