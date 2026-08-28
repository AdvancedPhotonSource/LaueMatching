"""Anisotropic Gaussian splat.

This is the primitive that paints a streaked reflection, so the tests check the
two things that would silently corrupt a joint fit: that the rendered shape
really has the covariance it was handed (orientation AND width), and that the
isotropic case is bit-compatible with the existing ``gaussian_splat`` so the new
path cannot drift from the old one.
"""
from __future__ import annotations

import math

import pytest
import torch

from laue_torch.rasterize import anisotropic_gaussian_splat, gaussian_splat


def _iso_cov(n: int, sigma: float, dtype=torch.float64) -> torch.Tensor:
    return (sigma**2) * torch.eye(2, dtype=dtype).expand(n, 2, 2).contiguous()


def _rot2(theta: float, dtype=torch.float64) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=dtype)


def _image_moments(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Intensity-weighted centroid and 2x2 second central moment of an image."""
    Nx, Ny = img.shape
    xs = torch.arange(Nx, dtype=img.dtype)[:, None].expand(Nx, Ny)
    ys = torch.arange(Ny, dtype=img.dtype)[None, :].expand(Nx, Ny)
    w = img / img.sum()
    mx, my = (w * xs).sum(), (w * ys).sum()
    dx, dy = xs - mx, ys - my
    cov = torch.stack([
        torch.stack([(w * dx * dx).sum(), (w * dx * dy).sum()]),
        torch.stack([(w * dx * dy).sum(), (w * dy * dy).sum()]),
    ])
    return torch.stack([mx, my]), cov


# ── compatibility with the existing isotropic splat ────────────────────────


def test_isotropic_case_matches_gaussian_splat_exactly():
    torch.manual_seed(0)
    n, sigma, window = 12, 2.0, 15
    px = torch.rand(n, dtype=torch.float64) * 40 + 10
    py = torch.rand(n, dtype=torch.float64) * 40 + 10
    inten = torch.rand(n, dtype=torch.float64) + 0.5

    old = gaussian_splat(px, py, inten, (64, 64), sigma=sigma, window=window)
    new = anisotropic_gaussian_splat(px, py, inten, _iso_cov(n, sigma),
                                     (64, 64), window=window)
    torch.testing.assert_close(new, old, rtol=1e-12, atol=1e-14)


def test_isotropic_stack_matches_gaussian_splat_stack():
    torch.manual_seed(1)
    n, sigma, window = 9, 1.5, 11
    px = torch.rand(n, dtype=torch.float64) * 30 + 10
    py = torch.rand(n, dtype=torch.float64) * 30 + 10
    inten = torch.ones(n, dtype=torch.float64)
    idx = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2])

    old = gaussian_splat(px, py, inten, (48, 48), sigma=sigma, window=window,
                         grain_idx=idx, n_grains=3)
    new = anisotropic_gaussian_splat(px, py, inten, _iso_cov(n, sigma),
                                     (48, 48), window=window,
                                     spot_idx=idx, n_stack=3)
    assert new.shape == (3, 48, 48)
    torch.testing.assert_close(new, old, rtol=1e-12, atol=1e-14)


# ── the rendered shape really has the requested covariance ─────────────────


def test_rendered_second_moment_recovers_covariance():
    """A wide window and a well-sampled kernel must reproduce Sigma."""
    N, window = 121, 81
    cov = torch.tensor([[36.0, 0.0], [0.0, 4.0]], dtype=torch.float64)[None]
    px = torch.tensor([60.0], dtype=torch.float64)
    py = torch.tensor([60.0], dtype=torch.float64)
    img = anisotropic_gaussian_splat(px, py, torch.ones(1, dtype=torch.float64),
                                     cov, (N, N), window=window)
    centroid, moments = _image_moments(img)
    torch.testing.assert_close(centroid, torch.tensor([60.0, 60.0],
                                                      dtype=torch.float64),
                               rtol=1e-8, atol=1e-6)
    torch.testing.assert_close(moments, cov[0], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("theta_deg", [0.0, 30.0, 45.0, 75.0, 120.0])
def test_rendered_moment_tracks_rotation(theta_deg):
    """Rotating Sigma must rotate the rendered streak by the same angle.

    This is the property the wrong-axis null tested statistically on real data;
    here it is pinned exactly, so a transposed or sign-flipped covariance cannot
    slip through.
    """
    N, window = 121, 81
    R = _rot2(math.radians(theta_deg))
    base = torch.diag(torch.tensor([36.0, 4.0], dtype=torch.float64))
    cov = (R @ base @ R.T)[None]
    px = py = torch.tensor([60.0], dtype=torch.float64)
    img = anisotropic_gaussian_splat(px, py, torch.ones(1, dtype=torch.float64),
                                     cov, (N, N), window=window)
    _, moments = _image_moments(img)
    torch.testing.assert_close(moments, cov[0], rtol=1e-4, atol=1e-3)

    # The principal axis of the rendered image points along the requested one.
    evals, evecs = torch.linalg.eigh(moments)
    major = evecs[:, int(evals.argmax())]
    want = R @ torch.tensor([1.0, 0.0], dtype=torch.float64)
    cos = abs(float(major @ want))
    assert cos > 0.999, f"major axis off: cos={cos} at {theta_deg} deg"


def test_elongated_footprint_is_not_round():
    """Guards the aa=0 failure mode end-to-end: a streak must not render round."""
    cov = torch.tensor([[400.0, 0.0], [0.0, 1.1]], dtype=torch.float64)[None]
    px = py = torch.tensor([60.0], dtype=torch.float64)
    img = anisotropic_gaussian_splat(px, py, torch.ones(1, dtype=torch.float64),
                                     cov, (121, 121), window=101)
    _, moments = _image_moments(img)
    evals = torch.linalg.eigvalsh(moments)
    assert (evals.max() / evals.min()).sqrt() > 15.0


# ── superposition, gradients, validation ───────────────────────────────────


def test_stack_sums_to_unstacked():
    """Contributions ADD -- the premise of the whole joint-fit approach."""
    torch.manual_seed(2)
    n = 8
    px = torch.rand(n, dtype=torch.float64) * 30 + 10
    py = torch.rand(n, dtype=torch.float64) * 30 + 10
    inten = torch.rand(n, dtype=torch.float64) + 0.2
    cov = _iso_cov(n, 2.0)
    idx = torch.arange(n) % 4
    stacked = anisotropic_gaussian_splat(px, py, inten, cov, (48, 48), window=15,
                                         spot_idx=idx, n_stack=4)
    flat = anisotropic_gaussian_splat(px, py, inten, cov, (48, 48), window=15)
    torch.testing.assert_close(stacked.sum(0), flat, rtol=1e-12, atol=1e-14)


def test_gradients_flow_to_position_and_covariance():
    """The outer loop optimizes through both -- neither may be detached."""
    px = torch.tensor([20.4], dtype=torch.float64, requires_grad=True)
    py = torch.tensor([20.7], dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(9.0, dtype=torch.float64, requires_grad=True)
    cov = (scale * torch.eye(2, dtype=torch.float64))[None]
    img = anisotropic_gaussian_splat(px, py, torch.ones(1, dtype=torch.float64),
                                     cov, (48, 48), window=21)
    (img**2).sum().backward()
    for name, t in (("px", px), ("py", py), ("cov scale", scale)):
        assert t.grad is not None and torch.isfinite(t.grad).all(), name
        assert float(t.grad.abs().sum()) > 0, f"{name} got a zero gradient"


def test_rejects_even_window():
    with pytest.raises(ValueError, match="window must be odd"):
        anisotropic_gaussian_splat(
            torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64), _iso_cov(1, 1.0), (16, 16), window=8,
        )


def test_rejects_singular_covariance():
    bad = torch.zeros(1, 2, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="positive-definite"):
        anisotropic_gaussian_splat(
            torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64), bad, (16, 16), window=9,
        )


def test_rejects_mismatched_cov_count():
    with pytest.raises(ValueError, match="cov has 2 spots"):
        anisotropic_gaussian_splat(
            torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64),
            torch.ones(3, dtype=torch.float64), _iso_cov(2, 1.0), (16, 16), window=9,
        )


def test_rejects_wrong_cov_shape():
    with pytest.raises(ValueError, match=r"cov must be \(N, 2, 2\)"):
        anisotropic_gaussian_splat(
            torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            torch.eye(3, dtype=torch.float64).expand(2, 3, 3),
            (16, 16), window=9,
        )
