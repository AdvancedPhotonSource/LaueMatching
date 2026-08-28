"""PSF-blurred segment splat -- the misorientation-gradient footprint.

Measured on real Ti-64 ID6 streaks: excess kurtosis median -0.786 along the
streak (Gaussian 0, top-hat -1.2), 66% flat-topped, top-hat(+PSF) beating a
Gaussian head-to-head 46% to 33%.  A Gaussian footprint cannot make that shape,
which is why the joint fit stalled at 35-75% of cloud intensity.

The tests pin the two properties that matter: the shape really is flat-topped
(negative kurtosis, like the data), and the whole thing degenerates EXACTLY to
the existing Gaussian splat at zero length so the new path cannot drift from the
old one.
"""
from __future__ import annotations

import math

import pytest
import torch

from laue_torch.rasterize import gaussian_splat, streak_splat


def _axis(n, theta=0.0):
    return torch.tensor([[math.cos(theta), math.sin(theta)]],
                        dtype=torch.float64).expand(n, 2).contiguous()


def _moments_along(img, axis, center):
    """Intensity-weighted mean/var/excess-kurtosis along `axis`."""
    Nx, Ny = img.shape
    xs = torch.arange(Nx, dtype=img.dtype)[:, None].expand(Nx, Ny)
    ys = torch.arange(Ny, dtype=img.dtype)[None, :].expand(Nx, Ny)
    u = (xs - center[0]) * axis[0] + (ys - center[1]) * axis[1]
    w = img / img.sum()
    mu = (w * u).sum()
    var = (w * (u - mu) ** 2).sum()
    kurt = (w * (u - mu) ** 4).sum() / var**2 - 3.0
    return float(mu), float(var), float(kurt)


# ── degenerate case: must reproduce the Gaussian exactly ───────────────────


def test_zero_length_reduces_to_gaussian_splat():
    """L=0 must reproduce the Gaussian -- to ~1e-10, not machine precision.

    The kernel is a difference of error functions divided by its own peak.  As
    L -> 0 that is 0/0, handled by flooring the half-length at 1e-6*sigma.  The
    residual disagreement is CANCELLATION, not a modelling error: subtracting
    two nearly-equal erf values loses about six digits.  The floor is already
    near optimal -- shape error goes as (h/sigma)^2 and cancellation as
    eps/(h/sigma), which balance around h/sigma ~ eps^(1/3) ~ 6e-6.  L=0 is not
    a physical case, so ~1e-10 is fine; it only has to not be zeros.
    """
    torch.manual_seed(0)
    n, sigma, window = 10, 2.0, 21
    px = torch.rand(n, dtype=torch.float64) * 30 + 15
    py = torch.rand(n, dtype=torch.float64) * 30 + 15
    inten = torch.rand(n, dtype=torch.float64) + 0.5

    old = gaussian_splat(px, py, inten, (64, 64), sigma=sigma, window=window)
    new = streak_splat(px, py, inten, _axis(n), torch.zeros(n, dtype=torch.float64),
                       sigma_long=sigma, sigma_perp=sigma, n_pix=(64, 64),
                       window=window)
    torch.testing.assert_close(new, old, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("theta", [0.0, 0.7, 2.0])
def test_zero_length_is_axis_independent(theta):
    """With no length there is no streak, so the axis must not matter."""
    n = 4
    px = torch.full((n,), 32.0, dtype=torch.float64)
    py = torch.linspace(20, 44, n, dtype=torch.float64)
    inten = torch.ones(n, dtype=torch.float64)
    ref = streak_splat(px, py, inten, _axis(n, 0.0),
                       torch.zeros(n, dtype=torch.float64), 2.0, 2.0, (64, 64), 21)
    got = streak_splat(px, py, inten, _axis(n, theta),
                       torch.zeros(n, dtype=torch.float64), 2.0, 2.0, (64, 64), 21)
    # Same ~1e-10 cancellation floor as the Gaussian-degeneracy test above.
    torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-9)


# ── the shape is flat-topped, like the data ────────────────────────────────


def test_long_segment_is_flat_topped():
    """Excess kurtosis must go NEGATIVE -- the property a Gaussian cannot have."""
    px = py = torch.tensor([60.0], dtype=torch.float64)
    img = streak_splat(px, py, torch.ones(1, dtype=torch.float64), _axis(1),
                       torch.tensor([40.0], dtype=torch.float64),
                       sigma_long=1.06, sigma_perp=1.06, n_pix=(121, 121), window=81)
    _, _, kurt = _moments_along(img, (1.0, 0.0), (60.0, 60.0))
    assert kurt < -1.0, f"expected top-hat-like kurtosis, got {kurt:+.3f}"


def test_kurtosis_interpolates_between_gaussian_and_tophat():
    """Short segment ~ Gaussian (kurt ~ 0); long segment ~ top-hat (kurt ~ -1.2).

    The Ti-64 data sits in between at -0.79 median, so the model must be able to
    reach that range continuously rather than only at the endpoints.
    """
    px = py = torch.tensor([60.0], dtype=torch.float64)
    seen = []
    for L in (0.0, 4.0, 10.0, 25.0, 60.0):
        img = streak_splat(px, py, torch.ones(1, dtype=torch.float64), _axis(1),
                           torch.tensor([L], dtype=torch.float64),
                           1.06, 1.06, (161, 161), window=121)
        seen.append(_moments_along(img, (1.0, 0.0), (60.0, 60.0))[2])
    assert seen[0] == pytest.approx(0.0, abs=0.05), f"L=0 should be Gaussian: {seen[0]}"
    assert all(b < a + 1e-9 for a, b in zip(seen, seen[1:])), (
        f"kurtosis must fall monotonically with length: {seen}"
    )
    assert min(seen) < -1.0
    # the measured Ti-64 value must be reachable
    assert min(seen) < -0.786 < max(seen)


def test_length_sets_the_measured_extent():
    px = py = torch.tensor([80.0], dtype=torch.float64)
    for L in (10.0, 30.0, 50.0):
        img = streak_splat(px, py, torch.ones(1, dtype=torch.float64), _axis(1),
                           torch.tensor([L], dtype=torch.float64),
                           1.06, 1.06, (161, 161), window=121)
        _, var, _ = _moments_along(img, (1.0, 0.0), (80.0, 80.0))
        # uniform segment variance = L^2/12, plus the PSF
        expect = L * L / 12.0 + 1.06**2
        assert var == pytest.approx(expect, rel=0.05), f"L={L}: {var} vs {expect}"


@pytest.mark.parametrize("theta_deg", [0.0, 30.0, 90.0, 135.0])
def test_streak_points_along_the_requested_axis(theta_deg):
    th = math.radians(theta_deg)
    px = py = torch.tensor([60.0], dtype=torch.float64)
    img = streak_splat(px, py, torch.ones(1, dtype=torch.float64), _axis(1, th),
                       torch.tensor([40.0], dtype=torch.float64),
                       1.06, 1.06, (121, 121), window=91)
    _, var_along, _ = _moments_along(img, (math.cos(th), math.sin(th)), (60.0, 60.0))
    _, var_across, _ = _moments_along(img, (-math.sin(th), math.cos(th)), (60.0, 60.0))
    assert var_along > 10 * var_across, (
        f"streak not aligned at {theta_deg} deg: {var_along} vs {var_across}"
    )


def test_plateau_height_equals_intensity():
    """Peak-normalized like the other splats, so amplitudes stay comparable."""
    px = py = torch.tensor([60.0], dtype=torch.float64)
    img = streak_splat(px, py, torch.tensor([3.5], dtype=torch.float64), _axis(1),
                       torch.tensor([40.0], dtype=torch.float64),
                       1.06, 1.06, (121, 121), window=81)
    assert float(img.max()) == pytest.approx(3.5, rel=1e-6)


# ── plumbing ───────────────────────────────────────────────────────────────


def test_superposition_and_stacking():
    torch.manual_seed(3)
    n = 6
    px = torch.rand(n, dtype=torch.float64) * 40 + 20
    py = torch.rand(n, dtype=torch.float64) * 40 + 20
    L = torch.rand(n, dtype=torch.float64) * 20
    inten = torch.rand(n, dtype=torch.float64) + 0.3
    idx = torch.arange(n) % 3
    st = streak_splat(px, py, inten, _axis(n, 0.4), L, 1.06, 1.5, (80, 80), 41,
                      spot_idx=idx, n_stack=3)
    flat = streak_splat(px, py, inten, _axis(n, 0.4), L, 1.06, 1.5, (80, 80), 41)
    assert st.shape == (3, 80, 80)
    torch.testing.assert_close(st.sum(0), flat, rtol=1e-12, atol=1e-14)


def test_gradients_reach_length_axis_and_position():
    px = torch.tensor([40.0], dtype=torch.float64, requires_grad=True)
    L = torch.tensor([20.0], dtype=torch.float64, requires_grad=True)
    ax = torch.tensor([[1.0, 0.3]], dtype=torch.float64, requires_grad=True)
    img = streak_splat(px, torch.tensor([40.0], dtype=torch.float64),
                       torch.ones(1, dtype=torch.float64), ax, L,
                       1.06, 1.5, (80, 80), 41)
    (img ** 2).sum().backward()
    for name, t in (("px", px), ("length", L), ("axis", ax)):
        assert t.grad is not None and torch.isfinite(t.grad).all(), name
        assert float(t.grad.abs().sum()) > 0, f"{name} got a zero gradient"


def test_rejects_bad_inputs():
    one = torch.ones(1, dtype=torch.float64)
    with pytest.raises(ValueError, match="window must be odd"):
        streak_splat(one, one, one, _axis(1), one, 1.0, 1.0, (16, 16), 8)
    with pytest.raises(ValueError, match=r"axis must be \(N, 2\)"):
        streak_splat(one, one, one, torch.ones(1, 3, dtype=torch.float64), one,
                     1.0, 1.0, (16, 16), 9)
    with pytest.raises(ValueError, match="must be positive"):
        streak_splat(one, one, one, _axis(1), one, 0.0, 1.0, (16, 16), 9)
