"""Tests for the 2D coded-aperture mask (single-shot designs).

Covers pattern generators (MURA, random binary, pinhole) and the
``CodedApertureMask2D`` forward, including a gradcheck on the
single-shot inverse-problem parameters.
"""
from __future__ import annotations

import math

import pytest
import torch

from laue_torch.coded_aperture import (
    CodedApertureMask2D,
    build_mura_pattern,
    build_pinhole_pattern,
    build_random_binary_pattern,
    mu_au,
)


DTYPE = torch.float64


def _wl_A_at_keV(E_keV: float) -> float:
    return 12.398419843320026 / E_keV


# ── Pattern generators ────────────────────────────────────────────────────


def test_mura_construction_basic():
    p = 13
    A = build_mura_pattern(p)
    assert A.shape == (p, p)
    assert A.dtype == torch.int64
    assert torch.all((A == 0) | (A == 1))
    # Open fraction near 0.5 (delta-autocorrelation MURA property).
    fill = float(A.sum()) / (p * p)
    assert 0.4 < fill < 0.6


def test_mura_requires_prime_one_mod_four():
    # 6 is not prime
    with pytest.raises(ValueError):
        build_mura_pattern(6)
    # 7 ≡ 3 (mod 4) — not a valid MURA order
    with pytest.raises(ValueError):
        build_mura_pattern(7)


def test_mura_autocorrelation_near_delta():
    """MURA periodic autocorrelation: (p+1)/2 peak, ~0 elsewhere up to scale.

    We just check the peak/floor ratio is high — the exact off-axis
    structure depends on the +1/−1 mapping convention but it should be
    far from a uniform constant.
    """
    p = 29
    A = build_mura_pattern(p)
    A_f = A.to(torch.float64) - 0.5    # zero-mean for AC computation
    # 2-D circular autocorrelation via FFT.
    F = torch.fft.fft2(A_f)
    AC = torch.fft.ifft2(F * F.conj()).real
    peak = AC[0, 0].item()
    off = AC[1:, 1:].abs().max().item()
    assert peak > 5.0 * off, f"peak/off = {peak / off:.3f}"


def test_random_binary_pattern_fill_factor():
    A = build_random_binary_pattern(64, fill=0.3, seed=42)
    assert A.shape == (64, 64)
    actual = float(A.sum()) / (64 * 64)
    assert abs(actual - 0.3) < 0.05


def test_pinhole_pattern_open_fraction():
    A = build_pinhole_pattern(64, grid_pitch=16, pinhole_radius_px=2)
    # 4 × 4 grid of holes, each ~13 pixels open → ~0.05 fill
    fill = float(A.sum()) / (64 * 64)
    assert 0.01 < fill < 0.10


# ── Mask construction + lookup ─────────────────────────────────────────────


def _build_simple_mask(
    *, pattern=None, au_thickness_um=4.6, sub_thickness_um=0.0,
    pixel_size_um=8.0, learnable=False, rotvec=None,
):
    if pattern is None:
        pattern = build_mura_pattern(13)
    if rotvec is None:
        rotvec = torch.tensor([0.02, -0.01, 0.015], dtype=DTYPE)
    return CodedApertureMask2D(
        pattern=pattern,
        pixel_size_um=pixel_size_um,
        au_thickness_um=au_thickness_um,
        sub_thickness_um=sub_thickness_um,
        position_um=torch.tensor([0.0, 0.0, 500.0], dtype=DTYPE),
        rotvec=rotvec,
        edge_softness_um=0.5,
        make_geometry_learnable=learnable,
        dtype=DTYPE,
    )


def test_construction_validates_shapes():
    with pytest.raises(ValueError):
        CodedApertureMask2D(
            pattern=torch.zeros(5, dtype=torch.int64),   # 1-D — wrong
            pixel_size_um=8.0,
            dtype=DTYPE,
        )


def test_thickness_lookup_at_pixel_centers():
    """Pattern value at integer-pixel centers ≈ the binary pattern."""
    pattern = build_mura_pattern(13)
    # Direct lookup test (not through ray geometry) — no rotvec dependence.
    mask = _build_simple_mask(pattern=pattern, au_thickness_um=6.0)
    px = mask.pixel_size_um
    # Center of pixel (i, j) at intrinsic (u, v) = ((i − (L−1)/2) px, (j − (L−1)/2) px)
    Lx = mask.Lx
    Ly = mask.Ly
    test_centers = [(0, 0), (3, 5), (6, 6), (Lx - 1, Ly - 1)]
    for i, j in test_centers:
        u = (i - (Lx - 1) * 0.5) * px
        v = (j - (Ly - 1) * 0.5) * px
        t_au = mask.au_thickness_at(
            torch.tensor(u, dtype=DTYPE), torch.tensor(v, dtype=DTYPE),
        )
        expected = 6.0 * float(pattern[i, j])
        assert math.isclose(t_au.item(), expected, abs_tol=1e-2)


def test_outside_pattern_returns_no_absorber():
    mask = _build_simple_mask()
    u_far = torch.tensor(1e4, dtype=DTYPE)
    v_far = torch.tensor(1e4, dtype=DTYPE)
    t = mask.au_thickness_at(u_far, v_far)
    assert t.abs().item() < 1e-3


# ── Forward + transmission ────────────────────────────────────────────────


def _make_normal_ray(x_um: float, y_um: float):
    origin = torch.tensor([[x_um, y_um, 0.0]], dtype=DTYPE)
    direction = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    return origin, direction


def test_zero_au_gives_unity_transmission():
    mask = _build_simple_mask(au_thickness_um=0.0, sub_thickness_um=0.0)
    origin, direction = _make_normal_ray(0.0, 0.0)
    lam = torch.tensor([1.0], dtype=DTYPE)
    T = mask(origin, direction, lam)
    assert torch.allclose(T, torch.ones_like(T), atol=1e-10)


def test_normal_incidence_on_open_pixel_passes():
    """Aim through a known-open MURA pixel; transmission ≈ 1."""
    pattern = build_mura_pattern(13)
    # Use a rotvec=0 mask so the intrinsic frame aligns with lab x,y,z
    # and ray-pixel correspondence is exact.
    mask = _build_simple_mask(
        pattern=pattern, au_thickness_um=8.0,
        rotvec=torch.zeros(3, dtype=DTYPE),
    )
    # MURA pattern[1, 0] = 1 (the convention used by the constructor).
    px = mask.pixel_size_um
    i, j = 1, 0
    Lx, Ly = mask.Lx, mask.Ly
    u = (i - (Lx - 1) * 0.5) * px
    v = (j - (Ly - 1) * 0.5) * px
    # Aim from below; mask center at (0, 0, 500), so origin offset by (u, v).
    origin = torch.tensor([[u, v, 0.0]], dtype=DTYPE)
    direction = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    lam = torch.tensor([_wl_A_at_keV(15.0)], dtype=DTYPE)
    T = mask(origin, direction, lam)
    # At a 1-bit (8 µm Au): T = exp(-µ_Au · 8 µm)
    expected = math.exp(-mu_au(lam).item() * 8.0)
    assert math.isclose(T.item(), expected, rel_tol=0.05)


def test_open_pixel_gives_high_transmission():
    """Probe a few definitely-zero pixels and verify near-unity transmission."""
    pattern = build_mura_pattern(13)
    mask = _build_simple_mask(
        pattern=pattern, au_thickness_um=8.0,
        rotvec=torch.zeros(3, dtype=DTYPE),
    )
    Lx, Ly = mask.Lx, mask.Ly
    px = mask.pixel_size_um
    # Find a (i, j) where pattern[i, j] == 0
    zero_idx = None
    for i in range(Lx):
        for j in range(Ly):
            if pattern[i, j].item() == 0 and 2 <= i <= Lx - 3 and 2 <= j <= Ly - 3:
                zero_idx = (i, j)
                break
        if zero_idx is not None:
            break
    assert zero_idx is not None
    i, j = zero_idx
    u = (i - (Lx - 1) * 0.5) * px
    v = (j - (Ly - 1) * 0.5) * px
    origin = torch.tensor([[u, v, 0.0]], dtype=DTYPE)
    direction = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    lam = torch.tensor([_wl_A_at_keV(15.0)], dtype=DTYPE)
    T = mask(origin, direction, lam)
    assert T.item() > 0.95


# ── Gradient flow ──────────────────────────────────────────────────────────


def _seed_rays():
    origin = torch.tensor([
        [-12.0, 5.0, 0.0],
        [3.0, -8.0, 0.0],
        [7.0, 12.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=DTYPE)
    direction = torch.tensor([
        [0.02, 0.0, 1.0],
        [0.0, 0.02, 1.0],
        [-0.01, 0.01, 1.0],
        [0.0, -0.02, 1.0],
    ], dtype=DTYPE)
    direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True)
    lam = torch.full((4,), _wl_A_at_keV(15.0), dtype=DTYPE)
    return origin, direction, lam


def test_gradient_through_mask_position():
    """Position is a refinable parameter — gradient is finite and non-zero."""
    pattern = build_mura_pattern(13)
    mask = _build_simple_mask(
        pattern=pattern, au_thickness_um=6.0, learnable=True,
    )
    origin, direction, lam = _seed_rays()
    T = mask(origin, direction, lam)
    loss = (1.0 - T).pow(2).sum()
    loss.backward()
    assert mask.position_um.grad is not None
    assert torch.isfinite(mask.position_um.grad).all()
    assert mask.position_um.grad.abs().sum() > 0


def test_gradcheck_position_via_closure():
    """gradcheck — strict numerical comparison of analytic vs FD gradient."""
    pattern = build_mura_pattern(13)
    origin, direction, lam = _seed_rays()
    sub_thickness_um = 0.0
    au_thickness_um = 4.6
    pixel_size_um = 8.0
    edge_softness_um = 1.0          # wider edges for FD-friendly landscape

    rotvec = torch.tensor([0.02, -0.01, 0.015], dtype=DTYPE)

    def f(pos):
        # Build a mask anchored at the current pos; the closure's grad
        # is strictly through pos because the mask reads it directly.
        mask = CodedApertureMask2D(
            pattern=pattern,
            pixel_size_um=pixel_size_um,
            au_thickness_um=au_thickness_um,
            sub_thickness_um=sub_thickness_um,
            position_um=pos,
            rotvec=rotvec,
            edge_softness_um=edge_softness_um,
            make_geometry_learnable=False,
            dtype=DTYPE,
        )
        return mask(origin, direction, lam).sum()

    pos = torch.tensor([0.5, -0.3, 500.0], dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(f, (pos,), eps=1e-4, atol=1e-4, rtol=1e-3)
