"""Unit + gradient tests for the differentiable coded-aperture mask.

These pin Phase 0 of the coded-aperture extension
(``laue_torch/implementation_plan_coded_aperture.md``).  The mask is
exercised in isolation, decoupled from ``LaueForwardModel`` — the
integration into the forward model is Phase 1.
"""
from __future__ import annotations

import math

import pytest
import torch

from laue_torch.coded_aperture import (
    CodedApertureMask,
    build_de_bruijn_sequence,
    mu_au,
    mu_si3n4,
)


DTYPE = torch.float64
torch.manual_seed(0)


def _energy_keV_to_lambda_A(E_keV: float) -> float:
    return 12.398419843320026 / E_keV


# ── absorption sanity ──────────────────────────────────────────────────────

def test_mu_au_monotonic_above_L_edges():
    """Above the L-edges μ_Au(λ) decreases with energy (i.e. increases with λ)."""
    energies_keV = torch.tensor([15.0, 20.0, 25.0, 30.0], dtype=DTYPE)
    lambdas = 12.398419843320026 / energies_keV
    mus = mu_au(lambdas)
    # higher energy → smaller μ (wavelengths here are monotone decreasing)
    assert torch.all(mus.diff() < 0)


def test_mu_si3n4_small_at_keV():
    """μ_{Si3N4} should be small (< 0.01 µm⁻¹) at 15 keV — substrate barely absorbs."""
    lam = torch.tensor(_energy_keV_to_lambda_A(15.0), dtype=DTYPE)
    val = mu_si3n4(lam)
    assert 0.0 < val.item() < 0.01


# ── de Bruijn sequence ─────────────────────────────────────────────────────

def test_de_bruijn_order8_length_256():
    seq = build_de_bruijn_sequence(order=8, alphabet=2)
    assert seq.numel() == 256
    assert torch.all((seq == 0) | (seq == 1))


def test_de_bruijn_substring_uniqueness():
    """Every 8-bit substring (taken cyclically) appears exactly once."""
    seq = build_de_bruijn_sequence(order=8, alphabet=2).tolist()
    L = len(seq)
    seen = set()
    for i in range(L):
        sub = tuple(seq[(i + j) % L] for j in range(8))
        assert sub not in seen, f"duplicate 8-bit substring at i={i}"
        seen.add(sub)
    assert len(seen) == 256


# ── mask construction ──────────────────────────────────────────────────────

def _make_simple_mask(
    seq: torch.Tensor,
    bar_width: float = 10.0,
    au_th: float = 4.6,
    sub_th: float = 0.0,           # disable substrate by default for clean tests
    edge_softness_um: float = 0.05,
    learnable: bool = False,
) -> CodedApertureMask:
    return CodedApertureMask(
        sequence=seq,
        bar_widths_um=bar_width,
        au_thickness_um=au_th,
        sub_thickness_um=sub_th,
        edge_softness_um=edge_softness_um,
        make_geometry_learnable=learnable,
        dtype=DTYPE,
    )


def test_au_coverage_at_bar_centers():
    seq = torch.tensor([1, 0, 1, 1, 0], dtype=torch.int64)
    mask = _make_simple_mask(seq, bar_width=10.0, edge_softness_um=0.02)
    edges = mask.bar_edges_um()
    centers = 0.5 * (edges[:-1] + edges[1:])
    cov = mask.au_coverage(centers)
    expected = seq.to(DTYPE)
    assert torch.allclose(cov, expected, atol=1e-3)


def test_au_coverage_outside_pattern_is_zero():
    seq = torch.tensor([1, 1, 1], dtype=torch.int64)
    mask = _make_simple_mask(seq, bar_width=10.0, edge_softness_um=0.02)
    u = torch.tensor([-1e3, 1e3], dtype=DTYPE)
    cov = mask.au_coverage(u)
    assert torch.all(cov < 1e-3)


def test_au_coverage_boundary_smooth():
    """At a 1↔0 boundary the coverage transitions monotonically through ~0.5."""
    seq = torch.tensor([1, 0], dtype=torch.int64)
    mask = _make_simple_mask(seq, bar_width=10.0, edge_softness_um=0.1)
    # Bar 0 is [-10, 0], bar 1 is [0, 10]. Sweep across the boundary.
    u = torch.linspace(-1.0, 1.0, 21, dtype=DTYPE)
    cov = mask.au_coverage(u)
    # endpoints saturate
    assert cov[0].item() > 0.9
    assert cov[-1].item() < 0.1
    # transition is monotonic in this regime
    assert torch.all(cov.diff() <= 1e-6)
    # midpoint is roughly 1/2
    assert 0.3 < cov[10].item() < 0.7


# ── transmission geometry ──────────────────────────────────────────────────


def _normal_incidence_ray(N: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Ray going straight along +n (normal incidence). Origin at -100 along n."""
    origin = torch.tensor([[0.0, 0.0, -100.0]] * N, dtype=DTYPE)
    direction = torch.tensor([[0.0, 0.0, 1.0]] * N, dtype=DTYPE)
    return origin, direction


def test_transmission_zero_au_thickness_gives_unity():
    seq = torch.tensor([1, 1, 1], dtype=torch.int64)
    mask = _make_simple_mask(seq, bar_width=10.0, au_th=0.0, sub_th=0.0)
    origin, direction = _normal_incidence_ray(N=1)
    lam = torch.tensor([_energy_keV_to_lambda_A(15.0)], dtype=DTYPE)
    T = mask(origin, direction, lam, scan_offset_um=0.0)
    assert torch.allclose(T, torch.ones_like(T), atol=1e-12)


def test_transmission_zero_bit_full_pass():
    """A ray through a 0-bit (no Au) should be ≈1 with zero substrate."""
    seq = torch.tensor([0, 1, 0], dtype=torch.int64)
    mask = _make_simple_mask(seq, bar_width=10.0, edge_softness_um=0.02)
    origin, direction = _normal_incidence_ray(N=1)
    # Center of bar 0 is at u = -10 (edges -15, -5). Default rotation, so
    # u_intrinsic = intersection.x.
    origin[0, 0] = -10.0
    direction = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    lam = torch.tensor([_energy_keV_to_lambda_A(15.0)], dtype=DTYPE)
    T = mask(origin, direction, lam, scan_offset_um=0.0)
    assert T.item() > 0.99


def test_transmission_one_bit_normal_incidence_matches_beer_lambert():
    """Through a 1-bit at normal incidence: T = exp(-μ·t_Au)."""
    seq = torch.tensor([1, 1, 1], dtype=torch.int64)
    au_th = 4.6
    mask = _make_simple_mask(seq, bar_width=20.0, au_th=au_th, sub_th=0.0,
                              edge_softness_um=0.02)
    origin, direction = _normal_incidence_ray(N=1)
    # Aim at the center of bar 1, at u = 0.
    origin[0, 0] = 0.0
    lam = torch.tensor([_energy_keV_to_lambda_A(15.0)], dtype=DTYPE)
    T = mask(origin, direction, lam, scan_offset_um=0.0)
    mu = mu_au(lam)
    expected = torch.exp(-mu * au_th)
    assert torch.allclose(T, expected, rtol=1e-3, atol=1e-6)


def test_transmission_oblique_incidence_path_length():
    """At 45° incidence the Au path length is √2 longer → T = exp(-μ·t·√2)."""
    seq = torch.tensor([1, 1, 1], dtype=torch.int64)
    au_th = 4.6
    mask = _make_simple_mask(seq, bar_width=20.0, au_th=au_th, sub_th=0.0,
                              edge_softness_um=0.02)
    # Aim at center of bar 1 (u=0) from below, but tilt the ray at 45° in x-z.
    origin = torch.tensor([[-50.0, 0.0, -100.0]], dtype=DTYPE)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    direction = torch.tensor([[inv_sqrt2, 0.0, inv_sqrt2]], dtype=DTYPE)
    # The ray intersects the z=0 plane at x = -50 + 100 = 50 — outside the
    # bar pattern. Pick a starting x so it lands on u=0.
    origin[0, 0] = -100.0  # ray ends at x = -100 + 100 = 0
    lam = torch.tensor([_energy_keV_to_lambda_A(15.0)], dtype=DTYPE)
    T = mask(origin, direction, lam, scan_offset_um=0.0)
    mu = mu_au(lam)
    expected = torch.exp(-mu * au_th * math.sqrt(2.0))
    assert torch.allclose(T, expected, rtol=1e-2, atol=1e-6)


def test_scan_offset_shifts_pattern_by_one_bar():
    """Translating the aperture by one bar-width shifts the sequence by 1."""
    seq = torch.tensor([1, 0, 1, 0, 1], dtype=torch.int64)
    bar_w = 10.0
    mask = _make_simple_mask(seq, bar_width=bar_w, sub_th=0.0,
                              edge_softness_um=0.02)
    # Probe at bar centers in the intrinsic frame: u = -20, -10, 0, 10, 20.
    centers = (mask.bar_edges_um()[:-1] + mask.bar_edges_um()[1:]) * 0.5
    origin = torch.zeros(5, 3, dtype=DTYPE)
    origin[:, 0] = centers
    origin[:, 2] = -100.0
    direction = torch.tensor([[0.0, 0.0, 1.0]] * 5, dtype=DTYPE)
    lam = torch.full((5,), _energy_keV_to_lambda_A(15.0), dtype=DTYPE)

    T0 = mask(origin, direction, lam, scan_offset_um=0.0)
    # With offset = +bar_w, the sequence shifts by one to the right (when
    # querying at the same intrinsic u, we sample the bit one to the left).
    T_shifted = mask(origin, direction, lam, scan_offset_um=bar_w)

    # T0 directly reads off seq: positions [1,0,1,0,1] → 1=opaque, 0=transparent.
    # T_shifted reads at u_query = u - bar_w → effectively shifted sequence
    # [_, 1, 0, 1, 0] (first element falls off the pattern → 0-bit / pass).
    # Verify: positions where (original) seq differs from (shifted) seq give
    # different transmission.
    mu = mu_au(lam)
    au_pass = torch.exp(-mu * 4.6)             # 1-bit transmission
    full_pass = torch.ones_like(au_pass)        # 0-bit transmission
    expect_0 = torch.where(seq.to(DTYPE) > 0.5, au_pass, full_pass)
    # shifted: query at center_k shifted by -bar_w → reads seq[k-1] for k>=1,
    # and outside-pattern for k=0.
    shifted_bits = torch.tensor([0, 1, 0, 1, 0], dtype=DTYPE)
    expect_shifted = torch.where(shifted_bits > 0.5, au_pass, full_pass)

    assert torch.allclose(T0, expect_0, atol=1e-2)
    assert torch.allclose(T_shifted, expect_shifted, atol=1e-2)


# ── gradient flow ──────────────────────────────────────────────────────────


def _seed_grad_problem():
    """A small ray batch usable as a gradcheck reference."""
    seq = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0], dtype=torch.int64)
    origin = torch.tensor([
        [-5.0, 0.0, -100.0],
        [ 0.0, 1.0, -100.0],
        [ 7.0, -2.0, -100.0],
        [12.0, 0.5, -100.0],
    ], dtype=DTYPE)
    # All near-normal but slightly off-axis to break symmetry.
    direction = torch.tensor([
        [0.05, 0.0, 1.0],
        [0.0, 0.05, 1.0],
        [-0.03, 0.02, 1.0],
        [0.02, -0.04, 1.0],
    ], dtype=DTYPE)
    direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True)
    lam = torch.full((4,), _energy_keV_to_lambda_A(15.0), dtype=DTYPE)
    return seq, origin, direction, lam


def test_gradcheck_position():
    """gradcheck on the mask position.

    Builds the transmission integrand by hand using the same canonical
    primitives the mask uses internally (midas_stress for the rotation,
    midas_hkls for μ_Au) so the gradient strictly traces through
    ``pos``.  The rotvec is non-zero so the axis-angle path is
    differentiable (the structural zero at ``rotvec = 0`` is documented
    in :func:`laue_torch.coded_aperture.mask._rotvec_to_matrix`).
    """
    seq, origin, direction, lam = _seed_grad_problem()
    mask = _make_simple_mask(seq, bar_width=10.0, sub_th=2.0,
                              edge_softness_um=0.5, learnable=False)
    mask.rotvec.copy_(torch.tensor([0.05, -0.03, 0.02], dtype=DTYPE))

    from laue_torch.coded_aperture.mask import _rotvec_to_matrix

    def f(pos):
        R = _rotvec_to_matrix(mask.rotvec)
        u_hat, _v_hat, n_hat = R[:, 0], R[:, 1], R[:, 2]
        denom = (direction * n_hat).sum(dim=-1)
        denom_safe = torch.where(denom.abs() > 1e-9, denom,
                                  torch.full_like(denom, 1e-9))
        t = ((pos - origin) * n_hat).sum(dim=-1) / denom_safe
        intersection = origin + t.unsqueeze(-1) * direction
        rel = intersection - pos
        u_intrinsic = (rel * u_hat).sum(dim=-1)
        cov = mask.au_coverage(u_intrinsic)
        norm_d = torch.linalg.norm(direction, dim=-1).clamp_min(1e-30)
        cos_inc = denom.abs() / norm_d
        cos_inc_safe = cos_inc.clamp_min(1e-6)
        path_au = mask.au_thickness_um / cos_inc_safe
        absorb = mu_au(lam) * path_au * cov
        return torch.exp(-absorb).sum()

    pos = mask.position_um.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(f, (pos,), eps=1e-4, atol=1e-4, rtol=1e-3)


def test_gradcheck_scan_offset():
    seq, origin, direction, lam = _seed_grad_problem()
    mask = _make_simple_mask(seq, bar_width=10.0, sub_th=0.0,
                              edge_softness_um=0.5)

    def f(p):
        return mask(origin, direction, lam, scan_offset_um=p).sum()

    p = torch.tensor(0.7, dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(f, (p,), eps=1e-4, atol=1e-4, rtol=1e-3)


def test_gradcheck_au_thickness():
    seq, origin, direction, lam = _seed_grad_problem()
    mask = _make_simple_mask(seq, bar_width=10.0, sub_th=0.0,
                              edge_softness_um=0.5)

    def f(t_au):
        mask.au_thickness_um = t_au  # not registered learnable for this test
        return mask(origin, direction, lam, scan_offset_um=0.0).sum()

    t = torch.tensor(4.6, dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(f, (t,), eps=1e-5, atol=1e-4, rtol=1e-3)


def test_gradcheck_rotvec():
    seq, origin, direction, lam = _seed_grad_problem()
    mask = _make_simple_mask(seq, bar_width=10.0, sub_th=0.0,
                              edge_softness_um=0.5)

    def f(rv):
        mask.rotvec = rv
        return mask(origin, direction, lam, scan_offset_um=0.0).sum()

    rv = torch.tensor([0.01, -0.02, 0.005], dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(f, (rv,), eps=1e-5, atol=1e-4, rtol=1e-3)


def test_gradcheck_wavelength():
    """μ(λ) lookup is differentiable through wavelength (via midas_hkls)."""
    seq, origin, direction, _ = _seed_grad_problem()
    mask = _make_simple_mask(seq, bar_width=10.0, sub_th=0.0,
                              edge_softness_um=0.5)

    def f(lam):
        return mask(origin, direction, lam, scan_offset_um=0.0).sum()

    lam = torch.full((4,), _energy_keV_to_lambda_A(15.0), dtype=DTYPE,
                     requires_grad=True)
    assert torch.autograd.gradcheck(f, (lam,), eps=1e-5, atol=1e-4, rtol=1e-3)
