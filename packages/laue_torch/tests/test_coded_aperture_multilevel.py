"""Phase 6: multi-level Au-thickness coded apertures.

Validates the energy-encoding generalisation of the coded aperture:
per-bar Au thickness in place of the scalar Phase 0 thickness.  This
unlocks the open future-work bullet from Gürsoy *et al.* RSI 2023 §IV
— "non-binary coded aperture with varying thicknesses can be used to
encode simultaneously the energy information of the diffracted beam".

The differentiable forward needs *zero* algorithmic changes for this:
``mu_au(λ)`` is already per-spot, the mask sums Beer–Lambert
contributions over bars in :meth:`au_thickness_at`, and the
gradient flows through the new per-bar Au-thickness tensor exactly
the same way as through any other parameter.
"""
from __future__ import annotations

import math

import pytest
import torch

from laue_torch.coded_aperture import CodedApertureMask, mu_au
from laue_torch.coded_aperture.mask import _RAD2DEG  # noqa: F401 — sanity


DTYPE = torch.float64


def _energy_keV_to_lambda_A(E_keV: float) -> float:
    return 12.398419843320026 / E_keV


def _normal_ray_at_x(x_um: float) -> tuple[torch.Tensor, torch.Tensor]:
    origin = torch.tensor([[x_um, 0.0, -100.0]], dtype=DTYPE)
    direction = torch.tensor([[0.0, 0.0, 1.0]], dtype=DTYPE)
    return origin, direction


# ── construction-time validation ───────────────────────────────────────────


def test_multilevel_requires_matching_length():
    seq = torch.tensor([1, 1, 1, 1], dtype=torch.int64)
    with pytest.raises(ValueError):
        CodedApertureMask(
            sequence=seq, bar_widths_um=10.0,
            au_thickness_um=torch.tensor([5.0, 10.0], dtype=DTYPE),  # wrong length
            sub_thickness_um=0.0, dtype=DTYPE,
        )


def test_scalar_au_thickness_unchanged():
    """Phase 0 (scalar) path must give the same transmission as before."""
    seq = torch.tensor([1, 1, 1, 1], dtype=torch.int64)
    mask_scalar = CodedApertureMask(
        sequence=seq, bar_widths_um=10.0,
        au_thickness_um=6.0, sub_thickness_um=0.0,
        edge_softness_um=0.02, dtype=DTYPE,
    )
    # Equivalent per-bar mask: same scalar broadcast 4 times.
    mask_vector = CodedApertureMask(
        sequence=seq, bar_widths_um=10.0,
        au_thickness_um=torch.full((4,), 6.0, dtype=DTYPE),
        sub_thickness_um=0.0,
        edge_softness_um=0.02, dtype=DTYPE,
    )
    origin, direction = _normal_ray_at_x(5.0)
    lam = torch.tensor([_energy_keV_to_lambda_A(15.0)], dtype=DTYPE)
    T_s = mask_scalar(origin, direction, lam, scan_offset_um=0.0)
    T_v = mask_vector(origin, direction, lam, scan_offset_um=0.0)
    assert torch.allclose(T_s, T_v, atol=1e-10)


# ── multi-level physics ────────────────────────────────────────────────────


def test_multilevel_per_bar_transmission_matches_beer_lambert():
    """A 4-level mask at normal incidence transmits exp(-µ_Au · t_k) per bar."""
    seq = torch.tensor([1, 1, 1, 1], dtype=torch.int64)
    thicknesses_um = torch.tensor([0.0, 4.0, 8.0, 16.0], dtype=DTYPE)
    mask = CodedApertureMask(
        sequence=seq, bar_widths_um=20.0,
        au_thickness_um=thicknesses_um,
        sub_thickness_um=0.0,
        edge_softness_um=0.02, dtype=DTYPE,
    )
    # Aim a normal-incidence ray at the center of each bar
    # (centers: -30, -10, 10, 30 for 20-µm bars centered on 0).
    centers = (mask.bar_edges_um()[:-1] + mask.bar_edges_um()[1:]) * 0.5
    lam = torch.full((4,), _energy_keV_to_lambda_A(15.0), dtype=DTYPE)
    origin = torch.zeros(4, 3, dtype=DTYPE)
    origin[:, 0] = centers
    origin[:, 2] = -100.0
    direction = torch.tensor([[0.0, 0.0, 1.0]] * 4, dtype=DTYPE)

    T = mask(origin, direction, lam, scan_offset_um=0.0)
    mu = mu_au(lam)                                # 1/µm
    expected = torch.exp(-mu * thicknesses_um)
    assert torch.allclose(T, expected, rtol=1e-3, atol=1e-6)


def test_multilevel_resolves_energy_via_thickness():
    """Variable-thickness mask: higher energy ⇒ less absorption.

    For a bar with Au thickness ``t``:
        T(E) = exp(-µ_Au(E) · t)
    Since µ_Au decreases with energy (above the L edges), T(E)
    *increases*.  The relative energy sensitivity is amplified by
    thickness — that's why thicker bars encode energy *better*.
    """
    seq = torch.tensor([1], dtype=torch.int64)
    mask_thin = CodedApertureMask(
        sequence=seq, bar_widths_um=100.0,
        au_thickness_um=2.0, sub_thickness_um=0.0,
        edge_softness_um=0.02, dtype=DTYPE,
    )
    mask_thick = CodedApertureMask(
        sequence=seq, bar_widths_um=100.0,
        au_thickness_um=12.0, sub_thickness_um=0.0,
        edge_softness_um=0.02, dtype=DTYPE,
    )
    origin, direction = _normal_ray_at_x(0.0)
    # Sample two energies on either side of the published 7–30 keV band.
    lam_lo = torch.tensor([_energy_keV_to_lambda_A(10.0)], dtype=DTYPE)
    lam_hi = torch.tensor([_energy_keV_to_lambda_A(25.0)], dtype=DTYPE)

    # Sensitivity = (T_hi - T_lo) / T_lo across this 15-keV span.
    T_lo_thin = mask_thin(origin, direction, lam_lo, 0.0).item()
    T_hi_thin = mask_thin(origin, direction, lam_hi, 0.0).item()
    T_lo_thick = mask_thick(origin, direction, lam_lo, 0.0).item()
    T_hi_thick = mask_thick(origin, direction, lam_hi, 0.0).item()

    s_thin = (T_hi_thin - T_lo_thin) / T_lo_thin
    s_thick = (T_hi_thick - T_lo_thick) / T_lo_thick

    # Higher energy must have higher transmission (both masks).
    assert T_hi_thin > T_lo_thin
    assert T_hi_thick > T_lo_thick
    # Thicker bar must amplify the energy-dependent contrast.
    assert s_thick > 2.0 * s_thin


# ── gradient flow: per-bar thickness is a refinable parameter ─────────────


def test_gradcheck_per_bar_au_thickness():
    """gradcheck against the per-bar thickness tensor (Phase 6 refinement)."""
    seq = torch.tensor([1, 1, 1, 0, 1, 1], dtype=torch.int64)
    bar_widths_um = 8.0
    # We don't bake the thicknesses into the mask — we treat them as a
    # free input to the closure so the gradcheck closure is clean and
    # we don't have to fiddle with nn.Parameter assignment under autograd.
    origin = torch.tensor([
        [-15.0, 0.0, -100.0],
        [ -5.0, 0.0, -100.0],
        [  3.0, 0.0, -100.0],
        [ 11.0, 0.0, -100.0],
    ], dtype=DTYPE)
    direction = torch.tensor([
        [0.01, 0.0, 1.0],
        [0.0, 0.01, 1.0],
        [-0.005, 0.0, 1.0],
        [0.005, -0.005, 1.0],
    ], dtype=DTYPE)
    direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True)
    lam = torch.full((4,), _energy_keV_to_lambda_A(15.0), dtype=DTYPE)

    def f(t_au_per_bar):
        mask = CodedApertureMask(
            sequence=seq, bar_widths_um=bar_widths_um,
            au_thickness_um=t_au_per_bar,
            sub_thickness_um=0.0,
            edge_softness_um=0.5, dtype=DTYPE,
        )
        return mask(origin, direction, lam, scan_offset_um=0.0).sum()

    t_init = torch.tensor([0.0, 4.0, 8.0, 12.0, 6.0, 3.0], dtype=DTYPE,
                          requires_grad=True)
    assert torch.autograd.gradcheck(f, (t_init,), eps=1e-5, atol=1e-4, rtol=1e-3)
