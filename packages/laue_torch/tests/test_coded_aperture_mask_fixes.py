"""Coded-aperture mask: ray direction gate, smooth rotvec at 0, no mutation.

* A ray that meets the aperture plane BEHIND its source (t <= 0) never passes
  through the mask; it used to be attenuated as if it did.
* ``_rotvec_to_matrix`` normalised the axis with a 1e-30 clamp, so its
  gradient at rotvec = 0 was -1e12 instead of the Rodrigues limit -1.
* ``DepthResolvedVoxelRefiner(mask_edge_softness_um=...)`` changed the
  CALLER's mask in place.
"""
from __future__ import annotations

import pytest
import torch

from laue_torch.coded_aperture import (CodedApertureMask, CodedApertureMask2D,
                                       build_de_bruijn_sequence)
from laue_torch.coded_aperture.mask import _rotvec_to_matrix
from laue_torch.coded_aperture.mask_spectral import CodedApertureMaskSpectral
from laue_torch.io import LaueParams
from laue_torch.realdata import DepthResolvedVoxelRefiner

DT = torch.float64


def _mask():
    return CodedApertureMask(
        sequence=build_de_bruijn_sequence(order=5, alphabet=2), bar_widths_um=12.0,
        au_thickness_um=6.0, sub_thickness_um=0.5,
        position_um=torch.tensor([0.0, 0.0, 500.0], dtype=DT),
        rotvec=torch.tensor([0.05, -0.03, 0.02], dtype=DT), edge_softness_um=2.0,
        make_geometry_learnable=False, dtype=DT)


def _rays():
    origin = torch.zeros(3, dtype=DT)
    d = torch.tensor([[0.01, 0.0, 1.0], [0.01, 0.0, -1.0]], dtype=DT)
    return origin, d / d.norm(dim=-1, keepdim=True), torch.full((2,), 0.5, dtype=DT)


def test_backward_ray_is_not_attenuated():
    o, d, wl = _rays()
    T = _mask()(o, d, wl)
    assert float(T[0]) < 1.0               # forward ray crosses the substrate at least
    assert float(T[1]) == 1.0              # backward ray never meets the mask


def test_backward_ray_not_attenuated_2d_and_spectral():
    o, d, wl = _rays()
    m2 = CodedApertureMask2D(pattern=torch.ones(4, 4, dtype=torch.long),
                             pixel_size_um=10.0, au_thickness_um=5.0,
                             sub_thickness_um=0.5,
                             position_um=torch.tensor([0.0, 0.0, 500.0], dtype=DT),
                             dtype=DT)
    assert float(m2(o, d, wl)[1]) == 1.0
    ms = CodedApertureMaskSpectral(
        ["Au", "Au"], 5.0, bar_widths_um=10.0, sub_thickness_um=0.5,
        position_um=torch.tensor([0.0, 0.0, 500.0], dtype=DT), dtype=DT)
    assert float(ms(o, d, wl)[1]) == 1.0


def test_rotvec_gradient_at_zero_is_the_rodrigues_limit():
    r = torch.zeros(3, dtype=DT, requires_grad=True)
    _rotvec_to_matrix(r)[0, 1].backward()
    assert torch.allclose(r.grad, torch.tensor([0.0, 0.0, -1.0], dtype=DT))


def test_refiner_does_not_mutate_the_callers_mask():
    p = LaueParams(sg_num=225, symmetry="F",
                   lattice=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
                   P=(0.028745, 0.002788, 0.513115),
                   R=(-1.20131258, -1.21399082, -1.21881158),
                   px_x=0.0064, px_y=0.0064, n_pix_x=64, n_pix_y=64,
                   E_lo=5.0, E_hi=12.0, psf_sigma=1.0)
    m = _mask()
    ref = DepthResolvedVoxelRefiner(p, mask=m, hkls=torch.tensor([[1, 1, 1]]),
                                    mask_edge_softness_um=7.0)
    assert float(m.edge_softness_um) == 2.0
    assert float(ref.mask.edge_softness_um) == 7.0
