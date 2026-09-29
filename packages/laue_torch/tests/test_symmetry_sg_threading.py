"""Misorientation and the TV prior honour the space group (no cubic default).

* ``MultiVoxelTVRefiner``: the orientation TV is ||U_{v+1} - U_v||_F. Two
  seeds that are the SAME orientation written as different symmetry variants
  (U and U @ S90z for cubic) used to cost ||S - I||_F = 2; seeds are now
  aligned to the nearest variant of their neighbour first, so the TV is ~0.
* ``VoxelODFRefiner`` / ``MultiGrainVoxelRefiner`` / ``plot_orientation_map``
  report misorientation under ``params.sg_num`` (or ``space_group=``), not
  the hard-wired cubic group.
"""
from __future__ import annotations

import math

import pytest
import torch

from midas_stress.orientation import quat_to_orient_mat

import laue_torch.symmetry as sym
from laue_torch.coded_aperture import CodedApertureMask, build_de_bruijn_sequence
from laue_torch.io import LaueParams, generate_hkls
from laue_torch.realdata import (
    CodedApertureVoxelMeasurement,
    MultiGrainVoxelRefiner,
    MultiVoxelTVRefiner,
    VoxelMeasurement,
    VoxelODFRefiner,
)

DT = torch.float64
U0 = quat_to_orient_mat(torch.tensor(
    [0.56153266089081, -0.1069242896544219, -0.7939419137346801, 0.2071340258144413],
    dtype=DT)).reshape(3, 3)
S90Z = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=DT)


def _params(sg=225, lattice=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0), n=(96, 64)):
    return LaueParams(
        sg_num=sg, symmetry="F" if sg == 225 else "P", lattice=lattice,
        P=(0.028745, 0.002788, 0.513115),
        R=(-1.20131258, -1.21399082, -1.21881158),
        px_x=0.006, px_y=0.006, n_pix_x=n[0], n_pix_y=n[1],
        E_lo=5.0, E_hi=15.0, psf_sigma=1.0)


def test_nearest_variant_undoes_a_symmetry_relabel():
    V = sym.nearest_variant(U0 @ S90Z, U0, 225)
    assert torch.allclose(V, U0, atol=1e-12)
    # Hexagonal group has no 90 deg z rotation: nothing to undo.
    V194 = sym.nearest_variant(U0 @ S90Z, U0, 194)
    assert not torch.allclose(V194, U0, atol=1e-3)


def test_tv_prior_is_symmetry_reduced():
    p = _params(n=(64, 64))
    hkls = generate_hkls(p.sg_num, p.lattice, p.E_hi)
    mask = CodedApertureMask(
        sequence=build_de_bruijn_sequence(order=5, alphabet=2), bar_widths_um=12.0,
        au_thickness_um=6.0, sub_thickness_um=0.0,
        position_um=torch.tensor([0.0, 0.0, 500.0], dtype=DT),
        rotvec=torch.tensor([0.05, -0.03, 0.02], dtype=DT), edge_softness_um=2.0,
        make_geometry_learnable=False, dtype=DT)
    scan = torch.linspace(-10.0, 10.0, 2, dtype=DT)
    frames = torch.zeros(2, 64, 64, dtype=DT)
    vox = [CodedApertureVoxelMeasurement(voxel_index=i, frame_stack=frames,
                                         scan_offsets_um=scan, U_seed=U, z_seed_um=0.0)
           for i, U in enumerate((U0, U0 @ S90Z))]
    res = MultiVoxelTVRefiner(p, mask=mask, hkls=hkls, n_steps=0,
                              lambda_U=1.0).refine(vox)
    assert res.final_loss_tv < 1e-9, res.final_loss_tv


def _record(monkeypatch):
    seen = []
    real = sym.misorientation_deg

    def spy(M1, M2, space_group=sym.CUBIC_SPACE_GROUP, lattice=None):
        seen.append(space_group)
        return real(M1, M2, space_group, lattice=lattice)
    monkeypatch.setattr(sym, "misorientation_deg", spy)
    return seen


HEX = (0.29505, 0.29505, 0.46826, 90.0, 90.0, 120.0)


def test_voxel_refiner_uses_params_space_group(monkeypatch):
    seen = _record(monkeypatch)
    p = _params(194, HEX)
    img = torch.zeros(64, 96, dtype=DT)
    img[30, 40] = 1.0
    VoxelODFRefiner(p, n_steps=1, M_render=2, compute_posterior=False).refine(
        VoxelMeasurement(0, img, U0.unsqueeze(0), {}, axis_order="YX"))
    assert seen and set(seen) == {194}


def test_multi_grain_uses_params_space_group(monkeypatch):
    seen = _record(monkeypatch)
    p = _params(194, HEX)
    img = torch.zeros(64, 96, dtype=DT)
    img[30, 40] = 1.0
    MultiGrainVoxelRefiner(p, n_steps=1, M_render=2).refine(
        img, U0.unsqueeze(0), axis_order="YX")
    assert seen and set(seen) == {194}


def test_plot_orientation_map_takes_space_group(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    from laue_torch.realdata import plot_orientation_map
    seen = _record(monkeypatch)

    class R:
        def __init__(self, U):
            self.U_mean = U
    plot_orientation_map([R(U0), R(U0 @ S90Z)], out_path=tmp_path / "o.png",
                         space_group=194)
    assert seen == [194]
    with pytest.raises(TypeError):
        plot_orientation_map([R(U0)], out_path=tmp_path / "o2.png")
