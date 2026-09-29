"""Every real-data path renders in the EXPERIMENT's band (no silent 5-30 keV).

* ``make_lauematching_params`` without a band records
  ``extras["energy_band_defaulted"]`` (it used to write 5/30 silently), and
  ``write_lauematching_config`` refuses to write such a config.
* The five refiners that used ``E_range or (params.E_lo, params.E_hi)``
  (autofocus_geometry, autofocus_hessian, ReferenceGrainParallaxRefiner,
  DepthResolvedVoxelRefiner, MultiVoxelTVRefiner) now go through
  ``experiment_band`` and refuse a defaulted band.
* ``validate._render_at_sigma`` and the checks built on it take a REQUIRED
  ``E_range`` (they rendered at the forward model's 5-30 keV default).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from midas_stress.orientation import quat_to_orient_mat

from laue_torch import LaueForwardModel
from laue_torch.coded_aperture import (CodedApertureMask, autofocus_geometry,
                                       build_de_bruijn_sequence)
from laue_torch.coded_aperture.landscape import autofocus_hessian
from laue_torch.io import LaueParams, experiment_band, generate_hkls
from laue_torch.realdata import (CodedApertureVoxelMeasurement, Crystal,
                                 DepthResolvedVoxelRefiner, GeoN,
                                 MultiVoxelTVRefiner,
                                 ReferenceGrainParallaxRefiner,
                                 make_lauematching_params,
                                 write_lauematching_config)
from laue_torch.realdata import validate as V

DT = torch.float64
U0 = quat_to_orient_mat(torch.tensor(
    [0.56153266089081, -0.1069242896544219, -0.7939419137346801, 0.2071340258144413],
    dtype=DT)).reshape(3, 3)


def _geon():
    return GeoN(detector_id="d0", P_mm=(28.745, 2.788, 513.115),
                R_rad=(-1.20131258, -1.21399082, -1.21881158),
                npx_x=96, npx_y=64, px_size_mm_x=0.006, px_size_mm_y=0.006,
                sample_R_rad=(0.0, 0.0, 0.0), sample_origin_um=(0.0, 0.0, 0.0))


CRYSTAL = Crystal(sg_num=225, lattice_nm=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0))


def test_make_params_flags_a_defaulted_band_and_writer_refuses(tmp_path):
    p = make_lauematching_params(_geon(), CRYSTAL)
    assert (p.E_lo, p.E_hi) == (5.0, 30.0)
    assert p.extras.get("energy_band_defaulted")
    with pytest.raises(ValueError, match="energy band"):
        experiment_band(p)
    with pytest.raises(ValueError, match="energy band"):
        write_lauematching_config(p, tmp_path / "cfg.txt")
    assert not (tmp_path / "cfg.txt").exists()

    q = make_lauematching_params(_geon(), CRYSTAL, E_lo=7.0, E_hi=28.0)
    assert not q.extras.get("energy_band_defaulted")
    txt = write_lauematching_config(q, tmp_path / "ok.txt").read_text()
    assert "Elo 7.0" in txt and "Ehi 28.0" in txt


def _defaulted_params():
    return dataclasses.replace(
        make_lauematching_params(_geon(), CRYSTAL), psf_sigma=1.0)


def _mask():
    return CodedApertureMask(
        sequence=build_de_bruijn_sequence(order=5, alphabet=2), bar_widths_um=12.0,
        au_thickness_um=6.0, sub_thickness_um=0.0,
        position_um=torch.tensor([0.0, 0.0, 500.0], dtype=DT),
        rotvec=torch.tensor([0.05, -0.03, 0.02], dtype=DT), edge_softness_um=2.0,
        make_geometry_learnable=False, dtype=DT)


def _vox():
    return [CodedApertureVoxelMeasurement(
        voxel_index=0, frame_stack=torch.zeros(2, 96, 64, dtype=DT),
        scan_offsets_um=torch.tensor([-5.0, 5.0], dtype=DT), U_seed=U0, z_seed_um=0.0)]


HKLS = torch.tensor([[1, 1, 1], [2, 0, 0], [2, 2, 0]])


@pytest.mark.parametrize("make", [
    lambda p: ReferenceGrainParallaxRefiner(p, hkls=HKLS),
    lambda p: DepthResolvedVoxelRefiner(p, mask=_mask(), hkls=HKLS),
    lambda p: MultiVoxelTVRefiner(p, mask=_mask(), hkls=HKLS),
    lambda p: autofocus_geometry(_vox(), _mask(), params=p, hkls=HKLS, n_steps=0),
    lambda p: autofocus_hessian(_vox(), _mask(), params=p, hkls=HKLS),
], ids=["reference_grain", "depth_resolved", "multi_voxel_tv", "autofocus", "landscape"])
def test_refiners_refuse_a_defaulted_band(make):
    with pytest.raises(ValueError, match="energy band"):
        make(_defaulted_params())


def test_refiners_take_the_params_band():
    p = dataclasses.replace(_defaulted_params(), E_lo=6.0, E_hi=40.0, extras={})
    assert ReferenceGrainParallaxRefiner(p, hkls=HKLS).E_range == (6.0, 40.0)
    assert DepthResolvedVoxelRefiner(p, mask=_mask(), hkls=HKLS).E_range == (6.0, 40.0)
    assert MultiVoxelTVRefiner(p, mask=_mask(), hkls=HKLS).E_range == (6.0, 40.0)


def test_validate_renders_in_the_given_band():
    """A spot above 30 keV is lit with E_range=(5, 40) and dark with (5, 30);
    omitting E_range is an error, not a silent 5-30 keV render."""
    p = LaueParams(sg_num=225, symmetry="F",
                   lattice=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
                   P=(0.028745, 0.002788, 0.513115),
                   R=(-1.20131258, -1.21399082, -1.21881158),
                   px_x=0.006, px_y=0.006, n_pix_x=96, n_pix_y=64,
                   E_lo=5.0, E_hi=40.0, psf_sigma=1.0)
    hkls = generate_hkls(p.sg_num, p.lattice, p.E_hi)
    t = p.to_tensors()
    model = LaueForwardModel(hkls=hkls, n_pix=t["n_pix"], px_size=t["px_size"],
                             psf_sigma=1.0, strain_mode="voigt", hard=False)
    with torch.no_grad():
        _, aux = model(U0.unsqueeze(0), t["lattice"], t["P"], t["R"],
                       strain=torch.zeros(1, 6, dtype=DT), E_range=(5.0, 40.0),
                       return_aux=True)
    high = ((aux.mask > 0.5) & (aux.energy > 31.0)).nonzero().reshape(-1)
    assert high.numel() > 0, "fixture has no reflection above 30 keV"
    h = int(high[0])
    px, py = int(round(float(aux.px[h]))), int(round(float(aux.py[h])))
    psi = torch.ones(hkls.shape[0], dtype=DT)
    kw = dict(sigma_deg=1e-6, M_render=2, target_psi=psi, seed=0)
    lit = V._render_at_sigma(model, U0, t["lattice"], t["P"], t["R"],
                             E_range=(5.0, 40.0), **kw)
    dark = V._render_at_sigma(model, U0, t["lattice"], t["P"], t["R"],
                              E_range=(5.0, 30.0), **kw)
    assert float(lit[px, py]) > 0.2 and float(dark[px, py]) < 1e-3 * float(lit[px, py])
    with pytest.raises(TypeError):
        V._render_at_sigma(model, U0, t["lattice"], t["P"], t["R"], **kw)
    with pytest.raises(TypeError):
        V.validate_recovery(model=model, U_seed=U0, lat=t["lattice"], P=t["P"],
                            R=t["R"], sigma_U_deg=0.1, M_render=2, sigma_psf_px=1.0,
                            n_visible_HKLs=10, detector_distance_mm=500.0,
                            px_size_mm=0.006, I_obs=lit, target_psi=psi,
                            spot_centers_xy_px=np.zeros((0, 2)), checks=("B",))
