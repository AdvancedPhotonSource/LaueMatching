"""The real-data refiners render in the EXPERIMENT's energy band.

``MultiGrainVoxelRefiner`` and ``VoxelODFRefiner`` used to call the mixture /
voxel renderers without ``E_range``, so every fit used their default
(5, 30) keV whatever the parameter file said: reflections the experiment
recorded above 30 keV were never predicted, and ones it did not record below
its E_lo were. They now use ``params.E_lo`` / ``params.E_hi`` (checked by
``laue_torch.io.experiment_band``) and refuse a band that
``parse_params`` had to default.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from midas_stress.orientation import quat_to_orient_mat

from laue_torch.io import LaueParams, experiment_band, parse_params
from laue_torch.realdata import (MultiGrainVoxelRefiner, VoxelMeasurement,
                                 VoxelODFRefiner)

DT = torch.float64
NX, NY = 96, 64


def _params(E_lo: float, E_hi: float) -> LaueParams:
    return LaueParams(
        sg_num=225, symmetry="F",
        lattice=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
        P=(0.028745, 0.002788, 0.513115),
        R=(-1.20131258, -1.21399082, -1.21881158),
        px_x=0.006, px_y=0.006, n_pix_x=NX, n_pix_y=NY,
        E_lo=E_lo, E_hi=E_hi, psf_sigma=1.0,
    )


U = quat_to_orient_mat(torch.tensor(
    [0.56153266089081, -0.1069242896544219, -0.7939419137346801, 0.2071340258144413],
    dtype=DT)).reshape(3, 3)


def _truth(ref, band):
    t = ref.tensors
    with torch.no_grad():
        img, aux = ref.model(U.unsqueeze(0), t["lattice"], t["P"], t["R"],
                             strain=torch.zeros(1, 6, dtype=DT), E_range=band,
                             return_aux=True)
    return img, aux


def test_multi_grain_target_includes_reflections_above_30_keV():
    ref = MultiGrainVoxelRefiner(_params(5.0, 40.0), sigma_init_deg=1e-4,
                                 n_steps=1, M_render=4)
    assert ref.E_range == (5.0, 40.0)
    img, aux = _truth(ref, (5.0, 40.0))
    target, _ = ref._compute_per_spot_target(None, U.unsqueeze(0), img)
    high = (aux.mask > 0.5) & (aux.energy > 31.0)
    assert int(high.sum()) > 0, "fixture has no reflection above 30 keV"
    assert int((target[0][high] > 0).sum()) > 0, \
        "no reflection above 30 keV got a target: fit is not using the params band"


@pytest.mark.parametrize("band", [(5.0, 40.0), (5.0, 12.0)])
def test_refiners_fit_in_params_band(band):
    """Observation rendered in the params band; at the seed the fit reproduces
    it. With the old fixed (5, 30) band the (5, 40) case misses every
    30-40 keV spot and the (5, 12) case adds every 12-30 keV spot."""
    p = _params(*band)
    mg = MultiGrainVoxelRefiner(p, sigma_init_deg=1e-4, n_steps=1, M_render=4)
    img, _ = _truth(mg, band)
    res = mg.refine(img.T.contiguous(), U.unsqueeze(0), axis_order="YX")
    # Not ~0: the target is the PIXEL max, below a sub-pixel peak by up to
    # ~22%. Measured 1.1e-4 (5-40) and 1.3e-3 (5-12) of peak^2; the band
    # itself is pinned for this refiner by the target test above.
    assert res.final_loss < 5e-3 * float(img.max() ** 2)

    dr = VoxelODFRefiner(p, sigma_init_deg=1e-4, n_steps=1, M_render=4,
                         compute_posterior=False)
    r = dr.refine(VoxelMeasurement(0, img.T.contiguous(), U.unsqueeze(0), {},
                                         axis_order="YX"))
    assert r.final_loss < 1e-3 * float((img ** 2).mean())

    # And the fixed (5, 30) band really would have been wrong here.
    t = dr.tensors
    with torch.no_grad():
        wrong = dr.model(U.unsqueeze(0), t["lattice"], t["P"], t["R"],
                         strain=torch.zeros(1, 6, dtype=DT), E_range=(5.0, 30.0))
    assert float(((wrong - img) ** 2).mean()) > 0.1 * float((img ** 2).mean())


def test_missing_band_raises(tmp_path):
    cfg = tmp_path / "p.txt"
    cfg.write_text(
        "LatticeParameter 0.35238 0.35238 0.35238 90 90 90\nSpaceGroup 225\n"
        "P_Array 0.028745 0.002788 0.513115\n"
        "R_Array -1.20131258 -1.21399082 -1.21881158\n"
        f"PxX 0.006\nPxY 0.006\nNrPxX {NX}\nNrPxY {NY}\n")
    p = parse_params(cfg)
    assert (p.E_lo, p.E_hi) == (5.0, 30.0)            # forward CLI default kept
    with pytest.raises(ValueError, match="energy band"):
        experiment_band(p)
    with pytest.raises(ValueError, match="energy band"):
        MultiGrainVoxelRefiner(p, n_steps=1, M_render=2)
    with pytest.raises(ValueError, match="energy band"):
        VoxelODFRefiner(p, n_steps=1, M_render=2)
    with pytest.raises(ValueError, match="energy band"):
        experiment_band(dataclasses.replace(_params(5.0, 30.0), E_hi=None))
