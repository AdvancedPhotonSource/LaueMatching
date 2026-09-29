"""VoxelODFRefiner: intensity scale / background and harmonic stacking.

* The loss was the raw MSE between a unit-intensity render and the frame, so
  the recovered spread depended on the data's counts and pedestal. The fit
  now solves a per-frame scale and background (a, b) in closed form inside
  the loss and the posterior residual: obs = I and obs = 1000 I + 50 give the
  same sigma_U.
* Harmonics (111), (222), (333)... land on one pixel; at unit intensity each
  they rendered that spot n times brighter than a single reflection. Only
  the lowest-order member of each seed-pixel group is rendered now (same
  grouping as MultiGrainVoxelRefiner), so the (111)-family peak is ~1.
"""
from __future__ import annotations

import math

import pytest
import torch

from midas_stress.orientation import quat_to_orient_mat

from laue_torch.io import LaueParams
from laue_torch.realdata import VoxelMeasurement, VoxelODFRefiner
from laue_torch.distributions import IndependentVoxelDistribution, TangentGaussianSO3, GaussianStrain

DT = torch.float64
U0 = quat_to_orient_mat(torch.tensor(
    [0.56153266089081, -0.1069242896544219, -0.7939419137346801, 0.2071340258144413],
    dtype=DT)).reshape(3, 3)


def _params(E_hi=12.0):
    return LaueParams(sg_num=225, symmetry="F",
                      lattice=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
                      P=(0.028745, 0.002788, 0.513115),
                      R=(-1.20131258, -1.21399082, -1.21881158),
                      px_x=0.006, px_y=0.006, n_pix_x=96, n_pix_y=64,
                      E_lo=5.0, E_hi=E_hi, psf_sigma=1.0)


def _truth_image(ref, sigma_deg):
    """Mosaic truth render (spread sigma_deg), harmonics deduplicated."""
    psi = ref.seed_spot_intensity(U0)
    v = IndependentVoxelDistribution(
        TangentGaussianSO3(U_init=U0, sigma_init=math.radians(sigma_deg)),
        GaussianStrain(sigma_init=1e-9))
    g = torch.Generator().manual_seed(123)
    t = ref.tensors
    with torch.no_grad():
        return v.render(ref.model, t["lattice"], t["P"], t["R"], M=256,
                        generator=g, E_range=ref.E_range, per_spot_intensity=psi)


def test_sigma_U_is_invariant_to_counts_and_pedestal():
    ref = VoxelODFRefiner(_params(), sigma_init_deg=0.6, n_steps=80, M_render=32,
                          compute_posterior=False)
    I = _truth_image(ref, 0.3)
    out = []
    for obs in (I, 1000.0 * I + 50.0):
        r = ref.refine(VoxelMeasurement(0, obs.T.contiguous(), U0.unsqueeze(0), {},
                                        axis_order="YX"))
        out.append(r.sigma_U_deg)
    assert abs(out[1] - out[0]) <= 0.02 * out[0], out
    assert r.metadata["intensity_scale"] == pytest.approx(1000.0, rel=0.2)


def test_harmonic_family_peak_is_one():
    ref = VoxelODFRefiner(_params(E_hi=30.0), n_steps=1, M_render=2,
                          compute_posterior=False)
    t = ref.tensors
    with torch.no_grad():
        img_all, aux = ref.model(U0.unsqueeze(0), t["lattice"], t["P"], t["R"],
                                 strain=torch.zeros(1, 6, dtype=DT),
                                 E_range=ref.E_range, return_aux=True)
    on = aux.mask > 0.5
    key = torch.stack([aux.px.round(), aux.py.round()], -1)
    # A pixel shared by >= 2 in-band reflections (a harmonic family).
    groups = {}
    for h in on.nonzero().reshape(-1).tolist():
        groups.setdefault(tuple(key[h].tolist()), []).append(h)
    fam = [g for g in groups.values() if len(g) >= 2]
    assert fam, "fixture has no harmonic family on the detector"
    x, y = (int(v) for v in key[fam[0][0]].tolist())
    psi = ref.seed_spot_intensity(U0)
    with torch.no_grad():
        img = ref.model(U0.unsqueeze(0), t["lattice"], t["P"], t["R"],
                        strain=torch.zeros(1, 6, dtype=DT), E_range=ref.E_range,
                        per_spot_intensity=psi.unsqueeze(0))
    peak = float(img[x - 1:x + 2, y - 1:y + 2].max())
    peak_all = float(img_all[x - 1:x + 2, y - 1:y + 2].max())
    assert 0.7 < peak <= 1.02, peak
    assert peak_all > 1.5 * peak          # what it used to render
