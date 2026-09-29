"""VoxelODFRefiner: which part of the orientation spread the image determines.

Reported: truth sigma_U 0.3 deg, init 0.6 -> 1.07 deg; init 0.15 -> ~0.3.
Not a loss or parametrisation bug: along an isotropic spread the loss is
minimised at the truth. It is identifiability. A Laue spot moves only with
the in-plane part of a rotation (per reflection the image sees J Sigma J^T),
so:

* one reflection: the spread about its normal is unobservable; Adam drifts
  along that flat direction (1.8 deg from init 0.6) while the two in-plane
  stds come back at the truth;
* 5 reflections on a small detector: one weak direction (sub-PSF
  displacement) keeps roughly its init while the two well-seen ones agree.

These tests pin that behaviour so a change that claims to "fix" it has to
show the unseen direction is actually constrained. See the Warnings section
of VoxelODFRefiner. Investigation scripts:
~/Desktop/analysis/lauematching_fixes_2026-09/voxelodf/.
"""
from __future__ import annotations

import math

import torch

from midas_stress.orientation import quat_to_orient_mat

from laue_torch.distributions import (GaussianStrain, IndependentVoxelDistribution,
                                      TangentGaussianSO3)
from laue_torch.geometry import rodrigues_to_matrix
from laue_torch.io import LaueParams
from laue_torch.realdata import VoxelMeasurement, VoxelODFRefiner
from laue_torch.realdata.driver import FIXED_PRED_SEED, affine_fit_residual

DT = torch.float64
U0 = quat_to_orient_mat(torch.tensor(
    [0.56153266089081, -0.1069242896544219, -0.7939419137346801, 0.2071340258144413],
    dtype=DT)).reshape(3, 3)
TRUTH = 0.3


def _params(nx, ny):
    return LaueParams(sg_num=225, symmetry="F",
                      lattice=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
                      P=(0.028745, 0.002788, 0.513115),
                      R=(-1.20131258, -1.21399082, -1.21881158),
                      px_x=0.006, px_y=0.006, n_pix_x=nx, n_pix_y=ny,
                      E_lo=5.0, E_hi=12.0, psf_sigma=1.0)


def _render(ref, sigma_deg, M, seed, strain_sigma):
    psi = ref.seed_spot_intensity(U0)
    v = IndependentVoxelDistribution(
        TangentGaussianSO3(U_init=U0, sigma_init=math.radians(sigma_deg)),
        GaussianStrain(sigma_init=strain_sigma))
    t = ref.tensors
    with torch.no_grad():
        return v.render(ref.model, t["lattice"], t["P"], t["R"], M=M,
                        generator=torch.Generator().manual_seed(seed),
                        E_range=ref.E_range, per_spot_intensity=psi)


def _spot_jacobian(ref):
    """(2K, 3) d(px, py)/d(body tangent rotation) for the rendered reflections."""
    psi = ref.seed_spot_intensity(U0)
    t = ref.tensors
    d = torch.zeros(3, dtype=DT, requires_grad=True)
    U = (U0 @ rodrigues_to_matrix(d.unsqueeze(0))[0]).unsqueeze(0)
    _, aux = ref.model(U, t["lattice"], t["P"], t["R"], strain=torch.zeros(1, 6, dtype=DT),
                       E_range=ref.E_range, return_aux=True)
    px, py = aux.px.reshape(-1), aux.py.reshape(-1)
    rows = []
    for h in (psi > 0.5).nonzero().reshape(-1).tolist():
        rows.append(torch.autograd.grad(px[h], d, retain_graph=True)[0])
        rows.append(torch.autograd.grad(py[h], d, retain_graph=True)[0])
    return torch.stack(rows)


def _std_along(cov, V):
    return [math.degrees(math.sqrt(float(v @ cov @ v))) for v in V]


def _fit(ref, I, init):
    ref.sigma_init_deg = init
    return ref.refine(VoxelMeasurement(0, I.T.contiguous(), U0.unsqueeze(0), {},
                                       axis_order="YX"))


def test_single_reflection_spread_about_its_normal_is_unrecoverable():
    ref = VoxelODFRefiner(_params(96, 64), n_steps=300, M_render=32,
                          compute_posterior=False)
    I = _render(ref, TRUTH, 256, 123, 1e-9)
    J = _spot_jacobian(ref)
    assert J.shape[0] == 2, "fixture must have exactly one reflection"
    _, _, Vh = torch.linalg.svd(J)          # Vh[2]: rotation about the normal

    # The loss is right: along an isotropic spread its minimum is the truth.
    psi = ref.seed_spot_intensity(U0)
    grid = [0.15, 0.2, 0.3, 0.4, 0.6]
    losses = []
    for s in grid:
        Ip = _render(ref, s, ref.M_render, FIXED_PRED_SEED, 1e-6)
        losses.append(float((affine_fit_residual(Ip, I)[0] ** 2).mean()))
    assert grid[losses.index(min(losses))] == TRUTH, losses

    # From above: in-plane recovered, the unseen direction drifts far up.
    # Measured: in-plane 0.318 / 0.281, about the normal 1.81, sigma_U 1.07.
    r = _fit(ref, I, 0.6)
    seen = _std_along(r.orient_cov, Vh[:2])
    unseen = _std_along(r.orient_cov, Vh[2:])[0]
    assert all(abs(s - TRUTH) < 0.06 for s in seen), seen
    assert unseen > 1.2, unseen
    assert r.sigma_U_deg > 0.9, r.sigma_U_deg


def test_weak_direction_tracks_init_with_few_reflections():
    ref = VoxelODFRefiner(_params(128, 128), n_steps=300, M_render=32,
                          compute_posterior=False)
    I = _render(ref, TRUTH, 256, 123, 1e-9)
    J = _spot_jacobian(ref)
    assert J.shape[0] == 10, "fixture must have 5 reflections"
    _, S, Vh = torch.linalg.svd(J)
    # Weak direction: sub-PSF displacement at the truth spread (measured
    # 1.95 px/deg vs 8.35 and 5.73).
    assert float(S[2]) * math.radians(TRUTH) < 0.7 < float(S[1]) * math.radians(TRUTH)

    lo, hi = _fit(ref, I, 0.15), _fit(ref, I, 0.6)
    # Measured (deg): well-seen 0.308 / 0.328 (init 0.15), 0.332 / 0.296
    # (init 0.6); weak 0.046 vs 0.402.
    for r in (lo, hi):
        seen = _std_along(r.orient_cov, Vh[:2])
        assert all(abs(s - TRUTH) < 0.07 for s in seen), seen
    w_lo = _std_along(lo.orient_cov, Vh[2:])[0]
    w_hi = _std_along(hi.orient_cov, Vh[2:])[0]
    assert w_lo < 0.15 and w_hi > 0.3, (w_lo, w_hi)
    assert hi.sigma_U_deg - lo.sigma_U_deg > 0.05, (lo.sigma_U_deg, hi.sigma_U_deg)
