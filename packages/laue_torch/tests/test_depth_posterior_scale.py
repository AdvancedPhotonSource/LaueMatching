"""DepthResolvedVoxelRefiner.posterior has the Gauss-Newton scale.

With residuals r(theta) over N pixels and plug-in noise variance nv, the
Laplace covariance at a zero-residual optimum is nv * inv(J^T J). The old
code took the Hessian of the MEAN squared residual (2 J^T J / N) and divided
by nv, so every sigma was sqrt(N / 2) too large. It now uses 0.5 * SSR.

The ridge that keeps the inverse finite is now relative to each parameter's
own curvature (Jacobi scaling). It used to be 1e-8 x the LARGEST diagonal,
and the depth curvature (per um) is ~1e12 below the rotation curvature (per
quaternion unit), so the ridge, not the data, set sigma_z (12x too small here).
"""
from __future__ import annotations

import math

import torch

from midas_stress.orientation import quat_to_orient_mat

from laue_torch import LaueForwardModel
from laue_torch.coded_aperture import CodedApertureMask, build_de_bruijn_sequence
from laue_torch.io import LaueParams, generate_hkls
from laue_torch.realdata import (CodedApertureVoxelMeasurement,
                                 DepthResolvedVoxelRefiner, DepthResolvedVoxelResult)

DT = torch.float64


def test_posterior_sigma_is_sqrt_nv_inv_JtJ():
    p = LaueParams(sg_num=225, symmetry="F",
                   lattice=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
                   P=(0.028745, 0.002788, 0.513115),
                   R=(-1.20131258, -1.21399082, -1.21881158),
                   px_x=0.0064, px_y=0.0064, n_pix_x=64, n_pix_y=64,
                   E_lo=5.0, E_hi=12.0, psf_sigma=1.0)
    hkls = generate_hkls(p.sg_num, p.lattice, p.E_hi)
    mask = CodedApertureMask(
        sequence=build_de_bruijn_sequence(order=5, alphabet=2), bar_widths_um=12.0,
        au_thickness_um=6.0, sub_thickness_um=0.0,
        position_um=torch.tensor([0.0, 0.0, 500.0], dtype=DT),
        rotvec=torch.tensor([0.05, -0.03, 0.02], dtype=DT), edge_softness_um=2.0,
        make_geometry_learnable=False, dtype=DT)
    U = quat_to_orient_mat(torch.tensor(
        [0.56153266089081, -0.1069242896544219, -0.7939419137346801, 0.2071340258144413],
        dtype=DT)).reshape(3, 3)
    t = p.to_tensors()
    model = LaueForwardModel(hkls=hkls, n_pix=(64, 64), px_size=(p.px_x, p.px_y),
                             psf_sigma=1.0, hard=False)
    scan = torch.linspace(-24.0, 24.0, 6, dtype=DT)
    z0 = 4.0

    def frames_at(theta):
        v = theta[1:4]
        w = torch.sqrt(1.0 - (v * v).sum())
        dR = quat_to_orient_mat(torch.stack([w, v[0], v[1], v[2]])).reshape(3, 3)
        src = torch.stack([torch.zeros((), dtype=DT), torch.zeros((), dtype=DT),
                           (z0 + theta[0]) * 1e-6])
        return model.forward_stack((dR @ U).unsqueeze(0), t["lattice"], t["P"], t["R"],
                                   coded_aperture=mask, scan_offsets_um=scan,
                                   source_xyz=src, E_range=(5.0, 12.0))

    with torch.no_grad():
        obs = frames_at(torch.zeros(4, dtype=DT))
    assert float(obs.sum()) > 0
    vox = CodedApertureVoxelMeasurement(voxel_index=0, frame_stack=obs,
                                        scan_offsets_um=scan, U_seed=U, z_seed_um=z0)
    res = DepthResolvedVoxelResult(voxel_index=0, U_refined=U, z_um=z0, z_init_um=z0,
                                   final_loss=0.0, initial_loss=0.0, n_steps=0, dt_s=0.0)
    ref = DepthResolvedVoxelRefiner(p, mask=mask, hkls=hkls)
    nv = 1e-4
    post = ref.posterior(vox, res, noise_variance=nv)

    # Central finite differences (4 parameters; reverse-mode would need one
    # backward pass per pixel).
    cols = []
    with torch.no_grad():
        for k, h in enumerate((1e-3, 1e-7, 1e-7, 1e-7)):
            e = torch.zeros(4, dtype=DT)
            e[k] = h
            cols.append(((frames_at(e) - frames_at(-e)) / (2 * h)).reshape(-1))
    J = torch.stack(cols, dim=1)
    cov = nv * torch.linalg.pinv(J.T @ J)
    sig = cov.diag().sqrt()
    N = obs.numel()
    got_z = post.z_sigma_um
    assert math.isclose(got_z, float(sig[0]), rel_tol=1e-3), (
        f"z sigma {got_z:.4g} vs sqrt(nv inv(JtJ)) {float(sig[0]):.4g} "
        f"(ratio {got_z / float(sig[0]):.3f}; sqrt(N/2) = {math.sqrt(N / 2):.1f})")
    for k in range(3):
        assert math.isclose(post.rot_sigma_deg[k], math.degrees(2 * float(sig[1 + k])),
                            rel_tol=1e-3)
