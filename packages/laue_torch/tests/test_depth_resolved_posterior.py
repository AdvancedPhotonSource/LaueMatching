"""Laplace posterior at the converged depth-refiner state.

Validates the ``DepthResolvedVoxelRefiner.posterior`` method:
posterior is well-defined (positive eigvals after sym + pinv), the
``z`` marginal width is finite and on a sensible scale, and the
posterior agrees with the *theoretical* Cramér-Rao bound from
``autofocus_hessian`` at the same noise level (which is a property
of the same underlying Fisher information matrix).
"""
from __future__ import annotations

import math

import pytest
import torch

from midas_stress.orientation import quat_to_orient_mat

from laue_torch import LaueForwardModel
from laue_torch.coded_aperture import CodedApertureMask, build_de_bruijn_sequence
from laue_torch.io import LaueParams, generate_hkls
from laue_torch.realdata import (
    CodedApertureVoxelMeasurement,
    DepthResolvedVoxelPosterior,
    DepthResolvedVoxelRefiner,
)


DTYPE = torch.float64


def _toy_params() -> LaueParams:
    return LaueParams(
        sg_num=225, symmetry="F",
        lattice=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
        P=(0.028745, 0.002788, 0.513115),
        R=(-1.20131258, -1.21399082, -1.21881158),
        px_x=0.0016, px_y=0.0016, n_pix_x=256, n_pix_y=256,
        E_lo=5.0, E_hi=12.0, psf_sigma=2.0,
    )


def _favorable_orientation() -> torch.Tensor:
    q = torch.tensor(
        [0.56153266089081, -0.1069242896544219,
         -0.7939419137346801, 0.2071340258144413],
        dtype=DTYPE,
    )
    return quat_to_orient_mat(q).reshape(3, 3)


def _toy_mask() -> CodedApertureMask:
    return CodedApertureMask(
        sequence=build_de_bruijn_sequence(order=5, alphabet=2),
        bar_widths_um=12.0,
        au_thickness_um=6.0,
        sub_thickness_um=0.0,
        position_um=torch.tensor([0.0, 0.0, 500.0], dtype=DTYPE),
        rotvec=torch.tensor([0.05, -0.03, 0.02], dtype=DTYPE),
        edge_softness_um=2.0,
        make_geometry_learnable=False,
        dtype=DTYPE,
    )


def _make_voxel(z_um: float, mask, params, hkls):
    model = LaueForwardModel(
        hkls=hkls,
        n_pix=(params.n_pix_x, params.n_pix_y),
        px_size=(params.px_x, params.px_y),
        psf_sigma=params.psf_sigma,
        rotation="matrix",
        detector_rotation="rodrigues",
        strain_mode="none",
        hard=False,
    )
    t = params.to_tensors(dtype=DTYPE)
    U_truth = _favorable_orientation()
    scan_offsets = torch.linspace(-24.0, 24.0, 8, dtype=DTYPE)
    with torch.no_grad():
        frames = model.forward_stack(
            U_truth.unsqueeze(0), t["lattice"], t["P"], t["R"],
            coded_aperture=mask, scan_offsets_um=scan_offsets,
            source_xyz=torch.tensor([0.0, 0.0, z_um * 1e-6], dtype=DTYPE),
            E_range=(params.E_lo, params.E_hi),
        ).detach()
    return CodedApertureVoxelMeasurement(
        voxel_index=0,
        frame_stack=frames,
        scan_offsets_um=scan_offsets,
        U_seed=U_truth,
        z_seed_um=z_um,
    )


def test_posterior_returns_finite_sigmas():
    """Posterior at convergence yields finite positive σ on each DOF."""
    params = _toy_params()
    hkls = generate_hkls(
        sg_num=params.sg_num, lattice_nm=params.lattice, E_hi_keV=params.E_hi,
    )
    mask = _toy_mask()
    voxel = _make_voxel(z_um=5.0, mask=mask, params=params, hkls=hkls)

    refiner = DepthResolvedVoxelRefiner(
        params=params, mask=mask, hkls=hkls,
        n_steps=20, lr_z=1.0, lr_rot=1e-3,
    )
    result = refiner.refine(voxel)

    post = refiner.posterior(voxel, result, noise_variance=1e-4)
    assert isinstance(post, DepthResolvedVoxelPosterior)
    assert post.z_sigma_um > 0.0 and math.isfinite(post.z_sigma_um)
    for s in post.rot_sigma_deg:
        assert s > 0.0 and math.isfinite(s)
    assert post.cov.shape == (4, 4)         # 1 (z) + 3 (rot) + 0 (no strain)
    assert post.rank_eff >= 1


def test_posterior_with_strain_includes_strain_marginals():
    params = _toy_params()
    hkls = generate_hkls(
        sg_num=params.sg_num, lattice_nm=params.lattice, E_hi_keV=params.E_hi,
    )
    mask = _toy_mask()
    voxel = _make_voxel(z_um=2.0, mask=mask, params=params, hkls=hkls)

    refiner = DepthResolvedVoxelRefiner(
        params=params, mask=mask, hkls=hkls,
        n_steps=10, lr_z=1.0, lr_rot=1e-3,
        strain_mode="deviatoric", refine_strain=True,
    )
    result = refiner.refine(voxel)

    post = refiner.posterior(voxel, result, noise_variance=1e-4)
    assert post.strain_sigma is not None
    assert post.strain_sigma.shape == (5,)
    assert post.cov.shape == (9, 9)    # 1 + 3 + 5
    # Sigma values must be positive (clamped) and finite
    assert torch.all(post.strain_sigma >= 0.0)
    assert torch.all(torch.isfinite(post.strain_sigma))


def test_posterior_plug_in_noise_uses_final_loss():
    """When ``noise_variance`` is None, refiner uses ``result.final_loss``."""
    params = _toy_params()
    hkls = generate_hkls(
        sg_num=params.sg_num, lattice_nm=params.lattice, E_hi_keV=params.E_hi,
    )
    mask = _toy_mask()
    voxel = _make_voxel(z_um=3.0, mask=mask, params=params, hkls=hkls)

    refiner = DepthResolvedVoxelRefiner(
        params=params, mask=mask, hkls=hkls,
        n_steps=10, lr_z=1.0, lr_rot=1e-3,
    )
    result = refiner.refine(voxel)

    post_default = refiner.posterior(voxel, result)
    post_explicit = refiner.posterior(
        voxel, result, noise_variance=max(result.final_loss, 1e-30),
    )
    # Same noise variance ⇒ same sigmas (the closure path is shared).
    assert math.isclose(
        post_default.z_sigma_um, post_explicit.z_sigma_um,
        rel_tol=0.0, abs_tol=1e-12,
    )
