"""Multi-voxel joint refinement with spatial TV regularization.

Validates that:

1. With ``lambda_z = lambda_U = 0`` the joint refiner reduces to
   per-voxel-independent refinement and recovers per-voxel depths
   to comparable precision.
2. With a non-zero TV weight, a *noisy* synthetic recovers a
   *smoother* depth profile than the unregularized run.  The noise
   is the kind of mismatch a per-voxel solver is most vulnerable
   to; TV is the differentiable approach's natural defence.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_stress.orientation import quat_to_orient_mat

from laue_torch import LaueForwardModel
from laue_torch.coded_aperture import (
    CodedApertureMask,
    build_de_bruijn_sequence,
)
from laue_torch.io import LaueParams, generate_hkls
from laue_torch.realdata import (
    CodedApertureVoxelMeasurement,
    MultiVoxelTVRefiner,
)


DTYPE = torch.float64
torch.manual_seed(0)


def _params() -> LaueParams:
    return LaueParams(
        sg_num=225, symmetry="F",
        lattice=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
        P=(0.028745, 0.002788, 0.513115),
        R=(-1.20131258, -1.21399082, -1.21881158),
        px_x=0.0016, px_y=0.0016, n_pix_x=512, n_pix_y=512,
        E_lo=5.0, E_hi=15.0, psf_sigma=2.0,
    )


def _orientation() -> torch.Tensor:
    q = torch.tensor(
        [0.56153266089081, -0.1069242896544219,
         -0.7939419137346801, 0.2071340258144413],
        dtype=DTYPE,
    )
    return quat_to_orient_mat(q).reshape(3, 3)


def _make_mask() -> CodedApertureMask:
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


def _build_voxels(*, params, hkls, mask, z_truth_list, n_frames=8,
                  pixel_noise_sigma=0.0, rng_seed=0):
    """Render a multi-voxel synthetic with optional Gaussian pixel noise."""
    rng = np.random.default_rng(rng_seed)
    U = _orientation()
    scan = torch.linspace(-24.0, 24.0, n_frames, dtype=DTYPE)
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
    voxels = []
    for vi, z in enumerate(z_truth_list):
        with torch.no_grad():
            frames = model.forward_stack(
                U.unsqueeze(0), t["lattice"], t["P"], t["R"],
                coded_aperture=mask, scan_offsets_um=scan,
                source_xyz=torch.tensor([0.0, 0.0, z * 1e-6], dtype=DTYPE),
                E_range=(params.E_lo, params.E_hi),
            ).detach()
        if pixel_noise_sigma > 0:
            noise = torch.tensor(
                rng.normal(scale=pixel_noise_sigma,
                            size=tuple(frames.shape)),
                dtype=DTYPE,
            )
            frames = frames + noise
        voxels.append(CodedApertureVoxelMeasurement(
            voxel_index=vi, frame_stack=frames, scan_offsets_um=scan,
            U_seed=U, z_seed_um=0.0,             # seed away from truth
        ))
    return voxels


def test_lambda_zero_matches_per_voxel_behaviour():
    """``lambda_z = lambda_U = 0`` ⇒ joint refiner is per-voxel-independent."""
    params = _params()
    hkls = generate_hkls(
        sg_num=params.sg_num, lattice_nm=params.lattice, E_hi_keV=params.E_hi,
    )
    mask = _make_mask()
    # Wider voxel separation so the per-voxel depth signal is well
    # above the noise floor at our 512² synthetic geometry.
    z_truth = [-15.0, 0.0, 15.0]
    voxels = _build_voxels(
        params=params, hkls=hkls, mask=mask, z_truth_list=z_truth,
        n_frames=12,
    )

    refiner = MultiVoxelTVRefiner(
        params=params, mask=mask, hkls=hkls,
        n_steps=200, lr_z=2.0, lr_rot=1e-3,
        lambda_z=0.0, lambda_U=0.0,
    )
    result = refiner.refine(voxels)
    assert len(result.per_voxel) == 3
    for r, z_true in zip(result.per_voxel, z_truth):
        assert abs(r.z_um - z_true) < 1.0, (
            f"voxel {r.voxel_index}: z {r.z_um:.3f} vs truth {z_true:.3f}"
        )
    # TV penalty is zero (no regularization applied).
    assert result.final_loss_tv == 0.0


def test_tv_smooths_a_noisy_profile():
    """A pixel-noise-corrupted synthetic produces a smoother z profile
    when ``lambda_z > 0`` than when it equals zero.

    "Smoother" is measured by the total-variation norm of the
    recovered depth field — that's exactly what the prior penalises.
    """
    params = _params()
    hkls = generate_hkls(
        sg_num=params.sg_num, lattice_nm=params.lattice, E_hi_keV=params.E_hi,
    )
    mask = _make_mask()
    # Smooth truth profile.
    z_truth = list(np.linspace(-3.0, 3.0, 5, dtype=np.float64))
    voxels = _build_voxels(
        params=params, hkls=hkls, mask=mask, z_truth_list=z_truth,
        pixel_noise_sigma=0.02,           # 2 % noise (well above signal floor)
        rng_seed=1,
    )

    def _tv(z_list):
        a = np.asarray(z_list, dtype=np.float64)
        return float(np.abs(a[1:] - a[:-1]).sum())

    unreg = MultiVoxelTVRefiner(
        params=params, mask=mask, hkls=hkls,
        n_steps=120, lr_z=2.0, lr_rot=1e-3,
        lambda_z=0.0, lambda_U=0.0,
    ).refine(voxels)
    z_unreg = [r.z_um for r in unreg.per_voxel]
    tv_unreg = _tv(z_unreg)

    reg = MultiVoxelTVRefiner(
        params=params, mask=mask, hkls=hkls,
        n_steps=120, lr_z=2.0, lr_rot=1e-3,
        lambda_z=1.0e-4, lambda_U=0.0,
    ).refine(voxels)
    z_reg = [r.z_um for r in reg.per_voxel]
    tv_reg = _tv(z_reg)

    print(
        f"\nunregularised z: {[f'{z:+.3f}' for z in z_unreg]}  TV = {tv_unreg:.3f}"
    )
    print(
        f"regularised  z: {[f'{z:+.3f}' for z in z_reg]}  TV = {tv_reg:.3f}"
    )
    print(f"truth         z: {[f'{z:+.3f}' for z in z_truth]}")

    # The regulariser should reduce the variation in the recovered
    # profile.  Tight bound would be brittle; we require the TV-
    # regularised profile to have TV strictly less than the
    # unregularised one.
    assert tv_reg < tv_unreg, (
        f"TV did not smooth: regularised TV {tv_reg:.3f} vs "
        f"unregularised {tv_unreg:.3f}"
    )
