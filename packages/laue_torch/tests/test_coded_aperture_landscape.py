"""Hessian / Cramér–Rao analysis of the coded-aperture autofocus loss.

This is the standalone-publishable analysis: the differentiable
forward + ``torch.autograd.functional.hessian`` give us the 6×6 mask-
pose Hessian at any chosen pose, and the Gaussian-noise CR bound
translates that into a precision floor for each DOF as a function of
the assumed per-pixel noise σ and the number of pixel observations.
"""
from __future__ import annotations

import math

import pytest
import torch

from midas_stress.orientation import quat_to_orient_mat

from laue_torch import LaueForwardModel
from laue_torch.coded_aperture import (
    CodedApertureMask,
    LandscapeReport,
    autofocus_hessian,
    build_de_bruijn_sequence,
    format_report,
)
from laue_torch.io import LaueParams, generate_hkls
from laue_torch.realdata import CodedApertureVoxelMeasurement


DTYPE = torch.float64


def _toy_params() -> LaueParams:
    return LaueParams(
        sg_num=225,
        symmetry="F",
        lattice=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
        P=(0.028745, 0.002788, 0.513115),
        R=(-1.20131258, -1.21399082, -1.21881158),
        px_x=0.0016, px_y=0.0016, n_pix_x=512, n_pix_y=512,
        E_lo=5.0, E_hi=15.0, psf_sigma=2.0,
    )


def _favorable_orientation() -> torch.Tensor:
    q = torch.tensor(
        [0.56153266089081, -0.1069242896544219,
         -0.7939419137346801, 0.2071340258144413],
        dtype=DTYPE,
    )
    return quat_to_orient_mat(q).reshape(3, 3)


def _make_mask(*, position_um, rotvec) -> CodedApertureMask:
    return CodedApertureMask(
        sequence=build_de_bruijn_sequence(order=5, alphabet=2),
        bar_widths_um=12.0,
        au_thickness_um=6.0,
        sub_thickness_um=0.0,
        position_um=position_um,
        rotvec=rotvec,
        edge_softness_um=2.0,
        make_geometry_learnable=False,
        dtype=DTYPE,
    )


def _synth_calibration_scan(n_voxels: int, n_frames: int):
    params = _toy_params()
    hkls = generate_hkls(
        sg_num=params.sg_num, lattice_nm=params.lattice, E_hi_keV=params.E_hi,
    )
    U_truth = _favorable_orientation()
    pos_truth = torch.tensor([5.0, 3.0, 500.0], dtype=DTYPE)
    rotvec_truth = torch.tensor([0.08, -0.04, 0.06], dtype=DTYPE)
    mask = _make_mask(position_um=pos_truth, rotvec=rotvec_truth)

    voxel_zs_um = torch.linspace(-3.0, 3.0, n_voxels, dtype=DTYPE).tolist()
    scan_offsets_um = torch.linspace(-36.0, 36.0, n_frames, dtype=DTYPE)

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

    measurements = []
    for vi, z in enumerate(voxel_zs_um):
        with torch.no_grad():
            frames = model.forward_stack(
                U_truth.unsqueeze(0), t["lattice"], t["P"], t["R"],
                coded_aperture=mask,
                scan_offsets_um=scan_offsets_um,
                source_xyz=torch.tensor([0.0, 0.0, z * 1e-6], dtype=DTYPE),
                E_range=(params.E_lo, params.E_hi),
            ).detach()
        measurements.append(CodedApertureVoxelMeasurement(
            voxel_index=vi,
            frame_stack=frames,
            scan_offsets_um=scan_offsets_um,
            U_seed=U_truth,
            z_seed_um=z,
        ))
    return measurements, mask, params, hkls


def test_hessian_at_truth_is_positive_semidefinite():
    """The MSE Hessian at the true pose must have non-negative eigenvalues."""
    measurements, mask, params, hkls = _synth_calibration_scan(
        n_voxels=2, n_frames=6,
    )
    report = autofocus_hessian(
        measurements, mask, params=params, hkls=hkls, sigma_pixel=0.01,
    )
    assert report.eigvals.shape == (6,)
    # Eigenvalues are sorted ascending; the smallest one tells us about
    # the floppy mode.  Tolerate small *negative* eigenvalues due to
    # finite-difference noise in the Hessian, but no large negatives.
    assert report.eigvals.min().item() > -1e-10, (
        f"min eigenvalue = {report.eigvals.min().item():.3e}; "
        f"Hessian is not PSD at truth"
    )
    assert report.eigvals.max().item() > 0.0


def test_cr_sigma_scales_with_pixel_noise():
    """CR-σ ∝ σ_pixel: doubling the noise doubles the precision bound."""
    measurements, mask, params, hkls = _synth_calibration_scan(
        n_voxels=2, n_frames=6,
    )
    r_lo = autofocus_hessian(
        measurements, mask, params=params, hkls=hkls, sigma_pixel=0.01,
    )
    r_hi = autofocus_hessian(
        measurements, mask, params=params, hkls=hkls, sigma_pixel=0.02,
    )
    ratio = r_hi.cr_sigma / r_lo.cr_sigma.clamp_min(1e-30)
    # Each DOF's bound should scale by exactly 2.0 (within fp64 noise).
    assert torch.allclose(ratio, 2.0 * torch.ones_like(ratio), atol=1e-3)


def test_cr_sigma_improves_with_more_frames():
    """Adding scan frames must lower the CR-σ on every DOF.

    The published autofocus protocol scans the mask through 2000 points;
    this test confirms our model agrees that *more frames → better
    pose precision*, validating the analysis's dependence on
    n_observations.
    """
    m6, mask, params, hkls = _synth_calibration_scan(n_voxels=2, n_frames=6)
    m18, _, _, _ = _synth_calibration_scan(n_voxels=2, n_frames=18)
    r6 = autofocus_hessian(m6, mask, params=params, hkls=hkls, sigma_pixel=0.01)
    r18 = autofocus_hessian(m18, mask, params=params, hkls=hkls, sigma_pixel=0.01)
    # At fixed σ_pixel, the bound improves at least sqrt(3)× when the
    # observation count triples (Fisher info is additive over independent
    # observations).
    sqrt3 = math.sqrt(3.0)
    improvement = r6.cr_sigma / r18.cr_sigma.clamp_min(1e-30)
    # Allow the actual ratio to fall below sqrt(3) if a particular DOF
    # is degenerate, but require *some* improvement across the board.
    assert (improvement > 1.0).all(), (
        f"adding frames did not improve every CR-σ: ratio = {improvement.tolist()}"
    )
    # The well-constrained DOFs (rotation, in-plane position) should
    # see close-to-the-theoretical √3 improvement.
    median_imp = improvement.median().item()
    assert median_imp > 1.3, (
        f"median CR-σ improvement only {median_imp:.2f}× (expected ≳ √3 ≈ 1.73)"
    )


def test_report_format_runs():
    """``format_report`` is the user-facing summary; just runs without error."""
    measurements, mask, params, hkls = _synth_calibration_scan(
        n_voxels=2, n_frames=4,
    )
    report = autofocus_hessian(
        measurements, mask, params=params, hkls=hkls, sigma_pixel=0.01,
    )
    text = format_report(report)
    assert "Eigenvalues" in text
    assert "Cramér-Rao σ" in text
    assert "softest modes" in text
