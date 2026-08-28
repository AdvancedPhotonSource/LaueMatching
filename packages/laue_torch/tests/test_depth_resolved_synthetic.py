"""Phase 2 acceptance: synthetic round-trip for the depth-resolved voxel refiner.

Generates a coded-aperture frame stack with a known ground-truth
``(U, z)``, then runs :class:`DepthResolvedVoxelRefiner` with a
perturbed seed and verifies recovery within target tolerances.

This is the only end-to-end correctness gate for Phase 2.  Real-data
parity vs. the Gürsoy *et al.* differential-aperture reference is
Phase 4.
"""
from __future__ import annotations

import math

import pytest
import torch

from midas_stress.orientation import (
    axis_angle_to_orient_mat,
    misorientation_om,
)

from laue_torch import LaueForwardModel
from laue_torch.coded_aperture import CodedApertureMask, build_de_bruijn_sequence
from laue_torch.realdata import (
    CodedApertureVoxelMeasurement,
    DepthResolvedVoxelRefiner,
)
from laue_torch.io import LaueParams


DTYPE = torch.float64
torch.manual_seed(0)


# ── synthetic geometry (toy LaueParams) ────────────────────────────────────

def _toy_params() -> LaueParams:
    return LaueParams(
        sg_num=225,
        symmetry="F",
        lattice=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),  # nm + deg
        P=(0.028745, 0.002788, 0.513115),
        R=(-1.20131258, -1.21399082, -1.21881158),
        px_x=0.0016,
        px_y=0.0016,
        n_pix_x=256,
        n_pix_y=256,
        E_lo=5.0,
        E_hi=30.0,
        psf_sigma=2.0,
    )


def _toy_hkls() -> torch.Tensor:
    return torch.tensor([
        [1, 1, 1], [-1, -1, -1], [2, 0, 0], [-2, 0, 0],
        [0, 2, 0], [0, -2, 0], [0, 0, 2], [0, 0, -2],
        [2, 2, 0], [-2, -2, 0], [1, -1, 3], [3, -1, 1],
        [1, -1, -3], [-3, 1, 1], [2, -2, 4], [4, 2, -2],
    ], dtype=torch.long)


def _toy_mask(*, learnable: bool = False, edge_softness_um: float = 2.0
              ) -> CodedApertureMask:
    """A 32-bar de-Bruijn-like mask sitting at z = 500 µm above the sample."""
    seq = build_de_bruijn_sequence(order=5, alphabet=2)   # length 32
    return CodedApertureMask(
        sequence=seq,
        bar_widths_um=12.0,
        au_thickness_um=6.0,
        sub_thickness_um=0.0,
        position_um=torch.tensor([0.0, 0.0, 500.0], dtype=DTYPE),
        edge_softness_um=edge_softness_um,
        make_geometry_learnable=learnable,
        dtype=DTYPE,
    )


def _seed_orientation() -> torch.Tensor:
    """An orthonormal rotation close to the hand-typed test orientation.

    The truncated literal is not orthonormal to fp64 precision, which
    contaminates the misorientation trace formula by ~1e-5.  SVD-project
    onto SO(3) to get a clean rotation matrix.
    """
    M = torch.tensor([
        [0.867151, 0.494088, 0.062670],
        [-0.052670, 0.216095, -0.974957],
        [-0.495254, 0.842135, 0.213410],
    ], dtype=DTYPE)
    U, _, Vh = torch.linalg.svd(M)
    R = U @ Vh
    if torch.det(R) < 0:
        R = U @ torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=DTYPE)) @ Vh
    return R


def _miso_deg(A: torch.Tensor, B: torch.Tensor, space_group: int = 225) -> float:
    """Cubic-symmetry-aware misorientation between two 3×3 rotations [deg].

    Uses the canonical :func:`midas_stress.orientation.misorientation_om`
    primitive — it returns radians (per memory
    ``feedback_midas_stress_miso_radians``) and accepts flat or (3,3)
    layouts (both packages already speak the (3,3) form).
    """
    angle_rad, _axis = misorientation_om(A.detach(), B.detach(), space_group)
    if isinstance(angle_rad, torch.Tensor):
        angle_rad = float(angle_rad.item())
    return math.degrees(angle_rad)


def _generate_frame_stack(
    *,
    U_truth: torch.Tensor,
    z_truth_um: float,
    mask: CodedApertureMask,
    scan_offsets_um: torch.Tensor,
    params: LaueParams,
    hkls: torch.Tensor,
) -> torch.Tensor:
    t = params.to_tensors(dtype=DTYPE)
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
    src = torch.tensor([0.0, 0.0, z_truth_um * 1e-6], dtype=DTYPE)
    with torch.no_grad():
        return model.forward_stack(
            U_truth.unsqueeze(0), t["lattice"], t["P"], t["R"],
            coded_aperture=mask,
            scan_offsets_um=scan_offsets_um,
            source_xyz=src,
            E_range=(params.E_lo, params.E_hi),
        ).detach()


# ── acceptance test ──────────────────────────────────────────────────────


def test_round_trip_recovers_depth_and_orientation():
    """End-to-end Phase 2 acceptance: recover (z, U) from a 16-frame stack.

    Tolerances:
      * z within 1 µm of truth (= mask scan step size)
      * miso < 1° from truth (orientation channel is intentionally
        soft-coupled to the depth fit; 1° is plenty for a "we found the
        right grain" check, and the synthetic exp5 tests already pin
        the sub-deg orientation recovery against ODF refinement).
    """
    params = _toy_params()
    hkls = _toy_hkls()

    U_truth = _seed_orientation()
    z_truth_um = 12.0

    mask = _toy_mask(learnable=False, edge_softness_um=2.0)
    # 16 scan frames over 96 µm (covers ~3 periods of the de-Bruijn pattern)
    scan_offsets_um = torch.linspace(-48.0, 48.0, 16, dtype=DTYPE)

    target = _generate_frame_stack(
        U_truth=U_truth,
        z_truth_um=z_truth_um,
        mask=mask,
        scan_offsets_um=scan_offsets_um,
        params=params,
        hkls=hkls,
    )
    assert target.sum().item() > 0, "synthetic forward produced an empty stack"

    # Perturb the seed by a known rotation about an arbitrary axis.
    # Canonical primitive: ``axis_angle_to_orient_mat`` (degrees) from
    # midas_stress — and unlike the laue_torch / midas_stress Rodrigues
    # paths it is smooth at small angles too, so we keep the same
    # primitive for both the perturbation (here) and the refinement
    # inner loop (via quaternion delta).
    perturb_axis = torch.tensor([0.8018, -0.5345, 0.2673], dtype=DTYPE)
    perturb_angle_deg = torch.tensor(0.4, dtype=DTYPE)
    U_seed = axis_angle_to_orient_mat(perturb_axis, perturb_angle_deg) @ U_truth
    seed_miso_deg = _miso_deg(U_seed, U_truth)
    assert seed_miso_deg > 0.1, f"seed perturbation collapsed to {seed_miso_deg}"

    voxel = CodedApertureVoxelMeasurement(
        voxel_index=0,
        frame_stack=target,
        scan_offsets_um=scan_offsets_um,
        U_seed=U_seed,
        z_seed_um=0.0,
    )

    refiner = DepthResolvedVoxelRefiner(
        params=params,
        mask=mask,
        hkls=hkls,
        n_steps=300,
        lr_z=2.0,
        lr_rot=2e-3,
    )
    result = refiner.refine(voxel)

    z_err = abs(result.z_um - z_truth_um)
    miso = _miso_deg(result.U_refined, U_truth)

    # Loss must improve substantially
    assert result.final_loss < 0.5 * result.initial_loss, (
        f"loss did not improve: initial={result.initial_loss:.3e}, "
        f"final={result.final_loss:.3e}"
    )
    assert z_err < 1.0, (
        f"depth not recovered: z_truth={z_truth_um}, z_refined={result.z_um}, "
        f"err={z_err:.3f} µm"
    )
    assert miso < 1.0, (
        f"orientation not recovered: seed_miso={seed_miso_deg:.3f}°, "
        f"final_miso={miso:.3f}°"
    )
