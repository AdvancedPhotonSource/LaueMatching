"""Strain-refinement extension to the depth-resolved voxel refiner.

Mirrors the Phase 2 acceptance test but recovers a known deviatoric
strain in addition to ``(z, U)``.  Deviatoric (5-vector) strain is used
because the hydrostatic component is formally degenerate with the
lattice parameter under polychromatic Laue (see
the exp6 landscape analysis).
"""
from __future__ import annotations

import math

import pytest
import torch

from midas_stress.orientation import (
    axis_angle_to_orient_mat,
    misorientation_om,
    quat_to_orient_mat,
)

from laue_torch import LaueForwardModel
from laue_torch.coded_aperture import CodedApertureMask, build_de_bruijn_sequence
from laue_torch.geometry import deviatoric5_to_symmetric
from laue_torch.io import LaueParams, generate_hkls
from laue_torch.realdata import (
    CodedApertureVoxelMeasurement,
    DepthResolvedVoxelRefiner,
)


DTYPE = torch.float64
torch.manual_seed(0)


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


def _make_mask() -> CodedApertureMask:
    return CodedApertureMask(
        sequence=build_de_bruijn_sequence(order=5, alphabet=2),
        bar_widths_um=12.0,
        au_thickness_um=6.0,
        sub_thickness_um=0.0,
        position_um=torch.tensor([0.0, 0.0, 500.0], dtype=DTYPE),
        rotvec=torch.tensor([0.05, -0.03, 0.02], dtype=DTYPE),
        edge_softness_um=4.0,
        make_geometry_learnable=False,
        dtype=DTYPE,
    )


def _miso_deg(A: torch.Tensor, B: torch.Tensor, sg: int = 225) -> float:
    ang_rad, _ = misorientation_om(A.detach(), B.detach(), sg)
    if isinstance(ang_rad, torch.Tensor):
        ang_rad = float(ang_rad.item())
    return math.degrees(ang_rad)


def test_strain_round_trip_deviatoric():
    """Recover a known deviatoric strain on a 16-frame coded-aperture stack."""
    params = _toy_params()
    hkls = generate_hkls(
        sg_num=params.sg_num,
        lattice_nm=params.lattice,
        E_hi_keV=params.E_hi,
    )

    U_truth = _favorable_orientation()
    z_truth_um = 8.0
    # Deviatoric 5-vector — first two diagonal components plus three
    # shears (see geometry.deviatoric5_to_symmetric for the exact
    # mapping).  Magnitudes ~5e-4 = ~500 microstrain, well within the
    # paper's regime for non-extreme deformations.
    strain_truth = torch.tensor(
        [4.0e-4, -2.0e-4, 1.5e-4, -1.0e-4, 0.8e-4],
        dtype=DTYPE,
    )

    mask = _make_mask()
    scan_offsets_um = torch.linspace(-36.0, 36.0, 16, dtype=DTYPE)

    # ── synthesize the target ────────────────────────────────────────────
    truth_model = LaueForwardModel(
        hkls=hkls,
        n_pix=(params.n_pix_x, params.n_pix_y),
        px_size=(params.px_x, params.px_y),
        psf_sigma=params.psf_sigma,
        rotation="matrix",
        detector_rotation="rodrigues",
        strain_mode="deviatoric",
        hard=False,
    )
    t = params.to_tensors(dtype=DTYPE)
    src = torch.tensor([0.0, 0.0, z_truth_um * 1e-6], dtype=DTYPE)
    with torch.no_grad():
        target = truth_model.forward_stack(
            U_truth.unsqueeze(0), t["lattice"], t["P"], t["R"],
            strain=strain_truth.unsqueeze(0),
            coded_aperture=mask,
            scan_offsets_um=scan_offsets_um,
            source_xyz=src,
            E_range=(params.E_lo, params.E_hi),
        ).detach()
    assert target.sum().item() > 0, "synthetic target empty"

    # ── seed with a small orientation perturbation, zero strain init,
    #    z seed = 0 (8 µm off truth) ─────────────────────────────────────
    perturb_axis = torch.tensor([0.8018, -0.5345, 0.2673], dtype=DTYPE)
    U_seed = axis_angle_to_orient_mat(perturb_axis, torch.tensor(0.3, dtype=DTYPE)) @ U_truth
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
        n_steps=400,
        lr_z=2.0,
        lr_rot=2e-3,
        lr_strain=5e-5,
        strain_mode="deviatoric",
        refine_strain=True,
    )
    result = refiner.refine(voxel)

    assert result.strain is not None
    assert result.strain.shape == (5,)
    assert result.strain_mode == "deviatoric"

    # Convert to symmetric 3×3 to compare in a frame-invariant way.
    eps_truth = deviatoric5_to_symmetric(strain_truth)
    eps_recv = deviatoric5_to_symmetric(result.strain)
    fro_truth = torch.linalg.matrix_norm(eps_truth).item()
    fro_err = torch.linalg.matrix_norm(eps_recv - eps_truth).item()
    rel_err = fro_err / fro_truth

    z_err = abs(result.z_um - z_truth_um)
    miso = _miso_deg(result.U_refined, U_truth)

    print(
        f"\nstrain round-trip:  z {result.z_init_um:+.2f} → {result.z_um:+.3f} "
        f"(truth {z_truth_um:.2f}, err {z_err:.3f} µm); "
        f"miso {miso:.3f}°; strain Frobenius err {fro_err:.3e} "
        f"(rel {rel_err:.1%}); loss {result.initial_loss:.3e} → "
        f"{result.final_loss:.3e}"
    )

    assert result.final_loss < 0.5 * result.initial_loss
    assert z_err < 1.0, f"depth off by {z_err:.3f} µm"
    assert miso < 1.0, f"orientation off by {miso:.3f}°"
    # Recover the strain tensor within 40 % Frobenius — generous bound
    # given the synthetic has only 16 frames × 16 active spots.  Real
    # data with 2000+ scan points will tighten this dramatically (the
    # Hessian / Cramér-Rao analysis quantifies the data-to-precision
    # relationship explicitly).
    assert rel_err < 0.40, (
        f"strain not recovered: ‖ε_recv−ε_truth‖_F / ‖ε_truth‖_F = {rel_err:.2%}"
    )


def test_refine_strain_requires_non_none_mode():
    """Sanity guard: refine_strain=True without strain_mode raises."""
    with pytest.raises(ValueError):
        DepthResolvedVoxelRefiner(
            params=_toy_params(),
            mask=_make_mask(),
            hkls=torch.tensor([[1, 1, 1]], dtype=torch.long),
            strain_mode="none",
            refine_strain=True,
        )
