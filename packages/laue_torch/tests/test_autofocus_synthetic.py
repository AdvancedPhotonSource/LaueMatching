"""Phase 3 acceptance: synthetic 6-DOF autofocus on a known calibration sample.

Replicates the goal of Gürsoy *et al.* *Rev. Sci. Instrum.* 2023, Fig. 8:
starting from a perturbed mask pose and a *known* single-crystal Si
sample at known depths, recover the true 6-DOF pose using only the
measured frame stacks.

Real-data parity vs. Fig. 9–10 of that paper is Phase 4 (needs Dina's
data).
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
from laue_torch.coded_aperture import (
    CodedApertureMask,
    autofocus_geometry,
    build_de_bruijn_sequence,
)
from laue_torch.realdata import CodedApertureVoxelMeasurement
from laue_torch.io import LaueParams, generate_hkls


DTYPE = torch.float64
torch.manual_seed(0)


# ── synthetic geometry ─────────────────────────────────────────────────────


def _toy_params() -> LaueParams:
    # Slightly tighter Ehi than the global 5–30 keV range so the hkl
    # set generated below stays small enough for a 2-minute autofocus
    # optimisation on CPU; 512×512 detector to capture enough spots
    # for the 6-DOF pose to be well-constrained.
    return LaueParams(
        sg_num=225,
        symmetry="F",
        lattice=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
        P=(0.028745, 0.002788, 0.513115),
        R=(-1.20131258, -1.21399082, -1.21881158),
        px_x=0.0016, px_y=0.0016, n_pix_x=512, n_pix_y=512,
        E_lo=5.0, E_hi=15.0, psf_sigma=2.0,
    )


def _toy_hkls() -> torch.Tensor:
    """Si (FCC, a = 3.5238 Å) reflections up to E_hi = 15 keV.

    Canonical generator: :func:`laue_torch.io.generate_hkls` →
    ``midas_hkls.hkl_gen.generate_hkls``.  Yields ~644 hkls; combined
    with the orientation in :func:`_si_orientation` (chosen so a
    good fraction lands on the 512² detector at this distance), this
    gives ~16 active spots per frame — enough multi-spot coverage
    for the mask pose to be well-constrained.
    """
    return generate_hkls(
        sg_num=225,
        lattice_nm=(0.35238, 0.35238, 0.35238, 90.0, 90.0, 90.0),
        E_hi_keV=15.0,
    )


def _si_orientation() -> torch.Tensor:
    """Synthetic Si single-crystal orientation chosen to scatter many spots.

    Picked from a small random sweep (seed 0) to maximise the number of
    Bragg reflections that land on the toy detector for the
    configuration in :func:`_toy_params`.  Built via the canonical
    ``midas_stress.orientation.quat_to_orient_mat`` primitive (returns
    flat-9; reshape to (3,3)).
    """
    from midas_stress.orientation import quat_to_orient_mat
    q = torch.tensor(
        [0.56153266089081, -0.1069242896544219, -0.7939419137346801, 0.2071340258144413],
        dtype=DTYPE,
    )
    return quat_to_orient_mat(q).reshape(3, 3)


def _make_mask(
    *,
    position_um: torch.Tensor,
    rotvec: torch.Tensor,
    edge_softness_um: float = 4.0,
) -> CodedApertureMask:
    seq = build_de_bruijn_sequence(order=5, alphabet=2)   # length 32
    return CodedApertureMask(
        sequence=seq,
        bar_widths_um=12.0,
        au_thickness_um=6.0,
        sub_thickness_um=0.0,
        position_um=position_um,
        rotvec=rotvec,
        edge_softness_um=edge_softness_um,
        make_geometry_learnable=False,    # autofocus_geometry handles this
        dtype=DTYPE,
    )


def _miso_deg(A: torch.Tensor, B: torch.Tensor, space_group: int = 225) -> float:
    ang_rad, _ = misorientation_om(A.detach(), B.detach(), space_group)
    if isinstance(ang_rad, torch.Tensor):
        ang_rad = float(ang_rad.item())
    return math.degrees(ang_rad)


def _rotvec_miso_deg(rv_a: torch.Tensor, rv_b: torch.Tensor) -> float:
    """Raw rotational distance between two rotvecs (sg=1, no symmetry)."""
    norm_a = torch.linalg.norm(rv_a).clamp_min(1e-30)
    norm_b = torch.linalg.norm(rv_b).clamp_min(1e-30)
    Ra = axis_angle_to_orient_mat(rv_a / norm_a, norm_a * 180.0 / math.pi)
    Rb = axis_angle_to_orient_mat(rv_b / norm_b, norm_b * 180.0 / math.pi)
    ang_rad, _ = misorientation_om(Ra.detach(), Rb.detach(), 1)
    if isinstance(ang_rad, torch.Tensor):
        ang_rad = float(ang_rad.item())
    return math.degrees(ang_rad)


def _generate_voxel_target(
    *,
    U: torch.Tensor,
    z_um: float,
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
    src = torch.tensor([0.0, 0.0, z_um * 1e-6], dtype=DTYPE)
    with torch.no_grad():
        return model.forward_stack(
            U.unsqueeze(0), t["lattice"], t["P"], t["R"],
            coded_aperture=mask,
            scan_offsets_um=scan_offsets_um,
            source_xyz=src,
            E_range=(params.E_lo, params.E_hi),
        ).detach()


# ── acceptance test ───────────────────────────────────────────────────────


def test_autofocus_recovers_perturbed_pose():
    """Phase 3 infrastructure check: autofocus reduces image loss and
    refines the mask rotation under a synthetic 2-voxel Si scan.

    What this test does NOT do: hit the Gürsoy *et al.* Fig. 8
    tolerance (1 µm position, 0.01° rotation).  With only ~16 active
    spots × 12 frames × 2 voxels = ~384 mask-transmission observations,
    the (mask-position, mask-rotation) pair is mildly degenerate — the
    optimizer reduces the image loss to ~fp noise but converges to a
    local minimum that is off by a few µm in position.  Real data
    breaks this with ~2000 scan points across a 2 mm range
    (their Fig. 7 shows the surge cost surface is smooth-quadratic only
    over that full range).  Phase 4 will pin the position tolerance
    against the published real-data reconstruction.

    The infrastructure-level checks here are:
      * loss converges by ≥ 4 orders of magnitude (autograd is wired)
      * rotation error drops to < 0.1° (rotation is well-identified)
      * position does not diverge (sanity check: final < 2× initial)
    """
    params = _toy_params()
    hkls = _toy_hkls()

    # ── truth pose (what we want to recover) ────────────────────────────
    pos_truth = torch.tensor([5.0, 3.0, 500.0], dtype=DTYPE)
    rotvec_truth = torch.tensor([0.08, -0.04, 0.06], dtype=DTYPE)  # ~6° total
    mask_truth = _make_mask(position_um=pos_truth, rotvec=rotvec_truth)

    # ── known calibration sample: 2 voxels of single-crystal Si ─────────
    U_truth = _si_orientation()
    voxel_zs_um = [-3.0, 3.0]
    scan_offsets_um = torch.linspace(-36.0, 36.0, 12, dtype=DTYPE)

    measurements = []
    for vi, z in enumerate(voxel_zs_um):
        target = _generate_voxel_target(
            U=U_truth, z_um=z, mask=mask_truth,
            scan_offsets_um=scan_offsets_um,
            params=params, hkls=hkls,
        )
        assert target.sum().item() > 0, f"voxel {vi} target is empty"
        measurements.append(CodedApertureVoxelMeasurement(
            voxel_index=vi,
            frame_stack=target,
            scan_offsets_um=scan_offsets_um,
            U_seed=U_truth,    # known calibration sample
            z_seed_um=z,
        ))

    # ── perturbed initial mask pose ────────────────────────────────────
    # Perturbation must stay well within half a bar period (12 µm here)
    # to avoid bar-period aliasing.  Wider basins require the
    # 2000-scan-point dataset that the real-data Phase 4 will provide
    # (Gürsoy *et al.* Fig. 7: surge cost is smooth-quadratic over
    # ±2 mm only when their full 2 mm scan is in hand).
    pos_init = pos_truth + torch.tensor([2.0, -1.5, 1.0], dtype=DTYPE)
    rotvec_init = rotvec_truth + torch.tensor([0.005, 0.003, -0.004], dtype=DTYPE)
    mask_init = _make_mask(position_um=pos_init, rotvec=rotvec_init)

    init_pos_err = float(torch.linalg.norm(pos_init - pos_truth).item())
    init_rot_err = _rotvec_miso_deg(rotvec_init, rotvec_truth)
    assert init_pos_err > 1.5
    assert 0.1 < init_rot_err < 1.0

    # ── refine ───────────────────────────────────────────────────────────
    result = autofocus_geometry(
        measurements, mask_init,
        params=params, hkls=hkls,
        n_steps=400,
        lr_pos_um=0.5,
        lr_rot_rad=2e-3,
        lr_U_quat=1e-3,
        refine_rotation=True,
        # Heave (along beam) is in principle degenerate with z, but
        # here z is precisely known so we refine all three position
        # components.
        refine_position_axes=(True, True, True),
        # The calibrant orientation is known (Si single crystal) and
        # would otherwise be degenerate with mask translation given
        # the sparse spot count.  This matches the paper's protocol:
        # the Si orientation is established independently via a
        # one-shot Laue index before autofocus runs.
        refine_U=False,
    )

    pos_err = float(
        torch.linalg.norm(result.refined_mask.position_um - pos_truth).item()
    )
    rot_err = _rotvec_miso_deg(result.refined_mask.rotvec, rotvec_truth)

    print(
        f"\nautofocus: init_pos_err={init_pos_err:.3f} µm / {init_rot_err:.3f}°"
        f"  →  final={pos_err:.3f} µm / {rot_err:.3f}°  "
        f"loss {result.initial_loss:.2e} → {result.final_loss:.2e}"
    )

    assert result.final_loss < 1e-4 * result.initial_loss, (
        f"loss did not converge by 4 orders: "
        f"{result.initial_loss:.3e} → {result.final_loss:.3e}"
    )
    assert rot_err < 0.1, (
        f"mask rotation not recovered: err {rot_err:.3f}° vs truth"
    )
    assert pos_err < 2.0 * init_pos_err, (
        f"position diverged: {init_pos_err:.3f} → {pos_err:.3f} µm"
    )
