"""Phase 3: 6-DOF digital autofocusing of the coded-aperture geometry.

Reproduces the goal of Gürsoy *et al.* *Rev. Sci. Instrum.* **94**, 013702
(2023) — recover the 6-DOF coded-aperture pose (surge / sway / heave +
yaw / pitch / roll) from a known calibration sample — but as a single
joint Adam optimisation over a differentiable forward model, rather
than the original sequential coordinate-descent + exhaustive search +
NNLS pipeline.

Loss
----

.. math::
   L(\\text{pose}, U) = \\sum_v\\sum_m
       \\|\\, f(U,\\,z_v;\\, \\text{pose})_m - d_{v,m}\\,\\|_2^2

* The calibration sample is a *known* single crystal (typically 10 µm
  strain-free Si), so a single orientation ``U`` is shared across all
  voxels.
* Voxel depths ``z_v`` are held fixed at their known values — that is
  what makes the calibration sample useful: it pins the depth ambiguity
  that would otherwise be degenerate with mask z translation.

Returned :class:`AutofocusResult` carries the refined mask (a fresh
:class:`CodedApertureMask` instance whose ``position_um`` and ``rotvec``
buffers are the recovered values) plus diagnostics.

Orientation handling uses the canonical primitives from
``midas_stress.orientation`` (see memory
``feedback_orientation_from_midas_stress``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import math
import time

import torch
from torch import Tensor, nn

from typing import TYPE_CHECKING

from midas_stress.orientation import quat_to_orient_mat

from ..forward import LaueForwardModel
from ..io import LaueParams
from .mask import CodedApertureMask

if TYPE_CHECKING:
    from ..realdata.depth_resolved import CodedApertureVoxelMeasurement


def _quat_to_rotmat(q: Tensor) -> Tensor:
    """Unit-normalised quaternion → (3, 3) rotation matrix (smooth at identity).

    Canonical primitive: ``midas_stress.orientation.quat_to_orient_mat``;
    we normalise and reshape (it returns flat-9 row-major per memory
    ``feedback_midas_stress_miso_radians``).
    """
    q = q / torch.linalg.norm(q).clamp_min(1e-30)
    return quat_to_orient_mat(q).reshape(3, 3)


@dataclass
class AutofocusResult:
    """Outcome of a digital-autofocus run."""

    refined_mask: CodedApertureMask
    U_refined: Tensor               # (3, 3) — shared orientation
    pose_position_init: Tensor      # initial (3,) µm
    pose_rotvec_init: Tensor        # initial (3,) rad
    final_loss: float
    initial_loss: float
    n_steps: int
    dt_s: float
    metadata: dict = field(default_factory=dict)


def autofocus_geometry(
    measurements: "Sequence[CodedApertureVoxelMeasurement]",
    mask: CodedApertureMask,
    *,
    params: LaueParams,
    hkls: Tensor,
    n_steps: int = 500,
    lr_pos_um: float = 1.0,
    lr_rot_rad: float = 2.0e-3,
    lr_U_quat: float = 1.0e-3,
    refine_rotation: bool = True,
    refine_position_axes: tuple[bool, bool, bool] = (True, True, True),
    refine_U: bool = True,
    psf_sigma: Optional[float] = None,
    E_range: Optional[tuple[float, float]] = None,
    annealing_schedule: Optional[Sequence[tuple[float, int]]] = None,
) -> AutofocusResult:
    """Jointly refine the mask pose and the calibration-sample orientation.

    Parameters
    ----------
    measurements
        One :class:`CodedApertureVoxelMeasurement` per voxel of the
        calibration scan.  ``frame_stack``, ``scan_offsets_um``, and
        ``z_seed_um`` are used; ``U_seed`` of the *first* voxel is
        used as the shared orientation seed.  Per-voxel ``z`` is
        treated as *known* (held fixed at ``z_seed_um``).
    mask
        Initial-guess mask.  Its ``rotvec`` must be non-zero (the
        axis-angle path is structurally singular at zero — see the
        note in :func:`laue_torch.coded_aperture.mask._rotvec_to_matrix`).
        A copy with ``make_geometry_learnable=True`` is created
        internally; the input is not modified.
    params
        :class:`LaueParams` for the underlying ``LaueForwardModel``.
    hkls
        Reflection list.
    n_steps, lr_pos_um, lr_rot_rad, lr_U_quat
        Adam optimisation knobs.  ``lr_pos_um`` is in micrometers,
        ``lr_rot_rad`` is in radians along ``rotvec``.
    refine_rotation
        If False, only ``position_um`` is refined.
    refine_position_axes
        ``(refine_x, refine_y, refine_z)`` — controls which mask
        position components are learnable.  Heave (along the beam
        direction) is degenerate with the *unknown* part of voxel z;
        for a calibration sample where z is precisely known there is
        no such degeneracy and all three may be refined together.
    refine_U
        If False, the shared orientation is held fixed at the seed.
        Use this when the calibration sample is a known-orientation
        single crystal — for a thin sample where the diffracted rays
        span a small angular range, ``U`` becomes degenerate with
        mask translation and the position recovery fails unless
        ``U`` is pinned.
    annealing_schedule
        Optional coarse-to-fine sequence of
        ``(edge_softness_um, n_substeps)`` tuples.  When supplied, the
        mask's ``edge_softness_um`` is set to each value in turn and
        Adam runs for ``n_substeps`` iterations at that softness.
        Widening edge softness early (e.g. one bar width or more)
        smooths the bar-period local minima and lets Adam slide across
        them, after which a tightening schedule pins the precise pose.
        ``n_steps`` is ignored when this is provided.

        Example: ``[(6.0, 100), (2.0, 100), (0.5, 200)]`` — start with
        softness wider than half a bar width, anneal to physical
        sharpness over 400 steps.

    Returns
    -------
    :class:`AutofocusResult` containing the refined mask, the recovered
    shared orientation, and diagnostics.
    """
    if not measurements:
        raise ValueError("autofocus_geometry needs at least one voxel measurement")

    dtype = measurements[0].frame_stack.dtype
    device = measurements[0].frame_stack.device

    # Build a fresh mask whose pose (position + rotvec) are nn.Parameters
    # so Adam can refine them directly.  This avoids the autograd break
    # that any ``mask.position_um.data = X`` workaround would introduce.
    refined_mask = CodedApertureMask(
        sequence=mask.sequence.detach().clone().to(torch.int64),
        bar_widths_um=mask.bar_widths_um.detach().clone(),
        au_thickness_um=float(mask.au_thickness_um.item()),
        sub_thickness_um=float(mask.sub_thickness_um.item()),
        position_um=mask.position_um.detach().clone(),
        rotvec=mask.rotvec.detach().clone(),
        edge_softness_um=mask.edge_softness_um,
        make_geometry_learnable=True,
        dtype=dtype,
    )

    pos_init = refined_mask.position_um.detach().clone()
    rotvec_init = refined_mask.rotvec.detach().clone()

    # Per-axis position freeze: register a gradient hook on the position
    # Parameter that zeros out frozen-axis gradient components after each
    # backward but before each optimiser step.  This keeps a single
    # ``position_um`` Parameter (clean API) while letting the caller
    # freeze, say, heave alone for cases where it is degenerate with the
    # voxel z (unknown sample depth).
    pos_freeze_mask = torch.tensor(
        [1.0 if f else 0.0 for f in refine_position_axes],
        dtype=dtype, device=device,
    )

    def _freeze_position_grad(grad: Tensor) -> Tensor:
        return grad * pos_freeze_mask

    refined_mask.position_um.register_hook(_freeze_position_grad)

    if not refine_rotation:
        # Freeze rotation by zeroing its gradient via the same trick;
        # cheaper than adjusting parameter groups around it.
        refined_mask.rotvec.register_hook(lambda g: torch.zeros_like(g))

    # Shared orientation perturbation as a delta quaternion (smooth at
    # identity — see ``DepthResolvedVoxelRefiner`` for the same pattern).
    U_seed = measurements[0].U_seed.to(dtype=dtype, device=device)
    delta_q_tensor = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
    if refine_U:
        delta_q = nn.Parameter(delta_q_tensor)
    else:
        delta_q = delta_q_tensor  # treated as a constant in the loss

    opt_groups = [
        {"params": [refined_mask.position_um], "lr": lr_pos_um},
        {"params": [refined_mask.rotvec], "lr": lr_rot_rad},
    ]
    if refine_U:
        opt_groups.append({"params": [delta_q], "lr": lr_U_quat})
    opt = torch.optim.Adam(opt_groups)

    sigma = psf_sigma if psf_sigma is not None else params.psf_sigma
    erange = E_range or (params.E_lo, params.E_hi)
    model = LaueForwardModel(
        hkls=hkls,
        n_pix=(params.n_pix_x, params.n_pix_y),
        px_size=(params.px_x, params.px_y),
        psf_sigma=sigma,
        rotation="matrix",
        detector_rotation="rodrigues",
        strain_mode="none",
        hard=False,
    )

    t = params.to_tensors(dtype=dtype, device=str(device))
    lat = t["lattice"]
    P = t["P"]
    R_det = t["R"]

    # Pre-broadcast: each voxel keeps its own (frame_stack, scan_offsets, z).
    voxel_data = []
    for v in measurements:
        voxel_data.append(dict(
            target=v.frame_stack.to(dtype=dtype, device=device),
            offsets=v.scan_offsets_um.to(dtype=dtype, device=device),
            z=float(v.z_seed_um),
        ))

    def _loss() -> Tensor:
        dR = _quat_to_rotmat(delta_q)
        U = (dR @ U_seed).unsqueeze(0)
        total = torch.zeros((), dtype=dtype, device=device)
        for v in voxel_data:
            src = torch.tensor(
                [0.0, 0.0, v["z"] * 1.0e-6], dtype=dtype, device=device,
            )
            pred = model.forward_stack(
                U, lat, P, R_det,
                coded_aperture=refined_mask,
                scan_offsets_um=v["offsets"],
                source_xyz=src,
                E_range=erange,
            )
            total = total + (pred - v["target"]).pow(2).mean()
        return total / len(voxel_data)

    t0 = time.perf_counter()
    with torch.no_grad():
        initial_loss = float(_loss().item())

    # Resolve the schedule: either a single (mask.edge_softness, n_steps)
    # block or the user-supplied annealing schedule.
    if annealing_schedule is None:
        schedule = [(float(refined_mask.edge_softness_um), int(n_steps))]
    else:
        schedule = [(float(es), int(ns)) for es, ns in annealing_schedule]

    total_steps = 0
    for edge_softness_um, n_substeps in schedule:
        refined_mask.edge_softness_um = float(edge_softness_um)
        for _step in range(n_substeps):
            opt.zero_grad()
            L = _loss()
            L.backward()
            opt.step()
        total_steps += n_substeps

    with torch.no_grad():
        final_loss = float(_loss().item())
        U_refined = (_quat_to_rotmat(delta_q) @ U_seed).detach().clone()

    return AutofocusResult(
        refined_mask=refined_mask,
        U_refined=U_refined,
        pose_position_init=pos_init.detach().clone(),
        pose_rotvec_init=rotvec_init.detach().clone(),
        final_loss=final_loss,
        initial_loss=initial_loss,
        n_steps=int(total_steps),
        dt_s=time.perf_counter() - t0,
        metadata={
            "n_voxels": len(measurements),
            "refine_rotation": refine_rotation,
            "refine_position_axes": refine_position_axes,
            "annealing_schedule": list(schedule),
        },
    )
