"""Joint multi-voxel refinement with spatial total-variation regularization.

For a scan where adjacent voxels are *physically* close (a thin
sample, a microbeam raster, a narrow z-slice through a single grain),
the recovered depth and orientation profiles are expected to be
smooth.  The per-voxel
:class:`DepthResolvedVoxelRefiner` treats each voxel independently
and so cannot exploit this spatial prior.  This module adds a
simple total-variation (TV) prior on the depth and orientation
fields:

.. math::
   \\mathcal{L}_{\\rm total} = \\sum_v \\mathcal{L}_v
       + \\lambda_z \\sum_v |z_{v+1} - z_v|
       + \\lambda_U \\sum_v \\|\\Delta U_{v,v+1}\\|_F

where the orientation TV is implemented as a Frobenius norm on the
difference of consecutive rotation matrices (the natural distance
on $\\mathrm{SO}(3)$ for small misorientations is proportional to
this).  No strain TV is added by default --- strain is per-voxel
physics and need not be smooth.

The differentiable forward + per-voxel parameter blocks let this be
a single Adam loop; the only addition vs the per-voxel refiner is
the regularizer term in the loss.

This is *one* concrete advantage of the differentiable approach over
the two-stage solver, which has no natural way to couple voxels at
all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import math
import time

import torch
from torch import Tensor, nn

from midas_stress.orientation import quat_to_orient_mat

from ..coded_aperture import CodedApertureMask
from ..forward import LaueForwardModel
from ..io import LaueParams
from .depth_resolved import (
    CodedApertureVoxelMeasurement,
    DepthResolvedVoxelResult,
)


def _quat_to_rotmat(q: Tensor) -> Tensor:
    q = q / torch.linalg.norm(q).clamp_min(1e-30)
    return quat_to_orient_mat(q).reshape(3, 3)


@dataclass
class MultiVoxelTVResult:
    """Aggregate output of a joint multi-voxel TV refinement."""

    per_voxel: list[DepthResolvedVoxelResult]
    lambda_z: float
    lambda_U: float
    final_loss_data: float        # sum of per-voxel data terms
    final_loss_tv: float          # sum of TV penalty terms
    initial_loss_total: float
    n_steps: int
    dt_s: float
    metadata: dict = field(default_factory=dict)


class MultiVoxelTVRefiner:
    """Joint refinement of all voxels in a scan with a spatial TV prior.

    Parameters mirror :class:`DepthResolvedVoxelRefiner`.  Extra knobs:

    Parameters
    ----------
    lambda_z
        TV regularization weight on the depth differences ``|z_{v+1} −
        z_v|``.  Setting to 0 recovers per-voxel-independent
        refinement (and the result should match the per-voxel
        refiner up to optimisation noise).
    lambda_U
        TV regularization weight on consecutive rotation matrices
        ``‖U_{v+1} − U_v‖_F``.  Same scaling note as ``lambda_z``.
    """

    def __init__(
        self,
        params: LaueParams,
        *,
        mask: CodedApertureMask,
        hkls: Tensor,
        n_steps: int = 200,
        lr_z: float = 2.0,
        lr_rot: float = 1.0e-3,
        lambda_z: float = 0.0,
        lambda_U: float = 0.0,
        psf_sigma: Optional[float] = None,
        E_range: Optional[tuple[float, float]] = None,
    ):
        self.params = params
        self.mask = mask
        self.hkls = hkls
        self.n_steps = int(n_steps)
        self.lr_z = float(lr_z)
        self.lr_rot = float(lr_rot)
        self.lambda_z = float(lambda_z)
        self.lambda_U = float(lambda_U)
        self.E_range = E_range or (params.E_lo, params.E_hi)

        sigma = psf_sigma if psf_sigma is not None else params.psf_sigma
        self.model = LaueForwardModel(
            hkls=hkls,
            n_pix=(params.n_pix_x, params.n_pix_y),
            px_size=(params.px_x, params.px_y),
            psf_sigma=sigma,
            rotation="matrix",
            detector_rotation="rodrigues",
            strain_mode="none",
            hard=False,
        )

    def refine(
        self,
        measurements: Sequence[CodedApertureVoxelMeasurement],
    ) -> MultiVoxelTVResult:
        V = len(measurements)
        if V == 0:
            raise ValueError("multi-voxel refiner requires at least one voxel")

        dtype = measurements[0].frame_stack.dtype
        device = measurements[0].frame_stack.device

        t = self.params.to_tensors(dtype=dtype, device=str(device))
        lat = t["lattice"]
        P = t["P"]
        R = t["R"]

        # Per-voxel state:
        U_seeds = [v.U_seed.to(dtype=dtype, device=device) for v in measurements]
        zs = nn.Parameter(torch.tensor(
            [float(v.z_seed_um) for v in measurements],
            dtype=dtype, device=device,
        ))
        # One quaternion delta per voxel.
        deltas = nn.Parameter(torch.stack([
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
            for _ in range(V)
        ], dim=0))

        opt = torch.optim.Adam([
            {"params": [zs], "lr": self.lr_z},
            {"params": [deltas], "lr": self.lr_rot},
        ])

        targets = [v.frame_stack.to(dtype=dtype, device=device)
                   for v in measurements]
        offsets = [v.scan_offsets_um.to(dtype=dtype, device=device)
                   for v in measurements]

        def _loss_components():
            data = torch.zeros((), dtype=dtype, device=device)
            U_list: list[Tensor] = []
            for vi in range(V):
                dR = _quat_to_rotmat(deltas[vi])
                U = (dR @ U_seeds[vi])
                U_list.append(U)
                src_xyz = torch.stack([
                    torch.zeros((), dtype=dtype, device=device),
                    torch.zeros((), dtype=dtype, device=device),
                    zs[vi] * 1.0e-6,
                ])
                pred = self.model.forward_stack(
                    U.unsqueeze(0), lat, P, R,
                    coded_aperture=self.mask,
                    scan_offsets_um=offsets[vi],
                    source_xyz=src_xyz,
                    E_range=self.E_range,
                )
                data = data + (pred - targets[vi]).pow(2).mean()
            data = data / V

            # TV penalties — only meaningful when V > 1.
            tv_z = torch.zeros((), dtype=dtype, device=device)
            tv_U = torch.zeros((), dtype=dtype, device=device)
            if V > 1:
                if self.lambda_z != 0:
                    diffs = zs[1:] - zs[:-1]
                    tv_z = diffs.abs().sum()
                if self.lambda_U != 0:
                    # ‖U_{v+1} − U_v‖_F for each consecutive pair
                    Us = torch.stack(U_list, dim=0)            # (V, 3, 3)
                    deltaU = Us[1:] - Us[:-1]                  # (V-1, 3, 3)
                    tv_U = torch.linalg.matrix_norm(deltaU).sum()
            return data, tv_z, tv_U

        t0 = time.perf_counter()
        with torch.no_grad():
            d0, tz0, tu0 = _loss_components()
            initial_total = float(
                (d0 + self.lambda_z * tz0 + self.lambda_U * tu0).item()
            )

        for _step in range(self.n_steps):
            opt.zero_grad()
            d, tz, tu = _loss_components()
            L = d + self.lambda_z * tz + self.lambda_U * tu
            L.backward()
            opt.step()

        with torch.no_grad():
            d, tz, tu = _loss_components()
            data_final = float(d.item())
            tv_z_final = float(tz.item())
            tv_U_final = float(tu.item())

            results: list[DepthResolvedVoxelResult] = []
            for vi, m in enumerate(measurements):
                U_ref = (_quat_to_rotmat(deltas[vi]) @ U_seeds[vi]).detach()
                results.append(DepthResolvedVoxelResult(
                    voxel_index=m.voxel_index,
                    U_refined=U_ref,
                    z_um=float(zs[vi].detach().item()),
                    z_init_um=float(m.z_seed_um),
                    final_loss=data_final,        # shared data term
                    initial_loss=initial_total,
                    n_steps=self.n_steps,
                    dt_s=time.perf_counter() - t0,
                    metadata=dict(m.metadata),
                ))

        return MultiVoxelTVResult(
            per_voxel=results,
            lambda_z=self.lambda_z,
            lambda_U=self.lambda_U,
            final_loss_data=data_final,
            final_loss_tv=self.lambda_z * tv_z_final + self.lambda_U * tv_U_final,
            initial_loss_total=initial_total,
            n_steps=self.n_steps,
            dt_s=time.perf_counter() - t0,
            metadata={"n_voxels": V},
        )
