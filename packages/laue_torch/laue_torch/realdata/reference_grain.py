"""Design #5: reference-grain parallax depth recovery.

Single-image, no-mask depth refinement using a *known-position*
reference single crystal (e.g.\\ a thin Si membrane at z = 0) co-loaded
with the sample.  The reference's Laue pattern is at a known
(orientation, depth); the sample's pattern is offset on the detector
by the parallax through the lab-detector geometry.  Fitting the
sample's depth jointly against the known reference resolves the
absolute z-scale that a single Laue image otherwise cannot pin.

Best paired with a small-pixel photon-counting detector (Eiger2 4M
CdTe at 75 µm).  Depth precision is approximately
``pixel_size_um · D / (kf_x / kf_z) / centroid_accuracy_fraction``
— so ~10 µm precision for a 200 mm working distance + 1/10-pixel
centroiding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math
import time

import torch
from torch import Tensor, nn

from midas_stress.orientation import quat_to_orient_mat

from ..forward import LaueForwardModel
from ..io import LaueParams


def _quat_to_rotmat(q: Tensor) -> Tensor:
    q = q / torch.linalg.norm(q).clamp_min(1e-30)
    return quat_to_orient_mat(q).reshape(3, 3)


@dataclass
class TwoSourceMeasurement:
    """A single Laue exposure containing reference + sample spots."""

    image: Tensor                       # (Nx, Ny) — single-frame Laue image
    U_reference: Tensor                 # (3, 3) — known reference orientation
    z_reference_um: float               # known reference depth (often 0)
    U_sample_seed: Tensor               # (3, 3) — indexer seed for sample
    z_sample_seed_um: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ReferenceGrainResult:
    """Outcome of a reference-grain parallax refinement."""

    U_sample_refined: Tensor
    z_sample_um: float
    z_sample_init_um: float
    final_loss: float
    initial_loss: float
    n_steps: int
    dt_s: float
    metadata: dict = field(default_factory=dict)


class ReferenceGrainParallaxRefiner:
    """Joint sample-vs-reference depth recovery from a single Laue image.

    The forward model renders the *sum* of the reference and sample
    Laue patterns.  Adam refines the sample's ``(z, U, ε)`` against
    the observed combined image while holding the reference's
    ``(z_ref, U_ref)`` fixed at their calibrated values.

    Parameters
    ----------
    params, hkls
        Standard ``LaueParams`` + reflection list shared between
        reference and sample (assumes both are the same crystal type;
        for cross-material experiments, the forward model accepts a
        second hkl list — extend later).
    n_steps, lr_z, lr_rot
        Adam knobs.  See :class:`DepthResolvedVoxelRefiner` for the
        learning-rate scale conventions.
    weight_reference
        Relative intensity of the reference Laue pattern in the
        rendered sum.  Defaults to 1.0 (sample and reference equal
        weight).  Reduce when the reference is much thinner / weaker
        than the sample.
    """

    def __init__(
        self,
        params: LaueParams,
        *,
        hkls: Tensor,
        n_steps: int = 250,
        lr_z: float = 0.01,
        lr_rot: float = 1e-3,
        weight_reference: float = 1.0,
        E_range: Optional[tuple[float, float]] = None,
    ):
        self.params = params
        self.hkls = hkls
        self.n_steps = int(n_steps)
        self.lr_z = float(lr_z)
        self.lr_rot = float(lr_rot)
        self.weight_reference = float(weight_reference)
        self.E_range = E_range or (params.E_lo, params.E_hi)
        self.model = LaueForwardModel(
            hkls=hkls,
            n_pix=(params.n_pix_x, params.n_pix_y),
            px_size=(params.px_x, params.px_y),
            psf_sigma=params.psf_sigma,
            rotation="matrix",
            detector_rotation="rodrigues",
            strain_mode="none",
            hard=False,
        )

    def refine(self, m: TwoSourceMeasurement) -> ReferenceGrainResult:
        dtype = m.image.dtype
        device = m.image.device

        t = self.params.to_tensors(dtype=dtype, device=str(device))
        lat = t["lattice"]
        P = t["P"]
        R = t["R"]

        U_ref = m.U_reference.to(dtype=dtype, device=device)
        U_seed = m.U_sample_seed.to(dtype=dtype, device=device)
        target = m.image.to(dtype=dtype, device=device)
        active = (target.abs() > 1e-9).to(dtype)
        n_active = active.sum().clamp_min(1.0)

        z_um = nn.Parameter(
            torch.tensor(float(m.z_sample_seed_um), dtype=dtype, device=device)
        )
        delta_q = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
        )
        opt = torch.optim.Adam([
            {"params": [z_um], "lr": self.lr_z},
            {"params": [delta_q], "lr": self.lr_rot},
        ])

        def _render_one(U_mat: Tensor, z_um_val: Tensor, weight: float) -> Tensor:
            src = torch.stack([
                torch.zeros((), dtype=dtype, device=device),
                torch.zeros((), dtype=dtype, device=device),
                z_um_val * 1e-6,
            ])
            img = self.model(
                U_mat.unsqueeze(0), lat, P, R,
                source_xyz=src,
                E_range=self.E_range,
            )
            return weight * img

        def _loss() -> Tensor:
            U_sample = _quat_to_rotmat(delta_q) @ U_seed
            img_ref = _render_one(
                U_ref,
                torch.tensor(m.z_reference_um, dtype=dtype, device=device),
                self.weight_reference,
            )
            img_sample = _render_one(U_sample, z_um, 1.0)
            pred = img_ref + img_sample
            return (pred - target).pow(2).sum() / n_active

        t0 = time.perf_counter()
        with torch.no_grad():
            initial_loss = float(_loss().item())
        for _ in range(self.n_steps):
            opt.zero_grad()
            L = _loss()
            L.backward()
            opt.step()
        with torch.no_grad():
            final_loss = float(_loss().item())
            U_sample_refined = (_quat_to_rotmat(delta_q) @ U_seed).detach().clone()
            z_final = float(z_um.detach().item())

        return ReferenceGrainResult(
            U_sample_refined=U_sample_refined,
            z_sample_um=z_final,
            z_sample_init_um=float(m.z_sample_seed_um),
            final_loss=final_loss,
            initial_loss=initial_loss,
            n_steps=self.n_steps,
            dt_s=time.perf_counter() - t0,
            metadata=dict(m.metadata),
        )
