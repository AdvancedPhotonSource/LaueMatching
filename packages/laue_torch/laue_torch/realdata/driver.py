"""Per-voxel ODF refinement driver.

Wraps the synthetic experiments' refinement pipeline (Adam +
multi-scale annealing + Laplace posterior) for use on real LaueMatching
output.

Typical usage::

    refiner = VoxelODFRefiner(
        params=parse_params("simulation/params_sim.txt"),
        sigma_init_deg=1.0,
    )
    for voxel in LaueScanLoader("/path/to/scan/"):
        result = refiner.refine(voxel)
        # result.U_mean, result.sigma_U_deg, result.posterior_sigma_U_deg, ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math
import time

import numpy as np
import torch
from torch import Tensor

from ..distributions import (
    GaussianStrain,
    IndependentVoxelDistribution,
    TangentGaussianSO3,
)
from ..forward import LaueForwardModel
from ..io import LaueParams, generate_hkls
from ..uncertainty import laplace_posterior
from .io import VoxelMeasurement


@dataclass
class VoxelODFResult:
    voxel_index: int
    U_mean: Tensor                              # (3, 3) recovered mean orientation
    sigma_U_deg: float                          # recovered isotropic mosaic spread (deg)
    sigma_U_full: Tensor                        # (3,) per-axis tangent covariance diagonal (rad)
    posterior_sigma_U_deg: float                # Laplace 1-σ on σ_U (deg)
    final_loss: float                           # converged image-MSE
    initial_seed_index: int                     # which U_seed_list entry was used
    initial_seed_miso_deg: float                # cubic miso between final and initial seed
    n_steps: int
    dt_s: float
    metadata: dict = field(default_factory=dict)


class VoxelODFRefiner:
    """Per-voxel ODF refinement using a tangent-Gaussian SO(3) model.

    The refiner takes a :class:`VoxelMeasurement` (image + indexer
    seed orientations) and returns a :class:`VoxelODFResult` containing
    the recovered ODF parameters and Laplace posterior.

    Parameters
    ----------
    params : :class:`LaueParams`
        Geometry and lattice parameters parsed from the LaueMatching
        config file.
    sigma_init_deg : float
        Initial mosaic spread for the refined model.  Should be a few
        times larger than the expected truth spread; 1° is a sensible
        default for typical samples.
    psf_sigma : float
        **Measured instrument resolution** in pixels — the detector+geometry
        point-spread width from a pristine single-crystal standard (zero
        intrinsic orientation spread).  This is a *fixed* input that the ODF
        deconvolution divides out, so recovered mosaic (``sigma_U``) reflects
        the material, not the instrument.  Load it from a beam-time calibration
        artefact with :func:`laue_torch.spectrum.load_instrument_psf_sigma`.
        NOTE: this is *not* the LaueMatching ``SimulationSmoothingWidth``
        (a rendering blur used during indexing); the ``params.psf_sigma``
        fallback (from that key) is only a placeholder.  For the 34-ID-E
        2023-03-28 setup the measured value is ~1.06 px (≈ 0.026° / 1.6 arcmin).
    n_steps : int
        Adam steps for ODF refinement.
    M_render : int
        Monte-Carlo phantom samples per render.  **Important for real
        data**: the synthetic experiments use *common-z*
        reparameterisation (shared standard-normal samples between
        truth and pred at every step) to eliminate MC-noise variance
        bias.  With a *fixed* observed image (real data), common-z is
        not available.  Instead we use a fixed RNG seed for pred
        renders so the gradient is deterministic, and we recommend
        ``M_render`` $\\geq$ 128 to drive the MC noise floor below
        pixel noise.  Smaller ``M_render`` will bias the recovered
        ``sigma_U`` upward.
    refine_mean : bool
        If True, also refine the mean orientation alongside spread.
        Use only if the seed is not very precise; otherwise freeze
        the seed-supplied mean.
    """

    def __init__(
        self,
        params: LaueParams,
        *,
        sigma_init_deg: float = 1.0,
        psf_sigma: Optional[float] = None,
        n_steps: int = 500,
        M_render: int = 128,
        refine_mean: bool = False,
        compute_posterior: bool = True,
        device: str = "cpu",
    ):
        self.params = params
        self.sigma_init_deg = sigma_init_deg
        self.psf_sigma = psf_sigma if psf_sigma is not None else params.psf_sigma
        self.n_steps = n_steps
        self.M_render = M_render
        self.refine_mean = refine_mean
        self.compute_posterior = compute_posterior
        self.device = device

        # Build the forward model once (HKLs and detector geometry are shared).
        self.hkls = generate_hkls(params.sg_num, params.lattice, params.E_hi)
        self.tensors = params.to_tensors(dtype=torch.float64, device=device)
        self.model = LaueForwardModel(
            hkls=self.hkls.to(device),
            n_pix=self.tensors["n_pix"],
            px_size=self.tensors["px_size"],
            psf_sigma=self.psf_sigma,
            rotation="matrix",
            detector_rotation="rodrigues",
            strain_mode="voigt",
            energy_image=False,
            hard=False,
            reduce="sum",
        )

    def _build_voxel(self, U_init: Tensor) -> IndependentVoxelDistribution:
        orient = TangentGaussianSO3(
            U_init=U_init,
            sigma_init=math.radians(self.sigma_init_deg),
        )
        strain = GaussianStrain(sigma_init=1e-6)
        v = IndependentVoxelDistribution(orient, strain)
        # Strain spread is invisible to position-only data; freeze.
        v.strain.cov.log_diag.requires_grad_(False)
        v.strain.cov.off_diag.requires_grad_(False)
        v.strain.mean.requires_grad_(False)
        if not self.refine_mean:
            v.orient.mean_d6.requires_grad_(False)
        return v

    def refine(self, measurement: VoxelMeasurement) -> VoxelODFResult:
        t0 = time.time()
        if measurement.U_seed_list.shape[0] == 0:
            raise ValueError(
                f"voxel {measurement.voxel_index}: no seed orientations")
        U_seed = measurement.U_seed_list[0].to(self.device)        # take first solution
        I_obs = measurement.image.to(self.device)

        voxel = self._build_voxel(U_seed)

        # Optimiser groups.
        groups = [
            {"params": [voxel.orient.cov.log_diag], "lr": 5e-3},
            {"params": [voxel.orient.cov.off_diag], "lr": 5e-3},
        ]
        if self.refine_mean:
            groups.append({"params": [voxel.orient.mean_d6], "lr": 1e-3})
        opt = torch.optim.Adam(groups)

        # Use a *fixed* RNG seed for pred renders across all Adam steps.
        # When the truth observation is *fixed* (real data — there is no
        # second render to share standard-normal samples with), this is
        # the right substitute for the common-z trick used in synthetic
        # experiments: the rendered pred image is then a *deterministic*
        # function of θ, so the gradient is unbiased and Adam converges
        # to a true MAP.  Without this fix, fresh seeds per step inject
        # noise that Adam reduces by inflating Σ_orient, biasing the
        # recovered mosaic spread up.
        FIXED_PRED_SEED = 0xC0FFEE
        last_loss = float("nan")
        for step in range(self.n_steps):
            opt.zero_grad()
            g = torch.Generator().manual_seed(FIXED_PRED_SEED)
            I_pred = voxel.render(self.model,
                                  self.tensors["lattice"],
                                  self.tensors["P"],
                                  self.tensors["R"],
                                  M=self.M_render, generator=g)
            loss = ((I_pred - I_obs) ** 2).mean()
            loss.backward()
            opt.step()
            last_loss = loss.item()

        with torch.no_grad():
            U_mean = voxel.orient.mean().detach()
            cov_diag = voxel.orient.covariance().diag().detach()
            sigma_U_full = cov_diag.sqrt()
            sigma_U_deg = math.degrees(math.sqrt(cov_diag.mean().item()))

        # Cubic misorientation between seed and final mean.
        from ..symmetry import cubic_misorientation_deg
        miso_seed = cubic_misorientation_deg(
            U_mean.unsqueeze(0), U_seed.unsqueeze(0)).item()

        # Laplace posterior on σ_U (only the Cholesky entries are free).
        posterior_sigma_U_deg = float("nan")
        if self.compute_posterior:
            posterior_sigma_U_deg = self._laplace_on_sigma(voxel, I_obs,
                                                          last_loss)

        return VoxelODFResult(
            voxel_index=measurement.voxel_index,
            U_mean=U_mean,
            sigma_U_deg=sigma_U_deg,
            sigma_U_full=sigma_U_full,
            posterior_sigma_U_deg=posterior_sigma_U_deg,
            final_loss=last_loss,
            initial_seed_index=0,
            initial_seed_miso_deg=miso_seed,
            n_steps=self.n_steps,
            dt_s=time.time() - t0,
            metadata=measurement.metadata,
        )

    def _laplace_on_sigma(
        self,
        voxel: IndependentVoxelDistribution,
        I_obs: Tensor,
        map_loss: float,
    ) -> float:
        """Compute the Laplace posterior on σ_U (averaged over the 3
        Cholesky entries of the orientation covariance)."""
        # Pack the converged Cholesky parameters into a flat vector.
        log_diag = voxel.orient.cov.log_diag.detach().clone()
        off_diag = voxel.orient.cov.off_diag.detach().clone()
        theta_map = torch.cat([log_diag, off_diag])

        from ..geometry import rodrigues_to_matrix, sixd_to_matrix
        U_mean = sixd_to_matrix(voxel.orient.mean_d6.detach())
        fixed_seed = 0xABCDEF

        def loss_fn_flat(theta: Tensor) -> Tensor:
            cov_log = theta[:3]
            cov_off = theta[3:6]
            L = torch.diag(cov_log.exp())
            tril_i, tril_j = voxel.orient.cov.tril_idx
            L = L.clone()
            L[tril_i, tril_j] = cov_off
            g = torch.Generator().manual_seed(fixed_seed)
            z = torch.randn(self.M_render, 3, dtype=theta.dtype, generator=g)
            delta = z @ L.T
            U_pert = rodrigues_to_matrix(delta)
            U_samples = U_mean.unsqueeze(0) @ U_pert
            eps = torch.zeros(self.M_render, 6, dtype=theta.dtype)
            weights = torch.full((self.M_render,), 1.0 / self.M_render, dtype=theta.dtype)
            I_pred = self.model(U_samples,
                                self.tensors["lattice"],
                                self.tensors["P"],
                                self.tensors["R"],
                                strain=eps, weights=weights)
            return ((I_pred - I_obs) ** 2).mean()

        try:
            posterior = laplace_posterior(
                loss_fn_flat, theta_map,
                noise_variance=max(map_loss, 1e-12),
            )
            sigma_orient_rad = float(log_diag.exp().mean().item())
            posterior_sigma_log_diag = posterior.sigma[:3].mean().item()
            posterior_sigma_orient_rad = sigma_orient_rad * posterior_sigma_log_diag
            return math.degrees(posterior_sigma_orient_rad)
        except Exception as exc:
            return float("nan")
