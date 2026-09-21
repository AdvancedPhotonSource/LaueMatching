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
from ..io import LaueParams, experiment_band, generate_hkls, to_model_layout
from ..uncertainty import LaplacePosterior, laplace_posterior_from_residuals

# Fixed RNG seed for the MC phantom samples. Shared by the fit and the Laplace
# posterior so the posterior is the curvature of the objective that was
# actually minimised (it used a different seed before).
FIXED_PRED_SEED = 0xC0FFEE
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
    # Full Laplace posterior over (3 log-diag, 3 off-diag) orientation-spread
    # Cholesky entries when compute_posterior=True and it could be computed.
    # posterior_sigma_U_deg is a summary of it; read posterior.eigvals,
    # cond_number, rank_eff and is_positive_definite before trusting it.
    posterior: Optional[LaplacePosterior] = None


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

        # Render in the experiment's band (raises if params has none; no
        # (5, 30) keV fallback). Used by the fit and the posterior.
        self.E_range = experiment_band(params)
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
        # AXIS ORDER: the loader hands over the frame as stored (a real frame is
        # image[row, col]); the forward model renders img[X, Y]. This is the one
        # place the conversion happens (see realdata/io.py); it is shape-checked
        # so a wrong declared layout fails on a non-square detector.
        I_obs = to_model_layout(
            measurement.image.to(self.device, dtype=torch.float64),
            measurement.axis_order,          # None raises: no guessing
            self.tensors["n_pix"])

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
        last_loss = float("nan")
        for step in range(self.n_steps):
            opt.zero_grad()
            g = torch.Generator().manual_seed(FIXED_PRED_SEED)
            I_pred = voxel.render(self.model,
                                  self.tensors["lattice"],
                                  self.tensors["P"],
                                  self.tensors["R"],
                                  M=self.M_render, generator=g,
                                  E_range=self.E_range)
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
        posterior = None
        metadata = dict(measurement.metadata)
        if self.compute_posterior:
            posterior_sigma_U_deg, posterior, err = self._laplace_on_sigma(voxel, I_obs)
            if err is not None:
                metadata["posterior_error"] = err
            # The posterior covers only the spread; the mean orientation is
            # held at its value (the refined mean if refine_mean, else the
            # seed). Same key as MultiGrainVoxelRefiner. Orientation and
            # strain/spread are coupled, so this understates uncertainty.
            metadata["posterior_conditional_on_fixed_means"] = True
            metadata["posterior_mean_source"] = ("fitted" if self.refine_mean
                                                 else "seed")

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
            metadata=metadata,
            posterior=posterior,
        )

    def _laplace_on_sigma(
        self,
        voxel: IndependentVoxelDistribution,
        I_obs: Tensor,
    ):
        """Laplace posterior over the 6 orientation-spread Cholesky entries.

        Returns ``(posterior_sigma_U_deg, posterior, error)``. The residual
        replays the fit exactly: same seed (``FIXED_PRED_SEED``), same draw
        order as ``IndependentVoxelDistribution.sample`` (orientation, then the
        frozen strain), same energy band. The curvature scale is
        :func:`laue_torch.uncertainty.laplace_posterior_from_residuals`
        (``0.5 * SSR`` with the plug-in per-pixel noise variance), the same as
        ``MultiGrainVoxelRefiner``.

        ``posterior_sigma_U_deg`` = sigma_U times the mean posterior std of the
        3 log-diagonal entries (delta method). It is NaN when those entries
        have no valid width (non-positive-definite Hessian; see
        ``posterior.is_positive_definite`` / ``n_negative_eigvals``). Only a
        ``torch.linalg.LinAlgError`` from the eigen/pinv step is caught (it is
        returned as ``error``); anything else propagates.
        """
        log_diag = voxel.orient.cov.log_diag.detach().clone()
        off_diag = voxel.orient.cov.off_diag.detach().clone()
        theta_map = torch.cat([log_diag, off_diag])

        from ..geometry import rodrigues_to_matrix
        U_mean = voxel.orient.mean().detach()
        L_strain = voxel.strain.cov.L().detach()
        eps_mean = voxel.strain.mean.detach()
        tril_i, tril_j = voxel.orient.cov.tril_idx
        M = self.M_render
        weights = torch.full((M,), 1.0 / M, dtype=theta_map.dtype,
                             device=theta_map.device)

        def residual_fn(theta: Tensor) -> Tensor:
            L = torch.diag(theta[:3].exp()).clone()
            L[tril_i, tril_j] = theta[3:6]
            g = torch.Generator().manual_seed(FIXED_PRED_SEED)
            z = torch.randn(M, 3, dtype=theta.dtype, device=theta.device, generator=g)
            U_samples = U_mean.unsqueeze(0) @ rodrigues_to_matrix(z @ L.T)
            zs = torch.randn(M, L_strain.shape[0], dtype=theta.dtype,
                             device=theta.device, generator=g)
            eps = eps_mean.unsqueeze(0) + zs @ L_strain.T
            I_pred = self.model(U_samples,
                                self.tensors["lattice"],
                                self.tensors["P"],
                                self.tensors["R"],
                                strain=eps, weights=weights,
                                E_range=self.E_range)
            return I_pred - I_obs

        try:
            posterior = laplace_posterior_from_residuals(residual_fn, theta_map)
        except torch.linalg.LinAlgError as exc:
            return float("nan"), None, f"LinAlgError: {exc}"
        sigma_orient_rad = float(log_diag.exp().mean().item())
        posterior_sigma_log_diag = float(posterior.sigma[:3].mean().item())
        return (math.degrees(sigma_orient_rad * posterior_sigma_log_diag),
                posterior, None)
