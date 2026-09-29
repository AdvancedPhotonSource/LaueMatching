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
from .harmonics import lowest_order_per_pixel

# Fixed RNG seed for the MC phantom samples. Shared by the fit and the Laplace
# posterior so the posterior is the curvature of the objective that was
# actually minimised (it used a different seed before).
FIXED_PRED_SEED = 0xC0FFEE
from .io import VoxelMeasurement


def affine_fit_residual(I_pred: Tensor, I_obs: Tensor):
    """Residual ``a * I_pred + b - I_obs`` with (a, b) the least-squares
    per-frame scale and background, solved in closed form.

    Returns ``(residual, a, b)``; ``a`` and ``b`` stay differentiable in
    ``I_pred`` (variable projection: the loss and the Laplace residual are
    both the fit with (a, b) profiled out). A flat prediction gets
    ``a = 0``, ``b = mean(I_obs)``.
    """
    p = I_pred.reshape(-1)
    o = I_obs.reshape(-1)
    pm, om = p.mean(), o.mean()
    dp = p - pm
    var = (dp * dp).sum()
    a = (dp * (o - om)).sum() / var.clamp_min(1e-300)
    a = torch.where(var > 0, a, torch.zeros_like(a))
    b = om - a * pm
    return (a * I_pred + b - I_obs), a, b


@dataclass
class VoxelODFResult:
    voxel_index: int
    U_mean: Tensor                              # (3, 3) recovered mean orientation
    sigma_U_deg: float                          # recovered isotropic mosaic spread (deg)
    sigma_U_full: Tensor                        # (3,) per-axis tangent covariance diagonal (rad)
    posterior_sigma_U_deg: float                # Laplace 1-σ on σ_U (deg)
    final_loss: float                           # converged MSE of a*render+b-obs
    initial_seed_index: int                     # which U_seed_list entry was used
    initial_seed_miso_deg: float                # miso (params.sg_num) between final and initial seed
    n_steps: int
    dt_s: float
    metadata: dict = field(default_factory=dict)
    # Full Laplace posterior over (3 log-diag, 3 off-diag) orientation-spread
    # Cholesky entries when compute_posterior=True and it could be computed.
    # posterior_sigma_U_deg is a summary of it; read posterior.eigvals,
    # cond_number, rank_eff and is_positive_definite before trusting it.
    posterior: Optional[LaplacePosterior] = None
    # (3, 3) fitted body-frame tangent covariance (rad^2). sigma_U_deg and
    # sigma_U_full are summaries of it; project it on the directions the
    # detector sees (see the VoxelODFRefiner identifiability warning).
    orient_cov: Optional[Tensor] = None


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
        Initial mosaic spread for the refined model. Not a neutral choice:
        in the tangent directions the image does not constrain (see the
        identifiability warning below) the fit keeps, or drifts away from,
        this value, so a start well above the truth reads high and one
        well below reads low.
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

    Warnings
    --------
    **The full 3-D orientation spread is not always identifiable, and
    ``sigma_U_deg`` (the RMS over the three tangent axes) then depends on
    ``sigma_init_deg``.** A Laue spot moves only with the in-plane part of a
    rotation: per reflection the image sees the 2x2 covariance
    ``J Sigma J^T`` (``J`` = d(spot px)/d(tangent rotation)), and rotation
    about the reflection's own normal does not move it at all. So:

    * One reflection on the detector: the spread about its normal is
      unobservable (3 of the 6 covariance entries). Measured
      (96x64, Ni, E 5-12 keV, truth 0.3 deg isotropic, M_render 32,
      300 steps): from init 0.6 the in-plane stds come back 0.318 / 0.281
      deg but the std about the normal drifts to 1.81 deg, so
      ``sigma_U_deg`` = 1.07; from init 0.15 it lands at 0.308, by chance
      (the unseen std drifted to 0.319). Adam is not rotation-invariant, so
      it moves along the flat direction instead of holding the init.
    * Several reflections on a small detector: one tangent direction (roughly
      rotation about the mean scattering vector) moves every spot by much
      less than the PSF, and its spread is weakly determined. Measured
      (128x128, 5 reflections, psf 1 px, weak direction 1.95 px/deg vs
      8.35 and 5.73): after 1000 steps (M_render 128) the two well-seen
      directions give 0.27-0.32 deg from both inits, the weak one 0.026
      (init 0.15) vs 0.328 (init 0.6); ``sigma_U_deg`` 0.247 vs 0.307.
      More steps do not remove it (M_render 32: 0.038 vs 0.444 at 3000).

    The loss itself is right: along an isotropic spread it is minimised at
    the truth (0.3 deg for psf 0.5-2 px, M_render 32/128). Before quoting
    ``sigma_U_deg``, project ``orient_cov`` on the singular vectors of the
    stacked spot Jacobian and quote only directions whose displacement per
    degree times the spread is not small against ``psf_sigma``; or treat
    the spread as isotropic. Pinned by
    ``tests/test_voxel_odf_identifiability.py``.
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
            sg_num=params.sg_num,
            detector_rotation="rodrigues",
            strain_mode="voigt",
            energy_image=False,
            hard=False,
            reduce="sum",
        )

    @torch.no_grad()
    def seed_spot_intensity(self, U_seed: Tensor) -> Tensor:
        """(H,) per-reflection intensity used by the fit: 1 for the
        lowest-order in-band reflection of each seed pixel, 0 for the other
        harmonics sharing it (and for reflections off the detector / out of
        band at the seed). Same grouping as ``MultiGrainVoxelRefiner``
        (``realdata.harmonics``); without it an (hhh) family rendered n times
        brighter than a single reflection.
        """
        U = U_seed.to(self.device, dtype=torch.float64).reshape(1, 3, 3)
        t = self.tensors
        _, aux = self.model(U, t["lattice"], t["P"], t["R"],
                            strain=torch.zeros(1, 6, dtype=torch.float64,
                                               device=self.device),
                            E_range=self.E_range, return_aux=True)
        Nx, Ny = t["n_pix"]
        cx = aux.px.round().long().clamp(0, Nx - 1)
        cy = aux.py.round().long().clamp(0, Ny - 1)
        keep_h = (aux.mask > 0.5).nonzero(as_tuple=False).reshape(-1)
        psi = torch.zeros(self.hkls.shape[0], dtype=torch.float64, device=self.device)
        idx = lowest_order_per_pixel(cx, cy, keep_h, self.hkls)
        if idx:
            psi[torch.tensor(idx, device=self.device)] = 1.0
        return psi

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
        # Harmonics deduplicated at the seed (fixed for the whole fit).
        psi = self.seed_spot_intensity(U_seed)

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
        last_ab = (float("nan"), float("nan"))
        for step in range(self.n_steps):
            opt.zero_grad()
            g = torch.Generator().manual_seed(FIXED_PRED_SEED)
            I_pred = voxel.render(self.model,
                                  self.tensors["lattice"],
                                  self.tensors["P"],
                                  self.tensors["R"],
                                  M=self.M_render, generator=g,
                                  E_range=self.E_range, per_spot_intensity=psi)
            # Per-frame scale and background (a, b) profiled out in closed
            # form: the spread no longer depends on the counts or pedestal.
            resid, a_fit, b_fit = affine_fit_residual(I_pred, I_obs)
            loss = (resid ** 2).mean()
            loss.backward()
            opt.step()
            last_loss = loss.item()
            last_ab = (float(a_fit.detach()), float(b_fit.detach()))

        with torch.no_grad():
            U_mean = voxel.orient.mean().detach()
            orient_cov = voxel.orient.covariance().detach()
            cov_diag = orient_cov.diag()
            sigma_U_full = cov_diag.sqrt()
            sigma_U_deg = math.degrees(math.sqrt(cov_diag.mean().item()))

        # Misorientation between seed and final mean, under the crystal's
        # own point group (was hard-wired cubic).
        from .. import symmetry
        miso_seed = symmetry.misorientation_deg(
            U_mean.unsqueeze(0), U_seed.unsqueeze(0), self.params.sg_num,
            lattice=self.params.lattice).item()

        # Laplace posterior on σ_U (only the Cholesky entries are free).
        posterior_sigma_U_deg = float("nan")
        posterior = None
        metadata = dict(measurement.metadata)
        # (a, b) at the last step: I_obs ~ a * render + b.
        metadata["intensity_scale"], metadata["intensity_offset"] = last_ab
        if self.compute_posterior:
            posterior_sigma_U_deg, posterior, err = self._laplace_on_sigma(
                voxel, I_obs, psi)
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
            orient_cov=orient_cov,
        )

    def _laplace_on_sigma(
        self,
        voxel: IndependentVoxelDistribution,
        I_obs: Tensor,
        psi: Tensor,
    ):
        """Laplace posterior over the 6 orientation-spread Cholesky entries.

        Returns ``(posterior_sigma_U_deg, posterior, error)``. The residual
        replays the fit exactly: same seed (``FIXED_PRED_SEED``), same draw
        order as ``IndependentVoxelDistribution.sample`` (orientation, then the
        frozen strain), same energy band, same harmonic-deduplicated
        per-reflection intensity, and the same closed-form per-frame scale and
        background (a, b) profiled out of the residual. The curvature scale is
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
                                E_range=self.E_range,
                                per_spot_intensity=psi.unsqueeze(0).expand(M, -1))
            # Same (a, b)-profiled residual as the fit.
            return affine_fit_residual(I_pred, I_obs)[0]

        try:
            posterior = laplace_posterior_from_residuals(residual_fn, theta_map)
        except torch.linalg.LinAlgError as exc:
            return float("nan"), None, f"LinAlgError: {exc}"
        sigma_orient_rad = float(log_diag.exp().mean().item())
        posterior_sigma_log_diag = float(posterior.sigma[:3].mean().item())
        return (math.degrees(sigma_orient_rad * posterior_sigma_log_diag),
                posterior, None)
