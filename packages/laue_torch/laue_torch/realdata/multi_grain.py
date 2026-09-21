"""Per-voxel multi-grain ODF refinement.

Real LaueMatching data: a single Laue exposure of an illuminated
voxel can carry signal from multiple grains.  This driver wraps the
existing ``MixtureOfTangentGaussianSO3`` /
``MixtureOfVoxelDistributions`` machinery in a use-friendly
front-end:

  * **Input**: one observed image plus ``K`` seed orientation
    matrices from the upstream indexer (e.g. from
    ``laue_postprocess.py``'s ``filtered_orientations`` group, or
    transcribed from a paper's published values).
  * **Output**: per-mode (``U_mean_k``, ``Σ_orient_k``, mixing
    weight ``π_k``); optionally per-mode strain (full Voigt-6 or
    deviatoric-5); optionally the Laplace posterior over the fitted
    parameters (``compute_posterior=True``).

Modes
-----

  * ``orient_only`` --- per-mode tangent Gaussian on SO(3); strain
    held at zero.  Use for FCC samples without expected strain
    (e.g. an undeformed reference).
  * ``strain_voigt`` --- per-mode strain Gaussian (Voigt-6 mean,
    spread frozen at zero per the paper's recommendation since
    position-only data can't see Σ_ε).  This is the EuAl2O4 case:
    each grain is a parent + characteristic strain.
  * ``strain_deviatoric`` --- 5-DOF trace-free strain mean in the
    ``geometry.deviatoric5_to_symmetric`` layout ``(e11, e22, e23, e13,
    e12)`` with ``e33 = -(e11 + e22)``.  The hydrostatic direction is
    not a parameter: in white-beam Laue a pure dilatation moves no spot
    (it only shifts the Bragg energy), so it is exactly unidentifiable
    from positions (``jointfit/footprint.py``, ``strain_jacobian``).
    Prefer this over ``strain_voigt``, whose 6th direction is that null.
    Orientation and deviatoric strain are still strongly coupled in
    white-beam Laue; read the posterior's eigenvalues, not a marginal
    sigma, before quoting a strain.

Axis order
----------

``refine`` requires ``axis_order``: ``"YX"`` for a frame in detector
layout ``image[row, col]`` (as the indexer and ``LaueScanLoader``
provide it), which it transposes to the forward model's ``img[X, Y]``,
or ``"XY"`` for a laue_torch render.  ``None`` raises; see
``laue_torch.io.to_model_layout`` and handbook invariant 38.

Note on M_render
----------------

For real measured data we cannot use common-z reparameterisation
(the observation is fixed, not regenerated).  Instead a fixed
RNG seed is used for pred renders so the gradient is deterministic;
``M_render`` should be ≥ 128 to keep MC noise below the pixel-noise
floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import math
import time

import numpy as np
import torch
from torch import Tensor

from ..distributions import (
    GaussianStrain,
    IndependentVoxelDistribution,
    MixtureOfTangentGaussianSO3,
    MixtureOfVoxelDistributions,
    TangentGaussianSO3,
)
from ..distributions import CholeskyCov
from ..forward import LaueForwardModel
from ..geometry import rodrigues_to_matrix
from ..io import LaueParams, experiment_band, generate_hkls, to_model_layout
from ..uncertainty import LaplacePosterior, laplace_posterior_from_residuals
from .driver import FIXED_PRED_SEED


def _dev5_to_voigt6(e5: Tensor) -> Tensor:
    """Deviatoric-5 ``(e11, e22, e23, e13, e12)`` -> trace-free Voigt-6
    ``(e11, e22, e33, e23, e13, e12)`` with ``e33 = -(e11 + e22)``.

    Same layout and tensor as ``geometry.deviatoric5_to_symmetric``; mapped to
    Voigt-6 only because the mixture renderers drive a ``strain_mode="voigt"``
    forward model.
    """
    e11, e22, e23, e13, e12 = e5.unbind(-1)
    return torch.stack([e11, e22, -(e11 + e22), e23, e13, e12], dim=-1)


class _DeviatoricGaussianStrain(torch.nn.Module):
    """Strain Gaussian restricted to the 5-D trace-free subspace.

    Drop-in for :class:`GaussianStrain` inside
    :class:`IndependentVoxelDistribution` (same ``mean`` / ``sample`` /
    ``covariance`` surface, Voigt-6 out), but the only free mean parameters
    are the 5 deviatoric components in ``mean5``; the hydrostatic direction
    cannot move.
    """

    def __init__(self, sigma_init: float = 1e-4,
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        self.mean5 = torch.nn.Parameter(torch.zeros(5, dtype=dtype))
        self.cov = CholeskyCov(5, init_scale=sigma_init, dtype=dtype)

    @property
    def mean(self) -> Tensor:
        return _dev5_to_voigt6(self.mean5)

    def sample(self, N: int, generator: Optional[torch.Generator] = None) -> Tensor:
        return _dev5_to_voigt6(self.mean5.unsqueeze(0)
                               + self.cov.sample(N, generator=generator))

    def covariance(self) -> Tensor:
        return self.cov.cov()


def _stratified_counts(M: int, K: int) -> list[int]:
    """Samples per mode, matching the mixtures' stratified ``sample``."""
    per_k, extra = M // K, M % K
    return [per_k + (1 if k < extra else 0) for k in range(K)]


def _per_sample_intensity(target_KH: Tensor, M: int) -> Tensor:
    """Expand a per-mode ``(K, H)`` target to the ``(M, H)`` per-sample
    ``per_spot_intensity`` in the mixtures' sample order (mode-major,
    modes with zero samples skipped)."""
    K = target_KH.shape[0]
    rows = [target_KH[k].unsqueeze(0).expand(m_k, -1)
            for k, m_k in enumerate(_stratified_counts(M, K)) if m_k > 0]
    return torch.cat(rows, dim=0)


@dataclass
class MultiGrainResult:
    """Recovered per-mode parameters for one voxel."""
    n_modes: int
    U_means: Tensor                      # (K, 3, 3) refined orientations
    sigma_U_deg: Tensor                  # (K,) per-mode mosaic spread (deg)
    pi: Tensor                           # (K,) mixing weights, sum 1
    eps_means: Optional[Tensor]          # (K, 6) Voigt strain or None;
                                         # trace-free for strain_deviatoric
    final_loss: float
    initial_seed_misos_deg: Tensor       # (K,) cubic miso between final and seed for each mode
    n_steps: int
    dt_s: float
    metadata: dict = field(default_factory=dict)
    # Laplace posterior at the optimum (compute_posterior=True), else None.
    # Read posterior.eigvals / cond_number / rank_eff / is_positive_definite
    # before any posterior.sigma entry: orientation spread and deviatoric
    # strain are strongly coupled in white-beam Laue, and a marginal sigma
    # on its own hides that. posterior_param_names labels posterior.theta.
    posterior: Optional[LaplacePosterior] = None
    posterior_param_names: Optional[list] = None


class MultiGrainVoxelRefiner:
    """ODF refinement for a single voxel containing K grains.

    Parameters
    ----------
    params : :class:`LaueParams`
        Geometry and lattice parameters.
    sigma_init_deg : float
        Initial mosaic spread for each mode.
    psf_sigma : float
        Geometric PSF in pixels.
    n_steps : int
        Adam iterations.
    M_render : int
        Monte-Carlo phantom samples per render.  Must be a multiple of
        ``K`` for clean stratification (otherwise a stride of
        ``M / K +/- 1`` is used internally).
    mode : str
        ``"orient_only"`` | ``"strain_voigt"`` | ``"strain_deviatoric"``.
    refine_means : bool
        If True, also refine each mode's mean orientation via 6-D
        rotation parameter.  Default False --- the indexer-supplied
        seed is held fixed.
    compute_posterior : bool
        If True, compute the Laplace posterior at the optimum
        (:func:`laue_torch.uncertainty.laplace_posterior`) over the
        fitted parameters and return it in ``MultiGrainResult.posterior``.
        With ``refine_means=False`` the means are not parameters, so the
        posterior is CONDITIONAL on them (``metadata
        ["posterior_conditional_on_fixed_means"]``); because orientation
        and strain are coupled, that understates strain uncertainty.
    """

    def __init__(
        self,
        params: LaueParams,
        *,
        sigma_init_deg: float = 1.0,
        psf_sigma: Optional[float] = None,
        psf_eta: float = 0.0,
        n_steps: int = 500,
        M_render: int = 128,
        mode: str = "orient_only",
        refine_means: bool = False,
        refine_psf: bool = False,
        refine_eta: bool = False,
        compute_posterior: bool = False,
        device: str = "cpu",
        optimizer: str = "adam",
    ):
        self.params = params
        self.sigma_init_deg = sigma_init_deg
        self.psf_sigma = psf_sigma if psf_sigma is not None else params.psf_sigma
        self.n_steps = n_steps
        self.M_render = M_render
        if mode not in ("orient_only", "strain_voigt", "strain_deviatoric"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.refine_means = refine_means
        self.compute_posterior = compute_posterior
        self.device = device
        if optimizer not in ("adam", "lbfgs"):
            raise ValueError(f"unknown optimizer {optimizer!r}; "
                             f"choose 'adam' or 'lbfgs'")
        self.optimizer = optimizer
        self.refine_psf = bool(refine_psf)
        self.refine_eta = bool(refine_eta)
        self.psf_eta = float(psf_eta)

        # Every render (target, fit, posterior) uses the experiment's band;
        # raises if params carries none (no (5, 30) keV fallback).
        self.E_range = experiment_band(params)
        self.hkls = generate_hkls(params.sg_num, params.lattice, params.E_hi)
        self.tensors = params.to_tensors(dtype=torch.float64, device=device)

        # Forward model: shared across all modes.  Voigt-6 for every strain
        # mode: "strain_deviatoric" constrains the MEAN to the trace-free
        # subspace (_DeviatoricGaussianStrain) and hands the model the
        # equivalent trace-free Voigt-6, identical to strain_mode="deviatoric"
        # on the 5-vector.
        self.model = LaueForwardModel(
            hkls=self.hkls.to(device),
            n_pix=self.tensors["n_pix"],
            px_size=self.tensors["px_size"],
            psf_sigma=self.psf_sigma,
            psf_eta=self.psf_eta,
            rotation="matrix",
            detector_rotation="rodrigues",
            strain_mode="voigt",
            energy_image=False,
            hard=False,
            reduce="sum",
        )

    def _strain_param(self, strain) -> Tensor:
        """The free strain-mean parameter: 5-vector (deviatoric) or 6 (Voigt)."""
        return strain.mean5 if self.mode == "strain_deviatoric" else strain.mean

    @torch.no_grad()
    def _compute_per_spot_target(self, mix, U_seed_list: Tensor, I_obs: Tensor
                                 ):
        """Returns ``(target_per_spot, patch_mask)``.

        - ``target_per_spot``: shape ``(K, H)``, one row PER MODE.
          ``target[k, h]`` is the peak amplitude (max of ``I_obs``) in a
          ``render_window×render_window`` patch around mode ``k``'s
          seed-predicted spot for reflection ``h``.  Used as the
          per-sample ``per_spot_intensity`` in the fit, so each mode's
          predicted peak amplitudes match the observed ones at the seed.
        - ``patch_mask``: shape ``(Nx, Ny)`` boolean tensor that is
          ``True`` inside the union of those W×W windows.  Used to
          restrict the loss to spot regions, ignoring the remainder of
          the image where post-background-subtraction residual is not a
          diffraction peak the model could fit.

        Three rules make the prediction match the observation in amplitude
        (each fixes a way the earlier version over-predicted):

        * PEAK, not window sum: the splat is peak-normalised (its peak is
          the intensity it is given), so a window-sum target rendered every
          spot ~2πσ² times too bright.  Same calibration as
          :meth:`_target_from_indexer_spots`.
        * ONE reflection per predicted pixel per mode: harmonics ((111),
          (222), ...) share a q-hat and so land on the same pixel; giving
          each the full observed amplitude predicted n× the observed peak.
          Only the lowest-order member (smallest |hkl|) of each pixel group
          gets the target; the rest get 0.  The group is the rounded seed
          pixel, which also merges distinct reflections that coincide on a
          pixel (the observed patch holds both, so one amplitude is right).
          Reflections that are near but not on the same pixel still see
          overlapping patches; that residual double-count is not removed.
        * PER MODE: each mode's reflection ``h`` lands at its own pixel, so
          the target is not shared (summed) across modes.

        Reflections off the detector or out of band at the seed (soft mask
        <= 0.5) get target 0; kept ones get ``peak / mask`` so that the
        render, which multiplies by the soft mask, reproduces the peak.

        Computed once at refinement start (not re-evaluated each step) so
        gradients through the optimisation are stable.  The seed
        orientation is accurate to a few pixels post-indexer refinement,
        so the W×W patches capture the full peak even when the converged
        orientation drifts slightly from the seed.
        """
        H = self.hkls.shape[0]
        K = U_seed_list.shape[0]
        Nx, Ny = self.tensors["n_pix"]
        W = self.model.render_window
        r = W // 2
        device = I_obs.device

        target = torch.zeros(K, H, dtype=torch.float64, device=device)
        patch_mask = torch.zeros(Nx, Ny, dtype=torch.bool, device=device)
        offsets = torch.arange(-r, r + 1, device=device, dtype=torch.long)
        I_obs_flat = I_obs.reshape(-1)
        eps_zero = torch.zeros(1, 6, dtype=torch.float64, device=device)
        weights_one = torch.ones(1, dtype=torch.float64, device=device)
        hkl_order = (self.hkls.to(device=device, dtype=torch.float64) ** 2).sum(-1)

        for k in range(K):
            U_k = U_seed_list[k:k + 1]  # (1, 3, 3)
            _, aux = self.model(U_k, self.tensors["lattice"],
                                self.tensors["P"], self.tensors["R"],
                                strain=eps_zero, weights=weights_one,
                                E_range=self.E_range, return_aux=True)
            cx = aux.px.detach().round().long().clamp(0, Nx - 1)
            cy = aux.py.detach().round().long().clamp(0, Ny - 1)
            tx = cx[:, None] + offsets[None, :]               # (H, W)
            ty = cy[:, None] + offsets[None, :]
            valid_x = (tx >= 0) & (tx < Nx)
            valid_y = (ty >= 0) & (ty < Ny)
            tx_c = tx.clamp(0, Nx - 1)
            ty_c = ty.clamp(0, Ny - 1)
            flat_idx = (tx_c[:, :, None] * Ny + ty_c[:, None, :])  # (H, W, W)
            valid = (valid_x[:, :, None] & valid_y[:, None, :])
            obs_tile = I_obs_flat[flat_idx.reshape(-1)].reshape(H, W, W)
            obs_tile = torch.where(valid, obs_tile,
                                   torch.full_like(obs_tile, -math.inf))
            patch_max = obs_tile.amax(dim=(1, 2)).clamp_min(0.0)   # (H,)
            mask_k = aux.mask.detach().reshape(H)
            keep_h = (mask_k > 0.5).nonzero(as_tuple=False).reshape(-1)

            # One reflection per rounded seed pixel: lowest order wins.
            if keep_h.numel() > 0:
                order = torch.argsort(hkl_order[keep_h], stable=True)
                seen = set()
                for h in keep_h[order].tolist():
                    key = (int(cx[h]), int(cy[h]))
                    if key in seen:
                        continue
                    seen.add(key)
                    # The render multiplies by the soft mask again, so divide
                    # it out (mask > 0.5 here): rendered peak = observed peak
                    # also for reflections near a band or detector edge.
                    target[k, h] = patch_max[h] / mask_k[h]

            # Build patch mask from valid HKLs only, expanded slightly so
            # the σ_U gradient has room to inflate without spilling out of
            # the loss region.
            if keep_h.numel() > 0:
                tx_full = tx_c[keep_h, :, None].expand(-1, W, W).reshape(-1)
                ty_full = ty_c[keep_h, None, :].expand(-1, W, W).reshape(-1)
                valid_keep = ((valid_x[keep_h, :, None] & valid_y[keep_h, None, :])
                              .reshape(-1))
                tx_v = tx_full[valid_keep]
                ty_v = ty_full[valid_keep]
                patch_mask[tx_v, ty_v] = True
        return target, patch_mask

    @torch.no_grad()
    def _target_from_indexer_spots(
        self,
        indexer_spots_hkl: Tensor,
        indexer_spots_xy: Tensor,
        indexer_spots_intensity: Tensor,
        I_obs: Tensor,
    ):
        """Build (target_per_spot, patch_mask) directly from the
        indexer's already-validated spot list.

        Parameters
        ----------
        indexer_spots_hkl : (S, 3) int tensor
            (h, k, l) of each indexer-confirmed spot.
        indexer_spots_xy : (S, 2) float tensor
            (X, Y) detector pixel coordinates of each indexer spot.
        indexer_spots_intensity : (S,) float tensor
            Per-spot integrated intensity from the indexer.
        I_obs : (Nx, Ny) tensor
            Background-subtracted observed image (used only for the
            patch_mask shape).

        Returns
        -------
        target_per_spot : (H,) tensor
            ``target_per_spot[h]`` = indexer intensity if our hkl list's
            ``h``-th row matches one of the indexer's confirmed spots,
            zero otherwise.
        patch_mask : (Nx, Ny) bool tensor
            True inside the W×W window around each indexer spot's
            (X, Y) pixel; loss is restricted to these pixels so we don't
            try to fit post-bg-subtraction residual that the indexer
            already classified as not-a-spot.
        """
        H = self.hkls.shape[0]
        S = indexer_spots_hkl.shape[0]
        Nx, Ny = self.tensors["n_pix"]
        W = self.model.render_window
        r = W // 2
        device = I_obs.device

        target = torch.zeros(H, dtype=torch.float64, device=device)
        patch_mask = torch.zeros(Nx, Ny, dtype=torch.bool, device=device)

        # Build (h,k,l) -> hkl-row-index lookup once.
        hkls_cpu = self.hkls.cpu().numpy().astype(int)
        from collections import defaultdict
        hkl_lookup = {tuple(row): i for i, row in enumerate(hkls_cpu.tolist())}

        spots_hkl_cpu = indexer_spots_hkl.cpu().numpy().astype(int)
        spots_inten_cpu = indexer_spots_intensity.cpu().numpy().astype(float)
        spots_xy_cpu = indexer_spots_xy.cpu().numpy().astype(int)

        # gaussian_splat produces an *unnormalised* 2-D Gaussian
        # (peak ≈ intensity, integral = intensity × 2πσ²).  So the
        # per-spot intensity that makes the predicted *peak amplitude*
        # match the observed peak amplitude is
        #     target[h] = max(I_obs in window around indexer (X, Y)).
        # The indexer's reported ``Intensity`` (column 10 of
        # /entry/results/spots in the RunImage layout, 11 in the stream
        # layout) is the value of the INDEXER'S processed image at the
        # truncated predicted pixel (``image[py * nrPxX + px]`` in
        # LaueMatchingHeaders.h), not a measurement on ``I_obs``, so it
        # is not used as the target here.  We deliberately use the
        # patch-max of ``I_obs`` as a peak-amplitude proxy -- this is the
        # calibration that makes the synthetic exp5l recovery converge
        # correctly and keeps the EuAl2O4 null-hypothesis recovery at the FWHM-derived bound.
        #
        # Coordinate convention: the LaueMatching indexer stores ``X`` as
        # the column index and ``Y`` as the row index of the cleaned image
        # in standard numpy layout.  ``I_obs`` here has *already been
        # transposed* in ``refine`` so it matches the forward model's
        # ``img[X, Y]`` convention; therefore we index it with X first
        # and Y second.
        n_matched = 0
        for s in range(S):
            h, k, l = spots_hkl_cpu[s].tolist()
            i = hkl_lookup.get((h, k, l), hkl_lookup.get((-h, -k, -l), -1))
            X, Y = spots_xy_cpu[s].tolist()
            x0, x1 = max(X - r, 0), min(X + r + 1, Nx)
            y0, y1 = max(Y - r, 0), min(Y + r + 1, Ny)
            if i >= 0:
                target[i] = float(I_obs[x0:x1, y0:y1].max().item())
                n_matched += 1
            patch_mask[x0:x1, y0:y1] = True

        if n_matched != S:
            print(f"  WARNING: {S - n_matched}/{S} indexer spots could not be "
                  "matched to the midas_hkls list (unexpected sign convention?).")
        return target, patch_mask

    def refine(
        self,
        image: Tensor,
        U_seed_list: Tensor,
        *,
        seed: int = FIXED_PRED_SEED,
        indexer_spots: Optional[dict] = None,
        axis_order: Optional[str] = None,
    ) -> MultiGrainResult:
        """Refine the K-grain mixture.

        ``U_seed_list``: (K, 3, 3) seed orientation matrices.
        ``image``: observed Laue pattern.  ``axis_order`` is REQUIRED
        (``None`` raises): ``"YX"`` for a real frame in detector layout
        ``image[row, col]`` = ``(NrPxY, NrPxX)`` (what ``LaueScanLoader``
        yields; pass ``voxel.axis_order``), ``"XY"`` for an image already
        in the forward model's ``img[X, Y]`` layout (a laue_torch render).
        The layout is shape-checked, so a wrong declaration fails on a
        non-square detector; on a square one only the declaration
        protects you.
        ``indexer_spots``: optional ``{"hkl", "xy", "intensity"}`` with
        ``xy`` = (X, Y) = (column, row); the target is then taken at
        those spots instead of at the seed predictions.  It is keyed by
        hkl only, so with K > 1 the same hkl of two grains shares one
        target (not per mode).
        """
        t0 = time.time()
        if U_seed_list.dim() != 3 or U_seed_list.shape[-2:] != (3, 3):
            raise ValueError(
                f"U_seed_list must be (K, 3, 3), got {tuple(U_seed_list.shape)}")
        K = U_seed_list.shape[0]
        # Convention reconciliation: gaussian_splat (and the C reference)
        # produce ``img[X_col, Y_row]`` (forward's "X" is the first axis,
        # contrary to standard numpy ``image[row, col] = image[Y, X]``).
        # RunImage.py writes the cleaned image with standard convention,
        # so we transpose the input here to align with the forward.  On a
        # square image (Nx=Ny=2048) this is invisible to existing parity
        # tests but critical for any pixel-wise comparison; the shape
        # check in to_model_layout catches it on a non-square one.
        I_obs = to_model_layout(image.to(self.device, dtype=torch.float64),
                                axis_order, self.tensors["n_pix"])
        U_seed_list = U_seed_list.to(self.device, dtype=torch.float64)

        # Build mixture model.  We construct the modules first, then move
        # the *whole module* onto the device (so all submodule parameters
        # and buffers share one device — CholeskyCov defaults to CPU
        # otherwise), and only THEN collect parameter references for the
        # optimiser.  Doing this in the other order leaves opt_param_groups
        # holding references to parameters that nn.Module.to() may have
        # replaced under us, in which case Adam steps on dead handles and
        # the loss stays exactly constant.
        if self.mode == "orient_only":
            mix = MixtureOfTangentGaussianSO3(
                U_inits=U_seed_list,
                sigma_init=math.radians(self.sigma_init_deg),
            )
            mix = mix.to(self.device)
            if not self.refine_means:
                for k in mix.kernels:
                    k.mean_d6.requires_grad_(False)
            opt_param_groups = [
                {"params": [k.cov.log_diag for k in mix.kernels], "lr": 5e-3},
                {"params": [k.cov.off_diag for k in mix.kernels], "lr": 5e-3},
                {"params": [mix.logits], "lr": 1e-2},
            ]
            if self.refine_means:
                opt_param_groups.append({
                    "params": [k.mean_d6 for k in mix.kernels], "lr": 1e-3,
                })
        elif self.mode in ("strain_voigt", "strain_deviatoric"):
            components = []
            for k in range(K):
                orient = TangentGaussianSO3(
                    U_init=U_seed_list[k],
                    sigma_init=math.radians(self.sigma_init_deg),
                )
                if self.mode == "strain_deviatoric":
                    strain = _DeviatoricGaussianStrain(sigma_init=1e-6)
                else:
                    strain = GaussianStrain(sigma_init=1e-6)
                components.append(IndependentVoxelDistribution(orient, strain))
            mix = MixtureOfVoxelDistributions(components)
            mix = mix.to(self.device)
            for c in mix.components:
                # Strain spread invisible to position-only data; freeze.
                c.strain.cov.log_diag.requires_grad_(False)
                c.strain.cov.off_diag.requires_grad_(False)
                if not self.refine_means:
                    c.orient.mean_d6.requires_grad_(False)
            opt_param_groups = [
                {"params": [c.orient.cov.log_diag for c in mix.components], "lr": 5e-3},
                {"params": [c.orient.cov.off_diag for c in mix.components], "lr": 5e-3},
                {"params": [self._strain_param(c.strain) for c in mix.components],
                 "lr": 1e-4},
                {"params": [mix.logits], "lr": 1e-2},
            ]
            if self.refine_means:
                opt_param_groups.append({
                    "params": [c.orient.mean_d6 for c in mix.components], "lr": 1e-3,
                })
        else:
            raise AssertionError(self.mode)

        # ── Diagnostic: I_obs scale ──────────────────────────────────
        print(f"  I_obs: shape={tuple(I_obs.shape)} sum={I_obs.sum().item():.3e} "
              f"max={I_obs.max().item():.3e} nonzero={(I_obs>0).sum().item()} "
              f"sq_mean={(I_obs**2).mean().item():.3e}", flush=True)

        # ── Per-spot target intensities ──────────────────────────────────
        # Two paths:
        # (1) indexer_spots given: use the upstream indexer's confirmed
        #     spot list directly.  ``target_per_spot[h]`` is the max of
        #     I_obs in the W×W window at the indexer's (X, Y) for the
        #     matching (h,k,l), zero everywhere else (NOT the indexer's
        #     reported intensity column; see _target_from_indexer_spots).
        #     ``patch_mask`` covers W×W around each indexer (X, Y).
        #     This is the cleanest signal — the indexer has already done
        #     the spot-vs-noise classification.
        # (2) fallback: per mode, the max of I_obs in the W×W window at
        #     each hkl's seed-predicted (px, py), one reflection per
        #     predicted pixel (see _compute_per_spot_target).  Vulnerable
        #     to confounding by post-bg residual when the obs image is
        #     noisy, so prefer path (1) when an indexer spot list is
        #     available.
        # The forward model gives every reflection an intrinsic intensity
        # of 1.  In real Laue data, |F_hkl|² varies by orders of magnitude
        # across the HKL list, and the sample-dependent absorption /
        # extinction further perturbs each spot.  A loss that compares a
        # uniform-intensity prediction to a wildly-varying observation
        # ends up dominated by amplitude mismatch and converges to
        # degenerate solutions (either I_pred≈0 with the raw MSE, or
        # σ_U→∞ with a single global LSQ scale).  Both failure modes were
        # observed before this fix.
        #
        # Solution: pre-compute a per-mode peak-amplitude target at the
        # seed-orientation predicted (px_h, py_h) and pass it as the
        # forward's per-sample per_spot_intensity.  Predicted peak
        # amplitudes then match observed peak amplitudes by construction at
        # the seed orientation; the remaining loss measures only the shape/
        # positional mismatch the optimizer is actually meant to fit
        # (mosaic spread, strain-induced position shift).
        if indexer_spots is not None:
            target_psi, patch_mask = self._target_from_indexer_spots(
                indexer_spots["hkl"].to(self.device),
                indexer_spots["xy"].to(self.device),
                indexer_spots["intensity"].to(self.device),
                I_obs,
            )
            target_psi = target_psi.unsqueeze(0).expand(K, -1)   # shared, see docstring
        else:
            target_psi, patch_mask = self._compute_per_spot_target(
                mix, U_seed_list, I_obs)
        # (K, H) per mode -> (M, H) per phantom sample, in the mixtures'
        # stratified sample order.
        psi_per_sample = _per_sample_intensity(target_psi, self.M_render)
        patch_mask_f = patch_mask.to(torch.float64)
        patch_mask_count = patch_mask_f.sum().clamp_min(1.0)
        print(f"  target_psi: shape={tuple(target_psi.shape)} "
              f"nonzero={(target_psi>0).sum().item()} "
              f"max={target_psi.max().item():.3e} "
              f"mean_nonzero={target_psi[target_psi>0].mean().item() if (target_psi>0).any() else 0:.3e}",
              flush=True)
        print(f"  patch_mask: True pixels={int(patch_mask_count.item())} "
              f"({patch_mask_count.item() / (patch_mask.numel()) * 100:.2f}% of image)",
              flush=True)

        pred_seed = seed
        last_loss = float("nan")
        log_every = max(1, self.n_steps // 20)
        gen_device = "cpu" if str(self.device).startswith("cpu") else self.device
        t_step0 = time.time()

        # Optional refinable PSF.  Parameterised as ``log(psf_sigma)`` so the
        # exponentiated value is positive by construction.  Initialised at the
        # configured PSF; included in opt_param_groups when refine_psf=True.
        if self.refine_psf:
            log_psf = torch.nn.Parameter(
                torch.tensor(math.log(self.psf_sigma),
                             dtype=torch.float64, device=self.device))
            opt_param_groups.append({"params": [log_psf], "lr": 1e-2})
        else:
            log_psf = None

        # Optional refinable pseudo-Voigt mixing fraction η ∈ [0, 1],
        # parameterised as a logit so sigmoid(logit) ∈ (0, 1) by
        # construction.  Initialised at the configured psf_eta.
        if self.refine_eta:
            eta_init = max(min(self.psf_eta, 0.999), 0.001)  # avoid logit at boundary
            logit_eta = torch.nn.Parameter(
                torch.tensor(math.log(eta_init / (1.0 - eta_init)),
                             dtype=torch.float64, device=self.device))
            opt_param_groups.append({"params": [logit_eta], "lr": 5e-2})
        else:
            logit_eta = None

        # Closure used by both Adam and L-BFGS branches.
        loss_iter = [0]   # mutable counter used to print step-0 diagnostics
        def loss_closure():
            opt.zero_grad()
            g = torch.Generator(device=gen_device).manual_seed(pred_seed)
            psf_arg = (torch.exp(log_psf) if log_psf is not None else None)
            eta_arg = (torch.sigmoid(logit_eta)
                       if logit_eta is not None else None)
            I_pred = mix.render(self.model,
                                self.tensors["lattice"],
                                self.tensors["P"],
                                self.tensors["R"],
                                M=self.M_render, generator=g,
                                E_range=self.E_range,
                                per_spot_intensity=psi_per_sample,
                                psf_sigma=psf_arg, psf_eta=eta_arg)
            loss = ((I_pred - I_obs) ** 2 * patch_mask_f).sum() / patch_mask_count
            if loss_iter[0] == 0:
                with torch.no_grad():
                    Iobs_sq_in_patches = ((I_obs ** 2) * patch_mask_f).sum().item() / patch_mask_count.item()
                    Ipred_sq_in_patches = ((I_pred ** 2) * patch_mask_f).sum().item() / patch_mask_count.item()
                    print(f"  step0 patch diagnostics: "
                          f"I_pred patch sq_mean={Ipred_sq_in_patches:.3e} "
                          f"I_obs patch sq_mean={Iobs_sq_in_patches:.3e} "
                          f"loss={loss.item():.3e} "
                          f"patch pixels={int(patch_mask_count.item())}", flush=True)
            loss_iter[0] += 1
            loss.backward()
            return loss

        if self.optimizer == "adam":
            opt = torch.optim.Adam(opt_param_groups)
            for step in range(self.n_steps):
                loss = loss_closure()
                opt.step()
                last_loss = loss.item()
                if step == 0 or (step + 1) % log_every == 0 or step == self.n_steps - 1:
                    elapsed = time.time() - t_step0
                    rate = (step + 1) / elapsed if elapsed > 0 else 0
                    eta = (self.n_steps - step - 1) / rate if rate > 0 else 0
                    print(f"  step {step+1:4d}/{self.n_steps}  loss={last_loss:.4g}  "
                          f"({rate:.2f} step/s, elapsed {elapsed/60:.1f} min, "
                          f"ETA {eta/60:.1f} min)", flush=True)
        else:  # lbfgs
            # L-BFGS is a quasi-Newton method with strong-Wolfe line search.
            # It uses past gradients to approximate the Hessian, so each
            # "step" can take many internal iterations.  Combined with the
            # line search, it auto-adapts step size to the loss landscape
            # and is much more robust than Adam on deterministic, smooth
            # losses (which is exactly our regime once the per-spot
            # intensity has fixed amplitude scale).  Note that the closure
            # is called multiple times per step (line search), so the
            # n_steps interpretation differs.
            params_flat = [p for grp in opt_param_groups for p in grp["params"]]
            opt = torch.optim.LBFGS(
                params_flat,
                lr=1.0,
                max_iter=20,
                history_size=20,
                tolerance_grad=1e-9,
                tolerance_change=1e-12,
                line_search_fn="strong_wolfe",
            )
            for step in range(self.n_steps):
                loss = opt.step(loss_closure)
                last_loss = float(loss)
                if step == 0 or (step + 1) % log_every == 0 or step == self.n_steps - 1:
                    elapsed = time.time() - t_step0
                    print(f"  outer {step+1:4d}/{self.n_steps}  loss={last_loss:.4g}  "
                          f"(elapsed {elapsed/60:.2f} min)", flush=True)

        # Extract per-mode results.
        if self.mode == "orient_only":
            U_means = torch.stack([k.mean().detach() for k in mix.kernels], dim=0)
            cov_diag = torch.stack(
                [k.covariance().diag().detach() for k in mix.kernels], dim=0)
            sigma_U_deg = torch.tensor(
                [math.degrees(math.sqrt(cd.mean().item())) for cd in cov_diag])
            eps_means = None
        else:
            U_means = torch.stack([c.orient.mean().detach() for c in mix.components], dim=0)
            cov_diag = torch.stack(
                [c.orient.covariance().diag().detach() for c in mix.components], dim=0)
            sigma_U_deg = torch.tensor(
                [math.degrees(math.sqrt(cd.mean().item())) for cd in cov_diag])
            eps_means = torch.stack([c.strain.mean.detach() for c in mix.components], dim=0)

        pi = mix.weights().detach()

        # Cubic miso between recovered means and seed means.
        from ..symmetry import cubic_misorientation_deg
        misos = torch.tensor(
            [cubic_misorientation_deg(U_means[k:k + 1],
                                      U_seed_list[k:k + 1]).item()
             for k in range(K)])

        recovered_psf_sigma_px = (float(torch.exp(log_psf).detach().item())
                                   if log_psf is not None else None)
        recovered_psf_eta = (float(torch.sigmoid(logit_eta).detach().item())
                              if logit_eta is not None else None)
        if recovered_psf_sigma_px is not None:
            print(f"  recovered psf_sigma = {recovered_psf_sigma_px:.3f} px "
                  f"(initial {self.psf_sigma:.3f} px)", flush=True)
        if recovered_psf_eta is not None:
            print(f"  recovered psf_eta   = {recovered_psf_eta:.3f} "
                  f"(initial {self.psf_eta:.3f}; pure Gaussian = 0, "
                  f"pure Lorentzian = 1)", flush=True)

        posterior, posterior_names = None, None
        if self.compute_posterior:
            posterior, posterior_names = self._laplace(
                mix, I_obs, psi_per_sample, patch_mask_f, pred_seed,
                gen_device, log_psf, logit_eta)

        return MultiGrainResult(
            n_modes=K,
            U_means=U_means,
            sigma_U_deg=sigma_U_deg,
            pi=pi,
            eps_means=eps_means,
            final_loss=last_loss,
            initial_seed_misos_deg=misos,
            n_steps=self.n_steps,
            dt_s=time.time() - t0,
            metadata={"psf_sigma_init_px": self.psf_sigma,
                      "psf_sigma_recovered_px": recovered_psf_sigma_px,
                      "psf_eta_init": self.psf_eta,
                      "psf_eta_recovered": recovered_psf_eta,
                      "posterior_conditional_on_fixed_means":
                          (None if posterior is None else not self.refine_means)},
            posterior=posterior,
            posterior_param_names=posterior_names,
        )

    def _laplace(self, mix, I_obs: Tensor, psi_per_sample: Tensor,
                 patch_mask_f: Tensor, seed: int, gen_device,
                 log_psf: Optional[Tensor], logit_eta: Optional[Tensor]):
        """Laplace posterior over the fitted parameters at the optimum.

        The flat vector holds, per mode ``k``: the 3 log-diagonal and 3
        off-diagonal Cholesky entries of the orientation spread; the strain
        mean (5 deviatoric or 6 Voigt) in strain modes; with
        ``refine_means``, a 3-vector tangent rotation composed on the fitted
        mean (0 at the optimum; the 6-D mean representation has 3 gauge
        directions, this does not).  Then ``K-1`` mixing logits relative to
        mode 0 (softmax is shift-invariant, so the absolute logits have a
        null direction), and ``log_psf`` / ``logit_eta`` if refined.

        The residual is the fit's own, over the patch-mask pixels, with the
        same fixed MC samples and energy band; the curvature scale is set by
        :func:`laue_torch.uncertainty.laplace_posterior_from_residuals`
        (``0.5 * SSR`` with the plug-in per-pixel noise variance).
        """
        K = len(mix.kernels) if self.mode == "orient_only" else len(mix.components)
        orients = (list(mix.kernels) if self.mode == "orient_only"
                   else [c.orient for c in mix.components])
        strains = (None if self.mode == "orient_only"
                   else [c.strain for c in mix.components])
        dtype = torch.float64
        dev = I_obs.device
        strain_names = (["e11", "e22", "e23", "e13", "e12"]
                        if self.mode == "strain_deviatoric"
                        else ["e11", "e22", "e33", "e23", "e13", "e12"])

        pieces, names, slices = [], [], []

        def add(t: Tensor, labels: list):
            start = sum(p.numel() for p in pieces)
            pieces.append(t.detach().reshape(-1).to(dtype))
            names.extend(labels)
            slices.append((start, start + t.numel()))
            return len(slices) - 1

        idx = []
        for k in range(K):
            entry = {
                "log_diag": add(orients[k].cov.log_diag,
                                [f"mode{k}.orient_chol_logdiag[{i}]" for i in range(3)]),
                "off_diag": add(orients[k].cov.off_diag,
                                [f"mode{k}.orient_chol_offdiag[{i}]" for i in range(3)]),
            }
            if strains is not None:
                entry["strain"] = add(self._strain_param(strains[k]),
                                      [f"mode{k}.strain_mean.{n}" for n in strain_names])
            if self.refine_means:
                entry["dtheta"] = add(torch.zeros(3, dtype=dtype, device=dev),
                                      [f"mode{k}.dtheta_rad[{i}]" for i in range(3)])
            idx.append(entry)
        logits = mix.logits.detach().to(dtype)
        i_logit = (add(logits[1:] - logits[0],
                       [f"mode{k}.logit_minus_mode0" for k in range(1, K)])
                   if K > 1 else None)
        i_psf = add(log_psf.reshape(1), ["log_psf_sigma"]) if log_psf is not None else None
        i_eta = add(logit_eta.reshape(1), ["logit_psf_eta"]) if logit_eta is not None else None
        theta_map = torch.cat(pieces)

        U_map = [o.mean().detach() for o in orients]
        L_strain = ([st.cov.L().detach() for st in strains]
                    if strains is not None else None)
        tril = orients[0].cov.tril_idx
        counts = _stratified_counts(self.M_render, K)

        def seg(theta: Tensor, i: int) -> Tensor:
            a, b = slices[i]
            return theta[a:b]

        patch_sel = patch_mask_f > 0.5

        def residual_fn(theta: Tensor) -> Tensor:
            # Replays mix.sample() draw-for-draw with the fit's fixed seed.
            if i_logit is not None:
                lg = torch.cat([torch.zeros(1, dtype=dtype, device=dev), seg(theta, i_logit)])
            else:
                lg = torch.zeros(1, dtype=dtype, device=dev)
            w = torch.softmax(lg, dim=0)
            g = torch.Generator(device=gen_device).manual_seed(seed)
            Us, epss, ws = [], [], []
            for k, m_k in enumerate(counts):
                if m_k == 0:
                    continue
                L = torch.diag(seg(theta, idx[k]["log_diag"]).exp())
                L = L.clone()
                L[tril[0], tril[1]] = seg(theta, idx[k]["off_diag"])
                z = torch.randn(m_k, 3, dtype=dtype, device=dev, generator=g)
                U_mean = U_map[k]
                if self.refine_means:
                    U_mean = U_mean @ rodrigues_to_matrix(seg(theta, idx[k]["dtheta"]))
                Us.append(U_mean.unsqueeze(0) @ rodrigues_to_matrix(z @ L.T))
                if strains is not None:
                    Ls = L_strain[k]
                    zs = torch.randn(m_k, Ls.shape[0], dtype=dtype, device=dev,
                                     generator=g)
                    e = seg(theta, idx[k]["strain"]).unsqueeze(0) + zs @ Ls.T
                    if self.mode == "strain_deviatoric":
                        e = _dev5_to_voigt6(e)
                    epss.append(e)
                else:
                    epss.append(torch.zeros(m_k, 6, dtype=dtype, device=dev))
                ws.append((w[k] / m_k).expand(m_k))
            psf_arg = seg(theta, i_psf).exp()[0] if i_psf is not None else None
            eta_arg = torch.sigmoid(seg(theta, i_eta))[0] if i_eta is not None else None
            I_pred = self.model(torch.cat(Us), self.tensors["lattice"],
                                self.tensors["P"], self.tensors["R"],
                                strain=torch.cat(epss), weights=torch.cat(ws),
                                E_range=self.E_range,   # same band as the fit
                                per_spot_intensity=psi_per_sample,
                                psf_sigma=psf_arg, psf_eta=eta_arg)
            return (I_pred - I_obs)[patch_sel]

        post = laplace_posterior_from_residuals(residual_fn, theta_map)
        return post, names
