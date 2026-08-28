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
    deviatoric-5).

Modes
-----

  * ``orient_only`` --- per-mode tangent Gaussian on SO(3); strain
    held at zero.  Use for FCC samples without expected strain
    (e.g. an undeformed reference).
  * ``strain_voigt`` --- per-mode strain Gaussian (Voigt-6 mean,
    spread frozen at zero per the paper's recommendation since
    position-only data can't see Σ_ε).  This is the EuAl2O4 case:
    each grain is a parent + characteristic strain.
  * ``strain_deviatoric`` --- 5-DOF deviatoric strain mean, drops the
    hydrostatic null direction (`ε33 = -(ε11+ε22)`).  Recommended
    when the user wants the paper's "full deviatoric strain"
    capability over the existing const-volume c/a refinement.

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
from ..forward import LaueForwardModel
from ..io import LaueParams, generate_hkls
from ..uncertainty import laplace_posterior


@dataclass
class MultiGrainResult:
    """Recovered per-mode parameters for one voxel."""
    n_modes: int
    U_means: Tensor                      # (K, 3, 3) refined orientations
    sigma_U_deg: Tensor                  # (K,) per-mode mosaic spread (deg)
    pi: Tensor                           # (K,) mixing weights, sum 1
    eps_means: Optional[Tensor]          # (K, 6) Voigt strain or None
    final_loss: float
    initial_seed_misos_deg: Tensor       # (K,) cubic miso between final and seed for each mode
    n_steps: int
    dt_s: float
    metadata: dict = field(default_factory=dict)


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

        self.hkls = generate_hkls(params.sg_num, params.lattice, params.E_hi)
        self.tensors = params.to_tensors(dtype=torch.float64, device=device)

        # Forward model: shared across all modes.
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

    @torch.no_grad()
    def _compute_per_spot_target(self, mix, U_seed_list: Tensor, I_obs: Tensor
                                 ):
        """Returns ``(target_per_spot, patch_mask)``.

        - ``target_per_spot``: shape ``(H,)`` per-HKL intrinsic spot
          intensity, equal to the sum of ``I_obs`` in a
          ``render_window×render_window`` patch around the
          seed-orientation predicted spot centre.  Used as
          ``per_spot_intensity`` in the Adam loop so predicted peak
          amplitudes match observed integrated counts at the seed.
        - ``patch_mask``: shape ``(Nx, Ny)`` boolean tensor that is
          ``True`` inside the union of those W×W windows.  Used to
          restrict the loss to spot regions, ignoring the remainder of
          the image where post-background-subtraction residual is not a
          diffraction peak the model could fit.

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

        target = torch.zeros(H, dtype=torch.float64, device=device)
        patch_mask = torch.zeros(Nx, Ny, dtype=torch.bool, device=device)
        offsets = torch.arange(-r, r + 1, device=device, dtype=torch.long)
        I_obs_flat = I_obs.reshape(-1)
        eps_zero = torch.zeros(1, 6, dtype=torch.float64, device=device)
        weights_one = torch.ones(1, dtype=torch.float64, device=device)

        for k in range(K):
            U_k = U_seed_list[k:k + 1]  # (1, 3, 3)
            _, aux = self.model(U_k, self.tensors["lattice"],
                                self.tensors["P"], self.tensors["R"],
                                strain=eps_zero, weights=weights_one,
                                return_aux=True)
            cx = aux.px.detach().round().long().clamp(0, Nx - 1)
            cy = aux.py.detach().round().long().clamp(0, Ny - 1)
            tx = cx[:, None] + offsets[None, :]               # (H, W)
            ty = cy[:, None] + offsets[None, :]
            valid_x = (tx >= 0) & (tx < Nx)
            valid_y = (ty >= 0) & (ty < Ny)
            tx_c = tx.clamp(0, Nx - 1)
            ty_c = ty.clamp(0, Ny - 1)
            flat_idx = (tx_c[:, :, None] * Ny + ty_c[:, None, :])  # (H, W, W)
            valid = (valid_x[:, :, None] & valid_y[:, None, :]).to(torch.float64)
            obs_tile = I_obs_flat[flat_idx.reshape(-1)].reshape(H, W, W) * valid
            patch_sum = obs_tile.sum(dim=(1, 2))               # (H,)
            mask_k = aux.mask.detach().reshape(H)
            target = target + patch_sum * mask_k

            # Build patch mask from valid HKLs only, expanded slightly so
            # the σ_U gradient has room to inflate without spilling out of
            # the loss region.
            keep_h = (mask_k > 0.5).nonzero(as_tuple=False).reshape(-1)
            if keep_h.numel() > 0:
                tx_keep = tx_c[keep_h].reshape(-1)             # (n_keep * W,)
                ty_keep = ty_c[keep_h].reshape(-1)
                # All (W,W) = 169 pixels per kept HKL on the patch_mask.
                # Use index_put_ for efficient scatter.
                tx_pairs = tx_keep.unsqueeze(-1).expand(-1, W).reshape(-1)
                ty_pairs = ty_c[keep_h, None, :].expand(-1, W, -1).reshape(-1)
                # Simpler: rebuild full coord grids
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
        # The indexer's reported ``intensity`` (column 10 of
        # /entry/results/spots) is the *integrated* intensity, not the
        # peak amplitude, so it would over-shoot when used as a peak
        # target.  We deliberately use the patch-max as a peak-amplitude
        # proxy — this is the calibration that makes the synthetic
        # exp5l recovery converge correctly and keeps the EuAl2O4
        # null-hypothesis recovery at the FWHM-derived bound.
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
        seed: int = 0xC0FFEE,
        indexer_spots: Optional[dict] = None,
    ) -> MultiGrainResult:
        """Refine the K-grain mixture.

        ``U_seed_list``: (K, 3, 3) seed orientation matrices.
        ``image``: (Nx, Ny) observed Laue pattern.
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
        # tests but critical for any pixel-wise comparison.
        I_obs = image.to(self.device, dtype=torch.float64).T.contiguous()
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
                {"params": [c.strain.mean for c in mix.components], "lr": 1e-4},
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
        #     spot list directly.  ``target_per_spot[h]`` is the indexer's
        #     reported intensity at the matching (h,k,l), zero everywhere
        #     else.  ``patch_mask`` covers W×W around each indexer (X, Y).
        #     This is the cleanest signal — the indexer has already done
        #     the spot-vs-noise classification.
        # (2) fallback: compute target as window-sum of I_obs at each
        #     hkl's seed-predicted (px, py).  Vulnerable to confounding
        #     by post-bg residual when the obs image is noisy, so prefer
        #     path (1) when an indexer spot list is available.
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
        # Solution: pre-compute target_per_spot[h] = ∑ I_obs(window) at
        # the seed-orientation predicted (px_h, py_h) and pass it as the
        # forward's per_spot_intensity.  Predicted peak amplitudes then
        # match observed peak amplitudes by construction at the seed
        # orientation; the remaining loss measures only the shape/
        # positional mismatch the optimizer is actually meant to fit
        # (mosaic spread, strain-induced position shift).
        if indexer_spots is not None:
            target_psi, patch_mask = self._target_from_indexer_spots(
                indexer_spots["hkl"].to(self.device),
                indexer_spots["xy"].to(self.device),
                indexer_spots["intensity"].to(self.device),
                I_obs,
            )
        else:
            target_psi, patch_mask = self._compute_per_spot_target(
                mix, U_seed_list, I_obs)
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

        FIXED_PRED_SEED = seed
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
            g = torch.Generator(device=gen_device).manual_seed(FIXED_PRED_SEED)
            psf_arg = (torch.exp(log_psf) if log_psf is not None else None)
            eta_arg = (torch.sigmoid(logit_eta)
                       if logit_eta is not None else None)
            I_pred = mix.render(self.model,
                                self.tensors["lattice"],
                                self.tensors["P"],
                                self.tensors["R"],
                                M=self.M_render, generator=g,
                                per_spot_intensity=target_psi,
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
                      "psf_eta_recovered": recovered_psf_eta},
        )
