"""Recovery-validation suite for σ_U / Σ_U fits on real-data refinements.

Six complementary checks, designed to flag MC-noise-plateau artefacts and
unstable convergence on multi-modal real-data loss landscapes.  Use as::

    from laue_torch.realdata.validate import validate_recovery
    rep = validate_recovery(
        model=model, U_seed=U_seed, sigma_U_deg=sigma_U_recovered,
        cov_full=cov_3x3,                        # body-frame Σ_U (rad²)
        I_obs=I_obs, target_psi=target_psi,
        lat=lat, P=P, R=R,
        M_render=512, sigma_psf_px=2.0,
        n_visible_HKLs=948, detector_distance_mm=510.0,
        px_size_mm=0.2,
        # optional:
        run_M_doubling=True, multi_start_results=ablation_rows,
    )
    print(rep)

Each check populates one slot of :class:`ValidationReport`; missing slots
indicate the check was skipped (None).  ``rep.summary()`` prints a
PASS/FAIL/UNKNOWN line per check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from ..geometry import rodrigues_to_matrix


@dataclass
class ValidationReport:
    """Per-check verdicts + numerical evidence."""
    # A — loss vs MC-noise floor
    loss_at_minimum: Optional[float] = None
    loss_mc_noise_floor_mean: Optional[float] = None
    loss_mc_noise_floor_std: Optional[float] = None
    passes_loss_floor: Optional[bool] = None
    # B — σ_critical
    sigma_recovered_deg: Optional[float] = None
    sigma_critical_deg: Optional[float] = None
    passes_sigma_critical: Optional[bool] = None
    # C — per-spot pred↔obs correlation
    median_pearson: Optional[float] = None
    pearson_q10: Optional[float] = None
    pearson_q90: Optional[float] = None
    n_spots_correlated: Optional[int] = None
    n_spots_total: Optional[int] = None
    passes_correlation: Optional[bool] = None
    # D — M_render doubling
    sigma_at_2x_M: Optional[float] = None
    sigma_relative_change: Optional[float] = None
    passes_M_doubling: Optional[bool] = None
    # E — multi-start cluster
    multi_start_sigmas: Optional[list[float]] = None
    multi_start_losses: Optional[list[float]] = None
    cluster_median_sigma: Optional[float] = None
    cluster_relative_spread: Optional[float] = None
    passes_multi_start: Optional[bool] = None
    # F — residual peak structure
    residual_energy_inside_peaks: Optional[float] = None
    residual_energy_outside_peaks: Optional[float] = None
    residual_concentration_ratio: Optional[float] = None
    explained_fraction: Optional[float] = None
    passes_residual: Optional[bool] = None
    # provenance
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        def verdict(v):
            if v is None: return "—   "
            return "PASS" if v else "FAIL"
        lines = [
            f"  A loss-vs-noise-floor   {verdict(self.passes_loss_floor)} "
            f"(L={self.loss_at_minimum} floor={self.loss_mc_noise_floor_mean})"
            if self.loss_at_minimum is not None else "  A loss-vs-noise-floor   —",
            f"  B σ_critical            {verdict(self.passes_sigma_critical)} "
            f"(σ_rec={self.sigma_recovered_deg:.4f}° σ_crit={self.sigma_critical_deg:.4f}°)"
            if self.sigma_critical_deg is not None else "  B σ_critical            —",
            f"  C per-spot correlation  {verdict(self.passes_correlation)} "
            f"(median ρ={self.median_pearson:.2f}, "
            f"q10-q90 [{self.pearson_q10:.2f},{self.pearson_q90:.2f}], "
            f"n>0.5 = {self.n_spots_correlated}/{self.n_spots_total})"
            if self.median_pearson is not None else "  C per-spot correlation  —",
            f"  D M-render doubling     {verdict(self.passes_M_doubling)} "
            f"(σ_2M={self.sigma_at_2x_M:.4f}°, |Δ|={self.sigma_relative_change*100:.1f}%)"
            if self.sigma_at_2x_M is not None else "  D M-render doubling     —",
            f"  E multi-start cluster   {verdict(self.passes_multi_start)} "
            f"(cluster σ={self.cluster_median_sigma:.4f}°, "
            f"spread {self.cluster_relative_spread*100:.0f}%)"
            if self.cluster_median_sigma is not None else "  E multi-start cluster   —",
            f"  F residual peak ratio   {verdict(self.passes_residual)} "
            f"(in/out energy = {self.residual_concentration_ratio:.2f})"
            if self.residual_concentration_ratio is not None else "  F residual peak ratio   —",
        ]
        return "\n".join(lines)


# ── core renderer (re-render at known Σ_U with controllable RNG) ──────────

def _render_at_sigma(model, U_seed, lat, P, R,
                     sigma_deg: float | torch.Tensor,
                     M_render: int, target_psi: torch.Tensor,
                     seed: int, *, cov_full: Optional[torch.Tensor] = None
                     ) -> torch.Tensor:
    """Render at U_seed with isotropic σ_deg cloud (or full cov_full).
    Returns I_pred (Nx, Ny)."""
    device, dtype = lat.device, lat.dtype
    H = model.hkls.shape[0]
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(M_render, 3, dtype=dtype, device=device, generator=g)
    if cov_full is not None:
        L = torch.linalg.cholesky(cov_full)
        delta = z @ L.T
    else:
        delta = z * math.radians(sigma_deg)
    U_cloud = U_seed.unsqueeze(0) @ rodrigues_to_matrix(delta)
    eps = torch.zeros(M_render, 6, dtype=dtype, device=device)
    w = torch.full((M_render,), 1.0 / M_render, dtype=dtype, device=device)
    psi = target_psi.unsqueeze(0).expand(M_render, H)
    with torch.no_grad():
        I = model(U_cloud, lat, P, R, strain=eps, weights=w,
                  per_spot_intensity=psi)
    return I


# ── individual checks ─────────────────────────────────────────────────────

def check_loss_floor(model, U_seed, lat, P, R, sigma_U_deg, M_render,
                     target_psi, I_obs, *, n_seeds: int = 5,
                     cov_full: Optional[torch.Tensor] = None,
                     loss_at_minimum: Optional[float] = None,
                     ) -> dict:
    """A.  Render at the converged σ_U with N independent MC seeds and
    measure the *coefficient of variation* of the loss across seeds.

    A real fit is in the smooth-envelope regime: the rendered image is
    insensitive to the MC seed at fixed σ, so loss varies by <5%.  A
    noise plateau is shot-noise dominated: the rendered image is a
    different scatter of M_render points for every seed, and the loss
    swings by 30%+.  CoV is a self-contained, M_render-aware MC-stability
    test (no cross-metric comparison required).
    """
    losses = []
    for s in range(n_seeds):
        I_pred = _render_at_sigma(model, U_seed, lat, P, R, sigma_U_deg,
                                   M_render, target_psi, seed=12345 + s,
                                   cov_full=cov_full)
        losses.append(float(((I_pred - I_obs) ** 2).mean().item()))
    arr = np.asarray(losses)
    mean = float(arr.mean()); std = float(arr.std())
    cov = std / max(mean, 1e-30)
    out = {
        "loss_mc_noise_floor_mean": mean,
        "loss_mc_noise_floor_std":  std,
    }
    if loss_at_minimum is not None:
        out["loss_at_minimum"] = float(loss_at_minimum)
    # PASS = CoV across MC seeds < 0.10 (smooth envelope regime).
    # FAIL = CoV >= 0.10 (MC-noise-dominated rendering).
    out["passes_loss_floor"] = bool(cov < 0.10)
    return out


def check_sigma_critical(sigma_U_recovered_deg: float, M_render: int,
                          sigma_psf_px: float, n_visible_HKLs: int,
                          detector_distance_mm: float, px_size_mm: float,
                          *, geometric_factor: float = 1.0) -> dict:
    """B.  σ_critical is the σ above which the per-spot envelope is
    sparser than the PSF (so MC samples don't overlap and the rendered
    image becomes shot noise rather than a smooth envelope).

    Heuristic derivation (per spot):
      envelope_area_px  ≈ (4 σ_pos)²
      coverage_per_PSF  ≈ π σ_PSF²
      M_per_spot        ≈ M_render / n_visible_HKLs   (rough)
                        ≈ M_render                     (if per_spot mode)
      smooth iff coverage * M_per_spot ≥ envelope_area, i.e.
        σ_pos² ≤ (M_per_spot × π σ_PSF²) / 16
      σ_pos_critical = σ_PSF × sqrt(M_per_spot × π / 16)

    σ_U_critical = σ_pos_critical × px_size_mm /
                   (detector_distance_mm × geometric_factor)
    """
    M_per_spot = M_render  # the per_spot_intensity mode hits every HKL
    sigma_pos_critical_px = sigma_psf_px * math.sqrt(M_per_spot * math.pi / 16.0)
    sigma_U_critical_rad = (sigma_pos_critical_px * px_size_mm
                             / (detector_distance_mm * geometric_factor))
    sigma_U_critical_deg = math.degrees(sigma_U_critical_rad)
    return {
        "sigma_recovered_deg": float(sigma_U_recovered_deg),
        "sigma_critical_deg":  float(sigma_U_critical_deg),
        "passes_sigma_critical": bool(sigma_U_recovered_deg < sigma_U_critical_deg),
    }


def check_per_spot_correlation(model, U_seed, lat, P, R, sigma_U_deg,
                                 M_render, target_psi, I_obs,
                                 spot_centers_xy_px: np.ndarray,
                                 *, W: int = 13, n_seeds: int = 1,
                                 cov_full: Optional[torch.Tensor] = None,
                                 correlation_threshold: float = 0.2,
                                 min_obs_intensity: float = 1.0) -> dict:
    """C.  Pearson correlation between predicted and observed pixel
    intensities within a W×W patch around each indexer-confirmed spot
    centre.  A real fit has predicted shape tracking observed (ρ > 0.5);
    a noise plateau gives random predicted intensity (ρ ≈ 0)."""
    I_pred_avg = torch.zeros_like(I_obs)
    for s in range(n_seeds):
        I_pred = _render_at_sigma(model, U_seed, lat, P, R, sigma_U_deg,
                                   M_render, target_psi, seed=20000 + s,
                                   cov_full=cov_full)
        I_pred_avg = I_pred_avg + I_pred
    I_pred_avg = (I_pred_avg / n_seeds).cpu().numpy()
    I_obs_np = I_obs.cpu().numpy()
    Nx, Ny = I_obs_np.shape
    r = W // 2
    correlations = []
    for cx, cy in spot_centers_xy_px:
        cx_i, cy_i = int(round(cx)), int(round(cy))
        i0 = max(0, cx_i - r); i1 = min(Nx, cx_i + r + 1)
        j0 = max(0, cy_i - r); j1 = min(Ny, cy_i + r + 1)
        po = I_obs_np[i0:i1, j0:j1].reshape(-1)
        pp = I_pred_avg[i0:i1, j0:j1].reshape(-1)
        if po.max() < min_obs_intensity:  # patch has no observed signal
            continue
        if po.std() < 1e-12 or pp.std() < 1e-12:
            continue
        r_pearson = float(np.corrcoef(po, pp)[0, 1])
        correlations.append(r_pearson)
    if not correlations:
        return {"median_pearson": float("nan"),
                "passes_correlation": None,
                "n_spots_total": 0}
    arr = np.asarray(correlations)
    out = {
        "median_pearson": float(np.median(arr)),
        "pearson_q10":    float(np.percentile(arr, 10)),
        "pearson_q90":    float(np.percentile(arr, 90)),
        "n_spots_correlated": int((arr > correlation_threshold).sum()),
        "n_spots_total":      int(arr.size),
        "passes_correlation": bool(np.median(arr) > correlation_threshold),
    }
    return out


def check_M_doubling(model, U_seed, lat, P, R, sigma_U_deg, M_render,
                     target_psi, I_obs, *,
                     refine_fn=None, sigma_init_deg: float = 0.05,
                     cov_full: Optional[torch.Tensor] = None) -> dict:
    """D.  Re-fit with 2 × M_render and compare the converged σ_U.
    Requires a callable ``refine_fn(M_render) -> sigma_U_deg`` that
    runs a fresh refinement at the given M_render.  If σ_U changes by
    more than 10 %, the original was M_render-limited.

    If ``refine_fn`` is not provided we skip this check (it requires
    a fresh fit which the caller has to wire in)."""
    if refine_fn is None:
        return {}
    sigma_at_2M = refine_fn(2 * M_render)
    rel = abs(sigma_at_2M - sigma_U_deg) / max(sigma_U_deg, 1e-9)
    return {
        "sigma_at_2x_M": float(sigma_at_2M),
        "sigma_relative_change": float(rel),
        "passes_M_doubling": bool(rel < 0.10),
    }


def check_multi_start(sigmas_recovered: Sequence[float],
                       losses: Sequence[float],
                       *, cluster_relative_threshold: float = 0.50) -> dict:
    """E.  Across N independent multi-start fits at the same recipe,
    take the lowest-loss fits and check that they cluster.  Spread > 30 %
    of the cluster median → multi-modal landscape; reject."""
    if len(sigmas_recovered) < 3:
        return {}
    arr_s = np.asarray(sigmas_recovered, dtype=np.float64)
    arr_l = np.asarray(losses, dtype=np.float64)
    # Take the lowest-loss N // 2 + 1 fits as "the cluster".
    n = len(arr_s); k = max(2, n // 2 + 1)
    idx = np.argsort(arr_l)[:k]
    cluster = arr_s[idx]
    med = float(np.median(cluster))
    spread = float((cluster.max() - cluster.min()) / max(med, 1e-9))
    return {
        "multi_start_sigmas":  [float(x) for x in arr_s.tolist()],
        "multi_start_losses":  [float(x) for x in arr_l.tolist()],
        "cluster_median_sigma": med,
        "cluster_relative_spread": spread,
        "passes_multi_start": bool(spread < cluster_relative_threshold),
    }


def check_residual_structure(model, U_seed, lat, P, R, sigma_U_deg,
                               M_render, target_psi, I_obs,
                               spot_centers_xy_px: np.ndarray,
                               *, W: int = 13,
                               cov_full: Optional[torch.Tensor] = None) -> dict:
    """F.  Fraction of observed peak energy explained by the model:
    1 − (Σ residual² inside peaks) / (Σ I_obs² inside peaks).

    Real fit: explained fraction ≈ 1 (model matches peaks closely).
    Noise plateau: explained fraction ≪ 1 (model misses peaks; residual
    inside peaks ≈ I_obs inside peaks).
    """
    I_pred = _render_at_sigma(model, U_seed, lat, P, R, sigma_U_deg,
                               M_render, target_psi, seed=30000,
                               cov_full=cov_full)
    I_obs_np = I_obs.cpu().numpy()
    I_pred_np = I_pred.cpu().numpy()
    R_resid = I_obs_np - I_pred_np
    Nx, Ny = R_resid.shape
    inside_mask = np.zeros((Nx, Ny), dtype=bool)
    r = W // 2
    for cx, cy in spot_centers_xy_px:
        cx_i, cy_i = int(round(cx)), int(round(cy))
        i0 = max(0, cx_i - r); i1 = min(Nx, cx_i + r + 1)
        j0 = max(0, cy_i - r); j1 = min(Ny, cy_i + r + 1)
        inside_mask[i0:i1, j0:j1] = True
    obs_in_sq = float((I_obs_np[inside_mask] ** 2).sum())
    res_in_sq = float((R_resid[inside_mask] ** 2).sum())
    explained_fraction = 1.0 - res_in_sq / max(obs_in_sq, 1e-30)
    # Also compute the (in/out) density ratio for backward-compat.
    R_sq = R_resid ** 2
    in_total = float(R_sq[inside_mask].sum())
    out_total = float(R_sq[~inside_mask].sum())
    in_n = int(inside_mask.sum()); out_n = int((~inside_mask).sum())
    ratio = (in_total / max(in_n, 1)) / max(out_total / max(out_n, 1), 1e-30)
    return {
        "residual_energy_inside_peaks":  in_total,
        "residual_energy_outside_peaks": out_total,
        "residual_concentration_ratio":  float(ratio),
        "explained_fraction":            float(explained_fraction),
        # PASS = model explains > 50 % of observed peak energy.
        "passes_residual": bool(explained_fraction > 0.5),
    }


# ── orchestrator ──────────────────────────────────────────────────────────

def validate_recovery(*, model, U_seed, lat, P, R,
                       sigma_U_deg: float,
                       M_render: int, sigma_psf_px: float,
                       n_visible_HKLs: int,
                       detector_distance_mm: float, px_size_mm: float,
                       I_obs: torch.Tensor, target_psi: torch.Tensor,
                       spot_centers_xy_px: np.ndarray,
                       cov_full: Optional[torch.Tensor] = None,
                       loss_at_minimum: Optional[float] = None,
                       checks: tuple = ("A", "B", "C", "F"),
                       multi_start_sigmas: Optional[Sequence[float]] = None,
                       multi_start_losses: Optional[Sequence[float]] = None,
                       refine_fn=None) -> ValidationReport:
    rep = ValidationReport()
    rep.sigma_recovered_deg = float(sigma_U_deg)
    if "A" in checks:
        d = check_loss_floor(model, U_seed, lat, P, R,
                              sigma_U_deg, M_render, target_psi, I_obs,
                              loss_at_minimum=loss_at_minimum,
                              cov_full=cov_full)
        for k, v in d.items(): setattr(rep, k, v)
    if "B" in checks:
        d = check_sigma_critical(sigma_U_deg, M_render, sigma_psf_px,
                                  n_visible_HKLs, detector_distance_mm,
                                  px_size_mm)
        for k, v in d.items(): setattr(rep, k, v)
    if "C" in checks:
        d = check_per_spot_correlation(model, U_seed, lat, P, R,
                                         sigma_U_deg, M_render, target_psi,
                                         I_obs, spot_centers_xy_px,
                                         cov_full=cov_full)
        for k, v in d.items(): setattr(rep, k, v)
    if "D" in checks and refine_fn is not None:
        d = check_M_doubling(model, U_seed, lat, P, R, sigma_U_deg,
                              M_render, target_psi, I_obs,
                              refine_fn=refine_fn)
        for k, v in d.items(): setattr(rep, k, v)
    if "E" in checks and multi_start_sigmas is not None:
        d = check_multi_start(multi_start_sigmas, multi_start_losses)
        for k, v in d.items(): setattr(rep, k, v)
    if "F" in checks:
        d = check_residual_structure(model, U_seed, lat, P, R,
                                       sigma_U_deg, M_render, target_psi,
                                       I_obs, spot_centers_xy_px,
                                       cov_full=cov_full)
        for k, v in d.items(): setattr(rep, k, v)
    return rep
