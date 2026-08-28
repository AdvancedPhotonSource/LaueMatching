"""Loss-landscape analysis for coded-aperture mask refinement.

Computes the Hessian of the autofocus loss at a chosen mask pose and
derives the Cramér–Rao information bound on the recoverable mask DOFs
given an assumed per-pixel noise level.  This is the differentiable
analogue of exp6_landscape_analysis.py — same machinery, but for the
mask-pose-only parameter set used by Phase 3 autofocusing.

What it tells you
-----------------

* **Which mask DOFs are well-constrained** — the eigenvalue spectrum
  of the Hessian.  Large eigenvalues = steep curvature = easy to recover.
* **Which DOFs are degenerate** — eigenvalues near zero plus their
  eigenvectors identify the parameter combinations that the data
  cannot distinguish.  (For a thin Si calibrant with rays clustered
  near the optical axis, pitch and sway are typically degenerate;
  see the docstring of :func:`run` for a worked example.)
* **The CR-σ bound per DOF** — the best precision achievable for each
  parameter at the assumed noise level.  Drops as 1/√N with the number
  of pixel observations (frames × voxels × Nx × Ny).
* **The parameter correlation matrix** — the off-diagonal structure of
  the CR covariance shows which DOF pairs trade off against each other.

Why this matters for the paper
-------------------------------

Gürsoy *et al.* (RSI 2023) report a 1 µm position / 0.01° rotation
recovery on Si calibration data but do not give the bound that defines
how many scan points are *needed* for that precision.  This analysis
fills that gap: given the forward model + mask + sample geometry, the
CR-σ tells you the minimum scan length for a given precision target,
*before* you run the experiment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from torch import Tensor

from midas_stress.orientation import quat_to_orient_mat

from typing import TYPE_CHECKING

from ..forward import LaueForwardModel
from ..io import LaueParams
from .mask import CodedApertureMask

if TYPE_CHECKING:
    from ..realdata.depth_resolved import CodedApertureVoxelMeasurement


_PARAM_NAMES = ("pos_x", "pos_y", "pos_z", "rotvec_x", "rotvec_y", "rotvec_z")
# Natural scales used to non-dimensionalise the Hessian for eigenanalysis.
# 1 µm for position, 1° (= π/180 rad) for rotation — roughly the
# precision target of the published Fig. 8.
_NATURAL_SCALES = torch.tensor(
    [1.0, 1.0, 1.0, math.pi / 180.0, math.pi / 180.0, math.pi / 180.0],
    dtype=torch.float64,
)


@dataclass
class LandscapeReport:
    """Eigenanalysis + Cramér–Rao bound on the autofocus loss landscape."""

    H: Tensor                       # (D, D) — Hessian (unscaled)
    H_scaled: Tensor                # (D, D) — D Hess D with D = diag(natural scales)
    eigvals: Tensor                 # (D,)   — ascending eigvals of H_scaled
    eigvecs: Tensor                 # (D, D) — corresponding eigenvectors
    cr_sigma: Tensor                # (D,)   — Cramér-Rao 1-σ in native units
                                    #           (pos [µm], rotvec [rad])
    correlation: Tensor             # (D, D) — symmetric, diag = 1
    loss_at_pose: float
    n_observations: int
    sigma_pixel: float
    param_names: list[str] = field(default_factory=lambda: list(_PARAM_NAMES))


def _build_loss_fn(
    measurements: "Sequence[CodedApertureVoxelMeasurement]",
    *,
    mask_template: CodedApertureMask,
    params: LaueParams,
    hkls: Tensor,
    dtype: torch.dtype,
    device: torch.device,
    E_range: tuple[float, float],
):
    """Closure: theta=(pos[3], rotvec[3]) → MSE loss on the given measurements.

    Reuses the mask buffers for everything *except* position + rotvec,
    which are taken from ``theta``.  The mask itself is rebuilt only
    once per call (cheap; we keep all per-bar buffers in the closure).
    """
    sequence = mask_template.sequence.to(dtype=torch.int64).clone()
    bar_widths_um = mask_template.bar_widths_um.detach().clone()
    au_thickness_um = mask_template.au_thickness_um.detach().clone()
    sub_thickness_um = float(mask_template.sub_thickness_um.item())
    edge_softness_um = mask_template.edge_softness_um

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
    t = params.to_tensors(dtype=dtype, device=str(device))
    lat = t["lattice"]
    P = t["P"]
    R_det = t["R"]

    targets = [v.frame_stack.to(dtype=dtype, device=device) for v in measurements]
    offsets = [v.scan_offsets_um.to(dtype=dtype, device=device) for v in measurements]
    zs = [float(v.z_seed_um) for v in measurements]
    U_seed = measurements[0].U_seed.to(dtype=dtype, device=device).unsqueeze(0)

    def _loss(theta: Tensor) -> Tensor:
        position_um = theta[:3]
        rotvec = theta[3:6]
        mask = CodedApertureMask(
            sequence=sequence,
            bar_widths_um=bar_widths_um,
            au_thickness_um=au_thickness_um,
            sub_thickness_um=sub_thickness_um,
            position_um=position_um,
            rotvec=rotvec,
            edge_softness_um=edge_softness_um,
            make_geometry_learnable=False,
            dtype=dtype,
        )
        # Move any newly-allocated mask buffers to ``device``.
        mask = mask.to(device)
        total = torch.zeros((), dtype=dtype, device=device)
        for tgt, off, z in zip(targets, offsets, zs):
            src = torch.tensor([0.0, 0.0, z * 1.0e-6], dtype=dtype, device=device)
            pred = model.forward_stack(
                U_seed, lat, P, R_det,
                coded_aperture=mask,
                scan_offsets_um=off,
                source_xyz=src,
                E_range=E_range,
            )
            total = total + (pred - tgt).pow(2).mean()
        return total / len(targets)

    return _loss


def autofocus_hessian(
    measurements: "Sequence[CodedApertureVoxelMeasurement]",
    mask: CodedApertureMask,
    *,
    params: LaueParams,
    hkls: Tensor,
    sigma_pixel: float = 0.01,
    E_range: Optional[tuple[float, float]] = None,
) -> LandscapeReport:
    """Compute the 6×6 mask-pose Hessian at the current mask pose.

    Parameters
    ----------
    measurements
        Voxel measurements, as fed to ``autofocus_geometry``.  The
        Hessian is evaluated at the mask's *current* pose — call this
        at the true pose (synthetic) or at convergence (real data) to
        get the local-precision bound.
    mask
        Provides the pose to evaluate at, plus all other mask state
        (sequence, bar widths, thicknesses, edge softness).
    params, hkls
        Forward-model setup.
    sigma_pixel
        Estimated per-pixel image-intensity noise σ.  For synthetic
        round-trips set this to a small fraction of the peak spot
        intensity (default 0.01); for real data use the measured
        Poisson + read-noise σ.
    E_range
        Energy window; default reads from ``params``.

    Returns
    -------
    :class:`LandscapeReport`
    """
    dtype = measurements[0].frame_stack.dtype
    device = measurements[0].frame_stack.device
    erange = E_range or (params.E_lo, params.E_hi)

    loss_fn = _build_loss_fn(
        measurements,
        mask_template=mask,
        params=params,
        hkls=hkls,
        dtype=dtype,
        device=device,
        E_range=erange,
    )

    theta_truth = torch.cat([
        mask.position_um.detach().clone(),
        mask.rotvec.detach().clone(),
    ]).to(dtype=dtype, device=device)

    H = torch.autograd.functional.hessian(
        loss_fn, theta_truth, create_graph=False, vectorize=False,
    )
    H = 0.5 * (H + H.T)             # symmetrise away numerical asymmetry

    # ── eigenanalysis in non-dimensional units ─────────────────────────────
    scales = _NATURAL_SCALES.to(dtype=dtype, device=device)
    D = torch.diag(scales)
    H_scaled = D @ H @ D
    H_scaled = 0.5 * (H_scaled + H_scaled.T)
    eigvals, eigvecs = torch.linalg.eigh(H_scaled)

    # ── Cramér-Rao bound on native parameters ──────────────────────────────
    # Loss is MSE: L = (1/N) Σ (pred - target)².  For Gaussian pixel noise
    # σ_pixel, Fisher info I = (N/(2σ²)) H, so cov(θ) ≥ (2σ²/N) H^{-1}.
    n_obs = 1
    for v in measurements:
        n_obs *= 1   # silence linter
    n_obs = sum(int(v.frame_stack.numel()) for v in measurements)
    H_pinv = torch.linalg.pinv(H.to(torch.float64), rtol=1e-12).to(dtype)
    cov = (2.0 * sigma_pixel ** 2 / n_obs) * H_pinv
    cr_sigma = cov.diag().clamp_min(0.0).sqrt()

    # Correlation matrix from the covariance.
    sd = cr_sigma.clamp_min(1e-30)
    correlation = cov / (sd.unsqueeze(0) * sd.unsqueeze(1))
    # Numerical clean-up
    correlation = correlation.clamp(-1.0, 1.0)

    loss_at = float(loss_fn(theta_truth).detach().item())

    return LandscapeReport(
        H=H.detach().cpu(),
        H_scaled=H_scaled.detach().cpu(),
        eigvals=eigvals.detach().cpu(),
        eigvecs=eigvecs.detach().cpu(),
        cr_sigma=cr_sigma.detach().cpu(),
        correlation=correlation.detach().cpu(),
        loss_at_pose=loss_at,
        n_observations=int(n_obs),
        sigma_pixel=float(sigma_pixel),
    )


def format_report(report: LandscapeReport, *, top_modes: int = 3) -> str:
    """Pretty-print a :class:`LandscapeReport` summary."""
    lines = []
    lines.append(f"loss @ pose: {report.loss_at_pose:.4e}")
    lines.append(
        f"n_observations: {report.n_observations}   "
        f"σ_pixel: {report.sigma_pixel}"
    )
    lines.append("")
    lines.append("Eigenvalues (ascending):")
    for i, e in enumerate(report.eigvals.tolist()):
        lines.append(f"  λ_{i+1} = {e:+.4e}")

    lines.append("")
    lines.append("Cramér-Rao σ per DOF:")
    pos_labels = ("pos_x", "pos_y", "pos_z")
    rot_labels = ("rotvec_x", "rotvec_y", "rotvec_z")
    for label, sigma in zip(pos_labels, report.cr_sigma[:3].tolist()):
        lines.append(f"  {label:>10s}: {sigma:.3e} µm")
    for label, sigma in zip(rot_labels, report.cr_sigma[3:].tolist()):
        deg = math.degrees(sigma)
        lines.append(f"  {label:>10s}: {sigma:.3e} rad  ({deg:.3e}°)")

    lines.append("")
    lines.append(f"Top {top_modes} softest modes (smallest eigenvalues):")
    for k in range(min(top_modes, report.eigvecs.shape[1])):
        v = report.eigvecs[:, k]
        comps = sorted(range(v.numel()), key=lambda j: -abs(v[j].item()))[:3]
        cstr = ", ".join(
            f"{report.param_names[j]}={v[j].item():+.3f}" for j in comps
        )
        lines.append(f"  λ_{k+1} = {report.eigvals[k].item():+.3e}: {cstr}")

    return "\n".join(lines)
