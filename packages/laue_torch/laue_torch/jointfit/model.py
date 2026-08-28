"""The joint model: grains fit together against continuous image intensity.

Outer parameters are per grain and few: a tangent correction to the orientation
(3) plus the spread (sigma_par, sigma_perp, axis).  The per-reflection
amplitudes are NOT outer parameters -- they are solved exactly at every step by
``amplitudes.solve_amplitudes``.  That is what keeps the nonlinear problem at
~6 parameters per grain instead of one per reflection.

The contrast with the peel is the point: the peel assigns each pixel to one
grain and deflates, so a large diffuse cloud gets carved up and its fragments
seed extra grains.  Here every grain that reaches a pixel contributes to it, the
amplitudes arbitrate, and a grain that is not needed simply gets amplitude zero
rather than being credited with a fragment.

JACOBIAN DETACHING.  The footprint shape depends on orientation through
J = d(pixel)/d(omega).  By default J is recomputed each step but DETACHED, so
gradients flow to the spread parameters (a direct, strong dependence) but not
through J's own orientation dependence (weak -- J varies slowly with
orientation, being a derivative of a smooth projection).  This avoids
second-order graphs through ``jacfwd`` at a cost that is negligible for the
step sizes involved.  Pass ``detach_jacobian=False`` to keep the full graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn

from .amplitudes import residual, solve_amplitudes
from .design import ROI, build_basis, spots_in_roi, suggested_window
from .footprint import pixel_covariance, pixel_jacobian, spread_covariance, tangent_rotation

__all__ = ["JointGrainFit", "SpotGeometry", "FitReport", "fit_grains"]

ProjectFn = Callable[[Tensor], Tensor]


@dataclass
class SpotGeometry:
    """Predicted spots for all grains, flattened.

    px, py     : (N,) detector pixel coordinates.
    cov        : (N, 2, 2) pixel-space footprint covariance.
    grain_idx  : (N,) which grain each spot came from.
    valid      : (N,) bool; False where the reflection misses the detector.
    """

    px: Tensor
    py: Tensor
    cov: Tensor
    grain_idx: Tensor
    valid: Tensor


@dataclass
class FitReport:
    """Outcome of an outer optimization.

    Convergence is reported, never asserted: a fit that quietly stopped early
    would show up as a worse residual and be misread as the model failing.
    """

    loss: float
    n_iter: int
    loss_history: list[float]
    n_amplitudes: int
    unconverged_solves: int


class JointGrainFit(nn.Module):
    """Grains with orientation + anisotropic spread, fit jointly to ROIs.

    base_orientations : (G, 3, 3) starting orientations, e.g. the peel output.
                        Refinement is parameterized as a tangent correction to
                        these, so the fit starts exactly at the seed.
    project_fn        : (3, 3) orientation -> (H, 2) pixel coordinates.  Rows
                        for reflections that miss the detector must be NaN, and
                        the row count must not change with orientation.
    psf_sigma         : instrument PSF sigma in pixels (1.06 measured on Si).
    sigma_par/perp    : initial spread in RADIANS.
    """

    def __init__(
        self,
        base_orientations: Tensor,
        project_fn: ProjectFn,
        psf_sigma: float = 1.06,
        sigma_par: float = 0.22 * torch.pi / 180,
        sigma_perp: float = 0.043 * torch.pi / 180,
        axis_init: Tensor | None = None,
        detach_jacobian: bool = True,
        fit_spread: bool = True,
    ) -> None:
        super().__init__()
        if base_orientations.dim() != 3 or base_orientations.shape[-2:] != (3, 3):
            raise ValueError(
                f"base_orientations must be (G, 3, 3), got "
                f"{tuple(base_orientations.shape)}"
            )
        g = base_orientations.shape[0]
        dtype, device = base_orientations.dtype, base_orientations.device
        self.register_buffer("base_orientations", base_orientations.clone())
        self.project_fn = project_fn
        self.psf_sigma = float(psf_sigma)
        self.detach_jacobian = bool(detach_jacobian)

        self.omega = nn.Parameter(torch.zeros(g, 3, dtype=dtype, device=device))
        # Spread is stored as a log so it stays positive without a constraint.
        log_par = torch.full((g,), float(torch.log(torch.tensor(sigma_par))),
                             dtype=dtype, device=device)
        log_perp = torch.full((g,), float(torch.log(torch.tensor(sigma_perp))),
                              dtype=dtype, device=device)
        axis = (torch.zeros(g, 3, dtype=dtype, device=device)
                if axis_init is None else axis_init.clone().to(dtype=dtype, device=device))
        if axis_init is None:
            axis[:, 0] = 1.0
        self.log_sigma_par = nn.Parameter(log_par, requires_grad=fit_spread)
        self.log_sigma_perp = nn.Parameter(log_perp, requires_grad=fit_spread)
        self.axis = nn.Parameter(axis, requires_grad=fit_spread)

    @property
    def n_grains(self) -> int:
        return int(self.base_orientations.shape[0])

    def orientations(self) -> Tensor:
        """Current orientations: tangent correction applied to the seeds."""
        return tangent_rotation(self.omega) @ self.base_orientations

    def spot_geometry(self) -> SpotGeometry:
        """Project every grain's reflections and build their footprints."""
        oms = self.orientations()
        px_l, py_l, cov_l, gidx_l = [], [], [], []
        for g in range(self.n_grains):
            om = oms[g]
            pix = self.project_fn(om)                       # (H, 2), NaN if off
            if pix.dim() != 2 or pix.shape[-1] != 2:
                raise ValueError(
                    f"project_fn must return (H, 2), got {tuple(pix.shape)}"
                )
            if self.detach_jacobian:
                with torch.no_grad():
                    jac = pixel_jacobian(self.project_fn, om.detach())
                jac = jac.detach()
            else:
                jac = pixel_jacobian(self.project_fn, om)
            cov_w = spread_covariance(
                torch.exp(self.log_sigma_par[g]),
                torch.exp(self.log_sigma_perp[g]),
                self.axis[g],
            )
            # jacfwd propagates NaN from off-detector rows; neutralize them for
            # the covariance so eigen-decompositions downstream stay finite.
            jac = torch.nan_to_num(jac, nan=0.0)
            cov_l.append(pixel_covariance(jac, cov_w, self.psf_sigma))
            px_l.append(pix[:, 0])
            py_l.append(pix[:, 1])
            gidx_l.append(torch.full((pix.shape[0],), g, dtype=torch.long,
                                     device=pix.device))
        px = torch.cat(px_l)
        py = torch.cat(py_l)
        valid = torch.isfinite(px) & torch.isfinite(py)
        return SpotGeometry(px, py, torch.cat(cov_l), torch.cat(gidx_l), valid)

    def loss(
        self,
        rois: Sequence[ROI],
        images: Sequence[Tensor],
        geom: SpotGeometry | None = None,
        ridge: float = 0.0,
        margin_sigma: float = 4.0,
        window: int | None = None,
    ) -> tuple[Tensor, dict]:
        """Total squared residual over all ROIs, plus bookkeeping.

        Each ROI is fitted independently in the amplitudes, but every ROI is a
        function of the SAME grain parameters -- that coupling is what makes
        this a joint fit rather than a set of local ones.
        """
        if len(rois) != len(images):
            raise ValueError(f"{len(rois)} rois but {len(images)} images")
        geom = self.spot_geometry() if geom is None else geom

        total = torch.zeros((), dtype=geom.px.dtype, device=geom.px.device)
        n_amp = 0
        unconverged = 0
        for roi, img in zip(rois, images):
            sel = self._select(geom, roi, margin_sigma)
            if sel.numel() == 0:
                # No grain reaches this ROI: all of its intensity is residual.
                total = total + (img.reshape(-1) ** 2).sum()
                continue
            win = window if window is not None else suggested_window(geom.cov[sel])
            basis = build_basis(geom.px[sel], geom.py[sel], geom.cov[sel], roi, win)
            sol = solve_amplitudes(basis, img, ridge=ridge)
            unconverged += int(not sol.converged)
            n_amp += int(sel.numel())
            total = total + residual(basis, img.reshape(-1), sol.amplitudes)
        return total, {"n_amplitudes": n_amp, "unconverged_solves": unconverged}

    @torch.no_grad()
    def grain_amplitudes(
        self,
        rois: Sequence[ROI],
        images: Sequence[Tensor],
        ridge: float = 0.0,
        margin_sigma: float = 4.0,
        window: int | None = None,
    ) -> Tensor:
        """Total fitted amplitude per grain, summed over all ROIs -> (G,).

        This is the joint fit's replacement for the peel's hit count, and it
        behaves better in exactly the case that broke the peel: a grain that is
        not needed to explain the image gets amplitude ~0 rather than being
        credited with a fragment of a cloud some other grain produced.  A near
        zero here is the evidence that a candidate grain is spurious.
        """
        geom = self.spot_geometry()
        out = torch.zeros(self.n_grains, dtype=geom.px.dtype, device=geom.px.device)
        for roi, img in zip(rois, images):
            sel = self._select(geom, roi, margin_sigma)
            if sel.numel() == 0:
                continue
            win = window if window is not None else suggested_window(geom.cov[sel])
            basis = build_basis(geom.px[sel], geom.py[sel], geom.cov[sel], roi, win)
            sol = solve_amplitudes(basis, img, ridge=ridge)
            out.index_add_(0, geom.grain_idx[sel], sol.amplitudes)
        return out

    def _select(self, geom: SpotGeometry, roi: ROI, margin_sigma: float) -> Tensor:
        """Valid spots whose footprint can reach ``roi``."""
        idx = torch.nonzero(geom.valid, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return idx
        # Detached: the ROI membership test is a discrete selection, not part of
        # the objective, and differentiating through it means nothing.
        sigma_max = torch.linalg.eigvalsh(geom.cov[idx].detach()).clamp_min(0).max().sqrt()
        margin = float(margin_sigma * sigma_max)
        local = spots_in_roi(geom.px[idx], geom.py[idx], roi, margin=margin)
        return idx[local]

    def subset(self, indices: Sequence[int]) -> "JointGrainFit":
        """A new model holding only ``indices``, keeping their fitted values.

        Used by model selection to test dropping a grain.  The retained grains
        carry their current orientation correction and spread across, so a drop
        test measures the cost of losing that grain -- not the cost of throwing
        away the fit.
        """
        idx = torch.as_tensor(list(indices), dtype=torch.long,
                              device=self.base_orientations.device)
        out = JointGrainFit(
            self.base_orientations[idx].clone(),
            self.project_fn,
            psf_sigma=self.psf_sigma,
            detach_jacobian=self.detach_jacobian,
            fit_spread=self.log_sigma_par.requires_grad,
        )
        with torch.no_grad():
            out.omega.copy_(self.omega[idx])
            out.log_sigma_par.copy_(self.log_sigma_par[idx])
            out.log_sigma_perp.copy_(self.log_sigma_perp[idx])
            out.axis.copy_(self.axis[idx])
        return out

    @torch.no_grad()
    def jacobian_scale(self) -> float:
        """Median |J| over valid spots, in pixels per radian.

        The outer step size must be set in PIXELS, not radians.  J is ~500
        px/rad on the synthetic scenes and ~5000 px/rad on the real 34-ID-E
        geometry, so an lr that is gentle on one is catastrophic on the other:
        lr=5e-3 moves every spot 2.5 px in the first case and 25 px in the
        second, which destroys the fit rather than refining it.  Use this to
        convert a desired pixel step into a learning rate.
        """
        geom = self.spot_geometry()
        idx = torch.nonzero(geom.valid, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return 1.0
        # Footprint covariance already encodes J; recover a scale from the
        # largest singular value of each spot's Jacobian via the projected
        # covariance is indirect, so recompute directly per grain instead.
        scales = []
        oms = self.orientations()
        for g in range(self.n_grains):
            jac = pixel_jacobian(self.project_fn, oms[g].detach())
            jac = torch.nan_to_num(jac, nan=0.0)
            s = torch.linalg.svdvals(jac)
            s = s[s > 0]
            if s.numel():
                scales.append(float(s.median()))
        return float(torch.tensor(scales).median()) if scales else 1.0

    def n_parameters(self, n_amplitudes: int) -> int:
        """Free parameters: 6 per grain (3 orientation, 3 spread) + amplitudes."""
        per_grain = 6 if self.log_sigma_par.requires_grad else 3
        return self.n_grains * per_grain + n_amplitudes


def fit_grains(
    model: JointGrainFit,
    rois: Sequence[ROI],
    images: Sequence[Tensor],
    n_iter: int = 200,
    lr: float | None = None,
    lr_pixels: float = 0.5,
    ridge: float = 0.0,
    tol: float = 0.0,
    verbose: bool = False,
) -> FitReport:
    """Run the outer optimization with Adam.

    lr        : learning rate in RADIANS.  Leave as None -- see ``lr_pixels``.
    lr_pixels : desired step size in PIXELS, converted via
                ``model.jacobian_scale()``.  This is the safe way to set the
                step, because the same radian lr behaves completely differently
                on different geometries (J ~ 500 px/rad synthetic vs ~5000 on
                34-ID-E).  Setting lr directly is supported but easy to get
                catastrophically wrong.
    tol       : stop when the relative loss improvement falls below this; 0
                disables.  The stopping iteration is reported either way.
    """
    if lr is None:
        lr = lr_pixels / max(model.jacobian_scale(), 1e-12)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    info = {"n_amplitudes": 0, "unconverged_solves": 0}
    prev = None
    it = 0
    for it in range(1, n_iter + 1):
        opt.zero_grad()
        loss, info = model.loss(rois, images, ridge=ridge)
        loss.backward()
        opt.step()
        value = float(loss.detach())
        history.append(value)
        if verbose:
            print(f"  iter {it:4d}  loss {value:.6e}")
        if prev is not None and tol > 0 and abs(prev - value) <= tol * abs(prev):
            break
        prev = value
    return FitReport(
        loss=history[-1] if history else float("nan"),
        n_iter=it,
        loss_history=history,
        n_amplitudes=info["n_amplitudes"],
        unconverged_solves=info["unconverged_solves"],
    )
