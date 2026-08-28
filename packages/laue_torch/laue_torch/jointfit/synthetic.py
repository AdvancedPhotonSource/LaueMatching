"""Synthetic scenes with known ground truth.

Nothing in this package points at real data until it can recover a field we
built ourselves.  The Ti-64 investigation produced several confident-looking
results that dissolved under a proper control, so the synthetic scenes here are
built to contain the specific structure that fooled the peel:

  * OVERLAPPING clouds -- two grains whose reflections land in one connected
    feature, which is what makes the greedy peel credit both with a fragment;
  * anisotropic STREAKS at the measured Ti-64 spread, not round blobs;
  * a spread large enough that one grain's footprint spans a whole feature.

The projection is a perspective (gnomonic) map of rotated reciprocal vectors.
It is not a full Laue model -- it does not need to be, because what is under
test is the FIT, not the projection -- but it is nonlinear in orientation and it
sends reflections off the detector.

SCALING.  The real detector is 2048 px at a focal length of ~2567 px (Lsd
513 mm / 200 um pixels), giving a ~21 degree acceptance and, with the Bragg
factor of two, J ~ 5000 px/rad.  Rendering 2048^2 in a unit test is wasteful, so
the defaults here use a 256 px detector with focal 500 -- the same ~27 degree
acceptance, but J ~ 500 px/rad.  The default spreads are scaled up by the same
factor (2.2 deg / 0.43 deg instead of 0.22 / 0.043) so the rendered footprint is
the SAME SIZE IN PIXELS as the real one: a ~19 px streak with 5:1 anisotropy.
Pixels are what the fit actually sees, so that is the quantity worth matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

from .design import ROI, build_basis, suggested_window
from .footprint import (
    pixel_covariance,
    pixel_jacobian,
    spread_covariance,
    tangent_rotation,
)

__all__ = ["SyntheticScene", "make_projection", "make_scene", "random_orientations"]

DEG = torch.pi / 180.0


def random_orientations(n: int, seed: int = 0, dtype=torch.float64) -> Tensor:
    """Uniform random rotations via normalized quaternions -> (n, 3, 3)."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        dim=1,
    )


def make_projection(
    gvectors: Tensor,
    n_pix: tuple[int, int],
    focal: float = 500.0,
) -> Callable[[Tensor], Tensor]:
    """Build a projection ``(3, 3) -> (H, 2)``, NaN where a reflection misses.

    The row count is fixed at ``len(gvectors)`` regardless of orientation, which
    the Jacobian machinery requires -- rows simply become NaN when they leave
    the detector.
    """
    nx, ny = n_pix
    cx, cy = 0.5 * (nx - 1), 0.5 * (ny - 1)

    def project(om: Tensor) -> Tensor:
        v = gvectors.to(om.dtype) @ om.transpose(-1, -2)          # (H, 3)
        vz = v[:, 2]
        # Guard the division so the NaN comes from the mask below, not from a
        # divide-by-zero that would poison the Jacobian of valid rows.
        safe_z = torch.where(vz.abs() < 1e-9, torch.ones_like(vz), vz)
        px = focal * v[:, 0] / safe_z + cx
        py = focal * v[:, 1] / safe_z + cy
        ok = (vz > 1e-9) & (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
        nan = torch.full_like(px, float("nan"))
        return torch.stack([torch.where(ok, px, nan), torch.where(ok, py, nan)], dim=-1)

    return project


@dataclass
class SyntheticScene:
    """A rendered field plus everything needed to score a fit against it."""

    image: Tensor                      # (Nx, Ny) noiseless + noise
    clean: Tensor                      # (Nx, Ny) noiseless
    orientations: Tensor               # (G, 3, 3) ground truth
    axes: Tensor                       # (G, 3) spread axes
    sigma_par: float                   # radians
    sigma_perp: float                  # radians
    amplitudes: Tensor                 # (N,) per-spot, ground truth
    project_fn: Callable[[Tensor], Tensor] = field(repr=False)
    gvectors: Tensor = field(repr=False)
    n_pix: tuple[int, int] = (256, 256)
    psf_sigma: float = 1.06

    @property
    def n_grains(self) -> int:
        return int(self.orientations.shape[0])

    def full_roi(self, pad: int = 0) -> ROI:
        return ROI(x0=pad, y0=pad,
                   nx=self.n_pix[0] - 2 * pad, ny=self.n_pix[1] - 2 * pad)


def make_scene(
    n_grains: int = 3,
    n_reflections: int = 14,
    n_pix: tuple[int, int] = (256, 256),
    sigma_par_deg: float = 2.2,
    sigma_perp_deg: float = 0.43,
    psf_sigma: float = 1.06,
    focal: float = 500.0,
    noise: float = 0.0,
    seed: int = 0,
    orientations: Tensor | None = None,
    overlap_pixels: list[tuple[float, float]] | None = None,
    dtype=torch.float64,
) -> SyntheticScene:
    """Render ``n_grains`` streaked grains into one image.

    noise          : standard deviation of additive Gaussian noise, in the same
                     units as the unit-peak footprints (amplitudes are in [1, 3]).
    overlap_pixels : detector positions where EVERY grain is given a reflection,
                     producing one connected cloud that several grains genuinely
                     contribute to.  This is the configuration that breaks the
                     greedy peel, so recovery tests should include it.
    """
    g = torch.Generator().manual_seed(seed)
    oms = (random_orientations(n_grains, seed=seed + 991, dtype=dtype)
           if orientations is None else orientations.to(dtype))

    # Reciprocal vectors are built BACKWARDS from target pixels: sampling them
    # at random and hoping they land does not work, because a random
    # orientation scatters them out of the ~14 degree acceptance cone (only
    # ~1.5% survive).  Choosing target pixels and back-rotating through each
    # grain's orientation guarantees every grain lights up, and lets
    # `overlap_pixels` place several grains on ONE cloud on purpose -- the
    # structure that made the greedy peel credit multiple grains for one
    # physical feature.
    nx, ny = n_pix
    cx, cy = 0.5 * (nx - 1), 0.5 * (ny - 1)
    margin = 0.15 * min(nx, ny)

    def _dirs_from_pixels(tx: Tensor, ty: Tensor) -> Tensor:
        d = torch.stack([(tx - cx) / focal, (ty - cy) / focal,
                         torch.ones_like(tx)], dim=-1)
        return d / d.norm(dim=-1, keepdim=True)

    gvec_list = []
    for i in range(n_grains):
        tx = margin + (nx - 2 * margin) * torch.rand(n_reflections, generator=g,
                                                     dtype=dtype)
        ty = margin + (ny - 2 * margin) * torch.rand(n_reflections, generator=g,
                                                     dtype=dtype)
        if overlap_pixels:
            ox = torch.tensor([p[0] for p in overlap_pixels], dtype=dtype)
            oy = torch.tensor([p[1] for p in overlap_pixels], dtype=dtype)
            tx = torch.cat([tx, ox])
            ty = torch.cat([ty, oy])
        dirs = _dirs_from_pixels(tx, ty)                     # (H_i, 3)
        # v = OM g = d  =>  g = OM^T d
        gvec_list.append(dirs @ oms[i])
    gvectors = torch.cat(gvec_list, dim=0)
    gvectors = gvectors / gvectors.norm(dim=-1, keepdim=True)
    axes = torch.randn(n_grains, 3, generator=g, dtype=dtype)
    axes = axes / axes.norm(dim=-1, keepdim=True)

    project = make_projection(gvectors, n_pix, focal=focal)
    s_par, s_perp = sigma_par_deg * DEG, sigma_perp_deg * DEG

    px_all, py_all, cov_all = [], [], []
    for i in range(n_grains):
        pix = project(oms[i])
        jac = pixel_jacobian(project, oms[i])
        jac = torch.nan_to_num(jac, nan=0.0)
        cov_w = spread_covariance(s_par, s_perp, axes[i])
        cov = pixel_covariance(jac, cov_w, psf_sigma)
        keep = torch.isfinite(pix[:, 0]) & torch.isfinite(pix[:, 1])
        px_all.append(pix[keep, 0])
        py_all.append(pix[keep, 1])
        cov_all.append(cov[keep])

    px = torch.cat(px_all)
    py = torch.cat(py_all)
    cov = torch.cat(cov_all)
    if px.numel() == 0:
        raise ValueError(
            "no reflections landed on the detector; raise n_reflections or "
            "lower the focal length"
        )
    amps = 1.0 + 2.0 * torch.rand(px.shape[0], generator=g, dtype=dtype)

    roi = ROI(x0=0, y0=0, nx=n_pix[0], ny=n_pix[1])
    basis = build_basis(px, py, cov, roi, window=suggested_window(cov))
    clean = (amps @ basis).reshape(n_pix)
    image = clean
    if noise > 0:
        image = clean + noise * torch.randn(*n_pix, generator=g, dtype=dtype)

    return SyntheticScene(
        image=image,
        clean=clean,
        orientations=oms,
        axes=axes,
        sigma_par=s_par,
        sigma_perp=s_perp,
        amplitudes=amps,
        project_fn=project,
        gvectors=gvectors,
        n_pix=n_pix,
        psf_sigma=psf_sigma,
    )


def perturb_orientations(oms: Tensor, angle_rad: float, seed: int = 0) -> Tensor:
    """Rotate each orientation by ``angle_rad`` about a random axis.

    Used to build a realistic starting point: the peel's seeds are close to the
    truth but not exact, and a fit that only works from the exact answer is not
    evidence of anything.
    """
    g = torch.Generator().manual_seed(seed)
    ax = torch.randn(oms.shape[0], 3, generator=g, dtype=oms.dtype)
    ax = ax / ax.norm(dim=-1, keepdim=True)
    return tangent_rotation(ax * angle_rad) @ oms
