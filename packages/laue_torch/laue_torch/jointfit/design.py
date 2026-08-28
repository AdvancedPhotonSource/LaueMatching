"""Design matrix: per-(grain, reflection) basis images over regions of interest.

The joint fit writes the observed image as a NON-NEGATIVE SUM of per-reflection
footprints,

    y(pixel) ~ sum_k  a_k * K_k(pixel)

where ``K_k`` is the anisotropic Gaussian footprint of reflection k (position
from the grain orientation, shape from the spread via
``footprint.pixel_covariance``) and ``a_k >= 0`` is its free amplitude.  Because
each reflection gets its own amplitude, absolute intensities -- which would need
|F|^2 * I0(E), and I0(E) is unavailable -- are never predicted.

The problem DECOMPOSES BY ROI: a reflection only touches pixels near its own
footprint, so the amplitude solve runs independently on each region of interest.
What couples the ROIs is the OUTER loop: one grain's orientation determines
reflections in many ROIs at once, which is exactly the constraint the greedy
peel throws away when it assigns each cloud to a single winner.

Memory note: a basis is (K, nx*ny) dense over its ROI only.  A full-detector
dense basis would be K x 2048^2 and is never built.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..rasterize import anisotropic_gaussian_splat

__all__ = [
    "ROI",
    "build_basis",
    "rois_from_labels",
    "spots_in_roi",
    "suggested_window",
]


@dataclass(frozen=True)
class ROI:
    """Axis-aligned region of interest, in pixels.

    ``x`` is the fast / first image dimension, matching ``rasterize``.
    """

    x0: int
    y0: int
    nx: int
    ny: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nx, self.ny)

    @property
    def n_pixels(self) -> int:
        return self.nx * self.ny

    def crop(self, image: Tensor) -> Tensor:
        """Extract this ROI from a full-detector image."""
        return image[self.x0:self.x0 + self.nx, self.y0:self.y0 + self.ny]

    def contains(self, px: Tensor, py: Tensor) -> Tensor:
        return (
            (px >= self.x0) & (px < self.x0 + self.nx)
            & (py >= self.y0) & (py < self.y0 + self.ny)
        )


def rois_from_labels(
    labels: Tensor,
    ids: list[int] | Tensor,
    pad: int = 24,
    clip_to: tuple[int, int] | None = None,
) -> list[ROI]:
    """Bounding-box ROI around each labelled connected feature, padded.

    labels  : (Nx, Ny) integer label image (e.g. from ``scipy.ndimage.label``).
    ids     : label values to build ROIs for.
    pad     : pixels of margin.  A footprint extends well beyond the thresholded
              feature -- the measured Ti-64 streak is ~19 px sigma -- so the pad
              must cover the kernel or the fit is scored on a truncated model.
    clip_to : (Nx, Ny) to clip against; defaults to the label image shape.
    """
    Nx, Ny = clip_to if clip_to is not None else tuple(labels.shape)
    out: list[ROI] = []
    for lid in [int(i) for i in ids]:
        idx = torch.nonzero(labels == lid, as_tuple=False)
        if idx.numel() == 0:
            raise ValueError(f"label {lid} not present in the label image")
        x0 = max(int(idx[:, 0].min()) - pad, 0)
        y0 = max(int(idx[:, 1].min()) - pad, 0)
        x1 = min(int(idx[:, 0].max()) + pad + 1, Nx)
        y1 = min(int(idx[:, 1].max()) + pad + 1, Ny)
        out.append(ROI(x0=x0, y0=y0, nx=x1 - x0, ny=y1 - y0))
    return out


def spots_in_roi(px: Tensor, py: Tensor, roi: ROI, margin: float = 0.0) -> Tensor:
    """Indices of spots whose centers fall in ``roi`` (optionally widened).

    ``margin`` admits spots just outside the ROI whose footprint still spills
    into it.  Pass a few times the largest footprint sigma; dropping those spots
    would leave real intensity unexplained and bias the amplitudes of whatever
    remains.
    """
    keep = (
        (px >= roi.x0 - margin) & (px < roi.x0 + roi.nx + margin)
        & (py >= roi.y0 - margin) & (py < roi.y0 + roi.ny + margin)
    )
    return torch.nonzero(keep, as_tuple=False).squeeze(-1)


def build_basis(
    px: Tensor,
    py: Tensor,
    cov: Tensor,
    roi: ROI,
    window: int,
) -> Tensor:
    """Per-reflection basis images over ``roi`` -> (K, nx*ny).

    px, py : (K,) spot centers in FULL-detector pixel coordinates.
    cov    : (K, 2, 2) pixel-space covariances from ``pixel_covariance``.
    window : odd rendering window; must cover the footprint (see
             :func:`suggested_window`).

    Row k is reflection k's unit-amplitude footprint, flattened.  Differentiable
    in ``px``, ``py`` and ``cov``, so the outer loop optimizes through it.
    """
    if px.shape != py.shape:
        raise ValueError(f"px {tuple(px.shape)} and py {tuple(py.shape)} differ")
    k = px.shape[0]
    if k == 0:
        return torch.zeros(0, roi.n_pixels, dtype=px.dtype, device=px.device)

    # Shift into ROI-local coordinates and splat one stack slice per reflection.
    local_x = px - roi.x0
    local_y = py - roi.y0
    stack = anisotropic_gaussian_splat(
        local_x,
        local_y,
        torch.ones(k, dtype=px.dtype, device=px.device),
        cov,
        roi.shape,
        window=window,
        spot_idx=torch.arange(k, device=px.device),
        n_stack=k,
    )                                              # (K, nx, ny)
    return stack.reshape(k, roi.n_pixels)


def suggested_window(cov: Tensor, n_sigma: float = 4.0) -> int:
    """Smallest odd window covering ``n_sigma`` of the widest footprint.

    Truncating the kernel biases amplitudes low and, worse, biases them
    UNEVENLY -- a long streak loses more of its tail than a compact spot -- so
    the window is sized from the data rather than fixed.
    """
    if cov.numel() == 0:
        return 3
    # Detached: the window size is a discrete rendering choice, not part of the
    # objective, so it must not drag the covariance into the graph.
    sigma_max = float(torch.linalg.eigvalsh(cov.detach()).clamp_min(0).max().sqrt())
    half = max(int(n_sigma * sigma_max + 0.5), 1)
    return 2 * half + 1
