"""Wire-scan (differential-aperture) depth gating for DAXM.

Complements the coded-aperture depth resolution in ``realdata/depth_resolved.py``
with the *classic* DAXM mechanism: a Pt wire is scanned across the diffracted
beams and progressively occludes deeper sub-volumes, so DIFFERENCING successive
wire positions isolates each depth's contribution (triangulation).

Model: a column of ``D`` sub-volumes at increasing depth, each with its own
out-of-plane strain ``eps(z)`` that shifts the Bragg peak radially
(``r(z) = r0 (1 - eps(z))``).  The wire scan gives a cumulative-visibility
matrix ``V`` (n_wire x D); the measured profile stack is ``M = V @ P(eps)`` with
``P[d]`` depth d's peak profile.

The depth-INTEGRATED peak (no wire) is degenerate -- many ``eps(z)`` give the
same broadened peak -- which is exactly why the wire scan is needed.  The fit
reuses the shared ``midas_invert`` gradient primitives.  Torch-differentiable.

(Originally prototyped as the standalone ``midas_daxm`` package; folded into
laue_torch per the MIDAS<->LaueMatching boundary: Laue-specific code lives here,
shared inversion primitives live in ``midas_invert``.)
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "visibility_matrix",
    "per_depth_profiles",
    "wire_scan",
    "integrated_profile",
    "triangulate_depths",
    "depth_centroid_strain",
    "recover_depth_strain",
]


def visibility_matrix(n_depth, *, dtype=None):
    """Cumulative wire visibility ``V`` (n_depth+1, n_depth): ``V[w, d] = 1`` if
    depth d is still illuminated at wire step w (here ``d >= w``), so
    ``M[w] - M[w+1]`` isolates depth ``w``."""
    import torch
    dtype = dtype or torch.float64
    D = int(n_depth)
    V = torch.zeros(D + 1, D, dtype=dtype)
    for w in range(D + 1):
        for d in range(D):
            if d >= w:
                V[w, d] = 1.0
    return V


def per_depth_profiles(eps_z, r_grid, *, r0, width, illum=None):
    """Per-depth radial peak profiles ``P[d, :]`` (Gaussian at ``r0(1-eps_d)``)."""
    import torch
    eps_z = torch.as_tensor(eps_z)
    r_grid = torch.as_tensor(r_grid, dtype=eps_z.dtype, device=eps_z.device)
    centers = float(r0) * (1.0 - eps_z)
    diff = r_grid[None, :] - centers[:, None]
    P = torch.exp(-0.5 * (diff / float(width)) ** 2)
    P = P / (float(width) * math.sqrt(2.0 * math.pi))
    if illum is not None:
        P = P * torch.as_tensor(illum, dtype=P.dtype, device=P.device)[:, None]
    return P


def wire_scan(eps_z, r_grid, *, r0, width, illum=None, V=None):
    """Wire-scan profile stack ``M = V @ P(eps)`` of shape (n_wire, R)."""
    import torch
    eps_z = torch.as_tensor(eps_z)
    if V is None:
        V = visibility_matrix(eps_z.shape[0], dtype=eps_z.dtype)
    V = V.to(dtype=eps_z.dtype, device=eps_z.device)
    P = per_depth_profiles(eps_z, r_grid, r0=r0, width=width, illum=illum)
    return V @ P


def integrated_profile(eps_z, r_grid, *, r0, width, illum=None):
    """Depth-integrated profile (no wire) -- the degenerate, broadened peak."""
    P = per_depth_profiles(eps_z, r_grid, r0=r0, width=width, illum=illum)
    return P.sum(dim=0)


def triangulate_depths(M):
    """Reconstruct per-depth profiles by differencing the wire-scan stack
    (classic DAXM triangulation): depth d = ``M[d] - M[d+1]``."""
    import torch
    M = torch.as_tensor(M)
    return M[:-1] - M[1:]


def depth_centroid_strain(P_rec, r_grid, *, r0):
    """Per-depth strain from each reconstructed profile centroid:
    ``eps_d = 1 - centroid_d / r0``."""
    import torch
    P_rec = torch.as_tensor(P_rec)
    r_grid = torch.as_tensor(r_grid, dtype=P_rec.dtype, device=P_rec.device)
    w = P_rec / P_rec.sum(dim=1, keepdim=True).clamp(min=1e-12)
    centroid = (w * r_grid[None, :]).sum(dim=1)
    return 1.0 - centroid / float(r0)


def recover_depth_strain(M_obs, r_grid, *, r0, width, n_depth, illum=None,
                         steps=800, lr=0.02, init=None):
    """Differentiable recovery of the depth strain profile ``eps(z)`` from a
    measured wire-scan stack ``M_obs`` (n_wire, R).  Returns dict with ``eps``
    (D,) and the loss.  Uses the shared ``midas_invert`` gradient fitter."""
    import torch
    from midas_invert.optimize import fit, relative_l2_loss

    M_obs = torch.as_tensor(M_obs)
    eps = (torch.zeros(n_depth, dtype=M_obs.dtype) if init is None
           else torch.as_tensor(init, dtype=M_obs.dtype).clone())
    eps.requires_grad_(True)

    def loss_fn():
        pred = wire_scan(eps, r_grid, r0=r0, width=width, illum=illum)
        return relative_l2_loss(pred, M_obs)

    out = fit([eps], loss_fn, steps=steps, lr=lr)
    return {"eps": eps.detach(), "loss": out["loss"]}
