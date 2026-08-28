"""Differentiable Laue forward projection — JAX port of laue_torch.forward.

Math identical to ``laue_torch/forward.py`` (and scripts/GenerateSimulation.py /
src/LaueMatchingCPU.c), as a single functional entry point ``laue_forward`` so
it composes with JAX-CPFEM in one autodiff graph (Plan A, Option 7.1).

Scope note: the coded-aperture path, ``forward_stack``, and the intensity
pre-filter of the torch model are intentionally omitted — none are needed for
the Plan A synthetic loop, and the full-splat path here is numerically identical
to the torch ``return_aux`` (diagnostic) path. Restore them when porting for
Plan B real-data scale.
"""

from __future__ import annotations

import math
from typing import Optional

import jax.numpy as jnp
from jax.nn import sigmoid

from .geometry import (
    HC_KEV_NM,
    reciprocal_matrix,
    rodrigues_to_matrix,
    quat_to_matrix,
    sixd_to_matrix,
    strain_to_B,
)
from .rasterize import hard_window, soft_window, pseudo_voigt_splat


def _U_to_matrix(U, rotation):
    if rotation == "matrix":
        return U
    if rotation == "quat":
        return quat_to_matrix(U)
    if rotation == "rodrigues":
        return rodrigues_to_matrix(U)
    if rotation == "6d":
        return sixd_to_matrix(U)
    raise ValueError(f"unknown rotation {rotation!r}")


def _R_to_matrix(R, detector_rotation):
    if detector_rotation == "matrix":
        return R
    if detector_rotation == "rodrigues":
        return rodrigues_to_matrix(R)
    if detector_rotation == "quat":
        return quat_to_matrix(R)
    raise ValueError(f"unknown detector_rotation {detector_rotation!r}")


def laue_forward(
    U,
    lattice,
    P,
    R,
    hkls,
    n_pix,
    px_size,
    *,
    strain=None,
    weights=None,
    per_spot_intensity: Optional[jnp.ndarray] = None,
    E_range=(5.0, 30.0),
    rotation: str = "matrix",
    detector_rotation: str = "rodrigues",
    strain_mode: str = "none",
    psf_sigma: float = 2.0,
    psf_eta: float = 0.0,
    render_window: Optional[int] = None,
    hard: bool = False,
    tau_z: float = 1e-3,
    tau_px: float = 0.5,
    tau_E: float = 0.05,
    reduce: str = "sum",
    source_xyz=None,
):
    """Render a differentiable Laue exposure.

    Parameters mirror ``laue_torch.LaueForwardModel`` (constructor + forward call
    merged). ``hkls`` is an integer (H, 3) array; ``n_pix=(Nx, Ny)``;
    ``px_size=(dx, dy)`` in meters. Returns (Nx, Ny) for ``reduce='sum'`` or
    (G, Nx, Ny) for ``reduce='stack'``.
    """
    if reduce not in ("sum", "stack"):
        raise ValueError(f"reduce must be 'sum' or 'stack', got {reduce!r}")
    dtype = lattice.dtype

    if render_window is None:
        rr = int(math.ceil(3 * psf_sigma))
        render_window = 2 * rr + 1
    if render_window % 2 == 0:
        render_window += 1

    U_mat = _U_to_matrix(U, rotation)
    R_mat = _R_to_matrix(R, detector_rotation)
    if U_mat.ndim == 2:
        U_mat = U_mat[None, :, :]
    G = U_mat.shape[0]
    H = hkls.shape[0]

    # B0 from lattice (shared or per-grain).
    if lattice.ndim == 1:
        B0 = reciprocal_matrix(lattice)                       # (3,3)
        B0_g = jnp.broadcast_to(B0[None], (G, 3, 3))
    else:
        B0_g = reciprocal_matrix(lattice)                     # (G,3,3)
        if B0_g.shape[0] != G:
            raise ValueError(f"lattice batch {B0_g.shape[0]} != grains {G}")

    B_g = strain_to_B(B0_g, strain, strain_mode)              # (G,3,3)

    M = jnp.matmul(U_mat, B_g)                                # (G,3,3)
    h = hkls.astype(dtype)                                    # (H,3)
    q = jnp.einsum("gij,hj->ghi", M, h)                       # (G,H,3)

    qlen = jnp.linalg.norm(q, axis=-1)                        # (G,H)
    qlen_safe = jnp.maximum(qlen, 1e-30)
    qhat = q / qlen_safe[..., None]                           # (G,H,3)

    # Bragg reflection of ki = (0,0,1).
    dot = qhat[..., 2]                                        # (G,H)
    kf = jnp.stack([
        -2 * dot * qhat[..., 0],
        -2 * dot * qhat[..., 1],
        1.0 - 2 * dot * qhat[..., 2],
    ], axis=-1)                                               # (G,H,3)

    # Into detector frame: xyz = R^T · kf.
    xyz = jnp.einsum("ji,ghj->ghi", R_mat, kf)                # (G,H,3)

    z = xyz[..., 2]
    z_pos = jnp.maximum(z, 1e-9)
    if source_xyz is None:
        scale = P[2] / z_pos
        proj_x = xyz[..., 0] * scale - P[0]
        proj_y = xyz[..., 1] * scale - P[1]
    else:
        s_det = jnp.einsum("ji,j->i", R_mat, source_xyz)
        t_param = (P[2] - s_det[2]) / z_pos
        proj_x = s_det[0] + t_param * xyz[..., 0] - P[0]
        proj_y = s_det[1] + t_param * xyz[..., 1] - P[1]

    Nx, Ny = n_pix
    dx, dy = px_size
    px = proj_x / dx + 0.5 * (Nx - 1)
    py = proj_y / dy + 0.5 * (Ny - 1)

    sin_theta = -qhat[..., 2]
    sin_theta_safe = jnp.maximum(sin_theta, 1e-30)
    energy = HC_KEV_NM * qlen / (4.0 * math.pi * sin_theta_safe)

    Elo, Ehi = E_range
    if hard:
        mask_z = (z > 0).astype(dtype)
        mask_x = hard_window(px, 0.0, Nx - 1.0)
        mask_y = hard_window(py, 0.0, Ny - 1.0)
        mask_E = hard_window(energy, Elo, Ehi)
        mask_st = (sin_theta > 0).astype(dtype)
    else:
        mask_z = sigmoid(z / tau_z)
        mask_x = soft_window(px, 0.0, Nx - 1.0, tau_px)
        mask_y = soft_window(py, 0.0, Ny - 1.0, tau_px)
        mask_E = soft_window(energy, Elo, Ehi, tau_E)
        mask_st = sigmoid(sin_theta / tau_z)
    mask = mask_z * mask_x * mask_y * mask_E * mask_st        # (G,H)

    if per_spot_intensity is None:
        inten = jnp.ones((G, H), dtype=dtype)
    else:
        inten = per_spot_intensity.astype(dtype)
        if inten.shape != (G, H):
            raise ValueError(f"per_spot_intensity must be (G,H)=({G},{H}), got {tuple(inten.shape)}")
    if weights is not None:
        inten = inten * weights.astype(dtype)[:, None]
    spot_int = inten * mask                                   # (G,H)

    px_f = px.reshape(-1)
    py_f = py.reshape(-1)
    int_f = spot_int.reshape(-1)

    if reduce == "stack":
        grain_idx = jnp.broadcast_to(jnp.arange(G)[:, None], (G, H)).reshape(-1)
        n_grains = G
    else:
        grain_idx = None
        n_grains = 1

    image = pseudo_voigt_splat(
        px_f, py_f, int_f,
        n_pix=n_pix, sigma=psf_sigma, eta=psf_eta, window=render_window,
        grain_idx=grain_idx, n_grains=n_grains,
    )
    return image
