"""Differentiable rasterization: pseudo-Voigt / Gaussian splat + soft/hard masks.

JAX port of ``laue_torch.rasterize``. The torch ``index_add_`` scatter becomes
``img.at[idx].add(...)``; the integer window-center choice is wrapped in
``stop_gradient`` so the gradient flows only through the sub-pixel offset, exactly
as the torch ``.detach()`` does.
"""

from __future__ import annotations

import math
import jax
import jax.numpy as jnp


def soft_window(x, lo, hi, tau):
    """Smooth indicator for ``lo < x < hi``: 1 inside, 0 outside."""
    return jax.nn.sigmoid((x - lo) / tau) * jax.nn.sigmoid((hi - x) / tau)


def hard_window(x, lo, hi):
    """Boolean indicator cast to float. Non-differentiable in x."""
    return ((x > lo) & (x < hi)).astype(x.dtype)


# Pseudo-Voigt FWHM-matching constant (see laue_torch.rasterize).
_FWHM_GAUSS_TO_HWHM_LORENTZ = math.sqrt(2.0 * math.log(2.0))


def pseudo_voigt_splat(px, py, intensity, n_pix, sigma, window, eta=0.0,
                       grain_idx=None, n_grains=1):
    """Sub-pixel-accurate pseudo-Voigt splat into an (Nx, Ny) image.

    ``pV(r) = (1 − η)·G(r; σ) + η·L(r; γ)`` with both components FWHM-matched
    (γ = σ·√(2 ln 2)) and unnormalised (peak amplitude 1). ``eta=0`` recovers a
    pure-Gaussian splat. Returns (Nx, Ny), or (n_grains, Nx, Ny) when
    ``grain_idx`` is given with ``n_grains > 1``.
    """
    Nx, Ny = n_pix
    dtype = px.dtype
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")
    r = window // 2

    # Window centers (integer pixel), gradient-stopped — matches torch .detach().
    cx = jax.lax.stop_gradient(jnp.round(px)).astype(jnp.int32)
    cy = jax.lax.stop_gradient(jnp.round(py)).astype(jnp.int32)

    offsets = jnp.arange(-r, r + 1)                              # (W,)
    tx = cx[:, None] + offsets[None, :]                         # (N, W)
    ty = cy[:, None] + offsets[None, :]                         # (N, W)

    valid_x = (tx >= 0) & (tx < Nx)
    valid_y = (ty >= 0) & (ty < Ny)

    dx2 = (tx.astype(dtype) - px[:, None]) ** 2
    dy2 = (ty.astype(dtype) - py[:, None]) ** 2

    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)
    gx = jnp.exp(-dx2 * inv_two_sigma2)                         # (N, W)
    gy = jnp.exp(-dy2 * inv_two_sigma2)                         # (N, W)

    if isinstance(eta, (int, float)) and float(eta) == 0.0:
        tile = gx[:, :, None] * gy[:, None, :] * intensity[:, None, None]
    else:
        gamma = sigma * _FWHM_GAUSS_TO_HWHM_LORENTZ
        inv_gamma2 = 1.0 / (gamma * gamma)
        lx = 1.0 / (1.0 + dx2 * inv_gamma2)
        ly = 1.0 / (1.0 + dy2 * inv_gamma2)
        gauss_2d = gx[:, :, None] * gy[:, None, :]
        lorentz_2d = lx[:, :, None] * ly[:, None, :]
        tile = ((1.0 - eta) * gauss_2d + eta * lorentz_2d) * intensity[:, None, None]

    valid = (valid_x[:, :, None] & valid_y[:, None, :]).astype(dtype)
    tile = tile * valid

    tx_c = jnp.clip(tx, 0, Nx - 1)
    ty_c = jnp.clip(ty, 0, Ny - 1)

    if grain_idx is None or n_grains == 1:
        flat = (tx_c[:, :, None] * Ny + ty_c[:, None, :]).reshape(-1)
        img = jnp.zeros(Nx * Ny, dtype=dtype)
        img = img.at[flat].add(tile.reshape(-1))
        return img.reshape(Nx, Ny)

    g_off = grain_idx.astype(jnp.int32)[:, None, None] * (Nx * Ny)
    flat = (g_off + tx_c[:, :, None] * Ny + ty_c[:, None, :]).reshape(-1)
    img = jnp.zeros(n_grains * Nx * Ny, dtype=dtype)
    img = img.at[flat].add(tile.reshape(-1))
    return img.reshape(n_grains, Nx, Ny)


# Pure-Gaussian alias for API symmetry with laue_torch.
def gaussian_splat(px, py, intensity, n_pix, sigma, window, grain_idx=None, n_grains=1):
    return pseudo_voigt_splat(px, py, intensity, n_pix, sigma, window,
                              eta=0.0, grain_idx=grain_idx, n_grains=n_grains)
