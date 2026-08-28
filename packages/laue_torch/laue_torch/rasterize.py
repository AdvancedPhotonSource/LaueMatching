"""Differentiable rasterization: pseudo-Voigt / Gaussian splat + soft / hard masks."""

from __future__ import annotations

import math
import torch
from torch import Tensor


def soft_window(x: Tensor, lo: Tensor | float, hi: Tensor | float, tau: float) -> Tensor:
    """Smooth indicator for `lo < x < hi`. Returns 1 inside, 0 outside.

    tau controls the width of the transition. Use a small tau (relative to
    the unit of x) for a near-hard mask, larger for smoother gradients.
    """
    return torch.sigmoid((x - lo) / tau) * torch.sigmoid((hi - x) / tau)


def hard_window(x: Tensor, lo: Tensor | float, hi: Tensor | float) -> Tensor:
    """Boolean indicator cast to float. Non-differentiable in x."""
    return ((x > lo) & (x < hi)).to(x.dtype)


def gaussian_splat(
    px: Tensor,
    py: Tensor,
    intensity: Tensor,
    n_pix: tuple[int, int],
    sigma: float,
    window: int,
    grain_idx: Tensor | None = None,
    n_grains: int = 1,
) -> Tensor:
    """Sub-pixel Gaussian splat into a (Nx, Ny) image.

    px, py        : (N,) float pixel coords (x is fast / first dim).
    intensity     : (N,) per-spot weight (already includes any soft mask).
    n_pix         : (Nx, Ny).
    sigma         : Gaussian PSF stddev in pixels.
    window        : odd integer; size of the per-spot rendering window.
    grain_idx     : (N,) integer per-spot grain id when n_grains > 1; if
                    given and n_grains>1, returns a stacked (G, Nx, Ny).
    n_grains      : number of grains for the stack mode.

    Returns a tensor:
      - (Nx, Ny)         if n_grains == 1 or grain_idx is None
      - (n_grains, Nx, Ny) otherwise (per-grain images, summing within grain).
    """
    Nx, Ny = n_pix
    device = px.device
    dtype = px.dtype
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")
    r = window // 2

    # Window centers (integer pixel) — detached so gradient flows only through
    # (offset = target - px), not through the index choice.
    cx = torch.round(px).to(torch.long).detach()
    cy = torch.round(py).to(torch.long).detach()

    offsets = torch.arange(-r, r + 1, device=device)            # (W,)
    tx = cx[:, None] + offsets[None, :]                         # (N, W)
    ty = cy[:, None] + offsets[None, :]                         # (N, W)

    valid_x = (tx >= 0) & (tx < Nx)
    valid_y = (ty >= 0) & (ty < Ny)

    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)
    gx = torch.exp(-((tx.to(dtype) - px[:, None]) ** 2) * inv_two_sigma2)  # (N, W)
    gy = torch.exp(-((ty.to(dtype) - py[:, None]) ** 2) * inv_two_sigma2)  # (N, W)

    # (N, W, W) per-spot rendering tile.
    tile = gx[:, :, None] * gy[:, None, :] * intensity[:, None, None]
    valid = (valid_x[:, :, None] & valid_y[:, None, :]).to(dtype)
    tile = tile * valid

    tx_c = tx.clamp(0, Nx - 1)
    ty_c = ty.clamp(0, Ny - 1)

    if grain_idx is None or n_grains == 1:
        flat = tx_c[:, :, None] * Ny + ty_c[:, None, :]                    # (N, W, W)
        img = torch.zeros(Nx * Ny, dtype=dtype, device=device)
        img.index_add_(0, flat.reshape(-1), tile.reshape(-1))
        return img.reshape(Nx, Ny)

    # Stacked: prepend grain offset.
    g_off = grain_idx.to(torch.long) * (Nx * Ny)                            # (N,)
    flat = (g_off[:, None, None]
            + tx_c[:, :, None] * Ny
            + ty_c[:, None, :])                                              # (N, W, W)
    img = torch.zeros(n_grains * Nx * Ny, dtype=dtype, device=device)
    img.index_add_(0, flat.reshape(-1), tile.reshape(-1))
    return img.reshape(n_grains, Nx, Ny)


def anisotropic_gaussian_splat(
    px: Tensor,
    py: Tensor,
    intensity: Tensor,
    cov: Tensor,
    n_pix: tuple[int, int],
    window: int,
    spot_idx: Tensor | None = None,
    n_stack: int = 1,
) -> Tensor:
    """Sub-pixel splat of per-spot ANISOTROPIC Gaussians into a (Nx, Ny) image.

    Generalizes :func:`gaussian_splat` from a single scalar sigma to a per-spot
    2x2 pixel-space covariance.  This is what a streaked reflection needs: an
    orientation spread of sigma_omega maps to a pixel covariance
    ``Sigma = J Sigma_omega J^T + sigma_psf^2 I`` via the Jacobian
    J = d(pixel)/d(omega), which is generally elongated and rotated.

    px, py    : (N,) float pixel coords (x is fast / first dim).
    intensity : (N,) per-spot weight.
    cov       : (N, 2, 2) symmetric positive-definite pixel-space covariance.
    n_pix     : (Nx, Ny).
    window    : odd integer; per-spot rendering window.
    spot_idx  : (N,) integer stack id when n_stack > 1; if given and n_stack > 1
                returns a stacked (n_stack, Nx, Ny).  Use a per-GRAIN id for
                per-grain images, or a per-REFLECTION id to build the design
                matrix of a joint fit with free per-reflection amplitudes.
    n_stack   : number of slices in the stack mode.

    Returns (Nx, Ny), or (n_stack, Nx, Ny) when stacking.

    Normalization matches ``gaussian_splat``: the peak of each spot's kernel is
    ``intensity`` (kernel value 1 at the exact center), NOT unit area, so the
    isotropic case reduces to ``gaussian_splat`` exactly.
    """
    Nx, Ny = n_pix
    device = px.device
    dtype = px.dtype
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")
    if cov.shape[-2:] != (2, 2):
        raise ValueError(f"cov must be (N, 2, 2), got {tuple(cov.shape)}")
    if cov.shape[0] != px.shape[0]:
        raise ValueError(
            f"cov has {cov.shape[0]} spots but px has {px.shape[0]}"
        )
    r = window // 2

    # Window centers detached: gradient flows through (target - px), not the
    # integer index choice.  Same contract as gaussian_splat.
    cx = torch.round(px).to(torch.long).detach()
    cy = torch.round(py).to(torch.long).detach()

    offsets = torch.arange(-r, r + 1, device=device)            # (W,)
    tx = cx[:, None] + offsets[None, :]                         # (N, W)
    ty = cy[:, None] + offsets[None, :]                         # (N, W)

    valid_x = (tx >= 0) & (tx < Nx)
    valid_y = (ty >= 0) & (ty < Ny)

    # Offsets from each spot's sub-pixel center, broadcast over the tile.
    ox = tx.to(dtype) - px[:, None]                             # (N, W)
    oy = ty.to(dtype) - py[:, None]                             # (N, W)

    # Precision matrix; inverting a 2x2 in closed form keeps this differentiable
    # and cheap, and lets us guard the determinant explicitly.
    a = cov[:, 0, 0]
    b = 0.5 * (cov[:, 0, 1] + cov[:, 1, 0])   # symmetrize defensively
    d = cov[:, 1, 1]
    det = a * d - b * b
    if bool(torch.any(det <= 0)):
        raise ValueError("cov must be positive-definite (got det <= 0)")
    inv_a = (d / det)[:, None, None]
    inv_b = (-b / det)[:, None, None]
    inv_d = (a / det)[:, None, None]

    OX = ox[:, :, None]                                         # (N, W, 1)
    OY = oy[:, None, :]                                         # (N, 1, W)
    # Mahalanobis quadratic form  o^T Sigma^-1 o  over the (W, W) tile.
    quad = inv_a * OX * OX + 2.0 * inv_b * OX * OY + inv_d * OY * OY
    tile = torch.exp(-0.5 * quad) * intensity[:, None, None]

    valid = (valid_x[:, :, None] & valid_y[:, None, :]).to(dtype)
    tile = tile * valid

    tx_c = tx.clamp(0, Nx - 1)
    ty_c = ty.clamp(0, Ny - 1)

    if spot_idx is None or n_stack == 1:
        flat = tx_c[:, :, None] * Ny + ty_c[:, None, :]         # (N, W, W)
        img = torch.zeros(Nx * Ny, dtype=dtype, device=device)
        img.index_add_(0, flat.reshape(-1), tile.reshape(-1))
        return img.reshape(Nx, Ny)

    s_off = spot_idx.to(torch.long) * (Nx * Ny)                 # (N,)
    flat = (s_off[:, None, None]
            + tx_c[:, :, None] * Ny
            + ty_c[:, None, :])                                 # (N, W, W)
    img = torch.zeros(n_stack * Nx * Ny, dtype=dtype, device=device)
    img.index_add_(0, flat.reshape(-1), tile.reshape(-1))
    return img.reshape(n_stack, Nx, Ny)


def streak_splat(
    px: Tensor,
    py: Tensor,
    intensity: Tensor,
    axis: Tensor,
    length: Tensor,
    sigma_long: Tensor | float,
    sigma_perp: Tensor | float,
    n_pix: tuple[int, int],
    window: int,
    spot_idx: Tensor | None = None,
    n_stack: int = 1,
) -> Tensor:
    """Splat PSF-blurred uniform SEGMENTS -- the misorientation-gradient shape.

    A Gaussian orientation SPREAD makes a bell-shaped streak.  A smooth
    misorientation GRADIENT (a bent grain) sweeps the reflection at nearly
    constant rate and makes a FLAT-TOPPED one.  Measured on Ti-64 ID6: excess
    kurtosis along the streak has median -0.786 (Gaussian 0, top-hat -1.2), 66%
    of streaks are flat-topped, and top-hat(+PSF) beats a Gaussian head-to-head
    46% to 33%.  An anisotropic Gaussian cannot fit that shape at any width.

    A uniform segment of length L convolved with a Gaussian has a closed form --
    a difference of error functions along the segment, Gaussian across it:

        I(u, v) = 1/2 [erf((u + L/2)/(s√2)) - erf((u - L/2)/(s√2))]
                  * exp(-v^2 / (2 sigma_perp^2))

    so this stays cheap and differentiable, and reduces EXACTLY to
    ``gaussian_splat`` as ``length -> 0`` with sigma_long == sigma_perp
    (asserted in the tests).

    px, py     : (N,) spot centers.
    axis       : (N, 2) streak direction; normalized internally.
    length     : (N,) segment length in pixels (the swept misorientation range).
    sigma_long : blur along the segment -- the PSF, NOT the spread.
    sigma_perp : width across the segment.
    window     : odd; must cover length/2 + several sigma.

    Peak-normalized like the other splats: the kernel is 1 at the center of a
    long segment, so ``intensity`` is the plateau height.
    """
    Nx, Ny = n_pix
    device, dtype = px.device, px.dtype
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")
    if axis.shape != (px.shape[0], 2):
        raise ValueError(
            f"axis must be (N, 2) matching px; got {tuple(axis.shape)}"
        )
    r = window // 2

    u_hat = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    s_long = torch.as_tensor(sigma_long, dtype=dtype, device=device).expand(px.shape[0])
    s_perp = torch.as_tensor(sigma_perp, dtype=dtype, device=device).expand(px.shape[0])
    if bool((s_long <= 0).any()) or bool((s_perp <= 0).any()):
        raise ValueError("sigma_long and sigma_perp must be positive")
    # As L -> 0 the peak-normalized kernel converges to a Gaussian, but the
    # closed form is 0/0 there (numerator and denominator both vanish linearly
    # in L).  Clamping the half-length to a tiny multiple of sigma makes the
    # expression exact to ~1e-12 at L = 0 instead of returning zeros.  The
    # gradient w.r.t. length is zero below that floor, which is harmless: the
    # floor is ~1e-6 px while real streaks are tens of pixels.
    half = 0.5 * length.clamp_min(0.0)

    cx = torch.round(px).to(torch.long).detach()
    cy = torch.round(py).to(torch.long).detach()
    offsets = torch.arange(-r, r + 1, device=device)
    tx = cx[:, None] + offsets[None, :]
    ty = cy[:, None] + offsets[None, :]
    valid_x = (tx >= 0) & (tx < Nx)
    valid_y = (ty >= 0) & (ty < Ny)

    ox = (tx.to(dtype) - px[:, None])[:, :, None]          # (N, W, 1)
    oy = (ty.to(dtype) - py[:, None])[:, None, :]          # (N, 1, W)
    ux = u_hat[:, 0][:, None, None]
    uy = u_hat[:, 1][:, None, None]
    u = ox * ux + oy * uy                                   # along the streak
    vv = -ox * uy + oy * ux                                 # across it

    sl = s_long[:, None, None]
    sp = s_perp[:, None, None]
    hl = torch.maximum(half[:, None, None], 1e-6 * sl)
    root2 = torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    along = 0.5 * (torch.erf((u + hl) / (sl * root2))
                   - torch.erf((u - hl) / (sl * root2)))
    # Peak of the along factor (at u = 0) -- normalize so `intensity` is the
    # plateau height rather than the integral, matching the other splats.
    peak = torch.erf(hl / (sl * root2)).clamp_min(1e-30)
    along = along / peak
    across = torch.exp(-0.5 * (vv / sp) ** 2)
    tile = along * across * intensity[:, None, None]
    tile = tile * (valid_x[:, :, None] & valid_y[:, None, :]).to(dtype)

    tx_c = tx.clamp(0, Nx - 1)
    ty_c = ty.clamp(0, Ny - 1)
    if spot_idx is None or n_stack == 1:
        flat = tx_c[:, :, None] * Ny + ty_c[:, None, :]
        img = torch.zeros(Nx * Ny, dtype=dtype, device=device)
        img.index_add_(0, flat.reshape(-1), tile.reshape(-1))
        return img.reshape(Nx, Ny)

    s_off = spot_idx.to(torch.long) * (Nx * Ny)
    flat = (s_off[:, None, None] + tx_c[:, :, None] * Ny + ty_c[:, None, :])
    img = torch.zeros(n_stack * Nx * Ny, dtype=dtype, device=device)
    img.index_add_(0, flat.reshape(-1), tile.reshape(-1))
    return img.reshape(n_stack, Nx, Ny)


# Pseudo-Voigt FWHM-matching constant: Gaussian FWHM = 2√(2 ln 2) σ,
# Lorentzian FWHM = 2 γ.  Setting γ = σ·√(2 ln 2) gives both components
# the same FWHM, which is the conventional pseudo-Voigt convention so
# that the η mixing parameter is interpreted in the standard sense.
_FWHM_GAUSS_TO_HWHM_LORENTZ = math.sqrt(2.0 * math.log(2.0))


def pseudo_voigt_splat(
    px: Tensor,
    py: Tensor,
    intensity: Tensor,
    n_pix: tuple[int, int],
    sigma,
    window: int,
    eta=0.0,
    grain_idx: Tensor | None = None,
    n_grains: int = 1,
) -> Tensor:
    """Sub-pixel-accurate pseudo-Voigt splat into an (Nx, Ny) image.

    The 2-D pseudo-Voigt is the linear combination
    ``pV(r) = (1 − η)·G(r; σ) + η·L(r; γ)`` with both components
    FWHM-matched (γ = σ·√(2 ln 2)).  Both components are *unnormalised*
    (peak amplitude = 1 at r=0), matching the Gaussian-splat convention
    so that ``intensity`` corresponds to peak amplitude.

    Setting ``eta=0`` recovers :func:`gaussian_splat` exactly.  ``eta``
    may be a Python float or a 0-D / 1-D tensor: a scalar tensor is
    fine for refinement (autograd works through the mixing fraction),
    or a per-spot ``(N,)`` tensor for spot-by-spot variation.

    Parameters
    ----------
    px, py        : (N,) float pixel coords.
    intensity     : (N,) per-spot peak amplitude.
    n_pix         : (Nx, Ny) image size.
    sigma         : Gaussian σ in pixels (scalar tensor or float).
    window        : odd integer; size of the per-spot rendering window.
    eta           : Lorentzian mixing fraction in [0, 1].  0 = pure
                    Gaussian; 1 = pure Lorentzian.  Default 0 keeps
                    the existing behaviour bit-identical to
                    :func:`gaussian_splat`.
    grain_idx     : (N,) integer per-spot grain id when n_grains > 1.
    n_grains      : number of grains for the stack mode.
    """
    Nx, Ny = n_pix
    device = px.device
    dtype = px.dtype
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")
    r = window // 2

    cx = torch.round(px).to(torch.long).detach()
    cy = torch.round(py).to(torch.long).detach()

    offsets = torch.arange(-r, r + 1, device=device)            # (W,)
    tx = cx[:, None] + offsets[None, :]                         # (N, W)
    ty = cy[:, None] + offsets[None, :]                         # (N, W)

    valid_x = (tx >= 0) & (tx < Nx)
    valid_y = (ty >= 0) & (ty < Ny)

    # Squared offsets in each axis (N, W).  Both Gaussian and Lorentzian
    # use these.
    dx2 = (tx.to(dtype) - px[:, None]) ** 2
    dy2 = (ty.to(dtype) - py[:, None]) ** 2

    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)
    gx = torch.exp(-dx2 * inv_two_sigma2)                       # (N, W)
    gy = torch.exp(-dy2 * inv_two_sigma2)                       # (N, W)

    if isinstance(eta, (int, float)) and float(eta) == 0.0:
        # Pure Gaussian path.  Equivalent to gaussian_splat.
        tile = gx[:, :, None] * gy[:, None, :] * intensity[:, None, None]
    else:
        # Pseudo-Voigt path.  γ = σ · √(2 ln 2) keeps both components
        # FWHM-matched, so η has its standard "Lorentzian fraction"
        # interpretation.  The 2-D Lorentzian here is the product of two
        # 1-D Lorentzians (separable), matching the Gaussian convention
        # used elsewhere; this is equivalent to a "2-D pseudo-Voigt" up
        # to overall normalisation.  Both components are unnormalised
        # (peak = 1 at r=0).
        gamma = sigma * _FWHM_GAUSS_TO_HWHM_LORENTZ
        inv_gamma2 = 1.0 / (gamma * gamma)
        lx = 1.0 / (1.0 + dx2 * inv_gamma2)                     # (N, W)
        ly = 1.0 / (1.0 + dy2 * inv_gamma2)                     # (N, W)
        # 2-D profiles
        gauss_2d = gx[:, :, None] * gy[:, None, :]              # (N, W, W)
        lorentz_2d = lx[:, :, None] * ly[:, None, :]            # (N, W, W)
        tile = ((1.0 - eta) * gauss_2d + eta * lorentz_2d) \
                * intensity[:, None, None]
    valid = (valid_x[:, :, None] & valid_y[:, None, :]).to(dtype)
    tile = tile * valid

    tx_c = tx.clamp(0, Nx - 1)
    ty_c = ty.clamp(0, Ny - 1)

    if grain_idx is None or n_grains == 1:
        flat = tx_c[:, :, None] * Ny + ty_c[:, None, :]                    # (N, W, W)
        img = torch.zeros(Nx * Ny, dtype=dtype, device=device)
        img.index_add_(0, flat.reshape(-1), tile.reshape(-1))
        return img.reshape(Nx, Ny)

    g_off = grain_idx.to(torch.long) * (Nx * Ny)                            # (N,)
    flat = (g_off[:, None, None]
            + tx_c[:, :, None] * Ny
            + ty_c[:, None, :])                                              # (N, W, W)
    img = torch.zeros(n_grains * Nx * Ny, dtype=dtype, device=device)
    img.index_add_(0, flat.reshape(-1), tile.reshape(-1))
    return img.reshape(n_grains, Nx, Ny)
