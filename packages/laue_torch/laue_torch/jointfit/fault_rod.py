"""Continuous-L forward model for one (h, k) reciprocal-lattice row.

A stacking fault smears Bragg intensity along a *rod*: fixed in-plane index
(h, k), continuous out-of-plane index L. ``LaueForwardModel`` cannot express
this directly -- it requires an integer ``hkls`` tensor, one row per discrete
reflection -- but the underlying per-spot physics (q = U B0 h, then project
to energy + detector pixel) is continuous in L. This module isolates that
physics for one row so a rod can be swept, differentiated, and windowed
without touching the discrete multi-grain rendering path.

The formulas here are DUPLICATED from ``laue_torch.forward.LaueForwardModel
.forward`` on purpose, not re-derived: an independent derivation could drift
from the package's own convention (incident beam k_i = (0, 0, 1), qlen in the
2*pi/nm convention, energy = HC_KEV_NM * qlen / (4*pi*sin_theta)) without
anyone noticing. ``test_fault_rod_matches_forward_model`` in
``tests/test_fault_rod.py`` cross-checks this module against
``LaueForwardModel`` itself at an integer L, and must keep passing if either
side changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor

from ..geometry import HC_KEV_NM, reciprocal_matrix

__all__ = ["RodPoint", "rod_forward", "rod_local_jacobian", "rod_accessible_mask"]


@dataclass
class RodPoint:
    """Per-L outputs of :func:`rod_forward`. Every field has ``L``'s shape."""

    L: Tensor
    q: Tensor            # (..., 3), 1/nm
    qlen: Tensor         # (...), 1/nm
    sin_theta: Tensor    # (...)
    energy: Tensor       # (...), keV
    px: Tensor           # (...), pixels
    py: Tensor           # (...), pixels
    z: Tensor            # (...); z > 0 is the forward-scatter condition


def rod_forward(
    U: Tensor,
    lattice: Tensor,
    h: float,
    k: float,
    L: Tensor,
    P: Tensor,
    R: Tensor,
    n_pix: tuple[int, int],
    px_size: tuple[float, float],
) -> RodPoint:
    """Energy and detector position for the row (h, k, L), L continuous.

    U        : (3, 3) orientation matrix.
    lattice  : (6,) (a, b, c [nm], alpha, beta, gamma [deg]) -- matches
               ``laue_torch.geometry.reciprocal_matrix``.
    h, k     : the fixed in-plane indices (python floats/ints).
    L        : Tensor, any shape -- the (possibly fractional) index swept
               along the rod. Every returned field has this shape.
    P        : (3,) detector translation [m].
    R        : (3, 3) detector rotation matrix (lab -> detector).
    n_pix    : (Nx, Ny) detector pixel count.
    px_size  : (dx, dy) pixel size [m].

    No aperture, energy-window, or forward-scatter gating is applied here --
    see :func:`rod_accessible_mask`. Reflections that are geometrically
    impossible (e.g. sin_theta <= 0) still produce a finite, if physically
    meaningless, point: callers must apply the mask before using the output.
    """
    if U.shape[-2:] != (3, 3):
        raise ValueError(f"U must be (3, 3), got {tuple(U.shape)}")
    if lattice.shape[-1] != 6:
        raise ValueError(f"lattice must be (..., 6), got {tuple(lattice.shape)}")

    dtype = L.dtype
    device = L.device
    B0 = reciprocal_matrix(lattice.to(dtype=dtype, device=device))  # (3, 3)
    M = U.to(dtype=dtype, device=device) @ B0                       # (3, 3)

    hk = torch.stack(
        [
            torch.full_like(L, float(h)),
            torch.full_like(L, float(k)),
            L,
        ],
        dim=-1,
    )  # (..., 3)
    q = torch.einsum("ij,...j->...i", M, hk)  # (..., 3)
    qlen = torch.linalg.norm(q, dim=-1).clamp_min(1e-30)
    qhat = q / qlen.unsqueeze(-1)

    # Bragg reflection of k_i = (0, 0, 1) -- identical to LaueForwardModel.
    dot = qhat[..., 2]
    kf = torch.stack(
        [-2 * dot * qhat[..., 0], -2 * dot * qhat[..., 1], 1.0 - 2 * dot * qhat[..., 2]],
        dim=-1,
    )

    R_ = R.to(dtype=dtype, device=device)
    xyz = torch.einsum("ji,...j->...i", R_, kf)

    z = xyz[..., 2]
    z_pos = z.clamp_min(1e-9)
    P_ = P.to(dtype=dtype, device=device)
    scale = P_[2] / z_pos
    proj_x = xyz[..., 0] * scale - P_[0]
    proj_y = xyz[..., 1] * scale - P_[1]

    Nx, Ny = n_pix
    dx, dy = px_size
    px = proj_x / dx + 0.5 * (Nx - 1)
    py = proj_y / dy + 0.5 * (Ny - 1)

    sin_theta = (-qhat[..., 2]).clamp_min(1e-30)
    energy = HC_KEV_NM * qlen / (4.0 * math.pi * sin_theta)

    return RodPoint(L=L, q=q, qlen=qlen, sin_theta=sin_theta, energy=energy, px=px, py=py, z=z)


def rod_local_jacobian(
    U: Tensor,
    lattice: Tensor,
    h: float,
    k: float,
    L0: Union[float, Tensor],
    P: Tensor,
    R: Tensor,
    n_pix: tuple[int, int],
    px_size: tuple[float, float],
) -> dict[str, float]:
    """d(energy, px, py) / dL at a single point L0, via forward-mode autodiff.

    This is what turns a Warren-type stacking-fault broadening (a width in
    *fractional L*) into a predicted energy or pixel width: multiply the
    fault-broadening sigma/FWHM in L by ``dE_dL`` or ``hypot(dpx_dL, dpy_dL)``.

    Same idea as ``jointfit.footprint.pixel_jacobian`` (evaluate the exact
    derivative of the forward map at a point), except the free parameter is
    the scalar L, not an orientation tangent vector.
    """
    from torch.func import jacfwd

    L0_t = torch.as_tensor(float(L0), dtype=torch.float64)

    def f(L: Tensor) -> Tensor:
        pt = rod_forward(U, lattice, h, k, L.reshape(()), P, R, n_pix, px_size)
        return torch.stack([pt.energy, pt.px, pt.py])

    value = f(L0_t)
    jac = jacfwd(f)(L0_t)  # (3,)
    return {
        "energy": float(value[0]),
        "px": float(value[1]),
        "py": float(value[2]),
        "dE_dL": float(jac[0]),
        "dpx_dL": float(jac[1]),
        "dpy_dL": float(jac[2]),
    }


def rod_accessible_mask(
    pt: RodPoint,
    n_pix: tuple[int, int],
    E_range: tuple[float, float],
) -> Tensor:
    """Boolean mask: forward-scattering, on-detector, and inside E_range.

    Does not know about a physical aperture (e.g. a DAC opening) beyond the
    detector footprint and energy window -- callers with a tighter geometric
    constraint (an aperture half-angle, say) must intersect this with their
    own mask.
    """
    Nx, Ny = n_pix
    Elo, Ehi = E_range
    return (
        (pt.z > 0)
        & (pt.px >= 0) & (pt.px <= Nx - 1)
        & (pt.py >= 0) & (pt.py <= Ny - 1)
        & (pt.energy >= Elo) & (pt.energy <= Ehi)
    )
