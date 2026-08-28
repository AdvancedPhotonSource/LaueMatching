"""Rotation, lattice, and strain helpers — all differentiable."""

from __future__ import annotations

import math
import torch
from torch import Tensor


# ── Constants ──────────────────────────────────────────────────────────────

HC_KEV_NM = 1.2398419739
"""Planck constant times c, expressed in keV*nm."""


# ── Rotation parameterizations → 3x3 matrix ────────────────────────────────

def rodrigues_to_matrix(rvec: Tensor) -> Tensor:
    """Axis-angle (Rodrigues) vector → rotation matrix.

    rvec shape (..., 3). The vector direction is the axis; magnitude is
    the angle in radians (matches the existing R_Array convention).

    Smooth at zero, deliberately. The obvious implementation normalises the
    axis and then patches the singularity with
    ``torch.where(theta < eps, I, R)`` -- but ``torch.where`` is a hard switch
    and autograd propagates only through the SELECTED branch, so at rvec = 0 it
    returns a gradient of 0 instead of the analytic Rodrigues limit
    ``dR[0,1]/d(rvec[2]) = -1``.

    That is not a corner case: refining an orientation by composing a delta onto
    a seed, ``rodrigues_to_matrix(dr) @ U0``, starts at exactly ``dr = 0``. The
    first gradient is zero, so the optimiser never takes a step and the fit
    silently returns its input. Measured before this fix: the gradient was 0 for
    |rvec| < 1e-12 and correct above it.

    This version instead evaluates ``sin(θ)/θ`` and ``(1-cos θ)/θ²`` directly,
    carrying θ through the ratios so no normalisation by a vanishing quantity is
    needed. Both are analytic at θ = 0 and autograd handles the limit. Same
    approach as ``midas_pf_odf.inversion._aa_to_R``, which exists because
    ``midas_grain_odf.odf.axis_angle_to_matrix`` has the ``torch.where`` bug.
    """
    eps = 1e-12
    theta_sq = (rvec ** 2).sum(dim=-1, keepdim=True)
    theta = (theta_sq + eps).sqrt()
    sinc = torch.sin(theta) / theta                       # sin(θ)/θ
    cosc = (1.0 - torch.cos(theta)) / (theta * theta)     # (1-cos θ)/θ²

    zero = torch.zeros_like(rvec[..., 0])
    K = torch.stack([
        torch.stack([zero,          -rvec[..., 2],  rvec[..., 1]], dim=-1),
        torch.stack([rvec[..., 2],   zero,         -rvec[..., 0]], dim=-1),
        torch.stack([-rvec[..., 1],  rvec[..., 0],  zero        ], dim=-1),
    ], dim=-2)

    eye = torch.eye(3, dtype=rvec.dtype, device=rvec.device)
    return eye + sinc.unsqueeze(-1) * K + cosc.unsqueeze(-1) * (K @ K)


def quat_to_matrix(q: Tensor) -> Tensor:
    """Quaternion (w, x, y, z) → rotation matrix. Auto-normalized."""
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-30)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z),  2 * (x * y - z * w),      2 * (x * z + y * w),
        2 * (x * y + z * w),      1 - 2 * (x * x + z * z),  2 * (y * z - x * w),
        2 * (x * z - y * w),      2 * (y * z + x * w),      1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def sixd_to_matrix(d6: Tensor) -> Tensor:
    """6D continuous representation (Zhou et al. 2019) → rotation matrix.

    d6 shape (..., 6). First 3 floats define an unnormalized x-axis; the
    next 3 are projected to be orthogonal and unit. y = (a2 - (b1·a2) b1).
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / torch.linalg.norm(a1, dim=-1, keepdim=True).clamp_min(1e-30)
    a2_proj = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2_proj / torch.linalg.norm(a2_proj, dim=-1, keepdim=True).clamp_min(1e-30)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def to_rotation_matrix(U: Tensor) -> Tensor:
    """Dispatch on last-dim size: 3→rodrigues, 4→quaternion, 6→6D, (3,3)→identity."""
    if U.dim() >= 2 and U.shape[-2:] == (3, 3):
        return U
    n = U.shape[-1]
    if n == 3:
        return rodrigues_to_matrix(U)
    if n == 4:
        return quat_to_matrix(U)
    if n == 6:
        return sixd_to_matrix(U)
    raise ValueError(f"Unknown rotation parameterization with shape {tuple(U.shape)}")


# ── Lattice → reciprocal B0 ────────────────────────────────────────────────

def reciprocal_matrix(lattice: Tensor) -> Tensor:
    """Reciprocal-lattice matrix B0 (columns are a*, b*, c*).

    lattice shape (..., 6) holds (a, b, c, alpha, beta, gamma). Lengths
    in nm, angles in degrees. Returns B0 in 1/nm.

    Mirrors the closed form in scripts/GenerateHKLs.py:55-100.
    """
    a, b, c = lattice[..., 0], lattice[..., 1], lattice[..., 2]
    alpha = lattice[..., 3] * (math.pi / 180.0)
    beta = lattice[..., 4] * (math.pi / 180.0)
    gamma = lattice[..., 5] * (math.pi / 180.0)
    ca, cb, cg = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sg = torch.sin(gamma)
    phi = torch.sqrt((1.0 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg).clamp_min(1e-30))
    Vc = a * b * c * phi
    pv = (2 * math.pi) / Vc

    z = torch.zeros_like(a)
    a0, a1, a2 = a, z, z
    b0, b1, b2 = b * cg, b * sg, z
    c0 = c * cb
    c1 = c * (ca - cb * cg) / sg
    c2 = c * phi / sg

    # Columns of B are (b×c, c×a, a×b) * pv.
    col0 = torch.stack([b1 * c2 - b2 * c1, b2 * c0 - b0 * c2, b0 * c1 - b1 * c0], dim=-1) * pv.unsqueeze(-1)
    col1 = torch.stack([c1 * a2 - c2 * a1, c2 * a0 - c0 * a2, c0 * a1 - c1 * a0], dim=-1) * pv.unsqueeze(-1)
    col2 = torch.stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], dim=-1) * pv.unsqueeze(-1)
    return torch.stack([col0, col1, col2], dim=-1)


# ── Strain ──────────────────────────────────────────────────────────────────

def voigt_to_symmetric(eps_v: Tensor) -> Tensor:
    """Voigt-6 (e11, e22, e33, e23, e13, e12) → symmetric 3×3."""
    e11, e22, e33, e23, e13, e12 = eps_v.unbind(-1)
    row0 = torch.stack([e11, e12, e13], dim=-1)
    row1 = torch.stack([e12, e22, e23], dim=-1)
    row2 = torch.stack([e13, e23, e33], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def deviatoric5_to_symmetric(eps_d: Tensor) -> Tensor:
    """Deviatoric-5 → symmetric 3×3 with tr=0.

    Layout: (e11, e22, e23, e13, e12). e33 is computed as -(e11+e22).
    """
    e11, e22, e23, e13, e12 = eps_d.unbind(-1)
    e33 = -(e11 + e22)
    row0 = torch.stack([e11, e12, e13], dim=-1)
    row1 = torch.stack([e12, e22, e23], dim=-1)
    row2 = torch.stack([e13, e23, e33], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def strain_to_B(B0: Tensor, strain: Tensor | None, mode: str) -> Tensor:
    """Apply a strain parameterization to a reference B0.

    B0     : (..., 3, 3) reference reciprocal-lattice matrix.
    strain : per-grain parameter tensor, shape depends on `mode`.
    mode   : "none" | "voigt" | "deviatoric" | "F".

    Returns the (per-grain-batched if needed) strained B matrix.

    Math:
      B = (I − ε) · B0           for "voigt" / "deviatoric"
      B = F⁻ᵀ · B0               for "F"
    """
    if mode == "none" or strain is None:
        return B0
    if mode == "voigt":
        eps = voigt_to_symmetric(strain)
    elif mode == "deviatoric":
        eps = deviatoric5_to_symmetric(strain)
    elif mode == "F":
        F = strain
        if F.shape[-2:] != (3, 3):
            raise ValueError(f"F mode expects (...,3,3), got {tuple(F.shape)}")
        Finv_T = torch.linalg.inv(F).transpose(-1, -2)
        # Broadcast B0 to match F's batch dims.
        return torch.matmul(Finv_T, B0)
    else:
        raise ValueError(f"unknown strain mode {mode!r}; expected none|voigt|deviatoric|F")
    eye = torch.eye(3, dtype=B0.dtype, device=B0.device)
    return torch.matmul(eye - eps, B0)
