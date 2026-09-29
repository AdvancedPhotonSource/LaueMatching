"""Rotation, lattice, and strain helpers — JAX port of laue_torch.geometry.

Math identical to ``laue_torch/geometry.py`` (which mirrors
``laue_index/pipeline/GenerateSimulation.py`` and
``packages/laue_index/c_src/LaueMatchingCPU.c``), expressed as
JAX ops so the whole chain is differentiable in one framework.

Run with ``jax.config.update("jax_enable_x64", True)`` to match the float64
numerics of the torch reference.
"""

from __future__ import annotations

import math
from typing import Optional

import jax.numpy as jnp
import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────

HC_KEV_NM = 1.2398419739
"""Planck constant times c, expressed in keV*nm."""


# ── Rotation parameterizations → 3x3 matrix ────────────────────────────────

def rodrigues_to_matrix(rvec):
    """Axis-angle (Rodrigues) vector → rotation matrix.

    rvec shape (..., 3). Direction is the axis; magnitude is the angle in
    radians (matches the R_Array convention).

    Smooth at zero, like ``laue_torch.geometry.rodrigues_to_matrix``. The old
    version normalised the axis and switched to I with ``jnp.where`` below
    1e-12: the value was right but the gradient at rvec = 0 was NaN (reverse
    mode: 0/0 through the unselected branch) or 0 (forward mode), where the
    analytic limit is ``dR/d rvec_k = [e_k]x``. Refining a delta composed onto
    a seed starts at exactly 0, so that gradient matters. This form evaluates
    ``sin(θ)/θ`` and ``(1-cos θ)/θ² = ½ (sin(θ/2)/(θ/2))²`` (no cancellation,
    also in float32) with θ = sqrt(|r|² + eps), so no quantity is divided by
    a vanishing norm.
    """
    eps = 1e-12
    theta = jnp.sqrt(jnp.sum(rvec * rvec, axis=-1, keepdims=True) + eps)
    sinc = jnp.sin(theta) / theta                                  # sin θ / θ
    half = 0.5 * theta
    cosc = 0.5 * (jnp.sin(half) / half) ** 2                      # (1 - cos θ) / θ²
    x, y, z = rvec[..., 0], rvec[..., 1], rvec[..., 2]
    zero = jnp.zeros_like(x)
    K = jnp.stack([
        jnp.stack([zero, -z, y], axis=-1),
        jnp.stack([z, zero, -x], axis=-1),
        jnp.stack([-y, x, zero], axis=-1),
    ], axis=-2)
    eye = jnp.eye(3, dtype=rvec.dtype)
    return eye + sinc[..., None] * K + cosc[..., None] * jnp.matmul(K, K)


def quat_to_matrix(q):
    """Quaternion (w, x, y, z) → rotation matrix. Auto-normalized."""
    q = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-30)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = jnp.stack([
        1 - 2 * (y * y + z * z),  2 * (x * y - z * w),      2 * (x * z + y * w),
        2 * (x * y + z * w),      1 - 2 * (x * x + z * z),  2 * (y * z - x * w),
        2 * (x * z - y * w),      2 * (y * z + x * w),      1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def sixd_to_matrix(d6):
    """6D continuous representation (Zhou et al. 2019) → rotation matrix."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / jnp.maximum(jnp.linalg.norm(a1, axis=-1, keepdims=True), 1e-30)
    a2_proj = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2_proj / jnp.maximum(jnp.linalg.norm(a2_proj, axis=-1, keepdims=True), 1e-30)
    b3 = jnp.cross(b1, b2, axis=-1)
    return jnp.stack([b1, b2, b3], axis=-1)


def to_rotation_matrix(U):
    """Dispatch on last-dim size: 3→rodrigues, 4→quaternion, 6→6D, (3,3)→identity."""
    if U.ndim >= 2 and U.shape[-2:] == (3, 3):
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

RHOMBOHEDRAL_SPACE_GROUPS = frozenset({146, 148, 155, 160, 161, 166, 167})
"""The R-centred trigonal space groups (hexagonal OR rhombohedral axes)."""

_SETTING_ANGLE_TOL_DEG = 1e-4
_SETTING_LENGTH_RTOL = 1e-6


def lattice_setting(lattice, sg_num: Optional[int]) -> str:
    """``"standard"`` or ``"rhombohedral"``; same rule as laue_torch.

    ``sg_num=None`` or a non-R space group -> ``"standard"`` (a along x). For
    SG 146/148/155/160/161/166/167: hexagonal axes (alpha = beta = 90,
    gamma = 120) -> ``"standard"``; rhombohedral axes (a = b = c,
    alpha = beta = gamma != 90) -> ``"rhombohedral"`` (3-fold along [111], as
    C ``calcRecipArray``); anything else raises ``ValueError``. The lattice
    must be concrete (not a traced value) when ``sg_num`` is an R group.
    """
    if sg_num is None or int(sg_num) not in RHOMBOHEDRAL_SPACE_GROUPS:
        return "standard"
    try:
        lat = np.asarray(lattice, dtype=np.float64).reshape(-1, 6)
    except Exception as exc:        # a jit tracer cannot be classified
        raise ValueError(
            f"sg_num={sg_num} needs a concrete lattice to choose between "
            f"hexagonal and rhombohedral axes; pass the lattice as a constant "
            f"(closure / static) rather than a traced argument") from exc
    a, b, c = lat[:, 0], lat[:, 1], lat[:, 2]
    al, be, ga = lat[:, 3], lat[:, 4], lat[:, 5]
    tol = _SETTING_ANGLE_TOL_DEG
    hexagonal = ((np.abs(al - 90.0) < tol) & (np.abs(be - 90.0) < tol)
                 & (np.abs(ga - 120.0) < tol))
    equal_len = ((np.abs(b - a) <= _SETTING_LENGTH_RTOL * np.abs(a))
                 & (np.abs(c - a) <= _SETTING_LENGTH_RTOL * np.abs(a)))
    rhombohedral = (equal_len & (np.abs(be - al) < tol) & (np.abs(ga - al) < tol)
                    & (np.abs(al - 90.0) >= tol))
    if hexagonal.all():
        return "standard"
    if rhombohedral.all():
        return "rhombohedral"
    raise ValueError(
        f"space group {int(sg_num)} is R-centred: give the cell on hexagonal "
        f"axes (a, a, c, 90, 90, 120) or rhombohedral axes (a, a, a, alpha, "
        f"alpha, alpha with alpha != 90); got {lat.tolist()}")


def reciprocal_matrix(lattice, sg_num: Optional[int] = None):
    """Reciprocal-lattice matrix B0 (columns are a*, b*, c*).

    lattice shape (..., 6) holds (a, b, c, alpha, beta, gamma). Lengths in nm,
    angles in degrees. Returns B0 in 1/nm. Mirrors ``GenerateHKLs.calcRecipArray``
    (packages/laue_index/laue_index/pipeline/GenerateHKLs.py).

    ``sg_num`` (optional): picks the embedding via :func:`lattice_setting`;
    ``None`` (default) is a along x for every cell, as before 0.1.2.
    """
    setting = lattice_setting(lattice, sg_num)
    a, b, c = lattice[..., 0], lattice[..., 1], lattice[..., 2]
    alpha = lattice[..., 3] * (math.pi / 180.0)
    beta = lattice[..., 4] * (math.pi / 180.0)
    gamma = lattice[..., 5] * (math.pi / 180.0)
    ca, cb, cg = jnp.cos(alpha), jnp.cos(beta), jnp.cos(gamma)
    sg = jnp.sin(gamma)
    phi = jnp.sqrt(jnp.maximum(1.0 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg, 1e-30))
    Vc = a * b * c * phi
    pv = (2 * math.pi) / Vc

    if setting == "standard":
        z = jnp.zeros_like(a)
        a0, a1, a2 = a, z, z
        b0, b1, b2 = b * cg, b * sg, z
        c0 = c * cb
        c1 = c * (ca - cb * cg) / sg
        c2 = c * phi / sg
    else:
        # C calcRecipArray rhombohedral branch: symmetric about [111].
        p = jnp.sqrt(1.0 + 2 * ca)
        q = jnp.sqrt(1.0 - ca)
        pmq = (a / 3.0) * (p - q)
        p2q = (a / 3.0) * (p + 2 * q)
        a0, a1, a2 = p2q, pmq, pmq
        b0, b1, b2 = pmq, p2q, pmq
        c0, c1, c2 = pmq, pmq, p2q

    col0 = jnp.stack([b1 * c2 - b2 * c1, b2 * c0 - b0 * c2, b0 * c1 - b1 * c0], axis=-1) * pv[..., None]
    col1 = jnp.stack([c1 * a2 - c2 * a1, c2 * a0 - c0 * a2, c0 * a1 - c1 * a0], axis=-1) * pv[..., None]
    col2 = jnp.stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], axis=-1) * pv[..., None]
    return jnp.stack([col0, col1, col2], axis=-1)


# ── Strain ──────────────────────────────────────────────────────────────────

def voigt_to_symmetric(eps_v):
    """Voigt-6 (e11, e22, e33, e23, e13, e12) → symmetric 3×3."""
    e11, e22, e33 = eps_v[..., 0], eps_v[..., 1], eps_v[..., 2]
    e23, e13, e12 = eps_v[..., 3], eps_v[..., 4], eps_v[..., 5]
    row0 = jnp.stack([e11, e12, e13], axis=-1)
    row1 = jnp.stack([e12, e22, e23], axis=-1)
    row2 = jnp.stack([e13, e23, e33], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def deviatoric5_to_symmetric(eps_d):
    """Deviatoric-5 (e11, e22, e23, e13, e12) → symmetric 3×3 with tr=0."""
    e11, e22 = eps_d[..., 0], eps_d[..., 1]
    e23, e13, e12 = eps_d[..., 2], eps_d[..., 3], eps_d[..., 4]
    e33 = -(e11 + e22)
    row0 = jnp.stack([e11, e12, e13], axis=-1)
    row1 = jnp.stack([e12, e22, e23], axis=-1)
    row2 = jnp.stack([e13, e23, e33], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def strain_to_B(B0, strain, mode):
    """Apply a strain parameterization to a reference B0.

    B = (I − ε)·B0 for voigt/deviatoric; B = F⁻ᵀ·B0 for F. mode 'none' → B0.
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
        Finv_T = jnp.swapaxes(jnp.linalg.inv(F), -1, -2)
        return jnp.matmul(Finv_T, B0)
    else:
        raise ValueError(f"unknown strain mode {mode!r}; expected none|voigt|deviatoric|F")
    eye = jnp.eye(3, dtype=B0.dtype)
    return jnp.matmul(eye - eps, B0)
