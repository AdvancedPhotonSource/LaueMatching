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
import jax.numpy as jnp


# ── Constants ──────────────────────────────────────────────────────────────

HC_KEV_NM = 1.2398419739
"""Planck constant times c, expressed in keV*nm."""


# ── Rotation parameterizations → 3x3 matrix ────────────────────────────────

def rodrigues_to_matrix(rvec):
    """Axis-angle (Rodrigues) vector → rotation matrix.

    rvec shape (..., 3). Direction is the axis; magnitude is the angle in
    radians (matches the R_Array convention).
    """
    theta = jnp.linalg.norm(rvec, axis=-1, keepdims=True)
    safe = jnp.maximum(theta, 1e-30)
    axis = rvec / safe
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c = jnp.cos(theta)[..., 0]
    s = jnp.sin(theta)[..., 0]
    C = 1.0 - c
    R = jnp.stack([
        c + x * x * C,         x * y * C - z * s,     x * z * C + y * s,
        y * x * C + z * s,     c + y * y * C,         y * z * C - x * s,
        z * x * C - y * s,     z * y * C + x * s,     c + z * z * C,
    ], axis=-1).reshape(*rvec.shape[:-1], 3, 3)
    eye = jnp.broadcast_to(jnp.eye(3, dtype=rvec.dtype), R.shape)
    near_zero = (theta[..., 0] < 1e-12)[..., None, None]
    return jnp.where(near_zero, eye, R)


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

def reciprocal_matrix(lattice):
    """Reciprocal-lattice matrix B0 (columns are a*, b*, c*).

    lattice shape (..., 6) holds (a, b, c, alpha, beta, gamma). Lengths in nm,
    angles in degrees. Returns B0 in 1/nm. Mirrors GenerateHKLs.py:55-100.
    """
    a, b, c = lattice[..., 0], lattice[..., 1], lattice[..., 2]
    alpha = lattice[..., 3] * (math.pi / 180.0)
    beta = lattice[..., 4] * (math.pi / 180.0)
    gamma = lattice[..., 5] * (math.pi / 180.0)
    ca, cb, cg = jnp.cos(alpha), jnp.cos(beta), jnp.cos(gamma)
    sg = jnp.sin(gamma)
    phi = jnp.sqrt(jnp.maximum(1.0 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg, 1e-30))
    Vc = a * b * c * phi
    pv = (2 * math.pi) / Vc

    z = jnp.zeros_like(a)
    a0, a1, a2 = a, z, z
    b0, b1, b2 = b * cg, b * sg, z
    c0 = c * cb
    c1 = c * (ca - cb * cg) / sg
    c2 = c * phi / sg

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
