"""Nye's dislocation density tensor from a per-voxel orientation field.

Given a field of crystal orientations ``R(x)`` recovered per voxel (e.g.
the mean orientation of each per-voxel ODF), Nye's tensor

.. math::

    \\alpha_{ij} = \\partial \\omega_i / \\partial x_j

quantifies the local lattice curvature.  In the small-rotation limit
this is the geometrically-necessary-dislocation (GND) density tensor
(Nye 1953; Pantleon 2008).  ``ω(x)`` is the local rotation vector
(axis × angle in radians).

Total GND density per voxel:

.. math::

    \\rho_\\mathrm{GND}(x) = \\| \\alpha(x) \\|_F \\, / \\, b ,

where ``b`` is the Burgers vector magnitude (m).  Per-slip-system GND
density requires a slip-system list and a different projection; this
module supplies the full tensor field, leaving the slip-system step
to the caller.

The orientation field ``R(x)`` may be 1-D, 2-D, or 3-D; spatial
gradients are computed by central differences with one-sided
differences at the boundaries.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from .geometry import rodrigues_to_matrix


# ── Rotation matrix ↔ axis-angle vector ────────────────────────────────────

def matrix_to_rotvec(R: Tensor) -> Tensor:
    """3×3 rotation matrix → axis-angle vector ``ω`` (||ω|| = θ).

    Differentiable; works for ``θ`` away from ±π. Batched on leading
    dims.
    """
    cos_t = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1.0, 1.0)
    theta = torch.arccos(cos_t)
    sin_t = torch.sin(theta)
    near_zero = sin_t.abs() < 1e-7
    safe = torch.where(near_zero, torch.ones_like(sin_t), sin_t)
    axis = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1) / (2.0 * safe.unsqueeze(-1))
    rvec = axis * theta.unsqueeze(-1)
    return torch.where(near_zero.unsqueeze(-1), torch.zeros_like(rvec), rvec)


# ── Spatial gradient via central differences ──────────────────────────────

def _central_diff(field: Tensor, axis: int, dx: float) -> Tensor:
    """Central-difference gradient along ``axis``; one-sided at boundaries."""
    n = field.shape[axis]
    if n < 2:
        raise ValueError(f"need at least 2 samples along axis {axis}, got {n}")
    forward = torch.roll(field, -1, axis)
    backward = torch.roll(field, 1, axis)
    grad = (forward - backward) / (2.0 * dx)
    # Forward / backward at endpoints.
    idx_first = [slice(None)] * field.dim()
    idx_first[axis] = 0
    idx_second = [slice(None)] * field.dim()
    idx_second[axis] = 1
    idx_last = [slice(None)] * field.dim()
    idx_last[axis] = n - 1
    idx_penult = [slice(None)] * field.dim()
    idx_penult[axis] = n - 2
    grad_clone = grad.clone()
    grad_clone[tuple(idx_first)] = (field[tuple(idx_second)] - field[tuple(idx_first)]) / dx
    grad_clone[tuple(idx_last)] = (field[tuple(idx_last)] - field[tuple(idx_penult)]) / dx
    return grad_clone


def relative_rotation_field(R_field: Tensor, R_ref: Tensor | None = None) -> Tensor:
    """Compute the per-voxel rotation **relative to a reference**.

    ``R_field``: (..., 3, 3) field of rotations.  ``R_ref``: (3, 3)
    reference; defaults to ``R_field`` at the first voxel.  Returns the
    per-voxel rotation ``R_rel(x) = R_ref^T · R(x)``, whose axis-angle
    vector ``ω(x)`` measures local curvature relative to the reference.

    This avoids the multi-valued nature of ``matrix_to_rotvec`` for
    rotations far from the identity (which can confuse finite
    differencing across crossings).
    """
    if R_ref is None:
        # Pick the central voxel of the field as reference.
        center = tuple(s // 2 for s in R_field.shape[:-2])
        R_ref = R_field[center]
    return torch.matmul(R_ref.transpose(-1, -2), R_field)


# ── Nye's tensor ───────────────────────────────────────────────────────────

def nye_tensor(
    R_field: Tensor,
    spacing: Sequence[float] | float,
    *,
    R_ref: Tensor | None = None,
) -> Tensor:
    """Compute Nye's dislocation density tensor field from a rotation field.

    ``R_field`` shape: ``(N_x, [N_y, [N_z]], 3, 3)`` — 1-, 2-, or 3-D
    spatial array of rotation matrices.

    ``spacing``: voxel edge length(s); a scalar applies isotropically to
    all spatial axes, otherwise a sequence of length matching the
    spatial dimensions.

    Returns ``α`` of shape ``(*spatial, 3, 3)`` where
    ``α_ij = ∂ω_i / ∂x_j``.

    Convention: the spatial axes are the leading dims; rotation matrix
    is the trailing 3×3.  ``j`` indexes spatial gradient; ``i`` indexes
    the components of ``ω``.
    """
    n_spatial = R_field.dim() - 2
    if isinstance(spacing, (int, float)):
        spacing = (float(spacing),) * n_spatial
    else:
        spacing = tuple(float(s) for s in spacing)
    if len(spacing) != n_spatial:
        raise ValueError(
            f"spacing length {len(spacing)} does not match n_spatial={n_spatial}")

    R_rel = relative_rotation_field(R_field, R_ref=R_ref)
    omega = matrix_to_rotvec(R_rel)                                       # (*spatial, 3)

    # Build ∂ω_i / ∂x_j for each spatial axis j.
    grads = []
    for j, dx_j in enumerate(spacing):
        grads.append(_central_diff(omega, axis=j, dx=dx_j))                # (*spatial, 3)
    # Stack along a new axis-j: result (*spatial, n_spatial, 3) → swap to (*spatial, 3, n_spatial).
    grad_stack = torch.stack(grads, dim=-2)                                # (*spatial, n_spatial, 3)
    alpha = grad_stack.transpose(-1, -2)                                    # (*spatial, 3, n_spatial)
    if n_spatial < 3:
        # Pad the missing spatial axes with zeros so α is always 3×3.
        pad_shape = list(alpha.shape)
        pad_shape[-1] = 3 - n_spatial
        zeros = torch.zeros(pad_shape, dtype=alpha.dtype, device=alpha.device)
        alpha = torch.cat([alpha, zeros], dim=-1)
    return alpha


# ── GND density ────────────────────────────────────────────────────────────

def gnd_density(alpha: Tensor, burgers_m: float) -> Tensor:
    """Total GND density per voxel from Nye's tensor.

    ``ρ_GND = ||α||_F / b``, where ``α`` is in units of 1/length and
    ``b`` is the Burgers vector magnitude in metres.  The result is in
    1/m² (areal dislocation density).
    """
    frob = torch.linalg.matrix_norm(alpha, ord="fro")
    return frob / burgers_m


def slip_system_gnd(
    alpha: Tensor,
    slip_systems: Tensor,
    burgers_m: float,
) -> Tensor:
    """Per-slip-system GND density via L2 minimisation against ``α``.

    ``slip_systems`` is a ``(S, 3, 3)`` tensor where each slice is the
    rank-1 projection ``b ⊗ ξ`` for one slip system (with ``b`` the
    Burgers direction and ``ξ`` the line direction, both unit
    vectors).  Returns ``ρ_α`` of shape ``(*spatial, S)`` —
    nonnegativity is *not* imposed; for an L1-positive solver use
    ``slip_system_gnd_l1`` (TODO).
    """
    # Flatten α and slip-system projections to 9-vectors and solve a
    # least-squares system per voxel.
    flat_alpha = alpha.reshape(*alpha.shape[:-2], 9)                       # (*spatial, 9)
    flat_sys = slip_systems.reshape(slip_systems.shape[0], 9)              # (S, 9)
    # Min-norm solution: ρ = (S Sᵀ)⁻¹ S α
    G = flat_sys @ flat_sys.T                                              # (S, S)
    rhs = torch.einsum("ij,...j->...i", flat_sys, flat_alpha)              # (*spatial, S)
    rho = torch.linalg.solve(G, rhs.unsqueeze(-1)).squeeze(-1)              # (*spatial, S)
    return rho / burgers_m


# ── Sanity-check helpers ───────────────────────────────────────────────────

def fcc_slip_systems(dtype: torch.dtype = torch.float64) -> Tensor:
    """The 12 FCC slip systems as rank-1 ``b ⊗ ξ`` projections.

    FCC slip is on ⟨110⟩{111} — 4 close-packed planes × 3 close-packed
    directions = 12 systems.  Each is returned as a 3×3 outer product
    of the unit Burgers vector with the unit line direction; ``α``
    decomposes as a non-negative weighted sum of these.

    Returns ``(12, 3, 3)``.
    """
    planes = torch.tensor([
        [ 1.0,  1.0,  1.0],
        [-1.0,  1.0,  1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0,  1.0, -1.0],
    ], dtype=dtype)
    directions_per_plane = [
        # plane (1,1,1):
        ([1.0, -1.0,  0.0], [1.0,  0.0, -1.0], [0.0,  1.0, -1.0]),
        # plane (-1,1,1):
        ([1.0,  1.0,  0.0], [1.0,  0.0,  1.0], [0.0,  1.0, -1.0]),
        # plane (1,-1,1):
        ([1.0,  1.0,  0.0], [1.0,  0.0, -1.0], [0.0,  1.0,  1.0]),
        # plane (1,1,-1):
        ([1.0, -1.0,  0.0], [1.0,  0.0,  1.0], [0.0,  1.0,  1.0]),
    ]
    systems = []
    for n_idx, plane in enumerate(planes):
        for b in directions_per_plane[n_idx]:
            b_t = torch.tensor(b, dtype=dtype)
            b_t = b_t / b_t.norm()
            xi = torch.linalg.cross(plane, b_t)
            xi = xi / xi.norm().clamp_min(1e-30)
            systems.append(b_t.unsqueeze(-1) * xi.unsqueeze(-2))
    return torch.stack(systems, dim=0)


def synthetic_linear_gradient_field(
    n_voxels: int,
    *,
    axis_index: int = 2,
    rate_per_voxel_deg: float = 0.1,
    spacing: float = 1.0,
    U_base: Tensor | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor]:
    """Build a 1-D rotation field varying linearly about a chosen axis.

    The constructed rotation field is

    .. math::
        R(z) = R_\\mathrm{axis}(\\theta(z)) \\, U_\\mathrm{base},

    so each voxel is the base orientation rotated about ``axis`` by
    ``θ(z) = z · rate``.  When :func:`nye_tensor` uses the central
    voxel as the reference, the resulting relative rotation is

    .. math::
        R_\\mathrm{rel}(z) = U_\\mathrm{base}^\\top \\, R_\\mathrm{axis}(\\theta(z))
                            \\, U_\\mathrm{base}
                          = R_{\\, U_\\mathrm{base}^\\top \\mathbf{e}_\\mathrm{axis}}(\\theta(z))

    so the expected lab-frame Nye tensor is
    ``α[:, 0] = (U_base^T · e_axis) · rate``.

    Returns ``(R_field, alpha_analytic)``: ``R_field`` shape
    ``(n_voxels, 3, 3)``; ``alpha_analytic`` shape ``(3, 3)`` is the
    constant Nye tensor expected from the linear gradient.
    """
    import math
    if U_base is None:
        U_base = torch.eye(3, dtype=dtype)
    angles = torch.arange(n_voxels, dtype=dtype) * math.radians(rate_per_voxel_deg)
    angles = angles - angles.mean()                                        # centred
    axis = torch.zeros(3, dtype=dtype)
    axis[axis_index] = 1.0
    rvecs = angles.unsqueeze(-1) * axis                                    # (N, 3)
    R_axis_field = rodrigues_to_matrix(rvecs)                              # (N, 3, 3)
    R_field = R_axis_field @ U_base.unsqueeze(0)                           # (N, 3, 3)

    # Effective rotation axis after conjugation by U_base.
    axis_eff = U_base.T @ axis                                             # (3,)
    rate_per_unit_length = math.radians(rate_per_voxel_deg) / spacing
    alpha = torch.zeros(3, 3, dtype=dtype)
    alpha[:, 0] = axis_eff * rate_per_unit_length
    return R_field, alpha
