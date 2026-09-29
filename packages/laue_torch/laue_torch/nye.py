"""Lattice curvature and Nye's dislocation density tensor from an orientation field.

Given a field of crystal orientations ``R(x)`` recovered per voxel (e.g.
the mean orientation of each per-voxel ODF), the lattice CURVATURE tensor
is

.. math::

    \\kappa_{ij} = \\partial \\omega_i / \\partial x_j ,

with ``omega(x)`` the local rotation vector (axis x angle, radians) of
``R_ref^T R(x)``. Because the orientation matrix maps crystal to lab
(``q_lab = U B h``), that relative rotation acts on the crystal side and
``omega`` is expressed in the REFERENCE CRYSTAL frame. ``x_j`` is the grid
axis ``j``; :func:`lattice_curvature` with ``frame="crystal"`` (default)
rotates that index into the same crystal frame, so both indices share one
frame.

Nye's tensor (Nye 1953; Pantleon 2008) follows from the curvature, neglecting
elastic-strain gradients:

.. math::

    \\alpha = \\kappa^T - \\mathrm{tr}(\\kappa)\\, I ,

(:func:`nye_alpha`). The two are NOT the same tensor: they differ whenever
``tr(kappa) != 0`` (twist) or ``kappa`` is not symmetric. Before 0.1.5 this
module returned ``kappa`` under the name ``nye_tensor`` (kept as a deprecated
alias of the old, mixed-frame quantity).

Total GND density per voxel:

.. math::

    \\rho_\\mathrm{GND}(x) = \\| \\alpha(x) \\|_F \\, / \\, b ,

where ``b`` is the Burgers vector magnitude (m).

The orientation field ``R(x)`` may be 1-D, 2-D, or 3-D; spatial
gradients are computed by central differences with one-sided
differences at the boundaries. Gradients along grid axes that the field
does not have (1-D / 2-D) are taken as zero, which is an assumption, not a
measurement.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import torch
from torch import Tensor

from .geometry import rodrigues_to_matrix


# ── Rotation matrix ↔ axis-angle vector ────────────────────────────────────

def matrix_to_rotvec(R: Tensor) -> Tensor:
    """3×3 rotation matrix → axis-angle vector ``ω`` (||ω|| = θ).

    Differentiable, including at the identity: ``ω = (θ / sin θ) · v`` with
    ``v = vee(R - Rᵀ)/2`` (= sin θ · axis), ``θ = atan2(|v|, (tr R - 1)/2)``.
    The ratio θ / sin θ is evaluated from its series ``1 + |v|²/6`` below
    1e-6, with guarded denominators, so the Jacobian at R = I is the identity
    (the old ``torch.where(..., 0, ...)`` returned 0 there). Valid for θ away
    from π. Batched on leading dims.
    """
    v = 0.5 * torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)
    c = (R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5
    s_sq = (v * v).sum(-1)
    small = s_sq < 1e-12
    s_safe = torch.sqrt(torch.where(small, torch.ones_like(s_sq), s_sq))
    theta = torch.atan2(s_safe, c)
    factor = torch.where(small, 1.0 + s_sq / 6.0, theta / s_safe)
    return factor.unsqueeze(-1) * v


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
    reference; defaults to ``R_field`` at the CENTRE voxel (index
    ``n // 2`` on every spatial axis).  Returns the per-voxel rotation
    ``R_rel(x) = R_ref^T · R(x)``, whose axis-angle vector ``ω(x)`` (in the
    reference crystal frame) measures local curvature relative to the
    reference.

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

def lattice_curvature(
    R_field: Tensor,
    spacing: Sequence[float] | float,
    *,
    R_ref: Tensor | None = None,
    frame: str = "crystal",
) -> Tensor:
    """Lattice curvature tensor field ``κ_ij = ∂ω_i / ∂x_j``.

    ``R_field`` shape: ``(N_x, [N_y, [N_z]], 3, 3)`` — 1-, 2-, or 3-D
    spatial array of crystal-to-lab orientation matrices. The grid axes
    are taken to be the lab x, y, z axes in that order.

    ``spacing``: voxel edge length(s); a scalar applies isotropically to
    all spatial axes, otherwise a sequence of length matching the
    spatial dimensions.

    ``R_ref``: reference orientation; default the centre voxel.

    ``frame``:
      * ``"crystal"`` (default): both indices in the reference crystal frame,
        ``κ = κ_mixed · R_ref`` (grid axis rotated into the crystal frame).
        Use this for :func:`nye_alpha`, which needs one frame.
      * ``"mixed"``: ``i`` in the reference crystal frame, ``j`` the grid
        (lab) axis. This is exactly what ``nye_tensor`` returned before 0.1.5.

    Returns ``κ`` of shape ``(*spatial, 3, 3)``, units rad / length. Missing
    spatial axes (1-D, 2-D fields) contribute zero gradient.
    """
    if frame not in ("crystal", "mixed"):
        raise ValueError(f"frame must be 'crystal' or 'mixed', got {frame!r}")
    n_spatial = R_field.dim() - 2
    if isinstance(spacing, (int, float)):
        spacing = (float(spacing),) * n_spatial
    else:
        spacing = tuple(float(s) for s in spacing)
    if len(spacing) != n_spatial:
        raise ValueError(
            f"spacing length {len(spacing)} does not match n_spatial={n_spatial}")

    if R_ref is None:
        center = tuple(s // 2 for s in R_field.shape[:-2])
        R_ref = R_field[center]
    R_rel = relative_rotation_field(R_field, R_ref=R_ref)
    omega = matrix_to_rotvec(R_rel)                                       # (*spatial, 3)

    # Build ∂ω_i / ∂x_j for each spatial axis j.
    grads = []
    for j, dx_j in enumerate(spacing):
        grads.append(_central_diff(omega, axis=j, dx=dx_j))                # (*spatial, 3)
    # Stack along a new axis-j: result (*spatial, n_spatial, 3) → swap to (*spatial, 3, n_spatial).
    grad_stack = torch.stack(grads, dim=-2)                                # (*spatial, n_spatial, 3)
    kappa = grad_stack.transpose(-1, -2)                                    # (*spatial, 3, n_spatial)
    if n_spatial < 3:
        # Pad the missing spatial axes with zeros so κ is always 3×3.
        pad_shape = list(kappa.shape)
        pad_shape[-1] = 3 - n_spatial
        zeros = torch.zeros(pad_shape, dtype=kappa.dtype, device=kappa.device)
        kappa = torch.cat([kappa, zeros], dim=-1)
    if frame == "crystal":
        # x_lab = R_ref x_crystal  =>  ∂/∂x_crystal_k = Σ_j ∂/∂x_lab_j R_ref[j, k].
        kappa = kappa @ R_ref.to(kappa.dtype)
    return kappa


def nye_alpha(kappa: Tensor) -> Tensor:
    """Nye's dislocation density tensor from the curvature:
    ``α = κᵀ − tr(κ) I`` (elastic-strain gradients neglected).

    ``kappa`` (..., 3, 3) must have both indices in ONE frame
    (``lattice_curvature(..., frame="crystal")``).
    """
    tr = kappa.diagonal(dim1=-2, dim2=-1).sum(-1)
    eye = torch.eye(3, dtype=kappa.dtype, device=kappa.device)
    return kappa.transpose(-1, -2) - tr[..., None, None] * eye


def nye_tensor(
    R_field: Tensor,
    spacing: Sequence[float] | float,
    *,
    R_ref: Tensor | None = None,
) -> Tensor:
    """DEPRECATED: returns the lattice CURVATURE ``κ_ij = ∂ω_i/∂x_j`` in mixed
    frames (``lattice_curvature(..., frame="mixed")``), not Nye's tensor.

    Kept so old scripts reproduce their numbers. For Nye's tensor use
    ``nye_alpha(lattice_curvature(R_field, spacing))``.
    """
    warnings.warn(
        "laue_torch.nye.nye_tensor returns the lattice curvature kappa in mixed "
        "frames, not Nye's alpha; use lattice_curvature(...) and "
        "nye_alpha(kappa) instead",
        DeprecationWarning, stacklevel=2)
    return lattice_curvature(R_field, spacing, R_ref=R_ref, frame="mixed")


# ── GND density ────────────────────────────────────────────────────────────

def gnd_density(alpha: Tensor, burgers_m: float) -> Tensor:
    """Total GND density per voxel from Nye's tensor (:func:`nye_alpha`).

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
    vectors).  Returns ``ρ_α`` of shape ``(*spatial, S)``, the minimum-norm
    least-squares densities (the FCC set is rank 8 of 12, so the solution is
    not unique; min-norm is one choice, not the physical one).
    Nonnegativity is *not* imposed. ``alpha`` must be Nye's tensor
    (:func:`nye_alpha`) in the same frame as the slip systems (the crystal
    frame for :func:`fcc_slip_systems`).
    """
    # Flatten α and slip-system projections to 9-vectors and solve
    # Σ_s ρ_s S_s = α per voxel in the least-squares, MIN-NORM sense. The 12
    # FCC systems span only 8 dimensions, so the 12x12 Gram matrix of the old
    # normal-equation solve was singular and returned an arbitrary member of
    # the solution family; the pseudo-inverse picks the minimum-norm one.
    flat_alpha = alpha.reshape(*alpha.shape[:-2], 9)                       # (*spatial, 9)
    flat_sys = slip_systems.reshape(slip_systems.shape[0], 9).to(flat_alpha.dtype)  # (S, 9)
    pinv = torch.linalg.pinv(flat_sys.T)                                   # (S, 9)
    rho = torch.einsum("sj,...j->...s", pinv, flat_alpha)                  # (*spatial, S)
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

    so the expected MIXED-frame curvature (``lattice_curvature(...,
    frame="mixed")``: ω in the crystal frame, j the grid axis) is
    ``κ[:, 0] = (U_base^T · e_axis) · rate``.

    Returns ``(R_field, kappa_analytic)``: ``R_field`` shape
    ``(n_voxels, 3, 3)``; ``kappa_analytic`` shape ``(3, 3)`` is that
    constant curvature (not Nye's tensor; see :func:`nye_alpha`).
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
