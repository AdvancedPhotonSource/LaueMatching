"""Symmetry-reduced misorientation, delegated to ``midas_stress``.

This module is a UNITS ADAPTER, not an implementation. The symmetry folding
lives in :mod:`midas_stress.orientation`, which dispatches on torch tensors
automatically, returns tensors on the input's device and dtype, and stays
differentiable end-to-end.

.. warning::

   ``midas_stress`` returns misorientation angles in **RADIANS** (as does the
   rest of MIDAS). Every caller in this package wants **DEGREES**. That
   conversion happens here, once. Doing it at the call sites instead is how a
   57x error gets in: a misorientation silently reported as 0.01 deg instead of
   0.6 deg reads as excellent convergence rather than as a bug.

The previous hand-rolled implementation was verified to be numerically identical
to this one before being deleted -- the two symmetry groups match as sets (24/24
proper rotations) and agree to 4.6e-13 deg over 500 random pairs. That
equivalence is locked in as a contract test in ``tests/test_symmetry_contract.py``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from midas_stress import make_symmetries, misorientation_om_batch
from midas_stress.orientation import quat_to_orient_mat

__all__ = ["misorientation_deg", "cubic_misorientation_deg",
           "symmetry_operators", "nearest_variant"]

#: Space group 225 (Fm-3m) -- the 24 proper rotations of the cubic point group.
CUBIC_SPACE_GROUP = 225

_RAD2DEG = 180.0 / math.pi


# A lattice per crystal system for callers that give only a space group. The
# frame change depends on the lattice only through its angles (and on
# rhombohedral axes, which are refused below), not its lengths.
_SYSTEM_LATTICE = ((2, (1.0, 1.1, 1.2, 80.0, 95.0, 105.0)), (15, (1.0, 1.1, 1.2, 90.0, 104.0, 90.0)),
                   (74, (1.0, 1.1, 1.2, 90.0, 90.0, 90.0)), (142, (1.0, 1.0, 1.2, 90.0, 90.0, 90.0)),
                   (194, (1.0, 1.0, 1.6, 90.0, 90.0, 120.0)), (230, (1.0, 1.0, 1.0, 90.0, 90.0, 90.0)))


def midas_frame_rotation(space_group: int, lattice=None) -> Tensor:
    """Rotation ``M`` (float64) with ``v_here = M @ v_midas`` for crystal vectors.

    laue_torch builds B with a along x (``geometry.reciprocal_matrix``, as the
    C indexer); midas_stress with a* along x. MIDAS's B is upper triangular
    with a positive diagonal, so ``B_here = M @ B_midas`` is the QR
    factorisation of ``B_here``. The identity for orthogonal cells; a 30 deg
    turn about c for trigonal/hexagonal ones. On rhombohedral axes
    midas_stress's operators (which assume hexagonal axes) do not apply.
    """
    from .geometry import reciprocal_matrix
    sg = int(space_group)
    if lattice is None:
        lattice = next(l for hi, l in _SYSTEM_LATTICE if sg <= hi)
    lat = [float(v) for v in lattice]
    if sg in (146, 148, 155, 160, 161, 166, 167) and abs(lat[3] - 90.0) >= 1.0 \
            and abs(lat[3] - lat[4]) < 1.0 and abs(lat[4] - lat[5]) < 1.0:
        raise ValueError(f"SG {sg} on rhombohedral axes: midas_stress operators assume "
                         f"hexagonal axes; give the lattice on hexagonal axes")
    B = reciprocal_matrix(torch.tensor(lat, dtype=torch.float64)).detach()
    Q, R = torch.linalg.qr(B)
    s = torch.sign(torch.diagonal(R))
    s[s == 0] = 1.0
    return Q * s


def misorientation_deg(M1: Tensor, M2: Tensor,
                       space_group: int = CUBIC_SPACE_GROUP, lattice=None) -> Tensor:
    """Misorientation modulo crystal symmetry, in DEGREES.

    Parameters
    ----------
    M1, M2 : Tensor
        Rotation matrices (crystal -> lab, laue_torch's crystal frame), shape
        (..., 3, 3). ``M1`` broadcastable against ``M2``.
    space_group : int
        Space group number 1-230, used to pick the symmetry operators.
        Defaults to 225 (cubic/FCC).
    lattice : sequence of 6, optional
        a, b, c, alpha, beta, gamma. Fixes the frame change for monoclinic,
        trigonal and hexagonal cells; without it a lattice of the right
        crystal system is assumed (the frame depends only on the angles).

    Returns
    -------
    Tensor
        Angle in degrees, batched over the leading dimensions.

    Notes
    -----
    midas_stress works with a* along x; both orientations are moved into its
    frame (``U @ M``) first. Before 0.1.5 they were not, and for the trigonal
    groups a symmetry-equivalent pair read as 60 deg apart.
    """
    M = midas_frame_rotation(space_group, lattice).to(dtype=M1.dtype, device=M1.device)
    angle_rad = misorientation_om_batch(M1 @ M, M2 @ M, space_group)
    if not isinstance(angle_rad, Tensor):
        angle_rad = torch.as_tensor(angle_rad, dtype=M1.dtype, device=M1.device)
    return angle_rad * _RAD2DEG


def cubic_misorientation_deg(M1: Tensor, M2: Tensor) -> Tensor:
    """Misorientation modulo cubic symmetry, in DEGREES.

    Thin alias for ``misorientation_deg(M1, M2, space_group=225)``, kept
    because it names the common case at the call sites.
    """
    return misorientation_deg(M1, M2, CUBIC_SPACE_GROUP)


def symmetry_operators(space_group: int, *, lattice=None, dtype: torch.dtype = torch.float64,
                       device=None) -> Tensor:
    """Proper rotations of the space group's Laue class, ``(n, 3, 3)``, in
    laue_torch's crystal frame (a along x). ``U @ S`` is the same orientation
    as ``U`` for every ``S`` returned.

    From :func:`midas_stress.make_symmetries`, conjugated into this frame
    (``M S M^T``, :func:`midas_frame_rotation`); ``lattice`` as in
    :func:`misorientation_deg`.
    """
    M = midas_frame_rotation(space_group, lattice)
    _, quats = make_symmetries(int(space_group))
    mats = [M @ torch.as_tensor(quat_to_orient_mat(list(q)), dtype=torch.float64).reshape(3, 3) @ M.T
            for q in quats]
    return torch.stack(mats).to(dtype=dtype, device=device)


def nearest_variant(U: Tensor, U_ref: Tensor, space_group: int, lattice=None) -> Tensor:
    """The symmetry-equivalent ``U @ S`` closest (Frobenius) to ``U_ref``.

    ``U`` and ``U_ref`` are (3, 3). Used to put neighbouring seeds on the same
    branch before a distance such as ``||U_a - U_b||_F`` is taken, which is
    otherwise not symmetry-reduced.
    """
    S = symmetry_operators(space_group, lattice=lattice, dtype=U.dtype, device=U.device)
    cand = U.unsqueeze(0) @ S                                     # (n, 3, 3)
    d = ((cand - U_ref.unsqueeze(0)) ** 2).sum(dim=(-1, -2))
    return cand[int(torch.argmin(d))]
