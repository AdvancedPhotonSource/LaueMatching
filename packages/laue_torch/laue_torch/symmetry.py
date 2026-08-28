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

from midas_stress import misorientation_om_batch

__all__ = ["misorientation_deg", "cubic_misorientation_deg"]

#: Space group 225 (Fm-3m) -- the 24 proper rotations of the cubic point group.
CUBIC_SPACE_GROUP = 225

_RAD2DEG = 180.0 / math.pi


def misorientation_deg(M1: Tensor, M2: Tensor,
                       space_group: int = CUBIC_SPACE_GROUP) -> Tensor:
    """Misorientation modulo crystal symmetry, in DEGREES.

    Parameters
    ----------
    M1, M2 : Tensor
        Rotation matrices, shape (..., 3, 3). ``M1`` broadcastable against
        ``M2``.
    space_group : int
        Space group number 1-230, used to pick the symmetry operators.
        Defaults to 225 (cubic/FCC).

    Returns
    -------
    Tensor
        Angle in degrees, batched over the leading dimensions.

    Notes
    -----
    Unlike the cubic-only helper this replaces, any space group works -- so
    hexagonal (e.g. Ti-64, space group 194) is correct here rather than
    silently reduced under the wrong point group.
    """
    angle_rad = misorientation_om_batch(M1, M2, space_group)
    if not isinstance(angle_rad, Tensor):
        angle_rad = torch.as_tensor(angle_rad, dtype=M1.dtype, device=M1.device)
    return angle_rad * _RAD2DEG


def cubic_misorientation_deg(M1: Tensor, M2: Tensor) -> Tensor:
    """Misorientation modulo cubic symmetry, in DEGREES.

    Thin alias for ``misorientation_deg(M1, M2, space_group=225)``, kept
    because it names the common case at the call sites.
    """
    return misorientation_deg(M1, M2, CUBIC_SPACE_GROUP)
