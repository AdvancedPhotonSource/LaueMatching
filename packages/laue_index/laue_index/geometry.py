"""Pure crystallographic geometry — orientation/CSL/disorientation helpers.

REFACTOR_PLAN §3.  ``Tensor``-free, state-free, I/O-free numpy functions lifted
verbatim from ``laue_stream_utils`` so they have a single home that
``filtering.py`` depends on (and the legacy ``lsu`` re-exports for back-compat).

Cubic (m-3m) and hexagonal (6/mmm) proper rotation groups; the CSL table is
cubic only. Other point groups are not tabulated here -- generalise via
point-group ops keyed on the config space group (REFACTOR_PLAN §4b);
``midas_stress.orientation.make_symmetries`` is space-group-aware and would
deliver that.

``# TODO(unify-after-publish)``: the canonical orientation/quat/misorientation
primitives live in ``midas_stress.orientation`` (make_symmetries,
misorientation_om, orient_mat_to_quat, fundamental_zone).  We deliberately do
NOT depend on it here: LaueMatching is a standalone, published, numpy+C indexer
and ``midas_stress`` pulls in ``torch``; also ``misorientation_om`` returns the
angle only (the CSL twin filter needs the axis too).  Decision (kept): duplicate
consciously now; the future single-source is a small numpy-only orientation leaf
that both ``midas_stress`` and this package depend on.
"""
from __future__ import annotations

import itertools as _it

import numpy as np

__all__ = [
    "cubic_proper_ops", "CUBIC_OPS", "hexagonal_proper_ops", "HEX_OPS",
    "proper_ops_for_space_group", "CSL_TABLE",
    "disorientation_deg_axis", "is_csl_related",
]


def cubic_proper_ops() -> np.ndarray:
    """24 proper (det=+1) rotation matrices of the cubic point group m-3m."""
    ops = []
    for perm in _it.permutations(range(3)):
        for signs in _it.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            if abs(np.linalg.det(M) - 1.0) < 1e-6:
                ops.append(M)
    return np.array(ops)


CUBIC_OPS = cubic_proper_ops()


def hexagonal_proper_ops() -> np.ndarray:
    """12 proper rotations of the hexagonal point group 6/mmm (group 622).

    In the indexer's crystal Cartesian frame ``c`` is along z (``calcRecipArray``
    in LaueMatchingHeaders.h puts a along x, b in the xy plane and c along z for
    alpha = beta = 90), so the group is the six rotations by k*60 deg about z
    plus six two-folds about in-plane axes at j*30 deg. That set of axes is the
    same whether a or a* is taken along x, so it does not depend on the in-plane
    convention.
    """
    ops = []
    for k in range(6):
        t = np.radians(60.0 * k)
        c, s = np.cos(t), np.sin(t)
        ops.append(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))
    for j in range(6):
        t = np.radians(30.0 * j)
        n = np.array([np.cos(t), np.sin(t), 0.0])
        ops.append(2.0 * np.outer(n, n) - np.eye(3))
    return np.array(ops)


HEX_OPS = hexagonal_proper_ops()


def proper_ops_for_space_group(space_group: int):
    """Proper rotation ops for the groups tabulated here, else None.

    Cubic (195-230) and hexagonal (168-194). Trigonal groups are NOT mapped to
    the hexagonal ops: in the rhombohedral setting the indexer uses a different
    Cartesian frame (``calcRecipArray``'s rhomb branch), so reusing them would
    be wrong rather than merely incomplete.
    """
    sg = int(space_group or 0)
    if 195 <= sg <= 230:
        return CUBIC_OPS
    if 168 <= sg <= 194:
        return HEX_OPS
    return None

# CSL boundaries for cubic: Sigma -> (disorientation angle deg, sorted-|axis|).
CSL_TABLE = {
    3:  (60.00, np.array([0.57735, 0.57735, 0.57735])),  # 60 / <111>
    9:  (38.94, np.array([0.00000, 0.70711, 0.70711])),  # 38.94 / <110>
    11: (50.48, np.array([0.00000, 0.70711, 0.70711])),  # 50.48 / <110>
}


def disorientation_deg_axis(A: np.ndarray, B: np.ndarray, ops: np.ndarray = CUBIC_OPS):
    """Symmetry-reduced disorientation angle (deg) and rotation-axis family
    (sorted |components|) between two 3x3 orientation matrices, minimised over
    both-sided point-group symmetry.  Crystal-frame misorientation M = A^T B."""
    M = A.T @ B
    best_ang, best_M = 999.0, M
    for O1 in ops:
        OM = O1 @ M
        for O2 in ops:
            m = OM @ O2
            tr = max(-1.0, min(1.0, (np.trace(m) - 1.0) / 2.0))
            ang = np.degrees(np.arccos(tr))
            if ang < best_ang:
                best_ang, best_M = ang, m
    w, v = np.linalg.eig(best_M)
    axis = np.real(v[:, int(np.argmin(np.abs(w - 1.0)))])
    n = np.linalg.norm(axis)
    axis = axis / n if n > 0 else axis
    return best_ang, np.sort(np.abs(axis))


def is_csl_related(A, B, sigmas=(3,), tol_deg: float = 3.0,
                   ops: np.ndarray = CUBIC_OPS) -> bool:
    """True if A,B are related by one of the requested cubic CSL boundaries."""
    ang, axfam = disorientation_deg_axis(A, B, ops)
    for s in sigmas:
        ref = CSL_TABLE.get(s)
        if ref is None:
            continue
        ang_ref, ax_ref = ref
        if abs(ang - ang_ref) < tol_deg and np.linalg.norm(axfam - ax_ref) < 0.08:
            return True
    return False
