"""Pure crystallographic geometry — orientation/CSL/disorientation helpers.

REFACTOR_PLAN §3.  ``Tensor``-free, state-free, I/O-free numpy functions lifted
verbatim from ``laue_stream_utils`` so they have a single home that
``filtering.py`` depends on (and the legacy ``lsu`` re-exports for back-compat).

Cubic point group only today; generalise via point-group ops keyed on the
config space group (REFACTOR_PLAN §4b).

``# TODO(unify-after-publish)``: this math overlaps the orientation utilities in
the paper-tied ``laue_torch``; per the §1.5 constraint box it is duplicated
consciously until a shared leaf can be extracted post-submission.
"""
from __future__ import annotations

import itertools as _it

import numpy as np

__all__ = [
    "cubic_proper_ops", "CUBIC_OPS", "CSL_TABLE",
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
