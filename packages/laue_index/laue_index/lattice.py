"""Lattice basis and symmetry operators in the indexer's crystal frame.

One Python copy of what ``calcRecipArray`` and ``MakeSymmetries`` in
``c_src/LaueMatchingHeaders.h`` do. GenerateHKLs, GenerateSimulation,
calibrate and pipeline/analysis/laue_material all take their B matrix from
here; before 0.8.0 each carried its own copy and they drifted (the rhombohedral
branch existed in two of four).

THE FRAME. The indexer puts a along Cartesian x and b in the xy plane. MIDAS,
and so ``midas_stress``, puts a* along x and b* in the xy plane. The two
coincide for orthogonal cells and differ by a rotation for monoclinic,
trigonal and hexagonal ones. A symmetry operator is only correct in the frame
its B matrix was built in, so ``laue_class_operators`` conjugates midas_stress's
operators into this frame rather than using them as they are.

THE SETTING. The R groups (146, 148, 155, 160, 161, 166, 167) can be given on
hexagonal axes (a a c 90 90 120) or rhombohedral axes (a a a al al al). The
setting is read from the angles, exactly as the C does; ``validate_setting``
refuses anything that is neither.

Lengths in nm, angles in degrees.
"""
from __future__ import annotations

from math import cos, pi, radians, sin, sqrt
from typing import Optional, Sequence

import numpy as np

R_GROUPS = frozenset({146, 148, 155, 160, 161, 166, 167})

# Trigonal groups whose basal 2-folds lie PERPENDICULAR to a (Laue class -31m),
# plus P3/P-3 (143-145, 147), kept on this set by choice as in the C. All other
# trigonal groups have them ALONG a.
_TRIG_PERP_A = frozenset({143, 144, 145, 147, 149, 151, 153, 157, 159, 162, 163})


def is_rhombohedral_group(sg: int) -> bool:
    return int(sg) in R_GROUPS


def uses_rhombohedral_axes(sg: Optional[int], lat: Sequence[float]) -> bool:
    """True when an R-group lattice is given on rhombohedral axes.

    Read from the angles only, as ``usesRhombohedralAxes`` in the C: a lattice
    fit moves the lengths, and the rhombohedral basis uses only a and alpha.
    """
    if sg is None or not is_rhombohedral_group(sg):
        return False
    al, be, ga = (float(v) for v in lat[3:6])
    return abs(al - be) < 1.0 and abs(be - ga) < 1.0 and abs(al - 90.0) >= 1.0


def validate_setting(sg: Optional[int], lat: Sequence[float]) -> None:
    """Raise ValueError for an R-group lattice on neither axis setting.

    Same test as ``validateTrigonalSetting`` in the C.
    """
    if sg is None or not is_rhombohedral_group(sg):
        return
    a, b, c, al, be, ga = (float(v) for v in lat)
    tl, ta = 1e-4, 1e-3
    hexa = (abs(a - b) <= tl * a and abs(al - 90) < ta and abs(be - 90) < ta
            and abs(ga - 120) < ta)
    rho = (abs(a - b) <= tl * a and abs(b - c) <= tl * a and abs(al - be) < ta
           and abs(be - ga) < ta and abs(al - 90) >= 1.0)
    if not (hexa or rho):
        raise ValueError(
            f"SpaceGroup {sg} is rhombohedral; LatticeParameter must be on hexagonal "
            f"axes (a a c 90 90 120) or rhombohedral axes (a a a al al al, al != 90); "
            f"got {tuple(lat)}")


def _zero_out(v: float) -> float:
    return 0.0 if abs(v) < 1e-11 else v


def direct_basis(lat: Sequence[float], sg: Optional[int] = None) -> np.ndarray:
    """Direct basis, columns a, b, c, in the indexer's Cartesian frame."""
    a, b, c, al, be, ga = (float(v) for v in lat)
    ca, cb, cg = cos(radians(al)), cos(radians(be)), cos(radians(ga))
    if uses_rhombohedral_axes(sg, lat):
        p, q = sqrt(1.0 + 2 * ca), sqrt(1.0 - ca)
        pmq, p2q = (a / 3.0) * (p - q), (a / 3.0) * (p + 2 * q)
        return np.array([[p2q, pmq, pmq], [pmq, p2q, pmq], [pmq, pmq, p2q]])
    sg_ = sin(radians(ga))
    phi = sqrt(1.0 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)
    av = [a, 0.0, 0.0]
    bv = [_zero_out(b * cg), _zero_out(b * sg_), 0.0]
    cv = [c * cb, c * (ca - cb * cg) / sg_, _zero_out(c * phi / sg_)]
    return np.column_stack([av, bv, cv])


def reciprocal_matrix(lat: Sequence[float], sg: Optional[int] = None) -> np.ndarray:
    """B, columns a*, b*, c*, in 1/nm WITH the 2*pi; q = U @ B @ hkl."""
    a, b, c, al, be, ga = (float(v) for v in lat)
    ca, cb, cg = cos(radians(al)), cos(radians(be)), cos(radians(ga))
    phi = sqrt(1.0 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)
    pv = 2 * pi / (a * b * c * phi)
    av, bv, cv = direct_basis(lat, sg).T
    B = np.column_stack([np.cross(bv, cv), np.cross(cv, av), np.cross(av, bv)]) * pv
    return np.vectorize(_zero_out)(B)


# --- symmetry ----------------------------------------------------------------

_H = sqrt(0.5)
_S3 = sqrt(3.0) / 2.0
_TABLES = {
    "tric": [(1, 0, 0, 0)],
    "mono": [(1, 0, 0, 0), (0, 0, 1, 0)],
    "orth": [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)],
    "tetr": [(1, 0, 0, 0), (_H, 0, 0, _H), (0, 0, 0, 1), (_H, 0, 0, -_H),
             (0, 1, 0, 0), (0, 0, 1, 0), (0, _H, _H, 0), (0, -_H, _H, 0)],
    "trig_perp": [(1, 0, 0, 0), (0, _S3, -0.5, 0), (0.5, 0, 0, _S3), (0, 0, 1, 0),
                  (0.5, 0, 0, -_S3), (0, _S3, 0.5, 0)],
    "trig_along": [(1, 0, 0, 0), (0.5, 0, 0, _S3), (0.5, 0, 0, -_S3), (0, 1, 0, 0),
                   (0, 0.5, _S3, 0), (0, -0.5, _S3, 0)],
    "trig_rhomb": [(1, 0, 0, 0), (0.5, 0.5, 0.5, 0.5), (0.5, -0.5, -0.5, -0.5),
                   (0, _H, -_H, 0), (0, 0, _H, -_H), (0, -_H, 0, _H)],
    "hexa": [(1, 0, 0, 0), (_S3, 0, 0, 0.5), (0.5, 0, 0, _S3), (0, 0, 0, 1),
             (0.5, 0, 0, -_S3), (_S3, 0, 0, -0.5), (0, 1, 0, 0), (0, _S3, 0.5, 0),
             (0, 0.5, _S3, 0), (0, 0, 1, 0), (0, -0.5, _S3, 0), (0, -_S3, 0.5, 0)],
}


def _cubic_table():
    q = [(1, 0, 0, 0)]
    for ax in np.eye(3):
        for ang in (90, 180, 270):
            h = radians(ang) / 2
            q.append((cos(h), *(sin(h) * ax)))
    for ax in ([1, 1, 0], [1, -1, 0], [1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, 1, -1]):
        ax = np.array(ax) / sqrt(2)
        q.append((0.0, *ax))
    for ax in ([1, 1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, -1]):
        ax = np.array(ax) / sqrt(3)
        for ang in (120, 240):
            h = radians(ang) / 2
            q.append((cos(h), *(sin(h) * ax)))
    return q


_TABLES["cubi"] = _cubic_table()


def _system_table(sg: int, lat: Sequence[float]) -> str:
    sg = int(sg)
    if sg <= 2:
        return "tric"
    if sg <= 15:
        return "mono"
    if sg <= 74:
        return "orth"
    if sg <= 142:
        return "tetr"
    if sg <= 167:
        if uses_rhombohedral_axes(sg, lat):
            return "trig_rhomb"
        return "trig_perp" if sg in _TRIG_PERP_A else "trig_along"
    if sg <= 194:
        return "hexa"
    return "cubi"


def symmetry_quaternions(sg: int, lat: Sequence[float]) -> np.ndarray:
    """What ``MakeSymmetries`` returns: the proper rotations of the crystal
    SYSTEM, (w, x, y, z), in this frame. The indexer merges on these."""
    return np.array(_TABLES[_system_table(sg, lat)], float)


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def midas_frame_rotation(lat: Sequence[float], sg: Optional[int] = None) -> np.ndarray:
    """Rotation M with v_here = M @ v_midas for crystal-frame vectors.

    MIDAS's B is upper triangular with a positive diagonal (a* along x, b* in
    the xy plane), so B_here = M @ B_midas is a QR factorisation of B_here.
    Derived here rather than imported, so it needs no midas_hkls.
    """
    Q, R = np.linalg.qr(reciprocal_matrix(lat, sg))
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    return Q * s


# The Laue class is a subgroup of the crystal-system group; which operators it
# keeps is a property of each operator, not of its row in a table.
def _about_c(m) -> bool:
    return abs(m[2, 2] - 1.0) < 1e-6          # identity or a rotation about z


def _laue_filter(sg: int):
    sg = int(sg)
    if 75 <= sg <= 88:      # 4/m
        return _about_c
    if 168 <= sg <= 176:    # 6/m
        return _about_c
    if 143 <= sg <= 148:    # -3: identity and the two 3-folds
        return lambda m: _order(m) in (1, 3)
    if 195 <= sg <= 206:    # m-3: no 4-folds, no <110> 2-folds
        return lambda m: _order(m) != 4 and not _is_face_diagonal_2fold(m)
    return None


def laue_class_operators(sg: int, lat: Sequence[float]) -> np.ndarray:
    """Proper rotations of the Laue class, (n, 3, 3), in THIS frame.

    For misorientation between orientations the indexer wrote. Uses
    ``midas_stress.make_symmetries`` conjugated into this frame when it is
    importable, else the tables here. On rhombohedral axes midas_stress (which
    assumes hexagonal axes) does not apply, so the table here is used.
    """
    sg = int(sg)
    if not uses_rhombohedral_axes(sg, lat):
        try:
            from midas_stress.orientation import make_symmetries
        except Exception:
            make_symmetries = None
        if make_symmetries is not None:
            M = midas_frame_rotation(lat, sg)
            n, ops = make_symmetries(sg)
            return np.array([M @ quat_to_matrix(q) @ M.T for q in ops[:n]])
    mats = [quat_to_matrix(q) for q in symmetry_quaternions(sg, lat)]
    keep = _laue_filter(sg)
    if keep is not None:
        mats = [m for m in mats if keep(m)]
    return np.array(mats)


def _order(m) -> int:
    ang = np.degrees(np.arccos(np.clip((np.trace(m) - 1) / 2, -1, 1)))
    return 1 if ang < 1 else int(round(360 / ang))


def _is_face_diagonal_2fold(m) -> bool:
    if _order(m) != 2:
        return False
    axis = np.linalg.eigh((m + m.T) / 2)[1][:, -1]
    return int(np.sum(np.abs(axis) > 1e-6)) == 2
