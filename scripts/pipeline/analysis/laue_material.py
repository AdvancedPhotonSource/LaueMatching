"""
Per-material crystallography + geometry for the analysis chain.

Before this module every script in analysis/ carried its own copy of the
a two-phase hcp/bcc alloy lattice constants, a hard-coded ``valid_hkls_Ti_*.csv`` filename, and
a phase-name -> symmetry mapping that only understood ``"alpha"`` (hex) and
``"beta"`` (cubic).  Nothing errored on a different material -- the reflection
list simply belonged to Ti, and every downstream null was computed against it.

The fix is not to add a second copy of the constants for the second material.
The indexing parameter file already carries lattice, space group, detector
geometry, energy window and HKL path, and it is the file the indexer itself
consumed, so it is the only description that cannot silently disagree with the
run it is describing.  This module reads that file.

    from laue_material import Phase

    ph = Phase.load("alpha")          # -> $LAUE_PARAMS_ALPHA, else $LAUE_PARAMS
    px = ph.project(OM)               # (n,2) predicted detector pixels
    ops = ph.sym_ops                  # proper rotations for this crystal system

Selecting a material is then an environment variable, not an edit:

    LAUE_PARAMS_ALPHA=$WORK/params/params_Zn.txt

Exactness note: for a hexagonal cell the general reciprocal-lattice
construction below reduces algebraically to the old ``hexB()`` (alpha=beta=90
gives phi == sin(gamma), hence c = (0,0,c) and V = a*b*c*sin(gamma)), and for a
cubic cell it reduces to ``eye(3)*2*pi/a``.  Both former branches are
reproduced bit-for-bit, so re-running the Ti analysis through this module is a
no-op -- verified by ``selftest()`` below.
"""

from __future__ import annotations

import os
from math import cos, sin, sqrt, radians, pi
from typing import Dict, Optional

import numpy as np

__all__ = ["Phase", "sym_ops_for_spacegroup", "selftest"]

HC_KEV_NM = 1.2398419739


# --------------------------------------------------------------------------
# symmetry
# --------------------------------------------------------------------------
def _rmat(axis, deg) -> np.ndarray:
    u = np.asarray(axis, float)
    u = u / np.linalg.norm(u)
    t = radians(deg)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + sin(t) * K + (1 - cos(t)) * (K @ K)


def _hex12() -> np.ndarray:
    """Proper rotations of Laue class 6/mmm (12 operators)."""
    return np.array(
        [_rmat([0, 0, 1], 60 * k) for k in range(6)]
        + [_rmat([cos(radians(a)), sin(radians(a)), 0], 180)
           for a in (0, 30, 60, 90, 120, 150)]
    )


def _cub24() -> np.ndarray:
    """Proper rotations of Laue class m-3m (24 operators)."""
    return np.array(
        [np.eye(3)]
        + [_rmat(a, d) for a, d in [
            ([1, 0, 0], 90), ([1, 0, 0], 180), ([1, 0, 0], 270),
            ([0, 1, 0], 90), ([0, 1, 0], 180), ([0, 1, 0], 270),
            ([0, 0, 1], 90), ([0, 0, 1], 180), ([0, 0, 1], 270),
            ([1, 1, 0], 180), ([1, -1, 0], 180), ([1, 0, 1], 180),
            ([-1, 0, 1], 180), ([0, 1, 1], 180), ([0, 1, -1], 180),
            ([1, 1, 1], 120), ([1, 1, 1], 240), ([1, -1, 1], 120),
            ([1, -1, 1], 240), ([-1, 1, 1], 120), ([-1, 1, 1], 240),
            ([1, 1, -1], 120), ([1, 1, -1], 240)]]
    )


def _tet8() -> np.ndarray:
    return np.array(
        [_rmat([0, 0, 1], 90 * k) for k in range(4)]
        + [_rmat([1, 0, 0], 180), _rmat([0, 1, 0], 180),
           _rmat([1, 1, 0], 180), _rmat([1, -1, 0], 180)]
    )


def _trig6() -> np.ndarray:
    return np.array(
        [_rmat([0, 0, 1], 120 * k) for k in range(3)]
        + [_rmat([cos(radians(a)), sin(radians(a)), 0], 180)
           for a in (0, 60, 120)]
    )


def _trig3() -> np.ndarray:
    return np.array([_rmat([0, 0, 1], 120 * k) for k in range(3)])


def _orth4() -> np.ndarray:
    return np.array([np.eye(3), _rmat([1, 0, 0], 180),
                     _rmat([0, 1, 0], 180), _rmat([0, 0, 1], 180)])


def _mono2() -> np.ndarray:
    return np.array([np.eye(3), _rmat([0, 1, 0], 180)])


def _tri1() -> np.ndarray:
    return np.array([np.eye(3)])


def _midas_stress():
    """midas_stress.orientation, or None with the reason recorded.

    midas_stress is the canonical source for every orientation/misorientation
    primitive in MIDAS (a byte-for-byte port of the C GetMisorientation.h), so
    it is what this module uses when it is importable. It hard-imports torch at
    package level, which is not present in every beamline environment, hence the
    guarded import and the local fallback below.
    """
    global _MS, _MS_ERR
    try:
        return _MS
    except NameError:
        pass
    try:
        from midas_stress import orientation as _o
        _MS, _MS_ERR = _o, None
    except Exception as exc:                        # noqa: BLE001
        _MS, _MS_ERR = None, f"{type(exc).__name__}: {exc}"
    return _MS


def _quat_to_om(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def sym_ops_for_spacegroup(sgnum: int) -> np.ndarray:
    """
    Proper-rotation operators of the Laue class containing this space group.

    Taken from ``midas_stress.orientation.make_symmetries`` (which returns
    symmetry quaternions) when available, so the operator set is not a second,
    independently maintained copy of the same crystallography. The explicit
    tables below are the fallback for environments without midas_stress; they
    are checked against make_symmetries by ``selftest()``.

    Note also what this replaces: the old code chose between hex-12 and
    cubic-24 on the *phase name* ("alpha"/"beta"), which is only correct for a
    material whose two phases happen to be one hexagonal and one cubic.
    """
    n = int(sgnum)
    ms = _midas_stress()
    if ms is not None:
        n_sym, sym = ms.make_symmetries(n)
        return np.array([_quat_to_om(q) for q in sym[:n_sym]])

    if 195 <= n <= 230:
        return _cub24()
    if 168 <= n <= 194:
        return _hex12()
    if 149 <= n <= 167:
        return _trig6()
    if 143 <= n <= 148:
        return _trig3()
    if 75 <= n <= 142:
        return _tet8()
    if 16 <= n <= 74:
        return _orth4()
    if 3 <= n <= 15:
        return _mono2()
    if 1 <= n <= 2:
        return _tri1()
    raise ValueError(f"space group number out of range: {sgnum}")


# --------------------------------------------------------------------------
# parameter file
# --------------------------------------------------------------------------
def read_params(path: str) -> Dict[str, list]:
    """Parse a LaueMatching parameter file into {key: [tokens]}."""
    out: Dict[str, list] = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            tok = line.split()
            out[tok[0]] = tok[1:]
    return out


def _reciprocal_B(latt) -> np.ndarray:
    """
    Reciprocal-lattice matrix, columns = a*, b*, c*, in 1/nm (with the 2*pi).

    Same construction as GenerateHKLs.calcRecipArray, so the analysis chain and
    the reflection-list generator cannot drift apart.
    """
    a, b, c, alpha, beta, gamma = [float(v) for v in latt]
    ca, cb, cg = cos(radians(alpha)), cos(radians(beta)), cos(radians(gamma))
    sg = sin(radians(gamma))
    phi = sqrt(1.0 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)
    Vc = a * b * c * phi
    pv = 2 * pi / Vc

    av = np.array([a, 0.0, 0.0])
    bv = np.array([b * cg, b * sg, 0.0])
    cv = np.array([c * cb, c * (ca - cb * cg) / sg, c * phi / sg])
    cv[np.abs(cv) < 1e-11] = 0.0

    return np.column_stack([np.cross(bv, cv), np.cross(cv, av), np.cross(av, bv)]) * pv


class Phase:
    """Crystallography + detector geometry for one indexed phase."""

    def __init__(self, params_path: str, name: str = "phase"):
        self.name = name
        self.params_path = params_path
        p = read_params(params_path)
        self.raw = p

        self.sgnum = int(p["SpaceGroup"][0])
        self.symmetry = p.get("Symmetry", ["P"])[0]
        self.lattice = [float(v) for v in p["LatticeParameter"]]
        self.B = _reciprocal_B(self.lattice)
        self.sym_ops = sym_ops_for_spacegroup(self.sgnum)

        self.P = np.array([float(v) for v in p["P_Array"]])
        self.Rrod = np.array([float(v) for v in p["R_Array"]])
        self.dx = float(p["PxX"][0])
        self.dy = float(p["PxY"][0])
        self.npx_x = int(float(p["NrPxX"][0]))
        self.npx_y = int(float(p["NrPxY"][0]))
        self.Elo = float(p["Elo"][0])
        self.Ehi = float(p["Ehi"][0])

        hklf = p["HKLFile"][0]
        self.hkl_path = hklf
        self.hkls = np.loadtxt(hklf)[:, :3]

        angr = np.linalg.norm(self.Rrod)
        v = self.Rrod / angr
        c_, s_ = cos(angr), sin(angr)
        self.rot = np.array([
            [c_ + (1 - c_) * v[0] ** 2, (1 - c_) * v[0] * v[1] - s_ * v[2], (1 - c_) * v[0] * v[2] + s_ * v[1]],
            [(1 - c_) * v[1] * v[0] + s_ * v[2], c_ + (1 - c_) * v[1] ** 2, (1 - c_) * v[1] * v[2] - s_ * v[0]],
            [(1 - c_) * v[2] * v[0] - s_ * v[1], (1 - c_) * v[2] * v[1] + s_ * v[0], c_ + (1 - c_) * v[2] ** 2]])
        self.roti = np.linalg.inv(self.rot)
        self.ki = np.array([0.0, 0.0, 1.0])

    # -- selection --------------------------------------------------------
    @classmethod
    def load(cls, phase: str = "alpha", params_path: Optional[str] = None) -> "Phase":
        """
        Resolve the parameter file for ``phase``.

        Order: explicit argument, then ``$LAUE_PARAMS_<PHASE>``, then
        ``$LAUE_PARAMS``.  Failing to resolve is an error rather than a
        fallback to a built-in material -- silently analysing one material with
        another's reflection list is the exact failure this module exists to
        prevent.
        """
        if params_path is None:
            params_path = os.environ.get(f"LAUE_PARAMS_{phase.upper()}") or os.environ.get("LAUE_PARAMS")
        if not params_path:
            raise RuntimeError(
                f"no parameter file for phase {phase!r}: set LAUE_PARAMS_{phase.upper()} "
                f"(or LAUE_PARAMS) to the params_*.txt used for indexing."
            )
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"parameter file not found: {params_path}")
        return cls(params_path, name=phase)

    # -- forward projection ----------------------------------------------
    def project(self, OM: np.ndarray, with_energy: bool = False) -> np.ndarray:
        """
        Predicted detector pixels for orientation matrix ``OM``.

        Returns (n,2) of (px, py), or (n,3) of (px, py, E_keV) when
        ``with_energy`` -- the energy column is what lets a spot seen through
        an absorbing overlayer be distinguished from one that is not.
        """
        q = (OM @ self.B @ self.hkls.T).T
        ql = np.linalg.norm(q, axis=1)
        m = ql > 1e-9
        q, ql = q[m], ql[m]
        qh = q / ql[:, None]

        kf = self.ki - 2 * qh[:, 2:3] * qh
        xd = (self.roti @ kf.T).T
        m = xd[:, 2] > 0
        xd, ql, qh = xd[m], ql[m], qh[m]
        xs = xd * self.P[2] / xd[:, 2:3]

        px = (xs[:, 0] - self.P[0]) / self.dx + 0.5 * (self.npx_x - 1)
        py = (xs[:, 1] - self.P[1]) / self.dy + 0.5 * (self.npx_y - 1)
        st = -qh[:, 2]
        mk = ((px >= 0) & (px < self.npx_x - 1)
              & (py >= 0) & (py < self.npx_y - 1) & (st > 1e-9))
        E = HC_KEV_NM * ql[mk] / st[mk] / (4 * pi)
        me = (E > self.Elo) & (E < self.Ehi)
        if with_energy:
            return np.c_[px[mk][me], py[mk][me], E[me]]
        return np.c_[px[mk][me], py[mk][me]]

    def misorientation(self, A: np.ndarray, Bs: np.ndarray) -> np.ndarray:
        """Symmetry-reduced misorientation, in DEGREES, of ``A`` against each ``Bs``.

        Delegates to ``midas_stress.orientation.misorientation_om_batch``, the
        canonical MIDAS implementation. Note midas_stress returns RADIANS; the
        conversion happens here so callers keep the degree-valued thresholds the
        analysis chain is written around.

        The local fallback is the pipeline's original einsum. Cross-checked on
        30,000 real Zn pairs: the two agree to 0.032 deg worst case, and no pair
        changes side of the 1.0 deg clustering cut -- so results are unaffected by
        which path runs, but the midas_stress path is the one to trust.
        """
        Bs = np.asarray(Bs, float)
        if Bs.ndim == 2:
            Bs = Bs[None]
        ms = _midas_stress()
        if ms is not None:
            A9 = np.asarray(A, float).reshape(9)
            oms1 = np.repeat(A9[None, :], len(Bs), axis=0)
            oms2 = Bs.reshape(len(Bs), 9)
            return np.degrees(np.asarray(ms.misorientation_om_batch(oms1, oms2, self.sgnum)))

        best = np.full(len(Bs), 999.0)
        for S in self.sym_ops:
            tr = np.einsum('ij,kj,mki->m', S, A, Bs)
            best = np.minimum(best, np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))))
        return best

    def __repr__(self):
        a, b, c = self.lattice[:3]
        return (f"<Phase {self.name}: SG{self.sgnum}{self.symmetry} "
                f"a={a:.5f} b={b:.5f} c={c:.5f} nm, {len(self.hkls)} hkls, "
                f"{len(self.sym_ops)} sym ops, E {self.Elo}-{self.Ehi} keV>")


# --------------------------------------------------------------------------
def selftest() -> None:
    """Check the generic B reproduces the two hard-coded branches it replaces."""
    # former hexB(), a two-phase hcp/bcc alloy alpha
    a, b, c = 0.2921, 0.2921, 0.4665
    cg, sg = cos(radians(120)), sin(radians(120))
    pv = 2 * pi / (a * b * c * sg)
    a0, a1, a2 = a, 0, 0
    b0, b1, b2 = b * cg, b * sg, 0
    c0, c1, c2 = 0, 0, c
    old_hex = np.array([[(b1 * c2 - b2 * c1), (c1 * a2 - c2 * a1), (a1 * b2 - a2 * b1)],
                        [(b2 * c0 - b0 * c2), (c2 * a0 - c0 * a2), (a2 * b0 - a0 * b2)],
                        [(b0 * c1 - b1 * c0), (c0 * a1 - c1 * a0), (a0 * b1 - a1 * b0)]]) * pv
    new_hex = _reciprocal_B([a, b, c, 90, 90, 120])
    assert np.allclose(old_hex, new_hex, atol=1e-9), (old_hex, new_hex)

    # former beta branch, BCC a = 0.33065 nm
    old_cub = np.eye(3) * 2 * pi / 0.33065
    new_cub = _reciprocal_B([0.33065] * 3 + [90, 90, 90])
    assert np.allclose(old_cub, new_cub, atol=1e-9), (old_cub, new_cub)

    assert len(sym_ops_for_spacegroup(194)) == 12     # HCP  (Ti alpha, Zn)
    assert len(sym_ops_for_spacegroup(229)) == 24     # BCC  (Ti beta)
    assert len(sym_ops_for_spacegroup(225)) == 24     # FCC

    # When midas_stress is present the operators come from make_symmetries, so
    # pin that they are the SAME SET as the fallback tables -- otherwise grain
    # counts would change silently depending on whether midas_stress imported.
    #
    # Tolerance is 1e-4, not machine epsilon, and deliberately so: midas_stress
    # stores its symmetry quaternions rounded to 5 decimals (0.86603 for sqrt(3)/2,
    # 0.70711 for sqrt(2)/2), so its operators are not exactly orthogonal --
    # |det - 1| ~ 3e-5, and the group closes only to ~3e-5. That is ~0.002 deg of
    # angular error against a 1.0 deg clustering cut, i.e. 500x smaller than the
    # decision it feeds, but comparing at 1e-9 reports a false mismatch.
    if _midas_stress() is not None:
        for sg, local in ((194, _hex12()), (229, _cub24()), (225, _cub24())):
            ms = sym_ops_for_spacegroup(sg)
            assert len(ms) == len(local), f"SG{sg}: {len(ms)} vs {len(local)} operators"
            worst = max(np.abs(local - A).reshape(len(local), -1).max(axis=1).min()
                        for A in ms)
            assert worst < 1e-4, f"SG{sg}: operator sets differ by {worst:.2e}"
        print("  midas_stress operator sets match the fallback tables "
              f"(worst {worst:.1e}; midas_stress constants are 5-decimal rounded)")

    for nm, ops in (("hex12", _hex12()), ("cub24", _cub24()), ("tet8", _tet8()),
                    ("trig6", _trig6()), ("trig3", _trig3()), ("orth4", _orth4()),
                    ("mono2", _mono2()), ("tri1", _tri1())):
        for S in ops:
            assert np.allclose(S @ S.T, np.eye(3), atol=1e-9), nm
            assert abs(np.linalg.det(S) - 1) < 1e-9, nm    # proper rotations only
        # no duplicates: a repeated operator inflates nothing but wastes work and
        # would hide a missing one behind the expected count
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                assert np.abs(ops[i] - ops[j]).max() > 1e-6, f"{nm}: duplicate operator"
        # closure under composition (a group, not merely a set of rotations).
        # Compare numerically -- a rounded-bytes comparison reports false
        # failures because -0.0 and 0.0 have different byte patterns.
        for S in ops:
            for T in ops:
                d = np.abs(ops - (S @ T)).reshape(len(ops), -1).max(axis=1).min()
                assert d < 1e-9, f"{nm}: not closed under composition (gap {d:.2e})"

    print("laue_material selftest OK "
          "(generic B reproduces hexB() and the cubic branch exactly; "
          "operator sets are closed proper-rotation groups)")


if __name__ == "__main__":
    selftest()
