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
                                      #    (the generic one only when a single
                                      #    phase is in use)
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

__all__ = ["Phase", "phase_name", "sym_ops_for_spacegroup", "selftest"]

HC_KEV_NM = 1.2398419739

# realpath(params file) -> the phase name that first loaded it in this process.
# Phase.load refuses a second, different name for the same file.
_RESOLVED: Dict[str, str] = {}


def phase_name() -> str:
    """THE phase a single-phase script works on, from the environment. No material default.

    * ``LAUE_PHASE`` set -> that name (and, if ``LAUE_PHASES`` is also set, it must
      be one of the phases listed there).
    * unset, and ``LAUE_PHASES`` lists exactly one phase -> that phase.
    * otherwise (``LAUE_PHASES`` lists several, or neither is set) -> exit naming
      ``LAUE_PHASE``.

    Before 2026-09 the scripts defaulted the singular variable to "alpha" in some
    places and "zn" in others, so the same unset environment meant a different
    material depending on which script ran.
    """
    listed = [p.strip() for p in os.environ.get("LAUE_PHASES", "").split(",") if p.strip()]
    v = os.environ.get("LAUE_PHASE", "").strip()
    if v:
        if listed and v not in listed:
            raise SystemExit(f"LAUE_PHASE={v!r} is not one of LAUE_PHASES={listed}")
        return v
    if len(listed) == 1:
        return listed[0]
    if listed:
        raise SystemExit(f"LAUE_PHASE is not set and LAUE_PHASES lists {len(listed)} phases "
                         f"{listed}: set LAUE_PHASE (or pass the phase argument) to choose one")
    raise SystemExit("LAUE_PHASE is not set (and LAUE_PHASES does not name a single phase): "
                     "set it to the phase whose LAUE_PARAMS_<PHASE> file describes this run")


# --------------------------------------------------------------------------
# symmetry and lattice: one copy, in laue_index.lattice
# --------------------------------------------------------------------------
# The indexer puts a along x (calcRecipArray); midas_stress puts a* along x.
# For trigonal cells the frames differ by 30 deg about c, so operators taken
# from midas_stress as they are were wrong here for every trigonal group
# (2026-09-28 code read). laue_index.lattice owns the frame change, the
# rhombohedral setting and the fallback tables, and is shared with the
# indexer's Python tools, so this module cannot drift from the indexer again.
try:
    from laue_index import lattice as _lattice
    from laue_index import artifacts as _artifacts
except ImportError as exc:                          # pragma: no cover
    raise SystemExit(f"laue_index is not importable ({exc}); laue_material takes its "
                     f"lattice basis and symmetry operators from laue_index.lattice")

# A lattice per crystal system for callers that give only a space group. The
# operators depend on the lattice only through its setting (rhombohedral axes)
# and its frame, which is fixed by the angles, not the lengths.
_SYSTEM_LATTICE = ((2, (1, 1.1, 1.2, 80, 95, 105)), (15, (1, 1.1, 1.2, 90, 104, 90)),
                   (74, (1, 1.1, 1.2, 90, 90, 90)), (142, (1, 1, 1.2, 90, 90, 90)),
                   (194, (1, 1, 1.6, 90, 90, 120)), (230, (1, 1, 1, 90, 90, 90)))


def _midas_stress():
    """midas_stress.orientation, or None with the reason recorded.

    midas_stress is the canonical source for every orientation/misorientation
    primitive in MIDAS. It is not importable in every beamline environment,
    hence the guarded import and the fallback tables in laue_index.lattice.
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
    return _lattice.quat_to_matrix(q)


def sym_ops_for_spacegroup(sgnum: int, latt=None) -> np.ndarray:
    """
    Proper-rotation operators of the Laue class containing this space group,
    in the INDEXER's crystal frame (a along x).

    ``latt`` decides the setting of an R group (hexagonal or rhombohedral
    axes); without it hexagonal axes are assumed. Symmetry follows the space
    group, never the phase name.
    """
    n = int(sgnum)
    if not 1 <= n <= 230:
        raise ValueError(f"space group number out of range: {sgnum}")
    if latt is None:
        latt = next(l for hi, l in _SYSTEM_LATTICE if n <= hi)
    return _lattice.laue_class_operators(n, latt)


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


def _direct_A(latt, sgnum=None) -> np.ndarray:
    """
    Direct-lattice matrix, columns = a, b, c, in nm, in the SAME Cartesian frame
    as :func:`_reciprocal_B` (a along x, b in the xy plane), so ``A.T @ B == 2*pi*I``.

    Use this for crystal DIRECTIONS [uvw] (``A @ uvw``); ``B @ hkl`` is a plane
    NORMAL. The two coincide only for cubic cells and for hex [0001] || (0001).
    ``sgnum`` selects the rhombohedral embedding for an R group on rhombohedral axes.
    """
    return _lattice.direct_basis(latt, sgnum)


def _reciprocal_B(latt, sgnum=None) -> np.ndarray:
    """
    Reciprocal-lattice matrix, columns = a*, b*, c*, in 1/nm (with the 2*pi).

    The indexer's own construction (laue_index.lattice mirrors calcRecipArray),
    so the analysis chain and the indexer cannot drift apart.
    """
    return _lattice.reciprocal_matrix(latt, sgnum)


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
        _lattice.validate_setting(self.sgnum, self.lattice)
        self.B = _reciprocal_B(self.lattice, self.sgnum)
        self.A = _direct_A(self.lattice, self.sgnum)  # direct lattice, for [uvw] directions
        self.sym_ops = sym_ops_for_spacegroup(self.sgnum, self.lattice)
        # midas_stress works in its own frame (a* along x): orientations go to it
        # as OM @ M. None on rhombohedral axes, which midas_stress does not model.
        self._to_midas = (None if _lattice.uses_rhombohedral_axes(self.sgnum, self.lattice)
                          else _lattice.midas_frame_rotation(self.lattice, self.sgnum))

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
        # The list's provenance record must describe THIS crystal (refuse
        # another lattice or space group; warn if the list has no record).
        _artifacts.require(hklf, "hkl_list", expect={
            "SpaceGroup": self.sgnum, "Symmetry": self.symmetry,
            "LatticeParameter": self.lattice})
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
        generic = False
        if params_path is None:
            params_path = os.environ.get(f"LAUE_PARAMS_{phase.upper()}")
            if not params_path and os.environ.get("LAUE_PARAMS"):
                params_path, generic = os.environ["LAUE_PARAMS"], True
        if not params_path:
            raise RuntimeError(
                f"no parameter file for phase {phase!r}: set LAUE_PARAMS_{phase.upper()} "
                f"(or LAUE_PARAMS, single-phase only) to the params_*.txt used for indexing."
            )
        if generic:
            # The generic LAUE_PARAMS names no phase, so it is only unambiguous when
            # one phase is in use. With two, Phase.load("alpha") and
            # Phase.load("beta") would both land on it and an "alpha vs beta"
            # analysis would silently compare one material with itself.
            listed = [p.strip() for p in os.environ.get("LAUE_PHASES", "").split(",") if p.strip()]
            if len(listed) > 1:
                raise RuntimeError(
                    f"LAUE_PARAMS is set but LAUE_PHASES lists {len(listed)} phases {listed}: "
                    f"the generic file cannot describe more than one. Set "
                    f"LAUE_PARAMS_{phase.upper()} (and one per other phase).")
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"parameter file not found: {params_path}")
        # Two different phase names resolving to ONE file in one process is the
        # same failure by another route (LAUE_PARAMS used for both, or two
        # LAUE_PARAMS_<PHASE> pointing at one file): refuse it.
        key = os.path.realpath(params_path)
        prev = _RESOLVED.get(key)
        if prev is not None and prev != phase:
            raise RuntimeError(
                f"phases {prev!r} and {phase!r} both resolve to {params_path}"
                f"{' (via the generic LAUE_PARAMS)' if generic else ''}: two phases "
                f"cannot share one material description. Set LAUE_PARAMS_{phase.upper()} "
                f"to that phase's own params file.")
        _RESOLVED[key] = phase
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
        canonical MIDAS implementation, after moving both orientations into its
        crystal frame (``OM @ M``; the identity except for monoclinic, trigonal
        and hexagonal cells). midas_stress returns RADIANS; the conversion
        happens here so callers keep degree-valued thresholds.

        The fallback (no midas_stress, or rhombohedral axes) is the crystal-side
        einsum over ``self.sym_ops``, which are already in this frame.
        """
        Bs = np.asarray(Bs, float)
        if Bs.ndim == 2:
            Bs = Bs[None]
        ms = _midas_stress()
        if ms is not None and self._to_midas is not None:
            M = self._to_midas
            A9 = (np.asarray(A, float) @ M).reshape(9)
            oms1 = np.repeat(A9[None, :], len(Bs), axis=0)
            oms2 = (Bs @ M).reshape(len(Bs), 9)
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

    # direct and reciprocal matrices share a frame: A^T B = 2 pi I
    for latt in ([a, b, c, 90, 90, 120], [0.33065] * 3 + [90, 90, 90]):
        assert np.allclose(_direct_A(latt).T @ _reciprocal_B(latt), 2 * pi * np.eye(3),
                           atol=1e-9)

    assert len(sym_ops_for_spacegroup(194)) == 12     # HCP  (Ti alpha, Zn)
    assert len(sym_ops_for_spacegroup(229)) == 24     # BCC  (Ti beta)
    assert len(sym_ops_for_spacegroup(225)) == 24     # FCC

    # Every operator of every Laue class maps the lattice onto itself (A^-1 S A
    # is an integer matrix) and each set is a closed group without duplicates.
    # Covers the frame (trigonal) and the finer Laue classes (4/m, -3, 6/m, m-3).
    for sg in (2, 12, 62, 88, 139, 148, 150, 162, 167, 176, 194, 206, 225):
        latt = next(l for hi, l in _SYSTEM_LATTICE if sg <= hi)
        ops, A = sym_ops_for_spacegroup(sg, latt), _direct_A(latt, sg)
        for S in ops:
            assert np.allclose(S @ S.T, np.eye(3), atol=1e-9) and abs(np.linalg.det(S) - 1) < 1e-9
            Mi = np.linalg.inv(A) @ S @ A
            assert np.allclose(Mi, np.rint(Mi), atol=1e-4), f"SG{sg}: operator is not a lattice symmetry"
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                assert np.abs(ops[i] - ops[j]).max() > 1e-6, f"SG{sg}: duplicate operator"
        for S in ops:
            for T in ops:
                d = np.abs(ops - (S @ T)).reshape(len(ops), -1).max(axis=1).min()
                assert d < 1e-6, f"SG{sg}: not closed under composition (gap {d:.2e})"

    print("laue_material selftest OK "
          "(generic B reproduces hexB() and the cubic branch exactly; "
          "operators are closed groups of lattice symmetries in the indexer frame)")


if __name__ == "__main__":
    selftest()
