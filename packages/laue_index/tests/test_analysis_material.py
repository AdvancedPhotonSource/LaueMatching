"""laue_material: one params file per phase, symmetry from the space group, and
direct-lattice texture directions.

Behaviours pinned (synthetic params files only):

* The generic ``LAUE_PARAMS`` is refused when two phases are in use, and two
  phase names resolving to ONE params file are refused -- otherwise
  ``Phase.load("alpha")`` and ``Phase.load("beta")`` silently describe the same
  material and an "alpha vs beta" analysis compares Zn with Zn.
* Symmetry follows the space group, not the phase name: a hexagonal phase called
  "zn" or "beta" gets 12 operators, a cubic one 24. The scripts that used to pick
  hex-vs-cubic from ``PHASE == "alpha"`` or carry their own tables no longer do.
* texture_null uses DIRECT-lattice directions: for a hexagonal cell [2-1-10] and
  <10-10> are 30 deg apart and [0001] lies along c. With the reciprocal vectors it
  used before, its two "a-axis" rows were both <10-10>.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    from conftest import _REPO_ROOT
except ImportError:  # pragma: no cover
    _REPO_ROOT = None

HEX = (0.2665, 0.2665, 0.4947, 90, 90, 120)
CUB = (0.33065, 0.33065, 0.33065, 90, 90, 90)


def _analysis_dir() -> Path:
    if _REPO_ROOT is None:
        pytest.skip("not running from a repo checkout (pipeline/analysis absent)")
    d = Path(_REPO_ROOT) / "pipeline" / "analysis"
    if not d.is_dir():
        pytest.skip(f"{d} not present")
    return d


@pytest.fixture
def lm(monkeypatch):
    d = str(_analysis_dir())
    if d not in sys.path:
        sys.path.insert(0, d)
    import laue_material
    # a fresh per-test registry, and no inherited selection variables
    monkeypatch.setattr(laue_material, "_RESOLVED", {})
    for k in [k for k in __import__("os").environ if k.startswith("LAUE_")]:
        monkeypatch.delenv(k)
    return laue_material


def _params(d: Path, name: str, sg: int, latt) -> Path:
    hkl = d / f"hkls_{name}.txt"
    np.savetxt(hkl, np.array([[1, 0, 0], [0, 0, 1], [1, 1, 1], [2, 2, 2]]), fmt="%d")
    p = d / f"params_{name}.txt"
    p.write_text(
        f"SpaceGroup {sg}\nLatticeParameter {' '.join(str(v) for v in latt)}\n"
        "P_Array 0.028828 0.002715 0.512993\nR_Array -1.20161887 -1.21404493 -1.21852276\n"
        "PxX 0.0002\nPxY 0.0002\nNrPxX 2048\nNrPxY 2048\nElo 5\nEhi 30\n"
        f"HKLFile {hkl}\n")
    return p


# ---------------------------------------------------------------------------
# one material per phase
# ---------------------------------------------------------------------------
def test_generic_params_refused_for_two_phases(lm, tmp_path, monkeypatch):
    p = _params(tmp_path, "zn", 194, HEX)
    monkeypatch.setenv("LAUE_PARAMS", str(p))
    monkeypatch.setenv("LAUE_PHASES", "alpha,beta")
    with pytest.raises(RuntimeError, match="LAUE_PHASES lists 2"):
        lm.Phase.load("alpha")


def test_generic_params_second_name_refused(lm, tmp_path, monkeypatch):
    """No LAUE_PHASES set: the first name may use LAUE_PARAMS, a second name
    resolving to the same file may not."""
    p = _params(tmp_path, "zn", 194, HEX)
    monkeypatch.setenv("LAUE_PARAMS", str(p))
    assert lm.Phase.load("alpha").sgnum == 194
    assert lm.Phase.load("alpha").sgnum == 194        # the same name again is fine
    with pytest.raises(RuntimeError, match="both resolve to"):
        lm.Phase.load("beta")


def test_two_phase_variables_pointing_at_one_file_refused(lm, tmp_path, monkeypatch):
    p = _params(tmp_path, "zn", 194, HEX)
    monkeypatch.setenv("LAUE_PARAMS_ALPHA", str(p))
    monkeypatch.setenv("LAUE_PARAMS_BETA", str(p))
    lm.Phase.load("alpha")
    with pytest.raises(RuntimeError, match="both resolve to"):
        lm.Phase.load("beta")


def test_single_phase_generic_params_still_works(lm, tmp_path, monkeypatch):
    p = _params(tmp_path, "zn", 194, HEX)
    monkeypatch.setenv("LAUE_PARAMS", str(p))
    monkeypatch.setenv("LAUE_PHASES", "zn")
    assert lm.Phase.load("zn").params_path == str(p)


def test_distinct_files_per_phase_load(lm, tmp_path, monkeypatch):
    monkeypatch.setenv("LAUE_PARAMS_ALPHA", str(_params(tmp_path, "a", 194, HEX)))
    monkeypatch.setenv("LAUE_PARAMS_BETA", str(_params(tmp_path, "b", 229, CUB)))
    assert lm.Phase.load("alpha").sgnum == 194 and lm.Phase.load("beta").sgnum == 229


# ---------------------------------------------------------------------------
# symmetry by space group
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,sg,latt,nops", [
    ("zn", 194, HEX, 12), ("beta", 194, HEX, 12), ("alpha", 229, CUB, 24), ("fcc", 225, CUB, 24),
])
def test_symmetry_follows_space_group_not_name(lm, tmp_path, monkeypatch, name, sg, latt, nops):
    monkeypatch.setenv(f"LAUE_PARAMS_{name.upper()}", str(_params(tmp_path, name, sg, latt)))
    ph = lm.Phase.load(name)
    assert len(ph.sym_ops) == nops
    # misorientation of an orientation with its own symmetry-equivalent is ~0,
    # in DEGREES
    rng = np.random.default_rng(1)
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    A = lm._quat_to_om(q)
    B = np.array([A @ S for S in ph.sym_ops])
    assert np.all(ph.misorientation(A, B) < 1e-3)


OWNED_WITH_SYMMETRY = ["scan_map.py", "parentbeta_backfill.py", "map_validate_cluster.py",
                       "anchor_null.py", "big_grain_diagnostic.py", "big_grain_split_test.py",
                       "parentbeta_reconstruct.py", "beta_map_validate.py",
                       "collect_scan_metrics.py"]


@pytest.mark.parametrize("name", OWNED_WITH_SYMMETRY)
def test_no_script_picks_symmetry_by_name_or_owns_a_table(name):
    """Source-level: the hex table (60*k about z) and the phase-name branch are gone."""
    src = (_analysis_dir() / name).read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"PHASE\s*==\s*[\"']alpha[\"']", code), "symmetry chosen by phase name"
    assert "60*k" not in code.replace(" ", ""), "own hexagonal operator table"
    assert "([1,1,1],120)" not in code.replace(" ", ""), "own cubic operator table"


# ---------------------------------------------------------------------------
# texture directions
# ---------------------------------------------------------------------------
@pytest.fixture
def tn():
    d = str(_analysis_dir())
    if d not in sys.path:
        sys.path.insert(0, d)
    import texture_null
    import laue_material
    return texture_null, laue_material


def _angle(u, v):
    return np.degrees(np.arccos(np.clip(abs(u @ v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))


def test_hex_direct_directions(tn):
    texture_null, laue_material = tn
    A = laue_material._direct_A(HEX)
    rows = dict(texture_null.texture_directions(194))
    I = np.eye(3)[None]
    c = texture_null.axis_directions(I, rows["c-axis [0001]"], A)[0]
    a1 = texture_null.axis_directions(I, rows["a-axis [2-1-10]"], A)[0]
    m = texture_null.axis_directions(I, rows["[10-10]"], A)[0]
    assert np.allclose(c, [0, 0, 1], atol=1e-12)                 # [0001] along c
    assert abs(_angle(a1, m) - 30.0) < 1e-9                      # [2-1-10] vs <10-10>
    assert abs(_angle(a1, c) - 90.0) < 1e-9 and abs(_angle(m, c) - 90.0) < 1e-9
    # a1 is the cell's a edge, of length a
    assert np.allclose(A @ np.array(rows["a-axis [2-1-10]"]), [HEX[0], 0, 0], atol=1e-12)


def test_reciprocal_vectors_were_the_wrong_family(tn):
    """What the old code measured: B @ (1,0,0) is 30 deg off the a axis, i.e. a
    <10-10> direction, the same family as its "[10-10]" row (B @ (0,1,0))."""
    texture_null, laue_material = tn
    A, B = laue_material._direct_A(HEX), laue_material._reciprocal_B(HEX)
    a_axis = A @ np.array([1.0, 0, 0])
    assert abs(_angle(B @ np.array([1.0, 0, 0]), a_axis) - 30.0) < 1e-9
    # and the c row was right: c* is parallel to c
    assert _angle(B @ np.array([0, 0, 1.0]), A @ np.array([0, 0, 1.0])) < 1e-9


def test_direct_and_reciprocal_share_a_frame(tn):
    _, laue_material = tn
    for latt in (HEX, CUB, (0.3, 0.4, 0.5, 80, 95, 105)):
        A, B = laue_material._direct_A(latt), laue_material._reciprocal_B(latt)
        assert np.allclose(A.T @ B, 2 * np.pi * np.eye(3), atol=1e-9)


# ---------------------------------------------------------------------------
# the singular LAUE_PHASE: one helper, no material default
# ---------------------------------------------------------------------------
def test_phase_name_rules(lm, monkeypatch):
    """Set -> itself; unset with one listed phase -> that phase; otherwise exit.
    It used to default to "alpha" in some scripts and "zn" in others."""
    with pytest.raises(SystemExit, match="LAUE_PHASE is not set"):
        lm.phase_name()
    monkeypatch.setenv("LAUE_PHASES", "zn")
    assert lm.phase_name() == "zn"
    monkeypatch.setenv("LAUE_PHASES", "alpha,beta")
    with pytest.raises(SystemExit, match="LAUE_PHASE is not set and LAUE_PHASES lists 2"):
        lm.phase_name()
    monkeypatch.setenv("LAUE_PHASE", "beta")
    assert lm.phase_name() == "beta"
    monkeypatch.setenv("LAUE_PHASE", "zn")
    with pytest.raises(SystemExit, match="not one of LAUE_PHASES"):
        lm.phase_name()
    monkeypatch.delenv("LAUE_PHASES")
    assert lm.phase_name() == "zn"


ALL_ANALYSIS = None


def _all_analysis_scripts():
    return sorted(p for p in _analysis_dir().glob("*.py") if p.name != "laue_material.py")


def test_no_script_reads_laue_phase_directly():
    """Every analysis script -- including the map scripts -- takes the singular phase
    through laue_material.phase_name(); none reads LAUE_PHASE or defaults a phase to
    a material name."""
    bad = []
    for p in _all_analysis_scripts():
        code = "\n".join(l.split("#")[0] for l in p.read_text().splitlines())
        if re.search(r"environ(\.get)?\s*[\(\[]\s*[\"']LAUE_PHASE[\"']", code):
            bad.append(f"{p.name}: reads LAUE_PHASE")
        if re.search(r"Phase\.load\(\s*[\"']zn[\"']", code) or re.search(
                r"else\s+[\"'](zn|alpha|beta)[\"']\s*(#.*)?$", code, re.M):
            bad.append(f"{p.name}: material/phase name as a default")
    assert not bad, bad


# ---- the crystal frame (2026-09-28 code read) --------------------------------
# The indexer builds lattices with a along x; midas_stress's operators assume
# a* along x. For trigonal cells the two frames differ by 30 deg about c, so the
# operators must be conjugated into the indexer's frame before use.

TRIG_HEX = (0.476, 0.476, 1.299, 90, 90, 120)
TRIG_RHOMB = (0.5128, 0.5128, 0.5128, 55.28, 55.28, 55.28)


def _rand_om(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def _rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


# A true crystal symmetry, written in the indexer's frame (a along x):
# -3m1 groups have a 2-fold ALONG a, -31m groups one PERPENDICULAR to a.
@pytest.mark.parametrize("sg,axis_deg", [(150, 0), (164, 0), (166, 0), (167, 0),
                                         (149, 90), (162, 90), (157, 90)])
@pytest.mark.parametrize("use_midas", [True, False])
def test_trigonal_symmetry_equivalent_orientations_are_zero_apart(
        lm, tmp_path, monkeypatch, sg, axis_deg, use_midas):
    if use_midas:
        pytest.importorskip("midas_stress.orientation")
    else:
        monkeypatch.setattr(lm, "_midas_stress", lambda: None)
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def no_ms(name, *a, **k):
            if name.startswith("midas_stress"):
                raise ImportError("blocked for the test")
            return real_import(name, *a, **k)
        monkeypatch.setattr("builtins.__import__", no_ms)
    monkeypatch.setenv("LAUE_PARAMS", str(_params(tmp_path, "t", sg, TRIG_HEX)))
    ph = lm.Phase.load("t")
    rng = np.random.default_rng(sg)
    S = _rot([np.cos(np.radians(axis_deg)), np.sin(np.radians(axis_deg)), 0], 180)
    for _ in range(5):
        A = _rand_om(rng)
        # q = OM @ B @ hkl, so a crystal symmetry S acts as OM -> OM @ S.
        d = ph.misorientation(A, (A @ S)[None])
        assert d[0] < 1e-4, f"SG {sg}: symmetry-equivalent pair reported {d[0]:.3f} deg apart"


@pytest.mark.parametrize("sg,latt", [(194, HEX), (225, CUB), (229, CUB)])
def test_hex_and_cubic_misorientation_unchanged(lm, tmp_path, monkeypatch, sg, latt):
    # The frame fix must not move any hexagonal or cubic result: compare against
    # the plain crystal-side einsum with the conventional operators.
    ms = pytest.importorskip("midas_stress.orientation")
    monkeypatch.setenv("LAUE_PARAMS", str(_params(tmp_path, "h", sg, latt)))
    ph = lm.Phase.load("h")
    rng = np.random.default_rng(7)
    A = _rand_om(rng)
    Bs = np.array([_rand_om(rng) for _ in range(50)])
    ref = np.degrees(np.asarray(ms.misorientation_om_batch(
        np.repeat(A.reshape(1, 9), 50, 0), Bs.reshape(50, 9), sg)))
    assert np.allclose(ph.misorientation(A, Bs), ref, atol=1e-6)


def test_rhombohedral_axes_give_the_indexers_b(lm, tmp_path, monkeypatch):
    # Before 0.8.0 laue_material had no rhombohedral branch, so on rhombohedral
    # axes its B disagreed with the indexer's and every projected spot moved.
    from laue_index.lattice import reciprocal_matrix
    monkeypatch.setenv("LAUE_PARAMS", str(_params(tmp_path, "r", 167, TRIG_RHOMB)))
    ph = lm.Phase.load("r")
    assert np.allclose(ph.B, reciprocal_matrix(TRIG_RHOMB, 167), atol=1e-12)
    rng = np.random.default_rng(3)
    A = _rand_om(rng)
    S = _rot([1, -1, 0], 180)       # a 2-fold of the rhombohedral embedding
    assert ph.misorientation(A, (A @ S)[None])[0] < 1e-4


def test_r_group_on_neither_setting_is_refused(lm, tmp_path, monkeypatch):
    monkeypatch.setenv("LAUE_PARAMS", str(_params(tmp_path, "x", 167, (0.5, 0.5, 0.7, 90, 90, 90))))
    with pytest.raises(ValueError, match="rhombohedral"):
        lm.Phase.load("x")


def test_phase_refuses_an_hkl_list_recorded_for_another_crystal(lm, tmp_path, monkeypatch):
    # data-artifact provenance (0.8.0): the analysis reads the reflection list
    # the params file names; a list recorded for another lattice would make
    # every projected spot wrong, so Phase refuses it.
    from laue_index import artifacts as A
    p = _params(tmp_path, "hx", 194, HEX)
    hkl = tmp_path / "hkls_hx.txt"
    A.write_record(hkl, "hkl_list", config={"SpaceGroup": 194, "LatticeParameter": [0.2921, 0.2921, 0.4665, 90, 90, 120]})
    monkeypatch.setenv("LAUE_PARAMS", str(p))
    with pytest.raises(A.ArtifactMismatch, match="LatticeParameter"):
        lm.Phase.load("hx")
    A.write_record(hkl, "hkl_list", config={"SpaceGroup": 194, "LatticeParameter": list(HEX)})
    assert lm.Phase.load("hx").sgnum == 194
