"""The C symmetry operators must be symmetries of the lattice the C builds.

LaueMatching builds every lattice with a along Cartesian x (calcRecipArray);
MIDAS, and so midas_stress, builds it with a* along x. The two frames coincide
except for trigonal/hexagonal and monoclinic cells, so an operator table is
only correct in the frame its B matrix was built in. Three defects this pins
(found in the 2026-09-28 code read, all in LaueMatchingHeaders.h):

- OrtSym[1] was {1,1,0,0}: normalised, a 90 deg turn about x, not a 2-fold.
- MonoSym[1] was a 2-fold about x; b lies along y, so the 2-fold is about y.
- TrigSym put the 2-folds perpendicular to a for EVERY trigonal group. That is
  -31m (149, 151, 153, 157, 159, 162, 163, and P-3 by choice); the -3m1 groups
  (150, 152, 154, 155, 156, 158, 160, 161, 164-167) have them ALONG a, and for
  the R groups on hexagonal axes the perpendicular set breaks R-centring.

And the rhombohedral lattice setting (C2): calcRecipArray chose the
rhombohedral embedding from the space-group number alone, so hexagonal axes
(a, a, c, 90, 90, 120) for SG 167 produced a cube of edge a and ignored c.
The setting is now read from the lattice angles, with a table whose 3-fold is
along [111] for rhombohedral axes, and anything else is refused.

Checks, per case: every operator is a unit quaternion; the set is closed under
multiplication (q ~ -q); A^-1 R A is an integer matrix for the direct basis A
the C itself builds; R-centring is preserved for R groups on hexagonal axes;
and the set equals midas_stress's operators conjugated into this frame (skipped
without midas_stress). The frame change M = A_LM A_MIDAS^-1 is computed from
the two bases, not assumed.
"""
import os
import re
import subprocess

import numpy as np
import pytest

from _cbuild import FIXTURES, HEADERS, build

FIXTURE = os.path.join(FIXTURES, "lattice_symmetry.c")

HEX = (0.476, 0.476, 1.299, 90.0, 90.0, 120.0)
RHOMB = (0.5128, 0.5128, 0.5128, 55.28, 55.28, 55.28)
LATTICES = {
    "tric": (0.5, 0.6, 0.7, 80.0, 95.0, 105.0),
    "mono": (0.5, 0.6, 0.7, 90.0, 104.0, 90.0),
    "orth": (0.5, 0.6, 0.7, 90.0, 90.0, 90.0),
    "tetr": (0.5, 0.5, 0.7, 90.0, 90.0, 90.0),
    "trig": HEX,
    "hexa": HEX,
    "cubi": (0.5, 0.5, 0.5, 90.0, 90.0, 90.0),
}
R_GROUPS = {146, 148, 155, 160, 161, 166, 167}
R_CENTRING = [np.array([2 / 3, 1 / 3, 1 / 3]), np.array([1 / 3, 2 / 3, 2 / 3])]


def _system(sg):
    for hi, name in ((2, "tric"), (15, "mono"), (74, "orth"), (142, "tetr"),
                     (167, "trig"), (194, "hexa"), (230, "cubi")):
        if sg <= hi:
            return name


def _q2m(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _direct_from_recip(recip):
    # recip columns are a*, b*, c* with the 2*pi: A = 2*pi * inv(B)^T.
    return 2 * np.pi * np.linalg.inv(np.asarray(recip)).T


def _is_lattice_symmetry(R, A, r_centred):
    M = np.linalg.inv(A) @ R @ A
    if not np.allclose(M, np.rint(M), atol=1e-4):
        return False
    if r_centred:
        M = np.rint(M)
        return all(any(np.allclose((M @ t) % 1, s, atol=1e-6) for s in R_CENTRING)
                   for t in R_CENTRING)
    return True


def _closed(ms):
    for a in ms:
        for b in ms:
            if not any(np.allclose(a @ b, c, atol=1e-4) for c in ms):
                return False
    return True


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    if not os.path.isfile(HEADERS):
        pytest.skip("c_src not present (installed wheel, not a checkout)")
    d = tmp_path_factory.mktemp("latsym")
    exe = str(d / "lattice_symmetry")
    err = build(exe, [FIXTURE])
    if err is not None and err.startswith("no toolchain"):
        pytest.skip(err)
    assert err is None, f"fixture failed to compile against the header:\n{err}"

    def _run(sg, lat):
        out = subprocess.run([exe, str(sg)] + [repr(float(v)) for v in lat],
                             capture_output=True, text=True, check=True).stdout
        res = {"valid": None, "recip": None, "quats": []}
        for line in out.splitlines():
            tag, *vals = line.split()
            if tag == "VALID":
                res["valid"] = int(vals[0])
            elif tag == "RECIP":
                res["recip"] = np.array(vals, float).reshape(3, 3)
            elif tag == "Q":
                res["quats"].append(np.array(vals, float))
        return res
    return _run


# ---- source-level: the static tables themselves ---------------------------

def _tables():
    txt = open(HEADERS).read()
    out = {}
    for name, body in re.findall(r"static double (\w+Sym\w*)\[\d+\]\[4\]\s*=\s*\{(.*?)\};",
                                 txt, re.S):
        rows = re.findall(r"\{([^{}]*)\}", body)
        out[name] = [np.array([float(v) for v in r.split(",")]) for r in rows]
    return out


def test_every_static_table_row_is_a_unit_quaternion():
    if not os.path.isfile(HEADERS):
        pytest.skip("c_src not present")
    for name, rows in _tables().items():
        for i, q in enumerate(rows):
            assert abs(np.linalg.norm(q) - 1) < 1e-4, f"{name}[{i}] = {q} is not unit"


def test_every_static_table_is_closed_under_multiplication():
    if not os.path.isfile(HEADERS):
        pytest.skip("c_src not present")
    for name, rows in _tables().items():
        ms = [_q2m(q) for q in rows]
        assert _closed(ms), f"{name} is not a group"


# ---- compiled: what MakeSymmetries / calcRecipArray actually return --------

@pytest.mark.parametrize("sg", list(range(1, 231)))
def test_operators_are_symmetries_of_the_lattice_the_c_builds(run, sg):
    lat = LATTICES[_system(sg)]
    res = run(sg, lat)
    assert res["valid"] == 0
    A = _direct_from_recip(res["recip"])
    ms = [_q2m(q) for q in res["quats"]]
    assert len(ms) >= 1 and _closed(ms), f"SG {sg}: operator set is not a group"
    for q, R in zip(res["quats"], ms):
        assert _is_lattice_symmetry(R, A, sg in R_GROUPS), (
            f"SG {sg}: {np.round(q, 5)} does not map the lattice onto itself")


@pytest.mark.parametrize("sg", sorted(R_GROUPS))
def test_rhombohedral_axes_use_the_rhombohedral_embedding(run, sg):
    res = run(sg, RHOMB)
    assert res["valid"] == 0
    A = _direct_from_recip(res["recip"])
    lengths = np.linalg.norm(A, axis=0)
    assert np.allclose(lengths, RHOMB[0], rtol=1e-9)
    cosang = [A[:, i] @ A[:, j] / RHOMB[0] ** 2 for i, j in ((1, 2), (0, 2), (0, 1))]
    assert np.allclose(cosang, np.cos(np.radians(RHOMB[3])), atol=1e-9)
    for R in (_q2m(q) for q in res["quats"]):
        assert _is_lattice_symmetry(R, A, False)


@pytest.mark.parametrize("sg", sorted(R_GROUPS))
def test_hexagonal_axes_on_an_r_group_keep_c(run, sg):
    # Before the fix this returned a cube of edge a: |c*| = 2 pi / a.
    res = run(sg, HEX)
    assert res["valid"] == 0
    cstar = np.linalg.norm(res["recip"][:, 2])
    assert np.isclose(cstar, 2 * np.pi / HEX[2], rtol=1e-9)


@pytest.mark.parametrize("lat", [(0.5, 0.5, 0.7, 90.0, 90.0, 90.0),
                                 (0.5, 0.5, 0.5, 90.0, 90.0, 125.0),
                                 (0.5, 0.5, 0.6, 60.0, 60.0, 60.0)])
def test_r_group_with_neither_setting_is_refused(run, lat):
    assert run(167, lat)["valid"] == 1


def _midas_frame_direct(lat):
    lattice_mod = pytest.importorskip("midas_hkls.lattice")
    nf = pytest.importorskip("midas_hkls.nf_hkls")
    a, b, c, al, be, ga = lat
    B = nf._b_matrix(lattice_mod.Lattice(a=a, b=b, c=c, alpha=al, beta=be, gamma=ga))
    return np.linalg.inv(B).T


@pytest.mark.parametrize("sg", [3, 16, 75, 89, 143, 149, 150, 155, 164, 166, 167,
                                168, 177, 191, 194, 195, 207, 225, 229])
def test_matches_midas_stress_conjugated_into_this_frame(run, sg):
    orient = pytest.importorskip("midas_stress.orientation")
    lat = LATTICES[_system(sg)]
    res = run(sg, lat)
    A_lm = _direct_from_recip(res["recip"])
    M = A_lm @ np.linalg.inv(_midas_frame_direct(lat))
    assert np.allclose(M @ M.T, np.eye(3), atol=1e-9) and np.linalg.det(M) > 0
    n, ops = orient.make_symmetries(sg)
    theirs = [M @ _q2m(q) @ M.T for q in ops[:n]]
    ours = [_q2m(q) for q in res["quats"]]
    # The C keeps the full rotation group of the crystal system (a design
    # choice: operators that are lattice but not crystal symmetries give the
    # same spot positions). midas_stress uses the Laue class. So: every
    # midas_stress operator must be one of ours.
    for R in theirs:
        assert any(np.allclose(R, S, atol=1e-4) for S in ours), (
            f"SG {sg}: midas_stress operator missing from the C set")
