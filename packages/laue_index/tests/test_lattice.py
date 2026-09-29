"""laue_index.lattice is the one Python copy of calcRecipArray / MakeSymmetries.

GenerateHKLs, GenerateSimulation, calibrate and pipeline/analysis/laue_material
take their B matrix from it; before 0.8.0 each had its own copy, and the
rhombohedral branch existed in two of the four. Pinned here:

* B, the setting refusal and the operator set agree with the COMPILED C (the
  lattice_symmetry fixture) for every crystal system and both R-group settings;
* the Laue-class operators are the same with and without midas_stress (the
  fallback tables are not a second opinion that can drift);
* the frame rotation derived by QR equals the one built from midas_hkls.
"""
import builtins
import os
import subprocess

import numpy as np
import pytest

from _cbuild import FIXTURES, HEADERS, build
from laue_index import lattice as L

HEX = (0.476, 0.476, 1.299, 90.0, 90.0, 120.0)
RHOMB = (0.5128, 0.5128, 0.5128, 55.28, 55.28, 55.28)
CASES = [
    (1, (0.5, 0.6, 0.7, 80.0, 95.0, 105.0)),
    (12, (0.5, 0.6, 0.7, 90.0, 104.0, 90.0)),
    (62, (0.5, 0.6, 0.7, 90.0, 90.0, 90.0)),
    (139, (0.5, 0.5, 0.7, 90.0, 90.0, 90.0)),
    (149, HEX), (150, HEX), (162, HEX), (164, HEX),
    (148, HEX), (148, RHOMB), (167, HEX), (167, RHOMB), (166, RHOMB),
    (194, HEX), (225, (0.36, 0.36, 0.36, 90.0, 90.0, 90.0)),
]


@pytest.fixture(scope="module")
def c_run(tmp_path_factory):
    if not os.path.isfile(HEADERS):
        pytest.skip("c_src not present (installed wheel, not a checkout)")
    d = tmp_path_factory.mktemp("lat")
    exe = str(d / "lattice_symmetry")
    err = build(exe, [os.path.join(FIXTURES, "lattice_symmetry.c")])
    if err is not None and err.startswith("no toolchain"):
        pytest.skip(err)
    assert err is None, err

    def _run(sg, lat):
        out = subprocess.run([exe, str(sg)] + [repr(float(v)) for v in lat],
                             capture_output=True, text=True, check=True).stdout
        res = {"quats": []}
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


@pytest.mark.parametrize("sg,lat", CASES)
def test_reciprocal_matrix_equals_the_c(c_run, sg, lat):
    res = c_run(sg, lat)
    assert np.allclose(L.reciprocal_matrix(lat, sg), res["recip"], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("sg,lat", CASES)
def test_symmetry_quaternions_equal_the_c(c_run, sg, lat):
    c = np.array(c_run(sg, lat)["quats"])
    py = L.symmetry_quaternions(sg, lat)
    assert c.shape == py.shape
    for q in py:  # as a set, q ~ -q; the C tables carry 5 decimals
        assert min(min(np.abs(q - r).max(), np.abs(q + r).max()) for r in c) < 1e-4


@pytest.mark.parametrize("lat", [(0.5, 0.5, 0.7, 90, 90, 90), (0.5, 0.5, 0.5, 90, 90, 125),
                                 (0.5, 0.5, 0.6, 60, 60, 60), HEX, RHOMB])
def test_setting_refusal_equals_the_c(c_run, lat):
    refused_c = c_run(167, lat)["valid"] == 1
    try:
        L.validate_setting(167, lat)
        refused_py = False
    except ValueError:
        refused_py = True
    assert refused_py == refused_c


def test_hexagonal_axes_on_an_r_group_keep_c():
    # Before 0.8.0 (every copy): SG 167 on hexagonal axes was a cube of edge a.
    B = L.reciprocal_matrix(HEX, 167)
    assert np.isclose(np.linalg.norm(B[:, 2]), 2 * np.pi / HEX[2], rtol=1e-12)
    assert np.allclose(B, L.reciprocal_matrix(HEX, None))


def _system_lattice(sg):
    for hi, lat in ((2, CASES[0][1]), (15, CASES[1][1]), (74, CASES[2][1]),
                    (142, CASES[3][1]), (194, HEX), (230, CASES[-1][1])):
        if sg <= hi:
            return lat


def _ops_without_midas(monkeypatch, sg, lat):
    real = builtins.__import__

    def blocked(name, *a, **k):
        if name.startswith("midas_stress"):
            raise ImportError("blocked for the test")
        return real(name, *a, **k)
    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", blocked)
        return L.laue_class_operators(sg, lat)


@pytest.mark.parametrize("sg", list(range(1, 231)))
def test_laue_class_operators_are_the_same_with_and_without_midas_stress(monkeypatch, sg):
    pytest.importorskip("midas_stress.orientation")
    lat = _system_lattice(sg)
    with_ms = L.laue_class_operators(sg, lat)
    without = _ops_without_midas(monkeypatch, sg, lat)
    assert len(with_ms) == len(without)
    for S in with_ms:
        assert any(np.allclose(S, T, atol=1e-4) for T in without)
    A = L.direct_basis(lat, sg)
    for S in with_ms:
        Mi = np.linalg.inv(A) @ S @ A
        assert np.allclose(Mi, np.rint(Mi), atol=1e-4)


def test_frame_rotation_matches_midas_hkls():
    nf = pytest.importorskip("midas_hkls.nf_hkls")
    lattice_mod = pytest.importorskip("midas_hkls.lattice")
    for lat in (HEX, (0.5, 0.6, 0.7, 90, 104, 90), (0.5, 0.6, 0.7, 80, 95, 105)):
        a, b, c, al, be, ga = lat
        Bm = nf._b_matrix(lattice_mod.Lattice(a=a, b=b, c=c, alpha=al, beta=be, gamma=ga))
        M = L.direct_basis(lat) @ np.linalg.inv(np.linalg.inv(Bm).T)
        assert np.allclose(L.midas_frame_rotation(lat), M, atol=1e-9)


# ---- every Python copy now delegates (C2) ------------------------------------

def _gen_hkls(sg, lat):
    import types
    import GenerateHKLs
    args = types.SimpleNamespace(resultFileName="unused", sym="R" if sg in L.R_GROUPS else "P",
                                 latticeParameter=list(lat), sgnum=sg, RArray=[0, 0, 0],
                                 PArray=[0, 0, 0.5], NumPxX=16, NumPxY=16, dx=2e-4, dy=2e-4, Ehi=30)
    return GenerateHKLs.LaueMatching(args)


@pytest.mark.parametrize("sg,lat", [(167, HEX), (167, RHOMB), (194, HEX), (12, CASES[1][1])])
def test_generate_hkls_uses_the_shared_b(sg, lat):
    assert np.allclose(_gen_hkls(sg, lat).calcRecipArray(), L.reciprocal_matrix(lat, sg),
                       rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("sg,lat", [(167, HEX), (167, RHOMB), (194, HEX), (12, CASES[1][1])])
def test_generate_simulation_uses_the_shared_b(sg, lat):
    import GenerateSimulation
    assert np.allclose(GenerateSimulation.calc_recip_array(lat, sg), L.reciprocal_matrix(lat, sg),
                       rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("sg,lat", [(167, HEX), (167, RHOMB), (194, HEX), (12, CASES[1][1])])
def test_calibrate_uses_the_shared_b(sg, lat):
    import importlib
    calibrate = importlib.import_module("laue_index.calibrate")  # the package exports a function of that name
    assert np.allclose(calibrate.reciprocal_matrix(lat, sg), L.reciprocal_matrix(lat, sg),
                       rtol=1e-10, atol=1e-10)


def test_generate_hkls_refuses_an_r_group_on_neither_setting():
    with pytest.raises((ValueError, SystemExit)):
        _gen_hkls(167, (0.5, 0.5, 0.7, 90, 90, 90))
