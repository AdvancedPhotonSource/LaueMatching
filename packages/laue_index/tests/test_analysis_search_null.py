"""The SEARCH null (invariant 29): the same indexer and validator on spot-scrambled frames.

End to end on a tiny synthetic scan through the REAL CPU indexer binary:

  (a) scrambled frames give a search-null maximum BELOW the true crystal's nhit;
  (b) the truth is still recovered on the real (unscrambled) frame by the same
      pipeline -- the driver's own positive control;
  (c) negative control: a "scramble" that KEEPS positions gives a null at the real
      crystal's level, so the gate would flag it -- the null can fail. Such a block
      is refused by every gate.

Plus the scrambler's invariants (component count, intensities, lit-pixel total, no
spot on a masked pixel or gap) and the gates' choice of null (search by default,
per-draw fallback with a loud warning, LAUE_NULL_KIND override).

Robust to a rebuilt binary (pixel rounding, symmetry tables): nothing here is a
pixel-exact golden number; the assertions are orderings and tolerances.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    from conftest import _REPO_ROOT
except ImportError:  # pragma: no cover
    _REPO_ROOT = None


def _analysis_dir() -> Path:
    if _REPO_ROOT is None:
        pytest.skip("not running from a repo checkout (pipeline/analysis absent)")
    d = Path(_REPO_ROOT) / "pipeline" / "analysis"
    if not d.is_dir():
        pytest.skip(f"{d} not present")
    return d


@pytest.fixture
def ana():
    d = str(_analysis_dir())
    if d not in sys.path:
        sys.path.insert(0, d)
    return d


def _clean_env(**kw):
    env = {k: v for k, v in os.environ.items() if not k.startswith("LAUE_")}
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env.update({k: str(v) for k, v in kw.items()})
    return env


def _rand_om(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _cubic_ops():
    import itertools
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for i, (j, s) in enumerate(zip(perm, signs)):
                m[i, j] = s
            if np.linalg.det(m) > 0:
                ops.append(m)
    return ops


def _miso_cubic_deg(a, b):
    best = 180.0
    for s in _cubic_ops():
        tr = np.trace(a @ s @ b.T)
        best = min(best, np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))))
    return best


NPX, PX = 256, 0.0016


def _binary():
    try:
        from laue_index import indexer
    except ImportError:
        pytest.skip("laue_index not importable")
    if not indexer.available("CPU"):
        pytest.skip("LaueMatchingCPU binary not available")
    return str(indexer.binary_path("CPU"))


@pytest.fixture(scope="module")
def scan(tmp_path_factory):
    """Two frames, each one true Ni crystal; a 3,000-orientation DB holding both truths."""
    _binary()
    h5py = pytest.importorskip("h5py")
    d = tmp_path_factory.mktemp("search_null")
    hk = sorted([(h, k, l) for h in range(-6, 7) for k in range(-6, 7) for l in range(-6, 7)
                 if (h, k, l) != (0, 0, 0) and h % 2 == k % 2 == l % 2],
                key=lambda t: t[0] ** 2 + t[1] ** 2 + t[2] ** 2)
    np.savetxt(d / "hkls.csv", np.array(hk), fmt="%d")
    params = d / "params_ni.txt"
    params.write_text(
        "SpaceGroup 225\nSymmetry F\nLatticeParameter 0.35238 0.35238 0.35238 90 90 90\n"
        "P_Array 0.028828 0.002715 0.512993\nR_Array -1.20161887 -1.21404493 -1.21852276\n"
        f"PxX {PX}\nPxY {PX}\nNrPxX {NPX}\nNrPxY {NPX}\nElo 5\nEhi 30\n"
        "MinNrSpots 3\nMinIntensity 0\nMaxAngle 2\nMaxNrLaueSpots 100\n"
        f"ForwardFile {d / 'unused_fwd.bin'}\nDoFwd 1\nOrientationSpacing 0.4\n"
        f"OrientationFile {d / 'db.bin'}\nHKLFile {d / 'hkls.csv'}\n")
    sys.path.insert(0, str(_analysis_dir()))
    from laue_material import Phase
    ph = Phase(str(params), "ni")
    rng = np.random.default_rng(7)
    truths = []
    while len(truths) < 2:
        om = _rand_om(rng)
        pr = ph.project(om)
        if len(np.unique(np.rint(pr).astype(int), axis=0)) >= 12:
            truths.append(om)
    db = np.array([_rand_om(rng) for _ in range(3000)])
    db[1234], db[2345] = truths
    db.astype(np.float64).tofile(d / "db.bin")

    data, run = d / "data", d / "run"
    (run / "results").mkdir(parents=True)
    data.mkdir()
    yy, xx = np.mgrid[0:NPX, 0:NPX]
    mapping = {}
    for i, om in enumerate(truths, start=1):
        raw = rng.normal(100, 3, size=(NPX, NPX))
        for x, y in ph.project(om):
            raw += 4000 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.3 ** 2))
        fn = f"ni_{i:05d}.h5"
        with h5py.File(data / fn, "w") as h:
            h.create_dataset("entry1/data/data", data=raw.astype(np.float32))
        # the indexer's segmented input, as RunImage writes it: background-subtracted,
        # thresholded, components below MinArea dropped
        sub = np.clip(raw - 100.0, 0, None)
        seg = np.where(sub > 300, sub, 0).astype(np.uint16)
        from scipy import ndimage as ndi
        lab, n = ndi.label(seg > 0, structure=np.ones((3, 3)))
        area = np.bincount(lab.ravel())
        seg[(area[lab] < 4) & (lab > 0)] = 0
        with h5py.File(run / "results" / f"image_{i:05d}.output.h5", "w") as h:
            h.create_dataset("entry/data/cleaned_data_threshold_filtered", data=seg)
        mapping[str(i)] = {"file": fn}
    (run / "frame_mapping.json").write_text(json.dumps(mapping))
    work = d / "work"
    (work / "peel_map").mkdir(parents=True)
    env = _clean_env(LAUE_WORK=work, LAUE_SCAN_DATA=data, LAUE_SCAN_NI=run,
                     LAUE_PHASES="ni", LAUE_PARAMS_NI=params, LAUE_OUT_PREFIX="syn")
    return {"dir": d, "env": env, "work": work, "truths": truths, "params": params}


def _run_search_null(scan, *extra):
    r = subprocess.run([sys.executable, "search_null.py", "ni", "2", "4", "2", *extra],
                       cwd=_analysis_dir(), env=scan["env"], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r


@pytest.fixture(scope="module")
def search_block(scan):
    r = _run_search_null(scan)
    js = json.loads((scan["work"] / "peel_map" / "syn_null.json").read_text())
    return js["phases"]["ni"]["search_null"], r.stdout


@pytest.fixture(scope="module")
def control_block(scan):
    out = scan["dir"] / "negative_control.json"
    _run_search_null(scan, "--keep-positions", "--out", str(out))
    return json.loads(out.read_text())["phases"]["ni"]["search_null"]


def test_real_frame_recovers_the_truth(search_block, scan):
    """(b) Positive control: the null's own pipeline finds the true crystal on the
    real frame, with a validator score well above anything a scramble reached."""
    blk, _ = search_block
    real = blk["real_control"]["per_frame"]
    assert len(real) == 2
    for rec, truth in zip(real, scan["truths"]):
        assert rec["n_solutions"] >= 1, rec
        assert rec["best_nhit"] >= 10, rec
        om = np.array(rec["best_om"]).reshape(3, 3)
        assert _miso_cubic_deg(om, truth) < 1.0, rec


def test_scrambled_search_stays_below_the_true_crystal(search_block):
    """(a) Every scrambled search's best is below the true crystal's nhit, and the
    scramble preserved what it must."""
    blk, out = search_block
    assert blk["n_searches"] == 8 and blk["kind"] == "search" and not blk["keep_positions"]
    true_min = min(r["best_nhit"] for r in blk["real_control"]["per_frame"])
    assert blk["nhit"]["max"] < true_min, (blk["nhit"], true_min)
    assert blk["nhit_distinct"]["max"] < min(
        r["best_nhit_distinct"] for r in blk["real_control"]["per_frame"])
    # NOT asserted zero: on this sparse frame the validator's analytic Poisson gate
    # (p < 1e-4) DOES pass scrambled solutions the search went looking for -- the
    # point of invariant 29. The bar that holds is the search maximum above.
    assert blk["n_validated"] >= 0 and blk["searches_with_solution"] > 0
    assert blk["n_orientations"] == 3000  # file size / 72 bytes
    for st in ("nhit", "nhit_distinct", "nmatches"):
        assert blk[st]["n_draws"] == 8 and blk[st]["statistic"] == st
    assert "SEARCH NULL" in out


def test_negative_control_keeps_positions_and_would_flag_the_truth(control_block, search_block):
    """(c) With positions kept the 'null' reaches the real crystal's score: a gate
    `nhit > null max` would then REJECT the true crystal. The method can fail."""
    blk, _ = search_block
    true_min = min(r["best_nhit"] for r in blk["real_control"]["per_frame"])
    assert control_block["keep_positions"] is True
    assert control_block["nhit"]["max"] >= true_min, control_block["nhit"]


def test_a_negative_control_block_can_never_gate(control_block, tmp_path, ana, monkeypatch):
    import frame_peaks as fp
    (tmp_path / "peel_map").mkdir()
    (tmp_path / "peel_map" / "c_null.json").write_text(json.dumps(
        {"schema": 1, "phases": {"ni": {"search_null": control_block}}}))
    monkeypatch.delenv("LAUE_NULL_KIND", raising=False)
    monkeypatch.delenv("LAUE_NULLMAX_NI", raising=False)
    with pytest.raises(SystemExit, match="negative control"):
        fp.load_null("ni", str(tmp_path), "c", "nhit")


def test_search_block_sits_beside_the_per_draw_null(scan, search_block, ana, monkeypatch):
    """The driver merges into <prefix>_null.json without touching null_model.py's
    entries, and the gate picks the search block by default."""
    import frame_peaks as fp
    blk, _ = search_block
    path = scan["work"] / "peel_map" / "syn_null.json"
    js = json.loads(path.read_text())
    draw = {"statistic": "nhit", "n_draws": 10, "mean": 1, "median": 1, "p99": 3,
            "p999": 3, "max": 3}
    js["phases"]["ni"]["nhit"] = draw
    js["phases"]["ni"]["nhit_distinct"] = dict(draw, statistic="nhit_distinct", max=2)
    path.write_text(json.dumps(js))
    monkeypatch.delenv("LAUE_NULL_KIND", raising=False)
    monkeypatch.delenv("LAUE_NULLMAX_NI", raising=False)
    got = fp.load_null("ni", str(scan["work"]), "syn", "nhit")
    assert got["kind"] == "search" and got["max"] == blk["nhit"]["max"]
    monkeypatch.setenv("LAUE_NULL_KIND", "draw")
    got = fp.load_null("ni", str(scan["work"]), "syn", "nhit")
    assert got["kind"] == "draw" and got["max"] == 3


# ---------------------------------------------------------------------------
# the scrambler alone
# ---------------------------------------------------------------------------
def test_scramble_preserves_components_intensities_and_avoids_masks(ana):
    import scramble_frames as sf
    rng = np.random.default_rng(3)
    img = np.zeros((180, 240), np.uint16)
    for _ in range(40):
        y, x = rng.integers(5, 175), rng.integers(5, 235)
        img[y:y + rng.integers(2, 7), x:x + rng.integers(2, 7)] = rng.integers(50, 9000)
    mask = np.zeros(img.shape, bool)
    mask[:, 100:112] = True              # module gap
    mask[20:60, 20:60] = True            # an excluded region
    img[mask] = 0
    for seed in range(5):
        out, _, info = sf.scramble_frame(img, np.random.default_rng(seed), allowed=~mask)
        assert info["n_components_out"] == info["n_components"]
        assert info["lit_px_out"] == info["lit_px"]
        assert sorted(out[out > 0].tolist()) == sorted(img[img > 0].tolist())
        assert not (out[mask] > 0).any()
        assert not np.array_equal(out, img)


def test_band_support_keeps_spots_in_the_frames_2theta_band(ana, tmp_path):
    import scramble_frames as sf
    from laue_material import Phase
    p = tmp_path / "p.txt"
    np.savetxt(tmp_path / "h.txt", np.array([[1, 1, 1], [2, 0, 0]]), fmt="%d")
    p.write_text("SpaceGroup 225\nSymmetry F\nLatticeParameter 0.35 0.35 0.35 90 90 90\n"
                 "P_Array 0.028828 0.002715 0.512993\nR_Array -1.2016 -1.2140 -1.2185\n"
                 f"PxX {PX}\nPxY {PX}\nNrPxX {NPX}\nNrPxY {NPX}\nElo 5\nEhi 30\n"
                 f"HKLFile {tmp_path / 'h.txt'}\n")
    tth = sf.twotheta_map(Phase(str(p), "x"))
    lit = np.zeros((NPX, NPX), bool)
    lit[100:110, 100:110] = True
    lit[150:155, 60:65] = True
    allowed = sf.support_mask(lit, None, tth)
    lo, hi = tth[lit].min(), tth[lit].max()
    assert allowed.any() and not allowed.all()
    assert (tth[allowed] >= lo - 1e-9).all() and (tth[allowed] <= hi + 1e-9).all()
    # the 2theta map is the inverse of Phase.project: a projected reflection's
    # pixel has the 2theta of its own scattering vector
    ph = Phase(str(p), "x")
    om = np.eye(3)
    q = (om @ ph.B @ ph.hkls.T).T
    pr = ph.project(om)
    if len(pr):
        x, y = pr[0]
        qh = q / np.linalg.norm(q, axis=1, keepdims=True)
        want = np.degrees(np.arccos(1 - 2 * qh[:, 2] ** 2))   # cos 2theta = 1 - 2 sin^2 theta
        got = tth[int(round(y)), int(round(x))]
        assert np.min(np.abs(want - got)) < 1.0


# ---------------------------------------------------------------------------
# the gates' choice of null
# ---------------------------------------------------------------------------
def _json(tmp_path, with_search=True, with_draw=True):
    rec = lambda s, mx, kind: {"statistic": s, "n_draws": 10, "mean": 1.0, "median": 1.0,
                               "p99": 3.0, "p999": 4.0, "max": mx, "kind": kind}
    ent = {}
    if with_draw:
        ent.update(nhit=rec("nhit", 10, "draw"), nhit_distinct=rec("nhit_distinct", 5, "draw"))
    if with_search:
        ent["search_null"] = {"keep_positions": False,
                              "nhit": rec("nhit", 7, "search"),
                              "nhit_distinct": rec("nhit_distinct", 4, "search")}
    (tmp_path / "peel_map").mkdir(exist_ok=True)
    (tmp_path / "peel_map" / "s_null.json").write_text(json.dumps({"schema": 1, "phases": {"zn": ent}}))


def test_gate_defaults_to_the_search_null(tmp_path, ana, monkeypatch, capsys):
    import frame_peaks as fp
    _json(tmp_path)
    monkeypatch.delenv("LAUE_NULL_KIND", raising=False)
    monkeypatch.delenv("LAUE_NULLMAX_ZN", raising=False)
    got = fp.load_null("zn", str(tmp_path), "s", "nhit")
    assert got["kind"] == "search" and got["max"] == 7
    assert "SEARCH null" in capsys.readouterr().out
    assert fp.load_null("zn", str(tmp_path), "s", "nhit_distinct")["max"] == 4


def test_gate_falls_back_to_the_per_draw_null_loudly(tmp_path, ana, monkeypatch, capsys):
    import frame_peaks as fp
    _json(tmp_path, with_search=False)
    monkeypatch.delenv("LAUE_NULL_KIND", raising=False)
    monkeypatch.delenv("LAUE_NULLMAX_ZN", raising=False)
    got = fp.load_null("zn", str(tmp_path), "s", "nhit")
    assert got["kind"] == "draw" and got["max"] == 10
    cap = capsys.readouterr()
    assert "WARNING: no SEARCH null" in cap.err and "PER-DRAW null" in cap.out


def test_null_kind_override(tmp_path, ana, monkeypatch):
    import frame_peaks as fp
    _json(tmp_path)
    monkeypatch.delenv("LAUE_NULLMAX_ZN", raising=False)
    monkeypatch.setenv("LAUE_NULL_KIND", "draw")
    assert fp.load_null("zn", str(tmp_path), "s", "nhit")["max"] == 10
    monkeypatch.setenv("LAUE_NULL_KIND", "bogus")
    with pytest.raises(SystemExit):
        fp.load_null("zn", str(tmp_path), "s", "nhit")
    _json(tmp_path, with_search=False)
    monkeypatch.setenv("LAUE_NULL_KIND", "search")
    with pytest.raises(SystemExit, match="search_null.py"):
        fp.load_null("zn", str(tmp_path), "s", "nhit")


def test_env_override_checks_the_other_statistic_of_the_same_kind(tmp_path, ana, monkeypatch):
    """LAUE_NULLMAX_<PHASE> equal to the OTHER statistic's SEARCH max is refused."""
    import frame_peaks as fp
    _json(tmp_path)
    monkeypatch.delenv("LAUE_NULL_KIND", raising=False)
    monkeypatch.setenv("LAUE_NULLMAX_ZN", "4")
    with pytest.raises(SystemExit, match="same statistic"):
        fp.load_null("zn", str(tmp_path), "s", "nhit")
