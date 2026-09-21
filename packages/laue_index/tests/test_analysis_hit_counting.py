"""Hit counting, the measured null, and the gates that consume it.

Behaviours pinned (all on synthetic inputs; no beamtime data needed):

* ``frame_peaks.count_matched_peaks`` returns ``(n_distinct, n_predicted)``:
  a HARMONIC pair -- two predicted reflections on one pixel, as (1,1,1) and
  (2,2,2) give, because ``Phase.project`` returns one row per hkl with no q-hat
  dedup -- is 1 distinct peak but 2 predicted hits. It used to sit after the
  module's ``__main__`` block, where the selftest never reached it.
* ``null_model.py`` writes ``peel_map/<prefix>_null.json`` carrying BOTH
  statistics, and ``frame_peaks.load_null`` reads back exactly what was written.
* A gate and its null are the same statistic: an ``LAUE_NULLMAX_<PHASE>`` equal
  to the OTHER statistic's measured maximum is refused.
* ``empirical_gate.py`` and ``validated_figures.py`` exit non-zero when no null
  exists for the scan -- they used to carry a hard-coded Ti null and apply it to
  every scan.
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
def fp():
    d = str(_analysis_dir())
    if d not in sys.path:
        sys.path.insert(0, d)
    import frame_peaks
    return frame_peaks


def _clean_env(**kw):
    """A child environment with none of the analysis variables inherited."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("LAUE_")}
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env.update({k: str(v) for k, v in kw.items()})
    return env


# ---------------------------------------------------------------------------
# count_matched_peaks
# ---------------------------------------------------------------------------
def test_harmonic_pair_is_one_distinct_peak_two_predicted(fp):
    from scipy.spatial import cKDTree
    tree = cKDTree(np.array([[100.0, 100.0], [300.0, 300.0]]))
    # (1,1,1) and (2,2,2) share a scattering direction: same pixel
    harmonic = np.array([[100.0, 100.0], [100.0, 100.0]])
    assert fp.count_matched_peaks(tree, harmonic, 8.0) == (1, 2)


def test_two_separate_peaks_count_two_both_ways(fp):
    from scipy.spatial import cKDTree
    tree = cKDTree(np.array([[100.0, 100.0], [300.0, 300.0]]))
    assert fp.count_matched_peaks(tree, np.array([[101.0, 99.0], [302.0, 300.0]]), 8.0) == (2, 2)


def test_misses_and_empty_prediction_count_zero(fp):
    from scipy.spatial import cKDTree
    tree = cKDTree(np.array([[100.0, 100.0]]))
    assert fp.count_matched_peaks(tree, np.array([[200.0, 200.0]]), 8.0) == (0, 0)
    assert fp.count_matched_peaks(tree, np.empty((0, 2)), 8.0) == (0, 0)


def test_exclude_mask_drops_claimed_peaks(fp):
    """The alpha-exclusion census counts only predictions whose nearest peak is
    unclaimed; the harmonic pair on the claimed peak must vanish from BOTH counts."""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.array([[100.0, 100.0], [300.0, 300.0]]))
    pr = np.array([[100.0, 100.0], [100.0, 100.0], [300.0, 300.0]])
    assert fp.count_matched_peaks(tree, pr, 8.0) == (2, 3)
    assert fp.count_matched_peaks(tree, pr, 8.0, exclude=np.array([True, False])) == (1, 1)


def test_count_matched_peaks_is_in_the_module_body_not_after_main(fp):
    """Defined above ``if __name__ == "__main__":`` so the selftest exercises it."""
    src = (_analysis_dir() / "frame_peaks.py").read_text()
    assert src.index("def count_matched_peaks(") < src.index('if __name__ == "__main__":')
    main_block = src[src.index('if __name__ == "__main__":'):]
    assert "count_matched_peaks(" in main_block


def test_gate_statistic_default_and_refusal(fp, monkeypatch):
    monkeypatch.delenv("LAUE_GATE_STAT", raising=False)
    assert fp.gate_statistic() == "nhit"
    monkeypatch.setenv("LAUE_GATE_STAT", "nhit_distinct")
    assert fp.gate_statistic() == "nhit_distinct"
    monkeypatch.setenv("LAUE_GATE_STAT", "distinct")
    with pytest.raises(SystemExit):
        fp.gate_statistic()


# ---------------------------------------------------------------------------
# null_model.py -> json -> load_null
# ---------------------------------------------------------------------------
def _write_params(d: Path, name="zn", sg=194, latt=(0.2665, 0.2665, 0.4947, 90, 90, 120),
                  npx=512, px=0.0008) -> Path:
    """A small but physically sensible params file: the Ni fixture's detector
    geometry at 4x coarser pixels, and a generated reflection list."""
    hk = []
    for h in range(-5, 6):
        for k in range(-5, 6):
            for l in range(-5, 6):
                if (h, k, l) != (0, 0, 0):
                    hk.append((h, k, l))
    hkl = d / f"hkls_{name}.txt"
    np.savetxt(hkl, np.array(hk), fmt="%d")
    p = d / f"params_{name}.txt"
    p.write_text(
        f"SpaceGroup {sg}\nSymmetry P\n"
        f"LatticeParameter {' '.join(str(v) for v in latt)}\n"
        "P_Array 0.028828 0.002715 0.512993\n"
        "R_Array -1.20161887 -1.21404493 -1.21852276\n"
        f"PxX {px}\nPxY {px}\nNrPxX {npx}\nNrPxY {npx}\n"
        "Elo 5.0\nEhi 30.0\nMinNrSpots 3\n"
        f"HKLFile {hkl}\n")
    return p


def _write_frames(data: Path, run: Path, n=2, npx=512, seed=0):
    import h5py
    rng = np.random.default_rng(seed)
    data.mkdir(parents=True, exist_ok=True)
    run.mkdir(parents=True, exist_ok=True)
    mapping = {}
    yy, xx = np.mgrid[0:npx, 0:npx]
    for i in range(1, n + 1):
        f = rng.normal(100, 3, size=(npx, npx))
        for _ in range(40):
            cy, cx = rng.uniform(20, npx - 20, size=2)
            f += 4000 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0)
        fn = f"frame_{i:05d}.h5"
        with h5py.File(data / fn, "w") as h:
            h.create_dataset("entry1/data/data", data=f.astype(np.float32))
        mapping[str(i)] = {"file": fn}
    (run / "frame_mapping.json").write_text(json.dumps(mapping))


@pytest.fixture
def null_json(tmp_path):
    """Run null_model.py end-to-end on a synthetic two-frame scan."""
    pytest.importorskip("h5py")
    ana = _analysis_dir()
    work, data, run = tmp_path / "work", tmp_path / "data", tmp_path / "run"
    work.mkdir()
    params = _write_params(tmp_path)
    _write_frames(data, run)
    env = _clean_env(LAUE_WORK=work, LAUE_SCAN_DATA=data, LAUE_SCAN_ZN=run,
                     LAUE_PHASES="zn", LAUE_PARAMS_ZN=params, LAUE_OUT_PREFIX="syn")
    # Plain interpreter, default start method: 'spawn' on macOS, which re-imports
    # the script in the worker -- this is also the test that its __main__ guard works.
    r = subprocess.run([sys.executable, "null_model.py", "2", "25", "1"], cwd=ana, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    p = work / "peel_map" / "syn_null.json"
    assert p.is_file(), r.stdout[-2000:]
    return work, p, r.stdout


def test_null_json_contract(null_json):
    work, p, out = null_json
    js = json.loads(p.read_text())
    assert js["schema"] == 1 and js["prefix"] == "syn" and js["tol_px"] == 8.0
    assert js["n_frames"] == 2 and js["draws_per_frame"] == 25
    ent = js["phases"]["zn"]
    for stat in ("nhit", "nhit_distinct"):
        rec = ent[stat]
        assert rec["statistic"] == stat
        assert rec["n_draws"] == 50
        assert set(rec) >= {"mean", "median", "p99", "p999", "max", "n_draws", "statistic"}
    # distinct can never exceed predicted, draw by draw, so neither can its max
    assert ent["nhit_distinct"]["max"] <= ent["nhit"]["max"]
    assert ent["nhit_distinct"]["mean"] <= ent["nhit"]["mean"]
    assert "wrote" in out


def test_load_null_round_trip(null_json, fp, monkeypatch):
    work, p, _ = null_json
    js = json.loads(p.read_text())["phases"]["zn"]
    monkeypatch.delenv("LAUE_NULLMAX_ZN", raising=False)
    for stat in ("nhit", "nhit_distinct"):
        got = fp.load_null("zn", str(work), "syn", stat)
        assert got["statistic"] == stat and got["max"] == js[stat]["max"]
        assert got["p999"] == js[stat]["p999"] and got["source"] == str(p)


def test_env_override_must_be_the_same_statistic(tmp_path, fp, monkeypatch):
    """An nhit_distinct maximum exported for an nhit gate is refused."""
    (tmp_path / "peel_map").mkdir()
    rec = lambda s, mx: {"statistic": s, "n_draws": 10, "mean": 1.0, "median": 1.0,
                         "p99": 3.0, "p999": 4.0, "max": mx}
    (tmp_path / "peel_map" / "s_null.json").write_text(json.dumps(
        {"schema": 1, "phases": {"zn": {"nhit": rec("nhit", 10),
                                        "nhit_distinct": rec("nhit_distinct", 5)}}}))
    monkeypatch.setenv("LAUE_NULLMAX_ZN", "5")
    with pytest.raises(SystemExit, match="same statistic"):
        fp.load_null("zn", str(tmp_path), "s", "nhit")
    assert fp.load_null("zn", str(tmp_path), "s", "nhit_distinct")["max"] == 5
    monkeypatch.setenv("LAUE_NULLMAX_ZN", "12")          # a genuine override
    got = fp.load_null("zn", str(tmp_path), "s", "nhit")
    assert got["max"] == 12 and got["source"] == "$LAUE_NULLMAX_ZN" and got["p999"] == 4.0


def test_load_null_with_nothing_available_exits(tmp_path, fp, monkeypatch):
    monkeypatch.delenv("LAUE_NULLMAX_ZN", raising=False)
    with pytest.raises(SystemExit, match="null_model.py"):
        fp.load_null("zn", str(tmp_path), "s", "nhit")


# ---------------------------------------------------------------------------
# gates refuse to run without a measured null
# ---------------------------------------------------------------------------
def _validated_npz(path: Path, n=20, seed=0):
    rng = np.random.default_rng(seed)
    np.savez(path, oms=np.tile(np.eye(3), (n, 1, 1)), X=rng.uniform(0, 5, n),
             Z=rng.uniform(0, 5, n), labels=np.arange(n) % 3,
             nhit=rng.integers(5, 30, n), nhit_distinct=rng.integers(3, 20, n))


@pytest.mark.parametrize("script", ["empirical_gate.py", "validated_figures.py"])
def test_gate_exits_when_no_null_available(script, tmp_path):
    pytest.importorskip("matplotlib")
    ana = _analysis_dir()
    work = tmp_path / "work"
    (work / "peel_map").mkdir(parents=True)
    (work / "figures").mkdir()
    for ph in ("alpha", "beta"):
        _validated_npz(work / "peel_map" / f"t_{ph}_validated.npz")
    env = _clean_env(LAUE_WORK=work, LAUE_OUT_PREFIX="t", LAUE_MOUNT_DEG=45)
    r = subprocess.run([sys.executable, script], cwd=ana, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode != 0
    assert "no measured nhit null" in (r.stderr + r.stdout)
    assert not (work / "figures" / "t_report_validated.png").exists()


@pytest.mark.parametrize("script", ["empirical_gate.py", "validated_figures.py"])
def test_gate_runs_with_the_measured_null(script, tmp_path):
    """Positive control for the test above: same inputs plus a null json -> runs,
    and prints the statistic in force."""
    pytest.importorskip("matplotlib")
    ana = _analysis_dir()
    work = tmp_path / "work"
    (work / "peel_map").mkdir(parents=True)
    (work / "figures").mkdir()
    rec = lambda s, mx: {"statistic": s, "n_draws": 10, "mean": 2.0, "median": 2.0,
                         "p99": 6.0, "p999": 8.0, "max": mx}
    phases = {}
    for ph in ("alpha", "beta"):
        _validated_npz(work / "peel_map" / f"t_{ph}_validated.npz")
        phases[ph] = {"nhit": rec("nhit", 12), "nhit_distinct": rec("nhit_distinct", 7)}
    (work / "peel_map" / "t_null.json").write_text(json.dumps({"schema": 1, "phases": phases}))
    env = _clean_env(LAUE_WORK=work, LAUE_OUT_PREFIX="t", LAUE_GATE_STAT="nhit_distinct",
                     LAUE_MOUNT_DEG=45, MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, script], cwd=ana, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "gate statistic: nhit_distinct" in r.stdout


# ---------------------------------------------------------------------------
# the alpha-exclusion census and its null
# ---------------------------------------------------------------------------
@pytest.fixture
def two_phase(tmp_path):
    """Synthetic two-phase scan: 4 frames, alpha/beta params, validated npz for both."""
    h5py = pytest.importorskip("h5py")
    data, run, work = tmp_path / "data", tmp_path / "run", tmp_path / "work"
    _write_frames(data, run, n=4)
    (work / "peel_map").mkdir(parents=True)
    pa = _write_params(tmp_path, "a", 194, (0.2921, 0.2921, 0.4665, 90, 90, 120))
    pb = _write_params(tmp_path, "b", 229, (0.33065,) * 3 + (90, 90, 90))
    fr = sorted(f.name for f in data.glob("*.h5"))
    rng = np.random.default_rng(0)
    for ph in ("alpha", "beta"):
        n = 12
        q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
        w, x, y, z = q.T
        oms = np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
                        np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
                        np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], -2)
        np.savez(work / "peel_map" / f"scan_{ph}_validated.npz", oms=oms,
                 frames=np.array([fr[i % 4] for i in range(n)]),
                 X=np.arange(n) % 2 * 1.0, Z=(np.arange(n) // 2) % 2 * 1.0,
                 labels=np.arange(n) % 5, nhit=rng.integers(3, 20, n),
                 nhit_distinct=rng.integers(2, 10, n))
    # LAUE_OUT_PREFIX deliberately UNSET: census and null must agree on the default
    return _clean_env(LAUE_WORK=work, LAUE_SCAN_DATA=data, LAUE_PARAMS_ALPHA=pa,
                      LAUE_PARAMS_BETA=pb)


def test_exclusion_null_reproducible_across_processes(two_phase):
    """The per-frame seed used to be hash(fn), which Python salts per process."""
    outs = []
    for seed in ("1", "2"):
        env = dict(two_phase, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "exclusion_null.py", "4", "20", "1"],
                           cwd=_analysis_dir(), env=env, capture_output=True, text=True,
                           timeout=600)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        outs.append([l for l in r.stdout.splitlines() if "mean" in l and "max" in l])
    assert outs[0] and outs[0] == outs[1]


def test_census_and_its_null_read_the_same_default_prefix(two_phase):
    """With LAUE_OUT_PREFIX unset the census ('env' key) used to read env_*.npz while
    its null read scan_*.npz. Both now default through frame_peaks.out_prefix()."""
    ana = _analysis_dir()
    r = subprocess.run([sys.executable, "beta_alpha_exclusion_census.py", "env", "1"], cwd=ana,
                       env=dict(two_phase, LAUE_GATE_STAT="nhit_distinct"),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "saved scan_census.npz" in r.stdout
    # analytic p-values stay on nhit, and the script says so rather than ignoring the variable
    assert "LAUE_GATE_STAT=nhit_distinct is NOT applied here" in r.stdout
    r = subprocess.run([sys.executable, "exclusion_null.py", "4", "10", "1"], cwd=ana,
                       env=two_phase, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def test_out_prefix_default_and_refusal(fp, monkeypatch):
    monkeypatch.delenv("LAUE_OUT_PREFIX", raising=False)
    assert fp.out_prefix() == "scan"
    monkeypatch.setenv("LAUE_OUT_PREFIX", "a/b")
    with pytest.raises(SystemExit):
        fp.out_prefix()


def test_detect_peaks_ignores_deprecated_npx(fp):
    rng = np.random.default_rng(3)
    f = rng.normal(100, 3, size=(256, 256))
    yy, xx = np.mgrid[0:256, 0:256]
    for cy, cx in ((50, 60), (120, 200), (200, 90)):
        f += 4000 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0)
    a = fp.detect_peaks(f)[:2]
    b = fp.detect_peaks(f, 99999)[:2]
    assert len(a[0]) == 3 and all(np.array_equal(u, v) for u, v in zip(a, b))
