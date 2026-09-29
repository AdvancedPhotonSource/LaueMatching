"""Analysis-chain robustness (group A1 of the 2026-09 code read).

* parentbeta_reconstruct.py with no significant alpha cluster writes a well-formed
  EMPTY reconstruction (it crashed on an empty vote cloud and a 0/0), and
  variant_coherence.py reads that as "no parent" and exits 0 (it raised IndexError).
* LAUE_SKIP_CLUSTER=1 leaves labels at -1. Every consumer refuses that up front,
  naming cluster_orientations.py, instead of crashing after a full frame pass
  (census), crashing (anchor_null), running silently on nothing (empirical_gate) or
  re-clustering O(n^2) (reconstruct).
* One clustering tolerance, LAUE_CLUSTER_TOL (default 1.0), recorded in each npz.
* The analytic lambda and the peel mask use both detector dimensions.
* anchor_null reads this scan's anchors from the reconstruction, not four
  numbers from one old scan.
* hardening_fullmap's row-block null pairs positions, not array offsets.
* Instance counts are not position counts.
* separate_layers / optical_overlay / reg_refine take the clustered npz from the
  prefix, not a hard-coded full_zn_clustered.npz.
"""
from __future__ import annotations

import json
import os
import re
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
    env["MPLBACKEND"] = "Agg"
    env.update({k: str(v) for k, v in kw.items()})
    return env


def _run(script, env, *args, timeout=300):
    return subprocess.run([sys.executable, script, *map(str, args)], cwd=_analysis_dir(),
                          env=env, capture_output=True, text=True, timeout=timeout)


def _write_params(d: Path, name, sg, latt, npx=128, px=0.0032):
    hk = [(h, k, l) for h in range(-3, 4) for k in range(-3, 4) for l in range(-3, 4)
          if (h, k, l) != (0, 0, 0)]
    hkl = d / f"hkls_{name}.txt"
    np.savetxt(hkl, np.array(hk), fmt="%d")
    p = d / f"params_{name}.txt"
    p.write_text(
        f"SpaceGroup {sg}\nSymmetry P\nLatticeParameter {' '.join(map(str, latt))}\n"
        "P_Array 0.028828 0.002715 0.512993\nR_Array -1.20161887 -1.21404493 -1.21852276\n"
        f"PxX {px}\nPxY {px}\nNrPxX {npx}\nNrPxY {npx}\nElo 5.0\nEhi 30.0\nHKLFile {hkl}\n")
    return p


def _rand_oms(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
                     np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
                     np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], -2)


@pytest.fixture
def two_phase(tmp_path):
    work = tmp_path / "work"
    (work / "peel_map").mkdir(parents=True)
    (work / "figures").mkdir()
    pa = _write_params(tmp_path, "a", 194, (0.2951, 0.2951, 0.4686, 90, 90, 120))
    pb = _write_params(tmp_path, "b", 229, (0.3306, 0.3306, 0.3306, 90, 90, 90))
    env = _clean_env(LAUE_WORK=work, LAUE_PARAMS_ALPHA=pa, LAUE_PARAMS_BETA=pb,
                     LAUE_PHASES="alpha,beta", LAUE_OUT_PREFIX="t", LAUE_MOUNT_DEG=45)
    return work, env


def _npz(path, oms, labels, rng):
    n = len(oms)
    np.savez(path, oms=oms, frames=np.array([f"f_{i + 1:05d}.h5" for i in range(n)]),
             X=np.arange(n) % 4 * 1.0, Z=np.arange(n) // 4 * 1.0, labels=labels,
             nhit=rng.integers(5, 30, n), nhit_distinct=rng.integers(3, 20, n))


# ---------------------------------------------------------------------------
# parentbeta_reconstruct -> variant_coherence with no significant parent
# ---------------------------------------------------------------------------
def test_reconstruct_with_no_significant_cluster_writes_empty_npz(two_phase):
    """3 alpha clusters of 2 instances, min cluster size 30: nothing to vote with.
    Used to crash (np.bincount on an empty cloud, then 0/0)."""
    work, env = two_phase
    rng = np.random.default_rng(0)
    base = _rand_oms(rng, 3)
    _npz(work / "peel_map" / "t_alpha_validated.npz", np.repeat(base, 2, axis=0),
         np.repeat(np.arange(3), 2), rng)
    _npz(work / "peel_map" / "t_beta_validated.npz", _rand_oms(rng, 4), np.arange(4), rng)
    r = _run("parentbeta_reconstruct.py", env, 30, "t")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    z = np.load(work / "peel_map" / "t_reconstruction.npz")
    assert len(z["parents_nv"]) == 0
    r = _run("variant_coherence.py", env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "no parent" in r.stdout.lower()


# ---------------------------------------------------------------------------
# LAUE_SKIP_CLUSTER=1 labels (-1) are refused up front
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("script, args", [
    ("empirical_gate.py", ()),
    ("anchor_null.py", ()),
    ("parentbeta_reconstruct.py", (2, "t")),
    ("beta_alpha_exclusion_census.py", ("env", 1)),
    ("regrain.py", ("alpha",)),
])
def test_unclustered_labels_are_refused(two_phase, tmp_path, script, args):
    work, env = two_phase
    rng = np.random.default_rng(1)
    for ph in ("alpha", "beta"):
        _npz(work / "peel_map" / f"t_{ph}_validated.npz", _rand_oms(rng, 6),
             np.full(6, -1), rng)
    rec = {"statistic": "nhit", "max": 9, "kind": "draw"}
    (work / "peel_map" / "t_null.json").write_text(json.dumps({"phases": {
        ph: {"nhit": rec, "nhit_distinct": dict(rec, statistic="nhit_distinct")}
        for ph in ("alpha", "beta")}}))
    (tmp_path / "data").mkdir(exist_ok=True)
    env = dict(env, LAUE_SCAN_DATA=str(tmp_path / "data"), LAUE_NULL_KIND="draw")
    r = _run(script, env, *args)
    assert r.returncode != 0, r.stdout[-1500:]
    assert "cluster_orientations.py" in (r.stdout + r.stderr), r.stdout[-1500:] + r.stderr[-1500:]


def test_phase4_doc_writes_the_clustered_labels_back(ana):
    """The SKIP_CLUSTER recipe must write back to the file the chain reads."""
    doc = (Path(ana).parents[1] / "manuals" / "laue" / "phase-4-analyse.md").read_text()
    blk = doc[doc.index("LAUE_SKIP_CLUSTER=1 python"):][:600]
    m = re.search(r"cluster_orientations\.py (\S+) (\S+)", blk)
    assert m and m.group(1) == m.group(2), m and m.groups()
    assert "_validated.npz" in m.group(2)


# ---------------------------------------------------------------------------
# one clustering tolerance
# ---------------------------------------------------------------------------
def test_cluster_tol_env(ana, monkeypatch):
    import frame_peaks as fp
    monkeypatch.delenv("LAUE_CLUSTER_TOL", raising=False)
    assert fp.cluster_tol() == 1.0
    monkeypatch.setenv("LAUE_CLUSTER_TOL", "0.7")
    assert fp.cluster_tol() == 0.7
    monkeypatch.setenv("LAUE_CLUSTER_TOL", "-1")
    with pytest.raises(SystemExit):
        fp.cluster_tol()


@pytest.mark.parametrize("name", ["parentbeta_validate.py", "parentbeta_backfill.py",
                                  "beta_map_validate.py", "map_validate_cluster.py",
                                  "scan_map.py", "cluster_orientations.py"])
def test_no_literal_clustering_tolerance(ana, name):
    """Greedy clustering cuts (`d < 1.0`, `< 0.7`, `< 1.5`) all go through
    frame_peaks.cluster_tol(); batch_peel's 0.7 DEDUP is a different thing."""
    src = (Path(ana) / name).read_text()
    lit = re.findall(r"(?:miso\w*|misorientation)\([^)]*\)\s*<\s*[0-9.]+\]", src)
    lit += re.findall(r"\bd\s*<\s*(?:0\.7|1\.0|1\.5)\]", src)
    assert not lit, lit
    assert "cluster_tol()" in src


# ---------------------------------------------------------------------------
# non-square detectors
# ---------------------------------------------------------------------------
def test_poisson_lambda_uses_both_dimensions(ana):
    import frame_peaks as fp
    a = fp.poisson_lambda(50, 100, 8.0, 100, 200)
    b = fp.poisson_lambda(50, 100, 8.0, 200, 100)
    assert a == pytest.approx(b) == pytest.approx(50 * 100 * np.pi * 64 / 20000)


@pytest.mark.parametrize("name", ["parentbeta_validate.py", "parentbeta_backfill.py",
                                  "grain_extent_backfill.py", "beta_alpha_exclusion_census.py",
                                  "beta_map_validate.py", "map_validate_cluster.py",
                                  "null_model.py"])
def test_no_square_detector_lambda(ana, name):
    src = (Path(ana) / name).read_text()
    assert "NPX*NPX" not in src.replace(" ", ""), name


@pytest.mark.parametrize("shape", [(100, 200), (200, 100)])
def test_mask_disks_non_square(ana, shape):
    import frame_peaks as fp
    img = np.ones(shape)
    H, W = shape
    pts = np.array([[W - 2, H - 2], [1, 1], [W // 2, H - 1]], float)
    fp.mask_disks(img, pts, 5, 0.0)
    for x, y in pts.astype(int):
        assert img[y, x] == 0.0
    assert img[H // 2, W // 2] == 1.0


def test_batch_peel_uses_the_shared_mask(ana):
    src = (Path(ana) / "batch_peel_driver.py").read_text()
    assert "mask_disks(" in src and "min(nPx, yi" not in src


# ---------------------------------------------------------------------------
# anchor_null reads this scan's anchors
# ---------------------------------------------------------------------------
def test_anchor_null_uses_this_scans_anchors(two_phase):
    work, env = two_phase
    rng = np.random.default_rng(2)
    _npz(work / "peel_map" / "t_beta_validated.npz", _rand_oms(rng, 8), np.arange(8) // 2, rng)
    np.savez(work / "peel_map" / "t_reconstruction.npz", parents_B=_rand_oms(rng, 2),
             parents_nv=np.array([9, 7]), parents_anchor=np.array([2.37, 5.11]),
             parents_ninst=np.array([10, 5]), inst_var=np.zeros(0, int),
             inst_par=np.zeros(0, int), aX=np.zeros(0), aZ=np.zeros(0),
             bX=np.zeros(0), bZ=np.zeros(0))
    r = _run("anchor_null.py", env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "2.37 deg" in r.stdout and "5.11 deg" in r.stdout
    assert "1.41" not in r.stdout and "4.40" not in r.stdout


# ---------------------------------------------------------------------------
# hardening_fullmap row-block permutation
# ---------------------------------------------------------------------------
def _hardening(ana):
    import importlib
    return importlib.import_module("hardening_fullmap")


def test_row_block_pairing_keeps_columns(ana):
    """Each a at (r, c) pairs with b at (perm[r], c); rows of unequal length drop
    only the pairs whose partner position does not exist."""
    hf = _hardening(ana)
    rows = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    cols = np.array([0, 1, 2, 0, 2, 0, 1, 2, 3])
    ident = hf.row_block_pairs(rows, cols, {0: 0, 1: 1, 2: 2})
    assert np.array_equal(ident, np.arange(9))
    sw = hf.row_block_pairs(rows, cols, {0: 2, 1: 0, 2: 1})
    for i, j in enumerate(sw):
        if j >= 0:
            assert cols[j] == cols[i] and rows[j] == {0: 2, 1: 0, 2: 1}[rows[i]]
    # (1, 1) does not exist, so a at (0, 1) mapped to row 1 has no partner
    assert hf.row_block_pairs(rows, cols, {0: 1, 1: 0, 2: 2})[1] == -1


def test_row_block_null_is_calibrated_on_unequal_rows(ana):
    """Two independent fields with row-level structure, rows of unequal length:
    the row-block null must give p < 0.05 about 5% of the time. The old pairing
    (argsort of permuted row ids) shifted values across rows and was not."""
    hf = _hardening(ana)
    rng = np.random.default_rng(5)
    hits_new = hits_old = 0
    reps = 150
    for _ in range(reps):
        nrow = 20
        lens = rng.integers(3, 25, nrow)
        rows = np.repeat(np.arange(nrow), lens)
        cols = np.concatenate([rng.choice(30, k, replace=False) for k in lens])
        a = rng.normal(size=nrow)[rows] + 0.3 * rng.normal(size=len(rows))
        b = rng.normal(size=nrow)[rows] + 0.3 * rng.normal(size=len(rows))
        hits_new += hf.blocked_perm_p(a, b, rows, cols=cols, n=199, seed=int(rng.integers(1e9))) < 0.05
        hits_old += _old_blocked_perm_p(hf, a, b, rows, n=199, seed=int(rng.integers(1e9))) < 0.05
    assert 0.01 <= hits_new / reps <= 0.10, hits_new / reps
    assert hits_old / reps > 0.10, hits_old / reps      # fail-before evidence


def _old_blocked_perm_p(hf, a, b, rows, n=5000, seed=0):
    r0 = abs(hf.pearson(a, b))
    rng = np.random.default_rng(seed)
    uniq = np.unique(rows)
    cnt = 0
    for _ in range(n):
        perm = rng.permutation(uniq)
        mapping = dict(zip(uniq, perm))
        order = np.argsort([mapping[r] for r in rows], kind="stable")
        if abs(hf.pearson(a, b[order])) >= r0:
            cnt += 1
    return (cnt + 1) / (n + 1)


# ---------------------------------------------------------------------------
# instances are not positions
# ---------------------------------------------------------------------------
def test_positions_per_label_counts_distinct_positions(ana):
    import raster
    lab = np.array([0, 0, 1, 1, 1])
    row = np.array([3, 3, 1, 1, 2])
    col = np.array([4, 4, 0, 1, 0])
    assert raster.positions_per_label(lab, row, col).tolist() == [1, 3]


@pytest.mark.parametrize("name", ["empirical_gate.py", "validated_figures.py",
                                  "catalog_figures.py", "substrate_deposit.py"])
def test_position_labels_use_distinct_positions(ana, name):
    src = (Path(ana) / name).read_text()
    assert "positions_per_label(" in src, name


# ---------------------------------------------------------------------------
# no hard-coded clustered-npz name / 5-digit image names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["separate_layers.py", "optical_overlay.py", "reg_refine.py"])
def test_clustered_npz_from_prefix(ana, name):
    src = (Path(ana) / name).read_text()
    assert "full_zn_clustered.npz" not in src
    assert "clustered_npz(" in src
    assert ":05d}" not in src


def test_clustered_npz_helper(ana, tmp_path, monkeypatch):
    import frame_peaks as fp
    (tmp_path / "peel_map").mkdir()
    monkeypatch.setenv("LAUE_OUT_PREFIX", "x")
    monkeypatch.delenv("LAUE_CLUSTERED_NPZ", raising=False)
    with pytest.raises(SystemExit):
        fp.clustered_npz(str(tmp_path))
    (tmp_path / "peel_map" / "x_zn_clustered.npz").write_bytes(b"")
    assert fp.clustered_npz(str(tmp_path)).endswith("x_zn_clustered.npz")
    (tmp_path / "peel_map" / "x_cu_clustered.npz").write_bytes(b"")
    with pytest.raises(SystemExit, match="LAUE_CLUSTERED_NPZ"):
        fp.clustered_npz(str(tmp_path))
    monkeypatch.setenv("LAUE_CLUSTERED_NPZ", str(tmp_path / "peel_map" / "x_cu_clustered.npz"))
    assert fp.clustered_npz(str(tmp_path)).endswith("x_cu_clustered.npz")


def test_variant_coherence_runs_with_the_cluster_null(two_phase):
    """Contiguous alpha clusters with RANDOM variants: the new null reports no
    coherence (|z| small) while the position shuffle it replaces reports a lot."""
    work, env = two_phase
    rng = np.random.default_rng(8)
    nx = nz = 24
    lab_map = np.full((nz, nx), -1)
    for k in range(30):
        cy, cx, r = rng.integers(nz), rng.integers(nx), rng.integers(1, 5)
        yy, xx = np.ogrid[:nz, :nx]
        lab_map[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = k
    gi, gj = np.nonzero(lab_map >= 0)
    lab = lab_map[gi, gj]
    lab = np.unique(lab, return_inverse=True)[1]
    var = rng.integers(0, 12, lab.max() + 1)[lab]
    n = len(lab)
    _npz(work / "peel_map" / "t_alpha_validated.npz", np.tile(np.eye(3), (n, 1, 1)), lab, rng)
    np.savez(work / "peel_map" / "t_reconstruction.npz", parents_B=np.eye(3)[None],
             parents_nv=np.array([10]), parents_anchor=np.array([3.0]),
             parents_ninst=np.array([n]), inst_var=var, inst_par=np.zeros(n, int),
             aX=gj.astype(float), aZ=gi.astype(float), bX=np.zeros(0), bZ=np.zeros(0))
    r = _run("variant_coherence.py", env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    z_new = float(re.search(r"^z = (-?[0-9.]+)", r.stdout, re.M).group(1))
    z_old = float(re.search(r"position-shuffle null.*z = (-?[0-9.]+)", r.stdout).group(1))
    assert abs(z_new) < 3.5 and z_old > 5, (z_new, z_old)


# ---------------------------------------------------------------------------
# backfills: measured false count from randomly rotated controls
# ---------------------------------------------------------------------------
def test_backfill_control_measures_an_absent_grains_false_rate(ana, tmp_path):
    """The randomly rotated control is exchangeable with a grain that is NOT in the
    frames, so its presence count measures the false-backfill rate at the gate
    actually used. (The plan expected the measured count to EXCEED the analytic
    n_grains * n_frames * PGATE on clustered synthetics; on the synthetics tried
    here -- frames built from real crystals' patterns -- the analytic count was
    the LARGER one, because the Poisson tail is discrete. Either way the analytic
    number is not a measurement; the control is.)"""
    from scipy.spatial import cKDTree
    import frame_peaks as fp
    from laue_material import Phase
    hk = [(h, k, l) for h in range(-8, 9) for k in range(-8, 9) for l in range(-8, 9)
          if (h, k, l) != (0, 0, 0)]
    np.savetxt(tmp_path / "h.txt", np.array(hk), fmt="%d")
    p = tmp_path / "p.txt"
    p.write_text("SpaceGroup 225\nSymmetry P\nLatticeParameter 0.3524 0.3524 0.3524 90 90 90\n"
                 "P_Array 0.028828 0.002715 0.512993\nR_Array -1.2016 -1.2140 -1.2185\n"
                 "PxX 0.0008\nPxY 0.0008\nNrPxX 512\nNrPxY 512\nElo 5\nEhi 30\n"
                 f"HKLFile {tmp_path / 'h.txt'}\n")
    ph = Phase(str(p), "c")
    rng = np.random.default_rng(0)
    master = _rand_oms(rng, 40)                       # grains absent from every frame
    absent = [ph.project(R) for R in master]
    ctrl = [ph.project(R) for R in fp.control_orientations(master, seed=1)]
    pgate, n_ctrl, n_abs, n_tests = 0.2, 0, 0, 0
    for _ in range(40):
        pk = np.unique(np.round(np.concatenate([ph.project(om) for om in _rand_oms(rng, 10)])),
                       axis=0)
        tree = cKDTree(pk)
        for pa, pc in zip(absent, ctrl):
            n_tests += 1
            n_abs += fp.presence_p(tree, len(pk), pa, 8.0, ph.npx_x, ph.npx_y) < pgate
            n_ctrl += fp.presence_p(tree, len(pk), pc, 8.0, ph.npx_x, ph.npx_y) < pgate
    print(f"\ncontrol {n_ctrl}, absent grains {n_abs}, analytic {n_tests * pgate:.0f} "
          f"({n_tests} tests at p < {pgate})")
    assert n_abs > 20
    assert abs(n_ctrl - n_abs) <= 3 * np.sqrt(n_ctrl + n_abs) + 2, (n_ctrl, n_abs)


@pytest.mark.parametrize("name", ["parentbeta_backfill.py", "grain_extent_backfill.py"])
def test_backfills_report_a_measured_false_count(ana, name):
    src = (Path(ana) / name).read_text()
    assert "control_orientations(" in src and "MEASURED false" in src
