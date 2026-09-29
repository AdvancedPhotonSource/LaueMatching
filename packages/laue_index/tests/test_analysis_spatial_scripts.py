"""The five scripts that now take a spatial (toroidal-shift) null run end to end.

Tiny synthetic inputs (a 16 x 16 raster), few null repetitions: these check the
plumbing -- the new null is computed and reported, the clustered npz comes from
the prefix, footprints are positions -- not the statistics, which
test_analysis_spatial_nulls.py calibrates.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

try:
    from conftest import _REPO_ROOT
except ImportError:  # pragma: no cover
    _REPO_ROOT = None

NR = NROWS = 16


def _analysis_dir() -> Path:
    if _REPO_ROOT is None:
        pytest.skip("not running from a repo checkout (pipeline/analysis absent)")
    d = Path(_REPO_ROOT) / "pipeline" / "analysis"
    if not d.is_dir():
        pytest.skip(f"{d} not present")
    return d


def _env(**kw):
    env = {k: v for k, v in os.environ.items() if not k.startswith("LAUE_")}
    env.update(KMP_DUPLICATE_LIB_OK="TRUE", MPLBACKEND="Agg", LAUE_NR=NR, LAUE_NROWS=NROWS,
               LAUE_NULL_REPS=20, LAUE_OUT_PREFIX="x")
    env.update({k: str(v) for k, v in kw.items()})
    return {k: str(v) for k, v in env.items()}


def _run(script, env, *args):
    r = subprocess.run([sys.executable, script, *map(str, args)], cwd=_analysis_dir(), env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    return r.stdout


def _rot(rng, n):
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
                     np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
                     np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], -2)


def _clustered(work, rng, name="x_zn_clustered.npz", ngrain=12):
    """Blob-shaped grains on the raster, 1-2 instances per position."""
    lab_map = np.full((NROWS, NR), -1)
    for k in range(ngrain):
        cy, cx = rng.integers(NROWS), rng.integers(NR)
        r = rng.integers(1, 5)
        yy, xx = np.ogrid[:NROWS, :NR]
        lab_map[((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r)] = k
    rr, cc = np.nonzero(lab_map >= 0)
    labs = lab_map[rr, cc]
    rep = rng.integers(1, 3, len(rr))                    # 1 or 2 instances per position
    rr, cc, labs = np.repeat(rr, rep), np.repeat(cc, rep), np.repeat(labs, rep)
    base = _rot(rng, ngrain)
    oms = base[labs]
    frames = np.array([f"s_{r * NR + c + 1:05d}.h5" for r, c in zip(rr, cc)])
    n = len(labs)
    np.savez(work / "peel_map" / name, oms=oms, labels=labs, frames=frames,
             X=cc.astype(float), Z=rr.astype(float), nhit=rng.integers(5, 30, n),
             nhit_distinct=rng.integers(3, 20, n))
    return labs, rr, cc


@pytest.fixture
def work(tmp_path):
    w = tmp_path / "work"
    (w / "peel_map").mkdir(parents=True)
    (w / "analysis_out").mkdir()
    (w / "figures").mkdir()
    return w


def test_substrate_deposit_spatial_null_and_positions(work, tmp_path):
    rng = np.random.default_rng(0)
    labs, rr, cc = _clustered(work, rng, ngrain=40)
    ped = ndi.gaussian_filter(rng.normal(size=(NROWS, NR)), 2)
    np.savez(tmp_path / "ped.npz", flat=ped)
    out = _run("substrate_deposit.py", _env(), work / "peel_map" / "x_zn_clustered.npz",
               tmp_path / "ped.npz", tmp_path / "sd")
    assert "toroidal-shift p" in out and "permutation p" not in out
    z = np.load(tmp_path / "sd" / "substrate_deposit.npz")
    ids = z["cluster_ids"]
    for k, npos in zip(ids, z["cluster_positions"]):
        assert npos == len(set(zip(rr[labs == k], cc[labs == k])))


def _optical(work, rng):
    from PIL import Image
    im = np.full((200, 200, 3), 200, np.uint8)
    blk = ndi.gaussian_filter(rng.normal(size=(200, 200)), 12) > 0
    im[blk] = (30, 30, 30)
    Image.fromarray(im).save(work / "optical.png")
    np.savez(work / "analysis_out" / "full_pedestal.npz",
             flat=ndi.gaussian_filter(rng.normal(size=(NROWS, NR)), 2))


_OPT = dict(LAUE_STEP_UM=5, LAUE_OPTICAL_CX=100, LAUE_OPTICAL_CY=100,
            LAUE_OPTICAL_PX_PER_UM=1.5, LAUE_OPTICAL_FLIP_Y=1)


def test_optical_overlay_reports_spatial_p(work):
    pytest.importorskip("PIL")
    rng = np.random.default_rng(1)
    _clustered(work, rng)
    _optical(work, rng)
    out = _run("optical_overlay.py", _env(LAUE_WORK=work, **_OPT))
    assert out.count("toroidal-shift p") == 2
    z = np.load(work / "analysis_out" / "optical_registration.npz")
    assert 0 < float(z["p_pb_toroidal"]) <= 1


def test_reg_refine_best_of_grid_has_a_null(work):
    pytest.importorskip("PIL")
    rng = np.random.default_rng(2)
    _clustered(work, rng)
    _optical(work, rng)
    out = _run("reg_refine.py", _env(LAUE_WORK=work, **_OPT))
    assert "p(best of grid)" in out
    z = np.load(work / "analysis_out" / "reg_refine.npz")
    assert 0 < float(z["p_best_toroidal"]) <= 1


def _params(d: Path, name, sg, latt):
    hk = [(h, k, l) for h in range(-3, 4) for k in range(-3, 4) for l in range(-3, 4)
          if (h, k, l) != (0, 0, 0)]
    np.savetxt(d / f"h_{name}.txt", np.array(hk), fmt="%d")
    p = d / f"params_{name}.txt"
    p.write_text(f"SpaceGroup {sg}\nSymmetry P\nLatticeParameter {' '.join(map(str, latt))}\n"
                 "P_Array 0.028828 0.002715 0.512993\nR_Array -1.2016 -1.2140 -1.2185\n"
                 "PxX 0.0032\nPxY 0.0032\nNrPxX 128\nNrPxY 128\nElo 5\nEhi 30\n"
                 f"HKLFile {d / f'h_{name}.txt'}\n")
    return p


def test_big_grain_split_test_spatial_verdict(work, tmp_path):
    rng = np.random.default_rng(3)
    pa = _params(tmp_path, "a", 194, (0.2951, 0.2951, 0.4686, 90, 90, 120))
    # one big cluster in two lobes, with a small smooth misorientation gradient
    base = _rot(rng, 1)[0]
    pos = [(r, c) for r in range(2, 7) for c in range(2, 7)] + \
          [(r, c) for r in range(9, 14) for c in range(9, 14)]
    oms = []
    for r, c in pos:
        ang = np.radians(0.05 * (r + c))
        R = np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]])
        oms.append(R @ base)
    lab = [0] * len(pos)
    other = _rot(rng, 1)[0]
    for r in range(NROWS):                       # a second, smaller grain fills the raster
        for c in range(NR):
            if (r, c) not in pos:
                pos.append((r, c)); oms.append(other); lab.append(1)
    lab = np.array(lab)
    fill = lab == 1                               # split the filler so cluster 0 is largest
    lab[fill] = 1 + np.arange(fill.sum()) % 6
    n = len(pos)
    np.savez(work / "peel_map" / "x_alpha_validated.npz", oms=np.array(oms),
             X=np.array([c for r, c in pos], float), Z=np.array([r for r, c in pos], float),
             labels=lab, nhit=np.full(n, 12), nhit_distinct=np.full(n, 9))
    out = _run("big_grain_split_test.py",
               _env(LAUE_WORK=work, LAUE_PARAMS_ALPHA=pa, LAUE_MOUNT_DEG=45))
    assert "SPATIAL NULL" in out and "VERDICT" in out


def test_separate_layers_map_level_null(work, tmp_path):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(4)
    labs, rr, cc = _clustered(work, rng)
    z = np.load(work / "peel_map" / "x_zn_clustered.npz", allow_pickle=True)
    pz = _params(tmp_path, "zn", 194, (0.2665, 0.2665, 0.4947, 90, 90, 120))
    shard = tmp_path / "shard_1"
    (shard / "results").mkdir(parents=True)
    (shard / "provenance.json").write_text("{}")
    mapping = {}
    by_frame = {}
    for i, f in enumerate(z["frames"]):
        by_frame.setdefault(str(f), []).append(i)
    for inum, (f, ix) in enumerate(sorted(by_frame.items()), start=100000):
        mapping[str(inum)] = {"file": f}
        ori = np.zeros((len(ix), 34)); sp = []
        for g, i in enumerate(ix):
            ori[g, 0] = g + 1
            ori[g, 22:31] = z["oms"][i].ravel()
            for s in range(4):
                row = np.zeros(11); row[0] = g + 1
                row[2:5] = rng.integers(-3, 4, 3); row[2] = row[2] or 1
                sp.append(row)
        with h5py.File(shard / "results" / f"image_{inum}.output.h5", "w") as h:
            h.create_dataset("entry/results/filtered_orientations", data=ori)
            h.create_dataset("entry/results/filtered_spots", data=np.array(sp))
    (shard / "frame_mapping.json").write_text(json.dumps(mapping))
    out = _run("separate_layers.py", _env(LAUE_WORK=work, LAUE_PARAMS_ZN=pz, LAUE_PHASES="zn",
                                          LAUE_SHARD_GLOB=str(tmp_path / "shard_*"), NW=1))
    assert "toroidal-shift p" in out
    zz = np.load(work / "analysis_out" / "layer_separation.npz")
    assert "toroidal_p" in zz.files
    # footprints are positions
    foot = zz["footprint"]
    for k in np.unique(labs):
        m = labs == k
        assert set(foot[m]) == {len(set(zip(rr[m], cc[m])))}
