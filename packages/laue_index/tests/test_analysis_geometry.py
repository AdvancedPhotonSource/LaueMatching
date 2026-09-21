"""Raster geometry in the figure scripts: the mount angle is REQUIRED, and steps are
measured from the stage coordinates.

Before 2026-09 big_grain_diagnostic.py, big_grain_split_test.py, validated_figures.py,
catalog_figures.py, variant_coherence.py and collect_scan_metrics.py hard-coded a
45-degree mount (``sqrt(2)``) and a 0.25 um step (half-steps 0.125 / 0.09 in plot
extents). Both were one scan's values. The mount now comes from
``raster.mount_deg()`` (``LAUE_MOUNT_DEG``, no default); steps are the median spacing
of the stage coordinates each script already holds.
"""
from __future__ import annotations

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

FIGURE_SCRIPTS = ["big_grain_diagnostic.py", "big_grain_split_test.py", "validated_figures.py",
                  "catalog_figures.py", "variant_coherence.py", "collect_scan_metrics.py"]


def _analysis_dir() -> Path:
    if _REPO_ROOT is None:
        pytest.skip("not running from a repo checkout (pipeline/analysis absent)")
    d = Path(_REPO_ROOT) / "pipeline" / "analysis"
    if not d.is_dir():
        pytest.skip(f"{d} not present")
    return d


def _env(**kw):
    env = {k: v for k, v in os.environ.items() if not k.startswith("LAUE_")}
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["MPLBACKEND"] = "Agg"
    env.update({k: str(v) for k, v in kw.items()})
    return env


@pytest.fixture
def raster(monkeypatch):
    d = str(_analysis_dir())
    if d not in sys.path:
        sys.path.insert(0, d)
    import raster
    monkeypatch.delenv("LAUE_MOUNT_DEG", raising=False)
    return raster


def test_mount_deg_required_and_ranged(raster, monkeypatch):
    with pytest.raises(SystemExit, match="LAUE_MOUNT_DEG is not set"):
        raster.mount_deg()
    for bad in ("90", "-5", "abc"):
        monkeypatch.setenv("LAUE_MOUNT_DEG", bad)
        with pytest.raises(SystemExit):
            raster.mount_deg()
    monkeypatch.setenv("LAUE_MOUNT_DEG", "45")
    assert raster.mount_deg() == 45.0
    # the de-projection the scripts apply reproduces the old hard-coded sqrt(2)
    assert abs(1 / np.cos(np.radians(raster.mount_deg())) - np.sqrt(2)) < 1e-12
    monkeypatch.setenv("LAUE_MOUNT_DEG", "0")
    assert raster.mount_deg() == 0.0


@pytest.mark.parametrize("name", FIGURE_SCRIPTS)
def test_no_hard_coded_mount_or_step(name):
    code = "\n".join(l.split("#")[0] for l in (_analysis_dir() / name).read_text().splitlines())
    assert "sqrt(2" not in code.replace(" ", ""), "hard-coded 45 deg mount"
    assert not re.search(r"\b0?\.125\b|\b0?\.09\b|\*\s*0\.25\b", code), "hard-coded step"
    assert "mount_deg()" in code


def _alpha_npz(path, n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = np.tile(np.arange(8) * 0.25, 5)[:n]
    Z = np.repeat(np.arange(5) * 0.177, 8)[:n]
    lab = np.zeros(n, int); lab[n // 2:] = 1
    np.savez(path, oms=np.tile(np.eye(3), (n, 1, 1)), X=X, Z=Z, labels=lab,
             nhit=rng.integers(5, 20, n), nhit_distinct=rng.integers(3, 10, n),
             frames=np.array([f"f_{i:05d}.h5" for i in range(n)]))


def _hex_params(d: Path) -> Path:
    hkl = d / "hkls.txt"
    np.savetxt(hkl, np.array([[1, 0, 0], [0, 0, 1]]), fmt="%d")
    p = d / "params_a.txt"
    p.write_text("SpaceGroup 194\nLatticeParameter 0.2921 0.2921 0.4665 90 90 120\n"
                 "P_Array 0.0288 0.0027 0.513\nR_Array -1.2 -1.21 -1.22\nPxX 0.0002\nPxY 0.0002\n"
                 f"NrPxX 2048\nNrPxY 2048\nElo 5\nEhi 30\nHKLFile {hkl}\n")
    return p


def test_big_grain_diagnostic_needs_mount_and_creates_figures_dir(tmp_path):
    ana = _analysis_dir()
    work = tmp_path / "work"
    (work / "peel_map").mkdir(parents=True)          # NO figures/ -- the script makes it
    _alpha_npz(work / "peel_map" / "scan_alpha_validated.npz")
    env = _env(LAUE_WORK=work, LAUE_PARAMS_ALPHA=_hex_params(tmp_path))
    r = subprocess.run([sys.executable, "big_grain_diagnostic.py", "0", "20"], cwd=ana, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and "LAUE_MOUNT_DEG is not set" in r.stderr
    r = subprocess.run([sys.executable, "big_grain_diagnostic.py", "0", "20"], cwd=ana,
                       env=dict(env, LAUE_MOUNT_DEG="45"), capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (work / "figures" / "scan_biggrain_rank0.png").is_file()
    # box size uses the MEASURED steps: 0.25 fast, 0.177 slow de-projected by sqrt(2)
    m = re.search(r"\(([0-9.]+) x ([0-9.]+) um in the sample-surface frame\)", r.stdout)
    assert m, r.stdout
    bw, bh = float(m.group(1)), float(m.group(2))
    assert abs(bw / 0.25 - round(bw / 0.25)) < 1e-6
    assert abs(bh / (0.177 * np.sqrt(2)) - round(bh / (0.177 * np.sqrt(2)))) < 1e-2
