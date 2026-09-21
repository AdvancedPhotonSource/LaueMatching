"""The map/figure scripts: kept-grain loop, static soundness, no baked-in raster or paths.

* ``within_grain.per_grain_split`` must visit exactly the KEPT grains. The loop it
  replaced ran ``for g in range(keep_g.sum())`` over ``np.unique``-inverse indices,
  i.e. grains 0..n_kept-1, which are not the kept ones.
* Every script in this set compiles and loads no name before binding it (the
  ``ast`` checker of ``test_analysis_imports``, which is itself tested to fail).
* None carries a hard-coded 201-column raster, a literal ``"$LAUE_WORK"`` path, or
  its own copy of the frame-number parse -- they go through ``raster.py``.
"""
from __future__ import annotations

import py_compile
import re
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    from conftest import _REPO_ROOT
except ImportError:  # pragma: no cover
    _REPO_ROOT = None

MAP_FILES = [
    "ipf_map.py", "optical_overlay.py", "reg_refine.py", "render_registered.py",
    "zn_report_figures.py", "separate_layers.py", "substrate_deposit.py",
    "hardening_fullmap.py", "drift_control.py", "fullped.py", "within_grain.py",
    "raster.py",
]
WINNER_FILES = ["ipf_map.py", "optical_overlay.py", "reg_refine.py",
                "render_registered.py", "zn_report_figures.py"]


def _analysis_dir() -> Path:
    if _REPO_ROOT is None:
        pytest.skip("not running from a repo checkout (pipeline/analysis absent)")
    d = Path(_REPO_ROOT) / "pipeline" / "analysis"
    if not d.is_dir():
        pytest.skip(f"{d} not present")
    return d


# ---------------------------------------------------------------------------
# within_grain: the per-grain split covers exactly the kept labels
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def within_grain():
    d = _analysis_dir()
    sys.path.insert(0, str(d))
    try:
        import within_grain as mod          # module body only defines functions
    finally:
        sys.path.remove(str(d))
    return mod


def test_within_grain_split_visits_exactly_the_kept_grains(within_grain):
    """Kept grains are unique-indices 1, 3, 4 (not 0, 1, 2): all three, and only they."""
    rng = np.random.default_rng(0)
    raw_labels = np.array([10, 20, 30, 40, 50])      # np.unique -> indices 0..4
    sizes = np.array([5, 30, 6, 25, 40])              # >= 20 kept: indices 1, 3, 4
    lab = np.repeat(raw_labels, sizes)
    labs, inv = np.unique(lab, return_inverse=True)
    counts = np.bincount(inv)
    keep_g = counts >= 20
    gmask = keep_g[inv]
    li = inv[gmask]
    pv = rng.normal(100, 10, gmask.sum())
    ev = rng.normal(20, 1, gmask.sum())
    deltas, grains = within_grain.per_grain_split(li, pv, ev, keep_g, 20)
    assert grains.tolist() == [1, 3, 4]
    assert len(deltas) == 3
    # the replaced loop would have visited 0, 1, 2: index 0 and 2 have no
    # instances in li at all, so it reported ONE grain out of three
    old = [g for g in range(keep_g.sum()) if (li == g).sum() >= 20]
    assert old == [1]


def test_within_grain_split_matches_a_direct_computation(within_grain):
    li = np.array([2] * 12 + [5] * 12)
    pv = np.r_[np.arange(12.0), np.arange(12.0)]
    ev = np.r_[np.r_[np.zeros(6), np.ones(6)], np.r_[np.ones(6), np.zeros(6)]]
    keep_g = np.zeros(6, bool); keep_g[[2, 5]] = True
    deltas, grains = within_grain.per_grain_split(li, pv, ev, keep_g, 10)
    assert grains.tolist() == [2, 5]
    assert deltas.tolist() == [1.0, -1.0]


# ---------------------------------------------------------------------------
# static soundness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", MAP_FILES)
def test_map_script_compiles(name, tmp_path):
    p = _analysis_dir() / name
    py_compile.compile(str(p), cfile=str(tmp_path / (name + "c")), doraise=True)


@pytest.mark.parametrize("name", MAP_FILES)
def test_map_script_has_no_undefined_names(name):
    try:
        from test_analysis_imports import undefined_names
    except ImportError:
        pytest.skip("test_analysis_imports (the ast checker) not present")
    bad = undefined_names((_analysis_dir() / name).read_text())
    assert not bad, f"{name}: names used before/without definition: {bad}"


@pytest.mark.parametrize("name", [f for f in MAP_FILES if f != "raster.py"])
def test_map_script_has_no_baked_in_raster_or_work_path(name):
    src = (_analysis_dir() / name).read_text()
    assert not re.search(r"^\s*NR\s*=\s*\d", src, re.M), "hard-coded raster width"
    assert '"LAUE_NR", "' not in src, "LAUE_NR must not have a default"
    assert '"$LAUE_WORK' not in src and '"$LAUE_DATA' not in src, "literal env-var path"
    assert '.split("_")[-1].split(".")[0]' not in src, "private frame-number parse"


@pytest.mark.parametrize("name", WINNER_FILES)
def test_winner_sites_use_the_shared_ranking(name):
    src = (_analysis_dir() / name).read_text()
    assert "winner_per_position(" in src and "ranking_counts(" in src
    assert "nh[i] >" not in src, "private nhit winner loop"


# ---------------------------------------------------------------------------
# micrometres come from LAUE_STEP_UM; registration and titles come from the input
# ---------------------------------------------------------------------------
UM_FILES = ["ipf_map.py", "optical_overlay.py", "reg_refine.py", "render_registered.py",
            "zn_report_figures.py", "drift_control.py"]


@pytest.mark.parametrize("name", UM_FILES)
def test_um_conversions_use_the_step(name):
    src = (_analysis_dir() / name).read_text()
    assert "step_um(" in src, "converts positions to um without LAUE_STEP_UM"
    assert "extent=[-100" not in src and "EXT = [-100" not in src, "fixed 200 um extent"
    assert "micrometres at 1 um steps" not in src


@pytest.mark.parametrize("name", ["optical_overlay.py", "reg_refine.py"])
def test_optical_registration_constants_are_not_built_in(name):
    src = (_analysis_dir() / name).read_text()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    assert "optical_registration()" in src
    for const in ("398", "284", "0.6,", "60.0 / 100.0"):
        assert const not in code, f"campaign registration constant {const!r} in code"


@pytest.fixture()
def zn_report(monkeypatch):
    d = _analysis_dir()
    sys.path.insert(0, str(d))
    try:
        import zn_report_figures as mod
    finally:
        sys.path.remove(str(d))
    return mod


def _background_title(mod, monkeypatch, tmp_path, **extra):
    rng = np.random.default_rng(0)
    m = rng.normal(100, 5, (5, 7))
    np.savez(tmp_path / "full_pedestal.npz", flat=m, halo=m * 0.1, i0=m * 0 + 1e4, **extra)
    got = {}

    def grab(fig, path):
        got["title"] = fig._suptitle.get_text()
        got["extent"] = fig.axes[0].get_images()[0].get_extent()

    monkeypatch.setattr(mod, "_save", grab)
    mod.plate_background(str(tmp_path), str(tmp_path))
    return got


def test_background_title_comes_from_the_input(zn_report, monkeypatch, tmp_path):
    monkeypatch.setenv("LAUE_STEP_UM", "10")
    monkeypatch.delenv("LAUE_SCAN_LABEL", raising=False)
    got = _background_title(zn_report, monkeypatch, tmp_path, scan_label="scanQ_survey")
    assert got["title"].startswith("scanQ_survey") and "sampleG" not in got["title"]
    assert "7×5 positions" in got["title"]
    assert list(got["extent"]) == [-30, 30, -20, 20]          # 10 um steps


def test_background_title_falls_back_to_env_then_path(zn_report, monkeypatch, tmp_path):
    monkeypatch.setenv("LAUE_STEP_UM", "1")
    monkeypatch.setenv("LAUE_SCAN_LABEL", "scanR")
    assert _background_title(zn_report, monkeypatch, tmp_path)["title"].startswith("scanR ")
    monkeypatch.delenv("LAUE_SCAN_LABEL")
    t = _background_title(zn_report, monkeypatch, tmp_path)["title"]
    assert t.startswith(str(tmp_path))
