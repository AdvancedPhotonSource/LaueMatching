"""Raster position and one-orientation-per-position, shared by the map scripts.

``pipeline/analysis/raster.py`` replaced a dozen private copies of two pieces of
logic that were wrong in the same ways:

* position from ``(frame-1) // 201`` with 201 hard-coded (or defaulted), so an
  81x81 or 201x101 scan was mis-placed or indexed out of range;
* "the highest-``nhit`` instance at each position", where ``nhit`` stacks
  harmonics and ties went to whichever instance came first in the file.

These tests pin the replacement: ``raster_positions`` (row-major, fast axis =
columns, explicit shape required) and ``winner_per_position`` / ``ranking_counts``
(distinct peaks first, deterministic ties, loud fallback on old files).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

try:
    from conftest import _REPO_ROOT
except ImportError:  # pragma: no cover
    _REPO_ROOT = None


@pytest.fixture(scope="module")
def raster():
    if _REPO_ROOT is None:
        pytest.skip("not running from a repo checkout (pipeline/analysis absent)")
    d = Path(_REPO_ROOT) / "pipeline" / "analysis"
    if not (d / "raster.py").is_file():
        pytest.skip(f"{d}/raster.py not present")
    sys.path.insert(0, str(d))
    try:
        import raster as mod
    finally:
        sys.path.remove(str(d))
    return mod


def _names(n, prefix="scanA_"):
    return [f"{prefix}{i:06d}.h5" for i in range(1, n + 1)]


def _stage(nrows, ncols, dx=1.0, dz=0.70710678):
    """Stage readbacks for a clean unidirectional raster, frame order."""
    n = np.arange(nrows * ncols)
    return -1436.0 + (n % ncols) * dx, -2803.0 + (n // ncols) * dz


# ---------------------------------------------------------------------------
# raster_positions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nrows,ncols", [(81, 81), (101, 201)])
def test_raster_positions_row_major_fast_axis_is_columns(raster, monkeypatch, nrows, ncols):
    """Frame N -> row (N-1)//ncols, col (N-1)%ncols; every cell hit exactly once."""
    monkeypatch.setenv("LAUE_NR", str(ncols))
    monkeypatch.setenv("LAUE_NROWS", str(nrows))
    X, Z = _stage(nrows, ncols)
    row, col = raster.raster_positions(_names(nrows * ncols), X=X, Z=Z)
    assert row.max() == nrows - 1 and col.max() == ncols - 1
    # consecutive frames step along the columns (fast axis) ...
    assert np.array_equal(col[:ncols], np.arange(ncols)) and (row[:ncols] == 0).all()
    # ... and the row advances once per ncols frames (slow axis)
    assert row[ncols] == 1 and col[ncols] == 0
    grid = np.zeros((nrows, ncols), int)
    np.add.at(grid, (row, col), 1)
    assert (grid == 1).all()


def test_raster_positions_reproduces_the_201_column_parse(raster, monkeypatch):
    """With LAUE_NR=201 the result is the old ``(n-1)//201, (n-1)%201`` exactly."""
    monkeypatch.setenv("LAUE_NR", "201")
    monkeypatch.setenv("LAUE_NROWS", "101")
    rng = np.random.default_rng(3)
    n = rng.integers(1, 201 * 101 + 1, 5000)
    names = [f"sampleH_scan1_{k:06d}.h5" for k in n]
    old = [(int(str(f).split("_")[-1].split(".")[0]) - 1) for f in names]
    row, col = raster.raster_positions(names)
    assert np.array_equal(row, np.array(old) // 201)
    assert np.array_equal(col, np.array(old) % 201)


@pytest.mark.parametrize("missing", ["LAUE_NR", "LAUE_NROWS"])
def test_raster_positions_exits_without_explicit_shape(raster, monkeypatch, missing):
    monkeypatch.setenv("LAUE_NR", "81")
    monkeypatch.setenv("LAUE_NROWS", "81")
    monkeypatch.delenv(missing)
    with pytest.raises(SystemExit) as e:
        raster.raster_positions(_names(10))
    assert missing in str(e.value)


def test_raster_positions_exits_when_frames_exceed_the_shape(raster, monkeypatch):
    """A 201x101 scan read with a 201x81 shape: frames past the raster must stop the run."""
    monkeypatch.setenv("LAUE_NR", "201")
    monkeypatch.setenv("LAUE_NROWS", "81")
    with pytest.raises(SystemExit) as e:
        raster.raster_positions(_names(201 * 101))
    assert "raster shape is wrong" in str(e.value)


def test_raster_positions_wrong_column_count_caught_by_stage_coordinates(raster, monkeypatch):
    """An 81x81 scan read as 201 columns fits the frame range; the slow axis catches it."""
    monkeypatch.setenv("LAUE_NR", "201")
    monkeypatch.setenv("LAUE_NROWS", "81")
    X, Z = _stage(81, 81)
    with pytest.raises(SystemExit) as e:
        raster.raster_positions(_names(81 * 81), X=X, Z=Z)
    assert "LAUE_NR=201" in str(e.value)


def test_raster_positions_tolerates_the_fast_axis_readback_race(raster, monkeypatch):
    """~1% of frames with X one step ahead (fix_positions.py) change nothing: X is not used."""
    monkeypatch.setenv("LAUE_NR", "201")
    monkeypatch.setenv("LAUE_NROWS", "101")
    X, Z = _stage(101, 201)
    X = X.copy()
    X[np.random.default_rng(0).choice(len(X), 200, replace=False)] += 1.0
    row, col = raster.raster_positions(_names(201 * 101), X=X, Z=Z)
    n = np.arange(201 * 101)
    assert np.array_equal(col, n % 201) and np.array_equal(row, n // 201)


def test_raster_positions_stage_fallback_without_frame_names(raster, capsys):
    X, Z = _stage(5, 7)
    row, col = raster.raster_positions(None, X=X, Z=Z)
    n = np.arange(35)
    assert np.array_equal(col, n % 7) and np.array_equal(row, n // 7)
    assert "WARNING" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# winner_per_position / ranking_counts
# ---------------------------------------------------------------------------
def test_winner_ranks_by_nhit_distinct_over_nhit(raster):
    """Harmonic-rich A (nhit 12, 5 distinct) loses to B (nhit 9, 7 distinct)."""
    z = {"nhit": np.array([12, 9, 4]), "nhit_distinct": np.array([5, 7, 4])}
    row, col = np.array([0, 0, 1]), np.array([0, 0, 1])
    prim, sec, name = raster.ranking_counts(z)
    assert name == "nhit_distinct"
    assert raster.winner_per_position(row, col, prim, sec) == {(0, 0): 1, (1, 1): 2}
    # the old ranking would have picked A -- the case really does disagree
    assert raster.winner_per_position(row, col, z["nhit"]) == {(0, 0): 0, (1, 1): 2}


def test_winner_ties_are_deterministic_and_independent_of_input_order(raster):
    """Equal primary: secondary decides; equal both: the orientation key decides."""
    rng = np.random.default_rng(7)
    n = 60
    row = rng.integers(0, 3, n)
    col = rng.integers(0, 3, n)
    prim = rng.integers(4, 6, n)             # many exact ties
    sec = rng.integers(8, 10, n)
    oms = rng.normal(size=(n, 9))            # a unique content key per instance
    ref = raster.winner_per_position(row, col, prim, sec, tiebreak=oms)
    ref_ids = {k: tuple(oms[i]) for k, i in ref.items()}
    for _ in range(20):
        p = rng.permutation(n)
        got = raster.winner_per_position(row[p], col[p], prim[p], sec[p], tiebreak=oms[p])
        assert {k: tuple(oms[p][i]) for k, i in got.items()} == ref_ids
    # and the winner really is the maximum by (primary, secondary)
    for (r, c), i in ref.items():
        m = (row == r) & (col == c)
        best = max(zip(prim[m], sec[m]))
        assert (prim[i], sec[i]) == best


def test_winner_secondary_breaks_primary_ties(raster):
    row = col = np.zeros(3, int)
    got = raster.winner_per_position(row, col, np.array([6, 6, 5]), np.array([8, 11, 20]))
    assert got == {(0, 0): 1}


def test_ranking_falls_back_to_nhit_with_a_warning_on_an_old_npz(raster, tmp_path, capsys):
    f = tmp_path / "old_validated.npz"
    np.savez(f, nhit=np.array([12, 9]), labels=np.array([0, 1]))
    z = np.load(f)
    prim, sec, name = raster.ranking_counts(z)
    assert name == "nhit" and sec is None
    assert np.array_equal(prim, [12, 9])
    err = capsys.readouterr().err
    assert "WARNING" in err and "harmonics" in err
    assert raster.winner_per_position(np.zeros(2, int), np.zeros(2, int), prim, sec) == {(0, 0): 0}


def test_ranking_prefers_distinct_silently_when_present(raster, capsys):
    raster.ranking_counts({"nhit": np.array([1]), "nhit_distinct": np.array([1])})
    assert "WARNING" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# step and optical registration: required, no defaults
# ---------------------------------------------------------------------------
def test_step_um_is_required_and_positive(raster, monkeypatch):
    monkeypatch.delenv("LAUE_STEP_UM", raising=False)
    with pytest.raises(SystemExit) as e:
        raster.step_um()
    assert "LAUE_STEP_UM" in str(e.value)
    for bad in ("0", "-1", "abc", "nan"):
        monkeypatch.setenv("LAUE_STEP_UM", bad)
        with pytest.raises(SystemExit) as e:
            raster.step_um()
        assert "LAUE_STEP_UM" in str(e.value)
    monkeypatch.setenv("LAUE_STEP_UM", "10")
    assert raster.step_um() == 10.0


def test_centred_extent_scales_with_step(raster):
    """201x201 at 1 um is the old [-100, 100, -100, 100]; a 10 um 151x101 survey is 10x larger."""
    assert raster.centred_extent(201, 201, 1.0) == [-100, 100, -100, 100]
    assert raster.centred_extent(101, 151, 10.0) == [-750, 750, -500, 500]


@pytest.mark.parametrize("missing", ["LAUE_OPTICAL_CX", "LAUE_OPTICAL_CY",
                                     "LAUE_OPTICAL_PX_PER_UM", "LAUE_OPTICAL_FLIP_Y"])
def test_optical_registration_has_no_built_in_constants(raster, monkeypatch, missing):
    env = {"LAUE_OPTICAL_CX": "398", "LAUE_OPTICAL_CY": "284",
           "LAUE_OPTICAL_PX_PER_UM": "0.6", "LAUE_OPTICAL_FLIP_Y": "+1"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert raster.optical_registration() == (398.0, 284.0, 0.6, 1)
    monkeypatch.delenv(missing)
    with pytest.raises(SystemExit) as e:
        raster.optical_registration()
    assert missing in str(e.value)


def test_optical_flip_rejects_anything_but_plus_minus_one(raster, monkeypatch):
    monkeypatch.setenv("LAUE_OPTICAL_FLIP_Y", "yes")
    with pytest.raises(SystemExit):
        raster.optical_flip_y()
    monkeypatch.setenv("LAUE_OPTICAL_FLIP_Y", "-1")
    assert raster.optical_flip_y() == -1
