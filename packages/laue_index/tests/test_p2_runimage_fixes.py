"""P2 (code read 2026-09-28): RunImage and the shared preprocessing.

Each test failed before its fix; the module each one pins is named in it.
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from conftest import _REPO_ROOT  # noqa: E402

NPX = 64

CONFIG = """SpaceGroup 225
Symmetry F
LatticeParameter 0.35238 0.35238 0.35238 90 90 90
P_Array 0.0 0.0 0.05
R_Array 0.0 0.0 0.0
Elo 5
Ehi 30
NrPxX {n}
NrPxY {n}
PxX 0.0002
PxY 0.0002
MaxNrLaueSpots 30
MinIntensity 0
NMeadianPasses 0
MinGoodSpots 1
MinNrSpots 1
RobustFilter 1
BackgroundFile {results}/bg.bin
MinArea 1
ThresholdMethod fixed
Threshold 5
WatershedImage 0
EnableVisualization 0
EnableSimulation 0
DoFwd 0
OrientationFile {orient}
HKLFile {hkl}
ResultDir {results}
"""


def _frame(path, blobs=((10, 10, 500.0), (40, 20, 900.0))):
    img = np.zeros((NPX, NPX))
    for x, y, v in blobs:
        img[y - 1:y + 2, x - 1:x + 2] = v
    with h5py.File(path, "w") as hf:
        hf.create_dataset("/entry/data/data", data=img)


@pytest.fixture
def runimage_setup(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (tmp_path / "o.bin").write_bytes(b"x")
    (tmp_path / "h.bin").write_bytes(b"x")
    frame = tmp_path / "frame.h5"
    _frame(frame)

    def cfg(extra=""):
        p = tmp_path / "cfg.txt"
        p.write_text(CONFIG.format(n=NPX, orient=tmp_path / "o.bin", hkl=tmp_path / "h.bin",
                                   results=results) + extra)
        return p
    return tmp_path, frame, results, cfg


def _failing_indexer(**kw):
    from laue_index.indexer import IndexerResult
    # what the real run_indexer writes before it returns
    for tag in ("stdout", "stderr"):
        with open(f"{kw['output_path']}.LaueMatching_{tag}.txt", "w") as f:
            f.write(f"indexer {tag}\n")
    return IndexerResult(success=False, returncode=1, error="stub")


# --------------------------------------------------------------------------- #
# repo_root(): the checkout, not the package directory                         #
# --------------------------------------------------------------------------- #

def test_repo_root_is_the_checkout_root():
    if _REPO_ROOT is None:
        pytest.skip("needs a checkout")
    from laue_index.pipeline import repo_root
    assert repo_root() == _REPO_ROOT


def test_daemon_lookup_passes_the_checkout_root(monkeypatch):
    if _REPO_ROOT is None:
        pytest.skip("needs a checkout")
    import laue_index.indexer as ix
    import laue_orchestrator as lo
    seen = {}

    def fake_require(kind, repo_root=None, **k):
        seen["repo_root"] = repo_root
        return "/x/LaueMatchingGPUStream"
    monkeypatch.setattr(ix, "require_binary", fake_require)
    lo._find_daemon_binary()
    assert seen["repo_root"] == str(_REPO_ROOT)


def test_default_orientation_db_is_looked_for_in_the_checkout(runimage_setup, monkeypatch):
    tmp_path, frame, results, cfg = runimage_setup
    import RunImage
    from laue_config import ConfigurationManager
    fake_root = tmp_path / "checkout"
    fake_root.mkdir()
    (fake_root / "100MilOrients.bin").write_bytes(b"db")
    monkeypatch.setattr(RunImage, "repo_root", lambda: fake_root)
    monkeypatch.delenv("LAUEMATCHING_ORIENT_DB", raising=False)
    monkeypatch.setattr(RunImage, "run_indexer", _failing_indexer)
    missing = tmp_path / "not_there.bin"
    p = cfg().read_text().replace(str(tmp_path / "o.bin"), str(missing))
    (tmp_path / "cfg.txt").write_text(p)
    proc = RunImage.EnhancedImageProcessor(ConfigurationManager(str(tmp_path / "cfg.txt")))
    proc.process_image(str(frame))
    assert missing.read_bytes() == b"db", "default DB not copied from the checkout root"


def test_missing_orientation_db_points_to_fetch_db(runimage_setup, monkeypatch):
    tmp_path, frame, results, cfg = runimage_setup
    import RunImage
    from laue_config import ConfigurationManager
    monkeypatch.setattr(RunImage, "repo_root", lambda: None)
    monkeypatch.delenv("LAUEMATCHING_ORIENT_DB", raising=False)
    p = cfg().read_text().replace(str(tmp_path / "o.bin"), str(tmp_path / "not_there.bin"))
    (tmp_path / "cfg.txt").write_text(p)
    proc = RunImage.EnhancedImageProcessor(ConfigurationManager(str(tmp_path / "cfg.txt")))
    res = proc.process_image(str(frame))
    assert not res["success"]
    assert "fetch-db" in res["error"]


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def test_both_threshold_flags_parse(monkeypatch, capsys):
    import RunImage
    monkeypatch.setattr(sys, "argv", ["RunImage.py", "process", "-c", "c.txt", "-i", "i.h5",
                                      "-t", "500", "--threshold-percentile", "95"])
    args = RunImage.parse_arguments()
    assert args.threshold == 500
    assert "--threshold value override will be used" in capsys.readouterr().err


def test_run_exits_nonzero_when_an_image_fails(runimage_setup):
    tmp_path, frame, results, cfg = runimage_setup
    from laue_index.cli import _cmd_run
    bad = tmp_path / "broken.h5"
    bad.write_bytes(b"not an hdf5 file")
    assert _cmd_run(["process", "-c", str(cfg()), "-i", str(bad)]) == 1


def test_run_exits_nonzero_when_the_config_fails(tmp_path):
    from laue_index.cli import _cmd_run
    bad = tmp_path / "cfg.txt"
    bad.write_text("SpaceGroup __SET_ME__\n")
    frame = tmp_path / "f.h5"
    _frame(frame)
    assert _cmd_run(["process", "-c", str(bad), "-i", str(frame)]) == 1


def test_dry_run_still_exits_zero(runimage_setup):
    tmp_path, frame, results, cfg = runimage_setup
    from laue_index.cli import _cmd_run
    assert _cmd_run(["process", "-c", str(cfg()), "-i", str(frame), "--dry-run"]) == 0


# --------------------------------------------------------------------------- #
# RunImage honours ExcludeSpotsFile / ExcludeSpotsDir                          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("how", ["file", "dir"])
def test_runimage_applies_exclusion(runimage_setup, monkeypatch, how):
    tmp_path, frame, results, cfg = runimage_setup
    import RunImage
    from laue_config import ConfigurationManager
    monkeypatch.setattr(RunImage, "run_indexer", _failing_indexer)
    if how == "file":
        (tmp_path / "excl.txt").write_text("40 20 3\n")
        extra = f"ExcludeSpotsFile {tmp_path / 'excl.txt'}\n"
    else:
        d = tmp_path / "excl"
        d.mkdir()
        (d / "frame.txt").write_text("40 20 3\n")      # <frame-stem>.txt
        extra = f"ExcludeSpotsDir {d}\n"
    proc = RunImage.EnhancedImageProcessor(ConfigurationManager(str(cfg(extra))))
    proc.process_image(str(frame))
    with h5py.File(results / "frame.output.h5", "r") as hf:
        centers = hf["/entry/data/component_centers"][()]
    xs = sorted(round(c[1]) for c in centers)
    assert xs == [10], f"excluded blob at x=40 still indexed: {xs}"


# --------------------------------------------------------------------------- #
# The indexer's own stdout/stderr are embedded in the output h5                #
# --------------------------------------------------------------------------- #

def test_indexer_logs_are_embedded(runimage_setup, monkeypatch):
    tmp_path, frame, results, cfg = runimage_setup
    import RunImage
    from laue_config import ConfigurationManager
    monkeypatch.setattr(RunImage, "run_indexer", _failing_indexer)
    proc = RunImage.EnhancedImageProcessor(ConfigurationManager(str(cfg())))
    proc.process_image(str(frame))
    with h5py.File(results / "frame.output.h5", "r") as hf:
        assert "/entry/logs/stdout" in hf
        assert hf["/entry/logs/stdout"][()].decode() == "indexer stdout\n"
        assert "/entry/logs/stderr" in hf


# --------------------------------------------------------------------------- #
# preprocess: no uint16 wrap, no loss of fractional counts                     #
# --------------------------------------------------------------------------- #

def _pp_cfg(**over):
    cfg = {"nr_px_x": NPX, "nr_px_y": NPX, "threshold_method": "fixed",
           "threshold_value": 0.1, "threshold_percentile": 90.0, "min_area": 1,
           "filter_radius": 0, "median_passes": 0, "px_x": 0.0002, "distance": 0.05,
           "orientation_spacing": 0.4, "gaussian_factor": 0.25}
    cfg.update(over)
    return cfg


def test_preprocess_keeps_counts_above_65535():
    from laue_index.preprocess import preprocess_image
    img = np.zeros((NPX, NPX))
    img[30:33, 30:33] = 70000.0
    out = preprocess_image(img, _pp_cfg(), background=np.zeros((NPX, NPX)),
                           return_intermediates=True)
    assert out["filt_img"].max() >= 65535, out["filt_img"].max()


def test_preprocess_keeps_fractional_counts():
    from laue_index.preprocess import preprocess_image
    img = np.zeros((NPX, NPX))
    img[30:33, 30:33] = 0.5
    out = preprocess_image(img, _pp_cfg(), background=np.zeros((NPX, NPX)),
                           return_intermediates=True)
    assert len(out["centers"]) == 1
    assert out["blurred"].max() > 0


def test_runimage_keeps_counts_above_65535(runimage_setup, monkeypatch):
    tmp_path, frame, results, cfg = runimage_setup
    import RunImage
    from laue_config import ConfigurationManager
    monkeypatch.setattr(RunImage, "run_indexer", _failing_indexer)
    _frame(frame, blobs=((20, 20, 70000.0),))
    proc = RunImage.EnhancedImageProcessor(ConfigurationManager(str(cfg())))
    proc.process_image(str(frame))
    with h5py.File(results / "frame.output.h5", "r") as hf:
        assert hf["/entry/data/cleaned_data_threshold_filtered"][()].max() >= 65535


# --------------------------------------------------------------------------- #
# GaussianFactor is read, not hard-coded                                       #
# --------------------------------------------------------------------------- #

def test_gaussian_factor_scales_the_blur():
    from laue_index.preprocess import calculate_gaussian_sigma
    centers = [[1, (float(x), 5.0), 4] for x in range(0, 400, 40)]
    s1 = calculate_gaussian_sigma(centers, 0.0002, 0.5, 0.4, factor=0.25)
    s2 = calculate_gaussian_sigma(centers, 0.0002, 0.5, 0.4, factor=0.5)
    assert s1 > 1.0 and s2 == pytest.approx(2 * s1)


def test_gaussian_factor_is_parsed_by_both_parsers(tmp_path):
    import laue_stream_utils as lsu
    from laue_config import ConfigurationManager
    p = tmp_path / "cfg.txt"
    p.write_text(CONFIG.format(n=NPX, orient="o", hkl="h", results="r") + "GaussianFactor 0.5\n")
    assert ConfigurationManager(str(p)).config.image_processing.gaussian_factor == 0.5
    assert lsu.parse_config(str(p))["gaussian_factor"] == 0.5


def test_streaming_preprocess_uses_gaussian_factor(monkeypatch):
    import laue_index.preprocess as pre
    seen = {}
    real = pre.calculate_gaussian_sigma

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)
    monkeypatch.setattr(pre, "calculate_gaussian_sigma", spy)
    img = np.zeros((NPX, NPX))
    img[10:13, 10:13] = 100.0
    img[40:43, 40:43] = 100.0
    pre.preprocess_image(img, _pp_cfg(gaussian_factor=0.5), background=np.zeros((NPX, NPX)))
    assert seen.get("factor") == 0.5


# --------------------------------------------------------------------------- #
# laue_visualization reads the column layout it is given                       #
# --------------------------------------------------------------------------- #

def test_interactive_plot_puts_stream_spots_at_their_pixels(tmp_path, monkeypatch):
    pytest.importorskip("plotly")
    import laue_visualization as lv
    figs = []
    real = lv.make_subplots

    def keep(*a, **k):
        figs.append(real(*a, **k))
        return figs[-1]
    monkeypatch.setattr(lv, "make_subplots", keep)
    orient = np.zeros((1, 35))
    orient[0, 0], orient[0, 1], orient[0, 5] = 7, 3, 10.0      # ImageNr, GrainNr, quality
    # ImageNr GrainNr SpotNr h k l X Y Q0 Q1 Q2 I
    spots = np.array([[7, 3, 0, 1, 1, 1, 21.0, 41.0, 0, 0, 1, 900.0]])
    r = lv.create_interactive_visualization(str(tmp_path / "x"), orient, spots,
                                            np.zeros((NPX, NPX), int), np.zeros((NPX, NPX)),
                                            None, (NPX, NPX))
    assert r["success"], r
    sc = [t for t in figs[0].data if t.type == "scatter"]
    assert len(sc) == 1
    assert list(sc[0].x) == [21.0] and list(sc[0].y) == [41.0]
    assert sc[0].name.startswith("Grain 3")


def test_simulation_comparison_does_not_call_predicted_spots_experimental():
    import inspect
    import laue_visualization as lv
    src = inspect.getsource(lv.create_simulation_comparison_visualization)
    assert 'f"Exp Grain' not in src and "Source: Exp" not in src
