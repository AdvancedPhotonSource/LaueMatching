"""Runs check their data artifacts against their provenance records.

Before launching the indexer, RunImage and the streaming orchestrator call
laue_index.artifacts.check_run_inputs: an HKL list recorded for another
crystal is REFUSED (the run stops before the C starts), an unrecorded file only
warns, and the identity of every artifact used is written into the run's
provenance under "artifacts" (lineage).
"""
import json
import shutil

import h5py
import pytest

from laue_index import artifacts as A
from test_p1_orchestrator import REQUIRED, _daemon_outputs, stubbed  # noqa: F401
import test_runimage_orchestration as TRO


def _hkl_record(path, lattice, sg=225, sym="F"):
    A.write_record(path, "hkl_list", config={"SpaceGroup": sg, "Symmetry": sym,
                                             "LatticeParameter": lattice})


# ---- streaming orchestrator --------------------------------------------------

def test_orchestrator_refuses_an_hkl_list_for_another_crystal(stubbed, tmp_path):
    state, go, out = stubbed
    (tmp_path / "h.bin").write_bytes(b"1 0 0 1\n")
    _hkl_record(tmp_path / "h.bin", [0.2921, 0.2921, 0.4665, 90, 90, 120], sg=194, sym="P")
    with pytest.raises(SystemExit) as e:
        go(REQUIRED)
    assert e.value.code not in (0, None)
    assert state["popen_cwd"] == [], "the daemon was started despite the mismatch"


def test_orchestrator_records_artifact_lineage(stubbed, tmp_path):
    state, go, out = stubbed
    (tmp_path / "h.bin").write_bytes(b"1 0 0 1\n")
    _hkl_record(tmp_path / "h.bin", [0.35238, 0.35238, 0.35238, 90, 90, 90])
    _daemon_outputs(out, "results_stream")
    go(REQUIRED)
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["artifacts"]["hkl_list"]["record"] is True
    assert prov["artifacts"]["params_file"]["sha256"]


# ---- RunImage ------------------------------------------------------------------

def _runimage(tmp_path, monkeypatch, hkl_lattice):
    from laue_config import ConfigurationManager
    results = tmp_path / "results"; results.mkdir()
    frame = tmp_path / "frame.h5"; TRO._synthetic_h5(frame)
    hkls = tmp_path / "hkls.txt"
    shutil.copy(str(TRO.fixture("sample.spots.txt")), hkls)
    if hkl_lattice is not None:
        _hkl_record(hkls, hkl_lattice)
    cfg = tmp_path / "cfg.txt"
    cfg.write_text(TRO._CONFIG.format(dummy=TRO.fixture("sample.solutions.txt"), hkls=hkls,
                                      fwd=tmp_path / "fwd.bin", results=results))
    called = []

    def fake_run_indexer(**kw):
        called.append(kw)
        ib = kw["image_bin"]
        shutil.copy(str(TRO.fixture("sample.solutions.txt")), ib + ".solutions.txt")
        shutil.copy(str(TRO.fixture("sample.spots.txt")), ib + ".spots.txt")
        return TRO.IndexerResult(success=True, returncode=0)
    monkeypatch.setattr(TRO.RunImage, "run_indexer", fake_run_indexer)
    proc = TRO.RunImage.EnhancedImageProcessor(ConfigurationManager(str(cfg)))
    return proc.process_image(str(frame)), called, results / "frame.output.h5"


def test_runimage_refuses_an_hkl_list_for_another_crystal(tmp_path, monkeypatch):
    res, called, _ = _runimage(tmp_path, monkeypatch, [0.2921, 0.2921, 0.4665, 90, 90, 120])
    assert not res["success"] and "LatticeParameter" in str(res)
    assert called == [], "the indexer ran despite the mismatch"


def test_runimage_records_artifact_lineage(tmp_path, monkeypatch):
    res, called, out_h5 = _runimage(tmp_path, monkeypatch, [0.35238, 0.35238, 0.35238, 90, 90, 90])
    assert res["success"], res
    import laue_provenance as lp
    with h5py.File(out_h5, "r") as hf:
        prov = lp.read_from_h5(hf, group="/entry/provenance")
    assert prov["artifacts"]["hkl_list"]["record"] is True
