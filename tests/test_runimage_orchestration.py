"""RunImage orchestration coverage WITHOUT the C indexer or orientation DB.

The gated e2e (test_char_e2e) needs the real binary + 6.7 GB DB.  This test
mocks the Indexer stage (run_indexer) so the rest of RunImage's per-image
pipeline — load -> background -> threshold -> connected-components -> blur ->
[mock index] -> PostProcessor -> HDF5 output — runs in CI on a synthetic frame.

Covers the wiring the e2e exercises but that nothing else does, with no GPU/DB.
"""
import shutil

import numpy as np
import pytest

import RunImage
from laue_index.indexer import IndexerResult
from _golden import fixture

h5py = pytest.importorskip("h5py")

_NPX = 2048
_CONFIG = """\
SpaceGroup 225
Symmetry F
LatticeParameter 0.352380 0.352380 0.352380 90.0 90.0 90.0
P_Array 0.028828 0.002715 0.512993
R_Array -1.20161887 -1.21404493 -1.21852276
PxX 0.000200
PxY 0.000200
NrPxX 2048
NrPxY 2048
Elo 5.0
Ehi 30.0
MinNrSpots 5
MinGoodSpots 2
MinIntensity 50
MaxAngle 5
MinArea 1
Threshold 1000
ThresholdMethod fixed
WatershedImage 0
NMeadianPasses 0
MaxNrLaueSpots 30
RobustFilter 1
OrientationFile {dummy}
HKLFile {hkls}
ForwardFile {fwd}
DoFwd 0
ResultDir {results}
EnableVisualization 0
EnableSimulation 0
"""


def _synthetic_h5(path):
    """A 2048^2 frame with bright blobs at the committed sample spots' (X,Y),
    so segmentation labels align with the canned solutions for the filter."""
    spots = np.atleast_2d(np.loadtxt(str(fixture("sample.spots.txt")), skiprows=1))
    img = np.zeros((_NPX, _NPX), dtype=np.float64)
    for r in spots:
        x, y = int(r[5]), int(r[6])  # runimage spots: X=col5, Y=col6
        if 1 <= x < _NPX - 1 and 1 <= y < _NPX - 1:
            img[y - 1:y + 2, x - 1:x + 2] = 5000.0
    with h5py.File(path, "w") as hf:
        hf.create_dataset("/entry/data/data", data=img)


def test_runimage_process_image_orchestration(tmp_path, monkeypatch):
    from laue_config import ConfigurationManager

    results = tmp_path / "results"
    results.mkdir()
    frame = tmp_path / "frame.h5"
    _synthetic_h5(frame)

    cfg_file = tmp_path / "cfg.txt"
    cfg_file.write_text(_CONFIG.format(
        dummy=fixture("sample.solutions.txt"),  # any existing file (indexer mocked)
        hkls=fixture("sample.spots.txt"),
        fwd=tmp_path / "fwd.bin", results=results))

    # Mock the Indexer stage: write the canned real C output instead of running
    # the binary.  RunImage._run_indexing calls run_indexer(image_bin=..., ...).
    def fake_run_indexer(**kw):
        ib = kw["image_bin"]
        shutil.copy(str(fixture("sample.solutions.txt")), ib + ".solutions.txt")
        shutil.copy(str(fixture("sample.spots.txt")), ib + ".spots.txt")
        return IndexerResult(success=True, returncode=0)
    monkeypatch.setattr(RunImage, "run_indexer", fake_run_indexer)

    proc = RunImage.EnhancedImageProcessor(ConfigurationManager(str(cfg_file)))
    res = proc.process_image(str(frame))

    assert res["success"], res
    out_h5 = results / "frame.output.h5"
    assert out_h5.exists()
    with h5py.File(out_h5, "r") as hf:
        assert "/entry/results/orientations" in hf
        assert "/entry/results/filtered_orientations" in hf
        # the canned solutions had 2 grains; the pipeline kept >=1 after filtering
        assert hf["/entry/results/filtered_orientations"].shape[0] >= 1


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    print("run via pytest (needs monkeypatch fixture)")
